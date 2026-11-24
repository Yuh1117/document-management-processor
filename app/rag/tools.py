"""Tools the assistant can call to look into the user's documents.

Every query is scoped to the asking user's documents and skips the ones in
their trash, whatever arguments the model sends.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import ELASTICSEARCH_INDEX
from app.core.es import es_client
from app.llm.base import ToolCall, ToolSpec, Usage
from app.rag.constants import (
    RAG_FIND_LIMIT,
    RAG_MAX_READS,
    RAG_MAX_TOOL_CALLS,
    RAG_READ_MAX_CHUNKS,
    RAG_SNIPPET_CHARS,
)
from app.rag.passages import Passage, normalize_text, truncate
from app.search.query_builder import SearchQueryBuilder

logger = logging.getLogger(__name__)


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    # Strict mode (OpenAI) needs every property required and no extras.
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SEARCH_DOCUMENTS = ToolSpec(
    name="search_documents",
    description=(
        "Search the content of the user's documents and return the most relevant "
        "passages with their document name and id. Use short, standalone queries with "
        "the key terms; call several times in parallel for multi-part questions."
    ),
    parameters=_schema(
        {"query": {"type": "string", "description": "Standalone search query."}}
    ),
)

FIND_DOCUMENTS = ToolSpec(
    name="find_documents",
    description=(
        "Find the user's documents by file name (partial or approximate names work). "
        "Returns document ids and names; use it before read_document when the user "
        "refers to a file by name."
    ),
    parameters=_schema(
        {"name": {"type": "string", "description": "File name or part of it."}}
    ),
)

READ_DOCUMENT = ToolSpec(
    name="read_document",
    description=(
        "Read the text of one document, from the start (long documents are cut off). "
        "Use it to summarize or answer questions about a whole document."
    ),
    parameters=_schema(
        {
            "document_id": {
                "type": "string",
                "description": "Id from search_documents or find_documents.",
            }
        }
    ),
)

TOOLS = [SEARCH_DOCUMENTS, FIND_DOCUMENTS, READ_DOCUMENT]


class SourceRegistry:
    """Passages shown to the model in this answer, numbered for the UI."""

    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self._seen: dict[str, Passage] = {}

    def add(self, passage: Passage) -> None:
        key = passage.document_id + "\0" + normalize_text(passage.text)[:500]
        if key not in self._seen:
            self._seen[key] = passage
            self.passages.append(passage)

    def payload(self) -> list[dict[str, Any]]:
        return [
            {
                "index": i,
                "document_id": document_id,
                "name": name,
                "snippet": truncate(p.text, RAG_SNIPPET_CHARS),
                "score": round(p.score, 3),
            }
            for i, p in enumerate(self.passages, start=1)
            for document_id, name in [(p.document_id, p.name), *p.copies]
        ]


@dataclass
class TurnContext:
    owner_id: str
    exclude_document_ids: list[str]
    read_max_chars: int
    sources: SourceRegistry = field(default_factory=SourceRegistry)
    usage: Usage = field(default_factory=Usage)
    # Document names seen so far, by id (for status messages).
    names: dict[str, str] = field(default_factory=dict)
    calls_left: int = RAG_MAX_TOOL_CALLS
    reads_left: int = RAG_MAX_READS


class ToolError(Exception):
    """A problem the model should read and react to (bad id, budget used up)."""


def _filters(ctx: TurnContext) -> tuple[list[dict], list[dict]]:
    filters = SearchQueryBuilder.owner_filter(ctx.owner_id)
    excluded = (
        [{"terms": {"document_id": [str(d) for d in ctx.exclude_document_ids]}}]
        if ctx.exclude_document_ids
        else []
    )
    return filters, excluded


def _chunk_number(chunk_id: str | None) -> int:
    match = re.search(r"_(\d+)$", chunk_id or "")
    return int(match.group(1)) if match else 0


def _join_chunks(chunks: list[str]) -> str:
    """Join consecutive chunks, dropping the words they overlap by."""
    words: list[str] = []
    for chunk in chunks:
        next_words = chunk.split()
        overlap = 0
        for size in range(min(len(words), len(next_words), 200), 0, -1):
            if words[-size:] == next_words[:size]:
                overlap = size
                break
        words.extend(next_words[overlap:])
    return " ".join(words)


class DocumentTools:
    def __init__(self, retrieve: Callable[[str, str, list[str]], list[Passage]]) -> None:
        self.retrieve = retrieve

    def run(self, call: ToolCall, ctx: TurnContext) -> str:
        """Run one tool call and return its result as JSON for the model."""
        ctx.usage.tool_calls += 1
        try:
            if ctx.calls_left <= 0 or (call.name == READ_DOCUMENT.name and ctx.reads_left <= 0):
                raise ToolError(
                    f"{call.name} limit reached for this answer; answer with what you have."
                )
            ctx.calls_left -= 1
            if call.name == READ_DOCUMENT.name:
                ctx.reads_left -= 1
            if call.name == SEARCH_DOCUMENTS.name:
                result = self.search(str(call.arguments.get("query", "")), ctx)
            elif call.name == FIND_DOCUMENTS.name:
                result = self.find(str(call.arguments.get("name", "")), ctx)
            elif call.name == READ_DOCUMENT.name:
                result = self.read(str(call.arguments.get("document_id", "")), ctx)
            else:
                raise ToolError(f"Unknown tool {call.name}")
        except ToolError as e:
            result = {"error": str(e)}
        except Exception:
            logger.exception("Tool %s failed", call.name)
            result = {"error": "The tool failed. Answer with what you have."}
        return json.dumps(result, ensure_ascii=False)

    def search(self, query: str, ctx: TurnContext) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ToolError("query is empty")
        passages = self.retrieve(query, ctx.owner_id, ctx.exclude_document_ids)
        for p in passages:
            ctx.sources.add(p)
            if p.name:
                ctx.names[p.document_id] = p.name
        if not passages:
            return {"results": [], "note": "No matching passages."}
        return {
            "results": [
                {"document": p.name or p.document_id, "document_id": p.document_id, "text": p.text}
                for p in passages
            ]
        }

    def find(self, name: str, ctx: TurnContext) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ToolError("name is empty")
        filters, excluded = _filters(ctx)
        body = {
            "size": 50,
            "track_total_hits": False,
            "_source": ["document_id", "name"],
            "query": {
                "bool": {
                    "should": [
                        {"match": {"name": {"query": name, "fuzziness": "AUTO"}}},
                        {"wildcard": {"name.keyword": {"value": f"*{name}*", "case_insensitive": True}}},
                    ],
                    "minimum_should_match": 1,
                    "filter": filters,
                    "must_not": excluded,
                }
            },
        }
        hits = es_client.get_client().search(index=ELASTICSEARCH_INDEX, body=body)["hits"]["hits"]
        documents: dict[str, str | None] = {}
        for hit in hits:
            source = hit.get("_source") or {}
            documents.setdefault(str(source.get("document_id")), source.get("name"))
            if len(documents) >= RAG_FIND_LIMIT:
                break
        if not documents:
            return {"documents": [], "note": "No document with a matching name."}
        ctx.names.update({d: n for d, n in documents.items() if n})
        return {"documents": [{"document_id": d, "name": n} for d, n in documents.items()]}

    def read(self, document_id: str, ctx: TurnContext) -> dict[str, Any]:
        document_id = document_id.strip()
        if not document_id:
            raise ToolError("document_id is empty")
        if document_id in ctx.exclude_document_ids:
            raise ToolError("No such document.")
        filters, _ = _filters(ctx)
        body = {
            "size": RAG_READ_MAX_CHUNKS,
            "track_total_hits": False,
            "_source": ["chunk_id", "name", "content"],
            "query": {"bool": {"filter": [*filters, {"term": {"document_id": document_id}}]}},
        }
        hits = es_client.get_client().search(index=ELASTICSEARCH_INDEX, body=body)["hits"]["hits"]
        if not hits:
            raise ToolError("No such document, or it has no readable text.")

        sources = sorted((h.get("_source") or {} for h in hits), key=lambda s: _chunk_number(s.get("chunk_id")))
        name = sources[0].get("name")
        text = _join_chunks([s.get("content") or "" for s in sources])
        truncated = len(text) > ctx.read_max_chars
        text = text[: ctx.read_max_chars]

        ctx.sources.add(Passage(document_id, name, text, 1.0))
        result: dict[str, Any] = {"document": name or document_id, "text": text}
        if truncated:
            result["note"] = "Only the beginning of the document is shown."
        return result
