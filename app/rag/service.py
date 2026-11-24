import copy
import json
import logging
from datetime import date
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from app.core.constants import DEFAULT_LANG
from app.core.es import es_client
from app.embedding.service import EmbeddingService, embedding_service
from app.llm.base import ChatTurn, LLMClient, Message, ToolCall, ToolsNotSupportedError
from app.llm.constants import LLM_PROVIDER_OLLAMA
from app.llm.factory import get_client
from app.rag.constants import (
    RAG_FULL_TEXT_QUERY_FILE,
    RAG_HISTORY_INPUT_SHARE,
    RAG_MAX_ROUNDS,
    RAG_MAX_PASSAGES,
    RAG_MAX_QUESTION_CHARS,
    RAG_MIN_COSINE,
    RAG_READ_INPUT_SHARE,
    RAG_RETRIEVE_SIZE,
    RAG_SEMANTIC_QUERY_FILE,
    RAG_TEMPERATURE,
    RAG_TOP_CHUNKS,
    RAG_WINDOW_WORDS,
)
from app.rag.passages import (
    Passage,
    add_copy,
    add_copy_to_passage,
    normalize_text,
    split_windows,
)
from app.rag.prompts import (
    CONDENSE_PROMPT,
    CONTEXT_MESSAGE,
    EMPTY_QUESTION,
    GAVE_UP,
    NO_ANSWER,
    ROLE_LABEL,
    SOURCE_LABEL,
    agent_system_prompt,
    pipeline_system_prompt,
)
from app.rag.schemas import HistoryMessage
from app.rag.tools import (
    FIND_DOCUMENTS,
    READ_DOCUMENT,
    SEARCH_DOCUMENTS,
    TOOLS,
    DocumentTools,
    TurnContext,
)
from app.search.query_builder import QueryTemplates, SearchQueryBuilder
from app.search.rrf import RRFMerger
from app.search.schemas import SearchMode
from app.search.service import SearchService
from app.search.types import SearchCandidate
from app.summarize.service import summarize_service

logger = logging.getLogger(__name__)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class RagService:
    def __init__(
        self,
        embedding: EmbeddingService,
        model_name: Callable[[], str | None],
        provider: Callable[[], str | None],
    ) -> None:
        self.embedding = embedding
        self.model_name = model_name
        self.provider = provider
        self.full_text_template = self.load_template(
            SearchMode.FULL_TEXT, RAG_FULL_TEXT_QUERY_FILE
        )
        self.semantic_template = self.load_template(
            SearchMode.SEMANTIC, RAG_SEMANTIC_QUERY_FILE
        )
        self.tools = DocumentTools(self.retrieve)

    @staticmethod
    def load_template(mode: SearchMode, path: str) -> dict[str, Any]:
        body = QueryTemplates.read(path)
        QueryTemplates.validate(mode, body, path)
        return body

    def stream(
        self,
        question: str,
        owner_id: str,
        language: str,
        exclude_document_ids: list[str] | None = None,
        history: list[HistoryMessage] | None = None,
    ) -> Iterator[str]:
        """Answer one question as server-sent events.

        Events: status (what the assistant is doing), sources (passages it has
        read so far), token (answer text), done (model and token usage), error.
        """
        lang = language if language in ROLE_LABEL else DEFAULT_LANG
        question = (question or "").strip()[:RAG_MAX_QUESTION_CHARS]
        if not question:
            yield sse("error", {"message": EMPTY_QUESTION[lang]})
            return

        model = self.model_name()
        provider = self.provider()
        if not model:
            yield sse("error", {"message": "No LLM model loaded from MLflow"})
            return
        try:
            client = get_client(provider)
        except Exception as e:
            logger.exception("Could not create the LLM client")
            yield sse("error", {"message": f"LLM error: {e}"})
            return

        ctx = TurnContext(
            owner_id=owner_id,
            exclude_document_ids=[str(d) for d in exclude_document_ids or []],
            read_max_chars=int(client.max_input_chars * RAG_READ_INPUT_SHARE),
        )
        past = self.history_messages(history or [], client.max_input_chars)

        try:
            try:
                yield from self.answer_with_tools(client, model, question, past, lang, ctx)
            except ToolsNotSupportedError:
                logger.info("Model %s cannot call tools, using the fixed RAG pipeline", model)
                yield from self.answer_with_pipeline(client, model, question, past, lang, ctx)
        except Exception as e:
            logger.exception("RAG answer failed")
            yield sse("error", {"message": f"LLM error: {e}"})
            return

        label = model if provider == LLM_PROVIDER_OLLAMA else f"{provider}/{model}"
        yield sse("done", {"model": label, "usage": ctx.usage.as_dict()})

    def answer_with_tools(
        self,
        client: LLMClient,
        model: str,
        question: str,
        past: list[Message],
        lang: str,
        ctx: TurnContext,
    ) -> Iterator[str]:
        """Let the model search and read documents itself, then answer."""
        messages: list[Message] = [
            {"role": "system", "content": agent_system_prompt(lang, date.today().isoformat())},
            *past,
            {"role": "user", "content": question},
        ]
        yield sse("status", {"stage": "thinking"})

        for round_no in range(RAG_MAX_ROUNDS):
            # Last round: no more tools, answer with what has been found.
            last = round_no == RAG_MAX_ROUNDS - 1
            turn = ChatTurn()
            for part in client.chat_stream(
                model, messages, TOOLS, RAG_TEMPERATURE, force_answer=last
            ):
                if isinstance(part, ChatTurn):
                    turn = part
                else:
                    yield sse("token", {"text": part})
            ctx.usage.add(turn.usage)

            if not turn.tool_calls:
                if not turn.content.strip():
                    yield sse("token", {"text": GAVE_UP[lang]})
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": turn.tool_calls,
                    "raw": turn.raw,
                }
            )
            for call in turn.tool_calls:
                yield sse("status", self.tool_status(call, ctx))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": self.tools.run(call, ctx),
                    }
                )
            yield sse("sources", {"sources": ctx.sources.payload()})
            yield sse("status", {"stage": "thinking"})

    def answer_with_pipeline(
        self,
        client: LLMClient,
        model: str,
        question: str,
        past: list[Message],
        lang: str,
        ctx: TurnContext,
    ) -> Iterator[str]:
        """Fallback for models without tools: rewrite, retrieve once, answer."""
        query = question
        if past:
            query = self.condense(client, model, question, past, lang, ctx) or question

        yield sse("status", {"stage": "searching", "query": query})
        passages = self.retrieve(query, ctx.owner_id, ctx.exclude_document_ids)
        for p in passages:
            ctx.sources.add(p)
        yield sse("sources", {"sources": ctx.sources.payload()})

        if not passages:
            yield sse("token", {"text": NO_ANSWER[lang]})
            return

        label = SOURCE_LABEL[lang]
        context = "\n\n".join(f"({label}: {p.name or p.document_id})\n{p.text}" for p in passages)
        messages: list[Message] = [
            {"role": "system", "content": pipeline_system_prompt(lang, date.today().isoformat())},
            *past,
            {"role": "user", "content": CONTEXT_MESSAGE[lang].format(context=context, question=question)},
        ]
        yield sse("status", {"stage": "answering"})
        for part in client.chat_stream(model, messages, None, RAG_TEMPERATURE):
            if isinstance(part, ChatTurn):
                ctx.usage.add(part.usage)
            else:
                yield sse("token", {"text": part})

    @staticmethod
    def condense(
        client: LLMClient,
        model: str,
        question: str,
        past: list[Message],
        lang: str,
        ctx: TurnContext,
    ) -> str:
        """Turn a follow-up question into a standalone search query."""
        labels = ROLE_LABEL[lang]
        transcript = "\n".join(f"{labels[m['role']]}: {m['content']}" for m in past)
        prompt = CONDENSE_PROMPT[lang].format(history=transcript, question=question)
        try:
            turn = client.chat(model, [{"role": "user", "content": prompt}], 0.0)
        except Exception:
            logger.warning("Could not rewrite the follow-up question", exc_info=True)
            return question
        ctx.usage.add(turn.usage)
        return turn.content.strip().strip('"').splitlines()[0] if turn.content.strip() else question

    @staticmethod
    def history_messages(history: list[HistoryMessage], max_input_chars: int) -> list[Message]:
        """Keep the latest turns that fit in the history share of the input budget."""
        budget = int(max_input_chars * RAG_HISTORY_INPUT_SHARE)
        kept: list[Message] = []
        for item in reversed(history):
            content = item.content.strip()
            if not content:
                continue
            if budget < 200:
                break
            if len(content) > budget:
                content = content[:budget].rstrip() + "..."
            budget -= len(content)
            kept.append({"role": item.role, "content": content})
        kept.reverse()
        # Start on a user turn so the model sees whole exchanges.
        while kept and kept[0]["role"] != "user":
            kept.pop(0)
        return kept

    @staticmethod
    def tool_status(call: ToolCall, ctx: TurnContext) -> dict[str, Any]:
        if call.name == SEARCH_DOCUMENTS.name:
            return {"stage": "searching", "query": str(call.arguments.get("query", ""))}
        if call.name == FIND_DOCUMENTS.name:
            return {"stage": "finding", "name": str(call.arguments.get("name", ""))}
        if call.name == READ_DOCUMENT.name:
            document_id = str(call.arguments.get("document_id", ""))
            return {"stage": "reading", "name": ctx.names.get(document_id, "")}
        return {"stage": "thinking"}

    def retrieve(
        self, question: str, owner_id: str, exclude_document_ids: list[str]
    ) -> list[Passage]:
        query_vector = self.embedding.encode_query(question)
        chunks = self.search_chunks(
            question, query_vector, owner_id, exclude_document_ids
        )
        return self.select_passages(query_vector, chunks)

    def search_chunks(
        self,
        question: str,
        query_vector: list[float],
        owner_id: str,
        exclude_document_ids: list[str],
    ) -> list[dict[str, Any]]:
        filters = SearchQueryBuilder.owner_filter(owner_id)
        excluded = (
            [{"terms": {"document_id": [str(d) for d in exclude_document_ids]}}]
            if exclude_document_ids
            else []
        )

        bm25_resp, knn_resp = SearchService.multi_search(
            es_client.get_client(),
            [
                self.full_text_body(question, filters, excluded),
                self.semantic_body(query_vector, filters, excluded),
            ],
        )
        return self.merge_chunks(bm25_resp["hits"]["hits"], knn_resp["hits"]["hits"])

    def full_text_body(
        self, question: str, filters: list[dict], excluded: list[dict]
    ) -> dict[str, Any]:
        body = copy.deepcopy(self.full_text_template)
        body["size"] = RAG_RETRIEVE_SIZE
        SearchQueryBuilder.set_text_clauses(body, question)
        body["query"]["bool"]["filter"] = filters
        body["query"]["bool"]["must_not"] = excluded
        return body

    def semantic_body(
        self, query_vector: list[float], filters: list[dict], excluded: list[dict]
    ) -> dict[str, Any]:
        body = copy.deepcopy(self.semantic_template)
        body["size"] = RAG_RETRIEVE_SIZE
        SearchQueryBuilder.apply_knn(body, query_vector, RAG_RETRIEVE_SIZE, filters)
        body["knn"]["filter"]["bool"]["must_not"] = excluded
        return body

    @staticmethod
    def merge_chunks(
        bm25_hits: list[dict[str, Any]], vector_hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for hit in (*bm25_hits, *vector_hits):
            by_id.setdefault(hit["_id"], hit)

        def candidates(hits: list[dict[str, Any]]) -> list[SearchCandidate]:
            return [
                SearchCandidate(document_id=h["_id"], score=h.get("_score") or 0.0)
                for h in hits
            ]

        merged = RRFMerger().merge(candidates(bm25_hits), candidates(vector_hits))
        top: list[dict[str, Any]] = []
        kept: dict[str, dict[str, Any]] = {}
        for c in merged:
            hit = by_id[c.document_id]
            source = hit.get("_source") or {}
            key = normalize_text(source.get("content") or "")
            if key in kept:
                add_copy(kept[key], source.get("document_id"), source.get("name"))
                continue
            if len(top) >= RAG_TOP_CHUNKS:
                continue
            hit["_copies"] = []
            kept[key] = hit
            top.append(hit)
        return top

    def select_passages(
        self, query_vector: list[float], chunks: list[dict[str, Any]]
    ) -> list[Passage]:
        candidates: list[Passage] = []
        seen: dict[str, Passage] = {}
        for hit in chunks:
            source = hit.get("_source") or {}
            content = source.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            document_id = str(source.get("document_id"))
            name = source.get("name")
            copies = hit.get("_copies") or []
            for window in split_windows(content, RAG_WINDOW_WORDS):
                key = normalize_text(window)
                existing = seen.get(key)
                if existing is not None:
                    for copy_id, copy_name in [(document_id, name), *copies]:
                        add_copy_to_passage(existing, copy_id, copy_name)
                    continue
                passage = Passage(document_id, name, window, 0.0, list(copies))
                seen[key] = passage
                candidates.append(passage)

        if not candidates:
            return []

        window_vectors = self.embedding.encode_many([c.text for c in candidates])
        similarities = self.cosine(np.asarray(query_vector), window_vectors)

        passages: list[Passage] = []
        for idx in np.argsort(-similarities):
            score = float(similarities[idx])
            if score < RAG_MIN_COSINE or len(passages) >= RAG_MAX_PASSAGES:
                break

            passage = candidates[idx]
            passage.score = score
            passages.append(passage)

        return passages

    @staticmethod
    def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query) or 1.0
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0
        return (matrix @ query) / (matrix_norms * query_norm)


rag_service = RagService(
    embedding_service,
    model_name=lambda: summarize_service.model_name,
    provider=lambda: summarize_service.provider,
)
