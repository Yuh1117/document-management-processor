import copy
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from app.core.config import OLLAMA_MODEL_NAME
from app.core.constants import DEFAULT_LANG
from app.core.es import es_client
from app.embedding.service import EmbeddingService, embedding_service
from app.llm.constants import LLM_PROVIDER_OLLAMA
from app.llm.factory import get_client
from app.rag.constants import (
    RAG_CONTEXT_CHARS,
    RAG_FULL_TEXT_QUERY_FILE,
    RAG_MAX_PASSAGES,
    RAG_MAX_QUESTION_CHARS,
    RAG_MIN_SIMILARITY,
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
    truncate,
)
from app.rag.prompts import EMPTY_QUESTION, NO_ANSWER, PROMPTS, SOURCE_LABEL
from app.search.query_builder import QueryTemplates, SearchQueryBuilder
from app.search.rrf import RRFMerger
from app.search.schemas import SearchMode
from app.search.service import SearchService
from app.search.types import SearchCandidate
from app.summarize.service import summarize_service

logger = logging.getLogger(__name__)


SNIPPET_CHARS = 280


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
    ) -> Iterator[str]:
        lang = language if language in PROMPTS else DEFAULT_LANG
        question = (question or "").strip()[:RAG_MAX_QUESTION_CHARS]

        if not question:
            yield sse("error", {"message": EMPTY_QUESTION[lang]})
            return

        try:
            passages = self.retrieve(question, owner_id, exclude_document_ids or [])
        except Exception as e:
            logger.exception("RAG retrieval failed")
            yield sse("error", {"message": f"Retrieval failed: {e}"})
            return

        yield sse("sources", {"sources": self.source_payload(passages)})

        if not passages:
            yield sse("token", {"text": NO_ANSWER[lang]})
            yield sse("done", {"model": None})
            return

        model = self.model_name()
        provider = self.provider() if model else LLM_PROVIDER_OLLAMA
        model = model or OLLAMA_MODEL_NAME
        prompt = self.build_prompt(question, passages, lang)

        try:
            for text in get_client(provider).stream(model, prompt, RAG_TEMPERATURE):
                yield sse("token", {"text": text})
        except Exception as e:
            logger.exception("RAG generation failed")
            yield sse("error", {"message": f"LLM error: {e}"})
            return

        yield sse(
            "done",
            {
                "model": model
                if provider == LLM_PROVIDER_OLLAMA
                else f"{provider}/{model}"
            },
        )

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
        used_chars = 0
        for idx in np.argsort(-similarities):
            score = float(similarities[idx])
            if score < RAG_MIN_SIMILARITY or len(passages) >= RAG_MAX_PASSAGES:
                break

            passage = candidates[idx]
            if passages and used_chars + len(passage.text) > RAG_CONTEXT_CHARS:
                continue

            passage.score = score
            passages.append(passage)
            used_chars += len(passage.text)

        return passages

    @staticmethod
    def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query) or 1.0
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0
        return (matrix @ query) / (matrix_norms * query_norm)

    @staticmethod
    def build_prompt(question: str, passages: list[Passage], lang: str) -> str:
        label = SOURCE_LABEL[lang]
        context = "\n\n".join(
            f"({label}: {p.name or p.document_id})\n{p.text}" for p in passages
        )
        return PROMPTS[lang].format(context=context, question=question)

    @staticmethod
    def source_payload(passages: list[Passage]) -> list[dict[str, Any]]:
        return [
            {
                "index": i,
                "document_id": document_id,
                "name": name,
                "snippet": truncate(p.text, SNIPPET_CHARS),
                "score": round(p.score, 3),
            }
            for i, p in enumerate(passages, start=1)
            for document_id, name in [(p.document_id, p.name), *p.copies]
        ]


rag_service = RagService(
    embedding_service,
    model_name=lambda: summarize_service.model_name,
    provider=lambda: summarize_service.provider,
)
