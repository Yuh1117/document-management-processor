import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import ollama

from app.constants.defaults import (
    DEFAULT_LANG,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
    RAG_CONTEXT_CHARS,
    RAG_FULL_TEXT_QUERY_FILE,
    RAG_MAX_PASSAGES,
    RAG_MAX_QUESTION_CHARS,
    RAG_MIN_SIMILARITY,
    RAG_RETRIEVE_SIZE,
    RAG_SEMANTIC_QUERY_FILE,
    RAG_TOP_CHUNKS,
    RAG_WINDOW_WORDS,
)
from app.core.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL_NAME,
)
from app.core.es import es_client
from app.services.embedding_service import EmbeddingService, embedding_service
from app.models.search import SearchMode
from app.services.search_service import (
    QueryTemplates,
    RRFMerger,
    SearchCandidate,
    SearchQueryBuilder,
    SearchService,
)
from app.services.summarize_service import summarize_service

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 280

PROMPTS = {
    "vi": (
        "Bạn là trợ lý hỏi đáp về tài liệu của người dùng. Chỉ trả lời dựa trên các "
        "đoạn trích dưới đây, không dùng kiến thức bên ngoài. Nếu các đoạn trích không "
        "có thông tin để trả lời, hãy nói rõ là bạn không tìm thấy thông tin trong tài "
        "liệu. Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng.\n\n"
        "Các đoạn trích:\n{context}\n\n"
        "Câu hỏi: {question}\n\nTrả lời:"
    ),
    "en": (
        "You are an assistant that answers questions about the user's documents. "
        "Answer only from the excerpts below and do not use outside knowledge. If the "
        "excerpts do not contain the answer, say clearly that you could not find it in "
        "the documents. Answer in English, concisely.\n\n"
        "Excerpts:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}

SOURCE_LABEL = {"vi": "Tài liệu", "en": "Document"}

NO_ANSWER = {
    "vi": "Tôi không tìm thấy thông tin liên quan trong tài liệu của bạn.",
    "en": "I could not find relevant information in your documents.",
}

EMPTY_QUESTION = {
    "vi": "Vui lòng nhập câu hỏi.",
    "en": "Please enter a question.",
}


@dataclass
class Passage:
    document_id: str
    name: str | None
    text: str
    score: float


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def split_windows(text: str, words_per_window: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_window])
        for i in range(0, len(words), words_per_window)
    ]


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


class RagService:
    def __init__(
        self,
        embedding: EmbeddingService,
        model_name: Callable[[], str | None],
    ) -> None:
        self.embedding = embedding
        self.model_name = model_name
        self.client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
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

        model = self.model_name() or OLLAMA_MODEL_NAME
        prompt = self.build_prompt(question, passages, lang)

        generation = None
        try:
            generation = self.client.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            for part in generation:
                if part.response:
                    yield sse("token", {"text": part.response})
        except Exception as e:
            logger.exception("RAG generation failed")
            yield sse("error", {"message": f"LLM error: {e}"})
            return
        finally:
            close = getattr(generation, "close", None)
            if close is not None:
                close()

        yield sse("done", {"model": model})

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
        return [by_id[c.document_id] for c in merged[:RAG_TOP_CHUNKS]]

    def select_passages(
        self, query_vector: list[float], chunks: list[dict[str, Any]]
    ) -> list[Passage]:
        candidates: list[tuple[str, str | None, str]] = []
        for hit in chunks:
            source = hit.get("_source") or {}
            content = source.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            for window in split_windows(content, RAG_WINDOW_WORDS):
                candidates.append(
                    (str(source.get("document_id")), source.get("name"), window)
                )

        if not candidates:
            return []

        window_vectors = self.embedding.encode_many([c[2] for c in candidates])
        similarities = self.cosine(np.asarray(query_vector), window_vectors)

        passages: list[Passage] = []
        used_chars = 0
        for idx in np.argsort(-similarities):
            score = float(similarities[idx])
            if score < RAG_MIN_SIMILARITY or len(passages) >= RAG_MAX_PASSAGES:
                break

            document_id, name, text = candidates[idx]
            if passages and used_chars + len(text) > RAG_CONTEXT_CHARS:
                continue

            passages.append(Passage(document_id, name, text, score))
            used_chars += len(text)

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
                "document_id": p.document_id,
                "name": p.name,
                "snippet": truncate(p.text, SNIPPET_CHARS),
                "score": round(p.score, 3),
            }
            for i, p in enumerate(passages, start=1)
        ]


rag_service = RagService(
    embedding_service,
    model_name=lambda: summarize_service.model_name,
)
