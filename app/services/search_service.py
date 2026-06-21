import json
import logging
from dataclasses import dataclass
from typing import Any
from fastapi import HTTPException
from app.constants.defaults import (
    FULL_TEXT_QUERY_FILE,
    SEARCH_CANDIDATE_MULTIPLIER,
    SEARCH_DEFAULT_CANDIDATE_SIZE,
    SEARCH_KNN_POOL_SIZE,
    SEARCH_MAX_CANDIDATE_SIZE,
    SEARCH_RRF_K,
    SEARCH_SEMANTIC_MIN_SCORE,
    SEMANTIC_QUERY_FILE,
    SNIPPET_MAX_CHARS,
)
from app.core.config import ELASTICSEARCH_INDEX
from app.core.es import es_client
from app.models.search import SearchHit, SearchMode
from app.services.embedding_service import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    document_id: str
    score: float
    snippet: str | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None


class SearchQueryBuilder:
    TEMPLATE_PATHS = {
        SearchMode.FULL_TEXT: FULL_TEXT_QUERY_FILE,
        SearchMode.SEMANTIC: SEMANTIC_QUERY_FILE,
    }

    def __init__(self, embedding: EmbeddingService) -> None:
        self.embedding = embedding

    def build(
        self,
        query: str,
        page: int,
        page_size: int,
        owner_id: int | None,
        mode: SearchMode,
    ) -> dict[str, Any]:
        body = self.load_template(mode)
        filters = self.owner_filter(owner_id)

        body["from"] = (page - 1) * page_size
        body["size"] = page_size

        if mode == SearchMode.FULL_TEXT:
            return self.apply_full_text(body, query, filters)

        if mode == SearchMode.SEMANTIC:
            query_vector = self.embedding.encode_text(query)
            knn_k = self.candidate_size(
                page=page,
                page_size=page_size,
                minimum=SEARCH_KNN_POOL_SIZE,
            )
            return self.apply_semantic(body, query_vector, knn_k, filters)

        raise ValueError(f"Unsupported build mode: {mode}")

    def build_full_text_candidate(
        self,
        query: str,
        candidate_size: int,
        owner_id: int | None,
    ) -> dict[str, Any]:
        body = self.load_template(SearchMode.FULL_TEXT)
        filters = self.owner_filter(owner_id)

        body["from"] = 0
        body["size"] = candidate_size

        return self.apply_full_text(body, query, filters)

    def build_semantic_candidate(
        self,
        query: str,
        candidate_size: int,
        owner_id: int | None,
    ) -> dict[str, Any]:
        body = self.load_template(SearchMode.SEMANTIC)
        filters = self.owner_filter(owner_id)

        body["from"] = 0
        body["size"] = candidate_size

        query_vector = self.embedding.encode_text(query)

        return self.apply_semantic(
            body=body,
            query_vector=query_vector,
            knn_k=candidate_size,
            filters=filters,
        )

    def load_template(self, mode: SearchMode) -> dict[str, Any]:
        path = self.TEMPLATE_PATHS[mode]
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def owner_filter(owner_id: int | None) -> list[dict[str, Any]]:
        if owner_id is None:
            return []
        return [{"term": {"owner_id": str(owner_id)}}]

    def apply_full_text(
        self,
        body: dict[str, Any],
        query: str,
        filters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.set_text_clauses(body, query, context="full_text")
        body["query"]["bool"]["filter"] = filters
        return body

    def apply_semantic(
        self,
        body: dict[str, Any],
        query_vector: list[float],
        knn_k: int,
        filters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if SEARCH_SEMANTIC_MIN_SCORE is not None:
            body["min_score"] = SEARCH_SEMANTIC_MIN_SCORE

        self.apply_knn(body, query_vector, knn_k, filters)
        return body

    @staticmethod
    def candidate_size(
        page: int,
        page_size: int,
        minimum: int = SEARCH_DEFAULT_CANDIDATE_SIZE,
    ) -> int:
        page = max(page, 1)
        page_size = max(page_size, 1)

        size = max(
            minimum,
            page * page_size * SEARCH_CANDIDATE_MULTIPLIER,
        )

        return min(size, SEARCH_MAX_CANDIDATE_SIZE)

    @staticmethod
    def apply_knn(
        body: dict[str, Any],
        query_vector: list[float],
        knn_k: int,
        filters: list[dict[str, Any]],
    ) -> None:
        knn = body["knn"]
        knn["query_vector"] = query_vector
        knn["k"] = knn_k
        knn["num_candidates"] = min(knn_k * 5, 10_000)

        if "filter" not in knn:
            if filters:
                raise KeyError("kNN template missing knn.filter.")
            return

        if "bool" not in knn["filter"] or "filter" not in knn["filter"]["bool"]:
            raise KeyError("kNN template has unsupported knn.filter shape.")

        knn["filter"]["bool"]["filter"] = filters

    @staticmethod
    def set_text_clauses(body: dict[str, Any], query: str, context: str) -> None:
        bool_query = body.get("query", {}).get("bool", {})
        clauses = bool_query.get("must", []) + bool_query.get("should", [])
        matched = False

        for clause in clauses:
            if "multi_match" in clause:
                clause["multi_match"]["query"] = query
                matched = True

            if "match" in clause:
                for field, value in clause["match"].items():
                    if isinstance(value, dict):
                        value["query"] = query
                    else:
                        clause["match"][field] = query
                    matched = True

        if not matched:
            raise KeyError(f"Unsupported {context} template: missing text query clause")


class SearchResponseParser:
    @staticmethod
    def hits(resp: dict[str, Any]) -> list[SearchHit]:
        return [
            SearchHit(
                document_id=hit["_source"]["document_id"],
                score=hit.get("_score") or 0.0,
                snippet=SearchResponseParser.snippet(hit),
            )
            for hit in resp["hits"]["hits"]
        ]

    @staticmethod
    def candidates(resp: dict[str, Any]) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                document_id=hit["_source"]["document_id"],
                score=hit.get("_score") or 0.0,
                snippet=SearchResponseParser.snippet(hit),
            )
            for hit in resp["hits"]["hits"]
        ]

    @staticmethod
    def snippet(hit: dict[str, Any]) -> str | None:
        inner_snippet = SearchResponseParser.inner_hit_snippet(hit)
        if inner_snippet:
            return inner_snippet

        return SearchResponseParser.content_snippet(hit)

    @staticmethod
    def inner_hit_snippet(hit: dict[str, Any]) -> str | None:
        for chunk in SearchResponseParser.inner_hits(hit):
            snippet = SearchResponseParser.highlight_snippet(chunk)
            if snippet:
                return snippet

        return None

    @staticmethod
    def inner_hits(hit: dict[str, Any]) -> list[dict[str, Any]]:
        inner_hits = hit.get("inner_hits") or {}
        best_chunks = inner_hits.get("best_chunks") or {}
        chunks = best_chunks.get("hits", {}).get("hits", [])
        return chunks if isinstance(chunks, list) else []

    @staticmethod
    def highlight_snippet(hit: dict[str, Any]) -> str | None:
        highlight = hit.get("highlight") or {}
        fragments = highlight.get("content")

        if isinstance(fragments, list) and fragments:
            return " ... ".join(str(f) for f in fragments)

        return None

    @staticmethod
    def content_snippet(hit: dict[str, Any]) -> str | None:
        content = SearchResponseParser.content(hit)
        if not isinstance(content, str):
            return None

        content = " ".join(content.split())

        if len(content) <= SNIPPET_MAX_CHARS:
            return content

        return f"{content[:SNIPPET_MAX_CHARS].rstrip()}..."

    @staticmethod
    def content(hit: dict[str, Any]) -> str | None:
        content = (hit.get("_source") or {}).get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def total(resp: dict[str, Any]) -> int:
        aggs = resp.get("aggregations") or {}
        cardinality = aggs.get("unique_document_count")

        if isinstance(cardinality, dict) and cardinality.get("value") is not None:
            return int(cardinality["value"])

        total = resp["hits"]["total"]

        if isinstance(total, dict):
            return int(total.get("value", 0))

        return int(total)


class RRFMerger:
    @staticmethod
    def rrf_score(rank: int) -> float:
        return 1.0 / (SEARCH_RRF_K + rank)

    def merge(
        self,
        bm25_hits: list[SearchCandidate],
        vector_hits: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        merged: dict[str, SearchCandidate] = {}

        self.add_bm25_hits(merged, bm25_hits)
        self.add_vector_hits(merged, vector_hits)
        self.calculate_scores(merged)

        return sorted(
            merged.values(),
            key=lambda item: (
                -item.score,
                item.bm25_rank or 10**9,
                item.vector_rank or 10**9,
            ),
        )

    def add_bm25_hits(
        self,
        merged: dict[str, SearchCandidate],
        hits: list[SearchCandidate],
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            candidate = self.get_or_create(merged, hit)

            candidate.bm25_rank = rank
            candidate.bm25_score = hit.score

            if not candidate.snippet and hit.snippet:
                candidate.snippet = hit.snippet

    def add_vector_hits(
        self,
        merged: dict[str, SearchCandidate],
        hits: list[SearchCandidate],
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            candidate = self.get_or_create(merged, hit)

            candidate.vector_rank = rank
            candidate.vector_score = hit.score

            if not candidate.snippet and hit.snippet:
                candidate.snippet = hit.snippet

    @staticmethod
    def get_or_create(
        merged: dict[str, SearchCandidate],
        hit: SearchCandidate,
    ) -> SearchCandidate:
        candidate = merged.get(hit.document_id)

        if candidate is None:
            candidate = SearchCandidate(
                document_id=hit.document_id,
                score=0.0,
                snippet=hit.snippet,
            )
            merged[hit.document_id] = candidate

        return candidate

    def calculate_scores(self, merged: dict[str, SearchCandidate]) -> None:
        for hit in merged.values():
            score = 0.0

            if hit.bm25_rank is not None:
                score += self.rrf_score(hit.bm25_rank)

            if hit.vector_rank is not None:
                score += self.rrf_score(hit.vector_rank)

            hit.score = score


class SearchService:
    def __init__(self, embedding: EmbeddingService) -> None:
        self.builder = SearchQueryBuilder(embedding)
        self.parser = SearchResponseParser()
        self.rrf_merger = RRFMerger()

    def search(
        self,
        query: str,
        owner_id: int | None = None,
        page: int = 1,
        page_size: int = 10,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> tuple[list[SearchHit], int]:
        try:
            page, page_size = self.normalize_pagination(page, page_size)

            if mode == SearchMode.HYBRID:
                return self.hybrid_search(
                    query=query,
                    owner_id=owner_id,
                    page=page,
                    page_size=page_size,
                )

            return self.single_mode_search(
                query=query,
                owner_id=owner_id,
                page=page,
                page_size=page_size,
                mode=mode,
            )

        except Exception as exc:
            logger.exception("Search failed")
            raise HTTPException(status_code=502, detail=f"Elasticsearch error: {exc}")

    @staticmethod
    def normalize_pagination(page: int, page_size: int) -> tuple[int, int]:
        if page < 1:
            page = 1

        if page_size < 1:
            page_size = 10

        return page, page_size

    def single_mode_search(
        self,
        query: str,
        owner_id: int | None,
        page: int,
        page_size: int,
        mode: SearchMode,
    ) -> tuple[list[SearchHit], int]:
        es = es_client.get_client()

        body = self.builder.build(
            query=query,
            page=page,
            page_size=page_size,
            owner_id=owner_id,
            mode=mode,
        )

        resp = es.search(index=ELASTICSEARCH_INDEX, body=body)
        total = self.parser.total(resp)

        if not resp["hits"]["hits"]:
            return [], total

        return self.parser.hits(resp), total

    def hybrid_search(
        self,
        query: str,
        owner_id: int | None,
        page: int,
        page_size: int,
    ) -> tuple[list[SearchHit], int]:
        es = es_client.get_client()

        bm25_body = self.builder.build_full_text_candidate(
            query=query,
            candidate_size=SEARCH_MAX_CANDIDATE_SIZE,
            owner_id=owner_id,
        )

        semantic_body = self.builder.build_semantic_candidate(
            query=query,
            candidate_size=SEARCH_MAX_CANDIDATE_SIZE,
            owner_id=owner_id,
        )

        bm25_resp = es.search(index=ELASTICSEARCH_INDEX, body=bm25_body)
        semantic_resp = es.search(index=ELASTICSEARCH_INDEX, body=semantic_body)

        bm25_hits = self.parser.candidates(bm25_resp)
        semantic_hits = self.parser.candidates(semantic_resp)

        merged_hits = self.rrf_merger.merge(
            bm25_hits=bm25_hits,
            vector_hits=semantic_hits,
        )

        page_hits = self.paginate(
            hits=merged_hits,
            page=page,
            page_size=page_size,
        )

        return self.to_search_hits(page_hits), len(merged_hits)

    @staticmethod
    def paginate(
        hits: list[SearchCandidate],
        page: int,
        page_size: int,
    ) -> list[SearchCandidate]:
        start = (page - 1) * page_size
        end = start + page_size
        return hits[start:end]

    @staticmethod
    def to_search_hits(candidates: list[SearchCandidate]) -> list[SearchHit]:
        return [
            SearchHit(
                document_id=hit.document_id,
                score=hit.score,
                snippet=hit.snippet,
            )
            for hit in candidates
        ]


search_service = SearchService(embedding_service)
