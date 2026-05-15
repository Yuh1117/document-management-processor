import logging
import json
from typing import Any
from fastapi import HTTPException
from app.core.config import (
    ELASTICSEARCH_INDEX,
    FULL_TEXT_QUERY_FILE,
    HYBRID_QUERY_FILE,
    SEMANTIC_QUERY_FILE,
    SEARCH_HYBRID_MIN_SCORE,
    SEARCH_KNN_POOL_SIZE,
    SEARCH_SEMANTIC_MIN_SCORE,
)
from app.core.es import es_client
from app.models.search import SearchMode, SearchHit
from app.services.embedding_service import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 320


class SearchQueryBuilder:
    TEMPLATE_PATHS = {
        SearchMode.FULL_TEXT: FULL_TEXT_QUERY_FILE,
        SearchMode.SEMANTIC: SEMANTIC_QUERY_FILE,
        SearchMode.HYBRID: HYBRID_QUERY_FILE,
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
        offset = (page - 1) * page_size
        body["from"] = offset
        body["size"] = page_size

        if mode == SearchMode.FULL_TEXT:
            return self.apply_full_text(body, query, filters)

        query_vector = self.embedding.encode_text(query)
        knn_k = SEARCH_KNN_POOL_SIZE
        return self.apply_vector(body, query, query_vector, knn_k, filters, mode)

    def load_template(self, mode: SearchMode) -> dict[str, Any]:
        path = self.TEMPLATE_PATHS[mode]
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def owner_filter(owner_id: int | None) -> list[dict]:
        if owner_id is None:
            return []
        return [{"term": {"owner_id": str(owner_id)}}]

    def apply_full_text(
        self, body: dict[str, Any], query: str, filters: list[dict]
    ) -> dict[str, Any]:
        self.set_text_clauses(body, query, context="full_text")
        body["query"]["bool"]["filter"] = filters
        return body

    def apply_vector(
        self,
        body: dict[str, Any],
        query: str,
        query_vector: list[float],
        knn_k: int,
        filters: list[dict],
        mode: SearchMode,
    ) -> dict[str, Any]:
        min_score = self.min_score(mode)
        if min_score is not None:
            body["min_score"] = min_score
        self.apply_knn(body, query_vector, knn_k, filters)

        if mode == SearchMode.HYBRID:
            body["query"]["bool"]["filter"] = filters
            self.set_text_clauses(body, query, context="hybrid")

        return body

    @staticmethod
    def min_score(mode: SearchMode) -> float | None:
        if mode == SearchMode.HYBRID:
            return SEARCH_HYBRID_MIN_SCORE
        return SEARCH_SEMANTIC_MIN_SCORE

    @staticmethod
    def apply_knn(
        body: dict[str, Any],
        query_vector: list[float],
        knn_k: int,
        filters: list[dict],
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

        for clause in clauses:
            if "match" in clause:
                for field, value in clause["match"].items():
                    if isinstance(value, dict):
                        value["query"] = query
                    else:
                        clause["match"][field] = query
                    matched = True

        if matched:
            return

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


class SearchService:
    def __init__(self, embedding: EmbeddingService) -> None:
        self.builder = SearchQueryBuilder(embedding)
        self.parser = SearchResponseParser()

    def search(
        self,
        query: str,
        owner_id: int | None = None,
        page: int = 1,
        page_size: int = 10,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> tuple[list[SearchHit], int]:
        try:
            es = es_client.get_client()

            body = self.builder.build(query, page, page_size, owner_id, mode)
            resp = es.search(index=ELASTICSEARCH_INDEX, body=body)
            total = self.parser.total(resp)

            if not resp["hits"]["hits"]:
                return [], total

            return self.parser.hits(resp), total

        except Exception as exc:
            logger.error("Search failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Elasticsearch error: {exc}")


search_service = SearchService(embedding_service)
