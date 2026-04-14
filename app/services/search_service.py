import logging
import json
from typing import Any
from fastapi import HTTPException
from app.core.config import (
    ELASTICSEARCH_INDEX,
    FULL_TEXT_QUERY_FILE,
    HYBRID_QUERY_FILE,
    SEMANTIC_QUERY_FILE,
)
from app.core.es import es_client
from app.models.search import SearchMode, SearchHit
from app.services.embedding_service import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)


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
        return self.apply_vector(
            body, query, query_vector, offset + page_size, filters, mode
        )

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
        clause = body["query"]["bool"]["must"][0]
        self.set_text_clause(clause, query, context="full_text")
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
        self.apply_knn(body, query_vector, knn_k, filters)

        if mode == SearchMode.HYBRID:
            body["query"]["bool"]["filter"] = filters
            clause = body["query"]["bool"]["should"][0]
            self.set_text_clause(clause, query, context="hybrid")

        return body

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
    def set_text_clause(clause: dict[str, Any], query: str, context: str) -> None:
        if "multi_match" in clause:
            clause["multi_match"]["query"] = query
            return

        if "match" in clause:
            value = clause["match"].get("content")
            if isinstance(value, dict):
                value["query"] = query
            else:
                clause["match"]["content"] = query
            return

        raise KeyError(f"Unsupported {context} template: expected match or multi_match")


class SearchResponseParser:
    @staticmethod
    def hits(resp: dict[str, Any]) -> list[SearchHit]:
        return [
            SearchHit(
                document_id=hit["_source"]["document_id"],
                score=hit["_score"],
            )
            for hit in resp["hits"]["hits"]
        ]

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
            return self.parser.hits(resp), self.parser.total(resp)
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Elasticsearch error: {exc}")


search_service = SearchService(embedding_service)
