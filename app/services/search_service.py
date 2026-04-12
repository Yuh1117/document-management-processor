import logging
import json
from typing import Any, Dict, List, Tuple
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


class SearchService:
    def __init__(self, embedding: EmbeddingService) -> None:
        self._embedding = embedding

    def search(
        self,
        query: str,
        owner_id: int | None = None,
        folder_id: int | None = None,
        page: int = 1,
        page_size: int = 10,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> Tuple[List[SearchHit], int]:
        try:
            es = es_client.get_client()
            filters = self._build_filters(owner_id, folder_id)
            body = self._build_search_body(query, page, page_size, filters, mode)
            resp = es.search(index=ELASTICSEARCH_INDEX, body=body)
            return self._to_hits(resp), self._total_hits(resp)
        except Exception as e:
            logger.error("Search failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Elasticsearch error: {str(e)}")

    @staticmethod
    def _build_filters(owner_id: int | None, folder_id: int | None) -> list[dict]:
        filters: list[dict] = []
        if owner_id is not None:
            filters.append({"term": {"owner_id": str(owner_id)}})
        if folder_id is not None:
            filters.append({"term": {"folder_id": str(folder_id)}})
        return filters

    def _build_search_body(
        self,
        query: str,
        page: int,
        page_size: int,
        filters: list[dict],
        mode: SearchMode,
    ) -> Dict[str, Any]:
        body = self._load_query_template(self._template_path_for_mode(mode))
        self._apply_pagination(body, page, page_size)

        if mode == SearchMode.FULL_TEXT:
            return self._fill_full_text_query(body, query, filters)

        offset = (page - 1) * page_size
        knn_k = offset + page_size
        query_vector = self._embedding.encode_text(query)
        return self._fill_semantic_or_hybrid_query(
            body=body,
            query=query,
            query_vector=query_vector,
            knn_k=knn_k,
            filters=filters,
            mode=mode,
        )

    def _load_query_template(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _template_path_for_mode(mode: SearchMode) -> str:
        if mode == SearchMode.FULL_TEXT:
            return FULL_TEXT_QUERY_FILE
        if mode == SearchMode.SEMANTIC:
            return SEMANTIC_QUERY_FILE
        return HYBRID_QUERY_FILE

    @staticmethod
    def _apply_pagination(body: Dict[str, Any], page: int, page_size: int) -> None:
        offset = (page - 1) * page_size
        body["from"] = offset
        body["size"] = page_size

    def _fill_full_text_query(
        self, body: Dict[str, Any], query: str, filters: list[dict]
    ) -> Dict[str, Any]:
        must0 = body["query"]["bool"]["must"][0]
        self._set_text_query(must0, query, context="full_text")
        self._apply_bool_filters(body, filters)
        return body

    def _fill_semantic_or_hybrid_query(
        self,
        body: Dict[str, Any],
        query: str,
        query_vector: list[float],
        knn_k: int,
        filters: list[dict],
        mode: SearchMode,
    ) -> Dict[str, Any]:
        self._apply_knn(body, query_vector, knn_k, filters)

        if mode == SearchMode.HYBRID:
            self._apply_bool_filters(body, filters)
            should0 = body["query"]["bool"]["should"][0]
            self._set_text_query(should0, query, context="hybrid")

        return body

    @staticmethod
    def _set_text_query(clause: Dict[str, Any], query: str, context: str) -> None:
        if "multi_match" in clause:
            clause["multi_match"]["query"] = query
            return

        if "match" in clause:
            match_body = clause["match"]
            content_value = match_body.get("content")
            if isinstance(content_value, dict):
                content_value["query"] = query
            else:
                match_body["content"] = query
            return

        raise KeyError(f"Unsupported {context} template: expected match or multi_match")

    @staticmethod
    def _apply_bool_filters(body: Dict[str, Any], filters: list[dict]) -> None:
        body["query"]["bool"]["filter"] = filters

    @staticmethod
    def _apply_knn(
        body: Dict[str, Any],
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
                raise KeyError(
                    "kNN template missing knn.filter (expected knn.filter.bool.filter)."
                )
            return

        if "bool" not in knn["filter"] or "filter" not in knn["filter"]["bool"]:
            raise KeyError(
                "kNN template has unsupported knn.filter shape (expected knn.filter.bool.filter)."
            )
        knn["filter"]["bool"]["filter"] = filters

    @staticmethod
    def _to_hits(resp: Dict[str, Any]) -> List[SearchHit]:
        return [
            SearchHit(
                document_id=hit["_source"]["document_id"],
                score=hit["_score"],
            )
            for hit in resp["hits"]["hits"]
        ]

    @staticmethod
    def _total_hits(resp: Dict[str, Any]) -> int:
        aggs = resp.get("aggregations") or {}
        card = aggs.get("unique_document_count")
        if isinstance(card, dict) and card.get("value") is not None:
            return int(card["value"])
        total = resp["hits"]["total"]
        if isinstance(total, dict):
            return int(total.get("value", 0))
        return int(total)


search_service = SearchService(embedding_service)
