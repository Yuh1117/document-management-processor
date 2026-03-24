from typing import Any, Dict, List
from app.core.config import (
    ELASTICSEARCH_INDEX,
    FULL_TEXT_QUERY_FILE,
    HYBRID_QUERY_FILE,
    SEMANTIC_QUERY_FILE,
)
from app.core.es import es_client
from app.models.search import SearchMode, SearchHit
from app.services.embedding_service import EmbeddingService, embedding_service


class SearchService:
    def __init__(self, embedding: EmbeddingService) -> None:
        self._embedding = embedding

    def _load_query_template(self, path: str) -> Dict[str, Any]:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_filters(self, owner_id: int | None, folder_id: int | None) -> list[dict]:
        filters: list[dict] = []
        if owner_id is not None:
            filters.append({"term": {"owner_id": str(owner_id)}})
        if folder_id is not None:
            filters.append({"term": {"folder_id": str(folder_id)}})
        return filters

    @staticmethod
    def _fill_full_text_query(
        body: Dict[str, Any], query: str, filters: list[dict]
    ) -> Dict[str, Any]:
        body["query"]["bool"]["must"][0]["multi_match"]["query"] = query
        body["query"]["bool"]["filter"] = filters
        return body

    @staticmethod
    def _fill_semantic_or_hybrid_query(
        body: Dict[str, Any],
        query: str,
        query_vector: list[float],
        top_k: int,
        filters: list[dict],
        mode: SearchMode,
    ) -> Dict[str, Any]:
        body["knn"]["query_vector"] = query_vector
        body["knn"]["k"] = top_k
        body["knn"]["num_candidates"] = top_k * 5
        body["query"]["bool"]["filter"] = filters
        if mode == SearchMode.HYBRID:
            body["query"]["bool"]["should"][0]["multi_match"]["query"] = query
        return body

    def _build_search_body(
        self, query: str, top_k: int, filters: list[dict], mode: SearchMode
    ) -> Dict[str, Any]:
        if mode == SearchMode.FULL_TEXT:
            body = self._load_query_template(FULL_TEXT_QUERY_FILE)
            return self._fill_full_text_query(body, query, filters)

        query_vector = self._embedding.encode_text(query)
        template_path = (
            SEMANTIC_QUERY_FILE if mode == SearchMode.SEMANTIC else HYBRID_QUERY_FILE
        )
        body = self._load_query_template(template_path)
        return self._fill_semantic_or_hybrid_query(
            body=body,
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            mode=mode,
        )

    @staticmethod
    def _to_hits(resp: Dict[str, Any]) -> List[SearchHit]:
        return [
            SearchHit(
                document_id=hit["_source"]["document_id"],
                score=hit["_score"],
            )
            for hit in resp["hits"]["hits"]
        ]

    def search(
        self,
        query: str,
        owner_id: int | None = None,
        folder_id: int | None = None,
        top_k: int = 5,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> List[SearchHit]:
        es = es_client.get_client()
        filters = self._build_filters(owner_id, folder_id)
        body = self._build_search_body(query, top_k, filters, mode)
        resp = es.search(index=ELASTICSEARCH_INDEX, body=body)
        return self._to_hits(resp)


search_service = SearchService(embedding_service)
