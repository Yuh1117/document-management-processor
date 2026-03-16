from typing import Any, Dict, List
from app.core.config import (
    ELASTICSEARCH_INDEX,
    FULL_TEXT_QUERY_FILE,
    HYBRID_QUERY_FILE,
    SEMANTIC_QUERY_FILE,
)
from app.core.es import get_es
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

    def search(
        self,
        query: str,
        owner_id: int | None = None,
        folder_id: int | None = None,
        top_k: int = 5,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> List[SearchHit]:
        es = get_es()
        filters = self._build_filters(owner_id, folder_id)

        if mode == SearchMode.FULL_TEXT:
            body = self._load_query_template(FULL_TEXT_QUERY_FILE)
            body["query"]["bool"]["must"][0]["multi_match"]["query"] = query
            body["query"]["bool"]["filter"] = filters
        else:
            query_vector = self._embedding.encode_text(query)
            num_candidates = top_k * 5

            if mode == SearchMode.SEMANTIC:
                body = self._load_query_template(SEMANTIC_QUERY_FILE)
            else:
                body = self._load_query_template(HYBRID_QUERY_FILE)

            body["knn"]["query_vector"] = query_vector
            body["knn"]["k"] = top_k
            body["knn"]["num_candidates"] = num_candidates
            body["query"]["bool"]["filter"] = filters
            if mode == SearchMode.HYBRID:
                body["query"]["bool"]["should"][0]["multi_match"]["query"] = query

        resp = es.search(index=ELASTICSEARCH_INDEX, body=body)

        hits = [
            SearchHit(
                document_id=hit["_source"]["document_id"],
                score=hit["_score"],
            )
            for hit in resp["hits"]["hits"]
        ]

        return hits


search_service = SearchService(embedding_service)
