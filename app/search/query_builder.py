import copy
import json
from typing import Any

from app.embedding.service import EmbeddingService
from app.search.constants import (
    FULL_TEXT_QUERY_FILE,
    SEARCH_CANDIDATE_MULTIPLIER,
    SEARCH_DEFAULT_CANDIDATE_SIZE,
    SEARCH_KNN_POOL_SIZE,
    SEARCH_MAX_CANDIDATE_SIZE,
    SEARCH_SEMANTIC_MIN_SCORE,
    SEMANTIC_QUERY_FILE,
)
from app.search.schemas import SearchMode
from app.search.types import SearchFilters

TEMPLATE_PATHS = {
    SearchMode.FULL_TEXT: FULL_TEXT_QUERY_FILE,
    SearchMode.SEMANTIC: SEMANTIC_QUERY_FILE,
}


def like_to_wildcard(pattern: str) -> str:
    out: list[str] = []
    for ch in pattern:
        if ch == "%":
            out.append("*")
        elif ch == "_":
            out.append("?")
        elif ch in "*?\\":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def text_clauses(body: dict[str, Any]) -> list[dict[str, Any]]:
    bool_query = body.get("query", {}).get("bool", {})
    clauses = bool_query.get("must", []) + bool_query.get("should", [])

    return [c for c in clauses if "multi_match" in c or "match" in c]


class QueryTemplates:
    def __init__(self, paths: dict[SearchMode, str]) -> None:
        self.templates: dict[SearchMode, dict[str, Any]] = {}

        for mode, path in paths.items():
            body = self.read(path)
            self.validate(mode, body, path)
            self.templates[mode] = body

    def get(self, mode: SearchMode) -> dict[str, Any]:
        template = self.templates.get(mode)

        if template is None:
            raise ValueError(f"Unsupported search mode: {mode}")

        return copy.deepcopy(template)

    @staticmethod
    def read(path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate(mode: SearchMode, body: dict[str, Any], path: str) -> None:
        if mode == SearchMode.FULL_TEXT and not text_clauses(body):
            raise ValueError(f"{path}: template has no match/multi_match clause")

        if mode != SearchMode.SEMANTIC:
            return

        knn = body.get("knn")
        if not isinstance(knn, dict):
            raise ValueError(f"{path}: template has no knn block")

        if "filter" not in knn.get("filter", {}).get("bool", {}):
            raise ValueError(f"{path}: template needs knn.filter.bool.filter")


class SearchQueryBuilder:
    def __init__(self, embedding: EmbeddingService, templates: QueryTemplates) -> None:
        self.embedding = embedding
        self.templates = templates

    def build(
        self,
        query: str,
        page: int,
        page_size: int,
        owner_id: str | None,
        mode: SearchMode,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        return self.build_body(
            query=query,
            mode=mode,
            owner_id=owner_id,
            filters=filters,
            offset=(page - 1) * page_size,
            size=page_size,
            knn_k=self.candidate_size(
                page=page,
                page_size=page_size,
                minimum=SEARCH_KNN_POOL_SIZE,
            ),
        )

    def build_candidate(
        self,
        query: str,
        candidate_size: int,
        owner_id: str | None,
        mode: SearchMode,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        return self.build_body(
            query=query,
            mode=mode,
            owner_id=owner_id,
            filters=filters,
            offset=0,
            size=candidate_size,
            knn_k=candidate_size,
        )

    def build_body(
        self,
        query: str,
        mode: SearchMode,
        owner_id: str | None,
        offset: int,
        size: int,
        knn_k: int,
        filters: SearchFilters | None = None,
    ) -> dict[str, Any]:
        body = self.templates.get(mode)
        filters = self.owner_filter(owner_id) + self.metadata_filter(filters)

        body["from"] = offset
        body["size"] = size

        if mode == SearchMode.FULL_TEXT:
            return self.apply_full_text(body, query, filters)

        return self.apply_semantic(body, query, knn_k, filters)

    @staticmethod
    def owner_filter(owner_id: str | None) -> list[dict[str, Any]]:
        if owner_id is None:
            return []
        return [{"term": {"owner_id": str(owner_id)}}]

    @staticmethod
    def metadata_filter(filters: SearchFilters | None) -> list[dict[str, Any]]:
        if filters is None:
            return []

        clauses: list[dict[str, Any]] = []

        if filters.mime_pattern:
            clauses.append(
                {"wildcard": {"mime_type": like_to_wildcard(filters.mime_pattern)}}
            )

        size_range: dict[str, int] = {}
        if filters.size_min is not None:
            size_range["gte"] = filters.size_min
        if filters.size_max is not None:
            size_range["lte"] = filters.size_max
        if size_range:
            clauses.append({"range": {"file_size": size_range}})

        if filters.exclude_document_ids:
            clauses.append(
                {
                    "bool": {
                        "must_not": [
                            {"terms": {"document_id": filters.exclude_document_ids}}
                        ]
                    }
                }
            )

        return clauses

    def apply_full_text(
        self,
        body: dict[str, Any],
        query: str,
        filters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.set_text_clauses(body, query)
        body["query"]["bool"]["filter"] = filters
        return body

    def apply_semantic(
        self,
        body: dict[str, Any],
        query: str,
        knn_k: int,
        filters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if SEARCH_SEMANTIC_MIN_SCORE is not None:
            body["min_score"] = SEARCH_SEMANTIC_MIN_SCORE

        self.apply_knn(body, self.embedding.encode_query(query), knn_k, filters)
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
        # Shape was checked by QueryTemplates.validate at startup.
        knn = body["knn"]
        knn["query_vector"] = query_vector
        knn["k"] = knn_k
        knn["num_candidates"] = min(knn_k * 5, 10_000)
        knn["filter"]["bool"]["filter"] = filters

    @staticmethod
    def set_text_clauses(body: dict[str, Any], query: str) -> None:
        for clause in text_clauses(body):
            if "multi_match" in clause:
                clause["multi_match"]["query"] = query

            for field_name, value in clause.get("match", {}).items():
                if isinstance(value, dict):
                    value["query"] = query
                else:
                    clause["match"][field_name] = query
