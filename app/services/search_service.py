import copy
import json
import logging
from dataclasses import dataclass, field
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

TEMPLATE_PATHS = {
    SearchMode.FULL_TEXT: FULL_TEXT_QUERY_FILE,
    SearchMode.SEMANTIC: SEMANTIC_QUERY_FILE,
}

BM25_SOURCE = "bm25"
VECTOR_SOURCE = "vector"

UNRANKED = 10**9


@dataclass
class SearchCandidate:
    document_id: str
    score: float
    snippet: str | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


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
    ) -> dict[str, Any]:
        return self.build_body(
            query=query,
            mode=mode,
            owner_id=owner_id,
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
    ) -> dict[str, Any]:
        return self.build_body(
            query=query,
            mode=mode,
            owner_id=owner_id,
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
    ) -> dict[str, Any]:
        body = self.templates.get(mode)
        filters = self.owner_filter(owner_id)

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

        self.add_ranked_hits(merged, BM25_SOURCE, bm25_hits)
        self.add_ranked_hits(merged, VECTOR_SOURCE, vector_hits)
        self.calculate_scores(merged)

        return sorted(merged.values(), key=self.sort_key)

    def add_ranked_hits(
        self,
        merged: dict[str, SearchCandidate],
        source: str,
        hits: list[SearchCandidate],
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            candidate = self.get_or_create(merged, hit)

            candidate.ranks[source] = rank
            candidate.scores[source] = hit.score

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
            hit.score = sum(self.rrf_score(rank) for rank in hit.ranks.values())

    @staticmethod
    def sort_key(candidate: SearchCandidate) -> tuple[float, int, int]:
        return (
            -candidate.score,
            candidate.ranks.get(BM25_SOURCE, UNRANKED),
            candidate.ranks.get(VECTOR_SOURCE, UNRANKED),
        )


class SearchService:
    def __init__(
        self,
        embedding: EmbeddingService,
        templates: QueryTemplates | None = None,
    ) -> None:
        self.builder = SearchQueryBuilder(
            embedding,
            templates if templates is not None else QueryTemplates(TEMPLATE_PATHS),
        )
        self.parser = SearchResponseParser()
        self.rrf_merger = RRFMerger()

    def search(
        self,
        query: str,
        owner_id: str | None = None,
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

        except HTTPException:
            raise
        except ValueError as exc:
            logger.exception("Search failed: unsupported request")
            raise HTTPException(status_code=400, detail=str(exc))
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
        owner_id: str | None,
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
        owner_id: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[SearchHit], int]:
        es = es_client.get_client()

        bodies = [
            self.builder.build_candidate(
                query=query,
                candidate_size=SEARCH_MAX_CANDIDATE_SIZE,
                owner_id=owner_id,
                mode=mode,
            )
            for mode in (SearchMode.FULL_TEXT, SearchMode.SEMANTIC)
        ]

        bm25_resp, semantic_resp = self.multi_search(es, bodies)

        merged_hits = self.rrf_merger.merge(
            bm25_hits=self.parser.candidates(bm25_resp),
            vector_hits=self.parser.candidates(semantic_resp),
        )

        page_hits = self.paginate(
            hits=merged_hits,
            page=page,
            page_size=page_size,
        )

        return self.to_search_hits(page_hits), len(merged_hits)

    @staticmethod
    def multi_search(es, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        searches: list[dict[str, Any]] = []
        for body in bodies:
            searches.append({})
            searches.append(body)

        responses = es.msearch(index=ELASTICSEARCH_INDEX, searches=searches)

        for response in responses["responses"]:
            if "error" in response:
                raise RuntimeError(f"Elasticsearch search failed: {response['error']}")

        return list(responses["responses"])

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
