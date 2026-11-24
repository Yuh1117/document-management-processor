import logging
from typing import Any

from fastapi import HTTPException

from app.core.config import ELASTICSEARCH_INDEX
from app.core.es import es_client
from app.embedding.service import EmbeddingService, embedding_service
from app.search.constants import SEARCH_HYBRID_WINDOW
from app.search.query_builder import TEMPLATE_PATHS, QueryTemplates, SearchQueryBuilder
from app.search.response_parser import SearchResponseParser
from app.search.rrf import RRFMerger
from app.search.schemas import SearchHit, SearchMode
from app.search.types import SearchCandidate, SearchFilters

logger = logging.getLogger(__name__)


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
        mime_pattern: str | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        exclude_document_ids: list[str] | None = None,
    ) -> tuple[list[SearchHit], int]:
        try:
            page, page_size = self.normalize_pagination(page, page_size)
            filters = SearchFilters(
                mime_pattern=mime_pattern,
                size_min=size_min,
                size_max=size_max,
                exclude_document_ids=exclude_document_ids,
            )

            if mode == SearchMode.HYBRID:
                return self.hybrid_search(
                    query=query,
                    owner_id=owner_id,
                    page=page,
                    page_size=page_size,
                    filters=filters,
                )

            return self.single_mode_search(
                query=query,
                owner_id=owner_id,
                page=page,
                page_size=page_size,
                mode=mode,
                filters=filters,
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
        filters: SearchFilters | None = None,
    ) -> tuple[list[SearchHit], int]:
        es = es_client.get_client()

        body = self.builder.build(
            query=query,
            page=page,
            page_size=page_size,
            owner_id=owner_id,
            mode=mode,
            filters=filters,
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
        filters: SearchFilters | None = None,
    ) -> tuple[list[SearchHit], int]:
        es = es_client.get_client()

        # RRF scores by rank only, so every candidate in either list gets a
        # score: a 1000-deep window lets weak tail hits flood the result. The
        # window must not depend on the page, or the merged ranking (and total)
        # shifts between pages and items repeat or go missing.
        bodies = [
            self.builder.build_candidate(
                query=query,
                candidate_size=SEARCH_HYBRID_WINDOW,
                owner_id=owner_id,
                mode=mode,
                filters=filters,
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
