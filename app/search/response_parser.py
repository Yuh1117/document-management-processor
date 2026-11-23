from typing import Any

from app.search.constants import SNIPPET_MAX_CHARS
from app.search.schemas import SearchHit
from app.search.types import SearchCandidate


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
