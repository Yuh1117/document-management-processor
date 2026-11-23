from app.search.constants import SEARCH_RRF_K
from app.search.types import BM25_SOURCE, UNRANKED, VECTOR_SOURCE, SearchCandidate


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
