from dataclasses import dataclass, field

BM25_SOURCE = "bm25"
VECTOR_SOURCE = "vector"

UNRANKED = 10**9


@dataclass
class SearchFilters:
    mime_pattern: str | None = None
    size_min: int | None = None
    size_max: int | None = None
    exclude_document_ids: list[str] | None = None


@dataclass
class SearchCandidate:
    document_id: str
    score: float
    snippet: str | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
