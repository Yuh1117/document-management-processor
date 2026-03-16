from enum import Enum
from typing import List
from pydantic import BaseModel


class SearchMode(str, Enum):
    FULL_TEXT = "full_text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    query: str
    owner_id: int | None = None
    folder_id: int | None = None
    top_k: int = 5
    mode: SearchMode = SearchMode.SEMANTIC


class SearchHit(BaseModel):
    document_id: str
    score: float


class SearchResponse(BaseModel):
    hits: List[SearchHit]
