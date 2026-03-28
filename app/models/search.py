from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    FULL_TEXT = "full_text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    query: str
    owner_id: int | None = None
    folder_id: int | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    mode: SearchMode = SearchMode.SEMANTIC


class SearchHit(BaseModel):
    document_id: str
    score: float


class SearchResponse(BaseModel):
    hits: List[SearchHit]
    total: int
    page: int
    page_size: int
