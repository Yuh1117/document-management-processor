from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RagRequest(BaseModel):
    question: str
    owner_id: str
    language: str = Field(default="vi", description="Answer language code: vi, en")
    exclude_document_ids: list[str] = Field(
        default_factory=list,
        description="Documents that must not be used (e.g. in the owner's trash)",
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Earlier messages of the conversation, oldest first",
    )
