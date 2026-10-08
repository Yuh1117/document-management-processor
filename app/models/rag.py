from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    question: str
    owner_id: str
    language: str = Field(default="vi", description="Answer language code: vi, en")
    exclude_document_ids: list[str] = Field(
        default_factory=list,
        description="Documents that must not be used (e.g. in the owner's trash)",
    )
