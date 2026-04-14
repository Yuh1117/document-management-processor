from pydantic import BaseModel


class ReindexRequest(BaseModel):
    text: str
    owner_id: int | None = None
    name: str | None = None


class ReindexResponse(BaseModel):
    success: bool
    chunks_indexed: int
    message: str | None = None
