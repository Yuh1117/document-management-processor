from pydantic import BaseModel


class ReindexRequest(BaseModel):
    text: str
    owner_id: str | None = None
    name: str | None = None


class RenameRequest(BaseModel):
    name: str


class ReindexResponse(BaseModel):
    success: bool
    chunks_indexed: int
    message: str | None = None
