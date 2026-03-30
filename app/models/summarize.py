from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary_text: str
    model_version: str
    prompt_version: str
