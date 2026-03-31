from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str
    language: str = Field(default="vi", description="Language code: vi, en, ...")


class SummarizeResponse(BaseModel):
    summary_text: str
    model_version: str
    prompt_version: str
