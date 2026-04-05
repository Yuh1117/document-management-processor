from typing import List

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str
    language: str = Field(default="vi", description="Language code: vi, en, ...")


class SummarizeResponse(BaseModel):
    summary_text: str
    model_name: str
    prompt_version: str


class ModelInfo(BaseModel):
    id: str
    provider: str
    is_default: bool
    prompt_version: str


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
