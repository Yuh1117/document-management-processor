from typing import List, Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str
    language: str = Field(default="vi", description="Language code: vi, en, ...")


class SummarizeResponse(BaseModel):
    summary_text: str
    model_name: str
    prompt_version: str


class ModelInfo(BaseModel):
    version: str
    model_name: Optional[str] = None
    is_active: bool = False
    created_at: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]


class ReloadModelResponse(BaseModel):
    previous_model: str
    current_model: str
    mlflow_model_uri: str
    status: str
