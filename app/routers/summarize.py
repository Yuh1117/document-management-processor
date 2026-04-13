from fastapi import APIRouter, Depends

from app.deps import get_summarize_service
from app.models.summarize import (
    ModelInfo,
    ModelsListResponse,
    ReloadModelResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.summarize_service import SummarizeService

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    req: SummarizeRequest,
    summarize_service: SummarizeService = Depends(get_summarize_service),
):
    result = summarize_service.summarize(req.text, req.language)
    return SummarizeResponse(**result)


@router.post("/models/reload", response_model=ReloadModelResponse)
def reload_model(summarize_service: SummarizeService = Depends(get_summarize_service)):
    result = summarize_service.reload_model()
    return ReloadModelResponse(**result)


@router.get("/models", response_model=ModelsListResponse)
def list_models(summarize_service: SummarizeService = Depends(get_summarize_service)):
    result = summarize_service.list_models()
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in result],
    )
