from fastapi import APIRouter, Depends

from app.deps import get_summarize_service
from app.models.summarize import (
    ModelInfo,
    ModelsListResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.summarize_service import SummarizeService

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(
    req: SummarizeRequest,
    svc: SummarizeService = Depends(get_summarize_service),
):
    result = svc.summarize(req.text, req.language)
    return SummarizeResponse(**result)


@router.get("/models", response_model=ModelsListResponse)
def list_models(svc: SummarizeService = Depends(get_summarize_service)):
    return ModelsListResponse(models=[ModelInfo(**m) for m in svc.list_models()])
