from fastapi import APIRouter, Depends, HTTPException

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
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    try:
        result = svc.summarize(req.text, req.language)
        return SummarizeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")


@router.get("/models", response_model=ModelsListResponse)
def list_models(svc: SummarizeService = Depends(get_summarize_service)):
    return ModelsListResponse(models=[ModelInfo(**m) for m in svc.list_models()])
