from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.deps import get_rag_service, verify_api_key
from app.models.rag import RagRequest
from app.services.rag_service import RagService

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/rag/ask")
def ask(req: RagRequest, rag: RagService = Depends(get_rag_service)):
    return StreamingResponse(
        rag.stream(req.question, req.owner_id, req.language, req.exclude_document_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
