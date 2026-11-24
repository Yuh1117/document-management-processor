from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_service, verify_api_key
from app.rag.schemas import RagRequest
from app.rag.service import RagService

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/rag/ask")
def ask(req: RagRequest, rag: RagService = Depends(get_rag_service)):
    return StreamingResponse(
        rag.stream(
            req.question,
            req.owner_id,
            req.language,
            req.exclude_document_ids,
            req.history,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
