from fastapi import APIRouter, Depends

from app.deps import get_search_service, verify_api_key
from app.models.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    hits, total = search_service.search(
        query=req.query,
        owner_id=req.owner_id,
        page=req.page,
        page_size=req.page_size,
        mode=req.mode,
        mime_pattern=req.mime_pattern,
        size_min=req.size_min,
        size_max=req.size_max,
        exclude_document_ids=req.exclude_document_ids,
    )
    return SearchResponse(
        hits=hits,
        total=total,
        page=req.page,
        page_size=req.page_size,
    )
