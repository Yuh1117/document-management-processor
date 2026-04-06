from fastapi import APIRouter, Depends

from app.deps import get_search_service
from app.models.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    hits, total = search_service.search(
        query=req.query,
        owner_id=req.owner_id,
        folder_id=req.folder_id,
        page=req.page,
        page_size=req.page_size,
        mode=req.mode,
    )
    return SearchResponse(
        hits=hits,
        total=total,
        page=req.page,
        page_size=req.page_size,
    )
