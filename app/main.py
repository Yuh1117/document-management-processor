from fastapi import Depends, HTTPException
from app import app
from app.deps import get_embedding_service, get_search_service, get_summarize_service
from app.models.embeddings import EmbedRequest, EmbedResponse
from app.models.search import SearchRequest, SearchResponse
from app.models.summarize import SummarizeRequest, SummarizeResponse
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.summarize_service import SummarizeService


@app.post("/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest,
    embedding: EmbeddingService = Depends(get_embedding_service),
):
    vector = embedding.encode_text(req.text)
    return EmbedResponse(embedding=vector)


@app.post("/search", response_model=SearchResponse)
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


@app.post("/summarize", response_model=SummarizeResponse)
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
