from fastapi import Depends
from app import app
from app.core.index_bootstrap import ensure_documents_index_exists
from app.deps import get_embedding_service, get_search_service
from app.models.embeddings import EmbedRequest, EmbedResponse
from app.models.search import SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService

ensure_documents_index_exists()


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
    hits = search_service.search(
        query=req.query,
        owner_id=req.owner_id,
        folder_id=req.folder_id,
        top_k=req.top_k,
        mode=req.mode,
    )

    return SearchResponse(hits=hits)
