from fastapi import Header, HTTPException, status

from app.core.config import PROCESSOR_API_KEY
from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.search_service import SearchService, search_service
from app.services.summarize_service import SummarizeService, summarize_service


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if not PROCESSOR_API_KEY or x_api_key != PROCESSOR_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


def get_embedding_service() -> EmbeddingService:
    return embedding_service


def get_search_service() -> SearchService:
    return search_service


def get_summarize_service() -> SummarizeService:
    return summarize_service
