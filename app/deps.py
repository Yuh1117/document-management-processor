from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.search_service import SearchService, search_service


def get_embedding_service() -> EmbeddingService:
    return embedding_service


def get_search_service() -> SearchService:
    return search_service
