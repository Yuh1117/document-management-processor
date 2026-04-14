import logging
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from app.core.config import SENTENCE_TRANSFORMER_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model: SentenceTransformer = self.load_model()

    def encode_text(self, text: str) -> list[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error("Embedding encode failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    def load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
        except Exception:
            logger.exception(
                "Failed to load SentenceTransformer model=%s",
                SENTENCE_TRANSFORMER_MODEL_NAME,
            )
            raise


embedding_service = EmbeddingService()
