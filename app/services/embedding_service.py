import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import SENTENCE_TRANSFORMER_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer = self._load_model()

    def encode_text(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
        except Exception:
            logger.exception(
                "Failed to load SentenceTransformer model=%s",
                SENTENCE_TRANSFORMER_MODEL_NAME,
            )
            raise


embedding_service = EmbeddingService()
