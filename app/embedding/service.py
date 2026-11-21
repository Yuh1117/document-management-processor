import logging
import time

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from app.core.config import (
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from app.embedding.cache import QueryEmbeddingCache
from app.embedding.constants import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, cache: QueryEmbeddingCache | None = None) -> None:
        self.model: SentenceTransformer = self.load_model()
        self.cache = cache if cache is not None else QueryEmbeddingCache()
        is_e5 = "e5" in (SENTENCE_TRANSFORMER_MODEL_NAME or "").lower()
        self.query_prefix = E5_QUERY_PREFIX if is_e5 else ""
        self.passage_prefix = E5_PASSAGE_PREFIX if is_e5 else ""

    def encode_text(self, text: str) -> list[float]:
        return self.encode(text, self.passage_prefix).tolist()

    def encode_many(self, texts: list[str]) -> np.ndarray:
        try:
            return self.model.encode([self.passage_prefix + t for t in texts])
        except Exception as e:
            logger.error("Embedding batch encode failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    def encode_query(self, text: str) -> list[float]:
        started = time.perf_counter()

        cached = self.cache.get(text)
        if cached is not None:
            self.log_outcome("HIT", text, started)
            return cached

        vector = self.encode(text, self.query_prefix)
        self.cache.set(text, vector)
        self.log_outcome("MISS", text, started)

        return vector.tolist()

    @staticmethod
    def log_outcome(outcome: str, text: str, started: float) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        preview = text if len(text) <= 60 else f"{text[:60]}..."
        logger.info(
            "Query embedding cache %s in %.1fms query=%r", outcome, elapsed_ms, preview
        )

    def encode(self, text: str, prefix: str = "") -> np.ndarray:
        try:
            return self.model.encode(prefix + text)
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
