import hashlib
import logging
import time

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from app.constants.defaults import QUERY_EMBEDDING_CACHE_PREFIX
from app.core.config import (
    QUERY_EMBEDDING_CACHE_TTL,
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from app.core.redis_client import RedisClient, redis_client

logger = logging.getLogger(__name__)

VECTOR_DTYPE = np.float32


class QueryEmbeddingCache:
    def __init__(
        self,
        client: RedisClient = redis_client,
        model_name: str | None = SENTENCE_TRANSFORMER_MODEL_NAME,
        ttl: int = QUERY_EMBEDDING_CACHE_TTL,
    ) -> None:
        self.redis = client
        self.model_name = model_name or "unknown"
        self.ttl = ttl

    def key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{QUERY_EMBEDDING_CACHE_PREFIX}:{self.model_name}:{digest}"

    def get(self, text: str) -> list[float] | None:
        client = self.redis.get_client()
        if client is None:
            return None

        try:
            raw = client.get(self.key(text))
        except Exception as e:
            self.redis.log_unavailable(e)
            return None

        if not raw:
            return None

        try:
            return np.frombuffer(raw, dtype=VECTOR_DTYPE).tolist()
        except ValueError as e:
            logger.warning("Discarding malformed cached embedding: %s", e)
            return None

    def set(self, text: str, vector: np.ndarray) -> None:
        client = self.redis.get_client()
        if client is None:
            return

        try:
            client.setex(
                self.key(text),
                self.ttl,
                vector.astype(VECTOR_DTYPE).tobytes(),
            )
        except Exception as e:
            self.redis.log_unavailable(e)


class EmbeddingService:
    def __init__(self, cache: QueryEmbeddingCache | None = None) -> None:
        self.model: SentenceTransformer = self.load_model()
        self.cache = cache if cache is not None else QueryEmbeddingCache()

    def encode_text(self, text: str) -> list[float]:
        return self.encode(text).tolist()

    def encode_query(self, text: str) -> list[float]:
        started = time.perf_counter()

        cached = self.cache.get(text)
        if cached is not None:
            self.log_outcome("HIT", text, started)
            return cached

        vector = self.encode(text)
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

    def encode(self, text: str) -> np.ndarray:
        try:
            return self.model.encode(text)
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
