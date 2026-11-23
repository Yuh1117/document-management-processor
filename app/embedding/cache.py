import hashlib
import logging

import numpy as np

from app.core.config import (
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from app.core.redis_client import RedisClient, redis_client
from app.embedding.constants import (
    QUERY_EMBEDDING_CACHE_PREFIX,
    QUERY_EMBEDDING_CACHE_TTL,
    VECTOR_DTYPE,
)

logger = logging.getLogger(__name__)


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
