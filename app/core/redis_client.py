import logging

import redis

from app.core.config import (
    REDIS_CONNECT_TIMEOUT,
    REDIS_SOCKET_TIMEOUT,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self, url: str | None = REDIS_URL) -> None:
        self.url = url
        self.client: redis.Redis | None = None
        self.unavailable_logged = False
        self.disabled_logged = False

    def get_client(self) -> redis.Redis | None:
        if not self.url:
            self.log_disabled()
            return None

        if self.client is None:
            try:
                self.client = redis.Redis.from_url(
                    self.url,
                    socket_timeout=REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                )
            except Exception as e:
                self.log_unavailable(e)
                return None

        return self.client

    def log_disabled(self) -> None:
        if not self.disabled_logged:
            logger.info("REDIS_URL is not set, caching is disabled")
            self.disabled_logged = True

    def log_unavailable(self, error: Exception) -> None:
        if not self.unavailable_logged:
            logger.warning("Redis unavailable, continuing without cache: %s", error)
            self.unavailable_logged = True
        else:
            logger.debug("Redis still unavailable: %s", error)


redis_client = RedisClient()
