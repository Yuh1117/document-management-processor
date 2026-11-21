"""Pick the LLM client for a provider ("ollama" or "openai")."""

import threading

from app.constants.defaults import LLM_PROVIDER_OLLAMA, LLM_PROVIDER_OPENAI
from app.utils.llm.base import LLMClient

_clients: dict[str, LLMClient] = {}
_lock = threading.Lock()


def get_client(provider: str | None) -> LLMClient:
    provider = (provider or LLM_PROVIDER_OLLAMA).strip().lower()
    with _lock:
        if provider not in _clients:
            if provider == LLM_PROVIDER_OLLAMA:
                from app.utils.llm.ollama_client import OllamaClient

                _clients[provider] = OllamaClient()
            elif provider == LLM_PROVIDER_OPENAI:
                from app.utils.llm.openai_client import OpenAIClient

                _clients[provider] = OpenAIClient()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return _clients[provider]


__all__ = ["LLMClient", "get_client"]
