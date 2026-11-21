from collections.abc import Iterator

import ollama

from app.constants.defaults import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_INPUT_CHARS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)
from app.core.config import OLLAMA_HOST
from app.utils.llm.base import LLMClient


class OllamaClient(LLMClient):
    max_input_chars = OLLAMA_MAX_INPUT_CHARS

    def __init__(self) -> None:
        self.client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)

    @staticmethod
    def options(temperature: float | None) -> dict:
        options = {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT}
        if temperature is not None:
            options["temperature"] = temperature
        return options

    def generate(self, model: str, prompt: str, temperature: float | None = None) -> str:
        response = self.client.generate(
            model=model,
            prompt=prompt,
            options=self.options(temperature),
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        return response.response

    def stream(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> Iterator[str]:
        generation = self.client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options=self.options(temperature),
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        try:
            for part in generation:
                if part.response:
                    yield part.response
        finally:
            close = getattr(generation, "close", None)
            if close is not None:
                close()
