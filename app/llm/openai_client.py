import logging
from collections.abc import Iterator

from openai import BadRequestError, OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_BASE_URL
from app.llm.base import LLMClient
from app.llm.constants import (
    OPENAI_MAX_INPUT_CHARS,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    max_input_chars = OPENAI_MAX_INPUT_CHARS

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT
        )

    def request(self, model: str, prompt: str, temperature: float | None, stream: bool):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            "stream": stream,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            # Reasoning models only accept the default temperature.
            if "temperature" not in kwargs or "temperature" not in str(e):
                raise
            logger.info("Model %s rejected temperature, retrying without it", model)
            kwargs.pop("temperature")
            return self.client.chat.completions.create(**kwargs)

    def generate(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> str:
        response = self.request(model, prompt, temperature, stream=False)
        return response.choices[0].message.content or ""

    def stream(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> Iterator[str]:
        response = self.request(model, prompt, temperature, stream=True)
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            response.close()
