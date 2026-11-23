from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    """Common interface for text generation, whatever the provider."""

    max_input_chars: int

    @abstractmethod
    def generate(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> str:
        """Return the full completion for the prompt."""

    @abstractmethod
    def stream(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> Iterator[str]:
        """Yield the completion piece by piece."""
