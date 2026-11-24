from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A function the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.llm_calls += other.llm_calls
        self.tool_calls += other.tool_calls

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


@dataclass
class ChatTurn:
    """What one chat call produced once the stream has finished."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # Provider-specific output items to send back next round (e.g. OpenAI's
    # encrypted reasoning); None when the provider needs nothing extra.
    raw: list[dict[str, Any]] | None = None


class ToolsNotSupportedError(RuntimeError):
    """The selected model cannot call tools; callers fall back to plain RAG."""


# Chat messages use one provider-neutral shape:
#   {"role": "system" | "user" | "assistant", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": list[ToolCall]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Message = dict[str, Any]


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

    @abstractmethod
    def chat_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        force_answer: bool = False,
    ) -> Iterator[str | ChatTurn]:
        """Yield text pieces as they arrive, then one final ChatTurn.

        With force_answer the model must reply in text and may not call tools
        (the tools stay declared so earlier tool messages remain valid).
        Raises ToolsNotSupportedError when tools are given but the model
        cannot use them.
        """

    def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float | None = None,
    ) -> ChatTurn:
        """Run a chat call without tools and return the finished turn."""
        turn = ChatTurn()
        for part in self.chat_stream(model, messages, None, temperature):
            if isinstance(part, ChatTurn):
                turn = part
        return turn
