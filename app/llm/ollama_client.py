from collections.abc import Iterator
from typing import Any

import ollama

from app.core.config import OLLAMA_HOST
from app.llm.base import (
    ChatTurn,
    LLMClient,
    Message,
    ToolCall,
    ToolsNotSupportedError,
    ToolSpec,
    Usage,
)
from app.llm.constants import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_INPUT_CHARS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT,
)
from app.llm.retry import retry_stream


def close_quietly(generation: Any) -> None:
    close = getattr(generation, "close", None)
    if close is not None:
        close()


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

    def generate(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> str:
        return "".join(self.stream(model, prompt, temperature))

    def stream(
        self, model: str, prompt: str, temperature: float | None = None
    ) -> Iterator[str]:
        def start() -> Iterator[str]:
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
                close_quietly(generation)

        yield from retry_stream("ollama generate", start)

    @staticmethod
    def to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
            elif m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or "",
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}}
                            for c in m["tool_calls"]
                        ],
                    }
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        force_answer: bool = False,
    ) -> Iterator[str | ChatTurn]:
        # Ollama has no tool_choice; leaving the tools out forces a text reply.
        ollama_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tools and not force_answer
            else None
        )

        def start() -> Iterator[str | ChatTurn]:
            try:
                generation = self.client.chat(
                    model=model,
                    messages=self.to_ollama(messages),
                    tools=ollama_tools,
                    stream=True,
                    options=self.options(temperature),
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                turn = ChatTurn(usage=Usage(llm_calls=1))
                text: list[str] = []
                try:
                    for part in generation:
                        message = part.message
                        if message.content:
                            text.append(message.content)
                            yield message.content
                        for i, call in enumerate(message.tool_calls or []):
                            turn.tool_calls.append(
                                ToolCall(
                                    id=f"call_{len(turn.tool_calls)}_{i}",
                                    name=call.function.name,
                                    arguments=dict(call.function.arguments or {}),
                                )
                            )
                        if part.done:
                            turn.usage.input_tokens = part.prompt_eval_count or 0
                            turn.usage.output_tokens = part.eval_count or 0
                finally:
                    close_quietly(generation)
            except ollama.ResponseError as e:
                if tools and "does not support tools" in str(e):
                    raise ToolsNotSupportedError(str(e)) from e
                raise
            turn.content = "".join(text)
            yield turn

        yield from retry_stream("ollama chat", start)
