import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from openai import BadRequestError, NotFoundError, OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_BASE_URL
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
    OPENAI_CHAT_MAX_OUTPUT_TOKENS,
    OPENAI_MAX_INPUT_CHARS,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_REASONING_EFFORT,
    OPENAI_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Optional request fields some models reject; they are dropped and the call retried.
OPTIONAL_PARAMS = ("temperature", "reasoning", "include", "parallel_tool_calls")


def parse_arguments(raw: str | None, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("Tool %s sent invalid JSON arguments", name)
        return {}
    return value if isinstance(value, dict) else {}


class OpenAIClient(LLMClient):
    max_input_chars = OPENAI_MAX_INPUT_CHARS

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # The SDK already retries 429, 5xx and connection errors (2 times, with backoff).
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=OPENAI_TIMEOUT,
        )
        # Set once the server turns out not to offer the Responses API.
        self.responses_unavailable = False

    @staticmethod
    def call(create: Callable[..., Any], kwargs: dict[str, Any]):
        """Call the API, dropping optional fields the model rejects."""
        while True:
            try:
                return create(**kwargs)
            except BadRequestError as e:
                message = str(e)
                rejected = next(
                    (p for p in OPTIONAL_PARAMS if p in kwargs and p in message), None
                )
                if rejected is None:
                    if "tools" in kwargs and "tool" in message.lower() and "support" in message.lower():
                        raise ToolsNotSupportedError(message) from e
                    raise
                logger.info("Model %s rejected %s, retrying without it", kwargs["model"], rejected)
                kwargs.pop(rejected)

    # --- plain prompt (summaries) ------------------------------------------

    def request(self, model: str, prompt: str, temperature: float | None, stream: bool):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            "stream": stream,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self.call(self.client.chat.completions.create, kwargs)

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

    # --- chat with tools ----------------------------------------------------

    def chat_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        force_answer: bool = False,
    ) -> Iterator[str | ChatTurn]:
        # Reasoning models only allow function tools on the Responses API.
        if not self.responses_unavailable:
            try:
                yield from self.responses_stream(model, messages, tools, temperature, force_answer)
                return
            except NotFoundError:
                logger.info("Responses API not available, using Chat Completions")
                self.responses_unavailable = True
        yield from self.completions_stream(model, messages, tools, temperature, force_answer)

    @staticmethod
    def to_responses_input(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                instructions.append(m["content"])
            elif role == "tool":
                items.append(
                    {"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]}
                )
            elif m.get("raw"):
                # Output items from an earlier round, including encrypted reasoning.
                items.extend(m["raw"])
            elif m.get("tool_calls"):
                if m.get("content"):
                    items.append({"role": "assistant", "content": m["content"]})
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": c.id,
                        "name": c.name,
                        "arguments": json.dumps(c.arguments, ensure_ascii=False),
                    }
                    for c in m["tool_calls"]
                )
            else:
                items.append({"role": role, "content": m["content"]})
        return "\n\n".join(instructions), items

    def responses_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float | None,
        force_answer: bool,
    ) -> Iterator[str | ChatTurn]:
        instructions, items = self.to_responses_input(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "input": items,
            "max_output_tokens": OPENAI_CHAT_MAX_OUTPUT_TOKENS,
            "stream": True,
            # Nothing is kept on OpenAI's side; reasoning is carried between
            # rounds in encrypted form instead.
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        }
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": True,
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "none" if force_answer else "auto"
            if not force_answer:
                kwargs["parallel_tool_calls"] = True
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.call(self.client.responses.create, kwargs)
        turn = ChatTurn(usage=Usage(llm_calls=1))
        text: list[str] = []
        raw: list[dict[str, Any]] = []
        try:
            for event in response:
                kind = event.type
                if kind == "response.output_text.delta":
                    text.append(event.delta)
                    yield event.delta
                elif kind == "response.output_item.done":
                    item = event.item
                    raw.append(item.model_dump(exclude_none=True))
                    if item.type == "function_call":
                        turn.tool_calls.append(
                            ToolCall(
                                id=item.call_id,
                                name=item.name,
                                arguments=parse_arguments(item.arguments, item.name),
                            )
                        )
                elif kind == "response.completed" and event.response.usage:
                    u = event.response.usage
                    turn.usage.input_tokens = u.input_tokens or 0
                    turn.usage.output_tokens = u.output_tokens or 0
                    details = getattr(u, "input_tokens_details", None)
                    turn.usage.cached_tokens = getattr(details, "cached_tokens", 0) or 0
                elif kind in ("response.failed", "error"):
                    raise RuntimeError(f"OpenAI response failed: {event}")
        finally:
            response.close()

        turn.content = "".join(text)
        turn.raw = raw
        yield turn

    @staticmethod
    def to_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
                )
            elif m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                                },
                            }
                            for c in m["tool_calls"]
                        ],
                    }
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def completions_stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float | None,
        force_answer: bool,
    ) -> Iterator[str | ChatTurn]:
        """Chat Completions fallback for OpenAI-compatible servers."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self.to_chat_messages(messages),
            "max_completion_tokens": OPENAI_CHAT_MAX_OUTPUT_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": True,
                    },
                }
                for t in tools
            ]
            if force_answer:
                kwargs["tool_choice"] = "none"
            else:
                kwargs["parallel_tool_calls"] = True
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.call(self.client.chat.completions.create, kwargs)
        turn = ChatTurn(usage=Usage(llm_calls=1))
        text: list[str] = []
        # Tool call fragments arrive spread over chunks, keyed by index.
        partial: dict[int, dict[str, str]] = {}
        try:
            for chunk in response:
                if chunk.usage:
                    u = chunk.usage
                    turn.usage.input_tokens = u.prompt_tokens or 0
                    turn.usage.output_tokens = u.completion_tokens or 0
                    details = getattr(u, "prompt_tokens_details", None)
                    turn.usage.cached_tokens = getattr(details, "cached_tokens", 0) or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text.append(delta.content)
                    yield delta.content
                for tc in delta.tool_calls or []:
                    slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        finally:
            response.close()

        turn.content = "".join(text)
        for index in sorted(partial):
            slot = partial[index]
            turn.tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=parse_arguments(slot["arguments"], slot["name"]),
                )
            )
        yield turn
