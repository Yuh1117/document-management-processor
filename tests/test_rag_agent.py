"""Agent loop of RagService with a scripted LLM and a fake retriever (no ES, no network)."""

import json

from app.llm.base import ChatTurn, LLMClient, ToolCall, ToolsNotSupportedError, Usage
from app.rag.passages import Passage
from app.rag.schemas import HistoryMessage
from app.rag.service import RagService
from app.rag.tools import TurnContext


class ScriptedClient(LLMClient):
    max_input_chars = 10000

    def __init__(self, turns, tools_supported=True):
        self.turns = list(turns)
        self.calls = []
        self.tools_supported = tools_supported

    def generate(self, model, prompt, temperature=None):
        raise NotImplementedError

    def stream(self, model, prompt, temperature=None):
        raise NotImplementedError

    def chat_stream(self, model, messages, tools=None, temperature=None, force_answer=False):
        if tools and not self.tools_supported:
            raise ToolsNotSupportedError("no tools")
        self.calls.append({"messages": list(messages), "tools": tools, "force_answer": force_answer})
        turn = self.turns.pop(0)
        if turn.content:
            yield turn.content
        yield turn


def make_service(passages):
    service = RagService.__new__(RagService)
    service.retrieve = lambda query, owner, excluded: passages
    from app.rag.tools import DocumentTools

    service.tools = DocumentTools(service.retrieve)
    return service


def events(stream):
    out = []
    for raw in stream:
        lines = raw.strip().split("\n")
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        out.append((name, data))
    return out


def ctx():
    return TurnContext(owner_id="u1", exclude_document_ids=[], read_max_chars=1000)


def test_agent_searches_then_answers():
    passages = [Passage("d1", "plan.pdf", "Budget is 10M.", 0.9)]
    client = ScriptedClient(
        [
            ChatTurn(tool_calls=[ToolCall("c1", "search_documents", {"query": "budget"})], usage=Usage(10, 2, 0, 1)),
            ChatTurn(content="Theo *plan.pdf*, ngân sách là 10M.", usage=Usage(30, 8, 0, 1)),
        ]
    )
    service = make_service(passages)
    c = ctx()
    out = events(service.answer_with_tools(client, "m", "Ngân sách?", [], "vi", c))

    names = [e[0] for e in out]
    assert ("status", {"stage": "searching", "query": "budget"}) in out
    assert "sources" in names
    assert out[-1] == ("token", {"text": "Theo *plan.pdf*, ngân sách là 10M."})
    # Tool result was sent back to the model.
    tool_msg = client.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "Budget is 10M." in tool_msg["content"]
    assert c.usage.input_tokens == 40 and c.usage.llm_calls == 2 and c.usage.tool_calls == 1


def test_last_round_forces_an_answer():
    looping = ChatTurn(tool_calls=[ToolCall("c", "search_documents", {"query": "x"})])
    client = ScriptedClient([looping] * 4 + [ChatTurn(content="done")])
    service = make_service([])
    out = events(service.answer_with_tools(client, "m", "q", [], "en", ctx()))
    assert client.calls[-1]["force_answer"] is True
    assert out[-1] == ("token", {"text": "done"})


def test_search_budget_is_enforced():
    calls = [ToolCall(f"c{i}", "search_documents", {"query": f"q{i}"}) for i in range(8)]
    client = ScriptedClient([ChatTurn(tool_calls=calls), ChatTurn(content="ok")])
    service = make_service([])
    list(service.answer_with_tools(client, "m", "q", [], "en", ctx()))
    tool_msgs = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert sum("limit reached" in m["content"] for m in tool_msgs) == 2


def test_history_keeps_latest_and_starts_with_user():
    history = [
        HistoryMessage(role="assistant", content="orphan"),
        HistoryMessage(role="user", content="a"),
        HistoryMessage(role="assistant", content="b"),
    ]
    kept = RagService.history_messages(history, 10000)
    assert [m["content"] for m in kept] == ["a", "b"]


def test_falls_back_to_pipeline_without_tools():
    passages = [Passage("d1", "plan.pdf", "Budget is 10M.", 0.9)]
    client = ScriptedClient(
        [
            ChatTurn(content="budget of plan", usage=Usage(llm_calls=1)),
            ChatTurn(content="10M.", usage=Usage(llm_calls=1)),
        ],
        tools_supported=False,
    )
    service = make_service(passages)
    service.model_name = lambda: "m"
    service.provider = lambda: "ollama"

    import app.rag.service as module

    original = module.get_client
    module.get_client = lambda provider: client
    try:
        history = [HistoryMessage(role="user", content="plan?"), HistoryMessage(role="assistant", content="...")]
        out = events(service.stream("how much?", "u1", "en", [], history))
    finally:
        module.get_client = original

    assert ("status", {"stage": "searching", "query": "budget of plan"}) in out
    assert ("token", {"text": "10M."}) in out
    assert out[-1][0] == "done" and out[-1][1]["usage"]["llm_calls"] == 2
