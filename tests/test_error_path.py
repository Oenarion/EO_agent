"""Error path, API and trace tests. No network, no real model."""
import json
import logging
from typing import Any

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException

from eo_agent.agent.errors import EMPTY_REPLY, MCP_DOWN_REPLY
from eo_agent.agent.runtime import AgentRuntime
from eo_agent.api.main import create_app
from eo_agent.config import Settings
from test_graph import BBOX, Anything, call, failing_tool, tools_ok

SECRET = "sk-secret-1234567890"


class ScriptedChatModel(BaseChatModel):
    """A real LangChain chat model that returns scripted replies (or raises scripted errors),
    so the callbacks that produce the llm_call trace events fire like with a real model."""

    replies: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self) -> ChatResult:
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._next()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._next()


def handled_error_tool(name: str, message: str) -> StructuredTool:
    """A tool that raises ToolException, which LangChain turns into a message with status=error.
    This is how the MCP adapter reports isError=True."""
    async def run(**kwargs):
        raise ToolException(message)

    return StructuredTool(name=name, description=name, args_schema=Anything, coroutine=run,
                          response_format="content_and_artifact", handle_tool_error=True)


def make_runtime(tmp_path, replies, tools=None, loader=None) -> AgentRuntime:
    async def default_loader(url):
        return tools if tools is not None else tools_ok()

    settings = Settings(trace_dir=str(tmp_path), llm_api_key=SECRET, max_steps=6)
    return AgentRuntime(settings, llm=ScriptedChatModel(replies=replies), tool_loader=loader or default_loader)


def post(client, session: str, message: str):
    return client.post("/chat", json={"session_id": session, "message": message})


UNKNOWN_ID = {"scene_id": "S2X_DOES_NOT_EXIST"}
EXPLANATION = "I tried to get the details of S2X_DOES_NOT_EXIST but the catalogue has no such scene."


@pytest.mark.parametrize("failing", [
    failing_tool("get_scene_details", ConnectionError("connection refused")),
    handled_error_tool("get_scene_details", "[not_found] No scene with id 'S2X_DOES_NOT_EXIST'"),
], ids=["tool raises", "tool returns isError"])
def test_forced_tool_error_gives_200_a_clear_reply_and_an_error_in_the_trace(tmp_path, failing):
    runtime = make_runtime(tmp_path, [call("get_scene_details", UNKNOWN_ID, "c1"), AIMessage(content=EXPLANATION)], [failing])
    with TestClient(create_app(runtime)) as client:
        response = post(client, "s1", "details of S2X_DOES_NOT_EXIST")
        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == EXPLANATION and body["turn"] == 1
        assert body["tool_calls"] == [{"tool": "get_scene_details", "args": UNKNOWN_ID, "ok": False}]

        events = client.get("/traces/s1").json()["events"]
    tool_event = next(e for e in events if e["event"] == "tool_call")
    assert tool_event["tool"] == "get_scene_details" and tool_event["args"] == UNKNOWN_ID
    assert tool_event["error"] and tool_event["duration_ms"] is not None
    assert events[-1]["event"] == "request_end" and events[-1]["final_answer"] == EXPLANATION


def test_the_session_keeps_working_after_a_tool_error(tmp_path):
    runtime = make_runtime(tmp_path, [call("get_scene_details", UNKNOWN_ID, "c1"), AIMessage(content=EXPLANATION), AIMessage(content="Hello again.")],
                           [failing_tool("get_scene_details", ConnectionError("boom"))])
    with TestClient(create_app(runtime)) as client:
        assert post(client, "s1", "details please").status_code == 200
        second = post(client, "s1", "hi")
        assert second.status_code == 200 and second.json()["turn"] == 2 and second.json()["reply"] == "Hello again."


def test_empty_model_answer_after_a_failure_gets_the_fixed_template(tmp_path):
    runtime = make_runtime(tmp_path, [call("get_scene_details", UNKNOWN_ID, "c1"), AIMessage(content="")],
                           [failing_tool("get_scene_details", ConnectionError("connection refused"))])
    with TestClient(create_app(runtime)) as client:
        reply = post(client, "s1", "details please").json()["reply"]
    assert reply == ('I tried to call get_scene_details with {"scene_id": "S2X_DOES_NOT_EXIST"} '
                     "and it failed: ConnectionError: connection refused.")


def test_empty_model_answer_without_any_failure_gets_a_generic_reply(tmp_path):
    runtime = make_runtime(tmp_path, [AIMessage(content="")])
    with TestClient(create_app(runtime)) as client:
        assert post(client, "s1", "hi").json()["reply"] == EMPTY_REPLY


def test_unreachable_mcp_server_gives_a_clear_reply_and_the_api_stays_up(tmp_path):
    attempts = {"n": 0}

    async def loader(url):
        attempts["n"] += 1
        if attempts["n"] <= 2:  # down at startup and on the first request, then it comes back
            raise ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("All connection attempts failed")])
        return tools_ok()

    runtime = make_runtime(tmp_path, [AIMessage(content="Back online.")], loader=loader)
    with TestClient(create_app(runtime)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["mcp_connected"] is False and "ConnectError" in health["mcp_error"]

        down = post(client, "s1", "find scenes")
        assert down.status_code == 200 and down.json()["reply"] == MCP_DOWN_REPLY and down.json()["tool_calls"] == []
        end = client.get("/traces/s1").json()["events"][-1]
        assert end["event"] == "request_end" and "ConnectError" in end["error"]  # the real cause, not "TaskGroup"

        back = post(client, "s1", "find scenes again")  # the next request retries the connection
        assert back.status_code == 200 and back.json()["reply"] == "Back online."
        assert client.get("/health").json()["mcp_connected"] is True


def test_unreachable_language_model_returns_503(tmp_path):
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://model.invalid"))
    runtime = make_runtime(tmp_path, [error, AIMessage(content="ok")])
    with TestClient(create_app(runtime)) as client:
        response = post(client, "s1", "hi")
        assert response.status_code == 503 and "language model" in response.json()["detail"]
        last = client.get("/traces/s1").json()["events"][-1]
        assert last["event"] == "request_end" and last["error"] and last["final_answer"] is None
        assert post(client, "s1", "hi again").status_code == 200  # the API is still up


def test_unexpected_failure_returns_500_and_the_api_stays_up(tmp_path):
    runtime = make_runtime(tmp_path, [RuntimeError("bug in my code"), AIMessage(content="fine")])
    with TestClient(create_app(runtime), raise_server_exceptions=False) as client:
        assert post(client, "s1", "hi").status_code == 500
        assert post(client, "s1", "hi again").json()["reply"] == "fine"


def test_failure_is_logged_with_session_id_tool_and_error(tmp_path, caplog):
    import asyncio

    runtime = make_runtime(tmp_path, [call("get_scene_details", UNKNOWN_ID, "c1"), AIMessage(content=EXPLANATION)],
                           [failing_tool("get_scene_details", ConnectionError("boom"))])
    with caplog.at_level(logging.ERROR, logger="eo_agent"):
        asyncio.run(runtime.chat("sess-42", "details please"))
    record = next(r for r in caplog.records if r.levelno == logging.ERROR and r.name == "eo_agent.agent")
    assert record.session_id == "sess-42"
    assert "get_scene_details" in record.getMessage() and "boom" in record.getMessage()


@pytest.mark.parametrize("bad", ["../secret", "a b", "x" * 65, "a/b"])
def test_bad_session_ids_are_rejected(tmp_path, bad):
    with TestClient(create_app(make_runtime(tmp_path, []))) as client:
        assert post(client, bad, "hi").status_code == 422
        assert client.get(f"/sessions/{bad}/memory").status_code in (404, 422)


def test_an_api_key_typed_by_the_user_never_reaches_the_trace_file(tmp_path):
    runtime = make_runtime(tmp_path, [AIMessage(content="I will not repeat that.")])
    with TestClient(create_app(runtime)) as client:
        post(client, "s1", f"my key is {SECRET}")
    text = (tmp_path / "s1.jsonl").read_text(encoding="utf-8")
    assert SECRET not in text and "[REDACTED]" in text
