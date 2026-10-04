"""Trace structure and the readable trace table."""
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from eo_agent.api.main import create_app
from eo_agent.observability.pretty import render
from test_error_path import UNKNOWN_ID, failing_tool, make_runtime, post
from test_graph import BBOX, SEARCH_DATA, call, fake_tool

FIELDS = {"ts", "session_id", "turn", "event", "node", "tool", "args", "duration_ms", "error",
          "result_summary", "final_answer", "data"}
SEARCH_ARGS = {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}


def three_turn_session(tmp_path):
    """The demo shape: a search, a follow-up, a forced error. Returns the events and the replies."""
    scene_args = {"scene_id": "SCENE_B"}
    tools = [
        fake_tool("search_scenes", lambda **kw: SEARCH_DATA),
        failing_tool("get_scene_details", ConnectionError("not found upstream")),
    ]
    replies = [
        call("search_scenes", SEARCH_ARGS, "c1"), AIMessage(content="Found 2 scenes."),
        AIMessage(content="The second one is SCENE_B."),                       # follow-up: answered from memory, no tool
        call("get_scene_details", UNKNOWN_ID, "c2"), AIMessage(content="That scene does not exist."),
    ]
    runtime = make_runtime(tmp_path, replies, tools)
    with TestClient(create_app(runtime)) as client:
        answers = [post(client, "demo", m).json()["reply"] for m in ("find scenes", "which is the second one?", "details of S2X_DOES_NOT_EXIST")]
        events = client.get("/traces/demo").json()["events"]
        memory = client.get("/sessions/demo/memory")
    return events, answers, memory


def test_every_line_has_the_same_fields_and_the_session_id(tmp_path):
    events, _, _ = three_turn_session(tmp_path)
    assert all(set(e) == FIELDS for e in events)
    assert {e["session_id"] for e in events} == {"demo"}


def test_each_turn_has_start_nodes_model_calls_tools_and_end_in_order(tmp_path):
    events, _, _ = three_turn_session(tmp_path)
    turn1 = [e for e in events if e["turn"] == 1]
    assert turn1[0]["event"] == "request_start" and turn1[0]["args"] == {"message": "find scenes"}
    assert turn1[-1]["event"] == "request_end"
    nodes = [e["node"] for e in turn1 if e["event"] == "node"]
    assert nodes == ["prepare_context", "agent", "tools", "update_memory", "prepare_context", "agent"]
    assert [e["tool"] for e in turn1 if e["event"] == "tool_call"] == ["search_scenes"]
    llm = [e for e in turn1 if e["event"] == "llm_call"]
    assert len(llm) == 2 and llm[0]["result_summary"] == "requested tools: search_scenes"
    assert llm[1]["result_summary"].startswith("text answer")
    assert all(e["duration_ms"] is not None for e in turn1 if e["event"] in ("node", "llm_call", "tool_call"))


def test_the_trace_matches_the_three_replies(tmp_path):
    events, answers, _ = three_turn_session(tmp_path)
    ends = [e for e in events if e["event"] == "request_end"]
    assert [e["turn"] for e in ends] == [1, 2, 3]
    assert [e["final_answer"] for e in ends] == answers
    assert [e["error"] for e in ends] == [None, None, None]  # the failed tool call does not fail the request
    # turn 2 needed no tool: the follow-up was answered from the kept context
    assert not [e for e in events if e["turn"] == 2 and e["event"] == "tool_call"]
    # turn 3 contains the error
    failed = [e for e in events if e["turn"] == 3 and e["event"] == "tool_call"]
    assert len(failed) == 1 and "ConnectionError" in failed[0]["error"]


def test_prepare_context_records_what_was_sent_to_the_model(tmp_path):
    events, _, _ = three_turn_session(tmp_path)
    first = next(e for e in events if e["event"] == "node" and e["node"] == "prepare_context")
    assert first["data"]["messages_sent"] == 1 and first["data"]["chars_sent"] > 0
    assert first["result_summary"].startswith("model input: 1 messages")
    model_call = next(e for e in events if e["event"] == "llm_call")
    assert model_call["data"]["messages_sent"] == 1


def test_memory_endpoint_returns_the_working_memory(tmp_path):
    _, _, memory = three_turn_session(tmp_path)
    body = memory.json()
    assert memory.status_code == 200 and body["turns"] == 3
    assert [s["id"] for s in body["working_memory"]["last_results"]] == ["SCENE_A", "SCENE_B"]


def test_unknown_session_has_no_memory_and_no_trace(tmp_path):
    with TestClient(create_app(make_runtime(tmp_path, []))) as client:
        assert client.get("/sessions/nobody/memory").status_code == 404
        assert client.get("/traces/nobody").status_code == 404


def test_pretty_table_shows_turns_errors_and_final_answers(tmp_path):
    events, answers, _ = three_turn_session(tmp_path)
    table = render(events)
    for turn in (1, 2, 3):
        assert f"=== Turn {turn}" in table
    assert "search_scenes" in table and "ERROR" in table and "ConnectionError" in table
    assert all(answer in table for answer in answers)
    assert render([]) == "(empty trace)"
