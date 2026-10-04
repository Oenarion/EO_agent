"""Tests for the context policy. No network, no real model."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from eo_agent.agent.context import (
    compact_old_tool_payloads, cut_for_summary, stub_for, template_summary, truncate_tool_content, window_start,
)
from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.config import Settings
from test_graph import BBOX, SEARCH_DATA, ScriptedLLM, call, tools_ok

SEARCH_FULL = {**SEARCH_DATA, "returned": 2}


def tool_turn(i: int, data: dict | None = None) -> list:
    """A turn with one tool call: user, assistant (call), tool result, assistant (answer)."""
    return [
        HumanMessage(content=f"question {i}", id=f"h{i}"),
        AIMessage(content="", id=f"a{i}", tool_calls=[{"name": "search_scenes", "args": {"bbox": BBOX}, "id": f"t{i}", "type": "tool_call"}]),
        ToolMessage(content=json.dumps(data or SEARCH_FULL), name="search_scenes", tool_call_id=f"t{i}", id=f"tm{i}"),
        AIMessage(content=f"answer {i}", id=f"f{i}"),
    ]


def chat_turn(i: int) -> list:
    return [HumanMessage(content=f"hello {i}", id=f"h{i}"), AIMessage(content=f"hi {i}", id=f"f{i}")]


# ---------- window ----------

@pytest.mark.parametrize("max_window", range(1, 21))
def test_window_never_splits_a_tool_call_from_its_result(max_window):
    messages = [m for i in range(5) for m in tool_turn(i)]
    window = messages[window_start(messages, max_window):]
    assert isinstance(window[0], HumanMessage)
    call_ids = {c["id"] for m in window if isinstance(m, AIMessage) for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in window if isinstance(m, ToolMessage)}
    assert call_ids == result_ids  # every call has its result and vice versa


def test_window_walks_back_to_the_start_of_the_turn():
    messages = [m for i in range(5) for m in tool_turn(i)]  # 20 messages, turns start at 0, 4, 8, 12, 16
    assert window_start(messages, 8) == 12   # exactly a turn start
    assert window_start(messages, 6) == 12   # 14 is mid-turn, walks back to 12
    assert window_start(messages, 100) == 0  # window bigger than the history


# ---------- summary trigger ----------

def test_summary_does_not_fire_at_the_threshold_and_fires_just_above():
    three_turns = [m for i in range(3) for m in tool_turn(i)]  # 12 messages
    assert cut_for_summary(three_turns, max_window=8, trigger=12) == 0
    four_turns = three_turns + tool_turn(3)  # 16 messages
    assert cut_for_summary(four_turns, max_window=8, trigger=12) == 8  # keeps the last 2 turns (8 messages)


def test_cut_is_always_at_the_start_of_a_turn():
    messages = [m for i in range(4) for m in tool_turn(i)] + chat_turn(4)
    cut = cut_for_summary(messages, max_window=5, trigger=6)
    assert cut > 0 and isinstance(messages[cut], HumanMessage)


def test_cut_never_touches_a_long_current_turn():
    long_turn = [HumanMessage(content="now", id="hc")]
    for k in range(5):  # five tool iterations in the current turn
        long_turn += [
            AIMessage(content="", id=f"ac{k}", tool_calls=[{"name": "search_scenes", "args": {}, "id": f"c{k}", "type": "tool_call"}]),
            ToolMessage(content="{}", name="search_scenes", tool_call_id=f"c{k}", id=f"tc{k}"),
        ]
    messages = tool_turn(0) + long_turn  # 4 + 11 messages
    assert cut_for_summary(messages, max_window=8, trigger=12) == 4  # old turn goes, current turn stays whole


def test_nothing_to_cut_when_the_history_is_a_single_turn():
    messages = [HumanMessage(content="now", id="h")] + [AIMessage(content="x", id=f"a{i}") for i in range(14)]
    assert cut_for_summary(messages, max_window=8, trigger=12) == 0


# ---------- stubs and truncation ----------

def test_old_tool_payloads_become_one_line_stubs_current_turn_stays_raw():
    messages = tool_turn(0) + tool_turn(1)
    compact = compact_old_tool_payloads(messages)
    assert compact[2].content == "search_scenes: 2 of 5 scenes returned for 2025-07-01..2025-07-31, cloud 0.0-10.0%"
    assert compact[6].content == messages[6].content  # current turn untouched
    assert messages[2].content.startswith("{")        # the originals are not modified


def test_stub_variants():
    geo = ToolMessage(content=json.dumps({"result": [{"name": "Ravenna", "region": "Emilia-Romagna", "country": "Italy"}]}),
                      name="geocode_place", tool_call_id="x")
    assert stub_for(geo, {"name": "Ravenna"}) == "geocode_place('Ravenna'): 1 candidate(s), first: Ravenna, Emilia-Romagna, Italy"
    none = ToolMessage(content='{"result":[]}', name="geocode_place", tool_call_id="x")
    assert "no match" in stub_for(none, {"name": "Zzz"})
    details = ToolMessage(content='{"id":"S1","cloud_cover":1.4,"platform":"sentinel-2a","datetime":"2025-07-31T10:00:00Z"}',
                          name="get_scene_details", tool_call_id="x")
    assert stub_for(details, {}).startswith("get_scene_details(S1): cloud 1.4%")
    error = ToolMessage(content="Tool 'x' failed. Error: boom", name="x", tool_call_id="x", status="error")
    assert "boom" in stub_for(error, {})
    broken = ToolMessage(content='{"query": {"bbox": [1, 2', name="search_scenes", tool_call_id="x")  # truncated JSON
    assert "result omitted" in stub_for(broken, {})


def test_truncate_tool_content():
    assert truncate_tool_content("short", 100) == "short"
    out = truncate_tool_content("x" * 500, 100)
    assert out.startswith("x" * 100) and "truncated, 500 chars" in out


def test_template_summary_keeps_the_most_recent_part_and_respects_the_limit():
    old = tool_turn(0) + chat_turn(1)
    out = template_summary("earlier facts", old, max_chars=120)
    assert len(out) <= 120 and "hello 1" in out


# ---------- the whole graph with the policy on ----------

class SmartLLM(ScriptedLLM):
    """Answers summary requests itself (or fails on purpose); everything else comes from the script."""

    def __init__(self, replies, fail_summary: bool = False):
        super().__init__(replies)
        self.fail_summary = fail_summary
        self.summary_calls = 0

    def _reply(self, messages, with_tools):
        if isinstance(messages[0], SystemMessage) and messages[0].content.startswith("You summarize"):
            self.summary_calls += 1
            if self.fail_summary:
                raise RuntimeError("model down")
            return AIMessage(content="SUMMARY-OF-OLD-TURNS")
        return super()._reply(messages, with_tools)


SEARCH_CALL = {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}


def small_settings(**kw) -> Settings:
    return Settings(max_window=kw.get("max_window", 4), summary_trigger=kw.get("summary_trigger", 6),
                    max_tool_chars=kw.get("max_tool_chars", 4000), max_steps=6)


async def long_conversation(llm) -> dict:
    graph = build_graph(llm, tools_ok(), small_settings())
    await run_turn(graph, "s", "find scenes")                   # turn 1: search (4 messages)
    for k in range(4):
        await run_turn(graph, "s", f"small talk {k}")           # turns 2 to 5: no tools
    return await run_turn(graph, "s", "details of the second one")


def script_for_long_conversation() -> list:
    return [call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content="found")] \
        + [AIMessage(content=f"ok {k}") for k in range(4)] \
        + [call("get_scene_details", {"scene_id": "SCENE_B"}, "c2"), AIMessage(content="details")]


async def test_memory_survives_trimming_and_the_state_stays_bounded():
    llm = SmartLLM(script_for_long_conversation())
    state = await long_conversation(llm)

    assert llm.summary_calls >= 1
    assert state["summary"] == "SUMMARY-OF-OLD-TURNS"
    # the search turn was folded into the summary and removed from the state...
    assert not any(isinstance(m, ToolMessage) and m.name == "search_scenes" for m in state["messages"])
    assert len(state["messages"]) <= 8
    # ...but the numbered results are still in the working memory, so "the second one" still resolves
    final_model_calls = [c for c in llm.calls if c["messages"][-1].content == "details of the second one"]
    system_text = final_model_calls[0]["messages"][0].content
    assert "2. SCENE_B" in system_text and "SUMMARY-OF-OLD-TURNS" in system_text
    assert state["working_memory"]["selected_scene_id"] == "SCENE_B"


async def test_without_the_policy_the_state_would_grow():
    """Sanity check on the numbers: 5 plain turns after the search would be 4 + 4 * 2 + 5 = 17 messages."""
    llm = SmartLLM(script_for_long_conversation())
    state = await long_conversation(llm)
    assert len(state["messages"]) < 17


async def test_summary_failure_falls_back_to_the_template_and_the_turn_still_answers():
    llm = SmartLLM(script_for_long_conversation(), fail_summary=True)
    state = await long_conversation(llm)
    assert state["messages"][-1].content == "details"
    assert state["summary"] and state["summary"] != "SUMMARY-OF-OLD-TURNS"
    assert "find scenes" in state["summary"]  # the template kept the first question


async def test_big_tool_result_is_truncated_after_the_memory_has_read_it():
    big = {**SEARCH_DATA, "notes": "x" * 5000}
    from test_graph import fake_tool
    tools = [fake_tool("search_scenes", lambda **kw: big)]
    llm = SmartLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content="found")])
    graph = build_graph(llm, tools, small_settings(max_tool_chars=500))
    state = await run_turn(graph, "s", "find scenes")

    stored = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert len(stored.content) < 600 and "truncated" in stored.content
    assert [s["id"] for s in state["working_memory"]["last_results"]] == ["SCENE_A", "SCENE_B"]  # memory saw it all
    assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 1  # replaced, not duplicated


async def test_context_stats_show_that_less_is_sent_than_is_stored():
    llm = SmartLLM(script_for_long_conversation())
    graph = build_graph(llm, tools_ok(), small_settings(max_window=4, summary_trigger=100))  # trigger never fires
    await run_turn(graph, "s", "find scenes")
    for k in range(4):
        state = await run_turn(graph, "s", f"small talk {k}")
    stats = state["context_stats"]
    assert stats["messages_in_state"] == 4 + 3 * 2 + 1  # search turn + 3 plain turns + the current question
    assert stats["messages_sent"] < stats["messages_in_state"]
    assert stats["chars_sent"] > 0 and stats["summarized"] is False


async def test_summarized_flag_is_reported_for_the_turn_that_triggered_it():
    """Trigger 6, window 4. History at the start of each plain turn: 5, 7 (fires), 5, 7 (fires)."""
    llm = SmartLLM(script_for_long_conversation())
    graph = build_graph(llm, tools_ok(), small_settings())
    await run_turn(graph, "s", "find scenes")
    seen = []
    for k in range(4):
        state = await run_turn(graph, "s", f"small talk {k}")
        seen.append(state["context_stats"]["summarized"])
    assert seen == [False, True, False, True]
    assert llm.summary_calls == 2


async def test_token_usage_is_recorded_when_the_provider_reports_it():
    reply = AIMessage(content="hello", usage_metadata={"input_tokens": 120, "output_tokens": 7, "total_tokens": 127})
    graph = build_graph(SmartLLM([reply]), tools_ok(), small_settings())
    state = await run_turn(graph, "s", "hi")
    assert state["context_stats"]["input_tokens"] == 120 and state["context_stats"]["output_tokens"] == 7
