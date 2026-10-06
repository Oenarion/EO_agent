"""Verification of final answers: the claims, the matching, the pairing, and the graph around them. Offline."""
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.agent.verify import build_facts, check, feedback_text, matches, warning_text
from eo_agent.api.main import create_app
from eo_agent.config import Settings
from test_citations import ID_A, ID_B, SEARCH_RESULT, tool as data_tool
from test_error_path import make_runtime, post
from test_graph import BBOX, ScriptedLLM, call

INVENTED = "S2A_32TQQ_20250101_0_L2A"
SEARCH_CALL = {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}


def facts_of(*messages, memory=None, summary=""):
    return build_facts(list(messages), memory or {}, summary)


def search_tool_message() -> ToolMessage:
    return ToolMessage(content=json.dumps(SEARCH_RESULT, separators=(",", ":")), name="search_scenes", tool_call_id="t1")


FACTS = facts_of(HumanMessage(content="scenes over Ravenna in July 2025 with less than 10% cloud cover"), search_tool_message())


# ---------- rounding ----------

def test_the_matching_rule_on_real_numbers():
    assert matches(1.402733, 1.40, 2) and matches(1.402733, 1.4, 1) and matches(1.402733, 1, 0)
    assert matches(62.2487, 62.25, 2) and matches(62.2487, 62.24, 2)   # rounded or truncated
    assert not matches(1.402733, 1.9, 1) and not matches(1.402733, 1.41, 2) and not matches(1.402733, 2, 0)


# ---------- what passes ----------

def test_a_correct_answer_passes_with_dots_or_commas():
    answer = f"The clearest is {ID_A} on 2025-07-31 with 1.40% cloud cover, then {ID_B} with 2,4%. The limit was 10%."
    verdict = check(answer, FACTS)
    assert verdict.ok and verdict.checked >= 5


def test_an_answer_with_nothing_to_check_passes():
    verdict = check("That is outside what I do. I can help you find scenes.", FACTS)
    assert verdict.ok and verdict.checked == 0


def test_the_sources_block_is_not_checked():
    answer = f"Scene {ID_A}.\n\nSources:\n- [{INVENTED}](https://example.test)\nPlace data: Open-Meteo.com."
    assert check(answer, FACTS).ok


def test_numbers_the_user_wrote_may_be_repeated():
    assert check("You asked for less than 10%, and I found two scenes.", FACTS).ok


def test_earlier_answers_count_as_facts_but_the_arguments_of_tool_calls_do_not():
    earlier = AIMessage(content="An earlier verified answer said 33.3% cloud over 2025-06-01.")
    args_only = AIMessage(content="", tool_calls=[{"name": "search_scenes", "args": {"start_date": "2025-05-05"}, "id": "x", "type": "tool_call"}])
    facts = facts_of(HumanMessage(content="hi"), earlier, args_only)
    assert check("It was 33.3% on 2025-06-01.", facts).ok
    assert not check("The search was on 2025-05-05.", facts).ok  # the model's own arguments are not evidence


def test_the_memory_and_the_summary_are_facts_too():
    facts = facts_of(HumanMessage(content="hi"), memory={"last_results": [{"id": ID_A, "cloud_cover": 1.402733}]},
                     summary=f"Earlier the user looked at {INVENTED} with 77.7% cloud cover.")
    assert check(f"{ID_A} has 1.4% cloud cover and {INVENTED} had 77.7%.", facts).ok


# ---------- what does not pass ----------

def test_an_invented_scene_id_is_caught():
    verdict = check(f"The best is {INVENTED}.", FACTS)
    assert not verdict.ok and INVENTED in verdict.problems[0]


def test_a_date_inside_a_datetime_counts_and_its_time_digits_do_not_support_numbers():
    facts = facts_of(ToolMessage(content='{"id":"S2C_32TQQ_20250731_0_L2A","datetime":"2025-07-31T10:18:46.243000Z"}', name="get_scene_details", tool_call_id="t"))
    assert check("It was acquired on 2025-07-31.", facts).ok            # a date written inside a datetime is a known date
    assert not check("It has 18% cloud cover.", facts).ok and not check("It has 46% cloud cover.", facts).ok


def test_a_date_no_tool_returned_is_caught():
    verdict = check("There is a scene on 2025-07-04.", FACTS)
    assert not verdict.ok and "2025-07-04" in verdict.problems[0]


def test_a_percentage_that_no_tool_returned_is_caught():
    assert not check("It has 17.3% cloud cover.", FACTS).ok


def test_degrees_are_checked_like_percentages():
    facts = facts_of(ToolMessage(content='{"id":"S2A_32TQQ_20250723_0_L2A","sun_elevation":62.4881095758032}', name="get_scene_details", tool_call_id="t"))
    assert check("The sun elevation is 62.49°.", facts).ok
    assert not check("The sun elevation is 71.2°.", facts).ok


def test_digits_of_ids_and_dates_do_not_support_a_percentage():
    """20250731 and 2025-07-31 contain 20 and 31: a claim of 31% must not pass because of them."""
    facts = facts_of(ToolMessage(content=json.dumps({"id": ID_A, "datetime": "2025-07-31T10:18:46Z"}), name="get_scene_details", tool_call_id="t"))
    assert not check("It has 31% cloud cover.", facts).ok and not check("It has 20% cloud cover.", facts).ok


# ---------- the number of one scene attached to another ----------

def test_the_cloud_cover_of_another_scene_on_a_line_is_caught():
    answer = f"1. `{ID_A}` | 2025-07-31 | cloud 2.4%"           # 2.4% exists, but it belongs to ID_B
    verdict = check(answer, FACTS)
    assert not verdict.ok and "its cloud cover is 1.40%" in verdict.problems[0]


def test_the_right_pairing_passes_and_the_unclear_lines_are_not_judged():
    assert check(f"1. `{ID_A}` | cloud 1.40%\n2. `{ID_B}` | cloud 2.4%", FACTS).ok
    assert check(f"{ID_A} and {ID_B} have 1.4% and 2.4%", FACTS).ok              # two ids on one line: no pairing
    assert check(f"{ID_B} has 2.4% cloud, below your 10% limit", FACTS).ok       # two percentages on one line: no pairing


def test_a_scene_whose_cloud_cover_is_unknown_is_not_paired():
    facts = facts_of(HumanMessage(content=f"look at {INVENTED}"), memory={})
    assert check(f"{INVENTED} | cloud 2.4%", facts).problems == ["the value 2.4% is not in any tool result"]  # only the existence check


# ---------- the texts ----------

def test_feedback_and_warning_name_the_problems_and_stay_short():
    problems = [f"problem {i}" for i in range(9)]
    assert "problem 0" in feedback_text(problems) and "problem 5" in feedback_text(problems) and "problem 6" not in feedback_text(problems)
    assert feedback_text(problems).startswith("CORRECTION:")
    assert warning_text(problems).startswith("Warning:") and "problem 4" in warning_text(problems) and "problem 5" not in warning_text(problems)


# ---------- inside the graph ----------

def graph_of(llm, settings=None):
    return build_graph(llm, [data_tool("search_scenes", SEARCH_RESULT)], settings or Settings())


WRONG = f"The clearest is {INVENTED} with 1.4% cloud cover."
RIGHT = f"The clearest is {ID_A} with 1.40% cloud cover."


async def test_a_correct_answer_costs_no_extra_model_call():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=RIGHT)])
    state = await run_turn(graph_of(llm), "s", "find scenes")
    assert len(llm.calls) == 2 and state["verify_report"]["action"] == "ok"
    assert not any("CORRECTION" in c["messages"][0].content for c in llm.calls)


async def test_a_wrong_answer_is_written_again_with_a_correction():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG), AIMessage(content=RIGHT)])
    state = await run_turn(graph_of(llm), "s", "find scenes")
    assert state["messages"][-1].content.startswith(RIGHT)
    assert not any(INVENTED in str(m.content) for m in state["messages"])      # the wrong answer is gone from the state
    answers = [m for m in state["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert len(answers) == 1
    rewrite_call = llm.calls[2]["messages"][0].content
    assert "CORRECTION:" in rewrite_call and INVENTED in rewrite_call and "CORRECTION" not in llm.calls[1]["messages"][0].content
    assert state["verify_report"]["action"] == "ok" and state["verify_feedback"] == ""


async def test_still_wrong_after_the_rewrite_the_answer_stays_with_a_visible_warning():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG), AIMessage(content=WRONG)])
    state = await run_turn(graph_of(llm), "s", "find scenes")
    final = state["messages"][-1].content
    assert final.startswith(WRONG) and "Warning: I could not verify these values" in final and INVENTED in final
    assert state["verify_report"]["action"] == "warning" and len(llm.calls) == 3   # one rewrite, not an endless loop


async def test_zero_rewrites_allowed_means_a_warning_at_once():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG)])
    state = await run_turn(graph_of(llm, Settings(max_verify_retries=0)), "s", "find scenes")
    assert "Warning:" in state["messages"][-1].content and len(llm.calls) == 2


async def test_the_rewrite_may_call_a_tool_before_answering():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG),
                       call("search_scenes", SEARCH_CALL, "c2"), AIMessage(content=RIGHT)])
    state = await run_turn(graph_of(llm), "s", "find scenes")
    assert state["messages"][-1].content.startswith(RIGHT) and state["verify_report"]["action"] == "ok"


async def test_the_allowance_for_a_rewrite_comes_back_every_turn():
    llm = ScriptedLLM([
        call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG), AIMessage(content=RIGHT),   # turn 1: one rewrite
        AIMessage(content="The first one is " + ID_B + " at 99.9%."), AIMessage(content=f"The first one is {ID_A}."),  # turn 2: again
    ])
    graph = graph_of(llm)
    await run_turn(graph, "s", "find scenes")
    state = await run_turn(graph, "s", "which is the first?")
    assert state["messages"][-1].content.startswith(f"The first one is {ID_A}.") and "Warning" not in state["messages"][-1].content


async def test_a_follow_up_answered_from_the_memory_is_checked_against_the_memory():
    llm = ScriptedLLM([call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=RIGHT),
                       AIMessage(content=f"The first scene, {ID_A}, has 1.40% cloud cover.")])
    graph = graph_of(llm)
    await run_turn(graph, "s", "find scenes")
    state = await run_turn(graph, "s", "cloud cover of the first one?")
    assert state["verify_report"]["action"] == "ok" and len(llm.calls) == 3


# ---------- through the API and the trace ----------

def test_the_trace_shows_what_verification_found(tmp_path):
    replies = [call("search_scenes", SEARCH_CALL, "c1"), AIMessage(content=WRONG), AIMessage(content=RIGHT)]
    runtime = make_runtime(tmp_path, replies, [data_tool("search_scenes", SEARCH_RESULT)])
    with TestClient(create_app(runtime)) as client:
        body = post(client, "s", "find scenes").json()
        events = client.get("/traces/s").json()["events"]
    assert body["reply"].startswith(RIGHT)
    verify_events = [e for e in events if e["event"] == "node" and e["node"] == "verify"]
    assert [e["data"]["action"] for e in verify_events] == ["rewrite", "ok"]
    assert "asked to rewrite" in verify_events[0]["result_summary"] and INVENTED in verify_events[0]["data"]["problems"][0]
    assert "supported by the tool results" in verify_events[1]["result_summary"]


def test_the_trace_says_so_when_an_answer_has_nothing_to_verify(tmp_path):
    runtime = make_runtime(tmp_path, [AIMessage(content="That is outside what I do.")])
    with TestClient(create_app(runtime)) as client:
        post(client, "s", "what is the capital of France?")
        events = client.get("/traces/s").json()["events"]
    verify_event = next(e for e in events if e["event"] == "node" and e["node"] == "verify")
    assert verify_event["result_summary"] == "nothing to verify in this answer (no scene ids, dates or values)"
    assert verify_event["data"]["action"] == "ok" and verify_event["data"]["checked"] == 0
