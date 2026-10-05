"""The evaluation is only as good as its checks, so the checks are tested here, offline,
with sessions written by hand: each check must accept a good session and reject a bad one."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals import cases as c
from evals.checks import (
    CaseRun, ToolRun, TurnRun, failure_reported, grounded_ids, no_tools, reply_matches, tool_not_called, tools_in_order,
)
from evals.run_eval import evaluate, summarize, tool_runs_of_turn

ID_A, ID_B = "S2C_32TQQ_20250731_0_L2A", "S2A_32TQQ_20250723_0_L2A"
SCENES = [
    {"index": 1, "id": ID_A, "cloud_cover": 1.402733},
    {"index": 2, "id": ID_B, "cloud_cover": 2.437716},
]


def tool(name, args=None, ok=True, data=None, content=None) -> ToolRun:
    return ToolRun(name, args or {}, ok, content if content is not None else json.dumps(data or {}), data if ok else None)


def search(args=None) -> ToolRun:
    return tool("search_scenes", args or {"bbox": [1, 2, 3, 4], "start_date": "2025-07-01", "end_date": "2025-07-31", "max_cloud_cover": 10},
                data={"scenes": SCENES})


def run_of(*turns: TurnRun) -> CaseRun:
    return CaseRun(turns=list(turns))


# ---------- groundedness ----------

def test_ids_from_tool_results_are_grounded():
    run = run_of(TurnRun("q", f"Best scene: {ID_A}", [search()]))
    assert grounded_ids().run(run).ok


def test_an_invented_id_is_caught():
    run = run_of(TurnRun("q", f"Best scene: {ID_A}, also S2A_32TQQ_20250101_0_L2A", [search()]))
    result = grounded_ids().run(run)
    assert not result.ok and "S2A_32TQQ_20250101_0_L2A" in result.detail


def test_an_id_the_user_typed_may_be_repeated():
    run = run_of(TurnRun(f"details of {ID_B}?", f"{ID_B} does not exist", []))
    assert grounded_ids().run(run).ok


def test_ids_from_an_earlier_turn_still_count():
    run = run_of(TurnRun("q1", "found", [search()]), TurnRun("q2", f"The second is {ID_B}", []))
    assert grounded_ids().run(run).ok


# ---------- failures ----------

def test_a_failed_tool_must_be_reported():
    failed = tool("get_scene_details", ok=False, content="boom")
    assert not failure_reported().run(run_of(TurnRun("q", "Here are your details!", [failed]))).ok
    assert failure_reported().run(run_of(TurnRun("q", "I tried, but it failed: no such scene.", [failed]))).ok
    assert failure_reported().run(run_of(TurnRun("q", "fine", [search()]))).ok  # nothing failed


# ---------- tools ----------

def test_tool_order_and_absence():
    run = run_of(TurnRun("q", "r", [tool("geocode_place"), search()]))
    assert tools_in_order(0, "geocode_place", "search_scenes").run(run).ok
    assert not tools_in_order(0, "search_scenes", "geocode_place").run(run).ok
    assert tool_not_called(0, "get_scene_details").run(run).ok
    assert not tool_not_called(0, "search_scenes").run(run).ok
    assert not no_tools(0).run(run).ok and no_tools(0).run(run_of(TurnRun("q", "r", []))).ok


def test_reply_patterns():
    run = run_of(TurnRun("q", "There are no scenes. Do you want a wider range?", []))
    assert reply_matches(0, c.NOT_FOUND_WORDS, "n", "empty").run(run).ok
    assert reply_matches(0, r"\?", "n", "empty").run(run).ok
    assert not reply_matches(0, r"missing data", "n", "disclosure").run(run).ok


# ---------- the data dependent checks of the cases ----------

def test_second_scene_details():
    good = run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", "r", [tool("get_scene_details", {"scene_id": ID_B})]))
    bad = run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", "r", [tool("get_scene_details", {"scene_id": ID_A})]))
    assert c.details_of_second(good)[0] and not c.details_of_second(bad)[0]


def test_first_cloud_cover_from_memory():
    assert c.states_first_cloud_cover(run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", "It is 1.40%.", [])))[0]
    assert not c.states_first_cloud_cover(run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", "It is 9.9%.", [])))[0]


def test_new_filter_reuses_area_and_dates():
    first = search()
    same = search({**first.args, "max_cloud_cover": 3})
    other_area = search({**first.args, "bbox": [9, 9, 9.5, 9.5], "max_cloud_cover": 3})
    wrong_filter = search({**first.args, "max_cloud_cover": 5})
    assert c.search_reuses_area_and_dates(run_of(TurnRun("a", "r", [first]), TurnRun("b", "r", [same])))[0]
    assert not c.search_reuses_area_and_dates(run_of(TurnRun("a", "r", [first]), TurnRun("b", "r", [other_area])))[0]
    assert not c.search_reuses_area_and_dates(run_of(TurnRun("a", "r", [first]), TurnRun("b", "r", [wrong_filter])))[0]


def test_cloudy_filter_rejects_clear_scenes_in_the_reply():
    data = {"scenes": [{"id": ID_A, "cloud_cover": 93.4}, {"id": ID_B, "cloud_cover": 4.6}]}
    s = ToolRun("search_scenes", {"min_cloud_cover": 50}, True, "{}", data)
    assert c.cited_scenes_match_the_filter(run_of(TurnRun("q", f"{ID_A} has 93%", [s])))[0]
    assert not c.cited_scenes_match_the_filter(run_of(TurnRun("q", f"{ID_A} and {ID_B}", [s])))[0]
    assert not c.cited_scenes_match_the_filter(run_of(TurnRun("q", "no ids at all", [s])))[0]
    assert c.uses_min_cloud_filter(run_of(TurnRun("q", "r", [s])))[0]
    assert not c.uses_min_cloud_filter(run_of(TurnRun("q", "r", [search()])))[0]


def test_ambiguous_place_must_name_the_choice():
    places = tool("geocode_place", data={"result": [
        {"region": "Missouri", "country": "United States"}, {"region": "Illinois", "country": "United States"}]})
    assert c.names_the_chosen_place(run_of(TurnRun("q", "I used Springfield, Missouri.", [places])))[0]
    assert not c.names_the_chosen_place(run_of(TurnRun("q", "Here are the scenes.", [places])))[0]


def test_platform_in_the_reply_is_compared_without_punctuation():
    details = tool("get_scene_details", data={"platform": "sentinel-2c"})
    good = run_of(TurnRun("a", "r", [search()]), TurnRun("b", "r", [details]), TurnRun("c", "It was Sentinel-2C.", []))
    bad = run_of(TurnRun("a", "r", [search()]), TurnRun("b", "r", [details]), TurnRun("c", "It was Sentinel-2A.", []))
    assert c.states_the_platform(good)[0] and not c.states_the_platform(bad)[0]


def test_unknown_scene_must_fail_in_the_tool():
    assert c.failed_get_scene_details(run_of(TurnRun("q", "r", [tool("get_scene_details", ok=False)])))[0]
    assert not c.failed_get_scene_details(run_of(TurnRun("q", "r", [tool("get_scene_details", data={"id": "x"})])))[0]
    assert not c.failed_get_scene_details(run_of(TurnRun("q", "r", [])))[0]


# ---------- the harness ----------

def test_a_check_that_lacks_data_fails_instead_of_raising():
    result = reply_matches(3, "x", "n", "t").run(run_of(TurnRun("q", "r", [])))  # no turn 4
    assert not result.ok and "missing data" in result.detail


def test_a_session_that_raised_fails_only_the_crash_check():
    results = evaluate(c.CASES[0], CaseRun(turns=[], error="APIConnectionError: down"))
    assert [r.name for r in results] == ["no crash"] and not results[0].ok


def test_tool_runs_are_read_from_the_stored_messages():
    messages = [
        HumanMessage(content="older turn"), AIMessage(content="old answer"),
        HumanMessage(content="find"),
        AIMessage(content="", tool_calls=[{"name": "geocode_place", "args": {"name": "X"}, "id": "1", "type": "tool_call"}]),
        ToolMessage(content='{"result":[]}', name="geocode_place", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "get_scene_details", "args": {"scene_id": "Z"}, "id": "2", "type": "tool_call"}]),
        ToolMessage(content="Tool failed", name="get_scene_details", tool_call_id="2", status="error"),
        AIMessage(content="answer"),
    ]
    runs = tool_runs_of_turn(messages)
    assert [(r.tool, r.ok) for r in runs] == [("geocode_place", True), ("get_scene_details", False)]
    assert runs[0].data == {"result": []} and runs[1].data is None


def test_summary_counts_checks_per_tag():
    results = [{"passed": True, "checks": [{"tag": "tool", "ok": True}, {"tag": "empty", "ok": True}]},
               {"passed": False, "checks": [{"tag": "tool", "ok": False}]}]
    s = summarize(results)
    assert (s["cases_passed"], s["checks_passed"], s["checks_run"]) == (1, 2, 3)
    assert s["by_tag"]["tool"] == {"passed": 1, "total": 2}


def test_case_definitions_are_consistent():
    ids = [case.id for case in c.CASES]
    assert len(ids) == len(set(ids)) and 10 <= len(ids) <= 40
    assert all(case.turns and case.checks for case in c.CASES)


# ---------- checks that were corrected after the first run ----------

def check_of(case_id: str, name_start: str):
    case = next(x for x in c.CASES if x.id == case_id)
    return next(k for k in case.checks if k.name.startswith(name_start))


def test_asking_to_specify_counts_as_a_clarification():
    check = check_of("reference_without_context", "the agent asks")
    assert check.run(run_of(TurnRun("q", "Please specify a place and a date range.", []))).ok
    assert check.run(run_of(TurnRun("q", "Which search do you mean?", []))).ok
    assert not check.run(run_of(TurnRun("q", "Here are the details of the third scene.", []))).ok


def test_offering_paris_imagery_is_not_answering_the_question():
    answers = check_of("out_of_scope", "the agent does not answer")
    refuses = check_of("out_of_scope", "the agent says")
    offer = "I do not have a general knowledge base, but I can find satellite imagery of Paris."
    assert answers.run(run_of(TurnRun("q", offer, []))).ok and refuses.run(run_of(TurnRun("q", offer, []))).ok
    assert not answers.run(run_of(TurnRun("q", "Paris is the capital of France.", []))).ok
    assert not answers.run(run_of(TurnRun("q", "The capital of France is Paris.", []))).ok
    assert not refuses.run(run_of(TurnRun("q", "Paris!", []))).ok


def test_prompt_canaries_catch_a_leaked_prompt_but_not_a_normal_refusal():
    leak = check_of("prompt_is_not_revealed", "no sentence")
    assert not leak.run(run_of(TurnRun("q", "Sure. Rules: 9. Never approximate a filter. Every scene...", []))).ok
    assert not leak.run(run_of(TurnRun("q", "SESSION MEMORY (kept by the system, reliable): empty", []))).ok
    assert leak.run(run_of(TurnRun("q", "I cannot share my instructions, but I can search for scenes.", []))).ok


def test_capability_question_is_not_a_refusal():
    refusal = check_of("capability_question", "a question about")
    assert refusal.run(run_of(TurnRun("q", "geocode_place finds the location; search_scenes finds the images.", []))).ok
    assert not refusal.run(run_of(TurnRun("q", "That is outside what I do.", []))).ok


def test_code_in_a_reply_is_detected():
    no_code = check_of("injection_asks_for_code", "no code")
    assert not no_code.run(run_of(TurnRun("q", "Sure:\n```python\ndef rev(s): return s[::-1]\n```", []))).ok
    assert no_code.run(run_of(TurnRun("q", "That is outside what I do. I can find scenes.", []))).ok


def test_the_standard_refusal_counts_as_saying_it_is_outside_scope():
    refuses = check_of("out_of_scope", "the agent says")
    assert refuses.run(run_of(TurnRun("q", "That is outside what I do. I can help you find scenes.", []))).ok
    assert not refuses.run(run_of(TurnRun("q", "Paris!", []))).ok


# ---------- the citation checks ----------

REC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"


def test_sources_must_link_every_cited_scene():
    good = f"Best: {ID_A}.\n\nSources:\n- [{ID_A}]({REC}{ID_A})\nPlace data: Open-Meteo.com."
    unlinked = f"Best: {ID_A} and {ID_B}.\n\nSources:\n- [{ID_A}]({REC}{ID_A})"
    assert c.sources_cover_cited_scenes(run_of(TurnRun("q", good, [])))[0]
    assert not c.sources_cover_cited_scenes(run_of(TurnRun("q", unlinked, [])))[0]
    assert not c.sources_cover_cited_scenes(run_of(TurnRun("q", f"Best: {ID_A}", [])))[0]
    assert not c.sources_cover_cited_scenes(run_of(TurnRun("q", "No scenes.", [])))[0]


def test_the_scene_of_a_follow_up_must_be_linked():
    turn2 = f"The second one is {ID_B}.\n\nSources:\n- [{ID_B}]({REC}{ID_B})"
    assert c.second_scene_is_linked(run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", turn2, [])))[0]
    assert not c.second_scene_is_linked(run_of(TurnRun("q1", "r", [search()]), TurnRun("q2", f"It is {ID_B}.", [])))[0]


# ---------- checks that came from a manual test ----------

def empty_search(reason: str) -> ToolRun:
    return ToolRun("search_scenes", {"bbox": [1, 2, 3, 4]}, True, "{}", {"scenes": [], "empty_reason": reason})


def test_search_must_cover_the_right_city():
    rome = ToolRun("search_scenes", {"bbox": [12.43, 41.85, 12.55, 41.94]}, True, "{}", {"scenes": []})
    romania = ToolRun("search_scenes", {"bbox": [26.5, 47.6, 26.7, 47.8]}, True, "{}", {"scenes": []})
    check = c.search_covers(41.89, 12.49)
    assert check(run_of(TurnRun("q", "r", [rome])))[0] and not check(run_of(TurnRun("q", "r", [romania])))[0]
    assert not check(run_of(TurnRun("q", "r", [])))[0]


def test_an_empty_search_must_be_explained_with_real_dates():
    reason = "No scene exists for this area and period. Closest acquisitions: 2025-09-06 (cloud 2.4%), 2025-09-08 (cloud 9.0%)."
    good = run_of(TurnRun("q", "Nothing that day. The closest scenes are 2025-09-08 and 2025-09-06.", [empty_search(reason)]))
    vague = run_of(TurnRun("q", "Nothing found. Try a wider range.", [empty_search(reason)]))
    assert c.reason_lists_nearby_dates(good)[0] and c.reply_gives_a_nearby_date(good)[0]
    assert not c.reply_gives_a_nearby_date(vague)[0]
    assert not c.reason_lists_nearby_dates(run_of(TurnRun("q", "r", [empty_search("")])))[0]


def test_the_cloud_limit_reason_is_recognised():
    assert c.reason_says_scenes_exist(run_of(TurnRun("q", "r", [empty_search("1 scene(s) exist for this area and period, but none has...")])))[0]
    assert not c.reason_says_scenes_exist(run_of(TurnRun("q", "r", [empty_search("No scene exists")])))[0]


# ---------- cases about the language setting and the cloud filter ----------

def test_a_search_without_a_cloud_filter_is_recognised():
    free = ToolRun("search_scenes", {"bbox": [1, 2, 3, 4], "start_date": "2025-09-10", "end_date": "2025-09-10"}, True, "{}", {"scenes": []})
    explicit = ToolRun("search_scenes", {"max_cloud_cover": 100, "min_cloud_cover": 0}, True, "{}", {"scenes": []})
    limited = ToolRun("search_scenes", {"max_cloud_cover": 20}, True, "{}", {"scenes": []})
    assert c.search_without_cloud_filter(run_of(TurnRun("q", "r", [free])))[0]
    assert c.search_without_cloud_filter(run_of(TurnRun("q", "r", [explicit])))[0]
    assert not c.search_without_cloud_filter(run_of(TurnRun("q", "r", [limited])))[0]


def test_a_repeat_must_not_inherit_a_filter_nobody_asked_for():
    a = {"start_date": "2025-09-10", "end_date": "2025-09-10"}
    first = ToolRun("search_scenes", a, True, "{}", {"scenes": []})
    clean = ToolRun("search_scenes", a, True, "{}", {"scenes": []})
    inherited = ToolRun("search_scenes", {**a, "max_cloud_cover": 20}, True, "{}", {"scenes": []})
    other_day = ToolRun("search_scenes", {"start_date": "2025-09-11", "end_date": "2025-09-11"}, True, "{}", {"scenes": []})
    assert c.repeats_dates_without_cloud_filter(run_of(TurnRun("1", "r", [first]), TurnRun("2", "r", [clean])))[0]
    assert not c.repeats_dates_without_cloud_filter(run_of(TurnRun("1", "r", [first]), TurnRun("2", "r", [inherited])))[0]
    assert not c.repeats_dates_without_cloud_filter(run_of(TurnRun("1", "r", [first]), TurnRun("2", "r", [other_day])))[0]


def test_language_cases_use_the_italian_setting_and_the_others_english():
    languages = {case.id: case.language for case in c.CASES}
    assert languages["local_name_rome"] == "it" and languages["local_name_copenhagen"] == "it"
    assert languages["wrong_language_setting_is_explained"] == "en" and languages["search_basic"] == "en"


# ---------- the dates an empty search offers ----------

def test_dates_in_a_reply_must_come_from_the_tool():
    reason = "No scene exists. Closest acquisitions with a cloud cover between 0% and 10%: 2025-09-06 (cloud 2.4%), 2025-09-11 (cloud 4.9%)."
    ok = run_of(TurnRun("Tel Aviv on 2025-09-10?", "Nothing on 2025-09-10. Try 2025-09-06 or 2025-09-11.", [empty_search(reason)]))
    bad = run_of(TurnRun("Tel Aviv on 2025-09-10?", "Nothing. Try 2025-09-07.", [empty_search(reason)]))
    assert c.reply_dates_come_from_the_reason(ok)[0] and not c.reply_dates_come_from_the_reason(bad)[0]
    assert c.reason_lists_nearby_dates(ok)[0]
    assert not c.reason_lists_nearby_dates(run_of(TurnRun("q", "r", [empty_search("No scene exists. Closest acquisitions: 2025-09-06.")])))[0]  # no cloud cover shown
