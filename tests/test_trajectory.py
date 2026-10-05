"""Trajectory metrics, with sessions written by hand. Offline."""
from evals import cases as c
from evals.checks import CaseRun, ToolRun, TurnRun
from evals.trajectories import (
    TRAJECTORIES, both, cloud_max, covers, dates, geocode, no_cloud_filter, scene_number, search,
)
from evals.trajectory import ExpectedCall, measure, summarize

RAVENNA = (44.41, 12.20)
BBOX = [12.14, 44.37, 12.26, 44.46]
SEARCH_ARGS = {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31", "max_cloud_cover": 10}


def tool(name, args=None, ok=True, data=None) -> ToolRun:
    return ToolRun(name, args or {}, ok, "{}", data)


def search_run(args=None) -> ToolRun:
    return tool("search_scenes", args or SEARCH_ARGS, data={"scenes": [{"id": "S2A_32TQQ_20250723_0_L2A"}, {"id": "S2C_32TQQ_20250731_0_L2A"}]})


EXPECTED = [geocode("Ravenna"), search(both(covers(*RAVENNA), dates("2025-07-01", "2025-07-31"), cloud_max(10)))]


def test_a_perfect_path():
    run = CaseRun(turns=[TurnRun("q", "r", [tool("geocode_place", {"name": "Ravenna, Italy"}), search_run()])])
    m = measure(EXPECTED, run)
    assert (m["tool_coverage"], m["order_ok"], m["param_accuracy"], m["extra_calls"]) == (1.0, True, 1.0, 0)
    assert summarize([m])["perfect_paths"] == 1


def test_a_missing_tool_lowers_the_coverage():
    m = measure(EXPECTED, CaseRun(turns=[TurnRun("q", "r", [search_run()])]))
    assert m["tool_coverage"] == 0.5 and m["missing"] == ["geocode_place"]


def test_the_wrong_order_is_detected():
    run = CaseRun(turns=[TurnRun("q", "r", [search_run(), tool("geocode_place", {"name": "Ravenna"})])])
    m = measure(EXPECTED, run)
    assert m["tool_coverage"] == 1.0 and m["order_ok"] is False


def test_wrong_arguments_lower_the_parameter_accuracy():
    wrong = {**SEARCH_ARGS, "max_cloud_cover": 50}                       # the user said 10
    elsewhere = {**SEARCH_ARGS, "bbox": [26.5, 47.6, 26.7, 47.8]}        # another Roma
    for args in (wrong, elsewhere):
        run = CaseRun(turns=[TurnRun("q", "r", [tool("geocode_place", {"name": "Ravenna"}), search_run(args)])])
        m = measure(EXPECTED, run)
        assert m["param_accuracy"] == 0.5 and m["wrong_args"] == ["search_scenes"]


def test_extra_calls_are_counted():
    run = CaseRun(turns=[TurnRun("q", "r", [tool("geocode_place", {"name": "Ravenna"}), search_run(), search_run(), tool("get_scene_details")])])
    assert measure(EXPECTED, run)["extra_calls"] == 2


def test_calls_are_matched_in_the_turn_they_were_expected():
    expected = [ExpectedCall("get_scene_details", scene_number(1), turn=1)]
    asked = tool("get_scene_details", {"scene_id": "S2C_32TQQ_20250731_0_L2A"})
    right = CaseRun(turns=[TurnRun("q1", "r", [search_run()]), TurnRun("q2", "r", [asked])])
    wrong_turn = CaseRun(turns=[TurnRun("q1", "r", [search_run(), asked]), TurnRun("q2", "r", [])])
    assert measure(expected, right)["param_accuracy"] == 1.0
    assert measure(expected, wrong_turn)["tool_coverage"] == 0.0


def test_the_second_scene_of_the_previous_search_is_checked_by_id():
    expected = [ExpectedCall("get_scene_details", scene_number(1), turn=1)]
    first_scene = tool("get_scene_details", {"scene_id": "S2A_32TQQ_20250723_0_L2A"})  # the first, not the second
    run = CaseRun(turns=[TurnRun("q1", "r", [search_run()]), TurnRun("q2", "r", [first_scene])])
    assert measure(expected, run)["param_accuracy"] == 0.0


def test_a_check_that_cannot_be_evaluated_counts_as_wrong_not_as_a_crash():
    expected = [ExpectedCall("search_scenes", scene_number(5))]   # no such scene: the check raises inside
    run = CaseRun(turns=[TurnRun("q", "r", [search_run()])])
    assert measure(expected, run)["param_accuracy"] == 0.0


def test_no_cloud_filter_is_recognised_in_the_arguments():
    expected = [search(no_cloud_filter())]
    free = CaseRun(turns=[TurnRun("q", "r", [search_run({"bbox": BBOX, "start_date": "2025-09-10", "end_date": "2025-09-10"})])])
    limited = CaseRun(turns=[TurnRun("q", "r", [search_run({"bbox": BBOX, "max_cloud_cover": 20})])])
    assert measure(expected, free)["param_accuracy"] == 1.0 and measure(expected, limited)["param_accuracy"] == 0.0


def test_summary_averages_the_runs():
    perfect = {"tool_coverage": 1.0, "order_ok": True, "param_accuracy": 1.0, "extra_calls": 0}
    flawed = {"tool_coverage": 0.5, "order_ok": False, "param_accuracy": 0.5, "extra_calls": 2}
    s = summarize([perfect, flawed])
    assert s == {"cases_measured": 2, "tool_coverage": 0.75, "order_ok_rate": 0.5, "param_accuracy": 0.75,
                 "extra_calls_per_case": 1.0, "perfect_paths": 1}
    assert summarize([]) == {}


def test_every_declared_trajectory_belongs_to_a_case_and_is_attached():
    ids = {case.id for case in c.CASES}
    assert set(TRAJECTORIES) <= ids
    assert all(case.trajectory for case in c.CASES if case.id in TRAJECTORIES)


def test_an_unordered_call_may_come_before_or_after_the_others():
    expected = [geocode("Ravenna"), ExpectedCall("load_skill", None, unordered=True), search(covers(*RAVENNA))]
    skill = tool("load_skill", {"name": "scene-selection"})
    geo = tool("geocode_place", {"name": "Ravenna"})
    for order in ([geo, skill, search_run()], [skill, geo, search_run()], [geo, search_run(), skill]):
        m = measure(expected, CaseRun(turns=[TurnRun("q", "r", order)]))
        assert m["order_ok"] and m["tool_coverage"] == 1.0, [t.tool for t in order]
    wrong = measure(expected, CaseRun(turns=[TurnRun("q", "r", [search_run(), geo, skill])]))
    assert wrong["order_ok"] is False  # geocode still has to come before the search
