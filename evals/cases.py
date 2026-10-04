"""The evaluation cases. Each case is one fresh session with one to three scripted turns.

Every case also gets two generic checks (no crash, scene ids grounded) and a check
that a failed tool is reported. The checks below are specific to the case.
All dates are in the past, so the catalogue answers do not change between runs.
"""
from evals.checks import (
    NOT_FOUND_WORDS, SCENE_ID, Case, CaseRun, custom, no_tools, normalize, reply_matches, reply_not_matches, scenes,
    successful, tool_not_called, tools_in_order,
)

RAVENNA = "Find Sentinel-2 scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover."
VENICE = "Find scenes over Venice, Italy in August 2025 with less than 5% cloud cover."


# ---------- check bodies that need the data of the session ----------

def mentions_a_result(run: CaseRun) -> tuple[bool, str]:
    ids = {s["id"] for s in scenes(run, 0)}
    cited = ids & set(SCENE_ID.findall(run.turns[0].reply))
    return bool(cited), f"{len(cited)} of {len(ids)} result ids cited"


def details_of_second(run: CaseRun) -> tuple[bool, str]:
    found = scenes(run, 0)
    if len(found) < 2:
        return False, "the search returned fewer than 2 scenes"
    wanted = found[1]["id"]
    asked = [t.args.get("scene_id") for t in run.turns[1].tools if t.tool == "get_scene_details"]
    return wanted in asked, f"expected {wanted}, called with {asked}"


def states_first_cloud_cover(run: CaseRun) -> tuple[bool, str]:
    wanted = f"{scenes(run, 0)[0]['cloud_cover']:.1f}"
    return wanted in run.turns[1].reply, f"expected {wanted} in the reply"


def search_reuses_area_and_dates(run: CaseRun) -> tuple[bool, str]:
    before = successful(run, 0, "search_scenes").args
    after = successful(run, 1, "search_scenes")
    if after is None:
        return False, "no successful search in turn 2"
    same = (after.args["bbox"] == before["bbox"] and after.args["start_date"] == before["start_date"]
            and after.args["end_date"] == before["end_date"])
    return same and after.args["max_cloud_cover"] == 3, f"turn 2 search args: {after.args}"


def search_returned_nothing(run: CaseRun) -> tuple[bool, str]:
    found = successful(run, 0, "search_scenes")
    return bool(found and found.data["scenes"] == []), "search ran and returned an empty list"


def uses_min_cloud_filter(run: CaseRun) -> tuple[bool, str]:
    found = successful(run, 0, "search_scenes")
    return bool(found and found.args.get("min_cloud_cover", 0) >= 50), f"args: {found.args if found else None}"


def cited_scenes_match_the_filter(run: CaseRun) -> tuple[bool, str]:
    cloud = {s["id"]: s["cloud_cover"] for s in scenes(run, 0)}
    cited = SCENE_ID.findall(run.turns[0].reply)
    wrong = [i for i in cited if cloud.get(i, 0) < 50]
    return bool(cited) and not wrong, f"cited {len(cited)}, below 50% cloud: {wrong}"


def geocode_found_nothing(run: CaseRun) -> tuple[bool, str]:
    found = successful(run, 0, "geocode_place")
    return bool(found and found.data["result"] == []), "geocoding ran and returned no candidate"


def names_the_chosen_place(run: CaseRun) -> tuple[bool, str]:
    candidates = successful(run, 0, "geocode_place").data["result"]
    if len(candidates) < 2:
        return False, "the place was not ambiguous in the geocoding result"
    names = [c["region"] for c in candidates if c.get("region")] + [c["country"] for c in candidates if c.get("country")]
    hit = [n for n in names if n.lower() in run.turns[0].reply.lower()]
    return bool(hit), f"candidates: {names}, mentioned: {hit}"


def failed_get_scene_details(run: CaseRun) -> tuple[bool, str]:
    calls = [t for t in run.turns[0].tools if t.tool == "get_scene_details"]
    return bool(calls) and not calls[-1].ok, f"calls: {[(t.args, t.ok) for t in calls]}"


def states_the_platform(run: CaseRun) -> tuple[bool, str]:
    platform = successful(run, 1, "get_scene_details").data["platform"]
    return normalize(platform) in normalize(run.turns[2].reply), f"expected {platform}"


def details_were_requested(run: CaseRun) -> tuple[bool, str]:
    return successful(run, 1, "get_scene_details") is not None, "get_scene_details ran in turn 2"


CASES = [
    Case("search_basic", [RAVENNA], [
        tools_in_order(0, "geocode_place", "search_scenes"),
        custom("the reply lists scenes from the results", "tool", mentions_a_result),
        reply_matches(0, r"\bItaly\b", "the reply says which Ravenna it used", "disclosure"),
    ]),
    Case("follow_up_second_one", [RAVENNA, "Give me the details of the second one."], [
        custom("details requested for the second scene of the first search", "memory", details_of_second),
        tool_not_called(1, "search_scenes"),
    ]),
    Case("follow_up_value_from_memory", [RAVENNA, "What is the cloud cover of the first scene?"], [
        no_tools(1),
        custom("the reply gives the cloud cover from the first search", "memory", states_first_cloud_cover),
    ]),
    Case("same_area_new_filter", [RAVENNA, "Same area and dates, but only scenes under 3% cloud cover."], [
        custom("the second search reuses area and dates and uses the new filter", "memory", search_reuses_area_and_dates),
    ]),
    Case("empty_result", ["Find scenes over Ravenna, Italy in July 2025 with 0% cloud cover."], [
        custom("the search ran and found nothing", "empty", search_returned_nothing),
        reply_matches(0, NOT_FOUND_WORDS, "the reply says that nothing was found", "empty"),
        reply_matches(0, r"\?", "the reply asks a follow-up question", "empty"),
    ]),
    Case("min_cloud_filter", ["Find scenes over Imola, Italy in September 2026 with more than 50% cloud cover."], [
        custom("the search uses min_cloud_cover of at least 50", "tool", uses_min_cloud_filter),
        custom("every scene in the reply has at least 50% cloud cover", "groundedness", cited_scenes_match_the_filter),
    ]),
    Case("unknown_place", ["Find scenes over Xyzzyqwkj in July 2025."], [
        custom("geocoding ran and found nothing", "empty", geocode_found_nothing),
        tool_not_called(0, "search_scenes"),
        reply_matches(0, NOT_FOUND_WORDS + r"|more precise|different name", "the reply says the place was not found", "empty"),
    ]),
    Case("ambiguous_place", ["Find scenes over Springfield in June 2025 with less than 20% cloud cover."], [
        custom("the reply names the Springfield it used", "disclosure", names_the_chosen_place),
    ]),
    Case("unknown_scene_id", ["Give me the details of scene S2X_DOES_NOT_EXIST."], [
        custom("get_scene_details was called and failed", "invalid", failed_get_scene_details),
        reply_not_matches(0, r"\d+(\.\d+)?\s*%", "the reply gives no invented cloud cover", "groundedness"),
    ]),
    Case("area_too_large", ["Find scenes with less than 10% cloud cover in July 2025 for the area between longitude 5 and 12 and latitude 40 and 46."], [
        reply_matches(0, r"too (large|big|wide)|smaller|2 degrees|220 km|limit", "the reply explains the area is too large", "invalid"),
    ]),
    Case("impossible_date", ["Find scenes over Ravenna, Italy from 2025-02-30 to 2025-03-05."], [
        reply_matches(0, r"date|invalid|february|valid", "the reply deals with the date that does not exist", "invalid"),
    ]),
    Case("missing_data_is_declared", [RAVENNA, "What is the vegetation percentage of the first scene?"], [
        reply_matches(1, r"missing data", "the reply ends with a Missing data line", "disclosure"),
    ]),
    Case("platform_from_details", [VENICE, "Give me the details of the third one.", "Which satellite took it?"], [
        custom("details were requested in turn 2", "memory", details_were_requested),
        custom("the reply names the satellite from the details", "groundedness", states_the_platform),
    ]),
    Case("reference_without_context", ["Give me the details of the third one."], [
        tool_not_called(0, "get_scene_details"),
        reply_matches(0, r"\?|please (specify|provide|tell|clarify)|could you|can you", "the agent asks what the user means", "memory"),
    ]),
    Case("out_of_scope", ["What is the capital of France?"], [
        no_tools(0),
        reply_not_matches(0, r"capital[^.]{0,40}\bParis\b|\bParis\b[^.]{0,40}capital|\bParis is\b", "the agent does not answer from its own knowledge", "groundedness"),
        reply_matches(0, r"cannot|can't|do not|don't|not able|designed to|only", "the agent says the question is outside what it does", "groundedness"),
    ]),
]
