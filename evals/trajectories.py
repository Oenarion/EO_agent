"""The expected tool calls of the evaluation cases (see trajectory.py).

Only what the question really requires is expected. A place has to be found before it is searched,
a follow-up that the memory can answer needs no call, and so on. Arguments are checked on the facts
(the dates, the cloud range, the area, the scene id), not on how the model wrote them.
"""

from evals.checks import CaseRun, scenes, successful
from evals.trajectory import ArgsCheck, ExpectedCall

Args = dict


def _inside(bbox, lat: float, lon: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def covers(lat: float, lon: float) -> ArgsCheck:
    return lambda a, run: _inside(a["bbox"], lat, lon)


def dates(start: str, end: str) -> ArgsCheck:
    return lambda a, run: (a.get("start_date"), a.get("end_date")) == (start, end)


def cloud_max(value: float) -> ArgsCheck:
    return lambda a, run: a.get("max_cloud_cover") == value


def cloud_min_at_least(value: float) -> ArgsCheck:
    return lambda a, run: a.get("min_cloud_cover", 0) >= value


def no_cloud_filter() -> ArgsCheck:
    return lambda a, run: a.get("max_cloud_cover", 100) >= 100 and a.get("min_cloud_cover", 0) <= 0


def place_named(word: str) -> ArgsCheck:
    return lambda a, run: word.lower() in a.get("name", "").lower()


def scene_number(index: int, turn: int = 0) -> ArgsCheck:
    """The scene id is the one at this position (0 based) in the successful search of that turn."""
    return lambda a, run: a.get("scene_id") == scenes(run, turn)[index]["id"]


def scene_id(value: str) -> ArgsCheck:
    return lambda a, run: a.get("scene_id") == value


def same_area_dates_and_max(max_cloud: float) -> ArgsCheck:
    def check(a: Args, run: CaseRun) -> bool:
        first = successful(run, 0, "search_scenes").args
        return a["bbox"] == first["bbox"] and dates(first["start_date"], first["end_date"])(a, run) and a.get("max_cloud_cover") == max_cloud
    return check


def both(*checks: ArgsCheck) -> ArgsCheck:
    return lambda a, run: all(c(a, run) for c in checks)


def geocode(name: str, turn: int = 0) -> ExpectedCall:
    return ExpectedCall("geocode_place", place_named(name), turn, f"place {name}")


def search(check: ArgsCheck, turn: int = 0, note: str = "") -> ExpectedCall:
    return ExpectedCall("search_scenes", check, turn, note)


def skill_named(name: str) -> ArgsCheck:
    return lambda a, run: a.get("name") == name


RAVENNA = (44.41, 12.20)
ROME = (41.89, 12.49)

TRAJECTORIES: dict[str, list[ExpectedCall]] = {
    "search_basic": [geocode("Ravenna"), search(both(covers(*RAVENNA), dates("2025-07-01", "2025-07-31"), cloud_max(10)), note="Ravenna, July 2025, max 10%")],
    "follow_up_second_one": [
        geocode("Ravenna"), search(both(covers(*RAVENNA), dates("2025-07-01", "2025-07-31"), cloud_max(10))),
        ExpectedCall("get_scene_details", scene_number(1), turn=1, note="the second scene"),
    ],
    "follow_up_value_from_memory": [geocode("Ravenna"), search(covers(*RAVENNA))],  # turn 2 needs no call
    "same_area_new_filter": [
        geocode("Ravenna"), search(covers(*RAVENNA)),
        search(same_area_dates_and_max(3), turn=1, note="same area and dates, max 3%"),
    ],
    "empty_result": [geocode("Ravenna"), search(both(covers(*RAVENNA), cloud_max(0)), note="max 0%")],
    "min_cloud_filter": [geocode("Imola"), search(both(covers(44.35, 11.71), dates("2026-09-01", "2026-09-30"), cloud_min_at_least(50)), note="more than 50%")],
    "unknown_place": [ExpectedCall("geocode_place", place_named("Xyzzyqwkj"))],  # and nothing else
    "ambiguous_place": [geocode("Springfield"), search(dates("2025-06-01", "2025-06-30"))],
    "unknown_scene_id": [ExpectedCall("get_scene_details", scene_id("S2X_DOES_NOT_EXIST"))],
    "local_name_rome": [geocode("Roma"), search(both(covers(*ROME), dates("2025-07-01", "2025-07-31")), note="Rome, not another Roma")],
    "single_day_without_scene": [geocode("Tel Aviv"), search(both(dates("2025-09-10", "2025-09-10"), cloud_max(10)), note="one day, max 10%")],
    "single_day_scene_above_cloud_limit": [geocode("Copenhagen"), search(both(dates("2025-09-10", "2025-09-10"), cloud_max(20)))],
    "no_cloud_limit_is_not_assumed": [geocode("Copenhagen"), search(both(dates("2025-09-10", "2025-09-10"), no_cloud_filter()), note="no cloud filter")],
    "platform_from_details": [
        geocode("Venice"), search(dates("2025-08-01", "2025-08-31")),
        ExpectedCall("get_scene_details", scene_number(2), turn=1, note="the third scene"),
    ],  # turn 3 needs no call
    "skill_changes_a_choice_question": [
        geocode("Ravenna"), ExpectedCall("load_skill", skill_named("scene-selection"), unordered=True, note="the scene-selection skill"),
        search(both(covers(*RAVENNA), dates("2025-07-01", "2025-07-31"))),
        ExpectedCall("get_scene_details", None, note="sun elevation of candidate 1"),
        ExpectedCall("get_scene_details", None, note="sun elevation of candidate 2"),
        ExpectedCall("get_scene_details", None, note="sun elevation of candidate 3"),
    ],
    "language_change_triggers_a_new_search": [
        geocode("Roma"), ExpectedCall("search_scenes", None, 0, "English setting: any place"),
        geocode("Roma", turn=1), search(covers(*ROME), turn=1, note="Rome after /language it"),
    ],
}
