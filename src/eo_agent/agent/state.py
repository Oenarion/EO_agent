"""Graph state and the working memory.

The working memory is a small dict that deterministic code (no LLM) keeps up to
date from tool results. It is what lets the agent resolve "the second one" or
"same dates" even after old messages are gone.
"""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from eo_agent.agent.citations import remember_links


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    working_memory: dict[str, Any]
    summary: str  # rolling summary of older turns (Phase 3)
    step_count: int  # tool loop iterations in the current turn
    place_language: str  # language in which place names are searched: a setting of the session
    verify_retries: int  # rewrites already asked in the current turn
    verify_feedback: str  # the correction given to the model for the next call (empty: none)
    verify_report: dict[str, Any]  # what the last verification found (for the trace)
    model_input: list[AnyMessage]  # exactly what the model sees this call; overwritten each time
    context_stats: dict[str, Any]  # size of the last model call, for the logs (overwritten each time)


def update_working_memory(memory: dict[str, Any], tool: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Return a new working memory after one successful tool call.

    memory keys: place_candidates, place, date_range, max_cloud_cover,
    last_results, total_found, selected_scene_id, scene_links.
    """
    wm = dict(memory)
    if tool == "geocode_place":
        wm["place_candidates"] = [
            {k: p.get(k) for k in ("name", "country", "region", "bbox")} for p in data.get("result", [])
        ]
    elif tool == "search_scenes":
        bbox = args.get("bbox")
        # The model picks which candidate to search. The bbox it used tells us which one.
        match = next((c for c in wm.get("place_candidates", []) if c["bbox"] == bbox), None)
        wm["place"] = match or {"bbox": bbox}
        wm["date_range"] = {"start": data["query"]["start_date"], "end": data["query"]["end_date"]}
        wm["min_cloud_cover"] = data["query"].get("min_cloud_cover", 0)
        wm["max_cloud_cover"] = data["query"]["max_cloud_cover"]
        wm["last_results"] = [
            {k: s.get(k) for k in ("index", "id", "datetime", "cloud_cover", "tile")} for s in data["scenes"]
        ]
        wm["total_found"] = data["total_found"]
        wm["scene_links"] = remember_links(wm.get("scene_links", {}), data["scenes"])
        wm["selected_scene_id"] = None  # a new search invalidates the old selection
    elif tool == "get_scene_details":
        wm["selected_scene_id"] = data["id"]
        wm["scene_links"] = remember_links(wm.get("scene_links", {}), [data])
    return wm
