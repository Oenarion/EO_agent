"""Graph state and the working memory.

The working memory is a small dict that deterministic code (no LLM) keeps up to
date from tool results. It is what lets the agent resolve "the second one" or
"same dates" even after old messages are gone.
"""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    working_memory: dict[str, Any]
    summary: str  # rolling summary of older turns (Phase 3)
    step_count: int  # tool loop iterations in the current turn
    model_input: list[AnyMessage]  # exactly what the model sees this call; overwritten each time


def update_working_memory(memory: dict[str, Any], tool: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Return a new working memory after one successful tool call.

    memory keys: place_candidates, place, date_range, max_cloud_cover,
    last_results, total_found, selected_scene_id.
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
        wm["selected_scene_id"] = None  # a new search invalidates the old selection
    elif tool == "get_scene_details":
        wm["selected_scene_id"] = data["id"]
    return wm
