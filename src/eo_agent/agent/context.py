"""Builds the exact input sent to the model.

Phase 2 version: system prompt + session memory block + all messages.
Phase 3 adds the window, the summary and the truncation.
"""
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from eo_agent.agent.prompts import LIMIT_NOTE, system_prompt
from eo_agent.agent.state import AgentState
from eo_agent.config import Settings


def render_memory(memory: dict[str, Any], summary: str = "") -> str:
    lines = ["SESSION MEMORY (kept by the system, reliable):"]
    if not memory and not summary:
        return lines[0] + " empty, nothing has been searched yet."
    place = memory.get("place")
    if place:
        label = ", ".join(str(place[k]) for k in ("name", "region", "country") if place.get(k)) or "unnamed area"
        lines.append(f"- Place: {label}, bbox {place.get('bbox')}")
    if memory.get("date_range"):
        lines.append(f"- Date range: {memory['date_range']['start']} to {memory['date_range']['end']}")
    if memory.get("max_cloud_cover") is not None:
        lines.append(f"- Cloud cover filter: {memory.get('min_cloud_cover', 0)}% to {memory['max_cloud_cover']}%")
    if "last_results" in memory:
        results = memory["last_results"]
        if results:
            lines.append(f"- Latest search results ({len(results)} shown, {memory.get('total_found')} found in total):")
            for s in results:
                lines.append(f"  {s['index']}. {s['id']} | {s['datetime']} | cloud {s['cloud_cover']}% | tile {s['tile']}")
        else:
            lines.append("- Latest search returned no scenes.")
    if memory.get("selected_scene_id"):
        lines.append(f"- Selected scene: {memory['selected_scene_id']}")
    if summary:
        lines.append(f"- Summary of earlier conversation: {summary}")
    return "\n".join(lines)


def build_model_input(state: AgentState, settings: Settings) -> list[BaseMessage]:
    system = system_prompt() + "\n\n" + render_memory(state.get("working_memory") or {}, state.get("summary", ""))
    if state.get("step_count", 0) >= settings.max_steps:
        system += LIMIT_NOTE
    return [SystemMessage(content=system), *state["messages"]]
