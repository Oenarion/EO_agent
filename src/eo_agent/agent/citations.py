"""Citations: a Sources block added to an answer by code, not by the model.

Every scene id that the answer cites, and that a tool really returned in this session,
gets a link to its catalogue record and to its preview. The model is not asked to copy
URLs, so a link can never be mistyped or invented. An id that no tool returned gets no link.
"""
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

SCENE_ID = re.compile(r"\bS2[A-D]_[0-9A-Z]{5}_\d{8}_\d+_L2A\b")
SOURCES_MARK = "\nSources:"
MAX_LINKS_KEPT = 200  # scene links remembered per session


def strip_sources(text: str) -> str:
    """Remove a trailing Sources block, so the same answer is never cited twice."""
    cut = text.rfind(SOURCES_MARK)
    return text[:cut].rstrip() if cut != -1 else text


def cited_ids(text: str, links: dict[str, dict[str, str]]) -> list[str]:
    """Scene ids in the text, in order of first appearance, that have a known record link."""
    seen: list[str] = []
    for scene_id in SCENE_ID.findall(text):
        if scene_id in links and scene_id not in seen:
            seen.append(scene_id)
    return seen


def used_geocoding(messages: list[BaseMessage]) -> bool:
    """Was geocode_place called successfully in the current turn?"""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return any(isinstance(m, ToolMessage) and m.name == "geocode_place" and m.status != "error" for m in messages[start:])


def sources_block(ids: list[str], links: dict[str, dict[str, str]], geocoded: bool) -> str:
    lines = ["Sources:"]
    for scene_id in ids:
        link = links[scene_id]
        preview = f" ([preview]({link['preview']}))" if link.get("preview") else ""
        lines.append(f"- [{scene_id}]({link['record']}){preview}")
    attribution = "Scene data: Sentinel-2 L2A through Earth Search (Element 84)."
    if geocoded:
        attribution += " Place data: Open-Meteo.com."
    lines.append(attribution)
    return "\n".join(lines)


def add_sources(text: str, links: dict[str, dict[str, str]], geocoded: bool) -> str:
    """The answer with a Sources block, or unchanged if it cites no known scene."""
    body = strip_sources(text)
    ids = cited_ids(body, links)
    if not ids:
        return body
    return f"{body}\n\n{sources_block(ids, links, geocoded)}"


def remember_links(links: dict[str, dict[str, str]], scenes: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Add the links of these scenes (dicts as returned by the tools) to the remembered ones."""
    updated = dict(links)
    for s in scenes:
        if s.get("id") and s.get("record_url"):
            updated.pop(s["id"], None)  # move it to the end: the oldest are dropped first
            updated[s["id"]] = {"record": s["record_url"], "preview": s.get("thumbnail_url") or ""}
    while len(updated) > MAX_LINKS_KEPT:
        updated.pop(next(iter(updated)))
    return updated
