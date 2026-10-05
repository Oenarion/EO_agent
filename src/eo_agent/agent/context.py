"""Context policy: what is stored, what is sent to the model, how it is bounded.

STORED per session (the checkpoint):
  - the message list, with every tool message capped at MAX_TOOL_CHARS
  - the working memory (place, dates, cloud range, numbered results, selection)
  - the summary of older turns

SENT to the model on every call:
  - the system prompt, then the SESSION MEMORY block (working memory + summary)
  - only the last MAX_WINDOW messages, extended backwards to the start of a turn,
    so a tool call is never separated from its tool result
  - tool results from earlier turns are replaced by one-line stubs

BOUNDED: when the stored history has more than SUMMARY_TRIGGER messages, the oldest
whole turns are folded into the summary and removed from the state. The current turn
is never touched.
"""
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from eo_agent.agent.prompts import LIMIT_NOTE, SUMMARY_PROMPT, system_prompt
from eo_agent.config import DEFAULT_PLACE_LANGUAGE, LANGUAGE_NAMES, Settings

log = logging.getLogger("eo_agent.context")


# ---------- small helpers ----------

def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def message_chars(message: BaseMessage) -> int:
    size = len(message_text(message))
    if isinstance(message, AIMessage) and message.tool_calls:
        size += len(json.dumps(message.tool_calls))
    return size


def last_human_index(messages: list[BaseMessage]) -> int:
    """Index of the message that started the current turn, or -1."""
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)


# ---------- the window ----------

def window_start(messages: list[BaseMessage], max_window: int) -> int:
    """First index of the messages to send.

    Take the last max_window messages, then walk back to the nearest user message.
    A turn is [user, assistant (tool call), tool result(s), assistant (answer)], so
    starting at a user message can never split a tool call from its result.
    """
    start = max(0, len(messages) - max_window)
    while start > 0 and not isinstance(messages[start], HumanMessage):
        start -= 1
    return start


# ---------- one-line stubs for old tool results ----------

def stub_for(message: ToolMessage, args: dict[str, Any]) -> str:
    name = message.name or "tool"
    if message.status == "error":
        return message_text(message)[:300]
    if name == "load_skill":  # plain text, not JSON
        return f"load_skill({args.get('name')!r}): instructions loaded ({len(message_text(message))} chars)"
    try:
        data = json.loads(message_text(message))
        if name == "geocode_place":
            places = data.get("result", [])
            if not places:
                return f"geocode_place({args.get('name')!r}): no match"
            first = places[0]
            label = ", ".join(str(first[k]) for k in ("name", "region", "country") if first.get(k))
            return f"geocode_place({args.get('name')!r}): {len(places)} candidate(s), first: {label}"
        if name == "search_scenes":
            q = data["query"]
            return (
                f"search_scenes: {data['returned']} of {data['total_found']} scenes returned for "
                f"{q['start_date']}..{q['end_date']}, cloud {q.get('min_cloud_cover', 0)}-{q['max_cloud_cover']}%"
            )
        if name == "get_scene_details":
            return f"get_scene_details({data['id']}): cloud {data.get('cloud_cover')}%, {data.get('platform')}, {data.get('datetime')}"
    except (ValueError, KeyError, TypeError, AttributeError):
        pass  # truncated or unexpected content: fall through to the generic stub
    return f"{name}: result omitted ({len(message_text(message))} chars)"


def _tool_args_by_id(messages: list[BaseMessage]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c["args"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}


def compact_old_tool_payloads(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replace tool results from earlier turns with stubs. The current turn stays raw."""
    boundary = last_human_index(messages)
    args = _tool_args_by_id(messages)
    return [
        m.model_copy(update={"content": stub_for(m, args.get(m.tool_call_id, {}))})
        if isinstance(m, ToolMessage) and i < boundary
        else m
        for i, m in enumerate(messages)
    ]


# ---------- truncation of stored tool messages ----------

def truncate_tool_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}... [truncated, {len(content)} chars in total]"


# ---------- the SESSION MEMORY block ----------

def render_memory(memory: dict[str, Any], summary: str = "", place_language: str = DEFAULT_PLACE_LANGUAGE) -> str:
    lines = ["SESSION MEMORY (kept by the system, reliable):"]
    language = (f"- Place names are searched in {LANGUAGE_NAMES.get(place_language, place_language)}. "
                "The user can change this setting with the /language command of the chat.")
    if not memory and not summary:
        return lines[0] + " empty, nothing has been searched yet.\n" + language
    lines.append(language)
    used = memory.get("place_language_used")
    if used and used != place_language:
        lines.append(f"- The place name language was changed from {LANGUAGE_NAMES.get(used, used)} after the last place search: "
                     "call geocode_place again for any place the user asks about, even one searched before.")
    place = memory.get("place")
    if place:
        label = ", ".join(str(place[k]) for k in ("name", "region", "country") if place.get(k)) or "unnamed area"
        lines.append(f"- Place: {label}, bbox {place.get('bbox')}")
    if memory.get("date_range"):
        lines.append(f"- Date range: {memory['date_range']['start']} to {memory['date_range']['end']}")
    if memory.get("max_cloud_cover") is not None:
        low, high = memory.get("min_cloud_cover", 0), memory["max_cloud_cover"]
        lines.append("- Cloud cover filter in the last search: none" if (low, high) == (0, 100)
                     else f"- Cloud cover filter in the last search: {low}% to {high}%")
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


def build_model_input(
    messages: list[BaseMessage], memory: dict[str, Any], summary: str, step_count: int, settings: Settings,
    place_language: str = DEFAULT_PLACE_LANGUAGE, skills_index: str = "", correction: str = "",
) -> list[BaseMessage]:
    system = system_prompt() + ("\n\n" + skills_index if skills_index else "") + "\n\n" + render_memory(memory, summary, place_language)
    if correction:
        system += "\n\n" + correction
    if step_count >= settings.max_steps:
        system += LIMIT_NOTE
    window = compact_old_tool_payloads(messages)[window_start(messages, settings.max_window):]
    return [SystemMessage(content=system), *window]


# ---------- bounding: summarize the oldest whole turns ----------

def cut_for_summary(messages: list[BaseMessage], max_window: int, trigger: int) -> int:
    """How many of the oldest messages to fold into the summary (0 = none).

    Fires only when the history is longer than `trigger`. The cut is always at the
    start of a turn, it leaves at most `max_window` messages, and it never goes past
    the start of the current turn.
    """
    if len(messages) <= trigger:
        return 0
    current = last_human_index(messages)
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and i <= current and len(messages) - i <= max_window:
            return i
    return max(current, 0)  # even the current turn alone is bigger than the window: keep just that


def render_for_summary(messages: list[BaseMessage]) -> str:
    args = _tool_args_by_id(messages)
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {message_text(m)}")
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                lines.append("Assistant called: " + "; ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls))
            if message_text(m).strip():
                lines.append(f"Assistant: {message_text(m)}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result: {stub_for(m, args.get(m.tool_call_id, {}))}")
    return "\n".join(lines)


def template_summary(previous: str, old_messages: list[BaseMessage], max_chars: int) -> str:
    """Deterministic fallback when the model cannot summarize. Keeps the most recent part."""
    parts = [previous] if previous else []
    for line in render_for_summary(old_messages).splitlines():
        parts.append(line[:200])
    return " | ".join(parts)[-max_chars:]


async def summarize(llm: BaseChatModel, previous: str, old_messages: list[BaseMessage], settings: Settings) -> str:
    """One short model call. If it fails or returns nothing, use the template. Never raises."""
    prompt = [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=f"Previous summary:\n{previous or '(none)'}\n\nNew messages to fold in:\n{render_for_summary(old_messages)}"),
    ]
    text = ""
    try:
        text = message_text(await llm.ainvoke(prompt)).strip()
    except Exception as exc:  # the turn must go on even if summarizing fails
        log.warning("summary call failed (%s: %s), using the template", type(exc).__name__, exc)
    if not text:
        return template_summary(previous, old_messages, settings.summary_max_chars)
    return text[: settings.summary_max_chars]
