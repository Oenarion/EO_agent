"""Verification of a final answer, in plain code (no model).

The answer is searched for claims that can be checked: scene ids, dates (YYYY-MM-DD), percentages
and degrees. Each one must be supported by a fact of the conversation: a tool result, the working
memory, the summary, an earlier answer, or something the user wrote. A percentage on a line with
a single scene id must also be the cloud cover of that scene, so the number of one scene cannot
be attached to another.

This checks that values exist and belong where they are written. It does not understand the
sentences around them.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from eo_agent.agent.citations import SCENE_ID, strip_sources
from eo_agent.agent.context import message_text

ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")  # also inside 2025-07-31T10:18:46Z
ISO_DATETIME = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}T[\d:.]+Z?")
PERCENT = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)?)\s*%")
DEGREES = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)?)\s*°")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _number(text: str) -> tuple[float, int]:
    """A number as written in an answer (dot or comma) and how many decimals were written."""
    text = text.replace(",", ".")
    return float(text), len(text.split(".")[1]) if "." in text else 0


def matches(known: float, claim: float, decimals: int) -> bool:
    """Is `claim`, written with that many decimals, a rounding or a truncation of `known`?
    1.402733 supports 1.40, 1.4 and 1; 62.2487 supports 62.25 and 62.24; 1.9 does not support 1.4."""
    step = 10 ** -decimals
    return claim - step / 2 - 1e-9 <= known < claim + step + 1e-9


@dataclass
class Facts:
    ids: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    numbers: list[float] = field(default_factory=list)
    cloud_by_id: dict[str, float] = field(default_factory=dict)


@dataclass
class Verdict:
    checked: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def build_facts(messages: list[BaseMessage], memory: dict[str, Any], summary: str) -> Facts:
    """Everything that may be quoted. `messages` must not contain the answer that is being checked."""
    parts = [message_text(m) for m in messages if not (isinstance(m, AIMessage) and m.tool_calls)]
    parts += [json.dumps(memory, default=str), summary]
    text = "\n".join(parts)
    facts = Facts(ids=set(SCENE_ID.findall(text)), dates=set(ISO_DATE.findall(text)))
    # numbers are read after removing ids, dates and times, so that their digits do not support a percentage
    bare = ISO_DATE.sub(" ", ISO_DATETIME.sub(" ", SCENE_ID.sub(" ", text)))
    facts.numbers = [float(n) for n in NUMBER.findall(bare)]

    for scene in memory.get("last_results", []):
        if scene.get("cloud_cover") is not None:
            facts.cloud_by_id[scene["id"]] = scene["cloud_cover"]
    for m in messages:
        if isinstance(m, ToolMessage) and m.status != "error":
            try:
                data = json.loads(message_text(m))
            except ValueError:
                continue  # plain text (a skill) or a truncated message
            for scene in data.get("scenes", []) if isinstance(data, dict) else []:
                if scene.get("cloud_cover") is not None:
                    facts.cloud_by_id[scene["id"]] = scene["cloud_cover"]
            if isinstance(data, dict) and "id" in data and data.get("cloud_cover") is not None:
                facts.cloud_by_id[data["id"]] = data["cloud_cover"]
    return facts


def check(answer: str, facts: Facts) -> Verdict:
    body = strip_sources(answer)
    problems: list[str] = []
    checked = 0

    for scene_id in dict.fromkeys(SCENE_ID.findall(body)):
        checked += 1
        if scene_id not in facts.ids:
            problems.append(f"scene id {scene_id} is not in any tool result")
    for day in dict.fromkeys(ISO_DATE.findall(body)):
        checked += 1
        if day not in facts.dates:
            problems.append(f"date {day} is not in any tool result or message")
    for unit, pattern in (("%", PERCENT), ("°", DEGREES)):
        for written in dict.fromkeys(pattern.findall(body)):
            checked += 1
            value, decimals = _number(written)
            if not any(matches(k, value, decimals) for k in facts.numbers):
                problems.append(f"the value {written}{unit} is not in any tool result")

    for line in body.splitlines():  # a percentage next to one scene id must be the cloud cover of that scene
        ids, percents = list(dict.fromkeys(SCENE_ID.findall(line))), PERCENT.findall(line)
        if len(ids) == 1 and len(percents) == 1 and ids[0] in facts.cloud_by_id:
            checked += 1
            value, decimals = _number(percents[0])
            real = facts.cloud_by_id[ids[0]]
            if not matches(real, value, decimals):
                problems.append(f"{percents[0]}% is written next to {ids[0]}, but its cloud cover is {real:.2f}%")
    return Verdict(checked, problems)


def feedback_text(problems: list[str]) -> str:
    """What the model is told when its answer did not pass."""
    return ("CORRECTION: your last answer contained values that the tool results of this conversation do not support: "
            + "; ".join(problems[:6]) + ". Write the answer again, using only the ids, dates and numbers that appear in the "
            "tool results (call a tool if you need one). Do not mention this correction.")


def warning_text(problems: list[str]) -> str:
    return "Warning: I could not verify these values against the tool results: " + "; ".join(problems[:5]) + "."
