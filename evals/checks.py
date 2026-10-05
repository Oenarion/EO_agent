"""Deterministic checks for the evaluation.

No model judges anything. Each check is a plain function over what happened in a
session: the questions, the replies, and the tools that ran with their full results.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from eo_agent.agent.citations import SCENE_ID  # one definition, shared with the product
NOT_FOUND_WORDS = (r"no scenes|none|no results|did not find|didn't find|couldn't find|could not find|not find|nothing|zero|"
                   r"no match|not found|unable to find|no place|no location|no imagery")
FAILURE_WORDS = (r"fail|error|could not|couldn't|unable|cannot|can't|not exist|does not exist|doesn't exist|not found|"
                 r"no scene|invalid|too large|no such|not valid|isn't valid")


@dataclass
class ToolRun:
    tool: str
    args: dict[str, Any]
    ok: bool
    content: str
    data: dict[str, Any] | None  # the parsed result, when the tool succeeded


@dataclass
class TurnRun:
    question: str
    reply: str
    tools: list[ToolRun] = field(default_factory=list)


@dataclass
class CaseRun:
    turns: list[TurnRun]
    error: str | None = None  # set when a request raised instead of answering
    verify: list[dict[str, Any]] = field(default_factory=list)  # the verification steps that did NOT pass at once (from the trace)

    def questions_text(self) -> str:
        return "\n".join(t.question for t in self.turns)

    def outputs_text(self) -> str:
        return "\n".join(tool.content for t in self.turns for tool in t.tools)


@dataclass
class CheckResult:
    name: str
    tag: str
    ok: bool
    detail: str


@dataclass
class Check:
    name: str
    tag: str  # groundedness, tool, memory, empty, invalid, disclosure
    fn: Callable[[CaseRun], tuple[bool, str]]

    def run(self, case_run: CaseRun) -> CheckResult:
        try:
            ok, detail = self.fn(case_run)
        except (IndexError, KeyError, TypeError, AttributeError) as exc:  # the session did not produce what the check needs
            ok, detail = False, f"missing data for this check ({type(exc).__name__}: {exc})"
        return CheckResult(self.name, self.tag, ok, detail)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---------- helpers for case definitions ----------

def successful(run: CaseRun, turn: int, tool: str) -> ToolRun | None:
    return next((t for t in run.turns[turn].tools if t.tool == tool and t.ok), None)


def scenes(run: CaseRun, turn: int = 0) -> list[dict[str, Any]]:
    found = successful(run, turn, "search_scenes")
    return found.data["scenes"] if found and found.data else []


# ---------- generic checks, added to every case ----------

def no_crash() -> Check:
    return Check("no crash", "tool", lambda r: (r.error is None and all(t.reply.strip() for t in r.turns),
                                                r.error or "every turn returned a non-empty reply"))


def grounded_ids() -> Check:
    """Every scene id in a reply must come from a tool result of the same session (or from the user)."""
    def fn(run: CaseRun) -> tuple[bool, str]:
        in_replies = {i for t in run.turns for i in SCENE_ID.findall(t.reply)}
        allowed = set(SCENE_ID.findall(run.outputs_text())) | set(SCENE_ID.findall(run.questions_text()))
        invented = sorted(in_replies - allowed)
        return not invented, f"invented ids: {invented}" if invented else f"{len(in_replies)} id(s) cited, all found in tool results"
    return Check("scene ids come from tool results", "groundedness", fn)


def verification_quiet() -> Check:
    """The verify step of the agent must have had nothing to correct. If it did, the model first wrote
    something that the tool results do not support (and the step fixed it), or the verifier raised a false alarm:
    either way it has to be looked at."""
    def fn(run: CaseRun) -> tuple[bool, str]:
        if not run.verify:
            return True, "every answer passed verification at once"
        return False, "; ".join(f"turn {v['turn']}: {v['action']}: {v['problems']}" for v in run.verify)
    return Check("verification had nothing to correct", "verify", fn)


def failure_reported() -> Check:
    """If a tool failed in a turn, the reply of that turn must say that something failed."""
    def fn(run: CaseRun) -> tuple[bool, str]:
        failed_turns = [i for i, t in enumerate(run.turns) if any(not tool.ok for tool in t.tools)]
        missing = [i for i in failed_turns if not re.search(FAILURE_WORDS, run.turns[i].reply, re.I)]
        return not missing, ("no tool failed" if not failed_turns else
                             f"turns {missing} failed without saying so" if missing else "every failed tool was reported")
    return Check("a failed tool is reported in the reply", "invalid", fn)


# ---------- specific checks ----------

def tools_in_order(turn: int, *names: str) -> Check:
    def fn(run: CaseRun) -> tuple[bool, str]:
        called = [t.tool for t in run.turns[turn].tools]
        it = iter(called)
        return all(n in it for n in names), f"called: {called}"
    return Check(f"turn {turn + 1}: tools called in order {list(names)}", "tool", fn)


def tool_not_called(turn: int, name: str) -> Check:
    return Check(f"turn {turn + 1}: {name} not called", "tool",
                 lambda r: (all(t.tool != name for t in r.turns[turn].tools), f"called: {[t.tool for t in r.turns[turn].tools]}"))


def no_tools(turn: int) -> Check:
    return Check(f"turn {turn + 1}: answered without any tool", "memory",
                 lambda r: (not r.turns[turn].tools, f"called: {[t.tool for t in r.turns[turn].tools]}"))


def reply_matches(turn: int, pattern: str, name: str, tag: str) -> Check:
    return Check(name, tag, lambda r: (bool(re.search(pattern, r.turns[turn].reply, re.I)), f"reply: {r.turns[turn].reply[:160]!r}"))


def reply_not_matches(turn: int, pattern: str, name: str, tag: str) -> Check:
    return Check(name, tag, lambda r: (not re.search(pattern, r.turns[turn].reply, re.I), f"reply: {r.turns[turn].reply[:160]!r}"))


def custom(name: str, tag: str, fn: Callable[[CaseRun], tuple[bool, str]]) -> Check:
    return Check(name, tag, fn)


@dataclass
class Case:
    id: str
    turns: list[str]
    checks: list[Check]
    trajectory: list = field(default_factory=list)  # expected tool calls, see trajectory.py
    language: str | list[str] = "en"  # the session setting for place names (one per turn if a list), as /language would set it
