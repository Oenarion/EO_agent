"""Trajectory metrics: not only whether the answer is right, but whether the agent took the right path.

Each case can declare the tool calls it expects, in order. After a run, four numbers are computed
(the same idea as the step-by-step metrics of the Earth-Bench benchmark):

  tool_coverage   share of the expected calls (tool and turn) that happened at all
  order_ok        whether the expected calls that happened are in the expected order
  param_accuracy  share of the expected calls that happened with the right arguments
  extra_calls     calls beyond the expected ones (a measure of wasted steps)

A right answer reached by a wrong path is a warning sign: it may not repeat.
"""
from dataclasses import dataclass
from typing import Any, Callable

from evals.checks import CaseRun

ArgsCheck = Callable[[dict[str, Any], CaseRun], bool]


@dataclass
class ExpectedCall:
    tool: str
    check: ArgsCheck | None = None  # a test of the arguments, given the arguments and the whole session
    turn: int = 0                   # the turn (0 based) in which the call is expected
    note: str = ""
    unordered: bool = False         # expected, but it may come before or after the others (it is left out of the order test)


def _holds(check: ArgsCheck | None, args: dict[str, Any], run: CaseRun) -> bool:
    if check is None:
        return True
    try:
        return bool(check(args, run))
    except (KeyError, IndexError, TypeError, AttributeError):
        return False  # the arguments or the data the check needs are not there


def measure(expected: list[ExpectedCall], run: CaseRun) -> dict[str, Any]:
    actual = [(turn, tool) for turn, t in enumerate(run.turns) for tool in t.tools]
    in_turn = lambda e: [(k, tool) for k, (turn, tool) in enumerate(actual) if turn == e.turn and tool.tool == e.tool]

    happened = [bool(in_turn(e)) for e in expected]
    right_args = [any(_holds(e.check, tool.args, run) for _, tool in in_turn(e)) for e in expected]

    # order: walk through the actual calls once, matching the expected calls that happened
    position, ordered = -1, True
    for e, did in zip(expected, happened):
        if not did or e.unordered:
            continue
        later = [k for k, _ in in_turn(e) if k > position]
        if not later:
            ordered = False
            break
        position = later[0]

    n = len(expected)
    return {
        "expected": [f"turn {e.turn + 1}: {e.tool}" + (f" ({e.note})" if e.note else "") for e in expected],
        "actual": [f"turn {turn + 1}: {tool.tool}" + ("" if tool.ok else " (failed)") for turn, tool in actual],
        "tool_coverage": sum(happened) / n,
        "order_ok": bool(any(happened)) and ordered,
        "param_accuracy": sum(right_args) / n,
        "extra_calls": max(len(actual) - n, 0),
        "missing": [e.tool for e, did in zip(expected, happened) if not did],
        "wrong_args": [e.tool for e, did, ok in zip(expected, happened, right_args) if did and not ok],
    }


def summarize(measured: list[dict[str, Any]]) -> dict[str, Any]:
    if not measured:
        return {}
    n = len(measured)
    return {
        "cases_measured": n,
        "tool_coverage": round(sum(m["tool_coverage"] for m in measured) / n, 3),
        "order_ok_rate": round(sum(m["order_ok"] for m in measured) / n, 3),
        "param_accuracy": round(sum(m["param_accuracy"] for m in measured) / n, 3),
        "extra_calls_per_case": round(sum(m["extra_calls"] for m in measured) / n, 2),
        "perfect_paths": sum(1 for m in measured if m["tool_coverage"] == 1 and m["order_ok"] and m["param_accuracy"] == 1 and m["extra_calls"] == 0),
    }
