"""Small evaluation: scripted questions, deterministic checks, no model as judge.

Start the MCP server first (see the README). The agent runs in this process, with the
real model and the real MCP tools. Each case is a fresh session.

    python -m evals.run_eval                  # all cases, once
    python -m evals.run_eval --repeat 3       # three runs of each case (the model is not deterministic)
    python -m evals.run_eval --cases search_basic,unknown_scene_id

The result is printed and saved to evals/last_run.json.
"""
import argparse
import asyncio
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from eo_agent.agent.context import last_human_index, message_text
from eo_agent.agent.runtime import AgentRuntime
from eo_agent.config import get_settings
from eo_agent.observability.tracing import setup_logging
from evals.cases import CASES
from evals import trajectory
from evals.checks import (
    Case, CaseRun, CheckResult, ToolRun, TurnRun, failure_reported, grounded_ids, no_crash, verification_quiet,
)

OUT = Path("evals/last_run.json")


def tool_runs_of_turn(messages: list) -> list[ToolRun]:
    """The tools called in the current turn, with the full result of each."""
    turn = messages[max(last_human_index(messages), 0):]
    results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
    runs = []
    for m in turn:
        if not isinstance(m, AIMessage):
            continue
        for call in m.tool_calls:
            result = results.get(call["id"])
            ok = result is not None and result.status != "error"
            content = message_text(result) if result is not None else ""
            data = None
            if ok:
                try:
                    data = json.loads(content)
                except ValueError:
                    data = None  # the stored message was truncated, there is nothing to parse
            runs.append(ToolRun(call["name"], call["args"], ok, content, data))
    return runs


async def run_case(runtime: AgentRuntime, case: Case, session_id: str) -> CaseRun:
    run = CaseRun(turns=[])
    for number, question in enumerate(case.turns):
        language = case.language[number] if isinstance(case.language, list) else case.language
        try:
            result = await runtime.chat(session_id, question, language)
        except Exception as exc:  # a model outage or a bug: record it, the check "no crash" fails
            run.error = f"{type(exc).__name__}: {exc}"
            break
        messages = await runtime.session_messages(session_id)
        run.turns.append(TurnRun(question, result.reply, tool_runs_of_turn(messages)))
    for event in runtime.trace(session_id):  # the verification steps that asked for a rewrite or added a warning
        if event["event"] == "node" and event["node"] == "verify" and (event.get("data") or {}).get("action") != "ok":
            run.verify.append({"turn": event["turn"], **(event.get("data") or {})})
    return run


def evaluate(case: Case, run: CaseRun) -> list[CheckResult]:
    checks = [no_crash(), grounded_ids(), failure_reported(), verification_quiet(), *case.checks]
    if run.error:  # nothing more can be judged if a request raised
        return [no_crash().run(run)]
    return [c.run(run) for c in checks]


def summarize(results: list[dict]) -> dict:
    all_checks = [c for r in results for c in r["checks"]]
    measured = [r["trajectory"] for r in results if r.get("trajectory")]
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for c in all_checks:
        by_tag[c["tag"]].append(c["ok"])
    return {
        "cases_run": len(results),
        "cases_passed": sum(r["passed"] for r in results),
        "checks_run": len(all_checks),
        "checks_passed": sum(c["ok"] for c in all_checks),
        "by_tag": {t: {"passed": sum(v), "total": len(v)} for t, v in sorted(by_tag.items())},
        "trajectory": trajectory.summarize(measured),
    }


def print_report(results: list[dict], summary: dict, model: str) -> None:
    print(f"\n{'case':<30} {'checks':>7}  result")
    print("-" * 52)
    for r in results:
        passed = sum(c["ok"] for c in r["checks"])
        print(f"{r['id'] + (' #' + str(r['run']) if r['run'] > 1 else ''):<30} {passed:>3}/{len(r['checks']):<3}  {'PASS' if r['passed'] else 'FAIL'}")
    print("-" * 52)
    print(f"Model: {model}")
    print(f"Cases:  {summary['cases_passed']}/{summary['cases_run']} passed")
    print(f"Checks: {summary['checks_passed']}/{summary['checks_run']} passed")
    for tag, v in summary["by_tag"].items():
        print(f"  {tag:<14} {v['passed']}/{v['total']}")
    t = summary["trajectory"]
    if t:
        print(f"Trajectory ({t['cases_measured']} runs with expected tool calls): tool coverage {t['tool_coverage']:.0%}, "
              f"order {t['order_ok_rate']:.0%}, arguments {t['param_accuracy']:.0%}, "
              f"extra calls per run {t['extra_calls_per_case']}, perfect paths {t['perfect_paths']}/{t['cases_measured']}")
    off_path = [(r, r["trajectory"]) for r in results if r.get("trajectory") and (
        r["trajectory"]["missing"] or r["trajectory"]["wrong_args"] or not r["trajectory"]["order_ok"] or r["trajectory"]["extra_calls"])]
    if off_path:
        print("\nRuns that left the expected path:")
        for r, m in off_path:
            print(f"  [{r['id']}] missing {m['missing']}, wrong arguments {m['wrong_args']}, extra calls {m['extra_calls']}, "
                  f"order ok {m['order_ok']}\n      expected {m['expected']}\n      actual   {m['actual']}")
    failures = [(r, c) for r in results for c in r["checks"] if not c["ok"]]
    if failures:
        print("\nFailed checks:")
        for r, c in failures:
            print(f"  [{r['id']}] {c['name']}\n      {c['detail']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1, help="runs of each case")
    parser.add_argument("--cases", default="", help="comma separated case ids (default: all)")
    args = parser.parse_args()

    selected = [c for c in CASES if not args.cases or c.id in args.cases.split(",")]
    if not selected:
        sys.exit(f"No such case. Available: {[c.id for c in CASES]}")

    setup_logging()
    settings = dataclasses.replace(get_settings(), trace_dir="evals/traces")  # keep eval traces out of traces/
    runtime = AgentRuntime(settings)
    if await runtime.ensure_graph() is None:
        sys.exit(f"Cannot reach the MCP server at {settings.mcp_url} ({runtime.mcp_error}). Start it first, see the README.")

    stamp = int(time.time())
    results: list[dict] = []
    for repeat in range(1, args.repeat + 1):
        for case in selected:
            run = await run_case(runtime, case, f"eval-{case.id}-{repeat}-{stamp}")
            checks = evaluate(case, run)
            measured = trajectory.measure(case.trajectory, run) if case.trajectory and not run.error else None
            results.append({
                "trajectory": measured,
                "id": case.id, "run": repeat, "passed": all(c.ok for c in checks),
                "checks": [dataclasses.asdict(c) for c in checks],
                "turns": [{"question": t.question, "reply": t.reply,
                           "tools": [{"tool": x.tool, "args": x.args, "ok": x.ok} for x in t.tools]} for t in run.turns],
                "error": run.error,
            })
            print(f"{case.id:<30} run {repeat}: {'PASS' if results[-1]['passed'] else 'FAIL'}", flush=True)

    summary = summarize(results)
    print_report(results, summary, settings.llm_model)
    OUT.write_text(json.dumps({"model": settings.llm_model, "date": time.strftime("%Y-%m-%d"), "repeats": args.repeat,
                               "summary": summary, "cases": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
