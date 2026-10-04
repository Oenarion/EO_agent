"""Print a trace as a readable table, grouped by turn.

    python -m eo_agent.observability.pretty traces/my-session.jsonl
    python -m eo_agent.observability.pretty my-session          (looks in TRACE_DIR)
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from eo_agent.config import get_settings

COLUMNS = ("t+", "event", "step", "ms", "status", "details")


def _seconds(ts: str, start: str) -> float:
    return (datetime.fromisoformat(ts) - datetime.fromisoformat(start)).total_seconds()


def _row(event: dict[str, Any], start: str) -> tuple[str, ...]:
    kind = event["event"]
    step = event.get("tool") or event.get("node") or ""
    if kind == "llm_call" and step:
        step = f"model ({step})"
    status = "ERROR" if event.get("error") else "ok"
    if kind == "tool_call":
        detail = json.dumps(event.get("args"), ensure_ascii=False)
        detail += f" -> {event['error']}" if event.get("error") else f" -> {event.get('result_summary') or ''}"
    elif kind == "request_start":
        detail = (event.get("args") or {}).get("message", "")
    else:
        detail = event.get("error") or event.get("result_summary") or ""
        data = event.get("data") or {}
        if kind == "llm_call" and data.get("input_tokens"):
            detail += f" [sent {data.get('messages_sent')} msgs, {data.get('input_tokens')} tokens in]"
    ms = "" if event.get("duration_ms") is None else f"{event['duration_ms']:.0f}"
    return (f"{_seconds(event['ts'], start):6.2f}", kind, step, ms, status, detail)


def render(events: list[dict[str, Any]], width: int = 150) -> str:
    if not events:
        return "(empty trace)"
    out: list[str] = [f"Session: {events[0]['session_id']}"]
    for turn in sorted({e["turn"] for e in events}):
        turn_events = [e for e in events if e["turn"] == turn]
        start = turn_events[0]["ts"]
        out += ["", f"=== Turn {turn} " + "=" * 60]
        rows = [_row(e, start) for e in turn_events]
        widths = [max(len(COLUMNS[i]), *(len(r[i]) for r in rows)) for i in range(5)]
        out.append("  ".join(COLUMNS[i].ljust(widths[i]) for i in range(5)) + "  " + COLUMNS[5])
        room = max(width - sum(widths) - 10, 30)
        for r in rows:
            detail = r[5] if len(r[5]) <= room else r[5][: room - 3] + "..."
            out.append("  ".join(r[i].ljust(widths[i]) for i in range(5)) + "  " + detail)
        for e in turn_events:
            if e["event"] == "request_end":
                out += ["", "FINAL ANSWER:", e.get("final_answer") or "(none)"]
                if e.get("error"):
                    out.append(f"REQUEST ERROR: {e['error']}")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    path = Path(arg) if arg.endswith(".jsonl") else Path(get_settings().trace_dir) / f"{arg}.jsonl"
    if not path.exists():
        sys.exit(f"No trace file at {path}")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(render(events))


if __name__ == "__main__":
    main()
