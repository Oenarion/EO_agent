"""Tracing and logging.

One JSONL file per session, traces/<session_id>.jsonl. One line per step, with the
same fields on every line (null when not applicable):

  ts, session_id, turn, event, node, tool, args, duration_ms, error,
  result_summary, final_answer, data

event is one of: request_start, node, llm_call, tool_call, request_end.

How it works: a LangChain callback handler is attached to the graph run. LangGraph and
LangChain call it for every node, model call and tool call, so the node and tool code
does not know that tracing exists. The session id and the turn travel in contextvars,
which is also how every log line gets its session id.
"""
import contextvars
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage

current_session: contextvars.ContextVar[str] = contextvars.ContextVar("current_session", default="-")
current_turn: contextvars.ContextVar[int] = contextvars.ContextVar("current_turn", default=0)

SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
NODE_NAMES = {"prepare_context", "agent", "tools", "update_memory", "verify", "cite"}
SUMMARY_MARK = "You summarize"  # first words of SUMMARY_PROMPT: tells a summary call from an agent call


# ---------- logging with the session id ----------

def _install_session_field() -> None:
    """Every log record gets .session_id, read from the contextvar."""
    old = logging.getLogRecordFactory()
    if getattr(old, "_eo_session", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old(*args, **kwargs)
        record.session_id = current_session.get()
        return record

    factory._eo_session = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


_install_session_field()


def setup_logging(level: int = logging.INFO) -> None:
    """Console logging for a process (API, scripts). Safe to call twice."""
    logger = logging.getLogger("eo_agent")
    logger.setLevel(level)
    if any(getattr(h, "_eo_handler", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler._eo_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [session=%(session_id)s] %(name)s: %(message)s"))
    logger.addHandler(handler)


# ---------- the trace file ----------

def _short(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text).split())  # one line: tool results are indented JSON
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class TraceWriter:
    def __init__(self, trace_dir: str | Path, session_id: str, secrets: list[str] | None = None):
        if not SAFE_SESSION_ID.match(session_id):
            raise ValueError("session_id may only contain letters, digits, '_', '.', '-' (max 64)")
        self.session_id = session_id
        self.path = Path(trace_dir) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._secrets = [s for s in (secrets or []) if len(s) >= 8]  # never write an API key to disk

    def emit(
        self, event: str, *, node: str | None = None, tool: str | None = None, args: Any = None,
        duration_ms: float | None = None, error: str | None = None, result_summary: str | None = None,
        final_answer: str | None = None, data: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "turn": current_turn.get(),
            "event": event,
            "node": node,
            "tool": tool,
            "args": args,
            "duration_ms": None if duration_ms is None else round(duration_ms, 1),
            "error": error,
            "result_summary": result_summary,
            "final_answer": final_answer,
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        for secret in self._secrets:
            line = line.replace(secret, "[REDACTED]")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_trace(trace_dir: str | Path, session_id: str) -> list[dict[str, Any]]:
    if not SAFE_SESSION_ID.match(session_id):
        return []
    path = Path(trace_dir) / f"{session_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- the callback handler ----------

def _summarize_node(node: str, outputs: Any) -> tuple[str | None, dict[str, Any] | None]:
    """A short description of what a node returned (and its raw numbers, if any)."""
    if not isinstance(outputs, dict):
        return None, None
    if node == "prepare_context":
        stats = outputs.get("context_stats") or {}
        return (f"model input: {stats.get('messages_sent')} messages, {stats.get('chars_sent')} chars"
                + (", summarized older turns" if stats.get("removed") else "")), stats
    if node == "agent":
        reply = (outputs.get("messages") or [None])[0]
        if isinstance(reply, AIMessage):
            if reply.tool_calls:
                return "requested tools: " + ", ".join(c["name"] for c in reply.tool_calls), None
            return f"final answer ({len(_text(reply.content))} chars)", None
    if node == "tools":
        return ", ".join(f"{m.name}:{m.status}" for m in outputs.get("messages", [])), None
    if node == "verify":
        report = outputs.get("verify_report") or {}
        action = report.get("action")
        if action == "ok":
            if report["checked"] == 0:
                return "nothing to verify in this answer (no scene ids, dates or values)", report
            return f"all {report['checked']} claims are supported by the tool results", report
        if action == "rewrite":
            return f"{len(report['problems'])} unsupported claim(s): the model is asked to rewrite", report
        if action == "warning":
            return f"still {len(report['problems'])} unsupported claim(s): a warning was added to the answer", report
        return None, None
    if node == "cite":
        added = bool(outputs.get("messages"))
        return ("added a Sources block to the answer" if added else "no known scene id in the answer, nothing added"), None
    if node == "update_memory":
        return "working memory keys: " + ", ".join(sorted((outputs.get("working_memory") or {}).keys())), None
    return None, None


class TraceHandler(AsyncCallbackHandler):
    """Turns LangChain/LangGraph callbacks into trace events. Never raises into the run."""

    raise_error = False

    def __init__(self, writer: TraceWriter):
        self.writer = writer
        self._started: dict[UUID, float] = {}
        self._info: dict[UUID, dict[str, Any]] = {}

    def _begin(self, run_id: UUID, **info: Any) -> None:
        self._started[run_id] = time.perf_counter()
        self._info[run_id] = info

    def _end(self, run_id: UUID) -> tuple[float | None, dict[str, Any]]:
        started = self._started.pop(run_id, None)
        info = self._info.pop(run_id, {})
        return (None if started is None else (time.perf_counter() - started) * 1000), info

    # nodes
    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node in NODE_NAMES and kwargs.get("name") == node:  # skips the graph itself and the routing function
            self._begin(run_id, node=node)

    async def on_chain_end(self, outputs, *, run_id, **kwargs):
        if run_id not in self._info:
            return
        duration, info = self._end(run_id)
        summary, data = _summarize_node(info["node"], outputs)
        self.writer.emit("node", node=info["node"], duration_ms=duration, result_summary=summary, data=data)

    async def on_chain_error(self, error, *, run_id, **kwargs):
        if run_id not in self._info:
            return
        duration, info = self._end(run_id)
        self.writer.emit("node", node=info["node"], duration_ms=duration, error=f"{type(error).__name__}: {error}")

    # model calls
    async def on_chat_model_start(self, serialized, messages: list[list[BaseMessage]], *, run_id, **kwargs):
        batch = messages[0] if messages else []
        is_summary = bool(batch) and _text(batch[0].content).startswith(SUMMARY_MARK)
        self._begin(
            run_id, purpose="summary" if is_summary else "agent", messages_sent=max(len(batch) - 1, 0),
            chars_sent=sum(len(_text(m.content)) for m in batch),
        )

    async def on_llm_end(self, response, *, run_id, **kwargs):
        duration, info = self._end(run_id)
        message = response.generations[0][0].message if response.generations and response.generations[0] else None
        usage = getattr(message, "usage_metadata", None) or {}
        calls = getattr(message, "tool_calls", None) or []
        summary = ("requested tools: " + ", ".join(c["name"] for c in calls)) if calls else (
            f"text answer ({len(_text(message.content))} chars)" if message is not None else None)
        self.writer.emit(
            "llm_call", node=info.get("purpose"), duration_ms=duration, result_summary=summary,
            data={**{k: info.get(k) for k in ("messages_sent", "chars_sent")},
                  "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
        )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        duration, info = self._end(run_id)
        self.writer.emit("llm_call", node=info.get("purpose"), duration_ms=duration, error=f"{type(error).__name__}: {error}",
                         data={k: info.get(k) for k in ("messages_sent", "chars_sent")})

    # tools
    async def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        self._begin(run_id, tool=(serialized or {}).get("name") or kwargs.get("name"), args=inputs if inputs is not None else input_str)

    async def on_tool_end(self, output, *, run_id, **kwargs):
        duration, info = self._end(run_id)
        failed = getattr(output, "status", None) == "error"  # the MCP adapter returns errors as a message with status=error
        text = _text(getattr(output, "content", output))
        self.writer.emit(
            "tool_call", tool=info.get("tool"), args=info.get("args"), duration_ms=duration,
            error=_short(text) if failed else None, result_summary=None if failed else _short(text, 200),
        )

    async def on_tool_error(self, error, *, run_id, **kwargs):
        duration, info = self._end(run_id)
        self.writer.emit("tool_call", tool=info.get("tool"), args=info.get("args"), duration_ms=duration,
                         error=f"{type(error).__name__}: {_short(error)}")
