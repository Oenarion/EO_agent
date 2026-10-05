"""The object the API talks to. It owns what must live as long as the process:
the checkpointer, the compiled graph, and one lock per session.

One chat() call is one request: it sets the session id for the logs, writes the
request_start and request_end lines of the trace, runs the graph with the trace
handler attached, and turns the final state into a reply.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from eo_agent.agent.context import last_human_index, message_text
from eo_agent.agent.errors import MCP_DOWN_REPLY, describe
from eo_agent.agent.graph import RECURSION_LIMIT, build_graph
from eo_agent.agent.mcp_client import load_mcp_tools
from eo_agent.agent.skills import discover, index_text, make_load_skill_tool
from eo_agent.config import DEFAULT_PLACE_LANGUAGE, Settings, get_settings
from eo_agent.llm import build_llm
from eo_agent.observability.tracing import TraceHandler, TraceWriter, current_session, current_turn, read_trace

log = logging.getLogger("eo_agent.runtime")

ToolLoader = Callable[[str], Awaitable[list[BaseTool]]]


@dataclass
class ChatResult:
    session_id: str
    reply: str
    turn: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def tool_calls_of_turn(messages: list) -> list[dict[str, Any]]:
    """The tools called in the current turn, with whether each one worked."""
    turn = messages[max(last_human_index(messages), 0):]
    status = {m.tool_call_id: m.status != "error" for m in turn if isinstance(m, ToolMessage)}
    return [
        {"tool": c["name"], "args": c["args"], "ok": status.get(c["id"], False)}
        for m in turn if isinstance(m, AIMessage) for c in m.tool_calls
    ]


class AgentRuntime:
    def __init__(self, settings: Settings | None = None, llm: Any = None, tool_loader: ToolLoader = load_mcp_tools):
        self.settings = settings or get_settings()
        self._llm = llm
        self._tool_loader = tool_loader
        self._checkpointer = InMemorySaver()  # sessions live as long as this process
        self._graph: Any = None
        self._graph_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._turns: dict[str, int] = {}
        self.mcp_error: str | None = None

    @property
    def mcp_connected(self) -> bool:
        return self._graph is not None

    async def ensure_graph(self) -> Any:
        """Load the tools from the MCP server and compile the graph, once.
        If the server is down, return None and try again on the next request."""
        if self._graph is not None:
            return self._graph
        async with self._graph_lock:
            if self._graph is not None:
                return self._graph
            try:
                tools = await self._tool_loader(self.settings.mcp_url)
            except Exception as exc:  # connection refused, timeout, protocol error
                self.mcp_error = describe(exc)
                log.error("cannot load the tools from the MCP server at %s: %s", self.settings.mcp_url, self.mcp_error)
                return None
            skills = discover(self.settings.skills_dir)
            if skills:
                tools = [*tools, make_load_skill_tool(skills)]
                log.info("skills available: %s", list(skills))
            self._graph = build_graph(self._llm or build_llm(self.settings), tools, self.settings, self._checkpointer, index_text(skills))
            self.mcp_error = None
            log.info("loaded %d tools from the MCP server: %s", len(tools), [t.name for t in tools])
            return self._graph

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def chat(self, session_id: str, message: str, language: str | None = None) -> ChatResult:
        turn = self._turns.get(session_id, 0) + 1
        self._turns[session_id] = turn
        session_token, turn_token = current_session.set(session_id), current_turn.set(turn)
        writer = TraceWriter(self.settings.trace_dir, session_id, secrets=[self.settings.llm_api_key])
        started = time.perf_counter()
        writer.emit("request_start", args={"message": message})
        try:
            async with self._lock_for(session_id):  # two requests of one session never run at the same time
                graph = await self.ensure_graph()
                if graph is None:
                    writer.emit("request_end", error=f"MCP server unreachable: {self.mcp_error}", final_answer=MCP_DOWN_REPLY,
                                duration_ms=(time.perf_counter() - started) * 1000)
                    return ChatResult(session_id, MCP_DOWN_REPLY, turn)
                state = await graph.ainvoke(
                    {"messages": [("user", message)], "step_count": 0, "verify_retries": 0,  # both restart every turn
                     **({"place_language": language} if language else {})},  # no language given: the session keeps its setting
                    config={"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT,
                            "callbacks": [TraceHandler(writer)]},
                )
            reply = message_text(state["messages"][-1])
            calls = tool_calls_of_turn(state["messages"])
            writer.emit("request_end", final_answer=reply, duration_ms=(time.perf_counter() - started) * 1000,
                        data={"tool_calls": calls})
            return ChatResult(session_id, reply, turn, calls)
        except Exception as exc:
            writer.emit("request_end", error=describe(exc), duration_ms=(time.perf_counter() - started) * 1000)
            raise
        finally:
            current_session.reset(session_token)
            current_turn.reset(turn_token)

    async def session_info(self, session_id: str) -> dict[str, Any] | None:
        if self._graph is None:
            return None
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": session_id}})
        if not snapshot.values:
            return None
        return {
            "session_id": session_id,
            "turns": self._turns.get(session_id, 0),
            "messages_in_state": len(snapshot.values.get("messages", [])),
            "working_memory": snapshot.values.get("working_memory") or {},
            "summary": snapshot.values.get("summary", ""),
            "place_language": snapshot.values.get("place_language", DEFAULT_PLACE_LANGUAGE),
        }

    async def session_messages(self, session_id: str) -> list:
        """The messages stored for a session (used by the evaluation to read the full tool results)."""
        if self._graph is None:
            return []
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": session_id}})
        return list(snapshot.values.get("messages", []))

    def trace(self, session_id: str) -> list[dict[str, Any]]:
        return read_trace(self.settings.trace_dir, session_id)
