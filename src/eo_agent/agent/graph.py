"""The LangGraph agent.

    START -> prepare_context -> agent -+-> END               (no tool calls: final answer)
                  ^                    |
                  |                    +-> tools -> update_memory
                  +----------------------------------------+

The loop also stops when step_count reaches max_steps: the agent node then
calls the model WITHOUT tools and with a note telling it to answer and say it
stopped.
"""
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from eo_agent.agent.context import (
    build_model_input, cut_for_summary, message_chars, message_text, summarize, truncate_tool_content,
)
from eo_agent.agent.errors import describe, safety_net_reply, tool_error_message
from eo_agent.agent.state import AgentState, update_working_memory
from eo_agent.config import Settings, get_settings

log = logging.getLogger("eo_agent.agent")

# One tool iteration is 4 graph steps (prepare, agent, tools, update_memory).
RECURSION_LIMIT = 100


def _text_of(content: Any) -> str:
    """Join the text blocks of a tool message content."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def _error_message(call: dict[str, Any], detail: str) -> ToolMessage:
    # The session id is added to this log line automatically (see observability/tracing.py).
    log.error("tool=%s args=%s error=%s", call["name"], json.dumps(call["args"]), detail)
    return tool_error_message(call["name"], call["id"], call["args"], detail)


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    settings: Settings | None = None,
    checkpointer: Any = None,
):
    settings = settings or get_settings()
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    async def prepare_context(state: AgentState) -> dict:
        """Apply the context policy: summarize old turns if needed, then build the model input."""
        update: dict[str, Any] = {}
        messages = state["messages"]
        summary = state.get("summary", "")
        in_state = len(messages)
        cut = cut_for_summary(messages, settings.max_window, settings.summary_trigger)
        if cut:
            old = messages[:cut]
            summary = await summarize(llm, summary, old, settings)
            update["summary"] = summary
            update["messages"] = [RemoveMessage(id=m.id) for m in old]  # deletes them from the state
            messages = messages[cut:]
        step_count = state.get("step_count", 0)
        model_input = build_model_input(messages, state.get("working_memory") or {}, summary, step_count, settings)
        # "summarized" stays true for the rest of the turn, so the last call of the turn still reports it
        carried = step_count > 0 and (state.get("context_stats") or {}).get("summarized", False)
        update["model_input"] = model_input
        update["context_stats"] = {
            "messages_in_state": in_state,        # stored history, before this call's removal
            "removed": cut,                       # messages folded into the summary on this call
            "messages_sent": len(model_input) - 1,  # without the system message
            "chars_sent": sum(message_chars(m) for m in model_input),
            "summarized": bool(cut) or carried,
            "summary_chars": len(summary),
        }
        return update

    async def agent(state: AgentState) -> dict:
        limit_reached = state.get("step_count", 0) >= settings.max_steps
        model = llm if limit_reached else llm_with_tools  # no tools bound once the limit is hit
        reply = await model.ainvoke(state["model_input"])
        if not getattr(reply, "tool_calls", None) and not message_text(reply).strip():
            # Safety net: never return an empty answer. After a tool error this says what was attempted.
            log.warning("the model returned an empty answer, using the fixed reply")
            reply = AIMessage(content=safety_net_reply(state["messages"]))
        stats = dict(state.get("context_stats") or {})
        usage = getattr(reply, "usage_metadata", None) or {}  # tokens, only if the provider reports them
        if usage:
            stats["input_tokens"], stats["output_tokens"] = usage.get("input_tokens"), usage.get("output_tokens")
        log.info("model_call %s", json.dumps(stats))
        return {"messages": [reply], "context_stats": stats}

    async def tools_node(state: AgentState) -> dict:
        results: list[ToolMessage] = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                results.append(_error_message(call, "no such tool"))
                continue
            try:
                raw = await tool.ainvoke(call)
            except Exception as exc:  # connection refused, timeout, protocol error: never crash the turn
                results.append(_error_message(call, describe(exc)))
                continue
            if raw.status == "error":
                results.append(_error_message(call, _text_of(raw.content)))
                continue
            data = (raw.artifact or {}).get("structured_content")
            content = json.dumps(data, separators=(",", ":")) if data is not None else _text_of(raw.content)
            results.append(ToolMessage(content=content, name=call["name"], tool_call_id=call["id"]))
        return {"messages": results, "step_count": state.get("step_count", 0) + 1}

    async def update_memory(state: AgentState) -> dict:
        """No LLM here: read this iteration's tool results and update the working memory."""
        messages = state["messages"]
        ai_index = max(i for i, m in enumerate(messages) if isinstance(m, AIMessage) and m.tool_calls)
        calls = {c["id"]: c for c in messages[ai_index].tool_calls}
        memory = dict(state.get("working_memory") or {})
        for m in messages[ai_index + 1:]:
            if isinstance(m, ToolMessage) and m.status != "error":
                try:
                    memory = update_working_memory(memory, m.name, calls[m.tool_call_id]["args"], json.loads(m.content))
                except (KeyError, ValueError, TypeError) as exc:
                    log.warning("could not update memory from %s: %s", m.name, exc)
        # Truncate only now: the memory above has already read the full JSON.
        capped = [
            ToolMessage(id=m.id, content=truncate_tool_content(m.content, settings.max_tool_chars),
                        name=m.name, tool_call_id=m.tool_call_id, status=m.status)
            for m in messages[ai_index + 1:]
            if isinstance(m, ToolMessage) and len(m.content) > settings.max_tool_chars
        ]
        return {"working_memory": memory, "messages": capped}  # same id: replaces the stored message

    def route_after_agent(state: AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("update_memory", update_memory)
    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "update_memory")
    graph.add_edge("update_memory", "prepare_context")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


async def run_turn(graph, session_id: str, message: str) -> dict:
    """Run one user turn. thread_id is the session id, so the checkpointer keeps the session."""
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=message)], "step_count": 0},  # step_count resets every turn
        config={"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT},
    )
