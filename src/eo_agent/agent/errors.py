"""Everything about the error path in one place.

The rules:
  - a failing tool never crashes the turn: it becomes a tool message with status "error"
  - the model then writes the reply for the user
  - if the model returns nothing, a fixed template says what was attempted and that it failed
  - if the MCP server cannot be reached at all, the reply is fixed (the model has no tools)
"""
import json

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

MCP_DOWN_REPLY = (
    "I cannot reach the Sentinel-2 catalogue service (the MCP server) right now, so I cannot search "
    "for scenes or read scene details. Please check that the MCP server is running and try again."
)
EMPTY_REPLY = "I could not produce an answer. Please try rephrasing your question."


def root_cause(exc: BaseException) -> BaseException:
    """The async libraries wrap the real error in an ExceptionGroup ("unhandled errors in a
    TaskGroup"). Dig down to the exception that explains what went wrong."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def describe(exc: BaseException) -> str:
    cause = root_cause(exc)
    return f"{type(cause).__name__}: {cause}"


def failure_text(tool: str, args: dict, reason: str) -> str:
    return f"I tried to call {tool} with {json.dumps(args)} and it failed: {reason}."


def tool_error_message(tool: str, call_id: str, args: dict, reason: str) -> ToolMessage:
    """The tool message the model sees when a tool failed. The fields are also kept
    in additional_kwargs, so the fixed template does not have to parse text."""
    return ToolMessage(
        content=f"Tool '{tool}' failed. Arguments: {json.dumps(args)}. Error: {reason}",
        name=tool, tool_call_id=call_id, status="error",
        additional_kwargs={"tool": tool, "args": args, "error": reason},
    )


def safety_net_reply(messages: list[BaseMessage]) -> str:
    """Used only when the model returned an empty answer."""
    current = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    for m in reversed(messages[current:]):
        if isinstance(m, ToolMessage) and m.status == "error" and m.additional_kwargs.get("tool"):
            k = m.additional_kwargs
            return failure_text(k["tool"], k["args"], k["error"])
    return EMPTY_REPLY
