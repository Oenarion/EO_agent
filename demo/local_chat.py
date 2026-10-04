"""Phase 2 check: run the agent in this process, tools through the MCP server.

Start the MCP server first:   python -m eo_agent.mcp_server.server
Then run:                     python demo/local_chat.py
Or with your own questions:   python demo/local_chat.py "question 1" "question 2"
Low thresholds, to see the context policy work:   MAX_WINDOW=4 SUMMARY_TRIGGER=6 python demo/local_chat.py ...

All turns share one session, so follow-ups use the kept context.
"""
import asyncio
import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.agent.mcp_client import load_mcp_tools
from eo_agent.config import get_settings
from eo_agent.llm import build_llm

DEFAULT_TURNS = [
    "Find Sentinel-2 scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover.",
    "Give me the details of the second one.",
]


def print_turn(messages: list, start: int) -> str:
    """Print the tool calls of this turn and return the final answer."""
    for m in messages[start:]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                print(f"   tool call: {c['name']}({json.dumps(c['args'])})")
        elif isinstance(m, ToolMessage):
            print(f"   tool result [{m.status}]: {str(m.content)[:150]}")
    return messages[-1].content


async def main() -> None:
    settings = get_settings()
    tools = await load_mcp_tools(settings.mcp_url)
    print("Tools from MCP:", [t.name for t in tools])
    graph = build_graph(build_llm(settings), tools, settings)

    session = "local-demo"
    turns = sys.argv[1:] or DEFAULT_TURNS
    for n, question in enumerate(turns, start=1):
        print(f"\n=== Turn {n}\nUser: {question}")
        before = len((await graph.aget_state({"configurable": {"thread_id": session}})).values.get("messages", []))
        state = await run_turn(graph, session, question)
        answer = print_turn(state["messages"], before + 1)
        print(f"Agent: {answer}")
        st = state["context_stats"]
        print(
            f"   context: in state={st['messages_in_state']}, sent={st['messages_sent']}, chars sent={st['chars_sent']}, "
            f"summarized this turn={st['summarized']}"
            + (f", input tokens={st['input_tokens']}" if st.get("input_tokens") else "")
        )

    print("\n=== Summary at the end")
    print(state.get("summary") or "(none)")
    print("\n=== Working memory at the end")
    print(json.dumps(state["working_memory"], indent=1))


if __name__ == "__main__":
    asyncio.run(main())
