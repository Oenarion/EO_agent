"""Verification at work, by fault injection.

The real model almost never writes an unsupported value, so there is nothing to catch in a normal run.
This demo wraps the real model so that its FIRST final answer is damaged on purpose (a scene id and a
percentage are replaced by values that no tool returned). Then the verify step of the agent is on its own:
it should catch the damage, ask the model to write the answer again, and the real model should give a good one.

The MCP server must be running (see the README).

    python demo/verify_demo.py
"""
import asyncio
import dataclasses
import re
import sys

from langchain_core.messages import AIMessage

from eo_agent.agent.citations import SCENE_ID
from eo_agent.agent.runtime import AgentRuntime
from eo_agent.agent.verify import PERCENT
from eo_agent.config import get_settings
from eo_agent.llm import build_llm
from eo_agent.observability.tracing import setup_logging

QUESTION = "Find Sentinel-2 scenes over Ravenna, Italy in July 2025 with less than 10% cloud cover."
INVENTED_ID = "S2A_32TQQ_20250101_0_L2A"


def damage(text: str) -> str:
    text = SCENE_ID.sub(INVENTED_ID, text, count=1)
    return PERCENT.sub(lambda m: "17.3%", text, count=1)


class Saboteur:
    """Wraps a chat model. The first final answer (a reply without tool calls) is damaged, once."""

    def __init__(self, model):
        self.model, self.damaged = model, False

    def bind_tools(self, tools):
        return _Bound(self.model.bind_tools(tools), self)

    async def ainvoke(self, messages):
        return await self.model.ainvoke(messages)


class _Bound:
    def __init__(self, bound, parent: Saboteur):
        self.bound, self.parent = bound, parent

    async def ainvoke(self, messages):
        reply = await self.bound.ainvoke(messages)
        if not reply.tool_calls and not self.parent.damaged:
            self.parent.damaged = True
            print(f"  [the model wrote]\n    {reply.content[:300]!r}")
            reply = AIMessage(content=damage(reply.content), id=reply.id)
            print(f"  [after the damage, this is what goes to verification]\n    {reply.content[:300]!r}\n")
        return reply


async def main() -> None:
    setup_logging()
    settings = dataclasses.replace(get_settings(), trace_dir="evals/traces")
    runtime = AgentRuntime(settings, llm=Saboteur(build_llm(settings)))
    if await runtime.ensure_graph() is None:
        sys.exit(f"Cannot reach the MCP server at {settings.mcp_url} ({runtime.mcp_error}). Start it first, see the README.")
    print(f"Question: {QUESTION}\n")
    result = await runtime.chat("verify-demo", QUESTION)
    print("=== What the verify step did")
    for e in runtime.trace("verify-demo"):
        if e["event"] == "node" and e["node"] == "verify":
            print(f"  verify: {e['result_summary']}")
            for problem in (e.get("data") or {}).get("problems", []):
                print(f"      - {problem}")
    print("\n=== The answer the user gets\n")
    print(result.reply.split("\n\nSources:")[0])


if __name__ == "__main__":
    asyncio.run(main())
