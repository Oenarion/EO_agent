"""The same question, asked twice: without the skill and with it.

The agent runs in this process, with the real model and the MCP server, so the MCP server must be running.

    python demo/skill_demo.py
    python demo/skill_demo.py "your own question about choosing a scene"

The skill is not put in the prompt: the prompt only has its name and a one-line description. The model
decides to load it (a call to load_skill, visible below) when the question matches the description.
"""
import asyncio
import dataclasses
import sys

from eo_agent.agent.runtime import AgentRuntime
from eo_agent.config import get_settings
from eo_agent.observability.tracing import setup_logging

QUESTION = "Which scene should I use over Ravenna, Italy in July 2025 for monitoring crops?"


async def ask(settings, session: str, question: str) -> None:
    runtime = AgentRuntime(settings)
    if await runtime.ensure_graph() is None:
        sys.exit(f"Cannot reach the MCP server at {settings.mcp_url} ({runtime.mcp_error}). Start it first, see the README.")
    result = await runtime.chat(session, question)
    print("  tools called:", [c["tool"] for c in result.tool_calls])
    print("  answer:\n")
    print("    " + result.reply.split("\n\nSources:")[0].replace("\n", "\n    "))
    print()


async def main() -> None:
    setup_logging()
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    base = dataclasses.replace(get_settings(), trace_dir="evals/traces")
    print(f"Question: {question}\n")
    print("=== WITHOUT the skill (no skills directory)")
    await ask(dataclasses.replace(base, skills_dir=""), "skill-demo-off", question)
    print("=== WITH the skill (scene-selection, loaded on demand)")
    await ask(base, "skill-demo-on", question)


if __name__ == "__main__":
    asyncio.run(main())
