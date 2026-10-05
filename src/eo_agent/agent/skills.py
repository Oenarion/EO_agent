"""Agent Skills: instructions that are loaded only when a request needs them.

A skill is a directory with a SKILL.md file. The file starts with a header between two lines of
three dashes, with a name and a description. The agent always sees the name and the description
(a few lines in the system prompt). The full text is loaded by the load_skill tool, only when the
request matches the description, so a skill costs nothing on the turns that do not need it. The
text arrives as a tool result, so the context policy replaces it with a one-line stub on the next turn.

This is a part of the agent and not of the MCP server: a skill is guidance for the model, and not
a source of data.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel

log = logging.getLogger("eo_agent.skills")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def parse(text: str) -> Skill | None:
    """The header is read by hand (name and description, one line each): no extra dependency."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None
    header = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()
    if not header.get("name") or not header.get("description"):
        return None
    return Skill(header["name"], header["description"], "\n".join(lines[end + 1:]).strip())


def discover(directory: str | Path | None) -> dict[str, Skill]:
    """All the valid skills under a directory. A broken one is skipped with a warning."""
    skills: dict[str, Skill] = {}
    if not directory or not Path(directory).is_dir():
        return skills
    for path in sorted(Path(directory).glob("*/SKILL.md")):
        skill = parse(path.read_text(encoding="utf-8"))
        if skill is None:
            log.warning("skipping %s: it needs a header with a name and a description", path)
        elif skill.name != path.parent.name:
            log.warning("skipping %s: the name '%s' differs from the folder name", path, skill.name)
        else:
            skills[skill.name] = skill
    return skills


def index_text(skills: dict[str, Skill]) -> str:
    """What the system prompt carries about the skills: only names and descriptions."""
    if not skills:
        return ""
    lines = ["Skills. Each one is a set of instructions for a kind of request. Load one with load_skill only when the request "
             "matches its description. Do not load a skill for other requests. When you load one, follow it:"]
    lines += [f"- {s.name}: {s.description}" for s in skills.values()]
    return "\n".join(lines)


class LoadSkillArgs(BaseModel):
    name: str


def make_load_skill_tool(skills: dict[str, Skill]) -> StructuredTool:
    async def load_skill(name: str) -> str:
        skill = skills.get(name.strip())
        if skill is None:
            raise ToolException(f"Unknown skill '{name}'. Available skills: {', '.join(skills) or 'none'}.")
        return skill.body

    return StructuredTool(
        name="load_skill",
        description="Load the full instructions of a skill, by name. Use it only when the user's request matches the "
                    "description of that skill, and then follow the instructions.",
        args_schema=LoadSkillArgs, coroutine=load_skill, handle_tool_error=True,
    )
