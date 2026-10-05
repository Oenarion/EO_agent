"""Agent Skills: discovery, the index in the prompt, loading on demand, and the context policy. Offline."""
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.agent.runtime import AgentRuntime
from eo_agent.agent.skills import Skill, discover, index_text, make_load_skill_tool, parse
from eo_agent.api.main import create_app
from eo_agent.config import DEFAULT_SKILLS_DIR, Settings
from test_error_path import ScriptedChatModel, post
from test_graph import ScriptedLLM, call, tools_ok

GOOD = "---\nname: demo\ndescription: Use it when the user asks for a demo.\n---\n\n# Demo\nBODY-OF-THE-DEMO-SKILL: do this, then that.\n"


def write_skill(root: Path, folder: str, text: str) -> None:
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(text, encoding="utf-8")


@pytest.fixture
def skills_dir(tmp_path):
    write_skill(tmp_path, "demo", GOOD)
    return tmp_path


# ---------- reading the files ----------

def test_a_skill_header_gives_name_description_and_body():
    skill = parse(GOOD)
    assert skill == Skill("demo", "Use it when the user asks for a demo.", "# Demo\nBODY-OF-THE-DEMO-SKILL: do this, then that.")


@pytest.mark.parametrize("text", [
    "no header at all",
    "---\nname: x\n---\nbody",                       # no description
    "---\ndescription: d\n---\nbody",                # no name
    "---\nname: x\ndescription: d\nbody without a closing line",
    "",
])
def test_a_file_without_a_valid_header_is_not_a_skill(text):
    assert parse(text) is None


def test_discover_finds_the_good_skills_and_skips_the_broken_ones_with_a_warning(tmp_path, caplog):
    write_skill(tmp_path, "demo", GOOD)
    write_skill(tmp_path, "broken", "no header")
    write_skill(tmp_path, "other-folder", GOOD)  # the name inside is "demo", the folder says otherwise
    with caplog.at_level(logging.WARNING, logger="eo_agent.skills"):
        skills = discover(tmp_path)
    assert list(skills) == ["demo"]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_no_directory_or_an_empty_setting_means_no_skills(tmp_path):
    assert discover(tmp_path / "missing") == {} and discover("") == {} and discover(None) == {}


# ---------- the index in the prompt carries names and descriptions only ----------

def test_the_index_has_the_description_and_never_the_body(skills_dir):
    text = index_text(discover(skills_dir))
    assert "- demo: Use it when the user asks for a demo." in text and "load_skill" in text
    assert "BODY-OF-THE-DEMO-SKILL" not in text
    assert index_text({}) == ""


# ---------- the tool ----------

async def test_load_skill_returns_the_body_and_an_unknown_name_is_an_error_message(skills_dir):
    tool = make_load_skill_tool(discover(skills_dir))
    ok = await tool.ainvoke({"type": "tool_call", "id": "1", "name": "load_skill", "args": {"name": "demo"}})
    assert "BODY-OF-THE-DEMO-SKILL" in ok.content and ok.status != "error"
    bad = await tool.ainvoke({"type": "tool_call", "id": "2", "name": "load_skill", "args": {"name": "nope"}})
    assert bad.status == "error" and "Available skills: demo" in bad.content


# ---------- the skill of the repository ----------

def test_the_scene_selection_skill_of_the_repository_is_valid_and_honest():
    skills = discover(DEFAULT_SKILLS_DIR)
    skill = skills["scene-selection"]
    assert "Do not use it for a plain search" in skill.description  # tells when NOT to load it
    assert len(skill.body) < 3000  # well under the cap of a stored tool message (6000), and cheap to load
    assert "whole scene" in skill.body and "Never say \"the best scene\"" in skill.body  # the limit of the cloud figure is part of it
    assert "get_scene_details" in skill.body


# ---------- inside the graph ----------

def graph_with_skill(llm, skills_dir):
    skills = discover(skills_dir)
    return build_graph(llm, [*tools_ok(), make_load_skill_tool(skills)], Settings(), skills_index=index_text(skills))


async def test_a_plain_turn_sees_the_index_but_not_the_body(skills_dir):
    llm = ScriptedLLM([AIMessage(content="hello")])
    await run_turn(graph_with_skill(llm, skills_dir), "s", "hi")
    system = llm.calls[0]["messages"][0].content
    assert "- demo: Use it when the user asks for a demo." in system and "BODY-OF-THE-DEMO-SKILL" not in system


async def test_the_body_arrives_as_a_tool_result_and_the_next_turn_only_keeps_a_stub(skills_dir):
    llm = ScriptedLLM([call("load_skill", {"name": "demo"}, "c1"), AIMessage(content="done with the skill"), AIMessage(content="next")])
    graph = graph_with_skill(llm, skills_dir)
    await run_turn(graph, "s", "give me the demo")
    # inside the turn, the model got the full text with the tool result
    assert any("BODY-OF-THE-DEMO-SKILL" in str(m.content) for m in llm.calls[1]["messages"] if isinstance(m, ToolMessage))
    await run_turn(graph, "s", "and now?")
    next_turn = llm.calls[2]["messages"]
    assert not any("BODY-OF-THE-DEMO-SKILL" in str(m.content) for m in next_turn)  # gone from the context
    size = len(parse(GOOD).body)
    assert any(isinstance(m, ToolMessage) and m.content == f"load_skill('demo'): instructions loaded ({size} chars)" for m in next_turn)


async def test_a_skill_does_not_trouble_the_working_memory(skills_dir, caplog):
    llm = ScriptedLLM([call("load_skill", {"name": "demo"}, "c1"), AIMessage(content="ok")])
    with caplog.at_level(logging.WARNING, logger="eo_agent"):
        state = await run_turn(graph_with_skill(llm, skills_dir), "s", "demo please")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # plain text is not parsed as JSON
    assert not state.get("working_memory")


# ---------- the runtime adds the tool and the index ----------

def test_the_runtime_gives_the_agent_the_skills_it_finds(skills_dir, tmp_path):
    async def loader(url):
        return tools_ok()

    settings = Settings(trace_dir=str(tmp_path / "t"), skills_dir=str(skills_dir))
    model = ScriptedChatModel(replies=[call("load_skill", {"name": "demo"}, "c1"), AIMessage(content="Followed the skill.")])
    runtime = AgentRuntime(settings, llm=model, tool_loader=loader)
    with TestClient(create_app(runtime)) as client:
        body = post(client, "s", "give me the demo").json()
        assert body["reply"] == "Followed the skill."
        assert body["tool_calls"] == [{"tool": "load_skill", "args": {"name": "demo"}, "ok": True}]
        events = client.get("/traces/s").json()["events"]
    assert any(e["event"] == "tool_call" and e["tool"] == "load_skill" for e in events)


def test_no_skills_directory_means_no_skill_tool(tmp_path):
    async def loader(url):
        return tools_ok()

    settings = Settings(trace_dir=str(tmp_path), skills_dir="")
    runtime = AgentRuntime(settings, llm=ScriptedChatModel(replies=[AIMessage(content="hi")]), tool_loader=loader)
    with TestClient(create_app(runtime)) as client:
        assert post(client, "s", "hi").status_code == 200
    assert runtime._graph is not None  # started fine without skills
