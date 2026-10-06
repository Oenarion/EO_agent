"""Graph tests with a scripted model and fake tools. No network, no real LLM."""
import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.agent.state import update_working_memory
from eo_agent.config import Settings

BBOX = [12.1381, 44.3684, 12.2643, 44.4585]
GEOCODE_DATA = {"result": [
    {"name": "Ravenna", "country": "Italy", "region": "Emilia-Romagna", "bbox": BBOX},
    {"name": "Ravenna", "country": "United States", "region": "Ohio", "bbox": [-81.3, 41.1, -81.2, 41.2]},
]}
SEARCH_DATA = {
    "query": {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31", "min_cloud_cover": 0.0, "max_cloud_cover": 10.0},
    "total_found": 5,
    "scenes": [
        {"index": 1, "id": "SCENE_A", "datetime": "2025-07-31T10:00:00Z", "cloud_cover": 1.4, "tile": "32TQQ"},
        {"index": 2, "id": "SCENE_B", "datetime": "2025-07-23T10:00:00Z", "cloud_cover": 2.4, "tile": "32TQQ"},
    ],
}


class Anything(BaseModel):
    model_config = {"extra": "allow"}


class SceneArgs(BaseModel):
    scene_id: str


def fake_tool(name: str, handler, schema=Anything) -> StructuredTool:
    """Mimics what langchain-mcp-adapters returns: text blocks plus structured_content as artifact."""
    async def run(**kwargs):
        data = handler(**kwargs)
        return [{"type": "text", "text": json.dumps(data, indent=1)}], {"structured_content": data}

    return StructuredTool(name=name, description=name, args_schema=schema, coroutine=run, response_format="content_and_artifact")


def failing_tool(name: str, exc: Exception) -> StructuredTool:
    async def run(**kwargs):
        raise exc

    return StructuredTool(name=name, description=name, args_schema=Anything, coroutine=run, response_format="content_and_artifact")


class ScriptedLLM:
    """Returns pre-written replies in order. Records every input and whether tools were bound."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def bind_tools(self, tools):
        return _Bound(self, with_tools=True)

    async def ainvoke(self, messages):
        return self._reply(messages, with_tools=False)

    def _reply(self, messages, with_tools):
        self.calls.append({"messages": messages, "with_tools": with_tools})
        return self.replies.pop(0)


class _Bound:
    def __init__(self, parent: ScriptedLLM, with_tools: bool):
        self.parent, self.with_tools = parent, with_tools

    async def ainvoke(self, messages):
        return self.parent._reply(messages, self.with_tools)


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def tools_ok():
    return [
        fake_tool("geocode_place", lambda **kw: GEOCODE_DATA),
        fake_tool("search_scenes", lambda **kw: SEARCH_DATA),
        fake_tool("get_scene_details", lambda scene_id: {"id": scene_id}, SceneArgs),
    ]


def settings(**kw) -> Settings:
    return Settings(max_steps=kw.get("max_steps", 6))


async def test_tool_loop_runs_tools_then_answers_and_fills_memory():
    llm = ScriptedLLM([
        call("geocode_place", {"name": "Ravenna"}, "c1"),
        call("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}, "c2"),
        AIMessage(content="Found 2 scenes."),
    ])
    graph = build_graph(llm, tools_ok(), settings())
    state = await run_turn(graph, "s1", "find scenes")

    assert state["messages"][-1].content == "Found 2 scenes."
    assert state["step_count"] == 2
    wm = state["working_memory"]
    assert wm["place"]["country"] == "Italy"  # candidate matched by the bbox the model used
    assert [s["id"] for s in wm["last_results"]] == ["SCENE_A", "SCENE_B"]
    assert wm["date_range"] == {"start": "2025-07-01", "end": "2025-07-31"}
    # stored tool messages are compact JSON, not the indented blocks
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert all(isinstance(m.content, str) and "\n" not in m.content for m in tool_msgs)


async def test_follow_up_in_same_session_sees_memory_in_model_input():
    llm = ScriptedLLM([
        call("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}, "c1"),
        AIMessage(content="done"),
        call("get_scene_details", {"scene_id": "SCENE_B"}, "c2"),
        AIMessage(content="details"),
    ])
    graph = build_graph(llm, tools_ok(), settings())
    await run_turn(graph, "s1", "search")
    state = await run_turn(graph, "s1", "details of the second one")

    # the first model call of turn 2 already had the numbered list in its system message
    turn2_first_input = llm.calls[2]["messages"]
    assert isinstance(turn2_first_input[0], SystemMessage)
    assert "2. SCENE_B" in turn2_first_input[0].content
    assert state["working_memory"]["selected_scene_id"] == "SCENE_B"
    assert state["step_count"] == 1  # reset at the start of the turn


async def test_sessions_are_isolated():
    llm = ScriptedLLM([
        call("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}, "c1"),
        AIMessage(content="done"),
        AIMessage(content="hello"),
    ])
    graph = build_graph(llm, tools_ok(), settings())
    await run_turn(graph, "A", "search")
    await run_turn(graph, "B", "hi")
    assert "empty" in llm.calls[2]["messages"][0].content  # session B has no memory of A


async def test_step_limit_forces_a_final_answer_without_tools():
    llm = ScriptedLLM([
        call("geocode_place", {"name": "x"}, "c1"),
        call("geocode_place", {"name": "x"}, "c2"),
        AIMessage(content="I stopped early, here is what I have."),
    ])
    graph = build_graph(llm, tools_ok(), settings(max_steps=2))
    state = await run_turn(graph, "s1", "loop forever")

    assert state["messages"][-1].content.startswith("I stopped early")
    last = llm.calls[-1]
    assert last["with_tools"] is False
    assert "maximum number of tool calls" in last["messages"][0].content
    assert llm.calls[0]["with_tools"] is True


async def test_tool_exception_becomes_error_message_and_turn_continues():
    llm = ScriptedLLM([
        call("search_scenes", {"bbox": BBOX}, "c1"),
        AIMessage(content="The search failed, sorry."),
    ])
    tools = [failing_tool("search_scenes", ConnectionError("connection refused"))]
    graph = build_graph(llm, tools, settings())
    state = await run_turn(graph, "s1", "search")

    err = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert err.status == "error"
    assert "search_scenes" in err.content and "ConnectionError" in err.content and "bbox" in err.content
    assert state["messages"][-1].content == "The search failed, sorry."
    assert "last_results" not in (state.get("working_memory") or {})


async def test_unknown_tool_is_an_error_message_not_a_crash():
    llm = ScriptedLLM([call("does_not_exist", {}, "c1"), AIMessage(content="no such tool")])
    graph = build_graph(llm, tools_ok(), settings())
    state = await run_turn(graph, "s1", "x")
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).status == "error"


# update_working_memory on its own

def test_new_search_clears_selected_scene():
    wm = update_working_memory({"selected_scene_id": "OLD"}, "search_scenes", {"bbox": BBOX}, SEARCH_DATA)
    assert wm["selected_scene_id"] is None


def test_empty_search_replaces_last_results_with_empty_list():
    empty = {**SEARCH_DATA, "scenes": [], "total_found": 0}
    wm = update_working_memory({"last_results": [{"id": "OLD"}]}, "search_scenes", {"bbox": BBOX}, empty)
    assert wm["last_results"] == []


def test_search_with_unknown_bbox_keeps_only_the_bbox():
    wm = update_working_memory({}, "search_scenes", {"bbox": [1, 2, 3, 4]}, SEARCH_DATA)
    assert wm["place"] == {"bbox": [1, 2, 3, 4]}


def test_memory_keeps_the_cloud_range():
    data = {**SEARCH_DATA, "query": {**SEARCH_DATA["query"], "min_cloud_cover": 50.0, "max_cloud_cover": 100.0}}
    wm = update_working_memory({}, "search_scenes", {"bbox": BBOX}, data)
    assert (wm["min_cloud_cover"], wm["max_cloud_cover"]) == (50.0, 100.0)
