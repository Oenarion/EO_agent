"""The language of place names is a setting of the session, applied by code. No network, no real model."""
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from eo_agent.agent.context import render_memory
from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.api.main import create_app
from eo_agent.config import Settings
from test_error_path import make_runtime, post
from test_graph import ScriptedLLM, call


class GeoArgs(BaseModel):
    name: str
    language: str = "en"


def geocode_tool(seen: list[str]) -> StructuredTool:
    """A geocoding tool that records the language it was called with."""
    async def run(name: str, language: str = "en"):
        seen.append(language)
        data = {"result": []}
        return [{"type": "text", "text": "{}"}], {"structured_content": data}

    return StructuredTool(name="geocode_place", description="geocode", args_schema=GeoArgs, coroutine=run,
                          response_format="content_and_artifact")


GEO_CALL = {"name": "Roma"}


async def test_english_is_the_default():
    seen: list[str] = []
    llm = ScriptedLLM([call("geocode_place", GEO_CALL, "c1"), AIMessage(content="not found")])
    await run_turn(build_graph(llm, [geocode_tool(seen)], Settings()), "s", "scenes over Roma")
    assert seen == ["en"]


async def test_the_session_language_is_applied_even_if_the_model_passes_another():
    seen: list[str] = []
    llm = ScriptedLLM([call("geocode_place", {"name": "Roma", "language": "de"}, "c1"), AIMessage(content="ok")])
    await run_turn(build_graph(llm, [geocode_tool(seen)], Settings()), "s", "scenes over Roma", language="it")
    assert seen == ["it"]  # code decides, not the model


async def test_the_language_persists_across_turns_until_it_is_changed():
    seen: list[str] = []
    llm = ScriptedLLM([
        call("geocode_place", GEO_CALL, "c1"), AIMessage(content="a"),
        call("geocode_place", GEO_CALL, "c2"), AIMessage(content="b"),   # no language sent: the session keeps "it"
        call("geocode_place", GEO_CALL, "c3"), AIMessage(content="c"),   # changed to French
    ])
    graph = build_graph(llm, [geocode_tool(seen)], Settings())
    await run_turn(graph, "s", "one", language="it")
    await run_turn(graph, "s", "two")
    await run_turn(graph, "s", "three", language="fr")
    assert seen == ["it", "it", "fr"]


async def test_two_sessions_have_their_own_language():
    seen: list[str] = []
    llm = ScriptedLLM([call("geocode_place", GEO_CALL, "c1"), AIMessage(content="a"),
                       call("geocode_place", GEO_CALL, "c2"), AIMessage(content="b")])
    graph = build_graph(llm, [geocode_tool(seen)], Settings())
    await run_turn(graph, "A", "one", language="it")
    await run_turn(graph, "B", "two")
    assert seen == ["it", "en"]


async def test_the_model_is_told_the_language_in_the_memory_block():
    llm = ScriptedLLM([AIMessage(content="hello")])
    await run_turn(build_graph(llm, [], Settings()), "s", "hi", language="de")
    assert "Place names are searched in German" in llm.calls[0]["messages"][0].content


def test_memory_block_shows_language_and_whether_a_cloud_filter_was_used():
    assert "searched in English" in render_memory({}, "", "en") and "empty" in render_memory({}, "", "en")
    base = {"place": {"name": "X", "bbox": [1, 2, 3, 4]}}
    assert "Cloud cover filter in the last search: none" in render_memory({**base, "min_cloud_cover": 0, "max_cloud_cover": 100})
    assert "10% to 40%" in render_memory({**base, "min_cloud_cover": 10, "max_cloud_cover": 40})
    assert "searched in Italian" in render_memory(base, "", "it")


def test_the_api_sets_the_language_and_reports_it(tmp_path):
    seen: list[str] = []
    runtime = make_runtime(tmp_path, [call("geocode_place", GEO_CALL, "c1"), AIMessage(content="ok"), AIMessage(content="again")],
                           [geocode_tool(seen)])
    with TestClient(create_app(runtime)) as client:
        assert client.post("/chat", json={"session_id": "s", "message": "scenes over Roma", "language": "it"}).status_code == 200
        assert seen == ["it"]
        assert post(client, "s", "and now?").status_code == 200  # no language field: it stays Italian
        assert client.get("/sessions/s/memory").json()["place_language"] == "it"
        assert client.get("/health").json()["place_languages"]["it"] == "Italian"


def test_an_unknown_language_is_rejected_by_the_api(tmp_path):
    with TestClient(create_app(make_runtime(tmp_path, []))) as client:
        for bad in ("xx", "english", "IT", ""):
            r = client.post("/chat", json={"session_id": "s", "message": "hi", "language": bad})
            assert r.status_code == 422, bad


# ---------- a change of language invalidates the places already searched ----------

async def test_the_memory_warns_the_model_when_the_language_changed_after_a_place_search():
    seen: list[str] = []
    llm = ScriptedLLM([call("geocode_place", GEO_CALL, "c1"), AIMessage(content="found"),
                       call("geocode_place", GEO_CALL, "c2"), AIMessage(content="found again")])
    graph = build_graph(llm, [geocode_tool(seen)], Settings())
    await run_turn(graph, "s", "scenes over Roma", language="en")
    await run_turn(graph, "s", "scenes over Roma", language="it")
    first_call_of_turn_2 = llm.calls[2]["messages"][0].content
    assert "changed from English after the last place search" in first_call_of_turn_2
    # once the place is searched again, in the new language, the warning is gone
    third = ScriptedLLM([AIMessage(content="ok")])
    await run_turn(build_graph(third, [geocode_tool(seen)], Settings()), "t", "hi", language="it")
    assert "changed from" not in third.calls[0]["messages"][0].content


async def test_no_warning_when_the_language_never_changed():
    seen: list[str] = []
    llm = ScriptedLLM([call("geocode_place", GEO_CALL, "c1"), AIMessage(content="a"), AIMessage(content="b")])
    graph = build_graph(llm, [geocode_tool(seen)], Settings())
    await run_turn(graph, "s", "one", language="it")
    await run_turn(graph, "s", "two", language="it")
    assert "changed from" not in llm.calls[2]["messages"][0].content


def test_the_stale_language_line_in_the_memory_block():
    memory = {"place": {"name": "Roma", "bbox": [1, 2, 3, 4]}, "place_language_used": "en"}
    assert "changed from English" in render_memory(memory, "", "it")
    assert "changed from" not in render_memory(memory, "", "en")
