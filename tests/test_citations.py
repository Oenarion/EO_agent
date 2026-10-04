"""Citations and the stdio transport. No network, no real model."""
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from eo_agent.agent.citations import MAX_LINKS_KEPT, add_sources, cited_ids, remember_links, strip_sources
from eo_agent.agent.graph import build_graph, run_turn
from eo_agent.agent.state import update_working_memory
from eo_agent.config import Settings
from eo_agent.mcp_server import stac
from test_graph import BBOX, Anything, ScriptedLLM, call

FIXTURES = Path(__file__).parent / "fixtures"
ID_A, ID_B = "S2C_32TQQ_20250731_0_L2A", "S2A_32TQQ_20250723_0_L2A"
REC_A = f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{ID_A}"
REC_B = f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{ID_B}"
LINKS = {ID_A: {"record": REC_A, "preview": "https://example.test/a.jpg"}, ID_B: {"record": REC_B, "preview": ""}}


# ---------- the links the server returns are the real ones ----------

def test_record_url_is_the_self_link_of_the_catalogue_record():
    item = json.loads((FIXTURES / "stac_item.json").read_text(encoding="utf-8"))
    self_link = next(link["href"] for link in item["links"] if link["rel"] == "self")
    assert stac.record_url(item["id"]) == self_link


def test_every_scene_in_the_real_search_fixture_gets_its_own_link():
    data = json.loads((FIXTURES / "stac_search_ravenna.json").read_text(encoding="utf-8"))
    for feature in data["features"]:
        self_link = next(link["href"] for link in feature["links"] if link["rel"] == "self")
        assert stac.record_url(feature["id"]) == self_link


# ---------- the Sources block ----------

def test_a_cited_scene_gets_a_record_link_and_a_preview_link():
    out = add_sources(f"The best scene is {ID_A}.", LINKS, geocoded=False)
    assert out.startswith(f"The best scene is {ID_A}.\n\nSources:\n")
    assert f"- [{ID_A}]({REC_A}) ([preview](https://example.test/a.jpg))" in out
    assert "Open-Meteo" not in out and "Earth Search" in out


def test_geocoding_adds_the_open_meteo_attribution():
    assert "Place data: Open-Meteo.com." in add_sources(f"See {ID_A}", LINKS, geocoded=True)


def test_an_id_no_tool_returned_gets_no_link():
    invented = "S2A_32TQQ_20250101_0_L2A"
    out = add_sources(f"{ID_A} and {invented}", LINKS, geocoded=False)
    assert REC_A in out and invented not in out.split("Sources:")[1]


def test_an_answer_without_known_ids_is_left_alone():
    assert add_sources("No scenes matched your filter.", LINKS, geocoded=True) == "No scenes matched your filter."
    assert add_sources(f"Only {'S2A_32TQQ_20250101_0_L2A'} here", LINKS, geocoded=False).count("Sources:") == 0


def test_adding_sources_twice_does_not_repeat_the_block():
    once = add_sources(f"Scene {ID_A}", LINKS, geocoded=False)
    assert add_sources(once, LINKS, geocoded=False) == once


def test_a_sources_block_written_by_the_model_is_replaced_not_doubled():
    text = f"Scene {ID_A}\n\nSources:\n- made up by the model"
    out = add_sources(text, LINKS, geocoded=False)
    assert out.count("Sources:") == 1 and "made up" not in out and REC_A in out


def test_ids_are_cited_once_in_order_of_appearance():
    assert cited_ids(f"{ID_B} then {ID_A} then {ID_B} again", LINKS) == [ID_B, ID_A]
    assert strip_sources("no block here") == "no block here"


# ---------- the memory remembers the links across searches ----------

def scene(i: int, **extra) -> dict:
    return {"id": f"S2A_32TQQ_2025{i:04d}_0_L2A", "record_url": f"https://r.test/{i}", "thumbnail_url": f"https://p.test/{i}", **extra}


def test_links_from_every_search_are_kept():
    first = remember_links({}, [scene(701), scene(702)])
    second = remember_links(first, [scene(801)])
    assert len(second) == 3 and second["S2A_32TQQ_20250701_0_L2A"]["record"] == "https://r.test/701"


def test_old_links_are_dropped_beyond_the_limit_and_recent_ones_survive():
    links = remember_links({}, [scene(i) for i in range(1, MAX_LINKS_KEPT + 11)])
    assert len(links) == MAX_LINKS_KEPT
    assert "S2A_32TQQ_20250001_0_L2A" not in links and f"S2A_32TQQ_2025{MAX_LINKS_KEPT + 10:04d}_0_L2A" in links


def test_scenes_without_a_record_link_are_not_remembered():
    assert remember_links({}, [{"id": ID_A}]) == {}


def test_search_and_details_results_feed_scene_links_in_the_working_memory():
    search = {"query": {"start_date": "2025-07-01", "end_date": "2025-07-31", "max_cloud_cover": 10}, "total_found": 1,
              "scenes": [{"index": 1, "id": ID_A, "datetime": "d", "cloud_cover": 1.4, "tile": "32TQQ",
                          "record_url": REC_A, "thumbnail_url": "https://example.test/a.jpg"}]}
    wm = update_working_memory({}, "search_scenes", {"bbox": BBOX}, search)
    assert wm["scene_links"][ID_A]["preview"] == "https://example.test/a.jpg"
    wm = update_working_memory(wm, "get_scene_details", {}, {"id": ID_B, "record_url": REC_B})
    assert set(wm["scene_links"]) == {ID_A, ID_B}


# ---------- the cite node, inside the graph ----------

SEARCH_RESULT = {
    "query": {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31", "min_cloud_cover": 0.0, "max_cloud_cover": 10.0},
    "total_found": 2, "returned": 2,
    "scenes": [{"index": 1, "id": ID_A, "datetime": "d1", "cloud_cover": 1.4, "tile": "32TQQ", "record_url": REC_A, "thumbnail_url": "https://example.test/a.jpg"},
               {"index": 2, "id": ID_B, "datetime": "d2", "cloud_cover": 2.4, "tile": "32TQQ", "record_url": REC_B, "thumbnail_url": ""}],
}


def tool(name: str, data: dict) -> StructuredTool:
    async def run(**kwargs):
        return [{"type": "text", "text": json.dumps(data, indent=1)}], {"structured_content": data}

    return StructuredTool(name=name, description=name, args_schema=Anything, coroutine=run, response_format="content_and_artifact")


async def test_the_final_answer_comes_back_with_its_sources():
    llm = ScriptedLLM([
        call("geocode_place", {"name": "Ravenna"}, "c1"),
        call("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}, "c2"),
        AIMessage(content=f"Found 2 scenes: {ID_A} (1.4%) and {ID_B} (2.4%)."),
    ])
    tools = [tool("geocode_place", {"result": [{"name": "Ravenna", "bbox": BBOX}]}), tool("search_scenes", SEARCH_RESULT)]
    graph = build_graph(llm, tools, Settings())
    state = await run_turn(graph, "s", "find scenes")

    answer = state["messages"][-1].content
    assert f"[{ID_A}]({REC_A})" in answer and f"[{ID_B}]({REC_B})" in answer
    assert "Place data: Open-Meteo.com." in answer
    assert sum(1 for m in state["messages"] if isinstance(m, AIMessage) and not m.tool_calls) == 1  # replaced, not duplicated


async def test_a_follow_up_answered_from_memory_is_cited_too():
    llm = ScriptedLLM([
        call("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"}, "c1"),
        AIMessage(content="Found two."),
        AIMessage(content=f"The first one is {ID_A}, with 1.4% clouds."),
    ])
    graph = build_graph(llm, [tool("search_scenes", SEARCH_RESULT)], Settings())
    await run_turn(graph, "s", "search")
    state = await run_turn(graph, "s", "which is the first one?")
    assert f"[{ID_A}]({REC_A})" in state["messages"][-1].content
    assert "Open-Meteo" not in state["messages"][-1].content  # no geocoding in this turn


async def test_no_sources_when_the_answer_cites_no_scene():
    llm = ScriptedLLM([AIMessage(content="That is outside what I do.")])
    state = await run_turn(build_graph(llm, [], Settings()), "s", "capital of France?")
    assert state["messages"][-1].content == "That is outside what I do."


# ---------- the server also speaks stdio ----------

async def test_the_server_works_over_stdio_in_a_real_subprocess():
    params = StdioServerParameters(command=sys.executable, args=["-m", "eo_agent.mcp_server.server", "--transport", "stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"geocode_place", "search_scenes", "get_scene_details"}
            bad = await session.call_tool("geocode_place", {"name": "   "})  # rejected before any network call
            assert bad.isError and "invalid_input" in bad.content[0].text
