"""Tests for the MCP tools. All HTTP is mocked with the saved real responses."""
import json
from pathlib import Path

import httpx
import pytest
import respx
from mcp.shared.memory import create_connected_server_and_client_session

from eo_agent.mcp_server import geocode as geocode_mod
from eo_agent.mcp_server import http as http_mod
from eo_agent.mcp_server import stac
from eo_agent.mcp_server.schemas import ToolFailure
from eo_agent.mcp_server.server import mcp

FIXTURES = Path(__file__).parent / "fixtures"
BBOX = [12.15, 44.37, 12.25, 44.45]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(http_mod, "RETRY_DELAY_S", 0)


@pytest.fixture
def mock_http():
    with respx.mock(assert_all_called=False) as m:
        yield m


# geocode_place

async def test_geocode_returns_candidates_with_bbox(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).respond(json=fixture("geocode_ravenna.json"))
    places = await geocode_mod.geocode("Ravenna")
    assert [p.country for p in places] == ["Italy", "United States", "United States"]
    west, south, east, north = places[0].bbox
    assert west < places[0].longitude < east and south < places[0].latitude < north


async def test_geocode_no_match_is_empty_list_not_error(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).respond(json=fixture("geocode_empty.json"))
    assert await geocode_mod.geocode("Xyzzyqwkj") == []


async def test_geocode_empty_name_is_invalid_input():
    with pytest.raises(ToolFailure) as exc:
        await geocode_mod.geocode("   ")
    assert exc.value.kind == "invalid_input"


# search_scenes

async def test_search_normal(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_ravenna.json"))
    result = await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 2)
    assert result.returned == 2 and result.total_found == 5 and result.more_available
    assert [s.index for s in result.scenes] == [1, 2]
    first = result.scenes[0]
    assert first.id == "S2C_32TQQ_20250731_0_L2A"
    assert first.tile == "32TQQ"
    assert first.thumbnail_url.endswith("preview.jpg")
    # the request sent to the catalogue
    sent = json.loads(route.calls.last.request.content)
    assert sent["query"] == {"eo:cloud_cover": {"gte": 0, "lte": 10}}
    assert sent["datetime"] == "2025-07-01T00:00:00Z/2025-07-31T23:59:59Z"
    assert sent["limit"] == 2


async def test_search_caps_limit_at_10(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_empty.json"))
    result = await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 500)
    assert json.loads(route.calls[0].request.content)["limit"] == 10  # calls[0]: the search; later calls explain the empty result
    assert result.query.limit == 10


async def test_search_empty_is_normal_result(mock_http):
    mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_empty.json"))
    result = await stac.search(BBOX, "2025-07-01", "2025-07-02", 5, 5)
    assert result.scenes == [] and result.total_found == 0 and not result.more_available


@pytest.mark.parametrize(
    "bbox, start, end, cloud",
    [
        (BBOX, "01/07/2025", "2025-07-31", 10),      # bad date format
        (BBOX, "2025-07-31", "2025-07-01", 10),      # start after end
        (BBOX, "2025-02-30", "2025-03-01", 10),      # impossible date
        ([12.0, 44.0, 12.5], "2025-07-01", "2025-07-31", 10),        # 3 numbers
        ([12.5, 44.0, 12.0, 45.0], "2025-07-01", "2025-07-31", 10),  # west > east
        ([12.0, 44.0, 12.5, 95.0], "2025-07-01", "2025-07-31", 10),  # latitude out of range
        ([10.0, 40.0, 13.0, 41.0], "2025-07-01", "2025-07-31", 10),  # wider than 2 degrees
        (BBOX, "2025-07-01", "2025-07-31", 150),     # cloud out of range
        (BBOX, "2025-07-01", "2025-07-31", -5),      # cloud below 0
    ],
)
async def test_search_invalid_input_makes_no_http_call(mock_http, bbox, start, end, cloud):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    with pytest.raises(ToolFailure) as exc:
        await stac.search(bbox, start, end, cloud, 5)
    assert exc.value.kind == "invalid_input"
    assert route.call_count == 0


async def test_search_upstream_timeout_retries_once_then_fails(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ToolFailure) as exc:
        await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 5)
    assert exc.value.kind == "upstream_error"
    assert route.call_count == 2


async def test_search_5xx_then_success_uses_the_retry(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json=fixture("stac_search_ravenna.json"))]
    result = await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 2)
    assert result.returned == 2 and route.call_count == 2


# get_scene_details

async def test_scene_details_hides_jp2_duplicates(mock_http):
    item = fixture("stac_item.json")
    mock_http.get(f"{stac.STAC_URL}/collections/{stac.COLLECTION}/items/{item['id']}").respond(json=item)
    details = await stac.get_scene(item["id"])
    assert details.id == item["id"] and details.tile == "32TQQ"
    assert "red" in details.assets and "visual" in details.assets
    assert not any(a.endswith("-jp2") for a in details.assets)


async def test_scene_details_unknown_id_is_not_found_without_retry(mock_http):
    route = mock_http.get(f"{stac.STAC_URL}/collections/{stac.COLLECTION}/items/S2X_DOES_NOT_EXIST")
    route.respond(404, json=fixture("stac_item_404.json"))
    with pytest.raises(ToolFailure) as exc:
        await stac.get_scene("S2X_DOES_NOT_EXIST")
    assert exc.value.kind == "not_found"
    assert route.call_count == 1


@pytest.mark.parametrize("bad_id", ["", "../../etc/passwd", "a b", "id?x=1"])
async def test_scene_details_rejects_unsafe_ids(bad_id):
    with pytest.raises(ToolFailure) as exc:
        await stac.get_scene(bad_id)
    assert exc.value.kind == "invalid_input"


# through MCP: a failure must come back as isError and the server must stay usable

async def test_mcp_error_is_iserror_and_server_survives(mock_http):
    mock_http.get(f"{stac.STAC_URL}/collections/{stac.COLLECTION}/items/S2X_DOES_NOT_EXIST").respond(
        404, json=fixture("stac_item_404.json")
    )
    mock_http.get(geocode_mod.GEOCODING_URL).respond(json=fixture("geocode_ravenna.json"))
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {"geocode_place", "search_scenes", "get_scene_details"}

        bad = await client.call_tool("get_scene_details", {"scene_id": "S2X_DOES_NOT_EXIST"})
        assert bad.isError
        assert "not_found" in bad.content[0].text

        good = await client.call_tool("geocode_place", {"name": "Ravenna"})  # same session, still works
        assert not good.isError


# missing data is reported, not hidden

async def test_search_reports_missing_fields(mock_http):
    data = fixture("stac_search_ravenna.json")
    del data["features"][1]["properties"]["eo:cloud_cover"]
    del data["features"][1]["assets"]["thumbnail"]
    mock_http.post(f"{stac.STAC_URL}/search").respond(json=data)
    result = await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 2)
    assert result.missing_data == ["scene 2: cloud_cover", "scene 2: thumbnail_url"]
    assert result.scenes[1].cloud_cover is None


async def test_search_complete_scenes_report_nothing_missing(mock_http):
    mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_ravenna.json"))
    assert (await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 2)).missing_data == []


async def test_scene_details_reports_missing_fields(mock_http):
    item = fixture("stac_item.json")
    del item["properties"]["platform"]
    del item["properties"]["view:sun_elevation"]
    mock_http.get(f"{stac.STAC_URL}/collections/{stac.COLLECTION}/items/{item['id']}").respond(json=item)
    details = await stac.get_scene(item["id"])
    assert details.missing_fields == ["platform", "sun_elevation"]


async def test_geocode_reports_missing_region(mock_http):
    data = fixture("geocode_ravenna.json")
    del data["results"][0]["admin1"]
    mock_http.get(geocode_mod.GEOCODING_URL).respond(json=data)
    places = await geocode_mod.geocode("Ravenna")
    assert places[0].missing_fields == ["region"]
    assert places[1].missing_fields == []


# min_cloud_cover: "more than 50% cloud" must be expressible

async def test_search_min_cloud_cover_is_sent_and_echoed(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_empty.json"))
    result = await stac.search(BBOX, "2025-09-01", "2025-09-30", 100, 5, min_cloud_cover=50)
    assert json.loads(route.calls[0].request.content)["query"] == {"eo:cloud_cover": {"gte": 50, "lte": 100}}
    assert result.query.min_cloud_cover == 50 and result.query.max_cloud_cover == 100


@pytest.mark.parametrize("min_cloud, max_cloud", [(60, 40), (-1, 50), (50, 120)])
async def test_search_invalid_cloud_range(mock_http, min_cloud, max_cloud):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    with pytest.raises(ToolFailure) as exc:
        await stac.search(BBOX, "2025-09-01", "2025-09-30", max_cloud, 5, min_cloud_cover=min_cloud)
    assert exc.value.kind == "invalid_input" and route.call_count == 0


# ---------- geocoding: one language per search, set by the user ----------

def place(pid: int, name: str, region: str, country: str, population: int | None, lat: float = 41.9, lon: float = 12.5) -> dict:
    p = {"id": pid, "name": name, "admin1": region, "country": country, "latitude": lat, "longitude": lon}
    if population is not None:
        p["population"] = population
    return p


def by_language(answers: dict[str, list[dict]]):
    """A respx side effect: the geocoding API answers differently for each requested language."""
    def side_effect(request: httpx.Request) -> httpx.Response:
        results = answers.get(request.url.params["language"], [])
        return httpx.Response(200, json={"results": results} if results else {"generationtime_ms": 0.1})
    return side_effect


ROMA_EN = [place(1, "Roma", "Botosani County", "Romania", 900, 47.7, 26.6)]
ROMA_IT = [place(2, "Roma", "Lazio", "Italia", 2_800_000), place(1, "Roma", "distretto di Botosani", "Romania", 900, 47.7, 26.6)]


async def test_geocode_makes_one_call_in_the_requested_language(mock_http):
    route = mock_http.get(geocode_mod.GEOCODING_URL).mock(side_effect=by_language({"it": ROMA_IT}))
    places = await geocode_mod.geocode("Roma", "it")
    assert route.call_count == 1 and route.calls[0].request.url.params["language"] == "it"
    assert (places[0].region, places[0].country) == ("Lazio", "Italia")


async def test_the_default_language_is_english(mock_http):
    route = mock_http.get(geocode_mod.GEOCODING_URL).respond(json=fixture("geocode_ravenna.json"))
    await geocode_mod.geocode("Ravenna")
    assert route.call_count == 1 and route.calls[0].request.url.params["language"] == "en"


async def test_a_local_name_is_found_only_in_its_own_language(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).mock(side_effect=by_language(
        {"it": [place(3, "Copenaghen", "Region Hovedstaden", "Danimarca", 600_000, 55.68, 12.57)]}))
    assert await geocode_mod.geocode("Copenaghen") == []  # English: no match
    assert [p.name for p in await geocode_mod.geocode("Copenaghen", "it")] == ["Copenaghen"]


async def test_an_unknown_language_is_invalid_input_without_a_call(mock_http):
    route = mock_http.get(geocode_mod.GEOCODING_URL)
    with pytest.raises(ToolFailure) as exc:
        await geocode_mod.geocode("Roma", "xx")
    assert exc.value.kind == "invalid_input" and route.call_count == 0


async def test_exact_name_first_then_the_most_populous(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).mock(side_effect=by_language({"en": [
        place(10, "Ravenna Greater Area", "X", "Y", 5_000_000),
        place(12, "Ravenna", "Ohio", "United States", 11_000),
        place(11, "Ravenna", "Emilia-Romagna", "Italy", 80_000)]}))
    places = await geocode_mod.geocode("ravenna")
    assert [p.region for p in places] == ["Emilia-Romagna", "Ohio", "X"]


async def test_places_sharing_a_name_are_flagged_and_different_names_are_not(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).respond(json=fixture("geocode_ravenna.json"))
    assert all(p.shares_name_with_others for p in await geocode_mod.geocode("Ravenna"))
    mock_http.get(geocode_mod.GEOCODING_URL).mock(side_effect=by_language({"en": [
        place(20, "Tel Aviv", "Tel Aviv", "Israel", 460_000), place(21, "Tel Aviv Luna Park", "Tel Aviv", "Israel", None)]}))
    assert not any(p.shares_name_with_others for p in await geocode_mod.geocode("Tel Aviv"))


async def test_a_geocoding_failure_is_an_upstream_error(mock_http):
    mock_http.get(geocode_mod.GEOCODING_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ToolFailure) as exc:
        await geocode_mod.geocode("Roma")
    assert exc.value.kind == "upstream_error"


async def test_search_has_no_cloud_filter_unless_the_caller_asks_for_one(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_ravenna.json"))
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("search_scenes", {"bbox": BBOX, "start_date": "2025-07-01", "end_date": "2025-07-31"})
    assert not result.isError
    assert json.loads(route.calls[0].request.content)["query"] == {"eo:cloud_cover": {"gte": 0, "lte": 100}}


# ---------- an empty search explains itself ----------

def feature_on(day: str, cloud: float = 50.0) -> dict:
    return {"id": f"S2A_32TQQ_{day.replace('-', '')}_0_L2A", "properties": {"datetime": f"{day}T10:00:00Z", "eo:cloud_cover": cloud}, "assets": {}}


def stac_answer(matched: int, days: tuple[str, ...] = (), clouds: dict[str, float] | None = None) -> httpx.Response:
    clouds = clouds or {}
    return httpx.Response(200, json={"numberMatched": matched, "features": [feature_on(d, clouds.get(d, 50.0)) for d in days]})


async def test_empty_search_says_when_scenes_exist_with_another_cloud_cover(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    route.side_effect = [stac_answer(0), stac_answer(1)]
    result = await stac.search(BBOX, "2025-09-10", "2025-09-10", 20, 5)
    assert result.scenes == [] and "1 scene(s) exist" in result.empty_reason and "between 0% and 20%" in result.empty_reason
    second = json.loads(route.calls[1].request.content)
    assert second["query"] == {"eo:cloud_cover": {"gte": 0, "lte": 100}} and second["limit"] == 1  # the same area and dates, any cloud


async def test_empty_search_lists_the_closest_acquisitions_when_nothing_exists_that_day(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    route.side_effect = [stac_answer(0), stac_answer(0),
                         stac_answer(5, ("2025-09-04", "2025-09-06", "2025-09-08", "2025-09-11", "2025-09-13", "2025-09-16"))]
    result = await stac.search(BBOX, "2025-09-10", "2025-09-10", 100, 5)  # no cloud limit
    assert "No scene exists for this area and period" in result.empty_reason and "every 2 to 5 days" in result.empty_reason
    assert "2025-09-08 (cloud 50.0%), 2025-09-11 (cloud 50.0%)" in result.empty_reason
    widened = json.loads(route.calls[2].request.content)["datetime"]
    assert widened == "2025-09-03T00:00:00Z/2025-09-17T23:59:59Z"  # 7 days on each side


async def test_empty_search_with_nothing_nearby_says_to_check_area_and_dates(mock_http):
    mock_http.post(f"{stac.STAC_URL}/search").respond(json={"numberMatched": 0, "features": []})
    assert "within 7 days" in (await stac.search(BBOX, "2025-09-10", "2025-09-10", 20, 5)).empty_reason


async def test_a_failed_explanation_does_not_break_the_search(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search")
    route.side_effect = [stac_answer(0), httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")]
    result = await stac.search(BBOX, "2025-09-10", "2025-09-10", 20, 5)
    assert result.scenes == [] and result.empty_reason is None


async def test_a_search_with_results_makes_exactly_one_catalogue_call(mock_http):
    route = mock_http.post(f"{stac.STAC_URL}/search").respond(json=fixture("stac_search_ravenna.json"))
    result = await stac.search(BBOX, "2025-07-01", "2025-07-31", 10, 2)
    assert route.call_count == 1 and result.empty_reason is None


async def test_nearby_dates_offered_respect_the_cloud_limit_the_user_gave(mock_http):
    """The real Tel Aviv case: with a 10% limit, the 21% date must not be offered as a date to ask for."""
    days = ("2025-09-06", "2025-09-08", "2025-09-11", "2025-09-16")
    clouds = {"2025-09-06": 2.4, "2025-09-08": 21.0, "2025-09-11": 4.9, "2025-09-16": 7.6}
    mock_http.post(f"{stac.STAC_URL}/search").side_effect = [stac_answer(0), stac_answer(0), stac_answer(4, days, clouds)]
    reason = (await stac.search(BBOX, "2025-09-10", "2025-09-10", 10, 5)).empty_reason
    assert "between 0% and 10%: 2025-09-06 (cloud 2.4%), 2025-09-11 (cloud 4.9%), 2025-09-16 (cloud 7.6%)" in reason
    assert "2025-09-08" not in reason


async def test_when_no_nearby_date_fits_the_limit_the_reason_says_so_and_shows_the_cloud_cover(mock_http):
    mock_http.post(f"{stac.STAC_URL}/search").side_effect = [
        stac_answer(0), stac_answer(0), stac_answer(2, ("2025-09-08", "2025-09-11"), {"2025-09-08": 21.0, "2025-09-11": 40.0})]
    reason = (await stac.search(BBOX, "2025-09-10", "2025-09-10", 10, 5)).empty_reason
    assert "none has a cloud cover between 0% and 10%" in reason and "2025-09-08 (cloud 21.0%)" in reason and "2025-09-11 (cloud 40.0%)" in reason


async def test_one_date_with_two_tiles_shows_the_clearest_scene_of_that_day(mock_http):
    wider = httpx.Response(200, json={"numberMatched": 2, "features": [feature_on("2025-09-11", 30.0), feature_on("2025-09-11", 3.0)]})
    mock_http.post(f"{stac.STAC_URL}/search").side_effect = [stac_answer(0), stac_answer(0), wider]
    assert "2025-09-11 (cloud 3.0%)" in (await stac.search(BBOX, "2025-09-10", "2025-09-10", 10, 5)).empty_reason
