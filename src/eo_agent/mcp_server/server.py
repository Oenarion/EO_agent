"""MCP server (FastMCP, streamable HTTP). Run as its own process:

    python -m eo_agent.mcp_server.server
"""
import argparse
import logging
import sys
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from eo_agent.config import get_settings
from eo_agent.mcp_server import geocode as geocode_mod
from eo_agent.mcp_server import stac
from eo_agent.mcp_server.schemas import Place, SceneDetails, SceneSearchResult

_url = urlparse(get_settings().mcp_url)
mcp = FastMCP("eo-scenes", host=_url.hostname or "127.0.0.1", port=_url.port or 8001)


@mcp.tool()
async def geocode_place(name: str, language: str = "en") -> list[Place]:
    """Find the coordinates of a place by name (city, town, region).

    Pass the name as the user wrote it. The name is matched in one language,
    given by language (en, it, es, fr or de): the system sets it from the
    user's setting, so leave the default. Returns up to 3 candidates, an exact
    name match first and then the most populous, each with country, region,
    latitude, longitude and a bbox of about 10 km around the point
    ([west, south, east, north]). shares_name_with_others is true when another candidate has exactly the same
    name (an ambiguous place). An empty list means the place was not found: that
    is a normal result, say so and ask for a more precise name. missing_fields
    lists what the API did not provide for a candidate.
    """
    return await geocode_mod.geocode(name, language)


@mcp.tool()
async def search_scenes(
    bbox: list[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 100,
    limit: int = 5,
    min_cloud_cover: float = 0,
) -> SceneSearchResult:
    """Search Sentinel-2 L2A satellite scenes over an area and a date range.

    bbox is [west, south, east, north] in degrees (use the bbox from
    geocode_place). Dates are YYYY-MM-DD, inclusive. Cloud cover is in percent
    (0 to 100). By default there is NO cloud filter: pass min_cloud_cover and
    max_cloud_cover only if the user asked for a cloud limit. min_cloud_cover is
    the lowest accepted value and max_cloud_cover the highest, so "less than 10%" is max_cloud_cover=10, "more than 50%" is
    min_cloud_cover=50 together with max_cloud_cover=100, and "between 20 and 40"
    is min 20 and max 40. limit is how many scenes to return (max 10).

    Scenes are sorted by cloud cover, clearest first, and numbered from 1
    (index). total_found is the number of matches in the catalogue and
    more_available tells whether more exist than were returned. An empty
    scenes list means nothing matched, and empty_reason says why (scenes exist
    but with another cloud cover, or there is no acquisition in that period and
    these are the closest dates): tell the user that reason. bbox sides are limited to 2 degrees (about 220 km). missing_data
    lists fields the catalogue did not provide, per scene.
    """
    return await stac.search(bbox, start_date, end_date, max_cloud_cover, limit, min_cloud_cover)


@mcp.tool()
async def get_scene_details(scene_id: str) -> SceneDetails:
    """Get the details of one Sentinel-2 L2A scene by its exact id.

    Use an id returned by search_scenes, copied exactly. Returns acquisition
    time, cloud cover, tile, satellite, sun elevation, footprint bbox,
    thumbnail URL and the names of the available bands and files. Fails with
    not_found if the id does not exist. missing_fields lists what the catalogue
    did not provide for this scene.
    """
    return await stac.get_scene(scene_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="EO scenes MCP server")
    parser.add_argument(
        "--transport", choices=["streamable-http", "stdio"], default="streamable-http",
        help="streamable-http: a service on MCP_URL (default, what the agent uses). "
             "stdio: the MCP client starts this process itself and talks through its input and output",
    )
    args = parser.parse_args()
    # logs go to stderr: in stdio mode stdout carries the protocol and must stay clean
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
