"""Phase 1 check: connect to the running MCP server, list the tools, call each one.

Start the server first:   python -m eo_agent.mcp_server.server
Then run:                 python demo/mcp_smoke.py

Edit the parameters below to try other cases (empty result, big bbox, ...).
"""
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from eo_agent.config import get_settings

# --- parameters to play with ---
PLACE = "Ravenna"
START_DATE, END_DATE = "2025-07-01", "2025-07-31"
MAX_CLOUD_COVER = 10
LIMIT = 3
BBOX_OVERRIDE: list[float] | None = None  # e.g. [5, 40, 12, 46] to test the size limit
# --------------------------------

PRINT_LIMIT = 1500  # characters printed per result


def show(title: str, result) -> dict | None:
    """Print a tool result. Successful results carry the full data in structuredContent
    (a list return value is wrapped as {"result": [...]}); errors only carry text."""
    print(f"\n--- {title}  (isError={result.isError})")
    if result.isError:
        print(result.content[0].text)
        return None
    data = result.structuredContent
    text = json.dumps(data, indent=1)
    print(text[:PRINT_LIMIT] + (" ..." if len(text) > PRINT_LIMIT else ""))
    return data


async def main() -> None:
    async with streamable_http_client(get_settings().mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            geo = show(f"geocode_place({PLACE!r})", await session.call_tool("geocode_place", {"name": PLACE}))
            places = (geo or {}).get("result", [])
            if places:
                place = places[0]
                print(f"\nUsing: {place['name']}, {place['country']}")
            elif not BBOX_OVERRIDE:
                print("\nNo place found and no BBOX_OVERRIDE set: stopping here.")
                return
            bbox = BBOX_OVERRIDE or place["bbox"]

            found = show(
                f"search_scenes(bbox={bbox}, {START_DATE}..{END_DATE}, cloud <= {MAX_CLOUD_COVER}, limit={LIMIT})",
                await session.call_tool("search_scenes", {
                    "bbox": bbox, "start_date": START_DATE, "end_date": END_DATE,
                    "max_cloud_cover": MAX_CLOUD_COVER, "limit": LIMIT,
                }),
            )
            if found and found["scenes"]:
                scene_id = found["scenes"][0]["id"]
                show(f"get_scene_details({scene_id})", await session.call_tool("get_scene_details", {"scene_id": scene_id}))
            else:
                print("\n(no scenes to get details for: skipping get_scene_details)")

            show("get_scene_details(S2X_DOES_NOT_EXIST)  <- expected error", await session.call_tool("get_scene_details", {"scene_id": "S2X_DOES_NOT_EXIST"}))
            show("search_scenes(bad date)  <- expected error", await session.call_tool("search_scenes", {
                "bbox": bbox, "start_date": "31/07/2025", "end_date": "2025-07-31"}))


if __name__ == "__main__":
    asyncio.run(main())
