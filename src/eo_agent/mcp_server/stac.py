"""Sentinel-2 L2A scenes from the Element 84 Earth Search STAC API (no key)."""
import re
from datetime import date

from eo_agent.mcp_server.http import send
from eo_agent.mcp_server.schemas import (
    SceneDetails, SceneQuery, SceneSearchResult, SceneSummary, ToolFailure, missing,
)

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_LIMIT = 10
MAX_BBOX_SPAN_DEG = 2.0  # per side, about 220 km: roughly two Sentinel-2 tiles
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _parse_date(label: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolFailure("invalid_input", f"{label} '{value}' is not a valid date. Use YYYY-MM-DD.") from None


def _validate(bbox: list[float], start: str, end: str, max_cloud: float, limit: int) -> None:
    if len(bbox) != 4:
        raise ToolFailure("invalid_input", "bbox must have 4 numbers: [west, south, east, north].")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180) or not (-90 <= south < north <= 90):
        raise ToolFailure(
            "invalid_input",
            "bbox out of range: longitudes must be in -180..180 with west < east, latitudes in -90..90 with south < north.",
        )
    if east - west > MAX_BBOX_SPAN_DEG or north - south > MAX_BBOX_SPAN_DEG:
        raise ToolFailure(
            "invalid_input",
            f"bbox too large: each side must be at most {MAX_BBOX_SPAN_DEG} degrees (about 220 km). Use a smaller area.",
        )
    if _parse_date("start_date", start) > _parse_date("end_date", end):
        raise ToolFailure("invalid_input", "start_date must not be after end_date.")
    if not 0 <= max_cloud <= 100:
        raise ToolFailure("invalid_input", "max_cloud_cover must be between 0 and 100.")
    if limit < 1:
        raise ToolFailure("invalid_input", "limit must be at least 1.")


def _tile(props: dict) -> str | None:
    code = props.get("grid:code")  # e.g. "MGRS-32TQQ"
    return code.removeprefix("MGRS-") if code else None


def _thumbnail(feature: dict) -> str | None:
    return (feature.get("assets", {}).get("thumbnail") or {}).get("href")


def _error_detail(response) -> str:
    try:
        return response.json().get("description", "")
    except ValueError:
        return ""


async def search(bbox: list[float], start_date: str, end_date: str, max_cloud_cover: float, limit: int) -> SceneSearchResult:
    _validate(bbox, start_date, end_date, max_cloud_cover, limit)
    limit = min(limit, MAX_LIMIT)
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": limit,
    }
    response = await send("Earth Search STAC", "POST", f"{STAC_URL}/search", json=body)
    if response.status_code != 200:
        raise ToolFailure(
            "upstream_error",
            f"Earth Search STAC rejected the search (HTTP {response.status_code}) {_error_detail(response)}".strip(),
        )
    data = response.json()
    features = data.get("features", [])
    total = data.get("numberMatched", len(features))
    scenes = [
        SceneSummary(
            index=i,
            id=f["id"],
            datetime=f["properties"]["datetime"],
            cloud_cover=f["properties"].get("eo:cloud_cover"),
            tile=_tile(f["properties"]),
            thumbnail_url=_thumbnail(f),
        )
        for i, f in enumerate(features, start=1)
    ]
    missing_data = [
        f"scene {s.index}: {name}"
        for s in scenes
        for name in missing(cloud_cover=s.cloud_cover, tile=s.tile, thumbnail_url=s.thumbnail_url)
    ]
    return SceneSearchResult(
        query=SceneQuery(bbox=bbox, start_date=start_date, end_date=end_date, max_cloud_cover=max_cloud_cover, limit=limit),
        total_found=total,
        returned=len(scenes),
        more_available=total > len(scenes),
        scenes=scenes,
        missing_data=missing_data,
    )


async def get_scene(scene_id: str) -> SceneDetails:
    scene_id = scene_id.strip()
    if not scene_id or not SCENE_ID_RE.match(scene_id):
        raise ToolFailure("invalid_input", "scene_id is empty or contains invalid characters.")
    response = await send("Earth Search STAC", "GET", f"{STAC_URL}/collections/{COLLECTION}/items/{scene_id}")
    if response.status_code == 404:
        raise ToolFailure("not_found", f"No Sentinel-2 L2A scene with id '{scene_id}' exists in the catalogue.")
    if response.status_code != 200:
        raise ToolFailure("upstream_error", f"Earth Search STAC rejected the request (HTTP {response.status_code}).")
    f = response.json()
    p = f["properties"]
    fields = {
        "cloud_cover": p.get("eo:cloud_cover"),
        "tile": _tile(p),
        "platform": p.get("platform"),
        "sun_elevation": p.get("view:sun_elevation"),
        "bbox": f.get("bbox"),
        "thumbnail_url": _thumbnail(f),
    }
    return SceneDetails(
        id=f["id"],
        datetime=p["datetime"],
        assets=[name for name in f.get("assets", {}) if not name.endswith("-jp2")],
        missing_fields=missing(**fields),
        **fields,
    )
