"""Sentinel-2 L2A scenes from the Element 84 Earth Search STAC API (no key)."""
import logging
import re
from datetime import date, timedelta

from eo_agent.mcp_server.http import send
from eo_agent.mcp_server.schemas import (
    SceneDetails, SceneQuery, SceneSearchResult, SceneSummary, ToolFailure, missing,
)

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_LIMIT = 10
NEARBY_DAYS = 7  # how far an empty search looks for the closest acquisitions
MAX_BBOX_SPAN_DEG = 2.0  # per side, about 220 km: roughly two Sentinel-2 tiles
log = logging.getLogger("eo_agent.mcp")
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _parse_date(label: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolFailure("invalid_input", f"{label} '{value}' is not a valid date. Use YYYY-MM-DD.") from None


def _validate(bbox: list[float], start: str, end: str, min_cloud: float, max_cloud: float, limit: int) -> None:
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
    if not (0 <= min_cloud <= 100 and 0 <= max_cloud <= 100):
        raise ToolFailure("invalid_input", "min_cloud_cover and max_cloud_cover must be between 0 and 100.")
    if min_cloud > max_cloud:
        raise ToolFailure("invalid_input", "min_cloud_cover must not be greater than max_cloud_cover.")
    if limit < 1:
        raise ToolFailure("invalid_input", "limit must be at least 1.")


def record_url(scene_id: str) -> str:
    """Where the catalogue serves this scene. Same as the 'self' link of the record."""
    return f"{STAC_URL}/collections/{COLLECTION}/items/{scene_id}"


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


async def _query(bbox: list[float], start_date: str, end_date: str, min_cloud: float, max_cloud: float, limit: int) -> dict:
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": {"eo:cloud_cover": {"gte": min_cloud, "lte": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": limit,
    }
    response = await send("Earth Search STAC", "POST", f"{STAC_URL}/search", json=body)
    if response.status_code != 200:
        raise ToolFailure(
            "upstream_error",
            f"Earth Search STAC rejected the search (HTTP {response.status_code}) {_error_detail(response)}".strip(),
        )
    return response.json()


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


async def _explain_empty(bbox: list[float], start_date: str, end_date: str, min_cloud: float, max_cloud: float) -> str | None:
    """Why did a search find nothing? Up to two more catalogue calls, made only when the result is empty.

    Sentinel-2 passes over a place every 2 to 5 days, so a single date often has no scene at all,
    and a user who sees "nothing found" easily blames the place or the spelling."""
    try:
        anywhere = (await _query(bbox, start_date, end_date, 0, 100, 1)).get("numberMatched", 0)
        if anywhere:
            return (f"{anywhere} scene(s) exist for this area and period, but none has a cloud cover "
                    f"between {min_cloud:g}% and {max_cloud:g}%.")
        wider = await _query(bbox, _shift(start_date, -NEARBY_DAYS), _shift(end_date, NEARBY_DAYS), 0, 100, 50)
    except ToolFailure as exc:  # the explanation is a bonus: its failure must not break the search
        log.warning("could not explain the empty search: %s", exc)
        return None
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    days = {date.fromisoformat(f["properties"]["datetime"][:10]) for f in wider.get("features", [])}
    nearest = sorted(sorted(days, key=lambda d: (start - d).days if d < start else (d - end).days)[:4])
    if not nearest:
        return f"No scene exists for this area within {NEARBY_DAYS} days of this period either. Check the area and the dates."
    return ("No scene exists for this area and period. Sentinel-2 images a location every 2 to 5 days, "
            "so a short period can have none. Closest acquisitions: " + ", ".join(d.isoformat() for d in nearest) + ".")


async def search(
    bbox: list[float], start_date: str, end_date: str, max_cloud_cover: float, limit: int, min_cloud_cover: float = 0
) -> SceneSearchResult:
    _validate(bbox, start_date, end_date, min_cloud_cover, max_cloud_cover, limit)
    limit = min(limit, MAX_LIMIT)
    data = await _query(bbox, start_date, end_date, min_cloud_cover, max_cloud_cover, limit)
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
            record_url=record_url(f["id"]),
        )
        for i, f in enumerate(features, start=1)
    ]
    missing_data = [
        f"scene {s.index}: {name}"
        for s in scenes
        for name in missing(cloud_cover=s.cloud_cover, tile=s.tile, thumbnail_url=s.thumbnail_url)
    ]
    empty_reason = None if scenes else await _explain_empty(bbox, start_date, end_date, min_cloud_cover, max_cloud_cover)
    return SceneSearchResult(
        query=SceneQuery(
            bbox=bbox, start_date=start_date, end_date=end_date,
            min_cloud_cover=min_cloud_cover, max_cloud_cover=max_cloud_cover, limit=limit,
        ),
        total_found=total,
        returned=len(scenes),
        more_available=total > len(scenes),
        scenes=scenes,
        empty_reason=empty_reason,
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
        record_url=record_url(f["id"]),
        assets=[name for name in f.get("assets", {}) if not name.endswith("-jp2")],
        missing_fields=missing(**fields),
        **fields,
    )
