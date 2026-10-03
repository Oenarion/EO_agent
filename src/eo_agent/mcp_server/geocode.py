"""Place name to coordinates, using the Open-Meteo geocoding API (no key)."""
import math

from eo_agent.mcp_server.http import send
from eo_agent.mcp_server.schemas import Place, ToolFailure, missing

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HALF_SIDE_KM = 5.0  # box of about 10 km
KM_PER_DEG_LAT = 111.0


def make_bbox(lat: float, lon: float) -> list[float]:
    dlat = HALF_SIDE_KM / KM_PER_DEG_LAT
    dlon = HALF_SIDE_KM / (KM_PER_DEG_LAT * max(math.cos(math.radians(lat)), 0.01))
    return [
        round(max(lon - dlon, -180), 4),
        round(max(lat - dlat, -90), 4),
        round(min(lon + dlon, 180), 4),
        round(min(lat + dlat, 90), 4),
    ]


async def geocode(name: str) -> list[Place]:
    name = name.strip()
    if not name:
        raise ToolFailure("invalid_input", "Place name is empty.")
    response = await send(
        "Open-Meteo geocoding", "GET", GEOCODING_URL,
        params={"name": name, "count": 3, "language": "en", "format": "json"},
    )
    if response.status_code != 200:
        raise ToolFailure("upstream_error", f"Open-Meteo geocoding rejected the request (HTTP {response.status_code}).")
    # With no match the API omits the "results" key entirely.
    results = response.json().get("results") or []
    return [
        Place(
            name=r["name"],
            country=r.get("country"),
            region=r.get("admin1"),
            latitude=r["latitude"],
            longitude=r["longitude"],
            bbox=make_bbox(r["latitude"], r["longitude"]),
            missing_fields=missing(country=r.get("country"), region=r.get("admin1")),
        )
        for r in results
    ]
