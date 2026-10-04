"""Place name to coordinates, using the Open-Meteo geocoding API (no key).

The API matches a name in the language you ask for: "Copenhagen" is found in English,
"Copenaghen" only in Italian. The language is a setting of the user's session, so it is an
argument of the tool, one language per search.
"""
import logging
import math
import unicodedata

from eo_agent.config import DEFAULT_PLACE_LANGUAGE, PLACE_LANGUAGES
from eo_agent.mcp_server.http import send
from eo_agent.mcp_server.schemas import Place, ToolFailure, missing

log = logging.getLogger("eo_agent.mcp")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FETCHED = 5  # candidates asked from the API, the best MAX_CANDIDATES are returned
MAX_CANDIDATES = 3
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


def _plain(text: str) -> str:
    """Lower case, without accents: 'Köln' and 'koln' compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


async def geocode(name: str, language: str = DEFAULT_PLACE_LANGUAGE) -> list[Place]:
    name = name.strip()
    if not name:
        raise ToolFailure("invalid_input", "Place name is empty.")
    if language not in PLACE_LANGUAGES:
        raise ToolFailure("invalid_input", f"language must be one of {', '.join(PLACE_LANGUAGES)}.")
    response = await send(
        "Open-Meteo geocoding", "GET", GEOCODING_URL,
        params={"name": name, "count": FETCHED, "language": language, "format": "json"},
    )
    if response.status_code != 200:
        raise ToolFailure("upstream_error", f"Open-Meteo geocoding rejected the request (HTTP {response.status_code}).")
    # With no match the API omits the "results" key entirely.
    results = response.json().get("results") or []
    wanted = _plain(name)
    # An exact name match first, then the most populous place.
    ranked = sorted(results, key=lambda r: (_plain(r["name"]) != wanted, -(r.get("population") or 0)))[:MAX_CANDIDATES]
    names = [_plain(r["name"]) for r in ranked]
    return [
        Place(
            name=r["name"],
            country=r.get("country"),
            region=r.get("admin1"),
            latitude=r["latitude"],
            longitude=r["longitude"],
            bbox=make_bbox(r["latitude"], r["longitude"]),
            shares_name_with_others=names.count(_plain(r["name"])) > 1,
            missing_fields=missing(country=r.get("country"), region=r.get("admin1")),
        )
        for r in ranked
    ]
