"""Pydantic models for tool outputs, and the one error type the tools raise.

Outputs are compact on purpose: this is the first layer of the context policy.
"""
from pydantic import BaseModel, Field


class ToolFailure(Exception):
    """A failure the model can explain to the user.

    kind is one of: invalid_input, not_found, upstream_error.
    The message always starts with the kind, so it reads well in the trace.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(f"[{kind}] {message}")


def missing(**fields: object) -> list[str]:
    """Names of the fields the upstream API did not provide (value is None)."""
    return [name for name, value in fields.items() if value is None]


class Place(BaseModel):
    name: str
    country: str | None = None
    region: str | None = Field(None, description="First-level admin area, e.g. a state or region")
    latitude: float
    longitude: float
    bbox: list[float] = Field(description="[west, south, east, north], about 10 km around the point")
    missing_fields: list[str] = Field(default_factory=list, description="Fields the API did not provide")


class SceneSummary(BaseModel):
    index: int = Field(description="1-based position in this result list")
    id: str
    datetime: str = Field(description="Acquisition time, UTC, as returned by the catalogue")
    cloud_cover: float | None = Field(None, description="Percent, 0 to 100")
    tile: str | None = Field(None, description="Sentinel-2 MGRS tile, e.g. 32TQQ")
    thumbnail_url: str | None = None


class SceneQuery(BaseModel):
    """Echo of the parameters actually used."""

    bbox: list[float]
    start_date: str
    end_date: str
    min_cloud_cover: float
    max_cloud_cover: float
    limit: int
    sorted_by: str = "cloud_cover ascending"


class SceneSearchResult(BaseModel):
    query: SceneQuery
    total_found: int = Field(description="Scenes matching the query in the catalogue")
    returned: int
    more_available: bool
    scenes: list[SceneSummary]
    missing_data: list[str] = Field(
        default_factory=list,
        description="Fields the catalogue did not provide, per scene, e.g. 'scene 2: cloud_cover'",
    )


class SceneDetails(BaseModel):
    id: str
    datetime: str
    cloud_cover: float | None = None
    tile: str | None = None
    platform: str | None = Field(None, description="e.g. sentinel-2a, sentinel-2b, sentinel-2c")
    sun_elevation: float | None = Field(None, description="Degrees")
    bbox: list[float] | None = None
    thumbnail_url: str | None = None
    assets: list[str] = Field(description="Names of the available bands and files (names only)")
    missing_fields: list[str] = Field(default_factory=list, description="Fields the catalogue did not provide")
