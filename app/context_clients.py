"""External situational context clients for wildfire rescue planning."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class IncidentLocation:
    """Latitude/longitude and query radius for incident context lookups."""

    latitude: float
    longitude: float
    radius_meters: int = 5000


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class ContextConfig:
    """Environment-driven configuration for external context APIs."""

    openweather_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY")
    )
    openweather_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENWEATHER_BASE_URL", "https://api.openweathermap.org"
        )
    )
    overpass_url: str = field(
        default_factory=lambda: os.getenv(
            "OVERPASS_URL", "https://overpass-api.de/api/interpreter"
        )
    )
    usgs_epqs_url: str = field(
        default_factory=lambda: os.getenv(
            "USGS_EPQS_URL", "https://epqs.nationalmap.gov/v1/json"
        )
    )
    landfire_base_url: str = field(
        default_factory=lambda: os.getenv("LANDFIRE_BASE_URL", "https://lfps.usgs.gov")
    )
    landfire_email: str | None = field(
        default_factory=lambda: os.getenv("LANDFIRE_EMAIL")
    )
    landfire_layers: str = field(
        default_factory=lambda: os.getenv("LANDFIRE_LAYERS", "EVT;EVC;FBFM40;CC;CH")
    )
    landfire_resolution: int = field(
        default_factory=lambda: _env_int("LANDFIRE_RESAMPLE_RESOLUTION", 90)
    )
    landfire_auto_submit: bool = field(
        default_factory=lambda: os.getenv("LANDFIRE_AUTO_SUBMIT", "false").lower()
        == "true"
    )
    terrain_sample_spacing_meters: float = field(
        default_factory=lambda: _env_float("TERRAIN_SAMPLE_SPACING_METERS", 90.0)
    )
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("CONTEXT_TIMEOUT_SECONDS", 20.0)
    )


def extract_location(
    cnn_output: dict[str, Any], latitude: float | None, longitude: float | None
) -> IncidentLocation | None:
    """Build an incident location from explicit form fields or common CNN JSON keys."""

    lat = latitude
    lon = longitude
    if lat is None:
        lat = _first_float(cnn_output, ("latitude", "lat", "center_lat", "incident_lat"))
    if lon is None:
        lon = _first_float(cnn_output, ("longitude", "lon", "lng", "center_lon", "incident_lon"))

    location = cnn_output.get("location")
    if isinstance(location, dict):
        if lat is None:
            lat = _first_float(location, ("latitude", "lat"))
        if lon is None:
            lon = _first_float(location, ("longitude", "lon", "lng"))

    if lat is None or lon is None:
        return None
    return IncidentLocation(latitude=lat, longitude=lon)


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


class ExternalContextClient:
    """Fetch geospatial and weather context around a wildfire incident location."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    async def gather(self, location: IncidentLocation) -> dict[str, Any]:
        """Return best-effort context from configured public data APIs."""

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            weather = await self._safe_call(self._fetch_openweather(client, location))
            osm = await self._safe_call(self._fetch_osm_context(client, location))
            terrain = await self._safe_call(self._fetch_terrain_context(client, location))
            landfire = await self._safe_call(self._fetch_landfire_context(client, location))

        return {
            "incident_location": {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "radius_meters": location.radius_meters,
            },
            "openweather": weather,
            "openstreetmap": osm,
            "usgs_national_map": terrain,
            "landfire": landfire,
        }

    async def _safe_call(self, coroutine: Any) -> dict[str, Any]:
        try:
            return await coroutine
        except httpx.HTTPStatusError as exc:
            return {
                "status": "error",
                "detail": f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
            }
        except httpx.HTTPError as exc:
            return {"status": "error", "detail": str(exc)}
        except ValueError as exc:
            return {"status": "error", "detail": str(exc)}

    async def _fetch_openweather(
        self, client: httpx.AsyncClient, location: IncidentLocation
    ) -> dict[str, Any]:
        if not self.config.openweather_api_key:
            return {
                "status": "skipped",
                "detail": "Set OPENWEATHER_API_KEY to include wind and weather context.",
            }

        response = await client.get(
            f"{self.config.openweather_base_url.rstrip('/')}/data/2.5/weather",
            params={
                "lat": location.latitude,
                "lon": location.longitude,
                "appid": self.config.openweather_api_key,
                "units": "metric",
            },
        )
        response.raise_for_status()
        raw = response.json()
        wind = raw.get("wind", {})
        main = raw.get("main", {})
        weather = raw.get("weather", [{}])
        return {
            "status": "ok",
            "temperature_c": main.get("temp"),
            "relative_humidity_percent": main.get("humidity"),
            "weather": weather[0].get("description") if weather else None,
            "wind_speed_mps": wind.get("speed"),
            "wind_direction_degrees": wind.get("deg"),
            "wind_gust_mps": wind.get("gust"),
        }

    async def _fetch_osm_context(
        self, client: httpx.AsyncClient, location: IncidentLocation
    ) -> dict[str, Any]:
        query = _build_overpass_query(location)
        response = await client.post(self.config.overpass_url, data={"data": query})
        response.raise_for_status()
        raw = response.json()
        features = [_summarize_osm_element(element) for element in raw.get("elements", [])]
        return {
            "status": "ok",
            "radius_meters": location.radius_meters,
            "feature_count": len(features),
            "features": features[:75],
            "categories": _categorize_osm_features(features),
        }

    async def _fetch_terrain_context(
        self, client: httpx.AsyncClient, location: IncidentLocation
    ) -> dict[str, Any]:
        spacing = self.config.terrain_sample_spacing_meters
        samples = await self._sample_elevations(client, location, spacing)
        center = samples["center"]
        neighbor_values = [value for key, value in samples.items() if key != "center"]
        slopes = [abs(value - center) / spacing for value in neighbor_values]
        max_slope_grade = max(slopes) if slopes else None
        return {
            "status": "ok",
            "source": "USGS National Map Elevation Point Query Service",
            "center_elevation_meters": center,
            "sample_spacing_meters": spacing,
            "max_sampled_slope_grade": max_slope_grade,
            "max_sampled_slope_degrees": math.degrees(math.atan(max_slope_grade))
            if max_slope_grade is not None
            else None,
            "samples": samples,
        }

    async def _sample_elevations(
        self, client: httpx.AsyncClient, location: IncidentLocation, spacing_meters: float
    ) -> dict[str, float]:
        lat_offset = spacing_meters / 111_320
        lon_offset = spacing_meters / (
            111_320 * max(math.cos(math.radians(location.latitude)), 0.01)
        )
        points = {
            "center": (location.latitude, location.longitude),
            "north": (location.latitude + lat_offset, location.longitude),
            "south": (location.latitude - lat_offset, location.longitude),
            "east": (location.latitude, location.longitude + lon_offset),
            "west": (location.latitude, location.longitude - lon_offset),
        }
        results: dict[str, float] = {}
        for name, (lat, lon) in points.items():
            results[name] = await self._fetch_elevation(client, lat, lon)
        return results

    async def _fetch_elevation(
        self, client: httpx.AsyncClient, latitude: float, longitude: float
    ) -> float:
        response = await client.get(
            self.config.usgs_epqs_url,
            params={
                "x": longitude,
                "y": latitude,
                "units": "Meters",
                "wkid": 4326,
                "includeDate": "False",
            },
        )
        response.raise_for_status()
        raw = response.json()
        value = raw.get("value")
        if value is None:
            value = raw.get("USGS_Elevation_Point_Query_Service", {}).get(
                "Elevation_Query", {}
            ).get("Elevation")
        return float(value)

    async def _fetch_landfire_context(
        self, client: httpx.AsyncClient, location: IncidentLocation
    ) -> dict[str, Any]:
        aoi = _bbox_from_location(location)
        payload = {
            "Email": self.config.landfire_email,
            "Layer_List": self.config.landfire_layers,
            "Area_of_Interest": aoi,
            "Output_Projection": "4326",
            "Resample_Resolution": self.config.landfire_resolution,
        }
        if not self.config.landfire_email:
            return {
                "status": "configured_request_only",
                "detail": "Set LANDFIRE_EMAIL and LANDFIRE_AUTO_SUBMIT=true to submit LFPS jobs.",
                "submit_endpoint": f"{self.config.landfire_base_url.rstrip('/')}/api/job/submit",
                "layer_purpose": "vegetation cover, canopy/fuel conditions, and fuel model context",
                "request_payload": payload,
            }
        if not self.config.landfire_auto_submit:
            return {
                "status": "ready_to_submit",
                "detail": "LANDFIRE_AUTO_SUBMIT is false; payload prepared but not submitted.",
                "submit_endpoint": f"{self.config.landfire_base_url.rstrip('/')}/api/job/submit",
                "request_payload": payload,
            }

        response = await client.post(
            f"{self.config.landfire_base_url.rstrip('/')}/api/job/submit", data=payload
        )
        response.raise_for_status()
        return {
            "status": "submitted",
            "request_payload": payload,
            "raw": response.json(),
        }


def _bbox_from_location(location: IncidentLocation) -> str:
    lat_delta = location.radius_meters / 111_320
    lon_delta = location.radius_meters / (
        111_320 * max(math.cos(math.radians(location.latitude)), 0.01)
    )
    west = location.longitude - lon_delta
    south = location.latitude - lat_delta
    east = location.longitude + lon_delta
    north = location.latitude + lat_delta
    return f"{west:.6f} {south:.6f} {east:.6f} {north:.6f}"


def _build_overpass_query(location: IncidentLocation) -> str:
    radius = location.radius_meters
    lat = location.latitude
    lon = location.longitude
    return f"""
[out:json][timeout:25];
(
  node["amenity"~"^(hospital|fire_station|clinic|doctors|shelter)$"](around:{radius},{lat},{lon});
  way["amenity"~"^(hospital|fire_station|clinic|doctors|shelter)$"](around:{radius},{lat},{lon});
  relation["amenity"~"^(hospital|fire_station|clinic|doctors|shelter)$"](around:{radius},{lat},{lon});
  node["emergency"~"^(fire_hydrant|water_tank|suction_point|assembly_point)$"](around:{radius},{lat},{lon});
  way["natural"="water"](around:{radius},{lat},{lon});
  way["waterway"](around:{radius},{lat},{lon});
  way["highway"](around:{radius},{lat},{lon});
  node["amenity"~"^(school|kindergarten|nursing_home|social_facility|prison)$"](around:{radius},{lat},{lon});
  way["amenity"~"^(school|kindergarten|nursing_home|social_facility|prison)$"](around:{radius},{lat},{lon});
  node["tourism"~"^(camp_site|caravan_site)$"](around:{radius},{lat},{lon});
  way["tourism"~"^(camp_site|caravan_site)$"](around:{radius},{lat},{lon});
);
out center tags qt 75;
"""


def _summarize_osm_element(element: dict[str, Any]) -> dict[str, Any]:
    tags = element.get("tags", {})
    lat = element.get("lat") or element.get("center", {}).get("lat")
    lon = element.get("lon") or element.get("center", {}).get("lon")
    return {
        "osm_type": element.get("type"),
        "osm_id": element.get("id"),
        "name": tags.get("name"),
        "latitude": lat,
        "longitude": lon,
        "amenity": tags.get("amenity"),
        "emergency": tags.get("emergency"),
        "highway": tags.get("highway"),
        "natural": tags.get("natural"),
        "waterway": tags.get("waterway"),
        "tourism": tags.get("tourism"),
    }


def _categorize_osm_features(features: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    categories = {
        "hospitals": [],
        "fire_stations": [],
        "roads": [],
        "water_sources": [],
        "threatened_assets": [],
    }
    threatened_amenities = {
        "school",
        "kindergarten",
        "nursing_home",
        "social_facility",
        "prison",
        "shelter",
    }
    for feature in features:
        amenity = feature.get("amenity")
        if amenity in {"hospital", "clinic", "doctors"}:
            categories["hospitals"].append(feature)
        if amenity == "fire_station":
            categories["fire_stations"].append(feature)
        if feature.get("highway"):
            categories["roads"].append(feature)
        if feature.get("emergency") in {"fire_hydrant", "water_tank", "suction_point"}:
            categories["water_sources"].append(feature)
        if feature.get("natural") == "water" or feature.get("waterway"):
            categories["water_sources"].append(feature)
        if amenity in threatened_amenities or feature.get("tourism") in {
            "camp_site",
            "caravan_site",
        }:
            categories["threatened_assets"].append(feature)
    return {key: value[:25] for key, value in categories.items()}
