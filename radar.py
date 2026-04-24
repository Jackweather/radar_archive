from __future__ import annotations

import argparse
import gc
import gzip
import io
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygrib
import requests
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import box


DEFAULT_COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
DEFAULT_STATES_URL = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
DEFAULT_RADAR_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
DEFAULT_NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
DEFAULT_OUTPUT = Path("mrms_conus_radar.png")
BASE_DIR = Path("/var/data")
ARCHIVE_ROOT = BASE_DIR / "mrms_radar_archive"
GRIB_ARCHIVE_ROOT = BASE_DIR / "mrms_grib_archive"
CONUS_EXTENT = (-124.0, -67.5, 25.0, 49.5)
MIN_DBZ = 5.0
MAX_DBZ = 70.0
OUTPUT_DPI = 300
DEFAULT_PNG_MAX_COLORS = 192
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
REGION_PADDING_FRACTION = 0.05
WARNING_REGION_PADDING_FRACTION = 0.18
WARNING_REGION_MIN_SPAN_DEGREES = 5.0
STORM_MOTION_DBZ_THRESHOLD = 30.0
STORM_MOTION_TARGET_GRID_SIZE = 160
STORM_MOTION_MIN_ACTIVE_PIXELS = 12
STORM_MOTION_LOOKAHEAD_MINUTES = 30.0
STORM_MOTION_MAX_TIME_DELTA_MINUTES = 30.0
STANDARD_RETENTION_DAYS = 10
WARNING_RETENTION_DAYS: int | None = 90
WARNING_REGION_PREFIX = "warning_region"
NWS_REQUEST_HEADERS = {
    "Accept": "application/geo+json",
    "User-Agent": "RadarArchiverWebsite/1.0 (contact: local-use)",
}
WARNING_EVENT_STYLES: dict[str, dict[str, object]] = {
    "Tornado Warning": {
        "edgecolor": "#ff0000",
        "facecolor": "#ff0000",
        "linewidth": 1.4,
        "label": "Tornado Warning",
    },
    "Severe Thunderstorm Warning": {
        "edgecolor": "#ffa500",
        "facecolor": "#ffa500",
        "linewidth": 1.4,
        "label": "Severe Tstm Warning",
    },
}
WARNING_EVENT_REGION_KEYS = {
    "Tornado Warning": "tornado",
    "Severe Thunderstorm Warning": "severe",
}
NORTHEAST_STATE_NAMES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Pennsylvania", "Ohio", "Virginia", "Maryland", "Delaware",
    "District of Columbia", "West Virginia",
]
NORTHEAST_STATE_FIPS = ["23", "33", "50", "25", "44", "09", "36", "34", "42", "39", "51", "24", "10", "11", "54"]
SOUTHEAST_STATE_NAMES = [
    "Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina", "Georgia", "Florida",
    "Alabama", "Mississippi",
]
SOUTHEAST_STATE_FIPS = ["51", "21", "47", "37", "45", "13", "12", "01", "28"]
SOUTH_CENTRAL_STATE_NAMES = ["Texas", "Oklahoma", "Arkansas", "Louisiana", "New Mexico"]
SOUTH_CENTRAL_STATE_FIPS = ["48", "40", "05", "22", "35"]
NORTH_CENTRAL_STATE_NAMES = [
    "North Dakota", "South Dakota", "Nebraska", "Kansas", "Minnesota", "Iowa", "Missouri",
    "Wisconsin", "Illinois", "Indiana", "Michigan",
]
NORTH_CENTRAL_STATE_FIPS = ["38", "46", "31", "20", "27", "19", "29", "55", "17", "18", "26"]
WESTERN_STATE_NAMES = [
    "Washington", "Oregon", "California", "Nevada", "Idaho", "Montana", "Wyoming", "Utah",
    "Colorado", "Arizona",
]
WESTERN_STATE_FIPS = ["53", "41", "06", "32", "16", "30", "56", "49", "08", "04"]


def build_radar_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    bounds = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], dtype=float)
    colors = [
        "#04e9e7",
        "#019ff4",
        "#0300f4",
        "#02fd02",
        "#01c501",
        "#008e00",
        "#fdf802",
        "#e5bc00",
        "#fd9500",
        "#fd0000",
        "#d40000",
        "#bc0000",
        "#f800fd",
    ]
    cmap = ListedColormap(colors, name="traditional_radar")
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under((0, 0, 0, 0))
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def load_geojson(url: str) -> gpd.GeoDataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    for feature in payload.get("features", []):
        properties = feature.setdefault("properties", {})
        if "id" in feature and "fips" not in properties:
            properties["fips"] = str(feature["id"])

    geo_dataframe = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if geo_dataframe.empty:
        raise ValueError(f"The GeoJSON at {url} did not contain any features.")

    return geo_dataframe


def fetch_active_warning_polygons(alerts_url: str = DEFAULT_NWS_ALERTS_URL) -> gpd.GeoDataFrame:
    warning_features: list[dict[str, object]] = []

    for event_name in WARNING_EVENT_STYLES:
        try:
            response = requests.get(
                alerts_url,
                params={"event": event_name},
                headers=NWS_REQUEST_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            print(f"Unable to fetch NWS {event_name} polygons: {exc}")
            continue

        for feature in payload.get("features", []):
            geometry = feature.get("geometry")
            if geometry is None or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                continue
            warning_features.append(feature)

    if not warning_features:
        return gpd.GeoDataFrame({"event": pd.Series(dtype="object")}, geometry=[], crs="EPSG:4326")

    warnings_gdf = gpd.GeoDataFrame.from_features(warning_features, crs="EPSG:4326")
    if warnings_gdf.empty:
        return gpd.GeoDataFrame({"event": pd.Series(dtype="object")}, geometry=[], crs="EPSG:4326")

    warnings_gdf = warnings_gdf.to_crs("EPSG:4326")
    warnings_gdf = warnings_gdf[warnings_gdf.geometry.notna()].copy()
    if "event" not in warnings_gdf.columns:
        warnings_gdf["event"] = ""
    if "id" in warnings_gdf.columns:
        warnings_gdf = warnings_gdf.drop_duplicates(subset=["id"]).reset_index(drop=True)
    else:
        warnings_gdf = warnings_gdf.reset_index(drop=True)

    tornado_count = int((warnings_gdf["event"] == "Tornado Warning").sum())
    severe_count = int((warnings_gdf["event"] == "Severe Thunderstorm Warning").sum())
    print(f"Loaded NWS warning polygons: tornado={tornado_count}, severe_thunderstorm={severe_count}")
    return warnings_gdf


def warnings_for_extent(
    warnings_gdf: gpd.GeoDataFrame,
    subset_extent: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    if warnings_gdf.empty:
        return warnings_gdf

    min_lon, max_lon, min_lat, max_lat = subset_extent
    extent_polygon = box(min_lon, min_lat, max_lon, max_lat)
    extent_warnings = warnings_gdf[warnings_gdf.intersects(extent_polygon)].copy()
    return extent_warnings.reset_index(drop=True)


@lru_cache(maxsize=1)
def get_county_geodataframe() -> gpd.GeoDataFrame:
    counties = load_geojson(DEFAULT_COUNTY_URL)
    if "fips" not in counties.columns:
        raise ValueError("County GeoJSON is missing a FIPS identifier column.")

    counties["fips"] = counties["fips"].astype(str).str.zfill(5)
    return counties


@lru_cache(maxsize=1)
def get_census_states_geodataframe() -> gpd.GeoDataFrame:
    states = gpd.read_file(DEFAULT_STATES_URL)
    if states.crs is None:
        states = states.set_crs("EPSG:4326")
    else:
        states = states.to_crs("EPSG:4326")
    return states


def get_region_geodata(
    state_names: list[str],
    region_fips: list[str],
    padding_frac: float = REGION_PADDING_FRACTION,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float], object, gpd.GeoDataFrame]:
    counties = get_county_geodataframe()
    region_counties = counties[counties["fips"].str[:2].isin(region_fips)].reset_index(drop=True)
    if region_counties.empty:
        raise RuntimeError("No region counties found in GeoJSON.")

    min_lon, min_lat, max_lon, max_lat = region_counties.total_bounds
    lon_padding = (max_lon - min_lon) * padding_frac
    lat_padding = (max_lat - min_lat) * padding_frac
    extent = (
        max(CONUS_EXTENT[0], float(min_lon - lon_padding)),
        min(CONUS_EXTENT[1], float(max_lon + lon_padding)),
        max(CONUS_EXTENT[2], float(min_lat - lat_padding)),
        min(CONUS_EXTENT[3], float(max_lat + lat_padding)),
    )

    states_census_gdf = get_census_states_geodataframe()
    states_census_gdf = states_census_gdf[states_census_gdf["NAME"].isin(state_names)].reset_index(drop=True)
    state_outline = states_census_gdf.unary_union
    return region_counties, extent, state_outline, states_census_gdf


northeast_gdf, NORTHEAST_EXTENT, northeast_outline, northeast_states_gdf = get_region_geodata(
    NORTHEAST_STATE_NAMES,
    NORTHEAST_STATE_FIPS,
)
southeast_gdf, SOUTHEAST_EXTENT, southeast_outline, southeast_states_gdf = get_region_geodata(
    SOUTHEAST_STATE_NAMES,
    SOUTHEAST_STATE_FIPS,
)
south_central_gdf, SOUTH_CENTRAL_EXTENT, south_central_outline, south_central_states_gdf = get_region_geodata(
    SOUTH_CENTRAL_STATE_NAMES,
    SOUTH_CENTRAL_STATE_FIPS,
)
north_central_gdf, NORTH_CENTRAL_EXTENT, north_central_outline, north_central_states_gdf = get_region_geodata(
    NORTH_CENTRAL_STATE_NAMES,
    NORTH_CENTRAL_STATE_FIPS,
)
western_gdf, WESTERN_EXTENT, western_outline, western_states_gdf = get_region_geodata(
    WESTERN_STATE_NAMES,
    WESTERN_STATE_FIPS,
)

BASE_REGION_CONFIGS: dict[str, dict[str, object]] = {
    "northeast": {
        "label": "Northeast",
        "title": "Northeast/Mid-Atlantic US",
        "extent": NORTHEAST_EXTENT,
        "counties_gdf": northeast_gdf,
        "states_gdf": northeast_states_gdf,
        "state_outline": northeast_outline,
    },
    "southeast": {
        "label": "Southeast",
        "title": "Southeast US",
        "extent": SOUTHEAST_EXTENT,
        "counties_gdf": southeast_gdf,
        "states_gdf": southeast_states_gdf,
        "state_outline": southeast_outline,
    },
    "south_central": {
        "label": "South Central",
        "title": "South Central US",
        "extent": SOUTH_CENTRAL_EXTENT,
        "counties_gdf": south_central_gdf,
        "states_gdf": south_central_states_gdf,
        "state_outline": south_central_outline,
    },
    "north_central": {
        "label": "North Central",
        "title": "North Central US",
        "extent": NORTH_CENTRAL_EXTENT,
        "counties_gdf": north_central_gdf,
        "states_gdf": north_central_states_gdf,
        "state_outline": north_central_outline,
    },
    "western": {
        "label": "Western",
        "title": "Western US",
        "extent": WESTERN_EXTENT,
        "counties_gdf": western_gdf,
        "states_gdf": western_states_gdf,
        "state_outline": western_outline,
    },
    "conus": {
        "label": "CONUS",
        "title": "CONUS",
        "extent": CONUS_EXTENT,
        "counties_gdf": get_county_geodataframe(),
        "states_gdf": get_census_states_geodataframe(),
    },
}


def normalize_extent(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    padding_frac: float,
    min_span_degrees: float,
) -> tuple[float, float, float, float]:
    lon_span = max(max_lon - min_lon, min_span_degrees)
    lat_span = max(max_lat - min_lat, min_span_degrees)
    lon_center = (min_lon + max_lon) / 2.0
    lat_center = (min_lat + max_lat) / 2.0
    lon_half_span = (lon_span * (1.0 + padding_frac)) / 2.0
    lat_half_span = (lat_span * (1.0 + padding_frac)) / 2.0

    return (
        max(CONUS_EXTENT[0], lon_center - lon_half_span),
        min(CONUS_EXTENT[1], lon_center + lon_half_span),
        max(CONUS_EXTENT[2], lat_center - lat_half_span),
        min(CONUS_EXTENT[3], lat_center + lat_half_span),
    )


def subset_geodata_to_extent(
    counties: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    subset_extent: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    min_lon, max_lon, min_lat, max_lat = subset_extent
    extent_polygon = box(min_lon, min_lat, max_lon, max_lat)

    subset_counties = counties[counties.intersects(extent_polygon)].reset_index(drop=True)
    subset_states = states[states.intersects(extent_polygon)].reset_index(drop=True)
    return subset_counties, subset_states


def slugify_warning_state_name(state_name: str) -> str:
    slug_characters: list[str] = []
    for character in state_name.lower():
        if character.isalnum():
            slug_characters.append(character)
        elif character in {" ", "-", "/"}:
            slug_characters.append("_")

    slug = "".join(slug_characters).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "state"


def build_warning_region_key(event_name: str, state_name: str) -> str:
    event_key = WARNING_EVENT_REGION_KEYS.get(event_name, slugify_warning_state_name(event_name))
    state_key = slugify_warning_state_name(state_name)
    return f"{WARNING_REGION_PREFIX}_{event_key}_{state_key}"


def build_warning_region_title(state_name: str, event_name: str) -> str:
    return f"{state_name} {event_name}s"


def build_warning_region_configs(warnings_gdf: gpd.GeoDataFrame) -> dict[str, dict[str, object]]:
    if warnings_gdf.empty:
        return {}

    counties = get_county_geodataframe()
    states = get_census_states_geodataframe()
    region_configs: dict[str, dict[str, object]] = {}

    for event_name in WARNING_EVENT_STYLES:
        event_warnings = warnings_gdf[warnings_gdf["event"] == event_name].copy()
        if event_warnings.empty:
            continue

        event_coverage = event_warnings.geometry.unary_union
        impacted_states = states[states.intersects(event_coverage)].sort_values("NAME").reset_index(drop=True)

        for state_row in impacted_states.itertuples(index=False):
            state_name = str(state_row.NAME)
            state_geometry = state_row.geometry
            state_warnings = event_warnings[event_warnings.intersects(state_geometry)].copy().reset_index(drop=True)
            if state_warnings.empty:
                continue

            min_lon, min_lat, max_lon, max_lat = state_geometry.bounds
            extent = normalize_extent(
                float(min_lon),
                float(max_lon),
                float(min_lat),
                float(max_lat),
                padding_frac=WARNING_REGION_PADDING_FRACTION,
                min_span_degrees=WARNING_REGION_MIN_SPAN_DEGREES,
            )
            region_counties, region_states = subset_geodata_to_extent(counties, states, extent)
            region_key = build_warning_region_key(event_name, state_name)
            region_title = build_warning_region_title(state_name, event_name)

            region_configs[region_key] = {
                "label": state_name,
                "title": region_title,
                "extent": extent,
                "counties_gdf": region_counties,
                "states_gdf": region_states,
                "warnings_gdf": state_warnings,
                "metadata": {
                    "key": region_key,
                    "label": state_name,
                    "title": region_title,
                    "event": event_name,
                    "events": [event_name],
                    "grouping": "state",
                    "state": state_name,
                },
            }

    return region_configs


def build_region_configs(warnings_gdf: gpd.GeoDataFrame) -> dict[str, dict[str, object]]:
    region_configs = {
        region_key: region_config
        for region_key, region_config in BASE_REGION_CONFIGS.items()
        if region_key != "conus"
    }
    region_configs.update(build_warning_region_configs(warnings_gdf))
    return region_configs


def is_warning_region_key(region_key: str) -> bool:
    legacy_prefixes = ("tornado_warnings", "severe_thunderstorm_warnings")
    return region_key.startswith(f"{WARNING_REGION_PREFIX}_") or any(
        region_key == base_region_key or region_key.startswith(f"{base_region_key}_")
        for base_region_key in legacy_prefixes
    )


def write_warning_region_metadata(region_key: str, metadata: dict[str, object]) -> None:
    if not is_warning_region_key(region_key):
        return

    region_dir = ARCHIVE_ROOT / region_key
    region_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = region_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_output_path(output_path: Path, region_key: str) -> Path:
    if region_key == "conus":
        return output_path

    return output_path.with_name(f"{output_path.stem}_{region_key}{output_path.suffix}")


def build_archive_output_path(region_key: str, valid_time: pd.Timestamp) -> Path:
    eastern_time = valid_time.tz_convert(EASTERN_TIMEZONE)
    timestamp = valid_time.tz_convert(timezone.utc).strftime("%Y%m%d_%H%M")
    region_dir = ARCHIVE_ROOT / region_key
    if is_warning_region_key(region_key):
        region_dir = region_dir / eastern_time.strftime("%y-%m-%d")
    return region_dir / f"mrms_{region_key}_{timestamp}.png"


def build_grib_archive_path(valid_time: pd.Timestamp) -> Path:
    eastern_time = valid_time.tz_convert(EASTERN_TIMEZONE)
    date_dir = GRIB_ARCHIVE_ROOT / eastern_time.strftime("%y-%m-%d")
    timestamp = valid_time.tz_convert(timezone.utc).strftime("%Y%m%d_%H%M")
    return date_dir / f"mrms_{timestamp}.grib2"


def prune_archived_files(archive_root: Path, pattern: str, retention_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    if not archive_root.exists():
        return

    for file_path in archive_root.rglob(pattern):
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        if modified_time < cutoff:
            file_path.unlink(missing_ok=True)


def prune_empty_directories(archive_root: Path) -> None:
    if not archive_root.exists():
        return

    for directory in sorted((path for path in archive_root.rglob("*") if path.is_dir()), reverse=True):
        try:
            next(directory.iterdir())
        except StopIteration:
            directory.rmdir()


def prune_radar_archives(archive_root: Path) -> None:
    if not archive_root.exists():
        return

    for region_dir in (path for path in archive_root.iterdir() if path.is_dir()):
        retention_days = WARNING_RETENTION_DAYS if is_warning_region_key(region_dir.name) else STANDARD_RETENTION_DAYS
        if retention_days is None:
            continue
        prune_archived_files(region_dir, "*.png", retention_days)

    prune_empty_directories(archive_root)


def download_and_archive_grib(url: str) -> tuple[Path, pd.Timestamp]:
    valid_time: pd.Timestamp

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        compressed_path = temp_path / "mrms.grib2.gz"
        grib_path = temp_path / "mrms.grib2"

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with compressed_path.open("wb") as compressed_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        compressed_file.write(chunk)

        with gzip.open(compressed_path, "rb") as src, grib_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        grib_file = pygrib.open(str(grib_path))
        try:
            message = grib_file.message(1)
            valid_time = pd.Timestamp(message.validDate, tz="UTC")
        finally:
            grib_file.close()

        grib_archive_path = build_grib_archive_path(valid_time)
        grib_archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(grib_path, grib_archive_path)
        print(f"Archived GRIB: {grib_archive_path}")

    return grib_archive_path, valid_time


def load_radar_subset(
    grib_path: Path,
    subset_extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, pd.Timestamp]:
    min_lon, max_lon, min_lat, max_lat = subset_extent
    lon_bounds = [lon if lon >= 0.0 else lon + 360.0 for lon in (min_lon, max_lon)]

    grib_file = pygrib.open(str(grib_path))
    try:
        message = grib_file.message(1)
        values, latitudes, longitudes = message.data(
            lat1=min_lat,
            lat2=max_lat,
            lon1=min(lon_bounds),
            lon2=max(lon_bounds),
        )
        valid_time = pd.Timestamp(message.validDate, tz="UTC")
    finally:
        grib_file.close()

    if values.size == 0:
        raise ValueError("The requested extent did not intersect the MRMS grid.")

    lon_grid = np.asarray(longitudes, dtype=np.float32)
    lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)
    lat_grid = np.asarray(latitudes, dtype=np.float32)
    reflectivity = np.asarray(values, dtype=np.float32)
    reflectivity = np.ma.masked_where((reflectivity < MIN_DBZ) | (reflectivity <= -99.0), reflectivity)
    return lon_grid, lat_grid, reflectivity, valid_time


def find_previous_grib_archive(current_grib_path: Path, current_valid_time: pd.Timestamp) -> Path | None:
    previous_candidates: list[tuple[pd.Timestamp, Path]] = []
    for archive_path in GRIB_ARCHIVE_ROOT.rglob("*.grib2"):
        if archive_path == current_grib_path:
            continue

        timestamp_text = archive_path.stem.removeprefix("mrms_")
        try:
            archive_time = pd.Timestamp(datetime.strptime(timestamp_text, "%Y%m%d_%H%M"), tz="UTC")
        except ValueError:
            continue

        if archive_time < current_valid_time:
            previous_candidates.append((archive_time, archive_path))

    if not previous_candidates:
        return None

    previous_candidates.sort(key=lambda item: item[0])
    return previous_candidates[-1][1]


def build_storm_centroid(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    reflectivity: np.ma.MaskedArray,
) -> tuple[float, float] | None:
    if reflectivity.size == 0:
        return None

    stride = max(1, int(np.ceil(max(reflectivity.shape) / STORM_MOTION_TARGET_GRID_SIZE)))
    sampled_reflectivity = np.ma.filled(reflectivity[::stride, ::stride], fill_value=0.0)
    weights = np.clip(sampled_reflectivity - STORM_MOTION_DBZ_THRESHOLD, 0.0, None).astype(np.float32, copy=False)
    active_pixel_count = int(np.count_nonzero(weights))
    if active_pixel_count < STORM_MOTION_MIN_ACTIVE_PIXELS:
        return None

    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return None

    sampled_lon = lon_grid[::stride, ::stride]
    sampled_lat = lat_grid[::stride, ::stride]
    centroid_lon = float((sampled_lon * weights).sum() / total_weight)
    centroid_lat = float((sampled_lat * weights).sum() / total_weight)
    return centroid_lon, centroid_lat


def clamp_point_to_extent(
    point: tuple[float, float],
    subset_extent: tuple[float, float, float, float],
) -> tuple[float, float]:
    min_lon, max_lon, min_lat, max_lat = subset_extent
    point_lon, point_lat = point
    return (
        min(max(point_lon, min_lon), max_lon),
        min(max(point_lat, min_lat), max_lat),
    )


def estimate_storm_motion_line(
    previous_grib_path: Path | None,
    valid_time: pd.Timestamp,
    subset_extent: tuple[float, float, float, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    reflectivity: np.ma.MaskedArray,
) -> dict[str, tuple[float, float] | float] | None:
    if previous_grib_path is None:
        return None

    previous_lon_grid, previous_lat_grid, previous_reflectivity, previous_valid_time = load_radar_subset(
        previous_grib_path,
        subset_extent,
    )

    current_centroid = build_storm_centroid(lon_grid, lat_grid, reflectivity)
    previous_centroid = build_storm_centroid(previous_lon_grid, previous_lat_grid, previous_reflectivity)
    del previous_lon_grid, previous_lat_grid, previous_reflectivity
    gc.collect()

    if current_centroid is None or previous_centroid is None:
        return None

    delta_minutes = (valid_time - previous_valid_time).total_seconds() / 60.0
    if delta_minutes <= 0.0 or delta_minutes > STORM_MOTION_MAX_TIME_DELTA_MINUTES:
        return None

    motion_lon = current_centroid[0] - previous_centroid[0]
    motion_lat = current_centroid[1] - previous_centroid[1]
    if abs(motion_lon) < 0.01 and abs(motion_lat) < 0.01:
        return None

    lookahead_scale = STORM_MOTION_LOOKAHEAD_MINUTES / delta_minutes
    motion_endpoint = (
        current_centroid[0] + (motion_lon * lookahead_scale),
        current_centroid[1] + (motion_lat * lookahead_scale),
    )
    motion_endpoint = clamp_point_to_extent(motion_endpoint, subset_extent)

    return {
        "start": current_centroid,
        "end": motion_endpoint,
        "minutes": delta_minutes,
    }


def subset_radar_grid(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    values: np.ndarray,
    subset_extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:

    min_lon, max_lon, min_lat, max_lat = subset_extent
    row_mask = (lat_1d >= min_lat) & (lat_1d <= max_lat)
    col_mask = (lon_1d >= min_lon) & (lon_1d <= max_lon)
    if not np.any(row_mask) or not np.any(col_mask):
        raise ValueError("The requested extent did not intersect the MRMS grid.")

    subset = values[np.ix_(row_mask, col_mask)]

    subset_lats = lat_1d[row_mask]
    subset_lons = lon_1d[col_mask]
    lon_grid, lat_grid = np.meshgrid(subset_lons, subset_lats)
    subset = np.ma.masked_where((subset < MIN_DBZ) | (subset <= -99.0), subset)
    return lon_grid, lat_grid, subset


def build_title(valid_time: pd.Timestamp, region_title: str) -> str:
    eastern_time = valid_time.tz_convert(EASTERN_TIMEZONE)
    return (
        "MRMS Reflectivity At Lowest Altitude\n"
        f"{region_title} | Valid {eastern_time:%Y-%m-%d %I:%M %p %Z}"
    )


def build_optimized_png_bytes(image: Image.Image, max_colors: int) -> bytes:
    working_image = image.copy()
    has_alpha = "A" in working_image.getbands() or working_image.info.get("transparency") is not None

    if max_colors > 0:
        if has_alpha:
            working_image = working_image.convert("RGBA").quantize(
                colors=max_colors,
                method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.NONE,
            )
        else:
            working_image = working_image.convert("P", palette=Image.Palette.ADAPTIVE, colors=max_colors)

    output_buffer = io.BytesIO()
    working_image.save(output_buffer, format="PNG", optimize=True, compress_level=9)
    return output_buffer.getvalue()


def optimize_png_file(path: Path, max_colors: int) -> bool:
    original_size = path.stat().st_size

    with Image.open(path) as image:
        optimized_bytes = build_optimized_png_bytes(image, max_colors=max_colors)

    optimized_size = len(optimized_bytes)
    if optimized_size >= original_size:
        print(f"PNG optimization skipped for {path}: no size improvement")
        return False

    path.write_bytes(optimized_bytes)
    savings = original_size - optimized_size
    percent = (savings / original_size) * 100 if original_size else 0.0
    print(f"Optimized PNG for {path}: {original_size} -> {optimized_size} bytes ({percent:.1f}% smaller)")
    return True


def plot_radar(
    counties: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    warnings_gdf: gpd.GeoDataFrame,
    region_key: str,
    region_title: str,
    subset_extent: tuple[float, float, float, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    reflectivity: np.ma.MaskedArray,
    valid_time: pd.Timestamp,
    output_path: Path,
    show: bool,
    png_max_colors: int,
    warning_metadata: dict[str, object] | None = None,
    storm_motion_line: dict[str, tuple[float, float] | float] | None = None,
) -> Path:
    figure = plt.figure(figsize=(10, 10))
    projection: ccrs.CRS
    if region_key == "conus":
        projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=39)
    else:
        projection = ccrs.PlateCarree()

    axis = plt.axes(projection=projection)
    axis.set_extent(subset_extent, crs=ccrs.PlateCarree())

    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1e8")
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dceaf7")
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dceaf7", edgecolor="#9ab6d3")
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)

    axis.add_geometries(
        counties.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#8f8f8f",
        linewidth=0.72,
        alpha=0.45,
        zorder=3,
    )

    if not states.empty:
        axis.add_geometries(
            states.boundary.geometry,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#4f4f4f",
            linewidth=0.7,
            zorder=4,
        )
    else:
        axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6, edgecolor="#4f4f4f", zorder=4)

    legend_handles: list[Line2D] = []
    region_warnings = warnings_for_extent(warnings_gdf, subset_extent)
    if not region_warnings.empty:
        for event_name, style in WARNING_EVENT_STYLES.items():
            event_warnings = region_warnings[region_warnings["event"] == event_name]
            if event_warnings.empty:
                continue

            axis.add_geometries(
                event_warnings.geometry,
                crs=ccrs.PlateCarree(),
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                alpha=0.12,
                zorder=5,
            )
            axis.add_geometries(
                event_warnings.geometry,
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                alpha=0.95,
                zorder=6,
            )
            legend_handles.append(
                Line2D([0], [0], color=style["edgecolor"], linewidth=style["linewidth"], label=style["label"])
            )

    if storm_motion_line is not None:
        start_lon, start_lat = storm_motion_line["start"]
        end_lon, end_lat = storm_motion_line["end"]
        axis.plot(
            [start_lon, end_lon],
            [start_lat, end_lat],
            transform=ccrs.PlateCarree(),
            color="white",
            linewidth=3.6,
            alpha=0.95,
            zorder=7,
        )
        axis.plot(
            [start_lon, end_lon],
            [start_lat, end_lat],
            transform=ccrs.PlateCarree(),
            color="#1f1f1f",
            linewidth=2.0,
            linestyle=(0, (8, 4)),
            zorder=8,
        )
        axis.scatter(
            [start_lon],
            [start_lat],
            transform=ccrs.PlateCarree(),
            s=24,
            color="#1f1f1f",
            edgecolors="white",
            linewidths=0.8,
            zorder=9,
        )
        legend_handles.append(
            Line2D([0], [0], color="#1f1f1f", linewidth=2.0, linestyle=(0, (8, 4)), label="Est. 30-min Motion")
        )

    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=8,
            framealpha=0.88,
            facecolor="white",
        )

    cmap, norm = build_radar_colormap()

    mesh = axis.pcolormesh(
        lon_grid,
        lat_grid,
        reflectivity,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=2,
    )

    colorbar = plt.colorbar(
        mesh,
        ax=axis,
        orientation="horizontal",
        pad=0.02,
        shrink=0.68,
        aspect=40,
        ticks=norm.boundaries,
    )
    colorbar.set_label("Reflectivity (dBZ)")

    axis.set_title(build_title(valid_time, region_title), fontsize=13, pad=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_output_path = build_archive_output_path(region_key, valid_time)
    archive_output_path.parent.mkdir(parents=True, exist_ok=True)
    if warning_metadata is not None:
        write_warning_region_metadata(region_key, warning_metadata)

    figure.savefig(archive_output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    optimize_png_file(archive_output_path, max_colors=png_max_colors)
    shutil.copy2(archive_output_path, output_path)
    print(f"Created PNG for {region_key}: {archive_output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    plt.close(figure)
    gc.collect()
    return archive_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MRMS lowest-altitude reflectivity over the CONUS domain."
    )
    parser.add_argument(
        "--radar-url",
        default=DEFAULT_RADAR_URL,
        help="MRMS GRIB2 GZ URL to plot.",
    )
    parser.add_argument(
        "--counties-url",
        default=DEFAULT_COUNTY_URL,
        help="County boundary GeoJSON URL to overlay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path. Defaults to mrms_conus_radar.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving the PNG.",
    )
    parser.add_argument(
        "--png-max-colors",
        type=int,
        default=DEFAULT_PNG_MAX_COLORS,
        help="Palette size used when optimizing output PNGs. Use 0 to disable quantization. Default: 192.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.png_max_colors < 0 or args.png_max_colors > 256:
        raise SystemExit("--png-max-colors must be between 0 and 256")

    grib_archive_path, valid_time = download_and_archive_grib(args.radar_url)
    previous_grib_path = find_previous_grib_archive(grib_archive_path, valid_time)
    warnings_gdf = fetch_active_warning_polygons()
    region_configs = build_region_configs(warnings_gdf)

    for region_key, region_config in region_configs.items():
        print(f"Generating PNG for {region_key}...")
        extent = region_config["extent"]
        lon_grid, lat_grid, reflectivity, _ = load_radar_subset(grib_archive_path, extent)
        storm_motion_line = estimate_storm_motion_line(
            previous_grib_path=previous_grib_path,
            valid_time=valid_time,
            subset_extent=extent,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            reflectivity=reflectivity,
        )
        plot_radar(
            counties=region_config["counties_gdf"],
            states=region_config["states_gdf"],
            warnings_gdf=region_config.get("warnings_gdf", warnings_gdf),
            region_key=region_key,
            region_title=str(region_config["title"]),
            subset_extent=extent,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            reflectivity=reflectivity,
            valid_time=valid_time,
            output_path=build_output_path(args.output, region_key),
            show=args.show,
            png_max_colors=args.png_max_colors,
            warning_metadata=region_config.get("metadata"),
            storm_motion_line=storm_motion_line,
        )
        del lon_grid, lat_grid, reflectivity, storm_motion_line
        gc.collect()

    prune_radar_archives(ARCHIVE_ROOT)
    prune_archived_files(GRIB_ARCHIVE_ROOT, "*.grib2", STANDARD_RETENTION_DAYS)
    prune_empty_directories(GRIB_ARCHIVE_ROOT)


if __name__ == "__main__":
    main()
