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
from scipy.ndimage import gaussian_filter
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
DEFAULT_STORM_MOTION_FORECAST_MINUTES = 20
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
REGION_PADDING_FRACTION = 0.05
WARNING_REGION_PADDING_FRACTION = 0.18
WARNING_REGION_MIN_SPAN_DEGREES = 5.0
STANDARD_RETENTION_DAYS = 10
WARNING_RETENTION_DAYS: int | None = 90
WARNING_REGION_PREFIX = "warning_region"
STORM_MOTION_SAMPLE_DBZ = 20.0
STORM_MOTION_ANCHOR_DBZ = 35.0
STORM_MOTION_MIN_SIGNAL_PIXELS = 120
STORM_MOTION_MAX_SHIFT_PER_MINUTE = 2.5
STORM_MOTION_MAX_SHIFT_CELLS = 32
STORM_MOTION_MAX_ANCHORS = 18
STORM_MOTION_MAX_WORKING_GRID_DIMENSION = 600
STORM_MOTION_LINE_FRACTION_OF_REGION = 0.12
STORM_MOTION_MIN_LINE_DEGREES = 0.6
STORM_MOTION_MAX_LINE_DEGREES = 2.4
STORM_MOTION_LINE_COLOR = "#ffffff"
STORM_MOTION_LINE_OUTLINE_COLOR = "#202020"
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
    region_configs = dict(BASE_REGION_CONFIGS)
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


def parse_grib_archive_time(grib_path: Path) -> pd.Timestamp | None:
    stem = grib_path.stem
    if not stem.startswith("mrms_"):
        return None

    timestamp_text = stem.removeprefix("mrms_")
    try:
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d_%H%M")
    except ValueError:
        return None

    return pd.Timestamp(timestamp, tz="UTC")


def load_grib_grid_from_file(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    grib_file = pygrib.open(str(grib_path))
    try:
        message = grib_file.message(1)
        values = np.asarray(message.values, dtype=np.float32)
        valid_time = pd.Timestamp(message.validDate, tz="UTC")
        latitudes, longitudes = message.latlons()
    finally:
        grib_file.close()

    lon_1d = longitudes[0].astype(np.float32, copy=True)
    lon_1d = np.where(lon_1d > 180.0, lon_1d - 360.0, lon_1d)
    lat_1d = latitudes[:, 0].astype(np.float32, copy=True)
    return lon_1d, lat_1d, values, valid_time


def find_previous_grib_archive(current_valid_time: pd.Timestamp, current_grib_path: Path) -> Path | None:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for grib_path in GRIB_ARCHIVE_ROOT.rglob("*.grib2"):
        if grib_path == current_grib_path:
            continue

        archive_time = parse_grib_archive_time(grib_path)
        if archive_time is None or archive_time >= current_valid_time:
            continue
        candidates.append((archive_time, grib_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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


def load_radar_grid(url: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, Path]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        compressed_path = temp_path / "mrms.grib2.gz"
        grib_path = temp_path / "mrms.grib2"
        compressed_path.write_bytes(response.content)

        with gzip.open(compressed_path, "rb") as src, grib_path.open("wb") as dst:
            dst.write(src.read())

        lon_1d, lat_1d, values, valid_time = load_grib_grid_from_file(grib_path)

        grib_archive_path = build_grib_archive_path(valid_time)
        grib_archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(grib_path, grib_archive_path)
        print(f"Archived GRIB: {grib_archive_path}")

    return lon_1d, lat_1d, values, valid_time, grib_archive_path


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


def determine_motion_stride(reflectivity_shape: tuple[int, int]) -> int:
    largest_dimension = max(reflectivity_shape)
    return max(1, int(np.ceil(largest_dimension / STORM_MOTION_MAX_WORKING_GRID_DIMENSION)))


def prepare_motion_field(
    reflectivity: np.ma.MaskedArray,
    stride: int,
    minimum_signal_pixels: int,
) -> np.ndarray | None:
    prepared = np.asarray(np.ma.filled(reflectivity[::stride, ::stride], 0.0), dtype=np.float32)
    np.putmask(prepared, prepared < STORM_MOTION_SAMPLE_DBZ, 0.0)
    if int(np.count_nonzero(prepared)) < minimum_signal_pixels:
        return None

    max_value = float(prepared.max(initial=0.0))
    if max_value <= 0.0:
        return None

    prepared /= max_value
    gaussian_filter(prepared, sigma=1.2, output=prepared)
    return prepared


def overlapping_slice_pair(length: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(shift, length), slice(0, length - shift)
    return slice(0, length + shift), slice(-shift, length)


def estimate_storm_motion(
    previous_reflectivity: np.ma.MaskedArray,
    current_reflectivity: np.ma.MaskedArray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    previous_valid_time: pd.Timestamp,
    current_valid_time: pd.Timestamp,
) -> tuple[float, float] | None:
    stride = determine_motion_stride(current_reflectivity.shape)
    minimum_signal_pixels = max(24, STORM_MOTION_MIN_SIGNAL_PIXELS // (stride * stride))
    current_field = prepare_motion_field(
        current_reflectivity,
        stride=stride,
        minimum_signal_pixels=minimum_signal_pixels,
    )
    previous_field = prepare_motion_field(
        previous_reflectivity,
        stride=stride,
        minimum_signal_pixels=minimum_signal_pixels,
    )
    if current_field is None or previous_field is None:
        return None

    minutes_between_frames = (current_valid_time - previous_valid_time).total_seconds() / 60.0
    if minutes_between_frames <= 0:
        return None

    max_shift = int(round(minutes_between_frames * STORM_MOTION_MAX_SHIFT_PER_MINUTE / stride))
    max_shift = max(3, min(STORM_MOTION_MAX_SHIFT_CELLS, max_shift))

    best_score = -1.0
    best_offset = (0, 0)
    for row_shift in range(-max_shift, max_shift + 1):
        current_rows, previous_rows = overlapping_slice_pair(current_field.shape[0], row_shift)
        for column_shift in range(-max_shift, max_shift + 1):
            current_columns, previous_columns = overlapping_slice_pair(current_field.shape[1], column_shift)
            current_window = current_field[current_rows, current_columns]
            previous_window = previous_field[previous_rows, previous_columns]

            if max(int(np.count_nonzero(current_window)), int(np.count_nonzero(previous_window))) < minimum_signal_pixels:
                continue

            current_energy = float(np.einsum("ij,ij->", current_window, current_window, optimize=True))
            previous_energy = float(np.einsum("ij,ij->", previous_window, previous_window, optimize=True))
            denominator = float(np.sqrt(current_energy * previous_energy))
            if denominator <= 0.0:
                continue

            score = float(np.einsum("ij,ij->", current_window, previous_window, optimize=True) / denominator)
            if score > best_score:
                best_score = score
                best_offset = (row_shift * stride, column_shift * stride)

    if best_score <= 0.0 or best_offset == (0, 0):
        return None

    if lon_grid.shape[1] < 2 or lat_grid.shape[0] < 2:
        return None

    lon_step = float(np.median(np.diff(lon_grid[0, :])))
    lat_step = float(np.median(np.diff(lat_grid[:, 0])))
    if lon_step == 0.0 and lat_step == 0.0:
        return None

    return (
        (best_offset[1] * lon_step) / minutes_between_frames,
        (best_offset[0] * lat_step) / minutes_between_frames,
    )


def select_motion_anchor_indices(reflectivity: np.ma.MaskedArray) -> list[tuple[int, int]]:
    reflectivity_values = np.asarray(np.ma.filled(reflectivity, 0.0), dtype=np.float32)
    candidate_indices = np.argwhere(reflectivity_values >= STORM_MOTION_ANCHOR_DBZ)
    if candidate_indices.size == 0:
        return []

    anchor_spacing = max(12, min(reflectivity_values.shape) // 10)
    anchor_spacing_squared = anchor_spacing * anchor_spacing
    candidate_strengths = reflectivity_values[candidate_indices[:, 0], candidate_indices[:, 1]]
    sorted_candidate_order = np.argsort(candidate_strengths)[::-1]

    selected_indices: list[tuple[int, int]] = []
    for candidate_index in sorted_candidate_order:
        row_index, column_index = candidate_indices[candidate_index]
        if any(
            (row_index - selected_row) ** 2 + (column_index - selected_column) ** 2 < anchor_spacing_squared
            for selected_row, selected_column in selected_indices
        ):
            continue

        selected_indices.append((int(row_index), int(column_index)))
        if len(selected_indices) >= STORM_MOTION_MAX_ANCHORS:
            break

    return selected_indices


def draw_storm_motion_lines(
    axis: plt.Axes,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    reflectivity: np.ma.MaskedArray,
    subset_extent: tuple[float, float, float, float],
    storm_motion_per_minute: tuple[float, float] | None,
    forecast_minutes: int,
) -> bool:
    if storm_motion_per_minute is None or forecast_minutes <= 0:
        return False

    delta_lon = storm_motion_per_minute[0] * forecast_minutes
    delta_lat = storm_motion_per_minute[1] * forecast_minutes
    motion_length = float(np.hypot(delta_lon, delta_lat))
    if motion_length <= 0.0:
        return False

    region_span = max(subset_extent[1] - subset_extent[0], subset_extent[3] - subset_extent[2])
    max_line_length = min(
        STORM_MOTION_MAX_LINE_DEGREES,
        max(STORM_MOTION_MIN_LINE_DEGREES, region_span * STORM_MOTION_LINE_FRACTION_OF_REGION),
    )
    if motion_length > max_line_length:
        scale_factor = max_line_length / motion_length
        delta_lon *= scale_factor
        delta_lat *= scale_factor

    anchor_indices = select_motion_anchor_indices(reflectivity)
    if not anchor_indices:
        return False

    for row_index, column_index in anchor_indices:
        anchor_lon = float(lon_grid[row_index, column_index])
        anchor_lat = float(lat_grid[row_index, column_index])
        line_lons = [anchor_lon, anchor_lon + delta_lon]
        line_lats = [anchor_lat, anchor_lat + delta_lat]
        axis.plot(
            line_lons,
            line_lats,
            transform=ccrs.PlateCarree(),
            color=STORM_MOTION_LINE_OUTLINE_COLOR,
            linewidth=3.2,
            alpha=0.92,
            solid_capstyle="round",
            zorder=7,
        )
        axis.plot(
            line_lons,
            line_lats,
            transform=ccrs.PlateCarree(),
            color=STORM_MOTION_LINE_COLOR,
            linewidth=1.7,
            alpha=0.96,
            solid_capstyle="round",
            zorder=8,
        )

    return True


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
    storm_motion_per_minute: tuple[float, float] | None,
    storm_motion_forecast_minutes: int,
    warning_metadata: dict[str, object] | None = None,
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

            # Draw storm motion for each warning polygon
            for _, warning_row in event_warnings.iterrows():
                poly = warning_row.geometry
                if poly is None or poly.is_empty:
                    continue
                min_lon, min_lat, max_lon, max_lat = poly.bounds
                poly_extent = (
                    max(CONUS_EXTENT[0], float(min_lon)),
                    min(CONUS_EXTENT[1], float(max_lon)),
                    max(CONUS_EXTENT[2], float(min_lat)),
                    min(CONUS_EXTENT[3], float(max_lat)),
                )
                try:
                    poly_lon_grid, poly_lat_grid, poly_reflectivity = subset_radar_grid(
                        lon_grid[0], lat_grid[:,0], reflectivity.data, poly_extent
                    )
                except Exception:
                    continue
                # Mask out points not in the polygon
                from shapely.geometry import Point
                mask = np.array([
                    [poly.contains(Point(lon, lat)) for lon in poly_lon_grid[0]]
                    for lat in poly_lat_grid[:,0]
                ])
                poly_reflectivity = np.ma.masked_where(~mask, poly_reflectivity)
                # Use the global storm_motion_per_minute for now (could be improved to estimate per-polygon)
                draw_storm_motion_lines(
                    axis=axis,
                    lon_grid=poly_lon_grid,
                    lat_grid=poly_lat_grid,
                    reflectivity=poly_reflectivity,
                    subset_extent=poly_extent,
                    storm_motion_per_minute=storm_motion_per_minute,
                    forecast_minutes=storm_motion_forecast_minutes,
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

    if draw_storm_motion_lines(
        axis=axis,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        reflectivity=reflectivity,
        subset_extent=subset_extent,
        storm_motion_per_minute=storm_motion_per_minute,
        forecast_minutes=storm_motion_forecast_minutes,
    ):
        legend_handles.append(
            Line2D([0], [0], color=STORM_MOTION_LINE_COLOR, linewidth=2.2, label="Estimated Storm Motion")
        )

    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=8,
            framealpha=0.88,
            facecolor="white",
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
    parser.add_argument(
        "--no-storm-motion-lines",
        action="store_true",
        help="Disable estimated storm motion line overlays.",
    )
    parser.add_argument(
        "--storm-motion-forecast-minutes",
        type=int,
        default=DEFAULT_STORM_MOTION_FORECAST_MINUTES,
        help="Projected duration used when drawing storm motion lines. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.png_max_colors < 0 or args.png_max_colors > 256:
        raise SystemExit("--png-max-colors must be between 0 and 256")
    if args.storm_motion_forecast_minutes < 0 or args.storm_motion_forecast_minutes > 120:
        raise SystemExit("--storm-motion-forecast-minutes must be between 0 and 120")

    lon_1d, lat_1d, values, valid_time, current_grib_archive_path = load_radar_grid(args.radar_url)
    previous_grid: tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp] | None = None
    if not args.no_storm_motion_lines:
        previous_grib_archive_path = find_previous_grib_archive(valid_time, current_grib_archive_path)
        if previous_grib_archive_path is not None:
            previous_lon_1d, previous_lat_1d, previous_values, previous_valid_time = load_grib_grid_from_file(
                previous_grib_archive_path
            )
            if np.array_equal(previous_lon_1d, lon_1d) and np.array_equal(previous_lat_1d, lat_1d):
                previous_grid = (previous_lon_1d, previous_lat_1d, previous_values, previous_valid_time)
            else:
                print("Skipping storm motion lines because the previous GRIB grid did not match the current grid.")
        else:
            print("Skipping storm motion lines because no previous archived GRIB was available.")

    warnings_gdf = fetch_active_warning_polygons()
    region_configs = build_region_configs(warnings_gdf)

    for region_key, region_config in region_configs.items():
        print(f"Generating PNG for {region_key}...")
        extent = region_config["extent"]
        lon_grid, lat_grid, reflectivity = subset_radar_grid(lon_1d, lat_1d, values, extent)
        storm_motion_per_minute: tuple[float, float] | None = None
        if previous_grid is not None and args.storm_motion_forecast_minutes > 0:
            _, _, previous_values, previous_valid_time = previous_grid
            _, _, previous_reflectivity = subset_radar_grid(lon_1d, lat_1d, previous_values, extent)
            storm_motion_per_minute = estimate_storm_motion(
                previous_reflectivity=previous_reflectivity,
                current_reflectivity=reflectivity,
                lon_grid=lon_grid,
                lat_grid=lat_grid,
                previous_valid_time=previous_valid_time,
                current_valid_time=valid_time,
            )
            del previous_reflectivity

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
            storm_motion_per_minute=storm_motion_per_minute,
            storm_motion_forecast_minutes=args.storm_motion_forecast_minutes,
            warning_metadata=region_config.get("metadata"),
        )
        del lon_grid, lat_grid, reflectivity
        gc.collect()

    prune_radar_archives(ARCHIVE_ROOT)
    prune_archived_files(GRIB_ARCHIVE_ROOT, "*.grib2", STANDARD_RETENTION_DAYS)
    prune_empty_directories(GRIB_ARCHIVE_ROOT)


if __name__ == "__main__":
    main()
