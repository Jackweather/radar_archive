from __future__ import annotations

import argparse
import gc
import gzip
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


DEFAULT_COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
DEFAULT_STATES_URL = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
DEFAULT_RADAR_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
DEFAULT_OUTPUT = Path("mrms_conus_radar.png")
BASE_DIR = Path("/var/data")
ARCHIVE_ROOT = BASE_DIR / "mrms_radar_archive"
CONUS_EXTENT = (-124.0, -67.5, 25.0, 49.5)
MIN_DBZ = 5.0
MAX_DBZ = 70.0
OUTPUT_DPI = 300
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
REGION_PADDING_FRACTION = 0.05
RETENTION_DAYS = 10

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
    "Wisconsin", "Illinois", "Michigan",
]
NORTH_CENTRAL_STATE_FIPS = ["38", "46", "31", "20", "27", "19", "29", "55", "17", "26"]
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

REGION_CONFIGS: dict[str, dict[str, object]] = {
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


def build_output_path(output_path: Path, region_key: str) -> Path:
    if region_key == "conus":
        return output_path

    return output_path.with_name(f"{output_path.stem}_{region_key}{output_path.suffix}")


def build_archive_output_path(region_key: str, valid_time: pd.Timestamp) -> Path:
    timestamp = valid_time.tz_convert(timezone.utc).strftime("%Y%m%d_%H%M")
    region_dir = ARCHIVE_ROOT / region_key
    return region_dir / f"mrms_{region_key}_{timestamp}.png"


def prune_archived_pngs(archive_root: Path, retention_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    if not archive_root.exists():
        return

    for png_path in archive_root.rglob("*.png"):
        modified_time = datetime.fromtimestamp(png_path.stat().st_mtime, tz=timezone.utc)
        if modified_time < cutoff:
            png_path.unlink(missing_ok=True)


def load_radar_grid(url: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        compressed_path = temp_path / "mrms.grib2.gz"
        grib_path = temp_path / "mrms.grib2"
        compressed_path.write_bytes(response.content)

        with gzip.open(compressed_path, "rb") as src, grib_path.open("wb") as dst:
            dst.write(src.read())

        grib_file = pygrib.open(str(grib_path))
        message = grib_file.message(1)
        values = np.asarray(message.values, dtype=np.float32)
        valid_time = pd.Timestamp(message.validDate, tz="UTC")
        latitudes, longitudes = message.latlons()
        grib_file.close()

    lon_1d = longitudes[0].astype(np.float32, copy=True)
    lon_1d = np.where(lon_1d > 180.0, lon_1d - 360.0, lon_1d)
    lat_1d = latitudes[:, 0].astype(np.float32, copy=True)

    return lon_1d, lat_1d, values, valid_time


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


def build_title(valid_time: pd.Timestamp, region_key: str) -> str:
    eastern_time = valid_time.tz_convert(EASTERN_TIMEZONE)
    region_label = str(REGION_CONFIGS[region_key]["title"])
    return (
        "MRMS Reflectivity At Lowest Altitude\n"
        f"{region_label} | Valid {eastern_time:%Y-%m-%d %I:%M %p %Z}"
    )


def plot_radar(
    counties: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    region_key: str,
    subset_extent: tuple[float, float, float, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    reflectivity: np.ma.MaskedArray,
    valid_time: pd.Timestamp,
    output_path: Path,
    show: bool,
) -> None:
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

    axis.set_title(build_title(valid_time, region_key), fontsize=13, pad=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_output_path = build_archive_output_path(region_key, valid_time)
    archive_output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(archive_output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    shutil.copy2(archive_output_path, output_path)

    if show:
        plt.show()
    else:
        plt.close(figure)

    plt.close(figure)
    gc.collect()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lon_1d, lat_1d, values, valid_time = load_radar_grid(args.radar_url)

    for region_key, region_config in REGION_CONFIGS.items():
        extent = region_config["extent"]
        lon_grid, lat_grid, reflectivity = subset_radar_grid(lon_1d, lat_1d, values, extent)
        plot_radar(
            counties=region_config["counties_gdf"],
            states=region_config["states_gdf"],
            region_key=region_key,
            subset_extent=extent,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            reflectivity=reflectivity,
            valid_time=valid_time,
            output_path=build_output_path(args.output, region_key),
            show=args.show,
        )

    prune_archived_pngs(ARCHIVE_ROOT, RETENTION_DAYS)


if __name__ == "__main__":
    main()
