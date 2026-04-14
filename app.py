from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort, jsonify, render_template, send_from_directory


BASE_DIR = Path("/var/data")
ARCHIVE_ROOT = BASE_DIR / "mrms_radar_archive"
GRIB_ARCHIVE_ROOT = BASE_DIR / "mrms_grib_archive"
LOG_ROOT = BASE_DIR / "logs"
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{4})$")
WARNING_REGION_PREFIX = "warning_region_"
WARNING_METADATA_FILENAME = "metadata.json"
LEGACY_WARNING_REGION_PREFIXES = ("tornado_warnings", "severe_thunderstorm_warnings")


app = Flask(__name__)

REGION_ORDER = {
    "conus": 0,
    "northeast": 1,
    "southeast": 2,
    "south_central": 3,
    "north_central": 4,
    "western": 5,
}

WARNING_VIEW_DEFINITIONS = {
    "tornado": {
        "label": "Tornado Warnings",
        "event": "Tornado Warning",
    },
    "severe": {
        "label": "Severe Thunderstorm Warnings",
        "event": "Severe Thunderstorm Warning",
    },
}


def split_region_key(region_key: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_(\d+)$", region_key)
    if not match:
        return region_key, None
    return match.group(1), int(match.group(2))


def run_script(script_path: str, working_directory: str, retries: int) -> None:
    script_name = Path(script_path).stem
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        timestamp = datetime.now(EASTERN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        log_path = LOG_ROOT / f"{script_name}_{timestamp}_attempt{attempt}.log"

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Running {script_path}\n")
            log_file.write(f"Working directory: {working_directory}\n")
            log_file.write(f"Attempt: {attempt}/{retries}\n\n")
            log_file.flush()

            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=working_directory,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if completed.returncode == 0:
            return

    raise RuntimeError(f"Failed to run {script_path} after {retries} attempt(s).")


def run_scripts(
    scripts: list[tuple[str, str]],
    retries: int,
    parallel: bool = False,
    max_parallel: int = 3,
) -> None:
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [
                executor.submit(run_script, script_path, working_directory, retries)
                for script_path, working_directory in scripts
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return

    for script_path, working_directory in scripts:
        run_script(script_path, working_directory, retries)


def region_label(region_key: str) -> str:
    metadata = load_warning_region_metadata(region_key)
    if metadata.get("label"):
        return str(metadata["label"])

    base_region_key, region_index = split_region_key(region_key)
    labels = {
        "tornado_warnings": "Tornado Warnings",
        "severe_thunderstorm_warnings": "Severe Thunderstorm Warnings",
        "conus": "CONUS",
        "northeast": "Northeast / Mid-Atlantic US",
        "southeast": "Southeast US",
        "south_central": "South Central US",
        "north_central": "North Central US",
        "western": "Western US",
    }
    label = labels.get(base_region_key, base_region_key.replace("_", " ").title())
    if region_index is None:
        return label
    return f"{label} {region_index}"


def parse_frame_time(png_path: Path) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(png_path.stem)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M").replace(tzinfo=ZoneInfo("UTC"))


def is_warning_region_key(region_key: str) -> bool:
    if region_key.startswith(WARNING_REGION_PREFIX):
        return True
    return any(
        region_key == prefix or region_key.startswith(f"{prefix}_")
        for prefix in LEGACY_WARNING_REGION_PREFIXES
    )


def warning_metadata_path(region_key: str) -> Path:
    return ARCHIVE_ROOT / region_key / WARNING_METADATA_FILENAME


def load_warning_region_metadata(region_key: str) -> dict[str, object]:
    if not region_key.startswith(WARNING_REGION_PREFIX):
        return {}

    metadata_path = warning_metadata_path(region_key)
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def list_region_frames(region_key: str) -> list[dict[str, str]]:
    region_dir = ARCHIVE_ROOT / region_key
    if not region_dir.exists():
        return []

    frames: list[dict[str, str]] = []
    for png_path in sorted(region_dir.rglob("*.png")):
        frame_time = parse_frame_time(png_path)
        if frame_time is None:
            continue
        eastern_time = frame_time.astimezone(EASTERN_TIMEZONE)
        relative_path = png_path.relative_to(region_dir).as_posix()
        frames.append(
            {
                "filename": relative_path,
                "timestamp": frame_time.isoformat(),
                "displayTime": eastern_time.strftime("%Y-%m-%d %I:%M %p %Z"),
                "url": f"/images/{region_key}/{relative_path}",
            }
        )
    frames.sort(key=lambda frame: frame["timestamp"])
    return frames


def list_regions() -> list[dict[str, str | int]]:
    if not ARCHIVE_ROOT.exists():
        return []

    regions: list[dict[str, str | int]] = []

    def region_sort_key(path: Path) -> tuple[int, int, str]:
        base_region_key, region_index = split_region_key(path.name)
        return (
            REGION_ORDER.get(base_region_key, 99),
            region_index or 0,
            region_label(path.name),
        )

    region_dirs = sorted(
        (path for path in ARCHIVE_ROOT.iterdir() if path.is_dir() and not is_warning_region_key(path.name)),
        key=region_sort_key,
    )
    for region_dir in region_dirs:
        frames = list_region_frames(region_dir.name)
        regions.append(
            {
                "key": region_dir.name,
                "label": region_label(region_dir.name),
                "frameCount": len(frames),
            }
        )
    return regions


def list_warning_regions(event_name: str) -> list[dict[str, str | int]]:
    if not ARCHIVE_ROOT.exists():
        return []

    warning_regions: list[dict[str, str | int]] = []

    for region_dir in (path for path in ARCHIVE_ROOT.iterdir() if path.is_dir() and is_warning_region_key(path.name)):
        metadata = load_warning_region_metadata(region_dir.name)
        events = {str(name) for name in metadata.get("events", []) if name}

        if region_dir.name.startswith(WARNING_REGION_PREFIX):
            if event_name not in events:
                continue
        elif event_name == WARNING_VIEW_DEFINITIONS["tornado"]["event"]:
            if not (region_dir.name == "tornado_warnings" or region_dir.name.startswith("tornado_warnings_")):
                continue
        elif event_name == WARNING_VIEW_DEFINITIONS["severe"]["event"]:
            if not (
                region_dir.name == "severe_thunderstorm_warnings"
                or region_dir.name.startswith("severe_thunderstorm_warnings_")
            ):
                continue

        frames = list_region_frames(region_dir.name)
        if not frames:
            continue

        latest_timestamp = frames[-1]["timestamp"]
        warning_regions.append(
            {
                "key": region_dir.name,
                "label": region_label(region_dir.name),
                "frameCount": len(frames),
                "latestTimestamp": latest_timestamp,
            }
        )

    warning_regions.sort(
        key=lambda region: (
            str(region["latestTimestamp"]),
            str(region["label"]),
        ),
        reverse=True,
    )

    for region in warning_regions:
        region.pop("latestTimestamp", None)

    return warning_regions


def list_viewer_options() -> dict[str, object]:
    return {
        "regions": list_regions(),
        "warnings": {
            warning_key: {
                "label": str(definition["label"]),
                "regions": list_warning_regions(str(definition["event"])),
            }
            for warning_key, definition in WARNING_VIEW_DEFINITIONS.items()
        },
    }


def list_grib_inventory() -> list[dict[str, object]]:
    if not GRIB_ARCHIVE_ROOT.exists():
        return []

    inventory: list[dict[str, object]] = []
    date_directories = sorted((path for path in GRIB_ARCHIVE_ROOT.iterdir() if path.is_dir()), reverse=True)
    for date_dir in date_directories:
        files: list[dict[str, str]] = []
        for grib_path in sorted(date_dir.glob("*.grib2"), reverse=True):
            files.append(
                {
                    "filename": grib_path.name,
                    "displayTime": date_dir.name,
                    "url": f"/grib-files/{date_dir.name}/{grib_path.name}",
                }
            )

        inventory.append(
            {
                "date": date_dir.name,
                "count": len(files),
                "files": files,
            }
        )

    return inventory


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/vault-archive")
def vault_archive() -> str:
    return render_template("archive_vault.html", inventory=list_grib_inventory())


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/radar.py", "/opt/render/project/src"),
        
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 1, parallel=True, max_parallel=1),
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/api/regions")
def api_regions():
    return jsonify({"regions": list_regions()})


@app.route("/api/viewer-options")
def api_viewer_options():
    return jsonify(list_viewer_options())


@app.route("/api/frames/<region_key>")
def api_frames(region_key: str):
    region_dir = ARCHIVE_ROOT / region_key
    if not region_dir.exists():
        abort(404, description="Unknown region")
    return jsonify({"region": region_key, "label": region_label(region_key), "frames": list_region_frames(region_key)})


@app.route("/images/<region_key>/<path:filename>")
def serve_image(region_key: str, filename: str):
    region_dir = ARCHIVE_ROOT / region_key
    if not region_dir.exists():
        abort(404, description="Unknown region")
    return send_from_directory(region_dir, filename)


@app.route("/grib-files/<date_key>/<path:filename>")
def serve_grib_file(date_key: str, filename: str):
    date_dir = GRIB_ARCHIVE_ROOT / date_key
    if not date_dir.exists():
        abort(404, description="Unknown GRIB date")
    return send_from_directory(date_dir, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
