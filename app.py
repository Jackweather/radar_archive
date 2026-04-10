from __future__ import annotations

import concurrent.futures
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
LOG_ROOT = BASE_DIR / "logs"
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{4})$")


app = Flask(__name__)


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
    labels = {
        "conus": "CONUS",
        "northeast": "Northeast / Mid-Atlantic US",
        "southeast": "Southeast US",
        "south_central": "South Central US",
        "north_central": "North Central US",
        "western": "Western US",
    }
    return labels.get(region_key, region_key.replace("_", " ").title())


def parse_frame_time(png_path: Path) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(png_path.stem)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M").replace(tzinfo=ZoneInfo("UTC"))


def list_region_frames(region_key: str) -> list[dict[str, str]]:
    region_dir = ARCHIVE_ROOT / region_key
    if not region_dir.exists():
        return []

    frames: list[dict[str, str]] = []
    for png_path in sorted(region_dir.glob("*.png")):
        frame_time = parse_frame_time(png_path)
        if frame_time is None:
            continue
        eastern_time = frame_time.astimezone(EASTERN_TIMEZONE)
        frames.append(
            {
                "filename": png_path.name,
                "timestamp": frame_time.isoformat(),
                "displayTime": eastern_time.strftime("%Y-%m-%d %I:%M %p %Z"),
                "url": f"/images/{region_key}/{png_path.name}",
            }
        )
    return frames


def list_regions() -> list[dict[str, str | int]]:
    if not ARCHIVE_ROOT.exists():
        return []

    regions: list[dict[str, str | int]] = []
    for region_dir in sorted(path for path in ARCHIVE_ROOT.iterdir() if path.is_dir()):
        frames = list_region_frames(region_dir.name)
        regions.append(
            {
                "key": region_dir.name,
                "label": region_label(region_dir.name),
                "frameCount": len(frames),
            }
        )
    return regions


@app.route("/")
def index() -> str:
    return render_template("index.html")


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
