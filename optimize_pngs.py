from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_INPUT_CANDIDATES = (
    Path("/var/data/mrms_radar_archive"),
    Path("/var/data"),
    Path("."),
)
DEFAULT_MAX_COLORS = 192
DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}


@dataclass
class OptimizationResult:
    path: Path
    original_size: int
    optimized_size: int
    updated: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    default_paths = [str(path) for path in get_default_input_paths()]
    parser = argparse.ArgumentParser(
        description="Reduce PNG file sizes by optimizing and optionally palette-quantizing them."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=get_default_input_paths(),
        help=(
            "PNG files or directories to process. Defaults to "
            f"{', '.join(default_paths)}."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of each directory input.",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=DEFAULT_MAX_COLORS,
        help="Palette size for quantization. Use 0 to disable quantization. Default: 192.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report potential savings without modifying files.",
    )
    parser.add_argument(
        "--include-excluded-dirs",
        action="store_true",
        help="Also scan normally excluded folders such as .venv and __pycache__.",
    )
    return parser.parse_args()


def get_default_input_paths() -> list[Path]:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return [candidate]
    return [Path(".")]


def iter_png_files(paths: list[Path], recursive: bool, include_excluded_dirs: bool) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in paths:
        if input_path.is_file() and input_path.suffix.lower() == ".png":
            discovered.add(input_path)
            continue

        if not input_path.is_dir():
            continue

        if recursive:
            for root, dir_names, file_names in walk_png_directories(input_path, include_excluded_dirs=include_excluded_dirs):
                root_path = Path(root)
                for file_name in file_names:
                    if file_name.lower().endswith(".png"):
                        discovered.add(root_path / file_name)
        else:
            for png_path in input_path.glob("*.png"):
                if png_path.is_file():
                    discovered.add(png_path)

    return sorted(discovered)


def walk_png_directories(input_path: Path, include_excluded_dirs: bool):
    for current_path, dir_names, file_names in __import__("os").walk(input_path):
        if not include_excluded_dirs:
            dir_names[:] = [
                dir_name for dir_name in dir_names if dir_name not in DEFAULT_EXCLUDED_DIR_NAMES
            ]
        yield current_path, dir_names, file_names


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


def optimize_png(path: Path, max_colors: int, dry_run: bool) -> OptimizationResult:
    original_size = path.stat().st_size

    try:
        with Image.open(path) as image:
            optimized_bytes = build_optimized_png_bytes(image, max_colors=max_colors)
    except Exception as exc:
        return OptimizationResult(path=path, original_size=original_size, optimized_size=original_size, updated=False, error=str(exc))

    optimized_size = len(optimized_bytes)
    should_update = optimized_size < original_size

    if should_update and not dry_run:
        path.write_bytes(optimized_bytes)

    return OptimizationResult(
        path=path,
        original_size=original_size,
        optimized_size=optimized_size,
        updated=should_update and not dry_run,
    )


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def main() -> None:
    args = parse_args()
    if args.max_colors < 0 or args.max_colors > 256:
        raise SystemExit("--max-colors must be between 0 and 256")

    png_paths = iter_png_files(
        args.paths,
        recursive=not args.no_recursive,
        include_excluded_dirs=args.include_excluded_dirs,
    )
    if not png_paths:
        raise SystemExit("No PNG files found.")

    total_original_size = 0
    total_optimized_size = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for png_path in png_paths:
        result = optimize_png(png_path, max_colors=args.max_colors, dry_run=args.dry_run)
        total_original_size += result.original_size
        total_optimized_size += result.optimized_size

        if result.error:
            error_count += 1
            print(f"ERROR  {result.path}: {result.error}")
            continue

        if result.optimized_size < result.original_size:
            savings = result.original_size - result.optimized_size
            percent = (savings / result.original_size) * 100 if result.original_size else 0.0
            status = "WOULD UPDATE" if args.dry_run else "UPDATED"
            if result.updated:
                updated_count += 1
            else:
                skipped_count += 1
            print(
                f"{status:<12} {result.path} | {format_bytes(result.original_size)} -> "
                f"{format_bytes(result.optimized_size)} ({percent:.1f}% smaller)"
            )
        else:
            skipped_count += 1
            print(f"SKIPPED      {result.path} | no size improvement")

    total_savings = total_original_size - total_optimized_size
    total_percent = (total_savings / total_original_size) * 100 if total_original_size else 0.0

    print()
    print(f"Processed: {len(png_paths)} PNG file(s)")
    print(f"Changed:   {updated_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Errors:    {error_count}")
    print(
        f"Savings:   {format_bytes(total_original_size)} -> {format_bytes(total_optimized_size)} "
        f"({total_percent:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()
