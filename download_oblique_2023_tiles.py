#!/usr/bin/env python3
"""
Download Vienna oblique 2023 pyramid tiles without stitching (mirrors server paths).

Tiles are stored as:
  {output_dir}/{image_name}/{zoom_level}/{x}/{y}.jpg

By default, every pyramid level listed in each image's \"tile-resolution\" is downloaded
(typically indices 0-4 for [16, 8, 4, 2, 1]). Use --zoom-level to fetch a single level.

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

# Same remote root as download_oblique_by_coordinates.py
REMOTE_BASE_URL = (
    "https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/"
    "91bf9860-34d7-4f9a-b841-32be753b09e5"
)
DEFAULT_TILES_DIR = Path("sources/2023/tiles")
TILE_SIZE = 1024
DEFAULT_MIN_BYTES = 100
DEFAULT_WORKERS = 4


def iter_images_from_tile(tile_file: Path) -> Iterable[dict]:
    """Same row filter as download_oblique.iter_images_from_tile."""
    with tile_file.open("r", encoding="utf-8") as f:
        tile_data = json.load(f)

    rows = tile_data.get("images", [])
    if len(rows) < 2:
        return

    header = rows[0]
    col = {name: idx for idx, name in enumerate(header)}

    required = ["name", "width", "height", "tile-resolution", "groundCoordinates"]
    for req in required:
        if req not in col:
            raise ValueError(f"{tile_file}: missing required column '{req}'")

    for row in rows[1:]:
        gc = row[col["groundCoordinates"]]
        if not gc or len(gc) < 3:
            continue
        yield {
            "name": row[col["name"]],
            "width": int(row[col["width"]] or 14144),
            "height": int(row[col["height"]] or 10560),
            "tile_resolution": list(row[col["tile-resolution"]] or [16, 8, 4, 2, 1]),
        }


def tile_grid_shape(width: int, height: int, resolution_factor: int) -> Tuple[int, int]:
    """Match pyramid_canvas_layout / reconstruct_image_at_zoom in download_oblique_by_coordinates.py."""
    cols = math.ceil(width / (TILE_SIZE * resolution_factor))
    rows = math.ceil(height / (TILE_SIZE * resolution_factor))
    return cols, rows


def collect_unique_images(tiles_dir: Path) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    for tf in sorted(tiles_dir.glob("*.json")):
        for img in iter_images_from_tile(tf):
            name = img["name"]
            if name in seen:
                continue
            seen.add(name)
            out.append(img)
    return out


def local_tile_path(output_dir: Path, image_name: str, zoom: int, x: int, y: int) -> Path:
    return output_dir / image_name / str(zoom) / str(x) / f"{y}.jpg"


def remote_tile_url(image_name: str, zoom: int, x: int, y: int) -> str:
    return f"{REMOTE_BASE_URL}/{image_name}/{zoom}/{x}/{y}.jpg"


def should_skip(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size > min_bytes
    except OSError:
        return False


def download_one(
    url: str,
    dest: Path,
    min_bytes: int,
    timeout: float,
) -> str:
    """Returns 'skipped', 'ok', or 'failed: ...'."""
    if should_skip(dest, min_bytes):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        dest.write_bytes(r.content)
        if dest.stat().st_size <= min_bytes:
            dest.unlink(missing_ok=True)
            return "failed: response too small"
        return "ok"
    except Exception as exc:
        return f"failed: {exc}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 2023 oblique JPEG tiles (one or all pyramid levels), mirroring server paths."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("oblique_2023_tiles"),
        help="Root folder for mirrored tile tree (default: ./oblique_2023_tiles)",
    )
    p.add_argument(
        "--tiles-dir",
        type=Path,
        default=DEFAULT_TILES_DIR,
        help="Directory with 2023 tile index JSON files",
    )
    p.add_argument(
        "--zoom-level",
        type=int,
        default=None,
        metavar="Z",
        help="If omitted: download every pyramid level (0 .. len(tile-resolution)-1) per image. "
        "If set: only this index (0 = coarsest). Use -1 for finest only.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel downloads (default: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--min-bytes",
        type=int,
        default=DEFAULT_MIN_BYTES,
        help=f"Skip re-download if file exists and size exceeds this (default: {DEFAULT_MIN_BYTES})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per tile in seconds (default: 120)",
    )
    p.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Optional max number of images to process (0 = all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        sys.exit(1)
    if args.min_bytes < 0:
        print("--min-bytes must be non-negative", file=sys.stderr)
        sys.exit(1)

    tiles_dir: Path = args.tiles_dir
    if not tiles_dir.is_dir():
        print(f"Tiles directory not found: {tiles_dir}", file=sys.stderr)
        sys.exit(1)

    images = collect_unique_images(tiles_dir)
    if not images:
        print(f"No images found under {tiles_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit_images > 0:
        images = images[: args.limit_images]

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    total_ok = total_skip = total_fail = 0

    for idx, img in enumerate(images, start=1):
        tr: List[int] = img["tile_resolution"]
        if args.zoom_level is None:
            level_indices = list(range(len(tr)))
        elif args.zoom_level < 0:
            level_indices = [len(tr) - 1]
        else:
            if args.zoom_level >= len(tr):
                print(
                    f"[{idx}/{len(images)}] {img['name']}: skip — "
                    f"zoom level {args.zoom_level} not in tile-resolution {tr}"
                )
                continue
            level_indices = [args.zoom_level]

        w, h = img["width"], img["height"]
        img_ok = img_skip = img_fail = 0

        for z in level_indices:
            factor = int(tr[z])
            cols, rows = tile_grid_shape(w, h, factor)
            n_tiles = cols * rows

            print(
                f"[{idx}/{len(images)}] {img['name']} z{z} "
                f"({cols}x{rows} = {n_tiles} tiles, factor {factor})"
            )

            tasks: List[Tuple[str, Path]] = []
            for y in range(rows):
                for x in range(cols):
                    url = remote_tile_url(img["name"], z, x, y)
                    dest = local_tile_path(out_root, img["name"], z, x, y)
                    tasks.append((url, dest))

            z_ok = z_skip = z_fail = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(download_one, url, dest, args.min_bytes, args.timeout): (url, dest)
                    for url, dest in tasks
                }
                for fut in as_completed(futs):
                    res = fut.result()
                    if res == "ok":
                        z_ok += 1
                    elif res == "skipped":
                        z_skip += 1
                    else:
                        z_fail += 1
                        url, _ = futs[fut]
                        print(f"  {res} — {url}")

            img_ok += z_ok
            img_skip += z_skip
            img_fail += z_fail

            total_ok += z_ok
            total_skip += z_skip
            total_fail += z_fail
            print(
                f"  z{z}: {z_ok} downloaded, {z_skip} skipped, {z_fail} failed "
                f"| total so far: {total_ok} downloaded, {total_skip} skipped, {total_fail} failed"
            )

        print(
            f"  {img['name']} total: {img_ok} downloaded, {img_skip} skipped, {img_fail} failed "
            f"| total so far: {total_ok} downloaded, {total_skip} skipped, {total_fail} failed"
        )

    print(
        f"Finished. Total tiles: {total_ok} downloaded, {total_skip} skipped, {total_fail} failed. "
        f"Output: {out_root.resolve()}"
    )
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
