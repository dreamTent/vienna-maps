#!/usr/bin/env python3
"""
Download Vienna oblique 2023 pyramid tiles without stitching (mirrors server paths).

Tiles are stored as:
  {output_dir}/{image_name}/{zoom_level}/{x}/{y}.jpg

By default, every pyramid level in each image's \"tile-resolution\" is downloaded
(typically 0-4 for [16, 8, 4, 2, 1]). Progress is a single global tile counter (done/total).
Use --zoom-level to limit to one pyramid level.

Throughput: line rate needs many concurrent downloads (--workers) and connection reuse (built-in).
The remote server may also cap per-IP bandwidth.

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
from requests.adapters import HTTPAdapter

# Same remote root as download_oblique.py
REMOTE_BASE_URL = (
    "https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/"
    "91bf9860-34d7-4f9a-b841-32be753b09e5"
)
DEFAULT_TILES_DIR = Path("sources/2023/tiles")
TILE_SIZE = 1024
DEFAULT_MIN_BYTES = 100
# High link speeds need many concurrent requests; also see thread-local Session below.
DEFAULT_WORKERS = 32
DEFAULT_CONNECT_TIMEOUT = 40.0
DEFAULT_CONNECT_RETRIES = 4

_thread_local = threading.local()

# Transient errors: many parallel connections can trigger short-lived connect timeouts server-side.
_RETRYABLE_HTTP = (
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    requests.exceptions.ChunkedEncodingError,
)


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
    """Match pyramid_canvas_layout / reconstruct_image_at_zoom in download_oblique.py."""
    cols = math.ceil(width / (TILE_SIZE * resolution_factor))
    rows = math.ceil(height / (TILE_SIZE * resolution_factor))
    return cols, rows


def resolve_level_indices(tr: List[int], zoom_level: int | None) -> List[int] | None:
    """Indices to download; None if --zoom-level is invalid for this image."""
    if zoom_level is None:
        return list(range(len(tr)))
    if zoom_level < 0:
        return [len(tr) - 1]
    if zoom_level >= len(tr):
        return None
    return [zoom_level]


def tile_count_for_image(img: dict, level_indices: List[int]) -> int:
    w, h = img["width"], img["height"]
    tr = img["tile_resolution"]
    n = 0
    for z in level_indices:
        cols, rows = tile_grid_shape(w, h, int(tr[z]))
        n += cols * rows
    return n


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


def _thread_http_session(pool_maxsize: int) -> requests.Session:
    """
    One Session per worker thread so connections to the same host are reused
    (avoids a new TCP+TLS handshake on every tile — that often caps throughput well below line rate).
    """
    s = getattr(_thread_local, "session", None)
    if s is not None and getattr(_thread_local, "pool_maxsize", None) == pool_maxsize:
        return s
    s = requests.Session()
    # pool_maxsize: concurrent connections to the same host in this pool (one thread uses one at a time).
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=pool_maxsize, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    _thread_local.session = s
    _thread_local.pool_maxsize = pool_maxsize
    return s


def should_skip(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size > min_bytes
    except OSError:
        return False


def download_one(
    url: str,
    dest: Path,
    min_bytes: int,
    connect_timeout: float,
    read_timeout: float,
    pool_maxsize: int,
    connect_retries: int,
) -> str:
    """Returns 'skipped', 'ok', or 'failed: ...'."""
    if should_skip(dest, min_bytes):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)

    attempts = 1 + max(0, connect_retries)
    for attempt in range(attempts):
        if attempt > 0:
            time.sleep(min(0.5 * (2 ** (attempt - 1)), 15.0))
        r = None
        try:
            s = _thread_http_session(pool_maxsize)
            r = s.get(url, timeout=(connect_timeout, read_timeout))
            r.raise_for_status()
            dest.write_bytes(r.content)
            if dest.stat().st_size <= min_bytes:
                dest.unlink(missing_ok=True)
                return "failed: response too small"
            return "ok"
        except requests.exceptions.HTTPError as exc:
            code = getattr(exc.response, "status_code", 0) or 0
            if code in (429, 502, 503, 504) and attempt < attempts - 1:
                continue
            return f"failed: {exc}"
        except _RETRYABLE_HTTP as exc:
            if attempt < attempts - 1:
                continue
            return f"failed: {exc}"
        except Exception as exc:
            return f"failed: {exc}"
        finally:
            if r is not None:
                with suppress(Exception):
                    r.close()

    return "failed: internal error"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 2023 oblique JPEG tiles; progress is a global tile counter."
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
        help="Concurrent tile downloads (higher uses more bandwidth; try 64–256 on fast links). "
        f"Default: {DEFAULT_WORKERS}",
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
        help="Read timeout per tile in seconds (after connection is established). Default: 120",
    )
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help=f"TCP/TLS connect timeout in seconds (default: {DEFAULT_CONNECT_TIMEOUT})",
    )
    p.add_argument(
        "--connect-retries",
        type=int,
        default=DEFAULT_CONNECT_RETRIES,
        help="Extra download attempts after connect/read/5xx failures (exponential backoff). "
        f"Default: {DEFAULT_CONNECT_RETRIES}",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="Max tile downloads queued at once per image (0 = auto from --workers). "
        "Raises this if the process feels stuck or uses too much RAM.",
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
    if args.connect_timeout <= 0 or args.timeout <= 0:
        print("--connect-timeout and --timeout must be positive", file=sys.stderr)
        sys.exit(1)
    if args.connect_retries < 0:
        print("--connect-retries must be non-negative", file=sys.stderr)
        sys.exit(1)
    batch_size = args.batch_size
    if batch_size < 0:
        print("--batch-size must be non-negative", file=sys.stderr)
        sys.exit(1)
    if batch_size == 0:
        batch_size = max(512, args.workers * 32)

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

    planned: List[Tuple[dict, List[int]]] = []
    for img in images:
        tr: List[int] = img["tile_resolution"]
        level_indices = resolve_level_indices(tr, args.zoom_level)
        if level_indices is None:
            continue
        planned.append((img, level_indices))

    if not planned:
        print("No tiles to download (empty index or --zoom-level invalid for every image).", file=sys.stderr)
        sys.exit(1)

    total_tiles = sum(tile_count_for_image(im, li) for im, li in planned)
    print(f"Tiles to process: {total_tiles}", flush=True)

    pool_maxsize = max(32, args.workers)

    total_ok = total_skip = total_fail = 0
    done = 0

    def print_progress() -> None:
        print(
            f"\r{done}/{total_tiles}  ok={total_ok}  skip={total_skip}  fail={total_fail}",
            end="",
            flush=True,
        )

    for img, level_indices in planned:
        tr = img["tile_resolution"]
        w, h = img["width"], img["height"]
        all_tasks: List[Tuple[str, Path, int]] = []

        for z in level_indices:
            factor = int(tr[z])
            cols, rows = tile_grid_shape(w, h, factor)
            for y in range(rows):
                for x in range(cols):
                    url = remote_tile_url(img["name"], z, x, y)
                    dest = local_tile_path(out_root, img["name"], z, x, y)
                    all_tasks.append((url, dest, z))

        if not all_tasks:
            continue

        # Submit in chunks so we never queue millions of futures at once (memory / scheduler stalls).
        for chunk_start in range(0, len(all_tasks), batch_size):
            chunk = all_tasks[chunk_start : chunk_start + batch_size]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(
                        download_one,
                        url,
                        dest,
                        args.min_bytes,
                        args.connect_timeout,
                        args.timeout,
                        pool_maxsize,
                        args.connect_retries,
                    ): (url, dest, z)
                    for url, dest, z in chunk
                }
                for fut in as_completed(futs):
                    res = fut.result()
                    url, _dest, _z = futs[fut]
                    if res == "ok":
                        total_ok += 1
                    elif res == "skipped":
                        total_skip += 1
                    else:
                        total_fail += 1
                        print(file=sys.stderr)
                        print(f"{res} — {url}", file=sys.stderr)

                    done += 1
                    print_progress()

    print()
    print(
        f"Finished. {total_ok} downloaded, {total_skip} skipped, {total_fail} failed. "
        f"Output: {out_root.resolve()}"
    )
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
