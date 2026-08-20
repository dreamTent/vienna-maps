#!/usr/bin/env python3
"""
Download and stitch Vienna oblique 2023 images by name (e.g. RI3605741).

Looks up each name in the local tile-index JSON files, downloads JPEG tiles at the
chosen pyramid zoom level, and writes a reconstructed image (PNG by default).

Examples:
    python download_oblique_by_name.py RI3605741
    python download_oblique_by_name.py RI3605741 RI3605742 --zoom-level 3 --format jpg
    python download_oblique_by_name.py RI3605741 --cdn --output-dir ./out

Requirements:
    pip install requests pillow shapely pyproj
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image

import download_oblique_by_coordinates as dob

CDN_BASE_URL = "https://cdn.map.dreamtent.dev/data/oblique/vienna/2023"

# Full-res oblique frames exceed Pillow's default decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None


def collect_images_by_name(tiles_dir: Path) -> Dict[str, dict]:
    """Map image name → metadata from tile indexes (first occurrence wins)."""
    by_name: Dict[str, dict] = {}
    for tf in sorted(tiles_dir.glob("*.json")):
        for img in dob.iter_images_from_tile(tf):
            name = img["name"]
            if name not in by_name:
                by_name[name] = img
    return by_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download/stitch Vienna 2023 oblique images by name."
    )
    p.add_argument(
        "names",
        nargs="+",
        help="One or more image names (e.g. RI3605741)",
    )
    p.add_argument(
        "--tiles-dir",
        type=Path,
        default=dob.DEFAULT_TILES_DIR,
        help="Directory with 2023 tile index JSON files (default: sources/2023/tiles)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for stitched images. Default: download_by_name_YYYYMMDD_HHMMSS/",
    )
    p.add_argument(
        "--zoom-level",
        type=int,
        default=4,
        help="Tile pyramid index z0…zN (default: 4 = finest when tile-resolution is [16,8,4,2,1])",
    )
    p.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output format (default: png)",
    )
    p.add_argument(
        "--cdn",
        action="store_true",
        help="Fetch tiles from cdn.map.dreamtent.dev instead of wien.gv.at",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cdn:
        dob.REMOTE_BASE_URL = CDN_BASE_URL

    tiles_dir: Path = args.tiles_dir
    if not tiles_dir.is_dir():
        raise SystemExit(f"Tiles directory not found: {tiles_dir}")

    catalog = collect_images_by_name(tiles_dir)
    if not catalog:
        raise SystemExit(f"No images found under {tiles_dir}")

    missing = [n for n in args.names if n not in catalog]
    if missing:
        raise SystemExit(
            "Unknown image name(s): "
            + ", ".join(missing)
            + f"\n(searched {len(catalog)} images in {tiles_dir})"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir if args.output_dir is not None else Path(f"download_by_name_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    print(
        f"Stitching {len(args.names)} image(s) at z{args.zoom_level} "
        f"({args.format}) into {out_dir.resolve()} ..."
    )

    for idx, name in enumerate(args.names, start=1):
        img = catalog[name]
        tr = img["tile_resolution"]
        factor = dob.get_resolution_factor_for_zoom_level(tr, args.zoom_level)
        if factor is None:
            print(
                f"[{idx}/{len(args.names)}] {name}: skip — "
                f"zoom level {args.zoom_level} not in tile-resolution {tr}",
                file=sys.stderr,
            )
            continue

        out_file = out_dir / f"{name}_z{args.zoom_level}.{args.format}"
        print(
            f"[{idx}/{len(args.names)}] {name} "
            f"(zoom {args.zoom_level}, factor {factor}, {img['width']}x{img['height']})"
        )
        w, h = dob.reconstruct_image_at_zoom(
            session=session,
            image_name=name,
            width=img["width"],
            height=img["height"],
            zoom_level=args.zoom_level,
            resolution_factor=factor,
            output_path=out_file,
            output_format=args.format,
        )
        mb = out_file.stat().st_size / 1e6
        print(f"  -> {out_file.resolve()} ({w}x{h}, {mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
