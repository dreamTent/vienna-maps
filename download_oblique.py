#!/usr/bin/env python3
"""
Download and reconstruct Vienna oblique 2023 images that contain a GPS point.

The script:
1) Converts input WGS84 lat/lon to dataset MGI coordinates.
2) Scans local 2023 tile index JSON files and selects images where the point is
   visible in the image frame (same as the viewer's "in frame" / out-of-frame filter),
   using the camera `p-to-image` matrix. Optionally you can use ground footprint only.
3) Downloads all JPEG tiles for each matching image at the chosen zoom level.
4) Reconstructs the full image and saves it to the output folder.

Requirements:
    pip install pyproj requests pillow
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import requests
from PIL import Image
from pyproj import CRS, Transformer


REMOTE_BASE_URL = "https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/91bf9860-34d7-4f9a-b841-32be753b09e5"
DEFAULT_IMAGE_JSON = Path("sources/2023/image.json")
DEFAULT_TILES_DIR = Path("sources/2023/tiles")
TILE_SIZE = 1024


def point_in_polygon(px: float, py: float, ring: Sequence[Sequence[float]]) -> bool:
    """Odd-even point-in-polygon test."""
    inside = False
    n = len(ring)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        denom = yj - yi
        if abs(denom) >= 1e-15 and ((yi > py) != (yj > py)):
            x_int = xi + ((py - yi) * (xj - xi)) / denom
            if px < x_int:
                inside = not inside
        j = i
    return inside


def order_footprint_ccw(ring: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Order potentially unordered footprint points in CCW order."""
    if len(ring) < 3:
        return [(p[0], p[1]) for p in ring]

    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    ordered = sorted(ring, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    area2 = 0.0
    for i in range(len(ordered)):
        j = (i + 1) % len(ordered)
        area2 += ordered[i][0] * ordered[j][1] - ordered[j][0] * ordered[i][1]
    if area2 < 0:
        ordered.reverse()

    return [(p[0], p[1]) for p in ordered]


def load_crs_transformer(image_json_path: Path) -> Transformer:
    with image_json_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    crs_str = manifest.get("generalImageInfo", {}).get("crs")
    if not crs_str:
        raise ValueError("Missing CRS in image.json at generalImageInfo.crs")

    mgi_crs = CRS.from_string(crs_str)
    wgs84_crs = CRS.from_epsg(4326)
    return Transformer.from_crs(wgs84_crs, mgi_crs, always_xy=True)


def iter_images_from_tile(tile_file: Path) -> Iterable[dict]:
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

    i_pti = col.get("p-to-image")

    for row in rows[1:]:
        gc = row[col["groundCoordinates"]]
        if not gc or len(gc) < 3:
            continue
        footprint = order_footprint_ccw([(pt[0], pt[1]) for pt in gc])
        p_to_image = row[i_pti] if i_pti is not None else None
        yield {
            "name": row[col["name"]],
            "width": int(row[col["width"]] or 14144),
            "height": int(row[col["height"]] or 10560),
            "tile_resolution": list(row[col["tile-resolution"]] or [16, 8, 4, 2, 1]),
            "footprint": footprint,
            "p_to_image": p_to_image,
        }


def project_to_pixel(
    p_to_image: Any,
    lon: float,
    lat: float,
    ground_z: float,
    transformer: Transformer,
) -> Tuple[float, float] | None:
    """
    World (MGI, metres) → image pixel; same model as oblique-viewer.html projectToPixel.
    y=0 at top of image.
    """
    if not p_to_image:
        return None
    P = p_to_image
    X, Y = transformer.transform(lon, lat)
    Z = ground_z
    u = P[0][0] * X + P[0][1] * Y + P[0][2] * Z + P[0][3]
    v = P[1][0] * X + P[1][1] * Y + P[1][2] * Z + P[1][3]
    w = P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3]
    if abs(w) < 1e-10:
        return None
    return (u / w, v / w)


def is_in_image_frame(px: float, py: float, width: int, height: int) -> bool:
    """Match applySort() in oblique-viewer.html (inclusive bounds)."""
    return 0 <= px <= width and 0 <= py <= height


def get_resolution_factor_for_zoom_level(tile_resolution: Sequence[int], zoom_level: int) -> int | None:
    if zoom_level < 0 or zoom_level >= len(tile_resolution):
        return None
    return int(tile_resolution[zoom_level])


def reconstruct_image_at_zoom(
    session: requests.Session,
    image_name: str,
    width: int,
    height: int,
    zoom_level: int,
    resolution_factor: int,
    output_path: Path,
    output_format: str,
) -> None:
    cols = math.ceil(width / (TILE_SIZE * resolution_factor))
    rows = math.ceil(height / (TILE_SIZE * resolution_factor))

    # Keep native z-level pixels (no upscaling) to avoid interpolation blur.
    full_w = cols * TILE_SIZE
    full_h = rows * TILE_SIZE
    canvas = Image.new("RGB", (full_w, full_h))

    for y in range(rows):
        for x in range(cols):
            url = f"{REMOTE_BASE_URL}/{image_name}/{zoom_level}/{x}/{y}.jpg"
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            tile = Image.open(io_bytes(resp.content)).convert("RGB")

            # Dataset rows start at bottom; place y=0 at bottom of stitched canvas.
            paste_y = (rows - 1 - y) * TILE_SIZE
            paste_x = x * TILE_SIZE
            canvas.paste(tile, (paste_x, paste_y))

    # Convert source-space image size to native z-level output pixels.
    target_w = math.ceil(width / resolution_factor)
    target_h = math.ceil(height / resolution_factor)
    top_pad = max(0, full_h - target_h)
    cropped = canvas.crop((0, top_pad, target_w, full_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        cropped.save(output_path, format="PNG")
    elif output_format == "jpg":
        # High-quality JPEG if explicitly requested.
        cropped.save(output_path, format="JPEG", quality=100, subsampling=0, optimize=True)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def io_bytes(content: bytes):
    from io import BytesIO

    return BytesIO(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/reconstruct all Vienna 2023 oblique images containing a GPS coordinate."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude in WGS84")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in WGS84")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads_2023_z4"),
        help="Directory where reconstructed images are saved",
    )
    parser.add_argument(
        "--image-json",
        type=Path,
        default=DEFAULT_IMAGE_JSON,
        help="Path to sources/2023/image.json",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=DEFAULT_TILES_DIR,
        help="Path to sources/2023/tiles directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images to download (0 = no limit)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output format. PNG is default to avoid extra compression loss.",
    )
    parser.add_argument(
        "--zoom-level",
        type=int,
        default=4,
        help="Tile zoom level used in URL path (.../<image>/<zoom>/<x>/<y>.jpg).",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Ground height (metres, MGI Z) for p-to-image projection — same as viewer height field.",
    )
    parser.add_argument(
        "--footprint-only",
        action="store_true",
        help="Select only by ground footprint polygon (old behaviour). Ignores camera projection.",
    )
    parser.add_argument(
        "--require-footprint",
        action="store_true",
        help="With default projection filter, also require the point inside the ground footprint.",
    )
    args = parser.parse_args()

    transformer = load_crs_transformer(args.image_json)
    qx, qy = transformer.transform(args.lon, args.lat)

    tile_files = sorted(args.tiles_dir.glob("*.json"))
    if not tile_files:
        raise FileNotFoundError(f"No tile index JSON files found in {args.tiles_dir}")

    mode = "footprint only" if args.footprint_only else "projection in frame"
    print(
        f"Searching {len(tile_files)} index files for point ({args.lat}, {args.lon}), "
        f"z={args.ground_z} m — {mode} ..."
    )
    matches: List[dict] = []
    seen_names = set()

    for tf in tile_files:
        for img in iter_images_from_tile(tf):
            name = img["name"]
            if name in seen_names:
                continue

            if args.footprint_only:
                ok = point_in_polygon(qx, qy, img["footprint"])
            else:
                proj = project_to_pixel(
                    img.get("p_to_image"),
                    args.lon,
                    args.lat,
                    args.ground_z,
                    transformer,
                )
                if proj is None:
                    continue
                px, py = proj
                ok = is_in_image_frame(px, py, img["width"], img["height"])
                if args.require_footprint:
                    ok = ok and point_in_polygon(qx, qy, img["footprint"])

            if not ok:
                continue

            z_factor = get_resolution_factor_for_zoom_level(img["tile_resolution"], args.zoom_level)
            if z_factor is None:
                continue
            img["zoom_level"] = args.zoom_level
            img["z_factor"] = z_factor
            matches.append(img)
            seen_names.add(name)

    if not matches:
        print("No matching images found.")
        return

    if args.limit > 0:
        matches = matches[: args.limit]

    print(f"Found {len(matches)} matching images. Downloading z4 reconstructions ...")
    with requests.Session() as session:
        for idx, img in enumerate(matches, start=1):
            out_file = args.output_dir / f"{img['name']}_z{img['zoom_level']}.{args.format}"
            print(
                f"[{idx}/{len(matches)}] {img['name']} (zoom level {img['zoom_level']}, factor {img['z_factor']})"
            )
            try:
                reconstruct_image_at_zoom(
                    session=session,
                    image_name=img["name"],
                    width=img["width"],
                    height=img["height"],
                    zoom_level=img["zoom_level"],
                    resolution_factor=img["z_factor"],
                    output_path=out_file,
                    output_format=args.format,
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")

    print(f"Done. Images saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
