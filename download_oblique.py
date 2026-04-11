#!/usr/bin/env python3
"""
Download and reconstruct Vienna oblique 2023 images for a GPS point or a ground area.

The script:
1) Converts input WGS84 lat/lon to dataset MGI coordinates.
2) Scans local 2023 tile index JSON files and selects images by AOI footprint coverage and/or
   the same centre test as point mode (projection in frame), using `p-to-image`.
3) Downloads JPEG tiles at the chosen pyramid zoom level and reconstructs each image.
4) Optionally writes binary masks in masks/ (white = keep, black = exclude), composited
   RGB in masked_images/, and COLMAP sparse reconstruction text.

Requirements:
    pip install pyproj requests pillow shapely
    For --colmap: pip install numpy opencv-python-headless
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import requests
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

try:
    from shapely.geometry import Polygon as ShapelyPolygon
except ImportError as e:
    raise ImportError("This script requires shapely. Install with: pip install shapely") from e

try:
    import cv2
except ImportError as e:
    cv2 = None  # type: ignore

REMOTE_BASE_URL = "https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/91bf9860-34d7-4f9a-b841-32be753b09e5"
DEFAULT_IMAGE_JSON = Path("sources/2023/image.json")
DEFAULT_TILES_DIR = Path("sources/2023/tiles")
TILE_SIZE = 1024


@dataclass
class AOI:
    """Axis-aligned area of interest in MGI metres (E, N)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def polygon_xy(self) -> List[Tuple[float, float]]:
        return [
            (self.min_x, self.min_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
            (self.min_x, self.max_y),
        ]

    def shapely_polygon(self) -> ShapelyPolygon:
        return ShapelyPolygon(self.polygon_xy())

    def area_m2(self) -> float:
        return float(self.shapely_polygon().area)


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
    cy = sum(p[1] for p in ring)
    ordered = sorted(ring, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    area2 = 0.0
    for i in range(len(ordered)):
        j = (i + 1) % len(ordered)
        area2 += ordered[i][0] * ordered[j][1] - ordered[j][0] * ordered[i][1]
    if area2 < 0:
        ordered.reverse()

    return [(p[0], p[1]) for p in ordered]


def load_transformers(image_json_path: Path) -> Tuple[Transformer, Transformer]:
    """Load WGS84↔MGI transformers (always_xy: lon, lat)."""
    with image_json_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    crs_str = manifest.get("generalImageInfo", {}).get("crs")
    if not crs_str:
        raise ValueError("Missing CRS in image.json at generalImageInfo.crs")

    mgi_crs = CRS.from_string(crs_str)
    wgs84_crs = CRS.from_epsg(4326)
    wgs_to_mgi = Transformer.from_crs(wgs84_crs, mgi_crs, always_xy=True)
    mgi_to_wgs = Transformer.from_crs(mgi_crs, wgs84_crs, always_xy=True)
    return wgs_to_mgi, mgi_to_wgs


def aoi_center_lon_lat(
    aoi: AOI,
    aoi_type: str,
    lon: float | None,
    lat: float | None,
    mgi_to_wgs: Transformer,
) -> Tuple[float, float]:
    """WGS84 center of the AOI (square uses given lat/lon; rect uses MGI bbox centre)."""
    if aoi_type == "square":
        if lat is None or lon is None:
            raise ValueError("square AOI requires lat/lon")
        return lon, lat
    cx_m = (aoi.min_x + aoi.max_x) / 2.0
    cy_m = (aoi.min_y + aoi.max_y) / 2.0
    return mgi_to_wgs.transform(cx_m, cy_m)


def selected_origin_mgi(
    aoi: AOI | None,
    lon: float | None,
    lat: float | None,
    wgs_to_mgi: Transformer,
) -> Tuple[float, float, float]:
    """COLMAP world origin in MGI metres, shifted so selected location becomes (0,0,0)."""
    if aoi is not None:
        cx_m = (aoi.min_x + aoi.max_x) / 2.0
        cy_m = (aoi.min_y + aoi.max_y) / 2.0
        return float(cx_m), float(cy_m), 0.0
    if lon is None or lat is None:
        raise ValueError("Point mode requires --lon and --lat for COLMAP origin")
    x_m, y_m = wgs_to_mgi.transform(lon, lat)
    return float(x_m), float(y_m), 0.0


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


def project_mgi_xy_to_pixel(
    p_to_image: Any,
    x_mgi: float,
    y_mgi: float,
    ground_z: float,
) -> Tuple[float, float] | None:
    """MGI ground point (X, Y) at height ground_z → full-resolution image pixel."""
    if not p_to_image:
        return None
    P = p_to_image
    X, Y, Z = x_mgi, y_mgi, ground_z
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


def footprint_shapely(footprint: Sequence[Tuple[float, float]]) -> ShapelyPolygon:
    return ShapelyPolygon(footprint)


def coverage_percent_aoi(aoi: AOI, footprint_ring: Sequence[Tuple[float, float]]) -> float:
    """Percentage of AOI area covered by image ground footprint (intersection / AOI area)."""
    fp = footprint_shapely(footprint_ring)
    if fp.is_empty or not fp.is_valid:
        return 0.0
    aoi_poly = aoi.shapely_polygon()
    inter = aoi_poly.intersection(fp)
    if inter.is_empty:
        return 0.0
    a = float(aoi_poly.area)
    if a < 1e-9:
        return 0.0
    return 100.0 * float(inter.area) / a


def aoi_from_rect_corners(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    transformer: Transformer,
) -> AOI:
    """Axis-aligned rectangle in MGI from two WGS84 corners (any diagonal)."""
    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)
    return AOI(
        min_x=min(x1, x2),
        min_y=min(y1, y2),
        max_x=max(x1, x2),
        max_y=max(y1, y2),
    )


def aoi_from_center_square(
    center_lon: float,
    center_lat: float,
    half_side_m: float,
    transformer: Transformer,
) -> AOI:
    """
    Square aligned with MGI axes (cardinal E–W / N–S in projected metres).
    half_side_m is half the side length (user 'radius' = half side).
    """
    cx, cy = transformer.transform(center_lon, center_lat)
    return AOI(
        min_x=cx - half_side_m,
        min_y=cy - half_side_m,
        max_x=cx + half_side_m,
        max_y=cy + half_side_m,
    )


def buffered_aoi_polygon(aoi: AOI, buffer_m: float) -> ShapelyPolygon:
    return aoi.shapely_polygon().buffer(buffer_m, join_style=2)


def reconstruct_image_at_zoom(
    session: requests.Session,
    image_name: str,
    width: int,
    height: int,
    zoom_level: int,
    resolution_factor: int,
    output_path: Path,
    output_format: str,
) -> Tuple[int, int]:
    """
    Returns (target_w, target_h) of the saved image.
    """
    cols = math.ceil(width / (TILE_SIZE * resolution_factor))
    rows = math.ceil(height / (TILE_SIZE * resolution_factor))

    full_w = cols * TILE_SIZE
    full_h = rows * TILE_SIZE
    canvas = Image.new("RGB", (full_w, full_h))

    for y in range(rows):
        for x in range(cols):
            url = f"{REMOTE_BASE_URL}/{image_name}/{zoom_level}/{x}/{y}.jpg"
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            tile = Image.open(io_bytes(resp.content)).convert("RGB")

            paste_y = (rows - 1 - y) * TILE_SIZE
            paste_x = x * TILE_SIZE
            canvas.paste(tile, (paste_x, paste_y))

    target_w = math.ceil(width / resolution_factor)
    target_h = math.ceil(height / resolution_factor)
    top_pad = max(0, full_h - target_h)
    cropped = canvas.crop((0, top_pad, target_w, full_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        cropped.save(output_path, format="PNG")
    elif output_format == "jpg":
        cropped.save(output_path, format="JPEG", quality=100, subsampling=0, optimize=True)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return target_w, target_h


def full_pixel_to_output_xy(
    px: float,
    py: float,
    width: int,
    height: int,
    target_w: int,
    target_h: int,
) -> Tuple[float, float]:
    """Map full-resolution image pixel (origin top-left) to reconstructed output pixel coords."""
    ox = px * (target_w / max(width, 1))
    oy = py * (target_h / max(height, 1))
    return ox, oy


def build_aoi_mask_l(
    rgb: Image.Image,
    p_to_image: Any,
    ground_z: float,
    aoi_buffered: ShapelyPolygon,
    width: int,
    height: int,
) -> Image.Image:
    """
    L-mode mask: 255 = include (inside buffered AOI projection), 0 = exclude.
    On degenerate geometry or failed projection, returns full-white (keep entire image).
    """
    w, h = rgb.size
    ext = aoi_buffered.exterior
    coords = list(ext.coords)
    if len(coords) < 4:
        return Image.new("L", (w, h), 255)

    poly_pts: List[Tuple[float, float]] = []
    for x_mgi, y_mgi in coords[:-1]:
        pr = project_mgi_xy_to_pixel(p_to_image, float(x_mgi), float(y_mgi), ground_z)
        if pr is None:
            return Image.new("L", (w, h), 255)
        ox, oy = full_pixel_to_pixel(pr[0], pr[1], width, height, w, h)
        poly_pts.append((ox, oy))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(poly_pts, fill=255)
    return mask


def full_pixel_to_pixel(
    px: float,
    py: float,
    width: int,
    height: int,
    out_w: int,
    out_h: int,
) -> Tuple[float, float]:
    return full_pixel_to_output_xy(px, py, width, height, out_w, out_h)


def projection_matrix_scaled(
    p_to_image: Sequence[Sequence[float]],
    width: int,
    height: int,
    out_w: int,
    out_h: int,
) -> Any:
    """Scale image rows of P to match reconstructed output pixel dimensions."""
    import numpy as np

    sx = out_w / float(width)
    sy = out_h / float(height)
    P = np.array(p_to_image, dtype=np.float64).reshape(3, 4)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return S @ P


def _camera_center_from_projection_matrix(P: Any) -> Any:
    """
    Euclidean camera centre C in world coords (MGI metres): P @ [C,1]^T = 0.
    Right null vector of a 3×4 projection matrix (last row of Vh from SVD); avoids
    OpenCV's translation from decomposeProjectionMatrix, which is often wrong scale.
    """
    import numpy as np

    P = np.asarray(P, dtype=np.float64)
    _, _, vh = np.linalg.svd(P, full_matrices=True)
    X = vh[-1, :]
    if abs(X[3]) < 1e-30:
        raise ValueError("Degenerate projection: camera centre at infinity")
    return X[:3] / X[3]


def colmap_ground_transform_matrix() -> Any:
    """
    World transform for easier top-down inspection:
    - Input world:  X=east, Y=north, Z=up
    - COLMAP world: X=west, Y=up,    Z=north
    This keeps the ground plane flat at Y=0.
    """
    import numpy as np

    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def colmap_export_for_image(
    image_id: int,
    camera_id: int,
    image_name: str,
    out_w: int,
    out_h: int,
    p_to_image: Sequence[Sequence[float]],
    width: int,
    height: int,
    world_origin_xyz: Tuple[float, float, float],
) -> Tuple[str, str]:
    """Return (cameras_line, images_line) for COLMAP text format."""
    import numpy as np

    if cv2 is None:
        raise ImportError("COLMAP export requires opencv-python-headless: pip install opencv-python-headless")

    P = projection_matrix_scaled(p_to_image, width, height, out_w, out_h)
    P = np.asarray(P, dtype=np.float64)
    # K, R from decomposition; ignore OpenCV t (often wrong scale vs COLMAP).
    K, R, _, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    R = R[:, :3]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
        K[:, 2] *= -1.0

    C = _camera_center_from_projection_matrix(P)
    # Shift world origin so selected location becomes (0, 0, 0) in COLMAP world.
    C = C - np.asarray(world_origin_xyz, dtype=np.float64)
    # Rotate world so ground is flat for top-down viewing (Y-up frame).
    T = colmap_ground_transform_matrix()
    C = T @ C
    R = R @ T.T
    # COLMAP: world point X_w to camera X_c = R * X_w + t, and C = -R^T * t  →  t = -R * C
    t = -R @ C

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    r00, r01, r02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    r10, r11, r12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    r20, r21, r22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    qw = math.sqrt(max(0.0, 1.0 + r00 + r11 + r22)) / 2.0
    qx = (r21 - r12) / (4.0 * qw) if abs(qw) > 1e-12 else 0.0
    qy = (r02 - r20) / (4.0 * qw) if abs(qw) > 1e-12 else 0.0
    qz = (r10 - r01) / (4.0 * qw) if abs(qw) > 1e-12 else 0.0
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-12:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    cam_line = f"{camera_id} PINHOLE {out_w} {out_h} {fx} {fy} {cx} {cy}"
    img_line = f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}"
    return cam_line, img_line


def colmap_flat_grid_points3d_lines(
    aoi: AOI,
    world_origin_xyz: Tuple[float, float, float],
    target_points: int = 100000,
) -> List[str]:
    """
    Generate a flat Z=0 grid in AOI, approximately target_points samples.
    Coordinates are shifted into the COLMAP world frame (selected origin at 0,0,0).
    """
    target_points = max(1, int(target_points))
    dx = max(1e-9, aoi.max_x - aoi.min_x)
    dy = max(1e-9, aoi.max_y - aoi.min_y)
    ratio = dx / dy

    nx = max(1, int(math.ceil(math.sqrt(target_points * ratio))))
    ny = max(1, int(math.ceil(target_points / nx)))

    ox, oy, oz = world_origin_xyz
    import numpy as np

    T = colmap_ground_transform_matrix()
    lines: List[str] = []
    pid = 1

    # Uniform grid over AOI bounds; use sample positions (including edges).
    for iy in range(ny):
        y = aoi.min_y if ny == 1 else aoi.min_y + (dy * iy) / (ny - 1)
        for ix in range(nx):
            x = aoi.min_x if nx == 1 else aoi.min_x + (dx * ix) / (nx - 1)
            p = np.array([x - ox, y - oy, 0.0 - oz], dtype=np.float64)
            xw, yw, zw = (T @ p).tolist()
            # Empty track is valid in text format: just omit IMAGE_ID/POINT2D_IDX pairs.
            lines.append(f"{pid} {xw} {yw} {zw} 255 255 255 0.0")
            pid += 1
    return lines


def io_bytes(content: bytes):
    from io import BytesIO

    return BytesIO(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/reconstruct Vienna 2023 oblique images for a point or ground area."
    )
    parser.add_argument(
        "--aoi-type",
        choices=["point", "rect", "square"],
        default="point",
        help="point: single WGS84 coordinate (--lat/--lon). rect: bounding box from two corners. "
        "square: axis-aligned square in MGI (--lat/--lon center, --half-side-m).",
    )
    parser.add_argument("--lat", type=float, help="Latitude in WGS84 (center for point/square)")
    parser.add_argument("--lon", type=float, help="Longitude in WGS84 (center for point/square)")
    parser.add_argument(
        "--corner1-lon",
        type=float,
        help="First corner longitude (rect)",
    )
    parser.add_argument(
        "--corner1-lat",
        type=float,
        help="First corner latitude (rect)",
    )
    parser.add_argument(
        "--corner2-lon",
        type=float,
        help="Second corner longitude (rect)",
    )
    parser.add_argument(
        "--corner2-lat",
        type=float,
        help="Second corner latitude (rect)",
    )
    parser.add_argument(
        "--half-side-m",
        type=float,
        default=None,
        help="Half the square side length in metres (square AOI); equals 'radius' from center to each edge.",
    )
    parser.add_argument(
        "--min-coverage-percent",
        type=float,
        default=0.0,
        help="For rect/square: require at least this %% of AOI area under the ground footprint, "
        "unless the AOI centre matches point mode (projection in frame or centre in footprint). "
        "Use 0 for any footprint overlap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory; a subfolder named from download start time is created inside. "
        "If omitted, a folder download_YYYYMMDD_HHMMSS is created in the current directory.",
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
        help="Tile pyramid index z0…zN (URL .../<image>/<z>/<x>/<y>.jpg). Must exist in tile-resolution.",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Ground height (metres, MGI Z) for projection and AOI mask.",
    )
    parser.add_argument(
        "--footprint-only",
        action="store_true",
        help="Point mode: select only by ground footprint (ignore p-to-image). "
        "AOI rect/square already uses footprint intersection with AOI only.",
    )
    parser.add_argument(
        "--require-footprint",
        action="store_true",
        help="With projection filter, also require the point inside the ground footprint (point mode).",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Write masks/ (PNG, white=keep, black=exclude) and masked_images/ (RGB composited).",
    )
    parser.add_argument(
        "--mask-buffer-m",
        type=float,
        default=50.0,
        help="Ground buffer (metres) added to AOI for mask keep region.",
    )
    parser.add_argument(
        "--colmap",
        action="store_true",
        help="Write COLMAP sparse text (cameras.txt, images.txt, points3D.txt) under colmap/sparse/0/.",
    )
    parser.add_argument(
        "--colmap-grid-points",
        type=int,
        default=100000,
        help="Approximate number of flat AOI grid points to write into points3D.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.aoi_type == "point":
        if args.lat is None or args.lon is None:
            raise SystemExit("--aoi-type point requires --lat and --lon")
    elif args.aoi_type == "rect":
        if None in (
            args.corner1_lon,
            args.corner1_lat,
            args.corner2_lon,
            args.corner2_lat,
        ):
            raise SystemExit("--aoi-type rect requires --corner1-lon/lat and --corner2-lon/lat")
    else:
        if args.lat is None or args.lon is None or args.half_side_m is None:
            raise SystemExit("--aoi-type square requires --lat, --lon, and --half-side-m")

    started = datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    root = args.output_dir if args.output_dir is not None else Path.cwd()
    run_dir = root / f"download_{stamp}"
    images_dir = run_dir / "images"
    masked_images_dir = run_dir / "masked_images"
    masks_dir = run_dir / "masks"
    colmap_dir = run_dir / "colmap" / "sparse" / "0"

    wgs_to_mgi, mgi_to_wgs = load_transformers(args.image_json)

    aoi: AOI | None = None
    qx = qy = None
    if args.aoi_type == "rect":
        aoi = aoi_from_rect_corners(
            args.corner1_lon,
            args.corner1_lat,
            args.corner2_lon,
            args.corner2_lat,
            wgs_to_mgi,
        )
    elif args.aoi_type == "square":
        aoi = aoi_from_center_square(args.lon, args.lat, args.half_side_m, wgs_to_mgi)
    else:
        qx, qy = wgs_to_mgi.transform(args.lon, args.lat)

    if args.masks and aoi is None:
        raise SystemExit("--masks requires --aoi-type rect or square (a defined ground area).")
    colmap_origin_xyz = selected_origin_mgi(aoi, args.lon, args.lat, wgs_to_mgi)

    tile_files = sorted(args.tiles_dir.glob("*.json"))
    if not tile_files:
        raise FileNotFoundError(f"No tile index JSON files found in {args.tiles_dir}")

    mode = "footprint only" if args.footprint_only else "projection in frame"
    if aoi is not None:
        mode = (
            f"AOI: coverage ≥ {args.min_coverage_percent}% of AOI "
            f"OR same centre test as point mode (projection / footprint)"
        )
    print(
        f"Searching {len(tile_files)} index files — {mode} ..."
    )
    matches: List[dict] = []
    seen_names = set()

    for tf in tile_files:
        for img in iter_images_from_tile(tf):
            name = img["name"]
            if name in seen_names:
                continue

            if aoi is not None:
                cov = coverage_percent_aoi(aoi, img["footprint"])
                if args.min_coverage_percent <= 0:
                    pass_cov = cov > 1e-9
                else:
                    pass_cov = cov + 1e-9 >= args.min_coverage_percent

                lon_c, lat_c = aoi_center_lon_lat(
                    aoi, args.aoi_type, args.lon, args.lat, mgi_to_wgs
                )
                if args.footprint_only:
                    cx_m = (aoi.min_x + aoi.max_x) / 2.0
                    cy_m = (aoi.min_y + aoi.max_y) / 2.0
                    pass_center = point_in_polygon(cx_m, cy_m, img["footprint"])
                else:
                    proj = project_to_pixel(
                        img.get("p_to_image"),
                        lon_c,
                        lat_c,
                        args.ground_z,
                        wgs_to_mgi,
                    )
                    pass_center = proj is not None and is_in_image_frame(
                        proj[0], proj[1], img["width"], img["height"]
                    )

                # Coverage alone misses narrow E/W oblique footprints: the AOI centre can
                # project in-frame (same as point search) while footprint∩AOI area is < 1%.
                if not pass_cov and not pass_center:
                    continue
                ok = True
            else:
                if args.footprint_only:
                    ok = point_in_polygon(qx, qy, img["footprint"])
                else:
                    proj = project_to_pixel(
                        img.get("p_to_image"),
                        args.lon,
                        args.lat,
                        args.ground_z,
                        wgs_to_mgi,
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

    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
        masked_images_dir.mkdir(parents=True, exist_ok=True)
    if args.colmap:
        colmap_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_utc": started.isoformat(),
        "run_dir": str(run_dir.resolve()),
        "aoi_type": args.aoi_type,
        "zoom_level": args.zoom_level,
        "min_coverage_percent": args.min_coverage_percent,
        "masks": args.masks,
        "mask_buffer_m": args.mask_buffer_m if args.masks else None,
        "colmap": args.colmap,
        "image_count": len(matches),
    }
    if args.colmap:
        manifest["colmap_origin_mgi_xyz"] = {
            "x": colmap_origin_xyz[0],
            "y": colmap_origin_xyz[1],
            "z": colmap_origin_xyz[2],
        }
        manifest["colmap_world_axes"] = {
            "x": "west",
            "y": "up",
            "z": "north",
            "ground_plane": "y=0",
        }
        manifest["colmap_grid_points_target"] = int(args.colmap_grid_points)
    if aoi is not None:
        manifest["aoi_mgi"] = {
            "min_x": aoi.min_x,
            "min_y": aoi.min_y,
            "max_x": aoi.max_x,
            "max_y": aoi.max_y,
        }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Found {len(matches)} matching images. Downloading z{args.zoom_level} reconstructions into {run_dir} ...")

    camera_lines: List[str] = []
    image_lines: List[str] = []
    projection_records: List[dict[str, Any]] = []

    with requests.Session() as session:
        for idx, img in enumerate(matches, start=1):
            out_file = images_dir / f"{img['name']}_z{img['zoom_level']}.{args.format}"
            print(
                f"[{idx}/{len(matches)}] {img['name']} (zoom level {img['zoom_level']}, factor {img['z_factor']})"
            )
            try:
                target_w, target_h = reconstruct_image_at_zoom(
                    session=session,
                    image_name=img["name"],
                    width=img["width"],
                    height=img["height"],
                    zoom_level=img["zoom_level"],
                    resolution_factor=img["z_factor"],
                    output_path=out_file,
                    output_format=args.format,
                )
                img["target_w"] = target_w
                img["target_h"] = target_h
                projection_records.append(
                    {
                        "image_file": out_file.name,
                        "name": img["name"],
                        "width_full": img["width"],
                        "height_full": img["height"],
                        "width_output": target_w,
                        "height_output": target_h,
                        "zoom_level": img["zoom_level"],
                        "resolution_factor": img["z_factor"],
                        "p_to_image": img.get("p_to_image"),
                    }
                )

                if args.masks and aoi is not None and img.get("p_to_image"):
                    buf_poly = buffered_aoi_polygon(aoi, args.mask_buffer_m)
                    loaded = Image.open(out_file).convert("RGB")
                    mask_l = build_aoi_mask_l(
                        loaded,
                        img["p_to_image"],
                        args.ground_z,
                        buf_poly,
                        img["width"],
                        img["height"],
                    )
                    mask_png = masks_dir / f"{out_file.stem}.png"
                    mask_l.save(mask_png, format="PNG")
                    black = Image.new("RGB", loaded.size, (0, 0, 0))
                    masked = Image.composite(loaded, black, mask_l)
                    masked_out = masked_images_dir / out_file.name
                    if args.format == "jpg":
                        masked.save(
                            masked_out,
                            format="JPEG",
                            quality=100,
                            subsampling=0,
                            optimize=True,
                        )
                    else:
                        masked.save(masked_out, format="PNG")

                if args.colmap and img.get("p_to_image"):
                    cl, il = colmap_export_for_image(
                        idx,
                        idx,
                        out_file.name,
                        target_w,
                        target_h,
                        img["p_to_image"],
                        img["width"],
                        img["height"],
                        colmap_origin_xyz,
                    )
                    camera_lines.append(cl)
                    image_lines.append(il)
            except Exception as exc:
                print(f"  FAILED: {exc}")

    if args.colmap and camera_lines:
        ncam = len(camera_lines)
        nimg = len(image_lines)
        cam_body = "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                f"# Number of cameras: {ncam}",
            ]
            + camera_lines
        )
        (colmap_dir / "cameras.txt").write_text(cam_body + "\n", encoding="utf-8")
        img_parts: List[str] = [
            "# Image list with two lines of data per image:",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)",
            f"# Number of images: {nimg}, mean observations per image: 0",
        ]
        for il in image_lines:
            img_parts.append(il)
            img_parts.append("")
        (colmap_dir / "images.txt").write_text("\n".join(img_parts) + "\n", encoding="utf-8")
        if aoi is not None:
            pts_lines = colmap_flat_grid_points3d_lines(
                aoi=aoi,
                world_origin_xyz=colmap_origin_xyz,
                target_points=args.colmap_grid_points,
            )
            pts_body = "\n".join(
                [
                    "# 3D point list with one line of data per point:",
                    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
                    f"# Number of points: {len(pts_lines)}, mean track length: 0",
                ]
                + pts_lines
            )
            (colmap_dir / "points3D.txt").write_text(pts_body + "\n", encoding="utf-8")
        else:
            (colmap_dir / "points3D.txt").write_text(
                "# 3D point list (empty — no AOI selected, grid not generated)\n", encoding="utf-8"
            )

    if projection_records:
        (run_dir / "projection_matrices.json").write_text(
            json.dumps(projection_records, indent=2), encoding="utf-8"
        )

    print(f"Done. Output folder: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
