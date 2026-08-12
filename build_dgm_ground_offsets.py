#!/usr/bin/env python3
"""
Build ground-height assets from sources/dgm/*.tif (+ .tfw) for oblique stitching.

Outputs (committed, small):
  sources/ground/heightmap.json   — mosaic metadata
  sources/ground/heightmap.bin    — Int16 LE elevations in decimetres (z_m = v/10)
  sources/ground/image-offsets.json — per-image {z, cz[4], plane{a,b,c}}

Decision for stitching (see infinite-scroll.html):
  Prefer a *shared local tilted plane* fitted from 4 DGM samples around the marker
  (not a single whole-image Z). A CSS/homography warp can only represent one plane;
  four local corners give slope without the pops of per-image average heights.
  Cameras are calibrated with ground ≈ model Z=0, so the viewer applies DGM
  *relative to the marker* (absolute DGM is shown in the UI for reference only).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DGM_DIR = ROOT / "sources" / "dgm"
OUT_DIR = ROOT / "sources" / "ground"
NODATA_SRC = -9999.0
NODATA_OUT = -32768  # Int16 sentinel (decimetres)
RES_M = 20.0  # mosaic cell size
ORIGIN_SNAP = 20.0


def read_tfw(path: Path):
    a, b, c, d, e, f = [float(x) for x in path.read_text().splitlines()[:6]]
    return a, b, c, d, e, f


def parse_ifd(f, endian: str):
    """Read TIFF IFD at current file; return dict of tag -> value(s)."""
    n = struct.unpack(endian + "H", f.read(2))[0]
    type_size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
    entries = []
    for _ in range(n):
        tag, typ, count, val = struct.unpack(endian + "HHII", f.read(12))
        entries.append((tag, typ, count, val))
    tags = {}
    for tag, typ, count, val in entries:
        size = type_size.get(typ, 1) * count
        if size <= 4:
            buf = struct.pack(endian + "I", val)[:size]
        else:
            pos = f.tell()
            f.seek(val)
            buf = f.read(size)
            f.seek(pos)
        if typ == 3:
            vals = struct.unpack(endian + f"{count}H", buf)
        elif typ == 4:
            vals = struct.unpack(endian + f"{count}I", buf)
        elif typ == 12:
            vals = struct.unpack(endian + f"{count}d", buf)
        elif typ == 11:
            vals = struct.unpack(endian + f"{count}f", buf)
        else:
            vals = (val,)
        tags[tag] = vals[0] if count == 1 else vals
    return tags


class DgmTile:
    __slots__ = (
        "path", "endian", "w", "h", "a", "d", "e", "f",
        "rows_per_strip", "strip_offsets", "bps", "sample_format",
        "_fh", "_strip_cache",
    )

    def __init__(self, tif: Path, tfw: Path):
        a, _b, _c, d, e, f = read_tfw(tfw)
        self.path = tif
        self.a, self.d, self.e, self.f = a, d, e, f
        with open(tif, "rb") as fh:
            hdr = fh.read(8)
            self.endian = "<" if hdr[:2] == b"II" else ">"
            off = struct.unpack_from(self.endian + "I", hdr, 4)[0]
            fh.seek(off)
            tags = parse_ifd(fh, self.endian)
        self.w = int(tags[256])
        self.h = int(tags[257])
        self.bps = int(tags[258])
        self.sample_format = int(tags.get(339, 1))
        self.rows_per_strip = int(tags.get(278, self.h))
        so = tags[273]
        self.strip_offsets = so if isinstance(so, tuple) else (so,)
        self._fh = None
        self._strip_cache = {}

    @property
    def xmin(self):
        return self.e - self.a * 0.5

    @property
    def xmax(self):
        return self.e + (self.w - 0.5) * self.a

    @property
    def ymax(self):
        # d < 0 (north-up): f is UL centre → max Y
        return self.f - self.d * 0.5

    @property
    def ymin(self):
        return self.f + (self.h - 0.5) * self.d

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def open(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._strip_cache.clear()

    def _read_strip(self, sidx: int) -> bytes:
        if sidx in self._strip_cache:
            return self._strip_cache[sidx]
        self.open()
        off = self.strip_offsets[sidx]
        rows = min(self.rows_per_strip, self.h - sidx * self.rows_per_strip)
        nbytes = rows * self.w * (self.bps // 8)
        self._fh.seek(off)
        buf = self._fh.read(nbytes)
        # Keep only a few strips — mosaic build streams one tile at a time.
        if len(self._strip_cache) > 4:
            self._strip_cache.clear()
        self._strip_cache[sidx] = buf
        return buf

    def sample(self, x: float, y: float) -> float | None:
        """Nearest-neighbour sample in metres, or None if nodata / OOB."""
        col = int(round((x - self.e) / self.a))
        row = int(round((y - self.f) / self.d))
        if col < 0 or col >= self.w or row < 0 or row >= self.h:
            return None
        sidx = row // self.rows_per_strip
        row_in = row % self.rows_per_strip
        buf = self._read_strip(sidx)
        bps = self.bps // 8
        idx = (row_in * self.w + col) * bps
        if self.bps == 64 and self.sample_format == 3:
            z = struct.unpack_from(self.endian + "d", buf, idx)[0]
        elif self.bps == 32 and self.sample_format == 3:
            z = struct.unpack_from(self.endian + "f", buf, idx)[0]
        else:
            raise RuntimeError(f"Unsupported sample format in {self.path.name}")
        if not math.isfinite(z) or abs(z - NODATA_SRC) < 0.5 or z < -500:
            return None
        return float(z)


def load_tiles() -> list[DgmTile]:
    tiles = []
    for tfw in sorted(DGM_DIR.glob("*.tfw")):
        tif = tfw.with_suffix(".tif")
        if not tif.exists():
            continue
        tiles.append(DgmTile(tif, tfw))
    if not tiles:
        raise SystemExit(f"No DGM tiles in {DGM_DIR}")
    return tiles


def sample_z(tiles: list[DgmTile], x: float, y: float) -> float | None:
    for t in tiles:
        if t.contains(x, y):
            z = t.sample(x, y)
            if z is not None:
                return z
    return None


def fit_plane(points: list[tuple[float, float, float]]):
    """Least-squares Z = aX + bY + c. Returns (a,b,c) or None."""
    if len(points) < 3:
        return None
    # Normal equations for [X,Y,1] → Z
    sxx = syy = sxy = sx = sy = sz = sxz = syz = n = 0.0
    for x, y, z in points:
        sxx += x * x
        syy += y * y
        sxy += x * y
        sx += x
        sy += y
        sz += z
        sxz += x * z
        syz += y * z
        n += 1.0
    # Solve 3×3
    A = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx, sy, n],
    ]
    b = [sxz, syz, sz]
    det = (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )
    if abs(det) < 1e-9:
        return None
    def det3(M):
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        )
    A0 = [[b[0], A[0][1], A[0][2]], [b[1], A[1][1], A[1][2]], [b[2], A[2][1], A[2][2]]]
    A1 = [[A[0][0], b[0], A[0][2]], [A[1][0], b[1], A[1][2]], [A[2][0], b[2], A[2][2]]]
    A2 = [[A[0][0], A[0][1], b[0]], [A[1][0], A[1][1], b[1]], [A[2][0], A[2][1], b[2]]]
    return det3(A0) / det, det3(A1) / det, det3(A2) / det


def build_heightmap(tiles: list[DgmTile]):
    xmin = min(t.xmin for t in tiles)
    xmax = max(t.xmax for t in tiles)
    ymin = min(t.ymin for t in tiles)
    ymax = max(t.ymax for t in tiles)
    # Snap origin to nice grid
    origin_x = math.floor(xmin / ORIGIN_SNAP) * ORIGIN_SNAP
    origin_y = math.ceil(ymax / ORIGIN_SNAP) * ORIGIN_SNAP  # top (north)
    width = int(math.ceil((xmax - origin_x) / RES_M)) + 1
    height = int(math.ceil((origin_y - ymin) / RES_M)) + 1
    print(f"heightmap {width}×{height} @ {RES_M}m  origin=({origin_x},{origin_y})", flush=True)

    grid = [NODATA_OUT] * (width * height)
    # Fill by streaming each tile once (nearest cell)
    for ti, tile in enumerate(tiles):
        print(f"  tile {ti+1}/{len(tiles)} {tile.path.name}", flush=True)
        tile.open()
        step = max(1, int(round(RES_M / abs(tile.a))))
        try:
            for row in range(0, tile.h, step):
                sidx = row // tile.rows_per_strip
                row_in = row % tile.rows_per_strip
                buf = tile._read_strip(sidx)
                y = tile.f + row * tile.d
                gy = int(round((origin_y - y) / RES_M))
                if gy < 0 or gy >= height:
                    continue
                bps = tile.bps // 8
                for col in range(0, tile.w, step):
                    x = tile.e + col * tile.a
                    gx = int(round((x - origin_x) / RES_M))
                    if gx < 0 or gx >= width:
                        continue
                    idx = (row_in * tile.w + col) * bps
                    if tile.bps == 64:
                        z = struct.unpack_from(tile.endian + "d", buf, idx)[0]
                    else:
                        z = struct.unpack_from(tile.endian + "f", buf, idx)[0]
                    if not math.isfinite(z) or abs(z - NODATA_SRC) < 0.5 or z < -500:
                        continue
                    # store decimetres
                    v = int(round(z * 10.0))
                    v = max(-32767, min(32767, v))
                    grid[gy * width + gx] = v
        finally:
            tile.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / "heightmap.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack(f"<{len(grid)}h", *grid))
    meta = {
        "crs": "MGI / Austria Gauss-Krueger M34 (same as oblique image.json)",
        "originX": origin_x,
        "originY": origin_y,
        "res": RES_M,
        "width": width,
        "height": height,
        "dtype": "int16",
        "unit": "decimetres",
        "nodata": NODATA_OUT,
        "zMetres": "value / 10",
        "note": "North-up: row 0 is originY; col 0 is originX. Sample: "
                "col=(X-originX)/res, row=(originY-Y)/res.",
    }
    (OUT_DIR / "heightmap.json").write_text(json.dumps(meta, indent=2) + "\n")
    filled = sum(1 for v in grid if v != NODATA_OUT)
    print(f"wrote {bin_path} ({bin_path.stat().st_size/1e6:.2f} MB), filled {filled}/{len(grid)}", flush=True)
    return meta


def look_at_from_p(Pm, width, height):
    """Ground point (Z=0) that image centre looks at — mirrors infinite-scroll.html."""
    px_c = width / 2
    py_c = height / 2
    a1 = Pm[0][0] - px_c * Pm[2][0]
    b1 = Pm[0][1] - px_c * Pm[2][1]
    c1 = px_c * Pm[2][3] - Pm[0][3]
    a2 = Pm[1][0] - py_c * Pm[2][0]
    b2 = Pm[1][1] - py_c * Pm[2][1]
    c2 = py_c * Pm[2][3] - Pm[1][3]
    det = a1 * b2 - b1 * a2
    if abs(det) <= 1e-10:
        return None
    return (c1 * b2 - b1 * c2) / det, (a1 * c2 - c1 * a2) / det


def order_footprint_ccw(ring):
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    ordered = sorted(ring, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    area = 0.0
    for i in range(len(ordered)):
        j = (i + 1) % len(ordered)
        area += ordered[i][0] * ordered[j][1] - ordered[j][0] * ordered[i][1]
    if area < 0:
        ordered.reverse()
    return ordered


def build_image_offsets(tiles: list[DgmTile]):
    out = {}
    tile_dirs = [
        ROOT / "sources" / "2023" / "tiles",
        ROOT / "sources" / "2020" / "tiles",
    ]
    files = []
    for d in tile_dirs:
        if d.is_dir():
            files.extend(sorted(d.glob("*.json")))
    print(f"image offsets from {len(files)} tile JSON files", flush=True)

    for fi, path in enumerate(files):
        data = json.loads(path.read_text())
        rows = data.get("images") or []
        if len(rows) < 2:
            continue
        hdr = rows[0]
        iN = hdr.index("name")
        iW = hdr.index("width")
        iH = hdr.index("height")
        iGC = hdr.index("groundCoordinates")
        iCtr = hdr.index("centerPointOnGround")
        iPTI = hdr.index("p-to-image")
        for r in rows[1:]:
            name = r[iN]
            if not name or name in out:
                continue
            gc = r[iGC] or []
            center = r[iCtr]
            Pm = r[iPTI]
            w = r[iW] or 14144
            h = r[iH] or 10560
            look = look_at_from_p(Pm, w, h) if Pm else None
            if look is None and center:
                look = (center[0], center[1])
            if look is None:
                continue
            corners_xy = []
            if gc and len(gc) >= 3:
                ring = order_footprint_ccw([[c[0], c[1]] for c in gc])
                # Take 4 extremes if more than 4
                if len(ring) == 4:
                    corners_xy = ring
                else:
                    # bbox corners of footprint
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    corners_xy = [
                        [min(xs), min(ys)],
                        [max(xs), min(ys)],
                        [max(xs), max(ys)],
                        [min(xs), max(ys)],
                    ]
            pts = []
            corner_zs = []
            for xy in corners_xy:
                z = sample_z(tiles, xy[0], xy[1])
                if z is None:
                    corner_zs.append(None)
                else:
                    corner_zs.append(round(z, 2))
                    pts.append((xy[0], xy[1], z))
            z_look = sample_z(tiles, look[0], look[1])
            if z_look is not None:
                pts.append((look[0], look[1], z_look))
            plane = fit_plane(pts) if len(pts) >= 3 else None
            entry = {}
            if z_look is not None:
                entry["z"] = round(z_look, 2)
            if corners_xy:
                entry["cz"] = corner_zs
            if plane:
                entry["plane"] = {
                    "a": round(plane[0], 8),
                    "b": round(plane[1], 8),
                    "c": round(plane[2], 3),
                }
            if entry:
                out[name] = entry
        if (fi + 1) % 6 == 0:
            print(f"  {fi+1}/{len(files)} files, {len(out)} images", flush=True)
        # Close tile file handles periodically
        if (fi + 1) % 12 == 0:
            for t in tiles:
                t.close()

    for t in tiles:
        t.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "image-offsets.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size/1e6:.2f} MB), {len(out)} images", flush=True)
    return out


def main():
    if not DGM_DIR.is_dir():
        raise SystemExit(f"Missing {DGM_DIR}")
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tiles = load_tiles()
    print(f"loaded {len(tiles)} DGM tiles", flush=True)
    if mode in ("all", "heightmap"):
        build_heightmap(tiles)
        for t in tiles:
            t.close()
    if mode in ("all", "images"):
        build_image_offsets(tiles)


if __name__ == "__main__":
    main()
