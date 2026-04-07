"""
Parse orthofotos/2023/*.jgw (ESRI world files) and emit ortho-tiles.json.
Full rasters are 16667×16667; previews share the same ground extent as the full raster.
Prefer *_s0.jpg (1024×1024); fall back to *_s1.jpg (2381×2381) if s0 is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

FULL_W = 16667
FULL_H = 16667
PREVIEW_S0 = 1024
PREVIEW_S1 = 2381


def extent_from_jgw(lines: list[float]) -> tuple[float, float, float, float]:
    """Return xmin, xmax, ymin, ymax in map units (center of upper-left pixel at line5, line6)."""
    a, _b, _c, d, e, f = lines[:6]
    # Upper-left pixel center (e, f); pixel width a; row step d (negative = north-up).
    xmin = e - 0.5 * a
    xmax = e + (FULL_W - 0.5) * a
    ymax = f - 0.5 * d  # d < 0 => ymax > f
    ymin = f + (FULL_H - 0.5) * d
    return xmin, xmax, ymin, ymax


def main() -> None:
    root = Path(__file__).resolve().parent / "orthofotos" / "2023"
    tiles: list[dict] = []
    global_xmin = global_ymin = float("inf")
    global_xmax = global_ymax = float("-inf")

    for jgw_path in sorted(root.glob("*.jgw")):
        stem = jgw_path.stem  # e.g. 15_1_op_2023
        text = jgw_path.read_text(encoding="utf-8", errors="replace")
        lines = [float(x.strip()) for x in text.splitlines() if x.strip()]
        if len(lines) < 6:
            continue
        a, b, c, d, e, f = lines[:6]
        xmin, xmax, ymin, ymax = extent_from_jgw(lines)

        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)

        s0_path = root / f"{stem}_s0.jpg"
        s1_path = root / f"{stem}_s1.jpg"
        if s0_path.is_file():
            preview_file = f"{stem}_s0.jpg"
            preview_w, preview_h = PREVIEW_S0, PREVIEW_S0
        elif s1_path.is_file():
            preview_file = f"{stem}_s1.jpg"
            preview_w, preview_h = PREVIEW_S1, PREVIEW_S1
        else:
            continue

        tiles.append(
            {
                "id": stem,
                "previewUrl": f"orthofotos/2023/{preview_file}",
                "upperLeftCenterX": e,
                "upperLeftCenterY": f,
                "pixelWidth": a,
                "pixelHeight": d,
                "rotationB": b,
                "rotationC": c,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "fullWidth": FULL_W,
                "fullHeight": FULL_H,
                "previewWidth": preview_w,
                "previewHeight": preview_h,
            }
        )

    out = {
        "crs": "MGI / Austria (EPSG:31256 area — metres, matches Stadt Wien oblique dataset)",
        "fullRasterPx": [FULL_W, FULL_H],
        "previewRasterPx": [PREVIEW_S0, PREVIEW_S0],
        "bbox": {
            "xmin": global_xmin,
            "xmax": global_xmax,
            "ymin": global_ymin,
            "ymax": global_ymax,
        },
        "tiles": tiles,
    }

    out_path = root / "ortho-tiles.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(tiles)} tiles to {out_path}")


if __name__ == "__main__":
    main()
