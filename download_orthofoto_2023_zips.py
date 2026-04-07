#!/usr/bin/env python3
"""
Download all Vienna orthophoto 2023 zip tiles from the public viewer.

URL pattern (not every combination exists; many return 404):
  https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/op_img/{XY}_{Z}_op_2023.zip
  where X ∈ [1–5], Y ∈ [2–8], Z ∈ [1–4]
  Example: 35_3_op_2023.zip

Requires: Python 3.8+ (stdlib only).
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

BASE_URL = "https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/op_img/"
DEFAULT_OUTPUT_DIR = Path("Orthofoto2023")
USER_AGENT = (
    "download_orthofoto_2023_zips/1.0 (+https://www.wien.gv.at/ma41datenviewer/)"
)


def iter_filenames() -> List[str]:
    names: List[str] = []
    for d1 in range(1, 6):  # 1–5
        for d2 in range(2, 9):  # 2–8
            for d3 in range(1, 5):  # 1–4
                names.append(f"{d1}{d2}_{d3}_op_2023.zip")
    return names


def download_one(
    out_dir: Path,
    filename: str,
    timeout: float,
) -> Tuple[str, str, str]:
    """
    Returns (status, filename, detail).
    status: 'ok' | 'skip' | '404' | 'error'
    """
    url = BASE_URL + filename
    dest = out_dir / filename
    if dest.is_file():
        return ("skip", filename, "")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return ("ok", filename, "")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ("404", filename, "HTTP 404 Not Found")
        return ("error", filename, f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        return ("error", filename, str(e.reason))
    except OSError as e:
        return ("error", filename, str(e))
    except Exception as e:  # noqa: BLE001 — surface unexpected failures
        return ("error", filename, repr(e))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Vienna orthophoto 2023 zip tiles (pattern XY_Z_op_2023.zip)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to store zips (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download threads (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=220.0,
        help="Per-request timeout in seconds (default: 220)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    names = iter_filenames()
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Candidate files: {len(names)}")
    print()

    not_found: List[str] = []
    errors: List[Tuple[str, str]] = []
    skipped: List[str] = []
    new_ok = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(download_one, out_dir, n, args.timeout): n for n in names
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            status, filename, detail = fut.result()
            if status == "ok":
                new_ok += 1
                print(f"[{done}/{len(names)}] OK   {filename}")
            elif status == "skip":
                skipped.append(filename)
                print(f"[{done}/{len(names)}] SKIP {filename} (already exists)")
            elif status == "404":
                not_found.append(filename)
                print(f"[{done}/{len(names)}] 404  {filename}")
            else:
                errors.append((filename, detail))
                print(f"[{done}/{len(names)}] ERR  {filename} — {detail}")

    print()
    print("=" * 60)
    print(f"404 Not Found ({len(not_found)}):")
    for f in sorted(not_found):
        print(f"  {f}")

    print()
    print(f"Other errors ({len(errors)}):")
    for fname, detail in sorted(errors, key=lambda x: x[0]):
        print(f"  {fname}: {detail}")

    print()
    print(
        f"Summary: new downloads {new_ok}, skipped (already present) {len(skipped)}, "
        f"404 {len(not_found)}, other errors {len(errors)}."
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
