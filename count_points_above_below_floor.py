#!/usr/bin/env python3
"""Count and optionally filter COLMAP sparse points by floor threshold."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


AXIS_TO_INDEX = {"x": 1, "y": 2, "z": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many points in a COLMAP points3D.txt file are above or "
            "below a floor value."
        )
    )
    parser.add_argument(
        "points_file",
        type=Path,
        help="Path to COLMAP points3D.txt",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Coordinate axis used for floor test (default: z)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="Floor coordinate on selected axis (default: 0.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help=(
            "Tolerance around floor to treat points as on-floor and exclude from "
            "above/below counts (default: 1e-9)"
        ),
    )
    parser.add_argument(
        "--remove-below-floor",
        action="store_true",
        help="Write an output file that removes points below floor",
    )
    parser.add_argument(
        "--remove-above-floor",
        action="store_true",
        help="Write an output file that removes points above floor",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output path for filtered points3D.txt (required with "
            "--remove-below-floor unless --in-place is used)"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input file directly (creates .bak backup)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axis_index = AXIS_TO_INDEX[args.axis]

    if not args.points_file.exists():
        raise SystemExit(f"File not found: {args.points_file}")

    total = 0
    above = 0
    below = 0
    on_floor = 0
    skipped = 0
    kept_lines: list[str] = []
    header_lines: list[str] = []
    filter_enabled = args.remove_below_floor or args.remove_above_floor

    with args.points_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if filter_enabled:
                    header_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) < 4:
                skipped += 1
                continue

            try:
                value = float(parts[axis_index])
            except ValueError:
                skipped += 1
                continue

            total += 1
            delta = value - args.floor
            if delta > args.epsilon:
                above += 1
                if args.remove_below_floor:
                    kept_lines.append(line)
            elif delta < -args.epsilon:
                below += 1
                if args.remove_above_floor:
                    kept_lines.append(line)
            else:
                on_floor += 1
                if filter_enabled:
                    kept_lines.append(line)

    print(f"File: {args.points_file}")
    print(f"Axis: {args.axis}")
    print(f"Floor value: {args.floor}")
    print(f"Tolerance (epsilon): {args.epsilon}")
    print(f"Total valid points: {total}")
    print(f"Above floor: {above}")
    print(f"Below floor: {below}")
    print(f"On floor (within epsilon): {on_floor}")
    if skipped:
        print(f"Skipped malformed lines: {skipped}")

    if args.remove_below_floor and args.remove_above_floor:
        raise SystemExit(
            "Use only one of --remove-below-floor or --remove-above-floor."
        )

    if not filter_enabled:
        return

    remove_below = args.remove_below_floor

    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or --output, not both.")
    if not args.in_place and args.output is None:
        raise SystemExit(
            "Filtering requires --output or --in-place."
        )

    output_path = args.points_file if args.in_place else args.output
    assert output_path is not None

    if args.in_place:
        backup_path = args.points_file.with_suffix(args.points_file.suffix + ".bak")
        shutil.copyfile(args.points_file, backup_path)
        print(f"Backup written to: {backup_path}")

    with output_path.open("w", encoding="utf-8") as out:
        for line in header_lines:
            out.write(line)
        for line in kept_lines:
            out.write(line)

    print(f"Filtered file written to: {output_path}")
    if remove_below:
        print(f"Removed below-floor points: {below}")
    else:
        print(f"Removed above-floor points: {above}")


if __name__ == "__main__":
    main()
