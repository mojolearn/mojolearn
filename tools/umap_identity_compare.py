#!/usr/bin/env python3
"""Compare complete named UMAP stage captures by uint32 bits, never tolerance.

Source/build and hardware provenance must accompany the captures; equality
here covers only this fixture, not other shapes or installed artifacts.
"""

import argparse
from pathlib import Path

PROFILE = "UMAP_PROFILE umap.identical.8x1.2d.e4.seed19.v1"
COUNTS = {"input": 8, "rho": 8, "sigma": 8, "directed": 64,
          "weights": 64, "curve": 2, "initial": 16, "layout": 16}
EXPECTED = {(stage, i) for stage, count in COUNTS.items() for i in range(count)}


def read_capture(path):
    records = {}
    lines = Path(path).read_text().splitlines()
    if lines.count(PROFILE) != 1 or lines.count("UMAP identity fixture PASS") != 1:
        raise ValueError(f"{path}: missing or repeated fixture profile/completion")
    for line in lines:
        if not line.startswith("UMAP_BITS "):
            continue
        _, stage, index, value = line.split()
        key = stage, int(index)
        bits = int(value)
        if key in records or key not in EXPECTED:
            raise ValueError(f"{path}: duplicate or unknown record {key}")
        if not 0 <= bits <= 0xFFFFFFFF or bits & 0x7F800000 == 0x7F800000:
            raise ValueError(f"{path}: invalid/non-finite bits at {key}")
        records[key] = bits
    if records.keys() != EXPECTED:
        raise ValueError(f"{path}: missing {len(EXPECTED - records.keys())} records")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    try:
        left, right = read_capture(args.left), read_capture(args.right)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"INVALID: {exc}\n")
    differences = [key for key in left if left[key] != right[key]]
    if differences:
        key = differences[0]
        parser.exit(1, f"FAIL: {len(differences)} cells differ; first {key}: "
                    f"0x{left[key]:08x} != 0x{right[key]:08x}\n")
    print(f"PASS: {len(left)} uint32 cells identical across {len(COUNTS)} stages")


if __name__ == "__main__":
    main()
