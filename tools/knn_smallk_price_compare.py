#!/usr/bin/env python3
"""Validate retained native public k-NN timings; never execute a benchmark.

Usage: python tools/knn_smallk_price_compare.py DIRECTORY [--other-vendor DIR]
An optional --output FILE writes the same JSON printed to stdout. A second
vendor is compared for exact output bytes, not a cross-vendor speed claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

QUERIES = (32, 128, 1000)
ARMS = ("legacy", "experimental")
ROUNDS = range(9)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_log(path, queries, arm):
    data = path.read_bytes()
    lines = [line.split() for line in data.decode("utf-8").splitlines() if line.strip()]
    header = ["SMALLK_PRICE", "experimental", str(ARMS.index(arm)), "mode", "IDENTICAL",
              "fixture", "dyadic-v1", "index", "100000", "queries", str(queries),
              "features", "32", "k", "10", "index_salt", "0", "query_salt", "593",
              "requested_tile", "256", "warmups", "2", "timed_calls", "1", "scope",
              "native-public-upload-search-download-synchronize"]
    require([line for line in lines if line[0] == "SMALLK_PRICE"] == [header],
            f"{path}: missing, duplicated or incorrect activation/fixture header")
    marker = ["KNN", "SMALL-K", "PUBLIC", "PRICE", "PASS", "queries", str(queries),
              "selected_pairs", str(queries * 10)]
    require([line for line in lines if line[0] == "KNN"] == [marker],
            f"{path}: missing, duplicated or incorrect pass marker")
    require(lines[-1] == marker, f"{path}: pass marker must be final record")
    require(lines.index(header) < lines.index(marker), f"{path}: invalid record order")
    cells = {}
    timing = []
    for line in lines:
        if line[0] == "PRICE_CELL":
            require(len(line) == 9 and line[1:6] ==
                    ["dyadic-v1", "100000", str(queries), "32", "10"],
                    f"{path}: malformed cell fixture")
            cell, bits, neighbor = map(int, line[6:])
            require(0 <= cell < queries * 10 and cell not in cells,
                    f"{path}: duplicate or out-of-range cell")
            require(0 <= bits <= 0xffffffff and 0 <= neighbor < 100000,
                    f"{path}: invalid distance bits or neighbor bounds")
            distance = struct.unpack("!f", struct.pack("!I", bits))[0]
            require(math.isfinite(distance) and distance >= 0,
                    f"{path}: nonfinite or negative distance")
            cells[cell] = (bits, neighbor)
        elif line[0] == "PRICE_MS":
            require(len(line) == 9 and line[1:6] ==
                    ["dyadic-v1", "100000", str(queries), "32", "10"]
                    and line[7] == "used_tile", f"{path}: malformed timing fixture")
            elapsed, tile = float(line[6]), int(line[8])
            require(math.isfinite(elapsed) and elapsed > 0,
                    f"{path}: timing must be positive and finite")
            require(1 <= tile <= min(256, queries), f"{path}: invalid used tile")
            timing.append((elapsed, tile))
        elif line[0].startswith("PRICE_"):
            raise ValueError(f"{path}: unknown price record")
    require(len(cells) == queries * 10, f"{path}: incomplete selected output")
    require(len(timing) == 1, f"{path}: expected exactly one timing")
    return tuple(cells[cell] for cell in range(queries * 10)), {
        "milliseconds": timing[0][0], "used_tile": timing[0][1],
        "log": path.name, "sha256": hashlib.sha256(data).hexdigest(),
    }


def distribution(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "minimum": min(values), "maximum": max(values)}


def validate_directory(directory):
    directory = Path(directory)
    require((directory / "completion.txt").read_text() == "COMPLETE\n",
            f"{directory}: missing exact campaign completion marker")
    names = [f"q{q}-r{r}-{arm}" for q in QUERIES for r in ROUNDS for arm in ARMS]
    expected_status = {"build-legacy", "build-experimental", *names}
    status = {}
    for line in (directory / "status.tsv").read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == 2 and fields[0] not in status,
                f"{directory}: malformed or duplicate status")
        status[fields[0]] = fields[1]
    require(set(status) == expected_status and set(status.values()) == {"0"},
            f"{directory}: incomplete or unsuccessful build/run status")
    require({p.stem for p in directory.glob("q*-r*-*.log")} == set(names),
            f"{directory}: expected exactly 54 campaign logs")
    witnesses = {}
    shapes = {}
    for q in QUERIES:
        samples = {arm: [] for arm in ARMS}
        for r in ROUNDS:
            for arm in ARMS:
                cells, sample = parse_log(directory / f"q{q}-r{r}-{arm}.log", q, arm)
                if q not in witnesses:
                    witnesses[q] = cells
                require(cells == witnesses[q],
                        f"{directory}: selected bytes differ for q{q}, round {r}, {arm}")
                samples[arm].append({"round": r, **sample})
        ratios = [samples["legacy"][r]["milliseconds"] /
                  samples["experimental"][r]["milliseconds"] for r in ROUNDS]
        shapes[str(q)] = {
            "selected_pairs": q * 10,
            "output_sha256": hashlib.sha256(b"".join(
                struct.pack("!II", bits, index) for bits, index in witnesses[q])).hexdigest(),
            "samples": samples,
            "milliseconds": {arm: distribution([s["milliseconds"] for s in samples[arm]])
                             for arm in ARMS},
            "paired_legacy_over_experimental": {"raw": ratios, **distribution(ratios)},
        }
    return {"directory": str(directory.resolve()), "status": "PASS", "log_count": 54,
            "scope": "native-public-upload-search-download-synchronize",
            "quartile_method": "inclusive linear interpolation",
            "ratio_direction": "legacy milliseconds / experimental milliseconds; above 1 favors experimental",
            "shapes": shapes}, witnesses


def compare_directories(directory, other=None):
    report, witnesses = validate_directory(directory)
    result = {"status": "PASS", "primary": report}
    if other is not None:
        other_report, other_witnesses = validate_directory(other)
        require(witnesses == other_witnesses, "cross-vendor selected bytes differ")
        result.update(other_vendor=other_report, cross_vendor_bits="PASS",
                      cross_vendor_speed_comparison="not performed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--other-vendor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_directories(args.directory, args.other_vendor)
    except (ValueError, OSError, UnicodeError, OverflowError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
