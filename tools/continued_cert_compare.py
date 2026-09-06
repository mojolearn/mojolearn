#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate serial ordered/kNN checks and compare exact records across GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


ARMS = {"baseline": (0, 0), "selector": (1, 0), "transpose": (0, 1), "both": (1, 1)}
KNN_FIXTURES = {(d, p, metric) for d in (1, 3, 17, 33, 65) for p in range(3) for metric in (0, 1)} | {(17, 0, 2), (17, 0, 3)}


def knn_records(directory):
    records, digests = {}, {}
    for arm, flags in ARMS.items():
        lines = (directory / f"knn-{arm}.log").read_text().splitlines()
        assert lines[0] == f"ADVERSARIAL_FLAGS IDENTICAL {flags[0]} {flags[1]}", f"wrong {arm} flags"
        assert lines[-1] == "KNN ADVERSARIAL PASS cases 32 selected_pairs 5440", f"incomplete {arm}"
        cells = [line for line in lines if line.startswith("ADVERSARIAL_CELL ")]
        assert len(cells) == 5440, f"wrong {arm} cell count"
        keys = []
        for line in cells:
            fields = line.split()
            assert len(fields) == 7
            dimension, profile, metric, cell, bits, neighbor = map(int, fields[1:])
            assert 0 <= cell < 170 and 0 <= bits < 2**32 and 0 <= neighbor < 129
            keys.append((dimension, profile, metric, cell))
        assert len(set(keys)) == 5440, f"duplicate {arm} cell"
        assert set(keys) == {(*fixture, cell) for fixture in KNN_FIXTURES for cell in range(170)}, f"wrong {arm} fixture inventory"
        records[arm] = cells
        digests[arm] = hashlib.sha256(("\n".join(cells) + "\n").encode()).hexdigest()
    assert all(records[arm] == records["baseline"] for arm in ARMS), "kNN flag output bytes differ"
    return records["baseline"], digests


def load(directory):
    commit = (directory / "commit.txt").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "missing frozen source commit"
    expected = {"ordered-build", "ordered-check", "ctr-build", "ctr-check"}
    expected.update(f"knn-build-{arm}" for arm in ARMS)
    expected.update(f"knn-{arm}" for arm in ARMS)
    statuses = [line.split("\t") for line in (directory / "status.tsv").read_text().splitlines()]
    assert len(statuses) == len(expected) and {row[0] for row in statuses} == expected
    assert all(row[1] == "0" for row in statuses), "a native build/check failed"
    assert (directory / "completion.txt").read_text().strip() == "continued_exit=0"
    knn, digests = knn_records(directory)
    ordered_log = (directory / "ordered-check.log").read_text()
    assert "ORDERED NUMERIC MODE IDENTICAL\n" in ordered_log
    assert "ORDERED RMSE PASS:" in ordered_log
    ordered = [line for line in ordered_log.splitlines() if line.startswith("ORDERED_BITS ")]
    assert len(ordered) == len(set(ordered)), "duplicate ordered record"
    for kind, count in (("prefix", 12), ("prediction", 32), ("fold_cursor", 48),
                        ("zero_prefix", 12), ("constant_model", 8)):
        assert sum(line.startswith(f"ORDERED_BITS {kind} ") for line in ordered) == count, f"wrong {kind} coverage"
    for kind in ("split", "leaf"):
        assert {int(line.split()[2]) for line in ordered if line.startswith(f"ORDERED_BITS {kind} ")} == {0, 1, 2}
    ctr = (directory / "ctr-check.log").read_text()
    assert "WEIGHTED LEAF PASS:" in ctr
    return {"directory": str(directory), "commit": commit, "knn_cells": len(knn),
            "knn_arm_sha256": digests, "ordered_records": len(ordered),
            "ordered_sha256": hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()}, knn, ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    left, knn, ordered = load(args.left)
    result = {"left": left, "status": "PASS", "scope": "Native correctness; no timing or external CatBoost parity"}
    if args.right:
        right, right_knn, right_ordered = load(args.right)
        assert left["commit"] == right["commit"], "different frozen source commits"
        assert knn == right_knn, "cross-vendor kNN bits differ"
        assert ordered == right_ordered, "cross-vendor ordered bits differ"
        result.update(right=right, cross_vendor_bitwise=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
