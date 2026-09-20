#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strictly compare baseline/candidate LM shard probes on one and two devices."""
import argparse
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text())


def compare(paths):
    runs = {name: _load(path) for name, path in paths.items()}
    reference = runs["baseline_one"]
    for name, run in runs.items():
        for key in ("schema", "shape", "parameters", "steps_requested", "seed"):
            if run.get(key) != reference.get(key):
                raise ValueError(f"{name}: experiment field {key} differs")
    arms = {name: {row["logical_shards"]: row for row in run["arms"]}
            for name, run in runs.items()}
    shard_sets = {name: set(rows) for name, rows in arms.items()}
    if any(value != shard_sets["baseline_one"] for value in shard_sets.values()):
        raise ValueError("logical shard arms differ")
    output = []
    for shards in sorted(shard_sets["baseline_one"]):
        rows = {name: values[shards] for name, values in arms.items()}
        for name, row in rows.items():
            if row.get("refused"):
                raise ValueError(f"{name}: shards={shards} refused: {row['refused']}")
        expected_hash = rows["baseline_one"]["final_sha256"]
        expected_losses = [step["losses_sha256"] for step in rows["baseline_one"]["steps"]]
        for name, row in rows.items():
            if row["final_sha256"] != expected_hash:
                raise ValueError(f"{name}: shards={shards} final hashes differ")
            if [step["losses_sha256"] for step in row["steps"]] != expected_losses:
                raise ValueError(f"{name}: shards={shards} loss hashes differ")
        b1 = rows["baseline_one"]["steady_median_seconds"]
        b2 = rows["baseline_two"]["steady_median_seconds"]
        c1 = rows["candidate_one"]["steady_median_seconds"]
        c2 = rows["candidate_two"]["steady_median_seconds"]
        output.append(dict(logical_shards=shards, exact=True, hashes=expected_hash,
                           baseline_one_seconds=b1, baseline_two_seconds=b2,
                           candidate_one_seconds=c1, candidate_two_seconds=c2,
                           baseline_two_device_speedup=b1 / b2,
                           candidate_two_device_speedup=c1 / c2,
                           candidate_vs_baseline_one=b1 / c1,
                           candidate_vs_baseline_two=b2 / c2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline_one", "baseline_two", "candidate_one", "candidate_two"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(vars(args)), indent=2))


if __name__ == "__main__":
    main()
