#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time ET shared counts against private accumulators with exact model checks.

First run extratrees/tools/check_shared_score.sh to build the six binaries.
Then use tools/with_build_lock.sh python3 extratrees/bench/shared_score_ab.py OUTPUT.
The build lock must span timing; this driver additionally takes the bench lock.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    if os.environ.get("MOJOLEARN_BUILD_LOCK_HELD") != "1":
        raise SystemExit("Run via tools/with_build_lock.sh")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    cases = [(65536, 2, 16, 10), (262144, 2, 16, 10),
             (65536, 5, 8, 8), (65536, 9, 8, 8),
             (65536, 17, 8, 8), (65536, 32, 8, 8)]
    if os.environ.get("MOJOLEARN_ET_BENCH_MULTICLASS_ONLY") == "1":
        cases = [case for case in cases if case[1] >= 5]
    wanted_classes = os.environ.get("MOJOLEARN_ET_BENCH_CLASSES")
    if wanted_classes:
        selected = {int(v) for v in wanted_classes.split(",")}
        cases = [case for case in cases if case[1] in selected]
    modes = os.environ.get("MOJOLEARN_ET_BENCH_MODES", "fast,identical").split(",")
    if not cases or any(mode not in ("fast", "identical", "deterministic") for mode in modes):
        raise SystemExit("Invalid or empty benchmark mode/class selection")
    results = []
    subprocess.run(["tools/bench_lock.sh", "acquire", "et-shared-counts", "exact model ABBA fits", "2 minutes"], check=True)
    try:
        for mode in modes:
            for rows, classes, trees, depth in cases:
                samples = {"baseline": [], "candidate": []}
                medians_by_pass = {"baseline": [], "candidate": []}
                fingerprints = set()
                for rep, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
                    cmd = [f"/tmp/et-shared-{mode}-{arm}", str(rows), str(classes), str(trees), str(depth), "3"]
                    run = subprocess.run(cmd, text=True, capture_output=True)
                    (output / f"{mode}-{rows}-{classes}-{arm}-{rep}.log").write_text(run.stdout + run.stderr)
                    run.check_returncode()
                    lines = run.stdout.splitlines()
                    assert f"numeric_mode {mode.upper()}" in lines
                    assert f"shared_class_counts_mask {15 if arm == 'candidate' else 0}" in lines
                    hashes = [s for s in lines if s.startswith("fingerprint ")]
                    times = [float(s.split()[-1]) for s in lines if s.startswith("fit_ms ")]
                    assert len(hashes) == 1 and len(times) == 3
                    fingerprints.update(hashes)
                    samples[arm].extend(times)
                    medians_by_pass[arm].append(statistics.median(times))
                assert len(fingerprints) == 1, fingerprints
                medians = {arm: statistics.median(values) for arm, values in samples.items()}
                drift = max(medians_by_pass["baseline"]) / min(medians_by_pass["baseline"])
                candidate_drift = max(medians_by_pass["candidate"]) / min(medians_by_pass["candidate"])
                result = dict(mode=mode, warmup_fits=int(os.environ.get("MOJOLEARN_ET_WARMUP_FITS", "1")), rows=rows, classes=classes, cols=13, trees=trees, depth=depth,
                              medians_ms=medians, speedup=medians["baseline"] / medians["candidate"],
                              baseline_drift_ratio=drift, candidate_drift_ratio=candidate_drift,
                              stable=max(drift, candidate_drift) <= 1.10,
                              samples_ms=samples, fingerprint=fingerprints.pop())
                results.append(result)
                print(json.dumps(result), flush=True)
                (output / "timing.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        subprocess.run(["tools/bench_lock.sh", "release"], check=True)


if __name__ == "__main__":
    main()
