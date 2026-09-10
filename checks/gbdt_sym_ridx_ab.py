#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare separately built symmetric2031 FAST binaries under both repo locks.

Arguments: baseline_binary candidate_binary output_directory.
Compile checks/gbdt_sym_ridx_fit_check.mojo with/without
-D MOJOLEARN_2031_SYM_RIDX_SPLITS=1. Run this driver through
 tools/with_build_lock.sh (the driver refuses otherwise).
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    if os.environ.get("MOJOLEARN_BUILD_LOCK_HELD") != "1":
        raise SystemExit("Run this driver through tools/with_build_lock.sh")
    baseline, candidate, output = sys.argv[1:]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        dict(rows=65537, cols=8, depth=6, trees=12, borders=15, loss="Logloss"),
        dict(rows=262145, cols=28, depth=8, trees=12, borders=32, loss="Logloss"),
        dict(rows=262145, cols=28, depth=8, trees=12, borders=64, loss="RMSE"),
        dict(rows=1000003, cols=28, depth=8, trees=8, borders=32, loss="Logloss"),
    ]
    subprocess.run(["tools/bench_lock.sh", "acquire", "symmetric2031", "full-model ABBA", "3 minutes"], check=True)
    results = []
    try:
        for case_id, case in enumerate(cases):
            env = os.environ.copy()
            env.update({"MOJOLEARN_SYM_" + k.upper(): str(v) for k, v in case.items()})
            env["MOJOLEARN_SYM_REPS"] = "3"
            fingerprints = set()
            samples = {"baseline": [], "candidate": []}
            pass_medians = {"baseline": [], "candidate": []}
            for run, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
                binary = baseline if arm == "baseline" else candidate
                completed = subprocess.run([binary], env=env, text=True, capture_output=True, check=True)
                (out / f"case{case_id}.{run}.{arm}.log").write_text(completed.stdout + completed.stderr)
                lines = completed.stdout.splitlines()
                if "numeric_mode FAST" not in lines:
                    raise AssertionError("wrong compiled numeric mode")
                expected = "False" if arm == "baseline" else "True"
                if not any(s.startswith(f"ridx {expected} ") for s in lines):
                    raise AssertionError("wrong compiled2031 route")
                hashes = [s.split()[-1] for s in lines if s.startswith("fingerprint ")]
                times = [float(s.split()[-1]) for s in lines if s.startswith("fit_ms ")]
                if len(hashes) != 4 or len(times) != 3:
                    raise AssertionError("incomplete run")
                fingerprints.update(hashes)
                samples[arm].extend(times)
                pass_medians[arm].append(statistics.median(times))
            if len(fingerprints) != 1:
                raise AssertionError(f"model/prediction/loss fingerprint mismatch case{case_id}: {fingerprints}")
            medians = {arm: statistics.median(values) for arm, values in samples.items()}
            drift = max(pass_medians["baseline"]) / min(pass_medians["baseline"])
            result = dict(case=case, fingerprint=next(iter(fingerprints)), medians_ms=medians,
                          speedup=medians["baseline"] / medians["candidate"], baseline_drift_ratio=drift,
                          stable=drift <= 1.10, samples_ms=samples)
            results.append(result)
            print(json.dumps(result), flush=True)
            (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        subprocess.run(["tools/bench_lock.sh", "release"], check=True)


if __name__ == "__main__":
    main()
