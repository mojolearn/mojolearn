#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Locked IDENTICAL Lossguide fit timing: five warmups and ABBA order.

    python3 tools/gbdt_partition_cache_bench.py --out build/partition_cache_bench

Native train() timing includes quantization/upload; prediction and complete
model fingerprinting are outside the timer. Five trees, depth limit 10,
31 leaves, eight columns, RMSE. This is a local synthetic workload, not
CatBoost parity or a claim about other devices/losses.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.run_only:
        for arm in ("reference", "cached"):
            cmd = ["tools/with_build_lock.sh", "pixi", "run", "mojo", "build", "-I", ".",
                   "-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"]
            if arm == "reference":
                cmd += ["-D", "MOJOLEARN_GBDT_FULL_PARTITION_STATS=1"]
            cmd += ["checks/gbdt_partition_cache_bench.mojo", "-o", str(out / arm)]
            with (out / f"{arm}.build.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["tools/with_build_lock.sh", sys.executable, str(Path(__file__).resolve()),
                        "--out", str(out), "--run-only"], cwd=ROOT, check=True)
        return
    env = dict(os.environ, BENCH_LOCK_PID=str(os.getpid()))
    subprocess.run(["tools/bench_lock.sh", "acquire", "gbdt-partition-cache", "Lossguide IDENTICAL", "2 minutes"],
                   cwd=ROOT, env=env, check=True)
    records = []
    try:
        for rows in (65537, 262145):
            samples = {arm: [] for arm in ("reference", "cached")}
            hashes = set()
            for index, arm in enumerate(("reference", "cached", "cached", "reference")):
                run = subprocess.run([str(out / arm), str(rows)], cwd=ROOT, capture_output=True, text=True, check=True)
                (out / f"{rows}.{index}.{arm}.log").write_text(run.stdout + run.stderr)
                lines = run.stdout.splitlines()
                if "numeric_mode IDENTICAL" not in lines:
                    raise RuntimeError("wrong numeric mode")
                expected = f"incremental {arm == 'cached'} rows {rows}"
                if expected not in lines:
                    raise RuntimeError(f"wrong compiled cache configuration; expected {expected}")
                values = [float(line.split()[1]) for line in lines if line.startswith("fit_ms ")]
                model_hashes = [line.split()[1] for line in lines if line.startswith("fingerprint ")]
                warmups = [line for line in lines if line.startswith("warmup_ms ")]
                if len(values) != 3 or len(model_hashes) != 8 or len(warmups) != 5:
                    raise RuntimeError("incomplete timing/model evidence")
                hashes.update(model_hashes)
                samples[arm].extend(values)
            if len(hashes) != 1:
                raise RuntimeError(f"model bits changed at {rows} rows")
            before, after = (statistics.median(samples[arm]) for arm in ("reference", "cached"))
            record = dict(rows=rows, reference_ms=before, cached_ms=after,
                          speedup=before / after, samples=samples, fingerprint=next(iter(hashes)))
            records.append(record)
            print("RESULT", json.dumps(record), flush=True)
    finally:
        subprocess.run(["tools/bench_lock.sh", "release"], cwd=ROOT, env=env, check=True)
    (out / "summary.json").write_text(json.dumps(dict(platform=platform.platform(), records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
