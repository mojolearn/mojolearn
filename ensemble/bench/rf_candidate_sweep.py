#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Build isolated RF candidate binaries, then compare time AND full-model bits.

    python ensemble/bench/rf_candidate_sweep.py build --out build/rf_candidates
    python ensemble/bench/quiet_window.py run -- \
        python ensemble/bench/rf_candidate_sweep.py run --out build/rf_candidates

Builds hold the repository build lock. Timings must run alone, after all builds;
quiet_window supplies the process/lock gate and checks the emitted canaries.
Every timed fit emits a hash of all forest node fields and leaf values. Hashes
must agree with the baseline separately in each numeric mode, including repeat
fits. This validates these datasets; it does not replace the RF identity suite.
The first fit per process is discarded from timing, but still identity-checked.
No tuning option changes the shipped defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "baseline": (),
    "sorted": ("MOJOLEARN_2010_ROWS_SORTED",),
    "items4": ("MOJOLEARN_2011_HIST_ITEMS4",),
    "copies4": ("MOJOLEARN_2012_SMEM_COPIES4",),
    "combined": ("MOJOLEARN_2011_HIST_ITEMS4", "MOJOLEARN_2012_SMEM_COPIES4"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("fast", "identical"), default=["fast", "identical"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--cols", type=int, default=50)
    parser.add_argument("--tasks", nargs="+", choices=("clf", "reg"), default=["clf", "reg"])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if "baseline" not in args.variants:
        parser.error("--variants must include baseline for the identity and timing comparisons")
    if args.rows < 128 or args.cols < 3 or args.rounds < 1:
        parser.error("rows >= 128, cols >= 3, rounds >= 1 required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = [(m, v) for m in args.modes for v in args.variants]
    if args.action == "build":
        for mode, variant in entries:
            key = f"{mode}_{variant}"
            defines = list(VARIANTS[variant])
            if mode == "identical":
                defines.append("MOJOLEARN_NUMERIC_IDENTICAL")
            cmd = ["tools/with_build_lock.sh", "pixi", "run", "mojo", "build", "-I", "."]
            for define in defines:
                cmd += ["-D", define + "=1"]
            cmd += ["ensemble/bench/rf_bench.mojo", "-o", str(out / key)]
            with (out / f"{key}.build.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("BUILT", key, flush=True)
        return
    # A failed rerun must not leave a previous successful summary in place.
    (out / "summary.json").unlink(missing_ok=True)
    binary_sha256 = {f"{m}_{v}": hashlib.sha256((out / f"{m}_{v}").read_bytes()).hexdigest()
                     for m, v in entries}
    samples: dict[str, list[float]] = {}
    hashes: dict[str, set[str]] = {}
    canaries: dict[str, list[float]] = {}
    for round_index in range(args.rounds):
        # Reverse the order on alternate passes to reduce order bias.
        for mode, variant in entries[::1 if round_index % 2 == 0 else -1]:
            binary_key = f"{mode}_{variant}"
            for task in args.tasks:
                key = f"{binary_key}_{task}"
                cmd = [str(out / binary_key), "--tune", task, str(args.rows), str(args.cols)]
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                (out / f"{key}.{round_index}.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    sys.stderr.write(result.stderr)
                    raise RuntimeError(f"{key} exited {result.returncode}")
                values, model_hashes = [], []
                expected = ["CONFIG", "1" if mode == "identical" else "0", "sorted",
                            "True" if "MOJOLEARN_2010_ROWS_SORTED" in VARIANTS[variant] else "False",
                            "items", "4" if "MOJOLEARN_2011_HIST_ITEMS4" in VARIANTS[variant] else "1",
                            "copies", "4" if "MOJOLEARN_2012_SMEM_COPIES4" in VARIANTS[variant] else "1"]
                config = []
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "CONFIG":
                        config = parts
                    elif parts[0] == "ARM":
                        values.append(float(parts[2]))
                    elif parts[0] == "MODEL":
                        model_hashes.append(parts[2])
                    elif parts[0] == "CANARY":
                        if parts[1] != "warmup":
                            canaries.setdefault(binary_key, []).append(float(parts[2]))
                        parts[1] = binary_key + ":" + parts[1]
                        line = " ".join(parts)
                    print(line, flush=True)
                if config != expected:
                    raise RuntimeError(f"{key}: compiled config {config!r} != requested {expected!r}")
                if len(values) != 5 or len(model_hashes) != 5:
                    raise RuntimeError(f"{key}: expected five timings and five full-model hashes")
                samples.setdefault(key, []).extend(values[1:])
                hashes.setdefault(key, set()).update(model_hashes)
    failures = []
    records = []
    for mode, variant in entries:
        binary_key = f"{mode}_{variant}"
        for task in args.tasks:
            key, baseline = f"{binary_key}_{task}", f"{mode}_baseline_{task}"
            same = len(hashes[key]) == 1 and hashes[key] == hashes[baseline]
            if not same:
                failures.append(key)
            noise = max(max(canaries[k]) / min(canaries[k])
                        for k in (binary_key, f"{mode}_baseline"))
            ratio = statistics.median(samples[baseline]) / statistics.median(samples[key])
            records.append(dict(mode=mode, variant=variant, task=task,
                                median_ms=statistics.median(samples[key]), samples_ms=samples[key],
                                model_hashes=sorted(hashes[key]), identity_pass=same,
                                baseline_over_candidate=ratio, canary_spread=noise,
                                exceeds_canary_noise=ratio > noise))
    summary = dict(platform=platform.platform(), binary_sha256=binary_sha256,
                   rows=args.rows, cols=args.cols, rounds=args.rounds, records=records,
                   identity_failures=failures, timing_requires_clean_quiet_window=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for record in records:
        print("RESULT", json.dumps({k: v for k, v in record.items() if k != "samples_ms"}), flush=True)
    if failures:
        raise RuntimeError("full-model identity mismatch: " + ", ".join(failures))


if __name__ == "__main__":
    main()
