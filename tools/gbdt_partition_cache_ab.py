#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""A/B full-model and work-count gate for unchanged-leaf partition stats.

    python3 tools/gbdt_partition_cache_ab.py --out build/partition_cache_ab

Builds/runs hold the repository build lock. Tests all three numeric modes,
then checks that omitting right-child invalidation changes an IDENTICAL model.
This is a correctness/work-count gate, not a wall-time benchmark.
Use checks/gbdt_partition_cache_bench.mojo for isolated fit timing.
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference_models = None
    for mode in ("identical", "deterministic", "fast"):
        results = {}
        for arm in ("reference", "cached"):
            name = f"{mode}-{arm}"
            defines = ["MOJOLEARN_GBDT_PART_STATS_WORK"]
            if mode != "fast":
                defines.append("MOJOLEARN_NUMERIC_" + mode.upper())
            if arm == "reference":
                defines.append("MOJOLEARN_GBDT_FULL_PARTITION_STATS")
            build_and_run(out, name, defines)
            results[arm] = parse(out / f"{name}.log", mode)
        if results["reference"][0] != results["cached"][0]:
            raise RuntimeError(f"{mode}: full-model fingerprints changed")
        before = sum(results["reference"][1])
        after = sum(results["cached"][1])
        if mode == "identical":
            if (len(results["reference"][1]) != len(results["cached"][1])
                    or any(a > b for a, b in zip(results["cached"][1], results["reference"][1]))
                    or not 0 < after < before):
                raise RuntimeError("IDENTICAL candidate did not reduce row visits without per-tree regressions")
            reference_models = results["reference"][0]
        elif results["reference"][1] != results["cached"][1]:
            raise RuntimeError(f"{mode}: inert cache changed work schedule")
        print(f"PASS {mode}: {len(results['cached'][0])} model fingerprints; "
              f"partition row visits {before} -> {after}", flush=True)
    name = "identical-sabotage"
    build_and_run(out, name, ["MOJOLEARN_NUMERIC_IDENTICAL",
                             "MOJOLEARN_GBDT_PART_STATS_WORK",
                             "MOJOLEARN_GBDT_SAB_SKIP_RIGHT_PART_STATS"], allow_failure=True)
    sabotaged = (out / f"{name}.log").read_text()
    lines = [line for line in sabotaged.splitlines() if line.startswith("fingerprint ")]
    # Require an actually observed model change, not a build/runtime failure.
    if not any(line != ref for line, ref in zip(lines, reference_models)):
        raise RuntimeError("negative control did not move a model fingerprint")
    print("PASS negative control: missing right-child invalidation moves model bits", flush=True)


def build_and_run(out, name, defines, allow_failure=False):
    binary = out / name
    cmd = ["tools/with_build_lock.sh", "pixi", "run", "mojo", "build", "-I", "."]
    for define in defines:
        cmd += ["-D", define + "=1"]
    cmd += ["checks/gbdt_partition_cache_check.mojo", "-o", str(binary)]
    with (out / f"{name}.build.log").open("w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / f"{name}.log").open("w") as log:
        subprocess.run(["tools/with_build_lock.sh", str(binary)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=not allow_failure)


def parse(path, mode):
    lines = path.read_text().splitlines()
    if f"numeric_mode {mode.upper()}" not in lines:
        raise RuntimeError(f"{path}: compiled numeric mode mismatch")
    models = [line for line in lines if line.startswith("fingerprint ")]
    work = [int(line.split()[-1]) for line in lines if line.startswith("part_stats_work ")]
    if len(models) != 14 or not work:
        raise RuntimeError(f"{path}: incomplete model/work output")
    return models, work


if __name__ == "__main__":
    main()
