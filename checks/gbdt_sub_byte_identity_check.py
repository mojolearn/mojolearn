#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate on DEVIATION 2600: IDENTICAL GBDT on the sub-byte histogram arms.

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
        python3 checks/gbdt_sub_byte_identity_check.py [--json out.json]

WHAT BROKE. The greedy searcher's binary, half-byte and hist_2 5-/6-bit
histogram kernels peel a partition's unaligned head and tail with a loop
copied from CatBoost, `for (idx = tid; idx < alignSize; idx += BlockSize)`.
A thread at or past `alignSize` makes no trip, so only part of the block
issues the write-turn syncs inside `AddPoint`. On a 32-lane column those
syncs are warp-local and nothing is wrong. On the 64-lane AMD column they
are threadgroup barriers (`sub_byte_lane_sync_for`), part of the block skips
them, and the fit moved between repeats in one process on the MI325X
(identity_break `ties`, taxi, Istella-S). `gbdt/methods/greedy_subsets_searcher/
kernel/lane_sync.mojo` carries the argument.

WHAT THIS CHECKS. Every cell is fitted TWICE in one process and must give
one hash (STABLE), and that hash must equal the NVIDIA H100 reference below.
A cell with no reference FAILS: a column that cannot be compared is not a
pass. The cell hash is `tools/identity_break.py`'s, so the `ties` references
are that tool's retained H100 cells.

ONE FIXTURE PER ARM THE FIX TOUCHES under IDENTICAL, because reach is per
branch (ENGINEERING_RULES 8). At border_count 128 an integer column with k
distinct values gets k-1 borders and k-1 folds (`gbdt/train.mojo`), and the
fold count picks the arm (`gbdt/gpu_data/grid_policy.mojo`, then the maxBins
ladder in `greedy_search_helper.mojo`):

    bin2   integers 0..1    1 fold    binary kernel
    ties   integers 0..5    5 folds   half-byte kernel (identity_break `ties`)
    bits5  integers 0..20  20 folds   hist_2 5-bit arm
    bits6  integers 0..40  40 folds   hist_2 6-bit arm

The 7-bit and 8-bit IDENTICAL arms issue no turn sync and every float
fixture in identity_break already covers them. Labels are identity_break's
rule (`labels_for`), so every vendor is handed the same bytes; the fixture
hashes are printed beside the verdicts.

SABOTAGE (rule 7). On the AMD column, restoring any one peel bound
(`while pe < PEEL_END` back to `while pe < ALIGN_SIZE`) in one kernel file
must make that file's fixture fail here. A green run on a 32-lane column
proves nothing about the defect; it is the reference side.
"""
import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("identity_break", ROOT / "tools" / "identity_break.py")
ib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ib)

LANES = ["gbdt-symmetric", "gbdt-depthwise", "gbdt-lossguide", "gbdt-rmse"]

#: fixture -> (distinct integer values, the arm it must reach)
WIDTHS = {
    "bin2": (2, "binary"),
    "ties": (6, "half-byte"),
    "bits5": (21, "hist_2 5-bit"),
    "bits6": (41, "hist_2 6-bit"),
}

#: NVIDIA H100 80GB, IDENTICAL, cell hash per lane. `ties` is the retained
#: identity_break set (bench/results/trees_identical/
#: mi325x_2026-09-11_taxi_istella/leg1/ib/h100_baseline.json, equal to
#: bench/results/identity_break/apple-m4.identical.json). `bin2`, `bits5`
#: and `bits6` were recorded on a RunPod H100 80GB HBM3 (pod
#: b8rlb4ggl30u4o, driver 580.126.09) at source 8020a0c9 on 2026-09-11,
#: both fits of every cell equal; the same build reproduced all 36 retained
#: identity_break gbdt cells.
REFERENCE = {
    "bin2": {
        "gbdt-symmetric": "9c9c2617381e9485",
        "gbdt-depthwise": "70c30f14ccb609d5",
        "gbdt-lossguide": "b502d21f4bb0a0c9",
        "gbdt-rmse": "c052ad32dd3c6c19",
    },
    "ties": {
        "gbdt-symmetric": "b32c469f812adf25",
        "gbdt-depthwise": "465cb10f303e005a",
        "gbdt-lossguide": "19d199f7bfa16c25",
        "gbdt-rmse": "88cba1574daa922e",
    },
    "bits5": {
        "gbdt-symmetric": "63fe2511e3730346",
        "gbdt-depthwise": "c8b57b95de2052f7",
        "gbdt-lossguide": "e69d957f80a7a128",
        "gbdt-rmse": "00856961d10e9f11",
    },
    "bits6": {
        "gbdt-symmetric": "c5507a51d232bd6f",
        "gbdt-depthwise": "caf1feee1cff20be",
        "gbdt-lossguide": "654bcd575d6b3c2e",
        "gbdt-rmse": "4ba67877a807c141",
    },
}


def fixture(name):
    if name == "ties":
        return ib.fixture("ties")
    k, _ = WIDTHS[name]
    rng = np.random.default_rng(0)
    X = rng.integers(0, k, size=(ib.N, ib.D)).astype(np.float32)
    return (X,) + ib.labels_for(X, 0)


def cell_hash(parts):
    blob = "|".join(f"{k}={v}" for k, v in sorted(parts.items())).encode()
    return ib._h(np.frombuffer(blob, dtype=np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--fixtures", default=",".join(WIDTHS))
    ap.add_argument("--lanes", default=",".join(LANES))
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    if args.repeats < 2:
        raise SystemExit("REFUSING: the run-to-run half of this check needs at least two fits")

    import mojolearn as ml
    mode = ml.numeric_mode()
    if mode != "identical":
        raise SystemExit(f"REFUSING: this gate is about IDENTICAL; loaded {mode!r}")

    fixtures = [f for f in args.fixtures.split(",") if f]
    lanes = [n for n in args.lanes.split(",") if n]
    print(f"# gbdt_sub_byte_identity_check  mode={mode}  {platform.platform()}")
    cells, bad = {}, 0
    for f in fixtures:
        X, yc, yr = fixture(f)
        k, arm = WIDTHS[f]
        fx = dict(X=ib._h(X), y_clf=ib._h(yc), y_reg=ib._h(yr))
        print(f"fixture {f}: {k} values per column -> {arm}; X={fx['X']} y_clf={fx['y_clf']} y_reg={fx['y_reg']}")
        for lane in lanes:
            hs = [cell_hash(ib.LANES[lane](ml, X, yc, yr)) for _ in range(args.repeats)]
            ref = REFERENCE.get(f, {}).get(lane)
            if len(set(hs)) != 1:
                verdict = "FAIL MOVED"
            elif ref is None:
                verdict = "FAIL NO REFERENCE"
            elif hs[0] != ref:
                verdict = "FAIL DIVERGENT"
            else:
                verdict = "PASS"
            bad += verdict != "PASS"
            cells[f"{lane}/{f}"] = dict(verdict=verdict, hashes=hs, reference=ref, arm=arm, fixture=fx)
            print(f"  {lane:<15} {f:<6} {verdict:<18} hashes={hs} reference={ref}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(mode=mode, platform=platform.platform(), repeats=args.repeats,
                           commit=os.environ.get("MOJOLEARN_COMMIT", ""), cells=cells), fh, indent=1)
    total = len(cells)
    print(f"gbdt_sub_byte_identity_check: {total - bad}/{total} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
