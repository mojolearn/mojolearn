#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/knn-selector-speed: the interleaved kNN arm race over k.

Arms are binding trees `/root/t-<arm>` (tools/knn_selector_body.sh). Per
outer round the arm order is rotated; each arm runs ONE process per dataset
per round: load the block, fit (the resident upload), then for every
(k, query rows) cell one warmup call and `--rounds` timed `kneighbors` calls,
host array in and host arrays out. Every cell's sha256 over the distance and
index bytes must be equal across arms, rounds and calls, which at 4,000
queries x k 64 is a byte-for-byte compare of 256,000 (distance, index) pairs
per dataset against the base arm's selector.

A cell's spread is max/min over the arm's per-round medians; an arm outside
1.10 is marked `u` and no ratio is quoted from it. The paired ratio is the
median over rounds of (arm round median / first arm's round median).
Test tooling: NumPy is allowed here, the package stays NumPy-free.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

WORKER = r'''
import hashlib, json, sys, time
import numpy as np, mojolearn as ml
ds, ks, rows_list, rounds, data = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
with np.load("%s/knn-%s.npz" % (data, ds)) as z:
    index = np.ascontiguousarray(z["index"]); queries = np.ascontiguousarray(z["queries"])
out = []
for k in [int(v) for v in ks.split(",")]:
    m = ml.NearestNeighbors(n_neighbors=k).fit(index)
    for rows in [int(v) for v in rows_list.split(",")]:
        q = np.ascontiguousarray(queries[:rows])
        m.kneighbors(q)
        ms, digests = [], []
        for _ in range(rounds):
            t0 = time.perf_counter(); d, i = m.kneighbors(q); ms.append((time.perf_counter() - t0) * 1000.0)
            digests.append(hashlib.sha256(np.ascontiguousarray(d).tobytes() + np.ascontiguousarray(i).tobytes()).hexdigest())
        out.append({"k": k, "rows": rows, "ms": ms, "digests": digests})
print("WORKER_JSON " + json.dumps({"vendor": ml.vendor(), "mode": ml.numeric_mode(), "cells": out}, separators=(",", ":")), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma list; the first is the reference of the paired ratio")
    ap.add_argument("--ks", default="32,64")
    ap.add_argument("--rows", default="4000")
    ap.add_argument("--datasets", default="istella,taxi")
    ap.add_argument("--data", default="/root/ctd-data")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    arms = args.arms.split(",")
    os.makedirs(args.out, exist_ok=True)
    raw = []  # (round, arm, dataset, worker json)
    for rnd in range(args.outer):
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
        for ds in args.datasets.split(","):
            for arm in order:
                env = dict(os.environ)
                env["PYTHONPATH"] = "/root/t-%s/python:/root/t-%s/tools" % (arm, arm)
                env["MOJOLEARN_NUMERIC_MODE"] = "identical"
                r = subprocess.run([args.python, "-c", WORKER, ds, args.ks, args.rows, str(args.rounds), args.data],
                                   capture_output=True, text=True, env=env, cwd="/root/t-%s" % arm)
                got = None
                for ln in r.stdout.splitlines():
                    if ln.startswith("WORKER_JSON "):
                        got = json.loads(ln[len("WORKER_JSON "):])
                if got is None:
                    print("RACE FAILED round=%d arm=%s ds=%s rc=%d\n%s" % (rnd, arm, ds, r.returncode, r.stderr[-1500:]), flush=True)
                    sys.exit(3)
                raw.append({"round": rnd, "arm": arm, "dataset": ds, "worker": got})
                print("round %d %s %s %s" % (rnd, ds, arm, " ".join("k%d/r%d=%.2f" % (c["k"], c["rows"], statistics.median(c["ms"])) for c in got["cells"])), flush=True)
    with open(os.path.join(args.out, "race_raw.json"), "w") as f:
        json.dump(raw, f, indent=1)
    rows_out = []
    bad = 0
    for ds in args.datasets.split(","):
        for k in [int(v) for v in args.ks.split(",")]:
            for rows in [int(v) for v in args.rows.split(",")]:
                per_arm, digests = {}, set()
                for arm in arms:
                    meds, allms = [], []
                    for rec in raw:
                        if rec["arm"] == arm and rec["dataset"] == ds:
                            for c in rec["worker"]["cells"]:
                                if c["k"] == k and c["rows"] == rows:
                                    meds.append(statistics.median(c["ms"])); allms.extend(c["ms"]); digests.update(c["digests"])
                    per_arm[arm] = (meds, allms)
                equal = len(digests) == 1
                bad += 0 if equal else 1
                ref = per_arm[arms[0]][0]
                for arm in arms:
                    meds, allms = per_arm[arm]
                    spread = max(meds) / min(meds)
                    paired = statistics.median([a / b for a, b in zip(meds, ref)])
                    rows_out.append({"dataset": ds, "k": k, "rows": rows, "arm": arm, "median_ms": statistics.median(meds),
                                     "min_ms": min(allms), "max_ms": max(allms), "round_medians": meds,
                                     "spread_round_medians": spread, "spread_all_calls": max(allms) / min(allms),
                                     "gate": "ok" if spread <= 1.10 else "u", "paired_ratio_vs_first": paired,
                                     "digests_equal": equal, "digest": sorted(digests)[0][:16]})
    with open(os.path.join(args.out, "race_summary.json"), "w") as f:
        json.dump(rows_out, f, indent=1)
    with open(os.path.join(args.out, "race_summary.tsv"), "w") as f:
        f.write("dataset\tk\trows\tarm\tmedian_ms\tmin_ms\tmax_ms\tspread_rounds\tspread_calls\tgate\tpaired_ratio\tdigests\n")
        for r in rows_out:
            f.write("%s\t%d\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%.4f\t%s\n" % (
                r["dataset"], r["k"], r["rows"], r["arm"], r["median_ms"], r["min_ms"], r["max_ms"],
                r["spread_round_medians"], r["spread_all_calls"], r["gate"], r["paired_ratio_vs_first"],
                "equal" if r["digests_equal"] else "DIFFERENT"))
    print(open(os.path.join(args.out, "race_summary.tsv")).read(), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
