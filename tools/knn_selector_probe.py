#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/knn-selector-speed: the per-cell probe of one binding arm.

One child process per (dataset, k, query rows) cell: fit (the resident
upload), one warmup call, `--calls` timed `kneighbors` calls from a host
array to host arrays, the sha256 of the distance and index bytes, and, on a
`-D MOJOLEARN_KNN_PHASE_TIMERS=1` build, the `KNN_PHASE_TIMERS` lines each
timed call printed (the class split; a timer build serializes the queue, so
its call time is not a price). The selector arm of a
`-D MOJOLEARN_KNN_SELECT_TRIAL=1` build comes from the environment
(`MOJOLEARN_KNN_SELECT`), which the caller sets. Test tooling: NumPy is
allowed here, the package stays NumPy-free.

    python3 tools/knn_selector_probe.py --arm NAME --json OUT.json \
        [--datasets istella,taxi] [--ks 1,10,32,64] [--rows 4000,1] [--calls 5]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

CHILD = r'''
import hashlib, json, sys, time
import numpy as np, mojolearn as ml
ds, k, rows, calls, data = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
with np.load("%s/knn-%s.npz" % (data, ds)) as z:
    index = np.ascontiguousarray(z["index"]); q = np.ascontiguousarray(z["queries"][:rows])
m = ml.NearestNeighbors(n_neighbors=k).fit(index)
m.kneighbors(q)
ms = []
for _ in range(calls):
    print("PROBE_CALL_BEGIN", flush=True)
    t0 = time.perf_counter(); d, i = m.kneighbors(q); ms.append((time.perf_counter() - t0) * 1000.0)
    print("PROBE_CALL_END", flush=True)
h = hashlib.sha256(np.ascontiguousarray(d).tobytes() + np.ascontiguousarray(i).tobytes()).hexdigest()
print("PROBE_MS", json.dumps(ms, separators=(",", ":")), "DIGEST", h, "VENDOR", ml.vendor(), flush=True)
'''


def phase_fields(line):
    parts = line.split()
    out = {}
    i = 1
    while i + 1 < len(parts):
        try:
            out[parts[i]] = float(parts[i + 1])
        except ValueError:
            out[parts[i]] = parts[i + 1]
        i += 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--data", default="/root/ctd-data")
    ap.add_argument("--datasets", default=os.environ.get("KSS_DATASETS", "istella,taxi"))
    ap.add_argument("--ks", default=os.environ.get("KSS_KS", "1,10,32,64"))
    ap.add_argument("--rows", default=os.environ.get("KSS_ROWS", "4000,1"))
    ap.add_argument("--calls", type=int, default=int(os.environ.get("KSS_CALLS", "5")))
    args = ap.parse_args()
    out = {"arm": args.arm, "select_env": os.environ.get("MOJOLEARN_KNN_SELECT", ""), "cells": []}
    for ds in args.datasets.split(","):
        for k in [int(v) for v in args.ks.split(",")]:
            for rows in [int(v) for v in args.rows.split(",")]:
                r = subprocess.run([sys.executable, "-c", CHILD, ds, str(k), str(rows), str(args.calls), args.data],
                                   capture_output=True, text=True)
                calls, cur, ms, digest = [], None, None, None
                for ln in r.stdout.splitlines():
                    if ln.startswith("PROBE_CALL_BEGIN"):
                        cur = []
                    elif ln.startswith("PROBE_CALL_END"):
                        calls.append(cur); cur = None
                    elif cur is not None and (ln.startswith("KNN_PHASE_TIMERS") or ln.startswith("KNN_ADMIT_RATE") or ln.startswith("KNN_SELECT_FALLBACK")):
                        cur.append(ln)
                    elif ln.startswith("PROBE_MS"):
                        parts = ln.split(); ms = json.loads(parts[1]); digest = parts[3]
                cell = {"dataset": ds, "k": k, "rows": rows, "ms": ms, "digest": digest, "rc": r.returncode,
                        "phase_lines": calls, "stderr_tail": r.stderr[-1500:]}
                line = "PROBE %s %s k=%d rows=%d rc=%d" % (args.arm, ds, k, rows, r.returncode)
                if ms:
                    cell["median_ms"] = statistics.median(ms); cell["min_ms"] = min(ms); cell["max_ms"] = max(ms)
                    line += " median=%.3f min=%.3f max=%.3f digest=%s" % (cell["median_ms"], cell["min_ms"], cell["max_ms"], digest[:16])
                # the class split: the median over the timed calls of each field of the tiled arm's line
                tiled = [phase_fields(l) for c in calls for l in c if l.startswith("KNN_PHASE_TIMERS distance_ms")]
                if tiled:
                    for f in ("distance_ms", "select_ms", "merge_ms"):
                        cell[f] = statistics.median([t[f] for t in tiled])
                    line += " distance=%.2f select=%.2f merge=%.2f" % (cell["distance_ms"], cell["select_ms"], cell["merge_ms"])
                fb = [l for c in calls for l in c if l.startswith("KNN_SELECT_FALLBACK")]
                if fb:
                    line += " | " + fb[-1]
                print(line, flush=True)
                if r.returncode != 0:
                    print("   stderr:", r.stderr[-600:], flush=True)
                out["cells"].append(cell)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
