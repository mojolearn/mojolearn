# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane cluster-apple3 diagnostic: where a cluster board fit spends its wall
time between Python and the binding calls (cProfile, one warm fit per case).

    python bench/cluster_apple3_prof.py [--dataset taxi,higgs] [--only name,...] [--top 10]

Lines: `CPROF <dataset> <case> <rows> total=<s>` then `CPROF <dataset> <case>
<tottime s> <cumtime s> <calls> <function>` for the top functions by tottime.
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import x_cluster_speed as board  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi")
    ap.add_argument("--only", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import mojolearn as ml
    table = board.cases(ml, a.seed)
    names = [s for s in a.only.split(",") if s] or list(table)
    for ds in a.dataset.split(","):
        full = board.load(ds, max(r for r, _, _ in table.values()))
        for name in names:
            rows, build, _ = table[name]
            x = np.ascontiguousarray(full[:rows])
            try:
                build().fit(x)
                est = build()
                pr = cProfile.Profile()
                t0 = time.perf_counter()
                pr.enable()
                est.fit(x)
                pr.disable()
                total = time.perf_counter() - t0
                print(f"CPROF {ds} {name} {len(x)} total={total:.4f}", flush=True)
                st = pstats.Stats(pr)
                rowsv = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[: a.top]
                for (fn, line, func), (cc, nc, tt, ct, _) in rowsv:
                    if tt < 0.0005:
                        continue
                    print(f"CPROF {ds} {name} {tt:.4f} {ct:.4f} {nc} {os.path.basename(fn)}:{line}:{func}", flush=True)
            except Exception as e:
                print(f"CPROF {ds} {name} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
