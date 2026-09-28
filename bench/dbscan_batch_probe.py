# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DBSCAN batch invariance probe (lane cluster-apple2).

The board's DBSCAN case (taxi 100k, eps 0.5, min_samples 10) at several
`max_mbytes_per_batch` budgets. The labels must not depend on the budget; the
digest is the board's (`bench/x_cluster_speed.py`: labels, then b"-" for the
absent centers), so the default line is comparable to its records
(one batch 9c8ea257cb04e118).

    python bench/dbscan_batch_probe.py [--budgets 0,38000,20000,8000] [--dataset taxi]

Lines: `DBPROBE <dataset> budget=<MB> <seconds> <digest> <n_clusters> <n_noise>`.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from x_cluster_speed import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="0,38000,20000,8000")
    ap.add_argument("--dataset", default="taxi")
    ap.add_argument("--rows", type=int, default=100_000)
    a = ap.parse_args()
    import mojolearn as ml
    for ds in a.dataset.split(","):
        x = load(ds, a.rows)
        for b in [int(s) for s in a.budgets.split(",") if s]:
            try:
                est = ml.DBSCAN(eps=0.5, min_samples=10, max_mbytes_per_batch=(b or None))
                t0 = time.perf_counter()
                est.fit(x)
                dt = time.perf_counter() - t0
                lab = np.asarray(est.labels_)
                h = hashlib.sha256()
                h.update(np.ascontiguousarray(lab).tobytes())
                h.update(b"-")
                print(f"DBPROBE {ds} budget={b} {dt:.4f} {h.hexdigest()[:16]} "
                      f"{len(set(lab.tolist()) - {-1})} {int((lab < 0).sum())}", flush=True)
            except Exception as e:
                print(f"DBPROBE {ds} budget={b} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
