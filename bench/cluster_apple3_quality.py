# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane cluster-apple3: the paired quality check of a FAST change.

Each case of the cluster board (bench/x_cluster_speed.py) is fitted twice on
the same rows and seed in one process: on the tier of the environment (FAST)
and on the IDENTICAL reference (`numeric_mode='identical'`). Printed per case:
the board's quality number of each (its own direction: inertia lower, mean
log-likelihood and silhouette higher), whether the labels are equal, the
adjusted Rand index of the two labelings, and for the agglomerative cases
whether the merge tree is equal and how far the merge values are apart.

    python bench/cluster_apple3_quality.py [--dataset taxi,higgs] [--only name,...] [--seeds 0]

Lines: `XCQUAL <dataset> <case> seed=<s> rows=<n> fast_q=<> ident_q=<> labels_equal=<0|1> ari=<> [tree_equal=<0|1> max_rel_dist=<>]`.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import x_cluster_speed as board  # noqa: E402


def ari(a, b):
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    na, nb = int(ai.max()) + 1, int(bi.max()) + 1
    cont = np.bincount(ai * nb + bi, minlength=na * nb).astype(np.float64)
    comb = lambda v: v * (v - 1.0) / 2.0
    s = comb(cont).sum()
    sa = comb(np.bincount(ai).astype(np.float64)).sum()
    sb = comb(np.bincount(bi).astype(np.float64)).sum()
    tot = comb(float(len(a)))
    exp = sa * sb / tot if tot > 0 else 0.0
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else float((s - exp) / (mx - exp))


def labels_of(est, x):
    return board._np(est.labels_) if hasattr(est, "labels_") else board._np(est.predict(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    import mojolearn as ml
    for seed in [int(s) for s in a.seeds.split(",")]:
        table = board.cases(ml, seed)
        names = [s for s in a.only.split(",") if s] or list(table)
        for ds in a.dataset.split(","):
            full = board.load(ds, max(table[nm][0] for nm in names))
            for name in names:
                rows, build, qual = table[name]
                x = np.ascontiguousarray(full[:rows])
                try:
                    agglo = name.startswith("agglomerative")
                    fast = build()
                    if agglo:
                        fast.compute_distances = True
                    fast.fit(x)
                    ref = build()
                    ref.numeric_mode = "identical"
                    if agglo:
                        ref.compute_distances = True
                    ref.fit(x)
                    lf, lr = labels_of(fast, x), labels_of(ref, x)
                    line = (f"XCQUAL {ds} {name} seed={seed} rows={len(x)} fast_q={qual(x, fast):.8g} "
                            f"ident_q={qual(x, ref):.8g} labels_equal={int(np.array_equal(lf, lr))} "
                            f"ari={ari(lf, lr):.8f}")
                    if agglo:
                        cf, cr = board._np(fast.children_), board._np(ref.children_)
                        df = board._np(fast.distances_).astype(np.float64)
                        dr = board._np(ref.distances_).astype(np.float64)
                        rel = np.max(np.abs(df - dr) / np.maximum(np.abs(dr), 1e-30))
                        line += f" tree_equal={int(np.array_equal(cf, cr))} max_rel_dist={rel:.3e}"
                    print(line, flush=True)
                except Exception as e:  # one broken case never hides the others
                    print(f"XCQUAL {ds} {name} seed={seed} ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
