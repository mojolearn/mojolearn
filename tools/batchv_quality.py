# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""batchv_quality.py: quality-only check for the lane/apple-fast-batchv
defines (SI_ONEPASS, KDE_DIMTILE; PTIMPUTE_ALL deleted 2026-10-09). It times nothing.

  python tools/batchv_quality.py dump prep|kde <out.npz>   fit on seeded data with the installed .so
  python tools/batchv_quality.py cmp <off.npz> <on.npz> <label>   one BATCHV-Q line per output

Fixtures are elementwise float64 (no BLAS), seed 0: PowerTransformer
(yeo-johnson, standardize) on 100k x 220 and 100k x 11, SimpleImputer mean
and median on the same with 10% NaN cells, GaussianNB theta_ / var_ (every col_stats
stage moves under SI_ONEPASS), KernelDensity gaussian / exponential bw 1.0
score_samples, 20k train x 2k query at d = 220 and d = 11."""
import sys

import numpy as np


def _data(n, d, seed):
    r = np.random.default_rng(seed)
    k = np.arange(d) % 4
    X = np.empty((n, d), dtype=np.float64)
    X[:, k == 0] = r.standard_normal((n, int((k == 0).sum())))
    X[:, k == 1] = r.lognormal(0.0, 0.75, (n, int((k == 1).sum())))
    X[:, k == 2] = r.exponential(2.0, (n, int((k == 2).sum()))) - 1.0
    X[:, k == 3] = r.uniform(-3.0, 5.0, (n, int((k == 3).sum())))
    return X.astype(np.float32)


def _np(v):
    try:
        return np.asarray(v, dtype=np.float64)
    except Exception:
        return np.asarray(v.tolist(), dtype=np.float64)


def dump_prep(out):
    from mojolearn._expansion_prep import GaussianNB, PowerTransformer, SimpleImputer
    res = {}
    for d in (220, 11):
        X = _data(100_000, d, d)
        pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)
        res["pt%d_lambdas" % d] = _np(pt.lambdas_)
        res["pt%d_transform" % d] = _np(pt.transform(X))
        nb = GaussianNB().fit(X, (X[:, 0] > np.median(X[:, 0])).astype(np.int32))
        res["nb%d_theta_mean" % d] = _np(nb.theta_)
        res["nb%d_var_scale" % d] = _np(nb.var_)
        Xn = X.copy()
        Xn[np.random.default_rng(7).random(X.shape) < 0.1] = np.nan
        for strat in ("mean", "median"):
            si = SimpleImputer(strategy=strat).fit(Xn)
            res["si%d_%s_stats" % (d, strat)] = _np(si.statistics_)
            res["si%d_%s_out" % (d, strat)] = _np(si.transform(Xn))
    np.savez(out, **res)


def dump_sk(out):
    """sklearn PowerTransformer (float64) on the same fixtures: the reference
    both arms' lambdas are measured against (`cmp` with label sk-*)."""
    from sklearn.preprocessing import PowerTransformer as SkPT
    res = {}
    for d in (220, 11):
        X = _data(100_000, d, d).astype(np.float64)
        pt = SkPT(method="yeo-johnson", standardize=True).fit(X)
        res["pt%d_lambdas" % d] = pt.lambdas_
        res["pt%d_transform" % d] = pt.transform(X)
    np.savez(out, **res)


def dump_kde(out):
    from mojolearn.density import KernelDensity
    res = {}
    for d in (220, 11):
        X = _data(22_000, d, 100 + d)
        X = (X - X.mean(0)) / X.std(0)
        for kern in ("gaussian", "exponential"):
            kd = KernelDensity(kernel=kern, bandwidth=1.0).fit(X[:20_000])
            res["kde%d_%s" % (d, kern)] = _np(kd.score_samples(X[20_000:]))
    np.savez(out, **res)


def cmp(a_path, b_path, label):
    a, b = np.load(a_path), np.load(b_path)
    worst = {}
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print("BATCHV-Q %s %s SHAPE %s vs %s" % (label, k, x.shape, y.shape))
            worst[k] = np.inf
            continue
        fin = np.isfinite(x) & np.isfinite(y)
        nanmis = int((np.isfinite(x) != np.isfinite(y)).sum())
        diff = np.abs(x[fin] - y[fin])
        mad = float(diff.max()) if diff.size else 0.0
        if k.endswith("lambdas") or k.endswith("_mean") or k.endswith("_scale") or k.endswith("_stats"):
            # elementwise relative
            den = np.maximum(np.abs(x[fin]), 1e-12)
            rel = float((diff / den).max()) if diff.size else 0.0
            kind = "elem_rel"
        else:
            scale = float(np.abs(x[fin]).max()) if diff.size else 1.0
            rel = mad / max(scale, 1e-30)
            kind = "rel_to_scale"
        exact = bool(np.array_equal(x, y, equal_nan=True))
        worst[k] = rel if not nanmis else np.inf
        print("BATCHV-Q %s %s max_abs=%.3e %s=%.3e exact=%s nan_mismatch=%d" % (
            label, k, mad, kind, rel, exact, nanmis))
    m = max(worst.values()) if worst else 0.0
    print("BATCHV-Q-SUMMARY %s worst_rel=%.3e %s" % (label, m, "PASS" if m < 1e-4 else "FAIL"))


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        {"prep": dump_prep, "kde": dump_kde, "sk": dump_sk}[sys.argv[2]](sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3], sys.argv[4])
