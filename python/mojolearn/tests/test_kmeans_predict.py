# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `KMeans.predict` (2026-09-15), mirroring cuML's
`KMeans.predict` (`kmeans.pyx:1071-1082`): the nearest fitted center under
the model's metric, the fit's own final assignment. On the GPU set it runs
`cluster/estimator.mojo::kmeans_predict`; on a CPU-only install the core
host binding's `host_kmeans_predict` (public inference; the fits here run
inside the private reference context, as the identity harness's do).

    cd python && python3 -m mojolearn.tests.test_kmeans_predict

Exit 2 naming the build script when the binding is unbuilt. Small fixtures,
one process: cheap on one core.
"""
import sys

import numpy as np

from mojolearn import KMeans
from mojolearn._cpu_reference import reference_training
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _x(seed=0, n=512, d=4):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.random((n, d), dtype=np.float32) * 4.0).astype(np.float32))


def _grid(seed=2):
    """Integer grid rows: many rows sit at the same distance from two or
    more centers once the centers are fitted on the grid."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 4, size=(512, 3)).astype(np.float32))


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _fit(**kw):
    x = kw.pop("x")
    with reference_training():
        return KMeans(**kw).fit(x)


def _bit_or_report(rep, arm, cond, what, detail=""):
    if mode() == "identical":
        rep.check(arm, cond, what, detail)
    else:
        rep.report_only(arm, cond, what)


def arm_equal(rep):
    for name, x in (("uniform [0, 4)", _x()), ("integer grid", _grid())):
        for metric in ("euclidean", "l2_sqrt_expanded"):
            m = _fit(x=x, n_clusters=5, random_state=3, metric=metric)
            p = m.predict(x)
            a = np.asarray(p)
            rep.check("EQUAL", a.dtype == np.int32 and a.shape == (len(x),), f"{metric} on {name}: int32, one label per row", (a.dtype, a.shape))
            rep.check("EQUAL", _bits(p) == _bits(m.labels_), f"{metric} on {name}: predict(X) is labels_ bit for bit",
                      int(np.sum(a != np.asarray(m.labels_))))
    w = np.linspace(0.5, 1.5, 512, dtype=np.float32)
    x = _x(4)
    with reference_training():
        m = KMeans(n_clusters=6, random_state=1).fit(x, sample_weight=w)
    rep.check("EQUAL", _bits(m.predict(x)) == _bits(m.labels_), "weighted fit: predict(X) is labels_")


def arm_ties(rep):
    # An exact tie in float32: the row (1, 0) is at squared distance 1 from
    # both (0, 0) and (2, 0), which the expanded form reproduces exactly
    # (1 + 0 - 0 and 1 + 4 - 4). The fused kernel keeps the LOWEST index.
    x = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [1.0, 5.0]], dtype=np.float32)
    for order, centers in (("(0,0) first", [[0.0, 0.0], [2.0, 0.0]]), ("(2,0) first", [[2.0, 0.0], [0.0, 0.0]])):
        m = KMeans(n_clusters=2)
        m.cluster_centers_ = np.array(centers, dtype=np.float32)
        p = np.asarray(m.predict(x))
        rep.check("TIES", int(p[0]) == 0 and int(p[3]) == 0, f"{order}: equidistant rows take center 0", p.tolist())
        rep.check("TIES", int(p[1]) == (0 if order == "(0,0) first" else 1) and int(p[2]) == (1 if order == "(0,0) first" else 0),
                  f"{order}: rows on a center take that center", p.tolist())


def arm_dtype(rep):
    x = _x(5)
    m = _fit(x=x, n_clusters=4, random_state=3)
    p32 = m.predict(x)
    x64 = x.astype(np.float64)
    rep.check("DTYPE", _bits(m.predict(x64)) == _bits(p32), "float64 input is cast to the centers' float32, the same labels")
    rep.check("DTYPE", _bits(m.predict(np.asfortranarray(x))) == _bits(p32), "a Fortran-order input gives the same labels")
    rep.check("DTYPE", _bits(m.predict(x.tolist())) == _bits(p32), "a nested list gives the same labels")


def arm_heldout(rep):
    x, xh = _x(6, n=2048), _x(7, n=777)
    m = _fit(x=x, n_clusters=8, random_state=3)
    whole = np.asarray(m.predict(xh))
    xd, c = xh.astype(np.float64), np.asarray(m.cluster_centers_, dtype=np.float64)
    d2 = ((xd[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    chosen, best = d2[np.arange(len(xd)), whole.astype(np.int64)], d2.min(1)
    bad = int(np.sum(chosen > best + 1e-4 * np.maximum(best, 1.0)))
    rep.check("HELDOUT", bad == 0, "every held-out label is the float64 argmin up to float32 round-off", f"{bad} rows")
    alone = [np.asarray(m.predict(xh[i:i + 1])) for i in range(16)]
    split = np.concatenate([np.asarray(m.predict(xh[a:b])) for a, b in ((0, 1), (1, 8), (8, len(xh)))])
    _bit_or_report(rep, "HELDOUT", all(_bits(alone[i]) == _bits(whole[i:i + 1]) for i in range(16)),
                   "each of the first 16 held-out rows alone is its whole-batch label")
    _bit_or_report(rep, "HELDOUT", _bits(split) == _bits(whole), "the split 1, 7, rest concatenates to the whole batch")


def arm_refuse(rep):
    x = _x(8, n=64)
    rep.raises("REFUSE", RuntimeError, "not fitted", "predict before fit", KMeans(n_clusters=2).predict, x)
    m = _fit(x=x, n_clusters=2, random_state=3)
    rep.raises("REFUSE", ValueError, "features", "a feature count other than the fit's", m.predict, x[:, :3])
    rep.raises("REFUSE", Exception, "", "a 1-D X", m.predict, x[0])
    rep.raises("REFUSE", ValueError, "X is empty", "an empty X", m.predict, np.zeros((0, 4), dtype=np.float32))
    cos = KMeans(n_clusters=2, metric="cosine")
    cos.cluster_centers_ = np.asarray(m.cluster_centers_)
    rep.raises("REFUSE", Exception, "L2Expanded or L2SqrtExpanded",
               "metric='cosine' predict is refused BY NAME, as its fit is", cos.predict, x)
    bad = KMeans(n_clusters=2, metric="manhattan")
    bad.cluster_centers_ = np.asarray(m.cluster_centers_)
    rep.raises("REFUSE", ValueError, "metric must be", "an unknown metric name", bad.predict, x)


def main(out=sys.stdout):
    bind_or_exit("_mojolearn", "build.sh")
    rep = Report("test_kmeans_predict")
    return run("test_kmeans_predict", [("EQUAL", arm_equal), ("TIES", arm_ties), ("DTYPE", arm_dtype),
                                       ("HELDOUT", arm_heldout), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
