# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `KMeans.transform` (2026-09-15). The reference is cuML's
`KMeans.transform` (`kmeans.pyx:1084`), cuVS `kmeans_transform`
(`detail/kmeans.cuh:1178-1219`): the distance from every row to every
center under the model's metric, squared for `L2Expanded`, rooted for
`L2SqrtExpanded`. On the GPU set it runs
`cluster/estimator.mojo::kmeans_transform`; on a CPU-only install the core
host binding's `host_kmeans_transform` (public inference; the fits here run
inside the private reference context, as the identity harness's do).

    cd python && python3 -m mojolearn.tests.test_kmeans_transform

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


def arm_shape(rep):
    x = _x()
    m = _fit(x=x, n_clusters=5, random_state=3)
    t = np.asarray(m.transform(x))
    rep.check("SHAPE", t.dtype == np.float32 and t.shape == (len(x), 5), "float32, one row per sample, one column per center",
              (t.dtype, t.shape))
    rep.check("SHAPE", bool(np.all(t >= 0.0)) and bool(np.all(np.isfinite(t))), "every distance is finite and non-negative")
    xh = _x(11, n=7)
    rep.check("SHAPE", _bits(m.transform(xh.astype(np.float64))) == _bits(m.transform(xh)),
              "float64 input is cast to the centers' float32, the same distances")
    rep.check("SHAPE", _bits(m.transform(np.asfortranarray(xh))) == _bits(m.transform(xh)), "a Fortran-order input gives the same distances")
    with reference_training():
        ft = KMeans(n_clusters=5, random_state=3).fit_transform(x)
    rep.check("SHAPE", _bits(ft) == _bits(t), "fit_transform is fit then transform")


def arm_min(rep):
    for metric in ("euclidean", "l2_sqrt_expanded"):
        x = _x(1)
        m = _fit(x=x, n_clusters=6, random_state=3, metric=metric)
        t = np.asarray(m.transform(x))
        lab = np.asarray(m.labels_, dtype=np.int64)
        at = t[np.arange(len(x)), lab]
        rep.check("MIN", _bits(at) == _bits(t.min(axis=1)), f"{metric}: transform at labels_ is each row's minimum bit for bit",
                  int(np.sum(at != t.min(axis=1))))
        p = np.asarray(m.predict(x), dtype=np.int64)
        rep.check("MIN", _bits(t[np.arange(len(x)), p]) == _bits(t.min(axis=1)), f"{metric}: transform at predict(X) is the row minimum")


def arm_reference(rep):
    x, xh = _x(2, n=2048), _x(3, n=333)
    m = _fit(x=x, n_clusters=8, random_state=3)
    t = np.asarray(m.transform(xh), dtype=np.float64)
    c = np.asarray(m.cluster_centers_, dtype=np.float64)
    d2 = ((xh.astype(np.float64)[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    bad = int(np.sum(np.abs(t - d2) > 1e-4 * np.maximum(d2, 1.0)))
    rep.check("REFERENCE", bad == 0, "euclidean: squared distances equal the float64 reference up to float32 round-off", f"{bad} cells")
    ms = KMeans(n_clusters=8, metric="l2_sqrt_expanded")
    ms.cluster_centers_ = np.asarray(m.cluster_centers_)
    ts = np.asarray(ms.transform(xh))
    root = np.sqrt(np.asarray(m.transform(xh)))
    rep.check("REFERENCE", _bits(ts) == _bits(root),
              "l2_sqrt_expanded is the correctly rounded root of the euclidean cell, bit for bit",
              int(np.sum(ts != root)))


def arm_exact(rep):
    # (1, 0) is at squared distance 1 from (0, 0) and from (2, 0), and the
    # expanded form reproduces it exactly; a row on a center clamps to 0.
    x = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [1.0, 5.0]], dtype=np.float32)
    m = KMeans(n_clusters=2)
    m.cluster_centers_ = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    t = np.asarray(m.transform(x))
    want = np.array([[1.0, 1.0], [0.0, 4.0], [4.0, 0.0], [26.0, 26.0]], dtype=np.float32)
    rep.check("EXACT", _bits(t) == _bits(want), "small integer rows give the exact squared distances", t.tolist())


def arm_heldout(rep):
    x, xh = _x(6, n=2048), _x(7, n=777)
    m = _fit(x=x, n_clusters=8, random_state=3)
    whole = np.asarray(m.transform(xh))
    alone = [np.asarray(m.transform(xh[i:i + 1])) for i in range(16)]
    split = np.concatenate([np.asarray(m.transform(xh[a:b])) for a, b in ((0, 1), (1, 8), (8, len(xh)))])
    _bit_or_report(rep, "HELDOUT", all(_bits(alone[i]) == _bits(whole[i:i + 1]) for i in range(16)),
                   "each of the first 16 held-out rows alone is its whole-batch row")
    _bit_or_report(rep, "HELDOUT", _bits(split) == _bits(whole), "the split 1, 7, rest concatenates to the whole batch")


def arm_refuse(rep):
    x = _x(8, n=64)
    rep.raises("REFUSE", RuntimeError, "not fitted", "transform before fit", KMeans(n_clusters=2).transform, x)
    m = _fit(x=x, n_clusters=2, random_state=3)
    rep.raises("REFUSE", ValueError, "features", "a feature count other than the fit's", m.transform, x[:, :3])
    rep.raises("REFUSE", Exception, "", "a 1-D X", m.transform, x[0])
    rep.raises("REFUSE", ValueError, "X is empty", "an empty X", m.transform, np.zeros((0, 4), dtype=np.float32))
    # cosine was deleted 2026-09-18; an unsupported metric CODE must still be
    # refused by the kernel rather than accepted and ignored.
    bad = KMeans(n_clusters=2, metric=2)
    bad.cluster_centers_ = np.asarray(m.cluster_centers_)
    rep.raises("REFUSE", Exception, "L2Expanded (0) and L2SqrtExpanded",
               "an unsupported metric code transform is refused BY NAME, as its fit is", bad.transform, x)


def main(out=sys.stdout):
    bind_or_exit("_mojolearn", "build.sh")
    rep = Report("test_kmeans_transform")
    return run("test_kmeans_transform", [("SHAPE", arm_shape), ("MIN", arm_min), ("REFERENCE", arm_reference),
                                         ("EXACT", arm_exact), ("HELDOUT", arm_heldout), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
