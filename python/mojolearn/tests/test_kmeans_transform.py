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

The numerical arms (finite non-negative cells, fit_transform, the row
minimum at labels_, the float64 reference and the rooted metric, exact small
cells, rows alone and split batches) are Mojo checks:
`checks/native_tests/cluster_kmeans_query_*` (`pixi run
check-native-cluster-kmeans-query-cpu` / `-gpu`). This gate keeps the output
type, the input conversions and the refusals.

Exit 2 naming the build script when the binding is unbuilt. Small fixtures,
one process: cheap on one core.
"""
import sys

import numpy as np

from mojolearn import KMeans
from mojolearn._cpu_reference import reference_training
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run


def _x(seed=0, n=512, d=4):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.random((n, d), dtype=np.float32) * 4.0).astype(np.float32))


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _fit(**kw):
    x = kw.pop("x")
    with reference_training():
        return KMeans(**kw).fit(x)


def arm_shape(rep):
    x = _x()
    m = _fit(x=x, n_clusters=5, random_state=3)
    t = np.asarray(m.transform(x))
    rep.check("SHAPE", t.dtype == np.float32 and t.shape == (len(x), 5), "float32, one row per sample, one column per center",
              (t.dtype, t.shape))
    xh = _x(11, n=7)
    rep.check("SHAPE", _bits(m.transform(xh.astype(np.float64))) == _bits(m.transform(xh)),
              "float64 input is cast to the centers' float32, the same distances")
    rep.check("SHAPE", _bits(m.transform(np.asfortranarray(xh))) == _bits(m.transform(xh)), "a Fortran-order input gives the same distances")


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
    return run("test_kmeans_transform", [("SHAPE", arm_shape), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
