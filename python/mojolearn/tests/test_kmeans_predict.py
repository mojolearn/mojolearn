# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `KMeans.predict` (2026-09-15), mirroring cuML's
`KMeans.predict` (`kmeans.pyx:1071-1082`): the nearest fitted center under
the model's metric, the fit's own final assignment. On the GPU set it runs
`cluster/estimator.mojo::kmeans_predict`; on a CPU-only install the core
host binding's `host_kmeans_predict` (public inference; the fits here run
inside the private reference context, as the identity harness's do).

    cd python && python3 -m mojolearn.tests.test_kmeans_predict

The numerical arms (predict(X) is labels_, center ties, held-out argmin,
rows alone and split batches) are Mojo checks:
`checks/native_tests/cluster_kmeans_query_*` (`pixi run
check-native-cluster-kmeans-query-cpu` / `-gpu`). This gate keeps the input
conversions and the refusals.

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


def arm_dtype(rep):
    x = _x(5)
    m = _fit(x=x, n_clusters=4, random_state=3)
    p32 = m.predict(x)
    x64 = x.astype(np.float64)
    rep.check("DTYPE", _bits(m.predict(x64)) == _bits(p32), "float64 input is cast to the centers' float32, the same labels")
    rep.check("DTYPE", _bits(m.predict(np.asfortranarray(x))) == _bits(p32), "a Fortran-order input gives the same labels")
    rep.check("DTYPE", _bits(m.predict(x.tolist())) == _bits(p32), "a nested list gives the same labels")


def arm_refuse(rep):
    x = _x(8, n=64)
    rep.raises("REFUSE", RuntimeError, "not fitted", "predict before fit", KMeans(n_clusters=2).predict, x)
    m = _fit(x=x, n_clusters=2, random_state=3)
    rep.raises("REFUSE", ValueError, "features", "a feature count other than the fit's", m.predict, x[:, :3])
    rep.raises("REFUSE", Exception, "", "a 1-D X", m.predict, x[0])
    rep.raises("REFUSE", ValueError, "X is empty", "an empty X", m.predict, np.zeros((0, 4), dtype=np.float32))
    # cosine was deleted 2026-09-18; an unsupported metric CODE must still be
    # refused by the kernel rather than accepted and ignored.
    bad = KMeans(n_clusters=2, metric=2)
    bad.cluster_centers_ = np.asarray(m.cluster_centers_)
    rep.raises("REFUSE", Exception, "L2Expanded (0) and L2SqrtExpanded",
               "an unsupported metric code predict is refused BY NAME, as its fit is", bad.predict, x)
    bad = KMeans(n_clusters=2, metric="manhattan")
    bad.cluster_centers_ = np.asarray(m.cluster_centers_)
    rep.raises("REFUSE", ValueError, "metric must be", "an unknown metric name", bad.predict, x)


def main(out=sys.stdout):
    bind_or_exit("_mojolearn", "build.sh")
    rep = Report("test_kmeans_predict")
    return run("test_kmeans_predict", [("DTYPE", arm_dtype), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
