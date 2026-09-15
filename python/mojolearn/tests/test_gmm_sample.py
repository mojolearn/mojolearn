# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `GaussianMixture.sample` (2026-09-15). The reference is
scikit-learn's `BaseMixture.sample`: a multinomial draw of the component
counts over `weights_`, rows grouped by component ascending, `y` naming each
row's component. The draws are position-mapped Philox (DEVIATION 2791) and
the normals go through `precisions_cholesky_` (DEVIATION 2792,
`mixture/checks/sample.mojo`). On the GPU set it runs
`mixture/estimator.mojo::gaussian_mixture_sample`; on a CPU-only install the
mixture host binding's `gmm_sample_host` (the fits run inside the private
reference context, as the identity harness's do).

    cd python && python3 -m mojolearn.tests.test_gmm_sample

Exit 2 naming the build script when the binding is unbuilt. Small fixtures,
one process: cheap on one core.
"""
import sys

import numpy as np

from mojolearn import GaussianMixture
from mojolearn._cpu_reference import reference_training
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run


def _blobs(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0, 0.0], [1.0, 0.5, 2.0], size=(n // 3, 3))
    b = rng.normal([8.0, -6.0, 3.0], [0.7, 1.5, 0.4], size=(n - n // 3, 3))
    return np.ascontiguousarray(np.vstack([a, b]).astype(np.float32))


def _fit(x, **kw):
    with reference_training():
        return GaussianMixture(**kw).fit(x)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def arm_shape(rep):
    m = _fit(_blobs(), n_components=2, random_state=3)
    x, y = m.sample(500)
    xa, ya = np.asarray(x), np.asarray(y)
    rep.check("SHAPE", xa.dtype == np.float32 and xa.shape == (500, 3), "X is float32 (n_samples, n_features)", (xa.dtype, xa.shape))
    rep.check("SHAPE", ya.dtype == np.int32 and ya.shape == (500,), "y is int32 (n_samples,)", (ya.dtype, ya.shape))
    rep.check("SHAPE", bool(np.all(np.diff(ya) >= 0)) and int(ya.min()) >= 0 and int(ya.max()) < 2,
              "y is grouped by component ascending, as scikit-learn stacks it")
    rep.check("SHAPE", bool(np.all(np.isfinite(xa))), "every sampled value is finite")
    x1, _ = m.sample(1)
    rep.check("SHAPE", np.asarray(x1).shape == (1, 3), "sample(1) is one row")


def arm_repeat(rep):
    m = _fit(_blobs(1), n_components=2, random_state=3)
    a, b = m.sample(300), m.sample(300)
    rep.check("REPEAT", _bits(a[0]) == _bits(b[0]) and _bits(a[1]) == _bits(b[1]), "two calls with one random_state give the same bits")
    m.random_state = 4
    c = m.sample(300)
    rep.check("REPEAT", _bits(c[0]) != _bits(a[0]), "another random_state gives other rows")
    m.random_state = 2 ** 64 - 1
    d = m.sample(300)
    rep.check("REPEAT", bool(np.all(np.isfinite(np.asarray(d[0])))), "the largest random_state is a valid key")


def arm_moments(rep):
    m = _fit(_blobs(2, n=6000), n_components=2, random_state=3)
    n = 20000
    x, y = m.sample(n)
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y)
    w = np.asarray(m.weights_, dtype=np.float64)
    for j in range(2):
        cnt = int(np.sum(ya == j))
        sd = np.sqrt(n * w[j] * (1.0 - w[j]))
        rep.check("MOMENTS", abs(cnt - n * w[j]) < 5.0 * sd + 1.0, f"component {j}: count within 5 sd of n * weight", (cnt, n * w[j]))
        rows = xa[ya == j]
        mu = np.asarray(m.means_, dtype=np.float64)[j]
        cov = np.asarray(m.covariances_, dtype=np.float64)[j]
        se = np.sqrt(np.diag(cov) / len(rows))
        rep.check("MOMENTS", bool(np.all(np.abs(rows.mean(0) - mu) < 6.0 * se)), f"component {j}: sample mean within 6 standard errors of means_",
                  (rows.mean(0).tolist(), mu.tolist()))
        emp = np.cov(rows.T)
        rel = np.abs(emp - cov) / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        rep.check("MOMENTS", bool(np.all(rel < 0.1)), f"component {j}: sample covariance within 0.1 correlation units of covariances_",
                  float(rel.max()))


def arm_whiten(rep):
    m = _fit(_blobs(3), n_components=2, random_state=3)
    x, y = m.sample(4000)
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y)
    p = np.asarray(m.precisions_cholesky_, dtype=np.float64)
    mu = np.asarray(m.means_, dtype=np.float64)
    z = np.vstack([(xa[ya == j] - mu[j]) @ p[j] for j in range(2)])
    rep.check("WHITEN", bool(np.all(np.abs(z.mean(0)) < 0.08)) and bool(np.all(np.abs(z.std(0) - 1.0) < 0.06)),
              "(X - mean) P is standard normal: the draw solves P^T y = z", (z.mean(0).tolist(), z.std(0).tolist()))


def arm_refuse(rep):
    rep.raises("REFUSE", ValueError, "call fit before sample", "sample before fit", GaussianMixture(n_components=2).sample, 4)
    m = _fit(_blobs(4, n=600), n_components=2, random_state=3)
    rep.raises("REFUSE", Exception, "requires at least one sample", "n_samples=0 is refused by name in Mojo", m.sample, 0)
    rep.raises("REFUSE", TypeError, "must be an int", "a float n_samples", m.sample, 2.0)
    m.random_state = -1
    rep.raises("REFUSE", ValueError, "random_state", "a negative random_state", m.sample, 4)


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_mixture", "build_mixture.sh")
    rep = Report("test_gmm_sample")
    return run("test_gmm_sample", [("SHAPE", arm_shape), ("REPEAT", arm_repeat), ("MOMENTS", arm_moments),
                                   ("WHITEN", arm_whiten), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
