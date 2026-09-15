# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `GaussianProcessRegressor.sample_y` (2026-09-15). The
reference is scikit-learn's `sample_y`: draws from the posterior at `X`, mean
plus a factor of the predictive covariance times standard normals, shape
`(n_rows, n_samples)`, un-normalized under `normalize_y`. The factor is the
identical Cholesky at `2^-20` and the normals are position-mapped Philox
(DEVIATION 2793, `gaussian_process/checks/sample_y.mojo`). On the GPU set it
runs `gaussian_process/estimator.mojo::gpr_sample_y_host`; on a CPU-only
install the gp host binding's internal verifier
`gaussian_process/host/sample_y_oracle.mojo` (the fits run inside the private
reference context, as the identity harness's do).

    cd python && python3 -m mojolearn.tests.test_gp_sample_y

Exit 2 naming the build script when the binding is unbuilt. Small fixtures,
one process: cheap on one core.
"""
import sys
from pathlib import Path

import numpy as np

from mojolearn._cpu_reference import reference_training
from mojolearn._gp_impl import RBF, ConstantKernel, GaussianProcessRegressor, WhiteKernel
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run

ROOT = Path(__file__).resolve().parents[3]


def _data(seed=4, n=96, q=24):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = (50.0 + 12.0 * np.sin(x[:, 0]) + 3.0 * x[:, 1]).astype(np.float32)
    xq = rng.normal(size=(q, 3)).astype(np.float32)
    return x, y, xq


def _fit(x, y, normalize_y=False):
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
    with reference_training():
        return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y).fit(x, y)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def arm_shape(rep):
    x, y, q = _data()
    m = _fit(x, y)
    s = np.asarray(m.sample_y(q, n_samples=5, random_state=1))
    rep.check("SHAPE", s.dtype == np.float32 and s.shape == (24, 5), "float32 (n_rows, n_samples)", (s.dtype, s.shape))
    rep.check("SHAPE", bool(np.all(np.isfinite(s))), "every draw is finite")
    one = np.asarray(m.sample_y(q))
    rep.check("SHAPE", one.shape == (24, 1), "the default is one draw, still two-dimensional as the reference's", one.shape)


def arm_repeat(rep):
    x, y, q = _data(5)
    m = _fit(x, y)
    a, b = m.sample_y(q, 3, random_state=7), m.sample_y(q, 3, random_state=7)
    rep.check("REPEAT", _bits(a) == _bits(b), "two calls with one random_state give the same bits")
    c = m.sample_y(q, 3, random_state=8)
    rep.check("REPEAT", _bits(c) != _bits(a), "another random_state gives other draws")
    hi = m.sample_y(q, 3, random_state=2 ** 32 + 7)
    rep.check("REPEAT", _bits(hi) != _bits(a), "the key's high word reaches the stream")
    top = np.asarray(m.sample_y(q, 3, random_state=2 ** 64 - 1))
    rep.check("REPEAT", bool(np.all(np.isfinite(top))), "the largest random_state is a valid key")
    wide = np.asarray(m.sample_y(q, 5, random_state=7))
    rep.check("REPEAT", _bits(wide[:, :3]) == _bits(np.asarray(a)),
              "draw s does not depend on n_samples (the normals are position-mapped)")


def arm_moments(rep):
    x, y, q = _data(6, q=6)
    m = _fit(x, y)
    mean, std = m.predict(q, return_std=True)
    mean = np.asarray(mean, np.float64)
    std = np.asarray(std, np.float64)
    n = 4000
    s = np.asarray(m.sample_y(q, n, random_state=3), np.float64)
    se = std / np.sqrt(n)
    rep.check("MOMENTS", bool(np.all(np.abs(s.mean(1) - mean) < 6.0 * se + 1e-4)),
              "the draw mean is within 6 standard errors of predict's mean", (s.mean(1).tolist(), mean.tolist()))
    rel = np.abs(s.std(1) - std) / std
    rep.check("MOMENTS", bool(np.all(rel < 0.08)), "the draw std is within 8 percent of predict's std", rel.tolist())


def arm_reference(rep):
    """The joint covariance against scikit-learn's predict(return_cov=True),
    when scikit-learn is installed (float64 there, float32 here)."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
        from sklearn.gaussian_process import kernels as skk
    except ImportError:
        rep.check("REFERENCE", True, "SKIPPED: scikit-learn is not installed")
        return
    x, y, q = _data(7, q=5)
    for normalize_y in (False, True):
        m = _fit(x, y, normalize_y)
        k = skk.ConstantKernel(1.0, "fixed") * skk.RBF(1.0, "fixed") + skk.WhiteKernel(0.1, "fixed")
        ref = SkGPR(kernel=k, alpha=2.0 ** -20, optimizer=None, normalize_y=normalize_y).fit(
            x.astype(np.float64), y.astype(np.float64))
        rmean, rcov = ref.predict(q.astype(np.float64), return_cov=True)
        s = np.asarray(m.sample_y(q, 6000, random_state=9), np.float64)
        emp = np.cov(s)
        scale = np.sqrt(np.outer(np.diag(rcov), np.diag(rcov)))
        rep.check("REFERENCE", float(np.max(np.abs(emp - rcov) / scale)) < 0.08,
                  f"normalize_y={normalize_y}: the draw covariance is within 0.08 correlation units of the reference's",
                  float(np.max(np.abs(emp - rcov) / scale)))
        se = np.sqrt(np.diag(rcov) / s.shape[1])
        rep.check("REFERENCE", bool(np.all(np.abs(s.mean(1) - rmean) < 6.0 * se + 1e-3)),
                  f"normalize_y={normalize_y}: the draw mean is within 6 standard errors of the reference's mean")


def arm_refuse(rep):
    x, y, q = _data(8, n=48)
    rep.raises("REFUSE", ValueError, "call fit() first", "sample_y before fit",
               GaussianProcessRegressor().sample_y, q)
    m = _fit(x, y)
    rep.raises("REFUSE", Exception, "n_samples must be at least 1", "n_samples=0 is refused by name in Mojo",
               m.sample_y, q, 0)
    rep.raises("REFUSE", TypeError, "must be an int", "a float n_samples", m.sample_y, q, 2.0)
    rep.raises("REFUSE", ValueError, "random_state", "random_state=None (the key is an int)",
               m.sample_y, q, 2, None)
    rep.raises("REFUSE", ValueError, "random_state", "a negative random_state", m.sample_y, q, 2, -1)
    rep.raises("REFUSE", ValueError, "features", "a query with the wrong width", m.sample_y, q[:, :2], 2)
    tsv = (ROOT / "gaussian_process/NOT_IMPLEMENTED.tsv").read_text()
    rep.check("REFUSE", "sklearn GaussianProcessRegressor.sample_y " not in tsv,
              "the NOT_IMPLEMENTED row for sample_y is gone")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_gp", "build_gp.sh")
    rep = Report("test_gp_sample_y")
    return run("test_gp_sample_y", [("SHAPE", arm_shape), ("REPEAT", arm_repeat), ("MOMENTS", arm_moments),
                                    ("REFERENCE", arm_reference), ("REFUSE", arm_refuse)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
