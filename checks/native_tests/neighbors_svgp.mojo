# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""SVGP's latent mean, latent variance and ELBO against the float64 closed form (lane py2mojo-D6, 2026-11-22; ported
from python/mojolearn/tests/test_x_neighbors_sanity.py `test_svgp`).

SVGP(inducing_points=X[::10], kernel_variance=1.5, lengthscale=1.2, noise_variance=0.2, jitter=1e-6) on the
original's `_data(150, 3)` (seed 0, the first five rows x 6) with y = sin(x0) + 0.1 x1 (float32), queried on
`_data(20, 3, seed=9)`. The fit and the query are the ones the door runs (neighbors/svgp_run.mojo `svgp_fit_run`
with the caller's inducing points, `svgp_query_run` with the latent variance, predict_f) on the host x_neighbors
image, gamma = 1 / (2 ls^2) as `xn_svgp_gamma`. The references are the original's float64 closed form (Kuu + jitter,
S = Kuu + Kuf Kuf^T / nv, alpha, the predictive mean and variance, the collapsed ELBO), stored in fixtures/xn_svgp/
by tools/py2mojo_native_fixtures_b6.py; tolerances the original's: mean rtol = atol = 2e-3, variance rtol 5e-2,
atol 5e-3, ELBO 1e-3 |elbo| + 0.05. The public attributes and dtypes stay in Python.
"""
from core.neighbors_native_driver import XnDriver
from neighbors.svgp_run import SvgpFitRequest, SvgpQueryRequest, svgp_fit_run, svgp_query_run, SVGP_VARIANCE_LATENT
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite

comptime XN_SVGP_DIR = "checks/native_tests/fixtures/xn_svgp/"


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def test_svgp() raises:
    var n = 150
    var m = 15
    var d = 3
    var nq = 20
    var kv = Float64(1.5)
    var ls = Float64(1.2)
    var nv = Float64(0.2)
    var gamma = 1.0 / (2.0 * (ls * ls))
    var x = fixture_f32(XN_SVGP_DIR + "x.f32", n * d)
    var y = fixture_f32(XN_SVGP_DIR + "y.f32", n)
    var z = fixture_f32(XN_SVGP_DIR + "z.f32", m * d)
    var q = fixture_f32(XN_SVGP_DIR + "q.f32", nq * d)
    var want_mean = fixture_f64(XN_SVGP_DIR + "mean.f64", nq)
    var want_var = fixture_f64(XN_SVGP_DIR + "var.f64", nq)
    var want_elbo = fixture_f64(XN_SVGP_DIR + "elbo.f64", 1)[0]
    var alpha = List[Float32](length=m, fill=0)
    var c = List[Float32](length=m * m, fill=0)
    var qmu = List[Float32](length=m, fill=0)
    var qsqrt = List[Float32](length=m * m, fill=0)
    var driver = XnDriver(xn_native_v1)
    var fitted = svgp_fit_run(driver, SvgpFitRequest(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(z.unsafe_ptr()),
        Int(alpha.unsafe_ptr()), Int(c.unsafe_ptr()), Int(qmu.unsafe_ptr()), Int(qsqrt.unsafe_ptr()), n, m, d, False,
        gamma, kv, nv, 1e-6))
    require(fitted[1], "SVGP inducing system not positive definite")
    var elbo = fitted[0]
    require(abs(elbo - want_elbo) < 1e-3 * abs(want_elbo) + 0.05,
            "SVGP elbo_ " + String(elbo) + " expected " + String(want_elbo))
    var mean = List[Float32](length=nq, fill=0)
    var variance = List[Float32](length=nq, fill=0)
    svgp_query_run(driver, SvgpQueryRequest(Int(q.unsafe_ptr()), Int(z.unsafe_ptr()), Int(alpha.unsafe_ptr()),
        Int(c.unsafe_ptr()), Int(mean.unsafe_ptr()), Int(variance.unsafe_ptr()), nq, m, d, gamma, kv, nv,
        SVGP_VARIANCE_LATENT, False))
    for i in range(nq):  # small-loop: twenty query rows
        var a = Float64(mean[i])
        require(abs(a - want_mean[i]) <= 2e-3 + 2e-3 * abs(want_mean[i]),
                "predict_f mean " + String(i) + ": " + String(mean[i]) + " expected " + String(want_mean[i]))
        var v = Float64(variance[i])
        require(abs(v - want_var[i]) <= 5e-3 + 5e-2 * abs(want_var[i]),
                "predict_f variance " + String(i) + ": " + String(variance[i]) + " expected " + String(want_var[i]))
    _ = x^
    _ = y^
    _ = z^
    _ = q^
    _ = alpha^
    _ = c^
    _ = qmu^
    _ = qsqrt^
    _ = mean^
    _ = variance^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
