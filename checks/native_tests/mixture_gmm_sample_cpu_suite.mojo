# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture.sample statistics on the host fit and host sample entries
the mixture host binding calls; the three fixtures run across host cores and
each fixture's independent draws run across host cores too."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr
from core.mixture_native_api import GmmDriver, GmmFitRequest
from mixture.host.gmm_native_host import gmm_native_host_v1
from mixture.host.gmm_host_entries import gmm_sample_into
from checks.native_tests.mixture_gmm_sample_cases import (
    GMM_SAMPLE_FIXTURES, GMM_K, GMM_D, GmmRun, GmmSampleCalls, fixture, require,
)


struct HostGmmSample(GmmSampleCalls):
    @staticmethod
    def run(x: List[Float32], n_fit: Int, n_draw: Int, keys: List[UInt64]) raises -> GmmRun:
        var weights = List[Float32](length=GMM_K, fill=0)
        var means = List[Float32](length=GMM_K * GMM_D, fill=0)
        var covariances = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
        var precisions = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
        var log_det = List[Float32](length=GMM_K, fill=0)
        var scalars = List[Float64](length=3, fill=0)
        var covariance_type = String("full")
        var init_params = String("kmeans")
        var driver = GmmDriver(gmm_native_host_v1)
        _ = driver.fit(GmmFitRequest(Int(x.unsafe_ptr()), Int(weights.unsafe_ptr()), Int(means.unsafe_ptr()),
            Int(covariances.unsafe_ptr()), Int(precisions.unsafe_ptr()), Int(log_det.unsafe_ptr()),
            Int(scalars.unsafe_ptr()), 0, Int(covariance_type.as_c_string_slice().unsafe_ptr()),
            Int(init_params.as_c_string_slice().unsafe_ptr()), n_fit, GMM_D, GMM_K, 100, 3, 1e-3, 1e-6))
        _ = covariance_type
        _ = init_params
        var draws = len(keys)
        var rows = List[Float32](length=draws * n_draw * GMM_D, fill=0)
        var labels = List[Int32](length=draws * n_draw, fill=-1)
        var failures = List[Int](length=draws, fill=0)
        var fp = failures.unsafe_ptr()
        var kp = keys.unsafe_ptr()
        var rp = rows.unsafe_ptr()
        var lp = labels.unsafe_ptr()
        var wp = weights.unsafe_ptr()
        var mp = means.unsafe_ptr()
        var pp = precisions.unsafe_ptr()
        def draw(s: Int) {imm fp, imm kp, imm rp, imm lp, imm wp, imm mp, imm pp, imm n_draw}:
            try:
                var w = List[Float32](length=GMM_K, fill=0)
                var m = List[Float32](length=GMM_K * GMM_D, fill=0)
                var p = List[Float32](length=GMM_K * GMM_D * GMM_D, fill=0)
                for i in range(GMM_K * GMM_D * GMM_D):  # this task's private copy of the 18 parameter words
                    p[i] = pp[i]
                    if i < GMM_K * GMM_D:
                        m[i] = mp[i]
                    if i < GMM_K:
                        w[i] = wp[i]
                gmm_sample_into(w, m, p, GMM_K, GMM_D, n_draw, kp[s],
                    f32_ptr(Int(rp) + 4 * s * n_draw * GMM_D),
                    i32_ptr(Int(lp) + 4 * s * n_draw))
            except error:
                fp.unsafe_store(s, 1)
                print("FAIL mixture_gmm_sample_cpu draw", s, String(error))
        host_parallelize(draw, draws)
        for value in failures:
            require(value == 0, "GMM host sample draw failed")
        _ = keys
        _ = x
        _ = log_det^
        _ = scalars^
        return GmmRun(weights^, means^, covariances^, precisions^, rows^, labels^)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=GMM_SAMPLE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture[HostGmmSample](unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL mixture_gmm_sample_cpu fixture", unit, String(error))
    host_parallelize(task, GMM_SAMPLE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "mixture_gmm_sample_cpu failed " + String(failed) + " fixtures")
    print("PASS mixture_gmm_sample_cpu:", GMM_SAMPLE_FIXTURES, "GaussianMixture.sample statistics")
