# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture.sample statistics on the device fit and sample entries.

The three fixtures, and the draws inside each, run in order: they submit to
the shared family DeviceContext, and concurrent submission from several host
threads to one context is a separately recorded open failure. The EM and
sample kernels keep their own parallelism; the statistics use host cores.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mixture.estimator import GmmParams, gaussian_mixture_fit, gaussian_mixture_sample
from checks.native_tests.mixture_gmm_sample_cases import (
    GMM_SAMPLE_FIXTURES, GMM_K, GMM_D, GmmRun, GmmSampleCalls, fixture, require,
)


struct DeviceGmmSample(GmmSampleCalls):
    @staticmethod
    def run(x: List[Float32], n_fit: Int, n_draw: Int, keys: List[UInt64]) raises -> GmmRun:
        var params = GmmParams.default()
        params.n_components = GMM_K
        params.random_state = UInt64(3)
        var model = gaussian_mixture_fit(x, n_fit, GMM_D, params)
        var rows = List[Float32]()
        var labels = List[Int32]()
        for s in range(len(keys)):  # driver-loop: over the requested keys, ordered on one device
            var drawn = List[Int32](length=n_draw, fill=-1)
            var values = gaussian_mixture_sample(model, n_draw, keys[s], drawn)
            require(len(values) == n_draw * GMM_D and len(drawn) == n_draw, "GMM device sample extents")
            rows.extend(values^)
            labels.extend(drawn^)
        var result = GmmRun(model.weights.copy(), model.means.copy(), model.covariances.copy(),
                            model.precisions_cholesky.copy(), rows^, labels^)
        _ = model^
        return result^


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GMM_SAMPLE_FIXTURES):
        try:
            fixture[DeviceGmmSample](unit)
        except error:
            failed += 1
            print("FAIL mixture_gmm_sample_gpu fixture", unit, String(error))
    require(failed == 0, "mixture_gmm_sample_gpu failed " + String(failed) + " fixtures")
    print("PASS mixture_gmm_sample_gpu:", GMM_SAMPLE_FIXTURES, "GaussianMixture.sample statistics")
