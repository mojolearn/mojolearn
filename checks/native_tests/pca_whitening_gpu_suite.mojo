# SPDX-License-Identifier: Apache-2.0
"""Importable native GPU PCA whitening checks through existing tuned kernels.

The production estimators context/pools are shared, so fixture graphs submit in
order. Product kernels retain their parallel implementation; independent host
oracle cells use host_parallelize. No concurrency or timing claim.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import process_ctx
from core.hostptr import f32_ptr
from decomposition.estimator import pca_fit_host, pca_whiten_transform_host, pca_whiten_inverse_transform_host
from checks.native_tests.pca_whitening_cases import PCA_WHITENING_FIXTURES, PcaWhiteningCalls, fixture, require

comptime _CONTEXT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


struct DevicePcaWhitening(PcaWhiteningCalls):
    @staticmethod
    def fit(x: Int, components: Int, mean: Int, explained: Int, ratio: Int,
        singular: Int, rows: Int) raises:
        var ctx = process_ctx[_CONTEXT]()
        var status: List[Int32] = [1234567, -1, 1234567]
        _ = pca_fit_host(ctx, f32_ptr(x), f32_ptr(components), f32_ptr(mean), f32_ptr(explained),
            f32_ptr(ratio), f32_ptr(singular), rows, 2, 2, Int(status.unsafe_ptr()) + 4)
        ctx.synchronize()
        require(status[0] == 1234567 and status[2] == 1234567, "PCA finite status guards")
        require(status[1] == 0, "finite PCA fixture refused")
        _ = status
        _ = ctx^

    @staticmethod
    def query(x: Int, mean: Int, components: Int, singular: Int, dst: Int,
        rows: Int, fit_rows: Int, inverse: Bool) raises:
        var ctx = process_ctx[_CONTEXT]()
        if inverse:
            pca_whiten_inverse_transform_host(ctx, f32_ptr(x), f32_ptr(components), f32_ptr(singular),
                f32_ptr(mean), f32_ptr(dst), rows, 2, 2, fit_rows, True)
        else:
            pca_whiten_transform_host(ctx, f32_ptr(x), f32_ptr(mean), f32_ptr(components), f32_ptr(singular),
                f32_ptr(dst), rows, 2, 2, fit_rows, True)
        ctx.synchronize()
        _ = ctx^


def run_checks() raises:
    var failed = 0
    for fixture_id in range(PCA_WHITENING_FIXTURES):
        try:
            fixture[DevicePcaWhitening](fixture_id)
        except error:
            failed += 1
            print("FAIL pca_whitening_gpu fixture", fixture_id, String(error))
    require(failed == 0, "pca_whitening_gpu failed " + String(failed) + " fixtures")
    print("PASS pca_whitening_gpu:", PCA_WHITENING_FIXTURES, "native whitening state fixtures")
