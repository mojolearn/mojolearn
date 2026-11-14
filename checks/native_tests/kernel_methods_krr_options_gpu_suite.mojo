# SPDX-License-Identifier: Apache-2.0
"""KernelRidge exact properties on the device estimator entries.

The three fixtures run in order: each submits its fit and predictions to the
shared family DeviceContext, and concurrent submission from several host
threads to one context is a separately recorded open failure. The device
kernels keep their own parallelism; the word comparisons use host cores.
"""
from std.math import sqrt
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import kernel_ridge_fit_host, kernel_ridge_predict_host
from svm.impl.svm_parameter import KernelParams
from checks.native_tests.kernel_methods_krr_options_cases import (
    KRR_OPTIONS_FIXTURES, KrrOptionsCalls, fixture, require,
)


struct DeviceKrrOptions(KrrOptionsCalls):
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], weights: List[Float64], n: Int, d: Int,
                    kernel: Int, gamma: Float64, alpha: Float32, query: List[Float32], q: Int,
                    ) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
        # sqrt(sample_weight) in binary64, rounded once to float32: the
        # binding's km_sqrt_weights rule.
        var roots = List[Float32](length=len(weights), fill=0)
        var wp = weights.unsafe_ptr()
        var rp = roots.unsafe_ptr()
        def root(index: Int) {imm wp, imm rp}:
            rp.unsafe_store(index, Float32(sqrt(wp[index])))
        host_parallelize(root, len(weights))
        var trace = IdentityTrace()
        var model = kernel_ridge_fit_host(x, y, n, d, 1, KernelParams(kernel, 3, gamma, 1.0), alpha, trace, sw=roots)
        require(model.info == 0, "KernelRidge device fit info")
        var dual = model.dual_coef.copy()
        var first = kernel_ridge_predict_host(model, query, q, trace)
        var second = kernel_ridge_predict_host(model, query, q, trace)
        _ = model^
        _ = roots^
        return (dual^, first^, second^)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(KRR_OPTIONS_FIXTURES):
        try:
            fixture[DeviceKrrOptions](unit)
        except error:
            failed += 1
            print("FAIL kernel_methods_krr_options_gpu fixture", unit, String(error))
    require(failed == 0, "kernel_methods_krr_options_gpu failed " + String(failed) + " fixtures")
    print("PASS kernel_methods_krr_options_gpu:", KRR_OPTIONS_FIXTURES, "exact KernelRidge properties")
