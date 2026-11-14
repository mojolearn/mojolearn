# SPDX-License-Identifier: Apache-2.0
"""KernelRidge exact properties on the host entries the host binding calls,
the three independent fixtures across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, f64_ptr
from kernel_methods.host.kernel_methods_host_run import kernel_ridge_fit_host_run, kernel_ridge_predict_host_run
from checks.native_tests.kernel_methods_krr_options_cases import (
    KRR_OPTIONS_FIXTURES, KrrOptionsCalls, fixture, require,
)


struct HostKrrOptions(KrrOptionsCalls):
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], weights: List[Float64], n: Int, d: Int,
                    kernel: Int, gamma: Float64, alpha: Float32, query: List[Float32], q: Int,
                    ) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
        var dual = List[Float32](length=n, fill=0)
        var scalars = List[Float64](length=1, fill=0)
        var weighted = len(weights) > 0
        var info = kernel_ridge_fit_host_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), weighted,
            Int(weights.unsafe_ptr()) if weighted else 0, n, d, 1, kernel, 3, gamma, 1.0, alpha,
            f32_ptr(Int(dual.unsafe_ptr())), f64_ptr(Int(scalars.unsafe_ptr())))
        require(info == 0, "KernelRidge host fit info")
        var first = List[Float32](length=q, fill=0)
        var second = List[Float32](length=q, fill=0)
        kernel_ridge_predict_host_run(Int(x.unsafe_ptr()), Int(dual.unsafe_ptr()), Int(query.unsafe_ptr()),
            n, d, 1, kernel, 3, gamma, 1.0, q, f32_ptr(Int(first.unsafe_ptr())))
        kernel_ridge_predict_host_run(Int(x.unsafe_ptr()), Int(dual.unsafe_ptr()), Int(query.unsafe_ptr()),
            n, d, 1, kernel, 3, gamma, 1.0, q, f32_ptr(Int(second.unsafe_ptr())))
        _ = x
        _ = y
        _ = weights
        _ = query
        _ = scalars^
        return (dual^, first^, second^)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=KRR_OPTIONS_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture[HostKrrOptions](unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kernel_methods_krr_options_cpu fixture", unit, String(error))
    host_parallelize(task, KRR_OPTIONS_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kernel_methods_krr_options_cpu failed " + String(failed) + " fixtures")
    print("PASS kernel_methods_krr_options_cpu:", KRR_OPTIONS_FIXTURES, "exact KernelRidge properties")
