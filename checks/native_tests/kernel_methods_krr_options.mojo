# SPDX-License-Identifier: Apache-2.0
"""KernelRidge exact properties from test_krr_options.py, on the actual entries.

The original 96x5 integer workload (NumPy seed 3, recorded under
fixtures/kernel_methods_krr_options/) gives an exact linear Gram matrix under
any summation order, so:

  0  kernel='precomputed' on that Gram (alpha 2) returns kernel='linear''s
     dual_coef_ words, its prediction from the 20x96 cross Gram returns the
     linear prediction of X[:20] word for word, and a second prediction
     repeats the first;
  1  a unit sample_weight (the weighted arm, sqrt(1) = 1) returns the
     unweighted rbf (gamma 0.1, alpha 1) dual_coef_ words;
  2  two weighted rbf fits (gamma 0.2, alpha 0.5, w = 0.25 + i % 3) return
     the same dual_coef_ and predict(X) words.

The Gram matrices are Int64 sums rounded once to Float32, the original
`(a.astype(int64) @ a.astype(int64).T).astype(float32)`. The scikit-learn
tolerance comparison stays in Python as an external oracle.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from kernel_methods.host.kernel_methods_host_run import kernel_ridge_fit_host_run, kernel_ridge_predict_host_run
from std.math import sqrt
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import kernel_ridge_fit_host, kernel_ridge_predict_host
from svm.impl.svm_parameter import KernelParams
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime KRR_OPTIONS_FIXTURES = 3
comptime KRR_ROWS = 96
comptime KRR_FEATURES = 5
comptime KRR_QUERIES = 20
comptime KRR_KERNEL_LINEAR = 0
comptime KRR_KERNEL_RBF = 2
comptime KRR_KERNEL_PRECOMPUTED = 4


trait KrrOptionsCalls:
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], weights: List[Float64], n: Int, d: Int,
                    kernel: Int, gamma: Float64, alpha: Float32, query: List[Float32], q: Int,
                    ) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
        """The fit, then two predictions of `query`: (dual_coef_, first, second).
        An empty `weights` is the unweighted fit."""
        ...


def recorded(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/kernel_methods_krr_options/" + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "KernelRidge options fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def integer_gram(x: List[Float32], rows: Int, n: Int, d: Int) raises -> List[Float32]:
    """rows x n Gram of X[:rows] against X, exact Int64 sums, one rounding."""
    var result = List[Float32](unsafe_uninit_length=rows * n)
    var xp = x.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm xp, imm rp, imm n, imm d}:
        var i = index // n
        var j = index % n
        var total = Int64(0)
        for f in range(d):  # one dot product's integer terms; exact in any order
            total += Int64(xp[i * d + f]) * Int64(xp[j * d + f])
        rp.unsafe_store(index, Float32(total))
    host_parallelize(cell, rows * n)
    _ = x
    return result^


def same_words(left: List[Float32], right: List[Float32], name: String) raises:
    require(len(left) == len(right), name + ": extents differ")
    var failures = List[Int](length=len(left), fill=0)
    var fp = failures.unsafe_ptr()
    var lp = left.unsafe_ptr()
    var rp = right.unsafe_ptr()
    def cell(index: Int) {imm fp, imm lp, imm rp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](lp[index]) != bitcast[DType.uint32](rp[index])))
    host_parallelize(cell, len(left))
    var differ = 0
    for value in failures:
        differ += value
    require(differ == 0, name + ": " + String(differ) + " words differ")
    _ = left
    _ = right


def fixture[Calls: KrrOptionsCalls](unit: Int) raises:
    var x = recorded("x-96x5.f32", KRR_ROWS * KRR_FEATURES)
    var y = recorded("y-96.f32", KRR_ROWS)
    var none = List[Float64]()
    if unit == 0:
        var query = List[Float32](unsafe_uninit_length=KRR_QUERIES * KRR_FEATURES)
        var sp = x.unsafe_ptr()
        var qp = query.unsafe_ptr()
        def prefix(index: Int) {imm sp, imm qp}:  # X[:20]
            qp.unsafe_store(index, sp[index])
        host_parallelize(prefix, KRR_QUERIES * KRR_FEATURES)
        var linear = Calls.fit_predict(x, y, none, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_LINEAR, 1.0, 2.0,
                                       query, KRR_QUERIES)
        var gram = integer_gram(x, KRR_ROWS, KRR_ROWS, KRR_FEATURES)
        var cross = integer_gram(x, KRR_QUERIES, KRR_ROWS, KRR_FEATURES)
        var pre = Calls.fit_predict(gram, y, none, KRR_ROWS, KRR_ROWS, KRR_KERNEL_PRECOMPUTED, 1.0, 2.0,
                                    cross, KRR_QUERIES)
        same_words(pre[0], linear[0], "precomputed vs linear dual_coef_")
        same_words(pre[1], linear[1], "precomputed vs linear predict")
        same_words(pre[1], pre[2], "precomputed repeated predict")
    elif unit == 1:
        var ones = List[Float64](length=KRR_ROWS, fill=1.0)
        var base = Calls.fit_predict(x, y, none, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.1, 1.0, x, KRR_ROWS)
        var one = Calls.fit_predict(x, y, ones, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.1, 1.0, x, KRR_ROWS)
        same_words(one[0], base[0], "unit weight vs unweighted dual_coef_")
    else:
        var weights = List[Float64](unsafe_uninit_length=KRR_ROWS)
        var wp = weights.unsafe_ptr()
        def weight(index: Int) {imm wp}:  # the original 0.25 + (arange % 3)
            wp.unsafe_store(index, 0.25 + Float64(index % 3))
        host_parallelize(weight, KRR_ROWS)
        var first = Calls.fit_predict(x, y, weights, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.2, 0.5, x, KRR_ROWS)
        var second = Calls.fit_predict(x, y, weights, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.2, 0.5, x, KRR_ROWS)
        same_words(first[0], second[0], "repeated weighted dual_coef_")
        same_words(first[1], second[1], "repeated weighted predict")


# ---- The host column (was kernel_methods_krr_options_cpu_suite.mojo) ----
# KernelRidge exact properties on the host entries the host binding calls,
# the three independent fixtures across host cores.
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


def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(KRR_OPTIONS_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture[HostKrrOptions](unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, KRR_OPTIONS_FIXTURES)


# ---- The device column (was kernel_methods_krr_options_gpu_suite.mojo) ----
# KernelRidge exact properties on the device estimator entries.
#
# The three fixtures run in order: each submits its fit and predictions to the
# shared family DeviceContext, and concurrent submission from several host
# threads to one context is a separately recorded open failure. The device
# kernels keep their own parallelism; the word comparisons use host cores.
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


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(KRR_OPTIONS_FIXTURES)
    for unit in range(KRR_OPTIONS_FIXTURES):
        try:
            fixture[DeviceKrrOptions](unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("kernel_methods_krr_options", cpu=True, gpu=True, serial="", run=run_cases)
