# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""KernelDensity's fused FAST score pass (DEVIATION 2490) against the staged pass, its FLOAT_MIN sentinel and its
batch invariance (lane py2mojo-D, batch P, 2026-11-22; ported from python/mojolearn/tests/test_kde_fused_fast.py
`test_fused_equals_staged_within_float32_fold_tolerance`, `test_all_out_of_range_row_scores_float_min_not_minus_inf`
and `test_query_score_does_not_depend_on_its_batch`).

The calls are the ones the public KernelDensity fit and score_samples run on the device column: the resident fit
handle (`kde/resident_fit.mojo` `kde_fit_prepare`) and its score (`kde_score_samples_resident`), which reach
`kde_score_samples_device`. As in the original, the staged arm is selected by recording an identity trace
(`MOJOLEARN_IDENTITY_TRACE` set before the call; `IdentityTrace()` reads it once per call) and the fused arm by
leaving it empty. The cases share the one device context and the process environment, so they run in order.

    fused_equals_staged  12 rows: the six (n_train, n_query, d, seed) sets x weighted False / True; per row the six
                         metrics x six kernels at bandwidth 0.9 (minkowski p = 2): per query the sentinels
                         (<= -1e30 or NaN) agree exactly, otherwise |fused - staged| / max(1, |staged|) < 2e-4, and
                         the worst gap is > 0 unless n_train < 300 (the two folds are different code)
    float_min_sentinel   4 rows: tophat, epanechnikov, linear, cosine; train 10..39 (one feature), queries 0, 100,
                         25.1 at bandwidth 0.25: rows 0 and 1 are FLOAT_MIN (rel 1e-6) on both arms and equal; row 2
                         finite, > -1e30 and within rel 1e-4 of the staged arm
    batch_invariance     1 row: gaussian at bandwidth 0.8 over 400 x 5; the query alone and at position 1500 of a
                         batch of 3000 give the same bytes

Inputs: the original tests' `random.Random` draws rounded to float32 as `Array.from_list(dtype="<f4")` rounds them,
stored by `tools/py2mojo_native_fixtures_bP.py` in `fixtures/kde_fused_fast/`; the sentinel case's literals are
written here. The arm exists only in a FAST build (Apple, `MOJOLEARN_NUMERIC_FAST`); in an identical build both
calls take the staged path and the comparison is trivially exact, so the tests are skipped there, as the original
skips. The public dtype, refusals and the scikit-learn comparison stay in the Python file.
"""
from std.math import isfinite, isnan
from std.memory import bitcast
from std.os import getenv, setenv
from std.testing import TestSuite
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.hostptr import f32_ptr
from kde.resident_fit import kde_fit_prepare, kde_fit_release, kde_score_samples_resident
from checks.native_tests.testsuite_cases import require, CPU_COLUMN_BUILD, check_cases_serial, fixture_f32

# DEVIATION 2490 is taken only by a FAST device build (kde/impl/neighbors/kernel_density.mojo).
comptime FUSED_ARM = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not CPU_COLUMN_BUILD
comptime DIR = "checks/native_tests/fixtures/kde_fused_fast/"
comptime TRACE_ENV = "MOJOLEARN_IDENTITY_TRACE"  # core/identity_trace.mojo
comptime FLOAT_MIN = -3.4028234663852886e38
comptime FE_SETS = 6


def kernel_at(index: Int) -> String:
    var names: List[String] = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    return names[index]


def metric_at(index: Int) -> String:
    var names: List[String] = ["euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "minkowski"]
    return names[index]


def fe_set(p: Int) -> Tuple[Int, Int, Int]:
    """(n_train, n_query, d) of the original's parameter set p (its seed only drew the stored inputs)."""
    if p == 0:
        return (700, 333, 1)
    if p == 1:
        return (900, 257, 7)
    if p == 2:
        return (500, 129, 33)
    if p == 3:
        return (300, 65, 64)
    if p == 4:
        return (200, 41, 70)
    return (120, 17, 150)


def _trace_path() -> String:
    var base = String(getenv("TMPDIR", "/tmp"))
    return base + "/mojolearn_kde_fused_fast_staged.trace"


def scores(handle: Int, mut query: List[Float32], n_query: Int, d: Int, bandwidth: Float32, kernel: String,
           metric: String, staged: Bool) raises -> List[Float32]:
    """`score_samples` through the resident handle; `staged` records a trace, which selects the staged arm."""
    var dst = List[Float32](length=n_query, fill=Float32(0))
    _ = setenv(TRACE_ENV, _trace_path() if staged else String(""), True)
    try:
        _ = kde_score_samples_resident(handle, f32_ptr(Int(query.unsafe_ptr())), n_query, d, bandwidth, kernel,
                                       metric, f32_ptr(Int(dst.unsafe_ptr())))
    except problem:
        _ = setenv(TRACE_ENV, "", True)
        raise problem^
    _ = setenv(TRACE_ENV, "", True)
    _ = query.unsafe_ptr()  # the query words stay alive past the call
    return dst^


def fit(mut train: List[Float32], n_train: Int, d: Int, bandwidth: Float32, kernel: String, metric: String,
        weights: List[Float32], has_weights: Bool) raises -> Int:
    var handle = kde_fit_prepare(f32_ptr(Int(train.unsafe_ptr())), n_train, d, bandwidth, kernel, metric,
                                 weights, has_weights)
    _ = train.unsafe_ptr()
    return handle


def is_sentinel(v: Float32) -> Bool:
    return Float64(v) <= -1e30 or isnan(v)


def fused_equals_staged_unit(unit: Int) raises:
    var p = unit // 2
    var weighted = unit % 2
    var shape = fe_set(p)
    var n_train = shape[0]
    var n_query = shape[1]
    var d = shape[2]
    var worst = Float64(0)
    var tag = String("fe-") + String(p) + "-" + String(weighted) + "-"
    for m in range(6):  # driver-loop: over the original's metrics, in its order
        var xt = fixture_f32(DIR + tag + String(m) + "-xt.f32", n_train * d)
        var xq = fixture_f32(DIR + tag + String(m) + "-xq.f32", n_query * d)
        var w = List[Float32]()
        if weighted == 1:
            w = fixture_f32(DIR + tag + String(m) + "-w.f32", n_train)
        for k in range(6):  # driver-loop: over the original's kernels, in its order
            var kernel = kernel_at(k)
            var metric = metric_at(m)
            var handle = fit(xt, n_train, d, Float32(0.9), kernel, metric, w, weighted == 1)
            var fused = List[Float32]()
            var staged = List[Float32]()
            try:
                fused = scores(handle, xq, n_query, d, Float32(0.9), kernel, metric, False)
                staged = scores(handle, xq, n_query, d, Float32(0.9), kernel, metric, True)
            except problem:
                kde_fit_release(handle)
                raise problem^
            kde_fit_release(handle)
            require(len(fused) == n_query and len(staged) == n_query, "score count")
            for q in range(n_query):  # driver-loop: over the downloaded scores (test oracle, host)
                var a = fused[q]
                var b = staged[q]
                var at = metric + "/" + kernel + " query " + String(q) + ": fused " + String(a) + " staged " + String(b)
                if is_sentinel(a) or is_sentinel(b):
                    require(a == b, "sentinels differ, " + at)
                    continue
                var rel = abs(Float64(a) - Float64(b)) / max(Float64(1.0), abs(Float64(b)))
                worst = max(worst, rel)
                require(rel < 2e-4, "rel " + String(rel) + " >= 2e-4, " + at)
    require(worst > 0.0 or n_train < 300,
        "the fused and staged arms gave identical scores everywhere (n_train " + String(n_train)
        + "): the fused arm did not run")


def near(a: Float64, want: Float64, rel: Float64) -> Bool:
    """pytest.approx(want, rel=rel): |a - want| <= max(rel |want|, 1e-12)."""
    return abs(a - want) <= max(rel * abs(want), Float64(1e-12))


def float_min_unit(unit: Int) raises:
    var kernels: List[String] = ["tophat", "epanechnikov", "linear", "cosine"]
    var kernel = kernels[unit]
    var xt = List[Float32](length=30, fill=Float32(0))
    for i in range(30):  # driver-loop: over the original's literal train rows
        xt[i] = Float32(10 + i)
    var xq: List[Float32] = [0.0, 100.0, 25.1]
    var handle = fit(xt, 30, 1, Float32(0.25), kernel, "euclidean", List[Float32](), False)
    var fused = List[Float32]()
    var staged = List[Float32]()
    try:
        fused = scores(handle, xq, 3, 1, Float32(0.25), kernel, "euclidean", False)
        staged = scores(handle, xq, 3, 1, Float32(0.25), kernel, "euclidean", True)
    except problem:
        kde_fit_release(handle)
        raise problem^
    kde_fit_release(handle)
    for r in range(2):  # driver-loop: over the two out-of-range queries
        require(fused[r] == staged[r] and near(Float64(staged[r]), FLOAT_MIN, 1e-6),
            kernel + " query " + String(r) + ": fused " + String(fused[r]) + " staged " + String(staged[r])
            + ", expected both FLOAT_MIN")
    require(isfinite(fused[2]) and Float64(fused[2]) > -1e30,
        kernel + " query 2: fused " + String(fused[2]) + " is not a finite score")
    require(near(Float64(fused[2]), Float64(staged[2]), 1e-4),
        kernel + " query 2: fused " + String(fused[2]) + " staged " + String(staged[2]) + " beyond rel 1e-4")


def batch_invariance_unit(unit: Int) raises:
    var xt = fixture_f32(DIR + "bi-xt.f32", 400 * 5)
    var q = fixture_f32(DIR + "bi-q.f32", 5)
    var filler = fixture_f32(DIR + "bi-filler.f32", 2999 * 5)
    var big = List[Float32](capacity=3000 * 5)
    for i in range(1500 * 5):  # driver-loop: assembling the original's batch on the host
        big.append(filler[i])
    for f in range(5):  # driver-loop: the query row
        big.append(q[f])
    for i in range(1500 * 5, 2999 * 5):  # driver-loop: the rest of the batch
        big.append(filler[i])
    var handle = fit(xt, 400, 5, Float32(0.8), "gaussian", "euclidean", List[Float32](), False)
    var alone = List[Float32]()
    var batch = List[Float32]()
    try:
        alone = scores(handle, q, 1, 5, Float32(0.8), "gaussian", "euclidean", False)
        batch = scores(handle, big, 3000, 5, Float32(0.8), "gaussian", "euclidean", False)
    except problem:
        kde_fit_release(handle)
        raise problem^
    kde_fit_release(handle)
    require(bitcast[DType.uint32](alone[0]) == bitcast[DType.uint32](batch[1500]),
        "the query alone scores " + String(alone[0]) + ", at 1500 of a batch of 3000 " + String(batch[1500]))


def test_fused_equals_staged() raises:
    check_cases_serial("fused_equals_staged", 2 * FE_SETS, fused_equals_staged_unit)


def test_float_min_sentinel() raises:
    check_cases_serial("float_min_sentinel", 4, float_min_unit)


def test_batch_invariance() raises:
    check_cases_serial("batch_invariance", 1, batch_invariance_unit)


def main() raises:
    # Tests by name: an identical build has no fused arm, so it lists the three as skipped (the original skips
    # outside FAST); a FAST Apple build runs them.
    var suite = TestSuite()
    comptime if FUSED_ARM:
        suite.test[test_fused_equals_staged]()
        suite.test[test_float_min_sentinel]()
        suite.test[test_batch_invariance]()
    else:
        suite.test[test_fused_equals_staged]()
        suite.skip[test_fused_equals_staged]()
        suite.test[test_float_min_sentinel]()
        suite.skip[test_float_min_sentinel]()
        suite.test[test_batch_invariance]()
        suite.skip[test_batch_invariance]()
    suite^.run()
