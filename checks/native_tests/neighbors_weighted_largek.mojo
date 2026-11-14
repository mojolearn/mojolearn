# SPDX-License-Identifier: Apache-2.0
"""Distance-weighted k-NN votes and large-k ranked selection on the device.

Ported from `python/mojolearn/tests/test_neighbors_weighted_largek.py` (the
CUDA/HIP opt-in gate): the three numerical tests call the estimator's own
entries (`neighbors/estimator.mojo` `knn_classifier_predict`,
`knn_regressor_predict`, `knn_search`) with the original inputs, options,
expected values and tolerances. The kd_tree and k > 1024 refusals stay in
the Python gate (public admission wording).

1. Exact matches replace the whole weight row: two training rows at distance
   0 outvote nothing else, the distant row contributes exactly zero.
2. Inverse distance changes the uniform winner: 3/7 and 4/7, and 30/7.
3. Strided rank boundaries and index ties (IDENTICAL; the pixi task builds
   with it): 1,056 rows whose Manhattan distances repeat three times across
   every thread-stride boundary; the ids are the (distance, index) order of an
   independent FP64 oracle, the distances its exact FP32 rounding, and a
   repeat call and query tile 2 give the same bits, for k = 256, 257, 300,
   513 and 1024.

The fixtures run in order on the one physical device; every search and vote
keeps its own kernel parallelism. The FP64 oracle ranks rows across host
cores.
"""
from max.gpu.host import DeviceContext
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr, u32_ptr
from neighbors.estimator import knn_classifier_predict, knn_regressor_predict, knn_search
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO
from neighbors.impl.distance.detail.distance_ops import DIST_L1
from neighbors.impl.selection.distance_weights import WEIGHTS_UNIFORM, WEIGHTS_DISTANCE
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was neighbors_weighted_largek_gpu_suite.mojo) ----
comptime WEIGHTED_LARGEK_FIXTURES = 3


def _f(values: List[Float32]) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(Int(values.unsafe_ptr()))


def _i(values: List[Int32]) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(Int(values.unsafe_ptr()))


def _u(values: List[UInt32]) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    return u32_ptr(Int(values.unsafe_ptr()))


def _close(value: Float32, expected: Float64, rtol: Float64, atol: Float64) -> Bool:
    return abs(Float64(value) - expected) <= atol + rtol * abs(expected)


def classify(ctx: DeviceContext, x: List[Float32], y: List[Int32], query: List[Float32], k: Int,
             weights: Int, mut labels: List[Int32], mut proba: List[Float32], mut uniq: List[Int32]) raises:
    var n_classes: List[Int] = [len(uniq)]
    _ = knn_classifier_predict(ctx, _f(x), len(x), _f(query), len(query), 1, k, _i(y), 1, n_classes,
        _i(labels), _f(proba), _i(uniq), True, weights=weights)
    _ = x
    _ = y
    _ = query


def regress(ctx: DeviceContext, x: List[Float32], y: List[Float32], outputs: Int, query: List[Float32],
            k: Int, mut dst: List[Float32]) raises:
    _ = knn_regressor_predict(ctx, _f(x), len(x), _f(query), len(query), 1, k, _f(y), outputs, _f(dst),
        weights=WEIGHTS_DISTANCE)
    _ = x
    _ = y
    _ = query


def case_exact_matches(ctx: DeviceContext) raises:
    var x: List[Float32] = [0.0, 0.0, 2.0]
    var q: List[Float32] = [0.0]
    var y: List[Int32] = [7, 3, 7]
    var labels = List[Int32](length=1, fill=-9)
    var proba = List[Float32](length=2, fill=-9.0)
    var uniq = List[Int32](length=2, fill=-9)
    classify(ctx, x, y, q, 3, WEIGHTS_DISTANCE, labels, proba, uniq)
    require(uniq[0] == 3 and uniq[1] == 7, "exact matches: classes are 3 and 7")
    require(labels[0] == 3, "exact matches: the tie between the two zero-distance rows goes to class 3")
    require(proba[0] == 0.5 and proba[1] == 0.5, "exact matches: predict_proba is [[0.5, 0.5]]")
    # Two outputs and signed targets: the distant value contributes exactly
    # zero, rather than a small inverse-distance weight.
    var targets: List[Float32] = [2.0, -8.0, 6.0, 4.0, 100.0, 100.0]
    var got = List[Float32](length=2, fill=-9.0)
    var again = List[Float32](length=2, fill=-9.0)
    regress(ctx, x, targets, 2, q, 3, got)
    require(got[0] == 4.0 and got[1] == -2.0, "exact matches: the regressor predicts [[4, -2]] exactly")
    regress(ctx, x, targets, 2, q, 3, again)
    require(got[0].to_bits() == again[0].to_bits() and got[1].to_bits() == again[1].to_bits(),
        "exact matches: a repeated regressor predict is the same bits")
    _ = x^
    _ = q^
    _ = y^
    _ = labels^
    _ = proba^
    _ = uniq^
    _ = targets^
    _ = got^
    _ = again^


def case_inverse_distance(ctx: DeviceContext) raises:
    var x: List[Float32] = [1.0, 2.0, 4.0]
    var q: List[Float32] = [0.0]
    var y: List[Int32] = [2, 1, 1]
    var weighted = List[Int32](length=1, fill=-9)
    var weighted_proba = List[Float32](length=2, fill=-9.0)
    var weighted_uniq = List[Int32](length=2, fill=-9)
    classify(ctx, x, y, q, 3, WEIGHTS_DISTANCE, weighted, weighted_proba, weighted_uniq)
    var uniform = List[Int32](length=1, fill=-9)
    var uniform_proba = List[Float32](length=2, fill=-9.0)
    var uniform_uniq = List[Int32](length=2, fill=-9)
    classify(ctx, x, y, q, 3, WEIGHTS_UNIFORM, uniform, uniform_proba, uniform_uniq)
    require(weighted[0] == 2, "inverse distance: the weighted vote predicts 2")
    require(uniform[0] == 1, "inverse distance: the uniform vote predicts 1")
    require(weighted_uniq[0] == 1 and weighted_uniq[1] == 2, "inverse distance: classes are 1 and 2")
    require(_close(weighted_proba[0], 3.0 / 7.0, 2e-6, 1e-7) and _close(weighted_proba[1], 4.0 / 7.0, 2e-6, 1e-7),
        "inverse distance: predict_proba is [[3/7, 4/7]]")
    var targets: List[Float32] = [2.0, 6.0, 10.0]
    var got = List[Float32](length=1, fill=-9.0)
    regress(ctx, x, targets, 1, q, 3, got)
    require(_close(got[0], 30.0 / 7.0, 2e-6, 1e-7), "inverse distance: the regressor predicts 30/7")
    _ = x^
    _ = q^
    _ = y^
    _ = weighted^
    _ = weighted_proba^
    _ = weighted_uniq^
    _ = uniform^
    _ = uniform_proba^
    _ = uniform_uniq^
    _ = targets^
    _ = got^


def search(ctx: DeviceContext, x: List[Float32], q: List[Float32], k: Int, tile: Int,
           mut distances: List[Float32], mut ids: List[UInt32]) raises:
    _ = knn_search(ctx, _f(x), len(x), _f(q), len(q), 1, k, _f(distances), _u(ids), True, tile,
        KNN_METHOD_AUTO, DIST_L1, Float32(2.0), True)
    _ = x
    _ = q


def case_strided_ranks(ctx: DeviceContext) raises:
    comptime N = 1056
    var x = List[Float32](length=N, fill=0)
    for i in range(N):
        x[i] = Float32(i // 3) / 16.0  # exact dyadics
    var q: List[Float32] = [0.0, 4.0]
    # Independent FP64 oracle: each row's rank in (distance, index) order.
    var order = List[Int](length=2 * N, fill=-1)
    var op = order.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var qp = q.unsafe_ptr()
    def rank(unit: Int) {imm op, imm xp, imm qp}:
        var row = unit // N
        var i = unit % N
        var di = abs(Float64(qp[row]) - Float64(xp[i]))
        var r = 0
        for j in range(N):
            var dj = abs(Float64(qp[row]) - Float64(xp[j]))
            if dj < di or (dj == di and j < i):
                r += 1
        op[row * N + r] = i
    host_parallelize(rank, 2 * N)
    var ks: List[Int] = [256, 257, 300, 513, 1024]
    for k in ks:
        var d = List[Float32](length=2 * k, fill=-9.0)
        var ids = List[UInt32](length=2 * k, fill=UInt32(0xFFFFFFFF))
        var dr = List[Float32](length=2 * k, fill=-9.0)
        var ir = List[UInt32](length=2 * k, fill=UInt32(0xFFFFFFFF))
        var dt = List[Float32](length=2 * k, fill=-9.0)
        var it = List[UInt32](length=2 * k, fill=UInt32(0xFFFFFFFF))
        search(ctx, x, q, k, 1, d, ids)
        search(ctx, x, q, k, 1, dr, ir)
        search(ctx, x, q, k, 2, dt, it)
        var wrong = 0
        for row in range(2):
            for s in range(k):
                var want = order[row * N + s]
                var want_distance = Float32(abs(Float64(q[row]) - Float64(x[want])))
                var c = row * k + s
                if Int(ids[c]) != want or d[c].to_bits() != want_distance.to_bits():
                    wrong += 1
                if ir[c] != ids[c] or it[c] != ids[c]:
                    wrong += 1
                if dr[c].to_bits() != d[c].to_bits() or dt[c].to_bits() != d[c].to_bits():
                    wrong += 1
        require(wrong == 0, "strided ranks k=" + String(k) + ": " + String(wrong)
            + " slots differ from the FP64 (distance, index) oracle or between repeat/tile-2 calls")
        _ = d^
        _ = ids^
        _ = dr^
        _ = ir^
        _ = dt^
        _ = it^
    _ = x^
    _ = q^
    _ = order^


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(WEIGHTED_LARGEK_FIXTURES)
    for unit in range(WEIGHTED_LARGEK_FIXTURES):
        try:
            if unit == 0:
                case_exact_matches(ctx)
            elif unit == 1:
                case_inverse_distance(ctx)
            else:
                case_strided_ranks(ctx)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("neighbors_weighted_largek", cpu=False, gpu=True, serial="", run=run_cases)
