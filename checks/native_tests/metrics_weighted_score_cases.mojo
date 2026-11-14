# SPDX-License-Identifier: Apache-2.0
"""Weighted accuracy and weighted R^2 against their definitions (ported
from python/mojolearn/tests/test_weighted_score.py
`test_weighted_accuracy_and_r2_match_the_definition`).

The original draw (`np.random.default_rng(5)`, n = 4099: three-class labels
and predictions that agree with probability 0.7, uniform weights in
[0.25, 4) with every eleventh weight zero, a normal target and a noisy
prediction) is NumPy PCG64 output, stored as raw words in
`fixtures/metrics/weighted-*` by `tools/tsa_metrics_native_fixtures.py`.

    WEIGHTED_UNIT 0  accuracy == sum(w [y_true == y_pred]) / sum(w) in
                     Float64, to 2e-6 relative
    WEIGHTED_UNIT 1  R^2 == 1 - sum(w (y - y_hat)^2) / sum(w (y - avg)^2)
                     in Float64 (the residual squared in float32, as NumPy
                     computes it), to 1e-5 relative; a repeat is bitwise
    WEIGHTED_UNIT 2  a constant target predicted exactly scores 1.0
    WEIGHTED_UNIT 3  a constant target predicted off by one scores 0.0

The refusals, the normalize=False counts and the source checks stay in the
Python file. Importable; no `main()`.
"""
from std.memory import bitcast
from checks.native_tests.tsa_metrics_fixture_words import METRICS_FIXTURE_DIR, fixture_f32, fixture_i32

comptime WEIGHTED_N = 4099
comptime WEIGHTED_UNITS = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait WeightedCalls:
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        ...

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        ...


def isclose(a: Float64, b: Float64, rel: Float64) -> Bool:
    """`math.isclose(a, b, rel_tol=rel)`."""
    return abs(a - b) <= rel * max(abs(a), abs(b))


def weighted_unit[K: WeightedCalls](unit: Int) raises:
    var w = fixture_f32(METRICS_FIXTURE_DIR + "weighted-w.f32", WEIGHTED_N)
    if unit == 0:
        var yt = fixture_i32(METRICS_FIXTURE_DIR + "weighted-yt.i32", WEIGHTED_N)
        var yp = fixture_i32(METRICS_FIXTURE_DIR + "weighted-yp.i32", WEIGHTED_N)
        var hit = Float64(0)
        var total = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            total += Float64(w[i])
            if yt[i] == yp[i]:
                hit += Float64(w[i])
        var want = hit / total
        var got = Float64(K.accuracy(yt, yp, w, WEIGHTED_N))
        require(isclose(got, want, 2e-6), "weighted accuracy " + String(got) + " != " + String(want))
    elif unit == 1:
        var y = fixture_f32(METRICS_FIXTURE_DIR + "weighted-y.f32", WEIGHTED_N)
        var yh = fixture_f32(METRICS_FIXTURE_DIR + "weighted-yh.f32", WEIGHTED_N)
        var swy = Float64(0)
        var sw = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            swy += Float64(y[i]) * Float64(w[i])
            sw += Float64(w[i])
        var avg = swy / sw
        var sse = Float64(0)
        var ssto = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            var d1 = y[i] - yh[i]
            sse += Float64(w[i]) * Float64(d1 * d1)
            var d2 = Float64(y[i]) - avg
            ssto += Float64(w[i]) * d2 * d2
        var want = 1.0 - sse / ssto
        var first = K.r2(y, yh, w, WEIGHTED_N)
        require(isclose(Float64(first), want, 1e-5), "weighted r2 " + String(first) + " != " + String(want))
        var again = K.r2(y, yh, w, WEIGHTED_N)
        require(bitcast[DType.uint32](again) == bitcast[DType.uint32](first), "weighted r2 repeat moved")
    else:
        var c = List[Float32](length=8, fill=Float32(2.0))
        var ones = List[Float32](length=8, fill=Float32(1.0))
        if unit == 2:
            require(K.r2(c, c, ones, 8) == 1.0, "constant target predicted exactly does not score 1")
        else:
            var off = List[Float32](length=8, fill=Float32(3.0))
            require(K.r2(c, off, ones, 8) == 0.0, "constant target predicted off by one does not score 0")
