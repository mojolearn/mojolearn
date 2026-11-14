# SPDX-License-Identifier: Apache-2.0
"""precision_recall_curve against the counts at every threshold (ported
from python/mojolearn/tests/test_metrics_selection_host_p2.py
`test_precision_recall_curve_matches_counts_at_each_threshold`).

The original 40 binary labels and scores (`np.random.default_rng(2)`:
labels, then `labels * 0.6 + uniform` in float32) are NumPy PCG64 output,
stored as raw words in `fixtures/metrics/prc-*` by
`tools/tsa_metrics_native_fixtures.py`. Through the column's
`precision_recall_curve_run` (the body the public call's door runs):

- precision and recall hold one more value than the thresholds, and end
  with precision 1 and recall 0;
- at every threshold t, with the prediction `score >= t` (float32),
  precision is tp / predicted and recall is tp / positives, to 1e-5
  relative.

The MAE / RMSE and precision / recall / F1 comparisons of the same file
are already independent native gates (metrics/checks/regression_errors_check.mojo
against a Float64 reference, metrics/checks/classification_metrics_check.mojo
against per-class integer counts, every average). Importable; no `main()`.
"""
from std.math import max
from checks.native_tests.tsa_metrics_fixture_words import METRICS_FIXTURE_DIR, fixture_f32, fixture_i32

comptime PRC_N = 40


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait PrcCalls:
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        ...


def approx(actual: Float64, expected: Float64, rel: Float64) -> Bool:
    """`actual == pytest.approx(expected, rel=rel)` (absolute floor 1e-12)."""
    return abs(actual - expected) <= max(rel * abs(expected), 1e-12)


def check_curve[K: PrcCalls]() raises:
    var y = fixture_i32(METRICS_FIXTURE_DIR + "prc-y.i32", PRC_N)
    var s = fixture_f32(METRICS_FIXTURE_DIR + "prc-s.f32", PRC_N)
    var precision = List[Float32](length=PRC_N + 1, fill=Float32(-1))
    var recall = List[Float32](length=PRC_N + 1, fill=Float32(-1))
    var thresholds = List[Float32](length=PRC_N + 1, fill=Float32(-1))
    var m = K.curve(y, s, PRC_N, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(precision.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(recall.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(thresholds.unsafe_ptr())))
    require(m >= 1 and m <= PRC_N, "precision_recall_curve threshold count " + String(m))
    require(precision[m] == 1.0 and recall[m] == 0.0, "the curve does not end at precision 1, recall 0")
    var positives = 0
    for i in range(PRC_N):  # small-loop(40 labels)
        positives += 1 if y[i] == 1 else 0
    for g in range(m):  # small-loop(one threshold of a 40-point curve)
        var t = thresholds[g]
        var tp = 0
        var predicted = 0
        for i in range(PRC_N):  # small-loop(40 points at one threshold)
            if s[i] >= t:
                predicted += 1
                tp += 1 if y[i] == 1 else 0
        require(approx(Float64(precision[g]), Float64(tp) / Float64(predicted), 1e-5),
                "precision at threshold " + String(g) + ": " + String(precision[g]))
        require(approx(Float64(recall[g]), Float64(tp) / Float64(positives), 1e-5),
                "recall at threshold " + String(g) + ": " + String(recall[g]))
