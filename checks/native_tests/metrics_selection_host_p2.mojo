# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
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

`test_binary_ranking` (lane py2mojo-D, batch 5, 2026-11-22; ported from checks/binary_ranking_smoke.py): the same
column bodies (`precision_recall_curve_run`, and the AUC body of the column's roc_auc_score binding:
`binary_ranking_checked(..., False)` on the host, `binary_ranking_host[False]` on the device) against the smoke's
independent oracle written here: the curve at every distinct score in ascending order (precision tp / selected,
recall tp / positives or 1 with no positive, then 1 and 0), the thresholds equal as values (subnormals kept, the
sign of zero not), and the AUC as the pairwise win (1) / tie (1/2) credit over positives x negatives. The original's
eight cases, with the labels already encoded as the door encodes them (`y == pos_label`): a plain case, all ties,
the signed zeros and the smallest subnormals, the string and the 2^62 labels (now 0/1 codes), 513 rows of 19 tied
values, one-class positive and negative (no AUC). Tolerances are the original's: rtol 3e-6, atol 1e-7. The label
encoding and the `fast` arm stay in Python.
"""
from std.math import max
from std.memory import bitcast
from metrics.host.metrics_host_entries_impl import binary_ranking_checked
from metrics.estimator import binary_ranking_host
from checks.native_tests.testsuite_cases import check_cases
from checks.native_tests.tsa_metrics_fixture_words import METRICS_FIXTURE_DIR, fixture_f32, fixture_i32
from metrics.mojolearn_metrics_host_run import precision_recall_curve_run
from metrics.mojolearn_metrics_run import precision_recall_curve_run as precision_recall_curve_run_gpu
from std.testing import TestSuite, assert_true as require
from std.sys.compile import is_defined


comptime PRC_N = 40


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


# ---- The host column (was metrics_selection_host_p2_cpu_suite.mojo) ----
# precision_recall_curve on the host column (`metrics/mojolearn_metrics_host_run.mojo`).
#
# One fixture: a single curve over 40 points, nothing independent to split
# across cores (the per-threshold comparison is 40 x 40 integer counts).
struct HostPrc(PrcCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run(y, scores, n, precision, recall, thresholds)


def _run_cpu() raises:
    check_curve[HostPrc]()
    print("PASS metrics_selection_host_p2_cpu: precision_recall_curve counts at every threshold")


# ---- The device column (was metrics_selection_host_p2_gpu_suite.mojo) ----
# precision_recall_curve on the device column (`metrics/mojolearn_metrics_run.mojo`).
#
# One fixture on the process's one metrics device context; the ranking keeps
# its own device parallelism.
struct DevicePrc(PrcCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run_gpu(y, scores, n, precision, recall, thresholds)


def _run_gpu() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        check_curve[DevicePrc]()
        print("PASS metrics_selection_host_p2_gpu: precision_recall_curve counts at every threshold")


def test_precision_recall_curve() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _run_cpu()
    else:
        _run_gpu()


# ---- test_binary_ranking (from checks/binary_ranking_smoke.py) ----

comptime RANKING_CASES = 8


trait RankCalls(PrcCalls):
    @staticmethod
    def auc(y: List[Int32], scores: List[Float32], n: Int) raises -> Float32:
        ...


struct HostRank(RankCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run(y, scores, n, precision, recall, thresholds)

    @staticmethod
    def auc(y: List[Int32], scores: List[Float32], n: Int) raises -> Float32:
        # bindings/_mojolearn_metrics_host.mojo roc_auc_score_binding
        return binary_ranking_checked(y, scores, n, False)[0][0]


struct DeviceRank(RankCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run_gpu(y, scores, n, precision, recall, thresholds)

    @staticmethod
    def auc(y: List[Int32], scores: List[Float32], n: Int) raises -> Float32:
        # bindings/_mojolearn_metrics.mojo roc_auc_score_binding
        return binary_ranking_host[False](y, scores, n)[0][0]


def ranking_case(which: Int, mut y: List[Int32], mut s: List[Float32]):
    """Case `which` of the smoke, its labels encoded as `y == pos_label`."""
    var tiny = bitcast[DType.float32](UInt32(1))  # np.nextafter(0, 1) in float32
    if which == 0:
        y = [0, 0, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8]
    elif which == 1:
        y = [0, 1, 0, 1]
        s = [2.0, 2.0, 2.0, 2.0]
    elif which == 2:
        y = [0, 1, 0, 1, 0, 1]
        s = [-0.0, 0.0, -tiny, tiny, -2.0, 2.0]
    elif which == 3:
        y = [0, 1, 0, 1]  # ["a", "z", "a", "z"], pos_label "z"
        s = [-4.0, -3.0, -2.0, -1.0]
    elif which == 4:
        y = [0, 1, 1]  # [2**62, 2**62 + 1, 2**62 + 1], pos_label 2**62 + 1
        s = [3.0, 2.0, 1.0]
    elif which == 5:
        y = List[Int32](length=513, fill=Int32(0))
        s = List[Float32](length=513, fill=Float32(0))
        for i in range(513):  # small-loop(the smoke's 513 literal rows)
            y[i] = Int32(i % 2)
            s[i] = Float32((i * 37) % 19 - 9)
    elif which == 6:
        y = [1, 1, 1]
        s = [0.0, 1.0, 2.0]
    else:
        y = [0, 0, 0]
        s = [0.0, 1.0, 2.0]


def all_close(actual: Float32, expected: Float64, label: String) raises:
    """numpy.testing.assert_allclose(rtol=3e-6, atol=1e-7) for one value."""
    require(abs(Float64(actual) - expected) <= 1e-7 + 3e-6 * abs(expected),
            label + ": " + String(actual) + " expected " + String(expected))


def check_ranking_case[K: RankCalls](which: Int) raises:
    var y = List[Int32]()
    var s = List[Float32]()
    ranking_case(which, y, s)
    var n = len(y)
    var what = "binary ranking case " + String(which)
    # the oracle's thresholds: the distinct values, ascending (-0.0 == 0.0)
    var thresholds = List[Float64]()
    for i in range(n):  # small-loop(oracle over at most 513 scores)
        var v = Float64(s[i])
        var at = 0
        while at < len(thresholds) and thresholds[at] < v:
            at += 1
        if at < len(thresholds) and thresholds[at] == v:
            continue
        thresholds.append(v)
        var j = len(thresholds) - 1
        while j > at:  # small-loop(insertion into the sorted oracle thresholds)
            thresholds[j] = thresholds[j - 1]
            j -= 1
        thresholds[at] = v
    var positives = 0
    for i in range(n):  # small-loop(oracle)
        positives += Int(y[i])
    var precision = List[Float32](length=n + 1, fill=Float32(-1))
    var recall = List[Float32](length=n + 1, fill=Float32(-1))
    var got_t = List[Float32](length=n + 1, fill=Float32(-1))
    var m = K.curve(y, s, n, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(precision.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(recall.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(got_t.unsafe_ptr())))
    require(m == len(thresholds), what + ": " + String(m) + " thresholds, the oracle " + String(len(thresholds)))
    for g in range(m):  # small-loop(one threshold of the oracle curve)
        var t = thresholds[g]
        require(Float64(got_t[g]) == t, what + ": threshold " + String(g) + " is " + String(got_t[g]) + " expected "
                + String(t))
        var tp = 0
        var selected = 0
        for i in range(n):  # small-loop(the scores at one threshold)
            if Float64(s[i]) >= t:
                selected += 1
                tp += Int(y[i])
        all_close(precision[g], Float64(tp) / Float64(selected), what + " precision " + String(g))
        all_close(recall[g], Float64(tp) / Float64(positives) if positives > 0 else 1.0, what + " recall " + String(g))
    all_close(precision[m], 1.0, what + " final precision")
    all_close(recall[m], 0.0, what + " final recall")
    if positives > 0 and positives < n:
        var credit = Float64(0)
        for i in range(n):  # small-loop(pairwise oracle: positives)
            if y[i] != 1:
                continue
            for j in range(n):  # small-loop(pairwise oracle: negatives)
                if y[j] == 1:
                    continue
                if s[i] > s[j]:
                    credit += 1.0
                elif s[i] == s[j]:
                    credit += 0.5
        var want = credit / Float64(positives * (n - positives))
        var got = Float64(K.auc(y, s, n))
        require(abs(got - want) <= max(3e-6 * max(abs(got), abs(want)), 1e-7),
                what + ": roc_auc " + String(got) + " expected " + String(want))


def _host_rank_unit(which: Int) raises:
    check_ranking_case[HostRank](which)


def _run_ranking_cpu() raises:
    # the eight independent cases across the host cores
    check_cases("binary_ranking", RANKING_CASES, _host_rank_unit)


def _run_ranking_gpu() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        var failed = String("")
        for which in range(RANKING_CASES):  # driver-loop: over the cases on the one device context
            try:
                check_ranking_case[DeviceRank](which)
            except error:
                failed += "\n      " + String(error)
        if failed.byte_length() > 0:
            raise Error("binary ranking failed:" + failed)


def test_binary_ranking() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _run_ranking_cpu()
    else:
        _run_ranking_gpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
