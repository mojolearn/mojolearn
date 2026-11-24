# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The device-input form of the `_mojolearn_metrics` kernel entries
(metrics/slot_entry_device.mojo, lane vdf-xm-kernels 2026-11-24) against
the host-input form and the CPU column, byte for byte.

Each case puts its inputs into the resident store (x_prep/store_slot.mojo,
the store `resident()` and DeviceRows use) and runs the SAME operation three
ways: every input a slot (read in place), every input a host address (the
form a door takes outside `resident()`), and the CPU column's host oracle
(metrics/host/*_oracle.mojo, the `_mojolearn_metrics_host` bodies). The
three answers must have the same bits.

  0 accuracy        raw int32 labels, 70,000 rows
  1 rand index      raw int32 labels
  2 entropy         int32 labels over [lower, upper]
  3 r2              float32, 70,000 values
  4 kl divergence   float32 positive
  5 mse / mae / rmse
  6 roc auc         0/1 labels, float32 scores with ties
  7 refusals        a NaN in a slot input (mse) and a label 2 (roc) are
                    refused on the device with the host entries' words; a
                    roc auc with one class is refused

What a sabotage changes here: a view over the wrong slot or a slot read as
the wrong width (every case), a scan that misses a NaN or a label out of
range, or flags a good value (case 7, cases 5-6).
"""
from std.testing import TestSuite
from std.memory import bitcast

from x_prep.store_slot import X_PREP_STORE, x_prep_ctx
from metrics.slot_entry_device import metric_slot_run
from metrics.slot_ops import MS_ACCURACY, MS_RAND, MS_ENTROPY, MS_R2, MS_KL, MS_MSE, MS_MAE, MS_RMSE, MS_ROC_AUC
from metrics.host.metrics_oracle import host_accuracy_score, host_r2_score, host_entropy
from metrics.host.classification_oracle import (
    host_rand_score, host_kl_divergence, host_regression_error, host_binary_ranking,
)
from checks.native_tests.testsuite_cases import require

comptime MSE_CASES = 8
comptime N = 70000


def _put_f32(values: List[Float32]) raises -> Int:
    return X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), Int(values.unsafe_ptr()), len(values))


def _put_i32(values: List[Int32]) raises -> Int:
    return X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), Int(values.unsafe_ptr()), len(values))


def _free(id: Int) raises:
    X_PREP_STORE.get_or_create_ptr()[].free(x_prep_ctx(), id)


def _labels(seed: Int, classes: Int) -> List[Int32]:
    var out = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # fixture
        out[i] = Int32(((i * 7919 + seed * 104729) // 7) % classes)
    return out^


def _floats(seed: Int, positive: Bool) -> List[Float32]:
    var out = List[Float32](length=N, fill=Float32(0.0))
    for i in range(N):  # fixture: a few hundred distinct values, so ties occur
        var v = Float32(((i * 2654435761 + seed * 40503) % 997)) / Float32(97.0)
        out[i] = v + Float32(0.001) if positive else v - Float32(5.0)
    return out^


def _two(op: Int, a_addr: Int, b_addr: Int, a_slot: Int, b_slot: Int, ints: List[Int]) raises -> Float64:
    var addrs: List[Int] = [a_addr, b_addr]
    var slots: List[Int] = [a_slot, b_slot]
    var values = List[Float64]()
    _ = metric_slot_run(op, addrs, slots, ints, List[Float64](), List[Int](), values)
    return values[0]


def _same(name: String, got: Float64, want: Float64) raises:
    require(bitcast[DType.uint64](got) == bitcast[DType.uint64](want),
            name + ": " + String(got) + " vs " + String(want) + " (bits differ)")


def _pair_case(op: Int, name: String, a_i: List[Int32], b_i: List[Int32], a_f: List[Float32], b_f: List[Float32],
               labels: Bool, ints: List[Int], want_cpu: Float64) raises:
    var sa: Int
    var sb: Int
    var aa: Int
    var ab: Int
    if labels:
        sa = _put_i32(a_i)
        sb = _put_i32(b_i)
        aa = Int(a_i.unsafe_ptr())
        ab = Int(b_i.unsafe_ptr())
    else:
        sa = _put_f32(a_f)
        sb = _put_f32(b_f)
        aa = Int(a_f.unsafe_ptr())
        ab = Int(b_f.unsafe_ptr())
    try:
        var device = _two(op, 0, 0, sa, sb, ints)
        var host = _two(op, aa, ab, -1, -1, ints)
        var mixed = _two(op, aa, 0, -1, sb, ints)
        _same(name + " slot vs host input", device, host)
        _same(name + " mixed vs host input", mixed, host)
        _same(name + " slot vs CPU column", device, want_cpu)
    finally:
        _free(sa)
        _free(sb)
    _ = len(a_i)
    _ = len(b_i)
    _ = len(a_f)
    _ = len(b_f)


def _refused(op: Int, a_i: List[Int32], a_f: List[Float32], b_f: List[Float32], labels_first: Bool,
             words: String) raises:
    var sa = _put_i32(a_i) if labels_first else _put_f32(a_f)
    var sb = _put_f32(b_f)
    var ints: List[Int] = [N]
    var raised = False
    try:
        _ = _two(op, 0, 0, sa, sb, ints)
    except problem:
        raised = True
        require(String(problem) == words, "refusal words: got '" + String(problem) + "', want '" + words + "'")
    finally:
        _free(sa)
        _free(sb)
    require(raised, "not refused: " + words)
    _ = len(a_i)
    _ = len(a_f)
    _ = len(b_f)


def _case(which: Int) raises:
    var none_i = List[Int32]()
    var none_f = List[Float32]()
    var n_only: List[Int] = [N]
    if which == 0:
        var a = _labels(1, 5)
        var b = _labels(2, 5)
        _pair_case(MS_ACCURACY, "accuracy", a, b, none_f, none_f, True, n_only, Float64(host_accuracy_score(a, b, N)))
    elif which == 1:
        var a = _labels(3, 9)
        var b = _labels(4, 7)
        _pair_case(MS_RAND, "rand", a, b, none_f, none_f, True, n_only, host_rand_score(a, b, N))
    elif which == 2:
        var a = _labels(5, 11)
        var s = _put_i32(a)
        try:
            var ints: List[Int] = [N, 0, 10]
            var addrs: List[Int] = [0]
            var slots: List[Int] = [s]
            var values = List[Float64]()
            _ = metric_slot_run(MS_ENTROPY, addrs, slots, ints, List[Float64](), List[Int](), values)
            var host_addrs: List[Int] = [Int(a.unsafe_ptr())]
            var host_slots: List[Int] = [-1]
            var host_values = List[Float64]()
            _ = metric_slot_run(MS_ENTROPY, host_addrs, host_slots, ints, List[Float64](), List[Int](), host_values)
            _same("entropy slot vs host input", values[0], host_values[0])
            _same("entropy slot vs CPU column", values[0], host_entropy(a, N, Int32(0), Int32(10)))
        finally:
            _free(s)
        _ = len(a)
    elif which == 3:
        var a = _floats(1, False)
        var b = _floats(2, False)
        _pair_case(MS_R2, "r2", none_i, none_i, a, b, False, n_only, Float64(host_r2_score(a, b, N)))
    elif which == 4:
        var a = _floats(3, True)
        var b = _floats(4, True)
        _pair_case(MS_KL, "kl", none_i, none_i, a, b, False, n_only, Float64(host_kl_divergence(a, b, N)))
    elif which == 5:
        var a = _floats(5, False)
        var b = _floats(6, False)
        _pair_case(MS_MSE, "mse", none_i, none_i, a, b, False, n_only,
                   Float64(host_regression_error(a, b, N, False, False)))
        _pair_case(MS_MAE, "mae", none_i, none_i, a, b, False, n_only,
                   Float64(host_regression_error(a, b, N, True, False)))
        _pair_case(MS_RMSE, "rmse", none_i, none_i, a, b, False, n_only,
                   Float64(host_regression_error(a, b, N, False, True)))
    elif which == 6:
        var y = _labels(7, 2)
        var s = _floats(8, False)
        var cpu = host_binary_ranking(y, s, N, False)
        var ys = _put_i32(y)
        var ss = _put_f32(s)
        try:
            var device = _two(MS_ROC_AUC, 0, 0, ys, ss, n_only)
            var host = _two(MS_ROC_AUC, Int(y.unsafe_ptr()), Int(s.unsafe_ptr()), -1, -1, n_only)
            _same("roc auc slot vs host input", device, host)
            _same("roc auc slot vs CPU column", device, Float64(cpu[0][0]))
        finally:
            _free(ys)
            _free(ss)
        _ = len(y)
        _ = len(s)
    else:
        var a = _floats(9, False)
        var b = _floats(10, False)
        b[60001] = bitcast[DType.float32](UInt32(0x7FC00000))
        _refused(MS_MSE, none_i, a, b, False, "regression_error: inputs must be finite Float32")
        var y = _labels(11, 2)
        var s = _floats(12, False)
        y[65000] = 2
        _refused(MS_ROC_AUC, y, none_f, s, True, "binary ranking: labels must encode 0 or 1")
        var one = List[Int32](length=N, fill=Int32(1))
        _refused(MS_ROC_AUC, one, none_f, s, True, "roc_auc_score: both classes required")


def test_cases() raises:
    var failed = String("")
    for which in range(MSE_CASES):  # driver-loop: over the check's cases, one device queue
        try:
            _case(which)
        except problem:
            failed += "\n      metric_slot_entries row " + String(which) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("metric_slot_entries: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
