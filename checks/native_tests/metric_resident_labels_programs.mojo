# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The device encode + map of ONE resident label vector feeding a program slot
(lane vdf-xm-wire4, 2026-11-24): log_loss and the curve doors against the
host-input route and the CPU column, byte for byte.

A metric door inside `resident()` whose y_true is a resident signed label
vector (bindings/metric_resident_labels_door.mojo `metric_resident_labels`)
encodes it ONCE on the device (MS_ENCODE into a codes slot, the k sorted
classes back once), applies its O(k) per-class table there (MS_GATHER: the
index of each class in the door's label order, or the 0/1 flag of the positive
label) into a second slot, and the x_metrics program reads that slot in place
(`metric_slots_run`, device to device). The host-input route: the host codes
(the index of each row's label among the sorted classes, then the same table)
as host inputs of the GPU image. The CPU column: `metrics_program_host_v1`
over the same host codes.

  0 encode           int32 signed sparse labels (n = 600, k = 4): k and the
                     class words from MS_ENCODE equal the host sort
  1 log_loss multi   the typed builder `mc_log_loss_stages` (n x k proba,
                     the mapped codes, weights): code slot == GPU host input
                     == CPU column (the flag word and the binary64 result)
  2 log_loss binary  the flags table (class == positive) mapped on the device
  3 roc binary fold  `mc_roc_binary_stages` (the curve doors' fold) on the
                     device flags, weighted: the folded AUC's bits
  4 roc ovr micro    `roc_ovr_micro_run_slot` on the codes slot vs
                     `roc_ovr_micro_run` on host codes, both columns
  5 roc ovo          `roc_ovo_run_slot` vs `roc_ovo_run`
  6 whole-arena form case 1 with MOJOLEARN_ARENA_RANGES off (route `ranges`
                     False, `store` True): equal to the CPU column
  7 int64 labels     the same values as 8-byte labels: same k, same mapped codes
  8 miss row         a table marking a class -1 answers the first row of that
                     class, as the host gather; every slot freed (`XP_DEV_LIVE`)

What a sabotage changes here: a gather that writes the table index instead of
the table word, an encode that does not sort, a slot read as the label words
instead of the mapped codes, the whole-arena form uploading over a slot span.

Importable; `main()` only runs the suite. GPU only: a host-column build never
instantiates the device cases.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run, metric_slots_run
from bindings.mcls_classify_run import mc_log_loss_stages
from bindings.mcls_ranking_run import (
    McAuc, RocMicro, RocOvo, mc_fold_auc_value, mc_roc_binary_stages, roc_ovr_micro_run, roc_ovr_micro_run_slot,
    roc_ovo_run, roc_ovo_run_slot,
)
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.cast_in import CAST_KIND_SIGNED
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_LIVE
from x_prep.transport_device import x_prep_transport_device_v1
from metrics.slot_inputs import metric_slot_create
from metrics.slot_entry_device import metric_slot_run
from metrics.slot_ops import MS_ENCODE, MS_GATHER

comptime _N = 600
comptime _K = 4


def _class_value(c: Int) -> Int:
    # signed, sparse, ascending in c: the host code of value c is c
    return c * 37 - 500


def _codes_host() -> List[Int32]:
    var out = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # driver-loop: fixture codes (every class present)
        out[i] = Int32(((i * 7919) // 7 + i) % _K)
    return out^


def _labels32(codes: List[Int32]) -> List[Int32]:
    var out = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # driver-loop: fixture labels from the codes
        out[i] = Int32(_class_value(Int(codes[i])))
    return out^


def _labels64(codes: List[Int32]) -> List[Int64]:
    var out = List[Int64](length=_N, fill=Int64(0))
    for i in range(_N):  # driver-loop: the same values as 8-byte labels
        out[i] = Int64(_class_value(Int(codes[i])))
    return out^


def _mapped(codes: List[Int32], table: List[Int32]) -> List[Int32]:
    # the host gather the device gather must equal
    var out = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # driver-loop: the host map of the fixture codes
        out[i] = table[Int(codes[i])]
    return out^


def _proba(cols: Int) -> List[Float32]:
    # dyadic rows (exact in float32) that sum to exactly 1 for cols == _K
    var out = List[Float32](length=_N * cols, fill=Float32(0.0))
    for i in range(_N):  # driver-loop: fixture probability rows
        if cols == 1:
            out[i] = Float32(Float64((i * 13) % 15 + 1) / 16.0)
            continue
        var a = (i * 3) % 5 + 1
        var b = (i * 5) % 4 + 1
        var c = (i * 7) % 3 + 1
        out[i * cols + 0] = Float32(Float64(a) / 16.0)
        out[i * cols + 1] = Float32(Float64(b) / 16.0)
        out[i * cols + 2] = Float32(Float64(c) / 16.0)
        out[i * cols + 3] = Float32(Float64(16 - a - b - c) / 16.0)
    return out^


def _weights() -> List[Float32]:
    var w = List[Float32](length=_N, fill=Float32(0.0))
    for i in range(_N):  # driver-loop: fixture weights, fractional and exact in float32
        w[i] = Float32(Float64((i * 31) % 17 + 1) / 8.0)
    return w^


def _same(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # driver-loop: bitwise comparison of the output words
        require(a[i] == b[i], what + ": word " + String(i) + " differs")


def _same_auc(a: McAuc, b: McAuc, what: String) raises:
    require(a.defined == b.defined, what + ": defined differs")
    require(bitcast[DType.uint64](a.value) == bitcast[DType.uint64](b.value), what + ": value bits differ")


# ---------------------------------------------------------------- device steps


def _encode(store: NativeDescriptorCall, address: Int, item: Int, mut taken: List[Int]) raises -> Tuple[Int, Int, List[Int64]]:
    """MS_ENCODE of a resident label vector: (codes slot, k, the k class
    values); every slot appended to `taken`."""
    var put = native_descriptor_call(store, XP_DEV_PUT, [address], [_N * item // 4], [])
    var codes = metric_slot_create(_N)
    taken.extend([put, codes])
    var words = List[UInt8](length=_N * item, fill=UInt8(0))
    var addrs: List[Int] = [0]
    var slots: List[Int] = [put]
    var ints: List[Int] = [_N, CAST_KIND_SIGNED, item, codes]
    var outs: List[Int] = [Int(words.unsafe_ptr())]
    var values = List[Float64]()
    var k = metric_slot_run(MS_ENCODE, addrs, slots, ints, List[Float64](), outs, values)
    var classes = List[Int64](length=max(k, 0), fill=Int64(0))
    for c in range(k):  # driver-loop: the k class words back once
        if item == 4:
            classes[c] = Int64(words.unsafe_ptr().bitcast[Int32]()[c])
        else:
            classes[c] = words.unsafe_ptr().bitcast[Int64]()[c]
    _ = words^
    return (codes, k, classes^)


def _gather(codes: Int, table: List[Int32], mut taken: List[Int]) raises -> Tuple[Int, Int]:
    """MS_GATHER of a codes slot through a per-class table: (the mapped
    slot, the first row mapped to -1, or -1)."""
    var dst = metric_slot_create(_N)
    taken.append(dst)
    var addrs: List[Int] = [0]
    var slots: List[Int] = [codes]
    var ints: List[Int] = [_N, len(table), dst]
    var outs: List[Int] = [Int(table.unsafe_ptr())]
    var values = List[Float64]()
    var first = metric_slot_run(MS_GATHER, addrs, slots, ints, List[Float64](), outs, values)
    _ = table
    return (dst, first)


# ---------------------------------------------------------------- programs


def _host_route(entry: NativeDescriptorCall, ranges: Bool) -> MetricRoute:
    return MetricRoute(entry, ranges, ranges)


def _log_loss(route: MetricRoute, proba: List[Float32], codes_at: Int, codes_slot: Int, w: List[Float32],
              binary: Bool) raises -> List[Int32]:
    """The log_loss door's program (proba, codes, weights): the flag word
    and the binary64 result."""
    var plan = MetricProgramPlan()
    var k = 2 if binary else _K
    var put = mc_log_loss_stages(plan, _N, k, binary, True, True)
    var addresses: List[Int] = [Int(proba.unsafe_ptr()), codes_at, Int(w.unsafe_ptr())]
    var sizes: List[Int] = [len(proba), _N, _N]
    var slots = List[Int]()
    if codes_slot >= 0:
        slots = [-1, codes_slot, -1]
    var arena = metric_slots_run(route, plan, addresses, sizes, slots)
    _ = proba
    _ = w
    var out = arena.i32_words(put[0], 1)
    out.extend(arena.i32_words(put[1], 2))
    return out^


def _roc_binary(route: MetricRoute, scores: List[Float32], flags_at: Int, flags_slot: Int,
                w: List[Float32]) raises -> McAuc:
    """The curve doors' binary fold program (scores, flags, weights)."""
    var plan = MetricProgramPlan()
    var out = mc_roc_binary_stages(plan, _N, True, False, 0.0)
    var addresses: List[Int] = [Int(scores.unsafe_ptr()), flags_at, Int(w.unsafe_ptr())]
    var sizes: List[Int] = [_N, _N, _N]
    var slots = List[Int]()
    if flags_slot >= 0:
        slots = [-1, flags_slot, -1]
    var arena = metric_slots_run(route, plan, addresses, sizes, slots)
    _ = scores
    _ = w
    return mc_fold_auc_value(arena, out, False, 0.0)


def _device_cases() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():
        var store: NativeDescriptorCall = x_prep_transport_device_v1
        var live0 = native_descriptor_call(store, XP_DEV_LIVE, [], [], [])
        var cpu = _host_route(metrics_program_host_v1, False)
        var gpu = _host_route(metrics_program_device_v1, True)
        var slot_route = MetricRoute(metrics_program_device_v1, True, True, True)
        var whole_route = MetricRoute(metrics_program_device_v1, False, True, True)
        var codes = _codes_host()
        var y = _labels32(codes)
        var w = _weights()
        var taken = List[Int]()
        # case 0: the device encode
        var enc = _encode(store, Int(y.unsafe_ptr()), 4, taken)
        require(enc[1] == _K, "case 0: k " + String(enc[1]))
        for c in range(_K):  # driver-loop: the sorted classes
            require(Int(enc[2][c]) == _class_value(c), "case 0: class " + String(c))
        # case 1: log_loss multiclass through the door's class-order table (the sorted classes: identity)
        var order: List[Int32] = [0, 1, 2, 3]
        var mapped = _gather(enc[0], order, taken)
        require(mapped[1] == -1, "case 1: every row inside the label order")
        var host_codes = _mapped(codes, order)
        var proba = _proba(_K)
        var at = Int(host_codes.unsafe_ptr())
        var ll_cpu = _log_loss(cpu, proba, at, -1, w, False)
        _same(_log_loss(gpu, proba, at, -1, w, False), ll_cpu, "case 1: GPU host input vs CPU column")
        _same(_log_loss(slot_route, proba, 0, mapped[0], w, False), ll_cpu, "case 1: GPU code slot vs CPU column")
        # case 6: the whole-arena slot form
        _same(_log_loss(whole_route, proba, 0, mapped[0], w, False), ll_cpu, "case 6: ranges off, code slot")
        # case 2: log_loss binary, the positive flags (class 2 positive)
        var eq: List[Int32] = [0, 0, 1, 0]
        var flags = _gather(enc[0], eq, taken)
        var host_flags = _mapped(codes, eq)
        var p1 = _proba(1)
        var fa = Int(host_flags.unsafe_ptr())
        var lb_cpu = _log_loss(cpu, p1, fa, -1, w, True)
        _same(_log_loss(gpu, p1, fa, -1, w, True), lb_cpu, "case 2: GPU host input vs CPU column")
        _same(_log_loss(slot_route, p1, 0, flags[0], w, True), lb_cpu, "case 2: GPU flag slot vs CPU column")
        # case 3: the curve doors' binary fold on the device flags
        var scores = _proba(1)
        for i in range(_N):  # driver-loop: distinct-ish fixture scores with ties
            scores[i] = Float32(Float64((i * 37) % 101) / 128.0)
        var r_cpu = _roc_binary(cpu, scores, fa, -1, w)
        _same_auc(_roc_binary(gpu, scores, fa, -1, w), r_cpu, "case 3: GPU host input vs CPU column")
        _same_auc(_roc_binary(slot_route, scores, 0, flags[0], w), r_cpu, "case 3: GPU flag slot vs CPU column")
        # case 4: the micro one-vs-rest typed run on the codes slot
        var sa = Int(proba.unsafe_ptr())
        var m_cpu = roc_ovr_micro_run(cpu, sa, _N * _K, at, _N, Int(w.unsafe_ptr()), _N, _N, _K)
        var m_gpu = roc_ovr_micro_run(gpu, sa, _N * _K, at, _N, Int(w.unsafe_ptr()), _N, _N, _K)
        var m_slot = roc_ovr_micro_run_slot(slot_route, sa, _N * _K, 0, _N, mapped[0], Int(w.unsafe_ptr()), _N, _N, _K)
        require(m_cpu.rows_ok and m_gpu.rows_ok and m_slot.rows_ok, "case 4: the fixture rows sum to 1")
        _same_auc(m_gpu.auc, m_cpu.auc, "case 4: GPU host input vs CPU column")
        _same_auc(m_slot.auc, m_cpu.auc, "case 4: GPU code slot vs CPU column")
        # case 5: one-vs-one
        var o_cpu = roc_ovo_run(cpu, sa, at, _N, _K, True)
        var o_gpu = roc_ovo_run(gpu, sa, at, _N, _K, True)
        var o_slot = roc_ovo_run_slot(slot_route, sa, 0, mapped[0], _N, _K, True)
        require(o_cpu.status == o_slot.status and o_cpu.status == o_gpu.status, "case 5: status")
        require(o_cpu.undefined == o_slot.undefined, "case 5: one-class pair count")
        require(bitcast[DType.uint64](o_gpu.value) == bitcast[DType.uint64](o_cpu.value),
                "case 5: GPU host input vs CPU column")
        require(bitcast[DType.uint64](o_slot.value) == bitcast[DType.uint64](o_cpu.value),
                "case 5: GPU code slot vs CPU column")
        # case 7: int64 labels: the same k and mapped codes as the int32 vector
        var y64 = _labels64(codes)
        var enc64 = _encode(store, Int(y64.unsafe_ptr()), 8, taken)
        require(enc64[1] == _K, "case 7: int64 k")
        var mapped64 = _gather(enc64[0], order, taken)
        _same(_log_loss(slot_route, proba, 0, mapped64[0], w, False), ll_cpu, "case 7: int64 code slot vs CPU column")
        # case 8: the first row of a class outside the order (class 3 marked -1)
        var miss: List[Int32] = [0, 1, 2, -1]
        var first_host = -1
        for i in range(_N):  # driver-loop: the host gather's first miss
            if codes[i] == 3:
                first_host = i
                break
        require(_gather(enc[0], miss, taken)[1] == first_host, "case 8: first row outside the order")
        for i in range(len(taken)):  # driver-loop: free every slot this check made, newest first
            _ = native_descriptor_call(store, XP_DEV_FREE, [], [taken[len(taken) - 1 - i]], [])
        require(native_descriptor_call(store, XP_DEV_LIVE, [], [], []) == live0, "case 8: every slot freed")
        _ = y^
        _ = y64^
        _ = w^
        _ = codes^
        _ = host_codes^
        _ = host_flags^
        _ = proba^
        _ = p1^
        _ = scores^


def test_metric_resident_labels_programs() raises:
    _device_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
