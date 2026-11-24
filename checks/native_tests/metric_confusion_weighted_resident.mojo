# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The weighted confusion program on resident label slots (lane vdf-xm-wire3,
2026-11-24) against the host-input route and the CPU column, byte for byte.

confusion_matrix with sample_weight inside `resident()` (bindings/mcls_classify_door.mojo
`_confusion_weighted`): both label vectors are slots of the ONE store
(x_prep/store_slot.mojo), the device union step (MS_ENCODE_PAIR) writes their
codes into two slots, and the x_metrics program (the typed builder
bindings/mcls_classify_run.mojo `mc_confusion_weighted_stages`: pair key,
k*k weighted grouping, `_normalize`) reads the two code slots in place
(`metric_slots_run`, device to device) with the weights as a host input. The
host-input route: the host union encoder's codes
(core/metrics_cluster_table.mojo `cluster_union_codes`) as host inputs of the
GPU image. The CPU column: `metrics_program_host_v1` over the same host codes.

  0 normalize None   int32 labels, signed and sparse, 4,000 rows, fractional
                     weights: GPU code slots == GPU host input == CPU column
  1 normalize 'true' the same, row-normalized (mode 2)
  2 normalize 'all'  the same, the extra total stage (mode 1)
  3 int64 labels     the same values as 8-byte labels: same k, same words
  4 slots freed      every slot the check made is freed (`XP_DEV_LIVE`)

What a sabotage changes here: code slots copied from the wrong half of the
union step, a program that reads the slot's owner words instead of the codes,
a weights input laid out before the key scratch (a different arena: the
builder is one body for the door and this check).

Importable; `main()` only runs the suite. GPU only: a host-column build never
instantiates the device cases.
"""
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run, metric_slots_run
from bindings.mcls_classify_run import mc_confusion_weighted_stages
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.metrics_cluster_table import cluster_union_codes
from core.cast_in import CAST_KIND_SIGNED
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_LIVE
from x_prep.transport_device import x_prep_transport_device_v1
from metrics.slot_inputs import metric_slot_create
from metrics.slot_entry_device import metric_slot_run
from metrics.slot_ops import MS_ENCODE_PAIR

comptime _N = 4000


def _labels(seed: Int, classes: Int) -> List[Int32]:
    # signed, sparse values (not 0 .. k-1): the encode has work to do
    var out = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # driver-loop: fixture labels
        out[i] = Int32((((i * 7919 + seed * 104729) // 7) % classes) * 37 - 500)
    return out^


def _wide(labels: List[Int32]) -> List[Int64]:
    var out = List[Int64](length=_N, fill=Int64(0))
    for i in range(_N):  # driver-loop: the same values as 8-byte labels
        out[i] = Int64(labels[i])
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


struct _Prog(Movable):
    """The door's layout: the two codes inputs, then the builder (key
    scratch, weights input, grouping, normalize)."""
    var plan: MetricProgramPlan
    var out: Int

    def __init__(out self, k: Int, mode: Int) raises:
        self.plan = MetricProgramPlan()
        var a = self.plan.input(_N)
        var b = self.plan.input(_N)
        var got = mc_confusion_weighted_stages(self.plan, a, b, _N, k, mode)
        self.out = got[1]


def _host_run(entry: NativeDescriptorCall, ranges: Bool, codes: List[Int32], w: List[Float32], k: Int,
              mode: Int) raises -> List[Int32]:
    var p = _Prog(k, mode)
    var base = Int(codes.unsafe_ptr())
    var addresses: List[Int] = [base, base + 4 * _N, Int(w.unsafe_ptr())]
    var sizes: List[Int] = [_N, _N, _N]
    var arena = metric_inputs_run(MetricRoute(entry, ranges, ranges), p.plan, addresses, sizes)
    _ = codes
    _ = w
    return arena.i32_words(p.out, 2 * k * k)


def _slot_run(ct: Int, cp: Int, w: List[Float32], k: Int, mode: Int) raises -> List[Int32]:
    var p = _Prog(k, mode)
    var addresses: List[Int] = [0, 0, Int(w.unsafe_ptr())]
    var sizes: List[Int] = [_N, _N, _N]
    var slots: List[Int] = [ct, cp, -1]
    var route = MetricRoute(metrics_program_device_v1, True, True, True)
    var arena = metric_slots_run(route, p.plan, addresses, sizes, slots)
    _ = w
    return arena.i32_words(p.out, 2 * k * k)


def _encode_pair(store: NativeDescriptorCall, t_addr: Int, p_addr: Int, item: Int,
                 mut taken: List[Int]) raises -> Tuple[Int, Int, Int]:
    """The device union step over two resident label slots: (codes_t slot,
    codes_p slot, k); every slot appended to `taken`."""
    var words = _N * item // 4
    var st = native_descriptor_call(store, XP_DEV_PUT, [t_addr], [words], [])
    var sp = native_descriptor_call(store, XP_DEV_PUT, [p_addr], [words], [])
    var ct = metric_slot_create(_N)
    var cp = metric_slot_create(_N)
    taken.extend([st, sp, ct, cp])
    var addrs: List[Int] = [0, 0]
    var slots: List[Int] = [st, sp]
    var ints: List[Int] = [_N, CAST_KIND_SIGNED, item, ct, cp]
    var outs: List[Int] = [0]
    var values = List[Float64]()
    var k = metric_slot_run(MS_ENCODE_PAIR, addrs, slots, ints, List[Float64](), outs, values)
    return (ct, cp, k)


def _device_cases() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():
        var store: NativeDescriptorCall = x_prep_transport_device_v1
        var live0 = native_descriptor_call(store, XP_DEV_LIVE, [], [], [])
        var yt = _labels(1, 7)
        var yp = _labels(2, 5)
        var w = _weights()
        var codes = List[Int32]()
        var k = cluster_union_codes(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), _N, codes)
        require(k >= 2 and k <= 4096, "fixture: a union of 2 .. 4096 classes")
        var taken = List[Int]()
        var pair = _encode_pair(store, Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, taken)
        require(pair[2] == k, "union k: device " + String(pair[2]) + " vs host " + String(k))
        var modes: List[Int] = [4, 2, 1]
        for j in range(len(modes)):  # driver-loop: cases 0-2, the normalize modes
            var mode = modes[j]
            var cpu = _host_run(metrics_program_host_v1, False, codes, w, k, mode)
            var gpu_host = _host_run(metrics_program_device_v1, True, codes, w, k, mode)
            var gpu_slot = _slot_run(pair[0], pair[1], w, k, mode)
            _same(gpu_host, cpu, "case " + String(j) + ": GPU host input vs CPU column")
            _same(gpu_slot, cpu, "case " + String(j) + ": GPU code slots vs CPU column")
        # case 3: the same values as 8-byte labels
        var wt = _wide(yt)
        var wp = _wide(yp)
        var wide = _encode_pair(store, Int(wt.unsafe_ptr()), Int(wp.unsafe_ptr()), 8, taken)
        require(wide[2] == k, "case 3: int64 union k")
        _same(_slot_run(wide[0], wide[1], w, k, 4), _host_run(metrics_program_host_v1, False, codes, w, k, 4),
              "case 3: int64 code slots vs CPU column")
        # case 4: every slot freed
        for i in range(len(taken)):  # driver-loop: free every slot this check made, newest first
            _ = native_descriptor_call(store, XP_DEV_FREE, [], [taken[len(taken) - 1 - i]], [])
        require(native_descriptor_call(store, XP_DEV_LIVE, [], [], []) == live0, "case 4: every slot freed")
        _ = yt^
        _ = yp^
        _ = wt^
        _ = wp^
        _ = w^
        _ = codes^


def test_metric_confusion_weighted_resident() raises:
    _device_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
