# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The x_metrics resident transport (lane vdf-xm-transport, 2026-11-24): a
metric program whose X input is a store slot reads the same bytes as the
host-input path and as the CPU column.

One program (the clustering metrics' grouped sums: `group` of X's d columns
by a label vector, bindings/metrics_exp_cluster_door.mojo) over a small
fixture (n = 37 rows of d = 5 float32 words with fractional and signed values,
k = 3 labels) runs four ways; the wanted output words must be bitwise equal:

    case 0  CPU column (`metrics_program_host_v1`), host inputs
    case 1  GPU image, host inputs through the range route
    case 2  GPU image, X put into the image's store (`MP_DEV_PUT`) and read
            as a slot (`metric_slots_run`, `upload_ranges` device to device)
    case 3  GPU image, X the rows [idx] of a 53-row base gathered on the
            device (`MP_DEV_PUT` of the base, `MP_DEV_TAKE_ROWS`), the host
            path reading the same rows gathered on the host
    case 4  the slot scans: `MP_DEV_FIRST_NONFINITE` is -1 on the fixture and
            the index of a planted NaN; `MP_DEV_FIRST_NEGATIVE` the first
            negative word; every slot freed (`MP_DEV_LIVE` back to its start)

Importable; `main()` only runs the suite. GPU only: a host-column build never
instantiates the device cases.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run, metric_slots_run
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from x_metrics.descriptor_schema import (
    MP_CAPABILITIES, MP_HAS_STORE, MP_DEV_PUT, MP_DEV_FREE, MP_DEV_TAKE_ROWS, MP_DEV_LIVE,
    MP_DEV_FIRST_NONFINITE, MP_DEV_FIRST_NEGATIVE,
)

comptime _N = 37
comptime _D = 5
comptime _K = 3
comptime _BASE_ROWS = 53


def _value(i: Int) -> Float32:
    # fractional, signed, distinct words (no NaN): exact float32 values
    return Float32(Float64((i * 7919) % 201 - 100) / 8.0 + Float64(i % 3) / 1024.0)


def _base() -> List[Float32]:
    var words = List[Float32](length=_BASE_ROWS * _D, fill=0.0)
    for i in range(_BASE_ROWS * _D):  # driver-loop: fixture words
        words[i] = _value(i)
    return words^


def _indices() -> List[Int64]:
    var idx = List[Int64](length=_N, fill=0)
    for r in range(_N):  # driver-loop: fixture row selection
        idx[r] = Int64((r * 17 + 5) % _BASE_ROWS)
    return idx^


def _gather_host(base: List[Float32], idx: List[Int64]) -> List[Float32]:
    var rows = List[Float32](length=_N * _D, fill=0.0)
    for r in range(_N):  # driver-loop: the host gather the device gather must equal
        for c in range(_D):  # driver-loop: one row's words
            rows[r * _D + c] = base[Int(idx[r]) * _D + c]
    return rows^


def _labels() -> List[Int32]:
    var lab = List[Int32](length=_N, fill=0)
    for r in range(_N):  # driver-loop: fixture labels
        lab[r] = Int32((r * 5) % _K)
    return lab^


struct _Prog(Movable):
    var plan: MetricProgramPlan
    var out: Int
    var words: Int

    def __init__(out self) raises:
        self.plan = MetricProgramPlan()
        var X0 = self.plan.input(_N * _D)
        var L = self.plan.input(_N)
        var g = self.plan.group(L, _N, _K, X0, _D, -1, _D)
        self.words = _K * _D
        self.out = self.plan.want(g[1], self.words, -1, 1)


def _words(arena: MetricArenaOwner, off: Int, count: Int) -> List[Int32]:
    return arena.i32_words(off, count)


def _same(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # driver-loop: bitwise comparison of the output words
        require(a[i] == b[i], what + ": word " + String(i) + " differs")


def _host_run(entry: NativeDescriptorCall, ranges: Bool, x: List[Float32], lab: List[Int32]) raises -> List[Int32]:
    var p = _Prog()
    var addresses: List[Int] = [Int(x.unsafe_ptr()), Int(lab.unsafe_ptr())]
    var sizes: List[Int] = [_N * _D, _N]
    var arena = metric_inputs_run(MetricRoute(entry, ranges, ranges), p.plan, addresses, sizes)
    _ = x
    _ = lab
    return _words(arena, p.out, p.words)


def _slot_run(x_slot: Int, lab: List[Int32]) raises -> List[Int32]:
    var p = _Prog()
    var addresses: List[Int] = [0, Int(lab.unsafe_ptr())]
    var sizes: List[Int] = [_N * _D, _N]
    var slots: List[Int] = [x_slot, -1]
    var route = MetricRoute(metrics_program_device_v1, True, True, True)
    var arena = metric_slots_run(route, p.plan, addresses, sizes, slots)
    _ = lab
    return _words(arena, p.out, p.words)


def _device_cases() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():
        var entry: NativeDescriptorCall = metrics_program_device_v1
        require((native_descriptor_call(entry, MP_CAPABILITIES, [], [], []) & MP_HAS_STORE) != 0,
                "the GPU image publishes its resident store")
        var live0 = native_descriptor_call(entry, MP_DEV_LIVE, [], [], [])
        var base = _base()
        var idx = _indices()
        var x = _gather_host(base, idx)
        var lab = _labels()
        var cpu = _host_run(metrics_program_host_v1, False, x, lab)            # case 0
        var gpu_host = _host_run(entry, True, x, lab)                          # case 1
        _same(gpu_host, cpu, "case 1: GPU host input vs CPU column")
        var put = native_descriptor_call(entry, MP_DEV_PUT, [Int(x.unsafe_ptr())], [_N * _D], [])
        _same(_slot_run(put, lab), cpu, "case 2: GPU slot input vs CPU column")  # case 2
        var base_slot = native_descriptor_call(entry, MP_DEV_PUT, [Int(base.unsafe_ptr())], [_BASE_ROWS * _D], [])
        var rows = native_descriptor_call(entry, MP_DEV_TAKE_ROWS, [Int(idx.unsafe_ptr())], [base_slot, _D, _N], [])
        _same(_slot_run(rows, lab), cpu, "case 3: GPU gathered rows vs CPU column")  # case 3
        # case 4: the slot scans
        require(native_descriptor_call(entry, MP_DEV_FIRST_NONFINITE, [], [rows, _N * _D], []) == -1,
                "case 4: no NaN in the fixture")
        var first_negative = -1
        for i in range(_N * _D):  # driver-loop: the host answer of the sign scan
            if x[i] < 0.0:
                first_negative = i
                break
        require(native_descriptor_call(entry, MP_DEV_FIRST_NEGATIVE, [], [rows, _N * _D], []) == first_negative,
                "case 4: first negative word")
        var planted = x.copy()
        planted[23] = bitcast[DType.float32](UInt32(0x7FC00000))
        var nan_slot = native_descriptor_call(entry, MP_DEV_PUT, [Int(planted.unsafe_ptr())], [_N * _D], [])
        require(native_descriptor_call(entry, MP_DEV_FIRST_NONFINITE, [], [nan_slot, _N * _D], []) == 23,
                "case 4: the planted NaN")
        var taken: List[Int] = [nan_slot, rows, base_slot, put]
        for i in range(len(taken)):  # driver-loop: free every slot this check put
            _ = native_descriptor_call(entry, MP_DEV_FREE, [], [taken[i]], [])
        require(native_descriptor_call(entry, MP_DEV_LIVE, [], [], []) == live0, "case 4: every slot freed")
        _ = base^
        _ = idx^
        _ = planted^


def test_x_metrics_resident_transport() raises:
    _device_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
