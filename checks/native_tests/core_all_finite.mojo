# SPDX-License-Identifier: Apache-2.0
"""The finiteness predicate behind `_buffer.all_finite`, on the literal vectors
of test_numpy_free_core.py `test_all_finite_host_path` (its native verdicts).

Each case is one literal vector and its expected verdict (1 finite, 0 not):

  0  f32 [1.0, 1e-45, -0.0, 3.4e38]                     -> 1
  1  case 0 widened to f64 (Array.astype("<f8"))        -> 1
  2-4  f32 [1.0, bad, 2.0], bad = NaN, +Inf, -Inf       -> 0
  5-7  the same widened to f64                          -> 0
  8  f64 [1.7e308, 1.7e308, -1.7e308] (sum overflows)   -> 1
  9  f64 [+Inf, -Inf] (sum is NaN)                      -> 0

The float32 values are the exact words Python's `Array.from_list(..., "<f4")`
stores (1e-45 is the smallest subnormal, 3.4e38 rounds to 0x7F7FC99E). The
host column calls the host bindings' `all_finite_f32` / `all_finite_f64`
(core/host_helpers_impl.mojo, bindings/host_helpers_pack.mojo); the device
column calls the base binding's `LabelsDeviceKernels.finite`
(bindings/_mojolearn.mojo `all_finite_f32_binding`). The empty-Array shortcut
and the integer-dtype TypeError stay in the Python test (boundary).
"""
from std.memory import bitcast

from core.host_helpers_impl import all_finite_f32, all_finite_f64
from core.hostptr import f32_ptr, f64_ptr
from core.labels_native_device import LabelsDeviceKernels
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require

comptime ALL_FINITE_CASES = 10


def _f32(word: UInt32) -> Float32:
    return bitcast[DType.float32](word)


def _f64(word: UInt64) -> Float64:
    return bitcast[DType.float64](word)


def _bad32(kind: Int) -> Float32:
    var words: List[UInt32] = [0x7FC00000, 0x7F800000, 0xFF800000]  # NaN, +Inf, -Inf
    return _f32(words[kind])


def _ok32() -> List[Float32]:
    return [_f32(0x3F800000), _f32(0x00000001), _f32(0x80000000), _f32(0x7F7FC99E)]


def _widen(values: List[Float32]) -> List[Float64]:
    var wide = List[Float64](length=len(values), fill=0.0)
    for i in range(len(values)):
        wide[i] = Float64(values[i])
    return wide^


def _verdict32(values: List[Float32]) raises -> Int:
    var address = Int(values.unsafe_ptr())
    var result: Int
    comptime if CPU_COLUMN_BUILD:
        result = all_finite_f32(f32_ptr(address), len(values))
    else:
        result = LabelsDeviceKernels.finite(address, len(values), False)
    return result


def _verdict64(values: List[Float64]) raises -> Int:
    var address = Int(values.unsafe_ptr())
    var result: Int
    comptime if CPU_COLUMN_BUILD:
        result = all_finite_f64(f64_ptr(address), len(values))
    else:
        result = LabelsDeviceKernels.finite(address, len(values), True)
    return result


def run_case(index: Int) raises:
    var label = "all_finite case " + String(index)
    if index == 0:
        require(_verdict32(_ok32()) == 1, label + ": finite f32 refused")
    elif index == 1:
        require(_verdict64(_widen(_ok32())) == 1, label + ": finite f64 refused")
    elif index < 8:
        var kind = (index - 2) % 3
        var values: List[Float32] = [_f32(0x3F800000), _bad32(kind), _f32(0x40000000)]
        if index < 5:
            require(_verdict32(values) == 0, label + ": non-finite f32 accepted")
        else:
            require(_verdict64(_widen(values)) == 0, label + ": non-finite f64 accepted")
    elif index == 8:
        var huge: List[Float64] = [_f64(0x7FEE42D130773B76), _f64(0x7FEE42D130773B76), _f64(0xFFEE42D130773B76)]
        require(_verdict64(huge) == 1, label + ": overflowing-sum finite f64 refused")
    else:
        var both: List[Float64] = [_f64(0x7FF0000000000000), _f64(0xFFF0000000000000)]
        require(_verdict64(both) == 0, label + ": +Inf/-Inf f64 accepted")


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(ALL_FINITE_CASES)
    def task(unit: Int) {imm cases}:
        try:
            run_case(unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, ALL_FINITE_CASES)


# The device column shares the base binding's one process-wide device context,
# so its cases are submitted in order.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(ALL_FINITE_CASES)
    for unit in range(ALL_FINITE_CASES):  # driver-loop: over the ordered shared-context cases
        try:
            run_case(unit)
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
    return Check("core_all_finite", cpu=True, gpu=True, serial="", run=run_cases)
