# SPDX-License-Identifier: Apache-2.0
"""Independent threshold comparison and refusal contracts for label ABI v3."""
from std.memory import bitcast
from core.labels_native_api import LabelsNativeDriver, LabelsThresholdRequest
from core.host_parallel import host_parallelize
from core.labels_native_host_provider import labels_host_native_v2
from core.labels_native_device import labels_device_native_v2
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report


comptime LABELS_THRESHOLD_CASES = 7


def fixture(driver: LabelsNativeDriver, unit: Int) raises:
    var source: List[Float32] = [-1.0, -0.0, 0.0, 1.0, bitcast[DType.float32](UInt32(0x7fc00001))]
    var wide: List[Float64] = [1.0, 1.0000000000009095, 1.000000000001819]
    var result = List[Int64](length=5, fill=123)
    var request = LabelsThresholdRequest(Int(source.unsafe_ptr()), Int(result.unsafe_ptr()),
        0, 5, unit != 1, -7, 9, 0)
    if unit == 2:
        request.source = Int(wide.unsafe_ptr())
        request.code = 1
        request.count = 3
        request.threshold = 1.0000000000009095
    elif unit == 3:
        request.source = 0
        request.dst = 0
        request.count = 0
        request.code = 99
    elif unit == 4:
        request.count = -1
        request.code = 99
    elif unit == 5:
        request.code = 99
    elif unit == 6:
        request.threshold = bitcast[DType.float64](UInt64(0x7ff8000000000001))
    var caught = False
    try:
        _ = driver.threshold(request)
    except problem:
        var expected = String("threshold_labels_i64: n must be non-negative") if unit == 4 else String("threshold_labels_i64: float32 or float64 scores only")
        if (unit != 4 and unit != 5) or String(problem) != expected:
            raise Error("threshold wrong refusal: " + String(problem))
        caught = True
    if caught != (unit == 4 or unit == 5):
        raise Error("threshold refusal missing")
    if unit == 0 or unit == 1:
        var zero = Int64(9 if unit == 1 else -7)
        if result[0] != -7 or result[1] != zero or result[2] != zero or result[3] != 9 or result[4] != -7:
            raise Error("threshold strict/equal/signed-zero/NaN comparison")
    elif unit == 2:
        if result[0] != -7 or result[1] != -7 or result[2] != 9 or result[3] != 123:
            raise Error("binary64 threshold must not narrow through Float32")
    elif unit == 3:
        if result[0] != 123:
            raise Error("empty threshold must not dereference either null pointer")
    elif unit == 6:
        if result[0] != -7 or result[1] != -7 or result[2] != -7 or result[3] != -7 or result[4] != -7:
            raise Error("NaN threshold always selects below")
    _ = source
    _ = wide
    _ = result


# ---- The host column (was labels_threshold_cpu_suite.mojo) ----
def _run_cpu(mut report: Report) raises:
    var failures = List[Int](length=LABELS_THRESHOLD_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(LabelsNativeDriver(labels_host_native_v2), unit)
        except problem:
            flags[unit] = 1
            print("FAIL labels_threshold_cpu fixture", unit, String(problem))
    host_parallelize(run, LABELS_THRESHOLD_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    if failed:
        raise Error("labels threshold CPU failed " + String(failed) + " cases")
    print("PASS labels_threshold_cpu:", LABELS_THRESHOLD_CASES)


# ---- The device column (was labels_threshold_gpu_suite.mojo) ----
# Ordered numerical coverage while shared-context GPU concurrency is parked.
def _run_gpu(mut report: Report) raises:
    var failed = 0
    for unit in range(LABELS_THRESHOLD_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LabelsNativeDriver(labels_device_native_v2), unit)
        except problem:
            failed += 1
            print("FAIL labels_threshold_gpu fixture", unit, String(problem))
    if failed:
        raise Error("labels threshold GPU failed " + String(failed) + " cases")
    print("PASS labels_threshold_gpu:", LABELS_THRESHOLD_CASES)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("labels_threshold", cpu=True, gpu=True, serial="", run=run_cases)
