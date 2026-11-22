# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Independent threshold comparison and refusal contracts for label ABI v3."""
from std.memory import bitcast
from core.labels_native_api import LabelsNativeDriver, LabelsThresholdRequest
from core.labels_native_host_provider import labels_host_native_v2
from core.labels_native_device import labels_device_native_v2
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases


comptime LABELS_THRESHOLD_CASES = 7
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


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
def _host_case(unit: Int) raises:
    fixture(LabelsNativeDriver(labels_host_native_v2), unit)


def test_threshold() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("threshold_cpu", LABELS_THRESHOLD_CASES, _host_case)
    else:
        # The device column (was labels_threshold_gpu_suite.mojo). Ordered numerical coverage while
        # shared-context GPU concurrency is parked. Inline in the comptime branch: TestSuite discovery
        # instantiates every function of the module, and the host column build has no device target.
        var failed = 0
        var text = String("")
        for unit in range(LABELS_THRESHOLD_CASES):  # ordered shared-context numerical fixtures
            try:
                fixture(LabelsNativeDriver(labels_device_native_v2), unit)
            except problem:
                failed += 1
                text += "\n      threshold_gpu row " + String(unit) + ": " + String(problem)
        if failed:
            raise Error("labels threshold GPU failed " + String(failed) + " of " + String(LABELS_THRESHOLD_CASES) + " cases" + text)



def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
