# SPDX-License-Identifier: Apache-2.0
"""Independent threshold comparison and refusal contracts for label ABI v3."""
from std.memory import bitcast
from core.labels_native_api import LabelsNativeDriver, LabelsThresholdRequest

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
        var zero = Int64(9) if unit == 1 else Int64(-7)
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
