# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast


def check_values(name: String, actual: List[Float32], expected: List[Float32]) raises:
    if len(actual) != len(expected):
        raise Error(name + ": wrong result length")
    var digest = UInt64(1469598103934665603)
    for i in range(len(actual)):
        var tolerance = Float32(0.0003) + Float32(0.0003) * abs(expected[i])
        if not (abs(actual[i] - expected[i]) <= tolerance):
            print("FAIL", name, "index", i, "actual", actual[i], "expected", expected[i])
            raise Error(name + ": independent gradient/value check failed")
        digest = (digest ^ UInt64(bitcast[DType.uint32](actual[i]))) * UInt64(1099511628211)
    print("PASS", name, "elements", len(actual), "digest", digest)
