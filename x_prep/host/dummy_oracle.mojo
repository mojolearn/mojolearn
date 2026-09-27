# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THE PREP LANE'S PROOF DUMMY, ON THE HOST (lane/algos-prep, 2026-09-27;
removed before merge): x_prep/dummy_device.mojo's two kernels as plain loops,
the same order and the same seams."""
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_div

comptime X_PREP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def host_l1_mean_fit(x: List[Float32], n: Int, d: Int) -> List[Float32]:
    var out = List[Float32](capacity=d)
    for c in range(d):
        var acc = Float32(0)
        for i in range(n):
            var row = n - 1 - i if X_PREP_HOST_SABOTAGE else i
            acc = ftz(acc + ftz(abs(ftz(x[row * d + c]))))
        var mean = ftz(identical_div(acc, Float32(n)))
        out.append(mean if mean > Float32(0) else Float32(1))
    return out^


def host_scale(x: List[Float32], s: List[Float32], n: Int, d: Int) -> List[Float32]:
    var out = List[Float32](capacity=n * d)
    for i in range(n * d):
        out.append(ftz(identical_div(ftz(x[i]), ftz(s[i % d]))))
    return out^
