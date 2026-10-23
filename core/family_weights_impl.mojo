# SPDX-License-Identifier: Apache-2.0
"""Sample-weight marshaling of the classical family owners
(bindings/family_common.mojo `family_sample_weight`): the scalar fill and
prepare_targets' weight checks (finiteness, then the minimum, then the
maximum) over float32 words."""
from std.math import isfinite

comptime FAMILY_WEIGHTS_OK = 0
comptime FAMILY_WEIGHTS_NONFINITE = -3
comptime FAMILY_WEIGHTS_NEGATIVE = -4
comptime FAMILY_WEIGHTS_ALL_ZERO = -5


def family_fill_f32(out: MutPointer[Float32, MutUntrackedOrigin], n: Int, value: Float32):
    for i in range(n):  # native input marshaling: a scalar weight fills every row
        out[i] = value


def family_weight_status(w: MutPointer[Float32, MutUntrackedOrigin], n: Int, allow_negative: Bool) -> Int:
    """prepare_targets' weight status: FAMILY_WEIGHTS_OK or the first refusal."""
    for i in range(n):  # native input marshaling: weight finiteness scan
        if not isfinite(w[i]):
            return FAMILY_WEIGHTS_NONFINITE
    var lo = Float32(0)
    var hi = Float32(0)
    for i in range(n):  # native input marshaling: weight minimum and maximum
        if i == 0 or w[i] < lo:
            lo = w[i]
        if i == 0 or w[i] > hi:
            hi = w[i]
    if not allow_negative and lo < 0:
        return FAMILY_WEIGHTS_NEGATIVE
    if not (hi > 0 or lo < 0):
        return FAMILY_WEIGHTS_ALL_ZERO
    return FAMILY_WEIGHTS_OK


def family_weight_status_f64(w: MutPointer[Float64, MutUntrackedOrigin], n: Int, allow_negative: Bool) -> Int:
    """`family_weight_status` over float64 words."""
    for i in range(n):  # native input marshaling: weight finiteness scan
        if not isfinite(w[i]):
            return FAMILY_WEIGHTS_NONFINITE
    var lo = Float64(0)
    var hi = Float64(0)
    for i in range(n):  # native input marshaling: weight minimum and maximum
        if i == 0 or w[i] < lo:
            lo = w[i]
        if i == 0 or w[i] > hi:
            hi = w[i]
    if not allow_negative and lo < 0:
        return FAMILY_WEIGHTS_NEGATIVE
    if not (hi > 0 or lo < 0):
        return FAMILY_WEIGHTS_ALL_ZERO
    return FAMILY_WEIGHTS_OK


def family_fill_f64(out: MutPointer[Float64, MutUntrackedOrigin], n: Int, value: Float64):
    for i in range(n):  # native input marshaling: a scalar weight fills every row
        out[i] = value
