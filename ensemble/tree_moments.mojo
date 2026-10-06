# SPDX-License-Identifier: Apache-2.0
"""T14 V1 arithmetic shared by RF GPU and host, default OFF at callers.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Input fixed-point quanta and scale admission are unchanged. A leaf assigns a
sample to logical partial row_id % 128, including each duplicate draw. Each
exact integer partial is dequantized with the incumbent Float32 division, FTZ,
then widened to portable binary64. Pair distances 64,32,...,1 are fixed, zero
padding is +0, additions and final division round nearest/even without FMA.
The final Float32 is FTZ. Weight is the incumbent exact integer/fixed sum.
NaN/nonfinite input refusal, missing-data refusal and split tie rules stay at
existing public boundaries. Nonfinite candidate gains fail the existing > gate.
"""
from checks.numerics import ftz
from checks.soft_f64 import sf64_add, sf64_div, sf64_from_f32, sf64_to_f32

@always_inline
def moment_pair(left: UInt64, right: UInt64) -> UInt64:
    return sf64_add(left,right)

@always_inline
def moment_finish(sum: UInt64, weight: Float32) -> Float32:
    if weight <= 0:
        return Float32(0)
    return ftz(sf64_to_f32(sf64_div(sum,sf64_from_f32(weight))))

@always_inline
def balanced_mse_gain(parent_weight: Float32, left_weight: Float32,
                      sum: Float32, left_sum: Float32) -> Float32:
    """V1: separately normalized means, then balanced weighted squares.

    Every product/division/add/subtract is explicitly rounded and FTZ; no FMA.
    Parent/left/right moments use existing exact histograms, so RNG, candidate
    sets, fixed-point bounds and missing direction do not change.
    """
    var right_weight = ftz(parent_weight-left_weight)
    if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return -Float32.MAX_FINITE
    var right_sum = ftz(sum-left_sum)
    var pm = ftz(sum/parent_weight)
    var lm = ftz(left_sum/left_weight)
    var rm = ftz(right_sum/right_weight)
    var parent = ftz(pm*pm)
    var left = ftz(ftz(lm*lm)*ftz(left_weight/parent_weight))
    var right = ftz(ftz(rm*rm)*ftz(right_weight/parent_weight))
    return ftz(ftz(ftz(left+right)-parent)*Float32(0.5))


def et_leaf_moment_host[ro: MutOrigin, lo: MutOrigin, //](
    rows: MutPointer[Int32,ro], labels: MutPointer[Int32,lo],
    begin: Int, count: Int, inv_scale: Float32,
) -> Float32:
    """ET V1 counterpart: partial quanta multiply by ET's pinned inv_scale."""
    var partials = List[Int32](length=128,fill=Int32(0))
    var moments = List[UInt64](length=128,fill=UInt64(0))
    for pos in range(begin,begin+count):
        var row = Int(rows[unsafe_offset=pos])
        partials[row%128] += labels[unsafe_offset=row]
    for logical in range(128):
        moments[logical] = sf64_from_f32(ftz(Float32(partials[logical])*inv_scale))
    var distance = 64
    while distance > 0:
        for logical in range(distance):
            moments[logical] = moment_pair(moments[logical],moments[logical+distance])
        distance //= 2
    return moment_finish(moments[0],Float32(count))
