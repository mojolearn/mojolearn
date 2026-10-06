# SPDX-License-Identifier: Apache-2.0
"""T31–T39 experiment controls and the T34 cross-column arithmetic contract.

All new switches are default OFF. NOT COMPILED — NOT TESTED — IDENTITY NOT
VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. No device-specific arithmetic.
"""
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_div, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime T31_PACKED_A = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T31_PACKED_A"]()
comptime T31_PACKED_B = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T31_PACKED_B"]()
comptime T32_SHARED_ROWS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T32_SHARED_ROWS"]()
comptime T33_COST_SCHEDULE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T33_COST_SCHEDULE"]()
comptime T34_CHUNK_FOLD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T34_CHUNK_FOLD"]()
comptime T35_LEAF_REUSE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T35_LEAF_REUSE"]()
comptime T36_FINITE_STAGE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T36_FINITE_STAGE"]()
comptime T37_WORKSPACE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T37_WORKSPACE"]()
comptime T38_FUSED_LABELS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T38_FUSED_LABELS"]()

# Fixed logical chunks are the numerical version, not launch or warp widths.
# Each chunk contains consecutive trees, zero-padded to 32; pair adjacent
# entries at strides 1,2,4,8,16. Chunk roots fold in increasing chunk order.
comptime FOREST_CHUNK = 32

@always_inline
def chunk_add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))

@always_inline
def forest_chunk_sum(mut values: InlineArray[Float32, FOREST_CHUNK]) -> Float32:
    var stride = 1
    while stride < FOREST_CHUNK:
        var i = 0
        while i < FOREST_CHUNK:
            values[i] = chunk_add(values[i], values[i + stride])
            i += 2 * stride
        stride *= 2
    return values[0]

@always_inline
def forest_chunk_finish(total: Float32, trees: Int) -> Float32:
    return ftz(identical_div(ftz(total), Float32(trees)))

comptime T44_METADATA_CACHE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T44_METADATA_CACHE"]()
comptime T39_AUXILIARY_WALK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T39_AUXILIARY_WALK"]()

comptime C50_GB_PACKED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C50_GB_PACKED"]()
