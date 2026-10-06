# SPDX-License-Identifier: Apache-2.0
"""T39 exact auxiliary-output walk, shared by host and every GPU column.
Default OFF. NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Prefix contract: strict tree-order sum / number of trees in that prefix.
Outputs are row-major. A zero output address requests no allocation/write.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_div
from core.forest_experiments import T39_AUXILIARY_WALK

comptime AuxI32 = MutPointer[Int32, MutAnyOrigin]
comptime AuxF32 = MutPointer[Float32, MutAnyOrigin]

@always_inline
def _aux_key(x: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](x)
    if (bits&UInt32(0x7fffffff)) == 0:
        bits = 0
    return ~bits if (bits&UInt32(0x80000000)) != 0 else bits^UInt32(0x80000000)

@always_inline
def _forest_auxiliary_row_shared[RF_INPUT: Bool](offsets: AuxI32, colid: AuxI32, threshold: AuxF32, left: AuxI32, leaves: AuxF32,
    x: AuxF32, leaf_out: AuxI32, prefix_out: AuxF32, prediction: AuxF32, totals: AuxF32,
    row: Int, features: Int, trees: Int, outputs: Int, emit_leaf: Bool, emit_prefix: Bool, emit_prediction: Bool) -> Bool:
    for c in range(features):
        if (bitcast[DType.uint32](x.unsafe_load(row*features+c))&UInt32(0x7f800000)) == UInt32(0x7f800000):
            return False
    if emit_prefix or emit_prediction:
        for c in range(outputs):
            totals.unsafe_store(row*outputs+c, Float32(0))
    for t in range(trees):
        var base = Int(offsets.unsafe_load(t))
        var limit = Int(offsets.unsafe_load(t+1))-base
        var node = 0
        var steps = 0
        while True:
            if node < 0 or node >= limit or steps > limit:
                return False
            var child = Int(left.unsafe_load(base+node))
            if child == -1:
                break
            var c = Int(colid.unsafe_load(base+node))
            if c < 0 or c >= features or child < 1 or child+1 >= limit:
                return False
            var value = x.unsafe_load(row*features+c)
            comptime if RF_INPUT:
                value = ftz(value)
            node = child + (0 if _aux_key(value) <= _aux_key(threshold.unsafe_load(base+node)) else 1)
            steps += 1
        if emit_leaf:
            leaf_out.unsafe_store(row*trees+t, Int32(node))
        if emit_prefix or emit_prediction:
            for c in range(outputs):
                var value = leaves.unsafe_load((base+node)*outputs+c)
                var total = ftz(ftz(totals.unsafe_load(row*outputs+c))+ftz(value))
                totals.unsafe_store(row*outputs+c, total)
                var mean = ftz(identical_div(ftz(total), Float32(t+1)))
                if (bitcast[DType.uint32](mean)&UInt32(0x7f800000)) == UInt32(0x7f800000):
                    return False
                if emit_prefix:
                    prefix_out.unsafe_store((row*trees+t)*outputs+c, mean)
                if emit_prediction and t == trees-1:
                    prediction.unsafe_store(row*outputs+c, mean)
    return True


@always_inline
def forest_auxiliary_row[RF_INPUT: Bool](offsets: AuxI32, colid: AuxI32, threshold: AuxF32, left: AuxI32, leaves: AuxF32,
    x: AuxF32, leaf_out: AuxI32, prefix_out: AuxF32, prediction: AuxF32, totals: AuxF32,
    row: Int, features: Int, trees: Int, outputs: Int, emit_leaf: Bool, emit_prefix: Bool, emit_prediction: Bool) -> Bool:
    comptime if T39_AUXILIARY_WALK:
        return _forest_auxiliary_row_shared[RF_INPUT](offsets, colid, threshold, left, leaves, x, leaf_out, prefix_out, prediction,
            totals, row, features, trees, outputs, emit_leaf, emit_prefix, emit_prediction)
    # B: requested output consumers perform independent incumbent ordered walks.
    # Prefix p repeats the first p+1 trees. No output-only work is fabricated.
    if emit_leaf:
        if not _forest_auxiliary_row_shared[RF_INPUT](offsets, colid, threshold, left, leaves, x, leaf_out, prefix_out, prediction,
            totals, row, features, trees, outputs, True, False, False):
            return False
    if emit_prefix:
        for prefix in range(trees):
            var target = prefix_out.unsafe_offset((row*trees+prefix-row)*outputs)
            if not _forest_auxiliary_row_shared[RF_INPUT](offsets, colid, threshold, left, leaves, x, leaf_out, prefix_out, target,
                totals, row, features, prefix+1, outputs, False, False, True):
                return False
    if emit_prediction:
        if not _forest_auxiliary_row_shared[RF_INPUT](offsets, colid, threshold, left, leaves, x, leaf_out, prefix_out, prediction,
            totals, row, features, trees, outputs, False, False, True):
            return False
    return True


def forest_auxiliary_host(forest: List[Int], x: Int, leaf_out: Int, prefix_out: Int, pred_out: Int,
                          rows: Int, features: Int, trees: Int, outputs: Int, rf_input: Bool) raises:
    var sums = List[Float32](length=max(1, rows*outputs if prefix_out != 0 or pred_out != 0 else 1), fill=0)
    var totals = sums.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for r in range(rows):
        var ok = True
        if rf_input:
            ok = forest_auxiliary_row[True](AuxI32(unsafe_from_address=forest[0]), AuxI32(unsafe_from_address=forest[1]),
                AuxF32(unsafe_from_address=forest[2]), AuxI32(unsafe_from_address=forest[3]), AuxF32(unsafe_from_address=forest[4]),
                AuxF32(unsafe_from_address=x), AuxI32(unsafe_from_address=leaf_out), AuxF32(unsafe_from_address=prefix_out),
                AuxF32(unsafe_from_address=pred_out), totals, r, features, trees, outputs, leaf_out!=0, prefix_out!=0, pred_out!=0)
        else:
            ok = forest_auxiliary_row[False](AuxI32(unsafe_from_address=forest[0]), AuxI32(unsafe_from_address=forest[1]),
                AuxF32(unsafe_from_address=forest[2]), AuxI32(unsafe_from_address=forest[3]), AuxF32(unsafe_from_address=forest[4]),
                AuxF32(unsafe_from_address=x), AuxI32(unsafe_from_address=leaf_out), AuxF32(unsafe_from_address=prefix_out),
                AuxF32(unsafe_from_address=pred_out), totals, r, features, trees, outputs, leaf_out!=0, prefix_out!=0, pred_out!=0)
        if not ok:
            raise Error("forest auxiliary outputs require valid finite trees and input")
