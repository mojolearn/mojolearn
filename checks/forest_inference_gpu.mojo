# SPDX-License-Identifier: Apache-2.0
"""Shared GPU flat-forest prototype, independent handcrafted oracle.

 tools/with_build_lock.sh pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/forest_inference_gpu.mojo
Repeat without mode define for FAST, or MOJOLEARN_NUMERIC_DETERMINISTIC=1.
No public inference dispatch changes. Expected leaves are chosen by fixture
row categories, not the production traversal/key helper. Host folds specify
ordered and fixed32-grove arithmetic independently; division is Float64 then
Float32, not the production portable_divf. Tests cover local offsets, stumps,
leaf-only trees, scalar/vector output, equality, +/-0, +/-subnormal thresholds,
31/32/33 tree tails, cancellation and invalid graphs.
"""
from std.memory import bitcast
from std.math import abs
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from core.forest_inference import forest_predict_gpu, validate_flat_forest


def reference_flush(x: Float32) -> Float32:
    var bits = bitcast[DType.uint32](x)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if (bits & UInt32(0x7f800000)) == 0:
            bits &= UInt32(0x80000000)
    return bitcast[DType.float32](bits)


def reference_add(a: Float32, b: Float32) -> Float32:
    return reference_flush(Float32(Float64(reference_flush(a))+Float64(reference_flush(b))))


def reference_mean(values: List[Float32], grove: Bool) -> Float32:
    var total = Float32(0)
    if grove:
        var sums = List[Float32](length=32,fill=Float32(0))
        # Tree-major scatter differs from the kernel's per-lane traversal loop.
        for tree in range(len(values)):
            sums[tree%32] = reference_add(sums[tree%32],values[tree])
        var active = 16
        while active > 0:
            for i in range(active):
                sums[i] = reference_add(sums[i],sums[i+active])
            active //= 2
        total = sums[0]
    else:
        for value in values:
            total = reference_add(total,value)
    return reference_flush(Float32(Float64(total)/Float64(len(values))))


def check_value(actual: Float32, expected: Float32) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        assert_equal(actual.to_bits(),expected.to_bits())
    else:
        # FAST device division may use vendor-specific lowering.
        assert_true(abs(actual-expected) <= Float32(0.000001))


def leaf_value(tree: Int, output: Int, outputs: Int, left_side: Bool) -> Float32:
    if outputs == 1:
        return Float32(tree%7-3)/Float32(8) if left_side else Float32(tree%5+1)/Float32(4)
    var first = Float32(tree%4+1)/Float32(8) if left_side else Float32(tree%3+1)/Float32(4)
    return first if output == 0 else Float32(1)-first


def expected_left[RF_INPUT: Bool](row: Int, threshold_kind: Int) -> Bool:
    # Rows: -1,+0,-0,+tiny,-tiny,+1. Threshold kind: -tiny,0,+tiny.
    if row == 0:
        return True
    if row == 5:
        return False
    comptime if RF_INPUT and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return threshold_kind != 0  # all four central input values flush/equal zero
    if row == 4:
        return True
    if row == 3:
        return threshold_kind == 2
    return threshold_kind != 0


def fixture[RF_INPUT: Bool](ctx: DeviceContext, trees: Int, outputs: Int) raises:
    var tiny = bitcast[DType.float32](UInt32(1))
    var negative_tiny = bitcast[DType.float32](UInt32(0x80000001))
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var x: List[Float32] = [-1,0,negative_zero,tiny,negative_tiny,1]
    var offsets: List[Int32] = [0]
    var cols = List[Int32]()
    var thresholds = List[Float32]()
    var children = List[Int32]()
    var leaves = List[Float32]()
    for t in range(trees):
        var singleton = t%5 == 0
        var count = 1 if singleton else 3
        for node in range(count):
            cols.append(Int32(0))
            thresholds.append(negative_tiny if t%3 == 0 else (Float32(0) if t%3 == 1 else tiny))
            children.append(Int32(1) if not singleton and node == 0 else Int32(-1))
            for c in range(outputs):
                leaves.append(leaf_value(t,c,outputs,singleton or node == 1))
        offsets.append(Int32(len(cols)))
    var ordered = forest_predict_gpu[RF_INPUT,False](ctx,offsets,cols,thresholds,children,leaves,x,6,1,outputs)
    var grove = forest_predict_gpu[RF_INPUT,True](ctx,offsets,cols,thresholds,children,leaves,x,6,1,outputs)
    for row in range(6):
        for c in range(outputs):
            var chosen = List[Float32]()
            for t in range(trees):
                chosen.append(leaf_value(t,c,outputs,t%5 == 0 or expected_left[RF_INPUT](row,t%3)))
            var index = row*outputs+c
            check_value(ordered[index],reference_mean(chosen,False))
            check_value(grove[index],reference_mean(chosen,True))
            print("BITS",RF_INPUT,trees,outputs,row,c,ordered[index].to_bits(),grove[index].to_bits())
            # Dyadic fixture totals sum exactly in either association.
            assert_equal(ordered[index].to_bits(),grove[index].to_bits())
    print("PASS fixture",RF_INPUT,trees,outputs)


def cancellation(ctx: DeviceContext) raises:
    var offsets: List[Int32] = [0]
    var cols = List[Int32]()
    var thresholds = List[Float32]()
    var children = List[Int32]()
    var leaves = List[Float32]()
    var x: List[Float32] = [0]
    for tree in range(33):
        offsets.append(Int32(tree+1))
        cols.append(0)
        thresholds.append(0)
        children.append(-1)
        leaves.append(Float32(16777216) if tree == 0 else (Float32(1) if tree == 1 else (Float32(-16777216) if tree == 2 else Float32(0))))
    var ordered = forest_predict_gpu[False,False](ctx,offsets,cols,thresholds,children,leaves,x,1,1,1)
    var grove = forest_predict_gpu[False,True](ctx,offsets,cols,thresholds,children,leaves,x,1,1,1)
    check_value(ordered[0],reference_mean(leaves,False))
    check_value(grove[0],reference_mean(leaves,True))
    assert_true(ordered[0].to_bits() != grove[0].to_bits())
    # Reassociation is intentionally observable; never assert blanket legacy parity.
    print("PASS cancellation: ordered and grove intentionally differ")
    var empty = List[Float32]()
    var zero = forest_predict_gpu[False,True](ctx,offsets,cols,thresholds,children,leaves,empty,0,1,1)
    assert_equal(len(zero),0)


def invalid_graph() raises:
    var offsets: List[Int32] = [0,3]
    var cols: List[Int32] = [0,0,0]
    var thresholds: List[Float32] = [0,0,0]
    var children: List[Int32] = [0,-1,-1]  # root revisits itself
    var leaves: List[Float32] = [0,1,2]
    var x: List[Float32] = [0]
    var caught = False
    try:
        validate_flat_forest(offsets,cols,thresholds,children,leaves,x,1,1,1)
    except:
        caught = True
    assert_true(caught)
    children[0] = 3
    caught = False
    try:
        validate_flat_forest(offsets,cols,thresholds,children,leaves,x,1,1,1)
    except:
        caught = True
    assert_true(caught)
    children[0] = 1
    x[0] = bitcast[DType.float32](UInt32(0x7f800000))
    caught = False
    try:
        validate_flat_forest(offsets,cols,thresholds,children,leaves,x,1,1,1)
    except:
        caught = True
    assert_true(caught)
    print("PASS graph/bounds/nonfinite refusals")


def main() raises:
    print("numeric_mode",numeric_mode_name())
    invalid_graph()
    var ctx = DeviceContext()
    var counts: List[Int] = [1,31,32,33]
    for trees in counts:
        fixture[False](ctx,trees,1)
        fixture[False](ctx,trees,2)
        fixture[True](ctx,trees,1)
        fixture[True](ctx,trees,2)
    cancellation(ctx)
    print("PASS shared GPU forest inference prototype")
