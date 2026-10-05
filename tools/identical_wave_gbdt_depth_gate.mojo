"""Grouped-four prediction: all-zero and mixed depths, including group tails.

Reuse the existing model-IO fixture's feature/border metadata and row fixture.
The reference is a direct untimed Float32 host tree walk, never an opponent.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.model_io_check import build_synthetic, _synthetic_rows
from gbdt.models.oblivious_model import (
    TAdditiveModel, TBinarySplit, TObliviousTreeModel, TObliviousTreeStructure,
    BIN_SPLIT_TAKE_BIN, BIN_SPLIT_TAKE_GREATER,
)
from gbdt.models.kernel.add_bin_values import IDN_PREDICT_FOUR
from gbdt.train import predict_floats


def check(ctx: DeviceContext, mixed: Bool) raises:
    var tm = build_synthetic()
    var m = TAdditiveModel()
    m.bias = Float64(0.125)
    var depths = List[Int]()
    # Nine trees exercise two full groups plus a trailing zero-depth tree.
    for t in range(9):
        var depth = 0
        if mixed and t % 3 != 0 and t != 8:
            depth = 1 + (t % 4)
        depths.append(depth)
        var structure = TObliviousTreeStructure()
        for level in range(depth):
            if level % 2 == 0:
                structure.splits.append(TBinarySplit(Int32(0), Int32(3 + level), Int32(BIN_SPLIT_TAKE_GREATER)))
            else:
                structure.splits.append(TBinarySplit(Int32(2), Int32(level % 3), Int32(BIN_SPLIT_TAKE_BIN)))
        var tree = TObliviousTreeModel(structure^)
        for leaf in range(1 << depth):
            tree.leaf_values.append(Float32((t + 1) * 3 - leaf) / Float32(8.0))
        m.add_weak_model(tree^)
    tm.model = m^
    var n = 257
    var x = _synthetic_rows(n)
    var got = predict_floats(ctx, tm, x, n)
    var changed = 0
    for row in range(n):
        var expected = Float32(tm.model.bias)
        for t in range(tm.model.size()):
            ref tree = tm.model.weak_models[t]
            var leaf = 0
            for level in range(depths[t]):
                ref split = tree.structure.splits[level]
                var feature = Int(split.feature_id)
                var bin = 0
                for border in range(len(tm.borders[feature])):
                    if x[feature * n + row] > tm.borders[feature][border]:
                        bin += 1
                var take = bin > Int(split.bin_idx)
                if split.split_type == Int32(BIN_SPLIT_TAKE_BIN):
                    take = bin == Int(split.bin_idx)
                if take:
                    leaf += 1 << level
            expected = expected + tree.leaf_values[leaf]
        if bitcast[DType.uint32](got[row]) != bitcast[DType.uint32](expected):
            raise Error("grouped prediction differs from zero/mixed-depth host walk")
        if got[row] != Float32(tm.model.bias):
            changed += 1
    if changed != n:
        raise Error("zero-depth contributions were vacuous")
    print("GBDT_DEPTH PASS mixed", mixed, "rows", n, "trees", tm.model.size(), "grouped_four", IDN_PREDICT_FOUR)


def main() raises:
    var ctx = DeviceContext()
    check(ctx, False)
    check(ctx, True)
