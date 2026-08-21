"""Epsilon at its NATIVE task: Logloss, both arms, interleaved.

    pixi run -e bench mojo run -I . \
        bench/interleaved/logloss_interleaved.mojo <dir> epsilon 400000

Epsilon is a binary-classification dataset (-1/+1) that the RMSE rows
trained as regression because that is the repository's pinned comparison.
This harness runs the task the dataset actually poses, and is the FIRST
benchmark of the loss chassis the feature lane landed (Logloss end to
end: the NeedEstimation arm, the Newton walker, the device oracle).

Discipline is `catboost_interleaved.mojo`'s, unchanged: same rows, same
CatBoost-built grid, arms alternating in one process, and the pinned
settings with exactly two changes that are THEIRS, not ours --
`loss_function="Logloss"`, and Logloss's own GPU estimation default of
Newton at 10 iterations (`GetEstimationMethodDefaults`,
`catboost_options.cpp:157-164`, transcribed in
`gbdt/options/catboost_options.mojo`). Pinning 1 iteration would be the
`leaf_estimation_iterations=1` cheat the RMSE rows already priced.

The loss columns compare as NUMBERS: both arms report
`mean(log(1 + e^approx) - t*approx)` over the same binarized target
(border 0.5, so -1 -> 0 and +1 -> 1), ours from `fit`'s returned per-
iteration losses, theirs computed in `tools/catboost_logloss_arm.py`
from raw predictions with the identical formula. Ratio convention as
everywhere: SPEEDUP, and > 1 means ours is faster.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit
from gbdt.options.catboost_options import LEAF_ESTIMATION_NEWTON
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_LOGLOSS

from std.sys import argv

comptime DEPTH = 6
comptime TREES = 20
comptime REPS = 3
comptime NEWTON_ITERATIONS = 10  # their Logloss GPU default, both arms


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def run_border(
    ctx: DeviceContext,
    arm: PythonObject,
    data_dir: String,
    name: String,
    n_rows: Int,
    border: Int,
) raises:
    var folds = List[Int]()
    var f = open(
        data_dir + "/" + name + "_folds_" + String(border) + ".txt", "r"
    )
    var txt = f.read()
    f.close()
    for line in txt.split("\n"):
        if line.byte_length() > 0:
            folds.append(Int(line))
    var n_feats = len(folds)

    var bins = read_all(
        data_dir + "/" + name + "_bins_" + String(border) + ".u8"
    )
    if len(bins) != n_feats * n_rows:
        raise Error("bins file size mismatch")

    var ybytes = read_all(data_dir + "/" + name + "_y.f32")
    if len(ybytes) != 4 * n_rows:
        raise Error("y file size mismatch")

    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var dbins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for feat in range(n_feats):
        ref cf = lay.features[feat]
        var base = feat * n_rows
        for r in range(n_rows):
            hb.unsafe_ptr().unsafe_store(r, bins[base + r])
        ctx.enqueue_copy(dst_buf=dbins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            dbins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=(WRITE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, yp[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    print(
        "border_count", border, ":", n_rows, "rows x", n_feats,
        "features, depth", DEPTH, ",", TREES,
        "trees, Logloss Newton-10 both arms, reps interleaved",
    )

    # warm-up, both arms, untimed
    _ = arm.fit_seconds(data_dir, name, border, 2, DEPTH)
    var wm = TAdditiveModel()
    _ = fit(
        wm, ctx, n_rows, folds, DEPTH, cindex, targets, weights, False,
        2, Float32(0.3), Float32(3.0), True,
        objective=OBJECTIVE_LOGLOSS,
        logloss_border=Float32(0.5),
        leaf_estimation_iterations=NEWTON_ITERATIONS,
        leaf_estimation_method=LEAF_ESTIMATION_NEWTON,
    )

    for rep in range(REPS):
        var res = arm.fit_seconds_and_logloss(
            data_dir, name, border, TREES, DEPTH
        )
        var theirs_ms = res[0].__float__() * 1000.0 / Float64(TREES)
        var their_loss = res[1].__float__()

        var model = TAdditiveModel()
        var t0 = perf_counter_ns()
        var losses = fit(
            model, ctx, n_rows, folds, DEPTH, cindex, targets, weights,
            False, TREES, Float32(0.3), Float32(3.0), True,
            objective=OBJECTIVE_LOGLOSS,
            logloss_border=Float32(0.5),
            leaf_estimation_iterations=NEWTON_ITERATIONS,
            leaf_estimation_method=LEAF_ESTIMATION_NEWTON,
        )
        var ours_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(
            TREES
        )
        print(
            "  rep", rep, " catboost-cpu", theirs_ms,
            "ms/tree   ours-gpu", ours_ms, "ms/tree   speedup",
            theirs_ms / ours_ms, "x", "  our final logloss",
            losses[len(losses) - 1], " catboost logloss", their_loss,
        )


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: mojo run -I . bench/interleaved/logloss_interleaved"
            ".mojo <data_dir> <dataset_name> <n_rows>"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools")
    var arm = Python.import_module("catboost_logloss_arm")
    run_border(ctx, arm, data_dir, name, n_rows, 254)
    run_border(ctx, arm, data_dir, name, n_rows, 128)
