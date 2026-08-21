"""The pinned RMSE comparison at a CALLER-CHOSEN depth: the depth-8 row.

    pixi run -e bench mojo run -I . \
        bench/interleaved/rmse_depth_interleaved.mojo <dir> epsilon 400000 8

`catboost_interleaved.mojo` pins DEPTH = 6 at comptime, gbm-bench pins
max_depth 8, and the density-cliff finding
(`SHAPE_SWEEP_2026-08-21_epsilon.md`) says depth is exactly the axis
where our accumulate degrades -- per-level cost 11.5 -> 27.8 ms across
levels 1-6 at constant built rows, trend rising. Depth 8 adds two MORE
levels at 128-256 leaves, where indexed-read density is worst. So this
row is the honest stress test: their CPU pays more depth too, and the
question of WHO pays more is a measurement, not an inference.

A separate file rather than an edit because the standard harness is
another lane's working file at the time of writing; the clone is the
COMMITTED harness verbatim -- same fixtures, same pinned settings, same
warm-ups, same ratio convention (SPEEDUP, > 1 means ours is faster) --
with `DEPTH` moved from comptime to argv (default 8) and the `bayesian`
arg dropped. When the lanes rejoin, the standard harness grows the argv
and this file is deleted.
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

from std.sys import argv

comptime TREES = 20
comptime REPS = 3


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
    depth: Int,
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
        "features, depth", depth, ",", TREES, "trees, reps interleaved",
    )

    # warm-up, both arms, untimed
    _ = arm.fit_seconds(data_dir, name, border, 2, depth)
    var wm = TAdditiveModel()
    _ = fit(
        wm, ctx, n_rows, folds, depth, cindex, targets, weights, False,
        2, Float32(0.3), Float32(3.0), True,
    )

    for rep in range(REPS):
        var res = arm.fit_seconds_and_mse(
            data_dir, name, border, TREES, depth
        )
        var secs = res[0].__float__()
        var their_mse = res[1].__float__()
        var theirs_ms = secs * 1000.0 / Float64(TREES)

        var model = TAdditiveModel()
        var t0 = perf_counter_ns()
        var losses = fit(
            model, ctx, n_rows, folds, depth, cindex, targets, weights,
            False, TREES, Float32(0.3), Float32(3.0), True,
        )
        var ours_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(
            TREES
        )
        print(
            "  rep", rep, " catboost-cpu", theirs_ms,
            "ms/tree   ours-gpu", ours_ms, "ms/tree   speedup",
            theirs_ms / ours_ms, "x", "  our final mse",
            losses[len(losses) - 1], " catboost mse", their_mse,
        )


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: mojo run -I . bench/interleaved/"
            "rmse_depth_interleaved.mojo <data_dir> <dataset_name>"
            " <n_rows> [depth]"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var depth = 8
    if len(args) > 4:
        depth = Int(String(args[4]))
    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools")
    var arm = Python.import_module("catboost_arm")
    run_border(ctx, arm, data_dir, name, n_rows, 254, depth)
    run_border(ctx, arm, data_dir, name, n_rows, 128, depth)
