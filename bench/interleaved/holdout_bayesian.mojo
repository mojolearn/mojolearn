"""The HOLDOUT quality gate for the stochastic mode: does a Bayesian fit
of ours GENERALIZE like a Bayesian fit of theirs?

    pixi run -e bench mojo run -I . bench/interleaved/holdout_bayesian.mojo <dir> <name> <rows>

Train mse under random reweighting says nothing about quality, so the
bayesian training tables cannot carry the quality claim on their own.
This harness trains BOTH arms on the first 80% of rows -- the SAME
quantization grid on both sides: theirs through `input_borders` on the
full-data tsv, ours through the prep bins that used that tsv -- at
Bayesian temperature 1, a fresh seed per rep on each side, and scores
the untouched last 20%. The comparison is BANDS of test mse across
seeds; each side stays bit-reproducible under its own seed.

The Mojo arm's holdout predictions run through the same `predict` the
probe suite pins to the learn cursor, over a compressed index built from
the held-out slice of the SAME prep bins.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit, predict

from std.sys import argv

comptime DEPTH = 6
comptime TREES = 100
comptime BORDER = 128
comptime REPS = 5


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def build_slice_cindex(
    ctx: DeviceContext,
    lay_columns: Int,
    folds: List[Int],
    bins: List[UInt8],
    n_rows_total: Int,
    row0: Int,
    n_rows: Int,
) raises -> DeviceBuffer[DType.uint32]:
    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var dbins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for feat in range(len(folds)):
        ref cf = lay.features[feat]
        var base = feat * n_rows_total + row0
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
    return cindex^


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: ... <data_dir> <name> <rows>")
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_total = Int(String(args[3]))
    var n_train = n_total * 4 // 5
    var n_test = n_total - n_train
    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools")
    var arm = Python.import_module("catboost_arm")

    var folds = List[Int]()
    var f = open(
        data_dir + "/" + name + "_folds_" + String(BORDER) + ".txt", "r"
    )
    var txt = f.read()
    f.close()
    for line in txt.split("\n"):
        if line.byte_length() > 0:
            folds.append(Int(line))

    var bins = read_all(
        data_dir + "/" + name + "_bins_" + String(BORDER) + ".u8"
    )
    var ybytes = read_all(data_dir + "/" + name + "_y.f32")
    var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()

    var lay = build_layout(folds)
    var cindex_train = build_slice_cindex(
        ctx, lay.columns, folds, bins, n_total, 0, n_train
    )
    var cindex_test = build_slice_cindex(
        ctx, lay.columns, folds, bins, n_total, n_train, n_test
    )

    var targets = ctx.enqueue_create_buffer[DType.float32](n_train)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_train)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_train)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_train)
    for r in range(n_train):
        ht.unsafe_ptr().unsafe_store(r, yp[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())

    var cursor_test = ctx.enqueue_create_buffer[DType.float32](n_test)
    var h_cur = ctx.enqueue_create_host_buffer[DType.float32](n_test)
    ctx.synchronize()

    print(
        name, "holdout:", n_train, "train /", n_test, "test rows,",
        TREES, "trees depth", DEPTH, "at", BORDER,
        "borders, Bayesian T=1, fresh seed per rep per arm",
    )

    for rep in range(REPS):
        var res = arm.fit_and_test_mse_bayesian(
            data_dir, name, BORDER, TREES, DEPTH, n_train, rep
        )
        var their_train = res[1].__float__()
        var their_test = res[2].__float__()

        var model = TAdditiveModel()
        var losses = fit(
            model, ctx, n_train, folds, DEPTH, cindex_train, targets,
            weights, False, TREES, Float32(0.3), Float32(3.0), True,
            bootstrap_bayesian=True,
            bagging_temperature=Float32(1.0),
            random_seed=UInt64(100 + rep),
        )
        predict(model, ctx, n_test, folds, cindex_test, cursor_test)
        ctx.enqueue_copy(dst_ptr=h_cur.unsafe_ptr(), src_buf=cursor_test)
        ctx.synchronize()
        var se = Float64(0.0)
        for r in range(n_test):
            var d = Float64(h_cur.unsafe_ptr().unsafe_load(r)) - Float64(
                yp[n_train + r]
            )
            se += d * d
        var our_test = se / Float64(n_test)
        print(
            "  rep", rep,
            " test mse: ours", our_test, " catboost", their_test,
            "  (train: ours", losses[len(losses) - 1],
            " catboost", their_train, ")",
        )
