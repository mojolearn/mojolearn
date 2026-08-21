"""GREEDY vs POINTWISE searcher parity at covtype -- CURRENTLY FAILING.

REPRODUCER, recorded 2026-08-22 by the performance lane. The two
searchers are bit-identical at eps500 (500 one-byte features) and on
the 4k-row oracle fixtures, but at covtype (10 one-byte + 43 BINARY
features, 581k rows) the pointwise arm diverges AND DEGRADES:

    greedy     mse 0.9486324077643835   (matches CatBoost's covtype run)
    pointwise  mse 1.1843180519507341   (stable under the Apple i32
               routing row; WITHOUT the row it also wobbles run to run:
               1.1843 vs 1.1636 -- the binary family flushes through a
               float atomicAdd)

Established with the routing row DISABLED in a clean worktree, so this
is NOT the row: the divergence is upstream, in the pointwise family's
BINARY-policy path (or a mixed-policy interaction) at scale -- a shape
the oracle fixtures never cover. The quality gap means WORSE SPLITS,
not last-bit wobble: something is mis-accumulated or mis-scored for
binary features.

Run: pixi run -e bench mojo run -I . \
    mojo_only/searcher_parity_covtype_check.mojo ~/.cache/mojolearn
Passes when every rep prints `identical True`.


    pixi run -e bench mojo run -I . \
        mojo_only/quantize_cost_probe.mojo <fixture_dir>

Every benchmark row quantizes OUTSIDE the timed region on both arms --
fair for kernel comparisons, but a user's `train(X, y)` (and the Python
`GradientBoosting.fit` above it) pays the host-side GreedyLogSum border
build and the binning INSIDE the call, against CatBoost's parallel C++
quantizer. That cost has never been measured. This probe times, on the
SAME 400k x 500 float slice of epsilon, interleaved in one process:

    A. train(X, y) end to end, 20 trees @128     (prep + trees)
    B. fit-only on prebuilt fixtures, same shape  (trees alone)

A - B is the preparation bill: borders + binning + cindex build + the
device uploads. Whether it is 0.5 s or 50 s decides whether the
user-facing path needs the device binarize kernel
(`binarize_float_feature_kernel`, already ported for inference) wired
into `train()`, or nothing at all.

The CatBoost side of the same question is one number from their arm
(`_pool` timing includes `pool.quantize`), collected when the Python
extension can be rebuilt; this probe answers OUR side without it.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit
from gbdt.train import train

from std.sys import argv

comptime DEPTH = 6
comptime TREES = 20
comptime REPS = 2
comptime N_ROWS = 581_012
comptime N_FEATS = 54


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: ... <fixture_dir>")
    var d = String(args[1])
    var ctx = DeviceContext()

    # raw floats: first N_FEATS columns of epsilon, column-major, from the
    # cached X (row-major on disk, so this transposes on load)
    var xbytes = read_all(d + "/covtype_Xcol.f32")
    if len(xbytes) != 4 * N_ROWS * N_FEATS:
        raise Error(
            "covtype_Xcol.f32 missing or wrong size; run"
            " tools/slice_x500.py first"
        )
    var xp = xbytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var x_colmajor = List[Float32]()
    for i in range(N_ROWS * N_FEATS):
        x_colmajor.append(xp[i])

    var ybytes = read_all(d + "/covtype_y.f32")
    var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var y = List[Float32]()
    for r in range(N_ROWS):
        y.append(yp[r])

    # fit-only fixtures: the standard 128-border binning, sliced to the
    # same 500 features (prefix slice, so the grid per feature matches
    # what train() will derive only if GreedyLogSum agrees with
    # CatBoost's -- which check-greedylogsum gates; small drift between
    # grids changes the trees, not the timing scale)
    var folds = List[Int]()
    var f = open(d + "/covtype_folds_128.txt", "r")
    var txt = f.read()
    f.close()
    for line in txt.split("\n"):
        if line.byte_length() > 0:
            folds.append(Int(line))
    var bins = read_all(d + "/covtype_bins_128.u8")

    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hbuf = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var dbins = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    for feat in range(len(folds)):
        ref cf = lay.features[feat]
        var base = feat * N_ROWS
        for r in range(N_ROWS):
            hbuf.unsafe_ptr().unsafe_store(r, bins[base + r])
        ctx.enqueue_copy(dst_buf=dbins, src_ptr=hbuf.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS), cf.mask, cf.shift,
            dbins.unsafe_ptr(), Int32(N_ROWS), cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=(WRITE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
    var targets = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    for r in range(N_ROWS):
        ht.unsafe_ptr().unsafe_store(r, yp[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    print(
        "quantize cost probe:", N_ROWS, "x", N_FEATS,
        "@128, 20 trees depth 6, train() vs fit-only, interleaved",
    )

    for rep in range(2):
        var model_g = TAdditiveModel()
        var t0 = perf_counter_ns()
        var losses_g = fit(
            model_g, ctx, N_ROWS, folds, DEPTH, cindex, targets, weights,
            False, TREES, Float32(0.3), Float32(3.0), True,
        )
        var greedy_s = Float64(perf_counter_ns() - t0) / 1e9
        var model_p = TAdditiveModel()
        var t1 = perf_counter_ns()
        var losses_p = fit(
            model_p, ctx, N_ROWS, folds, DEPTH, cindex, targets, weights,
            False, TREES, Float32(0.3), Float32(3.0), True,
            use_pointwise_searcher=True,
        )
        var pointwise_s = Float64(perf_counter_ns() - t1) / 1e9
        var lg = losses_g[len(losses_g) - 1]
        var lp = losses_p[len(losses_p) - 1]
        print(
            "rep", rep, " greedy", greedy_s, "s   pointwise", pointwise_s,
            "s   ratio", greedy_s / pointwise_s,
            "  mse greedy", lg, " pointwise", lp,
            " identical", lg == lp,
        )
