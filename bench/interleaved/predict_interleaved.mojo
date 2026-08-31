# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Our GPU inference vs CatBoost CPU inference, INTERLEAVED, same box.

    pixi run -e bench python tools/interleaved_prep.py <dir> synth 800000 100
    pixi run -e bench python tools/predict_prep.py <dir> synth 128
    pixi run -e bench mojo run -I . bench/interleaved/predict_interleaved.mojo <dir> synth <rows>

Their `model_evaluation_speed` methodology
(`catboost/benchmarks/model_evaluation_speed/model_evaluation_benchmark.ipynb`):
each library trains ITS OWN model of the same shape, the raw pool is built
OUTSIDE the timed region, and the timed line is `model.predict(pool)` --
quantization against the model's borders happens INSIDE the timed call.
Mirrored exactly: our timed region starts from raw device-resident floats
and covers the internal quantization (`binarize_float_feature_kernel`, their
`BinarizeFloatFeatureImpl`), the compressed-index build, every tree's
`AddObliviousTreeImpl`, and the copy of the predictions back to the host.
Their arm runs at full threads and at one thread, their notebook's two arms.

ACCURACY IS ASSERTED PER REP, not assumed: the mse of the predictions this
path returns must equal the train mse the fit reported, which it can only
do if the raw-float binarize reproduces the training bins bit for bit and
the packed ensemble walk reproduces the trees.
"""
from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    BINARIZE_BLOCK_SIZE,
    BINARIZE_DOCS_PER_THREAD,
    WRITE_BLOCK_SIZE,
    binarize_float_feature_kernel,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit, predict
from gbdt.models.cuda.evaluator import (
    GpuEvaluatorModel,
    launch_eval,
    launch_quantize,
    pack_model_for_evaluator,
    padded_results,
    quantized_buffer_u32s,
)

from std.sys import argv

comptime DEPTH = 6
comptime BORDER = 128
comptime REPS = 8


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: mojo run -I . bench/interleaved/predict_interleaved.mojo"
            " <data_dir> <dataset_name> <n_rows>"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    # optional 4th arg: tree count. 100 is the quick arena row; 8000 is
    # the model size of their own notebook's epsilon benchmark.
    var trees = 100
    if len(args) > 4:
        trees = Int(String(args[4]))
    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools")
    var arm = Python.import_module("catboost_arm")

    # ---- shared fixtures -------------------------------------------------
    var folds = List[Int]()
    var f = open(
        data_dir + "/" + name + "_folds_" + String(BORDER) + ".txt", "r"
    )
    var txt = f.read()
    f.close()
    for line in txt.split("\n"):
        if line.byte_length() > 0:
            folds.append(Int(line))
    var n_feats = len(folds)

    var bins = read_all(
        data_dir + "/" + name + "_bins_" + String(BORDER) + ".u8"
    )
    var ybytes = read_all(data_dir + "/" + name + "_y.f32")
    var xbytes = read_all(
        data_dir + "/" + name + "_X_colmajor_" + String(BORDER) + ".f32"
    )
    if len(xbytes) != 4 * n_feats * n_rows:
        raise Error("X colmajor size mismatch")
    var bbytes = read_all(
        data_dir + "/" + name + "_borders_" + String(BORDER) + ".f32bin"
    )

    var lay = build_layout(folds)

    # training cindex from the prep bins, exactly as the training harness
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

    # raw floats + borders, device-resident OUTSIDE the timed region (their
    # `cb.Pool(X_test)` is built outside theirs)
    var xdev = ctx.enqueue_create_buffer[DType.float32](n_feats * n_rows)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_feats * n_rows)
    var xp = xbytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n_feats * n_rows):
        hx.unsafe_ptr().unsafe_store(i, xp[i])
    ctx.enqueue_copy(dst_buf=xdev, src_ptr=hx.unsafe_ptr())

    var n_border_floats = len(bbytes) // 4
    var borders = ctx.enqueue_create_buffer[DType.float32](n_border_floats)
    var hbo = ctx.enqueue_create_host_buffer[DType.float32](n_border_floats)
    var bp = bbytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var border_off = List[Int]()
    var pos = 0
    for _ in range(n_feats):
        border_off.append(pos)
        var cnt = Int(bp[pos])
        pos += 1 + cnt
    if pos != n_border_floats:
        raise Error("borders file does not parse into n_feats features")
    for i in range(n_border_floats):
        hbo.unsafe_ptr().unsafe_store(i, bp[i])
    ctx.enqueue_copy(dst_buf=borders, src_ptr=hbo.unsafe_ptr())

    # the evaluator's model-data layout: values-only flat borders plus
    # separate offset/count arrays and the bucket -> feature identity
    # (`TGPUModelData.FlatBordersVector` / `BordersOffsets` /
    # `BordersCount` / `FloatFeatureForBucketIdx`)
    var ev_values = n_border_floats - n_feats
    var ev_borders = ctx.enqueue_create_buffer[DType.float32](ev_values)
    var ev_off = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var ev_cnt = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var ev_bucket_feat = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var he1 = ctx.enqueue_create_host_buffer[DType.float32](ev_values)
    var he2 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var he3 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var he4 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var vpos = 0
    for feat in range(n_feats):
        var base = border_off[feat]
        var cnt = Int(bp[base])
        he2.unsafe_ptr().unsafe_store(feat, UInt32(vpos))
        he3.unsafe_ptr().unsafe_store(feat, UInt32(cnt))
        he4.unsafe_ptr().unsafe_store(feat, UInt32(feat))
        for i in range(cnt):
            he1.unsafe_ptr().unsafe_store(vpos, bp[base + 1 + i])
            vpos += 1
    if vpos != ev_values:
        raise Error("evaluator borders repack mismatch")
    ctx.enqueue_copy(dst_buf=ev_borders, src_ptr=he1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_off, src_ptr=he2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_cnt, src_ptr=he3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_bucket_feat, src_ptr=he4.unsafe_ptr())

    var quantized = ctx.enqueue_create_buffer[DType.uint32](
        quantized_buffer_u32s(n_feats, n_rows)
    )
    var n_res = padded_results(n_rows)
    var results = ctx.enqueue_create_buffer[DType.float32](n_res)
    var h_res = ctx.enqueue_create_host_buffer[DType.float32](n_res)

    var pindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    var cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var h_cur = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.synchronize()

    # ---- train both models ONCE (untimed) ---------------------------------
    print(
        name, ":", n_rows, "rows x", n_feats, "features,", trees,
        "trees depth", DEPTH, "at", BORDER, "borders; predict reps",
        REPS, "interleaved",
    )
    _ = arm.predict_prep(data_dir, name, BORDER, trees, DEPTH)
    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, folds, DEPTH, cindex, targets, weights,
        False, trees, Float32(0.3), Float32(3.0), True,
    )
    var train_mse = losses[len(losses) - 1]
    # their TGPUModelData is built when the evaluator context is created,
    # not per predict call, so the pack sits OUTSIDE the timed region
    var gpu_model = pack_model_for_evaluator(ctx, model)

    comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD

    # warm-up, ALL arms, untimed (the first evaluator call pays kernel
    # compilation)
    _ = arm.predict_seconds(name, BORDER, -1)
    launch_quantize(
        ctx, xdev, n_rows, n_feats, ev_borders, ev_off, ev_cnt,
        ev_bucket_feat, quantized,
    )
    launch_eval(ctx, gpu_model, quantized, n_feats, n_rows, results)
    ctx.synchronize()

    for rep in range(REPS):
        var their_full = arm.predict_seconds(name, BORDER, -1).__float__()
        var their_one = arm.predict_seconds(name, BORDER, 1).__float__()

        # THE EVALUATOR ARM: their own GPU inference design, ported
        # (gbdt/models/cuda/evaluator.mojo) -- quantize once, one eval
        # kernel over every tree, results back.
        var t0 = perf_counter_ns()
        launch_quantize(
            ctx, xdev, n_rows, n_feats, ev_borders, ev_off, ev_cnt,
            ev_bucket_feat, quantized,
        )
        launch_eval(ctx, gpu_model, quantized, n_feats, n_rows, results)
        ctx.enqueue_copy(dst_ptr=h_res.unsafe_ptr(), src_buf=results)
        ctx.synchronize()
        var ours = Float64(perf_counter_ns() - t0) / 1e9

        # THE TREE-WISE ARM: the training-side apply
        # (`AddObliviousTreeImpl` per tree), for the record.
        var t1 = perf_counter_ns()
        ctx.enqueue_memset(pindex, UInt32(0))
        for feat in range(n_feats):
            ref cf = lay.features[feat]
            ctx.enqueue_function[binarize_float_feature_kernel](
                Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                xdev.unsafe_ptr() + feat * n_rows,
                Int32(n_rows),
                borders.unsafe_ptr() + border_off[feat],
                pindex.unsafe_ptr(),
                grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
            )
        predict(model, ctx, n_rows, folds, pindex, cursor)
        ctx.enqueue_copy(dst_ptr=h_cur.unsafe_ptr(), src_buf=cursor)
        ctx.synchronize()
        var ours_treewise = Float64(perf_counter_ns() - t1) / 1e9

        # per-rep accuracy: BOTH arms' predictions from RAW floats must
        # reproduce the fit's train mse, and each other
        var se = Float64(0.0)
        var se2 = Float64(0.0)
        var cross = Float64(0.0)
        for r in range(n_rows):
            var pe = Float64(h_res.unsafe_ptr().unsafe_load(r))
            var pt = Float64(h_cur.unsafe_ptr().unsafe_load(r))
            var dy = Float64(yp[r])
            se += (pe - dy) * (pe - dy)
            se2 += (pt - dy) * (pt - dy)
            var dd = pe - pt
            if dd < 0:
                dd = -dd
            if dd > cross:
                cross = dd
        var pmse = se / Float64(n_rows)
        var pmse2 = se2 / Float64(n_rows)
        for check in range(2):
            var m = pmse if check == 0 else pmse2
            var drift = m - train_mse
            if drift < 0:
                drift = -drift
            if drift > 1e-9 + 1e-5 * train_mse:
                raise Error(
                    "a raw-float predict arm does not reproduce the fit:"
                    " mse " + String(m) + " vs train " + String(train_mse)
                )
        # The two arms sum the same ~`trees` leaf values in different
        # orders, so their gap grows like sqrt(trees) in float32: measured
        # 9.5e-6 at 100 trees and 1.32e-4 at 8000 (the fixed 1e-4 bound
        # this replaces tripped on exactly that reorder noise). The bound
        # scales with the accumulation length and stays ~2x above the
        # measured points.
        var cross_tol = Float64(2e-6)
        var st = Float64(trees)
        # integer sqrt by doubling is enough for a tolerance
        var r = Float64(1.0)
        while r * r < st:
            r *= 2.0
        cross_tol *= r
        if cross > cross_tol:
            raise Error(
                "the evaluator and the tree-wise apply disagree by "
                + String(cross) + " (tolerance " + String(cross_tol) + ")"
            )

        var docs = Float64(n_rows)
        print(
            "  rep", rep,
            " catboost-cpu", their_full * 1000.0, "ms (",
            docs / their_full / 1e6, "M docs/s )  1-thread",
            their_one * 1000.0, "ms   ours-EVAL", ours * 1000.0, "ms (",
            docs / ours / 1e6, "M docs/s )  ours-treewise",
            ours_treewise * 1000.0, "ms   ratio EVAL vs full",
            ours / their_full, "  max cross-arm diff", cross,
        )
