"""Does boosting actually learn? Fit a signal and watch the loss fall.

Every check in this repository up to now has verified a PIECE: a histogram
cell, a partition, a leaf count. None of them can tell you whether the pieces
compose into gradient boosting, because a tree that splits correctly on
garbage gradients still splits correctly.

This is the first end-to-end assertion. It builds a dataset whose target is a
known function of a few features, runs `fit`, and requires:

- the loss to FALL from iteration 1 to the last
- the loss to fall BELOW the variance of the target, which is the loss of
  predicting the mean and is therefore the bar any real learner clears
- the loss to be monotonically non-increasing across iterations, which is
  what a correct gradient step on a convex objective gives

The third is the sharp one. A sign error in the gradient, a missing learning
rate, or a leaf value computed from the wrong stats plane all still produce a
loss that moves; only a correct step produces one that never goes back up.
"""

from max.gpu.host import DeviceContext

from ported.gpu_data.compressed_index_builder import build_layout
from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from ported.methods.doc_parallel_boosting import fit, predict
from ported.models.oblivious_model import TAdditiveModel


def check_boosting_learns(
    n_rows: Int = 8192, n_estimators: Int = 12, max_depth: Int = 4
) raises:
    var ctx = DeviceContext()

    # 16 one-byte features. The target depends on three of them, so most
    # features are noise and the split search has to find the signal.
    var folds = List[Int]()
    for _ in range(16):
        folds.append(15)
    var n_features = len(folds)

    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var host_bin = List[List[Int]]()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        var col = List[Int]()
        ref cf = lay.features[f]
        for r in range(n_rows):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var v = Int(x % UInt32(folds[f]))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # A LEARNABLE target: an additive function of features 0, 3 and 7 plus an
    # interaction, which a depth-4 symmetric tree can represent.
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var mean = Float64(0.0)
    for r in range(n_rows):
        var y = (
            Float32(host_bin[0][r]) * 1.5
            - Float32(host_bin[3][r]) * 0.75
            + Float32(host_bin[7][r]) * 0.5
        )
        if host_bin[0][r] > 7 and host_bin[3][r] > 7:
            y += 4.0
        ht.unsafe_ptr().unsafe_store(r, y)
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
        mean += Float64(y)
    mean /= Float64(n_rows)
    var variance = Float64(0.0)
    for r in range(n_rows):
        var d = Float64(ht.unsafe_ptr().unsafe_load(r)) - mean
        variance += d * d
    variance /= Float64(n_rows)

    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, folds, max_depth, cindex, targets, weights, False,
        n_estimators, Float32(0.3), Float32(3.0),
    )

    print("    predicting the mean would score", variance)
    for i in range(len(losses)):
        print("    iter", i + 1, "mse", losses[i])

    if len(losses) != n_estimators:
        raise Error("fit returned the wrong number of iterations")

    var rises = 0
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1] * 1.000001:
            rises += 1
    print("    iterations where the loss ROSE:", rises)

    if losses[len(losses) - 1] >= losses[0]:
        raise Error("boosting did not reduce the loss at all")
    if losses[len(losses) - 1] >= variance:
        raise Error(
            "boosting did not beat predicting the mean: "
            + String(losses[len(losses) - 1])
            + " against "
            + String(variance)
        )
    if rises != 0:
        raise Error(
            String("the loss rose on ") + String(rises)
            + " iterations; a correct gradient step on a convex objective"
            + " does not go back up"
        )
    print("  boosting learns: loss falls monotonically and beats the mean")

    # ---- the stored model must BE the trained model ----
    #
    # Training updated the cursor by reading each row's leaf off the
    # partition. Prediction walks the stored splits instead and never sees a
    # partition. If the two disagree, the thing that was stored is not the
    # thing that was trained, and every piece-wise check would still pass.
    #
    # This is also the only check that can see a reversed leaf-index bit
    # order: reading the split bits the other way round permutes the leaves,
    # so every total is preserved and every individual prediction is wrong.
    print("    tree 0 structure:")
    ref w0 = model.weak_models[0]
    var line = String("      ")
    for lv in range(w0.structure.get_depth()):
        line += (
            "(" + String(Int(w0.structure.splits[lv].feature_id)) + ","
            + String(Int(w0.structure.splits[lv].bin_idx)) + ") "
        )
    print(line)
    var nonzero = 0
    for i in range(len(w0.leaf_values)):
        if w0.leaf_values[i] != Float32(0.0):
            nonzero += 1
    print("      leaves with a non-zero value:", nonzero, "of",
          len(w0.leaf_values))

    var pred = ctx.enqueue_create_buffer[DType.float32](n_rows)
    predict(model, ctx, n_rows, folds, cindex, pred)
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(), src_buf=pred)
    ctx.synchronize()

    var sse_pred = Float64(0.0)
    var worst = Float64(0.0)
    for r in range(n_rows):
        var d = Float64(
            ht.unsafe_ptr().unsafe_load(r) - hp.unsafe_ptr().unsafe_load(r)
        )
        sse_pred += d * d
    var mse_pred = sse_pred / Float64(n_rows)
    print("    replayed from the stored model, mse", mse_pred)
    print("    training cursor mse            ", losses[len(losses) - 1])

    var gap = mse_pred - losses[len(losses) - 1]
    if gap < 0.0:
        gap = -gap
    var tol = 1.0e-3 * losses[len(losses) - 1]
    if gap > tol:
        raise Error(
            String("the stored model does not reproduce the trained one: ")
            + String(mse_pred) + " against "
            + String(losses[len(losses) - 1])
        )
    print("  the stored model reproduces training to", gap, "mse")
