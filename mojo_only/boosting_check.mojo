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
- the loss to reach 2% of that mean baseline, which is the LEVEL a correct
  depth-4 twelve-tree fit reaches on this target (it reaches about 1%)
- most splits to land on a feature the target actually depends on

The third is the sharp one for the GRADIENT. A sign error in the gradient, a
missing learning rate, or a leaf value computed from the wrong stats plane
all still produce a loss that moves; only a correct step produces one that
never goes back up.

The last two are the sharp ones for the SPLIT SEARCH, and they were added
after the first three passed a model that was 22x worse than it should be. A
corrupt histogram does not disturb the leaf values at all -- those come from
`compute_partition_stats`, which never reads the histogram -- so a tree built
on arbitrary splits with correct leaf values is still monotone and still
beats the mean. Nothing above the fourth assertion can see it. Whenever this
file grows a new assertion, ask which of the two halves it constrains.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import fit, predict
from gbdt.models.oblivious_model import TAdditiveModel


def check_boosting_learns(
    n_rows: Int = 8192, n_estimators: Int = 12, max_depth: Int = 4,
    use_subtraction: Bool = True,
) raises:
    var ctx = DeviceContext()

    # 16 HALF-BYTE features: 15 folds is `policy_max_folds(POLICY_HALF_BYTE)`,
    # so `policy_for_fold_count` puts every one of them under the half-byte
    # policy and this check exercises that kernel and no other. It said
    # "one-byte" here and that was simply wrong, which mattered: it is the
    # reason nobody looked at `hist_half_byte.mojo` when this check moved.
    # The target depends on three of them, so most
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
        n_estimators, Float32(0.3), Float32(3.0), use_subtraction,
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

    # ================== HOW WELL, NOT MERELY WHETHER ==================
    # Monotone and "beats the mean" are BOTH TRUE OF A BADLY BROKEN MODEL.
    # Measured: a fixed-point histogram that overflowed Int32 on the top two
    # levels of every tree scored 14.79 here, against 0.66 for the same code
    # with the scale bounded. Both pass every assertion above, because the
    # leaf VALUES come from `compute_partition_stats` and never touch the
    # histogram: only the SPLITS were garbage, and a tree with arbitrary
    # splits and correct leaf values still fits a monotone decreasing
    # sequence. The three assertions above are blind to the split search.
    #
    # So this asserts a LEVEL. Twelve depth-4 trees at learning rate 0.3 on a
    # target this tree class can represent reach 0.66, which is 1.0% of the
    # 66.46 mean baseline. The bar is 2%, double the measured value, and the
    # overflow run was 22.3%. It is expressed as a FRACTION of the baseline
    # rather than an absolute so it survives a change to the dataset size or
    # the target's scale.
    var reached = losses[len(losses) - 1] / variance
    print("    final loss as a fraction of the mean baseline:", reached)
    if reached > 0.02:
        raise Error(
            String("boosting learns, but far less than a correct depth-")
            + String(max_depth) + " " + String(n_estimators)
            + "-tree model does: final mse "
            + String(losses[len(losses) - 1]) + " is "
            + String(reached * 100.0)
            + "% of the mean baseline " + String(variance)
            + ", and a correct fit reaches about 1%. Monotone and"
            + " better-than-the-mean are already true above, so what this"
            + " catches is a split search reading a corrupt histogram"
            + " while the leaf values stay right"
        )

    # ================== AND ON WHICH FEATURES ==================
    # The target is built from features 0, 3 and 7. The other thirteen are
    # noise drawn from the same generator, so a search that is working picks
    # from the three and a search reading a corrupt histogram picks close to
    # uniformly, which is 3 of 16.
    #
    # This localizes what the loss bar only detects: it fails in the SPLIT
    # SEARCH and not in the leaf values, the boosting loop or the model
    # round-trip, all of which the assertions around it cover.
    var on_signal = 0
    var total_splits = 0
    for t in range(model.size()):
        ref wm = model.weak_models[t]
        for lv in range(wm.structure.get_depth()):
            var f = Int(wm.structure.splits[lv].feature_id)
            total_splits += 1
            if f == 0 or f == 3 or f == 7:
                on_signal += 1
    print(
        "    splits on a signal feature (0, 3, 7):", on_signal, "of",
        total_splits,
    )
    if total_splits == 0:
        raise Error("no tree in the ensemble carries a split at all")
    if on_signal * 2 <= total_splits:
        raise Error(
            String("only ") + String(on_signal) + " of "
            + String(total_splits)
            + " splits landed on a feature the target depends on; the"
            + " thirteen noise features are 13 of 16, so this is a split"
            + " search reading a histogram that does not rank candidates"
        )

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
