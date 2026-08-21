"""One-hot categorical splits, gated analytically.

A three-category feature with `y = (category == 1)` separates the classes
at DEPTH 1 only through an EQUALITY split: `cat == 1` puts category 1
alone on one side. No threshold over the ordered codes {0, 1, 2} can --
`> 0` mixes {1,2}, `> 1` mixes {0,1} -- so at depth 1 the ordered
handling of the SAME data must end with a visibly worse loss. That
asymmetry is the whole gate, and it needs no CatBoost fixture: the
correct answer is arithmetic.

What it reaches, branch by branch:
* `build_layout(one_hot=...)` -> the CFeature flag -> `flat_one_hot`
  (the scan SKIPS the prefix sum for the feature, so histogram cells
  stay per-category masses), the resolve kernel's bin-feature table, and
  `split_and_make_sequence`'s `==` predicate: with the flag ON the tree
  must split (feature 0, bin 1) and reach ~zero loss at depth 1; with
  the flag OFF, same data, the loss floor is provably ~2/9.
* `predict`'s per-level `takeEqual` (`add_bin_values.mojo`): applying
  the one-hot model must reproduce the fit's final loss -- a predict
  that still compares `>` reconstructs the wrong leaves and misses by a
  margin no tolerance absorbs.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.hist2_check import build_cindex
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit, predict

comptime OH_ROWS = 2048
comptime OH_FEATURES = 4
comptime OH_FOLDS = 3


def check_one_hot() raises:
    print("one-hot categorical splits (equality vs threshold, analytic):")
    var ctx = DeviceContext()

    var folds = List[Int]()
    for _ in range(OH_FEATURES):
        folds.append(OH_FOLDS)

    var bins = List[List[Int]]()
    for f in range(OH_FEATURES):
        var col = List[Int]()
        for r in range(OH_ROWS):
            if f == 0:
                col.append(r % 3)
            else:
                col.append(0)
        bins.append(col^)
    var y = List[Float32]()
    for r in range(OH_ROWS):
        y.append(Float32(1.0) if (r % 3) == 1 else Float32(0.0))

    var one_hot = List[Bool]()
    one_hot.append(True)
    for _ in range(OH_FEATURES - 1):
        one_hot.append(False)

    var results = List[Float64]()
    var first_split_feature = List[Int]()
    var first_split_bin = List[Int]()
    for arm in range(2):
        var lay = build_layout(folds)
        var cindex = build_cindex(ctx, lay, bins, OH_ROWS)
        var targets = ctx.enqueue_create_buffer[DType.float32](OH_ROWS)
        var weights = ctx.enqueue_create_buffer[DType.float32](OH_ROWS)
        var ht = ctx.enqueue_create_host_buffer[DType.float32](OH_ROWS)
        var hw = ctx.enqueue_create_host_buffer[DType.float32](OH_ROWS)
        for r in range(OH_ROWS):
            ht.unsafe_ptr().unsafe_store(r, y[r])
            hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
        ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
        ctx.synchronize()

        var flags = List[Bool]()
        if arm == 0:
            flags = one_hot.copy()
        var model = TAdditiveModel()
        var losses = fit(
            model, ctx, OH_ROWS, folds, 1, cindex, targets, weights,
            False, 1, Float32(1.0), Float32(3.0), True,
            one_hot=flags,
        )
        results.append(losses[len(losses) - 1])
        if model.size() > 0 and model.weak_models[0].structure.get_depth() > 0:
            first_split_feature.append(
                Int(model.weak_models[0].structure.splits[0].feature_id)
            )
            first_split_bin.append(
                Int(model.weak_models[0].structure.splits[0].bin_idx)
            )
        else:
            first_split_feature.append(-1)
            first_split_bin.append(-1)

        # the predict path must agree with the fit under BOTH predicates
        var cursor = ctx.enqueue_create_buffer[DType.float32](OH_ROWS)
        predict(model, ctx, OH_ROWS, folds, cindex, cursor, one_hot=flags)
        var hc = ctx.enqueue_create_host_buffer[DType.float32](OH_ROWS)
        ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=cursor)
        ctx.synchronize()
        var se = Float64(0.0)
        for r in range(OH_ROWS):
            var d = Float64(hc.unsafe_ptr().unsafe_load(r)) - Float64(y[r])
            se += d * d
        var pmse = se / Float64(OH_ROWS)
        var drift = pmse - results[arm]
        if drift < 0:
            drift = -drift
        if drift > 1e-9 + 1e-5 * results[arm]:
            raise Error(
                "predict does not reproduce the fit under "
                + ("EQUALITY" if arm == 0 else "threshold")
                + " splits: " + String(pmse) + " vs "
                + String(results[arm])
            )

    # one-hot arm: split (0, 1), loss near zero
    if first_split_feature[0] != 0 or first_split_bin[0] != 1:
        raise Error(
            "the one-hot arm did not split category 1 of feature 0: got"
            " feature " + String(first_split_feature[0]) + " bin "
            + String(first_split_bin[0])
        )
    if results[0] > 0.02:
        raise Error("the equality split did not separate the classes:"
                    " loss " + String(results[0]))
    # ordered arm on the SAME data: provably stuck near 2/9
    if results[1] < 0.05:
        raise Error(
            "the ordered arm separated classes a threshold cannot"
            " separate (loss " + String(results[1]) + "): the one-hot"
            " flag is leaking or the scan skip is applied to ordered"
            " features"
        )
    print(
        "  one-hot: split (0, 1), loss", results[0],
        " | ordered same data: loss", results[1],
        " -- equality reachable only through the flag, and predict"
        " agrees with the fit under both predicates",
    )


def main() raises:
    # STANDALONE DRIVER, the same call `probe_main.mojo` makes. The
    # cardinality sweep is a DIFFERENT module (`one_hot_cardinality_check`,
    # `pixi run check-onehot-cardinality`) and stays there.
    check_one_hot()
