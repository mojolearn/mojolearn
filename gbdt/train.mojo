"""The user-facing surface: train from raw floats, predict raw floats.

NOT A PORT -- this is the convenience layer over ported machinery, the
`fit(X, y)` shape callers actually hold. Everything under it is the
transliterated pipeline: borders from `grid_creator.binarization`
(their GreedyLogSum, heap semantics included), device quantization
through `binarize_float_feature_kernel` (their BinarizeFloatFeatureImpl,
the same kernel their own predict quantizes with), the compressed index
through `write` layout rules, and `doc_parallel_boosting.fit`.

One stated difference from CatBoost's default pipeline: their quantizer
subsamples large datasets before computing borders; this computes them
from ALL rows. Same rule, more data -- the grids agree wherever theirs
did not subsample, and `binarization_check` holds the border parity on
the oracle fixture.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    BINARIZE_BLOCK_SIZE,
    BINARIZE_DOCS_PER_THREAD,
    binarize_float_feature_kernel,
)
from gbdt.ctrs.ctr_binarization import (
    TBinarizationOptions,
    compute_ctr_borders,
)
from gbdt.ctrs.ctr_calcers import compute_simple_ctrs
from gbdt.grid_creator.binarization import best_split
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit, predict
from gbdt.options.catboost_options import (
    COUNTER_CALC_FULL,
    SCORE_FUNCTION_COSINE,
    TCatFeatureParams,
)


@fieldwise_init
struct TrainedModel(Movable):
    """The ensemble plus the quantization grid it was trained on, which
    is what applying it to NEW raw floats requires -- their model carries
    its borders for the same reason."""

    var model: TAdditiveModel
    var fold_counts: List[Int]
    var one_hot: List[Bool]
    var borders: List[List[Float32]]
    var losses: List[Float64]
    var ctr_column_count: Int
    """How many of the columns above are CTR values rather than raw input
    features. Nonzero means `predict_floats` cannot apply this model to new
    raw rows: the CTR values were computed from the LEARN pool, and scoring
    a new row needs the final CTR tables their model file carries
    (`ctr_data.hash_map` in `save_model(format='json')`). No model file
    format exists here for the numeric case either, so that is a
    prerequisite rather than a CTR task -- `RECON_CTRS.md` step 5."""


def _build_cindex_from_floats(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
) raises -> DeviceBuffer[DType.uint32]:
    """Quantize every column and pack it into the compressed index.

    **`fold_counts` IS AN ARGUMENT, and that is the whole point.** This
    function used to re-derive it as `len(borders[f])`, which is right for
    an ordered feature and WRONG for a one-hot one: a k-category feature is
    given `k - 1` synthetic borders and `k` folds, because its equality
    candidates have to reach bin `k - 1`. The layout that WROTE the
    compressed index therefore disagreed with the layout `fit` and
    `predict` READ it with, and `policy_for_fold_count`
    (`grid_policy.mojo:83`) is a step function, so the two picked different
    packing policies wherever the pair straddled a step -- 15 folds go to
    HalfByte and 16 to OneByte, 1 fold to Binary and 2 to HalfByte.

    A 16-category one-hot feature was silently unlearnable as a result. No
    exception, no crash: a returned model that could not see the feature.
    The fit/predict consistency assertion in `train_api_check` was blind to
    it BY CONSTRUCTION, because `predict_floats` came through this same
    function and read through the same wrong layout, so the two agreed on
    the wrong answer. `mojo_only/one_hot_cardinality_check.mojo` is the gate
    that can see it and it sweeps every policy boundary.
    """
    var n_features = len(borders)
    if len(fold_counts) != n_features:
        raise Error(
            "fold_counts has "
            + String(len(fold_counts))
            + " entries for "
            + String(n_features)
            + " feature border lists"
        )
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))

    var xdev = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    # the count-first borders layout `binarize_float_feature_kernel`
    # reads (their `sharedBorders[0]` broadcast)
    var hbo = ctx.enqueue_create_host_buffer[DType.float32](256)
    var bdev = ctx.enqueue_create_buffer[DType.float32](256)
    comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
    for f in range(n_features):
        if len(borders[f]) == 0:
            continue
        ref cf = lay.features[f]
        for r in range(n_rows):
            hx.unsafe_ptr().unsafe_store(r, x_colmajor[f * n_rows + r])
        ctx.enqueue_copy(dst_buf=xdev, src_ptr=hx.unsafe_ptr())
        hbo.unsafe_ptr().unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_ptr().unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=bdev, src_ptr=hbo.unsafe_ptr())
        ctx.enqueue_function[binarize_float_feature_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            xdev.unsafe_ptr(), Int32(n_rows),
            bdev.unsafe_ptr(), cindex.unsafe_ptr(),
            grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
            block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
    return cindex^


def train(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_count: Int = 128,
    n_estimators: Int = 100,
    max_depth: Int = 6,
    learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    one_hot: List[Bool] = List[Bool](),
    bootstrap_bayesian: Bool = False,
    bagging_temperature: Float32 = Float32(1.0),
    random_seed: UInt64 = UInt64(0),
    score_function: Int = SCORE_FUNCTION_COSINE,
    cat_features: List[Bool] = List[Bool](),
    cat_feature_params: List[TCatFeatureParams] = List[TCatFeatureParams](),
) raises -> TrainedModel:
    """Borders -> device quantization -> fit, one call.

    `x_colmajor` is `[feature * n_rows + row]`. A feature marked in
    `one_hot` skips border search: its values ARE dense category codes
    `0..k-1` and its fold count is `max + 1`, split by equality.

    ## The categorical path

    `cat_features[f]` declares column `f` to hold DENSE CATEGORY CODES
    `0..k-1`. Their dispatch decides what happens to it
    (`binarizations_manager.cpp:106-115`):

        1 < k <= one_hot_max_size   ->  ONE-HOT, equality splits, no CTR
        otherwise                   ->  CTR columns, and the raw column is
                                        NOT a split candidate at all
        k == 1                      ->  raises, their
                                        `CB_ENSURE(uniqueValues > 1,
                                        "Error: useless catFeature found")`

    so a CTR-bearing categorical feature is REPLACED by its CTR columns
    rather than joined by them. Under `cat_feature_params`' shipped value
    that is ONE column per feature (FeatureFreq at one prior); under
    CatBoost's own default it would be FOUR, three of them the ordered
    target statistic -- see `TCatFeatureParams.feature_freq_only` for why
    this port ships the frequency half only, and say so in any comparison.

    `cat_feature_params` is a list because Mojo default arguments cannot
    call a raising constructor; EMPTY means
    `TCatFeatureParams.feature_freq_only()`, and more than one entry is
    refused. Pass exactly one to override.

    A feature may be in `cat_features` OR in `one_hot`, not both: `one_hot`
    is the older hand-driven surface where the caller has already made the
    one-hot decision, and `cat_features` is the one that makes it the way
    their dispatch does.
    """
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if len(cat_features) != 0 and len(cat_features) != n_features:
        raise Error(
            "cat_features has "
            + String(len(cat_features))
            + " entries for "
            + String(n_features)
            + " features"
        )
    if len(cat_feature_params) > 1:
        raise Error("pass at most one TCatFeatureParams")

    var cat_params: TCatFeatureParams
    if len(cat_feature_params) == 1:
        cat_params = cat_feature_params[0].copy()
    else:
        cat_params = TCatFeatureParams.feature_freq_only()
    cat_params.check()

    # --- the categorical pass, host side, over `x_colmajor` -------------
    #
    # It runs BEFORE border search for the same reason their pipeline
    # computes CTRs before quantizing them: a CTR value is an ordinary
    # float feature from here on, and it gets a grid like any other -- its
    # OWN grid, `cat_params.ctr_binarization_for(config)`, which is
    # MinEntropy 15 for FeatureFreq and Uniform 15 for Borders, not the
    # `border_count` GreedyLogSum the numeric columns take.
    var columns = List[List[Float32]]()
    var column_one_hot = List[Bool]()
    var column_ctr_grid = List[Int]()
    """-1 for a raw column, otherwise an index into `ctr_grids`."""
    var ctr_grids = List[TBinarizationOptions]()
    var ctr_column_count = 0

    var configs = cat_params.simple_ctr_configs()

    for f in range(n_features):
        var is_cat = len(cat_features) == n_features and cat_features[f]
        var flagged_one_hot = len(one_hot) == n_features and one_hot[f]
        if is_cat and flagged_one_hot:
            raise Error(
                "feature "
                + String(f)
                + " is in both cat_features and one_hot; cat_features makes"
                " the one-hot decision itself, from one_hot_max_size"
            )
        var col = List[Float32]()
        for r in range(n_rows):
            col.append(x_colmajor[f * n_rows + r])

        if not is_cat:
            columns.append(col^)
            column_one_hot.append(flagged_one_hot)
            column_ctr_grid.append(-1)
            continue

        # dense codes: cardinality is max + 1
        var maxc = 0
        for r in range(n_rows):
            var c = Int(col[r])
            if c < 0:
                raise Error(
                    "cat_features column "
                    + String(f)
                    + " holds a negative code; codes must be dense 0..k-1"
                )
            if c > maxc:
                maxc = c
        var unique_values = maxc + 1
        if unique_values <= 1:
            # their `CB_ENSURE(uniqueValues > 1)`
            # (`batch_binarized_ctr_calcer.cpp:150`)
            raise Error(
                "Error: useless catFeature found (feature "
                + String(f)
                + " has one category)"
            )

        if unique_values <= cat_params.one_hot_max_size:
            # `UseForOneHotEncoding` (`binarizations_manager.cpp:106-109`):
            # one-hot features never get CTRs.
            columns.append(col^)
            column_one_hot.append(True)
            column_ctr_grid.append(-1)
            continue

        var codes = List[UInt32]()
        for r in range(n_rows):
            codes.append(UInt32(Int(col[r])))
        var ctr_columns = compute_simple_ctrs(
            codes,
            unique_values,
            configs,
            List[UInt8](),
            cat_params.counter_calc_method == COUNTER_CALC_FULL,
        )
        for c in range(len(ctr_columns)):
            columns.append(ctr_columns[c].copy())
            column_one_hot.append(False)
            ctr_grids.append(cat_params.ctr_binarization_for(configs[c]))
            column_ctr_grid.append(len(ctr_grids) - 1)
            ctr_column_count += 1

    var n_columns = len(columns)
    var flat = List[Float32]()
    for c in range(n_columns):
        for r in range(n_rows):
            flat.append(columns[c][r])

    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    for f in range(n_columns):
        var flagged = column_one_hot[f]
        if flagged:
            var maxc = 0
            for r in range(n_rows):
                var c = Int(columns[f][r])
                if c > maxc:
                    maxc = c
            if maxc > 254:
                raise Error("one-hot feature " + String(f)
                            + " has more than 255 categories")
            # synthetic integer 'borders' 0.5, 1.5, ... so the SAME
            # quantize kernel maps code k to bin k
            var bs = List[Float32]()
            for c in range(maxc):
                bs.append(Float32(c) + Float32(0.5))
            fold_counts.append(len(bs) + 1 if len(bs) > 0 else 0)
            borders.append(bs^)
        elif column_ctr_grid[f] >= 0:
            # A CTR column takes its OWN grid, not the numeric one:
            # `GetOrComputeBorders(featureId, binarizationDescription, ...)`
            # in `batch_binarized_ctr_calcer.cpp:57-63`, with the
            # description coming from the ctr config
            # (`CreateDefaultCounter` -> MinEntropy 15 for FeatureFreq,
            # the two-argument TCtrDescription constructor -> Uniform 15
            # for Borders). Reading `border_count` here instead would be
            # the numeric GreedyLogSum grid on a CTR column, which is what
            # `tools/ctr_prep.py` used to do.
            var bs = compute_ctr_borders(columns[f], ctr_grids[
                column_ctr_grid[f]
            ])
            fold_counts.append(len(bs))
            borders.append(bs^)
        else:
            var col = columns[f].copy()
            var bs = best_split(col^, border_count)
            fold_counts.append(len(bs))
            borders.append(bs^)

    # one-hot features occupy folds+? -- for ordered features CatBoost's
    # fold count IS the border count; a one-hot feature has k categories
    # = k bins reached by k-1 synthetic borders, and its fold count must
    # cover bin k-1 for the equality candidates, hence len+1 above.

    var cindex = _build_cindex_from_floats(
        ctx, flat, n_rows, borders, fold_counts
    )

    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, y[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, fold_counts, max_depth, cindex, targets,
        weights, False, n_estimators, learning_rate, l2_leaf_reg, True,
        bootstrap_bayesian=bootstrap_bayesian,
        bagging_temperature=bagging_temperature,
        random_seed=random_seed,
        one_hot=column_one_hot,
        score_function=score_function,
    )
    return TrainedModel(
        model^,
        fold_counts^,
        column_one_hot^,
        borders^,
        losses^,
        ctr_column_count,
    )


def predict_floats(
    ctx: DeviceContext,
    tm: TrainedModel,
    x_colmajor: List[Float32],
    n_rows: Int,
) raises -> List[Float32]:
    """Apply a trained model to NEW raw floats: quantize against the
    model's own grid (as their predict does internally), then the
    tree-wise apply the probe suite pins to the learn cursor."""
    var n_features = len(tm.fold_counts)
    if tm.ctr_column_count > 0:
        raise Error(
            "predict_floats cannot apply a model with "
            + String(tm.ctr_column_count)
            + " CTR columns: a CTR value is a statistic of the LEARN pool,"
            " and scoring a new row needs the final CTR tables their model"
            " file carries (ctr_data.hash_map in save_model(format='json')"
            "). No model file format exists in this port for the numeric"
            " case either, so that is the prerequisite -- RECON_CTRS.md"
            " step 5. Refused rather than scored against a grid the rows"
            " were never mapped onto"
        )
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    var cindex = _build_cindex_from_floats(
        ctx, x_colmajor, n_rows, tm.borders, tm.fold_counts
    )
    var cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    predict(
        tm.model, ctx, n_rows, tm.fold_counts, cindex, cursor,
        one_hot=tm.one_hot,
    )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=cursor)
    ctx.synchronize()
    var out = List[Float32]()
    for r in range(n_rows):
        out.append(hc.unsafe_ptr().unsafe_load(r))
    return out^
