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
from gbdt.ctrs.ctr import TCtrConfig, is_permutation_dependent_ctr_type
from gbdt.ctrs.ctr_binarization import (
    TBinarizationOptions,
    build_binarized_target,
    build_target_borders,
    compute_ctr_borders,
)
from gbdt.ctrs.ctr_calcers import compute_simple_ctrs, compute_simple_ctrs_gpu
from gbdt.data.permutation import (
    DEFAULT_PERMUTATION_COUNT,
    ctrs_estimation_permutation,
)
from gbdt.grid_creator.binarization import best_split
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    build_ctr_tables,
    column_plan,
    expand_raw_columns,
)
from std.math import exp
from gbdt.methods.doc_parallel_boosting import (
    TAdditiveModel,
    fit,
    model_approx_dim,
    predict,
)
from gbdt.gpu_util.kernel.bootstrap import (
    BOOTSTRAP_KERNEL_BAYESIAN,
    BOOTSTRAP_KERNEL_BERNOULLI,
    BOOTSTRAP_KERNEL_POISSON,
)

#: `subsample`'s default, 0.66 (`bootstrap_options.h:15`). Their own
#: `SetDefault(0.8)` at `catboost_options.cpp:798` is the MVS arm only,
#: and MVS does not reach their GPU oblivious searcher.
comptime DEFAULT_SUBSAMPLE = Float32(0.66)
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_RMSE,
)
from gbdt.options.loss_description import make_loss_description
from gbdt.options.catboost_options import (
    COUNTER_CALC_FULL,
    SCORE_FUNCTION_COSINE,
    TCatFeatureParams,
    set_leaves_estimation_default,
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
    features. It is a SAFETY count: a model that declares CTR columns and
    carries no tables for them cannot be applied to raw rows, and
    `predict_floats` refuses it. It survives serialization for exactly that
    reason -- a round trip that dropped it would turn a model that refuses
    into one that silently scores."""
    var ctr_tables: List[TCtrValueTable]
    """The apply-time CTR tables, one per CTR column, their
    `ctr_data.hash_map` (`libs/model/ctr_value_table.h`). With these
    present and covering every declared CTR column, `predict_floats` maps a
    raw category to the statistic the LEARN pool produced and scores the
    row; without them it refuses. See
    `gbdt/models/ctr_value_table.mojo`."""


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
    bootstrap_type: String = String(""),
    subsample: Float32 = Float32(-1.0),
    random_seed: UInt64 = UInt64(0),
    score_function: Int = SCORE_FUNCTION_COSINE,
    cat_features: List[Bool] = List[Bool](),
    cat_feature_params: List[TCatFeatureParams] = List[TCatFeatureParams](),
    ctr_estimation_permutation_id: Int = DEFAULT_PERMUTATION_COUNT - 1,
    loss: String = "RMSE",
    loss_alpha: Float32 = Float32(-1.0),
    loss_q: Float32 = Float32(-1.0),
    loss_delta: Float32 = Float32(-1.0),
    loss_variance_power: Float32 = Float32(-1.0),
    loss_border: Float32 = Float32(-1.0),
    leaf_estimation_iterations: Int = -1,
    leaf_estimation_method: Int = -1,
    class_weights: List[Float32] = List[Float32](),
) raises -> TrainedModel:
    """Borders -> device quantization -> fit, one call.

    `loss` takes their `ELossFunction` spellings, and every objective their
    GPU pointwise target ships is here:

        RMSE  Logloss  CrossEntropy  Quantile  MAE  LogLinQuantile  MAPE
        Poisson  Lq  Expectile  Tweedie  Huber

    Four of them REQUIRE a parameter, as theirs do: `Lq` needs `loss_q`,
    `Huber` needs `loss_delta`, `Tweedie` needs `loss_variance_power`,
    `Expectile` needs `loss_alpha`. `Quantile` and `LogLinQuantile` take
    `loss_alpha` and default it to 0.5; `Logloss` takes `loss_border` and
    defaults it to 0.5. A missing mandatory parameter raises here, where
    theirs raises (`catboost_options.cpp:82`, `:126`, `:222`).

    THE LEAF ESTIMATOR IS CHOSEN BY THE LOSS, not by this signature.
    `set_leaves_estimation_default` is the port of their
    `SetLeavesEstimationDefault` (`catboost_options.cpp:273-360`) and it
    is what decides Newton vs Gradient vs Exact and how many iterations --
    ten for Logloss, twenty for Tweedie, one for RMSE, and Exact for MAE /
    MAPE / Quantile. `leaf_estimation_method` and
    `leaf_estimation_iterations` override it and default to -1 meaning
    "unset", which is their `TOption::NotSet()`.

    PREDICTIONS STAY RAW APPROXES for every loss, exactly like their
    `predict` without a prediction_type: a Logloss caller applies the
    sigmoid to `predict_floats` output, a Poisson caller applies `exp`,
    and that is also what keeps the harness adapters honest about what
    they time.

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
    rather than joined by them. Under the default that is FOUR columns per
    feature -- three `Borders` priors plus one `FeatureFreq`, CatBoost's
    own GPU `simple_ctr` -- and under `TCatFeatureParams.feature_freq_only()`
    it is ONE.

    ## The two writers, and the two orders

    Their builder writes CTR columns TWICE, from the same driver and two
    different orders (`doc_parallel_dataset_builder.cpp:190-262`):

        MakeSequence(ctrEstimationOrder)            :206  identity
        writeCtrs(..., permutationIndependent)      :229  FeatureFreq
        for permutationId in [0, permutation_count):
            ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)  :255
            writeCtrs(..., permutationDependent)    :257  Borders

    This function does the same split, off
    `IsPermutationDependentCtrType` (`ctr_type.cpp:44-58`), which is what
    their `SplitByPermutationDependence` (`:81`) keys on. The independent
    half runs on the host (`compute_simple_ctrs`); the dependent half runs
    on the device (`compute_simple_ctrs_gpu`) over
    `ctrs_estimation_permutation(n_rows, ctr_estimation_permutation_id)`.

    **`ctr_estimation_permutation_id` defaults to `permutation_count - 1`,
    which is 3, and it is NOT the identity.** Their permutation 0 IS the
    identity (`permutation.cpp:14-17`) and is safe on their side only
    because the learn pool was already shuffled at load
    (`private/libs/algo/preprocess.cpp:183-199`, which shuffles whenever the
    data has a categorical feature and `has_time` is false). This port has
    no such stage, so a caller can hand us rows sorted by target, where the
    identity order makes every row's ordered statistic read its own
    neighbourhood. `permutation_count - 1` is their ESTIMATION permutation,
    `GetEstimationPermutation()` (`doc_parallel_boosting.h:101-103`), the
    one whose model `Run()` exports. Only ONE of their four column sets is
    built -- `PORTING.md` 55.

    `cat_feature_params` is a list because Mojo default arguments cannot
    call a raising constructor; EMPTY means `TCatFeatureParams.default()`,
    and more than one entry is refused. Pass exactly one to override.

    **THE FALLBACK IS `TCatFeatureParams.default()`, WHICH IS CATBOOST'S
    OWN GPU `simple_ctr`**, as of the commit that built the `Borders`
    apply-time tables. It was `feature_freq_only()` for one round, for one
    reason -- a `Borders` model could train and not score, because
    `build_ctr_tables` had no histogram arm -- and that reason is gone:
    `predict_floats` now maps a raw category through a `Borders` table the
    same way it does a `FeatureFreq` one. A switch that outlives its
    reason is a defect (`PORTING_RULES.md` 8), so both sides stay
    exercised: `mojo_only/ctr_apply_check.mojo` and
    `mojo_only/ctr_train_check.mojo` each run the default AND
    `feature_freq_only()` explicitly.

    What this changes for a caller who passes nothing: four columns where
    there was one, a device pass over the CTR estimation permutation where
    there was a host frequency count, and an applied model whose learn-row
    predictions no longer reproduce the fit's loss bit for bit -- because
    the ordered statistic a `Borders` column is trained on is not the
    full-learn-set histogram an applied model carries. That gap is
    CatBoost's design, not a defect here; see
    `gbdt/models/ctr_value_table.mojo`.

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
        cat_params = TCatFeatureParams.default()
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
    var ctr_tables = List[TCtrValueTable]()

    var configs = cat_params.simple_ctr_configs()

    # `SplitByPermutationDependence` (`doc_parallel_dataset_builder.cpp:81`)
    # over `IsPermutationDependentCtrType` (`ctr_type.cpp:44-58`), by config
    # slot so the columns can be put back in config order afterwards.
    var independent_slots = List[Int]()
    var dependent_slots = List[Int]()
    var independent_configs = List[TCtrConfig]()
    var dependent_configs = List[TCtrConfig]()
    for c in range(len(configs)):
        if is_permutation_dependent_ctr_type(configs[c].ctr_type):
            dependent_slots.append(c)
            dependent_configs.append(configs[c])
        else:
            independent_slots.append(c)
            independent_configs.append(configs[c])

    # `BuildCtrTarget` -> `BuildBinarizedTarget`
    # (`gpu_data/dataset_helpers.cpp:137-151`), the grid every
    # binarized-target CTR reads. Built once for the fit, because the GPU
    # refuses a per-CTR override outright (`catboost_options.cpp:505`).
    # Skipped when nothing is permutation dependent, which is what their
    # `CreateCtrConfigsFromDescription`'s `!HasTargetBinarization()`
    # `continue` (`binarizations_manager.cpp:397-399`) amounts to here.
    var binarized_target = List[UInt8]()
    var ctr_order = List[UInt32]()
    var target_classes_count = 0
    if len(dependent_configs) > 0:
        var target_borders = build_target_borders(
            y, cat_params.target_binarization
        )
        # `TTargetClassifier::GetClassesCount()` is `Borders.ysize() + 1`
        # (`libs/model/target_classifier.h:32-34`), and it is what
        # `CalcFinalCtrsImpl` allocates the apply-time blob's second axis
        # from (`private/libs/algo/online_ctr.cpp:909-910`). Taken from the
        # borders MinEntropy actually returned rather than from the option,
        # because their classifier counts the borders it holds.
        target_classes_count = len(target_borders) + 1
        binarized_target = build_binarized_target(y, target_borders)
        ctr_order = ctrs_estimation_permutation(
            n_rows, ctr_estimation_permutation_id
        ).fill_order()

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

        var ctr_columns = List[List[Float32]]()
        for _ in range(len(configs)):
            ctr_columns.append(List[Float32]())

        if len(independent_configs) > 0:
            # their `writeCtrs(..., permutationIndependent)` (`:229`),
            # over the identity `ctrEstimationOrder` (`:206`)
            var indep = compute_simple_ctrs(
                codes,
                unique_values,
                independent_configs,
                List[UInt8](),
                cat_params.counter_calc_method == COUNTER_CALC_FULL,
            )
            for c in range(len(independent_slots)):
                ctr_columns[independent_slots[c]] = indep[c].copy()

        if len(dependent_configs) > 0:
            # their `writeCtrs(..., permutationDependent)` (`:257`), over
            # the CTR ESTIMATION PERMUTATION written at `:255`
            var dep = compute_simple_ctrs_gpu(
                ctx,
                codes,
                unique_values,
                dependent_configs,
                binarized_target,
                ctr_order,
            )
            for c in range(len(dependent_slots)):
                ctr_columns[dependent_slots[c]] = dep[c].copy()
        # the APPLY-TIME half of the same statistic: their
        # `CalcFinalCtrs` writes a `TCtrValueTable` beside every CTR the
        # model uses, because the learn column cannot score a new row.
        # Built from the codes and the BINARIZED TARGET computed above --
        # the same one the Borders writer read -- rather than from the
        # columns, so it is the counts and the priors that travel and not
        # the divided values. See gbdt/models/ctr_value_table.mojo, and
        # note that the Borders table is the FULL-LEARN-SET histogram and
        # carries no permutation: the ordered statistic stops at training.
        var tables = build_ctr_tables(
            codes,
            unique_values,
            configs,
            binarized_target,
            target_classes_count,
            f,
            len(columns),
        )
        for c in range(len(tables)):
            ctr_tables.append(tables[c].copy())
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

    # `class_weights`: their `MakeClassificationWeights` applied at pool
    # build -- `weight_i = rawWeight_i * classWeights[class_i]`
    # (`target/data_providers.cpp:158-176`). Python's `scale_pos_weight=w`
    # is the two-class spelling `class_weights=[1, w]`, and their check
    # demands exactly two entries for Logloss (`catboost_options.cpp:
    # 617-623`). The class index takes the same 0.5 threshold the Logloss
    # border defaults to.
    var use_class_weights = len(class_weights) > 0
    if use_class_weights:
        if len(class_weights) != 2:
            raise Error(
                "class_weights takes exactly two entries [w0, w1] here,"
                " their binary-Logloss contract"
            )
        if loss == "RMSE":
            raise Error(
                "class weights take effect only with Logloss, their"
                " option check's words (catboost_options.cpp:617)"
            )
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, y[r])
        var w = Float32(1.0)
        if use_class_weights:
            w = class_weights[1 if y[r] > Float32(0.5) else 0]
        hw.unsafe_ptr().unsafe_store(r, w)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var loss_desc = make_loss_description(
        loss,
        alpha=loss_alpha,
        q=loss_q,
        delta=loss_delta,
        variance_power=loss_variance_power,
        border=loss_border,
    )
    var objective = loss_desc.loss_function

    # THE CLASS COUNT COMES FROM THE LABEL COLUMN, as their
    # `TClassificationTargetHelper` derives it, and it is derived rather
    # than taken as a parameter because a count that disagrees with the
    # data is a wrong model rather than an error.
    #
    # Their labels for MultiClass are DENSE CLASS CODES `0..k-1`, which is
    # what `MultiLogitValAndFirstDerImpl` reads with
    # `static_cast<ui16>(targetClasses[idx])` (`multilogit.cu:41`) and
    # indexes prediction planes with. A non-integral or negative label is
    # refused here rather than truncated silently on the device.
    var num_classes = 0
    if objective == OBJECTIVE_MULTICLASS:
        var mx = -1
        for r in range(n_rows):
            var v = y[r]
            if v < Float32(0.0):
                raise Error(
                    "MultiClass label at row " + String(r)
                    + " is negative; labels are dense class codes 0..k-1"
                )
            var iv = Int(v)
            if Float32(iv) != v:
                raise Error(
                    "MultiClass label at row " + String(r)
                    + " is not an integer; labels are dense class codes"
                    " 0..k-1"
                )
            if iv > mx:
                mx = iv
        num_classes = mx + 1
        if num_classes < 2:
            raise Error(
                "MultiClass needs at least two classes; the labels"
                " reach only " + String(num_classes)
            )
    var estimation = set_leaves_estimation_default(
        loss_desc,
        method_override=leaf_estimation_method,
        iterations_override=leaf_estimation_iterations,
    )

    # `TCatBoostOptions::SetLeavesEstimationDefault`'s sibling for
    # sampling (`catboost_options.cpp:779-800`), the two lines of it this
    # port can reach. `bootstrap_type` empty means "take the
    # `bootstrap_bayesian` shorthand", which is what every existing caller
    # passes; a name selects one of their three GPU draws.
    var boot_kind = -1
    var boot_param = bagging_temperature
    if bootstrap_type != String(""):
        if bootstrap_type == "Bayesian":
            boot_kind = BOOTSTRAP_KERNEL_BAYESIAN
            boot_param = bagging_temperature
            if subsample >= Float32(0.0):
                # their own validator, verbatim in intent
                # (`catboost_options.cpp:795`)
                raise Error(
                    "Error: default bootstrap type (bayesian) doesn't"
                    " support 'subsample' option"
                )
        elif bootstrap_type == "Bernoulli":
            boot_kind = BOOTSTRAP_KERNEL_BERNOULLI
            boot_param = (
                subsample if subsample >= Float32(0.0)
                else DEFAULT_SUBSAMPLE
            )
        elif bootstrap_type == "Poisson":
            boot_kind = BOOTSTRAP_KERNEL_POISSON
            boot_param = (
                subsample if subsample >= Float32(0.0)
                else DEFAULT_SUBSAMPLE
            )
        elif bootstrap_type == "No":
            boot_kind = -1
        elif bootstrap_type == "MVS":
            # `Y_ASSERT(config.GetBootstrapType() != EBootstrapType::MVS)`
            # (`weak_objective_impl.h:30`): their own GPU oblivious
            # searcher refuses MVS, so this port has nothing to port.
            raise Error(
                "MVS is not reachable from their GPU oblivious searcher"
                " (weak_objective_impl.h:30 asserts it away)"
            )
        else:
            raise Error(
                "unknown bootstrap_type '" + bootstrap_type
                + "': Bayesian, Bernoulli, Poisson, No"
            )

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, fold_counts, max_depth, cindex, targets,
        weights, use_class_weights, n_estimators, learning_rate,
        l2_leaf_reg, True,
        bootstrap_bayesian=bootstrap_bayesian,
        bagging_temperature=bagging_temperature,
        bootstrap_type=boot_kind,
        bootstrap_param=boot_param,
        random_seed=random_seed,
        one_hot=column_one_hot,
        score_function=score_function,
        objective=objective,
        num_classes=num_classes,
        logloss_border=loss_desc.get_logloss_border(),
        leaf_estimation_iterations=estimation.iterations,
        leaf_estimation_method=estimation.method,
        alpha=loss_desc.kernel_alpha(),
        # THEIR SECOND ALPHA. `ComputeWeightedQuantile` reads the quantile
        # level from the loss params map, default 0.5
        # (`leaves_estimation_helper.h:72-74`), NOT from the float the
        # target kernel receives. `get_alpha()` IS that accessor
        # (`loss_description.cpp:95-102`). They coincide for MAE and
        # Quantile and differ for MAPE, whose kernel alpha is 0.
        estimator_alpha=loss_desc.get_alpha(),
    )
    return TrainedModel(
        model^,
        fold_counts^,
        column_one_hot^,
        borders^,
        losses^,
        ctr_column_count,
        ctr_tables^,
    )


def model_input_features(tm: TrainedModel) raises -> Int:
    """How many RAW input columns `predict_floats` expects for this model.

    For a float-only or one-hot model that is just the column count. For a
    model with CTR tables it is fewer, because one categorical input stands
    behind one column per CTR config -- the shape their
    `TStaticCtrProvider` reconstructs from the model rather than being
    told."""
    if len(tm.ctr_tables) == 0:
        return len(tm.fold_counts)
    return column_plan(
        tm.ctr_tables, len(tm.fold_counts)
    ).n_input_features


def predict_floats(
    ctx: DeviceContext,
    tm: TrainedModel,
    x_colmajor: List[Float32],
    n_rows: Int,
) raises -> List[Float32]:
    """Apply a trained model to NEW raw floats: quantize against the
    model's own grid (as their predict does internally), then the
    tree-wise apply the probe suite pins to the learn cursor.

    ## What `x_colmajor` holds for a CATEGORICAL model

    RAW INPUT COLUMNS, not model columns. A high-cardinality categorical
    input is REPLACED by one CTR column per config
    (`binarizations_manager.cpp:106-115`), so a model can have more columns
    than the caller has features; the categorical column still arrives as
    its DENSE CODES and this function maps each code through the model's
    CTR tables before quantizing, which is their
    `TStaticCtrProvider::CalcCtrs` step
    (`libs/model/static_ctr_provider.cpp:14-71`) run ahead of the
    quantizer. For a model with no CTR tables the two spaces are the same
    and nothing about the old contract changes.

    ## The refusal, and when it lifts

    A CTR value is a statistic of the LEARN pool, so a model that declares
    CTR columns and does not carry their tables cannot score a new row at
    all, and this refuses rather than quantizing raw codes against a grid
    built for frequencies. It lifts when the tables are PRESENT and cover
    every declared CTR column -- never merely because the count went
    missing, which is why `ctr_column_count` travels through save and load
    beside them."""
    var n_features = len(tm.fold_counts)
    if tm.ctr_column_count != len(tm.ctr_tables):
        raise Error(
            "predict_floats cannot apply a model with "
            + String(tm.ctr_column_count)
            + " CTR columns and "
            + String(len(tm.ctr_tables))
            + " CTR tables: a CTR value is a statistic of the LEARN pool,"
            " and scoring a new row needs the final CTR tables their model"
            " file carries (ctr_data.hash_map in save_model(format='json')"
            "). Refused rather than scored against a grid the rows were"
            " never mapped onto"
        )
    var expanded: List[Float32]
    if len(tm.ctr_tables) == 0:
        if len(x_colmajor) != n_rows * n_features:
            raise Error("x_colmajor size mismatch")
        expanded = x_colmajor.copy()
    else:
        # their CalcCtrs, then the ordinary quantizer
        expanded = expand_raw_columns(
            tm.ctr_tables, n_features, x_colmajor, n_rows
        )
    var cindex = _build_cindex_from_floats(
        ctx, expanded, n_rows, tm.borders, tm.fold_counts
    )
    # A MULTI-DIMENSIONAL MODEL RETURNS `n_rows * approx_dim` VALUES, and
    # they come back PLANE-MAJOR -- `[dim * n_rows + row]` -- because that
    # is the cursor's layout and `predict_multi` is what reshapes it. This
    # entry point keeps its one-dimensional contract and refuses anything
    # wider rather than silently returning the first class's approxes.
    var approx_dim = model_approx_dim(tm.model)
    if approx_dim != 1:
        raise Error(
            "predict_floats is one-dimensional; this model has"
            " approx_dim " + String(approx_dim)
            + ". Use predict_multi_floats, which returns"
            " [row * approx_dim + dim]."
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


def predict_multi_floats(
    ctx: DeviceContext,
    tm: TrainedModel,
    x_colmajor: List[Float32],
    n_rows: Int,
) raises -> List[Float32]:
    """`predict_floats` for a multi-dimensional model, ROW-MAJOR out.

    Returns `n_rows * approx_dim` values as `[row * approx_dim + dim]`,
    which is the layout a caller iterating rows wants and the layout the
    MODEL stores its leaves in. The cursor is PLANE-MAJOR on the device;
    the transpose happens here, once, on the host.

    FOR MULTICLASS `approx_dim` IS `numClasses - 1`, not `numClasses`. The
    last class's approx is pinned at zero and is not stored -- that is the
    gauge the whole port trains in. A caller turning these into
    probabilities appends a zero and softmaxes over all `numClasses`;
    `multiclass_probabilities` does exactly that.

    RAW APPROXES, like every other predict here and like their `predict`
    without a `prediction_type`.
    """
    var approx_dim = model_approx_dim(tm.model)
    var expanded_x: List[Float32]
    if len(tm.ctr_tables) != 0:
        expanded_x = expand_raw_columns(
            tm.ctr_tables, len(tm.fold_counts), x_colmajor, n_rows
        )
    else:
        expanded_x = x_colmajor.copy()
    var cindex = _build_cindex_from_floats(
        ctx, expanded_x, n_rows, tm.borders, tm.fold_counts
    )
    var cursor = ctx.enqueue_create_buffer[DType.float32](
        approx_dim * n_rows
    )
    predict(
        tm.model, ctx, n_rows, tm.fold_counts, cindex, cursor,
        one_hot=tm.one_hot,
    )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](
        approx_dim * n_rows
    )
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=cursor)
    ctx.synchronize()
    var out = List[Float32]()
    for r in range(n_rows):
        for d in range(approx_dim):
            out.append(hc.unsafe_ptr().unsafe_load(d * n_rows + r))
    return out^


def multiclass_probabilities(
    approxes: List[Float32], n_rows: Int, num_classes: Int
) raises -> List[Float32]:
    """The softmax their `prediction_type='Probability'` applies.

    `approxes` is `predict_multi_floats`' output, `numClasses - 1` wide;
    the output is `numClasses` wide as `[row * numClasses + class]`. The
    LAST class is the pinned one, whose approx is zero by construction --
    which is why this function exists rather than the caller reshaping.

    The max subtraction is `multilogit.cu:41-53`'s, seeded at ZERO because
    zero IS the pinned class's approx, so it is a max over all
    `numClasses` and not only the stored ones.
    """
    var eff = num_classes - 1
    if len(approxes) != n_rows * eff:
        raise Error(
            "multiclass_probabilities: got " + String(len(approxes))
            + " approxes for " + String(n_rows) + " rows x "
            + String(eff) + " free classes"
        )
    var out = List[Float32]()
    for r in range(n_rows):
        var mx = Float64(0.0)
        for k in range(eff):
            var v = Float64(approxes[r * eff + k])
            if v > mx:
                mx = v
        var se = Float64(0.0)
        for k in range(eff):
            se += exp(Float64(approxes[r * eff + k]) - mx)
        se += exp(-mx)
        for k in range(eff):
            out.append(
                Float32(exp(Float64(approxes[r * eff + k]) - mx) / se)
            )
        out.append(Float32(exp(-mx) / se))
    return out^
