# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace
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
    dense_category_code,
    expand_raw_columns,
)
from std.math import log2

# DEVIATION 258: the probability links (double, as CatBoost computes them)
# go through the host-portable exp64 under IDENTICAL; FAST is the stdlib
from checks.numerics import identical_exp64
from gbdt.methods.doc_parallel_boosting import (
    TAdditiveModel,
    fit_with_test,
    make_test_arm,
    model_approx_dim,
    predict,
)
from gbdt.overfitting_detector.overfitting_detector import (
    OD_NONE,
    od_type_from_name,
)
from gbdt.gpu_util.kernel.radix_sort import DeviceFloatSorter
from std.memory import memcpy
from max.algorithm import sync_parallelize
from gbdt.data.permutation import TRandom
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
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
    OBJECTIVE_RMSE,
)
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_IS,
    calc_quantization,
    nan_substitution,
    nan_value_treatment,
)
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    nan_mode_from_name,
)
from gbdt.options.loss_description import make_loss_description
from gbdt.options.catboost_options import (
    COUNTER_CALC_FULL,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    SCORE_FUNCTION_COSINE,
    TCatFeatureParams,
    grow_policy_from_name,
    grow_policy_name,
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
    #: their `TFloatFeature::NanValueTreatment`, one per column
    #: (`libs/model/features.h:51-61`). `AsIs` for a column the learn pool
    #: had no NaN in, and a NaN arriving on such a column at apply time is
    #: their `CB_ENSURE` rather than a guess. It travels WITH the borders
    #: because it is part of the grid: the sentinel border is meaningless
    #: without the substitution that reaches it.
    var nan_treatment: List[Int]
    var losses: List[Float64]
    #: the HELD-OUT loss per iteration, empty when no eval set was given.
    #: Their `TestCursor`'s error curve, which is what the overfitting
    #: detector reads and what a caller should plot -- the learn curve
    #: falls almost by construction and says nothing about stopping.
    var test_losses: List[Float64]
    #: the index of the lowest TEST loss, or of the lowest learn loss with
    #: no eval set. Their `TLearnProgress` best-iteration bookkeeping.
    #: **-1 means NOT RECORDED**, which is what a model loaded from text
    #: carries: the text holds no held-out curve and
    #: `load_model_text` does not invent one.
    var best_iteration: Int
    #: True when the detector fired before `n_estimators` was reached
    var stopped_early: Bool
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
    nan_treatment: List[Int] = List[Int](),
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

    **`nan_treatment` IS WHERE NaN IS HANDLED, AND THE ONLY PLACE.** One
    entry per column, their `TFloatFeature::ENanValueTreatment`. A NaN is
    replaced by `-inf` (`AsFalse`) or `+inf` (`AsTrue`) as it is staged for
    the quantize kernel, which is exactly what their evaluator does
    (`libs/model/cpu/quantization.h:385-408`), and then the ordinary
    `value > border` comparison puts it in the sentinel bin. Nothing
    downstream -- histogram, score, partition, apply -- knows NaN exists.
    An empty list means every column is `AsIs`, which is the no-NaN case
    and the contract every caller had before this argument existed.

    A column whose treatment is `AsIs` and which CONTAINS a NaN is an
    error, their `CB_ENSURE(allowNans, "There are NaNs in test dataset
    (feature number N) but there were no NaNs in learn dataset")`
    (`private/libs/quantization/utils.h:74-78`). It is raised here rather
    than at the caller because this is the function that can see the value.

    A 16-category one-hot feature was silently unlearnable as a result. No
    exception, no crash: a returned model that could not see the feature.
    The fit/predict consistency assertion in `train_api_check` was blind to
    it BY CONSTRUCTION, because `predict_floats` came through this same
    function and read through the same wrong layout, so the two agreed on
    the wrong answer. `checks/one_hot_cardinality_check.mojo` is the gate
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
        var treat = NAN_TREATMENT_AS_IS
        if len(nan_treatment) == n_features:
            treat = nan_treatment[f]
        var sub = nan_substitution(treat)
        for r in range(n_rows):
            var v = x_colmajor[f * n_rows + r]
            if v != v:
                if treat == NAN_TREATMENT_AS_IS:
                    raise Error(
                        "There are NaNs in feature number " + String(f)
                        + " but there were no NaNs in the learn dataset"
                    )
                v = sub
            hx.unsafe_ptr().unsafe_store(r, v)
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
        _ = bdev^  # past the drain (step-33 race class, device side)
        _ = xdev^  # past the drain (step-33 race class, device side)
        _ = hx^  # past the drain (step-33 race class)
        _ = hbo^  # past the drain (step-33 race class)
    return cindex^


def _build_cindex_from_columns(
    ctx: DeviceContext,
    columns: List[List[Float32]],
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
    nan_treatment: List[Int],
) raises -> DeviceBuffer[DType.uint32]:
    """`_build_cindex_from_floats` without the flat pack and without the
    per-feature drain. The flat buffer exists so PERMUTATION-DEPENDENT
    columns can be substituted per permutation; when there are none, it
    is a 200M-element copy of data this function can read in place. The
    per-feature `synchronize` becomes a RING of `_CINDEX_SLOTS` staging
    slots drained once per ring revolution: a slot's host buffer may be
    overwritten only after its enqueued upload has run, and one drain
    before reusing slot 0 covers the whole previous revolution. The
    two-slot ping-pong this replaces still paid one drain (~0.2 ms) per
    FEATURE, which was ~0.4 s of the 1.69 s cindex bill at 2000
    features. Same kernels, same borders, same writes: bit-identical
    output to the flat-path builder, which the train-mse gate holds.
    """
    var n_features = len(borders)
    if len(fold_counts) != n_features:
        raise Error("fold_counts/borders length mismatch")
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))

    comptime _CINDEX_SLOTS = 8
    var xdevs = List[DeviceBuffer[DType.float32]]()
    var hxs = List[HostBuffer[DType.float32]]()
    var hbos = List[HostBuffer[DType.float32]]()
    var bdevs = List[DeviceBuffer[DType.float32]]()
    for _ in range(_CINDEX_SLOTS):
        xdevs.append(ctx.enqueue_create_buffer[DType.float32](n_rows))
        hxs.append(ctx.enqueue_create_host_buffer[DType.float32](n_rows))
        hbos.append(ctx.enqueue_create_host_buffer[DType.float32](256))
        bdevs.append(ctx.enqueue_create_buffer[DType.float32](256))
    ctx.synchronize()
    comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
    var staged = 0
    for f in range(n_features):
        if len(borders[f]) == 0:
            continue
        ref cf = lay.features[f]
        var treat = NAN_TREATMENT_AS_IS
        if len(nan_treatment) == n_features:
            treat = nan_treatment[f]
        var sub = nan_substitution(treat)
        var slot = staged % _CINDEX_SLOTS
        if staged >= _CINDEX_SLOTS and slot == 0:
            # one drain per revolution frees every slot in the ring
            ctx.synchronize()
        var hx = hxs[slot].unsafe_ptr()
        var hbo = hbos[slot].unsafe_ptr()
        var src = columns[f].unsafe_ptr()
        if treat == NAN_TREATMENT_AS_IS:
            # AS_IS means the border build's full-column NaN scan (the
            # sampled draw's explicit scan, or the full path's
            # calc_quantization over every value) saw none in THIS SAME
            # buffer, so a NaN here is unreachable and the checked
            # element-wise copy is a straight memcpy
            memcpy(dest=hx, src=src, count=n_rows)
        else:
            for r in range(n_rows):
                var v = src.unsafe_load(r)
                if v != v:
                    v = sub
                hx.unsafe_store(r, v)
        hbo.unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=xdevs[slot], src_ptr=hx)
        ctx.enqueue_copy(dst_buf=bdevs[slot], src_ptr=hbo)
        ctx.enqueue_function[binarize_float_feature_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            xdevs[slot].unsafe_ptr(), Int32(n_rows),
            bdevs[slot].unsafe_ptr(), cindex.unsafe_ptr(),
            grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
            block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
        )
        staged += 1
    ctx.synchronize()
    return cindex^


def generate_seed_for_borders(from_seed: UInt64) -> UInt64:
    """Their `TRandom::GenerateSeed` (`libs/helpers/cpu_random.h:88-94`):
    five LCG steps. Module level so `sample_indices_for_borders` and the
    check that gates it use THE SAME derivation, not two copies."""
    var sd = from_seed
    for _ in range(5):
        sd = 6364136223846793005 * sd + 1442695040888963407
    return sd


def sample_indices_for_borders(
    nrr: Int, sn: Int, sd0: UInt64
) -> List[UInt32]:
    """`SampleIndices<ui32>(n, k, rand)` -- `libs/helpers/sample.h:20-43`,
    both branches, their predicate `k > 1 && k > (n / log2(k))`.

    ONE subset for the whole dataset, drawn WITHOUT REPETITION. It is the
    sampling half of `GetSubsetForBuildBorders`
    (`libs/data/quantization.cpp:118-141`), whose subset every float
    column then gathers through.

    DEVIATION 135 covers what still differs: their engine is
    `TRestorableFastRng64` and ours is `TRandom`, so the SET drawn is not
    theirs at the same seed -- only the SEMANTICS (size, no repetition,
    shared across features) match. The rejection branch returns hash
    order upstream and insertion order here, which is order-equivalent
    because the sample is sorted before borders are built.

    MODULE LEVEL ON PURPOSE. It used to be inline in `train()`, which
    meant the only way to gate it was to re-type it into the check --
    and a check that builds its own copy of the thing it checks cannot
    catch the copy drifting. `checks/sample_indices_check.mojo`
    imports THIS function.
    """
    var sample_idx = List[UInt32]()
    var rnd0 = TRandom(generate_seed_for_borders(sd0))
    # their `CB_ENSURE_INTERNAL(n >= k)` (`sample.h:21`) has no
    # counterpart here, so this branch is `>=` where theirs is `n == k`:
    # the caller's invariant is `sn <= nrr`, and if it were ever broken
    # the Fisher-Yates branch would run `nrr - i` negative. Same
    # behavior under the invariant, a guard instead of a crash outside
    # it.
    if sn >= nrr:
        for i in range(nrr):
            sample_idx.append(UInt32(i))
    elif sn > 1 and Float64(sn) > Float64(nrr) / log2(Float64(sn)):
        # their partial Fisher-Yates over an iota: `std::swap(result[i],
        # result[rand->Uniform(i, n)])` for i in [0, k)
        sample_idx.resize(nrr, UInt32(0))
        for i in range(nrr):
            sample_idx[i] = UInt32(i)
        for i in range(sn):
            var j = i + Int(rnd0.next_uniform_l() % UInt64(nrr - i))
            var t = sample_idx[i]
            sample_idx[i] = sample_idx[j]
            sample_idx[j] = t
        sample_idx.resize(sn, UInt32(0))
    else:
        # their rejection loop into a set, `while (sampleSet.size() < k)
        # sampleSet.insert(rand->Uniform(n))`. A membership byte array
        # stands in for THashSet: same accept/reject sequence.
        var seen = List[Bool]()
        seen.resize(nrr, False)
        while len(sample_idx) < sn:
            var c = Int(rnd0.next_uniform_l() % UInt64(nrr))
            if not seen[c]:
                seen[c] = True
                sample_idx.append(UInt32(c))
        _ = seen^
    return sample_idx^


def train(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_count: Int = 128,
    border_build_max_samples: Int = 200_000,
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
    ctr_estimation_permutation_id: Int = -1,
    permutation_count: Int = -1,
    nan_mode: String = String("Min"),
    loss: String = "RMSE",
    loss_alpha: Float32 = Float32(-1.0),
    loss_q: Float32 = Float32(-1.0),
    loss_delta: Float32 = Float32(-1.0),
    loss_variance_power: Float32 = Float32(-1.0),
    loss_border: Float32 = Float32(-1.0),
    leaf_estimation_iterations: Int = -1,
    leaf_estimation_method: Int = -1,
    class_weights: List[Float32] = List[Float32](),
    sample_weight: List[Float32] = List[Float32](),
    eval_x_colmajor: List[Float32] = List[Float32](),
    eval_y: List[Float32] = List[Float32](),
    od_type: String = String("None"),
    od_pvalue: Float64 = 0.0,
    od_wait: Int = 20,
    use_best_model: Int = -1,
    best_model_min_trees: Int = 1,
    # `random_strength` (`oblivious_tree_options.cpp:17`). CatBoost's
    # default is 1.0 and this one is 0.0; see
    # `CatBoostOptions.random_strength`, and note that on the GREEDY
    # searcher this train() runs, CatBoost's own noise cancels in the gain
    # (`compute_scores.cu:84-134`) so a non-zero value here changes only
    # float rounding. It is a real knob on the doc-parallel searcher,
    # which `train` does not currently select.
    random_strength: Float32 = Float32(0.0),
    # `TDocParallelObliviousTreeSearcher`, CatBoost's SINGLE-TARGET
    # symmetric learner, in place of the greedy subsets searcher. Both are
    # theirs and both are reachable on their GPU; `fit_with_test` has taken
    # this since the pointwise family shipped and `train` did not pass it,
    # which made the whole arm unreachable from every caller that goes
    # through this function -- the CPython binding included. Forwarded at
    # the single `fit_with_test` call site below.
    #
    # The two arms differ in what they RETURN, not only in how they search:
    # the pointwise searcher returns the STRUCTURE ONLY (DEVIATION 104), so
    # it always runs the leaf estimator, where the greedy arm can reuse the
    # leaf it already grew. That is why it cannot take DEVIATION 64's
    # estimation shortcut.
    use_pointwise_searcher: Bool = False,
    # `boost_from_average`, tri-state exactly because THEIRS is: -1 is
    # "not set", resolved by the port of
    # `options_helper.cpp::AdjustBoostFromAverageDefaultValue` below --
    # auto-TRUE for the losses on their list whose CalcOptimumConstApprox
    # arm is ported (RMSE; their list also holds MAE/Quantile/MAPE, whose
    # constant needs the unported CalcSampleQuantile, so those resolve to
    # FALSE with the gap named in `gbdt/metrics/optimal_const_for_loss`).
    # NOTE their list does NOT hold Logloss: an unset option on a Logloss
    # fit is FALSE on their side too, which the higgs 2026-08-22 read
    # got wrong before this port. 0 and 1 are explicit; 1 raises by name
    # for losses without a ported constant.
    boost_from_average: Int = -1,
    # ============================ DEVIATION 259 ============================
    # `grow_policy` / `max_leaves` / `min_data_in_leaf`, their
    # `EGrowPolicy` spellings (`oblivious_tree_options.cpp:23-25`). The
    # three policies CatBoost's GPU learner grows: SymmetricTree (this
    # function's only arm until 2026-08-23), Depthwise and Lossguide, the
    # two that build `TNonSymmetricTree` through
    # `greedy_search_helper_depthwise.fit_non_symmetric_tree`. -1 for
    # `max_leaves` is their `IsDefault()`: `1 << depth` for every policy
    # but Lossguide (`catboost_options.cpp:993-1001`), 31 under Lossguide
    # (`oblivious_tree_options.cpp:24`). `min_data_in_leaf` is live under
    # the non-symmetric policies ONLY (`greedy_search_helper.cpp:685`) and
    # is REFUSED here at any value but 1 under SymmetricTree, where CatBoost
    # accepts and discards it -- their docs say the option "can be used only
    # with the Lossguide and Depthwise growing policies", and this port
    # refuses what it would otherwise silently drop.
    # =======================================================================
    grow_policy: String = String("SymmetricTree"),
    max_leaves: Int = -1,
    min_data_in_leaf: Int = 1,
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
    and it is NOT the identity.** Their permutation 0 IS the
    identity (`permutation.cpp:14-17`) and is safe on their side only
    because the learn pool was already shuffled at load
    (`private/libs/algo/preprocess.cpp:183-199`, which shuffles whenever the
    data has a categorical feature and `has_time` is false). This port has
    no such stage, so a caller can hand us rows sorted by target, where the
    identity order makes every row's ordered statistic read its own
    neighbourhood. `permutation_count - 1` is their ESTIMATION permutation,
    `GetEstimationPermutation()` (`doc_parallel_boosting.h:101-103`), the
    one whose model `Run()` exports.

    **ALL `permutation_count` COLUMN SETS ARE BUILT** as of 2026-08-21, one
    compressed index each (DEVIATION 89), where this used to build only the
    estimation permutation's -- the sentence archive/reference/PORTING.md 55 recorded, now
    false. `permutation_count` resolves the way `UpdateGpuSpecificDefaults`
    resolves it (`cuda/train_lib/train.cpp:99-108`): their default of 4,
    ASSIGNED down to 1 when no categorical feature feeds a CTR -- an
    assignment, so an explicit 4 is discarded too. The
    permutation-dependent columns are the only thing that varies between
    the sets, and they all take PERMUTATION 0'S BORDERS, because their
    border builder caches by feature id and permutation 0 is the one that
    fills the cache (`gpu_binarization_helpers.cpp:31-54`,
    `doc_parallel_dataset_builder.cpp:250`).

    **THE BOOSTING LOOP RUNS ALL OF THEM.** It searches the structure on a
    random non-estimation permutation, estimates and applies that structure
    against every permutation's compressed index and cursor, and exports
    the estimation permutation's weak model (`doc_parallel_boosting.h:
    345-398`). This is wired through `perm_cindexes` and
    `est_permutation` in `fit_with_test`; the one-permutation numeric path
    retains its original buffer and execution shape.

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
    exercised: `checks/ctr_apply_check.mojo` and
    `checks/ctr_train_check.mojo` each run the default AND
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

    ## The held-out set, the detector, and `use_best_model`

    `eval_x_colmajor` / `eval_y` are their test pool. It is quantized here,
    against the borders THIS fit built, and scored every iteration by
    `TestArm`; `od_type` / `od_pvalue` / `od_wait` drive the detector
    (`gbdt/overfitting_detector/overfitting_detector.mojo`).

    **`use_best_model` TRUNCATES THE RETURNED ENSEMBLE** to the trees up to
    and including the best held-out iteration, their `ShrinkToBestIteration`
    (`boosting_progress_tracker.h:113-125`) called from
    `train_template.h:127-137`. It is a TRI-STATE, because theirs is
    `TOption::NotSet()` and their default is data-dependent
    (`options_helper.cpp:100-113`):

        -1  unset -> TRUE when there is an eval set whose target is not
            constant, FALSE otherwise
         0  off
         1  on -> and with no eval set this raises, where they warn and
            switch it off

    `best_model_min_trees` is their `best_model_min_trees`, default 1
    (`output_file_options.cpp:77`): the shrink may not cut BELOW this many
    trees, because their second tracker only ever sees iterations at or
    past it (`boosting_progress_tracker.cpp:162`). At the default it is
    inert, which is why it is easy to get wrong and why it is ported now
    rather than later.

    `best_iteration` in the result is the ERROR tracker's, not the
    min-trees tracker's -- the same distinction their shrink log draws at
    `boosting_progress_tracker.h:119-122`. So a caller who sets
    `best_model_min_trees` above the best iteration gets a model with MORE
    trees than `best_iteration + 1`, and both numbers are correct.
    """
    # ---- the grow policy, resolved and refused BY NAME where theirs is ----
    var policy = grow_policy_from_name(grow_policy)
    if policy == GROW_SYMMETRIC and min_data_in_leaf != 1:
        raise Error(
            "min_data_in_leaf=" + String(min_data_in_leaf) + " does nothing"
            " under grow_policy=SymmetricTree: CatBoost guards its leaf-size"
            " test with `Policy != SymmetricTree`"
            " (greedy_search_helper.cpp:685) and discards the value; refused"
            " here rather than accepted and ignored. It is live under"
            " Depthwise and Lossguide."
        )
    if policy != GROW_LOSSGUIDE and max_leaves >= 0 and max_leaves != (
        1 << max_depth
    ):
        raise Error(
            "max_leaves option works only with lossguide tree growing"
            " (catboost_options.cpp:998): under grow_policy="
            + grow_policy_name(policy) + " CatBoost pins it to 1 << depth == "
            + String(1 << max_depth) + ", got " + String(max_leaves)
        )
    if policy == GROW_LOSSGUIDE and max_leaves >= 0 and max_leaves > 65536:
        # `CB_ENSURE(MaxLeaves <= 1 << 16, "Maximum leaves count for
        # Lossguide grow policy is 65536")` (`oblivious_tree_options.cpp:
        # 130-133`)
        raise Error(
            "Maximum leaves count for Lossguide grow policy is 65536; got "
            + String(max_leaves)
        )
    if policy != GROW_SYMMETRIC and use_pointwise_searcher:
        raise Error(
            "use_pointwise_searcher is TDocParallelObliviousTreeSearcher, an"
            " OBLIVIOUS searcher; grow_policy=" + grow_policy_name(policy)
            + " is grown by TGreedySubsetsSearcher<TNonSymmetricTree> only"
            " (pointwise_non_symmetric.cpp:5-29)"
        )

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

    #: the PERMUTATION-DEPENDENT columns, one set per permutation, and
    #: where each one sits in `columns`. Their
    #: `dataSet.PermutationDependentFeatures` is a separate compressed-index
    #: dataset per permutation while the float, one-hot and FeatureFreq
    #: columns are shared (`doc_parallel_dataset_builder.cpp:104-124`), so
    #: this is the only thing that varies. `columns` itself holds the
    #: ESTIMATION permutation's values, which is the set the exported model
    #: is trained on.
    var dep_col_index = List[Int]()
    var dep_by_perm = List[List[List[Float32]]]()

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
    # ---- `UpdateGpuSpecificDefaults` (`cuda/train_lib/train.cpp:99-108`)
    #
    #     if (!HasPermutationFeatures(featuresManager) &&
    #         options.BoostingOptions->BoostingType == EBoostingType::Plain) {
    #         options.BoostingOptions->PermutationCount = 1;
    #     }
    #
    # **That is an ASSIGNMENT, not a `SetDefault`**: with no CTR-bearing
    # categorical feature it overrides an explicit `permutation_count`
    # too, because four identical permutations of a dataset with no
    # permutation-dependent column are four identical datasets. This port
    # is Plain (archive/reference/PORTING.md 88), so the second half of their condition
    # holds unconditionally here.
    #
    # `HasPermutationFeatures` (`:86-98`) is "some cat feature is used for
    # a CTR". A feature is used for a CTR exactly when its cardinality is
    # above `one_hot_max_size` (`binarizations_manager.cpp:106-115`), so
    # this pre-pass reads cardinalities and nothing else -- their features
    # manager makes the same decision before their options are resolved,
    # in the same order.
    var has_permutation_features = False
    if len(dependent_configs) > 0:
        for f in range(n_features):
            if not (len(cat_features) == n_features and cat_features[f]):
                continue
            var maxc = 0
            for r in range(n_rows):
                var c = dense_category_code(
                    x_colmajor[f * n_rows + r], f, r
                )
                if c > maxc:
                    maxc = c
            if maxc + 1 > cat_params.one_hot_max_size:
                has_permutation_features = True
                break

    var perm_count = permutation_count
    if perm_count == -1:
        perm_count = DEFAULT_PERMUTATION_COUNT
    if perm_count < 1:
        # `CB_ENSURE(PermutationCount.Get() > 0)` (`boosting_options.cpp:67`)
        raise Error(
            "Permutation count should be positive, got "
            + String(perm_count)
        )
    if not has_permutation_features:
        perm_count = 1

    # their estimation permutation, `GetEstimationPermutation()`
    # (`doc_parallel_boosting.h:101-103`). It is a caller argument only so
    # that `ctr_device_check` can pin permutation 0; -1 means "theirs".
    var est_perm = ctr_estimation_permutation_id
    if est_perm == -1:
        est_perm = perm_count - 1
    if est_perm < 0 or est_perm >= perm_count:
        raise Error(
            "ctr_estimation_permutation_id " + String(est_perm)
            + " is outside the " + String(perm_count)
            + " permutations this fit builds"
        )

    for _ in range(perm_count):
        dep_by_perm.append(List[List[Float32]]())

    var binarized_target = List[UInt8]()
    var ctr_orders = List[List[UInt32]]()
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
        # ONE ORDER PER PERMUTATION, their loop's
        # `ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)`
        # (`doc_parallel_dataset_builder.cpp:255`) with the permutation
        # from `GetPermutation(DataProvider, permutationId)` (`:48`).
        for p in range(perm_count):
            ctr_orders.append(
                ctrs_estimation_permutation(n_rows, p).fill_order()
            )

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
        # one flat memcpy per column; the append loop this replaces was
        # ~0.5 s of every train() at 400k x 500 (200M bounds-checked
        # appends)
        var col = List[Float32]()
        col.resize(n_rows, Float32(0.0))
        memcpy(
            dest=col.unsafe_ptr(),
            src=x_colmajor.unsafe_ptr() + f * n_rows,
            count=n_rows,
        )

        if not is_cat:
            columns.append(col^)
            column_one_hot.append(flagged_one_hot)
            column_ctr_grid.append(-1)
            continue

        # dense codes: cardinality is max + 1
        var maxc = 0
        var seen = List[Bool]()
        seen.resize(1, False)
        var codes = List[UInt32]()
        for r in range(n_rows):
            var c = dense_category_code(col[r], f, r)
            if c > maxc:
                maxc = c
                seen.resize(maxc + 1, False)
            seen[c] = True
            codes.append(UInt32(c))
        var unique_values = maxc + 1
        if unique_values <= 1:
            # their `CB_ENSURE(uniqueValues > 1)`
            # (`batch_binarized_ctr_calcer.cpp:150`)
            raise Error(
                "Error: useless catFeature found (feature "
                + String(f)
                + " has one category)"
            )
        for c in range(unique_values):
            if not seen[c]:
                raise Error(
                    "cat_features column " + String(f)
                    + " is not densely coded: category " + String(c)
                    + " is absent from 0.." + String(maxc)
                )

        if unique_values <= cat_params.one_hot_max_size:
            # `UseForOneHotEncoding` (`binarizations_manager.cpp:106-109`):
            # one-hot features never get CTRs.
            columns.append(col^)
            column_one_hot.append(True)
            column_ctr_grid.append(-1)
            continue

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
            # their `writeCtrs(..., permutationDependent)` (`:257`), ONCE
            # PER PERMUTATION, over that permutation's order written at
            # `:255`. `columns` takes the estimation permutation's values;
            # the rest are kept beside it for the boosting loop, which
            # estimates leaf values separately on each.
            for p in range(perm_count):
                var dep = compute_simple_ctrs_gpu(
                    ctx,
                    codes,
                    unique_values,
                    dependent_configs,
                    binarized_target,
                    ctr_orders[p],
                )
                for c in range(len(dependent_slots)):
                    dep_by_perm[p].append(dep[c].copy())
                    if p == est_perm:
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
        var base_col = len(columns)
        for c in range(len(dependent_slots)):
            dep_col_index.append(base_col + dependent_slots[c])
        for c in range(len(ctr_columns)):
            columns.append(ctr_columns[c].copy())
            column_one_hot.append(False)
            ctr_grids.append(cat_params.ctr_binarization_for(configs[c]))
            column_ctr_grid.append(len(ctr_grids) - 1)
            ctr_column_count += 1

    var n_columns = len(columns)

    # column index -> its ordinal among the permutation-dependent columns,
    # or -1. Built here rather than carried, because the column positions
    # are only final once every feature has been walked.
    var dep_ordinal_of_column = List[Int]()
    for _ in range(n_columns):
        dep_ordinal_of_column.append(-1)
    for k in range(len(dep_col_index)):
        dep_ordinal_of_column[dep_col_index[k]] = k

    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    # their ComputeBorders' device RadixSort, scratch hoisted once for
    # every float column below (see DeviceFloatSorter's docstring for the
    # churn crash that makes the hoist load-bearing)
    # THEIR CPU QUANTIZER'S SUBSAMPLE, adopted for the user-facing path.
    #
    # CORRECTED 2026-08-22, DEVIATION 135. The three sentences that stood
    # here cited `GetSampleSizeForBorderSelectionType`
    # (`private/libs/quantization/utils.h:132-136`) and `SampleArray`
    # (`utils.cpp:14-24`) -- a REAL pair of functions ON THE WRONG CODE
    # PATH. `SampleArray` is reached from `NCB::BuildBorders`, which in
    # this tree is called only by a unit test and by the GPU CTR border
    # builder. The TRAINING pipeline goes
    # `GetSubsetForBuildBorders` (`libs/data/quantization.cpp:118-141`)
    # -> `GetArraySubsetForBuildBorders` (`utils.cpp:25-51`), and it
    # differs from that helper in all three of the ways that matter:
    #
    #   size          `TQuantizationOptions::MaxSubsetSizeForBuild
    #                 BordersAlgorithms = 200000` (`libs/data/
    #                 quantization.h:37`), not the helper's 100000
    #                 DEFAULT ARGUMENT, which the pipeline overrides.
    #   replacement   `SampleIndices` (`libs/helpers/sample.h:20-43`),
    #                 "Sample k element indices without repetition".
    #                 `SampleArray` draws WITH replacement.
    #   sharing       ONE subset for the whole dataset, built once at
    #                 `quantization.cpp:127` and reused by every float
    #                 column. Ours drew a fresh sample per feature.
    #
    # The old draw was therefore a different estimator of the border set
    # than CatBoost's on any pool above the cap: 100k with replacement
    # covers ~63.2% of distinct rows, 200k without replacement covers
    # exactly 200k. `border_build_max_samples = 0` still restores the
    # full-data GPU-pipeline behavior (`ComputeBorders`,
    # `gpu_binarization_helpers.cpp:10-16`, which full-sorts).
    #
    # NOT REACHED BY `bench/interleaved`, which quantizes with CATBOOST'S
    # OWN quantizer and hands both arms the same pre-binned uint8
    # (`tools/interleaved_prep.py:1-10`). This is the user-facing
    # `train()`/estimator path and the end-to-end arm, not the
    # standing benchmark numbers.
    var border_sample_n = n_rows
    if border_build_max_samples > 0 and border_build_max_samples < n_rows:
        border_sample_n = border_build_max_samples

    # phase-A/B scratch: which columns are float, and one flat buffer
    # holding every sorted float column back to back
    var n_float_prescan = 0
    var float_idx = List[Int]()
    for f in range(n_columns):
        if not column_one_hot[f] and column_ctr_grid[f] < 0:
            float_idx.append(f)
            n_float_prescan += 1
    var float_cols = List[Int]()
    # only the full-data path stages sorted columns here; the sampled path's
    # phase B reads `predrawn` directly
    var sorted_flat = List[Float32]()
    if border_sample_n == n_rows:
        sorted_flat.resize(n_float_prescan * border_sample_n, Float32(0.0))

    # PHASE 0, sampling only. ONE index subset for the whole dataset,
    # drawn WITHOUT REPLACEMENT, then every float column gathers through
    # it -- their `GetSubsetForBuildBorders`
    # (`libs/data/quantization.cpp:118-141`), which builds
    # `subsetIndexing` ONCE and hands the same one to every feature. The
    # per-column GATHER stays parallel (it is `n_float * 200k` loads);
    # only the DRAW is serial, because it is one draw now and not one
    # per feature. Per-slot error flags re-raised after the join.
    var predrawn = List[Float32]()
    if border_sample_n < n_rows and n_float_prescan > 0:
        predrawn.resize(n_float_prescan * border_sample_n, Float32(0.0))
        var pd = predrawn.unsafe_ptr()
        var fi = float_idx.unsafe_ptr()
        var colp = columns.unsafe_ptr()
        var sn = border_sample_n
        var nrr = n_rows
        var sd0 = random_seed
        var flags = List[Int]()
        flags.resize(n_float_prescan, 0)
        var flg = flags.unsafe_ptr()

        var sample_idx = sample_indices_for_borders(nrr, sn, sd0)
        var sidx = sample_idx.unsafe_ptr()

        def _draw_task(
            k: Int
        ) {imm pd, imm fi, imm colp, imm sn, imm nrr, imm sidx, imm flg}:
            var fcol = fi.unsafe_load(k)
            var src = (colp + fcol)[].unsafe_ptr()
            for i in range(sn):
                pd.unsafe_store(
                    k * sn + i,
                    src.unsafe_load(Int(sidx.unsafe_load(i))),
                )
            var has_nan = False
            for r in range(nrr):
                var v = src.unsafe_load(r)
                if v != v:
                    has_nan = True
                    break
            if has_nan:
                var sample_has = False
                for i in range(sn):
                    var v2 = pd.unsafe_load(k * sn + i)
                    if v2 != v2:
                        sample_has = True
                        break
                if not sample_has:
                    pd.unsafe_store(k * sn, Float32(0.0) / Float32(0.0))
                flg.unsafe_store(k, 1)

        sync_parallelize(_draw_task, n_float_prescan)
        _ = flags^
        _ = sample_idx^
    var border_sorter = DeviceFloatSorter(
        ctx, n_rows if border_sample_n == n_rows else 1
    )
    # their `TFloatFeature::NanValueTreatment`, one per COLUMN. One-hot and
    # CTR columns stay `AsIs`: a one-hot column holds dense codes and a CTR
    # column holds a computed statistic, and a NaN in either is a caller
    # error rather than a value to bin.
    var nan_mode_opt = nan_mode_from_name(nan_mode)
    var column_nan_treatment = List[Int]()
    for _ in range(n_columns):
        column_nan_treatment.append(NAN_TREATMENT_AS_IS)
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
        elif column_ctr_grid[f] >= 0 and dep_ordinal_of_column[f] >= 0:
            # A PERMUTATION-DEPENDENT CTR COLUMN TAKES PERMUTATION 0'S
            # BORDERS, and every other permutation is binarized against
            # them. That is `TGpuBordersBuilder::GetOrComputeBorders`
            # (`gpu_binarization_helpers.cpp:31-54`) caching by FEATURE ID
            # in the features manager, hit by whichever permutation was
            # written first -- and their loop starts at 0
            # (`doc_parallel_dataset_builder.cpp:250`). The grid is a
            # property of the feature, not of the permutation.
            var bs = compute_ctr_borders(
                dep_by_perm[0][dep_ordinal_of_column[f]],
                ctr_grids[column_ctr_grid[f]],
            )
            fold_counts.append(len(bs))
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
            # `CalcQuantization` (`libs/data/quantization.cpp:300-346`):
            # the column decides its own NaN mode, and a column that has
            # NaNs spends ONE of `border_count` on the sentinel.
            #
            # SORTED ON THE DEVICE FIRST, which is their own GPU
            # pipeline's design (`ComputeBorders`,
            # `gpu_binarization_helpers.cpp:10-16`: RadixSort on the
            # device, grid builder on the sorted readback). Measured
            # 2026-08-21 (PREP_BILL results): the host sort inside
            # `best_split` was 34 of 45.9 ms per 400k column, 74% of the
            # whole 24-second preparation bill at 400k x 500; on
            # presorted input the same call is 11.8 ms. The sorted
            # multiset is identical, so every border -- and the fit's
            # mse -- is bit-for-bit unchanged, which the quantize-cost
            # probe's recorded mse gates.
            # PHASE A OF THE SPLIT BORDER BUILD: the device pipeline
            # sorts this column (the previous float column's prefetch
            # already enqueued it; the next one is enqueued before the
            # host copy below, so the device stays busy), and the DP is
            # DEFERRED to phase B, which runs every float column's grid
            # builder in parallel on the host -- their per-feature
            # executor design (`calcBordersAndNanMode` on
            # `NPar::LocalExecutor`), with the device never touched
            # inside the parallel region (the sync_parallelize deadlock
            # rule mojotrees recorded).
            if border_sample_n == n_rows:
                # full-data path only: their GPU ComputeBorders device
                # RadixSort. The SAMPLED path skips the device entirely --
                # the subsample is their CPU quantizer's design, and their
                # CPU DP (`calcBordersAndNanMode`) sorts on the host inside
                # `BestSplit`; at 100k samples the device sort is
                # launch-floor-bound (measured 1.8 s for 500 columns) while
                # the host sort rides inside phase B's parallel region.
                if not border_sorter.has_pending:
                    border_sorter.begin(ctx, columns[f].copy())
                var col = border_sorter.finish(ctx)
                var g = f + 1
                while g < n_columns:
                    if not column_one_hot[g] and column_ctr_grid[g] < 0:
                        border_sorter.begin(ctx, columns[g].copy())
                        break
                    g += 1
                var slot = len(float_cols)
                var sfp = sorted_flat.unsafe_ptr()
                var cp2 = col.unsafe_ptr()
                for r in range(border_sample_n):
                    sfp.unsafe_store(
                        slot * border_sample_n + r, cp2.unsafe_load(r)
                    )
                _ = col^
            float_cols.append(f)
            fold_counts.append(0)  # placeholder, phase B fills it
            borders.append(List[Float32]())  # placeholder

    # one-hot features occupy folds+? -- for ordered features CatBoost's
    # fold count IS the border count; a one-hot feature has k categories
    # = k bins reached by k-1 synthetic borders, and its fold count must
    # cover bin k-1 for the equality candidates, hence len+1 above.

    # PHASE B: every float column's calc_quantization in parallel, disjoint
    # flat output slots, errors carried out through a per-slot flag and
    # re-raised after the join.
    var n_float = len(float_cols)
    if n_float > 0:
        var out_cap = border_count + 1
        var out_borders = List[Float32](capacity=n_float * out_cap)
        out_borders.resize(n_float * out_cap, Float32(0.0))
        var out_counts = List[Int](capacity=n_float)
        out_counts.resize(n_float, 0)
        var out_modes = List[Int](capacity=n_float)
        out_modes.resize(n_float, -1)
        # full path: device-sorted columns; sampled path: the raw parallel
        # draws (calc_quantization's BestSplit filters NaNs and sorts on the
        # host itself -- same multiset, bit-identical borders)
        var sfp2 = sorted_flat.unsafe_ptr()
        if border_sample_n < n_rows:
            sfp2 = rebind[type_of(sfp2)](predrawn.unsafe_ptr())
        var obp = out_borders.unsafe_ptr()
        var ocp = out_counts.unsafe_ptr()
        var omp = out_modes.unsafe_ptr()
        var nr2 = border_sample_n
        var bc2 = border_count
        var nm2 = nan_mode_opt
        var cap2 = out_cap

        def _dp_task(
            k: Int
        ) {imm sfp2, imm obp, imm ocp, imm omp, imm nr2, imm bc2, imm nm2, imm cap2}:
            try:
                var col2 = List[Float32]()
                col2.resize(nr2, Float32(0.0))
                memcpy(dest=col2.unsafe_ptr(), src=sfp2 + k * nr2, count=nr2)
                var q2 = calc_quantization(col2^, bc2, nm2)
                var nb = len(q2[0])
                if nb > cap2:
                    ocp.unsafe_store(k, -2)
                    return
                for j in range(nb):
                    obp.unsafe_store(k * cap2 + j, q2[0][j])
                ocp.unsafe_store(k, nb)
                omp.unsafe_store(k, q2[1])
            except:
                ocp.unsafe_store(k, -1)

        sync_parallelize(_dp_task, n_float)
        # ==================== THE STEP-33 RACE, FOUND =====================
        # The plane the tasks read must outlive the JOIN, not its last
        # textual use: `sfp2 = predrawn.unsafe_ptr()` above was
        # `predrawn`'s last use, so Mojo freed the whole drawn-sample
        # plane BEFORE the pool ran -- and each task's own `col2`
        # allocation (border_sample_n floats) could land inside the freed
        # pages and OVERWRITE them under a sibling task's read. Thread
        # scheduling decides who reads garbage: nondeterministic on a
        # QUIET box, dependent on allocator size classes (which is why
        # 8.8M-row fits diverged where 2M-row fits never did), and the
        # fork lands in the BORDERS, which is why divergent models differ
        # from tree 0 with coherent-but-wrong AUCs. Full record:
        # PREP_BILL_2026-08-22 steps 33-34.
        _ = sorted_flat^
        _ = predrawn^
        # ==================================================================

        for k in range(n_float):
            var nb = out_counts[k]
            if nb < 0:
                raise Error(
                    "parallel border build failed on float column "
                    + String(float_cols[k])
                )
            var f2 = float_cols[k]
            var bs2 = List[Float32](capacity=nb)
            for j in range(nb):
                bs2.append(out_borders[k * out_cap + j])
            column_nan_treatment[f2] = nan_value_treatment(out_modes[k])
            fold_counts[f2] = nb
            borders[f2] = bs2^

    # the fit's identity trace begins HERE so the border records and the
    # tree records share one seq space (a second IdentityTrace() later
    # would restart seq inside the same file). Disabled unless
    # MOJOLEARN_IDENTITY_TRACE is set.
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            "mojolearn train(): borders + " + grow_policy_name(policy)
            + " fit"
        )
        var border_counts = List[Int32]()
        var border_values = List[Float32]()
        for f in range(len(borders)):
            border_counts.append(Int32(len(borders[f])))
            for b in range(len(borders[f])):
                border_values.append(borders[f][b])
        trace.record_list_i32("borders.counts", border_counts)
        trace.record_list_f32("borders.values", border_values)

    # ONE COMPRESSED INDEX PER PERMUTATION. Theirs shares the
    # permutation-INDEPENDENT columns between them and gives each
    # permutation its own dataset for the dependent ones
    # (`doc_parallel_dataset_builder.cpp:104-124`); this port packs every
    # column into one buffer, so a permutation costs a whole index rather
    # than the dependent slice of one. DEVIATION 89.
    var cindexes = List[DeviceBuffer[DType.uint32]]()
    var any_dep = False
    for c in range(n_columns):
        if dep_ordinal_of_column[c] >= 0:
            any_dep = True
    for p in range(perm_count):
        if not any_dep:
            # no permutation-dependent columns: read `columns` in place,
            # skip the 200M-element flat pack and the per-feature drain
            # (see _build_cindex_from_columns)
            cindexes.append(
                _build_cindex_from_columns(
                    ctx, columns, n_rows, borders, fold_counts,
                    column_nan_treatment,
                )
            )
            continue
        var flat = List[Float32]()
        for c in range(n_columns):
            var ord = dep_ordinal_of_column[c]
            if ord >= 0:
                for r in range(n_rows):
                    flat.append(dep_by_perm[p][ord][r])
            else:
                for r in range(n_rows):
                    flat.append(columns[c][r])
        cindexes.append(
            _build_cindex_from_floats(
                ctx, flat, n_rows, borders, fold_counts,
                column_nan_treatment,
            )
        )
    var cindex = cindexes[est_perm].copy()

    # `class_weights` and `sample_weight`: their
    # `MakeClassificationWeights` applied at pool build. Both fold into
    # one weight column below; the checks that belong to `class_weights`
    # alone travel with it there, because the entry count now depends on
    # the loss.
    if len(class_weights) > 0 and loss == "RMSE":
        raise Error(
            "class weights take effect only with a classification loss,"
            " their option check's words (catboost_options.cpp:617)"
        )
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)

    # THEIR COMBINATION IS A PRODUCT (`private/libs/target/
    # data_providers.cpp:168`):
    #
    #     rawWeights[i] * rawGroupWeights[i] * classWeights[targetClass[i]]
    #
    # `rawGroupWeights` is the querywise family's and is 1 here -- this
    # port carries no `group_id`, so there is nothing to weight by. The
    # other two multiply, and a caller may pass either, both, or neither.
    var use_sample_weight = len(sample_weight) > 0
    if use_sample_weight and len(sample_weight) != n_rows:
        raise Error(
            "sample_weight has " + String(len(sample_weight))
            + " entries for " + String(n_rows) + " rows"
        )

    # `classWeights[(size_t)targetClassesArray[i]]` indexes by the TARGET
    # CLASS, so how many entries it needs depends on the loss: two for the
    # binarized classification targets, `numClasses` for MultiClass.
    var n_class_slots = 2
    if loss == "MultiClass" or loss == "MultiClassOneVsAll":
        var mxc = -1
        for r in range(n_rows):
            var iv = Int(y[r])
            if iv > mxc:
                mxc = iv
        n_class_slots = mxc + 1
    var use_class_weights = len(class_weights) > 0
    if use_class_weights and len(class_weights) != n_class_slots:
        raise Error(
            "class_weights takes " + String(n_class_slots)
            + " entries for loss '" + loss + "', got "
            + String(len(class_weights))
        )

    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, y[r])
        var w = Float32(1.0)
        if use_sample_weight:
            if sample_weight[r] < Float32(0.0):
                raise Error(
                    "sample_weight at row " + String(r)
                    + " is negative"
                )
            w = sample_weight[r]
        if use_class_weights:
            # their `targetClassesArray`: the dense class code for
            # MultiClass, the binarized target otherwise
            var cls: Int
            if loss == "MultiClass" or loss == "MultiClassOneVsAll":
                cls = Int(y[r])
            else:
                cls = 1 if y[r] > Float32(0.5) else 0
            w = w * class_weights[cls]
        hw.unsafe_ptr().unsafe_store(r, w)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    # past the drain (step-33 race class)
    _ = ht^
    _ = hw^

    var loss_desc = make_loss_description(
        loss,
        alpha=loss_alpha,
        q=loss_q,
        delta=loss_delta,
        variance_power=loss_variance_power,
        border=loss_border,
    )
    var objective = loss_desc.loss_function

    # ---- `AdjustBoostFromAverageDefaultValue` (`options_helper.cpp`),
    # ported 2026-08-22. Their rule, verbatim: if the option is SET,
    # keep it; else set TRUE on a single host with no baseline and no
    # continuation for RMSE, MAE, Quantile, MAPE (and three multi losses
    # this port does not have). Logloss is NOT on the list. This port has
    # no baseline column and no continuation, so those guards are
    # trivially met; MAE/Quantile/MAPE resolve FALSE here because their
    # constant needs the unported CalcSampleQuantile -- a named gap, not
    # their rule.
    var bfa: Bool
    if boost_from_average == 1:
        if not (
            objective == OBJECTIVE_RMSE
            or objective == OBJECTIVE_LOGLOSS
            or objective == OBJECTIVE_CROSSENTROPY
        ):
            # their CB_ENSURE names the allowed list; ours additionally
            # names the unported-constant gap for the quantile family
            raise Error(
                "boost_from_average: ported for RMSE, Logloss and"
                " CrossEntropy only. Their list also allows Quantile,"
                " MultiQuantile, MAE, MAPE, MultiRMSE (catboost_options"
                ".cpp:705-709); those need the unported CalcSampleQuantile"
                " and are refused by name."
            )
        bfa = True
    elif boost_from_average == 0:
        bfa = False
    elif boost_from_average == -1:
        bfa = objective == OBJECTIVE_RMSE
    else:
        raise Error(
            "boost_from_average must be -1 (their data-dependent"
            " default), 0 or 1; got " + String(boost_from_average)
        )

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
    if (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
    ):
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
                "the multiclass family needs at least two classes; the"
                " labels reach only " + String(num_classes)
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

    # ---- the HELD-OUT set, quantized against THIS MODEL'S BORDERS ----
    # That is the whole reason this lives in `train` and not in `fit`:
    # `borders` is built here, from the learn rows, and a `cindex` built
    # against any other borders would score every split against the wrong
    # bins with nothing to assert on it.
    var eval_rows = 0
    if len(eval_y) > 0:
        eval_rows = len(eval_y)
        var want = eval_rows * n_features
        if len(eval_x_colmajor) != want:
            raise Error(
                "eval_x_colmajor has " + String(len(eval_x_colmajor))
                + " values for " + String(eval_rows) + " rows x "
                + String(n_features) + " features"
            )
    elif len(eval_x_colmajor) > 0:
        raise Error("eval_x_colmajor given without eval_y")

    var od_kind = od_type_from_name(od_type)
    if od_kind != OD_NONE and eval_rows == 0:
        raise Error(
            "od_type='" + od_type + "' needs an eval set: pass"
            " eval_x_colmajor and eval_y. Stopping on the learn loss"
            " would stop on a curve that falls by construction."
        )

    # `UpdateUseBestModel` (`options_helper.cpp:100-113`). Their
    # `hasTestConstTarget` is the reason for the second half: a test set
    # whose target never varies cannot rank iterations, so they leave the
    # default off rather than shrink on a flat curve. `hasTestPairs` is
    # theirs and not ours -- this port carries no pairwise loss.
    var eval_const_target = True
    for r in range(1, eval_rows):
        if eval_y[r] != eval_y[0]:
            eval_const_target = False
            break
    var want_best_model = use_best_model
    if want_best_model == -1:
        want_best_model = (
            1 if (eval_rows > 0 and not eval_const_target) else 0
        )
    elif want_best_model == 1 and eval_rows == 0:
        # THEY WARN AND CONTINUE (`options_helper.cpp:109-112`); this
        # raises. DEVIATION 87. A warning on a returned model is invisible
        # from Python -- the caller asked for the best-iteration model and
        # would get the last-iteration one with no way to tell. Their
        # binary prints to a console a human is watching; this is a
        # library call.
        raise Error(
            "use_best_model=1 needs an eval set: pass eval_x_colmajor"
            " and eval_y, or leave it unset."
        )
    elif want_best_model != 0 and want_best_model != 1:
        raise Error(
            "use_best_model must be -1 (unset), 0 or 1, got "
            + String(use_best_model)
        )
    if best_model_min_trees < 1:
        raise Error(
            "best_model_min_trees must be at least 1, got "
            + String(best_model_min_trees)
        )

    var t_rows = eval_rows if eval_rows > 0 else 1
    var eval_expanded: List[Float32]
    if eval_rows > 0 and ctr_column_count != 0:
        eval_expanded = expand_raw_columns(
            ctr_tables, len(fold_counts), eval_x_colmajor, eval_rows
        )
    elif eval_rows > 0:
        eval_expanded = eval_x_colmajor.copy()
    else:
        eval_expanded = List[Float32]()
        for _ in range(len(fold_counts)):
            eval_expanded.append(Float32(0.0))

    var test_cindex = _build_cindex_from_floats(
        ctx, eval_expanded, t_rows, borders, fold_counts,
        column_nan_treatment,
    )
    var test_targets = ctx.enqueue_create_buffer[DType.float32](t_rows)
    var h_ty = ctx.enqueue_create_host_buffer[DType.float32](t_rows)
    for r in range(t_rows):
        h_ty.unsafe_ptr().unsafe_store(
            r, eval_y[r] if eval_rows > 0 else Float32(0.0)
        )
    ctx.enqueue_copy(dst_buf=test_targets, src_ptr=h_ty.unsafe_ptr())
    ctx.synchronize()
    _ = h_ty^  # past the drain (step-33 race class)

    var approx_dim = 1
    if objective == OBJECTIVE_MULTICLASS:
        approx_dim = num_classes - 1
    elif objective == OBJECTIVE_MULTICLASS_OVA:
        approx_dim = num_classes

    var test_arm = make_test_arm(
        ctx, eval_rows, test_cindex^, test_targets^,
        approx_dim, 1 + approx_dim, max_depth,
    )

    var model = TAdditiveModel()
    var fit_result = fit_with_test(
        model, ctx, n_rows, fold_counts, max_depth, cindex, targets,
        weights, use_class_weights or use_sample_weight,
        # `trace` rides POSITIONALLY (delta from the granted spec's
        # `trace=trace`: the later arguments here are positional, and a
        # positional argument may not follow a keyword one)
        n_estimators, trace, learning_rate,
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
        test=Optional(test_arm^),
        # their `permutationCount` datasets (`doc_parallel_boosting.h:
        # 137-141`). `cindex` above is `cindexes[est_perm]`, the same
        # handle, so the loop's estimation permutation and this one are the
        # same buffer.
        perm_cindexes=cindexes^,
        est_permutation=est_perm,
        od_type=od_kind,
        od_pvalue=od_pvalue,
        od_wait=od_wait,
        random_strength=random_strength,
        use_pointwise_searcher=use_pointwise_searcher,
        boost_from_average=bfa,
        grow_policy=policy,
        max_leaves=max_leaves,
        min_data_in_leaf=min_data_in_leaf,
    )
    var losses = fit_result.learn_losses.copy()
    var t_losses = fit_result.test_losses.copy()

    # ---- `ShrinkToBestIteration` (`boosting_progress_tracker.h:113-125`)
    #
    # Called here rather than inside `fit_with_test` because theirs is
    # called here: `train_template.h:127-137` shrinks the model the
    # boosting returned, after the loop, and only when there IS a test
    # set. The second tracker is separate from the detector's
    # (`boosting_progress_tracker.cpp:160-164`): it is fed only iterations
    # at or past `best_model_min_trees`, so its best iteration can differ
    # from `fit_result.best_iteration`, and it is THAT one the shrink
    # reads.
    if want_best_model == 1 and len(t_losses) > 0:
        var min_trees_best = -1
        var min_trees_err = Float64(0.0)
        for i in range(len(t_losses)):
            if i + 1 < best_model_min_trees:
                continue
            # their strict `<` (`error_tracker.h:58-64`), so the FIRST of
            # a tie wins and a plateau does not walk the cut rightwards
            if min_trees_best < 0 or t_losses[i] < min_trees_err:
                min_trees_err = t_losses[i]
                min_trees_best = i
        var best_iter = min_trees_best + 1
        if 0 < best_iter and best_iter < model.size():
            model.shrink(best_iter)

    return TrainedModel(
        model^,
        fold_counts^,
        column_one_hot^,
        borders^,
        column_nan_treatment^,
        losses^,
        t_losses^,
        fit_result.best_iteration,
        fit_result.stopped_early,
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
        ctx, expanded, n_rows, tm.borders, tm.fold_counts,
        tm.nan_treatment,
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
    _ = cursor^  # past the drain (step-33 race class, device side)
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
        ctx, expanded_x, n_rows, tm.borders, tm.fold_counts,
        tm.nan_treatment,
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
    _ = cursor^  # past the drain (step-33 race class, device side)
    var out = List[Float32]()
    for r in range(n_rows):
        for d in range(approx_dim):
            out.append(hc.unsafe_ptr().unsafe_load(d * n_rows + r))
    return out^


def one_vs_all_probabilities(
    approxes: List[Float32], n_rows: Int, num_classes: Int
) raises -> List[Float32]:
    """Their `MultiProbability` transform (`eval_processing.h:222-226`):

        CalcSigmoid(blockView, blockView)   -- ELEMENTWISE, no denominator

    `MultiClassOneVsAll` trains `numClasses` INDEPENDENT logistic
    regressions, so each plane's probability is its own sigmoid and
    **the columns do NOT sum to one**. That is not a defect to normalise
    away: `p_k` is "is this row class k", asked separately k times, and
    renormalising would assert an exclusivity the loss never fit.

    WHY NOT THEIR `Probability`, which for a multi-dimensional model is
    the SOFTMAX (`:214-221`): that branch is keyed on `ApproxDimension`
    rather than on the loss, so it would apply a softmax to independent
    sigmoid heads. `MultiProbability` is the transform that matches what
    OneVsAll actually fit, and it is theirs. A caller who wants the
    softmax can ask for it; `multiclass_probabilities` is right there.

    Input is `n_rows * num_classes`, and so is the output -- unlike
    MultiClass, nothing is reconstructed, because nothing was dropped.
    """
    if len(approxes) != n_rows * num_classes:
        raise Error(
            "one_vs_all_probabilities: got " + String(len(approxes))
            + " approxes for " + String(n_rows) + " rows x "
            + String(num_classes) + " classes"
        )
    var out = List[Float32]()
    for i in range(n_rows * num_classes):
        out.append(
            Float32(
                1.0 / (1.0 + identical_exp64(-Float64(approxes[i])))
            )
        )
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
            se += identical_exp64(Float64(approxes[r * eff + k]) - mx)
        se += identical_exp64(-mx)
        for k in range(eff):
            out.append(
                Float32(identical_exp64(Float64(approxes[r * eff + k]) - mx) / se)
            )
        out.append(Float32(identical_exp64(-mx) / se))
    return out^
