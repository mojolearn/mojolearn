# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The user-facing surface: train from raw floats, predict raw floats.

NO REFERENCE FILE -- this is the convenience layer over implemented machinery, the
`fit(X, y)` shape callers actually hold. Everything under it is the
implemented pipeline: borders from `grid_creator.binarization`
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

from gbdt.options.child_hessian import child_hessian_threshold, check_child_hessian_objective
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.device_zero import enqueue_fill

from core.identity_trace import IdentityTrace
from ensemble.instruments import StageTimes as HostStageTimes
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from max.gpu.host.device_attribute import DeviceAttribute
from gbdt.gpu_data.kernel.binarize import (
    BINARIZE_BLOCK_SIZE,
    BINARIZE_DOCS_PER_THREAD,
    binarize_float_feature_kernel,
)
from gbdt.ctrs.ctr import TCtrConfig, is_permutation_dependent_ctr_type
from gbdt.models.kernel.resident_link import (
    LINK_BLOCK,
    LINK_RAW,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    resident_link_kernel,
)
from gbdt.data.group_layout import device_group_layout
from gbdt.data.cat_code_scan import (
    cat_column_codes_resident,
    cat_column_max_code,
    onehot_column_max_code,
)
from gbdt.data.target_prep import (
    device_all_equal_first,
    fold_class_weights,
    upload_and_scan_targets,
)
from gbdt.ctrs.ctr_binarization import (
    TBinarizationOptions,
    build_binarized_target,
    build_target_borders,
    ctr_border_type_code,
)
from gbdt.ctrs.ctr_calcers import (
    compute_simple_ctrs_device_resident,
    compute_simple_ctrs_gpu_resident,
)
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ
from gbdt.ctrs.fast_prep import (
    CTR_FAST_PREP,
    CTR_INDEX_FUSED,
    CTR_ONEHOT_DEVICE,
    CTR_PREP_SHARED,
    CTR_SORT_ONCE,
    SYM_CTR_ANY,
    CtrPrepFast,
    ctr_borders_from_device,
    device_dense_codes,
    device_target_histogram,
    fast_dependent_ctrs,
    fast_freq_ctrs,
    read_codes,
    upload_codes,
)
from gbdt.ctrs.kernel.ctr_order import launch_ctr_estimation_order
from gbdt.data.permutation import DEFAULT_PERMUTATION_COUNT
from gbdt.grid_creator.binarization import (
    IDN_ORDERED_RMSE_DEVICE_GRID,
    BORDER_TYPE_GREEDY_LOG_SUM,
    best_split,
    border_type_from_name,
)
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    build_ctr_tables,
    build_ctr_tables_from_counts,
    column_plan,
    dense_category_code,
    expand_raw_columns,
)
from gbdt.models.tensor_ctr_value_table import TTensorCtrRegistry
from std.math import isfinite, log2
from std.os import getenv

# DEVIATION 258: the probability links (double, as CatBoost computes them)
# go through the host-portable exp64 under IDENTICAL; FAST is the stdlib
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    identical_exp64,
)
from checks.numerics import ftz as _hr2_ftz
from gbdt.grid_creator.gls_borders_device import device_float_borders
from core.device_scan import device_first_nonfinite
from checks.soft_f64 import (
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_gt,
    sf64_neg,
    sf64_sigmoid_f32,
    sf64_sub,
    sf64_to_f32,
)
from gbdt.methods.doc_parallel_boosting import (
    TAdditiveModel,
    fit_with_test,
    make_test_arm,
    model_approx_dim,
    predict,
)
from gbdt.methods.ordered_boosting import OrderedBoostingOptions, fit_ordered
from gbdt.data.ordered_plan import (
    ORDERED_MIN_FOLD_SIZE,
    ordered_permutation_block_size,
)
from gbdt.metrics.optimal_const_device import optimum_const_approx_device
from gbdt.overfitting_detector.overfitting_detector import (
    OD_NONE,
    od_type_from_name,
)
from std.memory import bitcast, memcpy
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.feature_sampling import check_feature_fraction
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: lane cpu4-gbdt (2026-10-04): the permutation-INDEPENDENT simple CTR
#: (FeatureFreq) is written by `compute_simple_ctrs_device_resident` in every
#: mode, on every vendor, for `counter_calc_method` SkipTest AND Full and at
#: any row count; the host calcer `compute_simple_ctrs` is the host column
#: only (checks). The routing switches that kept it in the fit
#: (`CTR_FAST_FREQ` / `MOJOLEARN_GBDT_CTR_FAST_FREQ_OFF`,
#: `IDN_CTR_FREQ_DEVICE` / `MOJOLEARN_IDN_GBDT_CTR_FREQ_DEVICE_OFF`, and the
#: `n_rows < 2^24` gate) are gone. The device calcer now divides the
#: INTEGER bin count (`ctr_calcers.mojo`, `TWeightedBinFreqCalcerGpu`), so
#: the column is exact at any size; below 2^24 rows it is the same word as
#: before, and the host column (`TWeightedBinFreqCalcer`,
#: `gbdt/host/gbdt_oracle_ctr.mojo::_feature_freq_column`) counts the same
#: way. The CTR columns themselves stay ON THE DEVICE through the border
#: build and the compressed-index build (`train`'s column loop).

#: lane/apple-fast-sym-feat (2026-10-03): the switches live in
#: `gbdt/gpu_data/sym_feat_switches.mojo` (FAST + Apple only, default OFF).
from gbdt.gpu_data.sym_feat_switches import (
    GBDT_EVAL_SKIP_EMPTY,
    GBDT_INDEX_PACK_DEVICE,
    GBDT_QUANT_DEVICE,
)
from gbdt.gpu_data.kernel.binarize import (
    PACK_BLOCK,
    PACK_BORDER_CAP,
    PACK_DOCS,
    PACK_MAX_ENTRIES,
    TR_MAX_GRID_Y,
    TR_ROWS_PER_PASS,
    TR_TILE,
    pack_cindex_words_kernel,
    transpose_rows_to_columns_kernel,
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

comptime CTR_TRACE_ENV = "MOJOLEARN_CTR_TRACE"
"""DEVIATION 2634's REACH MARKER (2026-09-12, lane harness-honesty).

`MOJOLEARN_CTR_TRACE=1` makes `train()` print, once per fit, the CTR config
split it computed and whether the target prep below actually RAN. Unset --
the shipping state, and what a user's fit does -- it costs one `getenv` per
fit and prints nothing.

WHY IT EXISTS. 2634's criteo A/B measured a clean 2.1% band between a build
with the gate ON and one with it OFF (11,790.0 ms against 12,045.3 ms,
non-overlapping, model hashes identical both sides) and the run NEVER
OBSERVED WHETHER THE BRANCH UNDER TEST EXECUTED. The gate is
`len(dependent_configs) > 0 and ctr_prep_wanted`, and `dependent_configs`
holds only the PERMUTATION-DEPENDENT CTR types; if that list is empty on the
dataset in hand then NEITHER build ran the prep and the 2.1% belongs to
something else entirely. A number whose mechanism was never witnessed is the
reached-but-inert trap (CONTRIBUTING.md (Non-default paths): a benchmark prints
which path it took, beside the timing), so the honest record said the gap was
real and its cause unknown. This is what makes the re-run attributable."""

comptime CTR_TARGET_PREP_NEEDS_CAT_2634 = not is_defined[
    "MOJOLEARN_2634_CTR_PREP_OFF"
]()
"""DEVIATION 2634 (2026-09-11, gbdt-speed lane), ours, host bookkeeping only.
`train()` built the CTR target prep whenever the default `simple_ctr`
carried a permutation-dependent config, which is always: the MinEntropy
target borders (a sort of a copy of y plus the DP), the binarized target
(one binarize per row) and one CTR estimation order per permutation. Their
only readers are `compute_simple_ctrs_gpu` and `build_ctr_tables`, both
inside the categorical-feature branch, so a fit with no `cat_features`
paid them for nothing on every call. CatBoost builds the binarized target
only when its feature manager holds CTR features. Under the switch the
prep runs only when some column is declared categorical; nothing a
numeric fit reads changes, so model bits cannot move. `-D
MOJOLEARN_2634_CTR_PREP_OFF=1` restores the unconditional build (the A/B
arm)."""

def nan_substitute_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n_in: Int32, sub: Float32
):
    """`BinarizeFloats<UseNanSubstitution=true>`'s prologue on the device
    (cpu-gpu-cleanup t-gbdt): every NaN of one staged column becomes the
    treatment's substitute (`nan_substitution`), every other value is left
    alone. The host loop it replaces tested `v != v` per row and stored the
    same constant; one thread per value, grid-stride, no arithmetic, so the
    column the binarize kernel reads is the same words."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n:
        var v = x.unsafe_load(i)
        if v != v:
            x.unsafe_store(i, sub)
        i += stride


comptime NAN_SUB_BLOCK = 256
comptime NAN_SUB_MAX_BLOCKS = 4096


def _enqueue_nan_substitute(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n_rows: Int, sub: Float32
) raises:
    ctx.enqueue_function[nan_substitute_kernel](
        x.unsafe_ptr(), Int32(n_rows), sub,
        grid_dim=min(
            (n_rows + NAN_SUB_BLOCK - 1) // NAN_SUB_BLOCK, NAN_SUB_MAX_BLOCKS
        ),
        block_dim=NAN_SUB_BLOCK,
    )


comptime BORROW_X_COLUMNS = True
"""DEVIATION 2550 (2026-09-11), the only path since cpu-gpu-cleanup t-gbdt
(the `-D MOJOLEARN_2550_HOST_COPY=1` arm that restored the two host copies
below is deleted; the name stays for `gbdt_per_round_paths`).
Ours, host bookkeeping only. `gbdt_fit` copied the caller's column-major X
into a `List` (`gbdt/estimator.mojo`) and `train` copied every raw column
again into its own `List` before quantization, so a 1M x 220 fit paid two
resize-and-memcpy passes over 880 MB before the first border (H100 HIGGS
ledger, 112 MB: 73 ms + 99 ms). Under the switch `gbdt_fit` hands `train`
the caller's pointer (`x_borrow`) and `train` reads every raw, non-
categorical column in place through `column_ptrs`; categorical and CTR
columns stay owned. The same bytes are read in the same order by the same
border builder and quantize kernel, so the model is bitwise the default's
by construction. Checks, one per side: `pixi run check-gbdt-per-round`
(the default, ON) and `check-gbdt-per-round-2550-host-copy` (the opt-out)
print one MODEL_HASH per lane, which must agree across the two builds."""
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
    OBJECTIVE_MULTIRMSE,
    OBJECTIVE_PAIR_LOGIT,
    OBJECTIVE_QUERY_RMSE,
    OBJECTIVE_YETI_RANK,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
    objective_from_name,
)
from gbdt.data.pairs import (
    PairList,
    generate_pairs,
    order_pairs_by_winner,
    prepare_pairs,
)
from gbdt.targets.kernel.pair_logit_group import PAIRLOGIT_GROUP_FUSED
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_IS,
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
    SCORE_FUNCTION_NEWTON_COSINE,
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
    var tensor_ctr_registry: TTensorCtrRegistry
    """Winning combination/tree CTR tables. Serialized model apply rebuilds
    these columns and their split-history bins in level order. Public
    combination training remains refused until this registry is populated by
    the production structure-search driver rather than focused checks."""


def _sym_feat_test_cindex(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
    nan_treatment: List[Int],
    eval_rows: Int,
) raises -> DeviceBuffer[DType.uint32]:
    """The held-out arm's compressed index: `_build_cindex_from_floats`,
    except under `GBDT_EVAL_SKIP_EMPTY` (lane/apple-fast-sym-feat, FAST +
    Apple only) with NO held-out rows, where nothing reads the index and
    the one-row dummy build (a launch and two uploads per bordered feature,
    a drain per eight) is replaced by a one-word allocation."""
    comptime if GBDT_EVAL_SKIP_EMPTY:
        if eval_rows == 0:
            return ctx.enqueue_create_buffer[DType.uint32](1)
    return _build_cindex_from_floats(
        ctx, x_colmajor, n_rows, borders, fold_counts, nan_treatment,
    )


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
    enqueue_fill(ctx, cindex, UInt32(0))

    # DEVIATION 2485 (boundary-tax WP5): mirror the existing columns
    # staging ring. Bulk copy unchanged AsIs values; retain NaN scanning
    # here because eval/predict buffers have not passed border validation.
    # Eight slots are retained until a drain before reuse and at return.
    # No quantization arithmetic/layout changes; local timing is OWED.
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
        var src = x_colmajor.unsafe_ptr() + f * n_rows
        if treat == NAN_TREATMENT_AS_IS:
            # Unlike the training-only columns path, this also consumes
            # eval/predict data. Preserve its unseen-NaN refusal before
            # copying the unchanged finite/Inf/signed-zero source bits.
            for r in range(n_rows):
                var v = src.unsafe_load(r)
                if v != v:
                    # Prior slots can still own queued uploads/kernels.
                    ctx.synchronize()
                    raise Error(
                        "There are NaNs in feature number " + String(f)
                        + " but there were no NaNs in the learn dataset"
                    )
            memcpy(dest=hx, src=src, count=n_rows)
        else:
            # a substituting treatment copies the raw bytes and substitutes
            # on the device after the upload (cpu-gpu-cleanup t-gbdt)
            memcpy(dest=hx, src=src, count=n_rows)
        hbo.unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=xdevs[slot], src_ptr=hx)
        ctx.enqueue_copy(dst_buf=bdevs[slot], src_ptr=hbo)
        if treat != NAN_TREATMENT_AS_IS:
            _enqueue_nan_substitute(ctx, xdevs[slot], n_rows, sub)
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


def _resolve_column_ptrs(
    columns: List[List[Float32]],
    column_ptrs: List[MutPointer[Float32, MutUntrackedOrigin]],
) raises -> List[MutPointer[Float32, MutUntrackedOrigin]]:
    """DEVIATION 2550: one read pointer per column. An empty `column_ptrs`
    means every column is owned by `columns` (every caller but `train`)."""
    if len(column_ptrs) == 0:
        var out = List[MutPointer[Float32, MutUntrackedOrigin]](
            capacity=len(columns)
        )
        for c in range(len(columns)):  # small-loop(columns: column pointers): one pointer per column, no element read
            out.append(
                rebind[MutPointer[Float32, MutUntrackedOrigin]](
                    columns[c].unsafe_ptr()
                )
            )
        return out^
    if len(column_ptrs) != len(columns):
        raise Error(
            "column_ptrs has " + String(len(column_ptrs)) + " entries for "
            + String(len(columns)) + " columns"
        )
    return column_ptrs.copy()


#: lane/apple-fast-sym-feat: the border slab stride of the device-resident
#: quantizer (count + up to 287 borders; a 255-border grid with both NaN
#: sentinels is 257), and the launch shape of its per-feature fallback.
comptime _QD_SLAB = 288
comptime _QD_SUB_BLOCK = 256
comptime _QD_SUB_MAX_BLOCKS = 4096


@fieldwise_init
struct _QdPackTable(Movable):
    """`pack_cindex_words_kernel`'s host-built tables (lane af-sym-feat):
    one entry per device-resident bordered feature, grouped by index word."""
    var word_start: List[UInt32]
    var col: List[UInt32]
    var shift: List[UInt32]
    var mask: List[UInt32]
    var slab: List[UInt32]
    var sub: List[Float32]
    var fits: Bool


def _qd_pack_word_table(
    lay: CompressedIndexLayout,
    dev_col_of: List[Int],
    borders: List[List[Float32]],
    nan_treatment: List[Int],
) raises -> _QdPackTable:
    """Group the device-resident bordered features by their compressed-index
    word, in word order. `fits` is the shared-page gate of
    `pack_cindex_words_kernel`: at most `PACK_MAX_ENTRIES` features and
    `PACK_BORDER_CAP` border values per word; a layout that exceeds either
    takes the per-feature launches instead."""
    var n_features = len(borders)
    var n_words = lay.columns
    var per_word = List[List[Int]]()
    for _ in range(n_words):
        per_word.append(List[Int]())
    for f in range(n_features):
        if len(borders[f]) == 0 or dev_col_of[f] < 0:
            continue
        per_word[Int(lay.features[f].offset)].append(f)
    var t = _QdPackTable(
        List[UInt32](), List[UInt32](), List[UInt32](), List[UInt32](),
        List[UInt32](), List[Float32](), True,
    )
    t.word_start.append(UInt32(0))
    for w in range(n_words):
        var total = 0
        if len(per_word[w]) > PACK_MAX_ENTRIES:
            t.fits = False
        for k in range(len(per_word[w])):
            var f = per_word[w][k]
            ref cf = lay.features[f]
            total += len(borders[f])
            var treat = NAN_TREATMENT_AS_IS
            if len(nan_treatment) == n_features:
                treat = nan_treatment[f]
            t.col.append(UInt32(dev_col_of[f]))
            t.shift.append(cf.shift)
            t.mask.append(cf.mask)
            t.slab.append(UInt32(f * _QD_SLAB))
            # a NaN substitute means "leave NaN alone" (bin 0, as AS_IS)
            t.sub.append(
                nan_substitution(treat) if treat != NAN_TREATMENT_AS_IS
                else bitcast[DType.float32](UInt32(0x7FC00000))
            )
        if total > PACK_BORDER_CAP:
            t.fits = False
        t.word_start.append(UInt32(len(t.col)))
    return t^


def _cindex_from_device_columns(
    ctx: DeviceContext,
    dev: MutPointer[Float32, MutAnyOrigin],
    dev_col_of: List[Int],
    n_rows: Int,
    borders: List[List[Float32]],
    nan_treatment: List[Int],
    lay: CompressedIndexLayout,
    mut cindex: DeviceBuffer[DType.uint32],
) raises:
    """lane/apple-fast-sym-feat (`GBDT_QUANT_DEVICE`): binarize every
    device-resident float column from the device copy. ONE upload carries
    every feature's border slab (`_QD_SLAB` floats a feature, the count
    first, the layout `binarize_float_feature_kernel` reads); a column whose
    NaN treatment substitutes does so in place on the device copy (the
    border build has already read it); then either ONE
    `pack_cindex_words_kernel` launch over every word
    (`GBDT_INDEX_PACK_DEVICE`, when the word table fits its shared page) or
    one `binarize_float_feature_kernel` launch per feature reading the
    device column and its slab. No pinned memcpy, no per-feature upload, no
    ring drain; the caller drains once. Same kernels and arithmetic as the
    staged path, so the words are the same."""
    var n_features = len(borders)
    var h_slab = ctx.enqueue_create_host_buffer[DType.float32](
        max(1, n_features * _QD_SLAB)
    )
    var hs = h_slab.unsafe_ptr()
    for f in range(n_features):
        if dev_col_of[f] < 0:
            continue
        if len(borders[f]) >= _QD_SLAB:
            raise Error(
                "feature " + String(f) + " has " + String(len(borders[f]))
                + " borders; the device quantizer's slab holds "
                + String(_QD_SLAB - 1)
            )
        hs.unsafe_store(f * _QD_SLAB, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hs.unsafe_store(f * _QD_SLAB + 1 + b, borders[f][b])
    var d_slab = ctx.enqueue_create_buffer[DType.float32](
        max(1, n_features * _QD_SLAB)
    )
    ctx.enqueue_copy(dst_buf=d_slab, src_ptr=hs)
    for f in range(n_features):
        if len(borders[f]) == 0 or dev_col_of[f] < 0:
            continue
        var treat = NAN_TREATMENT_AS_IS
        if len(nan_treatment) == n_features:
            treat = nan_treatment[f]
        if treat != NAN_TREATMENT_AS_IS:
            ctx.enqueue_function[nan_substitute_kernel](
                dev + dev_col_of[f] * n_rows, Int32(n_rows),
                nan_substitution(treat),
                grid_dim=min(
                    (n_rows + _QD_SUB_BLOCK - 1) // _QD_SUB_BLOCK,
                    _QD_SUB_MAX_BLOCKS,
                ),
                block_dim=_QD_SUB_BLOCK,
            )
    var packed = False
    comptime if GBDT_INDEX_PACK_DEVICE:
        var tab = _qd_pack_word_table(lay, dev_col_of, borders, nan_treatment)
        if tab.fits and lay.columns > 0 and len(tab.col) > 0:
            var ne = len(tab.col)
            var nw = lay.columns
            var h_ws = ctx.enqueue_create_host_buffer[DType.uint32](nw + 1)
            var h_meta = ctx.enqueue_create_host_buffer[DType.uint32](4 * ne)
            var h_sub = ctx.enqueue_create_host_buffer[DType.float32](ne)
            for w in range(nw + 1):
                h_ws.unsafe_ptr().unsafe_store(w, tab.word_start[w])
            for e in range(ne):
                h_meta.unsafe_ptr().unsafe_store(e, tab.col[e])
                h_meta.unsafe_ptr().unsafe_store(ne + e, tab.shift[e])
                h_meta.unsafe_ptr().unsafe_store(2 * ne + e, tab.mask[e])
                h_meta.unsafe_ptr().unsafe_store(3 * ne + e, tab.slab[e])
                h_sub.unsafe_ptr().unsafe_store(e, tab.sub[e])
            var d_ws = ctx.enqueue_create_buffer[DType.uint32](nw + 1)
            var d_meta = ctx.enqueue_create_buffer[DType.uint32](4 * ne)
            var d_sub = ctx.enqueue_create_buffer[DType.float32](ne)
            ctx.enqueue_copy(dst_buf=d_ws, src_ptr=h_ws.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_meta, src_ptr=h_meta.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_sub, src_ptr=h_sub.unsafe_ptr())
            comptime PACK_ROWS = PACK_BLOCK * PACK_DOCS
            # one table buffer, four views: an untracked origin so the
            # views are not seen as aliasing mutable arguments
            var pm = d_meta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            ctx.enqueue_function[pack_cindex_words_kernel](
                dev, Int32(n_rows),
                d_ws.unsafe_ptr(),
                pm,
                pm + ne,
                pm + 2 * ne,
                pm + 3 * ne,
                d_sub.unsafe_ptr(),
                d_slab.unsafe_ptr(),
                cindex.unsafe_ptr(),
                grid_dim=((n_rows + PACK_ROWS - 1) // PACK_ROWS, nw, 1),
                block_dim=(PACK_BLOCK, 1, 1),
            )
            # the tables and their staging outlive the launch: the caller
            # drains before anything reads `cindex`
            ctx.synchronize()
            _ = h_ws^
            _ = h_meta^
            _ = h_sub^
            _ = d_ws^
            _ = d_meta^
            _ = d_sub^
            packed = True
    if not packed:
        comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
        for f in range(n_features):
            if len(borders[f]) == 0 or dev_col_of[f] < 0:
                continue
            ref cf = lay.features[f]
            ctx.enqueue_function[binarize_float_feature_kernel](
                Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                dev + dev_col_of[f] * n_rows, Int32(n_rows),
                d_slab.unsafe_ptr() + f * _QD_SLAB, cindex.unsafe_ptr(),
                grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
            )
        ctx.synchronize()
    _ = h_slab^
    _ = d_slab^


def _qd_upload_float_columns(
    ctx: DeviceContext,
    column_ptrs: List[MutPointer[Float32, MutUntrackedOrigin]],
    dev_col_of: List[Int],
    n_float: Int,
    n_rows: Int,
    n_features: Int,
    x_row_major: Bool,
    x_src: MutPointer[Float32, MutUntrackedOrigin],
) raises -> DeviceBuffer[DType.float32]:
    """lane/apple-fast-sym-feat (`GBDT_QUANT_DEVICE`): the float columns
    resident on the device, column-major in `dev_col_of` order, uploaded
    ONCE. A column-major caller's columns go up straight from its pointers
    (no pinned memcpy, as `device_float_borders` already did per chunk); a
    row-major caller's rows go up as handed over and
    `transpose_rows_to_columns_kernel` lays them out (every column is then
    a raw float column, `train` has refused anything else)."""
    var mat = ctx.enqueue_create_buffer[DType.float32](
        max(1, n_float * n_rows)
    )
    if n_float == 0:
        return mat^
    if x_row_major:
        var raw = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
        ctx.enqueue_copy(
            dst_buf=raw,
            src_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](x_src),
        )
        var gy = min((n_rows + TR_TILE - 1) // TR_TILE, TR_MAX_GRID_Y)
        ctx.enqueue_function[transpose_rows_to_columns_kernel](
            mat.unsafe_ptr(), raw.unsafe_ptr(), Int32(n_rows), Int32(n_features),
            grid_dim=((n_features + TR_TILE - 1) // TR_TILE, gy, 1),
            block_dim=(TR_TILE, TR_ROWS_PER_PASS, 1),
        )
        # the raw rows are read by the transpose only; drain before they go
        ctx.synchronize()
        _ = raw^
        return mat^
    for c in range(len(column_ptrs)):
        var k = dev_col_of[c]
        if k < 0:
            continue
        var view = mat.create_sub_buffer[DType.float32](k * n_rows, n_rows)
        ctx.enqueue_copy(
            dst_buf=view,
            src_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](
                column_ptrs[c]
            ),
        )
    return mat^


def _build_cindex_from_columns(
    ctx: DeviceContext,
    columns: List[List[Float32]],
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
    nan_treatment: List[Int],
    column_ptrs: List[MutPointer[Float32, MutUntrackedOrigin]] = List[
        MutPointer[Float32, MutUntrackedOrigin]
    ](),
    dev_cols: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
    dev_col_of: List[Int] = List[Int](),
    # lane cpu4-gbdt: separately held resident columns (`train`'s CTR
    # columns) and each column's index into them, -1 for none
    dev_col_bufs: List[DeviceBuffer[DType.float32]] = List[
        DeviceBuffer[DType.float32]
    ](),
    column_dev_slot: List[Int] = List[Int](),
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

    RESIDENT COLUMNS (lane cpu4-gbdt): a column `c` with
    `column_dev_slot[c] >= 0` is already on the device, in
    `dev_col_bufs[column_dev_slot[c]]` (`n_rows` long; `train`'s CTR columns);
    it is binarized in place, never staged or uploaded. An empty
    `column_dev_slot` means no column is resident.
    """
    var n_features = len(borders)
    if len(fold_counts) != n_features:
        raise Error("fold_counts/borders length mismatch")
    var cps = _resolve_column_ptrs(columns, column_ptrs)
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    enqueue_fill(ctx, cindex, UInt32(0))

    # lane/apple-fast-sym-feat: the device-resident float columns are
    # binarized straight from the device (one slab upload, no pinned
    # memcpy, no per-feature upload, no ring drain); the staging ring below
    # then serves only the columns that have no device copy (CTR, one-hot).
    var dev_done = False
    comptime if GBDT_QUANT_DEVICE:
        if dev_cols.__bool__() and len(dev_col_of) == n_features:
            _cindex_from_device_columns(
                ctx, dev_cols.value(), dev_col_of, n_rows, borders,
                nan_treatment, lay, cindex,
            )
            dev_done = True
            var all_dev = True
            for f in range(n_features):
                if len(borders[f]) != 0 and dev_col_of[f] < 0:
                    all_dev = False
            if all_dev:
                ctx.synchronize()
                return cindex^

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
        comptime if GBDT_QUANT_DEVICE:
            if dev_done and dev_col_of[f] >= 0:
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
        var dev_slot = -1
        if len(column_dev_slot) == n_features:
            dev_slot = column_dev_slot[f]
        if dev_slot >= 0:
            # a RESIDENT column: only its borders are staged; the values
            # are binarized where they are (a NaN treatment, never set for
            # a CTR column, substitutes on a device-side copy)
            hbo.unsafe_store(0, Float32(len(borders[f])))
            for b in range(len(borders[f])):  # small-loop(borders: one column's borders, at most 255): kernel argument staging, model parameters
                hbo.unsafe_store(1 + b, borders[f][b])
            ctx.enqueue_copy(dst_buf=bdevs[slot], src_ptr=hbo)
            if treat != NAN_TREATMENT_AS_IS:
                ctx.enqueue_copy(
                    dst_buf=xdevs[slot], src_buf=dev_col_bufs[dev_slot]
                )
                _enqueue_nan_substitute(ctx, xdevs[slot], n_rows, sub)
                ctx.enqueue_function[binarize_float_feature_kernel](
                    Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                    xdevs[slot].unsafe_ptr(), Int32(n_rows),
                    bdevs[slot].unsafe_ptr(), cindex.unsafe_ptr(),
                    grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                    block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[binarize_float_feature_kernel](
                    Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                    dev_col_bufs[dev_slot].unsafe_ptr(), Int32(n_rows),
                    bdevs[slot].unsafe_ptr(), cindex.unsafe_ptr(),
                    grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                    block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
                )
            staged += 1
            continue
        var src = cps[f]
        # the raw column is a byte move into the pinned slot; a treatment
        # that substitutes NaNs does so on the device after the upload
        # (cpu-gpu-cleanup t-gbdt). AS_IS means the border build's
        # full-column NaN scan saw none in THIS SAME buffer, so it needs no
        # pass at all.
        memcpy(dest=hx, src=src, count=n_rows)
        hbo.unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=xdevs[slot], src_ptr=hx)
        ctx.enqueue_copy(dst_buf=bdevs[slot], src_ptr=hbo)
        if treat != NAN_TREATMENT_AS_IS:
            _enqueue_nan_substitute(ctx, xdevs[slot], n_rows, sub)
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


def _build_cindex_fused(
    ctx: DeviceContext,
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
    nan_treatment: List[Int],
    column_ptrs: List[MutPointer[Float32, MutUntrackedOrigin]],
    mut dev_cols: List[DeviceBuffer[DType.float32]],
    dev_slot: List[Int],
) raises -> DeviceBuffer[DType.uint32]:
    """lane/apple-fast-sym-ctr, `-D MOJOLEARN_CTR_INDEX_FUSED` (FAST +
    Apple only; reached from `train` under that flag alone).
    `_build_cindex_from_columns` with one change: a column whose
    `dev_slot` is set is a CTR column that already lives on the device, and
    the binarize kernel reads that buffer directly -- no host copy, no
    staging memcpy, no upload. A CTR column is NaN free (`AsIs`), so no
    substitution applies. Every other column takes the pointer path's ring
    unchanged. Same kernel, same borders, same values: the same index."""
    var n_features = len(borders)
    if len(fold_counts) != n_features:
        raise Error("fold_counts/borders length mismatch")
    if len(dev_slot) != n_features or len(column_ptrs) != n_features:
        raise Error("_build_cindex_fused: column lists disagree")
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    enqueue_fill(ctx, cindex, UInt32(0))

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
        var slot = staged % _CINDEX_SLOTS
        if staged >= _CINDEX_SLOTS and slot == 0:
            # one drain per revolution frees every slot in the ring
            ctx.synchronize()
        var hbo = hbos[slot].unsafe_ptr()
        hbo.unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=bdevs[slot], src_ptr=hbo)
        if dev_slot[f] >= 0:
            if treat != NAN_TREATMENT_AS_IS:
                raise Error(
                    "_build_cindex_fused: a device CTR column must be AsIs"
                )
            ctx.enqueue_function[binarize_float_feature_kernel](
                Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                dev_cols[dev_slot[f]].unsafe_ptr(), Int32(n_rows),
                bdevs[slot].unsafe_ptr(), cindex.unsafe_ptr(),
                grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
            )
            staged += 1
            continue
        var hx = hxs[slot].unsafe_ptr()
        memcpy(dest=hx, src=column_ptrs[f], count=n_rows)
        ctx.enqueue_copy(dst_buf=xdevs[slot], src_ptr=hx)
        if treat != NAN_TREATMENT_AS_IS:
            _enqueue_nan_substitute(
                ctx, xdevs[slot], n_rows, nan_substitution(treat)
            )
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
    five LCG steps. Module level so `sample_indices_for_borders_reference` and the
    check that gates it use THE SAME derivation, not two copies."""
    var sd = from_seed
    for _ in range(5):
        sd = 6364136223846793005 * sd + 1442695040888963407
    return sd


def sample_indices_for_borders_reference(
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
    order in the reference and insertion order here, which is order-equivalent
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
    # "not set", resolved by the implementation of
    # `options_helper.cpp::AdjustBoostFromAverageDefaultValue` below --
    # auto-TRUE for the losses on their list whose CalcOptimumConstApprox
    # arm is implemented (RMSE; their list also holds MAE/Quantile/MAPE, whose
    # constant needs the unimplemented CalcSampleQuantile, so those resolve to
    # FALSE with the gap named in `gbdt/metrics/optimal_const_for_loss`).
    # NOTE their list does NOT hold Logloss: an unset option on a Logloss
    # fit is FALSE on their side too, which the higgs 2026-08-22 read
    # got wrong before this implementation. 0 and 1 are explicit; 1 raises by name
    # for losses without a implemented constant.
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
    # with the Lossguide and Depthwise growing policies", and this implementation
    # refuses what it would otherwise silently drop.
    # =======================================================================
    grow_policy: String = String("SymmetricTree"),
    max_leaves: Int = -1,
    min_data_in_leaf: Int = 1,
    min_split_gain: Float64 = -1.0,
    min_child_hessian: Float64 = -1.0,
    feature_fraction: Float64 = 1.0,
    # DEVIATION 2550: the caller's column-major X read in place, with
    # `x_colmajor` empty. The binding entry (`gbdt_fit`) passes it; every
    # other caller passes the List.
    x_borrow: Optional[MutPointer[Float32, MutUntrackedOrigin]] = None,
    # lane/apple-fast-sym-feat: `x_borrow` holds ROW-MAJOR rows
    # (`x[row * n_features + feature]`). Served only by the FAST Apple
    # device quantizer (`GBDT_QUANT_DEVICE`), raw float columns only.
    x_row_major: Bool = False,
    # THE POOL'S GROUPING, their `TQueriesGrouping` sizes in row order: one
    # entry per query, each the number of CONSECUTIVE rows carrying that
    # `group_id` (`libs/data/objects.cpp:60-87` builds the groups from runs
    # and refuses a repeated id as "group Ids are not consecutive"; the
    # Python wrapper applies that rule before the sizes cross). Empty means
    # no grouping, which is every existing caller.
    group_sizes: List[UInt32] = List[UInt32](),
    # the caller's PairLogit pairs (the Pool's `pairs` and `pairs_weight`):
    # winner row, loser row and weight per pair, all empty to generate them
    # from `group_sizes` and `y` (`gbdt/data/pairs.mojo::generate_pairs`)
    pair_winners: List[UInt32] = List[UInt32](),
    pair_losers: List[UInt32] = List[UInt32](),
    pair_weights: List[Float32] = List[Float32](),
    # `feature_border_type` (`data_processing_options.cpp:15`, default
    # GreedyLogSum), their seven `EBorderSelectionType` spellings; the
    # float columns' border search (`gbdt/grid_creator/binarization.mojo`)
    feature_border_type: String = String("GreedyLogSum"),
    # `boosting_type` (`boosting_options.cpp:16`), "Plain" or "Ordered", as
    # the CALLER resolved it (their GPU default is data-dependent,
    # `catboost_options.cpp:802-807` then `defaults_helper.h:33-42`, and the
    # Python wrapper resolves it); Ordered runs `gbdt/methods/
    # ordered_boosting.mojo::fit_ordered`
    boosting_type: String = String("Plain"),
    # `fold_len_multiplier` (`boosting_options.cpp:11`, default 2) and
    # `fold_permutation_block` (`:12`, 0 unset: 64 on GPU,
    # `cuda/train_lib/train.cpp:115-118`); read by Ordered only
    fold_len_multiplier: Float64 = 2.0,
    fold_permutation_block: Int = 0,
    # THE TARGET DIMENSION (lane/algos-trees, 2026-09-27): 1 for every loss
    # but MultiRMSE, whose `y` is `target_dim` DIM-MAJOR planes of `n_rows`
    # (`y[dim * n_rows + row]`), the layout their device target is stored in
    # (`multilogit.cu:514`, `targets + idx + dim * targetAlignSize`).
    target_dim: Int = 1,
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
    `set_leaves_estimation_default` is the implementation of their
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
    half runs on the device (`compute_simple_ctrs_device_resident`); the
    dependent half runs on the device (`compute_simple_ctrs_gpu_resident`)
    over permutation `ctr_estimation_permutation_id`'s CTR estimation
    order, written on the device (`gbdt/ctrs/ctr_order.mojo`; lane
    cpu4-gbdt). Every CTR column stays on the device.

    **`ctr_estimation_permutation_id` defaults to `permutation_count - 1`,
    and it is NOT the identity.** Their permutation 0 IS the
    identity (`permutation.cpp:14-17`) and is safe on their side only
    because the learn pool was already shuffled at load
    (`private/libs/algo/preprocess.cpp:183-199`, which shuffles whenever the
    data has a categorical feature and `has_time` is false). This implementation has
    no such stage, so a caller can hand us rows sorted by target, where the
    identity order makes every row's ordered statistic read its own
    neighbourhood. `permutation_count - 1` is their ESTIMATION permutation,
    `GetEstimationPermutation()` (`doc_parallel_boosting.h:101-103`), the
    one whose model `Run()` exports.

    **ALL `permutation_count` COLUMN SETS ARE BUILT** as of 2026-08-21, one
    compressed index each (DEVIATION 89), where this used to build only the
    estimation permutation's, which is now
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
    reason is a defect (`CONTRIBUTING.md` (Non-default paths)), so both sides stay
    exercised: `checks/ctr_apply_check.mojo` and
    `checks/ctr_train_check.mojo` each run the default AND
    `feature_freq_only()` explicitly.

    What this changes for a caller who passes nothing: four columns where
    there was one, a device pass over the CTR estimation permutation where
    there was a host frequency count, and an applied model whose learn-row
    predictions no longer reproduce the fit's loss bit for bit -- because
    the ordered statistic a `Borders` column is trained on is not the
    full-learn-set histogram an applied model carries. That gap is
    a property of the reference, not a defect here; see
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
    inert, which is why it is easy to get wrong and why it is implemented now
    rather than later.

    `best_iteration` in the result is the ERROR tracker's, not the
    min-trees tracker's -- the same distinction their shrink log draws at
    `boosting_progress_tracker.h:119-122`. So a caller who sets
    `best_model_min_trees` above the best iteration gets a model with MORE
    trees than `best_iteration + 1`, and both numbers are correct.
    """
    # DEVIATION 2510 -- a host-stamped phase table for `train` itself
    # (MOJOLEARN_STAGE_TIMES=1 only, `stop_host`, no drains added): what the
    # fit spends before and after the boosting loop's own table. No
    # arithmetic changes.
    var host_times = HostStageTimes()
    var t_train = host_times.start()
    var t_phase = host_times.start()
    # ---- the grow policy, resolved and refused BY NAME where theirs is ----
    var policy = grow_policy_from_name(grow_policy)
    var border_type_code = border_type_from_name(feature_border_type)
    check_feature_fraction(feature_fraction)
    if feature_fraction < 1:
        for f in range(len(cat_features)):  # small-loop(cat_features: per-feature option flags): option validation, reads no data
            if cat_features[f]:
                raise Error("feature_fraction<1 supports numeric features only; categorical/CTR input refused")
        for f in range(len(one_hot)):  # small-loop(one_hot: per-feature option flags): option validation, reads no data
            if one_hot[f]:
                raise Error("feature_fraction<1 supports numeric features only; one_hot input refused")
    _ = child_hessian_threshold(min_child_hessian, policy, score_function)
    if not isfinite(min_split_gain) or (min_split_gain < 0 and min_split_gain != -1):
        raise Error("min_split_gain must be -1 (disabled) or finite and nonnegative")
    if policy == GROW_SYMMETRIC and min_split_gain >= 0:
        raise Error("min_split_gain requires Depthwise or Lossguide")
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

    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    var x_src: MutPointer[Float32, MutUntrackedOrigin]
    if x_borrow.__bool__():
        if len(x_colmajor) != 0:
            raise Error("pass x_colmajor or x_borrow, not both")
        x_src = x_borrow.value()
    else:
        if len(x_colmajor) != n_rows * n_features:
            raise Error("x_colmajor size mismatch")
        x_src = rebind[MutPointer[Float32, MutUntrackedOrigin]](
            x_colmajor.unsafe_ptr()
        )
    if target_dim < 1:
        raise Error("target_dim must be positive, got " + String(target_dim))
    if target_dim != 1 and loss != "MultiRMSE":
        raise Error(
            "a multi-dimensional target (target_dim=" + String(target_dim)
            + ") is read only by loss='MultiRMSE'; loss='" + loss
            + "' takes one target per row"
        )
    if len(y) != n_rows * target_dim:
        raise Error("y size mismatch")
    # ---- the pool grouping (`group_sizes`) and the querywise gate ----
    # QueryRMSE is the one loss here that reads the grouping; CatBoost's
    # other readers (the pairwise and the remaining querywise targets,
    # `cuda/targets/querywise_targets_impl.h`, `pair_logit_pairwise.h`) are
    # not implemented, so a grouping arriving with any other loss is refused
    # rather than carried and ignored. Everything here is decided before a
    # border is computed.
    var objective_code = objective_from_name(loss)
    # ---- the targets and weights, uploaded ONCE and checked on the device
    # (lane/cpu3-gbdt-b, `gbdt/data/target_prep.mojo`): every host walk of
    # `y` and `sample_weight` below reads this scan's five words instead.
    var use_sample_weight = len(sample_weight) > 0
    if use_sample_weight and len(sample_weight) != n_rows:
        raise Error(
            "sample_weight has " + String(len(sample_weight))
            + " entries for " + String(n_rows) + " rows"
        )
    var labels_multiclass = (
        loss == "MultiClass" or loss == "MultiClassOneVsAll"
        or objective_code == OBJECTIVE_MULTICLASS
        or objective_code == OBJECTIVE_MULTICLASS_OVA
    )
    # `target_dim` planes for MultiRMSE, dim-major; one for every other loss
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows * target_dim)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var tscan = upload_and_scan_targets(
        ctx, y, sample_weight, n_rows, target_dim, labels_multiclass,
        targets, weights,
    )
    var is_pair_logit = objective_code == OBJECTIVE_PAIR_LOGIT
    var is_yeti_rank = objective_code == OBJECTIVE_YETI_RANK
    var is_querywise = (
        objective_code == OBJECTIVE_QUERY_RMSE or is_pair_logit or is_yeti_rank
    )
    # ---- MultiRMSE: what this implementation carries, and what it refuses
    # BY NAME (lane/algos-trees, 2026-09-27). The fit is their
    # `TMultiClassificationTargets` with `NumClasses = GetTargetDimension()`
    # (`multiclass_targets.h:155-156`) on the greedy SymmetricTree searcher,
    # Plain boosting, numeric features, no held-out set.
    var is_multi_rmse = objective_code == OBJECTIVE_MULTIRMSE
    if is_multi_rmse:
        if target_dim < 2:
            # `CB_ENSURE(NumClasses > 1, ...)` (`multiclass_targets.h:167`)
            raise Error(
                "Only one class found, can't learn multiclass objective"
                " (MultiRMSE needs a target dimension >= 2,"
                " multiclass_targets.h:167); got target_dim="
                + String(target_dim)
            )
        if tscan.first_nonfinite >= 0:
            raise Error("MultiRMSE targets must be finite")
        if boosting_type == "Ordered":
            raise Error(
                "boosting_type='Ordered' with loss='MultiRMSE' is not carried"
                " here: the Ordered arm (gbdt/methods/ordered_boosting.mojo)"
                " is one-dimensional"
            )
        for f in range(len(cat_features)):  # small-loop(cat_features: per-feature option flags): option validation, reads no data
            if cat_features[f]:
                raise Error(
                    "loss='MultiRMSE' with cat_features is not carried here:"
                    " the CTR target binarization reads one target per row"
                )
        if len(eval_y) > 0 or len(eval_x_colmajor) > 0:
            raise Error(
                "loss='MultiRMSE' with eval_set is not carried here: the"
                " held-out arm's target is one value per row"
            )
        if len(class_weights) > 0:
            raise Error("class_weights do not apply to loss='MultiRMSE'")
    # ---- `boosting_type`, and what an Ordered fit refuses BY NAME ----
    # (lane/catboost-parity; `gbdt/methods/ordered_boosting.mojo` carries
    # the account). Decided here, before a border is computed.
    if boosting_type != "Plain" and boosting_type != "Ordered":
        raise Error(
            "boosting_type must be 'Plain' or 'Ordered', got '"
            + boosting_type + "'"
        )
    var ordered = boosting_type == "Ordered"
    if ordered:
        if policy != GROW_SYMMETRIC:
            raise Error(
                "Ordered boosting is not supported for nonsymmetric trees."
                " (catboost_options.cpp:757-759)"
            )
        if (
            objective_code == OBJECTIVE_MULTICLASS
            or objective_code == OBJECTIVE_MULTICLASS_OVA
        ):
            raise Error(
                "On GPU loss " + loss + " can't be used with ordered boosting"
                " (catboost_options.cpp:949-967: their GPU trains this loss"
                " doc-parallel and Plain only)"
            )
        if not (
            score_function == SCORE_FUNCTION_COSINE
            or score_function == SCORE_FUNCTION_NEWTON_COSINE
        ):
            raise Error(
                "Score function can't be used with ordered boosting"
                " (catboost_options.cpp:972-978); Cosine and NewtonCosine are"
                " the two with an ordered kernel"
            )
        if is_querywise:
            raise Error(
                "boosting_type='Ordered' with loss='" + loss + "' is not"
                " implemented here: the reference's folds follow the query"
                " grouping (dynamic_boosting.h:189-223), which this Ordered"
                " arm does not restate"
            )
        if use_pointwise_searcher:
            raise Error(
                "use_pointwise_searcher selects the doc-parallel Plain"
                " searcher; an Ordered fit always runs the feature-parallel"
                " fold searcher (feature_parallel_pointwise_oblivious_tree.h)"
            )
        if feature_fraction < 1:
            raise Error(
                "boosting_type='Ordered' with feature_fraction < 1 is not"
                " implemented here"
            )
    if len(group_sizes) > 0:
        # checked on the device (`gbdt/data/group_layout.mojo`), same words
        _ = device_group_layout(
            ctx, group_sizes, n_rows, "group_id: group ",
            "group_id: the group sizes cover ",
        )
        if not is_querywise:
            raise Error(
                "group_id is read only by the querywise and pairwise losses"
                " (QueryRMSE, PairLogit and YetiRank are trained here;"
                " QuerySoftMax and QueryCrossEntropy are not implemented);"
                " loss='" + loss + "' does not use it, so it is refused by"
                " name rather than carried and ignored"
            )
    # `TDocParallelSplit` (`gpu_data/doc_parallel_dataset.h:26-38`): the
    # pool's queries only when it has group ids AND fewer groups than rows;
    # otherwise `TWithoutQueriesGrouping` (`gpu_data/samples_grouping.h:
    # 28-54`), every row a query of one. A QueryRMSE fit on that grouping
    # has every query mean equal to its row's residual, so every derivative
    # is zero and the trees carry zero leaves: the reference's behavior for
    # a groupwise loss given no groups, kept as it is.
    var query_sizes = List[UInt32]()
    if is_querywise:
        if len(group_sizes) > 0 and len(group_sizes) < n_rows:
            query_sizes = group_sizes.copy()
        else:
            query_sizes.resize(n_rows, UInt32(1))
        # what the querywise arm of this implementation does not restate yet,
        # refused by name
        if policy != GROW_SYMMETRIC:
            raise Error(
                "loss='" + loss + "' is implemented on SymmetricTree only here;"
                " grow_policy=" + grow_policy_name(policy) + " is refused"
                " (the reference registers it, querywise_non_symmetric.cpp:5-14)"
            )
        if use_pointwise_searcher:
            raise Error(
                "loss='" + loss + "' with use_pointwise_searcher is not"
                " implemented here; the greedy subsets searcher only"
            )
        for f in range(len(cat_features)):  # small-loop(cat_features: per-feature option flags): option validation, reads no data
            if cat_features[f]:
                raise Error(
                    "loss='" + loss + "' with cat_features is not implemented"
                    " here: the reference shuffles whole queries for its CTR"
                    " permutations (permutation.cpp:9-11,"
                    " GenerateQueryDocsOrder), which this implementation does"
                    " not restate"
                )
        for f in range(len(one_hot)):  # small-loop(one_hot: per-feature option flags): option validation, reads no data
            if one_hot[f]:
                raise Error(
                    "loss='" + loss + "' with one_hot_features is not implemented"
                    " here"
                )
        if len(eval_y) > 0 or len(eval_x_colmajor) > 0:
            raise Error(
                "loss='" + loss + "' with eval_set is not implemented here: the"
                " held-out loss needs the eval pool's own grouping"
            )
        if len(class_weights) > 0:
            raise Error("class_weights do not apply to loss='" + loss + "'")
    # ---- the PairLogit pairs (`gbdt/data/pairs.mojo`) ----
    # Generated from the groups and grades unless the caller passed them;
    # either way put in the order the reference's device grouping flattens
    # them, winner row by winner row, and checked to stay inside one query.
    var pair_list = PairList(List[UInt32](), List[UInt32](), List[Float32]())
    if len(pair_winners) > 0 and not is_pair_logit:
        raise Error(
            "pairs are read only by loss='PairLogit' here; loss='" + loss
            + "' does not use them"
        )
    if is_pair_logit:
        if len(group_sizes) == 0:
            if len(pair_winners) > 0:
                raise Error(
                    "pairs without group_id are not implemented here: the"
                    " reference regroups and reorders the whole pool by the"
                    " pairs' connected components (data_providers.cpp:857-872,"
                    " 922-942)"
                )
            raise Error("Cannot generate pairs for data without groups")
        if len(pair_winners) > 0:
            if len(pair_losers) != len(pair_winners) or len(pair_weights) != len(pair_winners):
                raise Error("pairs: winners, losers and weights disagree in length")
            pair_list = order_pairs_by_winner(
                PairList(pair_winners.copy(), pair_losers.copy(), pair_weights.copy()),
                group_sizes, n_rows,
            )
        else:
            comptime if PAIRLOGIT_GROUP_FUSED:
                # FAST Apple: the pairs are enumerated on the device from
                # the grades (`gbdt/targets/kernel/pair_logit_group.mojo`),
                # no host list; its refusals are raised at the device setup
                pass
            else:
                pair_list = order_pairs_by_winner(
                    generate_pairs(group_sizes, y, sample_weight), group_sizes, n_rows
                )
    # Validate dense class codes before class-weight indexing or allocating
    # prediction planes. The later objective check was too late to protect
    # MakeClassificationWeights (reference data_providers.cpp:162-168).
    if loss == "MultiClass" or loss == "MultiClassOneVsAll":
        if tscan.first_bad_label() >= 0:
            if tscan.first_bad_label_is_range():
                raise Error("multiclass labels must be finite dense class codes 0..k-1")
            raise Error("multiclass labels must be integer class codes")
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
    #: DEVIATION 2550: the raw feature a BORROWED column reads from `x_src`,
    #: or -1 for a column `columns` owns.
    var column_src_feature = List[Int]()
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
    #: lane cpu4-gbdt: every CTR column is a DEVICE buffer, resident from
    #: its calcer through the border build and the compressed-index build.
    #: `dep_dev_by_perm[p][k]` is the k-th permutation-dependent column of
    #: permutation p; `ctr_dev_cols` holds the columns `columns` would have
    #: (the independent ones and the estimation permutation's dependent
    #: ones, sharing that permutation's buffers), and `column_dev_slot[c]`
    #: is column c's index into it, -1 for a raw or one-hot column.
    var dep_dev_by_perm = List[List[DeviceBuffer[DType.float32]]]()
    var ctr_dev_cols = List[DeviceBuffer[DType.float32]]()
    var column_dev_slot = List[Int]()

    var configs = cat_params.simple_ctr_configs()

    # `SplitByPermutationDependence` (`doc_parallel_dataset_builder.cpp:81`)
    # over `IsPermutationDependentCtrType` (`ctr_type.cpp:44-58`), by config
    # slot so the columns can be put back in config order afterwards.
    var independent_slots = List[Int]()
    var dependent_slots = List[Int]()
    var independent_configs = List[TCtrConfig]()
    var dependent_configs = List[TCtrConfig]()
    for c in range(len(configs)):  # small-loop(configs: CTR configs): splits the handful of CTR configs by type
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
    # permutation-dependent column are four identical datasets. This implementation
    # is Plain, so the second half of their condition
    # holds unconditionally here.
    #
    # `HasPermutationFeatures` (`:86-98`) is "some cat feature is used for
    # a CTR". A feature is used for a CTR exactly when its cardinality is
    # above `one_hot_max_size` (`binarizations_manager.cpp:106-115`), so
    # this pre-pass reads cardinalities and nothing else -- their features
    # manager makes the same decision before their options are resolved,
    # in the same order.
    # lane/apple-fast-sym-ctr, `-D MOJOLEARN_CTR_ONEHOT_DEVICE` (FAST + Apple
    # only, gbdt/ctrs/fast_prep.mojo): every declared categorical column's
    # dense-code pass on the device, ONCE, before the two cardinality
    # pre-passes below and the walk, which all read it. Off, these lists stay
    # empty and `symctr_host_prepass` is the constant True.
    var symctr_host_prepass = True
    var symctr_codes = List[DeviceBuffer[DType.uint32]]()
    var symctr_unique = List[Int]()
    var symctr_counts = List[List[Int]]()
    comptime if CTR_ONEHOT_DEVICE:
        symctr_host_prepass = False
        for f in range(n_features):
            if len(cat_features) == n_features and cat_features[f]:
                var dc = device_dense_codes(
                    ctx, x_src + f * n_rows, n_rows, f
                )
                symctr_unique.append(dc.unique_values)
                symctr_counts.append(dc.counts.copy())
                symctr_codes.append(dc.codes)
            else:
                symctr_unique.append(0)
                symctr_counts.append(List[Int]())
                symctr_codes.append(
                    ctx.enqueue_create_buffer[DType.uint32](1)
                )

    var has_permutation_features = False
    comptime if CTR_ONEHOT_DEVICE:
        if len(dependent_configs) > 0:
            for f in range(n_features):
                if symctr_unique[f] > 0 and (
                    symctr_unique[f] > cat_params.one_hot_max_size
                ):
                    has_permutation_features = True
                    break
    if len(dependent_configs) > 0 and symctr_host_prepass:
        for f in range(n_features):
            if not (len(cat_features) == n_features and cat_features[f]):
                continue
            var maxc = cat_column_max_code(ctx, x_src + f * n_rows, n_rows, f)
            if maxc + 1 > cat_params.one_hot_max_size:
                has_permutation_features = True
                break

    var ordered_ctr_column = False
    comptime if CTR_ONEHOT_DEVICE:
        if ordered:
            for f in range(n_features):
                if symctr_unique[f] > 0 and (
                    symctr_unique[f] > cat_params.one_hot_max_size
                ):
                    ordered_ctr_column = True
                    break
    if ordered and len(cat_features) == n_features and symctr_host_prepass:
        for f in range(n_features):
            if not cat_features[f]:
                continue
            var maxc_o = cat_column_max_code(ctx, x_src + f * n_rows, n_rows, f)
            if maxc_o + 1 > cat_params.one_hot_max_size:
                ordered_ctr_column = True
                break
    if ordered_ctr_column:
        # a categorical column above `one_hot_max_size` builds CTRs, and
        # those are the one categorical arm Ordered does not restate; a
        # column the dispatch makes one-hot is an ordinary split candidate
        raise Error(
            "boosting_type='Ordered' with a categorical feature that builds"
            " CTRs (cardinality above one_hot_max_size) is not implemented"
            " here: the reference builds a permutation-dependent CTR dataset"
            " per permutation (feature_parallel_dataset_builder.cpp:124-160),"
            " which this Ordered arm does not restate; use"
            " boosting_type='Plain'"
        )
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

    for _ in range(perm_count):  # small-loop(perm_count: permutations): one empty list per permutation, no data
        dep_dev_by_perm.append(List[DeviceBuffer[DType.float32]]())

    var binarized_target = List[UInt8]()
    #: the binarized target and one CTR estimation order per permutation,
    #: ON THE DEVICE, written once per fit (lane cpu4-gbdt)
    var d_btarget = ctx.enqueue_create_buffer[DType.uint8](1)
    var d_orders = List[DeviceBuffer[DType.uint32]]()
    var target_classes_count = 0
    # DEVIATION 2634 (see `CTR_TARGET_PREP_NEEDS_CAT_2634`): the binarized
    # target, its borders and the CTR orders are read only inside the
    # categorical-feature branch below, so a fit with no `cat_features`
    # skips building them. `-D MOJOLEARN_2634_CTR_PREP_OFF=1` restores the
    # unconditional build.
    var ctr_prep_wanted = True
    comptime if CTR_TARGET_PREP_NEEDS_CAT_2634:
        ctr_prep_wanted = False
        if len(cat_features) == n_features:
            for f in range(n_features):  # small-loop(n_features: per-feature option flags): reads the cat flag list only, no data
                if cat_features[f]:
                    ctr_prep_wanted = True
                    break
    if target_dim != 1:
        # MultiRMSE refuses cat_features above; the CTR target reads one
        # target per row and is never built from a multi-dimensional one
        ctr_prep_wanted = False
    if len(dependent_configs) > 0 and ctr_prep_wanted:
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
        if len(binarized_target) != n_rows:
            raise Error(
                "binarized target has " + String(len(binarized_target))
                + " entries for " + String(n_rows) + " rows"
            )
        # uploaded ONCE (lane cpu4-gbdt); every feature and permutation
        # copies it device to device
        d_btarget = ctx.enqueue_create_buffer[DType.uint8](n_rows)
        var h_bt = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
        ctx.synchronize()
        memcpy(
            dest=h_bt.unsafe_ptr(),
            src=binarized_target.unsafe_ptr(),
            count=n_rows,
        )
        ctx.enqueue_copy(dst_buf=d_btarget, src_ptr=h_bt.unsafe_ptr())
        ctx.synchronize()
        _ = h_bt^  # past the drain (step-33 race class)
        # ONE ORDER PER PERMUTATION, their loop's
        # `ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)`
        # (`doc_parallel_dataset_builder.cpp:255`), written ON THE DEVICE
        # one row per thread (`gbdt/ctrs/ctr_order.mojo`: identity for
        # permutation 0, a keyed Feistel bijection for the others) instead
        # of the host Fisher-Yates `fill_order` (lane cpu4-gbdt; bits move,
        # the host column takes the same order).
        for p in range(perm_count):  # small-loop(perm_count: permutations): one order launch per permutation, written on the device
            var d_ord = ctx.enqueue_create_buffer[DType.uint32](n_rows)
            launch_ctr_estimation_order(ctx, d_ord, n_rows, p)
            d_orders.append(d_ord^)

    # DEVIATION 2634, THE REACH MARKER (see `CTR_TRACE_ENV`). Every term of
    # the gate, plus what the prep produced, so an A/B on this switch can say
    # which side of it ran instead of inferring it from a time.
    if getenv(CTR_TRACE_ENV) == "1":
        var cat_columns = 0
        if len(cat_features) == n_features:
            for f in range(n_features):  # small-loop(n_features: per-feature option flags): trace-only count of cat flags, no data
                if cat_features[f]:
                    cat_columns += 1
        var prep_state = String("skipped")
        if len(dependent_configs) > 0 and ctr_prep_wanted:
            prep_state = String("ran")
        var gate_state = String("off")
        comptime if CTR_TARGET_PREP_NEEDS_CAT_2634:
            gate_state = String("on")
        print(
            String("[ctr-2634] simple_ctr_configs=")
            + String(len(configs))
            + " independent="
            + String(len(independent_configs))
            + " dependent="
            + String(len(dependent_configs))
            + " cat_columns="
            + String(cat_columns)
            + " ctr_prep_wanted="
            + String(ctr_prep_wanted)
            + " prep="
            + prep_state
            + " gate_2634="
            + gate_state
            + " permutations="
            + String(perm_count)
            + " target_classes="
            + String(target_classes_count)
            + " binarized_target_rows="
            + String(len(binarized_target))
            + " ctr_orders="
            + String(len(d_orders))
        )

    # lane/apple-fast-sym-ctr (FAST + Apple only; gbdt/ctrs/fast_prep.mojo,
    # docs/apple-fast/ab/sym-ctr.md). Under any of `-D MOJOLEARN_CTR_PREP_SHARED`,
    # `_SORT_ONCE`, `_INDEX_FUSED`, `_ONEHOT_DEVICE` (or SYM_CTR_ALL, default
    # since 2026-10-04, rollback `-D MOJOLEARN_SYM_CTR_ALL_OFF`) the
    # categorical walk below runs here instead and main's walk runs zero
    # times. Off, `symctr_walk_main` is the constant True and every list
    # stays empty.
    var symctr_walk_main = True
    var symctr_dev_cols = List[DeviceBuffer[DType.float32]]()
    #: [permutation][dependent ordinal] -> index into `symctr_dev_cols`
    var symctr_dep_dev = List[List[Int]]()
    #: a column whose device copy every permutation shares (FeatureFreq)
    var symctr_shared_col = List[Int]()
    var symctr_shared_dev = List[Int]()
    #: a column whose grid the walk already built
    var symctr_pre_col = List[Int]()
    var symctr_pre_borders = List[List[Float32]]()
    var symctr_pre_folds = List[Int]()
    comptime if SYM_CTR_ANY:
        symctr_walk_main = False
        for _ in range(perm_count):
            symctr_dep_dev.append(List[Int]())
        var fprep = CtrPrepFast(ctx, n_rows, perm_count)
        var use_fast_dep = False
        comptime if CTR_FAST_PREP:
            use_fast_dep = (
                len(dependent_configs) > 0
                and len(binarized_target) == n_rows
                and len(d_orders) == perm_count
            )
        var sort_once_ok = False
        comptime if CTR_SORT_ONCE:
            sort_once_ok = (
                use_fast_dep
                and len(independent_configs) > 0
                and cat_params.counter_calc_method != COUNTER_CALC_FULL
            )
            for c in range(len(independent_configs)):
                if independent_configs[c].ctr_type != CTR_FEATURE_FREQ:
                    sort_once_ok = False
        var device_out = False
        comptime if CTR_INDEX_FUSED:
            device_out = use_fast_dep
        comptime if CTR_PREP_SHARED:
            if use_fast_dep:
                # lane cpu4-gbdt: the fit's resident target and device
                # orders, shared by handle (no host order, no re-upload)
                fprep.target = d_btarget
                fprep.has_target = True
                for p in range(perm_count):
                    fprep.orders[p] = d_orders[p]
                    fprep.order_ready[p] = True
        var wants_histogram = False
        for c in range(len(configs)):
            if configs[c].ctr_type == CTR_BORDERS:
                wants_histogram = True

        for f in range(n_features):
            var is_cat = len(cat_features) == n_features and cat_features[f]
            var flagged_one_hot = len(one_hot) == n_features and one_hot[f]
            if is_cat and flagged_one_hot:
                raise Error(
                    "feature "
                    + String(f)
                    + " is in both cat_features and one_hot; cat_features"
                    " makes the one-hot decision itself, from"
                    " one_hot_max_size"
                )
            if not is_cat:
                columns.append(List[Float32]())
                column_src_feature.append(f)
                column_one_hot.append(flagged_one_hot)
                column_ctr_grid.append(-1)
                column_dev_slot.append(-1)
                continue

            var unique_values = 0
            var codes = List[UInt32]()
            var host_codes = False
            var dcodes = ctx.enqueue_create_buffer[DType.uint32](1)
            var have_dcodes = False
            var counts = List[Int]()
            comptime if CTR_ONEHOT_DEVICE:
                unique_values = symctr_unique[f]
                dcodes = symctr_codes[f]
                have_dcodes = True
                counts = symctr_counts[f].copy()
                if unique_values <= cat_params.one_hot_max_size:
                    # the raw column IS the code column (validated on the
                    # device): read it in place like a float column, and
                    # hand `_quantize_training_columns` the grid its
                    # one-hot branch would have computed with a host max
                    # loop (codes are dense, so the max is
                    # `unique_values - 1`)
                    if unique_values - 1 > 254:
                        raise Error(
                            "one-hot feature " + String(len(columns))
                            + " has more than 255 categories"
                        )
                    var bs = List[Float32]()
                    for c in range(unique_values - 1):
                        bs.append(Float32(c) + Float32(0.5))
                    symctr_pre_col.append(len(columns))
                    symctr_pre_folds.append(
                        len(bs) + 1 if len(bs) > 0 else 0
                    )
                    symctr_pre_borders.append(bs^)
                    columns.append(List[Float32]())
                    column_src_feature.append(f)
                    column_one_hot.append(True)
                    column_ctr_grid.append(-1)
                    column_dev_slot.append(-1)
                    continue
            else:
                var col = List[Float32]()
                col.resize(n_rows, Float32(0.0))
                memcpy(
                    dest=col.unsafe_ptr(),
                    src=x_src + f * n_rows,
                    count=n_rows,
                )
                var maxc = 0
                var seen = List[Bool]()
                seen.resize(1, False)
                for r in range(n_rows):
                    var c = dense_category_code(col[r], f, r)
                    if c > maxc:
                        maxc = c
                        seen.resize(maxc + 1, False)
                    seen[c] = True
                    codes.append(UInt32(c))
                host_codes = True
                unique_values = maxc + 1
                if unique_values <= 1:
                    raise Error(
                        "Error: useless catFeature found (feature "
                        + String(f)
                        + " has one category)"
                    )
                for c in range(unique_values):
                    if not seen[c]:
                        raise Error(
                            "cat_features column " + String(f)
                            + " is not densely coded: category "
                            + String(c)
                            + " is absent from 0.." + String(maxc)
                        )
                if unique_values <= cat_params.one_hot_max_size:
                    columns.append(col^)
                    column_src_feature.append(-1)
                    column_one_hot.append(True)
                    column_ctr_grid.append(-1)
                    column_dev_slot.append(-1)
                    continue

            # lane cpu4-gbdt: every calcer below reads the codes on the
            # device; the host codes feed only the host table builder
            if not have_dcodes:
                dcodes = upload_codes(ctx, codes)
                have_dcodes = True
            _ = host_codes

            var base_col = len(columns)
            # `slot_dev[c]`: config c's resident column, an index into
            # `ctr_dev_cols` (lane cpu4-gbdt; was a host column per config)
            var slot_dev = List[Int](length=len(configs), fill=-1)

            if len(independent_configs) > 0 and not sort_once_ok:
                # the device FeatureFreq writer, SkipTest and Full alike
                # (lane cpu4-gbdt; the host calcer is the host column only)
                var indep = compute_simple_ctrs_device_resident(
                    ctx, dcodes, n_rows, unique_values, independent_configs
                )
                for c in range(len(independent_slots)):  # small-loop(independent_slots: CTR config slots): one device column handle per config, no element copy
                    slot_dev[independent_slots[c]] = len(ctr_dev_cols)
                    ctr_dev_cols.append(indep[c])

            if len(dependent_configs) > 0:
                if use_fast_dep:
                    for p in range(perm_count):
                        comptime if not CTR_PREP_SHARED:
                            # one context per (feature, permutation): the
                            # scratch main pays per call; the target and
                            # order are the fit's resident ones
                            fprep = CtrPrepFast(ctx, n_rows, perm_count)
                            fprep.target = d_btarget
                            fprep.has_target = True
                            fprep.orders[p] = d_orders[p]
                            fprep.order_ready[p] = True
                        var got = fast_dependent_ctrs(
                            ctx, fprep, p, dcodes, unique_values,
                            dependent_configs, device_out,
                        )
                        for c in range(len(dependent_slots)):  # small-loop(dependent_slots: CTR config slots): one device column handle per config
                            var col_dev: DeviceBuffer[DType.float32]
                            if device_out:
                                symctr_dep_dev[p].append(
                                    len(symctr_dev_cols)
                                )
                                symctr_dev_cols.append(got.dev[c])
                                col_dev = got.dev[c]
                            else:
                                col_dev = _upload_ctr_column(
                                    ctx, got.host[c], n_rows
                                )
                            if p == est_perm:
                                slot_dev[dependent_slots[c]] = len(
                                    ctr_dev_cols
                                )
                                ctr_dev_cols.append(col_dev)
                            dep_dev_by_perm[p].append(col_dev)
                    if device_out:
                        # a dependent column's grid is permutation 0's
                        # (`GetOrComputeBorders`, see
                        # `_quantize_training_columns`), from its device
                        # copy
                        var k0 = len(symctr_dep_dev[0]) - len(
                            dependent_slots
                        )
                        for c in range(len(dependent_slots)):
                            var bs = ctr_borders_from_device(
                                ctx,
                                symctr_dev_cols[symctr_dep_dev[0][k0 + c]],
                                n_rows,
                                cat_params.ctr_binarization_for(
                                    configs[dependent_slots[c]]
                                ),
                            )
                            symctr_pre_col.append(
                                base_col + dependent_slots[c]
                            )
                            symctr_pre_folds.append(len(bs))
                            symctr_pre_borders.append(bs^)
                else:
                    if len(d_orders) != perm_count:
                        raise Error(
                            "a Borders ctr is a statistic OF the target and"
                            " cannot be computed without its grid and the"
                            " permutations' orders"
                        )
                    for p in range(perm_count):
                        var dep = compute_simple_ctrs_gpu_resident(
                            ctx,
                            dcodes,
                            n_rows,
                            unique_values,
                            dependent_configs,
                            d_btarget,
                            d_orders[p],
                        )
                        for c in range(len(dependent_slots)):  # small-loop(dependent_slots: CTR config slots): one device column handle per config
                            if p == est_perm:
                                slot_dev[dependent_slots[c]] = len(
                                    ctr_dev_cols
                                )
                                ctr_dev_cols.append(dep[c])
                            dep_dev_by_perm[p].append(dep[c])

            if sort_once_ok:
                # FeatureFreq off the LAST permutation's sorted order
                # (CTR_SORT_ONCE): same segments, same integer counts
                var gotf = fast_freq_ctrs(
                    ctx, fprep, unique_values, independent_configs,
                    device_out,
                )
                for c in range(len(independent_slots)):  # small-loop(independent_slots: CTR config slots): one device column handle per config
                    var fcol: DeviceBuffer[DType.float32]
                    if device_out:
                        symctr_shared_col.append(
                            base_col + independent_slots[c]
                        )
                        symctr_shared_dev.append(len(symctr_dev_cols))
                        symctr_dev_cols.append(gotf.dev[c])
                        fcol = gotf.dev[c]
                    else:
                        fcol = _upload_ctr_column(ctx, gotf.host[c], n_rows)
                    slot_dev[independent_slots[c]] = len(ctr_dev_cols)
                    ctr_dev_cols.append(fcol)

            var tables = List[TCtrValueTable]()
            comptime if CTR_ONEHOT_DEVICE:
                var hist = List[Int]()
                if wants_histogram and len(binarized_target) == n_rows and (
                    target_classes_count >= 1
                ):
                    if not fprep.has_target:
                        fprep.target = d_btarget
                        fprep.has_target = True
                    hist = device_target_histogram(
                        ctx, dcodes, n_rows, unique_values, fprep.target,
                        target_classes_count,
                    )
                tables = build_ctr_tables_from_counts(
                    counts,
                    hist,
                    unique_values,
                    configs,
                    target_classes_count,
                    f,
                    base_col,
                    n_rows,
                )
            else:
                tables = build_ctr_tables(
                    codes,
                    unique_values,
                    configs,
                    binarized_target,
                    target_classes_count,
                    f,
                    base_col,
                )
            for c in range(len(tables)):
                ctr_tables.append(tables[c].copy())
            for c in range(len(dependent_slots)):
                dep_col_index.append(base_col + dependent_slots[c])
            for c in range(len(configs)):  # small-loop(configs: CTR config slots): plan entries per config; the column itself stays on the device
                columns.append(List[Float32]())
                column_dev_slot.append(slot_dev[c])
                column_src_feature.append(-1)
                column_one_hot.append(False)
                ctr_grids.append(cat_params.ctr_binarization_for(configs[c]))
                column_ctr_grid.append(len(ctr_grids) - 1)
                ctr_column_count += 1
            _ = dcodes^  # past every launch that read it (drained above)

    for f in range(n_features if symctr_walk_main else 0):
        var is_cat = len(cat_features) == n_features and cat_features[f]
        var flagged_one_hot = len(one_hot) == n_features and one_hot[f]
        if is_cat and flagged_one_hot:
            raise Error(
                "feature "
                + String(f)
                + " is in both cat_features and one_hot; cat_features makes"
                " the one-hot decision itself, from one_hot_max_size"
            )
        if not is_cat:
            # DEVIATION 2550: under the switch a raw column is read in
            # place from the caller's buffer; the default copies it, one
            # flat memcpy per column (the append loop that replaced was
            # ~0.5 s of every train() at 400k x 500).
            columns.append(List[Float32]())
            column_src_feature.append(f)
            column_one_hot.append(flagged_one_hot)
            column_ctr_grid.append(-1)
            column_dev_slot.append(-1)
            continue

        # dense codes: cardinality is max + 1
        # validated, converted, maxed and density-checked on the device
        # (`gbdt/data/cat_code_scan.mojo`); the codes STAY on the device in
        # `d_codes` for the CTR calcers (lane cpu4-gbdt), and the host copy
        # in `codes` feeds the apply-time `build_ctr_tables` only
        var codes = List[UInt32]()
        var d_codes = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        var cscan = cat_column_codes_resident(
            ctx, x_src + f * n_rows, n_rows, f, codes, d_codes
        )
        var maxc = cscan.max_code
        var unique_values = maxc + 1
        if unique_values <= 1:
            # their `CB_ENSURE(uniqueValues > 1)`
            # (`batch_binarized_ctr_calcer.cpp:150`)
            raise Error(
                "Error: useless catFeature found (feature "
                + String(f)
                + " has one category)"
            )
        if cscan.first_absent >= 0:
            raise Error(
                "cat_features column " + String(f)
                + " is not densely coded: category " + String(cscan.first_absent)
                + " is absent from 0.." + String(maxc)
            )

        if unique_values <= cat_params.one_hot_max_size:
            # `UseForOneHotEncoding` (`binarizations_manager.cpp:106-109`):
            # one-hot features never get CTRs. The column is the caller's
            # values, read in place like a raw column (lane cpu4-gbdt; was a
            # host copy of the same floats).
            columns.append(List[Float32]())
            column_src_feature.append(f)
            column_one_hot.append(True)
            column_ctr_grid.append(-1)
            column_dev_slot.append(-1)
            continue

        # `slot_dev[c]`: config c's column, as an index into `ctr_dev_cols`
        var slot_dev = List[Int](length=len(configs), fill=-1)

        if len(independent_configs) > 0:
            # their `writeCtrs(..., permutationIndependent)` (`:229`),
            # over the identity `ctrEstimationOrder` (`:206`), on the
            # device for SkipTest and Full alike (see the note above
            # `gbdt.gpu_util.kernel.bootstrap`'s import)
            var indep = compute_simple_ctrs_device_resident(
                ctx, d_codes, n_rows, unique_values, independent_configs
            )
            for c in range(len(independent_slots)):  # small-loop(independent_slots: CTR config slots): one device column handle per config, no element copy
                slot_dev[independent_slots[c]] = len(ctr_dev_cols)
                ctr_dev_cols.append(indep[c])

        if len(dependent_configs) > 0:
            # their `writeCtrs(..., permutationDependent)` (`:257`), ONCE
            # PER PERMUTATION, over that permutation's order written at
            # `:255`. The model column takes the estimation permutation's
            # buffer (shared, not copied); every permutation's set stays
            # on the device for its own compressed index.
            if len(d_orders) != perm_count:
                raise Error(
                    "a Borders ctr is a statistic OF the target and cannot"
                    " be computed without its grid and the permutations'"
                    " orders"
                )
            for p in range(perm_count):
                var dep = compute_simple_ctrs_gpu_resident(
                    ctx,
                    d_codes,
                    n_rows,
                    unique_values,
                    dependent_configs,
                    d_btarget,
                    d_orders[p],
                )
                for c in range(len(dependent_slots)):  # small-loop(dependent_slots: CTR config slots): one device column handle per config, no element copy
                    if p == est_perm:
                        slot_dev[dependent_slots[c]] = len(ctr_dev_cols)
                        ctr_dev_cols.append(dep[c])
                    dep_dev_by_perm[p].append(dep[c])
        _ = d_codes^
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
        ctr_tables.extend(tables^)
        var base_col = len(columns)
        for c in range(len(dependent_slots)):  # small-loop(dependent_slots: CTR config slots): one column index per config
            dep_col_index.append(base_col + dependent_slots[c])
        for c in range(len(configs)):  # small-loop(configs: CTR config slots): appends plan entries per config; the column itself stays on the device
            # the CTR column lives on the device (`ctr_dev_cols`); its host
            # slot stays empty and its read pointer is never dereferenced
            columns.append(List[Float32]())
            column_dev_slot.append(slot_dev[c])
            column_src_feature.append(-1)
            column_one_hot.append(False)
            ctr_grids.append(cat_params.ctr_binarization_for(configs[c]))
            column_ctr_grid.append(len(ctr_grids) - 1)
            ctr_column_count += 1

    var n_columns = len(columns)

    # column index -> its ordinal among the permutation-dependent columns,
    # or -1. Built here rather than carried, because the column positions
    # are only final once every feature has been walked.
    var dep_ordinal_of_column = List[Int]()
    for _ in range(n_columns):  # small-loop(n_columns: column plan entries): one -1 per column, no data
        dep_ordinal_of_column.append(-1)
    for k in range(len(dep_col_index)):  # small-loop(dep_col_index: dependent column plan): one ordinal per dependent column, no data
        dep_ordinal_of_column[dep_col_index[k]] = k

    # DEVIATION 2550: one read pointer per column, into `x_src` for a
    # borrowed column and into `columns` for an owned one. `columns` takes
    # no further appends, so its element buffers stay put. A CTR column's
    # host slot is empty (its values are `ctr_dev_cols`' buffer) and its
    # pointer is never read: every reader takes the device buffer through
    # `column_dev_slot` first.
    var column_ptrs = List[MutPointer[Float32, MutUntrackedOrigin]](
        capacity=n_columns
    )
    for c in range(n_columns):  # small-loop(n_columns: column pointers): one pointer per column, no element read
        if column_src_feature[c] >= 0:
            column_ptrs.append(x_src + column_src_feature[c] * n_rows)
        else:
            column_ptrs.append(
                rebind[MutPointer[Float32, MutUntrackedOrigin]](
                    columns[c].unsafe_ptr()
                )
            )

    host_times.stop_host("train_pre_quantize", t_phase)
    t_phase = host_times.start()
    # lane/apple-fast-sym-feat (`GBDT_QUANT_DEVICE`): the float columns
    # resident on the device, uploaded once here and read by the border
    # build and the index build below; freed after the index build.
    var qd_mat: Optional[DeviceBuffer[DType.float32]] = None
    var qd_cols: Optional[MutPointer[Float32, MutAnyOrigin]] = None
    var qd_col_of = List[Int]()
    comptime if GBDT_QUANT_DEVICE:
        var qd_n_float = 0
        for c in range(n_columns):
            if not column_one_hot[c] and column_ctr_grid[c] < 0:
                qd_col_of.append(qd_n_float)
                qd_n_float += 1
            else:
                qd_col_of.append(-1)
        if x_row_major and qd_n_float != n_columns:
            raise Error(
                "gbdt_fit_rowmajor: a row-major fit takes raw float columns"
                " only (no categorical, one-hot or CTR columns)"
            )
        var mat = _qd_upload_float_columns(
            ctx, column_ptrs, qd_col_of, qd_n_float, n_rows, n_features,
            x_row_major, x_src,
        )
        qd_cols = Optional(mat.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
        qd_mat = Optional(mat^)
    else:
        if x_row_major:
            raise Error(
                "gbdt_fit_rowmajor is served only by the FAST Apple device"
                " quantizer (-D MOJOLEARN_GBDT_QUANT_DEVICE)"
            )
    # lane/apple-fast-sym-ctr: the grids the fast walk already built, by
    # column (empty unless a SYM_CTR define is on)
    var symctr_pre_has = List[Bool]()
    var symctr_pre_b = List[List[Float32]]()
    var symctr_pre_f = List[Int]()
    comptime if SYM_CTR_ANY:
        for _ in range(n_columns):
            symctr_pre_has.append(False)
            symctr_pre_b.append(List[Float32]())
            symctr_pre_f.append(0)
        for k in range(len(symctr_pre_col)):
            symctr_pre_has[symctr_pre_col[k]] = True
            symctr_pre_b[symctr_pre_col[k]] = symctr_pre_borders[k].copy()
            symctr_pre_f[symctr_pre_col[k]] = symctr_pre_folds[k]
    var grid = _quantize_training_columns(
        ctx, columns, column_one_hot, column_ctr_grid,
        dep_ordinal_of_column, dep_dev_by_perm, ctr_grids, n_rows,
        border_count, border_build_max_samples, random_seed, nan_mode,
        column_ptrs=column_ptrs,
        border_type=border_type_code,
        ctr_dev_cols=ctr_dev_cols,
        column_dev_slot=column_dev_slot,
        dev_cols=qd_cols,
        pre_has=symctr_pre_has,
        pre_borders=symctr_pre_b,
        pre_folds=symctr_pre_f,
    )
    var borders = grid[0].copy()
    var fold_counts = grid[1].copy()
    var column_nan_treatment = grid[2].copy()
    host_times.stop_host("train_quantize_borders", t_phase)
    t_phase = host_times.start()

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
    # (`doc_parallel_dataset_builder.cpp:104-124`); this implementation packs every
    # column into one buffer, so a permutation costs a whole index rather
    # than the dependent slice of one. DEVIATION 89.
    var cindexes = List[DeviceBuffer[DType.uint32]]()
    var any_dep = False
    for c in range(n_columns):  # small-loop(n_columns: column plan entries): one flag per column, no data
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
                    column_ptrs=column_ptrs,
                    dev_cols=qd_cols,
                    dev_col_of=qd_col_of,
                    dev_col_bufs=ctr_dev_cols,
                    column_dev_slot=column_dev_slot,
                )
            )
            continue
        # the dependent columns are read IN PLACE from this permutation's
        # RESIDENT CTR buffers (lane cpu4-gbdt), the other CTR columns from
        # `ctr_dev_cols`, every raw and one-hot column from `column_ptrs`
        # (or the resident float matrix under GBDT_QUANT_DEVICE), through
        # the same builder as above (same kernels, same borders, same
        # writes, so the same bits, with no column crossing the bus). This
        # subsumes lane/apple-fast-sym-ctr's CTR_INDEX_FUSED arm: its walk
        # now hands its device columns to the same resident lists.
        var perm_dev = List[DeviceBuffer[DType.float32]]()
        var perm_slot = List[Int](length=n_columns, fill=-1)
        for c in range(n_columns):  # small-loop(n_columns: column plan entries): one device buffer handle per CTR column, no element read
            var ord = dep_ordinal_of_column[c]
            if ord >= 0:
                perm_slot[c] = len(perm_dev)
                perm_dev.append(dep_dev_by_perm[p][ord])
            elif column_dev_slot[c] >= 0:
                perm_slot[c] = len(perm_dev)
                perm_dev.append(ctr_dev_cols[column_dev_slot[c]])
        cindexes.append(
            _build_cindex_from_columns(
                ctx, columns, n_rows, borders, fold_counts,
                column_nan_treatment,
                column_ptrs=column_ptrs,
                dev_cols=qd_cols,
                dev_col_of=qd_col_of,
                dev_col_bufs=perm_dev,
                column_dev_slot=perm_slot,
            )
        )
    var cindex = cindexes[est_perm].copy()
    # DEVIATION 2550: `column_ptrs` points into `columns`; hold both to here
    _ = len(columns)
    _ = len(column_ptrs)
    # lane/apple-fast-sym-feat: the index builders drained before returning,
    # so the resident float matrix can go now
    _ = qd_mat^
    _ = qd_cols^
    host_times.stop_host("train_cindex_build", t_phase)
    t_phase = host_times.start()

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
    # `targets` and `weights` were uploaded and scanned at entry (`tscan`).

    # THEIR COMBINATION IS A PRODUCT (`private/libs/target/
    # data_providers.cpp:168`):
    #
    #     rawWeights[i] * rawGroupWeights[i] * classWeights[targetClass[i]]
    #
    # `rawGroupWeights` is the querywise family's and is 1 here -- this
    # implementation carries no `group_id`, so there is nothing to weight by. The
    # other two multiply, and a caller may pass either, both, or neither.

    # `classWeights[(size_t)targetClassesArray[i]]` indexes by the TARGET
    # CLASS, so how many entries it needs depends on the loss: two for the
    # binarized classification targets, `numClasses` for MultiClass.
    var n_class_slots = 2
    if loss == "MultiClass" or loss == "MultiClassOneVsAll":
        n_class_slots = tscan.max_label + 1
    var use_class_weights = len(class_weights) > 0
    if use_class_weights and len(class_weights) != n_class_slots:
        raise Error(
            "class_weights takes " + String(n_class_slots)
            + " entries for loss '" + loss + "', got "
            + String(len(class_weights))
        )

    if tscan.first_negative_weight >= 0:
        raise Error(
            "sample_weight at row " + String(tscan.first_negative_weight)
            + " is negative"
        )
    if use_class_weights:
        # their `targetClassesArray`: the dense class code for MultiClass,
        # the binarized target otherwise; the one product on the device
        fold_class_weights(
            ctx, targets, weights, class_weights, n_rows,
            loss == "MultiClass" or loss == "MultiClassOneVsAll",
        )
    if is_pair_logit:
        comptime if PAIRLOGIT_GROUP_FUSED:
            # generated pairs (empty list): the device setup rewrites the
            # weights with the per-row pair weights; a caller's pairs keep
            # the host fold
            if len(pair_list.winners) > 0:
                var pair_prep = prepare_pairs(
                    pair_list.winners, pair_list.losers, pair_list.weights, n_rows
                )
                ctx.enqueue_copy(
                    dst_buf=weights, src_ptr=pair_prep.row_weights.unsafe_ptr()
                )
                ctx.synchronize()
                _ = len(pair_prep.row_weights)
        else:
            # `InitPairLogit` (`targets/querywise_targets_impl.h:326-346`): the
            # target weights become the per-row sums of the pair weights, folded
            # in the pinned endpoint order (`gbdt/data/pairs.mojo::prepare_pairs`)
            var pair_prep = prepare_pairs(
                pair_list.winners, pair_list.losers, pair_list.weights, n_rows
            )
            ctx.enqueue_copy(
                dst_buf=weights, src_ptr=pair_prep.row_weights.unsafe_ptr()
            )
            ctx.synchronize()
            _ = len(pair_prep.row_weights)
    host_times.stop_host("train_targets_upload", t_phase)
    t_phase = host_times.start()

    var loss_desc = make_loss_description(
        loss,
        alpha=loss_alpha,
        q=loss_q,
        delta=loss_delta,
        variance_power=loss_variance_power,
        border=loss_border,
    )
    var objective = loss_desc.loss_function
    check_child_hessian_objective(min_child_hessian, objective)

    # ---- `AdjustBoostFromAverageDefaultValue` (`options_helper.cpp:
    # 353-374`), implemented 2026-08-22, completed 2026-09-19
    # (lane/catboost-parity). Their rule, verbatim: if the option is SET,
    # keep it; else set TRUE on a single host with no baseline and no
    # continuation for RMSE, MAE, Quantile, MAPE (and three multi losses
    # this implementation does not have). Logloss is NOT on the list. This
    # implementation has no baseline column and no continuation, so those
    # guards are trivially met. MAE, Quantile and MAPE resolved FALSE here
    # until their constant (CalcSampleQuantile) existed; it does now
    # (`gbdt/metrics/sample_quantile.mojo`), so they take their rule.
    var bfa: Bool
    if boost_from_average == 1 and objective == OBJECTIVE_MULTIRMSE:
        # on their list (`catboost_options.cpp:705-709`), with a PER-DIMENSION
        # start (`optimal_const_for_loss.h:230-239`) this model's one-value
        # bias does not carry
        raise Error(
            "boost_from_average=True with loss MultiRMSE is not carried here:"
            " its StartingPoint is per-dimension"
            " (optimal_const_for_loss.h:230-239) and this model's bias is one"
            " value"
        )
    if boost_from_average == 1:
        if not (
            objective == OBJECTIVE_RMSE
            or objective == OBJECTIVE_LOGLOSS
            or objective == OBJECTIVE_CROSSENTROPY
            or objective == OBJECTIVE_QUANTILE
            or objective == OBJECTIVE_MAE
            or objective == OBJECTIVE_MAPE
        ):
            # their CB_ENSURE's list (`catboost_options.cpp:705-709`), the
            # losses of it this implementation trains
            raise Error(
                "You can use boost_from_average only for these loss"
                " functions now: RMSE, Logloss, CrossEntropy, Quantile, MAE,"
                " MAPE (catboost_options.cpp:705-709; their MultiQuantile"
                " and RMSPE are not trained here, and MultiRMSE's"
                " per-dimension start is refused above)."
            )
        bfa = True
    elif boost_from_average == 0:
        bfa = False
    elif boost_from_average == -1:
        # DEVIATION 5951: their rule sets TRUE for MultiRMSE too
        # (`options_helper.cpp:367`); its per-dimension start is not carried
        # (the refusal above), so an unset option resolves FALSE here and the
        # cursor starts at zero, as it does for MultiClass.
        bfa = (
            objective == OBJECTIVE_RMSE
            or objective == OBJECTIVE_QUANTILE
            or objective == OBJECTIVE_MAE
            or objective == OBJECTIVE_MAPE
        )
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
        # the entry scan's words (`tscan`): the lowest row that is not a
        # dense class code is refused, the largest code sets the count
        var bad_label = tscan.first_bad_label()
        if bad_label >= 0:
            raise Error(
                "MultiClass label at row " + String(bad_label)
                + " is not a dense class code 0..k-1 (negative, non-integer"
                " or non-finite)"
            )
        var mx = tscan.max_label
        num_classes = mx + 1
        if num_classes < 2:
            raise Error(
                "the multiclass family needs at least two classes; the"
                " labels reach only " + String(num_classes)
            )
    elif objective == OBJECTIVE_MULTIRMSE:
        # `NumClasses = GetTargetDimension()` (`multiclass_targets.h:155-156`)
        num_classes = target_dim
    var estimation = set_leaves_estimation_default(
        loss_desc,
        method_override=leaf_estimation_method,
        iterations_override=leaf_estimation_iterations,
        l2_override=l2_leaf_reg,
        ordered=ordered,
    )
    # `GradientBoosting(l2_leaf_reg=None)` resolves to the loss's default in
    # the wrapper (`catboost_options.cpp:34-37`, 3.0; 0 for YetiRank,
    # `:166-172`), so `l2_leaf_reg` arrives here nonnegative and the override
    # returns it unchanged, as the fit always read it.
    var resolved_l2 = estimation.l2_reg
    if is_yeti_rank and resolved_l2 == Float32(0.0):
        # `if (treeConfig.L2Reg == 0.0f) { treeConfig.L2Reg = 1e-20f; }`
        # (`catboost_options.cpp:357-359`). The reference applies it to every
        # loss; here only to YetiRank, whose default reaches it. For the other
        # losses an explicit 0 is NOT IMPLEMENTED (gbdt/NOT_IMPLEMENTED.tsv).
        resolved_l2 = Float32(1e-20)

    # `TCatBoostOptions::SetLeavesEstimationDefault`'s sibling for
    # sampling (`catboost_options.cpp:779-800`), the two lines of it this
    # implementation can reach. `bootstrap_type` empty means "take the
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
            # searcher refuses MVS, so this implementation has nothing to implement.
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
    # theirs and not ours -- this implementation carries no pairwise loss.
    var t_rows = eval_rows if eval_rows > 0 else 1
    var test_targets = ctx.enqueue_create_buffer[DType.float32](t_rows)
    # the caller's eval targets straight into the device buffer (one H2D),
    # or the one-row zero placeholder
    if eval_rows > 0:
        ctx.enqueue_copy(dst_buf=test_targets, src_ptr=eval_y.unsafe_ptr())
    else:
        enqueue_fill(ctx, test_targets, Float32(0.0))
    ctx.synchronize()
    _ = len(eval_y)  # past the drain (step-33 race class)
    # `eval_y[r] != eval_y[0]` for some row, decided on the device (one
    # word back); NaN compares unequal as on the host
    var eval_const_target = True
    if eval_rows > 1:
        eval_const_target = device_all_equal_first(ctx, test_targets, eval_rows)
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

    var eval_expanded: List[Float32]
    if eval_rows > 0 and ctr_column_count != 0:
        eval_expanded = expand_raw_columns(
            ctr_tables, len(fold_counts), eval_x_colmajor, eval_rows
        )
    elif eval_rows > 0:
        eval_expanded = eval_x_colmajor.copy()
    else:
        eval_expanded = List[Float32]()
        for _ in range(len(fold_counts)):  # small-loop(fold_counts: features of the one-row placeholder): one zero per feature for an empty eval set
            eval_expanded.append(Float32(0.0))

    var test_cindex = _sym_feat_test_cindex(
        ctx, eval_expanded, t_rows, borders, fold_counts,
        column_nan_treatment, eval_rows,
    )

    var approx_dim = 1
    if objective == OBJECTIVE_MULTICLASS:
        approx_dim = num_classes - 1
    elif objective == OBJECTIVE_MULTICLASS_OVA:
        approx_dim = num_classes
    elif objective == OBJECTIVE_MULTIRMSE:
        approx_dim = num_classes

    var test_arm = make_test_arm(
        ctx, eval_rows, test_cindex^, test_targets^,
        approx_dim, 1 + approx_dim, max_depth,
    )

    # ---- ORDERED BOOSTING (lane/catboost-parity) ----------------------
    # Everything above -- borders, the compressed index, the targets and
    # weights, the loss and its leaf estimator, the bootstrap -- is shared
    # with the Plain fit; the loop is `fit_ordered`.
    if ordered:
        var o_model = TAdditiveModel()
        var o_start = Float64(0.0)
        if bfa:
            # `StartingPoint = CalcOptimumConstApprox(...)`
            # (`dynamic_boosting.h:563-573`), the plain fit's own device
            # helper (gbdt/metrics/optimal_const_device.mojo)
            o_start = optimum_const_approx_device(
                ctx, objective, targets, weights, True, n_rows,
                Float64(loss_desc.get_alpha()),
            )
            o_model.bias = o_start
        var o_opts = OrderedBoostingOptions(
            objective,
            loss_desc.kernel_alpha(),
            loss_desc.get_alpha(),
            loss_desc.get_logloss_border(),
            estimation.method,
            estimation.iterations,
            score_function,
            learning_rate,
            resolved_l2,
            random_strength,
            random_seed,
            boot_kind,
            boot_param,
            DEFAULT_PERMUTATION_COUNT if permutation_count == -1
            else permutation_count,
            fold_len_multiplier,
            ordered_permutation_block_size(n_rows, fold_permutation_block),
            ORDERED_MIN_FOLD_SIZE,
            Float32(o_start),
        )
        var o_layout = build_layout(fold_counts, column_one_hot)
        var o_out = fit_ordered(
            ctx, o_layout, cindex, targets, weights, n_rows, n_estimators,
            max_depth, ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
            column_one_hot, o_opts, o_model, trace, test_arm,
            od_kind, od_pvalue, od_wait,
        )
        # `ShrinkToBestIteration`, the Plain fit's own block below
        if want_best_model == 1 and len(o_out.test_losses) > 0:
            var o_min_best = -1
            var o_min_err = Float64(0.0)
            for i in range(len(o_out.test_losses)):  # small-loop(test_losses: boosting iterations): argmin over the per-tree loss list the fit returned
                if i + 1 < best_model_min_trees:
                    continue
                if o_min_best < 0 or o_out.test_losses[i] < o_min_err:
                    o_min_err = o_out.test_losses[i]
                    o_min_best = i
            var o_best_iter = o_min_best + 1
            if 0 < o_best_iter and o_best_iter < o_model.size():
                o_model.shrink(o_best_iter)
        host_times.stop_host("train_ordered_fit", t_phase)
        host_times.stop_host("train_total", t_train)
        host_times.report()
        return TrainedModel(
            o_model^,
            fold_counts^,
            column_one_hot^,
            borders^,
            column_nan_treatment^,
            o_out.learn_losses.copy(),
            o_out.test_losses.copy(),
            o_out.best_iteration,
            o_out.stopped_early,
            ctr_column_count,
            ctr_tables^,
            TTensorCtrRegistry(len(fold_counts)),
        )


    host_times.stop_host("train_pre_fit", t_phase)
    t_phase = host_times.start()
    var model = TAdditiveModel()
    var fit_result = fit_with_test(
        model, ctx, n_rows, fold_counts, max_depth, cindex, targets,
        weights, use_class_weights or use_sample_weight or is_pair_logit,
        # `trace` rides POSITIONALLY (delta from the granted spec's
        # `trace=trace`: the later arguments here are positional, and a
        # positional argument may not follow a keyword one)
        n_estimators, trace, learning_rate,
        resolved_l2, True,
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
        min_split_gain=min_split_gain,
        min_child_hessian=min_child_hessian,
        feature_fraction=feature_fraction,
        group_sizes=query_sizes,
        pair_winners=pair_list.winners.copy(),
        pair_losers=pair_list.losers.copy(),
        pair_weights=pair_list.weights.copy(),
    )
    host_times.stop_host("train_fit_with_test", t_phase)
    t_phase = host_times.start()
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
        for i in range(len(t_losses)):  # small-loop(t_losses: boosting iterations): argmin over the per-tree loss list the fit returned
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

    host_times.stop_host("train_post_fit", t_phase)
    host_times.stop_host("train_total", t_train)
    host_times.report()
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
        TTensorCtrRegistry(len(fold_counts)),
    )


def _upload_ctr_column(
    ctx: DeviceContext, col: List[Float32], n_rows: Int
) raises -> DeviceBuffer[DType.float32]:
    """A host CTR column onto the device, one bulk copy (lane cpu4-gbdt:
    the FAST + Apple sym-ctr walk's host-returning drivers feed the same
    resident column lists as every other route)."""
    if len(col) != n_rows:
        raise Error(
            "CTR column has " + String(len(col)) + " values for "
            + String(n_rows) + " rows"
        )
    var h = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.synchronize()
    memcpy(dest=h.unsafe_ptr(), src=col.unsafe_ptr(), count=n_rows)
    var d = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^  # past the drain (step-33 race class)
    return d^


def _ctr_borders_device(
    ctx: DeviceContext,
    col: DeviceBuffer[DType.float32],
    n_rows: Int,
    description: TBinarizationOptions,
) raises -> List[Float32]:
    """`IDN_CTR_BORDERS_DEVICE` (`gbdt/ctrs/ctr_binarization.mojo`): one
    CTR column's grid from `device_float_borders` over every row at NaN mode
    Forbidden, with `compute_ctr_borders`' constant-feature 0.5. The host
    column's twin is `gbdt/host/gbdt_oracle_ctr.mojo::_ctr_borders_host_grid`.

    lane cpu4-gbdt: the column is RESIDENT (`train`'s CTR buffers), read
    device to device by `device_float_borders`' `dev_col_bufs` input; this is
    the CTR grid's only route in the GPU fit."""
    var one = List[DeviceBuffer[DType.float32]](capacity=1)
    one.append(col)
    var got = device_float_borders(
        ctx, List[MutPointer[Float32, MutUntrackedOrigin]](), n_rows, n_rows,
        description.border_count,
        NAN_MODE_FORBIDDEN, generate_seed_for_borders(UInt64(0)),
        ctr_border_type_code(description),
        dev_col_bufs=one,
    )
    var grids = got[0].copy()
    if len(grids) != 1:
        raise Error(
            "ctr borders: the device border build returned "
            + String(len(grids)) + " grids for one column"
        )
    var bs = grids[0].copy()
    if len(bs) == 0:
        # `//hack to work with constant features`
        # (`gpu_binarization_helpers.cpp:22-27`)
        bs.append(Float32(0.5))
    return bs^


def _quantize_training_columns(
    ctx: DeviceContext,
    columns: List[List[Float32]],
    column_one_hot: List[Bool],
    column_ctr_grid: List[Int],
    dep_ordinal_of_column: List[Int],
    dep_dev_by_perm: List[List[DeviceBuffer[DType.float32]]],
    ctr_grids: List[TBinarizationOptions],
    n_rows: Int,
    border_count: Int,
    border_build_max_samples: Int,
    random_seed: UInt64,
    nan_mode: String,
    column_ptrs: List[MutPointer[Float32, MutUntrackedOrigin]] = List[
        MutPointer[Float32, MutUntrackedOrigin]
    ](),
    # `feature_border_type` (`binarization.mojo` BORDER_TYPE_*), for the
    # float columns only: the CTR columns keep their own grids
    border_type: Int = BORDER_TYPE_GREEDY_LOG_SUM,
    # lane cpu4-gbdt: the RESIDENT CTR columns (`train`) and each column's
    # index into them (-1 for a raw or one-hot column); empty when there
    # is no CTR column
    ctr_dev_cols: List[DeviceBuffer[DType.float32]] = List[
        DeviceBuffer[DType.float32]
    ](),
    column_dev_slot: List[Int] = List[Int](),
    dev_cols: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
    # lane/apple-fast-sym-ctr: grids `train`'s fast categorical walk already
    # built (one-hot from the device cardinality, dependent CTR columns
    # from a device min/max). Read only under the FAST + Apple SYM_CTR
    # flags; empty for every other build and caller.
    pre_has: List[Bool] = List[Bool](),
    pre_borders: List[List[Float32]] = List[List[Float32]](),
    pre_folds: List[Int] = List[Int](),
) raises -> Tuple[List[List[Float32]], List[Int], List[Int]]:
    """Shared grid builder for ordinary training and reusable numeric pools.

    Keep sampling, sorting, NaN treatment and reduction order identical to
    the ordinary training path. Prepared pools freeze this result explicitly.

    DEVIATION 2550: raw columns are read through `column_ptrs` (empty means
    every column is owned by `columns`). CTR columns are device buffers
    (`dep_dev_by_perm`, `ctr_dev_cols`; lane cpu4-gbdt) and their grids
    are built on the device from them.
    """
    var n_columns = len(columns)
    var cps = _resolve_column_ptrs(columns, column_ptrs)
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
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
    var nan_mode_opt_early = nan_mode_from_name(nan_mode)

    var n_float_prescan = 0
    var float_idx = List[Int]()
    for f in range(n_columns):  # small-loop(n_columns: column plan entries): lists the float columns by flag, no data
        if not column_one_hot[f] and column_ctr_grid[f] < 0:
            float_idx.append(f)
            n_float_prescan += 1

    # ONE index subset for the whole dataset, drawn WITHOUT REPLACEMENT,
    # shared by every float column -- their `GetSubsetForBuildBorders`
    # (`libs/data/quantization.cpp:118-141`). LANE hr2-gbdt-host put the
    # GreedyLogSum grids on the device (`gbdt/grid_creator/gls_borders.mojo`:
    # a parallel Feistel subsample, device keys, a segmented sort, one
    # search thread per column); cpu-gpu-cleanup w2-trees put the six other
    # `feature_border_type`s on the same pipeline
    # (`gbdt/grid_creator/border_types.mojo`), so the host draw
    # (`_draw_task`), the host phase B (`_dp_task`) and the
    # `MOJOLEARN_HR2_OLD_BORDERS` arm are gone.
    var dev_borders = List[List[Float32]]()
    var dev_modes = List[Int]()
    if n_float_prescan > 0:
        var dcols = List[MutPointer[Float32, MutUntrackedOrigin]](
            capacity=n_float_prescan
        )
        for k in range(n_float_prescan):  # small-loop(n_float_prescan: column pointers): one pointer per float column, no element read
            dcols.append(cps[float_idx[k]])
        var got = device_float_borders(
            ctx, dcols, n_rows, border_sample_n, border_count, nan_mode_opt_early,
            generate_seed_for_borders(random_seed), border_type,
            dev_cols=dev_cols,
        )
        dev_borders = got[0].copy()
        dev_modes = got[1].copy()
        _ = len(cps)
    var dev_k = 0
    # their `TFloatFeature::NanValueTreatment`, one per COLUMN. One-hot and
    # CTR columns stay `AsIs`: a one-hot column holds dense codes and a CTR
    # column holds a computed statistic, and a NaN in either is a caller
    # error rather than a value to bin.
    var column_nan_treatment = List[Int]()
    for _ in range(n_columns):  # small-loop(n_columns: column plan entries): one AsIs code per column, no data
        column_nan_treatment.append(NAN_TREATMENT_AS_IS)
    for f in range(n_columns):
        comptime if SYM_CTR_ANY:
            if len(pre_has) == n_columns and pre_has[f]:
                fold_counts.append(pre_folds[f])
                borders.append(pre_borders[f].copy())
                continue
        var flagged = column_one_hot[f]
        if flagged:
            var maxc = onehot_column_max_code(ctx, cps[f], n_rows, f)
            if maxc > 254:
                raise Error("one-hot feature " + String(f)
                            + " has more than 255 categories")
            # synthetic integer 'borders' 0.5, 1.5, ... so the SAME
            # quantize kernel maps code k to bin k
            var bs = List[Float32]()
            for c in range(maxc):  # small-loop(maxc: one-hot categories, at most 254): synthetic borders, refused above 254 just before
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
            var bs = _ctr_borders_device(
                ctx,
                dep_dev_by_perm[0][dep_ordinal_of_column[f]],
                n_rows, ctr_grids[column_ctr_grid[f]],
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
            if len(column_dev_slot) != n_columns or column_dev_slot[f] < 0:
                raise Error(
                    "CTR column " + String(f) + " has no resident device"
                    " buffer"
                )
            var bs = _ctr_borders_device(
                ctx, ctr_dev_cols[column_dev_slot[f]], n_rows,
                ctr_grids[column_ctr_grid[f]],
            )
            fold_counts.append(len(bs))
            borders.append(bs^)
        else:
            # `CalcQuantization` (`libs/data/quantization.cpp:300-346`):
            # the column decides its own NaN mode, and a column that has
            # NaNs spends ONE of `border_count` on the sentinel. The grid
            # was built on the device above (`device_float_borders`).
            column_nan_treatment[f] = nan_value_treatment(dev_modes[dev_k])
            fold_counts.append(len(dev_borders[dev_k]))
            borders.append(dev_borders[dev_k].copy())
            dev_k += 1

    # one-hot features occupy folds+? -- for ordered features CatBoost's
    # fold count IS the border count; a one-hot feature has k categories
    # = k bins reached by k-1 synthetic borders, and its fold count must
    # cover bin k-1 for the equality candidates, hence len+1 above.

    return (borders^, fold_counts^, column_nan_treatment^)


def train_ordered_rmse(
    ctx: DeviceContext,
    x_colmajor: List[Float32], y: List[Float32],
    n_rows: Int, n_features: Int, permutation: List[UInt32],
    n_estimators: Int = 100, max_depth: Int = 6,
    border_count: Int = 128,
    learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    sample_weight: List[Float32] = List[Float32](),
) raises -> TrainedModel:
    """Train the supported numeric RMSE Ordered arm, with explicit ordering.

    One GPU, one permutation, finite numeric columns, zero starting point,
    Newton-1 leaves, no bootstrap or random score noise. Returns the usual
    serializable/predictable model. Other objectives, categorical CTRs,
    missing values and early stopping use distinct APIs; this entry does
    not silently interpret those options as supported Ordered features.
    """
    if n_rows != len(y) or n_features < 1 or len(x_colmajor) != n_rows * n_features:
        raise Error("train_ordered_rmse input shape mismatch")
    # lane/cpu3-gbdt-b: ONE route, the resident one. X goes up once, is
    # refused for NaN / infinity on the device and quantized from the
    # resident buffer (`train_ordered_rmse_ptr`); the host finiteness walk,
    # the host grid arm and the per-column host staging are gone. Same
    # grid, same kernels: the bits this list route returned under
    # `IDN_ORDERED_RMSE_DEVICE_GRID`.
    var model = train_ordered_rmse_ptr(
        ctx, Int(x_colmajor.unsafe_ptr()), y, n_rows, n_features, permutation,
        n_estimators, max_depth, border_count, learning_rate, l2_leaf_reg,
        sample_weight,
    )
    _ = len(x_colmajor)
    return model^


def _build_cindex_from_device_floats(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    n_rows: Int,
    borders: List[List[Float32]],
    fold_counts: List[Int],
) raises -> DeviceBuffer[DType.uint32]:
    """cpu2-l6-bindings: `_build_cindex_from_floats` for a column-major X
    already RESIDENT on the device and already refused for NaN / infinity
    there (every column `AsIs`). The same kernel quantizes the same words
    against the same borders, so the compressed index is the same bits; the
    per-column host staging (a host NaN walk and a memcpy of every column)
    is gone. The borders (k-sized) go up once, one 256-float slab a
    feature, in `_build_cindex_from_floats`' slab layout."""
    var n_features = len(borders)
    if len(fold_counts) != n_features:
        raise Error(
            "fold_counts has " + String(len(fold_counts)) + " entries for "
            + String(n_features) + " feature border lists"
        )
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    enqueue_fill(ctx, cindex, UInt32(0))
    comptime SLAB = 256
    var hb = ctx.enqueue_create_host_buffer[DType.float32](max(1, n_features) * SLAB)
    var db = ctx.enqueue_create_buffer[DType.float32](max(1, n_features) * SLAB)
    ctx.synchronize()
    for f in range(n_features):  # small-loop(n_features: per-feature border slabs): packs the fitted borders, at most 255 a feature, for one upload; no row data
        if len(borders[f]) + 1 > SLAB:
            raise Error("_build_cindex_from_device_floats: more than 255 borders")
        hb.unsafe_ptr().unsafe_store(f * SLAB, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hb.unsafe_ptr().unsafe_store(f * SLAB + 1 + b, borders[f][b])
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
    for f in range(n_features):
        if len(borders[f]) == 0:
            continue
        ref cf = lay.features[f]
        ctx.enqueue_function[binarize_float_feature_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            d_x.unsafe_ptr() + f * n_rows, Int32(n_rows),
            db.unsafe_ptr() + f * SLAB, cindex.unsafe_ptr(),
            grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
            block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
        )
    ctx.synchronize()
    _ = hb^
    _ = db^
    return cindex^


def train_ordered_rmse_ptr(
    ctx: DeviceContext,
    x_addr: Int, y: List[Float32],
    n_rows: Int, n_features: Int, permutation: List[UInt32],
    n_estimators: Int = 100, max_depth: Int = 6,
    border_count: Int = 128,
    learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    sample_weight: List[Float32] = List[Float32](),
) raises -> TrainedModel:
    """cpu2-l6-bindings: `train_ordered_rmse` with the column-major X read
    from the caller's address. X goes to the device ONCE, is refused for
    NaN / infinity there (`device_first_nonfinite`, one Int32 per block read
    back) and is quantized from the resident buffer; no host List copy of
    X, no host finiteness walk, no per-column host staging. The grid comes
    from the same device border build over the caller's columns, so the
    model is the list route's bits. lane/cpu3-gbdt-b: this is now the ONLY
    route (`train_ordered_rmse` delegates here); the host-grid `_OFF` arm is
    gone from the GPU path, in every mode."""
    from gbdt.methods.dynamic_boosting import fit_ordered_rmse

    if x_addr == 0:
        raise Error("train_ordered_rmse: null X address")
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_addr)
    if n_rows != len(y) or n_features < 1 or n_rows < 1:
        raise Error("train_ordered_rmse input shape mismatch")
    if border_count < 1 or border_count > 255:
        raise Error("train_ordered_rmse supports 1..255 borders")
    var n_cells = n_rows * n_features
    var d_x = ctx.enqueue_create_buffer[DType.float32](n_cells)
    ctx.enqueue_copy(
        dst_buf=d_x,
        src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=x_addr),
    )
    if device_first_nonfinite(ctx, d_x, n_cells) >= 0:
        raise Error("train_ordered_rmse requires finite numeric features")
    var grid_cols = List[MutPointer[Float32, MutUntrackedOrigin]](
        capacity=n_features
    )
    for f in range(n_features):  # small-loop(n_features: column pointers): one pointer per column for the border build, no element read
        grid_cols.append(xp + f * n_rows)
    var got = device_float_borders(
        ctx, grid_cols, n_rows, n_rows, border_count, NAN_MODE_FORBIDDEN,
        generate_seed_for_borders(UInt64(0)),
    )
    var dev_grids = got[0].copy()
    if len(dev_grids) != n_features:
        raise Error(
            "train_ordered_rmse: the device border build returned "
            + String(len(dev_grids)) + " grids for "
            + String(n_features) + " features"
        )
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    var one_hot = List[Bool]()
    var nan_treatment = List[Int]()
    for f in range(n_features):  # small-loop(n_features: model plan entries): per-feature border list, fold count and flags of the fitted grid, no row data
        fold_counts.append(len(dev_grids[f]))
        borders.append(dev_grids[f].copy())
        one_hot.append(False)
        nan_treatment.append(NAN_TREATMENT_AS_IS)
    var layout = build_layout(fold_counts)
    var cindex = _build_cindex_from_device_floats(
        ctx, d_x, n_rows, borders, fold_counts
    )
    _ = d_x^
    var result = fit_ordered_rmse(
        ctx, layout, cindex, y, sample_weight, permutation,
        n_estimators, max_depth,
        ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        learning_rate, l2_leaf_reg,
    )
    var exported_model = result.model.copy()
    return TrainedModel(
        exported_model^, fold_counts^, one_hot^, borders^, nan_treatment^,
        List[Float64](), List[Float64](), -1, False, 0,
        List[TCtrValueTable](), TTensorCtrRegistry(n_features),
    )


def model_input_features(tm: TrainedModel) raises -> Int:
    """How many RAW input columns `predict_floats` expects for this model.

    For a float-only or one-hot model that is just the column count. For a
    model with CTR tables it is fewer, because one categorical input stands
    behind one column per CTR config -- the shape their
    `TStaticCtrProvider` reconstructs from the model rather than being
    told."""
    if len(tm.tensor_ctr_registry.features) != 0:
        return tm.tensor_ctr_registry.first_model_column
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
    if len(tm.tensor_ctr_registry.features) != 0 and len(tm.ctr_tables) != 0:
        raise Error(
            "combined simple-CTR and tensor-CTR model apply needs a composed"
            " column plan and is not wired yet"
        )
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
    if len(tm.tensor_ctr_registry.features) != 0:
        expanded = tm.tensor_ctr_registry.expand_for_model_apply(
            x_colmajor, n_rows, tm.borders, tm.one_hot
        )
    elif len(tm.ctr_tables) == 0:
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
    # the cursor straight into the returned list (lane/cpu3-gbdt-b): no
    # host staging buffer, no per-row host copy
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    if n_rows > 0:
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=cursor)
    ctx.synchronize()
    _ = cursor^  # past the drain (step-33 race class, device side)
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
    gauge the whole implementation trains in. A caller turning these into
    probabilities appends a zero and softmaxes over all `numClasses`;
    `multiclass_probabilities` does exactly that.

    RAW APPROXES, like every other predict here and like their `predict`
    without a `prediction_type`.
    """
    var approx_dim = model_approx_dim(tm.model)
    var out = List[Float32](length=n_rows * approx_dim, fill=Float32(0.0))
    _ = predict_multi_linked_into(
        ctx, tm, x_colmajor, n_rows, LINK_RAW,
        rebind[MutPointer[Float32, MutUntrackedOrigin]](out.unsafe_ptr()),
    )
    return out^


def predict_multi_linked_into(
    ctx: DeviceContext,
    tm: TrainedModel,
    x_colmajor: List[Float32],
    n_rows: Int,
    link_mode: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """lane/cpu3-gbdt-b: `predict_multi_floats` with the row-major reshape
    and the probability link ON THE DEVICE, written straight into `dst`.
    `link_mode` is `LINK_RAW` (`approx_dim` wide, the reshape the host loop
    did), `LINK_SOFTMAX` (`approx_dim + 1` wide, `multiclass_probabilities`'
    words) or `LINK_SIGMOID` (`approx_dim` wide, `one_vs_all_probabilities`'
    words); `resident_link_kernel` computes them in soft binary64 exactly as
    the host functions do, so the bits are theirs. Returns the width."""
    var approx_dim = model_approx_dim(tm.model)
    var width = approx_dim
    if link_mode == LINK_SOFTMAX:
        width = approx_dim + 1
    elif link_mode != LINK_RAW and link_mode != LINK_SIGMOID:
        raise Error(
            "predict_multi_linked_into: link mode " + String(link_mode)
            + " is not RAW, SOFTMAX or SIGMOID"
        )
    var expanded_x: List[Float32]
    if len(tm.tensor_ctr_registry.features) != 0 and len(tm.ctr_tables) != 0:
        raise Error(
            "combined simple-CTR and tensor-CTR model apply needs a composed"
            " column plan and is not wired yet"
        )
    if len(tm.tensor_ctr_registry.features) != 0:
        expanded_x = tm.tensor_ctr_registry.expand_for_model_apply(
            x_colmajor, n_rows, tm.borders, tm.one_hot
        )
    elif len(tm.ctr_tables) != 0:
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
        max(1, approx_dim * n_rows)
    )
    predict(
        tm.model, ctx, n_rows, tm.fold_counts, cindex, cursor,
        one_hot=tm.one_hot,
    )
    if n_rows <= 0:
        ctx.synchronize()
        return width
    var d_out = ctx.enqueue_create_buffer[DType.float32](n_rows * width)
    var d_u64 = ctx.enqueue_create_buffer[DType.uint64](1)
    var d_i64 = ctx.enqueue_create_buffer[DType.int64](1)
    var link_blocks = min((n_rows + LINK_BLOCK - 1) // LINK_BLOCK, 65535)
    ctx.enqueue_function[resident_link_kernel](
        cursor.unsafe_ptr(),
        d_out.unsafe_ptr(),
        d_u64.unsafe_ptr(),
        d_i64.unsafe_ptr(),
        Int32(n_rows), Int32(approx_dim), Int32(link_mode),
        Int32(0), Int32(0),
        grid_dim=link_blocks, block_dim=LINK_BLOCK,
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_out)
    ctx.synchronize()
    # past the drain (step-33 race class)
    _ = cursor^
    _ = d_out^
    _ = d_u64^
    _ = d_i64^
    _ = len(expanded_x)
    return width


def one_vs_all_probabilities_reference(
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
        # lane hr2-gbdt-host: soft binary64, `resident_link_kernel`'s words
        out.append(_hr2_ftz(sf64_to_f32(sf64_sigmoid_f32(approxes[i]))))
    return out^


def multiclass_probabilities_reference(
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
        # lane hr2-gbdt-host: soft binary64, `resident_link_kernel`'s words
        var mx = SF64_ZERO
        for k in range(eff):
            var v = sf64_from_f32(approxes[r * eff + k])
            if sf64_gt(v, mx):
                mx = v
        var se = SF64_ZERO
        for k in range(eff):
            se = sf64_add(se, sf64_exp(sf64_sub(sf64_from_f32(approxes[r * eff + k]), mx)))
        var e_pin = sf64_exp(sf64_neg(mx))
        se = sf64_add(se, e_pin)
        for k in range(eff):
            out.append(_hr2_ftz(sf64_to_f32(sf64_div(
                sf64_exp(sf64_sub(sf64_from_f32(approxes[r * eff + k]), mx)), se
            ))))
        out.append(_hr2_ftz(sf64_to_f32(sf64_div(e_pin, se))))
    return out^
