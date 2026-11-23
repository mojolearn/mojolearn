# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the pointwise losses other than
Logloss and RMSE on symmetric trees, with the Exact leaf estimator and the
row bootstraps (lane/cpu-training-gbdt-losses, 2026-09-15): the
gbdt-parametric-losses and gbdt-exact-mae lanes of tools/identity_break.py.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
the GPU-free host modules the symmetric oracle already reuses, CatBoost's
GPU random stream (`gbdt/gpu_util/kernel/random_gen.mojo`, plain integer and
`checks/numerics` arithmetic with no kernel in it) and the symmetric oracle
itself, whose restatements of the grid, the histograms, the partition stats,
the Cosine score, the lane folds, the scale, the walker pieces and the model
text this fit reaches unchanged.

THE CONFIGURATIONS THIS COVERS, by name. `gbdt-parametric-losses`: eight
depth-4 trees each of Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq
(q 3), Expectile (alpha 0.3), Tweedie (variance power 1.5), Huber (delta 1)
and CrossEntropy, every other option at its default. `gbdt-exact-mae`: 20
depth-6 trees of MAE with `leaf_estimation_method="Exact"` and the Poisson
bootstrap at subsample 0.6. Bayesian bootstrap and score noise also reach
this oracle. QueryRMSE, PairLogit and YetiRank form their grouped gradients
first, then apply the same row bootstrap to both statistic planes, matching
CatBoost's greedy searcher and querywise_targets_impl.h::StochasticDer.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build, the
default flags the symmetric oracle names)

  1. `train`'s loss resolution (`gbdt/train.mojo:1560-1600`,
     `gbdt/options/loss_description.mojo:110-169`,
     `gbdt/options/catboost_options.mojo:1247-1444`): the kernel alpha, the
     estimator alpha, the leaf method and its iteration count, and
     `boost_from_average` unset resolving False for every loss here. The
     binding resolves them; `GbdtHostLoss` carries the result.
  2. The grid, the layout and the binarize: the symmetric oracle's.
  3. Per tree, the search pass `launch_approximate[False]`
     (`doc_parallel_boosting.mojo:1544-1552`): `pointwise_target_kernel
     [objective, False, False]` (`pointwise_targets.mojo:421-663`, the
     `target_score` / `target_der` / `target_der2` arms at `:203-413`
     restated below term for term) or `cross_entropy_kernel[False]` for
     CrossEntropy (`:866-1054`).
  4. Under a bootstrap, `launch_bootstrap` (`gbdt/gpu_util/kernel/
     bootstrap.mojo:88-311`): the splitmix64 seed fill, one draw per row
     per thread in the grid stride walk, both planes multiplied, the
     magnitudes reduced per block and folded, and the seeds carried from
     tree to tree.
  5. `run_tree_layout_traced` with `need_estimation` set, every level: a
     TWIN of the loop in `gbdt_oracle.mojo::gbdt_host_fit` (an edit to
     either must visit the other).
  6. The estimation task (`doc_parallel_boosting.mojo:622-850`) over
     `BinOptimizedOracle` (`pointwise_oracle.mojo`): for Newton and
     Gradient the walker with AnyImprovement over
     `write_value_and_first_derivatives`' single-dimensional arm (the
     estimation planes `ftz(w * Der)`, `ftz(w * Der2)`, the Float32 value
     fold) and `write_second_derivatives` (the cached Hessian plus lambda,
     or the leaf weight plus lambda under Gradient); for Exact
     `estimate_exact` (`:433-503`): `compute_exact_value_kernel`, the
     segmented radix sort on bits [10, 32) of the sortable key, the
     end-of-bins flags, the segmented inclusive scan in its three phases at
     the 768 block, the need-weights reduce at the 1024 block, the
     sixteen-step binary search, and the MAPE weight quotient
     (`leaves_estimation_helper.mojo`, `exact_estimation.mojo`,
     `segmented_sort.mojo`, `segmented_scan.mojo`).
  7. The cursor update through `identical_mul_add`, the rescale on append,
     the learn losses and the model text.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`
(`GBDT_ORACLE_HOST_SABOTAGE`) adds 1.0 to the walker's lambda, as the
symmetric oracle does, and one quarter to every Exact leaf, so every loss's
leaves move.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the two lanes is the measurement.
"""
from std.math import exp, fma, isfinite, log
from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
)
from core.host_parallel import host_parallelize
from gbdt.targets.pointwise_bodies import (
    BODY_OBJ_EXPECTILE,
    BODY_OBJ_HUBER,
    BODY_OBJ_LOGLINQUANTILE,
    BODY_OBJ_LQ,
    BODY_OBJ_MAE,
    BODY_OBJ_MAPE,
    BODY_OBJ_POISSON,
    BODY_OBJ_QUANTILE,
    BODY_OBJ_RMSE,
    BODY_OBJ_TWEEDIE,
    cross_entropy_row,
    cross_entropy_score,
    target_der,
    target_der2,
    target_score,
    weighted_der,
    weighted_der2,
    weighted_score,
)
from core.host_predict_threads import HostF32Ptr, host_predict_chunk
from gbdt.methods.cursor_rescale import rescaled_leaves
from gbdt.methods.random_score_bodies import score_model_length_mult
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.gpu_util.kernel.random_gen import next_poisson_f, next_uniform_f
from gbdt.data.pairs import (
    IDN_PAIRLOGIT_GROUP,
    PairList,
    generate_pairs,
    order_pairs_by_winner,
)
from gbdt.host.gbdt_oracle_pair import (
    HostPairs,
    host_pair_groups,
    host_pairs,
    pair_logit_eval,
    pair_logit_search_pass,
    pair_logit_value,
)
from gbdt.host.gbdt_oracle_query import (
    query_rmse_eval,
    query_rmse_search_pass,
    query_rmse_value,
)
from gbdt.host.gbdt_oracle_yeti import yeti_rank_eval, yeti_rank_search_pass
from gbdt.data.yeti_rank_tasks import YetiRankTasks, yeti_rank_tasks
from gbdt.data.permutation import TRandom
from gbdt.host.gbdt_oracle import (
    apply_scaled_leaves_host,
    _identity_rows,
    _split_level,
    _scan_histograms,
    _subtract_histograms,
    _partition_stats_level,
    GBDT_PARTITION_CHUNKS,
    GBDT_BOOT_BAYESIAN,
    GBDT_BOOT_BERNOULLI,
    GBDT_BOOT_BLOCK,
    GBDT_BOOT_POISSON,
    GBDT_BOOT_SEEDS,
    _bootstrap_pass,
    _snap_gradients,
    _target_std_dev,
    gbdt_bootstrap_seeds,
    GBDT_FLOAT32_MAX,
    GBDT_MSE_BLOCK,
    GBDT_PINNED_SM,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_SENTINEL,
    GbdtHostFitWithEval,
    GbdtHostModel,
    GbdtHostParams,
    _binarize_columns,
    _ns_project_index,
    _sample_tree_folds,
    _choose_scale_from_magnitudes,
    _gain_winner,
    _gbdt_units,
    _gather_rows,
    _leaf_bins,
    GbdtIntPtr,
    _deterministic_sum_lanes,
    _diagonal_direction,
    _binary_block,
    _half_byte_block,
    _halving_fold,
    _pinned_partition_stats,
    _one_byte_block,
    _oracle_move_to,
    _regularize,
    _second_derivatives,
    _walker_move,
    _greedy_gains,
    gbdt_host_grid,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    is_second_order_score_function,
)
from gbdt.host.gbdt_oracle_eval import (
    GbdtHostEval,
    GbdtHostEvalFit,
    gbdt_eval_add_tree,
    gbdt_eval_detector,
    gbdt_eval_shrink_point,
    gbdt_eval_test_bins,
)


#: The objective codes (`gbdt/targets/kernel/pointwise_targets.mojo`'s
#: `OBJECTIVE_*`; the pointwise ones are the body module's `BODY_OBJ_*`).
comptime GBDT_OBJ_RMSE = BODY_OBJ_RMSE
comptime GBDT_OBJ_LOGLOSS = 1
comptime GBDT_OBJ_CROSSENTROPY = 2
comptime GBDT_OBJ_QUANTILE = BODY_OBJ_QUANTILE
comptime GBDT_OBJ_MAE = BODY_OBJ_MAE
comptime GBDT_OBJ_LOGLINQUANTILE = BODY_OBJ_LOGLINQUANTILE
comptime GBDT_OBJ_MAPE = BODY_OBJ_MAPE
comptime GBDT_OBJ_POISSON = BODY_OBJ_POISSON
comptime GBDT_OBJ_LQ = BODY_OBJ_LQ
comptime GBDT_OBJ_EXPECTILE = BODY_OBJ_EXPECTILE
comptime GBDT_OBJ_TWEEDIE = BODY_OBJ_TWEEDIE
comptime GBDT_OBJ_HUBER = BODY_OBJ_HUBER
#: `OBJECTIVE_QUERY_RMSE` (`pointwise_targets.mojo`), the querywise target of
#: `gbdt/host/gbdt_oracle_query.mojo`
comptime GBDT_OBJ_QUERY_RMSE = 14
#: `OBJECTIVE_PAIR_LOGIT` (`pointwise_targets.mojo`), the pairwise-derivative
#: querywise target of `gbdt/host/gbdt_oracle_pair.mojo`
comptime GBDT_OBJ_PAIR_LOGIT = 15
#: `OBJECTIVE_YETI_RANK` (`pointwise_targets.mojo`), the sampled-permutation
#: querywise target of `gbdt/host/gbdt_oracle_yeti.mojo`
comptime GBDT_OBJ_YETI_RANK = 16

#: `LEAF_ESTIMATION_*` (`gbdt/options/catboost_options.mojo:269-271`).
comptime GBDT_LEAF_GRADIENT = 0
comptime GBDT_LEAF_NEWTON = 1
comptime GBDT_LEAF_EXACT = 2


#: `NEED_WEIGHTS_BLOCK`, `BINARY_SEARCH_ITERATIONS` (`exact_estimation.mojo`),
#: `EXACT_SORT_FIRST_BIT` (`leaves_estimation_helper.mojo`), and
#: `SEG_SCAN_BLOCK` (`segmented_scan.mojo`: 768 on every declared column).
comptime GBDT_NEED_WEIGHTS_BLOCK = 1024
comptime GBDT_QUANTILE_ITERATIONS = 16
comptime GBDT_EXACT_FIRST_BIT = 10
comptime GBDT_SEG_SCAN_BLOCK = 768


@fieldwise_init
struct GbdtHostLoss(ImplicitlyCopyable, Movable):
    """The loss as `train` resolves it (see the module docstring, item 1)."""

    var objective: Int
    var kernel_alpha: Float32
    var estimator_alpha: Float32
    var method: Int
    var iterations: Int
    var bootstrap_kind: Int
    var bootstrap_param: Float32
    #: Logloss's target border (`GetLogLossBorder`); unread by the others
    var border: Float32


# THE OBJECTIVES: `gbdt/targets/pointwise_bodies.mojo`, the bodies the device
# kernels call (lane gbdt-gaps, 2026-11-23: this file carried its own copy of
# `target_score` / `target_der` / `target_der2` until then). The host picks the
# objective at run time, so each body is reached through one switch here.


def _target_score(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_score[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return target_score[BODY_OBJ_RMSE](t, p, alpha)
    elif objective == GBDT_OBJ_QUANTILE:
        return target_score[BODY_OBJ_QUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAE:
        return target_score[BODY_OBJ_MAE](t, p, alpha)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return target_score[BODY_OBJ_LOGLINQUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAPE:
        return target_score[BODY_OBJ_MAPE](t, p, alpha)
    elif objective == GBDT_OBJ_POISSON:
        return target_score[BODY_OBJ_POISSON](t, p, alpha)
    elif objective == GBDT_OBJ_LQ:
        return target_score[BODY_OBJ_LQ](t, p, alpha)
    elif objective == GBDT_OBJ_EXPECTILE:
        return target_score[BODY_OBJ_EXPECTILE](t, p, alpha)
    elif objective == GBDT_OBJ_TWEEDIE:
        return target_score[BODY_OBJ_TWEEDIE](t, p, alpha)
    elif objective == GBDT_OBJ_HUBER:
        return target_score[BODY_OBJ_HUBER](t, p, alpha)
    return Float32(0.0)


def _target_der(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_der[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return target_der[BODY_OBJ_RMSE](t, p, alpha)
    elif objective == GBDT_OBJ_QUANTILE:
        return target_der[BODY_OBJ_QUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAE:
        return target_der[BODY_OBJ_MAE](t, p, alpha)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return target_der[BODY_OBJ_LOGLINQUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAPE:
        return target_der[BODY_OBJ_MAPE](t, p, alpha)
    elif objective == GBDT_OBJ_POISSON:
        return target_der[BODY_OBJ_POISSON](t, p, alpha)
    elif objective == GBDT_OBJ_LQ:
        return target_der[BODY_OBJ_LQ](t, p, alpha)
    elif objective == GBDT_OBJ_EXPECTILE:
        return target_der[BODY_OBJ_EXPECTILE](t, p, alpha)
    elif objective == GBDT_OBJ_TWEEDIE:
        return target_der[BODY_OBJ_TWEEDIE](t, p, alpha)
    elif objective == GBDT_OBJ_HUBER:
        return target_der[BODY_OBJ_HUBER](t, p, alpha)
    return Float32(0.0)


def _target_der2(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_der2[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return target_der2[BODY_OBJ_RMSE](t, p, alpha)
    elif objective == GBDT_OBJ_QUANTILE:
        return target_der2[BODY_OBJ_QUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAE:
        return target_der2[BODY_OBJ_MAE](t, p, alpha)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return target_der2[BODY_OBJ_LOGLINQUANTILE](t, p, alpha)
    elif objective == GBDT_OBJ_MAPE:
        return target_der2[BODY_OBJ_MAPE](t, p, alpha)
    elif objective == GBDT_OBJ_POISSON:
        return target_der2[BODY_OBJ_POISSON](t, p, alpha)
    elif objective == GBDT_OBJ_LQ:
        return target_der2[BODY_OBJ_LQ](t, p, alpha)
    elif objective == GBDT_OBJ_EXPECTILE:
        return target_der2[BODY_OBJ_EXPECTILE](t, p, alpha)
    elif objective == GBDT_OBJ_TWEEDIE:
        return target_der2[BODY_OBJ_TWEEDIE](t, p, alpha)
    elif objective == GBDT_OBJ_HUBER:
        return target_der2[BODY_OBJ_HUBER](t, p, alpha)
    return Float32(0.0)


def _weighted_der(objective: Int, t: Float32, p: Float32, alpha: Float32, weight: Float32) -> Float32:
    """`weighted_der[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return weighted_der[BODY_OBJ_RMSE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_QUANTILE:
        return weighted_der[BODY_OBJ_QUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAE:
        return weighted_der[BODY_OBJ_MAE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return weighted_der[BODY_OBJ_LOGLINQUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAPE:
        return weighted_der[BODY_OBJ_MAPE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_POISSON:
        return weighted_der[BODY_OBJ_POISSON](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LQ:
        return weighted_der[BODY_OBJ_LQ](t, p, alpha, weight)
    elif objective == GBDT_OBJ_EXPECTILE:
        return weighted_der[BODY_OBJ_EXPECTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_TWEEDIE:
        return weighted_der[BODY_OBJ_TWEEDIE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_HUBER:
        return weighted_der[BODY_OBJ_HUBER](t, p, alpha, weight)
    return Float32(0.0)


def _weighted_der2(objective: Int, t: Float32, p: Float32, alpha: Float32, weight: Float32) -> Float32:
    """`weighted_der2[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return weighted_der2[BODY_OBJ_RMSE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_QUANTILE:
        return weighted_der2[BODY_OBJ_QUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAE:
        return weighted_der2[BODY_OBJ_MAE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return weighted_der2[BODY_OBJ_LOGLINQUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAPE:
        return weighted_der2[BODY_OBJ_MAPE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_POISSON:
        return weighted_der2[BODY_OBJ_POISSON](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LQ:
        return weighted_der2[BODY_OBJ_LQ](t, p, alpha, weight)
    elif objective == GBDT_OBJ_EXPECTILE:
        return weighted_der2[BODY_OBJ_EXPECTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_TWEEDIE:
        return weighted_der2[BODY_OBJ_TWEEDIE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_HUBER:
        return weighted_der2[BODY_OBJ_HUBER](t, p, alpha, weight)
    return Float32(0.0)


def _weighted_score(objective: Int, t: Float32, p: Float32, alpha: Float32, weight: Float32) -> Float32:
    """`weighted_score[objective]` for a run-time objective."""
    if objective == GBDT_OBJ_RMSE:
        return weighted_score[BODY_OBJ_RMSE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_QUANTILE:
        return weighted_score[BODY_OBJ_QUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAE:
        return weighted_score[BODY_OBJ_MAE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        return weighted_score[BODY_OBJ_LOGLINQUANTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_MAPE:
        return weighted_score[BODY_OBJ_MAPE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_POISSON:
        return weighted_score[BODY_OBJ_POISSON](t, p, alpha, weight)
    elif objective == GBDT_OBJ_LQ:
        return weighted_score[BODY_OBJ_LQ](t, p, alpha, weight)
    elif objective == GBDT_OBJ_EXPECTILE:
        return weighted_score[BODY_OBJ_EXPECTILE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_TWEEDIE:
        return weighted_score[BODY_OBJ_TWEEDIE](t, p, alpha, weight)
    elif objective == GBDT_OBJ_HUBER:
        return weighted_score[BODY_OBJ_HUBER](t, p, alpha, weight)
    return Float32(0.0)


@fieldwise_init
struct _LossRow(ImplicitlyCopyable, Movable):
    #: the search plane 1 as stored, `ftz(w * Der)`
    var der: Float32
    #: `|plane 1|` as the magnitude reduce reads it
    var der_abs: Float32
    #: the estimation plane 1, `ftz(w * Der2)`
    var der2: Float32
    #: the score partial's element
    var score: Float32


def _loss_row(
    objective: Int, t: Float32, p: Float32, alpha: Float32,
    weight: Float32 = Float32(1.0),
) -> _LossRow:
    """One in-range thread of the kernel the objective reaches at the row's
    weight (unit unless the fit carries row weights):
    `pointwise_target_kernel` (`:553-663`) or, for CrossEntropy and Logloss,
    `cross_entropy_kernel[has_border]` (`:947-1054`; `alpha` carries
    Logloss's border)."""
    if objective == GBDT_OBJ_CROSSENTROPY or objective == GBDT_OBJ_LOGLOSS:
        # the device kernel's own body (`pointwise_bodies.cross_entropy_row`
        # / `cross_entropy_score`, lane gbdt-one-body 2026-11-23)
        var row = (
            cross_entropy_row[True](t, p, alpha) if objective == GBDT_OBJ_LOGLOSS
            else cross_entropy_row[False](t, p, alpha)
        )
        return _LossRow(
            ftz(weight * row.direction), abs(weight * row.direction),
            ftz(weight * row.scale), cross_entropy_score(row, p, weight),
        )
    var der = _weighted_der(objective, t, p, alpha, weight)
    return _LossRow(
        der, abs(der), _weighted_der2(objective, t, p, alpha, weight),
        _weighted_score(objective, t, p, alpha, weight),
    )


def _loss_search_pass(
    loss: GbdtHostLoss,
    targets: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
    weights: List[Float32] = List[Float32](),
    second_order: Bool = False,
):
    """`launch_approximate[False]` with `compute_fv` and
    `compute_magnitudes` set: plane 0 the row's weight, plane 1 the flushed
    weighted derivative, one score partial and two magnitude partials per
    256-thread block through the halving tree. `weights` empty is the unit
    weight, otherwise the kernels' `has_weights` arm (sample weights).

    `second_order` is `launch_approximate[False, True]`
    (`secondDerAsWeights`, the NewtonL2 / NewtonCosine search planes): plane
    0 is then the flushed weighted second derivative, and its magnitude
    bounds that plane as stored (`pointwise_target_kernel`'s `plane0`). Lane
    gbdt-rest-3 (2026-11-23): the pointwise searcher's own copy of this pass
    (`gbdt_oracle_pointwise.mojo::_pw_search_pass`) is deleted."""
    # THE SPLIT (lane gbdt-host-par), the unit of `_logloss_search_pass`:
    # one 256-thread device block owns its rows' two planes and its three
    # partials, and its halving trees are the kernel's. Blocks go round
    # robin to `_gbdt_units` tasks; a task holds three block slabs.
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var units = _gbdt_units(blocks, 16 * n_rows)
    var objective = loss.objective
    var alpha = loss.kernel_alpha
    var has_weights = len(weights) > 0
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    # an empty `weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())
    var mag_p = rebind[HostF32Ptr](mag_partials.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm objective, imm alpha, imm has_weights, imm second_order, imm targets_p, imm cursor_p, imm weights_p, imm stats_p, imm fv_p, imm mag_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var weight = Float32(1.0)
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slabs are reused)
                s_score[t] = Float32(0.0)
                s_w[t] = Float32(0.0)
                s_g[t] = Float32(0.0)
                if i < n_rows:
                    if has_weights:
                        weight = weights_p[i]
                    var r = _loss_row(objective, targets_p[i], cursor_p[i], alpha, weight)
                    var plane0 = weight
                    if second_order:
                        plane0 = r.der2
                    stats_p[i] = plane0
                    stats_p[n_rows + i] = r.der
                    s_score[t] = r.score
                    s_w[t] = abs(plane0)
                    s_g[t] = r.der_abs
            fv_p[b] = _halving_fold(s_score)
            mag_p[2 * b] = _halving_fold(s_w)
            mag_p[2 * b + 1] = _halving_fold(s_g)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(targets)
    _ = len(cursor)
    _ = len(weights)
    _ = len(stats)
    _ = len(fv_partials)
    _ = len(mag_partials)


def _loss_value(
    loss: GbdtHostLoss, targets: List[Float32], cursor: List[Float32], n_rows: Int,
    weights: List[Float32] = List[Float32](),
) -> Float32:
    """The final learn loss pass (each row at its weight; unit when
    `weights` is empty), folded by `deterministic_sum_lanes_kernel[1]`."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_partials = List[Float32](length=blocks, fill=Float32(0.0))
    # THE SPLIT (lane gbdt-host-par): one 256-thread device block per unit
    # (its own halving tree), round robin to `_gbdt_units` tasks; the lane
    # fold over the blocks runs after the join in block order.
    var units = _gbdt_units(blocks, 16 * n_rows)
    var objective = loss.objective
    var alpha = loss.kernel_alpha
    var has_weights = len(weights) > 0
    var targets_p = rebind[HostF32Ptr](targets.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](cursor.unsafe_ptr())
    # an empty `weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](weights.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv_partials.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm objective, imm alpha, imm has_weights, imm targets_p, imm cursor_p, imm weights_p, imm fv_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var weight = Float32(1.0)
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slab is reused)
                s_score[t] = Float32(0.0)
                if i < n_rows:
                    if has_weights:
                        weight = weights_p[i]
                    s_score[t] = _loss_row(
                        objective, targets_p[i], cursor_p[i], alpha, weight
                    ).score
            fv_p[b] = _halving_fold(s_score)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(targets)
    _ = len(cursor)
    _ = len(weights)
    return _deterministic_sum_lanes(fv_partials, 1, blocks)[0]


# ===========================================================================
# THE LEAF ESTIMATORS
# ===========================================================================


def _loss_eval(
    loss: GbdtHostLoss,
    g_target: List[Float32],
    g_cursor: List[Float32],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    lambda_reg: Float64,
    mut value: Float64,
    mut gradient: List[Float64],
    mut cached_der2: List[Float64],
    g_weights: List[Float32] = List[Float32](),
):
    """`write_value_and_first_derivatives`' single-dimensional arm
    (`pointwise_oracle.mojo:458-573`): the estimation planes, one score
    partial per 256-thread block, `compute_partition_stats` per leaf, the
    Hessian plus lambda, and the HOST Float32 fold of the value partials.
    `g_weights` (empty: unit) are the row weights gathered in bin order,
    as the targets and the cursor are."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    # THE SPLIT (lane gbdt-host-par), the unit of `_logloss_search_pass`:
    # one 256-thread device block owns its rows' two estimation planes and
    # its score partial (its own halving tree); blocks go round robin to
    # `_gbdt_units` tasks. This pass runs once per walker evaluation.
    var units = _gbdt_units(blocks, 16 * n_rows)
    var objective = loss.objective
    var alpha = loss.kernel_alpha
    var has_weights = len(g_weights) > 0
    var targets_p = rebind[HostF32Ptr](g_target.unsafe_ptr())
    var cursor_p = rebind[HostF32Ptr](g_cursor.unsafe_ptr())
    # an empty `g_weights` is never read (`has_weights` guards it)
    var weights_p = rebind[HostF32Ptr](g_weights.unsafe_ptr())
    var stats_p = rebind[HostF32Ptr](stats.unsafe_ptr())
    var fv_p = rebind[HostF32Ptr](fv.unsafe_ptr())

    def score_blocks(unit: Int) {imm units, imm blocks, imm n_rows, imm objective, imm alpha, imm has_weights, imm targets_p, imm cursor_p, imm weights_p, imm stats_p, imm fv_p}:
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var weight = Float32(1.0)
        var b = unit
        while b < blocks:
            for t in range(GBDT_MSE_BLOCK):
                var i = b * GBDT_MSE_BLOCK + t
                # an out-of-range thread adds 0.0 (the slab is reused)
                s_score[t] = Float32(0.0)
                if i < n_rows:
                    if has_weights:
                        weight = weights_p[i]
                    var r = _loss_row(objective, targets_p[i], cursor_p[i], alpha, weight)
                    stats_p[i] = r.der
                    stats_p[n_rows + i] = r.der2
                    s_score[t] = r.score
            fv_p[b] = _halving_fold(s_score)
            b += units

    host_parallelize(score_blocks, units)
    _ = len(g_target)
    _ = len(g_cursor)
    _ = len(g_weights)
    gradient.clear()
    cached_der2.clear()
    # the leaves' two sums (`compute_partition_stats`): the (leaf, plane)
    # launches across the cores (lane gbdt-host-par)
    var leaf_sums = _partition_stats_level(
        stats, n_rows, 2, GBDT_PARTITION_CHUNKS, offsets, sizes, len(sizes)
    )
    for leaf in range(len(sizes)):
        gradient.append(Float64(leaf_sums[2 * leaf]))
        cached_der2.append(Float64(leaf_sums[2 * leaf + 1]) + lambda_reg)
    # lane cpu4-gbdt: the oracle's value fold is the device's
    # `deterministic_sum_lanes_kernel[1]` order (was an ascending chain)
    value = Float64(_deterministic_sum_lanes(fv, 1, blocks)[0])


def _walker_estimate(
    loss: GbdtHostLoss,
    g_target: List[Float32],
    mut g_cursor: List[Float32],
    bins: List[Int],
    offsets: List[Int],
    sizes: List[Int],
    weights_cpu: List[Float64],
    n_rows: Int,
    l2_leaf_reg: Float32,
    targets_rows: List[Float32] = List[Float32](),
    row_index: List[Int] = List[Int](),
    group_sizes: List[Int] = List[Int](),
    pairs: Optional[HostPairs] = None,
    yeti_tasks: Optional[YetiRankTasks] = None,
    yeti_seed: UInt64 = UInt64(0),
    g_weights: List[Float32] = List[Float32](),
    row_weights: List[Float32] = List[Float32](),
) raises -> List[Float32]:
    """`newton_like_walker_estimate` with AnyImprovement
    (`descent_helpers.mojo:198-285`, `step_estimator.mojo:50-72`), the TWIN
    of `gbdt_oracle.mojo::_estimate_leaves` with the method's second
    derivatives. `g_weights` are the row weights in bin order (the
    pointwise evaluation's) and `row_weights` the same weights in row order
    (YetiRank's and QueryRMSE's, which read their groups in row order);
    both empty at unit weight."""
    var n_leaves = len(sizes)
    var lambda_reg = Float64(l2_leaf_reg)
    var has_row_weights = len(row_weights) > 0
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        lambda_reg = lambda_reg + 1.0
    var iterations = loss.iterations
    var current_point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var cur_point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var cur_value = Float64(0.0)
    var cur_grad = List[Float64]()
    var cached_der2 = List[Float64]()
    # this tree's YetiRank evaluation stream (`pointwise_oracle.mojo`'s
    # `yeti_rng`): one draw per evaluation
    var yeti_rng = TRandom(yeti_seed)
    _oracle_move_to(cur_point, current_point, bins, g_cursor, n_rows)
    if loss.objective == GBDT_OBJ_YETI_RANK:
        yeti_rank_eval(
            targets_rows, row_weights, has_row_weights, g_cursor, row_index,
            group_sizes, yeti_tasks.value(), yeti_rng.next_uniform_l(), 10,
            Float32(0.85), offsets, sizes, n_rows, lambda_reg, cur_value,
            cur_grad, cached_der2,
        )
    elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
        pair_logit_eval(
            pairs.value(), g_cursor, row_index, offsets, sizes, n_rows,
            lambda_reg, cur_value, cur_grad, cached_der2,
        )
    elif loss.objective == GBDT_OBJ_QUERY_RMSE:
        query_rmse_eval(
            targets_rows, g_cursor, row_index, group_sizes, offsets, sizes,
            n_rows, lambda_reg, cur_value, cur_grad, cached_der2, row_weights,
        )
    else:
        _loss_eval(
            loss, g_target, g_cursor, offsets, sizes, n_rows, lambda_reg,
            cur_value, cur_grad, cached_der2, g_weights,
        )
    var cur_hess = _second_derivatives(loss.method, weights_cpu, lambda_reg, cached_der2)
    var direction = _diagonal_direction(cur_grad, cur_hess)

    if iterations == 1:
        var result = _walker_move(cur_point, direction, 1.0)
        _regularize(weights_cpu, result)
        return result^

    var updated = False
    var iteration = 0
    while iteration < iterations:
        var function_value = cur_value
        var step = Float64(1.0)
        var accepted = False
        var next_value = Float64(0.0)
        var next_grad = List[Float64]()
        while iteration < iterations or ((not updated) and iteration < 100):
            var next_point = _walker_move(cur_point, direction, step)
            _regularize(weights_cpu, next_point)
            _oracle_move_to(next_point, current_point, bins, g_cursor, n_rows)
            if loss.objective == GBDT_OBJ_YETI_RANK:
                yeti_rank_eval(
                    targets_rows, row_weights, has_row_weights, g_cursor,
                    row_index, group_sizes, yeti_tasks.value(), yeti_rng.next_uniform_l(),
                    10, Float32(0.85), offsets, sizes, n_rows, lambda_reg,
                    next_value, next_grad, cached_der2,
                )
            elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
                pair_logit_eval(
                    pairs.value(), g_cursor, row_index, offsets, sizes, n_rows,
                    lambda_reg, next_value, next_grad, cached_der2,
                )
            elif loss.objective == GBDT_OBJ_QUERY_RMSE:
                query_rmse_eval(
                    targets_rows, g_cursor, row_index, group_sizes, offsets,
                    sizes, n_rows, lambda_reg, next_value, next_grad,
                    cached_der2, row_weights,
                )
            else:
                _loss_eval(
                    loss, g_target, g_cursor, offsets, sizes, n_rows,
                    lambda_reg, next_value, next_grad, cached_der2, g_weights,
                )
            if function_value <= next_value:
                cur_hess = _second_derivatives(loss.method, weights_cpu, lambda_reg, cached_der2)
                cur_point = next_point.copy()
                cur_value = next_value
                cur_grad = next_grad.copy()
                direction = _diagonal_direction(cur_grad, cur_hess)
                iteration += 1
                updated = True
                accepted = True
                break
            iteration += 1
            step /= 2
        if not accepted:
            break
    return cur_point^


#: A host task's view of a `List[Bool]` (the scan's flags).
comptime _BoolPtr = MutPointer[Bool, MutUntrackedOrigin]


def _float_to_sortable(bits: UInt32) -> UInt32:
    """`float_to_sortable` (`segmented_sort.mojo:88-93`)."""
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


def _segmented_inclusive_scan(
    values: List[Float32], flags: List[Bool], n: Int
) -> List[Float32]:
    """`launch_segmented_scan_vector(inclusive=True)`
    (`segmented_scan.mojo`): phase 1 the in-block doubling scan at the 768
    block with its snapshot reads, phase 2 the serial scan of the block
    aggregates, phase 3 the carry added to the elements that saw no start
    in their own block, then the inclusive copy."""
    comptime B = GBDT_SEG_SCAN_BLOCK
    var n_blocks = (n + B - 1) // B
    var scanned = List[Float32](length=n, fill=Float32(0.0))
    var has_flag = List[Bool](length=n, fill=False)
    var block_sums = List[Float32](length=n_blocks, fill=Float32(0.0))
    var block_flags = List[Bool](length=n_blocks, fill=False)
    # THE SPLIT (lane gbdt-host-par). Phase 1's unit is one 768-thread
    # device block: its doubling scan reads and writes its own elements and
    # its own aggregate. Blocks go round robin to `_gbdt_units` tasks; a
    # task holds the block's value / flag slabs and their snapshots (the
    # snapshot lanes `d..B` are rewritten before each step reads them).
    var units = _gbdt_units(n_blocks, 24 * n)
    var values_p = rebind[HostF32Ptr](values.unsafe_ptr())
    var flags_p = rebind[_BoolPtr](flags.unsafe_ptr())
    var scanned_p = rebind[HostF32Ptr](scanned.unsafe_ptr())
    var has_p = rebind[_BoolPtr](has_flag.unsafe_ptr())
    var sums_p = rebind[HostF32Ptr](block_sums.unsafe_ptr())
    var bflags_p = rebind[_BoolPtr](block_flags.unsafe_ptr())

    def scan_blocks(unit: Int) {imm units, imm n_blocks, imm n, imm values_p, imm flags_p, imm scanned_p, imm has_p, imm sums_p, imm bflags_p}:
        var s_val = List[Float32](length=B, fill=Float32(0.0))
        var s_flg = List[Int](length=B, fill=0)
        var lv = List[Float32](length=B, fill=Float32(0.0))
        var lf = List[Int](length=B, fill=0)
        var b = unit
        while b < n_blocks:
            for tid in range(B):
                var i = b * B + tid
                # an out-of-range thread holds 0.0 and no flag (reused slabs)
                s_val[tid] = Float32(0.0)
                s_flg[tid] = 0
                if i < n:
                    s_val[tid] = values_p[i]
                    s_flg[tid] = 1 if flags_p[i] else 0
            var d = 1
            while d < B:
                for tid in range(d, B):
                    lv[tid] = s_val[tid - d]
                    lf[tid] = s_flg[tid - d]
                for tid in range(d, B):
                    if s_flg[tid] == 0:
                        s_val[tid] = s_val[tid] + lv[tid]
                        s_flg[tid] = lf[tid]
                d *= 2
            for tid in range(B):
                var i = b * B + tid
                if i < n:
                    scanned_p[i] = s_val[tid]
                    has_p[i] = s_flg[tid] != 0
            sums_p[b] = s_val[B - 1]
            bflags_p[b] = s_flg[B - 1] != 0
            b += units

    host_parallelize(scan_blocks, units)
    _ = len(values)
    _ = len(flags)
    # KEPT SERIAL, by nature: phase 2 is the device's serial scan of the
    # block aggregates, each block's carry the running value of the blocks
    # before it (one add per 768 elements).
    var running = Float32(0.0)
    for b in range(n_blocks):
        var v = block_sums[b]
        var f = block_flags[b]
        block_sums[b] = running
        if f:
            running = v
        else:
            running = running + v
    # phase 3: one statement per element, in `_gbdt_units` ranges
    var c_units = _gbdt_units(n, 2 * n)
    var c_chunk = host_predict_chunk(n, c_units)

    def carry(unit: Int) {imm c_chunk, imm n, imm scanned_p, imm has_p, imm sums_p}:
        for i in range(unit * c_chunk, min(n, (unit + 1) * c_chunk)):
            if not has_p[i]:
                scanned_p[i] = scanned_p[i] + sums_p[i // B]

    host_parallelize(carry, c_units)
    _ = len(has_flag)
    _ = len(block_sums)
    return scanned^


def _exact_estimate(
    loss: GbdtHostLoss,
    g_target: List[Float32],
    g_cursor: List[Float32],
    offsets: List[Int],
    sizes: List[Int],
    weights_cpu: List[Float64],
    n_rows: Int,
    g_weights: List[Float32] = List[Float32](),
) raises -> List[Float32]:
    """`BinOptimizedOracle.estimate_exact` (`pointwise_oracle.mojo:433-503`)
    over `compute_exact_approx` (`leaves_estimation_helper.mojo`): the
    residual and its weight (`compute_exact_value_kernel`: the row weight in
    bin order, `g_weights`, or 1.0 when it is empty; lane gbdt-cpu-rest
    2026-11-23), the MAPE quotient `w / max(1, |r|)`, the stable sort of each leaf
    on bits [10, 32) of the sortable key, the prefix weights, the need
    weights and the binary search, then `RegularizeImpl`."""
    var n_leaves = len(sizes)
    var is_mape = loss.objective == GBDT_OBJ_MAPE
    if (not is_mape) and loss.objective != GBDT_OBJ_MAE and loss.objective != GBDT_OBJ_QUANTILE:
        raise Error(
            "Only MAPE, MAE and Quantile are supported for Exact leaves"
            " estimation on GPU"
        )
    var residuals = List[Float32](length=n_rows, fill=Float32(0.0))
    var weights = List[Float32](length=n_rows, fill=Float32(1.0))
    var has_weights = len(g_weights) > 0
    # an empty `g_weights` is never read (`has_weights` guards it)
    var gw_p = rebind[HostF32Ptr](g_weights.unsafe_ptr())
    # THE SPLITS (lane gbdt-host-par). The residual and the MAPE quotient
    # are one statement per position (`_gbdt_units` ranges). The sort's and
    # the need-weights' unit is one LEAF: the device's segmented sort orders
    # each leaf's range on its own keys and the need weight is the leaf's
    # own 1024-lane strided fold, so leaves go round robin to the tasks.
    var r_units = _gbdt_units(n_rows, 6 * n_rows)
    var r_chunk = host_predict_chunk(n_rows, r_units)
    var gt_p = rebind[HostF32Ptr](g_target.unsafe_ptr())
    var gc_p = rebind[HostF32Ptr](g_cursor.unsafe_ptr())
    var res_p = rebind[HostF32Ptr](residuals.unsafe_ptr())
    var w_p = rebind[HostF32Ptr](weights.unsafe_ptr())

    def residual_rows(unit: Int) {imm r_chunk, imm n_rows, imm is_mape, imm has_weights, imm gt_p, imm gc_p, imm gw_p, imm res_p, imm w_p}:
        for pos in range(unit * r_chunk, min(n_rows, (unit + 1) * r_chunk)):
            var r = ftz(gt_p[pos] - gc_p[pos])
            res_p[pos] = r
            var w = Float32(1.0)
            if has_weights:
                w = gw_p[pos]
            if is_mape:
                var delta = max(Float32(1.0), abs(r))
                w = ftz(w / delta)
            w_p[pos] = w

    host_parallelize(residual_rows, r_units)
    _ = len(g_target)
    _ = len(g_cursor)
    _ = len(g_weights)
    var ordered_targets = List[Float32](length=n_rows, fill=Float32(0.0))
    var ordered_weights = List[Float32](length=n_rows, fill=Float32(0.0))
    var flags = List[Bool](length=n_rows, fill=False)
    # 20 operations a row stands for the sort's compares at a leaf's size
    var l_units = _gbdt_units(n_leaves, 20 * n_rows)
    var off_p = rebind[GbdtIntPtr](offsets.unsafe_ptr())
    var sz_p = rebind[GbdtIntPtr](sizes.unsafe_ptr())
    var ot_p = rebind[HostF32Ptr](ordered_targets.unsafe_ptr())
    var ow_p = rebind[HostF32Ptr](ordered_weights.unsafe_ptr())
    var flags_p = rebind[_BoolPtr](flags.unsafe_ptr())

    def sort_leaves(unit: Int) {imm l_units, imm n_leaves, imm off_p, imm sz_p, imm res_p, imm w_p, imm ot_p, imm ow_p, imm flags_p}:
        var leaf = unit
        while leaf < n_leaves:
            var base = off_p[leaf]
            var size = sz_p[leaf]
            if size > 0:
                var keys = List[UInt64](capacity=size)
                for k in range(size):
                    var key = _float_to_sortable(bitcast[DType.uint32](res_p[base + k]))
                    keys.append((UInt64(key >> UInt32(GBDT_EXACT_FIRST_BIT)) << UInt64(32)) | UInt64(k))
                sort(keys)
                for k in range(size):
                    var src = base + Int(keys[k] & UInt64(0xFFFFFFFF))
                    ot_p[base + k] = res_p[src]
                    ow_p[base + k] = w_p[src]
                flags_p[base] = True
            leaf += l_units

    host_parallelize(sort_leaves, l_units)
    _ = len(residuals)
    _ = len(weights)
    var prefix = _segmented_inclusive_scan(ordered_weights, flags, n_rows)

    var point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var eps = Float32(1.1920929e-07)
    var alpha = loss.estimator_alpha
    var n_units = _gbdt_units(n_leaves, 2 * n_rows + 1024 * n_leaves)
    var prefix_p = rebind[HostF32Ptr](prefix.unsafe_ptr())
    var point_p = rebind[HostF32Ptr](point.unsafe_ptr())

    def need_leaves(unit: Int) {imm n_units, imm n_leaves, imm alpha, imm eps, imm off_p, imm sz_p, imm ot_p, imm ow_p, imm prefix_p, imm point_p}:
        # every lane is rewritten per leaf (the slab is reused)
        var slab = List[Float32](length=GBDT_NEED_WEIGHTS_BLOCK, fill=Float32(0.0))
        var leaf = unit
        while leaf < n_leaves:
            var base = off_p[leaf]
            var size = sz_p[leaf]
            if size <= 0:
                point_p[leaf] = Float32(0.0)
            else:
                for tid in range(GBDT_NEED_WEIGHTS_BLOCK):
                    var total_sum = Float32(0.0)
                    var idx = tid
                    while idx < size:
                        total_sum = ftz(total_sum + ow_p[base + idx])
                        idx += GBDT_NEED_WEIGHTS_BLOCK
                    slab[tid] = total_sum
                var need = ftz(_halving_fold(slab) * alpha)
                var left = base
                var right = base + size - 1
                for _ in range(GBDT_QUANTILE_ITERATIONS):
                    var middle = left + (right - left) // 2
                    if prefix_p[middle] < need - eps:
                        left = middle
                    else:
                        right = middle
                point_p[leaf] = ot_p[right]
            leaf += n_units

    host_parallelize(need_leaves, n_units)
    _ = len(offsets)
    _ = len(sizes)
    _ = len(ordered_targets)
    _ = len(ordered_weights)
    _ = len(prefix)
    comptime if GBDT_ORACLE_HOST_SABOTAGE:
        for leaf in range(n_leaves):
            point[leaf] = point[leaf] + Float32(0.25)
    _regularize(weights_cpu, point)
    return point^


def _estimate_leaves_for_loss(
    loss: GbdtHostLoss,
    targets: List[Float32],
    cursor: List[Float32],
    row_index: List[Int],
    offsets: List[Int],
    sizes: List[Int],
    n_rows: Int,
    l2_leaf_reg: Float32,
    group_sizes: List[Int] = List[Int](),
    pairs: Optional[HostPairs] = None,
    yeti_tasks: Optional[YetiRankTasks] = None,
    yeti_seed: UInt64 = UInt64(0),
    weights: List[Float32] = List[Float32](),
) raises -> List[Float32]:
    """`_estimate_and_apply`'s estimate (`doc_parallel_boosting.mojo:
    698-797`): the gathers by the row index, then Exact or the walker.
    `weights` (row order, empty: unit) are the fit's row weights: gathered
    with the targets, and their per-leaf sums are the leaf weights Gradient
    leaves and `RegularizeImpl` read, as PairLogit's pair row weights are."""
    var n_leaves = len(sizes)
    var g_target = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    # the gathers and the bins move values, one position a device thread:
    # `_gbdt_units` ranges of positions each (lane gbdt-host-par)
    _gather_rows(targets, row_index, n_rows, g_target)
    _gather_rows(cursor, row_index, n_rows, g_cursor)
    var bins = List[Int](length=n_rows, fill=0)
    _leaf_bins(offsets, sizes, n_rows, bins)
    var weights_cpu = List[Float64]()
    var has_weights = len(weights) > 0
    var g_weights = List[Float32]()
    if has_weights and loss.objective != GBDT_OBJ_PAIR_LOGIT:
        g_weights = List[Float32](length=n_rows, fill=Float32(0.0))
        _gather_rows(weights, row_index, n_rows, g_weights)
    # Both CPU searcher callers compiled on Apple M1 at O0 (2026-10-28);
    # runtime/identity remain pending. GPU setup replaces row weights with per-row
    # pair weights. Gradient leaves and Regularize need their partition sums,
    # not row counts; this also repairs the existing greedy grouped owner.
    var pair_row_weights = List[Float32]()
    if loss.objective == GBDT_OBJ_PAIR_LOGIT:
        pair_row_weights = List[Float32](length=n_rows, fill=Float32(0))
        if pairs.value().group_layout:
            _gather_rows(
                pairs.value().group_row_weights, row_index, n_rows, pair_row_weights
            )
        else:
            _gather_rows(
                pairs.value().prep.row_weights, row_index, n_rows, pair_row_weights
            )
    if loss.objective == GBDT_OBJ_PAIR_LOGIT:
        # the leaves' weight sums, the launches across the cores (lane
        # gbdt-host-par): `_pinned_partition_stat` per leaf, unchanged
        var leaf_weight_sums = List[Float32](length=n_leaves, fill=Float32(0.0))
        _pinned_partition_stats(
            pair_row_weights, offsets, sizes, 2 * GBDT_PINNED_SM, leaf_weight_sums
        )
        for leaf in range(n_leaves):
            weights_cpu.append(Float64(leaf_weight_sums[leaf]))
    elif len(g_weights) > 0:
        # the row weights' leaf sums, the PairLogit arm's partition stat
        var leaf_weight_sums = List[Float32](length=n_leaves, fill=Float32(0.0))
        _pinned_partition_stats(
            g_weights, offsets, sizes, 2 * GBDT_PINNED_SM, leaf_weight_sums
        )
        for leaf in range(n_leaves):
            weights_cpu.append(Float64(leaf_weight_sums[leaf]))
    else:
        for leaf in range(n_leaves):
            weights_cpu.append(Float64(sizes[leaf]))
    if loss.method == GBDT_LEAF_EXACT:
        return _exact_estimate(
            loss, g_target, g_cursor, offsets, sizes, weights_cpu, n_rows,
            g_weights,
        )
    var row_weights = List[Float32]()
    if loss.objective != GBDT_OBJ_PAIR_LOGIT:
        row_weights = weights.copy()
    return _walker_estimate(
        loss, g_target, g_cursor, bins, offsets, sizes, weights_cpu, n_rows,
        l2_leaf_reg, targets, row_index, group_sizes, pairs, yeti_tasks,
        yeti_seed, g_weights, row_weights,
    )


# ===========================================================================
# THE FIT
# ===========================================================================


def gbdt_losses_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    loss: GbdtHostLoss,
    eval: GbdtHostEval,
    group_sizes: List[Int] = List[Int](),
    pair_winners: List[UInt32] = List[UInt32](),
    pair_losers: List[UInt32] = List[UInt32](),
    pair_weights: List[Float32] = List[Float32](),
    start: Float64 = 0.0,
    random_strength: Float32 = Float32(0.0),
    weights: List[Float32] = List[Float32](),
    feature_fraction: Float64 = 1.0,
    score_function: Int = SCORE_FUNCTION_COSINE,
) raises -> GbdtHostFitWithEval:
    """`train` then `fit_with_test` on the covered configurations (see the
    module docstring). `group_sizes` is read by QueryRMSE alone, already
    resolved to their `TWithoutQueriesGrouping` rule by the binding.

    `weights` (empty: unit) are the sample weights, the device's one
    `weights` column (lane gbdt-gaps, 2026-11-23): the pointwise losses read
    them in the search pass, the leaf estimate and the learn loss; YetiRank
    through its own `has_weights` arm; PairLogit through the generated
    pairs' weights (`generate_pairs` / the group layout, as `train` builds
    them; a caller's pairs keep their own weights, the device replaces the
    row weights with the pair row weights); QueryRMSE through its weighted
    query means (`gbdt_oracle_query.mojo`, lane gbdt-cpu-rest).

    `eval` is the HELD-OUT arm (lane gbdt-cpu-rest-2, 2026-11-23), the
    same four pieces `gbdt_oracle.mojo::gbdt_host_boost` runs
    (`gbdt/host/gbdt_oracle_eval.mojo`: the test rows binarized against the
    fit's grid, the last tree onto the test cursor, the held-out loss
    through the learn loss's own target kernel at unit weight, the detector
    and `ShrinkToBestIteration`); the test cursor starts at `start`, as
    `CreateCursors` seeds it. `eval.n_rows == 0` is no eval set and the fit
    is the one this function ran before. The grouped losses never reach it
    with one: `train` refuses their eval set (their held-out loss needs the
    eval pool's grouping), and so does the binding.

    `feature_fraction` below 1 is the device's per-tree feature sample
    (`doc_parallel_boosting.mojo`, `sample_tree_folds` on its own stream,
    the tree layout and the learn index projected onto it), through the
    same two helpers `gbdt_host_boost` and the non-symmetric driver call
    (`_sample_tree_folds`, `_ns_project_index`). The search reads the
    tree's layout; the splits it returns name the fit's features and bins,
    so the estimation and the held-out apply read the full layout.

    `score_function` (lane gbdt-cpu-rest-4-sym, 2026-11-23) is the device's
    `enqueue_score` dispatch (`_greedy_gains`: L2 / NewtonL2 through the L2
    kernel, Cosine / NewtonCosine through the cosine kernel) over the
    search planes `secondDerAsWeights` keys: under NewtonL2 and
    NewtonCosine plane 0 is the weighted second derivative
    (`_loss_search_pass(second_order=True)`, PairLogit's
    `pair_logit_search_pass(second_order=True)`), as the grouped pointwise
    searcher's host fit reads them (QueryRMSE's and YetiRank's planes do
    not take the flag there either)."""
    var second_order = is_second_order_score_function(score_function)
    var has_weights = len(weights) > 0
    if has_weights and len(weights) != n_rows:
        raise Error("weights size mismatch")
    if (
        loss.objective == GBDT_OBJ_QUERY_RMSE
        or loss.objective == GBDT_OBJ_PAIR_LOGIT
        or loss.objective == GBDT_OBJ_YETI_RANK
    ) and len(group_sizes) < 1:
        raise Error("the querywise host fit needs the query sizes")
    # the YetiRank task table (`InitYetiRank`'s query size refusal inside) and
    # draw stream, as `doc_parallel_boosting.mojo::fit_with_test` builds them
    var yeti_tasks = Optional[YetiRankTasks]()
    var yeti_rand = TRandom(params.random_seed ^ UInt64(0x5945544952414E4B))
    if loss.objective == GBDT_OBJ_YETI_RANK:
        var yeti_sizes = List[UInt32](capacity=len(group_sizes))
        for g in range(len(group_sizes)):
            yeti_sizes.append(UInt32(group_sizes[g]))
        yeti_tasks = Optional(yeti_rank_tasks(yeti_sizes, n_rows))
    # the PairLogit pairs, generated or given, in the device order
    # (`gbdt/train.mojo::train` does the same with the same functions)
    var pairs = Optional[HostPairs]()
    var loss_norm = Float64(n_rows)
    if loss.objective == GBDT_OBJ_PAIR_LOGIT:
        var sizes_u32 = List[UInt32](capacity=len(group_sizes))
        for g in range(len(group_sizes)):
            sizes_u32.append(UInt32(group_sizes[g]))
        if IDN_PAIRLOGIT_GROUP and len(pair_winners) == 0:
            # lane/fam2-gbdt F2: generated pairs take the device's group
            # layout (`pair_logit_group.mojo`), no pair list
            var hg = host_pair_groups(sizes_u32, y, weights, n_rows)
            loss_norm = hg.total_weight()
            pairs = Optional(hg^)
        else:
            var ordered: PairList
            if len(pair_winners) > 0:
                ordered = order_pairs_by_winner(
                    PairList(pair_winners.copy(), pair_losers.copy(), pair_weights.copy()),
                    sizes_u32, n_rows,
                )
            else:
                ordered = order_pairs_by_winner(
                    generate_pairs(sizes_u32, y, weights), sizes_u32, n_rows
                )
            var hp = host_pairs(ordered.winners, ordered.losers, ordered.weights, n_rows)
            loss_norm = hp.total_weight()
            pairs = Optional(hp^)
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if params.max_depth < 0:
        raise Error("max_depth must not be negative")

    var grid = gbdt_host_grid(
        x_colmajor, n_rows, n_features, params.border_count,
        params.border_build_max_samples, params.random_seed, params.nan_mode,
        params.border_type,
    )
    var one_hot = List[Bool](length=n_features, fill=False)
    var layout = build_layout(grid.fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var hist_cells = layout.hist_cells

    var bf_feature = List[Int](length=hist_cells, fill=0)
    var bf_bin = List[Int](length=hist_cells, fill=0)
    for f in range(n_features):
        ref lf = layout.features[f]
        for b in range(Int(lf.folds)):
            bf_feature[Int(lf.first_fold_index) + b] = f
            bf_bin[Int(lf.first_fold_index) + b] = b

    # ---- the HELD-OUT arm (gbdt/host/gbdt_oracle_eval.mojo), as
    # `gbdt_host_fit_eval` sets it up: the test rows quantized against the
    # model's own borders and layout ----
    var has_test = eval.n_rows > 0
    if has_test:
        if len(eval.x_colmajor) != eval.n_rows * n_features:
            raise Error("eval_set x_colmajor size mismatch")
        if len(eval.y) != eval.n_rows:
            raise Error("eval_set y size mismatch")
        if (
            loss.objective == GBDT_OBJ_QUERY_RMSE
            or loss.objective == GBDT_OBJ_PAIR_LOGIT
            or loss.objective == GBDT_OBJ_YETI_RANK
        ):
            # `train`'s words (gbdt/train.mojo)
            raise Error(
                "an eval_set with a grouped loss is not implemented here: the"
                " held-out loss needs the eval pool's own grouping"
            )
    var n_eval = eval.n_rows if has_test else 1
    var test_cindex = List[UInt32]()
    if has_test:
        test_cindex = _binarize_columns(
            eval.x_colmajor, eval.n_rows, n_features, grid, layout
        )
    var test_cursor = List[Float32](length=n_eval, fill=Float32(start))
    var detector = gbdt_eval_detector(eval)
    var test_losses = List[Float64]()
    var stopped_early = False
    # the packed per-feature records `_apply_last_tree_to_test` writes
    var feat_offset = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_mask = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_shift = List[UInt32](length=n_features, fill=UInt32(0))
    var feat_one_hot = List[Bool](length=n_features, fill=False)
    for f in range(n_features):  # driver-loop: over the features (one record each)
        feat_offset[f] = layout.features[f].offset
        feat_mask[f] = layout.features[f].mask
        feat_shift[f] = layout.features[f].shift
        feat_one_hot[f] = layout.features[f].one_hot_feature

    # ---- the per-tree feature sample's state: the searcher's layout,
    # blocks and bin-feature tables, the fit's own at fraction 1 ----
    var ff_on = feature_fraction < 1.0
    var s_layout = layout.copy()
    var s_blocks = blocks.copy()
    var s_hist_cells = hist_cells
    var s_bf_feature = bf_feature.copy()
    var s_bf_bin = bf_bin.copy()
    # under a sample the learn index is projected per tree (as
    # `gbdt_host_boost` does to `cindexes[0]`), so the full one is kept
    var full_cindex = List[UInt32]()
    if ff_on:
        full_cindex = cindex.copy()
    var ff_one_hot = List[Bool](length=n_features, fill=False)
    var feature_random = TRandom(params.random_seed ^ UInt64(0x4645415455524553))

    var max_depth = params.max_depth
    var max_leaves = 1 << max_depth
    var lr = params.learning_rate

    # the starting point `boost_from_average` sets (`start_value`, the
    # binding's `calc_sample_quantile` constant for MAE / Quantile / MAPE,
    # lane/catboost-parity); 0 without it
    var cursor = List[Float32](length=n_rows, fill=Float32(start))
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var mse_blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_blocks = mse_blocks
    # the magnitudes are two per 256-row block, except on PairLogit's group
    # layout (`IDN_PAIRLOGIT_GROUP`): two per GROUP, as the value partials
    var mag_blocks = mse_blocks
    if pairs.__bool__():
        fv_blocks = pairs.value().blocks()
        if pairs.value().group_layout:
            mag_blocks = fv_blocks
    var fv_part = List[Float32](length=fv_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mag_blocks, fill=Float32(0.0))
    var bootstrap_on = loss.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(params.random_seed)
    # the score noise, the stochastic arm of `gbdt_oracle.mojo::
    # gbdt_host_fit_eval`: the same draws in the same order
    var noise_rand = TRandom(params.random_seed)

    var losses = List[Float64]()
    var tree_split_offsets = List[Int]()
    tree_split_offsets.append(0)
    var split_features = List[Int]()
    var split_bins = List[Int]()
    var tree_leaf_offsets = List[Int]()
    tree_leaf_offsets.append(0)
    var model_leaves = List[Float32]()

    for iteration in range(params.n_estimators):
        # ---- the gradients, the learn loss and the magnitudes ----
        if loss.objective == GBDT_OBJ_YETI_RANK:
            # the tree's first YetiRank draw, as the device's search pass
            yeti_rank_search_pass(
                y, weights, has_weights, cursor, group_sizes,
                yeti_tasks.value(), yeti_rand.next_uniform_l(), 10,
                Float32(0.85), n_rows, stats, fv_part, mag_part,
            )
        elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
            pair_logit_search_pass(
                pairs.value(), cursor, n_rows, stats, fv_part, mag_part,
                second_order,
            )
        elif loss.objective == GBDT_OBJ_QUERY_RMSE:
            query_rmse_search_pass(
                y, cursor, group_sizes, n_rows, stats, fv_part, mag_part, weights
            )
        else:
            _loss_search_pass(
                loss, y, cursor, n_rows, stats, fv_part, mag_part, weights,
                second_order,
            )
        var fv = _deterministic_sum_lanes(fv_part, 1, fv_blocks)[0]
        var noise_mult = Float64(0.0)
        if random_strength != Float32(0.0):
            # the device's one body (gbdt/methods/random_score_bodies.mojo;
            # this was a host libm restatement, lane gbdt-cpu-rest-2)
            noise_mult = score_model_length_mult(
                Float64(n_rows), iteration, params.learning_rate
            )
        var tree_seed = noise_rand.next_uniform_l()
        var fixed_scale: Float32
        if bootstrap_on:
            var bm = _bootstrap_pass(
                loss.bootstrap_kind, seeds, stats, n_rows, loss.bootstrap_param
            )
            fixed_scale = _choose_scale_from_magnitudes(bm[0], bm[1], n_rows)
        else:
            var mags = _deterministic_sum_lanes(mag_part, 2, mag_blocks)
            fixed_scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)
        _snap_gradients(stats, n_rows, 2, fixed_scale)  # lane/sym-quality
        var score_std_dev = Float32(0.0)
        if random_strength != Float32(0.0):
            score_std_dev = Float32(
                Float64(Float32(noise_mult * Float64(random_strength)))
                * _target_std_dev(stats, n_rows)
            )
        var level_rand = TRandom(tree_seed)

        # ---- the per-tree feature sample (`feature_fraction < 1`), the
        # statements of `gbdt_oracle.mojo::gbdt_host_boost` ----
        if ff_on:
            var tree_folds = _sample_tree_folds(
                grid.fold_counts, feature_fraction, feature_random
            )
            s_layout = build_layout(tree_folds, ff_one_hot)
            s_blocks = blocks_for(s_layout, n_rows)
            cindex = _ns_project_index(
                full_cindex, layout, s_layout, tree_folds, n_rows, n_features
            )
            s_hist_cells = s_layout.hist_cells
            s_bf_feature = List[Int](length=s_hist_cells, fill=0)
            s_bf_bin = List[Int](length=s_hist_cells, fill=0)
            for f in range(n_features):  # driver-loop: over the features and their borders (the tree layout)
                ref lf = s_layout.features[f]
                for b in range(Int(lf.folds)):
                    s_bf_feature[Int(lf.first_fold_index) + b] = f
                    s_bf_bin[Int(lf.first_fold_index) + b] = b
        # ---- `run_tree_layout_traced`, every level (TWIN: the loop in
        # `gbdt_oracle.mojo::gbdt_host_fit`; edit both) ----
        var row_index = List[Int](length=n_rows, fill=0)
        _identity_rows(row_index, n_rows)
        var p_off = List[Int](length=max_leaves, fill=0)
        var p_sz = List[Int](length=max_leaves, fill=0)
        p_sz[0] = n_rows
        var hist = List[Float32](
            length=max_leaves * 2 * s_hist_cells, fill=Float32(0.0)
        )
        var ids_compute = List[Int](length=max_leaves, fill=0)
        var sub_from = List[Int](length=max_leaves, fill=0)
        var sub_what = List[Int](length=max_leaves, fill=0)
        var winners_score = List[Float32]()
        var winners_bf = List[UInt32]()
        var n_live = 1
        for depth in range(max_depth):
            var level_seed = level_rand.next_uniform_l()
            var half = n_live // 2
            var planned = depth > 0
            var compute = List[Int]()
            if planned:
                for j in range(half):
                    compute.append(ids_compute[j])
            else:
                for j in range(n_live):
                    compute.append(j)

            var block_first_bin = 0
            for b in range(len(s_blocks)):
                ref blk = s_blocks[b]
                var total = 0
                for k in range(blk.count()):
                    total += Int(blk.folds[k])
                if blk.policy == POLICY_BINARY:
                    _binary_block(
                        blk, block_first_bin, s_hist_cells, compute, depth,
                        p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_HALF_BYTE:
                    _half_byte_block(
                        blk, block_first_bin, s_hist_cells, compute, depth,
                        p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_ONE_BYTE:
                    _one_byte_block(
                        blk, block_first_bin, s_hist_cells, compute, p_off, p_sz,
                        row_index, stats, cindex, s_layout, n_rows, fixed_scale,
                        hist,
                    )
                block_first_bin += total

            _scan_histograms(hist, s_hist_cells, 2, compute, s_layout, n_features)

            if planned and half > 0:
                _subtract_histograms(hist, s_hist_cells, sub_from, sub_what, half)

            var part_stats = _partition_stats_level(
                stats, n_rows, 2, GBDT_PARTITION_CHUNKS, p_off, p_sz, n_live
            )

            var gains = _greedy_gains(
                score_function, hist, s_hist_cells, part_stats, n_live,
                params.l2_leaf_reg, score_std_dev, level_seed, s_bf_feature,
            )
            var best_bin = _gain_winner(gains, s_hist_cells)
            var best_gain = -GBDT_FLOAT32_MAX
            if best_bin != GBDT_SENTINEL:
                best_gain = gains[Int(best_bin)]
            if best_bin != GBDT_SENTINEL:
                winners_score.append(ftz(best_gain))
            else:
                winners_score.append(ftz(-GBDT_FLOAT32_MAX))
            winners_bf.append(best_bin)
            var bf_split = 0
            if best_bin != GBDT_SENTINEL:
                bf_split = Int(best_bin)

            var split_f = s_bf_feature[bf_split] if s_hist_cells > 0 else 0
            var split_b = s_bf_bin[bf_split] if s_hist_cells > 0 else 0
            ref sfeat = s_layout.features[split_f]
            _split_level(
                cindex, Int(sfeat.offset) * n_rows, sfeat.mask << sfeat.shift,
                UInt32(split_b) << sfeat.shift, sfeat.one_hot_feature, n_rows,
                n_live, s_hist_cells, row_index, stats, p_off, p_sz, hist,
                ids_compute, sub_from, sub_what,
            )
            n_live = n_live * 2

        # ---- the gates, post-tree, and the rollback ----
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        var grown = 0
        for d in range(max_depth):
            var best_score = winners_score[d]
            var best_bin_u = winners_bf[d]
            if best_bin_u == GBDT_SENTINEL or Int(best_bin_u) >= s_hist_cells:
                raise Error(
                    "All splits have infinite score. Probably, numerical"
                    " overflow occurs in loss function and/or split score"
                    " calculation. Try increasing l2_leaf_reg, and/or"
                    " decreasing learning_rate, etc."
                    " [level " + String(d) + ", live leaves "
                    + String(1 << d) + "]"
                )
            var cf_ = s_bf_feature[Int(best_bin_u)]
            var cb_ = s_bf_bin[Int(best_bin_u)]
            if not (best_score > Float32(0.0)):
                break
            var repeated = False
            for i in range(len(tree_features)):
                if tree_features[i] == cf_ and tree_bins[i] == cb_:
                    repeated = True
            if repeated:
                break
            tree_features.append(cf_)
            tree_bins.append(cb_)
            grown += 1
        if grown < max_depth:
            var live = n_live
            for _ in range(max_depth - grown):
                var h2 = live // 2
                for i in range(h2):
                    p_sz[i] = p_sz[i] + p_sz[h2 + i]
                live = h2
            n_live = live
        var sizes = List[Int]()
        var offsets = List[Int]()
        var covered_rows = 0
        for i in range(n_live):
            sizes.append(p_sz[i])
            offsets.append(p_off[i])
            covered_rows += p_sz[i]
        if covered_rows != n_rows:
            raise Error(
                "gbdt host: the final leaf partitions cover "
                + String(covered_rows) + " of " + String(n_rows) + " rows"
            )

        # ---- the estimation task and `AppendModels` ----
        var yeti_tree_seed = UInt64(0)
        if loss.objective == GBDT_OBJ_YETI_RANK:
            # the tree's second YetiRank draw: the estimation stream's seed
            yeti_tree_seed = yeti_rand.next_uniform_l()
        var estimated = _estimate_leaves_for_loss(
            loss, y, cursor, row_index, offsets, sizes, n_rows,
            params.l2_leaf_reg, group_sizes, pairs, yeti_tasks, yeti_tree_seed,
            weights,
        )
        if loss.objective == GBDT_OBJ_PAIR_LOGIT or loss.objective == GBDT_OBJ_YETI_RANK:
            # `MakeZeroAverage` (`doc_parallel_leaves_estimator.cpp:25-37`),
            # restated from `device_walker.enqueue_zero_average`: minus
            # the unweighted mean over all `n_live` leaves, the double sum in
            # the kernel's order (lane cpu4-gbdt; was an ascending chain).
            _zero_average_host(estimated)
        var scaled = rescaled_leaves(estimated, lr)
        apply_scaled_leaves_host(cursor, scaled, row_index, offsets, sizes, n_live)
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(estimated)):
            model_leaves.append(scaled[i])
        tree_leaf_offsets.append(len(model_leaves))

        # ---- `AppendModels(..., learnCursors, testCursor)`: the SAME weak
        # model onto the held-out cursor, its loss through the SAME target
        # kernel at unit weight, and the detector (the statements of
        # `gbdt_oracle.mojo::gbdt_host_boost`). A depth-0 tree adds nothing;
        # the stored values already carry the learning rate. ----
        if has_test:
            if grown > 0:
                var test_bins = gbdt_eval_test_bins(
                    test_cindex, n_eval, tree_features, tree_bins,
                    feat_offset, feat_mask, feat_shift, feat_one_hot,
                )
                gbdt_eval_add_tree(test_cursor, test_bins, scaled)
            var t_loss = -Float64(
                _loss_value(loss, eval.y, test_cursor, n_eval)
            ) / Float64(n_eval)
            test_losses.append(t_loss)
            detector.add_error(t_loss)
            if detector.is_need_stop():
                # their break is BEFORE the learn loss append
                # (`doc_parallel_boosting.mojo:2395-2411`)
                stopped_early = True
                break

        if len(losses) < params.n_estimators:
            var v = Float64(fv)
            if iteration + 1 > 1:
                losses.append(-v / loss_norm)

    if loss.objective == GBDT_OBJ_YETI_RANK:
        # the device's final pass writes 0.0 partials and takes no draw
        losses.append(-Float64(Float32(0.0)) / Float64(n_rows))
    elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
        var p_fv = pair_logit_value(pairs.value(), cursor)
        losses.append(
            -Float64(_deterministic_sum_lanes(p_fv, 1, len(p_fv))[0]) / loss_norm
        )
    elif loss.objective == GBDT_OBJ_QUERY_RMSE:
        var q_fv = query_rmse_value(y, cursor, group_sizes, n_rows, weights)
        losses.append(
            -Float64(_deterministic_sum_lanes(q_fv, 1, len(q_fv))[0]) / Float64(n_rows)
        )
    else:
        losses.append(
            -Float64(_loss_value(loss, y, cursor, n_rows, weights)) / Float64(n_rows)
        )
    # ---- `ShrinkToBestIteration` (`gbdt/train.mojo:2241-2254`), the
    # held-out arm's own tracker ----
    var shrink_to = gbdt_eval_shrink_point(
        test_losses, eval.want_best_model, eval.best_model_min_trees
    )
    var n_trees = len(tree_split_offsets) - 1
    if 0 < shrink_to and shrink_to < n_trees:
        while len(tree_split_offsets) - 1 > shrink_to:
            _ = tree_split_offsets.pop()
            _ = tree_leaf_offsets.pop()
        split_features.resize(
            tree_split_offsets[len(tree_split_offsets) - 1], 0
        )
        split_bins.resize(tree_split_offsets[len(tree_split_offsets) - 1], 0)
        model_leaves.resize(
            tree_leaf_offsets[len(tree_leaf_offsets) - 1], Float32(0.0)
        )
    var model = GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, detector.best_iteration, stopped_early,
    )
    return GbdtHostFitWithEval(
        model^,
        GbdtHostEvalFit(test_losses^, detector.best_iteration, stopped_early),
    )


def _zero_average_host(mut estimated: List[Float32]):
    """`device_walker.enqueue_zero_average` (`gbdt/methods/leaves_estimation/
    device_walker.mojo`) on the host. Level 1: block `b` puts leaves
    `[256 b, 256 b + 256)` as doubles (absent ones +0.0) through the 256-lane
    halving tree. Level 2: lane `t` of 256 adds partials `t, t + 256, ...`
    ascending from +0.0, then the halving tree. `bias = -sum / count`;
    `leaf = float(double(leaf) + bias)`.

    KEPT SERIAL: the units are one tree's leaf values, a few adds each,
    far below one host task's work; the fold orders are the kernel's."""
    var n = len(estimated)
    if n == 0:
        return
    var parts = (n + 255) // 256
    var partials = List[Float64](length=parts, fill=Float64(0.0))
    for b in range(parts):
        var slab = List[Float64](length=256, fill=Float64(0.0))
        for t in range(256):
            var i = b * 256 + t
            if i < n:
                slab[t] = Float64(estimated[i])
        var step = 128
        while step > 0:
            for t in range(step):
                slab[t] = slab[t] + slab[t + step]
            step //= 2
        partials[b] = slab[0]
    var red = List[Float64](length=256, fill=Float64(0.0))
    for t in range(256):
        var acc = Float64(0.0)
        var i = t
        while i < parts:
            acc = acc + partials[i]
            i += 256
        red[t] = acc
    var step = 128
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    var bias = -red[0] / Float64(n)
    for i in range(n):
        estimated[i] = Float32(Float64(estimated[i]) + bias)
