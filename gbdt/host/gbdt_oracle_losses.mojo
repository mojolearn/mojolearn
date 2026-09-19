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
bootstrap at subsample 0.6. The binding refuses by name what the symmetric
oracle refuses, and the Bayesian bootstrap.

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
from std.math import isfinite
from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
)
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.gpu_util.kernel.random_gen import next_poisson_f, next_uniform_f
from gbdt.data.pairs import PairList, generate_pairs, order_pairs_by_winner
from gbdt.host.gbdt_oracle_pair import (
    HostPairs,
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
    GBDT_FLOAT32_MAX,
    GBDT_MSE_BLOCK,
    GBDT_ORACLE_HOST_SABOTAGE,
    GBDT_SENTINEL,
    GbdtHostModel,
    GbdtHostParams,
    _binarize_columns,
    _choose_scale_from_magnitudes,
    _cosine_gain,
    _deterministic_sum_lanes,
    _diagonal_direction,
    _half_byte_block,
    _halving_fold,
    _one_byte_block,
    _oracle_move_to,
    _partition_stat,
    _regularize,
    _walker_move,
    gbdt_host_grid,
)


#: The objective codes (`gbdt/targets/kernel/pointwise_targets.mojo:45-56`).
comptime GBDT_OBJ_RMSE = 0
comptime GBDT_OBJ_LOGLOSS = 1
comptime GBDT_OBJ_CROSSENTROPY = 2
comptime GBDT_OBJ_QUANTILE = 3
comptime GBDT_OBJ_MAE = 4
comptime GBDT_OBJ_LOGLINQUANTILE = 5
comptime GBDT_OBJ_MAPE = 6
comptime GBDT_OBJ_POISSON = 7
comptime GBDT_OBJ_LQ = 8
comptime GBDT_OBJ_EXPECTILE = 9
comptime GBDT_OBJ_TWEEDIE = 10
comptime GBDT_OBJ_HUBER = 11
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

#: `BOOTSTRAP_KERNEL_*` (`bootstrap.mojo:106-108`); -1 is no bootstrap.
comptime GBDT_BOOT_BERNOULLI = 1
comptime GBDT_BOOT_POISSON = 2
#: `BOOTSTRAP_BLOCK_SIZE`, `BOOTSTRAP_SEED_COUNT` (`bootstrap.mojo:112-115`).
comptime GBDT_BOOT_BLOCK = 256
comptime GBDT_BOOT_SEEDS = 65536

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


# ===========================================================================
# THE OBJECTIVES (`pointwise_targets.mojo:191-418`)
# ===========================================================================


def _target_sign(x: Float32) -> Float32:
    return Float32(1.0) if x > Float32(0.0) else Float32(-1.0)


def _target_score(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_score` (`pointwise_targets.mojo:203-265`)."""
    if objective == GBDT_OBJ_RMSE:
        return (t - p) * (t - p)
    elif objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_MAE:
        var val = t - p
        var multiplier = alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        return multiplier * val
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        var val = t - identical_exp(p)
        var multiplier = alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        return val * multiplier
    elif objective == GBDT_OBJ_MAPE:
        return abs(t - p) / max(Float32(1.0), abs(t))
    elif objective == GBDT_OBJ_POISSON:
        # `exp(p) - t * p`, CONTRACTED: the device columns fuse the
        # product into the subtraction (IDENTITY_PATHS row 9: Metal through
        # MAX contracts by default), measured on gbdt-parametric-losses
        return identical_mul_add(-t, p, identical_exp(p))
    elif objective == GBDT_OBJ_LQ:
        var abs_loss = abs(t - p)
        return identical_pow(abs_loss, alpha)
    elif objective == GBDT_OBJ_EXPECTILE:
        var val = t - p
        var multiplier = alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        return multiplier * val * val
    elif objective == GBDT_OBJ_TWEEDIE:
        var val = -t * identical_exp((Float32(1.0) - alpha) * p) / (Float32(1.0) - alpha)
        var delta = identical_exp((Float32(2.0) - alpha) * p) / (Float32(2.0) - alpha)
        return val + delta
    elif objective == GBDT_OBJ_HUBER:
        var mismatch = abs(t - p)
        if mismatch < alpha:
            return Float32(0.5) * mismatch * mismatch
        return alpha * (mismatch - Float32(0.5) * alpha)
    return Float32(0.0)


def _target_der(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_der` (`pointwise_targets.mojo:269-327`)."""
    if objective == GBDT_OBJ_RMSE:
        return t - p
    elif objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_MAE:
        var val = t - p
        return alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
    elif objective == GBDT_OBJ_LOGLINQUANTILE:
        var exp_pred = identical_exp(p)
        if t - exp_pred > Float32(0.0):
            return alpha * exp_pred
        return -(Float32(1.0) - alpha) * exp_pred
    elif objective == GBDT_OBJ_MAPE:
        if t - p > Float32(0.0):
            return Float32(1.0) / max(Float32(1.0), abs(t))
        return Float32(-1.0) / max(Float32(1.0), abs(t))
    elif objective == GBDT_OBJ_POISSON:
        return t - identical_exp(p)
    elif objective == GBDT_OBJ_LQ:
        var abs_loss = abs(t - p)
        var abs_loss_q = identical_pow(abs_loss, alpha - Float32(1.0))
        return alpha * _target_sign(t - p) * abs_loss_q
    elif objective == GBDT_OBJ_EXPECTILE:
        var val = t - p
        var multiplier = alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        return Float32(2.0) * multiplier * val
    elif objective == GBDT_OBJ_TWEEDIE:
        var der = t * identical_exp((Float32(1.0) - alpha) * p)
        var delta = identical_exp((Float32(2.0) - alpha) * p)
        return der - delta
    elif objective == GBDT_OBJ_HUBER:
        var diff = t - p
        if abs(diff) < alpha:
            return diff
        return alpha if diff > Float32(0.0) else -alpha
    return Float32(0.0)


def _target_der2(objective: Int, t: Float32, p: Float32, alpha: Float32) -> Float32:
    """`target_der2` (`pointwise_targets.mojo:331-413`)."""
    if objective == GBDT_OBJ_RMSE:
        return Float32(1.0)
    elif (
        objective == GBDT_OBJ_QUANTILE or objective == GBDT_OBJ_MAE
        or objective == GBDT_OBJ_LOGLINQUANTILE or objective == GBDT_OBJ_MAPE
    ):
        return Float32(0.0)
    elif objective == GBDT_OBJ_POISSON:
        return identical_exp(p)
    elif objective == GBDT_OBJ_LQ:
        var abs_loss = abs(t - p)
        if alpha >= Float32(2.0):
            return alpha * (alpha - Float32(1.0)) * identical_pow(abs_loss, alpha - Float32(2.0))
        return Float32(1.0)
    elif objective == GBDT_OBJ_EXPECTILE:
        var val = t - p
        var multiplier = alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        return Float32(2.0) * multiplier
    elif objective == GBDT_OBJ_TWEEDIE:
        var der2 = t * identical_exp((Float32(1.0) - alpha) * p) * (Float32(1.0) - alpha)
        var delta = identical_exp((Float32(2.0) - alpha) * p) * (Float32(2.0) - alpha)
        return -der2 + delta
    elif objective == GBDT_OBJ_HUBER:
        var diff = t - p
        if abs(diff) < alpha:
            return Float32(1.0)
        return Float32(0.0)
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


def _loss_row(objective: Int, t: Float32, p: Float32, alpha: Float32) -> _LossRow:
    """One in-range thread of the kernel the objective reaches, unit weight:
    `pointwise_target_kernel` (`:553-663`) or, for CrossEntropy and Logloss,
    `cross_entropy_kernel[has_border]` (`:947-1054`; `alpha` carries
    Logloss's border)."""
    var weight = Float32(1.0)
    if objective == GBDT_OBJ_CROSSENTROPY or objective == GBDT_OBJ_LOGLOSS:
        var exp_val = identical_exp(p)
        var prob = Float32(1.0)
        if isfinite(exp_val):
            prob = exp_val / (Float32(1.0) + exp_val)
        prob = max(min(prob, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
        var c = t
        if objective == GBDT_OBJ_LOGLOSS:
            c = Float32(1.0) if t > alpha else Float32(0.0)
        var direction = ftz(c - prob)
        var scale = ftz(prob * (Float32(1.0) - prob))
        var log_exp_val_plus_one = p
        if isfinite(exp_val):
            log_exp_val_plus_one = identical_log(Float32(1.0) + exp_val)
        return _LossRow(
            ftz(weight * direction), abs(weight * direction),
            ftz(weight * scale), weight * (c * p - log_exp_val_plus_one),
        )
    var der = ftz(weight * _target_der(objective, t, p, alpha))
    return _LossRow(
        der, abs(der), ftz(weight * _target_der2(objective, t, p, alpha)),
        -weight * _target_score(objective, t, p, alpha),
    )


def _loss_search_pass(
    loss: GbdtHostLoss,
    targets: List[Float32],
    cursor: List[Float32],
    n_rows: Int,
    mut stats: List[Float32],
    mut fv_partials: List[Float32],
    mut mag_partials: List[Float32],
):
    """`launch_approximate[False]` with `compute_fv` and
    `compute_magnitudes` set: plane 0 the unit weight, plane 1 the flushed
    weighted derivative, one score partial and two magnitude partials per
    256-thread block through the halving tree."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_w = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var r = _loss_row(loss.objective, targets[i], cursor[i], loss.kernel_alpha)
                stats[i] = Float32(1.0)
                stats[n_rows + i] = r.der
                s_score[t] = r.score
                s_w[t] = Float32(1.0)
                s_g[t] = r.der_abs
        fv_partials[b] = _halving_fold(s_score)
        mag_partials[2 * b] = _halving_fold(s_w)
        mag_partials[2 * b + 1] = _halving_fold(s_g)


def _loss_value(
    loss: GbdtHostLoss, targets: List[Float32], cursor: List[Float32], n_rows: Int
) -> Float32:
    """The final learn loss pass, folded by `deterministic_sum_lanes_kernel[1]`."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var fv_partials = List[Float32](length=blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                s_score[t] = _loss_row(
                    loss.objective, targets[i], cursor[i], loss.kernel_alpha
                ).score
        fv_partials[b] = _halving_fold(s_score)
    return _deterministic_sum_lanes(fv_partials, 1, blocks)[0]


# ===========================================================================
# THE BOOTSTRAP (`gbdt/gpu_util/kernel/bootstrap.mojo`)
# ===========================================================================


def gbdt_bootstrap_seeds(base_seed: UInt64) -> List[UInt64]:
    """`create_bootstrap_seeds` (`bootstrap.mojo:249-275`): splitmix64."""
    var seeds = List[UInt64](capacity=GBDT_BOOT_SEEDS)
    var x = base_seed
    for _ in range(GBDT_BOOT_SEEDS):
        x += UInt64(0x9E3779B97F4A7C15)
        var z = x
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        seeds.append(z)
    return seeds^


def _bootstrap_pass(
    kind: Int,
    mut seeds: List[UInt64],
    mut stats: List[Float32],
    n_rows: Int,
    param: Float32,
) raises -> Tuple[Float32, Float32]:
    """`launch_bootstrap` + `bootstrap_kernel` (`bootstrap.mojo:118-246`,
    `:292-348`) at two stat planes, then `deterministic_sum_lanes_kernel[2]`
    over the block magnitudes. Returns the two folded magnitudes."""
    var by_rows = (n_rows + GBDT_BOOT_BLOCK - 1) // GBDT_BOOT_BLOCK
    var blocks = GBDT_BOOT_SEEDS // GBDT_BOOT_BLOCK
    if by_rows < blocks:
        blocks = by_rows
    if blocks < 1:
        blocks = 1
    var stride = blocks * GBDT_BOOT_BLOCK
    var mag_part = List[Float32](length=2 * blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_w = List[Float32](length=GBDT_BOOT_BLOCK, fill=Float32(0.0))
        var s_g = List[Float32](length=GBDT_BOOT_BLOCK, fill=Float32(0.0))
        for tid in range(GBDT_BOOT_BLOCK):
            var gid = b * GBDT_BOOT_BLOCK + tid
            var s = seeds[gid]
            var mag_w = Float32(0.0)
            var mag_g = Float32(0.0)
            var i = gid
            while i < n_rows:
                var bw: Float32
                if kind == GBDT_BOOT_BERNOULLI:
                    var draw = next_uniform_f(s)
                    s = draw[1]
                    bw = Float32(1.0) if draw[0] < param else Float32(0.0)
                elif kind == GBDT_BOOT_POISSON:
                    var draw = next_poisson_f(s, param)
                    s = draw[1]
                    bw = draw[0]
                else:
                    raise Error("gbdt host: bootstrap kind " + String(kind) + " is not restated")
                var w = stats[i] * bw
                stats[i] = w
                mag_w += abs(w)
                var g = stats[n_rows + i] * bw
                stats[n_rows + i] = g
                mag_g += abs(g)
                i += stride
            seeds[gid] = s
            s_w[tid] = mag_w
            s_g[tid] = mag_g
        mag_part[2 * b] = _halving_fold(s_w)
        mag_part[2 * b + 1] = _halving_fold(s_g)
    var mags = _deterministic_sum_lanes(mag_part, 2, blocks)
    return (mags[0], mags[1])


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
):
    """`write_value_and_first_derivatives`' single-dimensional arm
    (`pointwise_oracle.mojo:458-573`): the estimation planes, one score
    partial per 256-thread block, `compute_partition_stats` per leaf, the
    Hessian plus lambda, and the HOST Float32 fold of the value partials."""
    var blocks = (n_rows + GBDT_MSE_BLOCK - 1) // GBDT_MSE_BLOCK
    var stats = List[Float32](length=2 * n_rows, fill=Float32(0.0))
    var fv = List[Float32](length=blocks, fill=Float32(0.0))
    for b in range(blocks):
        var s_score = List[Float32](length=GBDT_MSE_BLOCK, fill=Float32(0.0))
        for t in range(GBDT_MSE_BLOCK):
            var i = b * GBDT_MSE_BLOCK + t
            if i < n_rows:
                var r = _loss_row(loss.objective, g_target[i], g_cursor[i], loss.kernel_alpha)
                stats[i] = r.der
                stats[n_rows + i] = r.der2
                s_score[t] = r.score
        fv[b] = _halving_fold(s_score)
    gradient.clear()
    cached_der2.clear()
    for leaf in range(len(sizes)):
        gradient.append(
            Float64(_partition_stat(stats, n_rows, 0, offsets[leaf], sizes[leaf]))
        )
        cached_der2.append(
            Float64(_partition_stat(stats, n_rows, 1, offsets[leaf], sizes[leaf]))
            + lambda_reg
        )
    var fv32 = Float32(0.0)
    for b in range(blocks):
        fv32 += fv[b]
    value = Float64(fv32)


def _second_derivatives(
    method: Int, weights_cpu: List[Float64], lambda_reg: Float64,
    cached_der2: List[Float64],
) -> List[Float64]:
    """`write_second_derivatives` (`pointwise_oracle.mojo:341-431`): the
    leaf weight plus lambda under Gradient, the cached Hessian under Newton."""
    if method == GBDT_LEAF_GRADIENT:
        var out = List[Float64]()
        for leaf in range(len(weights_cpu)):
            out.append(weights_cpu[leaf] + lambda_reg)
        return out^
    return cached_der2.copy()


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
) raises -> List[Float32]:
    """`newton_like_walker_estimate` with AnyImprovement
    (`descent_helpers.mojo:198-285`, `step_estimator.mojo:50-72`), the TWIN
    of `gbdt_oracle.mojo::_estimate_leaves` with the method's second
    derivatives."""
    var n_leaves = len(sizes)
    var lambda_reg = Float64(l2_leaf_reg)
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
            targets_rows, List[Float32](), False, g_cursor, row_index,
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
            n_rows, lambda_reg, cur_value, cur_grad, cached_der2,
        )
    else:
        _loss_eval(
            loss, g_target, g_cursor, offsets, sizes, n_rows, lambda_reg,
            cur_value, cur_grad, cached_der2,
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
                    targets_rows, List[Float32](), False, g_cursor, row_index,
                    group_sizes, yeti_tasks.value(), yeti_rng.next_uniform_l(),
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
                    cached_der2,
                )
            else:
                _loss_eval(
                    loss, g_target, g_cursor, offsets, sizes, n_rows,
                    lambda_reg, next_value, next_grad, cached_der2,
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
    for b in range(n_blocks):
        var s_val = List[Float32](length=B, fill=Float32(0.0))
        var s_flg = List[Int](length=B, fill=0)
        for tid in range(B):
            var i = b * B + tid
            if i < n:
                s_val[tid] = values[i]
                s_flg[tid] = 1 if flags[i] else 0
        var d = 1
        while d < B:
            var lv = List[Float32](length=B, fill=Float32(0.0))
            var lf = List[Int](length=B, fill=0)
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
                scanned[i] = s_val[tid]
                has_flag[i] = s_flg[tid] != 0
        block_sums[b] = s_val[B - 1]
        block_flags[b] = s_flg[B - 1] != 0
    var running = Float32(0.0)
    for b in range(n_blocks):
        var v = block_sums[b]
        var f = block_flags[b]
        block_sums[b] = running
        if f:
            running = v
        else:
            running = running + v
    for i in range(n):
        if not has_flag[i]:
            scanned[i] = scanned[i] + block_sums[i // B]
    return scanned^


def _exact_estimate(
    loss: GbdtHostLoss,
    g_target: List[Float32],
    g_cursor: List[Float32],
    offsets: List[Int],
    sizes: List[Int],
    weights_cpu: List[Float64],
    n_rows: Int,
) raises -> List[Float32]:
    """`BinOptimizedOracle.estimate_exact` (`pointwise_oracle.mojo:433-503`)
    over `compute_exact_approx` (`leaves_estimation_helper.mojo`), unit row
    weights: the residual, the MAPE quotient, the stable sort of each leaf
    on bits [10, 32) of the sortable key, the prefix weights, the need
    weights and the binary search, then `RegularizeImpl`."""
    var n_leaves = len(sizes)
    var residuals = List[Float32](length=n_rows, fill=Float32(0.0))
    var weights = List[Float32](length=n_rows, fill=Float32(1.0))
    for pos in range(n_rows):
        residuals[pos] = ftz(g_target[pos] - g_cursor[pos])
    if loss.objective == GBDT_OBJ_MAPE:
        for pos in range(n_rows):
            var delta = max(Float32(1.0), abs(residuals[pos]))
            weights[pos] = ftz(Float32(1.0) / delta)
    elif loss.objective != GBDT_OBJ_MAE and loss.objective != GBDT_OBJ_QUANTILE:
        raise Error(
            "Only MAPE, MAE and Quantile are supported for Exact leaves"
            " estimation on GPU"
        )
    var ordered_targets = List[Float32](length=n_rows, fill=Float32(0.0))
    var ordered_weights = List[Float32](length=n_rows, fill=Float32(0.0))
    var flags = List[Bool](length=n_rows, fill=False)
    for leaf in range(n_leaves):
        var base = offsets[leaf]
        var size = sizes[leaf]
        if size <= 0:
            continue
        var keys = List[UInt64](capacity=size)
        for k in range(size):
            var key = _float_to_sortable(bitcast[DType.uint32](residuals[base + k]))
            keys.append((UInt64(key >> UInt32(GBDT_EXACT_FIRST_BIT)) << UInt64(32)) | UInt64(k))
        sort(keys)
        for k in range(size):
            var src = base + Int(keys[k] & UInt64(0xFFFFFFFF))
            ordered_targets[base + k] = residuals[src]
            ordered_weights[base + k] = weights[src]
        flags[base] = True
    var prefix = _segmented_inclusive_scan(ordered_weights, flags, n_rows)

    var point = List[Float32](length=n_leaves, fill=Float32(0.0))
    var eps = Float32(1.1920929e-07)
    for leaf in range(n_leaves):
        var base = offsets[leaf]
        var size = sizes[leaf]
        if size <= 0:
            point[leaf] = Float32(0.0)
            continue
        var slab = List[Float32](length=GBDT_NEED_WEIGHTS_BLOCK, fill=Float32(0.0))
        for tid in range(GBDT_NEED_WEIGHTS_BLOCK):
            var total_sum = Float32(0.0)
            var idx = tid
            while idx < size:
                total_sum = ftz(total_sum + ordered_weights[base + idx])
                idx += GBDT_NEED_WEIGHTS_BLOCK
            slab[tid] = total_sum
        var need = ftz(_halving_fold(slab) * loss.estimator_alpha)
        var left = base
        var right = base + size - 1
        for _ in range(GBDT_QUANTILE_ITERATIONS):
            var middle = left + (right - left) // 2
            if prefix[middle] < need - eps:
                left = middle
            else:
                right = middle
        point[leaf] = ordered_targets[right]
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
) raises -> List[Float32]:
    """`_estimate_and_apply`'s estimate (`doc_parallel_boosting.mojo:
    698-797`): the gathers by the row index, then Exact or the walker."""
    var n_leaves = len(sizes)
    var g_target = List[Float32](length=n_rows, fill=Float32(0.0))
    var g_cursor = List[Float32](length=n_rows, fill=Float32(0.0))
    for pos in range(n_rows):
        g_target[pos] = targets[row_index[pos]]
        g_cursor[pos] = cursor[row_index[pos]]
    var bins = List[Int](length=n_rows, fill=0)
    for leaf in range(n_leaves):
        for k in range(sizes[leaf]):
            bins[offsets[leaf] + k] = leaf
    var weights_cpu = List[Float64]()
    for leaf in range(n_leaves):
        weights_cpu.append(Float64(sizes[leaf]))
    if loss.method == GBDT_LEAF_EXACT:
        return _exact_estimate(
            loss, g_target, g_cursor, offsets, sizes, weights_cpu, n_rows
        )
    return _walker_estimate(
        loss, g_target, g_cursor, bins, offsets, sizes, weights_cpu, n_rows,
        l2_leaf_reg, targets, row_index, group_sizes, pairs, yeti_tasks,
        yeti_seed,
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
    group_sizes: List[Int] = List[Int](),
    pair_winners: List[UInt32] = List[UInt32](),
    pair_losers: List[UInt32] = List[UInt32](),
    pair_weights: List[Float32] = List[Float32](),
    start: Float64 = 0.0,
) raises -> GbdtHostModel:
    """`train` then `fit_with_test` on the covered configurations (see the
    module docstring). `group_sizes` is read by QueryRMSE alone, already
    resolved to their `TWithoutQueriesGrouping` rule by the binding."""
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
        var ordered: PairList
        if len(pair_winners) > 0:
            ordered = order_pairs_by_winner(
                PairList(pair_winners.copy(), pair_losers.copy(), pair_weights.copy()),
                sizes_u32, n_rows,
            )
        else:
            ordered = order_pairs_by_winner(
                generate_pairs(sizes_u32, y, List[Float32]()), sizes_u32, n_rows
            )
        var hp = host_pairs(ordered.winners, ordered.losers, ordered.weights, n_rows)
        loss_norm = hp.prep.total
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
    for b in range(len(blocks)):
        if blocks[b].policy == POLICY_BINARY:
            raise Error(
                "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a"
                " feature with exactly one border (the BinaryFeatures"
                " histogram policy, feature "
                + String(blocks[b].feature_ids[0])
                + "); the gbdt host binding restates the half-byte and"
                " one-byte policies only (gbdt/host/gbdt_oracle.mojo)"
            )
    var cindex = _binarize_columns(x_colmajor, n_rows, n_features, grid, layout)
    var hist_cells = layout.hist_cells

    var bf_feature = List[Int](length=hist_cells, fill=0)
    var bf_bin = List[Int](length=hist_cells, fill=0)
    for f in range(n_features):
        ref lf = layout.features[f]
        for b in range(Int(lf.folds)):
            bf_feature[Int(lf.first_fold_index) + b] = f
            bf_bin[Int(lf.first_fold_index) + b] = b

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
    if pairs.__bool__():
        fv_blocks = pairs.value().blocks()
    var fv_part = List[Float32](length=fv_blocks, fill=Float32(0.0))
    var mag_part = List[Float32](length=2 * mse_blocks, fill=Float32(0.0))
    var bootstrap_on = loss.bootstrap_kind >= 0
    var seeds = List[UInt64]()
    if bootstrap_on:
        seeds = gbdt_bootstrap_seeds(params.random_seed)

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
                y, List[Float32](), False, cursor, group_sizes,
                yeti_tasks.value(), yeti_rand.next_uniform_l(), 10,
                Float32(0.85), n_rows, stats, fv_part, mag_part,
            )
        elif loss.objective == GBDT_OBJ_PAIR_LOGIT:
            pair_logit_search_pass(
                pairs.value(), cursor, n_rows, stats, fv_part, mag_part
            )
        elif loss.objective == GBDT_OBJ_QUERY_RMSE:
            query_rmse_search_pass(
                y, cursor, group_sizes, n_rows, stats, fv_part, mag_part
            )
        else:
            _loss_search_pass(loss, y, cursor, n_rows, stats, fv_part, mag_part)
        var fv = _deterministic_sum_lanes(fv_part, 1, fv_blocks)[0]
        var fixed_scale: Float32
        if bootstrap_on:
            var bm = _bootstrap_pass(
                loss.bootstrap_kind, seeds, stats, n_rows, loss.bootstrap_param
            )
            fixed_scale = _choose_scale_from_magnitudes(bm[0], bm[1], n_rows)
        else:
            var mags = _deterministic_sum_lanes(mag_part, 2, mse_blocks)
            fixed_scale = _choose_scale_from_magnitudes(mags[0], mags[1], n_rows)

        # ---- `run_tree_layout_traced`, every level (TWIN: the loop in
        # `gbdt_oracle.mojo::gbdt_host_fit`; edit both) ----
        var row_index = List[Int](length=n_rows, fill=0)
        for r in range(n_rows):
            row_index[r] = r
        var p_off = List[Int](length=max_leaves, fill=0)
        var p_sz = List[Int](length=max_leaves, fill=0)
        p_sz[0] = n_rows
        var hist = List[Float32](
            length=max_leaves * 2 * hist_cells, fill=Float32(0.0)
        )
        var ids_compute = List[Int](length=max_leaves, fill=0)
        var sub_from = List[Int](length=max_leaves, fill=0)
        var sub_what = List[Int](length=max_leaves, fill=0)
        var winners_score = List[Float32]()
        var winners_bf = List[UInt32]()
        var n_live = 1
        for depth in range(max_depth):
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
            for b in range(len(blocks)):
                ref blk = blocks[b]
                var total = 0
                for k in range(blk.count()):
                    total += Int(blk.folds[k])
                if blk.policy == POLICY_HALF_BYTE:
                    _half_byte_block(
                        blk, block_first_bin, hist_cells, compute, depth,
                        p_off, p_sz, row_index, stats, cindex, n_rows,
                        fixed_scale, hist,
                    )
                elif blk.policy == POLICY_ONE_BYTE:
                    _one_byte_block(
                        blk, block_first_bin, hist_cells, compute, p_off, p_sz,
                        row_index, stats, cindex, layout, n_rows, fixed_scale,
                        hist,
                    )
                block_first_bin += total

            for j in range(len(compute)):
                var slot = compute[j]
                for z in range(2):
                    for f in range(n_features):
                        ref cf = layout.features[f]
                        var folds = Int(cf.folds)
                        if cf.one_hot_feature or folds <= 1:
                            continue
                        var base = (
                            slot * 2 * hist_cells + z * hist_cells
                            + Int(cf.first_fold_index)
                        )
                        var running = Float32(0.0)
                        for i in range(folds):
                            running = ftz(running + hist[base + i])
                            hist[base + i] = running

            if planned and half > 0:
                for j in range(half):
                    var from_slot = sub_from[j]
                    var what_slot = sub_what[j]
                    for z in range(2):
                        var from_base = from_slot * 2 * hist_cells + z * hist_cells
                        var what_base = what_slot * 2 * hist_cells + z * hist_cells
                        for bf in range(hist_cells):
                            var new_val = ftz(hist[from_base + bf] - hist[what_base + bf])
                            if z == 0:
                                new_val = max(new_val, Float32(0.0))
                            hist[from_base + bf] = new_val

            var part_stats = List[Float32](length=2 * n_live, fill=Float32(0.0))
            for i in range(n_live):
                part_stats[2 * i] = _partition_stat(stats, n_rows, 0, p_off[i], p_sz[i])
                part_stats[2 * i + 1] = _partition_stat(stats, n_rows, 1, p_off[i], p_sz[i])

            var best_gain = -GBDT_FLOAT32_MAX
            var best_bin = GBDT_SENTINEL
            for bf in range(hist_cells):
                var gain = _cosine_gain(
                    hist, hist_cells, part_stats, n_live, bf, params.l2_leaf_reg
                )
                if gain > best_gain:
                    best_gain = gain
                    best_bin = UInt32(bf)
            if best_bin != GBDT_SENTINEL:
                winners_score.append(ftz(best_gain))
            else:
                winners_score.append(ftz(-GBDT_FLOAT32_MAX))
            winners_bf.append(best_bin)
            var bf_split = 0
            if best_bin != GBDT_SENTINEL:
                bf_split = Int(best_bin)

            var split_f = bf_feature[bf_split] if hist_cells > 0 else 0
            var split_b = bf_bin[bf_split] if hist_cells > 0 else 0
            ref sfeat = layout.features[split_f]
            var new_rows = row_index.copy()
            var new_stats = stats.copy()
            for i in range(n_live):
                var off = p_off[i]
                var sz = p_sz[i]
                var zeros = List[Int]()
                var ones = List[Int]()
                for k in range(sz):
                    var row = row_index[off + k]
                    var word = cindex[Int(sfeat.offset) * n_rows + row]
                    var feature_val = word & (sfeat.mask << sfeat.shift)
                    var value = UInt32(split_b) << sfeat.shift
                    var goes_right: Bool
                    if sfeat.one_hot_feature:
                        goes_right = feature_val == value
                    else:
                        goes_right = feature_val > value
                    if goes_right:
                        ones.append(k)
                    else:
                        zeros.append(k)
                var dst = 0
                for k in range(len(zeros)):
                    var src = off + zeros[k]
                    new_rows[off + dst] = row_index[src]
                    new_stats[off + dst] = stats[src]
                    new_stats[n_rows + off + dst] = stats[n_rows + src]
                    dst += 1
                for k in range(len(ones)):
                    var src = off + ones[k]
                    new_rows[off + dst] = row_index[src]
                    new_stats[off + dst] = stats[src]
                    new_stats[n_rows + off + dst] = stats[n_rows + src]
                    dst += 1
                var src_base = i * 2 * hist_cells
                var dst_base = (n_live + i) * 2 * hist_cells
                for c in range(2 * hist_cells):
                    hist[dst_base + c] = hist[src_base + c]
                var left_sz = len(zeros)
                var right_sz = sz - left_sz
                p_sz[i] = left_sz
                p_off[n_live + i] = off + left_sz
                p_sz[n_live + i] = right_sz
                var small = n_live + i
                var big = i
                if left_sz < right_sz:
                    small = i
                    big = n_live + i
                ids_compute[i] = small
                sub_from[i] = big
                sub_what[i] = small
            row_index = new_rows^
            stats = new_stats^
            n_live = n_live * 2

        # ---- the gates, post-tree, and the rollback ----
        var tree_features = List[Int]()
        var tree_bins = List[Int]()
        var grown = 0
        for d in range(max_depth):
            var best_score = winners_score[d]
            var best_bin_u = winners_bf[d]
            if best_bin_u == GBDT_SENTINEL or Int(best_bin_u) >= hist_cells:
                raise Error(
                    "All splits have infinite score. Probably, numerical"
                    " overflow occurs in loss function and/or split score"
                    " calculation. Try increasing l2_leaf_reg, and/or"
                    " decreasing learning_rate, etc."
                    " [level " + String(d) + ", live leaves "
                    + String(1 << d) + "]"
                )
            var cf_ = bf_feature[Int(best_bin_u)]
            var cb_ = bf_bin[Int(best_bin_u)]
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
        )
        if loss.objective == GBDT_OBJ_PAIR_LOGIT or loss.objective == GBDT_OBJ_YETI_RANK:
            # `MakeZeroAverage` (`doc_parallel_leaves_estimator.cpp:25-37`),
            # restated from `_estimate_and_apply`: minus the unweighted mean
            # over all `n_live` leaves, summed in double in leaf order.
            var zero_sum = Float64(0.0)
            var zero_weight = Float64(0.0)
            for i in range(len(estimated)):
                zero_sum += Float64(estimated[i])
                zero_weight += Float64(1.0)
            var zero_bias = Float64(0.0)
            if zero_weight > Float64(0.0):
                zero_bias = -zero_sum / zero_weight
            for i in range(len(estimated)):
                estimated[i] = Float32(Float64(estimated[i]) + zero_bias)
        for leaf in range(n_live):
            for k in range(sizes[leaf]):
                var row = row_index[offsets[leaf] + k]
                cursor[row] = identical_mul_add(estimated[leaf], lr, cursor[row])
        for i in range(grown):
            split_features.append(tree_features[i])
            split_bins.append(tree_bins[i])
        tree_split_offsets.append(len(split_features))
        for i in range(len(estimated)):
            model_leaves.append(estimated[i] * lr)
        tree_leaf_offsets.append(len(model_leaves))

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
        var q_fv = query_rmse_value(y, cursor, group_sizes, n_rows)
        losses.append(
            -Float64(_deterministic_sum_lanes(q_fv, 1, len(q_fv))[0]) / Float64(n_rows)
        )
    else:
        losses.append(-Float64(_loss_value(loss, y, cursor, n_rows)) / Float64(n_rows))
    return GbdtHostModel(
        grid.fold_counts.copy(), grid.borders.copy(), grid.nan_treatment.copy(),
        tree_split_offsets^, split_features^, split_bins^, tree_leaf_offsets^,
        model_leaves^, losses^, 0, False,
    )
