"""Pointwise objectives: value, first derivative, second derivative.

PORT OF `catboost/cuda/targets/kernel/pointwise_targets.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

## The two kernels, and their fork

`TPointwiseTargetsImpl::Approximate` (`pointwise_target_impl.h:307-358`)
sends every objective down exactly one of two paths:

    Logloss, CrossEntropy   ->  ApproximateCrossEntropy  ->  CrossEntropyImpl
    everything else         ->  ApproximatePointwise     ->  PointwiseTargetImpl

so this file holds two kernels, one per arm, and `objective_is_cross_entropy`
is that fork. `pointwise_target_kernel` is the generic one, comptime-
specialized on the objective where theirs is a host switch that instantiates
a template (`pointwise_targets.cu:447-519`); `cross_entropy_kernel` is the
other. Eleven trainable objectives reach them:

    RMSE  Quantile  MAE  LogLinQuantile  MAPE  Poisson  Lq  Expectile
    Tweedie  Huber                        -> pointwise_target_kernel
    Logloss  CrossEntropy                 -> cross_entropy_kernel

RMSE and the cross-entropy pair came first because RMSE's derivatives are
exact and its Newton step needs no line search, and because the cross-entropy
pair built the estimation chassis the rest of them ride on: their Logloss
default is Newton with TEN estimation iterations
(`catboost_options.cpp:157-164`), so its leaves are the estimator's job, not
this file's. The other nine arrived after, as `target_score` / `target_der` /
`target_der2` arms, because the chassis was already there.

THE OBJECTIVE DOES NOT DECIDE THE LEAF ALONE. Four of the nine have a
second derivative that is identically zero, and CatBoost gives those a
GRADIENT or an EXACT leaf estimator rather than a Newton one
(`catboost_options.cpp:113-131`, `:289-300`).
`gbdt/options/leaf_estimation_defaults.mojo` is where that table lives; a
reader who takes the derivative formula here as the whole of a loss will get
the leaf values wrong.

Note the SIGN on `functionValue`: the pointwise kernel accumulates the
NEGATIVE score and the cross-entropy kernel accumulates the POSITIVE one,
because everything downstream maximizes and their `CrossEntropy::Score` is
already a log-likelihood. Both are theirs. Flipping either here would
silently invert every early-stopping comparison later.
"""

from std.atomic import Atomic
from std.math import exp, isfinite, log
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.block import sum as block_sum
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: The mode this build compiles against; see `mojo_only/numerics.mojo`. Same
#: declaration as the histogram kernels', and `pinned_block_sum` below is the
#: one reader in this file.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE

#: the boosting loop launches this kernel at 256 threads; the reduce below
#: needs the block size at comptime.
comptime MSE_BLOCK_SIZE = 256


@always_inline
def pinned_block_sum[block_size: Int](value: Float32) -> Float32:
    """The within-block float sum, with a MACHINE-INDEPENDENT fold shape
    under `NUMERIC_IDENTICAL`.

    IDENTITY_PATHS row 8's remainder (DEVIATION 251). The per-block
    partials that `deterministic_sum_lanes_kernel` folds are themselves
    produced by MAX's `block.sum`, whose internal cross-lane fold runs at
    the HARDWARE'S warp width -- 32 on Apple and NVIDIA, 64 on AMD's CDNA
    wavefront -- so the partial's last bits differ on the AMD column even
    with the partial COUNTS pinned and the cross-block combine fixed.

    Under `IDENTICAL` this is therefore a shared-memory halving tree at
    exactly `block_size` lanes, NO warp primitives -- the same fold shape
    `deterministic_sum_lanes_kernel` and `bootstrap.mojo`'s magnitude tail
    already use -- so the fold has ONE shape on every vendor. Under `FAST`
    it is the library call unchanged, bit for bit.

    CONTRACT, same as the `block.sum` call sites already hold: EVERY
    thread of the block must call this (out-of-range threads contribute
    zero), and only thread 0's return is meaningful. `block_size` must be
    a power of two, so the halving fold is exact. The trailing `barrier()`
    protects the shared slab across back-to-back calls (the magnitude
    sites call this twice in a row).
    """
    comptime if BUILD_MODE == NUMERIC_IDENTICAL:
        var tid = Int(thread_idx.x)
        var red = stack_allocation[
            block_size,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        red[tid] = value
        barrier()
        var step = block_size // 2
        while step > 0:
            if tid < step:
                red[tid] = red[tid] + red[tid + step]
            barrier()
            step //= 2
        var total = red[0]
        barrier()
        return total
    else:
        return block_sum[block_size=block_size](value)

#: the pointwise objectives this port trains. Their `ELossFunction`
#: spellings: RMSE dispatches `TRmseTarget`; Logloss and CrossEntropy both
#: dispatch `ApproximateCrossEntropy` (`pointwise_target_impl.h:333-345`),
#: differing only in `UseBorder()` -- Logloss thresholds the target at the
#: `GetLogLossBorder` default 0.5, CrossEntropy takes it as a probability.
comptime OBJECTIVE_RMSE = 0
comptime OBJECTIVE_LOGLOSS = 1
comptime OBJECTIVE_CROSSENTROPY = 2
comptime OBJECTIVE_QUANTILE = 3
comptime OBJECTIVE_MAE = 4
comptime OBJECTIVE_LOGLINQUANTILE = 5
comptime OBJECTIVE_MAPE = 6
comptime OBJECTIVE_POISSON = 7
comptime OBJECTIVE_LQ = 8
comptime OBJECTIVE_EXPECTILE = 9
comptime OBJECTIVE_TWEEDIE = 10
comptime OBJECTIVE_HUBER = 11

#: MultiClass reaches NEITHER kernel in this file. It is here so that one
#: enumeration covers every loss the trainer accepts, and so that
#: `launch_approximate` can refuse it by name rather than by falling into
#: the pointwise arm: their `Approximate` fork
#: (`pointwise_target_impl.h:307-358`) belongs to `TPointwiseTargetsImpl`,
#: and MultiClass is `TMultiClassificationTargets`
#: (`targets/multiclass_targets.h`) with its own der calcer and its own
#: kernels in `targets/kernel/multilogit.cu`.
comptime OBJECTIVE_MULTICLASS = 12

#: `MultiClassOneVsAll`: `numClasses` INDEPENDENT logistic regressions, one
#: per class, with NO pinned class and a DIAGONAL Hessian
#: (`multiclass_targets.h:118-134`). Reaches
#: `one_vs_all_val_and_first_der_kernel`, not either kernel in this file.
comptime OBJECTIVE_MULTICLASS_OVA = 13

#: `NumErrors` is in their kernel switch (`pointwise_targets.cu:497-501`)
#: and is deliberately NOT here: `TPointwiseTargetsImpl::Init`
#: (`pointwise_target_impl.h:259-299`) has no `NumErrors` case, so its
#: `default:` arm throws "Unsupported loss function" before training can
#: reach it. It is a METRIC that borrows the target kernel, and this port
#: has no metric path for it to arrive by. Porting the arm would leave a
#: branch no caller reaches, which PORTING_RULES 3 forbids.


def objective_from_name(name: String) raises -> Int:
    """Their `ELossFunction` spelling -> the comptime tag above.

    The spellings are theirs exactly (`enums.h`, `ELossFunction`), because
    a user who reads CatBoost's documentation must be able to paste the
    name across. Anything not listed raises rather than falling back to a
    default, mirroring their `Init` (`pointwise_target_impl.h:295-297`).
    """
    if name == "RMSE":
        return OBJECTIVE_RMSE
    if name == "Logloss":
        return OBJECTIVE_LOGLOSS
    if name == "CrossEntropy":
        return OBJECTIVE_CROSSENTROPY
    if name == "Quantile":
        return OBJECTIVE_QUANTILE
    if name == "MAE":
        return OBJECTIVE_MAE
    if name == "LogLinQuantile":
        return OBJECTIVE_LOGLINQUANTILE
    if name == "MAPE":
        return OBJECTIVE_MAPE
    if name == "Poisson":
        return OBJECTIVE_POISSON
    if name == "Lq":
        return OBJECTIVE_LQ
    if name == "Expectile":
        return OBJECTIVE_EXPECTILE
    if name == "Tweedie":
        return OBJECTIVE_TWEEDIE
    if name == "Huber":
        return OBJECTIVE_HUBER
    if name == "MultiClass":
        return OBJECTIVE_MULTICLASS
    if name == "MultiClassOneVsAll":
        return OBJECTIVE_MULTICLASS_OVA
    raise Error(
        "unknown loss '" + name + "': this port trains RMSE, Logloss,"
        " CrossEntropy, Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq,"
        " Expectile, Tweedie, Huber, MultiClass and MultiClassOneVsAll"
    )


def objective_name(objective: Int) -> String:
    """The inverse, for messages and for the benchmark's path line."""
    if objective == OBJECTIVE_RMSE:
        return String("RMSE")
    if objective == OBJECTIVE_LOGLOSS:
        return String("Logloss")
    if objective == OBJECTIVE_CROSSENTROPY:
        return String("CrossEntropy")
    if objective == OBJECTIVE_QUANTILE:
        return String("Quantile")
    if objective == OBJECTIVE_MAE:
        return String("MAE")
    if objective == OBJECTIVE_LOGLINQUANTILE:
        return String("LogLinQuantile")
    if objective == OBJECTIVE_MAPE:
        return String("MAPE")
    if objective == OBJECTIVE_POISSON:
        return String("Poisson")
    if objective == OBJECTIVE_LQ:
        return String("Lq")
    if objective == OBJECTIVE_EXPECTILE:
        return String("Expectile")
    if objective == OBJECTIVE_TWEEDIE:
        return String("Tweedie")
    if objective == OBJECTIVE_HUBER:
        return String("Huber")
    if objective == OBJECTIVE_MULTICLASS:
        return String("MultiClass")
    if objective == OBJECTIVE_MULTICLASS_OVA:
        return String("MultiClassOneVsAll")
    return String("<unknown>")


def objective_is_cross_entropy(objective: Int) -> Bool:
    """Which of their TWO kernels this objective reaches.

    `pointwise_target_impl.h:333-345` sends Logloss and CrossEntropy to
    `ApproximateCrossEntropy` and everything else to `ApproximatePointwise`
    (`:346-357`). That fork is the only reason this file holds two kernels.
    """
    return (
        objective == OBJECTIVE_LOGLOSS
        or objective == OBJECTIVE_CROSSENTROPY
    )


# =========================================================================
# THE OBJECTIVES: `Score`, `Der`, `Der2`, one comptime arm each.
#
# PORT OF the nine objective structs of `pointwise_targets.cu:11-240`.
# Theirs are C++ structs with three `__device__ __forceinline__` methods,
# instantiated by `PointwiseTargetKernel`'s switch (`:447-519`) and passed
# BY VALUE into the one generic kernel. Mojo has no zero-cost struct-by-
# value kernel argument of that shape, so the switch moves to a comptime
# parameter and the three methods become three comptime-dispatched
# functions. THE ARITHMETIC IS TRANSCRIBED, term for term, including the
# order of operations, which is why each arm carries its own line cite.
#
# THEIR ONE FLOAT PARAMETER. Their kernel takes a single `float alpha`
# (`:451`) and every parameterized objective reads it: Quantile/MAE/
# LogLinQuantile/Expectile as the quantile level, Lq as `q`, Huber as
# `delta`, Tweedie as `variancePower`. This port keeps that one slot.
# =========================================================================


@always_inline
def _target_sign(x: Float32) -> Float32:
    """Their `sign` (`pointwise_targets.cu:193-195`).

    NOTE `sign(0) == -1`, because theirs is `x > 0 ? 1.0f : -1.0f` with no
    zero arm. Lq's `Der` multiplies by it, so an exactly-zero residual
    takes the negative branch in their arithmetic and must here too.
    """
    return Float32(1.0) if x > Float32(0.0) else Float32(-1.0)


@always_inline
def target_score[objective: Int](
    t: Float32, p: Float32, alpha: Float32
) -> Float32:
    """`TTarget::Score(target, prediction)`, per objective."""

    @parameter
    if objective == OBJECTIVE_RMSE:
        # `TRmseTarget::Score` (`:180-182`)
        return (t - p) * (t - p)
    elif objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_MAE:
        # `TQuantileTarget::Score` (`:18-22`). MAE IS Quantile at
        # alpha 0.5: `Init` sets `Alpha = 0.5` for MAE
        # (`pointwise_target_impl.h:272-275`) and the kernel switch falls
        # MAE through to `TQuantileTarget` (`:483-489`).
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        )
        return multiplier * val
    elif objective == OBJECTIVE_LOGLINQUANTILE:
        # `TLogLinQuantileTarget::Score` (`:126-131`)
        var val = t - exp(p)
        var multiplier = (
            alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        )
        return val * multiplier
    elif objective == OBJECTIVE_MAPE:
        # `TMAPETarget::Score` (`:147-149`)
        return abs(t - p) / max(Float32(1.0), abs(t))
    elif objective == OBJECTIVE_POISSON:
        # `TPoissonTarget::Score` (`:164-166`)
        return exp(p) - t * p
    elif objective == OBJECTIVE_LQ:
        # `TLqTarget::Score` (`:204-207`); their `__powf`
        var abs_loss = abs(t - p)
        return abs_loss**alpha
    elif objective == OBJECTIVE_EXPECTILE:
        # `TExpectileTarget::Score` (`:102-106`)
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return multiplier * val * val
    elif objective == OBJECTIVE_TWEEDIE:
        # `TTweedieTarget::Score` (`:40-44`), `alpha` IS their
        # `VariancePower` -- see the deviation block on the kernel.
        var val = (
            -t
            * exp((Float32(1.0) - alpha) * p)
            / (Float32(1.0) - alpha)
        )
        var delta = exp((Float32(2.0) - alpha) * p) / (Float32(2.0) - alpha)
        return val + delta
    elif objective == OBJECTIVE_HUBER:
        # `THuberTarget::Score` (`:68-75`), `alpha` is their `Delta`
        var mismatch = abs(t - p)
        if mismatch < alpha:
            return Float32(0.5) * mismatch * mismatch
        return alpha * (mismatch - Float32(0.5) * alpha)
    else:
        return Float32(0.0)


@always_inline
def target_der[objective: Int](
    t: Float32, p: Float32, alpha: Float32
) -> Float32:
    """`TTarget::Der(target, prediction)`, per objective.

    SIGN CONVENTION, theirs: `Der` is the NEGATIVE gradient of the loss --
    `TRmseTarget::Der` is `target - prediction` (`:184-186`), not
    `prediction - target`. Everything downstream ascends. Each arm below
    keeps their spelling, so the convention is inherited rather than
    re-decided.
    """

    @parameter
    if objective == OBJECTIVE_RMSE:
        # `:184-186`
        return t - p
    elif objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_MAE:
        # `:24-27`
        var val = t - p
        return alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
    elif objective == OBJECTIVE_LOGLINQUANTILE:
        # `:133-136`
        var exp_pred = exp(p)
        if t - exp_pred > Float32(0.0):
            return alpha * exp_pred
        return -(Float32(1.0) - alpha) * exp_pred
    elif objective == OBJECTIVE_MAPE:
        # `:151-154`
        if t - p > Float32(0.0):
            return Float32(1.0) / max(Float32(1.0), abs(t))
        return Float32(-1.0) / max(Float32(1.0), abs(t))
    elif objective == OBJECTIVE_POISSON:
        # `:168-171`
        return t - exp(p)
    elif objective == OBJECTIVE_LQ:
        # `:209-213`
        var abs_loss = abs(t - p)
        var abs_loss_q = abs_loss ** (alpha - Float32(1.0))
        return alpha * _target_sign(t - p) * abs_loss_q
    elif objective == OBJECTIVE_EXPECTILE:
        # `:108-112`
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return Float32(2.0) * multiplier * val
    elif objective == OBJECTIVE_TWEEDIE:
        # `:46-50`
        var der = t * exp((Float32(1.0) - alpha) * p)
        var delta = exp((Float32(2.0) - alpha) * p)
        return der - delta
    elif objective == OBJECTIVE_HUBER:
        # `:77-84`
        var diff = t - p
        if abs(diff) < alpha:
            return diff
        return alpha if diff > Float32(0.0) else -alpha
    else:
        return Float32(0.0)


@always_inline
def target_der2[objective: Int](
    t: Float32, p: Float32, alpha: Float32
) -> Float32:
    """`TTarget::Der2(target, prediction)`, per objective.

    FOUR OF THESE RETURN A HARD ZERO -- Quantile, MAE, MAPE and
    LogLinQuantile (`:29-31`, `:138-140`, `:156-158`). That is not an
    omission: their second derivative is zero almost everywhere, which is
    exactly why `catboost_options.cpp:113-124` gives those losses a
    GRADIENT default and `:289-300` upgrades three of them to EXACT. A
    Newton step on them divides by `lambda` alone.
    """

    @parameter
    if objective == OBJECTIVE_RMSE:
        # `:188-190`
        return Float32(1.0)
    elif (
        objective == OBJECTIVE_QUANTILE
        or objective == OBJECTIVE_MAE
        or objective == OBJECTIVE_LOGLINQUANTILE
        or objective == OBJECTIVE_MAPE
    ):
        return Float32(0.0)
    elif objective == OBJECTIVE_POISSON:
        # `:173-175`
        return exp(p)
    elif objective == OBJECTIVE_LQ:
        # `:215-218`: BELOW q = 2 their second derivative is the constant
        # 1, not the true one. Copied, because it is what their Newton
        # step divides by.
        var abs_loss = abs(t - p)
        if alpha >= Float32(2.0):
            return (
                alpha
                * (alpha - Float32(1.0))
                * abs_loss ** (alpha - Float32(2.0))
            )
        return Float32(1.0)
    elif objective == OBJECTIVE_EXPECTILE:
        # `:114-118`
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return Float32(2.0) * multiplier
    elif objective == OBJECTIVE_TWEEDIE:
        # `:52-56`
        var der2 = (
            t
            * exp((Float32(1.0) - alpha) * p)
            * (Float32(1.0) - alpha)
        )
        var delta = (
            exp((Float32(2.0) - alpha) * p) * (Float32(2.0) - alpha)
        )
        return -der2 + delta
    elif objective == OBJECTIVE_HUBER:
        # `:86-93`. Theirs writes `-HUBER_DER2` where
        # `HUBER_DER2 = -1.0` (`:62`), i.e. 1.0; the double negation is
        # kept so a reader diffing against their file finds the same
        # expression.
        var diff = t - p
        if abs(diff) < alpha:
            return -Float32(HUBER_DER2)
        return Float32(0.0)
    else:
        return Float32(0.0)


#: `THuberTarget::HUBER_DER2` (`pointwise_targets.cu:62`). Their constant
#: is negative and every use negates it.
comptime HUBER_DER2 = -1.0


def pointwise_target_kernel[
    objective: Int,
    estimation: Bool = False,
    second_der_as_weights: Bool = False,
](
    relevs: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    alpha: Float32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`PointwiseTargetImpl<TTarget, BLOCK_SIZE>` (`pointwise_targets.cu:
    246-281`), the ONE generic kernel every non-cross-entropy pointwise
    loss reaches, writing straight into the stats planes.

    Their `PointwiseTargetKernel` (`:447-519`) is a switch that constructs
    a `TTarget` and calls `RunPointwiseTargetKernel<blockSize>` with it;
    here the switch IS the `objective` comptime parameter and the three
    methods are `target_score` / `target_der` / `target_der2` above.

    ## Which of their two MSE kernels this is, because they have two

    `MseImpl` (`pointwise_targets.cu:285-321`) is the obvious one to port
    and it is UNREACHED IN THEIR OWN TREE. Its only entry point is
    `MseTargetKernel` (`:415-428`) behind `ApproximateMse`
    (`targets/kernel.h:828`), and `ApproximateMse` has no caller anywhere
    in `catboost/`. This file used to cite it, and the citation was to
    dead code.

    What RMSE actually reaches is this generic kernel, dispatched at
    `case ELossFunction::RMSE: TRmseTarget target;` (`:496-500`) out of
    `PointwiseTargetKernel`, which `ApproximatePointwise`
    (`targets/kernel.h:840`) launches. **The two kernels compute the same
    three numbers**, term for term, which is why the RMSE port was
    numerically right while its citation was wrong.

    ================= DEVIATION BLOCK =================
    Theirs writes `der` and `der2` into two separate buffers and sums
    `functionValue` with a block reduce and an `atomicAdd`.

    Ours writes the two derivatives into the ONE `stats` buffer the
    histogram kernels already read, in the layout they already expect,
    IN TWO MODES, because their one kernel serves two callers whose
    outputs differ (the same two modes `cross_entropy_kernel` records):

      - `estimation=False`, the SEARCH: plane 1 is `weight * Der`, and
        plane 0 is keyed on `second_der_as_weights`, which is their
        `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
        (`greedy_search_helper.cpp:286-296`, `enum_helpers.cpp:829-846`):
        False (Cosine/L2, and Cosine is the shipped default) writes the
        WEIGHT -- their `weightsView.Copy(weightsForIndices)`
        (`pointwise_target_impl.h:203-213`); True (NewtonCosine/NewtonL2)
        writes `weight * Der2` -- their `StochasticDer` hands
        `&weightsView` to `Approximate` as the der2 output
        (`pointwise_target_impl.h:193-201`) and the kernel fills it as
        `der2[i] = weight * target.Der2(relev, val)`
        (`pointwise_targets.cu:268-269`), the sample weight FOLDED IN.
        No third plane is written in either mode; the search reads only
        two. For RMSE alone the two modes coincide bit for bit, because
        `TRmseTarget::Der2` returns `1.0f`.
      - `estimation=True`, the LEAVES ORACLE's `ApproximateAt`
        (`pointwise_oracle.cpp:73-78`, the rowSize==1 fast path): plane 0
        their `der` buffer and plane 1 their `der2`, which
        `WriteValueAndFirstDerivatives` reduces per bin and caches as the
        Newton gradient and Hessian diagonal. Magnitudes are not computed
        in this mode.

    DEVIATION 62: TWEEDIE'S `variance_power` REACHES THIS KERNEL, AND ON
    THEIR GPU IT DOES NOT. `TPointwiseTargetsImpl::Init` reads it into the
    member `VariancePower` (`pointwise_target_impl.h:288-291`) and NOTHING
    EVER READS THAT MEMBER AGAIN -- the only value handed to the kernel is
    `GetAlpha()` (`:151-166`, `:346-356`), and `Init`'s Tweedie case never
    sets `Alpha`, so it stays at its declared `0` (`:364`). Their GPU
    Tweedie therefore trains at `variancePower = 0` whatever the user
    asked for. Their CPU reads it correctly
    (`algo/tensor_search_helpers.cpp:308`).
    We thread the parameter, for two reasons that both have to hold: the
    arm we are measured against IS their CPU (their GPU cannot run on this
    machine at all, which is the entire thesis), and their CPU honours it;
    and a loss whose defining parameter is ignored is not the loss. The
    ARITHMETIC in `target_score`/`target_der`/`target_der2` is their GPU
    struct's, unchanged -- only the value of the parameter differs, and it
    differs toward their own CPU.

    DEVIATION 63: `alpha` IS A KERNEL ARGUMENT, as theirs is (`:451`), but
    the objective is a COMPTIME parameter where theirs is a runtime
    `ELossFunction` switch on the host that then instantiates a template.
    Same specialisation, moved one step earlier: theirs picks the template
    instantiation on the host, ours picks it at compile time. No
    arithmetic difference; one launch shape per objective instead of one.

    * PER-BLOCK PARTIALS instead of their block-reduce-plus-`atomicAdd`
      tail for `function_value` AND the two fixed-point plane magnitudes
      -- the 2026-08-21 determinism fix, same buffers, same
      `deterministic_sum_lanes_kernel` fold.
    * `exp` and `**` here are `std.math` / Mojo's operator, where theirs
      are a MIXTURE of CUDA fast-math and libm within a single file:
      `TPoissonTarget` and `TLogLinQuantileTarget` use `__expf`,
      `TTweedieTarget` uses `std::exp`, `TLqTarget::Score` uses `__powf`
      and its `Der`/`Der2` use `powf`. CatBoost accepts approximate
      transcendentals at this exact site, so the substitution is
      in-family, but the two arms' derivatives differ in last bits BY
      CONSTRUCTION and no check may expect bitwise der parity against a
      CatBoost fit.
    ===================================================

    ## The fixed-point scale, which is a NEW risk for the exponentials

    `plane_magnitudes` / `compute_magnitudes` have NO CATBOOST COUNTERPART,
    like the scale they feed: they are the two sums of absolute values --
    `plane_magnitudes[0] = sum |weight plane|`, `[1] = sum |der plane|` --
    that `mojo_only/fixed_point.choose_scale` is specified against,
    reduced here by the same shape as `functionValue`. `compute_magnitudes`
    stands in for a null-pointer test, exactly as `compute_fv` does.

    RMSE's der is a residual and Logloss's is bounded in [-1, 1]. POISSON,
    TWEEDIE AND LOGLINQUANTILE ARE EXPONENTIAL IN THE PREDICTION, so a few
    drifted rows can dominate `sum |der|` and drive the scale down until
    ordinary gradients quantize toward zero. CatBoost never meets this
    because it flushes histograms with a float `atomicAdd`, so there is no
    source to port an answer from. `mojo_only/pointwise_target_check.mojo`
    measures the surviving resolution per objective; read it before
    trusting a fit on one of the three.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)

    # their `(i < size) ? ... : 0` ternaries (`:255-257`): out-of-range
    # threads stay alive to take part in the reduce, contributing zero.
    var in_range = i < size
    var val = Float32(0.0)
    var relev = Float32(0.0)
    var weight = Float32(1.0)
    if in_range:
        val = predictions.unsafe_load(i)
        relev = relevs.unsafe_load(i)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)

    # their `ApproximateAt` path has no `secondDerAsWeights` flag; the
    # combination is a caller error, not a mode.
    comptime assert not (estimation and second_der_as_weights), (
        "second_der_as_weights is a SEARCH-mode flag: their StochasticDer"
        " takes it (`pointwise_target_impl.h:173-216`), their ApproximateAt"
        " does not"
    )

    # `der[i] = weight * target.Der(relev, val)` (`:265-266`)
    # `der2[i] = weight * target.Der2(relev, val)` (`:268-269`)
    var der = weight * target_der[objective](relev, val, alpha)

    # what SEARCH plane 0 holds, per `second_der_as_weights` -- see the
    # deviation block. The der2 arm is their `der2[i] = weight *
    # target.Der2(relev, val)` landing in `weightsView`.
    var plane0 = weight

    @parameter
    if second_der_as_weights:
        plane0 = weight * target_der2[objective](relev, val, alpha)

    if in_range:

        @parameter
        if estimation:
            stats.unsafe_store(i, der)
            stats.unsafe_store(
                size + i,
                weight * target_der2[objective](relev, val, alpha),
            )
        else:
            stats.unsafe_store(i, plane0)
            stats.unsafe_store(size + i, der)
            # no third plane in search mode; see the deviation block.

    # `if (functionValue) { tmpScores[tid] = -w * target.Score(...); ...
    # FastInBlockReduce; if (tid == 0) atomicAdd(functionValue, val); }`
    # (`pointwise_targets.cu:271-280`).
    # ============ DETERMINISM FIX, 2026-08-21 ============
    # These two reduces ENDED IN A FLOAT ATOMIC (their `atomicAdd`,
    # `:277`), so `functionValue` and, worse, the fixed-point scale's
    # magnitudes depended on block arrival order -- the same-seed-twice
    # bootstrap gate caught two fits differing, and the long-observed
    # rep-to-rep last-bit jitter of the loss column was exactly this.
    # Every block now STORES its partial at its own slot and
    # `deterministic_sum_lanes_kernel` folds them in one fixed order, so a
    # seeded fit is bit-reproducible end to end. `function_value` and
    # `plane_magnitudes` are therefore PER-BLOCK PARTIALS buffers
    # (`n_blocks` and `2 * n_blocks`), not scalars.
    #
    # NOTE THE SIGN: theirs accumulates the NEGATIVE score, because
    # everything downstream MAXIMIZES. Kept, because flipping it here
    # would silently invert every early-stopping comparison later. The
    # cross-entropy kernel below accumulates the POSITIVE score for the
    # same reason -- its `Score` already IS a log-likelihood.
    # =====================================================
    if compute_fv != Int32(0):
        var score = Float32(0.0)
        if in_range:
            score = -weight * target_score[objective](relev, val, alpha)
        var total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    # the fixed-point scale's inputs: sum of |plane| for both stat planes,
    # same reduce shape as `functionValue` above.
    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            # NO CATBOOST COUNTERPART (the fixed-point scale is ours):
            # the magnitudes bound the planes AS STORED, so plane 0's is
            # |weight * Der2| under `second_der_as_weights`.
            w_abs = abs(plane0)
            g_abs = abs(der)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x), w_total
            )
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x) + 1, g_total
            )


comptime REDUCE_LANES_BLOCK = 256


def deterministic_sum_lanes_kernel[
    lanes: Int
](
    partials: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """Fold interleaved per-block partials in ONE FIXED ORDER.

    NO CATBOOST COUNTERPART: they accept the float atomic's arrival-order
    nondeterminism in `functionValue`; this port cannot, because the
    magnitudes feed `fixed_scale` and the dithered histograms behind the
    bit-reproducibility claim. One block; thread `t` walks slots
    `t, t + 256, ...` in ascending order and the shared tree fold has one
    shape -- same inputs, same bits, every run.

    `partials` is `[i * lanes + lane]` for `count_in` blocks."""
    var tid = Int(thread_idx.x)
    var count = Int(count_in)

    var acc = InlineArray[Float32, lanes](fill=Float32(0.0))
    var i = tid
    while i < count:

        @parameter
        for lane in range(lanes):
            acc[lane] += partials.unsafe_load(i * lanes + lane)
        i += REDUCE_LANES_BLOCK

    var red = stack_allocation[
        lanes * REDUCE_LANES_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    @parameter
    for lane in range(lanes):
        red[lane * REDUCE_LANES_BLOCK + tid] = acc[lane]
    barrier()
    var step = REDUCE_LANES_BLOCK // 2
    while step > 0:
        if tid < step:

            @parameter
            for lane in range(lanes):
                red[lane * REDUCE_LANES_BLOCK + tid] = (
                    red[lane * REDUCE_LANES_BLOCK + tid]
                    + red[lane * REDUCE_LANES_BLOCK + tid + step]
                )
        barrier()
        step //= 2
    if tid == 0:

        @parameter
        for lane in range(lanes):
            dst.unsafe_store(lane, red[lane * REDUCE_LANES_BLOCK])


def cross_entropy_kernel[
    has_border: Bool,
    estimation: Bool = False,
    second_der_as_weights: Bool = False,
](
    target_classes: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    predictions: MutPointer[Float32, MutAnyOrigin],
    has_weights: Int32,
    border: Float32,
    stats: MutPointer[Float32, MutAnyOrigin],
    function_value: MutPointer[Float32, MutAnyOrigin],
    compute_fv: Int32,
    plane_magnitudes: MutPointer[Float32, MutAnyOrigin],
    compute_magnitudes: Int32,
):
    """`CrossEntropyImpl` (`pointwise_targets.cu:327-390`), the kernel
    Logloss and CrossEntropy actually reach: `pointwise_target_impl.h:333-345`
    dispatches both to `ApproximateCrossEntropy`, which launches this and
    nothing else (`targets/kernel.h:49`, `:876`). `has_border` is their
    `HAS_BORDER` template arm: Logloss thresholds the target at `border`
    (`GetLogLossBorder`, default 0.5, `pointwise_target_impl.h:285-287`);
    CrossEntropy takes the target as an already-soft class probability.

    Per element, THEIR arithmetic, term for term (`:353-360`):

        expVal = exp(val)                      // their __expf
        p  = clamp(isfinite ? expVal/(1+expVal) : 1, [1e-40, 1-1e-40])
        c  = has_border ? (targetClass > border) : targetClass
        der  = weight * (c - p)
        der2 = weight * (p * (1 - p))
        score += weight * (c*val - log(1+expVal))   // isfinite ? : w*c*val-val

    Note the SIGN convention matches `mse_kernel`: `score` is the
    log-LIKELIHOOD (their `tmpScore`, `:363`), so downstream maximizes, and
    the printed loss is its negation by the same reader that negates mse.

    ================= DEVIATION BLOCK =================
    Same three substitutions `mse_kernel` records, none new:

    * STATS PLANES instead of their separate output buffers, in TWO MODES,
      because their one kernel serves two callers whose outputs differ:
      - `estimation=False`, the SEARCH: plane 1 is `weight * der`, and
        plane 0 is keyed on `second_der_as_weights` exactly as the mse
        block lays out: False (Cosine/L2) writes the weight
        (`pointwise_target_impl.h:203-213`); True (NewtonCosine/NewtonL2)
        writes `weight * p * (1 - p)` -- their `der2[idx] = weight[j] *
        scale[j]` (`pointwise_targets.cu:373-375`) landing in
        `weightsView` (`pointwise_target_impl.h:193-201`). No third plane
        is written in either mode.
      - `estimation=True`, the LEAVES ORACLE's `ApproximateAt`
        (`pointwise_oracle.cpp:73-78`, the rowSize==1 "fast path" their
        pointwise losses take): plane 0 their `der` buffer
        (`weight * (c - p)`, `:370-372`) and plane 1 their `der2`
        (`weight * p * (1 - p)`, `:373-375`), which
        `WriteValueAndFirstDerivatives` reduces per bin and caches as the
        Newton gradient and Hessian diagonal. Magnitudes are not computed
        in this mode; the estimator's reduces run through
        `compute_partition_stats`, not the fixed-point histograms.
    * PER-BLOCK PARTIALS instead of their block-reduce-plus-`atomicAdd`
      tail (`:381-388`) for `function_value` AND the two fixed-point plane
      magnitudes -- the 2026-08-21 determinism fix, same buffers, same
      `deterministic_sum_lanes_kernel` fold.
    * GRID: theirs is 512 threads x 2 elements each (`:396-397`); ours is
      the file's one-element-per-thread shape at `MSE_BLOCK_SIZE`.
      Scheduling only -- every per-document store is identical -- and the
      fv/magnitude summation order is ours either way, because the partials
      buffers already are.

    And one of substance: `exp`/`log` here are `std.math`, where theirs are
    CUDA's `__expf`/`__logf` fast-math approximations (~2 ulp). CatBoost
    itself accepts approximate transcendentals at this exact site, so the
    substitution is in-family, but the two arms' derivatives differ in last
    bits BY CONSTRUCTION and no check may expect bitwise der parity against
    a CatBoost fit. (The host-side `std.math.log` tie-decoding defect is a
    different site and does not apply: nothing here decodes ties.)
    ===================================================
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)

    var in_range = i < size
    var val = Float32(0.0)
    var target_class = Float32(1.0)  # their `scale[j]` OOB default (`:346`)
    var weight = Float32(1.0)
    if in_range:
        val = predictions.unsafe_load(i)
        target_class = target_classes.unsafe_load(i)
        if has_weights != Int32(0):
            weight = weights.unsafe_load(i)

    # `const float expVal = idx < size ? __expf(val) : 0;` (`:353`)
    var exp_val = Float32(0.0)
    if in_range:
        exp_val = exp(val)
    # `p = max(min(isfinite(expVal) ? expVal / (1.0f + expVal) : 1.0f,
    #              1.0f - 1e-40f), 1e-40f)` (`:354`)
    var p = Float32(1.0)
    if isfinite(exp_val):
        p = exp_val / (Float32(1.0) + exp_val)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))

    # `const float c = HAS_BORDER ? targetClass > border : targetClass;`
    var c: Float32

    @parameter
    if has_border:
        c = Float32(1.0) if target_class > border else Float32(0.0)
    else:
        c = target_class

    # their `ApproximateAt` path has no `secondDerAsWeights` flag; the
    # combination is a caller error, not a mode.
    comptime assert not (estimation and second_der_as_weights), (
        "second_der_as_weights is a SEARCH-mode flag: their StochasticDer"
        " takes it (`pointwise_target_impl.h:173-216`), their ApproximateAt"
        " does not"
    )

    var direction = c - p  # their `direction[j] = c - p` (`:358`)
    var scale = p * (Float32(1.0) - p)  # their `scale[j]` (`:359`)

    # what SEARCH plane 0 holds, per `second_der_as_weights` -- see the
    # deviation block. The der2 arm is their `der2[idx] = weight[j] *
    # scale[j]` (`:373-375`) landing in `weightsView`.
    var plane0 = weight

    @parameter
    if second_der_as_weights:
        plane0 = weight * scale

    if in_range:

        @parameter
        if estimation:
            stats.unsafe_store(i, weight * direction)
            stats.unsafe_store(size + i, weight * scale)
        else:
            stats.unsafe_store(i, plane0)
            stats.unsafe_store(size + i, weight * direction)
            # no third plane in search mode; see the deviation block.
    _ = scale

    if compute_fv != Int32(0):
        # `tmpScore += w * (c * val - __logf(1 + expVal))`, with their
        # infinity fallback `logExpValPlusOne = isfinite ? ... : val`
        # (`:362-364`).
        var score = Float32(0.0)
        if in_range:
            var log_exp_val_plus_one = val
            if isfinite(exp_val):
                log_exp_val_plus_one = log(Float32(1.0) + exp_val)
            score = weight * (c * val - log_exp_val_plus_one)
        var total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](score)
        if thread_idx.x == 0:
            function_value.unsafe_store(Int(block_idx.x), total)

    if compute_magnitudes != Int32(0):
        var w_abs = Float32(0.0)
        var g_abs = Float32(0.0)
        if in_range:
            # NO CATBOOST COUNTERPART (the fixed-point scale is ours):
            # the magnitudes bound the planes AS STORED, so plane 0's is
            # |weight * p * (1 - p)| under `second_der_as_weights`.
            w_abs = abs(plane0)
            g_abs = abs(weight * direction)
        var w_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](w_abs)
        var g_total = pinned_block_sum[block_size=MSE_BLOCK_SIZE](g_abs)
        if thread_idx.x == 0:
            plane_magnitudes.unsafe_store(2 * Int(block_idx.x), w_total)
            plane_magnitudes.unsafe_store(
                2 * Int(block_idx.x) + 1, g_total
            )


# =========================================================================
# THE HOST DISPATCH: their `PointwiseTargetKernel` switch and the
# CrossEntropy/Pointwise fork above it, both ported as host functions.
#
# `PointwiseTargetKernel` (`pointwise_targets.cu:447-519`) is a HOST switch
# on `ELossFunction` that constructs the objective struct and launches the
# one generic template. `TPointwiseTargetsImpl::Approximate`
# (`pointwise_target_impl.h:307-358`) is the fork above it that decides
# whether the launch is that kernel or `CrossEntropyImpl` instead.
#
# Both are transcribed here rather than open-coded at each call site,
# because their two call sites -- the boosting loop's search pass and the
# leaves oracle -- differ only in `estimation`, exactly as their
# `StochasticDer` and `ApproximateAt` differ only in which buffers they
# hand over.
# =========================================================================


def launch_pointwise_target_kernel[
    estimation: Bool,
    second_der_as_weights: Bool = False,
](
    ctx: DeviceContext,
    objective: Int,
    mut relevs: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    size: Int32,
    mut predictions: DeviceBuffer[DType.float32],
    has_weights: Int32,
    alpha: Float32,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Int32,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Int32,
    blocks: Int,
) raises:
    """`PointwiseTargetKernel` (`pointwise_targets.cu:447-519`).

    Their switch picks a template instantiation on the host; this one picks
    a comptime specialization on the host. `NumErrors` is absent for the
    reason recorded beside the objective constants, and their `default:`
    `Y_ABORT_UNLESS(false, "Unknown target")` becomes a raise.
    """

    @parameter
    def _go[obj: Int]() raises:
        ctx.enqueue_function[
            pointwise_target_kernel[obj, estimation, second_der_as_weights]
        ](
            relevs, weights, size, predictions, has_weights, alpha,
            stats, function_value, compute_fv,
            plane_magnitudes, compute_magnitudes,
            grid_dim=(blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )

    # their case order, `:455-518`
    if objective == OBJECTIVE_EXPECTILE:
        _go[OBJECTIVE_EXPECTILE]()
    elif objective == OBJECTIVE_QUANTILE:
        _go[OBJECTIVE_QUANTILE]()
    elif objective == OBJECTIVE_MAE:
        # theirs falls MAE through to `TQuantileTarget` (`:483-489`);
        # the alpha it falls through WITH is 0.5, set by `Init`, and
        # `TLossDescription.kernel_alpha` is where that happens here
        _go[OBJECTIVE_MAE]()
    elif objective == OBJECTIVE_LOGLINQUANTILE:
        _go[OBJECTIVE_LOGLINQUANTILE]()
    elif objective == OBJECTIVE_MAPE:
        _go[OBJECTIVE_MAPE]()
    elif objective == OBJECTIVE_POISSON:
        _go[OBJECTIVE_POISSON]()
    elif objective == OBJECTIVE_LQ:
        _go[OBJECTIVE_LQ]()
    elif objective == OBJECTIVE_RMSE:
        _go[OBJECTIVE_RMSE]()
    elif objective == OBJECTIVE_TWEEDIE:
        _go[OBJECTIVE_TWEEDIE]()
    elif objective == OBJECTIVE_HUBER:
        _go[OBJECTIVE_HUBER]()
    elif (
        objective == OBJECTIVE_MULTICLASS
        or objective == OBJECTIVE_MULTICLASS_OVA
    ):
        raise Error(
            "the multiclass family does not reach PointwiseTargetKernel:"
            " it is TMultiClassificationTargets, whose kernels are"
            " targets/kernel/multilogit.cu"
        )
    else:
        raise Error(
            "Unknown target: objective " + String(objective)
            + " does not reach PointwiseTargetKernel"
        )


def launch_approximate[
    estimation: Bool,
    second_der_as_weights: Bool = False,
](
    ctx: DeviceContext,
    objective: Int,
    mut relevs: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    size: Int32,
    mut predictions: DeviceBuffer[DType.float32],
    has_weights: Int32,
    alpha: Float32,
    border: Float32,
    mut stats: DeviceBuffer[DType.float32],
    mut function_value: DeviceBuffer[DType.float32],
    compute_fv: Int32,
    mut plane_magnitudes: DeviceBuffer[DType.float32],
    compute_magnitudes: Int32,
    blocks: Int,
) raises:
    """`TPointwiseTargetsImpl::Approximate` (`pointwise_target_impl.h:
    307-358`): the fork between their two kernels.

    `UseBorder()` is `Type == ELossFunction::Logloss` (`:303-305`), so
    Logloss thresholds the target and CrossEntropy takes it soft; both
    land on the same kernel. Everything else goes to the generic one.
    """
    if objective == OBJECTIVE_LOGLOSS:
        ctx.enqueue_function[
            cross_entropy_kernel[True, estimation, second_der_as_weights]
        ](
            relevs, weights, size, predictions, has_weights, border,
            stats, function_value, compute_fv,
            plane_magnitudes, compute_magnitudes,
            grid_dim=(blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    elif objective == OBJECTIVE_CROSSENTROPY:
        ctx.enqueue_function[
            cross_entropy_kernel[False, estimation, second_der_as_weights]
        ](
            relevs, weights, size, predictions, has_weights, border,
            stats, function_value, compute_fv,
            plane_magnitudes, compute_magnitudes,
            grid_dim=(blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
    else:
        launch_pointwise_target_kernel[estimation, second_der_as_weights](
            ctx, objective, relevs, weights, size, predictions,
            has_weights, alpha, stats, function_value, compute_fv,
            plane_magnitudes, compute_magnitudes, blocks,
        )
