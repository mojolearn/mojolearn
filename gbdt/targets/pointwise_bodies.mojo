# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pointwise objectives' per-row math, written ONCE (AGENTS.md "One body
per operation", lane gbdt-gaps 2026-11-23).

`target_score`, `target_der` and `target_der2` are `TTarget::Score / Der /
Der2` (`pointwise_targets.cu`), and `weighted_der`, `weighted_der2` and
`weighted_score` are the three per-row values the kernels store or reduce
from them with the row weight folded in (`der[i] = weight * Der`, `der2[i] =
weight * Der2`, `-weight * Score`, each derived value flushed at derivation,
DEVIATION 256 / IDENTITY_PATHS row 10). Every column calls these bodies:
`pointwise_target_kernel` and the Ordered arm (`gbdt/methods/
ordered_boosting.mojo`) on the device, and the host oracles
(`gbdt/host/gbdt_oracle_losses.mojo`, the RMSE rows of
`gbdt/host/gbdt_oracle.mojo`) on the CPU column. Until this lane the host
carried its own copy of all three derivatives.

HOST SAFE: no `max.gpu` / `std.gpu` import and no kernel, so the CPU-only
host bindings import it. The objective codes are `pointwise_targets.mojo`'s
`OBJECTIVE_*` (that module defines each as the `BODY_OBJ_*` here, one
number per objective) and the host oracle's `GBDT_OBJ_*`.

`routed_exp` / `routed_pow` / `routed_log` (DEVIATION 254's shims) moved here
with them, word for word; see `pointwise_targets.mojo`'s deviation block for
why they are `@always_inline` shims.
"""

from std.math import isfinite

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_pow,
)

comptime BODY_OBJ_RMSE = 0
comptime BODY_OBJ_QUANTILE = 3
comptime BODY_OBJ_MAE = 4
comptime BODY_OBJ_LOGLINQUANTILE = 5
comptime BODY_OBJ_MAPE = 6
comptime BODY_OBJ_POISSON = 7
comptime BODY_OBJ_LQ = 8
comptime BODY_OBJ_EXPECTILE = 9
comptime BODY_OBJ_TWEEDIE = 10
comptime BODY_OBJ_HUBER = 11


@always_inline
def routed_exp(x: Float32) -> Float32:
    """DEVIATION 254's site call. FAST is `std.math.exp` inlined at the
    call site, bit for bit the direct call it replaced; IDENTICAL is
    row 12's `identical_exp`. See the deviation block above for why the
    inlining is load-bearing."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return identical_exp(x)
    from std.math import exp

    return exp(x)


@always_inline
def routed_pow(x: Float32, p: Float32) -> Float32:
    """DEVIATION 258's site call, `pow` side (Lq's `__powf` three ways:
    score, der, der2). FAST is the `**` the sites used, inlined;
    IDENTICAL is `identical_pow` (row 12). E2 round 1 measured gbdt_lq
    DIVERGENT at the FIRST histogram on both AMD and NVIDIA before this."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return identical_pow(x, p)
    return x**p


@always_inline
def routed_log(x: Float32) -> Float32:
    """DEVIATION 254's site call, `log` side. Same two-arm contract as
    `routed_exp`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return identical_log(x)
    from std.math import log

    return log(x)


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

    comptime if objective == BODY_OBJ_RMSE:
        # `TRmseTarget::Score` (`:180-182`)
        return (t - p) * (t - p)
    elif objective == BODY_OBJ_QUANTILE or objective == BODY_OBJ_MAE:
        # `TQuantileTarget::Score` (`:18-22`). MAE IS Quantile at
        # alpha 0.5: `Init` sets `Alpha = 0.5` for MAE
        # (`pointwise_target_impl.h:272-275`) and the kernel switch falls
        # MAE through to `TQuantileTarget` (`:483-489`).
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        )
        return multiplier * val
    elif objective == BODY_OBJ_LOGLINQUANTILE:
        # `TLogLinQuantileTarget::Score` (`:126-131`)
        var val = t - routed_exp(p)
        var multiplier = (
            alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
        )
        return val * multiplier
    elif objective == BODY_OBJ_MAPE:
        # `TMAPETarget::Score` (`:147-149`)
        return abs(t - p) / max(Float32(1.0), abs(t))
    elif objective == BODY_OBJ_POISSON:
        # `TPoissonTarget::Score` (`:164-166`), `exp(p) - t * p` in ONE
        # rounding: every default (contract=fast) build fused the product
        # into the subtraction, and the host oracle
        # (`gbdt_oracle_losses._target_score`) spells the same fma. Explicit,
        # so a contract=off build keeps CPU and GPU agreeing (lane/explicit-fma-contract-proof, 2026-09-26)
        return identical_mul_add(-t, p, routed_exp(p))
    elif objective == BODY_OBJ_LQ:
        # `TLqTarget::Score` (`:204-207`); their `__powf`
        var abs_loss = abs(t - p)
        return routed_pow(abs_loss, alpha)
    elif objective == BODY_OBJ_EXPECTILE:
        # `TExpectileTarget::Score` (`:102-106`)
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return multiplier * val * val
    elif objective == BODY_OBJ_TWEEDIE:
        # `TTweedieTarget::Score` (`:40-44`), `alpha` IS their
        # `VariancePower` -- see the deviation block on the kernel.
        var val = (
            -t
            * routed_exp((Float32(1.0) - alpha) * p)
            / (Float32(1.0) - alpha)
        )
        var delta = routed_exp((Float32(2.0) - alpha) * p) / (
            Float32(2.0) - alpha
        )
        return val + delta
    elif objective == BODY_OBJ_HUBER:
        # `THuberTarget::Score` (`:68-75`), `alpha` is their `Delta`
        var mismatch = abs(t - p)
        if mismatch < alpha:
            return Float32(0.5) * mismatch * mismatch
        # `mismatch - 0.5 * alpha` in ONE rounding, the default build's fusion
        return alpha * identical_mul_add(Float32(-0.5), alpha, mismatch)
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

    comptime if objective == BODY_OBJ_RMSE:
        # `:184-186`
        return t - p
    elif objective == BODY_OBJ_QUANTILE or objective == BODY_OBJ_MAE:
        # `:24-27`
        var val = t - p
        return alpha if val > Float32(0.0) else -(Float32(1.0) - alpha)
    elif objective == BODY_OBJ_LOGLINQUANTILE:
        # `:133-136`
        var exp_pred = routed_exp(p)
        if t - exp_pred > Float32(0.0):
            return alpha * exp_pred
        return -(Float32(1.0) - alpha) * exp_pred
    elif objective == BODY_OBJ_MAPE:
        # `:151-154`
        if t - p > Float32(0.0):
            return Float32(1.0) / max(Float32(1.0), abs(t))
        return Float32(-1.0) / max(Float32(1.0), abs(t))
    elif objective == BODY_OBJ_POISSON:
        # `:168-171`
        return t - routed_exp(p)
    elif objective == BODY_OBJ_LQ:
        # `:209-213`
        var abs_loss = abs(t - p)
        var abs_loss_q = routed_pow(abs_loss, alpha - Float32(1.0))
        return alpha * _target_sign(t - p) * abs_loss_q
    elif objective == BODY_OBJ_EXPECTILE:
        # `:108-112`
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return Float32(2.0) * multiplier * val
    elif objective == BODY_OBJ_TWEEDIE:
        # `:46-50`, `t * e1 - e2` in ONE rounding (the default build's
        # fusion, written out; the host oracle spells the same fma)
        var e1 = routed_exp((Float32(1.0) - alpha) * p)
        var delta = routed_exp((Float32(2.0) - alpha) * p)
        return identical_mul_add(t, e1, -delta)
    elif objective == BODY_OBJ_HUBER:
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

    comptime if objective == BODY_OBJ_RMSE:
        # `:188-190`
        return Float32(1.0)
    elif (
        objective == BODY_OBJ_QUANTILE
        or objective == BODY_OBJ_MAE
        or objective == BODY_OBJ_LOGLINQUANTILE
        or objective == BODY_OBJ_MAPE
    ):
        return Float32(0.0)
    elif objective == BODY_OBJ_POISSON:
        # `:173-175`
        return routed_exp(p)
    elif objective == BODY_OBJ_LQ:
        # `:215-218`: BELOW q = 2 their second derivative is the constant
        # 1, not the true one. Copied, because it is what their Newton
        # step divides by.
        var abs_loss = abs(t - p)
        if alpha >= Float32(2.0):
            return (
                alpha
                * (alpha - Float32(1.0))
                * routed_pow(abs_loss, alpha - Float32(2.0))
            )
        return Float32(1.0)
    elif objective == BODY_OBJ_EXPECTILE:
        # `:114-118`
        var val = t - p
        var multiplier = (
            alpha if val > Float32(0.0) else (Float32(1.0) - alpha)
        )
        return Float32(2.0) * multiplier
    elif objective == BODY_OBJ_TWEEDIE:
        # `:52-56`
        var der2 = (
            t
            * routed_exp((Float32(1.0) - alpha) * p)
            * (Float32(1.0) - alpha)
        )
        # `-der2 + e2 * (2 - alpha)` in ONE rounding on the SECOND product,
        # the one the default build fused (the host oracle spells the same)
        var e2 = routed_exp((Float32(2.0) - alpha) * p)
        return identical_mul_add(e2, Float32(2.0) - alpha, -der2)
    elif objective == BODY_OBJ_HUBER:
        # `:86-93`. Theirs writes `-HUBER_DER2` where
        # `HUBER_DER2 = -1.0` (`:62`), i.e. 1.0; the double negation is
        # kept so a reader diffing against their file finds the same
        # expression.
        #
        # THE ZERO ARM IS LOAD-BEARING AND THEIRS. Huber's default is
        # Newton at one iteration (`catboost_options.cpp:187-192`) and
        # the cursor starts at 0 (Huber is not on the boost-from-average
        # list, `train_lib/options_helper.cpp:362-368`), so a target
        # whose every |value| exceeds delta puts EVERY row in this arm on
        # tree 0: the leaf Hessian is `0 + l2` (`pointwise_oracle.cpp:
        # 86-89`) and the leaf is `sum(+-delta) / l2` -- the
        # order-of-n overshoot that CatBoost CPU 1.2.10 reproduces on
        # the E2 fixture (predictions in [-786, 669] for a target in
        # [3.7, 10.6]). Measured 2026-08-23; see DEVIATION 257
        # (`gbdt/options/catboost_options.mojo`) for the whole chain and
        # for the one piece of it that WAS ours. Not an implementation defect; do
        # not "fix" this arm.
        var diff = t - p
        if abs(diff) < alpha:
            return -Float32(HUBER_DER2)
        return Float32(0.0)
    else:
        return Float32(0.0)


#: `THuberTarget::HUBER_DER2` (`pointwise_targets.cu:62`). Their constant
#: is negative and every use negates it.
comptime HUBER_DER2 = -1.0


@always_inline
def weighted_der[objective: Int](
    t: Float32, p: Float32, alpha: Float32, weight: Float32
) -> Float32:
    """`der[i] = weight * target.Der(relev, val)` (`pointwise_targets.cu:
    265-266`), flushed at derivation."""
    return ftz(weight * target_der[objective](t, p, alpha))


@always_inline
def weighted_der2[objective: Int](
    t: Float32, p: Float32, alpha: Float32, weight: Float32
) -> Float32:
    """`der2[i] = weight * target.Der2(relev, val)` (`pointwise_targets.cu:
    268-269`), flushed at derivation."""
    return ftz(weight * target_der2[objective](t, p, alpha))


@always_inline
def weighted_score[objective: Int](
    t: Float32, p: Float32, alpha: Float32, weight: Float32
) -> Float32:
    """`-w * target.Score(...)`, the function value's element
    (`pointwise_targets.cu:271-280`)."""
    return -weight * target_score[objective](t, p, alpha)


# ---------------------------------------------------------------------------
# `CrossEntropyImpl`'s per-element statements (`pointwise_targets.cu:353-364`),
# Logloss (`has_border`) and CrossEntropy. Lane gbdt-one-body (2026-11-23):
# written once here and called by the device's `cross_entropy_kernel`
# (`gbdt/targets/kernel/pointwise_targets.mojo`), the Ordered arm's
# `_ord_point` (`gbdt/methods/ordered_boosting.mojo`) and the host oracles'
# rows (`gbdt/host/gbdt_oracle.mojo::_cross_entropy_row`,
# `gbdt/host/gbdt_oracle_losses.mojo::_loss_row`), which each carried their
# own copy. The host copies called `identical_exp` / `identical_log`
# directly, which is what `routed_exp` / `routed_log` are under IDENTICAL,
# so no bit moves on the identical columns.
# ---------------------------------------------------------------------------


@fieldwise_init
struct CrossEntropyRow(ImplicitlyCopyable, Movable):
    """One element's derived values, before the row weight."""

    #: `expVal = exp(val)`, routed (DEVIATION 254)
    var exp_val: Float32
    #: the class: `targetClass > border` under `has_border`, else the target
    var c: Float32
    #: `ftz(c - p)`, their `direction[j]`
    var direction: Float32
    #: `ftz(p * (1 - p))`, their `scale[j]`
    var scale: Float32


@always_inline
def cross_entropy_row[has_border: Bool](
    target_class: Float32, val: Float32, border: Float32
) -> CrossEntropyRow:
    """`expVal`, `p = clamp(isfinite ? expVal / (1 + expVal) : 1, [1e-40,
    1 - 1e-40])`, the class `c`, and the flushed direction and scale
    (DEVIATION 256 / IDENTITY_PATHS row 10: `c - p` cancels and `p * (1 -
    p)` is denormal at the clamp, so both are flushed at derivation)."""
    var exp_val = routed_exp(val)
    var p = Float32(1.0)
    if isfinite(exp_val):
        p = exp_val / (Float32(1.0) + exp_val)
    p = max(min(p, Float32(1.0) - Float32(1e-40)), Float32(1e-40))
    var c: Float32

    comptime if has_border:
        c = Float32(1.0) if target_class > border else Float32(0.0)
    else:
        c = target_class
    return CrossEntropyRow(
        exp_val, c, ftz(c - p), ftz(p * (Float32(1.0) - p))
    )


@always_inline
def cross_entropy_score(row: CrossEntropyRow, val: Float32, weight: Float32) -> Float32:
    """`tmpScore += w * (c * val - log(1 + expVal))` with their isfinite
    fallback `logExpValPlusOne = isfinite ? log(1 + expVal) : val`
    (`:362-364`); `c * val - log(1 + e)` in ONE rounding, the default
    build's fusion written out."""
    var log_exp_val_plus_one = val
    if isfinite(row.exp_val):
        log_exp_val_plus_one = routed_log(Float32(1.0) + row.exp_val)
    return weight * identical_mul_add(row.c, val, -log_exp_val_plus_one)
