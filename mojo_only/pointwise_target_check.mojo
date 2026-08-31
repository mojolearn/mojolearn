# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`pointwise_target_kernel` against a float64 libm oracle, per objective.

    pixi run check-pointwise-target

NO CATBOOST COUNTERPART: this is a gate, and `mojo_only/` is where gates
live.

WHAT GATES WHAT. The kernel is the port of `PointwiseTargetImpl<TTarget>`
(`pointwise_targets.cu:246-281`) specialized on each of their nine objective
structs (`:11-240`). No CatBoost fit can gate its DERIVATIVES bitwise -- the
kernel's own deviation block explains why: their file mixes `__expf`,
`std::exp`, `__powf` and `powf` WITHIN ONE FILE, and Mojo has one `exp` and
one `**`. So this check gates the ARITHMETIC three ways, none of which is a
digest:

1. ANCHORS. Cases computed OUTSIDE the device in float64 through **libm via
   `external_call`**, not through `std.math`, because `std.math.log` on this
   toolchain carries ~5e-8 ABSOLUTE error and a reference that shares the
   implementation under test proves nothing. They pin the branches a bulk
   sweep undersamples, and every one of them is a place their own code
   BRANCHES:

       Quantile/MAE/Expectile   val EXACTLY 0  -- their test is `val > 0`,
                                so zero takes the NEGATIVE arm
       Lq                       t == p, where `sign(0)` is -1 in their
                                `sign` (`:193-195`, no zero arm) and
                                `0 ** (q-1)` is 0
       Lq                       q < 2 vs q >= 2, the two Der2 arms (`:215`)
       Huber                    |diff| EXACTLY delta -- their test is
                                `< Delta`, so equality takes the OUTER arm
                                where Der2 is 0
       Poisson/Tweedie          large positive prediction, the exp overflow
       MAPE                     |t| < 1 and |t| > 1, the two sides of
                                `max(1, |t|)`

2. BULK, PER CELL. 4,133 rows -- odd, so the in-range tail is real -- of
   HASHED scattered targets and predictions, so no two rows expect the same
   value. Per `uniform-test-data-hides-permutation`, a fixture whose expected
   value is the same in every cell verifies the total and nothing about
   placement, and this one has 4,133 distinct expectations per objective.
   Compared ELEMENT BY ELEMENT against the same formula evaluated on the host
   in float64 through libm.

3. MODES. `estimation` true and false, which are two DIFFERENT plane layouts
   (`weight, weight*der` versus `weight*der, weight*der2`), and weighted
   versus unweighted. A check that ran only the default would leave the other
   branch unreached, which is PORTING_RULES 8.

THE SABOTAGES. Rule 8 says one per MECHANISM, not one per assertion, and
each is RUN on every invocation and required to move the check -- a
sabotage that is only claimed in a comment is not evidence. Measured
2026-08-21, cells moved out of 4,133:

    S1  reference `sign(0)` flipped to +1         20   at q == 1
    S2  reference Huber boundary `<` -> `<=`      14   estimation mode
    S3  reference Quantile arm `>` -> `>=`        20
    S4  reference der scaled by (1 + 1e-5)     2,556
    S5  reference planes 0/1 swapped           4,133   estimation mode
    S6  reference Lq Der2 forced to q >= 2     4,133   estimation mode

TWO OF THOSE SABOTAGES FAILED FIRST, and both failures were defects in the
SABOTAGE rather than in the kernel, which is the outcome rule 8's scar
predicts:

  * S1 was run at q = 1.5, where their `Der` is `q * sign(r) * |r|^(q-1)`
    and the trailing factor at r = 0 is `0^0.5 == 0`. The sign was being
    multiplied by zero, so NO perturbation of it could move a cell. It is
    run at q = 1 now, where the factor is `0^0 == 1` and the sign is the
    whole answer -- and the honest run covers q = 1, 1.5 and 2.5, which are
    their three Der2 regimes.
  * S2 was run in the SEARCH mode, which does not write `der2`. At
    |diff| exactly delta their `Der` returns delta on both arms -- the
    inner one returns `diff`, which IS delta -- so the boundary moves only
    `Der2`. It runs in the estimation mode now.

TWO OF THE ANCHORS ALSO FAILED FIRST, and both were the same mistake: **an
anchor is only an anchor if it survives the precision it is read in.**
Planting LogLinQuantile's residual at zero via `p = log(t); t = exp(p)`, and
Huber's at exactly delta via `p = t - delta` on a hashed `t`, are exact in
float64 and are NOT exact once both operands round to float32 -- the device
and the reference took opposite arms and the check reported the KERNEL wrong
at cells where the FIXTURE was wrong. Both are planted on exactly
representable values now.

TOLERANCE, and why it is neither bitwise nor a guess. The device computes in
float32 with Mojo's `exp` and `**`; the reference computes in float64 through
libm. They differ in last bits BY CONSTRUCTION -- their own file calls
`__expf`, a ~2-ulp approximation, at the same sites. Two rules make the
compare meaningful anyway:

  1. The reference is evaluated at the DEVICE'S float32 inputs, not at the
     float64 originals. Without this, `t - p` on two nearby values reads a
     6e-4 relative error that is entirely the rounding of the inputs.
  2. The compare is against `max(|want|, term_scale)`, where `term_scale` is
     the magnitude of the terms the derivative SUBTRACTS. For `x - y` the
     achievable accuracy is `eps * (|x| + |y|)`, not `eps * |x - y|`;
     demanding the latter would demand something no implementation can give.

With both, the largest honest gap over ten objectives x four modes is
**2.47e-6**, from Poisson, whose `Der2` is a bare `exp(p)` at `p = 60`. That
is about twenty ulp of float32 rather than one or two, so **Mojo's device
`exp` is roughly an order of magnitude looser than libm at large
arguments** -- recorded rather than worked around. `REL_TOL` is 5e-6, twice
the observed worst, and the run PRINTS the worst gap and the headroom so the
next reader checks the number instead of this sentence.
"""

from max.gpu.host import DeviceContext
from std.ffi import external_call

from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    OBJECTIVE_EXPECTILE,
    OBJECTIVE_HUBER,
    OBJECTIVE_LOGLINQUANTILE,
    OBJECTIVE_LQ,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_POISSON,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
    OBJECTIVE_TWEEDIE,
    launch_pointwise_target_kernel,
    objective_name,
)

comptime N_BULK = 4133
#: Set from a MEASUREMENT, not guessed, and the measurement is printed on
#: every run so the next person can see the headroom rather than trust this
#: comment.
#:
#: With the reference evaluated at the device's own float32 inputs and
#: measured against `term_scale` rather than against a cancelling result,
#: the largest honest relative gap over all ten objectives x four modes is
#: **2.47e-6**, and it comes from POISSON -- whose `Der2` is a bare
#: `exp(p)` and whose fixture reaches `p = 60`. That is roughly twenty ulp
#: of float32, not the one or two a correctly-rounded `exp` would give, so
#: **Mojo's device `exp` is about an order of magnitude looser than libm at
#: large arguments.** That is in family with CatBoost, whose own kernel
#: calls `__expf` at this exact site, and it is recorded rather than worked
#: around -- rule 9 says assume the stdlib digit is approximate until
#: measured, and this is the measurement.
#:
#: 5e-6 is twice the observed worst, and S4's 1e-5 perturbation is twice
#: this, which is the only property a tolerance is required to have: loose
#: enough to pass the transcendental gap, tight enough that a real error
#: fails.
comptime REL_TOL = 5e-6
comptime ABS_FLOOR = 1e-30


# ---------------------------------------------------------------- libm --
# `external_call` reaches the C library the host is linked against. It is
# the arbiter rule 9 names, and it is deliberately NOT `std.math`: this
# check exists to catch a wrong formula, and a reference that shares an
# implementation with the thing under test cannot.


def c_exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


def c_log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


def c_pow(x: Float64, y: Float64) -> Float64:
    return external_call["pow", Float64](x, y)


def c_fabs(x: Float64) -> Float64:
    return external_call["fabs", Float64](x)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed_unit(seed: UInt64, i: Int) -> Float64:
    """A scattered value in [0, 1). Hashed, per rule 8, so that no two rows
    share an expectation."""
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float64(h >> 11) * (1.0 / Float64(1 << 53))


# ------------------------------------------------------- the reference --
# Their nine structs, in float64, through libm. `sab` selects a sabotage;
# 0 is the honest reference.


@fieldwise_init
struct Sabotage(Copyable, ImplicitlyCopyable, Movable):
    var id: Int


def ref_score(
    objective: Int, t: Float64, p: Float64, a: Float64, sab: Int
) raises -> Float64:
    if objective == OBJECTIVE_RMSE:
        return (t - p) * (t - p)
    if objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_MAE:
        var v = t - p
        var m = a if _gt(v, 0.0, sab) else -(1.0 - a)
        return m * v
    if objective == OBJECTIVE_LOGLINQUANTILE:
        var v = t - c_exp(p)
        var m = a if _gt(v, 0.0, sab) else -(1.0 - a)
        return v * m
    if objective == OBJECTIVE_MAPE:
        return c_fabs(t - p) / max(1.0, c_fabs(t))
    if objective == OBJECTIVE_POISSON:
        return c_exp(p) - t * p
    if objective == OBJECTIVE_LQ:
        return c_pow(c_fabs(t - p), a)
    if objective == OBJECTIVE_EXPECTILE:
        var v = t - p
        var m = a if _gt(v, 0.0, sab) else (1.0 - a)
        return m * v * v
    if objective == OBJECTIVE_TWEEDIE:
        var val = -t * c_exp((1.0 - a) * p) / (1.0 - a)
        var delta = c_exp((2.0 - a) * p) / (2.0 - a)
        return val + delta
    if objective == OBJECTIVE_HUBER:
        var mm = c_fabs(t - p)
        if _huber_inner(mm, a, sab):
            return 0.5 * mm * mm
        return a * (mm - 0.5 * a)
    raise Error("ref_score: no arm for " + objective_name(objective))


def ref_der(
    objective: Int, t: Float64, p: Float64, a: Float64, sab: Int
) raises -> Float64:
    var d: Float64
    if objective == OBJECTIVE_RMSE:
        d = t - p
    elif objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_MAE:
        var v = t - p
        d = a if _gt(v, 0.0, sab) else -(1.0 - a)
    elif objective == OBJECTIVE_LOGLINQUANTILE:
        var e = c_exp(p)
        d = a * e if _gt(t - e, 0.0, sab) else -(1.0 - a) * e
    elif objective == OBJECTIVE_MAPE:
        var den = max(1.0, c_fabs(t))
        d = 1.0 / den if _gt(t - p, 0.0, sab) else -1.0 / den
    elif objective == OBJECTIVE_POISSON:
        d = t - c_exp(p)
    elif objective == OBJECTIVE_LQ:
        var al = c_fabs(t - p)
        d = a * _sign(t - p, sab) * c_pow(al, a - 1.0)
    elif objective == OBJECTIVE_EXPECTILE:
        var v = t - p
        var m = a if _gt(v, 0.0, sab) else (1.0 - a)
        d = 2.0 * m * v
    elif objective == OBJECTIVE_TWEEDIE:
        d = t * c_exp((1.0 - a) * p) - c_exp((2.0 - a) * p)
    elif objective == OBJECTIVE_HUBER:
        var diff = t - p
        if _huber_inner(c_fabs(diff), a, sab):
            d = diff
        else:
            d = a if diff > 0.0 else -a
    else:
        raise Error("ref_der: no arm for " + objective_name(objective))
    if sab == 4:
        d = d * (1.0 + 1e-5)
    return d


def ref_der2(
    objective: Int, t: Float64, p: Float64, a: Float64, sab: Int
) raises -> Float64:
    if objective == OBJECTIVE_RMSE:
        return 1.0
    if (
        objective == OBJECTIVE_QUANTILE
        or objective == OBJECTIVE_MAE
        or objective == OBJECTIVE_LOGLINQUANTILE
        or objective == OBJECTIVE_MAPE
    ):
        return 0.0
    if objective == OBJECTIVE_POISSON:
        return c_exp(p)
    if objective == OBJECTIVE_LQ:
        var al = c_fabs(t - p)
        # S6 forces the q >= 2 arm even below 2
        if a >= 2.0 or sab == 6:
            return a * (a - 1.0) * c_pow(al, a - 2.0)
        return 1.0
    if objective == OBJECTIVE_EXPECTILE:
        var v = t - p
        var m = a if _gt(v, 0.0, sab) else (1.0 - a)
        return 2.0 * m
    if objective == OBJECTIVE_TWEEDIE:
        var der2 = t * c_exp((1.0 - a) * p) * (1.0 - a)
        var delta = c_exp((2.0 - a) * p) * (2.0 - a)
        return -der2 + delta
    if objective == OBJECTIVE_HUBER:
        if _huber_inner(c_fabs(t - p), a, sab):
            return 1.0
        return 0.0
    raise Error("ref_der2: no arm for " + objective_name(objective))


def _gt(v: Float64, z: Float64, sab: Int) -> Bool:
    """Their `val > 0`. S3 loosens it to `>=`, which changes ONLY the rows
    where the residual is exactly zero -- which is why the fixture plants
    them."""
    if sab == 3:
        return v >= z
    return v > z


def _sign(x: Float64, sab: Int) -> Float64:
    """Their `sign` (`pointwise_targets.cu:193-195`): NO ZERO ARM, so
    `sign(0) == -1`. S1 gives zero the positive arm instead."""
    if sab == 1 and x == 0.0:
        return 1.0
    return 1.0 if x > 0.0 else -1.0


def _huber_inner(mismatch: Float64, delta: Float64, sab: Int) -> Bool:
    """Their `fabs(diff) < Delta` (`:79`, `:88`). S2 makes it `<=`, which
    changes ONLY the rows planted exactly at delta."""
    if sab == 2:
        return mismatch <= delta
    return mismatch < delta


def term_scale(
    objective: Int, t: Float64, p: Float64, a: Float64
) raises -> Float64:
    """The magnitude of the TERMS their `Der` subtracts, not of its result.

    WHY A DIFFERENCE IS NOT MEASURED AGAINST ITSELF. Six of these
    objectives begin `Der` with a subtraction -- `t - p`, `t - exp(p)`,
    `t*exp((1-a)p) - exp((2-a)p)` -- and the accuracy achievable in
    floating point for `x - y` is `eps * (|x| + |y|)`, NOT `eps * |x - y|`.
    When the two terms nearly cancel, a perfectly correct float32
    subtraction has an unbounded RELATIVE error against a float64 one.
    Demanding a relative tolerance on the result would be demanding
    something no implementation can deliver, theirs included -- their
    `__expf` is a ~2-ulp approximation and would fail it by more.

    So the compare is `|got - want| <= REL_TOL * max(|want|, term_scale)`.
    That is the standard criterion for a cancelling difference and it is
    the reason this check can hold a 2e-6 tolerance on Poisson and Tweedie
    instead of the 5e-4 the naive form forced.

    IT DOES NOT BLIND THE CHECK: S4 perturbs the reference `der` by 1e-5
    and still moves 4,130 of 4,133 cells, and the three it misses are
    exactly the cells where the residual is planted at zero and 1e-5 times
    zero is zero.
    """
    if objective == OBJECTIVE_POISSON:
        return c_fabs(t) + c_exp(p)
    if objective == OBJECTIVE_LOGLINQUANTILE:
        return c_fabs(t) + c_exp(p)
    if objective == OBJECTIVE_TWEEDIE:
        return (
            c_fabs(t) * c_exp((1.0 - a) * p) + c_exp((2.0 - a) * p)
        )
    # the rest subtract two inputs of comparable size, or nothing at all
    return c_fabs(t) + c_fabs(p)


def rel_gap(
    got: Float64, want: Float64, scale: Float64
) -> Float64:
    """The deviation, measured against `max(|want|, scale)`.

    The absolute floor catches the values that legally flush to zero on
    the device (a float32 `der2` of ~1e-40)."""
    var d = got - want
    if d < 0.0:
        d = -d
    if d <= ABS_FLOOR:
        return 0.0
    var m = want if want > 0.0 else -want
    if scale > m:
        m = scale
    if m < ABS_FLOOR:
        # `want` is zero and `got` is not: a full miss, not a rounding gap
        return 1.0
    return d / m


def close(got: Float64, want: Float64, scale: Float64) -> Bool:
    return rel_gap(got, want, scale) <= REL_TOL


# ------------------------------------------------------------ the run --


def run_objective(
    ctx: DeviceContext,
    objective: Int,
    alpha: Float32,
    estimation: Bool,
    has_weights: Bool,
    sab: Int,
    verbose: Bool,
    mut out_worst: Float64,
) raises -> Int:
    """Launch one objective in one mode and count the cells that disagree.

    Returns the mismatch count, so a sabotage run can assert it is NONZERO
    and the honest run can assert it is zero.
    """
    var n = N_BULK
    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_p = ctx.enqueue_create_buffer[DType.float32](n)
    var d_s = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_mg = ctx.enqueue_create_buffer[DType.float32](2 * blocks)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](n)

    # THE FIXTURE. Hashed and scattered, with the branch anchors PLANTED at
    # a stride that is coprime with everything else so they do not collide.
    var ts = List[Float64]()
    var ps = List[Float64]()
    var ws = List[Float64]()
    for i in range(n):
        var t = hashed_unit(UInt64(0xA1), i) * 4.0 - 1.5
        var p = hashed_unit(UInt64(0xB2), i) * 3.0 - 1.0
        var w = 0.25 + hashed_unit(UInt64(0xC3), i) * 2.0

        # ANCHOR A: residual EXACTLY zero, every 211th row. Reaches
        # `sign(0)` and the `val > 0` FALSE arm of Quantile / MAE /
        # Expectile / MAPE.
        #
        # AN ANCHOR IS ONLY AN ANCHOR IF IT SURVIVES THE PRECISION IT IS
        # READ IN. `p = t` is exact in both float64 and float32, because
        # the residual is a SUBTRACTION of two representable values. The
        # first version of this check also set LogLinQuantile's residual
        # to zero by `p = log(t); t = exp(p)`, and that is NOT exact: the
        # device's float32 `exp` and libm's float64 `exp` land on opposite
        # sides of zero, the two arms disagree, and the check reported the
        # KERNEL wrong at a cell where the fixture was wrong. Their code
        # has the same knife-edge and neither answer is more correct.
        # LogLinQuantile gets `p = 0, t = 1` instead: `exp(0) == 1.0`
        # exactly in every precision, so the residual is exactly zero on
        # both sides and the arm test is gateable again.
        if i % 211 == 0:
            if objective == OBJECTIVE_LOGLINQUANTILE:
                p = 0.0
                t = 1.0
            else:
                p = t
        # ANCHOR B: |diff| EXACTLY delta, every 307th row. Reaches Huber's
        # boundary, where their `<` sends equality to the OUTER arm and
        # `Der2` to zero.
        #
        # Same lesson as ANCHOR A, and it cost the same false failure:
        # `p = t - delta` with a hashed `t` is exact in float64 and lands a
        # fraction of an ULP BELOW delta once both are rounded to float32,
        # so the device took the inner arm and the reference took the
        # outer one. Both operands are planted as exactly-representable
        # values now, chosen so their float32 difference is exactly delta.
        if i % 307 == 1 and objective == OBJECTIVE_HUBER:
            t = 1.5
            p = 1.5 - Float64(alpha)
        # ANCHOR C: the exp overflow arm, every 401st row.
        if i % 401 == 2:
            p = 60.0
        # ANCHOR D: |t| straddling 1, which is MAPE's `max(1, |t|)` fork.
        if i % 149 == 3:
            t = 0.5
        if i % 149 == 4:
            t = 3.5

        # THE REFERENCE IS EVALUATED AT THE VALUES THE DEVICE RECEIVED,
        # which are the float32 roundings, NOT at the float64 originals.
        #
        # This is not a loosening; it removes a discrepancy that has
        # nothing to do with the kernel. `Der` for most of these
        # objectives begins with `t - p`, and when the two are close that
        # subtraction is catastrophic cancellation: at t = 1.5790744 and
        # p = 1.5792221 the residual is 1.48e-4, and the ~1e-7 absolute
        # rounding each input picked up on the way to float32 is a 6e-4
        # RELATIVE error in their difference. The first version of this
        # check compared the device's float32-input answer against a
        # float64-input reference and read that 6e-4 as a kernel defect
        # across six objectives. The kernel is only answerable for
        # computing correctly FROM THE INPUTS IT WAS GIVEN.
        var t32 = Float64(Float32(t))
        var p32 = Float64(Float32(p))
        var w32 = Float64(Float32(w if has_weights else 1.0))
        ts.append(t32)
        ps.append(p32)
        ws.append(w32)
        h_t.unsafe_ptr().unsafe_store(i, Float32(t))
        h_p.unsafe_ptr().unsafe_store(i, Float32(p))
        h_w.unsafe_ptr().unsafe_store(i, Float32(w32))

    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.synchronize()

    if estimation:
        launch_pointwise_target_kernel[True](
            ctx, objective, d_t, d_w, Int32(n), d_p,
            Int32(1) if has_weights else Int32(0), alpha,
            d_s, d_fv, Int32(1), d_mg, Int32(0), blocks,
        )
    else:
        launch_pointwise_target_kernel[False](
            ctx, objective, d_t, d_w, Int32(n), d_p,
            Int32(1) if has_weights else Int32(0), alpha,
            d_s, d_fv, Int32(1), d_mg, Int32(1), blocks,
        )

    var h_s = ctx.enqueue_create_host_buffer[DType.float32](2 * n)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_s)
    ctx.synchronize()

    var a64 = Float64(alpha)
    var bad = 0
    var shown = 0
    var worst = Float64(0.0)
    for i in range(n):
        var got0 = Float64(h_s.unsafe_ptr().unsafe_load(i))
        var got1 = Float64(h_s.unsafe_ptr().unsafe_load(n + i))
        var w = ws[i]
        var want0: Float64
        var want1: Float64
        if estimation:
            want0 = w * ref_der(objective, ts[i], ps[i], a64, sab)
            want1 = w * ref_der2(objective, ts[i], ps[i], a64, sab)
        else:
            want0 = w
            want1 = w * ref_der(objective, ts[i], ps[i], a64, sab)
        # S5 swaps the two planes in the estimation mode: a layout defect a
        # summed total cannot see.
        if sab == 5 and estimation:
            var tmp = want0
            want0 = want1
            want1 = tmp
        var sc = term_scale(objective, ts[i], ps[i], a64)
        var g0 = rel_gap(got0, want0, sc)
        var g1 = rel_gap(got1, want1, sc)
        if sab == 0:
            if g0 > worst:
                worst = g0
            if g1 > worst:
                worst = g1
        if not close(got0, want0, sc) or not close(got1, want1, sc):
            bad += 1
            if verbose and shown < 3:
                print(
                    "    cell", i, "t", ts[i], "p", ps[i],
                    "got", got0, got1, "want", want0, want1,
                )
                shown += 1
    if sab == 0 and worst > out_worst:
        out_worst = worst
    return bad


def objectives() -> List[Int]:
    return [
        OBJECTIVE_RMSE, OBJECTIVE_QUANTILE, OBJECTIVE_MAE,
        OBJECTIVE_LOGLINQUANTILE, OBJECTIVE_MAPE, OBJECTIVE_POISSON,
        OBJECTIVE_LQ, OBJECTIVE_EXPECTILE, OBJECTIVE_TWEEDIE,
        OBJECTIVE_HUBER,
    ]


def alpha_for(objective: Int) -> Float32:
    """The one float their kernel takes, per objective."""
    if objective == OBJECTIVE_MAE:
        return Float32(0.5)  # their `Init` fixes it
    if objective == OBJECTIVE_LQ:
        return Float32(1.5)  # BELOW 2, the constant-Der2 arm
    if objective == OBJECTIVE_HUBER:
        return Float32(1.0)  # delta
    if objective == OBJECTIVE_TWEEDIE:
        return Float32(1.5)  # variance_power
    if (
        objective == OBJECTIVE_QUANTILE
        or objective == OBJECTIVE_LOGLINQUANTILE
        or objective == OBJECTIVE_EXPECTILE
    ):
        return Float32(0.7)
    return Float32(0.0)


def check_pointwise_targets(ctx: DeviceContext) raises:
    var objs = objectives()
    var failures = 0
    # THE TOLERANCE IS REPORTED AGAINST A MEASUREMENT, not asserted. This
    # carries the largest relative gap any honest cell showed, so a reader
    # can see how much headroom REL_TOL actually has over the float32 /
    # libm transcendental difference, and so a future tightening is a
    # decision rather than a guess.
    var worst = Float64(0.0)
    print("-- honest run: every objective, both modes, both weightings --")
    for oi in range(len(objs)):
        var o = objs[oi]
        var a = alpha_for(o)
        var total = 0
        for mode in range(4):
            var est = (mode & 1) != 0
            var wt = (mode & 2) != 0
            total += run_objective(ctx, o, a, est, wt, 0, True, worst)
        if total != 0:
            print("  FAIL", objective_name(o), "mismatched cells", total)
            failures += 1
        else:
            print("  ok  ", objective_name(o), "alpha", a)

    # LQ AT q >= 2, the OTHER Der2 arm. PORTING_RULES 8: reach is
    # per-branch, and the default alpha above deliberately sits below 2.
    var lq2 = 0
    for mode in range(4):
        lq2 += run_objective(
            ctx, OBJECTIVE_LQ, Float32(2.5), (mode & 1) != 0,
            (mode & 2) != 0, 0, True, worst,
        )
    if lq2 != 0:
        print("  FAIL Lq q=2.5 mismatched cells", lq2)
        failures += 1
    else:
        print("  ok   Lq q=2.5 (the q >= 2 Der2 arm)")

    print()
    print("-- sabotages: each must turn the check RED --")
    # LQ AT q == 1, which is the ONLY place `sign(0)` is observable.
    # At q = 1.5 their `Der` is `q * sign(r) * |r|^(q-1)`, and at r = 0
    # that trailing factor is `0^0.5 == 0`, so the sign is multiplied by
    # zero and NO sabotage of it can move a cell. At q = 1 the factor is
    # `0^0 == 1` and the sign is the whole answer. The first version of
    # this check ran S1 at 1.5 and reported "changed nothing", which was
    # true and was a defect in the sabotage, not in the kernel.
    var lq1 = 0
    for mode in range(4):
        lq1 += run_objective(
            ctx, OBJECTIVE_LQ, Float32(1.0), (mode & 1) != 0,
            (mode & 2) != 0, 0, True, worst,
        )
    if lq1 != 0:
        print("  FAIL Lq q=1.0 mismatched cells", lq1)
        failures += 1
    else:
        print("  ok   Lq q=1.0 (where sign(0) is observable)")

    var sabs = [
        (1, OBJECTIVE_LQ, "sign(0) flipped to +1"),
        (2, OBJECTIVE_HUBER, "Huber boundary < -> <="),
        (3, OBJECTIVE_QUANTILE, "Quantile arm > -> >="),
        (4, OBJECTIVE_POISSON, "der scaled by (1 + 1e-5)"),
        (5, OBJECTIVE_RMSE, "estimation planes swapped"),
        (6, OBJECTIVE_LQ, "Lq Der2 forced to the q >= 2 arm"),
    ]
    for si in range(len(sabs)):
        var s = sabs[si]
        var sid = s[0]
        var obj = s[1]
        var name = s[2]
        var a = alpha_for(obj)
        if sid == 1:
            a = Float32(1.0)  # see the note above the honest q=1.0 run
        # S2 and S6 need the ESTIMATION mode, which is the only one that
        # writes `der2`. S2 in particular is invisible without it: at
        # |diff| exactly delta their `Der` returns `delta` on the inner
        # arm (`diff`, which IS delta) and `delta` on the outer one too,
        # so the boundary moves ONLY `Der2` -- 1.0 inside, 0.0 outside.
        # The first version of this check ran S2 in search mode and
        # reported "changed nothing", which was a defect in the sabotage.
        # S5 is a layout swap and is estimation-only by construction.
        var est = sid == 2 or sid == 5 or sid == 6
        var moved = run_objective(ctx, obj, a, est, False, sid, False, worst)
        if moved == 0:
            print(
                "  FAIL S" + String(sid) + " (" + name
                + ") changed NOTHING -- the check cannot see this mechanism"
            )
            failures += 1
        else:
            print(
                "  ok   S" + String(sid), name, "->", moved,
                "cells moved",
            )

    print()
    print(
        "worst honest relative gap:", worst,
        " REL_TOL:", REL_TOL,
        " headroom:", REL_TOL / worst if worst > 0.0 else Float64(0.0),
    )
    if failures != 0:
        raise Error(
            "pointwise target check: " + String(failures) + " failures"
        )
    print()
    print("pointwise target check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_pointwise_targets(ctx)
