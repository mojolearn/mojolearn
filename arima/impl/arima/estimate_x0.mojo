# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`estimate_x0`, `start_params`, `arma_least_squares`, `test_invparams`:
the starting point a `fit` needs before the optimizer can run.

PORT OF `cuml/cpp/src/arima/batched_arima.cu` at cuML 265b9da6 (v26.08.00):
`test_invparams` (:628-657), `_arma_least_squares` (:664-839), `_start_params`
(:845-946), `estimate_x0` (:948-1008). COPY, DO NOT IMPROVE, except at the
one closed call, which is `b_gels` and is DEVIATION 678 in
`arima/impl/linalg/batched/least_squares.mojo`.

NOT PORTED FROM THIS CHAIN, refused by name: the `order.n_exog > 0` block of
`_start_params` (:857-931), which regresses the endogenous series on the
exogenous ones and subtracts the fitted component. Exog is refused by
`validate_order` for the whole lane, so that block cannot be reached; the
refusal is not new here. `estimate_x0`'s `missing` arm (:975-983, `fillna`)
is likewise unreachable, because a non-finite `y` is refused by name.

=============================================================================
THE SEAM THIS FILE EXISTS TO GET RIGHT, AND WOULD GET WRONG BY COPYING
=============================================================================
`test_invparams` and `invtransform` are THE SAME MATHEMATICS WITH A
DIFFERENT ASSOCIATION, and the already-ported `invtransform` spelling is the
WRONG one to paste in here.

    test_invparams  (batched_arima.cu:645)
        tmp[k] = (new_params[k] + coef * a * new_params[j-k-1]) / (1 - (a*a));
        with  constexpr double coef = isAr ? 1 : -1;

    invtransform    (jones_transform.cuh:79)
        tmp[k] = (myNewParams[k] + sign * (a * myNewParams[j-k-1])) / (1 - (a*a));

C++ parses `coef * a * x` LEFT TO RIGHT as `(coef * a) * x`. `coef` is a
compile-time +-1, so `coef * a` is EXACT and folds away, and the surviving
multiply is the one that FEEDS the add -- so it contracts. **ONE ROUNDING**,
`fma(coef*a, x, new_params[k])`.

`sign * (a * x)` is parenthesized the other way: `a * x` feeds a MULTIPLY,
cannot fuse, and rounds; only the outer `sign * (...)` feeds the add.
**TWO ROUNDINGS**.

That is the exact defect the 2026-08-23 audit found in four places at once
(`arima/README.md`, "The Jones contraction was associated wrong"). Copying
`jones_transform.mojo`'s inner loop into this file -- which is the obvious
thing to do, because the two loops are otherwise character for character
identical -- would put it back. The spelling below is
`identical_mul_add(coef_a, x, new_params[k])` with `coef_a` formed first and
exactly, and `arima/SEAMS.tsv` carries the row.

The denominator is the same in both: `1 - (a*a)` has the multiply feeding
the subtract, so it fuses, `identical_mul_add(-a, a, 1)`, and it is hoisted
out of the `k` loop because it is loop-invariant in theirs too.

WHAT THE VERDICT IS WORTH GATING FOR. `test_invparams` returns a BOOLEAN, so
a one-ulp difference is invisible unless a value lands within an ulp of +-1.
`check_invparams_contraction_is_visible` therefore asserts on the host that
the two spellings differ SOMEWHERE on this fixture (the shape
`check_jones_contraction_is_visible` already uses), and the verdict itself is
recorded as the `x0.invparams` DECISION stage, one byte per series, so a
vendor that zeroes a series another vendor keeps is caught by the card even
when no float stage moves. That is the `guards` / `piv` argument again: a
decision the ALGORITHM makes from computed values belongs on the card.

=============================================================================
METAL'S 31-ARGUMENT CAP
=============================================================================
Counted, not assumed (`arima/README.md` records what it cost the holtwinters
lane). `arma_least_squares_kernel` takes 15 arguments, 16 to spare, and it
gets there by the fix that lane found: the EIGHT scratch matrices are ONE
buffer with named slices, and the slice offsets are computed inside the
kernel by `ls_off`, the same function the launcher sizes the buffer with, so
host and device cannot drift.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from arima.impl.linalg.batched.least_squares import (
    LS_MAX_COLS,
    householder_qr_solve,
)
from arima.impl.timeSeries.jones_transform import JONES_MAX_PARAMS
from arima.impl.tsa.arima_common import ARIMAOrder, ARIMAParams, validate_order
from checks.numerics import ftz, identical_mul_add
from tsa.impl.timeSeries.arima_helpers import prepare_data


comptime X0_TPB = 128

# `x0.invparams` decision bits, one byte per series per call.
comptime INVP_AR_TESTED = 1
comptime INVP_AR_VALID = 2
comptime INVP_MA_TESTED = 4
comptime INVP_MA_VALID = 8


# ---------------------------------------------------------------------------
# the scratch layout (ONE buffer, named slices) -- see the 31-argument note
# ---------------------------------------------------------------------------

comptime SL_LSAR = 0  # bm_ls_ar_res, m1 x ncols
comptime SL_LSARQ = 1  # its copy, destroyed by the QR (b_gels copies A)
comptime SL_PRE = 2  # bm_ls, the AR(p_ar) lag matrix, m2 x p_ar
comptime SL_PREQ = 3  # its copy
comptime SL_ARFIT = 4  # bm_ar_fit, m2 (holds the AR pre-fit solution)
comptime SL_RESID = 5  # bm_residual, m2
comptime SL_AFIT = 6  # bm_arma_fit, m1 (holds the ARMA solution)
comptime SL_FRES = 7  # bm_final_residual, m1
comptime SL_COUNT = 8


@always_inline
def ls_p_ar(p: Int, q: Int, s: Int) -> Int:
    """`int p_ar = std::max(ps, 2 * qs);` (`:678`)."""
    var ps = p * s
    var qs2 = 2 * q * s
    return ps if ps > qs2 else qs2


@always_inline
def ls_r(p: Int, q: Int, s: Int) -> Int:
    """`int r = std::max(p_ar + qs, ps);` (`:679`)."""
    var ps = p * s
    var a = ls_p_ar(p, q, s) + q * s
    return a if a > ps else ps


@always_inline
def ls_off(kind: Int, n_obs_d: Int, p: Int, q: Int, s: Int, k: Int) -> Int:
    """The offset of slice `kind` inside ONE series' scratch block, and with
    `kind = SL_COUNT` the block's total size. Used by the kernel AND by the
    launcher that allocates: one function, so the two cannot disagree."""
    var m1 = n_obs_d - ls_r(p, q, s)
    var ncols = p + q + k
    var m2 = (n_obs_d - ls_p_ar(p, q, s)) if q != 0 else 0
    var p_ar = ls_p_ar(p, q, s)
    var off = 0
    if kind <= SL_LSAR:
        return off
    off += m1 * ncols
    if kind <= SL_LSARQ:
        return off
    off += m1 * ncols
    if kind <= SL_PRE:
        return off
    off += m2 * p_ar
    if kind <= SL_PREQ:
        return off
    off += m2 * p_ar
    if kind <= SL_ARFIT:
        return off
    off += m2
    if kind <= SL_RESID:
        return off
    off += m2
    if kind <= SL_AFIT:
        return off
    off += m1
    if kind <= SL_FRES:
        return off
    off += m1
    return off


# ---------------------------------------------------------------------------
# test_invparams (:628-657)
# ---------------------------------------------------------------------------


@always_inline
def test_invparams(
    params: MutPointer[Float32, MutAnyOrigin], base: Int, pq: Int, is_ar: Bool
) -> Bool:
    """`test_invparams<isAr>(params, pq)` (`:634-657`): run the inverse
    recursion and STOP BEFORE the atanh step, then ask whether every value
    is strictly inside `(-1, 1)`.

    READ THE BANNER BEFORE TOUCHING THE INNER LINE. `coef * a * x` is
    `(coef*a) * x`, `coef` is an exact +-1, the surviving product feeds the
    add: ONE rounding. This is NOT `invtransform`'s `sign * (a * x)`."""
    var new_params = InlineArray[Float32, JONES_MAX_PARAMS](fill=Float32(0.0))
    var tmp = InlineArray[Float32, JONES_MAX_PARAMS](fill=Float32(0.0))
    for i in range(pq):
        var v = ftz(params.unsafe_load(base + i))
        tmp[i] = v
        new_params[i] = v
    var j = pq - 1
    while j > 0:
        var a = new_params[j]
        # `coef * a`: coef is a constexpr +-1, so this is exact and it is
        # the operand of the multiply that fuses into the add.
        var coef_a = a if is_ar else ftz(-a)
        var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
        for k in range(j):
            var num = ftz(
                identical_mul_add(coef_a, new_params[j - k - 1], new_params[k])
            )
            tmp[k] = ftz(num / den)
        for it in range(j):
            new_params[it] = tmp[it]
        j -= 1
    var result = True
    for i in range(pq):
        var v = new_params[i]
        result = result and not (v <= Float32(-1.0) or v >= Float32(1.0))
    return result


# ---------------------------------------------------------------------------
# _arma_least_squares (:664-839)
# ---------------------------------------------------------------------------


def ls_degenerate_kernel(
    d_ar: MutPointer[Float32, MutAnyOrigin],
    d_ma: MutPointer[Float32, MutAnyOrigin],
    d_sigma2: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    p_in: Int32,
    q_in: Int32,
    k_in: Int32,
    est_sigma2_in: Int32,
):
    """`:687-697`, their "too few observations for the estimate" arm: fill
    with 0, and 1 for `sigma2`. Theirs is a `cudaMemsetAsync` per kind plus
    a `thrust::fill`; one kernel here, same values, no arithmetic."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    if k_in != 0:
        d_mu.unsafe_store(bid, Float32(0.0))
    for i in range(Int(p_in)):
        d_ar.unsafe_store(Int(p_in) * bid + i, Float32(0.0))
    for i in range(Int(q_in)):
        d_ma.unsafe_store(Int(q_in) * bid + i, Float32(0.0))
    if est_sigma2_in != 0:
        d_sigma2.unsafe_store(bid, Float32(1.0))


def arma_least_squares_kernel(
    scratch: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    d_ar: MutPointer[Float32, MutAnyOrigin],
    d_ma: MutPointer[Float32, MutAnyOrigin],
    d_sigma2: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    verdict: MutPointer[UInt8, MutAnyOrigin],
    batch_size_in: Int32,
    n_obs_d_in: Int32,
    p_in: Int32,
    q_in: Int32,
    s_in: Int32,
    k_in: Int32,
    est_sigma2_in: Int32,
):
    """`_arma_least_squares`'s body (`:699-839`) for one series.

    Theirs is five kernels and four cuBLAS calls over batched matrices; ours
    is one thread per series doing the same nine steps in the same order.
    That is a REIMPLEMENTATION of the decomposition, not of the algorithm:
    the values written are theirs step for step, and every step is named
    with the upstream line it comes from."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var n_obs_d = Int(n_obs_d_in)
    var p = Int(p_in)
    var q = Int(q_in)
    var s = Int(s_in)
    var k = Int(k_in)
    var ps = p * s
    var qs = q * s
    var p_ar = ls_p_ar(p, q, s)
    var r_ls = ls_r(p, q, s)
    var m1 = n_obs_d - r_ls
    var ncols = p + q + k
    var sb = bid * ls_off(SL_COUNT, n_obs_d, p, q, s, k)
    var yb = bid * n_obs_d
    var lsar = sb + ls_off(SL_LSAR, n_obs_d, p, q, s, k)
    var lsarq = sb + ls_off(SL_LSARQ, n_obs_d, p, q, s, k)
    var afit = sb + ls_off(SL_AFIT, n_obs_d, p, q, s, k)
    var fres = sb + ls_off(SL_FRES, n_obs_d, p, q, s, k)

    info.unsafe_store(bid, Int32(0))
    verdict.unsafe_store(bid, UInt8(0))

    # -- 1. residuals of an AR(p_ar) fit, to stand in for the MA lags (:706-731)
    if q != 0:
        var pre = sb + ls_off(SL_PRE, n_obs_d, p, q, s, k)
        var preq = sb + ls_off(SL_PREQ, n_obs_d, p, q, s, k)
        var arfit = sb + ls_off(SL_ARFIT, n_obs_d, p, q, s, k)
        var resid = sb + ls_off(SL_RESID, n_obs_d, p, q, s, k)
        var m2 = n_obs_d - p_ar
        # `b_lagged_mat(bm_y, p_ar)` (:713, matrix.cuh:961-974): vec_offset
        # 0, lag period 1, so column `lag` is `y[p_ar - lag - 1 + i]`.
        for lag in range(p_ar):
            var src = yb + (p_ar - lag - 1)
            for i in range(m2):
                scratch.unsafe_store(
                    pre + lag * m2 + i, ftz(y.unsafe_load(src + i))
                )
        # `b_2dcopy(bm_y, p_ar, 0, ls_height, 1)` (:719) and the residual
        # "initialized as offset y to avoid one kernel call" (:722)
        for i in range(m2):
            var v = ftz(y.unsafe_load(yb + p_ar + i))
            scratch.unsafe_store(arfit + i, v)
            scratch.unsafe_store(resid + i, v)
        # `b_gels(bm_ls, bm_ar_fit)` (:725). b_gels COPIES A; so do we,
        # because the gemm below reads the original.
        for t in range(m2 * p_ar):
            scratch.unsafe_store(preq + t, scratch.unsafe_load(pre + t))
        var inf_pre = householder_qr_solve(scratch, preq, m2, p_ar, scratch, arfit)
        if inf_pre != Int32(0):
            # DEVIATION 678: cuML would carry a garbage solution forward
            # here with devInfoArray = nullptr. Negative info marks the AR
            # PRE-fit, so the card distinguishes the two solves.
            info.unsafe_store(bid, -inf_pre)
            if k != 0:
                d_mu.unsafe_store(bid, Float32(0.0))
            for i in range(p):
                d_ar.unsafe_store(p * bid + i, Float32(0.0))
            for i in range(q):
                d_ma.unsafe_store(q * bid + i, Float32(0.0))
            if est_sigma2_in != 0:
                d_sigma2.unsafe_store(bid, Float32(1.0))
            return
        # `b_gemm(false, false, ls_height, 1, p_ar, -1.0, bm_ls, bm_ar_fit,
        # 1.0, bm_residual)` (:728-729), "technically a gemv".
        # cublasgemmStridedBatched is CLOSED, so the association is CHOSEN:
        # serial ascending fma INTO the beta = 1 accumulator, the same
        # choice DEVIATION 674 made for the other two gemm shapes. alpha =
        # -1 is applied to the matrix operand, where it is exact.
        for i in range(m2):
            var acc = ftz(scratch.unsafe_load(resid + i))
            for c in range(p_ar):
                var av = ftz(scratch.unsafe_load(pre + c * m2 + i))
                var xv = ftz(scratch.unsafe_load(arfit + c))
                acc = ftz(identical_mul_add(-av, xv, acc))
            scratch.unsafe_store(resid + i, acc)
        # `b_lagged_mat(bm_residual, bm_ls_ar_res, q, n_obs - r, res_offset,
        # (n_obs - r) * (k + p), s)` (:732-733). A copy, no arithmetic.
        var res_offset = r_ls - p_ar - qs
        for lag in range(q):
            var src = resid + res_offset + s * (q - lag - 1)
            var dst = lsar + m1 * (k + p) + lag * m1
            for i in range(m1):
                scratch.unsafe_store(dst + i, scratch.unsafe_load(src + i))

    # -- 2. the intercept column (:736-746)
    if k != 0:
        for i in range(m1):
            scratch.unsafe_store(lsar + i, Float32(1.0))

    # -- 3. lags of y (:749-750): mat_offset (n_obs - r) * k, period s
    var ar_offset = r_ls - ps
    for lag in range(p):
        var src = yb + ar_offset + s * (p - lag - 1)
        var dst = lsar + m1 * k + lag * m1
        for i in range(m1):
            scratch.unsafe_store(dst + i, ftz(y.unsafe_load(src + i)))

    # -- 4. the target, and the residual it seeds (:754-755, :767-771)
    for i in range(m1):
        var v = ftz(y.unsafe_load(yb + r_ls + i))
        scratch.unsafe_store(afit + i, v)
        if est_sigma2_in != 0:
            scratch.unsafe_store(fres + i, v)

    # -- 5. the ARMA fit, `b_gels(bm_ls_ar_res, bm_arma_fit)` (:774)
    for t in range(m1 * ncols):
        scratch.unsafe_store(lsarq + t, scratch.unsafe_load(lsar + t))
    var inf = householder_qr_solve(scratch, lsarq, m1, ncols, scratch, afit)
    if inf != Int32(0):
        info.unsafe_store(bid, inf)
        if k != 0:
            d_mu.unsafe_store(bid, Float32(0.0))
        for i in range(p):
            d_ar.unsafe_store(p * bid + i, Float32(0.0))
        for i in range(q):
            d_ma.unsafe_store(q * bid + i, Float32(0.0))
        if est_sigma2_in != 0:
            d_sigma2.unsafe_store(bid, Float32(1.0))
        return

    # -- 6. copy the solution into the parameter vectors (:777-793)
    if k != 0:
        d_mu.unsafe_store(bid, scratch.unsafe_load(afit))
    for i in range(p):
        d_ar.unsafe_store(p * bid + i, scratch.unsafe_load(afit + i + k))
    for i in range(q):
        d_ma.unsafe_store(q * bid + i, scratch.unsafe_load(afit + i + p + k))

    # -- 7. sigma2 from the final residual (:795-820)
    if est_sigma2_in != 0:
        for i in range(m1):
            var acc = ftz(scratch.unsafe_load(fres + i))
            for c in range(ncols):
                var av = ftz(scratch.unsafe_load(lsar + c * m1 + i))
                var xv = ftz(scratch.unsafe_load(afit + c))
                acc = ftz(identical_mul_add(-av, xv, acc))
            scratch.unsafe_store(fres + i, acc)
        # `acc += res * res` (:816): the product feeds the += directly, so
        # it FUSES. The sum starts at q, not at 0, which is theirs.
        var acc2 = Float32(0.0)
        for i in range(q, m1):
            var res = ftz(scratch.unsafe_load(fres + i))
            acc2 = ftz(identical_mul_add(res, res, acc2))
        d_sigma2.unsafe_store(bid, ftz(acc2 / Float32(m1 - q)))

    # -- 8. zero anything the inverse transform would reject (:823-838)
    var v = 0
    if p != 0:
        v += INVP_AR_TESTED
        if test_invparams(d_ar, p * bid, p, True):
            v += INVP_AR_VALID
        else:
            for ip in range(p):
                d_ar.unsafe_store(p * bid + ip, Float32(0.0))
    if q != 0:
        v += INVP_MA_TESTED
        if test_invparams(d_ma, q * bid, q, False):
            v += INVP_MA_VALID
        else:
            for iq in range(q):
                d_ma.unsafe_store(q * bid + iq, Float32(0.0))
    verdict.unsafe_store(bid, UInt8(v))


@fieldwise_init
struct LeastSquaresResult(Movable):
    """What `arma_least_squares` decided, for the card and the gates.

    `degenerate` is a function of the ORDER and `n_obs` alone, so it cannot
    differ between two vendors and is NOT a card stage (the same rule that
    keeps `predict`'s shape off the card); it is returned so a gate can
    assert the arm was or was not taken."""

    var info: DeviceBuffer[DType.int32]
    var verdict: DeviceBuffer[DType.uint8]
    var degenerate: Bool


def arma_least_squares(
    ctx: DeviceContext,
    mut d_ar: DeviceBuffer[DType.float32],
    mut d_ma: DeviceBuffer[DType.float32],
    mut d_sigma2: DeviceBuffer[DType.float32],
    mut d_mu: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_d: Int,
    p: Int,
    q: Int,
    s: Int,
    estimate_sigma2: Bool,
    k: Int,
) raises -> LeastSquaresResult:
    """`_arma_least_squares` (`:664-839`). `s = 1` for the non-seasonal call,
    `order.s` for the seasonal one; their note "in this function the
    non-seasonal case has s=1, not s=0" is why."""
    var p_ar = ls_p_ar(p, q, s)
    var r_ls = ls_r(p, q, s)
    var info = ctx.enqueue_create_buffer[DType.int32](max(1, batch_size))
    var verdict = ctx.enqueue_create_buffer[DType.uint8](max(1, batch_size))
    var grid = (batch_size + X0_TPB - 1) // X0_TPB

    # THE REFUSAL IS THEIRS AND IT IS ON THE HOST IN THEIRS TOO (`:685`), so
    # this is the same branch in the same place, not a decomposition change.
    # The two clauses are exactly `b_gels`'s `m > n` requirement for the two
    # solves: the first says the AR pre-fit is overdetermined, the second
    # says the ARMA fit is.
    if (q != 0 and p_ar >= n_obs_d - p_ar) or (p + q + k >= n_obs_d - r_ls):
        ctx.enqueue_function[ls_degenerate_kernel](
            d_ar.unsafe_ptr(), d_ma.unsafe_ptr(), d_sigma2.unsafe_ptr(),
            d_mu.unsafe_ptr(), Int32(batch_size), Int32(p), Int32(q), Int32(k),
            Int32(1 if estimate_sigma2 else 0),
            grid_dim=(grid, 1, 1), block_dim=(X0_TPB, 1, 1),
        )
        var zi = ctx.enqueue_create_host_buffer[DType.int32](max(1, batch_size))
        var zb = ctx.enqueue_create_host_buffer[DType.uint8](max(1, batch_size))
        for i in range(batch_size):
            zi.unsafe_ptr().unsafe_store(i, Int32(0))
            zb.unsafe_ptr().unsafe_store(i, UInt8(0))
        ctx.enqueue_copy(dst_buf=info, src_ptr=zi.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=verdict, src_ptr=zb.unsafe_ptr())
        ctx.synchronize()
        _ = zi^
        _ = zb^
        return LeastSquaresResult(info=info^, verdict=verdict^, degenerate=True)

    if p + q + k > LS_MAX_COLS or p_ar > LS_MAX_COLS:
        raise Error(
            "estimate_x0: the least-squares system has "
            + String(max(p + q + k, p_ar))
            + " columns, above LS_MAX_COLS = " + String(LS_MAX_COLS)
            + "; refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )

    var per = ls_off(SL_COUNT, n_obs_d, p, q, s, k)
    var scratch = ctx.enqueue_create_buffer[DType.float32](max(1, per * batch_size))
    ctx.enqueue_function[arma_least_squares_kernel](
        scratch.unsafe_ptr(), d_y.unsafe_ptr(), d_ar.unsafe_ptr(),
        d_ma.unsafe_ptr(), d_sigma2.unsafe_ptr(), d_mu.unsafe_ptr(),
        info.unsafe_ptr(), verdict.unsafe_ptr(),
        Int32(batch_size), Int32(n_obs_d), Int32(p), Int32(q), Int32(s),
        Int32(k), Int32(1 if estimate_sigma2 else 0),
        grid_dim=(grid, 1, 1), block_dim=(X0_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = scratch^
    return LeastSquaresResult(info=info^, verdict=verdict^, degenerate=False)


# ---------------------------------------------------------------------------
# _start_params (:845-946) and estimate_x0 (:948-1008)
# ---------------------------------------------------------------------------


@fieldwise_init
struct StartParamsResult(Movable):
    """The two `_arma_least_squares` calls' decision stages, kept apart
    because the seasonal call solves a different system on the same data."""

    var ns: LeastSquaresResult
    var ns_run: Bool
    var seasonal: LeastSquaresResult
    var seasonal_run: Bool


def _empty_ls(ctx: DeviceContext) raises -> LeastSquaresResult:
    """A result object for a call that did not run. `degenerate = False`, so
    no gate can mistake "not called" for "the refusal arm fired"."""
    var i = ctx.enqueue_create_buffer[DType.int32](1)
    var b = ctx.enqueue_create_buffer[DType.uint8](1)
    return LeastSquaresResult(info=i^, verdict=b^, degenerate=False)


def start_params(
    ctx: DeviceContext,
    mut params: ARIMAParams,
    mut d_yd: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs_d: Int,
    order: ARIMAOrder,
) raises -> StartParamsResult:
    """`_start_params` (`:845-946`) with `n_exog = 0`, which is the only arm
    this lane can reach.

    Note `estimate_sigma2` on the seasonal call: `order.p + order.q +
    order.k == 0` (`:945`). The non-seasonal call estimates `sigma2` when it
    runs, and the seasonal one only estimates it when the non-seasonal one
    did NOT run, so exactly one of the two writes `sigma2`.

    Written as four explicit arms rather than two reassignments because a
    `LeastSquaresResult` owns device buffers and is move-only."""
    var ns_run = order.p + order.q + order.k != 0
    var seasonal_run = order.P + order.Q != 0
    if ns_run and seasonal_run:
        var a = arma_least_squares(
            ctx, params.ar, params.ma, params.sigma2, params.mu, d_yd,
            batch_size, n_obs_d, order.p, order.q, 1, True, order.k,
        )
        var b = arma_least_squares(
            ctx, params.sar, params.sma, params.sigma2, params.mu, d_yd,
            batch_size, n_obs_d, order.P, order.Q, order.s, False, 0,
        )
        return StartParamsResult(ns=a^, ns_run=True, seasonal=b^, seasonal_run=True)
    if ns_run:
        var a = arma_least_squares(
            ctx, params.ar, params.ma, params.sigma2, params.mu, d_yd,
            batch_size, n_obs_d, order.p, order.q, 1, True, order.k,
        )
        var b = _empty_ls(ctx)
        return StartParamsResult(ns=a^, ns_run=True, seasonal=b^, seasonal_run=False)
    if seasonal_run:
        var a = _empty_ls(ctx)
        var b = arma_least_squares(
            ctx, params.sar, params.sma, params.sigma2, params.mu, d_yd,
            batch_size, n_obs_d, order.P, order.Q, order.s, True, 0,
        )
        return StartParamsResult(ns=a^, ns_run=False, seasonal=b^, seasonal_run=True)
    # unreachable: `validate_order` refuses p + q + P + Q + k == 0
    raise Error(
        "start_params: the order has no parameters to estimate;"
        " validate_order should have refused it first"
    )


def estimate_x0(
    ctx: DeviceContext,
    mut params: ARIMAParams,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    order: ARIMAOrder,
) raises -> StartParamsResult:
    """`estimate_x0` (`:948-1008`) with `missing = false` and `n_exog = 0`:
    difference, then `_start_params`. Writes `params` in place.

    Their `RAFT_FAIL` on `n_obs <= d + s*D` (`:989`) is kept as a raise by
    name; `prepare_data` would otherwise be handed a negative length."""
    validate_order(order)
    var d_sD = order.n_diff()
    if n_obs <= d_sD:
        raise Error(
            "estimate_x0: n_obs (" + String(n_obs)
            + ") must be greater than d + s*D (" + String(d_sD)
            + ") for differencing"
        )
    var n_obs_d = n_obs - d_sD
    var yd = ctx.enqueue_create_buffer[DType.float32](max(1, n_obs_d * batch_size))
    prepare_data(ctx, yd, d_y, batch_size, n_obs, order.d, order.D, order.s)
    var out = start_params(ctx, params, yd, batch_size, n_obs_d, order)
    ctx.synchronize()
    _ = yd^
    return out^
