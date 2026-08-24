"""The batched Kalman filter for ARIMA: state-space matrices, initial state
and covariance, the per-series filter loop, the log-likelihood, the forecast.

PORT OF `cuml/cpp/src/arima/batched_kalman.cu` at cuML 265b9da6 (v26.08.00):
`Mv_l` / `MM_l` / `numerical_stability` (:34-92),
`batched_kalman_loop_kernel` (:117-333, the `rd <= 8` one-thread-per-series
kernel their dispatch takes at `:772`), `batched_kalman_loop` (:746-819),
`_lyapunov_wrapper` (:845-886, the `r <= 5` direct arm),
`_batched_kalman_filter` (:889-1139), `init_batched_kalman_matrices`
(:1141-1245), `batched_kalman_filter` (:1248-1303). COPY, DO NOT IMPROVE.
Layout is theirs: series `b` contiguous in `ys`/`pred` (`bid * nobs`),
`T` at `bid * rd * rd` column-major, `Z`/`R`/`alpha` at `bid * rd`.

NOT PORTED, each refused by name one layer up (`arima_common.mojo::
validate_order`, `batched_arima.mojo`): the `rd > 8` block-per-series
kernel (`_batched_kalman_device_loop_large_kernel`, `linalg/block.cuh`);
the `r > 5` Schur Lyapunov arm; exogenous regressors (`d_exog`, `d_beta`,
the two cuBLAS gemms at `:925-970`); confidence intervals (`level > 0`,
`d_F_fc`, `confidence_intervals` kernel, host `erfinv`); MISSING
OBSERVATIONS (`isnan(yt)` arms at `:191,193,219,236,246`; NaN is refused at the
surface so those arms would be unreachable, and an unreached branch is an
unchecked one -- PORTING_RULES 8). `arima/UNPORTED.tsv` lists each.

PRECISION: DEVIATION 670 (`arima_common.mojo`): Float32 where theirs is
`double`.

THE SEAMS, statement for statement (IDENTITY_PATHS rows 9/10/12):
  `sum += A[i + j*n] * v[j]`            Mv_l/MM_l  -> identical_mul_add, k ascending
  `_Fs += P[j*rd+i] * Z[i] * Z[j]`       -> t = P*Z[i]; identical_mul_add(t, Z[j], F)
  `log(_Fs)`                             -> identical_log
  `vs*vs / _Fs`                          -> (vs*vs) then / F, one local each
  `_1_Fs = 1.0 / _Fs`; `K[i] = _1_Fs * TP[i]`
  `alpha[i] = tmp[i] + K[i] * vs`        -> identical_mul_add(K, vs, tmp)
  `alpha[n_diff] += mu`                  (always, mu = 0 without intercept;
                                          so a -0.0 state becomes +0.0, as theirs)
  `L[j*rd+i] -= K[i] * Z[j]`             -> identical_mul_add(-K, Z, L)
  `P += RQR`; `0.5 * (A + A')`; `|A_ii|`  -> one local per op
  `-.5 * (sum_logFs + n * (ll_s2 + log(2 pi)))` -> identical_mul_add(n, ll_s2 + C, sum_logFs)
`log(2 * M_PI)` is a compile-time constant in theirs and here
(`LOG_2PI = 1.8378770664093453` rounded once to Float32); no transcendental
is evaluated for it.

=============================================================================
DEVIATION 673: A NON-POSITIVE INNOVATION VARIANCE IS REFUSED, NOT FILTERED
=============================================================================
THEIRS. `F = Z P Z'` goes straight into `log(_Fs)` and `vs*vs / _Fs`
(`batched_kalman.cu:209-210`); a non-positive `F` makes the log-likelihood
NaN/-inf, whose payload is the vendor's, in a recorded stage.
OURS (ADDENDUM 11: no computed NaN in a hashed stage). A non-positive `F`
at a summed step sets `info[bid]` to `it + 1` and the host RAISES BY NAME
after the launch; the loop still runs to the end so every series is
checked. With `sigma2 >= 1e-6` (the Jones floor) and the
symmetric/positive-diagonal stabilization it has not fired on any planted
fixture; the gate `check_kalman_refuses_by_name` reaches it with
`sigma2 = 0` and `trans = False`. MEASURED there: the raise names the
series and the step.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from arima.ported.linalg.batched.matrix import (
    LYAP_R2_MAX,
    kron_minus_identity,
    lu_inverse,
)
from arima.ported.timeSeries.arima_helpers import (
    param_to_poly,
    reduced_poly_indices,
    reduced_polynomial,
)
from arima.ported.tsa.arima_common import ARIMAOrder, ARIMAParams
from mojo_only.numerics import ftz, identical_log, identical_mul_add


comptime RD_MAX = 8
comptime RD2_MAX = RD_MAX * RD_MAX
comptime LOG_2PI = Float32(1.8378770664093453)
comptime KAPPA = Float32(1e6)
comptime KALMAN_TPB = 32
comptime INIT_TPB = 128


def _grid(n: Int, tpb: Int) -> Int:
    return (n + tpb - 1) // tpb


# ---------------------------------------------------------------------------
# init_batched_kalman_matrices (:1141-1245)
# ---------------------------------------------------------------------------


def init_batched_kalman_matrices_kernel(
    d_ar: MutPointer[Float32, MutAnyOrigin],
    d_ma: MutPointer[Float32, MutAnyOrigin],
    d_sar: MutPointer[Float32, MutAnyOrigin],
    d_sma: MutPointer[Float32, MutAnyOrigin],
    d_Z_b: MutPointer[Float32, MutAnyOrigin],
    d_R_b: MutPointer[Float32, MutAnyOrigin],
    d_T_b: MutPointer[Float32, MutAnyOrigin],
    nb_in: Int32,
    p_in: Int32, d_in: Int32, q_in: Int32,
    P_in: Int32, D_in: Int32, Q_in: Int32, s_in: Int32,
):
    """One thread per series (their `thrust::for_each` lambda, `:1166-1244`),
    after the three memsets (`:1158-1160`, done here per series)."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(nb_in):
        return
    var p = Int(p_in)
    var d = Int(d_in)
    var q = Int(q_in)
    var P = Int(P_in)
    var D = Int(D_in)
    var Q = Int(Q_in)
    var s = Int(s_in)
    var n_diff = d + s * D
    var n_phi = p + s * P
    var n_theta = q + s * Q
    var r = n_phi if n_phi > n_theta + 1 else n_theta + 1
    var rd = n_diff + r
    var zb = bid * rd
    var tb = bid * rd * rd
    for i in range(rd):
        d_Z_b.unsafe_store(zb + i, Float32(0.0))
        d_R_b.unsafe_store(zb + i, Float32(0.0))
    for i in range(rd * rd):
        d_T_b.unsafe_store(tb + i, Float32(0.0))

    # Z = [ 1 | 0 . . 0 1 0 . . 0 1 | 1 0 . . 0 ]
    for i in range(d):
        d_Z_b.unsafe_store(zb + i, Float32(1.0))
    for i in range(1, D + 1):
        d_Z_b.unsafe_store(zb + d + i * s - 1, Float32(1.0))
    d_Z_b.unsafe_store(zb + n_diff, Float32(1.0))

    # R = [ 0 .. 0 | 1 theta_1 .. theta_{r-1} ]
    d_R_b.unsafe_store(zb + n_diff, Float32(1.0))
    for i in range(n_theta):
        var ix = reduced_poly_indices(i + 1, s)
        var c0 = param_to_poly(False, d_ma.unsafe_load(bid * q + ix[0] - 1) if (ix[0] != 0 and ix[0] <= q) else Float32(0.0), ix[0], q)
        var c1 = param_to_poly(False, d_sma.unsafe_load(bid * Q + ix[1] - 1) if (ix[1] != 0 and ix[1] <= Q) else Float32(0.0), ix[1], Q)
        d_R_b.unsafe_store(zb + n_diff + i + 1, reduced_polynomial(False, c0, c1))

    # T: 1. differencing component
    for i in range(d):
        for j in range(i, d):
            d_T_b.unsafe_store(tb + j * rd + i, Float32(1.0))
    for id_ in range(d):
        d_T_b.unsafe_store(tb + n_diff * rd + id_, Float32(1.0))
        for iD in range(1, D + 1):
            d_T_b.unsafe_store(tb + (d + s * iD - 1) * rd + id_, Float32(1.0))
    # 2. seasonal differencing component
    for iD in range(D):
        var offset = d + iD * s
        for i in range(s - 1):
            d_T_b.unsafe_store(tb + (offset + i) * rd + offset + i + 1, Float32(1.0))
        d_T_b.unsafe_store(tb + (offset + s - 1) * rd + offset, Float32(1.0))
        d_T_b.unsafe_store(tb + n_diff * rd + offset, Float32(1.0))
    if D == 2:
        d_T_b.unsafe_store(tb + (n_diff - 1) * rd + d, Float32(1.0))
    # 3. auto-regressive component
    for i in range(n_phi):
        var ix = reduced_poly_indices(i + 1, s)
        var c0 = param_to_poly(True, d_ar.unsafe_load(bid * p + ix[0] - 1) if (ix[0] != 0 and ix[0] <= p) else Float32(0.0), ix[0], p)
        var c1 = param_to_poly(True, d_sar.unsafe_load(bid * P + ix[1] - 1) if (ix[1] != 0 and ix[1] <= P) else Float32(0.0), ix[1], P)
        d_T_b.unsafe_store(tb + n_diff * (rd + 1) + i, reduced_polynomial(True, c0, c1))
    for i in range(r - 1):
        d_T_b.unsafe_store(tb + (n_diff + i + 1) * rd + n_diff + i, Float32(1.0))

    # If rd=2 and phi_2=-1, I-TxT is singular (:1243-1244)
    if rd == 2 and p == 2:
        var t1 = ftz(d_T_b.unsafe_load(tb + 1))
        if abs(ftz(t1 + Float32(1.0))) < Float32(0.01):
            d_T_b.unsafe_store(tb + 1, Float32(-0.99))


# ---------------------------------------------------------------------------
# RQR, P0, alpha0 (:976-1109), one thread per series
# ---------------------------------------------------------------------------


def kalman_init_state_kernel(
    d_R: MutPointer[Float32, MutAnyOrigin],
    d_T: MutPointer[Float32, MutAnyOrigin],
    d_sigma2: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    d_RQ: MutPointer[Float32, MutAnyOrigin],
    d_RQR: MutPointer[Float32, MutAnyOrigin],
    d_P: MutPointer[Float32, MutAnyOrigin],
    d_alpha: MutPointer[Float32, MutAnyOrigin],
    d_ImAA: MutPointer[Float32, MutAnyOrigin],
    d_ImAA_inv: MutPointer[Float32, MutAnyOrigin],
    d_piv: MutPointer[Int32, MutAnyOrigin],
    d_vecq: MutPointer[Float32, MutAnyOrigin],
    d_ImT: MutPointer[Float32, MutAnyOrigin],
    d_ImT_inv: MutPointer[Float32, MutAnyOrigin],
    d_info: MutPointer[Int32, MutAnyOrigin],
    batch_size_in: Int32,
    rd_in: Int32,
    r_in: Int32,
    n_diff_in: Int32,
    intercept_in: Int32,
):
    """`_batched_kalman_filter` :976-1109 for one series: `RQ = R * sigma2`,
    `RQR = RQ R'` (a k=1 gemm: one product per cell), `P0` (diffuse `kappa`
    on the first `n_diff` diagonal cells, the stationary block from the
    direct Lyapunov solve -- DEVIATION 674), `alpha0` (`(I - T*)^-1 c` with
    the `r == 1` guard, or zeros). `d_info[bid]` is 0 or the failed column
    of a singular solve (then the host raises)."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var rd = Int(rd_in)
    var r = Int(r_in)
    var n_diff = Int(n_diff_in)
    var rd2 = rd * rd
    var r2 = r * r
    var vb = bid * rd
    var mb = bid * rd2
    var info = Int32(0)

    # RQ = R * sigma2 (:981-987); RQR = RQ * R' (:990)
    var sigma2 = ftz(d_sigma2.unsafe_load(bid))
    for i in range(rd):
        d_RQ.unsafe_store(vb + i, ftz(ftz(d_R.unsafe_load(vb + i)) * sigma2))
    for j in range(rd):
        var rj = ftz(d_R.unsafe_load(vb + j))
        for i in range(rd):
            d_RQR.unsafe_store(mb + i + j * rd, ftz(ftz(d_RQ.unsafe_load(vb + i)) * rj))

    # P0 (:993-1035)
    for i in range(rd2):
        d_P.unsafe_store(mb + i, Float32(0.0))
    for i in range(n_diff):
        d_P.unsafe_store(mb + (rd + 1) * i, KAPPA)
    # the stationary block: X solves Ts X Ts' - X + RQRs = 0 where Ts, RQRs
    # are the r x r blocks at offset n_diff (b_2dcopy folded into the read)
    var kb = bid * r2 * r2
    kron_minus_identity(d_T, mb, rd, n_diff, r, d_ImAA, kb)
    var inf1 = lu_inverse(d_ImAA, kb, d_ImAA_inv, kb, r2, d_piv, bid * LYAP_R2_MAX)
    if inf1 != 0:
        info = inf1
    else:
        var qb = bid * r2
        for j in range(r):
            for i in range(r):
                d_vecq.unsafe_store(qb + i + j * r, ftz(d_RQR.unsafe_load(mb + (i + n_diff) + (j + n_diff) * rd)))
        # X = inv * vec(Q), then b_2dcopy(Ps, P, 0, 0, r, r, n_diff, n_diff)
        # X = inv * vec(Q). NOT in place: matvec_serial(out == v) would let
        # row i+1 read row i's RESULT (their b_gemm writes a distinct
        # buffer); the product goes to a local first, matvec_serial's exact
        # spelling (ftz loads, ascending fma), then to P.
        var xloc = InlineArray[Float32, LYAP_R2_MAX](fill=Float32(0.0))
        for i in range(r2):
            var acc = Float32(0.0)
            for k in range(r2):
                var av = ftz(d_ImAA_inv.unsafe_load(kb + i + k * r2))
                var xv = ftz(d_vecq.unsafe_load(qb + k))
                acc = ftz(identical_mul_add(av, xv, acc))
            xloc[i] = acc
        for j in range(r):
            for i in range(r):
                d_P.unsafe_store(mb + (i + n_diff) + (j + n_diff) * rd, xloc[i + j * r])

    # alpha0 (:1043-1109)
    if intercept_in != 0:
        var ib = bid * r2
        for j in range(r):
            for i in range(r):
                var delta = Float32(1.0) if i == j else Float32(0.0)
                var tij = ftz(d_T.unsafe_load(mb + (i + n_diff) + (j + n_diff) * rd))
                d_ImT.unsafe_store(ib + i + j * r, ftz(delta - tij))
        if r == 1:
            var v = d_ImT.unsafe_load(ib)
            if abs(v) < Float32(1e-3):
                # raft::signPrim: signbit(x) ? -1 : +1
                var neg = (bitcast[DType.uint32](v) >> 31) != 0
                d_ImT.unsafe_store(ib, Float32(-1e-3) if neg else Float32(1e-3))
        var inf2 = lu_inverse(d_ImT, ib, d_ImT_inv, ib, r, d_piv, bid * LYAP_R2_MAX)
        if inf2 != 0 and info == 0:
            info = inf2
        var mu = ftz(d_mu.unsafe_load(bid))
        for i in range(n_diff):
            d_alpha.unsafe_store(vb + i, Float32(0.0))
        if inf2 == 0:
            for i in range(r):
                d_alpha.unsafe_store(vb + i + n_diff, ftz(ftz(d_ImT_inv.unsafe_load(ib + i)) * mu))
        else:
            for i in range(r):
                d_alpha.unsafe_store(vb + i + n_diff, Float32(0.0))
    else:
        for i in range(rd):
            d_alpha.unsafe_store(vb + i, Float32(0.0))
    d_info.unsafe_store(bid, info)


# ---------------------------------------------------------------------------
# the Kalman loop (:117-333), one thread per series, rd <= 8
# ---------------------------------------------------------------------------


@always_inline
def _mv(n: Int, alpha: Float32, a: InlineArray[Float32, RD2_MAX], v: InlineArray[Float32, RD_MAX], mut out_v: InlineArray[Float32, RD_MAX]):
    """`Mv_l(n, alpha, A, v, out)` (`:45-56`): `out[i] = alpha * sum_j
    A[i + j*n] v[j]`, j ascending.

    Their TWO overloads (`:34-43` unscaled, `:45-56` scaled) are one
    function here, called with `alpha = 1` where they call the unscaled one.
    `1.0 * x` is exact for every IEEE-754 value including subnormals, +-0
    and infinities, so no bit moves; the collapse is recorded, not silent."""
    for i in range(n):
        var acc = Float32(0.0)
        for j in range(n):
            acc = ftz(identical_mul_add(a[i + j * n], v[j], acc))
        out_v[i] = ftz(alpha * acc)


@always_inline
def _mm(n: Int, a: InlineArray[Float32, RD2_MAX], b: InlineArray[Float32, RD2_MAX], bT: Bool, mut out_v: InlineArray[Float32, RD2_MAX]):
    """`MM_l<false, bT>(n, A, B, out)` (`:57-70`): `out[i + j*n] =
    sum_k A[i + k*n] * (bT ? B[j + k*n] : B[k + j*n])`, k ascending. Their
    `aT` template parameter is never instantiated true anywhere this file
    reaches, so the `aT` read is not offered (`arima/UNPORTED.tsv`)."""
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for k in range(n):
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc = ftz(identical_mul_add(a[i + k * n], bkj, acc))
            out_v[i + j * n] = acc


@always_inline
def _numerical_stability(n: Int, mut a: InlineArray[Float32, RD2_MAX]):
    """`:76-92`: `A = 0.5 (A + A')`, `A_ii = |A_ii|`."""
    for i in range(n - 1):
        for j in range(i + 1, n):
            var s = ftz(a[j * n + i] + a[i * n + j])
            var new_val = ftz(Float32(0.5) * s)
            a[j * n + i] = new_val
            a[i * n + j] = new_val
    for i in range(n):
        a[i * n + i] = abs(a[i * n + i])


def batched_kalman_loop_kernel(
    ys: MutPointer[Float32, MutAnyOrigin],
    T: MutPointer[Float32, MutAnyOrigin],
    Z: MutPointer[Float32, MutAnyOrigin],
    RQR: MutPointer[Float32, MutAnyOrigin],
    P: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    d_pred: MutPointer[Float32, MutAnyOrigin],
    d_vs: MutPointer[Float32, MutAnyOrigin],
    d_loglike: MutPointer[Float32, MutAnyOrigin],
    d_fc: MutPointer[Float32, MutAnyOrigin],
    d_info: MutPointer[Int32, MutAnyOrigin],
    rd_in: Int32,
    nobs_in: Int32,
    batch_size_in: Int32,
    intercept_in: Int32,
    n_diff_in: Int32,
    fc_steps_in: Int32,
):
    """`batched_kalman_loop_kernel` (:117-333) without the `missing` arms,
    `d_obs_inter` (exog) and `conf_int`. `d_vs` (the innovations `y - pred`)
    is ours, for the card's `arima.resid`; theirs keeps `vs_it` in a
    register. `d_info[bid]` is ours: 0, or `it + 1` at the first summed step
    whose `F <= 0`."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var rd = Int(rd_in)
    var rd2 = rd * rd
    var nobs = Int(nobs_in)
    var n_diff = Int(n_diff_in)
    var fc_steps = Int(fc_steps_in)
    var l_RQR = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_T = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_Z = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_P = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_alpha = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_K = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_tmp = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_TP = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    # ONE VARIABLE OF THEIRS SPLIT INTO TWO OF OURS, deliberately and
    # recorded: their `l_tmp[rd2_max]` is BOTH the `T*alpha` vector of step 4
    # and the `L` matrix of step 5 (`:234,241`). Ours keeps `l_v` for the
    # vector. Step 5's first statement rewrites ALL rd2 cells of `l_tmp`
    # from `l_T` before any read, so no cell of theirs is ever read carrying
    # step 4's value and no bit depends on the reuse; the split is spelled
    # out here because a silent one is the class of defect the audit hunts.
    var l_v = InlineArray[Float32, RD_MAX](fill=Float32(0.0))

    var b_rd = bid * rd
    var b_rd2 = bid * rd2
    for i in range(rd2):
        l_RQR[i] = ftz(RQR.unsafe_load(b_rd2 + i))
        l_T[i] = ftz(T.unsafe_load(b_rd2 + i))
        l_P[i] = ftz(P.unsafe_load(b_rd2 + i))
    for i in range(rd):
        if n_diff > 0:
            l_Z[i] = ftz(Z.unsafe_load(b_rd + i))
        l_alpha[i] = ftz(alpha.unsafe_load(b_rd + i))

    var b_sum_logFs = Float32(0.0)
    var b_ll_s2 = Float32(0.0)
    var n_obs_ll = 0
    var info = Int32(0)
    var b_ys = bid * nobs
    var mu = ftz(d_mu.unsafe_load(bid)) if intercept_in != 0 else Float32(0.0)

    for it in range(nobs):
        # 1. v = y - Z*alpha
        var pred = Float32(0.0)
        if n_diff == 0:
            pred = ftz(pred + l_alpha[0])
        else:
            for i in range(rd):
                pred = ftz(identical_mul_add(l_alpha[i], l_Z[i], pred))
        d_pred.unsafe_store(b_ys + it, pred)
        var yt = ftz(ys.unsafe_load(b_ys + it))
        var vs_it = ftz(yt - pred)
        d_vs.unsafe_store(b_ys + it, vs_it)

        # 2. F = Z*P*Z'
        var _Fs = Float32(0.0)
        if n_diff == 0:
            _Fs = l_P[0]
        else:
            for i in range(rd):
                for j in range(rd):
                    var t0 = ftz(l_P[j * rd + i] * l_Z[i])
                    _Fs = ftz(identical_mul_add(t0, l_Z[j], _Fs))
        if it >= n_diff:
            if _Fs <= Float32(0.0):
                if info == 0:
                    info = Int32(it + 1)
            else:
                b_sum_logFs = ftz(b_sum_logFs + ftz(identical_log(_Fs)))
                var v2 = ftz(vs_it * vs_it)
                b_ll_s2 = ftz(b_ll_s2 + ftz(v2 / _Fs))
            n_obs_ll += 1

        # 3. K = 1/Fs * T*P*Z'
        _mm(rd, l_T, l_P, False, l_TP)
        var _1_Fs = ftz(Float32(1.0) / _Fs)
        if n_diff == 0:
            for i in range(rd):
                l_K[i] = ftz(_1_Fs * l_TP[i])
        else:
            _mv(rd, _1_Fs, l_TP, l_Z, l_K)

        # 4. alpha = T*alpha + K*vs + c
        _mv(rd, Float32(1.0), l_T, l_alpha, l_v)
        for i in range(rd):
            l_alpha[i] = ftz(identical_mul_add(l_K[i], vs_it, l_v[i]))
        l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)

        # 5. L = T - K*Z
        for i in range(rd2):
            l_tmp[i] = l_T[i]
        if n_diff == 0:
            for i in range(rd):
                l_tmp[i] = ftz(l_tmp[i] - l_K[i])
        else:
            for i in range(rd):
                for j in range(rd):
                    l_tmp[j * rd + i] = ftz(identical_mul_add(-l_K[i], l_Z[j], l_tmp[j * rd + i]))

        # 6. P = T*P*L' + R*Q*R'
        _mm(rd, l_TP, l_tmp, True, l_P)
        for i in range(rd2):
            l_P[i] = ftz(l_P[i] + l_RQR[i])
        _numerical_stability(rd, l_P)

    # log-likelihood
    var n_obs_ll_f = Float32(n_obs_ll)
    b_ll_s2 = ftz(b_ll_s2 / n_obs_ll_f)
    var inner = ftz(b_ll_s2 + LOG_2PI)
    var tot = ftz(identical_mul_add(n_obs_ll_f, inner, b_sum_logFs))
    d_loglike.unsafe_store(bid, ftz(Float32(-0.5) * tot))
    d_info.unsafe_store(bid, info)

    # forecast (no confidence intervals)
    var b_fc = bid * fc_steps
    for it in range(fc_steps):
        var pred = Float32(0.0)
        if n_diff == 0:
            pred = ftz(pred + l_alpha[0])
        else:
            for i in range(rd):
                pred = ftz(identical_mul_add(l_alpha[i], l_Z[i], pred))
        d_fc.unsafe_store(b_fc + it, pred)
        _mv(rd, Float32(1.0), l_T, l_alpha, l_v)
        for i in range(rd):
            l_alpha[i] = l_v[i]
        l_alpha[n_diff] = ftz(l_alpha[n_diff] + mu)


# ---------------------------------------------------------------------------
# host side: the buffers and the launch sequence
# ---------------------------------------------------------------------------


struct KalmanWorkspace(Movable):
    """Every device buffer `_batched_kalman_filter` reaches (their
    `ARIMAMemory` carve-outs), owned here so each outlives its last launch
    and can be recorded as a card stage."""

    var Z: DeviceBuffer[DType.float32]
    var R: DeviceBuffer[DType.float32]
    var T: DeviceBuffer[DType.float32]
    var RQ: DeviceBuffer[DType.float32]
    var RQR: DeviceBuffer[DType.float32]
    var P: DeviceBuffer[DType.float32]
    var alpha: DeviceBuffer[DType.float32]
    var ImAA: DeviceBuffer[DType.float32]
    var ImAA_inv: DeviceBuffer[DType.float32]
    var piv: DeviceBuffer[DType.int32]
    var vecq: DeviceBuffer[DType.float32]
    var ImT: DeviceBuffer[DType.float32]
    var ImT_inv: DeviceBuffer[DType.float32]
    var info_init: DeviceBuffer[DType.int32]
    var info_loop: DeviceBuffer[DType.int32]
    var pred: DeviceBuffer[DType.float32]
    var vs: DeviceBuffer[DType.float32]
    var loglike: DeviceBuffer[DType.float32]
    var fc: DeviceBuffer[DType.float32]
    var P0: DeviceBuffer[DType.float32]
    var alpha0: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, order: ARIMAOrder, batch_size: Int, n_obs: Int, fc_steps: Int) raises:
        var rd = order.rd()
        var r = order.r()
        var rd2 = rd * rd
        var r2 = r * r
        self.Z = ctx.enqueue_create_buffer[DType.float32](rd * batch_size)
        self.R = ctx.enqueue_create_buffer[DType.float32](rd * batch_size)
        self.T = ctx.enqueue_create_buffer[DType.float32](rd2 * batch_size)
        self.RQ = ctx.enqueue_create_buffer[DType.float32](rd * batch_size)
        self.RQR = ctx.enqueue_create_buffer[DType.float32](rd2 * batch_size)
        self.P = ctx.enqueue_create_buffer[DType.float32](rd2 * batch_size)
        self.alpha = ctx.enqueue_create_buffer[DType.float32](rd * batch_size)
        self.ImAA = ctx.enqueue_create_buffer[DType.float32](r2 * r2 * batch_size)
        self.ImAA_inv = ctx.enqueue_create_buffer[DType.float32](r2 * r2 * batch_size)
        self.piv = ctx.enqueue_create_buffer[DType.int32](LYAP_R2_MAX * batch_size)
        self.vecq = ctx.enqueue_create_buffer[DType.float32](r2 * batch_size)
        self.ImT = ctx.enqueue_create_buffer[DType.float32](r2 * batch_size)
        self.ImT_inv = ctx.enqueue_create_buffer[DType.float32](r2 * batch_size)
        self.info_init = ctx.enqueue_create_buffer[DType.int32](batch_size)
        self.info_loop = ctx.enqueue_create_buffer[DType.int32](batch_size)
        self.pred = ctx.enqueue_create_buffer[DType.float32](n_obs * batch_size)
        self.vs = ctx.enqueue_create_buffer[DType.float32](n_obs * batch_size)
        self.loglike = ctx.enqueue_create_buffer[DType.float32](batch_size)
        self.fc = ctx.enqueue_create_buffer[DType.float32](max(1, fc_steps * batch_size))
        self.P0 = ctx.enqueue_create_buffer[DType.float32](rd2 * batch_size)
        self.alpha0 = ctx.enqueue_create_buffer[DType.float32](rd * batch_size)


def init_batched_kalman_matrices(
    ctx: DeviceContext,
    mut params: ARIMAParams,
    batch_size: Int,
    order: ARIMAOrder,
    mut ws: KalmanWorkspace,
) raises:
    """`:1141-1245`."""
    ctx.enqueue_function[init_batched_kalman_matrices_kernel](
        params.ar.unsafe_ptr(), params.ma.unsafe_ptr(), params.sar.unsafe_ptr(),
        params.sma.unsafe_ptr(), ws.Z.unsafe_ptr(), ws.R.unsafe_ptr(), ws.T.unsafe_ptr(),
        Int32(batch_size), Int32(order.p), Int32(order.d), Int32(order.q),
        Int32(order.P), Int32(order.D), Int32(order.Q), Int32(order.s),
        grid_dim=(_grid(batch_size, INIT_TPB), 1, 1), block_dim=(INIT_TPB, 1, 1),
    )


def _read_info(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def batched_kalman_filter(
    ctx: DeviceContext,
    mut d_ys: DeviceBuffer[DType.float32],
    nobs: Int,
    mut params: ARIMAParams,
    order: ARIMAOrder,
    batch_size: Int,
    fc_steps: Int,
    kalman_tpb: Int = KALMAN_TPB,
) raises -> KalmanWorkspace:
    """`batched_kalman_filter` (:1248-1303) -> `_batched_kalman_filter`
    (:889-1139) -> `batched_kalman_loop` (:746-819, the `rd <= 8` arm),
    launch for launch; `level` (confidence intervals) and `exog` are not
    parameters because they are refused one layer up. `kalman_tpb` is the
    loop kernel's block width (their `dim3(32, 1)`): SCHEDULING, varied by
    the launch-invariance gate. Raises by name when a series' Lyapunov or
    intercept system is singular or an innovation variance is not positive."""
    var rd = order.rd()
    var r = order.r()
    var n_diff = order.n_diff()
    var ws = KalmanWorkspace(ctx, order, batch_size, nobs, fc_steps)
    init_batched_kalman_matrices(ctx, params, batch_size, order, ws)
    ctx.enqueue_function[kalman_init_state_kernel](
        ws.R.unsafe_ptr(), ws.T.unsafe_ptr(), params.sigma2.unsafe_ptr(), params.mu.unsafe_ptr(),
        ws.RQ.unsafe_ptr(), ws.RQR.unsafe_ptr(), ws.P.unsafe_ptr(), ws.alpha.unsafe_ptr(),
        ws.ImAA.unsafe_ptr(), ws.ImAA_inv.unsafe_ptr(), ws.piv.unsafe_ptr(), ws.vecq.unsafe_ptr(),
        ws.ImT.unsafe_ptr(), ws.ImT_inv.unsafe_ptr(), ws.info_init.unsafe_ptr(),
        Int32(batch_size), Int32(rd), Int32(r), Int32(n_diff), Int32(order.k),
        grid_dim=(_grid(batch_size, INIT_TPB), 1, 1), block_dim=(INIT_TPB, 1, 1),
    )
    var info0 = _read_info(ctx, ws.info_init, batch_size)
    for b in range(batch_size):
        if info0[b] != 0:
            raise Error(
                "batched_kalman_filter: series " + String(b) + ": the initial-state system (I - T (x) T, or I - T* for the intercept) is singular at column "
                + String(info0[b]) + "; a unit-root parameter set is refused by name rather than filtered with a non-finite P0"
            )
    # keep P0 / alpha0 (the loop kernel reads them into registers and never
    # writes them back, so the buffers hold the initial state already; the
    # copies are the card's stages by name)
    ctx.enqueue_copy(dst_buf=ws.P0, src_buf=ws.P)
    ctx.enqueue_copy(dst_buf=ws.alpha0, src_buf=ws.alpha)
    ctx.enqueue_function[batched_kalman_loop_kernel](
        d_ys.unsafe_ptr(), ws.T.unsafe_ptr(), ws.Z.unsafe_ptr(), ws.RQR.unsafe_ptr(),
        ws.P.unsafe_ptr(), ws.alpha.unsafe_ptr(), params.mu.unsafe_ptr(),
        ws.pred.unsafe_ptr(), ws.vs.unsafe_ptr(), ws.loglike.unsafe_ptr(), ws.fc.unsafe_ptr(),
        ws.info_loop.unsafe_ptr(),
        Int32(rd), Int32(nobs), Int32(batch_size), Int32(order.k), Int32(n_diff), Int32(fc_steps),
        grid_dim=(_grid(batch_size, kalman_tpb), 1, 1), block_dim=(kalman_tpb, 1, 1),
    )
    var info1 = _read_info(ctx, ws.info_loop, batch_size)
    for b in range(batch_size):
        if info1[b] != 0:
            raise Error(
                "batched_kalman_filter: series " + String(b) + ": innovation variance F <= 0 at step "
                + String(info1[b] - 1) + "; refused by name rather than carrying log(F) into the likelihood"
            )
    return ws^
