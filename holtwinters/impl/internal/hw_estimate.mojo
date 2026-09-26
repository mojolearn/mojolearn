# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`initialization_method="estimated"`: the initial level, trend and every
seasonal state are ESTIMATED jointly with alpha, beta and gamma, over all
`n` observations. OURS: cuML has no such path. It is the definition
statsmodels' `ExponentialSmoothing(..., initialization_method="estimated")`
fits (its default), and since 2026-09-22 it is mojolearn's default too.
`initialization_method="heuristic"` (alias "cuml") keeps cuML's fit,
bit for bit (`runner.mojo::holtwinters_fit_helper`, untouched).

THE MODEL. theta = [alpha, beta, gamma, l0, b0, s_0 .. s_{f-1}], `d = f +
5` values per series. From `(l, b, s) = (l0, b0, s)` the recurrence is
`hw_eval.mojo`'s, run from `t = 0` instead of `t = f`:

    leveltrend = l + b
    xhat_t     = leveltrend + s_{t % f}          | leveltrend * s_{t % f}
    l'         = alpha (y_t - s) + (1 - alpha) leveltrend     | alpha y_t / s + ...
    b'         = beta (l' - l) + (1 - beta) b
    s'_{t % f} = gamma (y_t - l') + (1 - gamma) s              | gamma y_t / l' + ...

and the objective is `SSE = sum_{t=0}^{n-1} (y_t - xhat_t)^2`, all `n`
points (cuML's drops the first `f`). The fitted components the estimator
returns are the states at `t = f .. n - 1`, so `level_`, `trend_` and
`season_` keep their `(ts_num, n - f)` shape and meaning and `forecast`,
`predict`, `save` and `load` read them unchanged.

THE OPTIMIZER: Levenberg-Marquardt on the residual vector, deterministic
and serial. One thread per series; every loop has a fixed order and a fixed
bound; no atomic, no reduction across threads, no libm.
  * The Jacobian of `xhat` in theta is carried forward through the
    recurrence (forward-mode, exact, `d` values per state), so each
    evaluation yields the SSE, `J^T J` (upper triangle, accumulated in `t`
    order) and `J^T e` in one pass.
  * Step: `(J^T J + lambda diag) delta = J^T e`, `diag_i = max(J^T J_ii,
    1e-6 * max_k J^T J_kk)` (Marquardt's scaling), solved by unpivoted
    Gaussian elimination on the symmetric positive definite system (no
    square root). A pivot that is not `> 0` rejects the step.
  * Bounds: alpha, beta, gamma in [0, 1]. The trial point is clamped by
    `bound_device`'s compare chain (DEVIATION 663); a bounded parameter
    sitting on its bound whose gradient points out is held fixed for that
    step (its row and column are replaced by the identity).
  * Accept iff the trial SSE is STRICTLY lower (a NaN never is); then
    `lambda /= 3` (floor 1e-7), else `lambda *= 4`. Stop on a relative
    decrease below 1e-6 (criterion MIN_ERROR_DIFF), on `lambda > 1e10`
    (MIN_PARAM_DIFF: no step lowers the SSE) or after 100 iterations
    (BFGS_ITER_LIMIT, the name the enum already has).
  * THREE FIXED STARTS for (alpha, beta, gamma): (0.4, 0.3, 0.3) (cuML's),
    (0.2, 0.01, 0.01) and (0.8, 0.1, 0.1), each with the initial states
    seeded from cuML's heuristic decomposition (`hw.start.*`): `l0 = level
    - f * trend`, `b0 = trend`, `s = season`. The lowest final SSE wins, the
    earlier start on a tie. Measured on the 30-series comparison suite
    (float32 restatement, 2026-09-22): one start lands in a worse local
    minimum on 2 of 30 series; three reach statsmodels' SSE or lower on all 30.
  * SCALING: the series is multiplied by a POWER OF TWO so its largest
    magnitude lies in [1, 2) and the states are scaled back by the inverse
    power at the end. Both are exact (barring the flush), so the scaling
    moves no bit of the recurrence; it keeps `J^T J` inside float32's range for
    series of any magnitude.

THE SPELLING (DEVIATION 698's rule, applied to every new seam): every
stored intermediate through `ftz`; at every `a * x + b * y` the FIRST
product fused (`identical_mul_add`) and the second stored; divisions are
IEEE `/` (correctly rounded on every column measured, IDENTITY_PATHS row
10). The per-element arithmetic lives in ONE set of helpers (`_est_step`,
`_sse_add`, `_est_dx`, `_est_dln`, `_est_dbn`, `_est_dsn`, `_est_row_acc`,
`hw_est_finish`) that every arm calls:

  * the host oracle (`holtwinters/host/hw_oracle.mojo`, the CPU column) and
    the serial device kernel run `hw_estimate_series`, one series at a time;
  * THE PARALLEL DEVICE KERNEL (`holtwinters_estimate_block_kernel`, taken
    when `f + 5 <= HW_EST_BLOCK`) runs one thread block per (series, start)
    and one thread per theta column. Every derivative update is per column,
    so thread `j` owns column `j`; row `i` of `J^T J` is accumulated by
    thread `i` in the same `t` order; the elimination updates row `i` in
    thread `i`; the scalar recurrence, the back substitution and every
    control decision are one thread's or computed redundantly from the same
    shared values. No value is ever summed across threads, so the parallel
    kernel performs exactly the host's operations on every element and
    `hw_estimate_check.mojo` compares their bits.
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.impl.internal.hw_utils import STMP_EPS, bound_device
from holtwinters.impl.tsa.holtwinters_params import (
    OPTIM_BFGS_ITER_LIMIT,
    OPTIM_MIN_ERROR_DIFF,
    OPTIM_MIN_PARAM_DIFF,
)
from checks.numerics import ftz, identical_mul_add

#: `initialization_method` codes. The integer crosses the binding in the
#: params list; HEURISTIC is cuML's fit and the default of every Mojo entry
#: (the gates), ESTIMATED is the Python default.
comptime HW_INIT_HEURISTIC = 0
comptime HW_INIT_ESTIMATED = 1

comptime HW_EST_MAX_ITER = 100
comptime HW_EST_LAMBDA0 = Float32(Float64(1e-3))
comptime HW_EST_LAMBDA_MIN = Float32(Float64(1e-7))
comptime HW_EST_LAMBDA_MAX = Float32(Float64(1e10))
comptime HW_EST_REL_TOL = Float32(Float64(1e-6))
comptime HW_EST_DIAG_FLOOR = Float32(Float64(1e-6))
comptime HW_EST_STARTS = 3
#: `-D MOJOLEARN_HOST_SABOTAGE=1`, which only the host build scripts pass
#: (`hw_oracle.mojo::HW_ORACLE_HOST_SABOTAGE`, the same define).
comptime HW_EST_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def hw_est_start(k: Int, j: Int) -> Float32:
    """Start `k`'s alpha (j=0), beta (1) or gamma (2)."""
    if k == 0:
        if j == 0:
            return Float32(Float64(0.4))
        return Float32(Float64(0.3))
    if k == 1:
        if j == 0:
            return Float32(Float64(0.2))
        return Float32(Float64(0.01))
    if j == 0:
        return Float32(Float64(0.8))
    return Float32(Float64(0.1))


def hw_est_dim(frequency: Int) -> Int:
    return frequency + 5


@always_inline
def _f(x: Float32) -> Float32:
    return ftz(x)


@always_inline
def _mad(a: Float32, b: Float32, c: Float32) -> Float32:
    return _f(identical_mul_add(a, b, c))


@always_inline
def _mix(a: Float32, x: Float32, one_minus_a: Float32, y: Float32) -> Float32:
    """`hw_eval.mojo::_mix`: the first product fused, the second stored."""
    return _f(identical_mul_add(a, x, _f(one_minus_a * y)))


def _pow2_scale(m: Float32) -> Float32:
    """`2^k` with `m * 2^k` in [1, 2); 1 for a zero (or flushed) maximum.
    Built from the exponent bits, so it is exact."""
    var be = Int((bitcast[DType.uint32](m) >> UInt32(23)) & UInt32(0xFF))
    if be == 0 or be == 0xFF:
        return Float32(1.0)
    var se = 254 - be
    if se < 1:
        se = 1
    return bitcast[DType.float32](UInt32(se) << UInt32(23))


def _inv_pow2(sc: Float32) -> Float32:
    var se = Int((bitcast[DType.uint32](sc) >> UInt32(23)) & UInt32(0xFF))
    return bitcast[DType.float32](UInt32(254 - se) << UInt32(23))


def _series_scale(
    tid: Int, ts: MutPointer[Float32, MutAnyOrigin], n: Int, batch_size: Int
) -> Float32:
    var m = Float32(0.0)
    for t in range(n):
        var v = abs(ts.unsafe_load(tid + t * batch_size))
        if v > m:
            m = v
    return _pow2_scale(m)


# ---------------------------------------------------------------------------
# The per-element arithmetic every arm shares
# ---------------------------------------------------------------------------


@fieldwise_init
struct _Step(Copyable, Movable, ImplicitlyCopyable):
    var e: Float32
    var sp: Float32
    var lt: Float32
    var ysp: Float32
    var ln: Float32
    var bn: Float32
    var ylv: Float32
    var sn: Float32
    var sp_eps: Float32
    var guarded: Bool


@always_inline
def _est_step(
    y: Float32, sp: Float32, l: Float32, tr: Float32,
    a: Float32, b: Float32, g: Float32,
    oma: Float32, omb: Float32, omg: Float32, additive: Bool,
) -> _Step:
    """One step of the recurrence (the header's equations)."""
    var lt = _f(l + tr)
    var xh: Float32
    if additive:
        xh = _f(lt + sp)
    else:
        xh = _f(lt * sp)
    var e = _f(y - xh)
    var sp_eps: Float32 = sp
    var guarded = False
    if not additive:
        if not (abs(sp) > STMP_EPS):
            sp_eps = STMP_EPS
            guarded = True
    var ysp: Float32
    if additive:
        ysp = _f(y - sp)
    else:
        ysp = _f(y / sp_eps)
    var ln = _mix(a, ysp, oma, lt)
    var bn = _mix(b, _f(ln - l), omb, tr)
    var ylv: Float32
    if additive:
        ylv = _f(y - ln)
    else:
        ylv = _f(y / ln)
    var sn = _mix(g, ylv, omg, sp)
    return _Step(e, sp, lt, ysp, ln, bn, ylv, sn, sp_eps, guarded)


@always_inline
def _sse_add(sse: Float32, e: Float32) -> Float32:
    comptime if HW_EST_HOST_SABOTAGE:
        # the CPU gate's negative control (hw_oracle.mojo's): the fused
        # SSE step split into two roundings. Host sabotage builds only.
        return _f(sse + _f(e * e))
    else:
        return _mad(e, e, sse)


@always_inline
def _est_ca(st: _Step, a: Float32, additive: Bool) -> Float32:
    """d level' / d s_p's coefficient."""
    if additive:
        return -a
    if st.guarded:
        return Float32(0.0)
    return -_f(a * _f(st.ysp / st.sp_eps))


@always_inline
def _est_cg(st: _Step, g: Float32, additive: Bool) -> Float32:
    """d season' / d level''s coefficient."""
    if additive:
        return -g
    return -_f(g * _f(st.ylv / st.ln))


@always_inline
def _est_dx(additive: Bool, st: _Step, dlj: Float32, dbj: Float32, dspj: Float32) -> Float32:
    var dlt = _f(dlj + dbj)
    if additive:
        return _f(dlt + dspj)
    return _mad(st.sp, dlt, _f(st.lt * dspj))


@always_inline
def _est_dln(
    j: Int, st: _Step, ca: Float32, oma: Float32, dlj: Float32, dbj: Float32, dspj: Float32
) -> Float32:
    var dlt = _f(dlj + dbj)
    var v = _mad(ca, dspj, _f(oma * dlt))
    if j == 0:
        v = _f(v + _f(st.ysp - st.lt))
    return v


@always_inline
def _est_dbn(
    j: Int, st: _Step, b: Float32, omb: Float32, dnj: Float32, dlj: Float32, dbj: Float32,
    l: Float32, tr: Float32,
) -> Float32:
    var v = _mad(b, _f(dnj - dlj), _f(omb * dbj))
    if j == 1:
        v = _f(v + _f(_f(st.ln - l) - tr))
    return v


@always_inline
def _est_dsn(j: Int, st: _Step, cg: Float32, omg: Float32, dlnew: Float32, dspj: Float32) -> Float32:
    var v = _mad(cg, dlnew, _f(omg * dspj))
    if j == 2:
        v = _f(v + _f(st.ylv - st.sp))
    return v


# ---------------------------------------------------------------------------
# The serial arm: the host oracle and the device fallback for f + 5 > 64
# ---------------------------------------------------------------------------


def _est_eval_plain(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    f: Int,
    sc: Float32,
    additive: Bool,
    th: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
    write: Bool,
    inv_sc: Float32,
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """The recurrence from t = 0 at `th` (scaled units); returns the SSE.
    `write`: the states at `t >= f`, scaled back, time-major at `tid + (t -
    f) * batch_size`."""
    var a = th.unsafe_load(0)
    var b = th.unsafe_load(1)
    var g = th.unsafe_load(2)
    var oma = _f(Float32(1.0) - a)
    var omb = _f(Float32(1.0) - b)
    var omg = _f(Float32(1.0) - g)
    var l = th.unsafe_load(3)
    var tr = th.unsafe_load(4)
    for p in range(f):
        sw.unsafe_store(p, th.unsafe_load(5 + p))
    var sse = Float32(0.0)
    # `p` is `t % f`, carried as the phase counter `ph` (see hw_eval.mojo: a signed
    # 64-bit floor modulus here made the gfx942 code object differ between
    # cold compiles).
    var ph = 0
    for t in range(n):
        var p = ph
        var y = _f(ts.unsafe_load(tid + t * batch_size) * sc)
        var st = _est_step(y, sw.unsafe_load(p), l, tr, a, b, g, oma, omb, omg, additive)
        sse = _sse_add(sse, st.e)
        l = st.ln
        tr = st.bn
        sw.unsafe_store(p, st.sn)
        if write and t >= f:
            var k = tid + (t - f) * batch_size
            level.unsafe_store(k, _f(st.ln * inv_sc))
            trend.unsafe_store(k, _f(st.bn * inv_sc))
            if additive:
                season.unsafe_store(k, _f(st.sn * inv_sc))
            else:
                season.unsafe_store(k, st.sn)
        ph += 1
        if ph >= f:
            ph = 0
    return sse


def _est_eval_jac(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    f: Int,
    sc: Float32,
    additive: Bool,
    th: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
    A: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    dl: MutPointer[Float32, MutAnyOrigin],
    db: MutPointer[Float32, MutAnyOrigin],
    dn: MutPointer[Float32, MutAnyOrigin],
    ds: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """`_est_eval_plain` plus `A = J^T J` (upper triangle, row-major `d x
    d`) and `grad = J^T e`. The SSE bits are `_est_eval_plain`'s."""
    var d = f + 5
    var a = th.unsafe_load(0)
    var b = th.unsafe_load(1)
    var g = th.unsafe_load(2)
    var oma = _f(Float32(1.0) - a)
    var omb = _f(Float32(1.0) - b)
    var omg = _f(Float32(1.0) - g)
    var l = th.unsafe_load(3)
    var tr = th.unsafe_load(4)
    for p in range(f):
        sw.unsafe_store(p, th.unsafe_load(5 + p))
    for i in range(d * d):
        A.unsafe_store(i, Float32(0.0))
    for j in range(d):
        grad.unsafe_store(j, Float32(0.0))
        dl.unsafe_store(j, Float32(1.0) if j == 3 else Float32(0.0))
        db.unsafe_store(j, Float32(1.0) if j == 4 else Float32(0.0))
    for p in range(f):
        for j in range(d):
            ds.unsafe_store(p * d + j, Float32(1.0) if j == 5 + p else Float32(0.0))
    var sse = Float32(0.0)
    # `p` is `t % f`, carried as the phase counter `ph` (see hw_eval.mojo: a signed
    # 64-bit floor modulus here made the gfx942 code object differ between
    # cold compiles).
    var ph = 0
    for t in range(n):
        var p = ph
        var pd = p * d
        var y = _f(ts.unsafe_load(tid + t * batch_size) * sc)
        var st = _est_step(y, sw.unsafe_load(p), l, tr, a, b, g, oma, omb, omg, additive)
        sse = _sse_add(sse, st.e)
        for j in range(d):
            dn.unsafe_store(j, _est_dx(additive, st, dl.unsafe_load(j), db.unsafe_load(j), ds.unsafe_load(pd + j)))
        for i in range(d):
            var xi = dn.unsafe_load(i)
            if xi != Float32(0.0):
                grad.unsafe_store(i, _mad(st.e, xi, grad.unsafe_load(i)))
                for jj in range(i, d):
                    A.unsafe_store(i * d + jj, _mad(xi, dn.unsafe_load(jj), A.unsafe_load(i * d + jj)))
        var ca = _est_ca(st, a, additive)
        var cg = _est_cg(st, g, additive)
        for j in range(d):
            var dlj = dl.unsafe_load(j)
            var dbj = db.unsafe_load(j)
            var dspj = ds.unsafe_load(pd + j)
            var dnj = _est_dln(j, st, ca, oma, dlj, dbj, dspj)
            db.unsafe_store(j, _est_dbn(j, st, b, omb, dnj, dlj, dbj, l, tr))
            dl.unsafe_store(j, dnj)
            ds.unsafe_store(pd + j, _est_dsn(j, st, cg, omg, dnj, dspj))
        l = st.ln
        tr = st.bn
        sw.unsafe_store(p, st.sn)
        ph += 1
        if ph >= f:
            ph = 0
    return sse


def _est_solve(
    d: Int,
    M: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
) -> Bool:
    """`M x = rhs` in place (`x` holds rhs on entry), unpivoted elimination
    in fixed order. False when a pivot is not `> 0` (or NaN)."""
    for k in range(d):
        var piv = M.unsafe_load(k * d + k)
        if not (piv > Float32(0.0)):
            return False
        for i in range(k + 1, d):
            var m = _f(M.unsafe_load(i * d + k) / piv)
            if m != Float32(0.0):
                var nm = -m
                for j in range(k + 1, d):
                    M.unsafe_store(i * d + j, _mad(nm, M.unsafe_load(k * d + j), M.unsafe_load(i * d + j)))
                x.unsafe_store(i, _mad(nm, x.unsafe_load(k), x.unsafe_load(i)))
    var kk = d - 1
    while kk >= 0:
        var acc = x.unsafe_load(kk)
        for j in range(kk + 1, d):
            acc = _mad(-M.unsafe_load(kk * d + j), x.unsafe_load(j), acc)
        x.unsafe_store(kk, _f(acc / M.unsafe_load(kk * d + kk)))
        kk -= 1
    return True


@always_inline
def _hold(v: Float32, gi: Float32) -> Bool:
    """A bounded parameter on its bound whose gradient points out."""
    return (not (v > Float32(0.0)) and gi < Float32(0.0)) or (not (v < Float32(1.0)) and gi > Float32(0.0))


@always_inline
def _seed(k: Int, j: Int, l0: Float32, b0: Float32, s0: Float32, sc: Float32, additive: Bool) -> Float32:
    """Start `k`'s theta_j; `s0` is the heuristic season of column `j >= 5`."""
    if j < 3:
        return hw_est_start(k, j)
    if j == 3:
        return l0
    if j == 4:
        return b0
    if additive:
        return _f(s0 * sc)
    return s0


def hw_est_finish(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    sc: Float32,
    inv_sc: Float32,
    cand: MutPointer[Float32, MutAnyOrigin],
    sse0: Float32, sse1: Float32, sse2: Float32,
    it0: Int, it1: Int, it2: Int,
    cr0: Int, cr1: Int, cr2: Int,
    sw: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    criterion: MutPointer[Int32, MutAnyOrigin],
    niter: MutPointer[Int32, MutAnyOrigin],
    theta_out: MutPointer[Float32, MutAnyOrigin],
):
    """The lowest final SSE of the three starts (`cand` holds their thetas,
    `d` apart), the earlier start on a tie and a finite SSE over a NaN; then
    the final pass writes the components and every per-series output."""
    var d = frequency + 5
    var bk = 0
    var best_sse = sse0
    var best_it = it0
    var best_cr = cr0
    for k in range(1, HW_EST_STARTS):
        var sk = sse1 if k == 1 else sse2
        if sk < best_sse or (not (best_sse == best_sse) and sk == sk):
            bk = k
            best_sse = sk
            best_it = it1 if k == 1 else it2
            best_cr = cr1 if k == 1 else cr2
    var best = cand.unsafe_offset(bk * d)
    var sse = _est_eval_plain(tid, ts, n, batch_size, frequency, sc, additive, best, sw,
                              True, inv_sc, level, trend, season)
    alpha.unsafe_store(tid, best.unsafe_load(0))
    beta.unsafe_store(tid, best.unsafe_load(1))
    gamma.unsafe_store(tid, best.unsafe_load(2))
    error.unsafe_store(tid, _f(_f(sse * inv_sc) * inv_sc))
    criterion.unsafe_store(tid, Int32(best_cr))
    niter.unsafe_store(tid, Int32(best_it))
    for j in range(d):
        var v = best.unsafe_load(j)
        if j == 3 or j == 4 or (j >= 5 and additive):
            v = _f(v * inv_sc)
        theta_out.unsafe_store(j * batch_size + tid, v)


def hw_est_scratch_len(frequency: Int) -> Int:
    """Float32 scratch per series for the serial arm: three candidate
    thetas, trial, grad, step, dl, db, dn (`9 d`), the season derivatives
    (`f d`), the working season (`f`), and `J^T J` and the system (`2 d^2`)."""
    var d = hw_est_dim(frequency)
    return 9 * d + frequency * d + frequency + 2 * d * d


def hw_estimate_series(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    start_level: Float32,
    start_trend: Float32,
    start_season: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    criterion: MutPointer[Int32, MutAnyOrigin],
    niter: MutPointer[Int32, MutAnyOrigin],
    theta_out: MutPointer[Float32, MutAnyOrigin],
):
    """The estimated fit of series `tid`, serially (see the header). `ts` is
    time-major (`[tid + t * batch_size]`), `start_season` this series'
    heuristic seasons at stride `batch_size`, `scratch` this series'
    `hw_est_scratch_len(frequency)` floats."""
    var f = frequency
    var d = f + 5
    var cand = scratch
    var tt = scratch.unsafe_offset(3 * d)
    var grad = scratch.unsafe_offset(4 * d)
    var step = scratch.unsafe_offset(5 * d)
    var dl = scratch.unsafe_offset(6 * d)
    var db = scratch.unsafe_offset(7 * d)
    var dn = scratch.unsafe_offset(8 * d)
    var ds = scratch.unsafe_offset(9 * d)
    var sw = ds.unsafe_offset(f * d)
    var A = sw.unsafe_offset(f)
    var M = A.unsafe_offset(d * d)

    var sc = _series_scale(tid, ts, n, batch_size)
    var inv_sc = _inv_pow2(sc)
    var l0 = _f(_mad(Float32(-f), start_trend, start_level) * sc)
    var b0 = _f(start_trend * sc)

    var sses = Array[Float32, HW_EST_STARTS](fill=Float32(0.0))
    var its = Array[Int, HW_EST_STARTS](fill=0)
    var crs = Array[Int, HW_EST_STARTS](fill=0)
    for k in range(HW_EST_STARTS):
        var th = cand.unsafe_offset(k * d)
        for j in range(d):
            var s0 = start_season.unsafe_load((j - 5) * batch_size) if j >= 5 else Float32(0.0)
            th.unsafe_store(j, _seed(k, j, l0, b0, s0, sc, additive))
        var sse = _est_eval_jac(tid, ts, n, batch_size, f, sc, additive, th, sw, A, grad, dl, db, dn, ds)
        var lam = HW_EST_LAMBDA0
        var crit = OPTIM_BFGS_ITER_LIMIT
        var it = 0
        while it < HW_EST_MAX_ITER:
            it += 1
            var maxdiag = Float32(0.0)
            for i in range(d):
                var v = A.unsafe_load(i * d + i)
                if v > maxdiag:
                    maxdiag = v
            var floor_ = _f(HW_EST_DIAG_FLOOR * maxdiag)
            for i in range(d):
                for j in range(i, d):
                    var v = A.unsafe_load(i * d + j)
                    M.unsafe_store(i * d + j, v)
                    M.unsafe_store(j * d + i, v)
                var aii = A.unsafe_load(i * d + i)
                var dg: Float32 = aii if aii > floor_ else floor_
                M.unsafe_store(i * d + i, _mad(lam, dg, aii))
                step.unsafe_store(i, grad.unsafe_load(i))
            for i in range(3):
                if _hold(th.unsafe_load(i), grad.unsafe_load(i)):
                    for j in range(d):
                        M.unsafe_store(i * d + j, Float32(0.0))
                        M.unsafe_store(j * d + i, Float32(0.0))
                    M.unsafe_store(i * d + i, Float32(1.0))
                    step.unsafe_store(i, Float32(0.0))
            if not _est_solve(d, M, step):
                lam = _f(lam * Float32(4.0))
                if lam > HW_EST_LAMBDA_MAX:
                    crit = OPTIM_MIN_PARAM_DIFF
                    break
                continue
            for j in range(d):
                tt.unsafe_store(j, _f(th.unsafe_load(j) + step.unsafe_load(j)))
            for j in range(3):
                tt.unsafe_store(j, bound_device(tt.unsafe_load(j)))
            var ns = _est_eval_plain(tid, ts, n, batch_size, f, sc, additive, tt, sw,
                                     False, inv_sc, level, trend, season)
            if ns < sse:
                var rel = _f(_f(sse - ns) / sse)
                for j in range(d):
                    th.unsafe_store(j, tt.unsafe_load(j))
                sse = _est_eval_jac(tid, ts, n, batch_size, f, sc, additive, th, sw, A, grad, dl, db, dn, ds)
                lam = _f(lam / Float32(3.0))
                if lam < HW_EST_LAMBDA_MIN:
                    lam = HW_EST_LAMBDA_MIN
                if rel < HW_EST_REL_TOL:
                    crit = OPTIM_MIN_ERROR_DIFF
                    break
            else:
                lam = _f(lam * Float32(4.0))
                if lam > HW_EST_LAMBDA_MAX:
                    crit = OPTIM_MIN_PARAM_DIFF
                    break
        sses[k] = sse
        its[k] = it
        crs[k] = crit
    hw_est_finish(tid, ts, n, batch_size, f, additive, sc, inv_sc, cand,
                  sses[0], sses[1], sses[2], its[0], its[1], its[2], crs[0], crs[1], crs[2],
                  sw, level, trend, season, alpha, beta, gamma, error, criterion, niter, theta_out)


def holtwinters_estimate_gpu_kernel(
    ts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    additive_in: Int32,
    start_level: MutPointer[Float32, MutAnyOrigin],
    start_trend: MutPointer[Float32, MutAnyOrigin],
    start_season: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    criterion: MutPointer[Int32, MutAnyOrigin],
    niter: MutPointer[Int32, MutAnyOrigin],
    theta_out: MutPointer[Float32, MutAnyOrigin],
):
    """The serial arm on the device (f + 5 > HW_EST_BLOCK): one thread per
    series, `hw_estimate_series`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch_size = Int(batch_size_in)
    var frequency = Int(frequency_in)
    if tid < batch_size:
        hw_estimate_series(
            tid, ts, Int(n_in), batch_size, frequency, additive_in != 0,
            start_level.unsafe_load(tid), start_trend.unsafe_load(tid),
            start_season.unsafe_offset(tid),
            scratch.unsafe_offset(tid * hw_est_scratch_len(frequency)),
            level, trend, season, alpha, beta, gamma, error, criterion, niter, theta_out,
        )


# ---------------------------------------------------------------------------
# The parallel arm: one block per (series, start), one thread per column
# ---------------------------------------------------------------------------

#: threads per block, and the largest `d = f + 5` the parallel arm takes
#: (the system matrix lives in threadgroup memory: 64 x 64 floats, 16 KiB)
comptime HW_EST_BLOCK = 64
#: threadgroup layout (floats)
comptime _SH_M = 0
comptime _SH_TH = HW_EST_BLOCK * HW_EST_BLOCK
comptime _SH_TT = _SH_TH + HW_EST_BLOCK
comptime _SH_RHS = _SH_TT + HW_EST_BLOCK
comptime _SH_G = _SH_RHS + HW_EST_BLOCK
comptime _SH_DX = _SH_G + HW_EST_BLOCK
comptime _SH_DIAG = _SH_DX + HW_EST_BLOCK
comptime _SH_S = _SH_DIAG + HW_EST_BLOCK      # 2 x 16 step scalars
comptime _SH_CTL = _SH_S + 32                  # broadcast SSE
comptime _SH_LEN = _SH_CTL + 4


def hw_est_parallel(frequency: Int) -> Bool:
    return frequency + 5 <= HW_EST_BLOCK


def hw_est_block_scratch_len(frequency: Int) -> Int:
    """Per (series, start) block: `J^T J` rows (`d^2`, row `i` thread `i`'s
    own), the season derivatives (`f d`, column `j` thread `j`'s own), the
    working season and a theta copy (thread 0's own)."""
    var d = hw_est_dim(frequency)
    return d * d + frequency * d + frequency + d


@always_inline
def _blk_eval_jac[
    origin: MutOrigin, //
](
    sh: MutPointer[Float32, origin, address_space = AddressSpace.SHARED],
    j: Int, s: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int, batch_size: Int, f: Int, sc: Float32, additive: Bool,
    Arow: MutPointer[Float32, MutAnyOrigin],
    dsj: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """`_est_eval_jac` at the shared theta, in parallel. Thread 0 runs the
    scalar recurrence and publishes each step's scalars (double-buffered by
    `t` parity); thread `j < d` carries column `j` of the derivatives and
    row `j` of `J^T J` and entry `j` of `J^T e` (into shared `_SH_G`).
    Returns the SSE on every thread."""
    var d = f + 5
    var a = sh.unsafe_load(_SH_TH + 0)
    var b = sh.unsafe_load(_SH_TH + 1)
    var g = sh.unsafe_load(_SH_TH + 2)
    var oma = _f(Float32(1.0) - a)
    var omb = _f(Float32(1.0) - b)
    var omg = _f(Float32(1.0) - g)
    var l = sh.unsafe_load(_SH_TH + 3)
    var tr = sh.unsafe_load(_SH_TH + 4)
    var dl = Float32(0.0)
    var db = Float32(0.0)
    if j < d:
        for jj in range(j, d):
            Arow.unsafe_store(jj, Float32(0.0))
        sh.unsafe_store(_SH_G + j, Float32(0.0))
        dl = Float32(1.0) if j == 3 else Float32(0.0)
        db = Float32(1.0) if j == 4 else Float32(0.0)
        for p in range(f):
            dsj.unsafe_store(p, Float32(1.0) if j == 5 + p else Float32(0.0))
    if j == 0:
        for p in range(f):
            sw.unsafe_store(p, sh.unsafe_load(_SH_TH + 5 + p))
    var sse = Float32(0.0)
    # `p` is `t % f`, carried as the phase counter `ph` (see hw_eval.mojo: a signed
    # 64-bit floor modulus here made the gfx942 code object differ between
    # cold compiles).
    var ph = 0
    for t in range(n):
        var p = ph
        var S = _SH_S + (t & 1) * 16
        if j == 0:
            var y = _f(ts.unsafe_load(s + t * batch_size) * sc)
            var st = _est_step(y, sw.unsafe_load(p), l, tr, a, b, g, oma, omb, omg, additive)
            sse = _sse_add(sse, st.e)
            sh.unsafe_store(S + 0, st.e)
            sh.unsafe_store(S + 1, st.sp)
            sh.unsafe_store(S + 2, st.lt)
            sh.unsafe_store(S + 3, st.ysp)
            sh.unsafe_store(S + 4, st.ln)
            sh.unsafe_store(S + 5, st.bn)
            sh.unsafe_store(S + 6, st.ylv)
            sh.unsafe_store(S + 7, st.sn)
            sh.unsafe_store(S + 8, st.sp_eps)
            sh.unsafe_store(S + 9, Float32(1.0) if st.guarded else Float32(0.0))
            sh.unsafe_store(S + 10, l)
            sh.unsafe_store(S + 11, tr)
            l = st.ln
            tr = st.bn
            sw.unsafe_store(p, st.sn)
        barrier()
        var st = _Step(
            sh.unsafe_load(S + 0), sh.unsafe_load(S + 1), sh.unsafe_load(S + 2), sh.unsafe_load(S + 3),
            sh.unsafe_load(S + 4), sh.unsafe_load(S + 5), sh.unsafe_load(S + 6), sh.unsafe_load(S + 7),
            sh.unsafe_load(S + 8), sh.unsafe_load(S + 9) != Float32(0.0),
        )
        var dspj = Float32(0.0)
        if j < d:
            dspj = dsj.unsafe_load(p)
            sh.unsafe_store(_SH_DX + j, _est_dx(additive, st, dl, db, dspj))
        barrier()
        if j < d:
            var xi = sh.unsafe_load(_SH_DX + j)
            if xi != Float32(0.0):
                sh.unsafe_store(_SH_G + j, _mad(st.e, xi, sh.unsafe_load(_SH_G + j)))
                for jj in range(j, d):
                    Arow.unsafe_store(jj, _mad(xi, sh.unsafe_load(_SH_DX + jj), Arow.unsafe_load(jj)))
            var lold = sh.unsafe_load(S + 10)
            var trold = sh.unsafe_load(S + 11)
            var ca = _est_ca(st, a, additive)
            var cg = _est_cg(st, g, additive)
            var dnj = _est_dln(j, st, ca, oma, dl, db, dspj)
            db = _est_dbn(j, st, b, omb, dnj, dl, db, lold, trold)
            dl = dnj
            dsj.unsafe_store(p, _est_dsn(j, st, cg, omg, dnj, dspj))
        ph += 1
        if ph >= f:
            ph = 0
    if j == 0:
        sh.unsafe_store(_SH_CTL, sse)
    barrier()
    var out = sh.unsafe_load(_SH_CTL)
    barrier()
    return out


@always_inline
def _blk_eval_plain[
    origin: MutOrigin, //
](
    sh: MutPointer[Float32, origin, address_space = AddressSpace.SHARED],
    at: Int, j: Int, s: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int, batch_size: Int, f: Int, sc: Float32, additive: Bool,
    sw: MutPointer[Float32, MutAnyOrigin],
    thp: MutPointer[Float32, MutAnyOrigin],
    inv_sc: Float32,
    level: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """`_est_eval_plain` of the shared theta at offset `at`, by thread 0
    alone (the scalar recurrence has no column to split); every thread
    returns its SSE."""
    var d = f + 5
    if j == 0:
        for jj in range(d):
            thp.unsafe_store(jj, sh.unsafe_load(at + jj))
        var v = _est_eval_plain(s, ts, n, batch_size, f, sc, additive, thp, sw,
                                False, inv_sc, level, level, level)
        sh.unsafe_store(_SH_CTL, v)
    barrier()
    var out = sh.unsafe_load(_SH_CTL)
    barrier()
    return out


def holtwinters_estimate_block_kernel(
    ts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    additive_in: Int32,
    start_level: MutPointer[Float32, MutAnyOrigin],
    start_trend: MutPointer[Float32, MutAnyOrigin],
    start_season: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
    cand_theta: MutPointer[Float32, MutAnyOrigin],
    cand_sse: MutPointer[Float32, MutAnyOrigin],
    cand_ints: MutPointer[Int32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
):
    """Block `s * 3 + k` runs start `k` of series `s`: `hw_estimate_series`'s
    start loop body with every per-element operation in the thread that owns
    that element. Writes the start's final theta (`cand_theta[(s*3+k)*d +
    j]`), SSE and (niter, criterion) (`cand_ints[2*(s*3+k)..]`)."""
    var blk = Int(block_idx.x)
    var j = Int(thread_idx.x)
    var n = Int(n_in)
    var batch_size = Int(batch_size_in)
    var f = Int(frequency_in)
    var additive = additive_in != 0
    var s = blk // HW_EST_STARTS
    var k = blk - s * HW_EST_STARTS
    var d = f + 5
    var sh = stack_allocation[_SH_LEN, Float32, address_space = AddressSpace.SHARED]()
    var base = scratch.unsafe_offset(blk * hw_est_block_scratch_len(f))
    var jr = j if j < d else 0
    var Arow = base.unsafe_offset(jr * d)
    var dsj = base.unsafe_offset(d * d + jr * f)
    var sw = base.unsafe_offset(d * d + f * d)
    var thp = sw.unsafe_offset(f)

    var sc = _series_scale(s, ts, n, batch_size)
    var inv_sc = _inv_pow2(sc)
    var l0 = _f(_mad(Float32(-f), start_trend.unsafe_load(s), start_level.unsafe_load(s)) * sc)
    var b0 = _f(start_trend.unsafe_load(s) * sc)
    if j < d:
        var s0 = start_season.unsafe_load((j - 5) * batch_size + s) if j >= 5 else Float32(0.0)
        sh.unsafe_store(_SH_TH + j, _seed(k, j, l0, b0, s0, sc, additive))
    barrier()
    var sse = _blk_eval_jac(sh, j, s, ts, n, batch_size, f, sc, additive, Arow, dsj, sw)
    var lam = HW_EST_LAMBDA0
    var crit = OPTIM_BFGS_ITER_LIMIT
    var it = 0
    while it < HW_EST_MAX_ITER:
        it += 1
        if j < d:
            sh.unsafe_store(_SH_DIAG + j, Arow.unsafe_load(j))
        barrier()
        var maxdiag = Float32(0.0)
        for i in range(d):
            var v = sh.unsafe_load(_SH_DIAG + i)
            if v > maxdiag:
                maxdiag = v
        var floor_ = _f(HW_EST_DIAG_FLOOR * maxdiag)
        if j < d:
            for jj in range(j, d):
                var v = Arow.unsafe_load(jj)
                sh.unsafe_store(_SH_M + j * d + jj, v)
                sh.unsafe_store(_SH_M + jj * d + j, v)
            var ajj = Arow.unsafe_load(j)
            var dg: Float32 = ajj if ajj > floor_ else floor_
            sh.unsafe_store(_SH_M + j * d + j, _mad(lam, dg, ajj))
            sh.unsafe_store(_SH_RHS + j, sh.unsafe_load(_SH_G + j))
        barrier()
        var h0 = _hold(sh.unsafe_load(_SH_TH + 0), sh.unsafe_load(_SH_G + 0))
        var h1 = _hold(sh.unsafe_load(_SH_TH + 1), sh.unsafe_load(_SH_G + 1))
        var h2 = _hold(sh.unsafe_load(_SH_TH + 2), sh.unsafe_load(_SH_G + 2))
        if j < d:
            for i in range(3):
                var hi = h0 if i == 0 else (h1 if i == 1 else h2)
                if hi:
                    sh.unsafe_store(_SH_M + i * d + j, Float32(0.0))
                    sh.unsafe_store(_SH_M + j * d + i, Float32(0.0))
        barrier()
        if j < 3:
            var hj = h0 if j == 0 else (h1 if j == 1 else h2)
            if hj:
                sh.unsafe_store(_SH_M + j * d + j, Float32(1.0))
                sh.unsafe_store(_SH_RHS + j, Float32(0.0))
        # the elimination: row i in thread i, pivot row k untouched at step k
        var ok = True
        for kk in range(d):
            barrier()
            var piv = sh.unsafe_load(_SH_M + kk * d + kk)
            if not (piv > Float32(0.0)):
                ok = False
                break
            if j > kk and j < d:
                var m = _f(sh.unsafe_load(_SH_M + j * d + kk) / piv)
                if m != Float32(0.0):
                    var nm = -m
                    for jj in range(kk + 1, d):
                        sh.unsafe_store(_SH_M + j * d + jj,
                                        _mad(nm, sh.unsafe_load(_SH_M + kk * d + jj), sh.unsafe_load(_SH_M + j * d + jj)))
                    sh.unsafe_store(_SH_RHS + j, _mad(nm, sh.unsafe_load(_SH_RHS + kk), sh.unsafe_load(_SH_RHS + j)))
        barrier()
        if not ok:
            lam = _f(lam * Float32(4.0))
            if lam > HW_EST_LAMBDA_MAX:
                crit = OPTIM_MIN_PARAM_DIFF
                break
            continue
        if j == 0:
            var r = d - 1
            while r >= 0:
                var acc = sh.unsafe_load(_SH_RHS + r)
                for jj in range(r + 1, d):
                    acc = _mad(-sh.unsafe_load(_SH_M + r * d + jj), sh.unsafe_load(_SH_RHS + jj), acc)
                sh.unsafe_store(_SH_RHS + r, _f(acc / sh.unsafe_load(_SH_M + r * d + r)))
                r -= 1
        barrier()
        if j < d:
            var v = _f(sh.unsafe_load(_SH_TH + j) + sh.unsafe_load(_SH_RHS + j))
            if j < 3:
                v = bound_device(v)
            sh.unsafe_store(_SH_TT + j, v)
        barrier()
        var ns = _blk_eval_plain(sh, _SH_TT, j, s, ts, n, batch_size, f, sc, additive, sw, thp, inv_sc, level)
        if ns < sse:
            var rel = _f(_f(sse - ns) / sse)
            if j < d:
                sh.unsafe_store(_SH_TH + j, sh.unsafe_load(_SH_TT + j))
            barrier()
            sse = _blk_eval_jac(sh, j, s, ts, n, batch_size, f, sc, additive, Arow, dsj, sw)
            lam = _f(lam / Float32(3.0))
            if lam < HW_EST_LAMBDA_MIN:
                lam = HW_EST_LAMBDA_MIN
            if rel < HW_EST_REL_TOL:
                crit = OPTIM_MIN_ERROR_DIFF
                break
        else:
            lam = _f(lam * Float32(4.0))
            if lam > HW_EST_LAMBDA_MAX:
                crit = OPTIM_MIN_PARAM_DIFF
                break
    if j < d:
        cand_theta.unsafe_store(blk * d + j, sh.unsafe_load(_SH_TH + j))
    if j == 0:
        cand_sse.unsafe_store(blk, sse)
        cand_ints.unsafe_store(2 * blk, Int32(it))
        cand_ints.unsafe_store(2 * blk + 1, Int32(crit))


def holtwinters_estimate_finish_kernel(
    ts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    additive_in: Int32,
    cand_theta: MutPointer[Float32, MutAnyOrigin],
    cand_sse: MutPointer[Float32, MutAnyOrigin],
    cand_ints: MutPointer[Int32, MutAnyOrigin],
    sw_all: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    criterion: MutPointer[Int32, MutAnyOrigin],
    niter: MutPointer[Int32, MutAnyOrigin],
    theta_out: MutPointer[Float32, MutAnyOrigin],
):
    """One thread per series: `hw_est_finish` over the three blocks' starts."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch_size = Int(batch_size_in)
    var f = Int(frequency_in)
    var n = Int(n_in)
    if s < batch_size:
        var d = f + 5
        var sc = _series_scale(s, ts, n, batch_size)
        var b3 = s * HW_EST_STARTS
        hw_est_finish(
            s, ts, n, batch_size, f, additive_in != 0, sc, _inv_pow2(sc),
            cand_theta.unsafe_offset(b3 * d),
            cand_sse.unsafe_load(b3), cand_sse.unsafe_load(b3 + 1), cand_sse.unsafe_load(b3 + 2),
            Int(cand_ints.unsafe_load(2 * b3)), Int(cand_ints.unsafe_load(2 * b3 + 2)),
            Int(cand_ints.unsafe_load(2 * b3 + 4)),
            Int(cand_ints.unsafe_load(2 * b3 + 1)), Int(cand_ints.unsafe_load(2 * b3 + 3)),
            Int(cand_ints.unsafe_load(2 * b3 + 5)),
            sw_all.unsafe_offset(s * f),
            level, trend, season, alpha, beta, gamma, error, criterion, niter, theta_out,
        )


def holtwinters_estimate_gpu(
    ctx: DeviceContext,
    mut ts: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut beta: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    mut criterion: DeviceBuffer[DType.int32],
    mut niter: DeviceBuffer[DType.int32],
    mut theta_out: DeviceBuffer[DType.float32],
    tpb: Int,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = Float32(0.0),
    force_serial: Bool = False,
) raises:
    """Launch the estimated fit. `ts` is time-major; `theta_out` holds
    `(frequency + 5) * batch_size`. `tpb` is the finish/serial kernels'
    block width (scheduling only). `force_serial` takes the serial arm at
    any `frequency` (the gate uses it to hold the two arms to each other)."""
    if tpb <= 0:
        raise Error("holtwinters_estimate_gpu: tpb must be positive")
    var total_blocks = (batch_size + tpb - 1) // tpb
    if hw_est_parallel(frequency) and not force_serial:
        var d = frequency + 5
        var blocks = batch_size * HW_EST_STARTS
        var scratch = ctx.enqueue_create_buffer[DType.float32](blocks * hw_est_block_scratch_len(frequency) + scratch_pad)
        var cand_theta = ctx.enqueue_create_buffer[DType.float32](blocks * d + scratch_pad)
        var cand_sse = ctx.enqueue_create_buffer[DType.float32](blocks + scratch_pad)
        var cand_ints = ctx.enqueue_create_buffer[DType.int32](2 * blocks)
        var sw_all = ctx.enqueue_create_buffer[DType.float32](batch_size * frequency + scratch_pad)
        scratch.enqueue_fill(scratch_poison)
        cand_theta.enqueue_fill(scratch_poison)
        cand_sse.enqueue_fill(scratch_poison)
        sw_all.enqueue_fill(scratch_poison)
        ctx.synchronize()
        ctx.enqueue_function[holtwinters_estimate_block_kernel](
            ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
            Int32(1 if additive else 0),
            start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
            scratch.unsafe_ptr(), cand_theta.unsafe_ptr(), cand_sse.unsafe_ptr(), cand_ints.unsafe_ptr(),
            level.unsafe_ptr(),
            grid_dim=(blocks, 1, 1),
            block_dim=(HW_EST_BLOCK, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_function[holtwinters_estimate_finish_kernel](
            ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
            Int32(1 if additive else 0),
            cand_theta.unsafe_ptr(), cand_sse.unsafe_ptr(), cand_ints.unsafe_ptr(), sw_all.unsafe_ptr(),
            level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(),
            alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(), error.unsafe_ptr(),
            criterion.unsafe_ptr(), niter.unsafe_ptr(), theta_out.unsafe_ptr(),
            grid_dim=(total_blocks, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        _ = scratch^
        _ = cand_theta^
        _ = cand_sse^
        _ = cand_ints^
        _ = sw_all^
        return
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        batch_size * hw_est_scratch_len(frequency) + scratch_pad
    )
    scratch.enqueue_fill(scratch_poison)
    ctx.synchronize()
    ctx.enqueue_function[holtwinters_estimate_gpu_kernel](
        ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
        Int32(1 if additive else 0),
        start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
        scratch.unsafe_ptr(),
        level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(),
        alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(), error.unsafe_ptr(),
        criterion.unsafe_ptr(), niter.unsafe_ptr(), theta_out.unsafe_ptr(),
        grid_dim=(total_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    _ = scratch^
