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
10). ONE FUNCTION, `hw_estimate_series`, is the arithmetic: the GPU kernel
calls it once per thread and the host oracle (`holtwinters/host/
hw_oracle.mojo`, the CPU column) calls it once per series, so the device
and the CPU run the same spelling by construction and the gate compares
their bits.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
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


def hw_est_scratch_len(frequency: Int) -> Int:
    """Float32 scratch per series: th, trial, best, grad, step, dl, db, dn
    (`8 d`), the season derivatives (`f d`), the working season (`f`), and
    `J^T J` and the system matrix (`2 d^2`)."""
    var d = hw_est_dim(frequency)
    return 8 * d + frequency * d + frequency + 2 * d * d


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


def _est_eval(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    f: Int,
    sc: Float32,
    additive: Bool,
    th: MutPointer[Float32, MutAnyOrigin],
    sw: MutPointer[Float32, MutAnyOrigin],
    jac: Bool,
    A: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    dl: MutPointer[Float32, MutAnyOrigin],
    db: MutPointer[Float32, MutAnyOrigin],
    dn: MutPointer[Float32, MutAnyOrigin],
    ds: MutPointer[Float32, MutAnyOrigin],
    write: Bool,
    inv_sc: Float32,
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """The recurrence from t = 0 at `th` (scaled units); returns the SSE.
    `jac`: also `A = J^T J` (upper triangle, row-major `d x d`) and `grad =
    J^T e`. `write`: the states at `t >= f`, scaled back, time-major at
    `tid + (t - f) * batch_size`. The SSE bits do not depend on either flag."""
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
    if jac:
        for i in range(d * d):
            A.unsafe_store(i, Float32(0.0))
        for j in range(d):
            grad.unsafe_store(j, Float32(0.0))
            dl.unsafe_store(j, Float32(0.0))
            db.unsafe_store(j, Float32(0.0))
        for p in range(f):
            for j in range(d):
                ds.unsafe_store(p * d + j, Float32(0.0))
            ds.unsafe_store(p * d + 5 + p, Float32(1.0))
        dl.unsafe_store(3, Float32(1.0))
        db.unsafe_store(4, Float32(1.0))
    var sse = Float32(0.0)
    for t in range(n):
        var p = t % f
        var y = _f(ts.unsafe_load(tid + t * batch_size) * sc)
        var sp = sw.unsafe_load(p)
        var lt = _f(l + tr)
        var xh: Float32
        if additive:
            xh = _f(lt + sp)
        else:
            xh = _f(lt * sp)
        var e = _f(y - xh)
        comptime if HW_EST_HOST_SABOTAGE:
            # the CPU gate's negative control (hw_oracle.mojo's): the fused
            # SSE step split into two roundings. Host sabotage builds only.
            sse = _f(sse + _f(e * e))
        else:
            sse = _mad(e, e, sse)
        var sp_eps: Float32 = sp
        var guarded = False
        if not additive:
            if not (abs(sp) > STMP_EPS):
                sp_eps = STMP_EPS
                guarded = True
        var ysp: Float32
        var ln: Float32
        if additive:
            ysp = _f(y - sp)
        else:
            ysp = _f(y / sp_eps)
        ln = _mix(a, ysp, oma, lt)
        var bn = _mix(b, _f(ln - l), omb, tr)
        var ylv: Float32
        if additive:
            ylv = _f(y - ln)
        else:
            ylv = _f(y / ln)
        var sn = _mix(g, ylv, omg, sp)
        if jac:
            var pd = p * d
            # dx = d xhat / d theta, into dn; then A += dx dx^T, grad += e dx
            for j in range(d):
                var dlt = _f(dl.unsafe_load(j) + db.unsafe_load(j))
                var dx: Float32
                if additive:
                    dx = _f(dlt + ds.unsafe_load(pd + j))
                else:
                    dx = _mad(sp, dlt, _f(lt * ds.unsafe_load(pd + j)))
                dn.unsafe_store(j, dx)
            for i in range(d):
                var xi = dn.unsafe_load(i)
                if xi != Float32(0.0):
                    grad.unsafe_store(i, _mad(e, xi, grad.unsafe_load(i)))
                    for j in range(i, d):
                        A.unsafe_store(i * d + j, _mad(xi, dn.unsafe_load(j), A.unsafe_load(i * d + j)))
            # the level's derivative, into dn (dx is no longer needed)
            var ca: Float32
            if additive:
                ca = -a
            elif guarded:
                ca = Float32(0.0)
            else:
                ca = -_f(a * _f(ysp / sp_eps))
            for j in range(d):
                var dlt = _f(dl.unsafe_load(j) + db.unsafe_load(j))
                dn.unsafe_store(j, _mad(ca, ds.unsafe_load(pd + j), _f(oma * dlt)))
            dn.unsafe_store(0, _f(dn.unsafe_load(0) + _f(ysp - lt)))
            # the trend's (reads the OLD dl), then dl <- dn
            for j in range(d):
                var dlj = dl.unsafe_load(j)
                var dnj = dn.unsafe_load(j)
                db.unsafe_store(j, _mad(b, _f(dnj - dlj), _f(omb * db.unsafe_load(j))))
                dl.unsafe_store(j, dnj)
            db.unsafe_store(1, _f(db.unsafe_load(1) + _f(_f(ln - l) - tr)))
            # the season slot's
            var cg: Float32
            if additive:
                cg = -g
            else:
                cg = -_f(g * _f(ylv / ln))
            for j in range(d):
                ds.unsafe_store(pd + j, _mad(cg, dl.unsafe_load(j), _f(omg * ds.unsafe_load(pd + j))))
            ds.unsafe_store(pd + 2, _f(ds.unsafe_load(pd + 2) + _f(ylv - sp)))
        l = ln
        tr = bn
        sw.unsafe_store(p, sn)
        if write and t >= f:
            var k = tid + (t - f) * batch_size
            level.unsafe_store(k, _f(ln * inv_sc))
            trend.unsafe_store(k, _f(bn * inv_sc))
            if additive:
                season.unsafe_store(k, _f(sn * inv_sc))
            else:
                season.unsafe_store(k, sn)
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
    """The estimated fit of series `tid` (see the header). `ts` is
    time-major (`[tid + t * batch_size]`), `start_season` this series'
    heuristic seasons at stride `batch_size`, `scratch` this series'
    `hw_est_scratch_len(frequency)` floats. Writes the components at `t >=
    frequency`, `alpha/beta/gamma/error/criterion/niter[tid]`, and the
    chosen theta with its states in the series' units at `theta_out[j *
    batch_size + tid]`."""
    var f = frequency
    var d = f + 5
    var th = scratch
    var tt = scratch.unsafe_offset(d)
    var best = scratch.unsafe_offset(2 * d)
    var grad = scratch.unsafe_offset(3 * d)
    var step = scratch.unsafe_offset(4 * d)
    var dl = scratch.unsafe_offset(5 * d)
    var db = scratch.unsafe_offset(6 * d)
    var dn = scratch.unsafe_offset(7 * d)
    var ds = scratch.unsafe_offset(8 * d)
    var sw = ds.unsafe_offset(f * d)
    var A = sw.unsafe_offset(f)
    var M = A.unsafe_offset(d * d)

    # the power-of-two scale
    var m = Float32(0.0)
    for t in range(n):
        var v = abs(ts.unsafe_load(tid + t * batch_size))
        if v > m:
            m = v
    var sc = _pow2_scale(m)
    var inv_sc = _inv_pow2(sc)

    # the heuristic seed, in scaled units
    var l0 = _f(_mad(Float32(-f), start_trend, start_level) * sc)
    var b0 = _f(start_trend * sc)

    var best_sse = Float32(0.0)
    var best_niter = 0
    var best_crit = OPTIM_BFGS_ITER_LIMIT
    for k in range(HW_EST_STARTS):
        th.unsafe_store(0, hw_est_start(k, 0))
        th.unsafe_store(1, hw_est_start(k, 1))
        th.unsafe_store(2, hw_est_start(k, 2))
        th.unsafe_store(3, l0)
        th.unsafe_store(4, b0)
        for p in range(f):
            var s0 = start_season.unsafe_load(p * batch_size)
            if additive:
                th.unsafe_store(5 + p, _f(s0 * sc))
            else:
                th.unsafe_store(5 + p, s0)
        var sse = _est_eval(tid, ts, n, batch_size, f, sc, additive, th, sw, True,
                            A, grad, dl, db, dn, ds, False, inv_sc, level, trend, season)
        var lam = HW_EST_LAMBDA0
        var crit = OPTIM_BFGS_ITER_LIMIT
        var it = 0
        while it < HW_EST_MAX_ITER:
            it += 1
            # Marquardt's diagonal, floored relative to the largest
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
            # the active bounds: held fixed this step
            for i in range(3):
                var v = th.unsafe_load(i)
                var gi = grad.unsafe_load(i)
                var hold = (not (v > Float32(0.0)) and gi < Float32(0.0)) or (
                    not (v < Float32(1.0)) and gi > Float32(0.0))
                if hold:
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
            var ns = _est_eval(tid, ts, n, batch_size, f, sc, additive, tt, sw, False,
                               A, grad, dl, db, dn, ds, False, inv_sc, level, trend, season)
            if ns < sse:
                var rel = _f(_f(sse - ns) / sse)
                for j in range(d):
                    th.unsafe_store(j, tt.unsafe_load(j))
                sse = _est_eval(tid, ts, n, batch_size, f, sc, additive, th, sw, True,
                                A, grad, dl, db, dn, ds, False, inv_sc, level, trend, season)
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
        var take = k == 0 or sse < best_sse or (not (best_sse == best_sse) and sse == sse)
        if take:
            best_sse = sse
            best_niter = it
            best_crit = crit
            for j in range(d):
                best.unsafe_store(j, th.unsafe_load(j))

    # the final pass at the chosen theta writes the components
    var sse = _est_eval(tid, ts, n, batch_size, f, sc, additive, best, sw, False,
                        A, grad, dl, db, dn, ds, True, inv_sc, level, trend, season)
    alpha.unsafe_store(tid, best.unsafe_load(0))
    beta.unsafe_store(tid, best.unsafe_load(1))
    gamma.unsafe_store(tid, best.unsafe_load(2))
    error.unsafe_store(tid, _f(_f(sse * inv_sc) * inv_sc))
    criterion.unsafe_store(tid, Int32(best_crit))
    niter.unsafe_store(tid, Int32(best_niter))
    for j in range(d):
        var v = best.unsafe_load(j)
        if j == 3 or j == 4 or (j >= 5 and additive):
            v = _f(v * inv_sc)
        theta_out.unsafe_store(j * batch_size + tid, v)


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
    """One thread per series; each thread owns its slice of `scratch`."""
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
) raises:
    """Launch the estimated fit. `ts` is time-major; `theta_out` holds
    `(frequency + 5) * batch_size`. `tpb` is scheduling only."""
    if tpb <= 0:
        raise Error("holtwinters_estimate_gpu: tpb must be positive")
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        batch_size * hw_est_scratch_len(frequency) + scratch_pad
    )
    scratch.enqueue_fill(scratch_poison)
    ctx.synchronize()
    var total_blocks = (batch_size + tpb - 1) // tpb
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
