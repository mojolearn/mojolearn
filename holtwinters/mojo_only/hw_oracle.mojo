"""The host oracle for Holt-Winters: serial, ascending, written to the same
pinned spelling as the device kernels -- and a Float64 reference through
the same code, for tolerance sanity.

NOT A PORT (cuML has no host arm). `oracle_fit[DType.float32]` is the
float32 arm: every multiply-add through `identical_mul_add`, every stored
intermediate through `ftz`, the step-size `sqrt` through `identical_sqrt`,
the clamp through DEVIATION 663's compare chain -- so under IDENTICAL the
device (`holtwinters/ported/...`) must match it BIT FOR BIT at every stage,
and `hw_check` asserts that. `oracle_fit[DType.float64]` is the same
algorithm in double (plain host arithmetic) and is what the float32 start
values, SSE and forecasts are compared against WITHIN TOLERANCE.

Written independently of the kernels (host style: time-major `List`s,
the C++ structure followed from `runner.cuh` down), so that a logic slip
in one is not mirrored in the other. The arithmetic SPELLING is shared by
design -- that is the pin -- and `hw_eval.mojo`'s header names it.
"""

from std.math import fma, sqrt

from holtwinters.ported.holtwinters.internal.hw_decompose import host_filter, host_r1qt
from holtwinters.ported.holtwinters.internal.hw_optim import (
    HW_DEC_ALPHA_HI,
    HW_DEC_ALPHA_LO,
    HW_DEC_BETA_HI,
    HW_DEC_BETA_LO,
    HW_DEC_BOTH_CRIT,
    HW_DEC_GAMMA_HI,
    HW_DEC_GAMMA_LO,
    HW_DEC_HALVING_SHIFT,
    HW_DEC_HESSIAN_RESET,
    HW_DEC_H_NAN,
    HW_DEC_ITER_LIMIT,
    HW_DEC_LS_LIMIT,
    HW_DEC_RHO_ZERO,
    HW_DEC_ZERO_DIR,
)
from holtwinters.ported.holtwinters.internal.hw_utils import STMP_EPS
from holtwinters.ported.holtwinters.runner import (
    HW_ALPHA0,
    HW_BETA0,
    HW_GAMMA0,
    default_optim_params,
)
from holtwinters.ported.tsa.holtwinters_params import (
    OPTIM_BFGS_ITER_LIMIT,
    OPTIM_MIN_ERROR_DIFF,
    OPTIM_MIN_GRAD_NORM,
    OPTIM_MIN_PARAM_DIFF,
    SEASONAL_ADDITIVE,
)
from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt


# ---------------------------------------------------------------------------
# The dtype-generic seams: float32 through the identical helpers, float64
# plain.
# ---------------------------------------------------------------------------


@always_inline
def _f[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](ftz(rebind[Float32](x)))
    else:
        return x


@always_inline
def _mad[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](
            identical_mul_add(rebind[Float32](a), rebind[Float32](b), rebind[Float32](c))
        )
    else:
        return a * b + c


@always_inline
def _sqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](identical_sqrt(rebind[Float32](x)))
    else:
        return sqrt(x)


@always_inline
def _bound[dt: DType](v: Scalar[dt]) -> Scalar[dt]:
    """DEVIATION 663's clamp: `v > 0 ? v : +0`, then `> 1 ? 1`."""
    var r: Scalar[dt]
    if v > Scalar[dt](0):
        r = v
    else:
        r = Scalar[dt](0)
    if r > Scalar[dt](1):
        r = Scalar[dt](1)
    return r


@always_inline
def _max3[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    if a > b:
        return a if a > c else c
    return b if b > c else c


@always_inline
def _mix[dt: DType](a: Scalar[dt], x: Scalar[dt], one_minus_a: Scalar[dt], y: Scalar[dt]) -> Scalar[dt]:
    """`a * x + (1 - a) * y`: first product fused, second stored."""
    return _f[dt](_mad[dt](a, x, _f[dt](one_minus_a * y)))


@always_inline
def _dot3[dt: DType](
    a1: Scalar[dt], a2: Scalar[dt], a3: Scalar[dt], b1: Scalar[dt], b2: Scalar[dt], b3: Scalar[dt]
) -> Scalar[dt]:
    var t = _f[dt](a1 * b1)
    t = _f[dt](_mad[dt](a2, b2, t))
    t = _f[dt](_mad[dt](a3, b3, t))
    return t


def _const[dt: DType](v: Float32) -> Scalar[dt]:
    """A float32 constant of the device build (already rounded as the
    device rounds it) widened exactly."""
    return Scalar[dt](v)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


struct HWOracleFit[dt: DType](Movable):
    var n: Int
    var batch_size: Int
    var frequency: Int
    var additive: Bool
    var ts: List[Scalar[Self.dt]]            # time-major, n x batch
    var decomp_trend: List[Scalar[Self.dt]]  # trend_len x batch
    var decomp_season: List[Scalar[Self.dt]]
    var trend_len: Int
    var start_level: List[Scalar[Self.dt]]
    var start_trend: List[Scalar[Self.dt]]
    var start_season: List[Scalar[Self.dt]]  # frequency x batch
    var alpha: List[Scalar[Self.dt]]
    var beta: List[Scalar[Self.dt]]
    var gamma: List[Scalar[Self.dt]]
    var criterion: List[Int]
    var niter: List[Int]
    #: DEVIATION 699's packed branch record (mirrors the device bit for bit).
    var decisions: List[Int]
    var iter_trace: List[Scalar[Self.dt]]    # [(iter*3+k)*batch + s]
    var trace_iters: Int
    var sse: List[Scalar[Self.dt]]
    var level: List[Scalar[Self.dt]]         # (n - f) x batch
    var trend: List[Scalar[Self.dt]]
    var season: List[Scalar[Self.dt]]

    def __init__(out self, n: Int, batch_size: Int, frequency: Int, additive: Bool, trace_iters: Int):
        self.n = n
        self.batch_size = batch_size
        self.frequency = frequency
        self.additive = additive
        self.ts = List[Scalar[Self.dt]]()
        self.decomp_trend = List[Scalar[Self.dt]]()
        self.decomp_season = List[Scalar[Self.dt]]()
        self.trend_len = 0
        self.start_level = List[Scalar[Self.dt]]()
        self.start_trend = List[Scalar[Self.dt]]()
        self.start_season = List[Scalar[Self.dt]]()
        self.alpha = List[Scalar[Self.dt]]()
        self.beta = List[Scalar[Self.dt]]()
        self.gamma = List[Scalar[Self.dt]]()
        self.criterion = List[Int]()
        self.niter = List[Int]()
        self.decisions = List[Int]()
        self.iter_trace = List[Scalar[Self.dt]]()
        self.trace_iters = trace_iters
        self.sse = List[Scalar[Self.dt]]()
        self.level = List[Scalar[Self.dt]]()
        self.trend = List[Scalar[Self.dt]]()
        self.season = List[Scalar[Self.dt]]()


def _zeros[dt: DType](n: Int) -> List[Scalar[dt]]:
    var out = List[Scalar[dt]]()
    out.reserve(n)
    for _ in range(n):
        out.append(Scalar[dt](0))
    return out^


# ---------------------------------------------------------------------------
# The recurrence (hw_eval.cuh), one series, returns the SSE
# ---------------------------------------------------------------------------


def oracle_eval[dt: DType](
    s: Int,
    ts: List[Scalar[dt]],
    n: Int,
    batch_size: Int,
    frequency: Int,
    shift: Int,
    plevel_in: Scalar[dt],
    ptrend_in: Scalar[dt],
    mut pseason: List[Scalar[dt]],     # frequency entries, this series' scratch
    start_season: List[Scalar[dt]],    # frequency x batch
    alpha_in: Scalar[dt],
    beta_in: Scalar[dt],
    gamma_in: Scalar[dt],
    additive: Bool,
    write: Bool,
    mut level: List[Scalar[dt]],
    mut trend: List[Scalar[dt]],
    mut season: List[Scalar[dt]],
) -> Scalar[dt]:
    var alpha_ = _bound[dt](alpha_in)
    var beta_ = _bound[dt](beta_in)
    var gamma_ = _bound[dt](gamma_in)
    var oma = _f[dt](Scalar[dt](1) - alpha_)
    var omb = _f[dt](Scalar[dt](1) - beta_)
    var omg = _f[dt](Scalar[dt](1) - gamma_)
    var stmp_eps_c = _const[dt](STMP_EPS)
    var plevel = plevel_in
    var ptrend = ptrend_in
    var error_ = Scalar[dt](0)
    var clevel = Scalar[dt](0)
    var ctrend = Scalar[dt](0)
    for i in range(n - shift):
        var cseason: Scalar[dt]
        var ph = i % frequency
        var pts = ts[s + (i + shift) * batch_size]
        var leveltrend = _f[dt](plevel + ptrend)
        var stmp: Scalar[dt]
        if i < frequency:
            stmp = start_season[s + i * batch_size]
        else:
            stmp = pseason[ph]
        var xhat_: Scalar[dt]
        if additive:
            xhat_ = _f[dt](leveltrend + stmp)
        else:
            xhat_ = _f[dt](leveltrend * stmp)
        var diff = _f[dt](pts - xhat_)
        error_ = _f[dt](_mad[dt](diff, diff, error_))
        if additive:
            clevel = _mix[dt](alpha_, _f[dt](pts - stmp), oma, leveltrend)
        else:
            var stmp_eps: Scalar[dt] = stmp if abs(stmp) > stmp_eps_c else stmp_eps_c
            clevel = _mix[dt](alpha_, _f[dt](pts / stmp_eps), oma, leveltrend)
        ctrend = _mix[dt](beta_, _f[dt](clevel - plevel), omb, ptrend)
        ptrend = ctrend
        if additive:
            cseason = _mix[dt](gamma_, _f[dt](pts - clevel), omg, stmp)
        else:
            cseason = _mix[dt](gamma_, _f[dt](pts / clevel), omg, stmp)
        pseason[ph] = cseason
        plevel = clevel
        if write:
            level[s + i * batch_size] = clevel
            trend[s + i * batch_size] = ctrend
            season[s + i * batch_size] = cseason
    return error_


def _loss[dt: DType](
    s: Int, fit: HWOracleFit[dt], mut pseason: List[Scalar[dt]],
    a: Scalar[dt], b: Scalar[dt], g: Scalar[dt],
) -> Scalar[dt]:
    var none_l = List[Scalar[dt]]()
    var none_t = List[Scalar[dt]]()
    var none_s = List[Scalar[dt]]()
    return oracle_eval[dt](
        s, fit.ts, fit.n, fit.batch_size, fit.frequency, fit.frequency,
        fit.start_level[s], fit.start_trend[s], pseason, fit.start_season,
        a, b, g, fit.additive, False, none_l, none_t, none_s,
    )


def _gradient[dt: DType](
    s: Int, fit: HWOracleFit[dt], mut pseason: List[Scalar[dt]],
    a: Scalar[dt], b: Scalar[dt], g: Scalar[dt], eps: Scalar[dt],
    mut g1: Scalar[dt], mut g2: Scalar[dt], mut g3: Scalar[dt],
):
    var two_eps = _f[dt](eps * Scalar[dt](2))
    var l = _loss[dt](s, fit, pseason, _f[dt](a - eps), b, g)
    var r = _loss[dt](s, fit, pseason, _f[dt](a + eps), b, g)
    g1 = _f[dt](_f[dt](r - l) / two_eps)
    l = _loss[dt](s, fit, pseason, a, _f[dt](b - eps), g)
    r = _loss[dt](s, fit, pseason, a, _f[dt](b + eps), g)
    g2 = _f[dt](_f[dt](r - l) / two_eps)
    l = _loss[dt](s, fit, pseason, a, b, _f[dt](g - eps))
    r = _loss[dt](s, fit, pseason, a, b, _f[dt](g + eps))
    g3 = _f[dt](_f[dt](r - l) / two_eps)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def oracle_fit[dt: DType](
    data: List[Float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    start_periods: Int,
    seasonal: Int,
    eps_f32: Float32,
    trace_iters: Int,
    zero_dir_guard: Bool = True,
    bfgs_iter_limit: Int = -1,
    linesearch_iter_limit: Int = -1,
) raises -> HWOracleFit[dt]:
    """`HoltWintersFitHelper` on the host: transpose, decompose, BFGS,
    final eval. `data` is series-major float32 (widened exactly for
    float64). `zero_dir_guard = False` reproduces their unguarded step
    (DEVIATION 662 off) for the sabotage comparison."""
    var additive = seasonal == SEASONAL_ADDITIVE
    var fit = HWOracleFit[dt](n, batch_size, frequency, additive, trace_iters)
    # Step 1: transpose to time-major
    fit.ts = _zeros[dt](n * batch_size)
    for s in range(batch_size):
        for t in range(n):
            fit.ts[s + t * batch_size] = Scalar[dt](data[s * n + t])

    # Step 2: stl_decomposition_gpu
    var end = start_periods * frequency
    var filter_size = (frequency // 2) * 2 + 1
    var trend_len = end - (filter_size - 1)
    fit.trend_len = trend_len
    var filter_f32 = host_filter(frequency)
    var filt = List[Scalar[dt]]()
    for i in range(filter_size):
        filt.append(Scalar[dt](filter_f32[i]))
    # conv1d
    fit.decomp_trend = _zeros[dt](trend_len * batch_size)
    for s in range(batch_size):
        for o in range(trend_len):
            var out = Scalar[dt](0)
            for i in range(filter_size):
                out = _f[dt](_mad[dt](filt[i], fit.ts[s + (i + o) * batch_size], out))
            fit.decomp_trend[s + o * batch_size] = out
    # the residual
    var ts_offset = (filter_size // 2) * batch_size
    fit.decomp_season = _zeros[dt](trend_len * batch_size)
    for k in range(trend_len * batch_size):
        if additive:
            fit.decomp_season[k] = _f[dt](fit.ts[ts_offset + k] - fit.decomp_trend[k])
        else:
            fit.decomp_season[k] = _f[dt](fit.ts[ts_offset + k] / fit.decomp_trend[k])
    # season_mean
    fit.start_season = _zeros[dt](frequency * batch_size)
    var half = filter_size // 2
    for s in range(batch_size):
        var mean = Scalar[dt](0)
        for i in range(frequency):
            var pm = Scalar[dt](0)
            var k = i
            while k < trend_len:
                pm = _f[dt](pm + fit.decomp_season[k * batch_size + s])
                k += frequency
            var count = 1 + ((trend_len - i - 1) // frequency)
            pm = _f[dt](pm / Scalar[dt](count))
            fit.start_season[((i + half) % frequency) * batch_size + s] = pm
            mean = _f[dt](mean + pm)
        mean = _f[dt](mean / Scalar[dt](frequency))
        for i in range(frequency):
            var v = fit.start_season[i * batch_size + s]
            if additive:
                fit.start_season[i * batch_size + s] = _f[dt](v - mean)
            else:
                fit.start_season[i * batch_size + s] = _f[dt](v / mean)
    # batched_ls (DEVIATION 660's R1Qt)
    var rq_f32 = host_r1qt(trend_len)
    var rq = List[Scalar[dt]]()
    for i in range(2 * trend_len):
        rq.append(Scalar[dt](rq_f32[i]))
    fit.start_level = _zeros[dt](batch_size)
    fit.start_trend = _zeros[dt](batch_size)
    for s in range(batch_size):
        var lv = Scalar[dt](0)
        var tr = Scalar[dt](0)
        for i in range(trend_len):
            var b = fit.decomp_trend[s + i * batch_size]
            lv = _f[dt](_mad[dt](rq[2 * i], b, lv))
            tr = _f[dt](_mad[dt](rq[2 * i + 1], b, tr))
        fit.start_level[s] = lv
        fit.start_trend[s] = tr

    # Step 3: BFGS per series
    var p = default_optim_params(eps_f32)
    # their override block's two gate-relevant members (see runner.mojo)
    if bfgs_iter_limit > 0:
        p.bfgs_iter_limit = bfgs_iter_limit
    if linesearch_iter_limit > 0:
        p.linesearch_iter_limit = linesearch_iter_limit
    var eps = _const[dt](p.eps)
    var min_param_diff = _const[dt](p.min_param_diff)
    var min_error_diff = _const[dt](p.min_error_diff)
    var min_grad_norm = _const[dt](p.min_grad_norm)
    var tau = _const[dt](p.linesearch_tau)
    var c_ls = _const[dt](p.linesearch_c)
    var ls_initial = _const[dt](Float32(Float64(0.866)))
    fit.iter_trace = _zeros[dt](trace_iters * 3 * batch_size)
    fit.alpha = _zeros[dt](batch_size)
    fit.beta = _zeros[dt](batch_size)
    fit.gamma = _zeros[dt](batch_size)
    fit.sse = _zeros[dt](batch_size)
    var comps = (n - frequency) * batch_size
    fit.level = _zeros[dt](comps)
    fit.trend = _zeros[dt](comps)
    fit.season = _zeros[dt](comps)
    for s in range(batch_size):
        var pseason = _zeros[dt](frequency)
        var x1 = _const[dt](HW_ALPHA0)
        var x2 = _const[dt](HW_BETA0)
        var x3 = _const[dt](HW_GAMMA0)
        var H11 = Scalar[dt](1)
        var H12 = Scalar[dt](0)
        var H13 = Scalar[dt](0)
        var H22 = Scalar[dt](1)
        var H23 = Scalar[dt](0)
        var H33 = Scalar[dt](1)
        var g1 = Scalar[dt](0)
        var g2 = Scalar[dt](0)
        var g3 = Scalar[dt](0)
        _gradient[dt](s, fit, pseason, x1, x2, x3, eps, g1, g2, g3)
        var crit = OPTIM_BFGS_ITER_LIMIT
        var niter = 0
        # DEVIATION 699, mirrored from the device term for term.
        var decisions = 0
        var ls_halvings = 0
        for it in range(p.bfgs_iter_limit):
            var p1 = -_dot3[dt](H11, H12, H13, g1, g2, g3)
            var p2 = -_dot3[dt](H12, H22, H23, g1, g2, g3)
            var p3 = -_dot3[dt](H13, H23, H33, g1, g2, g3)
            var phi = _dot3[dt](p1, p2, p3, g1, g2, g3)
            if phi > Scalar[dt](0):
                decisions |= HW_DEC_HESSIAN_RESET
                H11 = Scalar[dt](1)
                H12 = Scalar[dt](0)
                H13 = Scalar[dt](0)
                H22 = Scalar[dt](1)
                H23 = Scalar[dt](0)
                H33 = Scalar[dt](1)
                p1 = -g1
                p2 = -g2
                p3 = -g3
            var pp = _dot3[dt](p1, p2, p3, p1, p2, p3)
            if zero_dir_guard and pp == Scalar[dt](0):
                decisions |= HW_DEC_ZERO_DIR
                crit = OPTIM_MIN_GRAD_NORM
                break
            var step_size = _f[dt](ls_initial / _sqrt[dt](pp))
            var nx1 = _f[dt](_mad[dt](step_size, p1, x1))
            var nx2 = _f[dt](_mad[dt](step_size, p2, x2))
            var nx3 = _f[dt](_mad[dt](step_size, p3, x3))
            var cauchy = _f[dt](c_ls * _dot3[dt](g1, g2, g3, p1, p2, p3))
            var loss_ref = _loss[dt](s, fit, pseason, x1, x2, x3)
            var loss = _loss[dt](s, fit, pseason, nx1, nx2, nx3)
            var i = 0
            while i < p.linesearch_iter_limit and (loss > _f[dt](_mad[dt](step_size, cauchy, loss_ref))):
                step_size = _f[dt](step_size * tau)
                nx1 = _f[dt](_mad[dt](step_size, p1, x1))
                nx2 = _f[dt](_mad[dt](step_size, p2, x2))
                nx3 = _f[dt](_mad[dt](step_size, p3, x3))
                loss = _loss[dt](s, fit, pseason, nx1, nx2, nx3)
                i += 1
            ls_halvings += i
            if i >= p.linesearch_iter_limit:
                decisions |= HW_DEC_LS_LIMIT
            var dx1 = abs(_f[dt](x1 - nx1))
            var dx2 = abs(_f[dt](x2 - nx2))
            var dx3 = abs(_f[dt](x3 - nx3))
            var mx = _max3[dt](dx1, dx2, dx3)
            x1 = nx1
            x2 = nx2
            x3 = nx3
            niter = it + 1
            if it < trace_iters:
                fit.iter_trace[(it * 3 + 0) * batch_size + s] = x1
                fit.iter_trace[(it * 3 + 1) * batch_size + s] = x2
                fit.iter_trace[(it * 3 + 2) * batch_size + s] = x3
            var crit_param = min_param_diff > mx
            var crit_error = min_error_diff > abs(_f[dt](loss - loss_ref))
            if crit_param and crit_error:
                decisions |= HW_DEC_BOTH_CRIT
            if crit_param:
                crit = OPTIM_MIN_PARAM_DIFF
                break
            if crit_error:
                crit = OPTIM_MIN_ERROR_DIFF
                break
            var ng1 = Scalar[dt](0)
            var ng2 = Scalar[dt](0)
            var ng3 = Scalar[dt](0)
            _gradient[dt](s, fit, pseason, nx1, nx2, nx3, eps, ng1, ng2, ng3)
            mx = _max3[dt](abs(ng1), abs(ng2), abs(ng3))
            if min_grad_norm > mx:
                crit = OPTIM_MIN_GRAD_NORM
                break
            var s1 = _f[dt](step_size * p1)
            var s2 = _f[dt](step_size * p2)
            var s3 = _f[dt](step_size * p3)
            var y1 = _f[dt](ng1 - g1)
            var y2 = _f[dt](ng2 - g2)
            var y3 = _f[dt](ng3 - g3)
            var rho_ = _dot3[dt](y1, y2, y3, s1, s2, s3)
            if rho_ == Scalar[dt](0):
                decisions |= HW_DEC_RHO_ZERO
            var rho = _f[dt](Scalar[dt](1) / rho_)
            var Hy1 = _dot3[dt](H11, H12, H13, y1, y2, y3)
            var Hy2 = _dot3[dt](H12, H22, H23, y1, y2, y3)
            var Hy3 = _dot3[dt](H13, H23, H33, y1, y2, y3)
            var k = _f[dt](_f[dt](rho * rho) * _f[dt](_dot3[dt](y1, y2, y3, Hy1, Hy2, Hy3) + rho_))
            var two_rho = _f[dt](Scalar[dt](2) * rho)
            H11 = _f[dt](H11 + _f[dt](_f[dt](_f[dt](k * s1) * s1) - _f[dt](_f[dt](two_rho * s1) * Hy1)))
            H12 = _f[dt](H12 + _f[dt](_f[dt](_f[dt](k * s1) * s2) - _f[dt](rho * _f[dt](_mad[dt](s2, Hy1, _f[dt](s1 * Hy2))))))
            H13 = _f[dt](H13 + _f[dt](_f[dt](_f[dt](k * s1) * s3) - _f[dt](rho * _f[dt](_mad[dt](s3, Hy1, _f[dt](s1 * Hy3))))))
            H22 = _f[dt](H22 + _f[dt](_f[dt](_f[dt](k * s2) * s2) - _f[dt](_f[dt](two_rho * s2) * Hy2)))
            H23 = _f[dt](H23 + _f[dt](_f[dt](_f[dt](k * s2) * s3) - _f[dt](rho * _f[dt](_mad[dt](s3, Hy2, _f[dt](s2 * Hy3))))))
            H33 = _f[dt](H33 + _f[dt](_f[dt](_f[dt](k * s3) * s3) - _f[dt](_f[dt](two_rho * s3) * Hy3)))
            if not (H11 == H11 and H22 == H22 and H33 == H33
                    and H12 == H12 and H13 == H13 and H23 == H23):
                decisions |= HW_DEC_H_NAN
            g1 = ng1
            g2 = ng2
            g3 = ng3
        if crit == OPTIM_BFGS_ITER_LIMIT:
            # the loop ran to completion: the device's fall-through return.
            decisions |= HW_DEC_ITER_LIMIT
        if not (x1 > Scalar[dt](0)):
            decisions |= HW_DEC_ALPHA_LO
        elif x1 > Scalar[dt](1):
            decisions |= HW_DEC_ALPHA_HI
        if not (x2 > Scalar[dt](0)):
            decisions |= HW_DEC_BETA_LO
        elif x2 > Scalar[dt](1):
            decisions |= HW_DEC_BETA_HI
        if not (x3 > Scalar[dt](0)):
            decisions |= HW_DEC_GAMMA_LO
        elif x3 > Scalar[dt](1):
            decisions |= HW_DEC_GAMMA_HI
        decisions |= ls_halvings << HW_DEC_HALVING_SHIFT
        fit.criterion.append(crit)
        fit.niter.append(niter)
        fit.decisions.append(decisions)
        fit.alpha[s] = _bound[dt](x1)
        fit.beta[s] = _bound[dt](x2)
        fit.gamma[s] = _bound[dt](x3)
        # Final fit (with the UNBOUNDED x, as theirs: the eval bounds)
        fit.sse[s] = oracle_eval[dt](
            s, fit.ts, n, batch_size, frequency, frequency,
            fit.start_level[s], fit.start_trend[s], pseason, fit.start_season,
            x1, x2, x3, additive, True, fit.level, fit.trend, fit.season,
        )
    return fit^


def oracle_forecast[dt: DType](fit: HWOracleFit[dt], h: Int) -> List[Scalar[dt]]:
    """`HoltWintersForecastHelper`: from the last fitted row of level and
    trend and the last `frequency` rows of season; `h x batch`, time-major."""
    var bs = fit.batch_size
    var f = fit.frequency
    var n_minus = fit.n - f
    var lt_shift = (n_minus - 1) * bs
    var s_shift = (n_minus - f) * bs
    var out = _zeros[dt](h * bs)
    for s in range(bs):
        var level = fit.level[lt_shift + s]
        var trend = fit.trend[lt_shift + s]
        for i in range(h):
            var season = fit.season[s_shift + s + (i % f) * bs]
            var lt = _f[dt](_mad[dt](trend, Scalar[dt](i + 1), level))
            if fit.additive:
                out[s + i * bs] = _f[dt](lt + season)
            else:
                out[s + i * bs] = _f[dt](lt * season)
    return out^


def oracle_sse_at[dt: DType](
    fit: HWOracleFit[dt], s: Int, a: Scalar[dt], b: Scalar[dt], g: Scalar[dt]
) -> Scalar[dt]:
    """The SSE of series `s` at given parameters (the gates' "start SSE")."""
    var pseason = _zeros[dt](fit.frequency)
    return _loss[dt](s, fit, pseason, a, b, g)
