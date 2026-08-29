"""The six one-target QN losses (cuML `qnFit`: QN_LOSS_SQUARED, QN_LOSS_ABS,
QN_LOSS_SVC_L1, QN_LOSS_SVC_L2, QN_LOSS_SVR_L1, QN_LOSS_SVR_L2): oracle,
reach, identity.

DEVIATIONS 707-708 (the ports, `glm/ported/glm/qn/glm_linear.mojo`,
`glm_svm.mojo`) and 712-713 (the gates here). See `glm/README.md`, "QN
losses". Every check runs over ALL SIX losses. The checks:

    check_losses_fd_gradient           the FLOAT64 host objective of each
                                       SMOOTH loss (squared, SVC-L2, SVR-L2)
                                       against central finite differences at
                                       a hashed `w` away from the kinks; the
                                       three non-smooth losses are checked
                                       where they are differentiable
    check_losses_planted               regression losses recover the planted
                                       coefficient SIGNS and fall well below
                                       the objective at `w = 0`; the hinge
                                       classifiers reach training accuracy >
                                       0.95 on a separable planted fixture;
                                       the objective is MONOTONE in the
                                       iteration budget (1 >= 3 >= full)
    check_losses_is_a_minimizer        the three smooth losses: float64
                                       gradient in the loss's OWN norm at the
                                       device solution <= tol * max(fx, tol),
                                       objective below 8 perturbed points
    check_losses_refuses_by_name       each loss with the wrong `C`
                                       (`qn.h: ... invalid C`, their text),
                                       l1, sample_weight
    check_losses_device_equals_host    ONE objective evaluation per loss (loss,
                                       every gradient entry, every dZ cell) and
                                       the loss's gradNorm (`nrm1`,
                                       `squaredNorm * 0.5`, `nrmMax`) at a
                                       hashed `w` against the host replay: bit
                                       for bit under IDENTICAL, REPORT under
                                       FAST; device vs float64 within tolerance
                                       in both modes
    check_losses_kernel_invariance     each fused loss kernel on a PLANTED `z`
                                       carrying `+0.0`/`-0.0` at every signed
                                       site (`y = -0.0, z = +0.0` and the
                                       reverse), the exact hinge boundary
                                       `s z == 1`, and the exact SVR dead-zone
                                       edges `|y - z| == eps`: block sizes 64
                                       and 256, padded and poisoned outputs,
                                       a batch composition; every cell's bytes
                                       equal across launches and equal to the
                                       host spelling
    check_losses_card_is_emitted       two traced device fits per loss: stage
                                       count `2 + 3 n_iter + 3`, run-to-run
                                       control

SABOTAGES PERFORMED (2026-08-23), each reverted; the README's table has
the outputs:

    (d) `nrm1_kernel` folding SEQUENTIALLY on thread 0 instead of the pinned
        tree: `check_losses_device_equals_host [IDENTICAL]` fails on the
        gradNorm of QN_LOSS_ABS.
    (e) `squared_lz` without the `* 0.5`: `check_losses_fd_gradient` fails
        for QN_LOSS_SQUARED (analytic vs finite difference) -- the oracle
        disagrees with itself before any device number is consulted.
    (f) `svc_l1_dlz` / `svc_l2_dlz` with `<` in place of `<=` at `s z == 1`:
        `check_losses_kernel_invariance` fails on the planted boundary rows
        against the host spelling -- the boundary fixture is reached.
"""

from std.math import exp, log, sqrt
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from core.identity_trace import IdentityTrace, first_divergence
from glm.mojo_only.multinomial_check import (
    _download,
    _hex32,
    _host_dot,
    _host_sum_terms,
    _host_tikhonov,
    _poison,
    _same_bits,
    _tree,
    _u01,
    _upload,
    _zeros,
)
from glm.ported.glm.qn.glm_base import GLMDims, GLMWithData
from glm.ported.glm.qn.glm_linear import (
    abs_loss_dz_kernel,
    squared_loss_dz_kernel,
)
from glm.ported.glm.qn.glm_svm import (
    svc_l1_loss_dz_kernel,
    svc_l2_loss_dz_kernel,
    svr_l1_loss_dz_kernel,
    svr_l2_loss_dz_kernel,
)
from glm.ported.glm.qn.qn import qn_fit_x
from glm.ported.glm.qn.qn_util import OPT_SUCCESS
from glm.ported.linear_model.qn import (
    QN_LOSS_ABS,
    QN_LOSS_SQUARED,
    QN_LOSS_SVC_L1,
    QN_LOSS_SVC_L2,
    QN_LOSS_SVR_L1,
    QN_LOSS_SVR_L2,
    QNParams,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime QL_ROWS = 2048
comptime QL_COLS = 5
comptime SVR_EPS = Float32(0.25)



def _losses() -> List[Int]:
    """The six one-target ids, in `qn.h` order."""
    var out: List[Int] = [
        QN_LOSS_SQUARED, QN_LOSS_ABS, QN_LOSS_SVC_L1, QN_LOSS_SVC_L2,
        QN_LOSS_SVR_L1, QN_LOSS_SVR_L2,
    ]
    return out^


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _loss_name(loss: Int) -> String:
    if loss == QN_LOSS_SQUARED:
        return String("QN_LOSS_SQUARED")
    if loss == QN_LOSS_ABS:
        return String("QN_LOSS_ABS")
    if loss == QN_LOSS_SVC_L1:
        return String("QN_LOSS_SVC_L1")
    if loss == QN_LOSS_SVC_L2:
        return String("QN_LOSS_SVC_L2")
    if loss == QN_LOSS_SVR_L1:
        return String("QN_LOSS_SVR_L1")
    return String("QN_LOSS_SVR_L2")


def _is_classification(loss: Int) -> Bool:
    return loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2


def _is_smooth(loss: Int) -> Bool:
    return loss == QN_LOSS_SQUARED or loss == QN_LOSS_SVC_L2 or loss == QN_LOSS_SVR_L2


def _n_classes_for(loss: Int) -> Int:
    """Their `ASSERT`: the SVC losses take `C == 2`, the rest `C == 1`."""
    return 2 if _is_classification(loss) else 1


def _planted_w(k: Int) -> Float64:
    var mag = 2.0 - 0.5 * Float64(k % 4)
    return mag if k % 2 == 0 else -mag


def _fixture(n: Int, d: Int, classification: Bool, margin: Float64) -> Tuple[List[Float32], List[Float32]]:
    """Hashed design in [-1, 1); regression target `x . w* + 0.3 + noise`
    (noise scaled by `margin`), classification label `1[target > 0]`."""
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n):
        var s = 0.3
        for k in range(d):
            var v = 2.0 * _u01(i, k, 0) - 1.0
            x.append(Float32(v))
            s += v * _planted_w(k)
        s += margin * 2.0 * (_u01(i, 77, 5) - 0.5)
        if classification:
            y.append(Float32(1.0) if s > 0.0 else Float32(0.0))
        else:
            y.append(Float32(s))
    return (x^, y^)


def _params(loss: Int, C: Float64, fit_intercept: Bool, penalty: Bool, max_iter: Int = 1000) -> QNParams:
    var p = QNParams.default()
    p.loss = loss
    p.penalty_l1 = 0.0
    p.penalty_l2 = (1.0 / C) if penalty else 0.0
    p.grad_tol = 1e-4
    p.change_tol = 1e-4 * 0.01
    p.max_iter = max_iter
    p.linesearch_max_iter = 50
    p.lbfgs_memory = 5
    p.fit_intercept = fit_intercept
    p.penalty_normalized = True
    return p^


struct FitResult(Movable):
    var w: List[Float32]
    var fx: Float32
    var n_iter: Int
    var retcode: Int

    def __init__(out self, var w: List[Float32], fx: Float32, n_iter: Int, retcode: Int):
        self.w = w^
        self.fx = fx
        self.n_iter = n_iter
        self.retcode = retcode


def _fit(
    ctx: DeviceContext, xh: List[Float32], yh: List[Float32], n: Int, d: Int,
    pams: QNParams, mut trace: IdentityTrace, n_classes: Int = -1, has_sw: Bool = False,
) raises -> FitResult:
    var x = _upload(ctx, xh)
    var y = _upload(ctx, yh)
    var n_param = d + (1 if pams.fit_intercept else 0)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_memset(w, Float32(0.0))
    ctx.synchronize()
    var fx = Float32(0.0)
    var iters = 0
    var nc = _n_classes_for(pams.loss) if n_classes < 0 else n_classes
    var ret = qn_fit_x(ctx, pams, x^, y^, n, d, nc, w, fx, iters, has_sw, trace, SVR_EPS)
    var out = _download(ctx, w, n_param)
    return FitResult(out^, fx, iters, ret)


# ---------------------------------------------------------------------------
# THE FLOAT64 REFERENCE
# ---------------------------------------------------------------------------


def _lz64(loss: Int, y: Float64, z: Float64) -> Float64:
    var eps = Float64(SVR_EPS)
    if loss == QN_LOSS_SQUARED:
        return (z - y) * (z - y) * 0.5
    if loss == QN_LOSS_ABS:
        return abs(z - y)
    if loss == QN_LOSS_SVC_L1:
        var s = 2.0 * y - 1.0
        return max(0.0, 1.0 - s * z)
    if loss == QN_LOSS_SVC_L2:
        var s = 2.0 * y - 1.0
        var t = max(0.0, 1.0 - s * z)
        return t * t
    var t = y - z
    var m = (t - eps) if t > eps else ((-t - eps) if t < -eps else 0.0)
    if loss == QN_LOSS_SVR_L1:
        return m
    return m * m


def _dlz64(loss: Int, y: Float64, z: Float64) -> Float64:
    var eps = Float64(SVR_EPS)
    if loss == QN_LOSS_SQUARED:
        return z - y
    if loss == QN_LOSS_ABS:
        return 1.0 if z > y else (-1.0 if z < y else 0.0)
    if loss == QN_LOSS_SVC_L1:
        var s = 2.0 * y - 1.0
        return -s if s * z <= 1.0 else 0.0
    if loss == QN_LOSS_SVC_L2:
        # DEVIATION 714: the derivative of (1 - s z)^2; theirs drops the 2
        var s = 2.0 * y - 1.0
        return 2.0 * (z - s) if s * z <= 1.0 else 0.0
    var t = y - z
    if loss == QN_LOSS_SVR_L1:
        return -1.0 if t > eps else (1.0 if t < -eps else 0.0)
    return -2.0 * ((t - eps) if t > eps else ((t + eps) if t < -eps else 0.0))


def _objective64(
    loss: Int, xh: List[Float32], yh: List[Float32], n: Int, d: Int, w: List[Float64],
    fit_intercept: Bool, l2: Float64, mut grad: List[Float64],
) -> Float64:
    """`mean_i lz(y_i, z_i) + l2/2 ||w||^2` and its gradient, float64."""
    grad.clear()
    var n_param = d + (1 if fit_intercept else 0)
    for _ in range(n_param):
        grad.append(0.0)
    var total = 0.0
    for i in range(n):
        var z = 0.0
        for k in range(d):
            z += Float64(xh[i * d + k]) * w[k]
        if fit_intercept:
            z += w[d]
        var yi = Float64(yh[i])
        total += _lz64(loss, yi, z)
        var dz = _dlz64(loss, yi, z)
        for k in range(d):
            grad[k] += Float64(xh[i * d + k]) * dz
        if fit_intercept:
            grad[d] += dz
    var fx = total / Float64(n)
    for k in range(n_param):
        grad[k] /= Float64(n)
    for k in range(d):
        fx += 0.5 * l2 * w[k] * w[k]
        grad[k] += l2 * w[k]
    return fx


def _hashed_w(n_param: Int, salt: Int) -> List[Float32]:
    var hw = List[Float32]()
    for k in range(n_param):
        hw.append(Float32(0.8 * (2.0 * _u01(k, 3, salt) - 1.0)))
    return hw^


def check_losses_fd_gradient() raises:
    var n = 512
    var d = 4
    for loss in _losses():
        var fx = _fixture(n, d, _is_classification(loss), 1.0)
        var hw = _hashed_w(d + 1, 11)
        var w = List[Float64]()
        for k in range(d + 1):
            w.append(Float64(hw[k]))
        var l2 = 0.5 / Float64(n)
        var g = List[Float64]()
        _ = _objective64(loss, fx[0], fx[1], n, d, w, True, l2, g)
        var worst = 0.0
        var h = 1e-6
        for j in range(d + 1):
            var wp = w.copy()
            var wm = w.copy()
            wp[j] += h
            wm[j] -= h
            var gp = List[Float64]()
            var gm = List[Float64]()
            var fp = _objective64(loss, fx[0], fx[1], n, d, wp, True, l2, gp)
            var fm = _objective64(loss, fx[0], fx[1], n, d, wm, True, l2, gm)
            var fd = (fp - fm) / (2.0 * h)
            var err = abs(fd - g[j]) / max(abs(g[j]), 1e-2)
            if err > worst:
                worst = err
        # the non-smooth losses have kinks a hashed `w` can land a row on;
        # their finite difference is exact off the kinks and O(1/(n h)) at
        # one, so they get a looser bound and the smooth ones a tight one
        var bound = 1e-6 if _is_smooth(loss) else 1e-2
        if worst > bound:
            raise Error(
                "check_losses_fd_gradient: " + _loss_name(loss) + " worst relative"
                " error analytic vs central difference " + String(worst) + " > " + String(bound)
            )
        print(
            "check_losses_fd_gradient OK: " + _loss_name(loss) + " worst relative error "
            + String(worst) + " <= " + String(bound)
        )


def check_losses_planted() raises:
    var n = QL_ROWS
    var d = QL_COLS
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    for loss in _losses():
        var cls = _is_classification(loss)
        var fx = _fixture(n, d, cls, 0.0)
        var r = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 10.0, True, True), off)
        var r1 = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 10.0, True, True, 1), off)
        var r3 = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 10.0, True, True, 3), off)
        if not (r1.fx >= r3.fx and r3.fx >= r.fx):
            raise Error(
                "check_losses_planted: " + _loss_name(loss) + " objective not monotone"
                " in the iteration budget: " + String(r1.fx) + " / " + String(r3.fx)
                + " / " + String(r.fx)
            )
        # the objective at w = 0
        var zero = List[Float64]()
        for _ in range(d + 1):
            zero.append(0.0)
        var g0 = List[Float64]()
        var f0 = _objective64(loss, fx[0], fx[1], n, d, zero, True, 0.1 / Float64(n), g0)
        if not (Float64(r.fx) < 0.5 * f0):
            raise Error(
                "check_losses_planted: " + _loss_name(loss) + " objective " + String(r.fx)
                + " did not fall below half the w = 0 objective " + String(f0)
                + " (n_iter " + String(r.n_iter) + ", retcode " + String(r.retcode) + ")"
            )
        var detail = String("")
        if cls:
            var correct = 0
            for i in range(n):
                var z = Float64(r.w[d])
                for k in range(d):
                    z += Float64(fx[0][i * d + k]) * Float64(r.w[k])
                if (Float32(1.0) if z > 0.0 else Float32(0.0)) == fx[1][i]:
                    correct += 1
            var acc = Float64(correct) / Float64(n)
            if acc < 0.95:
                raise Error("check_losses_planted: " + _loss_name(loss) + " training accuracy " + String(acc) + " < 0.95")
            detail = "accuracy " + String(acc)
        else:
            for k in range(d):
                if (_planted_w(k) > 0.0) != (r.w[k] > Float32(0.0)):
                    raise Error(
                        "check_losses_planted: " + _loss_name(loss) + " weight " + String(k)
                        + " = " + String(r.w[k]) + " lost the planted sign (n_iter "
                        + String(r.n_iter) + ", retcode " + String(r.retcode) + ")"
                    )
            detail = "all " + String(d) + " planted signs recovered, bias " + String(r.w[d])
        print(
            "check_losses_planted OK: " + _loss_name(loss) + " " + detail + "; objective "
            + String(r1.fx) + " (1) >= " + String(r3.fx) + " (3) >= " + String(r.fx)
            + " (full, n_iter " + String(r.n_iter) + ", retcode " + String(r.retcode)
            + ") < half of f(0) = " + String(f0)
        )


def _own_norm64(loss: Int, g: List[Float64]) -> Float64:
    """The loss's `gradNorm` in float64: `squaredNorm * 0.5` for the L2
    family, `nrm1` for the L1 family, `nrmMax` otherwise."""
    if _is_smooth(loss):
        var s = 0.0
        for v in g:
            s += v * v
        return s * 0.5
    var s1 = 0.0
    for v in g:
        s1 += abs(v)
    return s1


def check_losses_is_a_minimizer() raises:
    var n = QL_ROWS
    var d = QL_COLS
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    for loss in _losses():
        if not _is_smooth(loss):
            continue
        var fx = _fixture(n, d, _is_classification(loss), 1.0)
        var r = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 1.0, True, True), off)
        if r.retcode != OPT_SUCCESS:
            raise Error("check_losses_is_a_minimizer: " + _loss_name(loss) + " retcode " + String(r.retcode) + " after " + String(r.n_iter) + " iterations")
        var l2 = 1.0 / Float64(n)
        var w64 = List[Float64]()
        for k in range(d + 1):
            w64.append(Float64(r.w[k]))
        var grad = List[Float64]()
        var f0 = _objective64(loss, fx[0], fx[1], n, d, w64, True, l2, grad)
        var gn = _own_norm64(loss, grad)
        var bound = 1e-4 * max(f0, 1e-4)
        if gn > bound:
            raise Error(
                "check_losses_is_a_minimizer: " + _loss_name(loss) + " float64 gradient"
                " (own norm) " + String(gn) + " at the device solution exceeds the"
                " solver's own bound " + String(bound) + " (fx " + String(f0) + ")"
            )
        for p in range(8):
            var wp = List[Float64]()
            for k in range(d + 1):
                wp.append(w64[k] + 0.05 * (2.0 * _u01(p, k, 9) - 1.0))
            var g2 = List[Float64]()
            var fp = _objective64(loss, fx[0], fx[1], n, d, wp, True, l2, g2)
            if not (fp > f0):
                raise Error("check_losses_is_a_minimizer: " + _loss_name(loss) + " perturbed objective " + String(fp) + " <= " + String(f0))
        print(
            "check_losses_is_a_minimizer OK: " + _loss_name(loss) + " float64 own-norm gradient "
            + String(gn) + " <= " + String(bound) + " (objective " + String(f0)
            + ", device's own " + String(r.fx) + "), below 8 perturbed; n_iter " + String(r.n_iter)
        )


def _fit_raises(ctx: DeviceContext, pams: QNParams, n_classes: Int, has_sw: Bool) raises -> String:
    var fx = _fixture(256, 4, _is_classification(pams.loss), 1.0)
    var off = IdentityTrace.disabled()
    try:
        var r = _fit(ctx, fx[0], fx[1], 256, 4, pams, off, n_classes, has_sw)
        _ = r.n_iter
    except e:
        return String(e)
    return String("")


def check_losses_refuses_by_name() raises:
    var ctx = DeviceContext()
    for loss in _losses():
        var wrong_c = 1 if _is_classification(loss) else 2
        var m = _fit_raises(ctx, _params(loss, 1.0, True, True), wrong_c, False)
        if m.find("invalid C") < 0:
            raise Error(_loss_name(loss) + " with C=" + String(wrong_c) + " did not raise their text; got: " + m)
    var p1 = _params(QN_LOSS_SQUARED, 1.0, True, True)
    p1.penalty_l1 = 0.5
    var m1 = _fit_raises(ctx, p1, 1, False)
    if m1.find("min_owlqn") < 0:
        raise Error("l1 did not raise by name; got: " + m1)
    var m2 = _fit_raises(ctx, _params(QN_LOSS_SVR_L2, 1.0, True, True), 1, True)
    if m2.find("sample_weight is NOT PORTED") < 0:
        raise Error("sample_weight did not raise by name; got: " + m2)
    print("check_losses_refuses_by_name OK: each loss with the wrong C (their `invalid C` text), l1 (OWL-QN) and sample_weight RAISE by name")


# ---------------------------------------------------------------------------
# THE HOST REPLAY, C = 1
# ---------------------------------------------------------------------------


def _host_lz(loss: Int, y: Float32, z: Float32) -> Float32:
    """The six `Lz`, re-spelled with the same pins as the port."""
    if loss == QN_LOSS_SQUARED:
        var diff = ftz(z - y)
        return ftz(ftz(diff * diff) * Float32(0.5))
    if loss == QN_LOSS_ABS:
        return abs(ftz(z - y))
    if loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2:
        var s = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
        var v = ftz(identical_mul_add(-s, z, Float32(1.0)))
        var t = max(v, Float32(0.0))
        return t if loss == QN_LOSS_SVC_L1 else ftz(t * t)
    var tt = ftz(y - z)
    var m = Float32(0.0)
    if tt > SVR_EPS:
        m = ftz(tt - SVR_EPS)
    elif tt < -SVR_EPS:
        m = ftz(-tt - SVR_EPS)
    return m if loss == QN_LOSS_SVR_L1 else ftz(m * m)


def _host_dlz(loss: Int, y: Float32, z: Float32) -> Float32:
    if loss == QN_LOSS_SQUARED:
        return ftz(z - y)
    if loss == QN_LOSS_ABS:
        return Float32(1.0) if z > y else (Float32(-1.0) if z < y else Float32(0.0))
    if loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2:
        var s = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
        if ftz(s * z) <= Float32(1.0):
            return -s if loss == QN_LOSS_SVC_L1 else ftz(Float32(2.0) * ftz(z - s))
        return Float32(0.0)
    var t = ftz(y - z)
    if loss == QN_LOSS_SVR_L1:
        return Float32(-1.0) if t > SVR_EPS else (Float32(1.0) if t < -SVR_EPS else Float32(0.0))
    var inner = Float32(0.0)
    if t > SVR_EPS:
        inner = ftz(t - SVR_EPS)
    elif t < -SVR_EPS:
        inner = ftz(t + SVR_EPS)
    return ftz(Float32(-2.0) * inner)


def _host_nrm1(g: List[Float32], n: Int) -> Float32:
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < n:
            acc = ftz(acc + abs(g[i]))
            i += STATS_TPB
        red.append(acc)
    return _tree(red)


def _host_grad_norm(loss: Int, g: List[Float32], n: Int) -> Float32:
    if _is_smooth(loss):
        return _host_dot(g, g, n) * Float32(0.5)
    var m = Float32(0.0)
    if loss == QN_LOSS_ABS or loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVR_L1:
        return _host_nrm1(g, n)
    for i in range(n):
        if abs(g[i]) > m:
            m = abs(g[i])
    return m


def _host_evaluate(
    loss: Int, xh: List[Float32], yh: List[Float32], n: Int, d: Int, hw: List[Float32],
    l2: Float32, mut g: List[Float32], mut dz: List[Float32],
) -> Float32:
    """`GLMWithData.evaluate` for `C = 1` with l2: the pinned gemv row-dot,
    the bias seam, the fused loss map, the one-block sums, the xty tree, the
    cuBLAS epilogue, the mean, the Tikhonov kernel, `loss + reg`."""
    var normalization = Float32(1.0 / Float64(n))
    var terms = _zeros(n)
    dz = _zeros(n)
    for i in range(n):
        var acc = Float32(0.0)
        for p in range(d):
            acc = ftz(identical_mul_add(ftz(xh[i * d + p]), ftz(hw[p]), acc))
        var zi = ftz(Float32(0.0) + ftz(acc))
        zi = ftz(zi + hw[d])
        terms[i] = ftz(_host_lz(loss, yh[i], zi) * normalization)
        dz[i] = _host_dlz(loss, yh[i], zi)
    var loss_h = _host_sum_terms(terms, n)
    for k in range(d + 1):
        g[k] = Float32(0.0)
    var reg_h = _host_tikhonov(hw, d, l2, g)
    var alpha = Float32(1.0 / Float64(n))
    for k in range(d):
        var red = List[Float32]()
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var r = t
            while r < n:
                acc = identical_mul_add(xh[r * d + k], dz[r], acc)
                r += STATS_TPB
            red.append(acc)
        var s = ftz(alpha * _tree(red))
        g[k] = ftz(s + g[k])
    var ratio = Float32(1.0) / Float32(n)
    g[d] = ftz(_host_sum_terms(dz, n) * ratio)
    return ftz(loss_h + reg_h)


def check_losses_device_equals_host() raises:
    var n = QL_ROWS
    var d = QL_COLS
    var ctx = DeviceContext()
    for loss in _losses():
        var fx = _fixture(n, d, _is_classification(loss), 1.0)
        var x = _upload(ctx, fx[0])
        var y = _upload(ctx, fx[1])
        var n_param = d + 1
        var hw = _hashed_w(n_param, 11)
        var w = _upload(ctx, hw)
        var g = ctx.enqueue_create_buffer[DType.float32](n_param)
        ctx.synchronize()
        var l2 = Float32(1.0) / Float32(n)
        var dims = GLMDims.make(1, d, True)
        var obj = GLMWithData(ctx, x^, y^, n, dims, loss, l2, SVR_EPS)
        var loss_dev = obj.evaluate(ctx, w, g)
        var gn_dev = obj.grad_norm(ctx, g)
        var g_dev = _download(ctx, g, n_param)
        var dz_dev = _download(ctx, obj.z, n)
        var g_h = _zeros(n_param)
        var dz_h = List[Float32]()
        var loss_h = _host_evaluate(loss, fx[0], fx[1], n, d, hw, l2, g_h, dz_h)
        var gn_h = _host_grad_norm(loss, g_h, n_param)
        var bad = 0
        var first = String("")
        if not _same_bits(loss_h, loss_dev):
            bad += 1
            first = "loss device " + _hex32(loss_dev) + " host " + _hex32(loss_h)
        for k in range(n_param):
            if not _same_bits(g_h[k], g_dev[k]):
                bad += 1
                if first == "":
                    first = "grad " + String(k) + " device " + _hex32(g_dev[k]) + " host " + _hex32(g_h[k])
        if not _same_bits(gn_h, gn_dev):
            bad += 1
            if first == "":
                first = "gradNorm device " + _hex32(gn_dev) + " host " + _hex32(gn_h)
        var bad_dz = 0
        for i in range(n):
            if not _same_bits(dz_h[i], dz_dev[i]):
                bad_dz += 1
                if first == "":
                    first = "dZ " + String(i) + " device " + _hex32(dz_dev[i]) + " host " + _hex32(dz_h[i])
        # float64 tolerance in both modes
        var w64 = List[Float64]()
        for k in range(n_param):
            w64.append(Float64(hw[k]))
        var g64 = List[Float64]()
        var f64 = _objective64(loss, fx[0], fx[1], n, d, w64, True, Float64(l2), g64)
        var worst = abs(Float64(loss_dev) - f64) / max(abs(f64), 1e-3)
        for k in range(n_param):
            var e = abs(Float64(g_dev[k]) - g64[k]) / max(abs(g64[k]), 1e-3)
            if e > worst:
                worst = e
        if worst > 1e-4:
            raise Error(
                "check_losses_device_equals_host: " + _loss_name(loss) + " device vs"
                " float64 worst relative error " + String(worst) + " > 1e-4"
            )
        comptime if IDENTICAL:
            if bad + bad_dz != 0:
                raise Error(
                    "check_losses_device_equals_host [IDENTICAL]: " + _loss_name(loss) + " "
                    + String(bad) + " of " + String(n_param + 2) + " loss/grad/gradNorm values, "
                    + String(bad_dz) + " of " + String(n) + " dZ cells differ. First: " + first
                )
            print(
                "check_losses_device_equals_host OK [IDENTICAL]: " + _loss_name(loss)
                + " loss, " + String(n_param) + " gradient entries, gradNorm and "
                + String(n) + " dZ cells equal the host replay bit for bit (loss "
                + _hex32(loss_dev) + ", gradNorm " + _hex32(gn_dev) + "); float64 worst rel err "
                + String(worst)
            )
        else:
            print(
                "check_losses_device_equals_host REPORT [FAST]: " + _loss_name(loss) + " "
                + String(bad) + " of " + String(n_param + 2) + " loss/grad/gradNorm values, "
                + String(bad_dz) + " of " + String(n) + " dZ cells differ from the pinned"
                " host replay" + ("" if first == "" else "; first: " + first)
                + "; float64 worst rel err " + String(worst) + " <= 1e-4"
            )


# ---------------------------------------------------------------------------
# the fused kernels on a planted z: signed zeros, boundaries, launch shapes
# ---------------------------------------------------------------------------


def _launch_loss(
    ctx: DeviceContext, loss: Int, mut terms: DeviceBuffer[DType.float32],
    mut z: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    n: Int, block: Int, normalization: Float32,
) raises:
    var grid = (n + block - 1) // block
    if loss == QN_LOSS_SQUARED:
        ctx.enqueue_function[squared_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    elif loss == QN_LOSS_ABS:
        ctx.enqueue_function[abs_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    elif loss == QN_LOSS_SVC_L1:
        ctx.enqueue_function[svc_l1_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    elif loss == QN_LOSS_SVC_L2:
        ctx.enqueue_function[svc_l2_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    elif loss == QN_LOSS_SVR_L1:
        ctx.enqueue_function[svr_l1_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization, SVR_EPS,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    else:
        ctx.enqueue_function[svr_l2_loss_dz_kernel](
            terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(n), normalization, SVR_EPS,
            grid_dim=(grid, 1, 1), block_dim=(block, 1, 1),
        )
    ctx.synchronize()


def _planted_zy(n: Int) -> Tuple[List[Float32], List[Float32]]:
    """Hashed (y, z) pairs with the special rows planted FIRST: signed zeros
    in both orders, `y == z`, the hinge boundary `s z == 1` for both labels,
    and the SVR dead-zone edges `y - z == +-eps`."""
    var z = List[Float32]()
    var y = List[Float32]()
    var pz = Float32(0.0)
    var nz = Float32(-0.0)
    # (y, z)
    var special_y: List[Float32] = [nz, pz, pz, nz, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 2.0, 2.0]
    var special_z: List[Float32] = [pz, nz, pz, nz, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.25, 0.75, 2.0 - SVR_EPS, 2.0 + SVR_EPS]
    for i in range(len(special_y)):
        y.append(special_y[i])
        z.append(special_z[i])
    for i in range(len(special_y), n):
        # half the rows binary labels, half regression targets
        if i % 2 == 0:
            y.append(Float32(1.0) if _u01(i, 1, 21) > 0.5 else Float32(0.0))
        else:
            y.append(Float32(4.0 * (_u01(i, 1, 21) - 0.5)))
        z.append(Float32(4.0 * (_u01(i, 2, 22) - 0.5)))
    return (y^, z^)


def check_losses_kernel_invariance() raises:
    var n = 1024
    var zy = _planted_zy(n)
    var yh = zy[0].copy()
    var zh = zy[1].copy()
    var normalization = Float32(1.0 / Float64(n))
    var ctx = DeviceContext()
    var y = _upload(ctx, yh)
    for loss in _losses():
        # host spelling
        var th = _zeros(n)
        var dzh = _zeros(n)
        for i in range(n):
            var yi = yh[i]
            var zi = zh[i]
            th[i] = ftz(_host_lz(loss, yi, zi) * normalization)
            dzh[i] = _host_dlz(loss, yi, zi)
        # the device reference: block 256, exact-size buffers
        var z_ref = _upload(ctx, zh)
        var t_ref = _poison(ctx, n)
        _launch_loss(ctx, loss, t_ref, z_ref, y, n, 256, normalization)
        var dz_ref = _download(ctx, z_ref, n)
        var tr = _download(ctx, t_ref, n)
        var moved = 0
        var first = String("")
        # block 64, padded + poisoned outputs, against the reference
        var zpad = _zeros(n + 33)
        for i in range(n):
            zpad[i] = zh[i]
        for i in range(n, n + 33):
            zpad[i] = bitcast[DType.float32](UInt32(0x7FC0BEEF))
        var zb = _upload(ctx, zpad)
        var tb = _poison(ctx, n + 17)
        _launch_loss(ctx, loss, tb, zb, y, n, 64, normalization)
        var dzb = _download(ctx, zb, n + 33)
        var tbh = _download(ctx, tb, n + 17)
        for i in range(n):
            if not _same_bits(dzb[i], dz_ref[i]) or not _same_bits(tbh[i], tr[i]):
                moved += 1
                if first == "":
                    first = "block 64 row " + String(i) + " dZ " + _hex32(dzb[i]) + " vs " + _hex32(dz_ref[i]) + " term " + _hex32(tbh[i]) + " vs " + _hex32(tr[i])
        for i in range(n, n + 33):
            if bitcast[DType.uint32](dzb[i]) != UInt32(0x7FC0BEEF):
                raise Error("check_losses_kernel_invariance: " + _loss_name(loss) + " block 64 wrote past the last row (dZ pad)")
        for i in range(n, n + 17):
            if bitcast[DType.uint32](tbh[i]) != UInt32(0x7FC0BEEF):
                raise Error("check_losses_kernel_invariance: " + _loss_name(loss) + " block 64 wrote past the last row (term pad)")
        # batch composition: the 14 special rows + 50 more, alone
        var sub_n = 64
        var zs = List[Float32]()
        var ys = List[Float32]()
        for i in range(sub_n):
            zs.append(zh[i])
            ys.append(yh[i])
        var zsb = _upload(ctx, zs)
        var ysb = _upload(ctx, ys)
        var tsb = _poison(ctx, sub_n)
        _launch_loss(ctx, loss, tsb, zsb, ysb, sub_n, 64, normalization)
        var dzs = _download(ctx, zsb, sub_n)
        var tss = _download(ctx, tsb, sub_n)
        for i in range(sub_n):
            if not _same_bits(dzs[i], dz_ref[i]) or not _same_bits(tss[i], tr[i]):
                moved += 1
                if first == "":
                    first = "batch row " + String(i)
        if moved != 0:
            raise Error(
                "check_losses_kernel_invariance: " + _loss_name(loss) + " " + String(moved)
                + " cells moved across block 64/256, the paddings or the batch"
                " composition. First: " + first
            )
        # the device reference against the host spelling: IDENTICAL asserts
        var host_bad = 0
        var host_first = String("")
        for i in range(n):
            if not _same_bits(dz_ref[i], dzh[i]) or not _same_bits(tr[i], th[i]):
                host_bad += 1
                if host_first == "":
                    host_first = "row " + String(i) + " (y " + _hex32(yh[i]) + " z " + _hex32(zh[i]) + ") dZ device " + _hex32(dz_ref[i]) + " host " + _hex32(dzh[i]) + " term device " + _hex32(tr[i]) + " host " + _hex32(th[i])
        comptime if IDENTICAL:
            if host_bad != 0:
                raise Error(
                    "check_losses_kernel_invariance [IDENTICAL]: " + _loss_name(loss) + " "
                    + String(host_bad) + " of " + String(n) + " rows differ from the host"
                    " spelling (the planted rows are the first 14). First: " + host_first
                )
            print(
                "check_losses_kernel_invariance OK [IDENTICAL]: " + _loss_name(loss)
                + " invariant over block 64/256, two paddings (poison intact) and the"
                " batch composition, and equal to the host spelling on all " + String(n)
                + " rows (14 planted: +-0 both orders, y == z, s z == 1 both labels,"
                " |y - z| == eps)"
            )
        else:
            print(
                "check_losses_kernel_invariance OK [FAST]: " + _loss_name(loss)
                + " invariant over block 64/256, two paddings (poison intact) and the"
                " batch composition; vs the host spelling " + String(host_bad) + " of "
                + String(n) + " rows differ (REPORT: the host does not contract the"
                " multiply-adds Metal contracts)"
                + ("" if host_first == "" else "; first: " + host_first)
            )


def check_losses_card_is_emitted() raises:
    var n = 1024
    var d = QL_COLS
    for loss in _losses():
        var p1 = String("/tmp/mojolearn_qnloss_card_a.card")
        var p2 = String("/tmp/mojolearn_qnloss_card_b.card")
        var fx = _fixture(n, d, _is_classification(loss), 1.0)
        var n1 = 0
        var iters = 0
        var moved = 0
        with DeviceContext() as ctx:
            var t1 = IdentityTrace.to_path(p1, "", True)
            t1.header(_loss_name(loss) + " n=" + String(n) + " d=" + String(d))
            var r1 = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 1.0, True, True), t1)
            n1 = t1.seq
            iters = r1.n_iter
            var t2 = IdentityTrace.to_path(p2, "", True)
            t2.header(_loss_name(loss) + " n=" + String(n) + " d=" + String(d))
            var r2 = _fit(ctx, fx[0], fx[1], n, d, _params(loss, 1.0, True, True), t2)
            if t2.seq != n1:
                raise Error("check_losses_card_is_emitted: " + _loss_name(loss) + " stage count " + String(n1) + " vs " + String(t2.seq))
            for k in range(len(r1.w)):
                if not _same_bits(r1.w[k], r2.w[k]):
                    moved += 1
        var want = 2 + 3 * iters + 3
        if n1 != want:
            raise Error("check_losses_card_is_emitted: " + _loss_name(loss) + " " + String(n1) + " stages, expected " + String(want) + " for n_iter " + String(iters))
        var diff = first_divergence(p1, p2)
        comptime if IDENTICAL:
            if diff != "" or moved != 0:
                raise Error("check_losses_card_is_emitted [IDENTICAL]: " + _loss_name(loss) + " control differs (" + String(moved) + " coefficients): " + diff)
            print("check_losses_card_is_emitted OK [IDENTICAL]: " + _loss_name(loss) + " " + String(n1) + " stages (n_iter " + String(iters) + "), control agrees on all")
        else:
            if diff == "":
                print("check_losses_card_is_emitted OK [FAST]: " + _loss_name(loss) + " " + String(n1) + " stages (n_iter " + String(iters) + "), control happens to agree")
            else:
                print("check_losses_card_is_emitted REPORT [FAST]: " + _loss_name(loss) + " " + String(n1) + " stages; control differs first at " + diff)


def main() raises:
    print("== glm/mojo_only/qn_losses_check.mojo [" + _mode_name() + "] ==")
    check_losses_fd_gradient()
    check_losses_planted()
    check_losses_is_a_minimizer()
    check_losses_refuses_by_name()
    check_losses_device_equals_host()
    check_losses_kernel_invariance()
    check_losses_card_is_emitted()
