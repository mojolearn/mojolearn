# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Logistic regression (cuML `qnFit`, QN_LOSS_LOGISTIC, L-BFGS): oracle, reach, identity.

DEVIATIONS 546-549. See `glm/README.md`. The checks:

    check_logistic_planted_separable   a planted-sign fixture: every fitted
                                       weight carries the planted SIGN and
                                       the training accuracy is > 0.97
    check_logistic_is_a_minimizer      the L-infinity gradient norm at the
                                       solution is below the solver's own
                                       tolerance (gnorm <= tol * max(fx,
                                       tol), the `check_convergence` rule),
                                       and the objective at the solution is
                                       below the objective at 8 perturbed
                                       points -- an oracle that needs no
                                       scikit-learn, only the objective
                                       (which is replayed on the HOST in
                                       Float64, a DIFFERENT program from the
                                       device's Float32 one)
    check_logistic_c_reaches           C = 1 vs C = 100 vs no penalty: the
                                       weight norm grows with C and the bits
                                       move; fit_intercept False zeroes the
                                       bias slot (it is not there)
    check_logistic_refuses_by_name     l1, softmax (3 classes), sample_weight,
                                       an unknown loss: each RAISES with the
                                       message naming the unported thing
    check_logistic_device_equals_host  ONE objective evaluation (loss +
                                       gradient) at a hashed `w`, replayed on
                                       the host through the same pinned
                                       shapes: IDENTICAL bit for bit on the
                                       loss and every gradient entry, FAST a
                                       report
    check_logistic_run_twice_identical two fits: IDENTICAL asserts the
                                       coefficients AND the iteration count
                                       agree byte for byte; FAST reports
    check_logistic_card_is_emitted     the certificate: stage count is a
                                       function of the fixture (n_iter), and
                                       the run-to-run control

SABOTAGES PERFORMED (2026-08-23), each reverted:

    (a) `logistic_dlz` returning `y - q` instead of `q - y` (the gradient
        SIGN): `check_logistic_planted_separable` fails "weight 0 =
        -1.8185203e-07 has sign -1, planted +1 (n_iter 10, retcode 0)" --
        with the gradient pointing uphill the first step fails the Armijo
        test at every width, the objective never changes, and
        `check_convergence`'s insufficient-change test declares SUCCESS at
        the zero model after 10 iterations. A wrong gradient converges
        cleanly; only the oracle sees it.
    (b) `sum_terms_kernel` folding SEQUENTIALLY on thread 0 instead of the
        pinned tree: under IDENTICAL `check_logistic_device_equals_host`
        fails "1 of 8 values differ. First: loss device 0x3f4aa44b host
        0x3f4aa44d" -- two ulps on the loss, the gradient still agreeing
        (it does not go through that kernel), which is the evidence this
        check separates the two folds and that the fold shape reaches the
        number the line search compares.
    (c) `gemm_epilogue_kernel` ignoring `beta_is_one` (the l2 gradient is
        dropped, the loss gradient overwrites it): `planted_separable`
        still passes (signs survive), `check_logistic_is_a_minimizer`
        fails "host float64 gradient Linf 0.00067819 at the device
        solution exceeds the solver's own bound 1.94e-05" -- the solver
        converged to the minimizer of the WRONG objective, and again only
        the oracle sees it.
"""

from std.math import exp, log, sqrt
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from core.identity_trace import IdentityTrace, first_divergence
from glm.impl.glm.qn.glm_base import GLMDims, GLMWithData
from glm.impl.glm.qn.qn import qn_decision_function, qn_fit_x
from glm.impl.glm.qn.qn_util import OPT_MAX_ITERS_REACHED, OPT_SUCCESS
from glm.impl.linear_model.qn import (
    QN_LOSS_LOGISTIC,
    QN_LOSS_SOFTMAX,
    QNParams,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime LR_ROWS = 4096
comptime LR_COLS = 6


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _planted_w(k: Int) -> Float64:
    """Mixed signs, none near zero: +2, -1.5, +1, -2.5, +1.5, -1, ..."""
    var mag = 2.0 - 0.5 * Float64(k % 4)
    return mag if k % 2 == 0 else -mag


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _fixture(n: Int, d: Int, margin: Float64) -> Tuple[List[Float32], List[Float32]]:
    """Hashed design in [-1, 1); label = 1 if x . w* + bias + noise > 0.
    `margin` scales the noise: 0 = separable, 1 = ~10% flipped."""
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n):
        var s = 0.3  # a planted positive bias
        for k in range(d):
            var v = 2.0 * _u01(i, k, 0) - 1.0
            x.append(Float32(v))
            s += v * _planted_w(k)
        s += margin * 2.0 * (_u01(i, 77, 5) - 0.5)
        y.append(Float32(1.0) if s > 0.0 else Float32(0.0))
    return (x^, y^)


def _params(C: Float64, fit_intercept: Bool, penalty: Bool, max_iter: Int = 1000) -> QNParams:
    """cuML's Python door for `LogisticRegression(penalty='l2', C, tol=1e-4,
    max_iter, linesearch_max_iter=50)`: `l2 = 1/C`, `grad_tol = tol`,
    `change_tol = tol * 0.01`, `lbfgs_memory = 5`, `penalty_normalized`."""
    var p = QNParams.default()
    p.loss = QN_LOSS_LOGISTIC
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
    ctx: DeviceContext,
    xh: List[Float32],
    yh: List[Float32],
    n: Int,
    d: Int,
    pams: QNParams,
    mut trace: IdentityTrace,
    n_classes: Int = 2,
    has_sw: Bool = False,
) raises -> FitResult:
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var n_param = d + (1 if pams.fit_intercept else 0)
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    var hx = xh.copy()
    var hy = yh.copy()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_memset(w, Float32(0.0))
    ctx.synchronize()
    var fx = Float32(0.0)
    var iters = 0
    var ret = qn_fit_x(ctx, pams, x^, y^, n, d, n_classes, w, fx, iters, has_sw, trace)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n_param):
        out.append(hw.unsafe_ptr().unsafe_load(i))
    _ = hx^
    _ = hy^
    _ = hw^
    return FitResult(out^, fx, iters, ret)


def _host_objective64(
    xh: List[Float32], yh: List[Float32], n: Int, d: Int, w: List[Float64],
    fit_intercept: Bool, l2: Float64, mut grad: List[Float64],
) -> Float64:
    """The objective cuML minimizes, in FLOAT64 on the host -- a different
    program from the device's, the ORACLE: `mean_i logloss_i + l2/2 ||w||^2`,
    with its gradient."""
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
        # -log sigmoid((2y-1) z), stable
        var t = (2.0 * yi - 1.0) * z
        var loss = 0.0
        if t < 0.0:
            loss = -t + log(1.0 + exp(t))
        else:
            loss = log(1.0 + exp(-t))
        total += loss
        var p = 1.0 / (1.0 + exp(-z))
        var dz = p - yi
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


def check_logistic_planted_separable() raises:
    var n = LR_ROWS
    var d = LR_COLS
    var fx = _fixture(n, d, 0.0)
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    var r = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, True), off)
    for k in range(d):
        var planted = _planted_w(k) > 0.0
        var got = r.w[k] > Float32(0.0)
        if planted != got:
            raise Error(
                "check_logistic_planted_separable: weight " + String(k)
                + " = " + String(r.w[k]) + " has sign " + ("+1" if got else "-1")
                + ", planted " + ("+1" if planted else "-1")
                + " (n_iter " + String(r.n_iter) + ", retcode " + String(r.retcode) + ")"
            )
    if not (r.w[d] > Float32(0.0)):
        raise Error("check_logistic_planted_separable: the bias " + String(r.w[d]) + " lost its planted + sign")
    # training accuracy
    var correct = 0
    for i in range(n):
        var z = Float64(r.w[d])
        for k in range(d):
            z += Float64(fx[0][i * d + k]) * Float64(r.w[k])
        var pred = Float32(1.0) if z > 0.0 else Float32(0.0)
        if pred == fx[1][i]:
            correct += 1
    var acc = Float64(correct) / Float64(n)
    if acc < 0.97:
        raise Error("check_logistic_planted_separable: training accuracy " + String(acc) + " < 0.97")
    print(
        "check_logistic_planted_separable OK: all " + String(d) + " weights and"
        " the bias carry the planted sign; training accuracy " + String(acc)
        + "; n_iter " + String(r.n_iter) + ", retcode " + String(r.retcode)
        + ", objective " + String(r.fx)
    )


def check_logistic_is_a_minimizer() raises:
    var n = LR_ROWS
    var d = LR_COLS
    var fx = _fixture(n, d, 1.0)
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    var pams = _params(1.0, True, True)
    var r = _fit(ctx, fx[0], fx[1], n, d, pams, off)
    if r.retcode != OPT_SUCCESS:
        raise Error("check_logistic_is_a_minimizer: retcode " + String(r.retcode) + " after " + String(r.n_iter) + " iterations")
    var l2 = (1.0 / 1.0) / Float64(n)
    var w64 = List[Float64]()
    for k in range(d + 1):
        w64.append(Float64(r.w[k]))
    var grad = List[Float64]()
    var f0 = _host_objective64(fx[0], fx[1], n, d, w64, True, l2, grad)
    var gmax = 0.0
    for k in range(d + 1):
        if abs(grad[k]) > gmax:
            gmax = abs(grad[k])
    # `check_convergence`'s own rule: gnorm <= epsilon * max(fx, epsilon)
    var bound = 1e-4 * max(f0, 1e-4)
    if gmax > bound:
        raise Error(
            "check_logistic_is_a_minimizer: host float64 gradient Linf "
            + String(gmax) + " at the device solution exceeds the solver's"
            " own bound " + String(bound) + " (fx " + String(f0) + ")"
        )
    # below the objective at 8 perturbed points
    for p in range(8):
        var wp = List[Float64]()
        for k in range(d + 1):
            wp.append(w64[k] + 0.05 * (2.0 * _u01(p, k, 9) - 1.0))
        var g2 = List[Float64]()
        var fp = _host_objective64(fx[0], fx[1], n, d, wp, True, l2, g2)
        if not (fp > f0):
            raise Error("check_logistic_is_a_minimizer: perturbed objective " + String(fp) + " <= " + String(f0))
    print(
        "check_logistic_is_a_minimizer OK: float64 gradient Linf " + String(gmax)
        + " <= " + String(bound) + " at the device solution (objective "
        + String(f0) + ", device's own " + String(r.fx) + "), below 8 perturbed"
        " objectives; n_iter " + String(r.n_iter)
    )


def _norm(w: List[Float32], d: Int) -> Float64:
    var s = 0.0
    for k in range(d):
        s += Float64(w[k]) * Float64(w[k])
    return sqrt(s)


def check_logistic_c_reaches() raises:
    var n = LR_ROWS
    var d = LR_COLS
    var fx = _fixture(n, d, 1.0)
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    var r1 = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, True), off)
    var r100 = _fit(ctx, fx[0], fx[1], n, d, _params(100.0, True, True), off)
    var rno = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, False), off)
    var rnb = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, False, True), off)
    var n1 = _norm(r1.w, d)
    var n100 = _norm(r100.w, d)
    var nno = _norm(rno.w, d)
    if not (n1 < n100 and n100 <= nno * 1.0001):
        raise Error(
            "check_logistic_c_reaches: ||w|| " + String(n1) + " (C=1), "
            + String(n100) + " (C=100), " + String(nno) + " (no penalty): not increasing"
        )
    var same = 0
    for k in range(d):
        if bitcast[DType.uint32](r1.w[k]) == bitcast[DType.uint32](r100.w[k]):
            same += 1
    if same == d:
        raise Error("check_logistic_c_reaches: C=1 and C=100 gave identical bits")
    if len(rnb.w) != d:
        raise Error("check_logistic_c_reaches: fit_intercept=False returned " + String(len(rnb.w)) + " parameters")
    print(
        "check_logistic_c_reaches OK: ||w|| " + String(n1) + " < " + String(n100)
        + " <= " + String(nno) + " at C = 1 / 100 / none; fit_intercept=False"
        " has " + String(d) + " parameters; n_iter " + String(r1.n_iter) + "/"
        + String(r100.n_iter) + "/" + String(rno.n_iter) + "/" + String(rnb.n_iter)
    )


def _fit_raises(ctx: DeviceContext, pams: QNParams, n_classes: Int, has_sw: Bool) raises -> String:
    var fx = _fixture(256, 4, 1.0)
    var off = IdentityTrace.disabled()
    try:
        var r = _fit(ctx, fx[0], fx[1], 256, 4, pams, off, n_classes, has_sw)
        _ = r.n_iter
    except e:
        return String(e)
    return String("")


def check_logistic_refuses_by_name() raises:
    var ctx = DeviceContext()
    var p1 = _params(1.0, True, True)
    p1.penalty_l1 = 0.5
    var m1 = _fit_raises(ctx, p1, 2, False)
    if m1.find("min_owlqn") < 0:
        raise Error("l1 did not raise by name; got: " + m1)
    var p2 = _params(1.0, True, True)
    p2.loss = QN_LOSS_SOFTMAX
    var m2 = _fit_raises(ctx, p2, 3, False)
    if m2.find("QN_LOSS_SOFTMAX") < 0:
        raise Error("softmax did not raise by name; got: " + m2)
    var m3 = _fit_raises(ctx, _params(1.0, True, True), 3, False)
    if m3.find("logistic loss invalid C") < 0:
        raise Error("3 classes under logistic did not raise their text; got: " + m3)
    var m4 = _fit_raises(ctx, _params(1.0, True, True), 2, True)
    if m4.find("sample_weight is NOT PORTED") < 0:
        raise Error("sample_weight did not raise by name; got: " + m4)
    var p5 = _params(1.0, True, True)
    p5.loss = 42
    var m5 = _fit_raises(ctx, p5, 2, False)
    if m5.find("unknown loss function type") < 0:
        raise Error("loss 42 did not raise their text; got: " + m5)
    print(
        "check_logistic_refuses_by_name OK: l1 (OWL-QN), softmax, 3 classes,"
        " sample_weight and an unknown loss each RAISE by name"
    )


# ---------------------------------------------------------------------------
# the host replay of ONE objective evaluation, pinned shapes
# ---------------------------------------------------------------------------


def _host_halving_sum(terms: List[Float32], n: Int) -> Float32:
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < n:
            acc = ftz(acc + terms[i])
            i += STATS_TPB
        red.append(acc)
    var step = STATS_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def _host_halving_xty(x: List[Float32], y: List[Float32], n: Int, d: Int, col: Int) -> Float32:
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var r = t
        while r < n:
            acc = identical_mul_add(x[r * d + col], y[r], acc)
            r += STATS_TPB
        red.append(acc)
    var step = STATS_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def _host_lz(y: Float32, z: Float32) -> Float32:
    var ytil = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
    var x = ftz(ytil * z)
    var e = ftz(identical_exp(x if x < Float32(0.0) else -x))
    var temp = ftz(identical_log(ftz(Float32(1.0) + e)))
    var ls = ftz(x - temp) if x < Float32(0.0) else -temp
    return -ls


def _host_dlz(y: Float32, z: Float32) -> Float32:
    var ez = ftz(identical_exp(z if z < Float32(0.0) else -z))
    var numerator = ez if z < Float32(0.0) else Float32(1.0)
    var q = ftz(numerator / ftz(Float32(1.0) + ez))
    return ftz(q - y)


def check_logistic_device_equals_host() raises:
    """One `GLMWithData.evaluate` at a hashed `w`, against the host replay
    of every pinned shape it runs: the gemv row-dot (ascending, fma), the
    bias add, the fused loss/dz map, the one-block halving sum, the xty
    halving tree, the cuBLAS epilogue, the mean, the Tikhonov kernel, and
    the host `loss + reg`."""
    var n = 2048
    var d = LR_COLS
    var fx = _fixture(n, d, 1.0)
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var hx = fx[0].copy()
    var hy = fx[1].copy()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    var n_param = d + 1
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    var g = ctx.enqueue_create_buffer[DType.float32](n_param)
    var hw = List[Float32]()
    for k in range(n_param):
        hw.append(Float32(2.0 * _u01(k, 3, 11) - 1.0))
    var hw2 = hw.copy()
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw2.unsafe_ptr())
    ctx.synchronize()
    var l2 = Float32(1.0) / Float32(n)  # C = 1, normalized
    var dims = GLMDims.make(1, d, True)
    var obj = GLMWithData(ctx, x^, y^, n, dims, QN_LOSS_LOGISTIC, l2)
    var loss_dev = obj.evaluate(ctx, w, g)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](n_param)
    ctx.enqueue_copy(dst_ptr=hg.unsafe_ptr(), src_buf=g)
    ctx.synchronize()
    var g_dev = List[Float32]()
    for k in range(n_param):
        g_dev.append(hg.unsafe_ptr().unsafe_load(k))
    # HOST
    var normalization = Float32(1.0 / Float64(n))
    var z = List[Float32]()
    var terms = List[Float32]()
    for i in range(n):
        var acc = Float32(0.0)
        for p in range(d):
            acc = ftz(identical_mul_add(ftz(fx[0][i * d + p]), ftz(hw[p]), acc))
        var zi = ftz(Float32(0.0) + ftz(acc))
        zi = ftz(zi + hw[d])
        terms.append(ftz(_host_lz(fx[1][i], zi) * normalization))
        z.append(_host_dlz(fx[1][i], zi))
    var loss_h = _host_halving_sum(terms, n)
    # reg
    var half_l2 = ftz(Float32(0.5) * l2)
    var reg_terms = List[Float32]()
    var g_h = List[Float32]()
    for k in range(d):
        g_h.append(ftz(l2 * hw[k]))
        var t = ftz(half_l2 * hw[k])
        reg_terms.append(ftz(t * hw[k]))
    g_h.append(Float32(0.0))
    # the Tikhonov kernel strides with STATS_TPB and sums per thread then folds
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var j = t
        while j < d:
            acc = ftz(acc + reg_terms[j])
            j += STATS_TPB
        red.append(acc)
    var step = STATS_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    var reg_h = ftz(red[0])
    var alpha = Float32(1.0 / Float64(n))
    for k in range(d):
        var s = ftz(alpha * _host_halving_xty(fx[0], z, n, d, k))
        g_h[k] = ftz(s + g_h[k])
    var ratio = Float32(1.0) / Float32(n)
    g_h[d] = ftz(_host_halving_sum(z, n) * ratio)
    var fx_h = ftz(loss_h + reg_h)
    var bad = 0
    var first = String("")
    if bitcast[DType.uint32](fx_h) != bitcast[DType.uint32](loss_dev):
        bad += 1
        first = "loss device " + _hex32(loss_dev) + " host " + _hex32(fx_h)
    for k in range(n_param):
        if bitcast[DType.uint32](g_h[k]) != bitcast[DType.uint32](g_dev[k]):
            bad += 1
            if first == "":
                first = "grad " + String(k) + " device " + _hex32(g_dev[k]) + " host " + _hex32(g_h[k])
    _ = hx^
    _ = hy^
    _ = hw2^
    _ = hg^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    comptime if IDENTICAL:
        if bad != 0:
            raise Error(
                "check_logistic_device_equals_host [IDENTICAL]: " + String(bad)
                + " of " + String(n_param + 1) + " values differ. First: " + first
            )
        print(
            "check_logistic_device_equals_host OK [IDENTICAL]: loss and all "
            + String(n_param) + " gradient entries equal the host replay bit"
            " for bit (loss " + _hex32(loss_dev) + ")"
        )
    else:
        print(
            "check_logistic_device_equals_host REPORT [FAST]: " + String(bad)
            + " of " + String(n_param + 1) + " values differ from the pinned"
            " host replay (vendor gemv, block.sum, device exp/log)"
            + ("" if first == "" else "; first: " + first)
        )


def check_logistic_run_twice_identical() raises:
    var n = LR_ROWS
    var d = LR_COLS
    var fx = _fixture(n, d, 1.0)
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    var r1 = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, True), off)
    var r2 = _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, True), off)
    var moved = 0
    for k in range(d + 1):
        if bitcast[DType.uint32](r1.w[k]) != bitcast[DType.uint32](r2.w[k]):
            moved += 1
    var same_iter = r1.n_iter == r2.n_iter
    comptime if IDENTICAL:
        if moved != 0 or not same_iter:
            raise Error(
                "check_logistic_run_twice_identical [IDENTICAL]: " + String(moved)
                + " parameters moved, n_iter " + String(r1.n_iter) + " vs " + String(r2.n_iter)
            )
        print("check_logistic_run_twice_identical OK [IDENTICAL]: " + String(d + 1) + " parameters and n_iter " + String(r1.n_iter) + " byte-identical across two fits")
    else:
        print(
            "check_logistic_run_twice_identical REPORT [FAST]: " + String(moved)
            + " of " + String(d + 1) + " parameters moved; n_iter " + String(r1.n_iter)
            + " vs " + String(r2.n_iter)
        )


def emit_logistic_card(ctx: DeviceContext, mut trace: IdentityTrace, n: Int, d: Int) raises -> FitResult:
    var fx = _fixture(n, d, 1.0)
    trace.header(String("logistic n=") + String(n) + " d=" + String(d))
    return _fit(ctx, fx[0], fx[1], n, d, _params(1.0, True, True), trace)


def check_logistic_card_is_emitted() raises:
    var p1 = String("/tmp/mojolearn_logistic_card_check_a.card")
    var p2 = String("/tmp/mojolearn_logistic_card_check_b.card")
    var n1 = 0
    var iters = 0
    with DeviceContext() as ctx:
        var t1 = IdentityTrace.to_path(p1, "", True)
        var r1 = emit_logistic_card(ctx, t1, 2048, LR_COLS)
        n1 = t1.seq
        iters = r1.n_iter
        var t2 = IdentityTrace.to_path(p2, "", True)
        var r2 = emit_logistic_card(ctx, t2, 2048, LR_COLS)
        if t2.seq != n1:
            raise Error("check_logistic_card_is_emitted: stage count " + String(n1) + " vs " + String(t2.seq))
        _ = r2.n_iter
    # 2 init + 3 per iteration + coef + n_iter + retcode
    var want = 2 + 3 * iters + 3
    if n1 != want:
        raise Error("check_logistic_card_is_emitted: " + String(n1) + " stages, expected " + String(want) + " for n_iter " + String(iters))
    var diff = first_divergence(p1, p2)
    comptime if IDENTICAL:
        if diff != "":
            raise Error("check_logistic_card_is_emitted [IDENTICAL]: control differs: " + diff)
        print("check_logistic_card_is_emitted OK [IDENTICAL]: " + String(n1) + " stages (n_iter " + String(iters) + "), control agrees on all")
    else:
        if diff == "":
            print("check_logistic_card_is_emitted OK [FAST]: " + String(n1) + " stages (n_iter " + String(iters) + "), control happens to agree")
        else:
            print("check_logistic_card_is_emitted REPORT [FAST]: " + String(n1) + " stages; control differs first at " + diff)


def main() raises:
    print("== glm/checks/logistic_check.mojo [" + _mode_name() + "] ==")
    check_logistic_planted_separable()
    check_logistic_is_a_minimizer()
    check_logistic_c_reaches()
    check_logistic_refuses_by_name()
    check_logistic_device_equals_host()
    check_logistic_run_twice_identical()
    check_logistic_card_is_emitted()
