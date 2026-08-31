# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Multinomial logistic regression (cuML `qnFit`, QN_LOSS_SOFTMAX, L-BFGS):
oracle, reach, identity.

DEVIATIONS 705-706 (the port, `glm/derived/glm/qn/glm_softmax.mojo`) and
709-711 (the gates here). See `glm/README.md`, "QN losses". The checks:

    check_softmax_fd_gradient           the FLOAT64 host objective's analytic
                                        gradient against central finite
                                        differences at a hashed `w` (3 and 5
                                        classes): the oracle is checked before
                                        anything is checked against it
    check_softmax_planted               3-class and 5-class planted fixtures:
                                        training accuracy on the planted
                                        labels > 0.95, the objective falls
                                        below `log C` (the value at `w = 0`),
                                        and it is MONOTONE in the iteration
                                        budget (max_iter 1 >= 3 >= full)
    check_softmax_is_a_minimizer        float64 gradient Linf at the device
                                        solution <= tol * max(fx, tol) (the
                                        solver's own `check_convergence`
                                        rule), objective below 8 perturbed
                                        points -- no scikit-learn needed
    check_softmax_refuses_by_name       softmax with 2 classes, logistic
                                        with 3, a too-small `w0`, l1,
                                        sample_weight: each RAISES naming
                                        the thing
    check_softmax_device_equals_host    ONE objective evaluation (loss,
                                        every gradient entry, every dZ
                                        cell) and the decision function at a
                                        hashed `w`, against the host replay
                                        of every pinned shape: bit for bit
                                        under IDENTICAL, a REPORT under FAST;
                                        and the device gradient against the
                                        float64 gradient within tolerance
                                        in BOTH modes
    check_softmax_signed_zero_selection the max SELECTION on rows holding
                                        +0.0 and -0.0 in both orders and an
                                        exact tie: the positional rule's
                                        answer, by bits, and the whole
                                        kernel on those rows against the
                                        host
    check_softmax_launch_invariance     the softmax kernel at block sizes
                                        32 / 64 / 256, two output paddings
                                        with poison, and a batch composition
                                        (64 rows alone vs the same rows
                                        inside 2048): every cell's bytes
                                        equal; the pinned forward product
                                        at two block sizes (IDENTICAL;
                                        RECORDED under FAST, vendor matmul)
    check_softmax_host_lbfgs_replay     THE ITERATION ORACLE: the L-BFGS
                                        loop, the line search and the
                                        objective replayed on the HOST
                                        through the same pinned shapes,
                                        writing a card with the SAME tags;
                                        the device card and the host card
                                        must agree at EVERY stage under
                                        IDENTICAL (`first_divergence` ==
                                        ""), FAST a report
    check_softmax_card_is_emitted       two traced device fits: stage count
                                        `2 + 3 n_iter + 3`, run-to-run
                                        control

SABOTAGES PERFORMED (2026-08-23), each reverted; the README's table has
the outputs:

    (a) `softmax_row_max` folding with the hardware `max(v, eta_max)`
        instead of the strict `>`: REQUIRED, and Apple-INERT as predicted
        (Metal returns the second operand on `max(+0, -0)`, which is the
        running value, i.e. the lower index -- the same answer the rule
        gives; NVIDIA/AMD's IEEE maximum would return `+0.0` on the
        `[-0.0, +0.0]` row and differ). `check_softmax_signed_zero_selection`
        passes under the sabotage on this device and that is exactly why
        the rule is a rule and not a measurement.
    (b) `std.math.exp` in phase 2 and 3 instead of `identical_exp`:
        `check_softmax_device_equals_host [IDENTICAL]` fails on the loss
        and on dZ cells.
    (c) the phase-2 sum starting at class `i % C` and wrapping (a rotated
        start, the block-index rotation of a serial fold): the same check
        fails on dZ/loss cells where the three-term float sum is order
        sensitive.
"""

from std.math import exp, isinf, isnan, log, sqrt
from std.memory import bitcast

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from core.gemm import PINNED_GEMM_TPB, gemm_nt, pinned_gemm_nt_kernel
from core.identity_trace import IdentityTrace, first_divergence
from glm.derived.glm.qn.glm_base import GLMDims, GLMWithData
from glm.derived.glm.qn.glm_softmax import (
    SOFTMAX_MAX_SEED,
    softmax_loss_dz_kernel,
    softmax_row_max,
)
from glm.derived.glm.qn.qn import qn_decision_function, qn_fit_x
from glm.derived.glm.qn.qn_util import (
    FLOAT_EPSILON,
    LBFGS_LS_BT_ARMIJO,
    LBFGS_LS_BT_WOLFE,
    LBFGSParam,
    LS_INVALID_DIR,
    LS_INVALID_STEP,
    LS_INVALID_STEP_MAX,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_LS_FAILED,
    OPT_MAX_ITERS_REACHED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
)
from glm.derived.linear_model.qn import (
    QN_LOSS_LOGISTIC,
    QN_LOSS_SOFTMAX,
    QNParams,
)
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime MN_ROWS = 2048
comptime MN_COLS = 5


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


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _same_bits(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _planted_w(c: Int, k: Int, n_classes: Int) -> Float64:
    """A hashed-but-structured planted weight: mixed signs, none near zero,
    different per class so the classes are separable in the design."""
    var mag = 2.0 - 0.5 * Float64((c * 3 + k) % 4)
    return mag if (c + k) % 2 == 0 else -mag


def _planted_b(c: Int, n_classes: Int) -> Float64:
    return 0.3 * (Float64(c) - 0.5 * Float64(n_classes - 1))


def _fixture(n: Int, d: Int, n_classes: Int, margin: Float64) -> Tuple[List[Float32], List[Float32]]:
    """Hashed design in [-1, 1); label = argmax_c (x . w*_c + b_c + noise),
    lowest index on a tie. `margin` scales the noise: 0 = separable."""
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n):
        var best = 0
        var best_v = -1e300
        for c in range(n_classes):
            var s = _planted_b(c, n_classes)
            for k in range(d):
                var v = 2.0 * _u01(i, k, 0) - 1.0
                s += v * _planted_w(c, k, n_classes)
            s += margin * 2.0 * (_u01(i, 77 + c, 5) - 0.5)
            if s > best_v:
                best_v = s
                best = c
        for k in range(d):
            x.append(Float32(2.0 * _u01(i, k, 0) - 1.0))
        y.append(Float32(best))
    return (x^, y^)


def _params(C: Float64, fit_intercept: Bool, penalty: Bool, max_iter: Int = 1000) -> QNParams:
    """cuML's Python door for `LogisticRegression(penalty='l2', C, tol=1e-4,
    max_iter, linesearch_max_iter=50)` with the softmax loss."""
    var p = QNParams.default()
    p.loss = QN_LOSS_SOFTMAX
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


def _upload(ctx: DeviceContext, h: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](len(h))
    var tmp = h.copy()
    ctx.enqueue_copy(dst_buf=buf, src_ptr=tmp.unsafe_ptr())
    ctx.synchronize()
    _ = tmp^
    return buf^


def _download(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(hb.unsafe_ptr().unsafe_load(i))
    _ = hb^
    return out^


def _fit(
    ctx: DeviceContext,
    xh: List[Float32],
    yh: List[Float32],
    n: Int,
    d: Int,
    n_classes: Int,
    pams: QNParams,
    mut trace: IdentityTrace,
    has_sw: Bool = False,
    w0_len: Int = -1,
) raises -> FitResult:
    """`qn_fit_x` through the softmax arm: `w0` is `C * dims` floats, zero
    initialized, cuML's column-major `W`. `w0_len` overrides the length
    (the refusal check hands in a too-small one)."""
    var x = _upload(ctx, xh)
    var y = _upload(ctx, yh)
    var dims = GLMDims.make(n_classes, d, pams.fit_intercept)
    var n_param = dims.n_param if w0_len < 0 else w0_len
    var w = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.enqueue_memset(w, Float32(0.0))
    ctx.synchronize()
    var fx = Float32(0.0)
    var iters = 0
    var ret = qn_fit_x(ctx, pams, x^, y^, n, d, n_classes, w, fx, iters, has_sw, trace)
    var out = _download(ctx, w, n_param)
    return FitResult(out^, fx, iters, ret)


# ---------------------------------------------------------------------------
# THE FLOAT64 REFERENCE: a different program from the device's
# ---------------------------------------------------------------------------


def _objective64(
    xh: List[Float32], yh: List[Float32], n: Int, d: Int, C: Int, w: List[Float64],
    fit_intercept: Bool, l2: Float64, mut grad: List[Float64],
) -> Float64:
    """`mean_i (lse_i - z_{i, y_i}) + l2/2 ||W_weights||^2` in FLOAT64 with
    its gradient, cuML's layout (`w[c + C*j]`, bias at `j = D`)."""
    var dims = d + (1 if fit_intercept else 0)
    grad.clear()
    for _ in range(C * dims):
        grad.append(0.0)
    var total = 0.0
    var z = List[Float64]()
    var p = List[Float64]()
    for _ in range(C):
        z.append(0.0)
        p.append(0.0)
    for i in range(n):
        var m = -1e300
        for c in range(C):
            var s = 0.0
            for k in range(d):
                s += Float64(xh[i * d + k]) * w[c + C * k]
            if fit_intercept:
                s += w[c + C * d]
            z[c] = s
            if s > m:
                m = s
        var se = 0.0
        for c in range(C):
            se += exp(z[c] - m)
        var lse = m + log(se)
        var yi = Int(yh[i])
        if yi >= 0 and yi < C:
            total += lse - z[yi]
        for c in range(C):
            p[c] = exp(z[c] - lse)
            var dz = p[c] - (1.0 if c == yi else 0.0)
            for k in range(d):
                grad[c + C * k] += Float64(xh[i * d + k]) * dz
            if fit_intercept:
                grad[c + C * d] += dz
    var fx = total / Float64(n)
    for j in range(C * dims):
        grad[j] /= Float64(n)
    for j in range(C * d):
        fx += 0.5 * l2 * w[j] * w[j]
        grad[j] += l2 * w[j]
    return fx


def check_softmax_fd_gradient() raises:
    var n = 512
    var d = 4
    for C in [3, 5]:
        var fx = _fixture(n, d, C, 1.0)
        var dims = d + 1
        var w = List[Float64]()
        for j in range(C * dims):
            w.append(0.8 * (2.0 * _u01(j, 2, 13) - 1.0))
        var l2 = 0.5 / Float64(n)
        var g = List[Float64]()
        var f0 = _objective64(fx[0], fx[1], n, d, C, w, True, l2, g)
        var worst = 0.0
        var h = 1e-5
        for j in range(C * dims):
            var wp = w.copy()
            var wm = w.copy()
            wp[j] += h
            wm[j] -= h
            var gp = List[Float64]()
            var gm = List[Float64]()
            var fp = _objective64(fx[0], fx[1], n, d, C, wp, True, l2, gp)
            var fm = _objective64(fx[0], fx[1], n, d, C, wm, True, l2, gm)
            var fd = (fp - fm) / (2.0 * h)
            var err = abs(fd - g[j]) / max(abs(g[j]), 1e-3)
            if err > worst:
                worst = err
        if worst > 1e-6:
            raise Error(
                "check_softmax_fd_gradient: C=" + String(C) + " worst relative"
                " error analytic vs central difference " + String(worst) + " > 1e-6"
            )
        print(
            "check_softmax_fd_gradient OK: C=" + String(C) + ", " + String(C * dims)
            + " entries, worst relative error " + String(worst) + " (f " + String(f0) + ")"
        )


# ---------------------------------------------------------------------------
# planted fixtures through the device solver
# ---------------------------------------------------------------------------


def _accuracy(xh: List[Float32], yh: List[Float32], n: Int, d: Int, C: Int, w: List[Float32], fit_intercept: Bool) -> Float64:
    var correct = 0
    for i in range(n):
        var best = 0
        var best_v = -1e300
        for c in range(C):
            var s = Float64(w[c + C * d]) if fit_intercept else 0.0
            for k in range(d):
                s += Float64(xh[i * d + k]) * Float64(w[c + C * k])
            if s > best_v:
                best_v = s
                best = c
        if Float32(best) == yh[i]:
            correct += 1
    return Float64(correct) / Float64(n)


def check_softmax_planted() raises:
    var n = MN_ROWS
    var d = MN_COLS
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    for C in [3, 5]:
        var fx = _fixture(n, d, C, 0.0)
        var r = _fit(ctx, fx[0], fx[1], n, d, C, _params(10.0, True, True), off)
        var r1 = _fit(ctx, fx[0], fx[1], n, d, C, _params(10.0, True, True, 1), off)
        var r3 = _fit(ctx, fx[0], fx[1], n, d, C, _params(10.0, True, True, 3), off)
        var acc = _accuracy(fx[0], fx[1], n, d, C, r.w, True)
        var log_c = Float32(log(Float64(C)))
        if acc < 0.95:
            raise Error(
                "check_softmax_planted: C=" + String(C) + " training accuracy "
                + String(acc) + " < 0.95 (n_iter " + String(r.n_iter)
                + ", retcode " + String(r.retcode) + ", objective " + String(r.fx) + ")"
            )
        if not (r.fx < log_c * Float32(0.5)):
            raise Error(
                "check_softmax_planted: C=" + String(C) + " objective " + String(r.fx)
                + " did not fall below half of log C = " + String(log_c)
            )
        if not (r1.fx >= r3.fx and r3.fx >= r.fx):
            raise Error(
                "check_softmax_planted: C=" + String(C) + " objective not monotone"
                " in the iteration budget: " + String(r1.fx) + " (1) / "
                + String(r3.fx) + " (3) / " + String(r.fx) + " (full)"
            )
        if len(r.w) != C * (d + 1):
            raise Error("check_softmax_planted: " + String(len(r.w)) + " parameters, want " + String(C * (d + 1)))
        print(
            "check_softmax_planted OK: C=" + String(C) + " accuracy " + String(acc)
            + ", objective " + String(r1.fx) + " (max_iter 1) >= " + String(r3.fx)
            + " (3) >= " + String(r.fx) + " (full, n_iter " + String(r.n_iter)
            + ", retcode " + String(r.retcode) + ") < log C = " + String(log_c)
        )


def check_softmax_is_a_minimizer() raises:
    var n = MN_ROWS
    var d = MN_COLS
    var C = 3
    var fx = _fixture(n, d, C, 1.0)
    var ctx = DeviceContext()
    var off = IdentityTrace.disabled()
    var r = _fit(ctx, fx[0], fx[1], n, d, C, _params(1.0, True, True), off)
    if r.retcode != OPT_SUCCESS:
        raise Error("check_softmax_is_a_minimizer: retcode " + String(r.retcode) + " after " + String(r.n_iter) + " iterations")
    var l2 = 1.0 / Float64(n)
    var w64 = List[Float64]()
    for j in range(C * (d + 1)):
        w64.append(Float64(r.w[j]))
    var grad = List[Float64]()
    var f0 = _objective64(fx[0], fx[1], n, d, C, w64, True, l2, grad)
    var gmax = 0.0
    for j in range(C * (d + 1)):
        if abs(grad[j]) > gmax:
            gmax = abs(grad[j])
    var bound = 1e-4 * max(f0, 1e-4)
    if gmax > bound:
        raise Error(
            "check_softmax_is_a_minimizer: host float64 gradient Linf " + String(gmax)
            + " at the device solution exceeds the solver's own bound " + String(bound)
            + " (fx " + String(f0) + ", n_iter " + String(r.n_iter) + ")"
        )
    for p in range(8):
        var wp = List[Float64]()
        for j in range(C * (d + 1)):
            wp.append(w64[j] + 0.05 * (2.0 * _u01(p, j, 9) - 1.0))
        var g2 = List[Float64]()
        var fp = _objective64(fx[0], fx[1], n, d, C, wp, True, l2, g2)
        if not (fp > f0):
            raise Error("check_softmax_is_a_minimizer: perturbed objective " + String(fp) + " <= " + String(f0))
    print(
        "check_softmax_is_a_minimizer OK: float64 gradient Linf " + String(gmax)
        + " <= " + String(bound) + " at the device solution (objective " + String(f0)
        + ", device's own " + String(r.fx) + "), below 8 perturbed objectives; n_iter "
        + String(r.n_iter)
    )


def _fit_raises(ctx: DeviceContext, pams: QNParams, n_classes: Int, has_sw: Bool, w0_len: Int = -1) raises -> String:
    var fx = _fixture(256, 4, 3, 1.0)
    var off = IdentityTrace.disabled()
    try:
        var r = _fit(ctx, fx[0], fx[1], 256, 4, n_classes, pams, off, has_sw, w0_len)
        _ = r.n_iter
    except e:
        return String(e)
    return String("")


def check_softmax_refuses_by_name() raises:
    var ctx = DeviceContext()
    var m1 = _fit_raises(ctx, _params(1.0, True, True), 2, False)
    if m1.find("softmax invalid C") < 0:
        raise Error("softmax with 2 classes did not raise their text; got: " + m1)
    var p2 = _params(1.0, True, True)
    p2.loss = QN_LOSS_LOGISTIC
    var m2 = _fit_raises(ctx, p2, 3, False)
    if m2.find("logistic loss invalid C") < 0:
        raise Error("logistic with 3 classes did not raise their text; got: " + m2)
    var m3 = _fit_raises(ctx, _params(1.0, True, True), 3, False, 5)
    if m3.find("QN_LOSS_SOFTMAX needs w0 of n_param") < 0:
        raise Error("a too-small w0 did not raise by name; got: " + m3)
    var p4 = _params(1.0, True, True)
    p4.penalty_l1 = 0.5
    var m4 = _fit_raises(ctx, p4, 3, False)
    if m4.find("min_owlqn") < 0:
        raise Error("l1 did not raise by name; got: " + m4)
    var m5 = _fit_raises(ctx, _params(1.0, True, True), 3, True)
    if m5.find("sample_weight is NOT PORTED") < 0:
        raise Error("sample_weight did not raise by name; got: " + m5)
    print(
        "check_softmax_refuses_by_name OK: softmax with C=2, logistic with C=3,"
        " a too-small w0, l1 (OWL-QN) and sample_weight each RAISE by name"
    )


# ---------------------------------------------------------------------------
# THE HOST REPLAY of every pinned shape the softmax objective runs
# ---------------------------------------------------------------------------


def _tree(mut red: List[Float32]) -> Float32:
    """`pinned_block_sum`'s IDENTICAL arm on `STATS_TPB` partials: halving,
    no flush inside, the result flushed."""
    var step = STATS_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def _host_sum_terms(terms: List[Float32], n: Int) -> Float32:
    """`sum_terms_kernel`: strided `acc = ftz(acc + t)`, then the tree."""
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < n:
            acc = ftz(acc + terms[i])
            i += STATS_TPB
        red.append(acc)
    return _tree(red)


def _host_dot(u: List[Float32], v: List[Float32], n: Int) -> Float32:
    """`dot_kernel`: strided `acc = fma(u, v, acc)` (no flush per step),
    then the tree."""
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < n:
            acc = identical_mul_add(u[i], v[i], acc)
            i += STATS_TPB
        red.append(acc)
    return _tree(red)


def _host_nrm_max(u: List[Float32], n: Int) -> Float32:
    var m = Float32(0.0)
    for i in range(n):
        var a = abs(u[i])
        if a > m:
            m = a
    return m


def _host_forward(
    xh: List[Float32], w: List[Float32], n: Int, d: Int, C: Int, fit_intercept: Bool
) -> List[Float32]:
    """`linear_fwd`'s `C > 1` arm: `pinned_gemm_nt_kernel` cell by cell
    (`acc = ftz(fma(ftz(x), ftz(w_rm), acc))`, `ftz(0 + ftz(acc))`) and the
    bias seam. Returns `z[c + C*i]`."""
    var z = _zeros(C * n)
    for i in range(n):
        for c in range(C):
            var acc = Float32(0.0)
            for p in range(d):
                acc = ftz(identical_mul_add(ftz(xh[i * d + p]), ftz(w[c + C * p]), acc))
            var v = ftz(Float32(0.0) + ftz(acc))
            if fit_intercept:
                v = ftz(v + w[c + C * d])
            z[c + C * i] = v
    return z^


def _host_softmax_rows(
    mut z: List[Float32], yh: List[Float32], n: Int, C: Int
) -> List[Float32]:
    """`softmax_loss_dz_kernel` re-spelled: the positional max, the serial
    ascending lse, `exp(eta - lse) - delta`, `(lse - eta_y) / N`. `z`
    becomes dZ in place; returns the per-row loss terms."""
    var terms = _zeros(n)
    for i in range(n):
        var label = yh[i]
        var eta_max = SOFTMAX_MAX_SEED
        for c in range(C):
            var v = z[c + C * i]
            if v > eta_max:
                eta_max = v
        var delta = False
        var eta_y = Float32(0.0)
        for c in range(C):
            if Float32(c) == label:
                delta = True
                eta_y = z[c + C * i]
        var s = Float32(0.0)
        for c in range(C):
            s = ftz(s + ftz(identical_exp(ftz(z[c + C * i] - eta_max))))
        var lse = ftz(eta_max + ftz(identical_log(s)))
        for c in range(C):
            var p = ftz(identical_exp(ftz(z[c + C * i] - lse)))
            var dlt = Float32(1.0) if Float32(c) == label else Float32(0.0)
            z[c + C * i] = ftz(p - dlt)
        terms[i] = ftz(ftz(lse - eta_y) / Float32(n)) if delta else Float32(0.0)
    return terms^


def _host_backward(
    xh: List[Float32], dz: List[Float32], n: Int, d: Int, C: Int, fit_intercept: Bool,
    mut g: List[Float32], set_zero: Bool,
):
    """`linear_bwd`'s `C > 1` arm: `xtdz_multi_kernel` per `(c, j)` (strided
    fma partials, tree), the cuBLAS epilogue (`ftz(alpha * s)`, `+ g`), and
    `mean_rows_multi_kernel` per class (`ftz(acc + dz)`, tree, `* ratio`)."""
    var alpha = Float32(1.0 / Float64(n))
    for b in range(C * d):
        var c = b % C
        var j = b // C
        var red = List[Float32]()
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var r = t
            while r < n:
                acc = identical_mul_add(xh[r * d + j], dz[c + C * r], acc)
                r += STATS_TPB
            red.append(acc)
        var s0 = _tree(red)
        var s = ftz(alpha * s0)
        g[b] = s if set_zero else ftz(s + g[b])
    if fit_intercept:
        var ratio = Float32(1.0) / Float32(n)
        for c in range(C):
            var red = List[Float32]()
            for t in range(STATS_TPB):
                var acc = Float32(0.0)
                var i = t
                while i < n:
                    acc = ftz(acc + dz[c + C * i])
                    i += STATS_TPB
                red.append(acc)
            var s0 = _tree(red)
            g[C * d + c] = ftz(s0 * ratio)


def _host_tikhonov(w: List[Float32], n_weights: Int, l2: Float32, mut g: List[Float32]) -> Float32:
    """`tikhonov_reg_grad_kernel`: `g[j] = l2 * w_j`, strided `acc += 0.5 *
    l2 * w_j * w_j` (two products, each flushed), tree."""
    var half_l2 = ftz(Float32(0.5) * l2)
    var red = List[Float32]()
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var j = t
        while j < n_weights:
            var wj = w[j]
            g[j] = ftz(l2 * wj)
            var tt = ftz(half_l2 * wj)
            acc = ftz(acc + ftz(tt * wj))
            j += STATS_TPB
        red.append(acc)
    return _tree(red)


struct HostSoftmaxObjective(Movable):
    """`GLMWithData` on the host: `evaluate(w, g)` and `grad_norm(g)`
    through the pinned shapes above. `dz` is left holding the last dZ."""

    var x: List[Float32]
    var y: List[Float32]
    var n: Int
    var d: Int
    var C: Int
    var fit_intercept: Bool
    var l2: Float32
    var dz: List[Float32]
    var n_evals: Int

    def __init__(
        out self, var x: List[Float32], var y: List[Float32], n: Int, d: Int,
        C: Int, fit_intercept: Bool, l2: Float32,
    ):
        self.x = x^
        self.y = y^
        self.n = n
        self.d = d
        self.C = C
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.dz = List[Float32]()
        self.n_evals = 0

    def n_param(self) -> Int:
        return self.C * (self.d + (1 if self.fit_intercept else 0))

    def loss_grad(mut self, w: List[Float32], mut g: List[Float32], init_grad_zero: Bool) -> Float32:
        var z = _host_forward(self.x, w, self.n, self.d, self.C, self.fit_intercept)
        var terms = _host_softmax_rows(z, self.y, self.n, self.C)
        var loss = _host_sum_terms(terms, self.n)
        _host_backward(self.x, z, self.n, self.d, self.C, self.fit_intercept, g, init_grad_zero)
        self.dz = z^
        return loss

    def evaluate(mut self, w: List[Float32], mut g: List[Float32]) -> Float32:
        self.n_evals += 1
        if self.l2 == Float32(0.0):
            return self.loss_grad(w, g, True)
        for j in range(len(g)):
            g[j] = Float32(0.0)
        var reg = _host_tikhonov(w, self.C * self.d, self.l2, g)
        var loss = self.loss_grad(w, g, False)
        return ftz(loss + reg)

    def grad_norm(self, g: List[Float32]) -> Float32:
        return _host_nrm_max(g, self.n_param())


def check_softmax_device_equals_host() raises:
    """One `GLMWithData.evaluate` at a hashed `w` (with l2): loss, every
    gradient entry and every dZ cell against the host replay; the decision
    function against the host forward; the device gradient against the
    float64 gradient within tolerance."""
    var n = MN_ROWS
    var d = MN_COLS
    var C = 3
    var fx = _fixture(n, d, C, 1.0)
    var ctx = DeviceContext()
    var x = _upload(ctx, fx[0])
    var y = _upload(ctx, fx[1])
    var dims = GLMDims.make(C, d, True)
    var n_param = dims.n_param
    var hw = List[Float32]()
    for j in range(n_param):
        hw.append(Float32(0.8 * (2.0 * _u01(j, 3, 11) - 1.0)))
    var w = _upload(ctx, hw)
    var g = ctx.enqueue_create_buffer[DType.float32](n_param)
    ctx.synchronize()
    var l2 = Float32(1.0) / Float32(n)
    var obj = GLMWithData(ctx, x^, y^, n, dims, QN_LOSS_SOFTMAX, l2)
    var loss_dev = obj.evaluate(ctx, w, g)
    var g_dev = _download(ctx, g, n_param)
    var dz_dev = _download(ctx, obj.z, C * n)
    # the decision function's multi arm (reach): scores = W X^T + b
    var x2 = _upload(ctx, fx[0])
    var scores = ctx.enqueue_create_buffer[DType.float32](C * n)
    ctx.synchronize()
    qn_decision_function(ctx, _params(1.0, True, True), x2, n, d, w, scores, C)
    var scores_dev = _download(ctx, scores, C * n)
    # HOST replay
    var host = HostSoftmaxObjective(fx[0].copy(), fx[1].copy(), n, d, C, True, l2)
    var g_h = _zeros(n_param)
    var loss_h = host.evaluate(hw, g_h)
    var z_h = _host_forward(fx[0], hw, n, d, C, True)
    var bad = 0
    var first = String("")
    if not _same_bits(loss_h, loss_dev):
        bad += 1
        first = "loss device " + _hex32(loss_dev) + " host " + _hex32(loss_h)
    for j in range(n_param):
        if not _same_bits(g_h[j], g_dev[j]):
            bad += 1
            if first == "":
                first = "grad " + String(j) + " device " + _hex32(g_dev[j]) + " host " + _hex32(g_h[j])
    var bad_dz = 0
    for k in range(C * n):
        if not _same_bits(host.dz[k], dz_dev[k]):
            bad_dz += 1
            if first == "":
                first = "dZ " + String(k) + " device " + _hex32(dz_dev[k]) + " host " + _hex32(host.dz[k])
    var bad_scores = 0
    for k in range(C * n):
        if not _same_bits(z_h[k], scores_dev[k]):
            bad_scores += 1
            if first == "":
                first = "scores " + String(k) + " device " + _hex32(scores_dev[k]) + " host " + _hex32(z_h[k])
    # float64 tolerance, both modes
    var w64 = List[Float64]()
    for j in range(n_param):
        w64.append(Float64(hw[j]))
    var g64 = List[Float64]()
    var f64 = _objective64(fx[0], fx[1], n, d, C, w64, True, Float64(l2), g64)
    var worst = abs(Float64(loss_dev) - f64) / max(abs(f64), 1e-3)
    for j in range(n_param):
        var e = abs(Float64(g_dev[j]) - g64[j]) / max(abs(g64[j]), 1e-3)
        if e > worst:
            worst = e
    if worst > 1e-4:
        raise Error(
            "check_softmax_device_equals_host: device loss/gradient vs float64"
            " worst relative error " + String(worst) + " > 1e-4 (loss device "
            + String(loss_dev) + " float64 " + String(f64) + ")"
        )
    var total = bad + bad_dz + bad_scores
    comptime if IDENTICAL:
        if total != 0:
            raise Error(
                "check_softmax_device_equals_host [IDENTICAL]: " + String(bad)
                + " of " + String(n_param + 1) + " loss/grad values, " + String(bad_dz)
                + " of " + String(C * n) + " dZ cells, " + String(bad_scores)
                + " of " + String(C * n) + " scores differ. First: " + first
            )
        print(
            "check_softmax_device_equals_host OK [IDENTICAL]: loss, all " + String(n_param)
            + " gradient entries, all " + String(C * n) + " dZ cells and all "
            + String(C * n) + " decision-function scores equal the host replay bit"
            " for bit (loss " + _hex32(loss_dev) + "); float64 worst rel err " + String(worst)
        )
    else:
        print(
            "check_softmax_device_equals_host REPORT [FAST]: " + String(bad) + " of "
            + String(n_param + 1) + " loss/grad values, " + String(bad_dz) + " of "
            + String(C * n) + " dZ cells, " + String(bad_scores) + " of " + String(C * n)
            + " scores differ from the pinned host replay (vendor matmul, block.sum,"
            " device exp/log)" + ("" if first == "" else "; first: " + first)
            + "; float64 worst rel err " + String(worst) + " <= 1e-4"
        )


# ---------------------------------------------------------------------------
# signed zero at the selection; launch invariance
# ---------------------------------------------------------------------------


def _row_max_probe_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    n_classes_in: Int32,
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_v.unsafe_store(i, softmax_row_max(z, i, Int(n_classes_in)))


def _launch_softmax(
    ctx: DeviceContext,
    mut terms: DeviceBuffer[DType.float32],
    mut z: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    C: Int,
    n: Int,
    block: Int,
) raises:
    ctx.enqueue_function[softmax_loss_dz_kernel](
        terms.unsafe_ptr(), z.unsafe_ptr(), y.unsafe_ptr(), Int32(C), Int32(n),
        grid_dim=((n + block - 1) // block, 1, 1), block_dim=(block, 1, 1),
    )
    ctx.synchronize()


def check_softmax_signed_zero_selection() raises:
    var C = 3
    var rows = List[Float32]()
    var want = List[Float32]()
    # [+0, -0, -1] -> +0 (index 0)     [-0, +0, -1] -> -0 (index 0)
    # [-1, +0, -0] -> +0 (index 1)     [-1, -0, +0] -> -0 (index 1)
    # [1, 1, 0]    -> 1 (index 0)      [-2, -2, -3] -> -2 (index 0)
    var pz = Float32(0.0)
    var nz = Float32(-0.0)
    var vals: List[Float32] = [pz, nz, -1.0, nz, pz, -1.0, -1.0, pz, nz, -1.0, nz, pz, 1.0, 1.0, 0.0, -2.0, -2.0, -3.0]
    for v in vals:
        rows.append(v)
    want.append(pz)
    want.append(nz)
    want.append(pz)
    want.append(nz)
    want.append(Float32(1.0))
    want.append(Float32(-2.0))
    var n = 6
    var labels: List[Float32] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    var ctx = DeviceContext()
    var z = _upload(ctx, rows)
    var out = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_function[_row_max_probe_kernel](
        out.unsafe_ptr(), z.unsafe_ptr(), Int32(C), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
    )
    ctx.synchronize()
    var got = _download(ctx, out, n)
    var bad = 0
    var first = String("")
    for i in range(n):
        if not _same_bits(got[i], want[i]):
            bad += 1
            if first == "":
                first = "row " + String(i) + " device " + _hex32(got[i]) + " positional rule " + _hex32(want[i])
    # the whole kernel on those rows vs the host
    var y = _upload(ctx, labels)
    var terms = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    _launch_softmax(ctx, terms, z, y, C, n, 32)
    var dz_dev = _download(ctx, z, C * n)
    var t_dev = _download(ctx, terms, n)
    var zh = rows.copy()
    var th = _host_softmax_rows(zh, labels, n, C)
    var bad2 = 0
    for k in range(C * n):
        if not _same_bits(zh[k], dz_dev[k]):
            bad2 += 1
            if first == "":
                first = "dZ " + String(k) + " device " + _hex32(dz_dev[k]) + " host " + _hex32(zh[k])
    for i in range(n):
        if not _same_bits(th[i], t_dev[i]):
            bad2 += 1
            if first == "":
                first = "term " + String(i) + " device " + _hex32(t_dev[i]) + " host " + _hex32(th[i])
    if bad != 0:
        raise Error(
            "check_softmax_signed_zero_selection: " + String(bad) + " of " + String(n)
            + " selected maxima differ from the positional rule (strict >, lower"
            " index wins). First: " + first
        )
    comptime if IDENTICAL:
        if bad2 != 0:
            raise Error(
                "check_softmax_signed_zero_selection [IDENTICAL]: " + String(bad2)
                + " kernel cells differ from the host on the signed-zero rows. First: " + first
            )
        print(
            "check_softmax_signed_zero_selection OK [IDENTICAL]: the max selection"
            " answers the positional rule by bits on +0/-0 in both orders and on an"
            " exact tie (" + _hex32(got[0]) + " " + _hex32(got[1]) + " " + _hex32(got[2])
            + " " + _hex32(got[3]) + "); the whole kernel agrees with the host on all "
            + String(C * n + n) + " cells"
        )
    else:
        print(
            "check_softmax_signed_zero_selection OK [FAST]: the max selection answers"
            " the positional rule by bits; kernel vs host on those rows: " + String(bad2)
            + " cells differ (REPORT: device exp/log)"
        )


def _poison(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var h = List[Float32]()
    for _ in range(n):
        h.append(bitcast[DType.float32](UInt32(0x7FC0BEEF)))
    return _upload(ctx, h)


def check_softmax_launch_invariance() raises:
    var n = MN_ROWS
    var d = MN_COLS
    var C = 3
    var fx = _fixture(n, d, C, 1.0)
    var hw = List[Float32]()
    for j in range(C * (d + 1)):
        hw.append(Float32(0.8 * (2.0 * _u01(j, 3, 11) - 1.0)))
    var z0 = _host_forward(fx[0], hw, n, d, C, True)
    var ctx = DeviceContext()
    var y = _upload(ctx, fx[1])
    # reference launch: block 256, exact-size buffers
    var z_ref = _upload(ctx, z0)
    var t_ref = _poison(ctx, n)
    _launch_softmax(ctx, t_ref, z_ref, y, C, n, 256)
    var dz_ref = _download(ctx, z_ref, C * n)
    var tr = _download(ctx, t_ref, n)
    var moved = 0
    var first = String("")
    # block sizes 32 and 64, padded + poisoned outputs
    for bs in [32, 64]:
        var zpad = _zeros(C * n + 97)
        for k in range(C * n):
            zpad[k] = z0[k]
        for k in range(C * n, C * n + 97):
            zpad[k] = bitcast[DType.float32](UInt32(0x7FC0BEEF))
        var zb = _upload(ctx, zpad)
        var tb = _poison(ctx, n + 41)
        _launch_softmax(ctx, tb, zb, y, C, n, bs)
        var dzb = _download(ctx, zb, C * n + 97)
        var tbh = _download(ctx, tb, n + 41)
        for k in range(C * n):
            if not _same_bits(dzb[k], dz_ref[k]):
                moved += 1
                if first == "":
                    first = "block " + String(bs) + " dZ " + String(k) + " " + _hex32(dzb[k]) + " vs " + _hex32(dz_ref[k])
        for i in range(n):
            if not _same_bits(tbh[i], tr[i]):
                moved += 1
                if first == "":
                    first = "block " + String(bs) + " term " + String(i) + " " + _hex32(tbh[i]) + " vs " + _hex32(tr[i])
        # the padding must still hold its poison
        for k in range(C * n, C * n + 97):
            if bitcast[DType.uint32](dzb[k]) != UInt32(0x7FC0BEEF):
                raise Error("check_softmax_launch_invariance: block " + String(bs) + " wrote past the last row (dZ pad " + String(k) + ")")
        for i in range(n, n + 41):
            if bitcast[DType.uint32](tbh[i]) != UInt32(0x7FC0BEEF):
                raise Error("check_softmax_launch_invariance: block " + String(bs) + " wrote past the last row (term pad " + String(i) + ")")
    # batch composition: rows 100..163 alone vs inside the 2048 launch
    var sub_n = 64
    var zs = List[Float32]()
    var ys = List[Float32]()
    for i in range(100, 100 + sub_n):
        for c in range(C):
            zs.append(z0[c + C * i])
        ys.append(fx[1][i])
    var zsb = _upload(ctx, zs)
    var ysb = _upload(ctx, ys)
    var tsb = _poison(ctx, sub_n)
    _launch_softmax(ctx, tsb, zsb, ysb, C, sub_n, 64)
    var dzs = _download(ctx, zsb, C * sub_n)
    var tss = _download(ctx, tsb, sub_n)
    for i in range(sub_n):
        for c in range(C):
            if not _same_bits(dzs[c + C * i], dz_ref[c + C * (100 + i)]):
                moved += 1
                if first == "":
                    first = "batch row " + String(100 + i) + " class " + String(c)
        # the per-row term carries `/ N`, so compare it with N re-applied:
        # `(lse - eta_y) / 64` vs `/ 2048` differ by design; compare dZ only
        # and the term's zero-ness
        if (tss[i] == Float32(0.0)) != (tr[100 + i] == Float32(0.0)):
            moved += 1
    # the forward product: the pinned kernel at two block sizes vs gemm_nt
    var x = _upload(ctx, fx[0])
    var w_rm = List[Float32]()
    for c in range(C):
        for p in range(d):
            w_rm.append(hw[c + C * p])
    var wrb = _upload(ctx, w_rm)
    var z_g = ctx.enqueue_create_buffer[DType.float32](C * n)
    ctx.synchronize()
    gemm_nt(ctx, z_g, x, wrb, n, C, d)
    ctx.synchronize()
    var zg = _download(ctx, z_g, C * n)
    var prod_moved = 0
    for bs in [64, 256]:
        var z_p = _poison(ctx, C * n)
        ctx.enqueue_function[pinned_gemm_nt_kernel](
            z_p.unsafe_ptr(), x.unsafe_ptr(), wrb.unsafe_ptr(), Int32(n), Int32(C), Int32(d),
            grid_dim=((C * n + bs - 1) // bs, 1, 1), block_dim=(bs, 1, 1),
        )
        ctx.synchronize()
        var zp = _download(ctx, z_p, C * n)
        for k in range(C * n):
            if not _same_bits(zp[k], zg[k]):
                prod_moved += 1
    # host forward without the bias must equal the pinned product
    var z_nb = _host_forward(fx[0], hw, n, d, C, False)
    var prod_host = 0
    for k in range(C * n):
        if not _same_bits(z_nb[k], zg[k]):
            prod_host += 1
    if moved != 0:
        raise Error(
            "check_softmax_launch_invariance: " + String(moved) + " cells moved across"
            " block sizes 32/64/256, paddings and the batch composition. First: " + first
        )
    comptime if IDENTICAL:
        if prod_moved != 0 or prod_host != 0:
            raise Error(
                "check_softmax_launch_invariance [IDENTICAL]: the forward product moved: "
                + String(prod_moved) + " cells across block sizes 64/256 vs gemm_nt, "
                + String(prod_host) + " cells vs the host"
            )
        print(
            "check_softmax_launch_invariance OK [IDENTICAL]: softmax kernel invariant"
            " over block 32/64/256, two paddings (poison intact) and the batch"
            " composition; forward product invariant over block 64/256 and equal to"
            " the host on all " + String(C * n) + " cells"
        )
    else:
        print(
            "check_softmax_launch_invariance OK [FAST]: softmax kernel invariant over"
            " block 32/64/256, paddings and batch composition; forward product"
            " RECORDED [FAST]: vendor matmul vs pinned kernel " + String(prod_moved)
            + " of " + String(2 * C * n) + " cells differ, vs host " + String(prod_host)
        )


# ---------------------------------------------------------------------------
# THE ITERATION ORACLE: L-BFGS replayed on the host, same tags
# ---------------------------------------------------------------------------


def _iter_tag(k: Int) -> String:
    var s = String(k)
    while s.byte_length() < 4:
        s = "0" + s
    return "qn.iter" + s


def _host_check_convergence(param: LBFGSParam, k: Int, fx: Float32, gnorm: Float32, mut fx_hist: List[Float32]) -> Bool:
    var fmag = max(fx, param.epsilon)
    if gnorm <= param.epsilon * fmag:
        return True
    if param.past > 0:
        if k >= param.past and abs(fx_hist[k % param.past] - fx) <= param.delta * fmag:
            return True
        fx_hist[k % param.past] = fx
    return False


def _host_axpy(mut out: List[Float32], a: Float32, x: List[Float32], y: List[Float32], n: Int):
    for i in range(n):
        out[i] = ftz(identical_mul_add(a, x[i], y[i]))


def _host_ls_success(
    param: LBFGSParam, fx_init: Float32, dg_init: Float32, fx: Float32, dg_test: Float32,
    step: Float32, grad: List[Float32], drt: List[Float32], n: Int, mut width: Float32,
) -> Bool:
    if fx > identical_mul_add(step, dg_test, fx_init):
        width = param.ls_dec
    else:
        if param.linesearch == LBFGS_LS_BT_ARMIJO:
            return True
        var dg = _host_dot(grad, drt, n)
        if dg < param.wolfe * dg_init:
            width = param.ls_inc
        else:
            if param.linesearch == LBFGS_LS_BT_WOLFE:
                return True
            if dg > -param.wolfe * dg_init:
                width = param.ls_dec
            else:
                return True
    return False


def _host_ls_backtrack(
    param: LBFGSParam, mut f: HostSoftmaxObjective, mut fx: Float32, mut x: List[Float32],
    mut grad: List[Float32], mut step: Float32, drt: List[Float32], xp: List[Float32],
    n: Int, mut ls_iters: Int,
) -> Int:
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    var dg_init = _host_dot(grad, drt, n)
    if dg_init > Float32(0.0):
        return LS_INVALID_DIR
    var dg_test = param.ftol * dg_init
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        _host_axpy(x, step, drt, xp, n)
        fx = f.evaluate(x, grad)
        ls_iters += 1
        if _host_ls_success(param, fx_init, dg_init, fx, dg_test, step, grad, drt, n, width):
            return LS_SUCCESS
        if step < param.min_step:
            return LS_INVALID_STEP_MIN
        if step > param.max_step:
            return LS_INVALID_STEP_MAX
        step *= width
    return LS_MAX_ITERS_REACHED


def _host_update_and_check(
    param: LBFGSParam, iter: Int, lsret: Int, mut fx: Float32, fxp: Float32, gnorm: Float32,
    mut x: List[Float32], xp: List[Float32], mut grad: List[Float32], gradp: List[Float32],
    mut fx_hist: List[Float32], mut outcode: Int,
) -> Bool:
    var stop = False
    var converged = False
    var is_ls_valid = (not isnan(fx)) and (not isinf(fx))
    var is_ls_non_critical = lsret == LS_INVALID_STEP_MIN or lsret == LS_MAX_ITERS_REACHED
    var is_ls_in_doubt = is_ls_valid and fx <= fxp + param.ftol and is_ls_non_critical
    var is_ls_success = lsret == LS_SUCCESS or is_ls_in_doubt
    if is_ls_valid:
        converged = _host_check_convergence(param, iter, fx, gnorm, fx_hist)
    if (not is_ls_success) and (not converged):
        outcode = OPT_LS_FAILED
        stop = True
    elif not is_ls_valid:
        outcode = OPT_NUMERIC_ERROR
        stop = True
    elif converged:
        outcode = OPT_SUCCESS
        stop = True
    elif is_ls_in_doubt and fx + param.ftol >= fxp:
        outcode = OPT_LS_FAILED
        stop = True
    if (not is_ls_success) or (not is_ls_valid):
        fx = fxp
        for i in range(len(x)):
            x[i] = xp[i]
            grad[i] = gradp[i]
    return stop


def _host_lbfgs_search_dir(
    param: LBFGSParam, mut n_vec: Int, end_prev: Int, S: List[List[Float32]],
    Y: List[List[Float32]], g: List[Float32], mut drt: List[Float32],
    mut yhist: List[Float32], mut alpha: List[Float32], n: Int,
) -> Int:
    var end = end_prev
    var ys = _host_dot(S[end], Y[end], n)
    var yy = _host_dot(Y[end], Y[end], n)
    if ys <= FLOAT_EPSILON * yy:
        return end
    n_vec += 1
    yhist[end] = ys
    for i in range(n):
        drt[i] = ftz(Float32(-1.0) * g[i])
    var bound = min(param.m, n_vec)
    end = (end + 1) % param.m
    var j = end
    for _ in range(bound):
        j = (j + param.m - 1) % param.m
        alpha[j] = _host_dot(S[j], drt, n) / yhist[j]
        var a = -alpha[j]
        for i in range(n):
            drt[i] = ftz(identical_mul_add(a, Y[j][i], drt[i]))
    var scale = ys / yy
    for i in range(n):
        drt[i] = ftz(scale * drt[i])
    for _ in range(bound):
        var beta = _host_dot(Y[j], drt, n) / yhist[j]
        var a = alpha[j] - beta
        for i in range(n):
            drt[i] = ftz(identical_mul_add(a, S[j][i], drt[i]))
        j = (j + 1) % param.m
    return end


def _host_min_lbfgs(
    param: LBFGSParam, mut f: HostSoftmaxObjective, mut x: List[Float32], mut fx: Float32,
    mut k: Int, n: Int, mut trace: IdentityTrace,
) raises -> Int:
    """`min_lbfgs` on the host, recording the same tags as the device."""
    var S = List[List[Float32]]()
    var Y = List[List[Float32]]()
    for _ in range(param.m):
        S.append(_zeros(n))
        Y.append(_zeros(n))
    var xp = _zeros(n)
    var grad = _zeros(n)
    var gradp = _zeros(n)
    var drt = _zeros(n)
    var ys = _zeros(param.m)
    var alpha = _zeros(param.m)
    var fx_hist = _zeros(param.past if param.past > 0 else 0)
    k = 0
    fx = f.evaluate(x, grad)
    var gnorm = f.grad_norm(grad)
    trace.record_scalar_f32("qn.init.loss", fx)
    trace.record_list_f32("qn.init.grad", grad)
    if param.past > 0:
        fx_hist[0] = fx
    if _host_check_convergence(param, k, fx, gnorm, fx_hist):
        return OPT_SUCCESS
    for i in range(n):
        drt[i] = ftz(Float32(-1.0) * grad[i])
    var step = Float32(1.0) / sqrt(_host_dot(drt, drt, n))
    var fxp = fx
    k = 1
    var end = 0
    var n_vec = 0
    var retcode = OPT_MAX_ITERS_REACHED
    var ls_iters = 0
    while k <= param.max_iterations:
        for i in range(n):
            xp[i] = x[i]
            gradp[i] = grad[i]
        fxp = fx
        var lsret = _host_ls_backtrack(param, f, fx, x, grad, step, drt, xp, n, ls_iters)
        gnorm = f.grad_norm(grad)
        var stop = _host_update_and_check(param, k, lsret, fx, fxp, gnorm, x, xp, grad, gradp, fx_hist, retcode)
        if trace.enabled:
            var tag = _iter_tag(k)
            trace.record_scalar_f32(tag + ".loss", fx)
            trace.record_list_f32(tag + ".grad", grad)
            var ls = List[Int32]()
            ls.append(Int32(lsret))
            ls.append(Int32(ls_iters))
            trace.record_list_i32(tag + ".ls", ls)
        if stop:
            return retcode
        for i in range(n):
            S[end][i] = ftz(identical_mul_add(Float32(-1.0), xp[i], x[i]))
            Y[end][i] = ftz(identical_mul_add(Float32(-1.0), gradp[i], grad[i]))
        end = _host_lbfgs_search_dir(param, n_vec, end, S, Y, grad, drt, ys, alpha, n)
        step = Float32(1.0)
        k += 1
    return OPT_MAX_ITERS_REACHED


def _host_fit(
    xh: List[Float32], yh: List[Float32], n: Int, d: Int, C: Int, pams: QNParams,
    mut trace: IdentityTrace,
) raises -> FitResult:
    """`qn_fit_x`'s softmax arm on the host: l2 normalized, L-BFGS, the
    closing stages."""
    var l2 = Float32(pams.penalty_l2)
    if pams.penalty_normalized:
        l2 = l2 / Float32(n)
    var f = HostSoftmaxObjective(xh.copy(), yh.copy(), n, d, C, pams.fit_intercept, l2)
    var n_param = f.n_param()
    var w = _zeros(n_param)
    var opt = LBFGSParam.from_params(pams)
    var fx = Float32(0.0)
    var k = 0
    var ret = _host_min_lbfgs(opt, f, w, fx, k, n_param, trace)
    trace.record_list_f32("qn.coef", w)
    var ints = List[Int32]()
    ints.append(Int32(k))
    trace.record_list_i32("qn.n_iter", ints)
    var rc = List[Int32]()
    rc.append(Int32(ret))
    trace.record_list_i32("qn.retcode", rc)
    return FitResult(w^, fx, k, ret)


def check_softmax_host_lbfgs_replay() raises:
    var n = 512
    var d = 4
    var C = 3
    var fx = _fixture(n, d, C, 1.0)
    var pams = _params(1.0, True, True)
    var p_dev = String("/tmp/mojolearn_multinomial_device.card")
    var p_host = String("/tmp/mojolearn_multinomial_host.card")
    var ctx = DeviceContext()
    var td = IdentityTrace.to_path(p_dev, "", True)
    td.header(String("softmax device n=") + String(n) + " d=" + String(d) + " C=" + String(C))
    var r_dev = _fit(ctx, fx[0], fx[1], n, d, C, pams, td)
    var th = IdentityTrace.to_path(p_host, "", True)
    th.header(String("softmax host replay n=") + String(n) + " d=" + String(d) + " C=" + String(C))
    var r_host = _host_fit(fx[0], fx[1], n, d, C, pams, th)
    var moved = 0
    for j in range(len(r_dev.w)):
        if not _same_bits(r_dev.w[j], r_host.w[j]):
            moved += 1
    var diff = first_divergence(p_dev, p_host)
    comptime if IDENTICAL:
        if diff != "" or moved != 0 or r_dev.n_iter != r_host.n_iter or r_dev.retcode != r_host.retcode:
            raise Error(
                "check_softmax_host_lbfgs_replay [IDENTICAL]: device and host L-BFGS"
                " disagree: " + String(moved) + " coefficients moved, n_iter "
                + String(r_dev.n_iter) + " vs " + String(r_host.n_iter) + ", retcode "
                + String(r_dev.retcode) + " vs " + String(r_host.retcode)
                + "; first diverging stage: " + diff
            )
        print(
            "check_softmax_host_lbfgs_replay OK [IDENTICAL]: device and host L-BFGS"
            " agree at every card stage (n_iter " + String(r_dev.n_iter) + ", retcode "
            + String(r_dev.retcode) + ", " + String(len(r_dev.w)) + " coefficients, objective "
            + String(r_dev.fx) + " = " + String(r_host.fx) + ")"
        )
    else:
        print(
            "check_softmax_host_lbfgs_replay REPORT [FAST]: " + String(moved) + " of "
            + String(len(r_dev.w)) + " coefficients differ from the host replay; n_iter "
            + String(r_dev.n_iter) + " vs " + String(r_host.n_iter) + "; first diverging"
            " stage: " + (diff if diff != "" else "none")
        )


def check_softmax_card_is_emitted() raises:
    var p1 = String("/tmp/mojolearn_multinomial_card_check_a.card")
    var p2 = String("/tmp/mojolearn_multinomial_card_check_b.card")
    var n = 1024
    var d = MN_COLS
    var C = 5
    var fx = _fixture(n, d, C, 1.0)
    var n1 = 0
    var iters = 0
    var moved = 0
    with DeviceContext() as ctx:
        var t1 = IdentityTrace.to_path(p1, "", True)
        t1.header(String("softmax n=") + String(n) + " d=" + String(d) + " C=" + String(C))
        var r1 = _fit(ctx, fx[0], fx[1], n, d, C, _params(1.0, True, True), t1)
        n1 = t1.seq
        iters = r1.n_iter
        var t2 = IdentityTrace.to_path(p2, "", True)
        t2.header(String("softmax n=") + String(n) + " d=" + String(d) + " C=" + String(C))
        var r2 = _fit(ctx, fx[0], fx[1], n, d, C, _params(1.0, True, True), t2)
        if t2.seq != n1:
            raise Error("check_softmax_card_is_emitted: stage count " + String(n1) + " vs " + String(t2.seq))
        for j in range(len(r1.w)):
            if not _same_bits(r1.w[j], r2.w[j]):
                moved += 1
    var want = 2 + 3 * iters + 3
    if n1 != want:
        raise Error("check_softmax_card_is_emitted: " + String(n1) + " stages, expected " + String(want) + " for n_iter " + String(iters))
    var diff = first_divergence(p1, p2)
    comptime if IDENTICAL:
        if diff != "" or moved != 0:
            raise Error("check_softmax_card_is_emitted [IDENTICAL]: control differs (" + String(moved) + " coefficients): " + diff)
        print("check_softmax_card_is_emitted OK [IDENTICAL]: " + String(n1) + " stages (n_iter " + String(iters) + "), control agrees on all")
    else:
        if diff == "":
            print("check_softmax_card_is_emitted OK [FAST]: " + String(n1) + " stages (n_iter " + String(iters) + "), control happens to agree")
        else:
            print("check_softmax_card_is_emitted REPORT [FAST]: " + String(n1) + " stages; control differs first at " + diff)


def main() raises:
    print("== glm/original/multinomial_check.mojo [" + _mode_name() + "] ==")
    check_softmax_fd_gradient()
    check_softmax_planted()
    check_softmax_is_a_minimizer()
    check_softmax_refuses_by_name()
    check_softmax_device_equals_host()
    check_softmax_signed_zero_selection()
    check_softmax_launch_invariance()
    check_softmax_host_lbfgs_replay()
    check_softmax_card_is_emitted()
