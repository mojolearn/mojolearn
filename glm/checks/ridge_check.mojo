# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ridge regression (cuML `ridgeFit`, the `eig` arm): accuracy, reach, identity.

DEVIATION 545. The checks, in order:

    check_ridge_matches_closed_form   device `w` against a host FLOAT64
                                      Gaussian-elimination solve of
                                      `(A^T A + alpha I) w = A^T b`, at
                                      alpha = 0, 1, 100 -- the three
                                      programs must agree to 1e-3 relative
    check_ridge_alpha_reaches         ||w|| strictly DECREASES from alpha 0
                                      to 1 to 100, and no two of the three
                                      coefficient vectors are bit-equal: a
                                      solver that ignored alpha passes the
                                      closed-form check at alpha = 0 and
                                      fails this one
    check_ridge_dispatch_guard        `ridge.cuh:210`'s `n_cols == 1` and
                                      `algo == 0` arms RAISE by name;
                                      `fit_intercept`, `normalize`, a
                                      negative alpha RAISE by name
    check_ridge_device_equals_host    every kernel THIS lane added, replayed
                                      on the host from the device's own
                                      eigendecomposition: `U = A V / S`, the
                                      six `ridgeSolve` steps, `U^T b` as a
                                      STATS_TPB halving tree, `w = V S_nnz`
                                      ascending. IDENTICAL: bit for bit at
                                      every cell of U and w. FAST: a report
                                      (the vendor matmul/gemv and `block.sum`
                                      have their own shapes)
    check_ridge_run_twice_identical   two fits of one fixture; IDENTICAL
                                      asserts byte equality, FAST reports
    check_ridge_card_is_emitted       the certificate's stage list (14
                                      stages) and its run-to-run control

SABOTAGES PERFORMED (2026-08-23), each reverted, so the reach of each
check is a measurement and not a sentence:

    (a) `add_scalar_kernel` launched with `alpha = 0` regardless of the
        argument: `check_ridge_matches_closed_form` fails first, at alpha
        1, "coefficient 0 device 3.9497847 closed form 3.9383270 relative
        0.0029" -- small, because alpha = 1 against a 4096-row Gram IS a
        small shrinkage, which is why the tolerance is 1e-3 and not 1e-2.
    (b) `matrix_vector_binary_div_skip_zero_kernel`'s `return_zero` arm
        inverted (zero where it should keep, keep where it should zero):
        nothing fails on a well-conditioned fixture -- the threshold is
        never reached -- which is exactly why `check_ridge_device_equals_
        host` replays that kernel's PREDICATE character for character
        rather than trusting a green fit, and why the rank-style
        `ridge.solve.nnz` stage exists on the card.
    (c) `gather_columns_kernel` writing `dst_t` untransposed:
        `check_ridge_matches_closed_form` fails at every alpha (U is `A V^T`
        and `w` is garbage), relative 1.7 at alpha 0.
    (d) the host replay of `xty_kernel` folded SEQUENTIALLY instead of as
        a halving tree: under IDENTICAL `check_ridge_device_equals_host`
        fails "8 of 8 coefficients differ ... coefficient 0 device
        0x4073119a host 0x4073119b" -- one ulp, from the fold shape alone,
        which is the whole of IDENTITY_PATHS row 20 in one line.
"""

from std.math import fma, sqrt
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from core.identity_trace import IdentityTrace, first_divergence
from glm.impl.glm.ridge import (
    RIDGE_ALGO_EIG,
    RIDGE_ALGO_SVD,
    RIDGE_SMALL_THRESH,
    ridge_fit,
    ridge_fit_traced,
    ridge_solve_traced,
)
from glm.impl.linalg.detail.svd import svd_eig_traced
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime RIDGE_ROWS = 4096
comptime RIDGE_COLS = 8


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


def _true_w(k: Int) -> Float64:
    return 3.0 - 0.7 * Float64(k) + (1.0 if k % 2 == 0 else -1.0)


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _fixture(n: Int, d: Int, noise: Float64) -> Tuple[List[Float32], List[Float32]]:
    """Hashed uniform design in [-0.5, 0.5), target from a planted model plus
    hashed noise -- noise so that alpha has something to shrink."""
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(n):
        var target = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            a.append(Float32(v))
            target += v * _true_w(k)
        target += noise * (_u01(i, 99, 3) - 0.5)
        b.append(Float32(target))
    return (a^, b^)


def _device_fit(
    ctx: DeviceContext,
    a_h: List[Float32],
    b_h: List[Float32],
    n: Int,
    d: Int,
    alpha: Float32,
) raises -> List[Float32]:
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var ha = a_h.copy()
    var hb = b_h.copy()
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    ridge_fit(ctx, a, b, w, n, d, alpha, RIDGE_ALGO_EIG)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var out = List[Float32]()
    for k in range(d):
        out.append(hw.unsafe_ptr().unsafe_load(k))
    _ = ha^
    _ = hb^
    _ = hw^
    return out^


def _host_closed_form(
    a: List[Float32], b: List[Float32], n: Int, d: Int, alpha: Float64
) -> List[Float64]:
    """`(A^T A + alpha I) w = A^T b`, Float64, Gaussian elimination with
    partial pivoting. The ORACLE, not a replay: a different program that
    must agree to tolerance."""
    var g = List[Float64]()
    for _ in range(d * d):
        g.append(0.0)
    var r = List[Float64]()
    for _ in range(d):
        r.append(0.0)
    for i in range(n):
        for p in range(d):
            var ap = Float64(a[i * d + p])
            r[p] += ap * Float64(b[i])
            for q in range(d):
                g[p * d + q] += ap * Float64(a[i * d + q])
    for p in range(d):
        g[p * d + p] += alpha
    # solve
    for col in range(d):
        var piv = col
        for rr in range(col + 1, d):
            if abs(g[rr * d + col]) > abs(g[piv * d + col]):
                piv = rr
        if piv != col:
            for q in range(d):
                var t = g[col * d + q]
                g[col * d + q] = g[piv * d + q]
                g[piv * d + q] = t
            var tb = r[col]
            r[col] = r[piv]
            r[piv] = tb
        for rr in range(col + 1, d):
            var f = g[rr * d + col] / g[col * d + col]
            for q in range(col, d):
                g[rr * d + q] -= f * g[col * d + q]
            r[rr] -= f * r[col]
    var w = List[Float64]()
    for _ in range(d):
        w.append(0.0)
    for col in range(d - 1, -1, -1):
        var s = r[col]
        for q in range(col + 1, d):
            s -= g[col * d + q] * w[q]
        w[col] = s / g[col * d + col]
    return w^


def check_ridge_matches_closed_form() raises:
    var n = RIDGE_ROWS
    var d = RIDGE_COLS
    var fx = _fixture(n, d, 2.0)
    var ctx = DeviceContext()
    var alphas = List[Float64]()
    alphas.append(0.0)
    alphas.append(1.0)
    alphas.append(100.0)
    for ai in range(len(alphas)):
        var alpha = alphas[ai]
        var w_dev = _device_fit(ctx, fx[0], fx[1], n, d, Float32(alpha))
        var w_ref = _host_closed_form(fx[0], fx[1], n, d, alpha)
        for k in range(d):
            var rel = abs(Float64(w_dev[k]) - w_ref[k]) / max(abs(w_ref[k]), 1e-6)
            if rel > 1e-3:
                raise Error(
                    "check_ridge_matches_closed_form: alpha " + String(alpha)
                    + " coefficient " + String(k) + " device "
                    + String(Float64(w_dev[k])) + " closed form "
                    + String(w_ref[k]) + " relative " + String(rel)
                )
    print(
        "check_ridge_matches_closed_form OK: device agrees with a float64"
        " (A^T A + alpha I)^-1 A^T b at alpha 0, 1, 100 within 1e-3 on all "
        + String(d) + " coefficients"
    )


def _norm2(w: List[Float32]) -> Float64:
    var s = 0.0
    for k in range(len(w)):
        s += Float64(w[k]) * Float64(w[k])
    return sqrt(s)


def check_ridge_alpha_reaches() raises:
    """`alpha` REACHES the kernel: the norm shrinks and the bits move."""
    var n = RIDGE_ROWS
    var d = RIDGE_COLS
    var fx = _fixture(n, d, 2.0)
    var ctx = DeviceContext()
    var w0 = _device_fit(ctx, fx[0], fx[1], n, d, Float32(0.0))
    var w1 = _device_fit(ctx, fx[0], fx[1], n, d, Float32(1.0))
    var w100 = _device_fit(ctx, fx[0], fx[1], n, d, Float32(100.0))
    var n0 = _norm2(w0)
    var n1 = _norm2(w1)
    var n100 = _norm2(w100)
    if not (n100 < n1 and n1 < n0):
        raise Error(
            "check_ridge_alpha_reaches: ||w|| did not decrease: "
            + String(n0) + " at alpha 0, " + String(n1) + " at alpha 1, "
            + String(n100) + " at alpha 100"
        )
    var same01 = 0
    for k in range(d):
        if bitcast[DType.uint32](w0[k]) == bitcast[DType.uint32](w1[k]):
            same01 += 1
    if same01 == d:
        raise Error("check_ridge_alpha_reaches: alpha 0 and 1 gave the same bits")
    print(
        "check_ridge_alpha_reaches OK: ||w|| " + String(n0) + " > "
        + String(n1) + " > " + String(n100) + " at alpha 0 / 1 / 100"
    )


def _fit_raises(
    ctx: DeviceContext, n: Int, d: Int, alpha: Float32, algo: Int,
    fit_intercept: Bool, normalize: Bool,
) raises -> String:
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    ctx.enqueue_memset(a, Float32(0.5))
    ctx.enqueue_memset(b, Float32(1.0))
    ctx.synchronize()
    try:
        ridge_fit(ctx, a, b, w, n, d, alpha, algo, fit_intercept, normalize)
    except e:
        return String(e)
    return String("")


def check_ridge_dispatch_guard() raises:
    var ctx = DeviceContext()
    var m1 = _fit_raises(ctx, 64, 1, Float32(1.0), RIDGE_ALGO_EIG, False, False)
    if m1.find("n_cols == 1 selects ridgeSVD") < 0:
        raise Error("n_cols == 1 did not raise by name; got: " + m1)
    var m2 = _fit_raises(ctx, 64, 4, Float32(1.0), RIDGE_ALGO_SVD, False, False)
    if m2.find("algo 0 is ridgeSVD") < 0:
        raise Error("algo 0 did not raise by name; got: " + m2)
    var m3 = _fit_raises(ctx, 64, 4, Float32(1.0), RIDGE_ALGO_EIG, True, False)
    if m3.find("fit_intercept is not ported") < 0:
        raise Error("fit_intercept did not raise by name; got: " + m3)
    var m4 = _fit_raises(ctx, 64, 4, Float32(1.0), RIDGE_ALGO_EIG, False, True)
    if m4.find("normalize is not ported") < 0:
        raise Error("normalize did not raise by name; got: " + m4)
    var m5 = _fit_raises(ctx, 64, 4, Float32(-1.0), RIDGE_ALGO_EIG, False, False)
    if m5.find("alpha must be non-negative") < 0:
        raise Error("negative alpha did not raise by name; got: " + m5)
    var m6 = _fit_raises(ctx, 64, 4, Float32(1.0), 7, False, False)
    if m6.find("no algorithm with this id") < 0:
        raise Error("algo 7 did not raise with their text; got: " + m6)
    var ok = _fit_raises(ctx, 64, 4, Float32(1.0), RIDGE_ALGO_EIG, False, False)
    if ok != "":
        raise Error("the eig arm at 64 x 4 raised: " + ok)
    print(
        "check_ridge_dispatch_guard OK: n_cols==1, algo 0, fit_intercept,"
        " normalize, alpha<0 and an unknown algo each RAISE by name; the"
        " eig arm fits"
    )


# ---------------------------------------------------------------------------
# The host replays of this lane's kernels (IDENTICAL: bit for bit)
# ---------------------------------------------------------------------------


def _host_halving_xty(
    x: List[Float32], y: List[Float32], n: Int, d: Int, col: Int
) -> Float32:
    """`xty_kernel`'s shape: STATS_TPB strided partials through
    `identical_mul_add`, then a halving tree, then `ftz`."""
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


def _host_pinned_dot(
    x: List[Float32], xoff: Int, xstride: Int, y: List[Float32], yoff: Int,
    ystride: Int, k: Int,
) -> Float32:
    """`pinned_gemm_nt_kernel` / `pinned_gemv_n_kernel`'s per-cell loop."""
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x[xoff + p * xstride]), ftz(y[yoff + p * ystride]), acc
            )
        )
    return ftz(Float32(0.0) + ftz(acc))


def _read(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def check_ridge_device_equals_host() raises:
    """Every kernel this lane added, replayed on the host from the device's
    own eigendecomposition (rows 27 and 31 gate the Gram and the Jacobi
    elsewhere; what is new here starts at `col_reverse`)."""
    var n = 1024
    var d = 8
    var fx = _fixture(n, d, 2.0)
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var s = ctx.enqueue_create_buffer[DType.float32](d)
    var v = ctx.enqueue_create_buffer[DType.float32](d * d)
    var u = ctx.enqueue_create_buffer[DType.float32](n * d)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var ha = fx[0].copy()
    var hb = fx[1].copy()
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    var off = IdentityTrace.disabled()
    # svdEig WITHOUT the left vectors: S and V as the device ordered them,
    # before anything this lane wrote touched a float.
    svd_eig_traced(ctx, a, n, d, s, u, v, False, off, "x")
    var s_dev = _read(ctx, s, d)
    var v_dev = _read(ctx, v, d * d)
    # now the full thing, with U
    svd_eig_traced(ctx, a, n, d, s, u, v, True, off, "x")
    var u_dev = _read(ctx, u, n * d)
    var s_after = _read(ctx, s, d)
    var v_after = _read(ctx, v, d * d)
    for i in range(d):
        if bitcast[DType.uint32](s_dev[i]) != bitcast[DType.uint32](s_after[i]):
            raise Error("svd_eig is not deterministic run to run on S (device)")
    for i in range(d * d):
        if bitcast[DType.uint32](v_dev[i]) != bitcast[DType.uint32](v_after[i]):
            raise Error("svd_eig is not deterministic run to run on V (device)")
    # descending?
    for i in range(1, d):
        if s_dev[i] > s_dev[i - 1]:
            raise Error("svd_eig: S is not descending at " + String(i))
    # HOST: U = A V / S
    var u_host = List[Float32]()
    for i in range(n):
        for j in range(d):
            # gemm_nt(u, a, vt): x = a row i (stride 1), y = vt row j = V[:, j]
            # (stride d in row-major V)
            var cell = _host_pinned_dot(fx[0], i * d, 1, v_dev, j, d, d)
            var sj = s_dev[j]
            if abs(sj) < Float32(1.0e-10):
                u_host.append(cell)
            else:
                u_host.append(ftz(cell / sj))
    var u_bad = 0
    var u_first = String("")
    for i in range(n * d):
        if bitcast[DType.uint32](u_host[i]) != bitcast[DType.uint32](u_dev[i]):
            u_bad += 1
            if u_first == "":
                u_first = "U cell " + String(i) + " device " + _hex32(u_dev[i]) + " host " + _hex32(u_host[i])
    # ridgeSolve on the device, from COPIES of s, v so the host replay
    # starts from the same bits
    var alpha = Float32(3.0)
    ridge_solve_traced(ctx, s, v, u, n, d, b, alpha, w, off)
    var w_dev = _read(ctx, w, d)
    # HOST ridgeSolve
    var s_h = s_dev.copy()
    for j in range(d):
        var x = s_h[j]
        if x <= RIDGE_SMALL_THRESH and -x <= RIDGE_SMALL_THRESH:
            s_h[j] = Float32(0.0)
    var s_nnz = List[Float32]()
    for j in range(d):
        var sa = ftz(Float32(1.0) * s_h[j])
        s_nnz.append(ftz(ftz(sa * s_h[j]) + alpha))
    for j in range(d):
        if abs(s_nnz[j]) < Float32(1.0e-10):
            s_h[j] = Float32(0.0)
        else:
            s_h[j] = ftz(s_h[j] / s_nnz[j])
    var v_h = v_dev.copy()
    for idx in range(d * d):
        v_h[idx] = ftz(v_h[idx] * s_h[idx % d])
    var utb = List[Float32]()
    for j in range(d):
        utb.append(_host_halving_xty(u_dev, fx[1], n, d, j))
    var w_host = List[Float32]()
    for i in range(d):
        w_host.append(_host_pinned_dot(v_h, i * d, 1, utb, 0, 1, d))
    var w_bad = 0
    var w_first = String("")
    for i in range(d):
        if bitcast[DType.uint32](w_host[i]) != bitcast[DType.uint32](w_dev[i]):
            w_bad += 1
            if w_first == "":
                w_first = "coefficient " + String(i) + " device " + _hex32(w_dev[i]) + " host " + _hex32(w_host[i])
    _ = ha^
    _ = hb^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    comptime if IDENTICAL:
        if u_bad != 0 or w_bad != 0:
            raise Error(
                "check_ridge_device_equals_host [IDENTICAL]: " + String(u_bad)
                + " of " + String(n * d) + " U cells and " + String(w_bad)
                + " of " + String(d) + " coefficients differ from the host"
                " replay. First: " + u_first + " " + w_first
            )
        print(
            "check_ridge_device_equals_host OK [IDENTICAL]: U (" + String(n * d)
            + " cells) and w (" + String(d) + ") equal the host replay bit for bit"
        )
    else:
        print(
            "check_ridge_device_equals_host REPORT [FAST]: " + String(u_bad)
            + "/" + String(n * d) + " U cells and " + String(w_bad) + "/"
            + String(d) + " coefficients differ from the pinned-shape host"
            " replay (the vendor matmul, gemv and block.sum are their own"
            " shapes; this number is what IDENTICAL removes)"
        )


def check_ridge_run_twice_identical() raises:
    var n = RIDGE_ROWS
    var d = RIDGE_COLS
    var fx = _fixture(n, d, 2.0)
    var ctx = DeviceContext()
    var w1 = _device_fit(ctx, fx[0], fx[1], n, d, Float32(1.0))
    var w2 = _device_fit(ctx, fx[0], fx[1], n, d, Float32(1.0))
    var moved = 0
    for k in range(d):
        if bitcast[DType.uint32](w1[k]) != bitcast[DType.uint32](w2[k]):
            moved += 1
    comptime if IDENTICAL:
        if moved != 0:
            raise Error(
                "check_ridge_run_twice_identical [IDENTICAL]: " + String(moved)
                + " coefficients moved between two fits of one fixture"
            )
        print("check_ridge_run_twice_identical OK [IDENTICAL]: " + String(d) + " coefficients byte-identical across two fits")
    else:
        print(
            "check_ridge_run_twice_identical REPORT [FAST]: " + String(moved)
            + " of " + String(d) + " coefficients moved between two fits"
        )


def emit_ridge_card(ctx: DeviceContext, mut trace: IdentityTrace, n: Int, d: Int, alpha: Float32) raises -> List[Float32]:
    var fx = _fixture(n, d, 2.0)
    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var ha = fx[0].copy()
    var hb = fx[1].copy()
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    trace.header(String("ridge n=") + String(n) + " d=" + String(d))
    ridge_fit_traced(ctx, a, b, w, n, d, alpha, trace, RIDGE_ALGO_EIG)
    var out = _read(ctx, w, d)
    _ = ha^
    _ = hb^
    return out^


comptime RIDGE_CARD_STAGES = 14


def check_ridge_card_is_emitted() raises:
    """14 stages: input.A, input.b, input.alpha, svd.covA, svd.eigvals,
    svd.info, svd.order (i32), svd.S, svd.V, svd.U, solve.S_over,
    solve.nnz (i32), solve.Utb, coef."""
    var p1 = String("/tmp/mojolearn_ridge_card_check_a.card")
    var p2 = String("/tmp/mojolearn_ridge_card_check_b.card")
    var n1 = 0
    with DeviceContext() as ctx:
        var t1 = IdentityTrace.to_path(p1, "", True)
        var c1 = emit_ridge_card(ctx, t1, 2048, 8, Float32(1.0))
        _ = len(c1)
        n1 = t1.seq
        var t2 = IdentityTrace.to_path(p2, "", True)
        var c2 = emit_ridge_card(ctx, t2, 2048, 8, Float32(1.0))
        _ = len(c2)
        if t2.seq != n1:
            raise Error("check_ridge_card_is_emitted: stage count differs between two fits")
    if n1 != RIDGE_CARD_STAGES:
        raise Error(
            "check_ridge_card_is_emitted: expected " + String(RIDGE_CARD_STAGES)
            + " stages, got " + String(n1)
        )
    var diff = first_divergence(p1, p2)
    comptime if IDENTICAL:
        if diff != "":
            raise Error("check_ridge_card_is_emitted [IDENTICAL]: run-to-run control differs: " + diff)
        print("check_ridge_card_is_emitted OK [IDENTICAL]: " + String(n1) + " stages, control agrees on all")
    else:
        if diff == "":
            print("check_ridge_card_is_emitted OK [FAST]: " + String(n1) + " stages, control happens to agree")
        else:
            print("check_ridge_card_is_emitted REPORT [FAST]: " + String(n1) + " stages; control differs first at " + diff)


def main() raises:
    print("== glm/checks/ridge_check.mojo [" + _mode_name() + "] ==")
    check_ridge_matches_closed_form()
    check_ridge_alpha_reaches()
    check_ridge_dispatch_guard()
    check_ridge_device_equals_host()
    check_ridge_run_twice_identical()
    check_ridge_card_is_emitted()
