# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGDOneClassSVM's tail replay (lane/apple-fast-w2-sgdoc, 2026-10-04):
MOJOLEARN_SGDOC_FAST_TAIL, FAST + Apple, opt-in (CANDIDATE, off on main).

Why the tail is the whole answer for learning_rate='optimal'. sgd_one with
eta_t = 1 / (alpha (t0 + t)) decays w by (1 - eta_t alpha) =
(t0 + t - 1) / (t0 + t) each step before the update eta_t g_t (g_t = x_i
when the row violates the margin, else 0), so the product telescopes and

    w_T = S_T / (alpha (t0 + T)),          S_T = sum_t g_t,
    rho_T = R_T / (alpha (t0 + T)),        R_t ~ R_{t-1} + alpha - v_t

(v_t the violation bit; rho = 1 - intercept, R's own 1/(t0 + t) growth term is
below 1e-7 per step at the board's t). Row i violates iff S.x_i < R: in
(S, R) the chain does not depend on t at all, and the decision function
(S.x - R) / (alpha (t0 + T)) has the sign of S.x - R. When the optimum is
w* = 0 (0 inside the reduced convex hull, caps 1 / (nu n): centered columns,
the board's standardized cls block) S has no drift away from 0, only the
selection's pull toward it (violators lie on the low side of S), so (S, R)
is a stationary chain that forgets its start within its mixing time
(taxi d 11: tens of steps; Istella d 220: hundreds to thousands along the
leading directions). scikit-learn's 20 epochs and main's 20 epochs at
tol=None end at one draw of that chain: a noise direction w ~ 1e-6, its
flagged fraction anywhere in 0.007 .. 0.093 on the board (sklearn taxi
0.00702 vs main 0.05557; Istella 0.09334 vs 0.04042), which no parallel
optimizer reproduces (lane/apple-fast-sgdoc-parallel's converged solution
w = 0 flags none: HOLD).

So the candidate runs main's own per-sample kernels, same statements, same
epoch orders (`sgd_perm_kernel` of the real epoch), same t (eta of the real
step), over the LAST K steps only, from w = 0, intercept = 1: an exact suffix
of main's schedule from the zero state, whose final (S, R) has the full run's
law once K exceeds the mixing time. Everything else (20 x n steps of
burn-in) is skipped. The gate below refuses data that are not centered:
there w* != 0, S_T grows like T and a short tail would give the wrong scale,
so those fits keep main's full run, bit for bit.

Gate (all, else main): one class, learning_rate optimal, penalty l2,
fit_intercept, shuffle, tol=None (no objective / no early stop), no sample
weights, max_iter n > K, and every column's mean within SGDOC_CENTER_TOL of
its standard deviation (`sgdoc_centered`, two grid kernels and one word home;
a constant column must be exactly 0)."""
from std.gpu import block_idx, thread_idx
from std.atomic import Atomic
from max.gpu.host import DeviceContext
from x_linear.ops import FP, IP, ld, st, fabs, fsqrt

comptime SGDOC_TAIL_TPB = 256
comptime SGDOC_MOM_BLOCKS = 512
comptime SGDOC_CENTER_TOL = Float32(1e-3)
"""|mean_j| <= 1e-3 sd_j for every column j: float32 standardization leaves
~1e-7; raw (uncentered) features fail by orders of magnitude."""


def sgdoc_mom_part_kernel(x: FP, n: Int32, d: Int32, g: Int32, part: FP):
    """Block b: the column sums and sums of squares of rows
    [b r, min(n, (b + 1) r)), r = ceil(n / g); thread j column j (+ TPB ...),
    rows ascending. part[2 (b d + j) ..] = sum, sum of squares."""
    var b = Int(block_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var rpb = (nn + Int(g) - 1) // Int(g)
    var r0 = b * rpb
    var r1 = min(nn, r0 + rpb)
    for j in range(Int(thread_idx.x), dd, SGDOC_TAIL_TPB):
        var s = Float32(0)
        var s2 = Float32(0)
        for i in range(r0, r1):
            var v = ld(x, i * dd + j)
            s += v
            s2 += v * v
        st(part, 2 * (b * dd + j), s)
        st(part, 2 * (b * dd + j) + 1, s2)


def sgdoc_mom_flag_kernel(part: FP, n: Int32, d: Int32, g: Int32, flag: IP):
    """One block: thread j folds column j's g partials ascending; adds one to
    flag[0] when the column is not centered (or not finite)."""
    var dd = Int(d)
    var nf = Float32(Int(n))
    for j in range(Int(thread_idx.x), dd, SGDOC_TAIL_TPB):
        var s = Float32(0)
        var s2 = Float32(0)
        for b in range(Int(g)):
            s += ld(part, 2 * (b * dd + j))
            s2 += ld(part, 2 * (b * dd + j) + 1)
        var mu = s / nf
        var var_ = s2 / nf - mu * mu
        var sd = fsqrt(var_) if var_ > Float32(0) else Float32(0)
        var finite = s == s and s2 == s2 and fabs(s) < Float32(3.0e38) and fabs(s2) < Float32(3.0e38)
        if not finite or fabs(mu) > SGDOC_CENTER_TOL * sd:
            _ = Atomic[DType.int32].fetch_add(flag, Int32(1))


def sgdoc_centered(ctx: DeviceContext, x: FP, n: Int, d: Int) raises -> Bool:
    """True when every column of the n x d row-major x (on the device) has
    |mean| <= SGDOC_CENTER_TOL sd: the tail replay's gate. One word home."""
    if n < 1 or d < 1:
        return False
    var g = min(SGDOC_MOM_BLOCKS, n)
    var dpart = ctx.enqueue_create_buffer[DType.float32](2 * g * d)
    var dflag = ctx.enqueue_create_buffer[DType.int32](1)
    dflag.enqueue_fill(Int32(0))
    ctx.enqueue_function[sgdoc_mom_part_kernel](
        x, Int32(n), Int32(d), Int32(g), dpart.unsafe_ptr(), grid_dim=g, block_dim=SGDOC_TAIL_TPB,
    )
    ctx.enqueue_function[sgdoc_mom_flag_kernel](
        dpart.unsafe_ptr(), Int32(n), Int32(d), Int32(g), dflag.unsafe_ptr(), grid_dim=1, block_dim=SGDOC_TAIL_TPB,
    )
    var h = List[Int32](length=1, fill=Int32(1))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dflag)
    ctx.synchronize()
    var ok = h[0] == 0
    _ = dpart^
    _ = dflag^
    _ = h^
    return ok
