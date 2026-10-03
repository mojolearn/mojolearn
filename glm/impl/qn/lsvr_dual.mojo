# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-linsvr, -D MOJOLEARN_LSVR_DUAL_CD (QN_FAST_DUAL_CD): a
parallel dual coordinate-descent warm start for LinearSVR on the device,
then the shipped L-BFGS from that point to its own tolerance.

Compiled and called under FAST on Apple only (`qn_solvers.qn_minimize`
calls `lsvr_dual_warm_start` inside `comptime if QN_FAST_DUAL_CD`);
IDENTICAL never references this module's functions.

THE DUAL. The objective the QN solver minimizes is
`F(w, b) = (1/N) sum L(y_i - w.x_i - b) + (l2 / 2) ||w||^2`; divided by l2
it is liblinear's `0.5 ||w||^2 + C' sum L_i` with `C' = 1 / (N l2)`. Its
dual (liblinear `solve_l2r_l1l2_svr`) is

    min_beta  0.5 ||sum_i beta_i x_i||^2 - y.beta + eps ||beta||_1
              + 0.5 lam ||beta||^2,      -U <= beta_i <= U,
    epsilon-insensitive (QN_LOSS_SVR_L1): lam = 0,        U = C'
    squared  (QN_LOSS_SVR_L2):            lam = 1 / (2C'), U = inf

and `w = sum beta_i x_i`. The intercept is NOT penalized in F; an
unpenalized bias puts `sum beta_i = 0` on the dual, which coordinate
descent cannot keep one coordinate at a time. Here the bias is liblinear's
augmented feature (value 1, so it carries a `0.5 b^2` penalty, negligible
against C' * N loss terms); the L-BFGS polish then removes that penalty.

WHY NOT liblinear's CD AS IS, AND WHAT RUNS INSTEAD. liblinear updates one
coordinate at a time against the CURRENT w. Updating many coordinates at
once against the same w diverges on dense data: every row of taxi shares
all 11 features, so the parallel steps add up along the same directions
(the safe damping of a Jacobi / ESO step is about the number of rows
updated at once). The closest correct parallel form, CoCoA-style with a
line search:

    dcd_local_kernel   thread t owns DCD_RPT rows (row-interleaved in the
                       block, as the fused objective pass); it commits the
                       previous round's accepted step to its rows' beta,
                       then runs liblinear's exact sequential CD update over
                       its rows against a REGISTER copy of w (w kept on the
                       device, never on the host), and emits its tile's
                       delta-w and, for every candidate global step gamma_c,
                       the change of the row-separable dual terms
    dcd_fold_kernel    folds the tiles (one block per output)
    dcd_pick_kernel    adds the quadratic term's change
                       `gamma (w.dw) + 0.5 gamma^2 ||dw||^2` and takes the
                       candidate with the smallest dual objective change;
                       none negative -> gamma = 0 (no move); w += gamma* dw

Candidates: gamma = 2^0 .. 2^-(DCD_G - 2) and 1/K (K = tiles). The
guarantee: CoCoA averaging (gamma = 1/K with exact local CD steps) decreases
the dual every round with a linear rate, and by convexity along the ray
every gamma in (0, 1/K] decreases it too; the pick is never worse than
gamma = 1/K, so the rounds inherit CoCoA's rate, and gamma = 1 is taken
whenever the tiles do not conflict. Monotone, no divergence. Candidate
changes are computed as DIFFERENCES from the current point, so the float32
fold compares small numbers, not two near-equal totals.

SHRINKING: not done. A row is 4 * d bytes inside a row-coalesced tile;
skipping a bound row saves its ALU, not its memory traffic, and the pass is
bandwidth-bound.

ROUNDS: DCD_ROUNDS = 16 fixed, three launches each, no synchronize; then
one synchronize, `w` copied into the L-BFGS start, and `min_lbfgs` runs to
the shipped tolerance on the shipped objective: the fitted answer is
certified by the same convergence test as without the define. The dual
phase only moves the starting point (the caller's w0 is ignored).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace

from checks.numerics import ftz
from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from core.strided_walk import strided_ftz_sum
from glm.impl.qn.glm_base import GLMWithData, QN_FAST_DUAL_CD, QNF_MAX_D, _qnb_barrier
from glm.impl.linear_model.qn import QN_LOSS_SVR_L1, QN_LOSS_SVR_L2

comptime DCD_TPB = 256
comptime DCD_RPT = 16
#: candidate global steps per round
comptime DCD_G = 20
comptime DCD_ROUNDS = 16
comptime DCD_PICK_TPB = 64


def dcd_blocks(n: Int) -> Int:
    return (n + DCD_TPB * DCD_RPT - 1) // (DCD_TPB * DCD_RPT)


def dcd_tiles(n: Int) -> Int:
    return dcd_blocks(n) * DCD_TPB


def dcd_local_kernel[DMAX: Int](
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    dbeta: MutPointer[Float32, MutAnyOrigin],
    wt: MutPointer[Float32, MutAnyOrigin],
    gam: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    fit_intercept: Int32,
    eps: Float32,
    lam: Float32,
    ub: Float32,
    inv_k: Float32,
    tiles_in: Int32,
):
    """One round's local solve (see the module docstring). Outputs per tile
    at `part[o * tiles + tile]`: o < d the delta of weight o, o == d (with
    an intercept) the bias delta, then DCD_G candidate row-term changes at
    o = dt + c, dt = d + fit_intercept."""
    var n = Int(n_in)
    var d = Int(d_in)
    var tiles = Int(tiles_in)
    var fi = Int(fit_intercept)
    var dt = d + fi
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var tile = blk * DCD_TPB + tid
    var gs = gam.unsafe_load(0)
    var fb = Float32(1.0) if fi != 0 else Float32(0.0)
    var wl = InlineArray[Float32, DMAX](fill=Float32(0.0))
    var w0 = InlineArray[Float32, DMAX](fill=Float32(0.0))
    comptime for j in range(DMAX):
        if j < d:
            wl[j] = wt.unsafe_load(j)
            w0[j] = wl[j]
    var wb = Float32(0.0)
    if fi != 0:
        wb = wt.unsafe_load(d)
    var wb0 = wb
    var gl = InlineArray[Float32, DCD_G](fill=Float32(0.0))
    var stp = Float32(1.0)
    comptime for c in range(DCD_G - 1):
        gl[c] = stp
        stp = stp * Float32(0.5)
    gl[DCD_G - 1] = inv_k
    var acc = InlineArray[Float32, DCD_G](fill=Float32(0.0))
    var r = blk * (DCD_TPB * DCD_RPT) + tid
    for _ in range(DCD_RPT):
        if r < n:
            var xr = InlineArray[Float32, DMAX](fill=Float32(0.0))
            var zi = wb * fb
            var h = fb * fb + lam
            comptime for j in range(DMAX):
                if j < d:
                    xr[j] = x.unsafe_load(r * d + j)
                    zi = xr[j] * wl[j] + zi
                    h = xr[j] * xr[j] + h
            var yi = y.unsafe_load(r)
            # commit the previous round's accepted global step
            var b = gs * dbeta.unsafe_load(r) + beta.unsafe_load(r)
            # liblinear solve_l2r_l1l2_svr's coordinate step
            var gr = zi - yi + lam * b
            var gp = gr + eps
            var gn = gr - eps
            var hb = h * b
            var z = Float32(0.0)
            if h > Float32(0.0):
                if gp < hb:
                    z = -gp / h
                elif gn > hb:
                    z = -gn / h
                else:
                    z = -b
            var bn = min(max(b + z, -ub), ub)
            var dd = ftz(bn - b)
            comptime for j in range(DMAX):
                if j < d:
                    wl[j] = xr[j] * dd + wl[j]
            wb = fb * dd + wb
            beta.unsafe_store(r, b)
            dbeta.unsafe_store(r, dd)
            var ab = abs(b)
            comptime for c in range(DCD_G):
                var dc = gl[c] * dd
                var t = -yi * dc + eps * (abs(b + dc) - ab) + lam * (b * dc + Float32(0.5) * dc * dc)
                acc[c] = acc[c] + t
        r += DCD_TPB
    comptime for j in range(DMAX):
        if j < d:
            part.unsafe_store(j * tiles + tile, ftz(wl[j] - w0[j]))
    if fi != 0:
        part.unsafe_store(d * tiles + tile, ftz(wb - wb0))
    comptime for c in range(DCD_G):
        part.unsafe_store((dt + c) * tiles + tile, acc[c])


def dcd_fold_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dsum: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    dt_in: Int32,
    tiles_in: Int32,
):
    """Block o folds output o's tile partials (lane t: tiles t, t +
    STATS_TPB, ...; the block fold): o < dt into dw[o], else the candidate
    row-term change into dsum[o - dt]."""
    var dt = Int(dt_in)
    var tiles = Int(tiles_in)
    var o = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, o * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        if o < dt:
            dw.unsafe_store(o, s0)
        else:
            dsum.unsafe_store(o - dt, s0)


def dcd_pick_kernel(
    wt: MutPointer[Float32, MutAnyOrigin],
    dw: MutPointer[Float32, MutAnyOrigin],
    dsum: MutPointer[Float32, MutAnyOrigin],
    gam: MutPointer[Float32, MutAnyOrigin],
    dt_in: Int32,
    inv_k: Float32,
):
    """Thread c < DCD_G prices candidate c's dual change: its row-term change
    plus `gamma_c (w.dw) + 0.5 gamma_c^2 ||dw||^2` (dt <= QNF_MAX_D + 1
    terms, a constant bound). Thread 0 takes the smallest; not negative ->
    gamma 0. Then thread j < dt: w[j] += gamma* dw[j]. gam[0] carries
    gamma* to the next round's local kernel, which commits it to beta."""
    var dt = Int(dt_in)
    var tid = Int(thread_idx.x)
    var price = stack_allocation[
        DCD_G, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var pick = stack_allocation[
        1, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    if tid < DCD_G:
        var g = Float32(1.0)
        if tid == DCD_G - 1:
            g = inv_k
        else:
            for _ in range(tid):
                g = g * Float32(0.5)
        var wd = Float32(0.0)
        var dd = Float32(0.0)
        for j in range(dt):
            var dj = dw.unsafe_load(j)
            wd = wt.unsafe_load(j) * dj + wd
            dd = dj * dj + dd
        price[tid] = dsum.unsafe_load(tid) + g * wd + Float32(0.5) * g * g * dd
    _qnb_barrier()
    if tid == 0:
        var best = Float32(0.0)
        var gbest = Float32(0.0)
        var g = Float32(1.0)
        for c in range(DCD_G):
            var gc = inv_k if c == DCD_G - 1 else g
            var pc = price[c]
            if pc < best:
                best = pc
                gbest = gc
            g = g * Float32(0.5)
        pick[0] = gbest
        gam.unsafe_store(0, gbest)
    _qnb_barrier()
    var gs = pick[0]
    if tid < dt:
        wt.unsafe_store(tid, ftz(gs * dw.unsafe_load(tid) + wt.unsafe_load(tid)))


def lsvr_dual_applies(f: GLMWithData) -> Bool:
    """The dual warm start serves this fit: FAST on Apple under the define,
    an SVR loss, one target, 1 <= d <= QNF_MAX_D (registers), l2 > 0 (the
    dual needs the penalty)."""
    comptime if not QN_FAST_DUAL_CD:
        return False
    else:
        return (
            (f.loss == QN_LOSS_SVR_L1 or f.loss == QN_LOSS_SVR_L2)
            and f.dims.C == 1
            and f.dims.D >= 1
            and f.dims.D <= QNF_MAX_D
            and f.l2 > Float32(0.0)
            and f.dims.n_param <= DCD_PICK_TPB
        )


def lsvr_dual_warm_start(
    ctx: DeviceContext,
    mut f: GLMWithData,
    mut w: DeviceBuffer[DType.float32],
) raises:
    """DCD_ROUNDS rounds of the damped parallel dual CD (module docstring),
    all on the device, then w (the L-BFGS start, n_param words: the D
    weights, then the bias) <- the dual's w. One synchronize at the end."""
    comptime if not QN_FAST_DUAL_CD:
        raise Error("qn: lsvr_dual_warm_start is compiled under MOJOLEARN_LSVR_DUAL_CD only")
    else:
        var n = f.n_rows
        var d = f.dims.D
        var fi = 1 if f.dims.fit_intercept else 0
        var dt = d + fi
        var tiles = dcd_tiles(n)
        var nb = dcd_blocks(n)
        var c64 = 1.0 / (Float64(n) * Float64(f.l2))
        var lam = Float32(0.0)
        var ub = Float32(c64)
        if f.loss == QN_LOSS_SVR_L2:
            lam = Float32(0.5 / c64)
            ub = Float32(3.0e38)
        var inv_k = Float32(1.0 / Float64(tiles))
        var beta = ctx.enqueue_create_buffer[DType.float32](n)
        var dbeta = ctx.enqueue_create_buffer[DType.float32](n)
        var part = ctx.enqueue_create_buffer[DType.float32](tiles * (dt + DCD_G))
        var wt = ctx.enqueue_create_buffer[DType.float32](dt)
        var dw = ctx.enqueue_create_buffer[DType.float32](dt)
        var dsum = ctx.enqueue_create_buffer[DType.float32](DCD_G)
        var gam = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.enqueue_memset(beta, Float32(0.0))
        ctx.enqueue_memset(dbeta, Float32(0.0))
        ctx.enqueue_memset(wt, Float32(0.0))
        ctx.enqueue_memset(gam, Float32(0.0))
        for _ in range(DCD_ROUNDS):
            if d <= 16:
                ctx.enqueue_function[dcd_local_kernel[16]](
                    f.x.unsafe_ptr(), f.y.unsafe_ptr(), beta.unsafe_ptr(),
                    dbeta.unsafe_ptr(), wt.unsafe_ptr(), gam.unsafe_ptr(),
                    part.unsafe_ptr(), Int32(n), Int32(d), Int32(fi),
                    f.svr_eps, lam, ub, inv_k, Int32(tiles),
                    grid_dim=(nb, 1, 1), block_dim=(DCD_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[dcd_local_kernel[QNF_MAX_D]](
                    f.x.unsafe_ptr(), f.y.unsafe_ptr(), beta.unsafe_ptr(),
                    dbeta.unsafe_ptr(), wt.unsafe_ptr(), gam.unsafe_ptr(),
                    part.unsafe_ptr(), Int32(n), Int32(d), Int32(fi),
                    f.svr_eps, lam, ub, inv_k, Int32(tiles),
                    grid_dim=(nb, 1, 1), block_dim=(DCD_TPB, 1, 1),
                )
            ctx.enqueue_function[dcd_fold_kernel](
                dw.unsafe_ptr(), dsum.unsafe_ptr(), part.unsafe_ptr(),
                Int32(dt), Int32(tiles),
                grid_dim=(dt + DCD_G, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            ctx.enqueue_function[dcd_pick_kernel](
                wt.unsafe_ptr(), dw.unsafe_ptr(), dsum.unsafe_ptr(),
                gam.unsafe_ptr(), Int32(dt), inv_k,
                grid_dim=(1, 1, 1), block_dim=(DCD_PICK_TPB, 1, 1),
            )
        var wsub = w.create_sub_buffer[DType.float32](0, dt)
        ctx.enqueue_copy(dst_buf=wsub, src_buf=wt)
        ctx.synchronize()
        _ = wsub^
        _ = len(beta)
        _ = len(dbeta)
        _ = len(part)
        _ = len(wt)
        _ = len(dw)
        _ = len(dsum)
        _ = len(gam)
