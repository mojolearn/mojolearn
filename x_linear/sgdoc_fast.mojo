# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-sgdoc-parallel (2026-10-03): SGDOneClassSVM solved to
convergence on the grid, FAST + Apple, behind -D MOJOLEARN_SGDOC_FAST_PAR.

THE PROBLEM. scikit-learn's SGDOneClassSVM (`_fit_one_class`: y = 1, hinge,
l2, alpha = nu, intercept = 1 - offset, the intercept step `- eta * alpha`)
minimizes, per sample summed over an epoch,

    J(w, rho) = nu/2 |w|^2 + nu (1 - rho) + 1/n sum_i max(0, rho - w.x_i)

(x_linear/sgd.mojo `sgd_one`'s objective words: the hinge `oc_hinge`, the
penalty alpha/2 |w|^2, `objective += intercept * alpha`). That is nu times
the linear one-class SVM primal 1/2 |w|^2 - rho + 1/(nu n) sum (rho - w.x)_+
plus the constant nu. Its dual is the minimum-norm point of the reduced
convex hull

    min_u 1/2 |u|^2,  u = sum_i b_i x_i,  0 <= b_i <= 1/(nu n),  sum b_i = 1,

with w = u at the optimum, J* = nu - nu/2 |u*|^2.

THE SOLVER (Frank-Wolfe on that dual, exact line search). From u0 = the
column mean (b_i = 1/n, feasible since nu <= 1), each iteration is

  1. scores sc_i = u . x_i                       (one pass over X, every row)
  2. tau = the R-th smallest score, R = ceil(nu n)   (an 8-pass 4-bit radix
     select: per-block digit counts, a 16-bin pick; no atomics)
  3. s = the linear minimizer of <u, .> over the hull: weight 1/(nu n) on
     every row below tau, the remaining mass spread evenly over the rows AT
     tau (a masked column sum over X, per-block partials folded ascending)
  4. gap g = <u, u - s> (the certified duality gap / nu); stop when
     min(g, |u|^2 / 2) <= SF_EPS * max(|u|^2, |s|^2) (|u|^2 / 2 certifies
     the zero primal) or after the iteration cap; else
     u += gamma (s - u), gamma = clip(g / |u - s|^2, 0, 1).

Every pass is a grid kernel; the small d-vector step is one block (O(d)
work). The host enqueues SF_BATCH iterations and reads one stop word per
batch; launches after the stop word return at once.

THE PRIMAL. For the final u, the best offset is rho = tau (the nu-quantile:
#{sc < tau} <= nu n <= #{sc <= tau}), and the primal along the ray t u is
P(t) = nu + nu/2 t^2 |u|^2 - nu t <u, s>, minimized at
t = max(0, <u, s>) / |u|^2. The fit returns coef = t u, offset = t tau:
never worse than P(u) and never worse than (w, rho) = (0, 0), J = nu.
On standardized X (the board's cls block: columns centered on the fit
rows) the column mean IS the origin, so u0 = 0 up to float32 rounding, the
gap certifies optimality at once, and t = 0: the exact optimum is
w = 0, rho = 0, J* = nu, which flags no row (decision 0 is an inlier).
scikit-learn's 20 epochs (tol=None) stop short of it (J > nu) with a
noise direction that flags 4-6 % of the held-out rows.

Deterministic: every sum is a fixed-order chain or tree; no atomics.
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_linear.ops import FP, IP, ld, st, i2f

comptime SGDOC_FAST_PAR = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                           and is_defined["MOJOLEARN_SGDOC_FAST_PAR"]())

comptime SF_TPB = 256
comptime SF_SEL_ROWS = 4096
"""Rows a block counts per radix pass."""
comptime SF_COL_ROWS = 1024
"""Rows a block sums in the masked column sum."""
comptime SF_BATCH = 8
"""Frank-Wolfe iterations queued per stop-word read."""
comptime SF_EPS = Float32(1e-6)
"""Relative duality-gap tolerance: g <= SF_EPS * max(|u|^2, |s|^2)."""
comptime SF_ITERS_PER_EPOCH = 50
"""The iteration cap is SF_ITERS_PER_EPOCH * max_iter (at least 50)."""
comptime SF_ST = 8
"""State words: 0 done, 1 iterations, 2 |u|^2, 3 <u,s>, 4 gap, 5 tau, 6 gamma, 7 |s|^2."""


@always_inline
def _key(v: Float32) -> UInt32:
    """The monotone uint32 image of a float (-0.0 as +0.0)."""
    var w = v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    return ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)


@always_inline
def _unkey(k: UInt32) -> Float32:
    var ub = k & UInt32(0x7FFFFFFF) if (k >> 31) == 1 else k ^ UInt32(0xFFFFFFFF)
    return bitcast[DType.float32](ub)


@always_inline
def _done(stt: FP) -> Bool:
    return ld(stt, 0) != Float32(0)


def sf_score_kernel(x: FP, n: Int32, d: Int32, u: FP, sc: FP, stt: FP, gl: Int32):
    """sc[i] = u . x_i, gl lanes a row (1 or 32; SF_TPB / gl rows a block),
    each lane's columns ascending, the lanes folded by a shared tree."""
    if _done(stt):
        return
    var nn = Int(n)
    var dd = Int(d)
    var g = Int(gl)
    var tid = Int(thread_idx.x)
    var rpb = SF_TPB // g
    var lane = tid % g
    var i = Int(block_idx.x) * rpb + tid // g
    var acc = Float32(0)
    if i < nn:
        var j = lane
        while j < dd:
            acc += ld(x, i * dd + j) * ld(u, j)
            j += g
    if g == 1:
        if i < nn:
            st(sc, i, acc)
        return
    var sh = stack_allocation[SF_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    sh[tid] = acc
    barrier()
    var h = g // 2
    while h > 0:
        if lane < h:
            sh[tid] = sh[tid] + sh[tid + h]
        barrier()
        h //= 2
    if lane == 0 and i < nn:
        st(sc, i, sh[tid])


def sf_hist_kernel(sc: FP, n: Int32, sel: IP, parts: IP, shift: Int32, first: Int32, stt: FP):
    """Block b: the 16 counts of digit (key >> shift) & 15 over rows
    [b SF_SEL_ROWS, ...) among the keys whose higher digits are the prefix
    sel[0] (every key on the first pass)."""
    if _done(stt):
        return
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var lo = Int(block_idx.x) * SF_SEL_ROWS
    var hi = min(nn, lo + SF_SEL_ROWS)
    var sft = UInt32(Int(shift))
    var every = Int(first) != 0
    var prefix = bitcast[DType.uint32](sel.unsafe_load(0))
    var cnt = InlineArray[Int32, 16](fill=Int32(0))
    var i = lo + tid
    while i < hi:
        var k = _key(ld(sc, i))
        if every or (k >> (sft + UInt32(4))) == prefix:
            cnt[Int((k >> sft) & UInt32(15))] += 1
        i += SF_TPB
    var sh = stack_allocation[SF_TPB * 16, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for b in range(16):
        sh[tid * 16 + b] = cnt[b]
    barrier()
    if tid < 16:
        var s = Int32(0)
        for t in range(SF_TPB):
            s += sh[t * 16 + tid]
        parts.unsafe_store(Int(block_idx.x) * 16 + tid, s)


def sf_pick_kernel(parts: IP, nb: Int32, sel: IP, first: Int32, r_rank: Int32, stt: FP):
    """One block: the 16 bin totals over the nb blocks (ascending), the
    digit holding the need-th key; sel = (prefix, need left, that bin's count)."""
    if _done(stt):
        return
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[16, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if tid < 16:
        var s = Int32(0)
        for b in range(Int(nb)):
            s += parts.unsafe_load(b * 16 + tid)
        sh[tid] = s
    barrier()
    if tid == 0:
        var fst = Int(first) != 0
        var prefix = UInt32(0) if fst else bitcast[DType.uint32](sel.unsafe_load(0))
        var need = Int(r_rank) if fst else Int(sel.unsafe_load(1))
        var acc = 0
        var dsel = 15
        var found = False
        for b in range(16):
            if not found:
                var nbin = Int(sh[b])
                if acc + nbin >= need:
                    dsel = b
                    found = True
                else:
                    acc += nbin
        need -= acc
        prefix = (prefix << UInt32(4)) | UInt32(dsel)
        sel.unsafe_store(0, bitcast[DType.int32](prefix))
        sel.unsafe_store(1, Int32(need))
        sel.unsafe_store(2, sh[dsel])


@always_inline
def _cls(sc: FP, i: Int, tk: UInt32, every: Bool) -> Int:
    """0: below tau (or every row when all), 1: at tau, 2: above."""
    if every:
        return 0
    var k = _key(ld(sc, i))
    if k < tk:
        return 0
    return 1 if k == tk else 2


def sf_colsum_kernel(x: FP, n: Int32, d: Int32, sc: FP, sel: IP, parts: FP, mode_all: Int32, stt: FP):
    """Block b over rows [b SF_COL_ROWS, ...): per column j the sum of x_ij
    over the rows below tau (parts[(2b) d + j]) and at tau
    (parts[(2b + 1) d + j]); mode_all: every row in the first sum. d >= 128:
    a thread a column (columns tid, tid + SF_TPB, ...); else P = SF_TPB / d
    row phases a column, folded ascending."""
    var every = Int(mode_all) != 0
    if not every and _done(stt):
        return
    var nn = Int(n)
    var dd = Int(d)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lo = b * SF_COL_ROWS
    var hi = min(nn, lo + SF_COL_ROWS)
    var tk = bitcast[DType.uint32](sel.unsafe_load(0))
    if dd >= SF_TPB // 2:
        var j = tid
        while j < dd:
            var a0 = Float32(0)
            var a1 = Float32(0)
            for i in range(lo, hi):
                var c = _cls(sc, i, tk, every)
                if c == 0:
                    a0 += ld(x, i * dd + j)
                elif c == 1:
                    a1 += ld(x, i * dd + j)
            st(parts, (2 * b) * dd + j, a0)
            st(parts, (2 * b + 1) * dd + j, a1)
            j += SF_TPB
        return
    var p = SF_TPB // dd
    var ph = tid // dd
    var j = tid % dd
    var a0 = Float32(0)
    var a1 = Float32(0)
    if ph < p:
        var i = lo + ph
        while i < hi:
            var c = _cls(sc, i, tk, every)
            if c == 0:
                a0 += ld(x, i * dd + j)
            elif c == 1:
                a1 += ld(x, i * dd + j)
            i += p
    var sh = stack_allocation[2 * SF_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    sh[tid] = a0
    sh[SF_TPB + tid] = a1
    barrier()
    if tid < dd:
        var s0 = Float32(0)
        var s1 = Float32(0)
        for q in range(p):
            s0 += sh[q * dd + tid]
            s1 += sh[SF_TPB + q * dd + tid]
        st(parts, (2 * b) * dd + tid, s0)
        st(parts, (2 * b + 1) * dd + tid, s1)


def sf_fold_kernel(parts: FP, nb: Int32, d: Int32, dst_sums: FP, mode_all: Int32, stt: FP):
    """out[c d + j] = sum over blocks b ascending of parts[(2b + c) d + j]."""
    if Int(mode_all) == 0 and _done(stt):
        return
    var dd = Int(d)
    var q = Int(block_idx.x) * SF_TPB + Int(thread_idx.x)
    if q < 2 * dd:
        var c = q // dd
        var j = q % dd
        var s = Float32(0)
        for b in range(Int(nb)):
            s += ld(parts, (2 * b + c) * dd + j)
        st(dst_sums, q, s)


def sf_step_kernel(sums: FP, d: Int32, n: Int32, nu: Float32, r_rank: Int32, u: FP, s: FP, sel: IP,
                   stt: FP, cap: Int32, init: Int32):
    """One block. init: u = column mean. Else s from the masked sums, the
    gap, the stop test and the Frank-Wolfe step on u (state words SF_ST)."""
    if _done(stt):
        return
    var dd = Int(d)
    var tid = Int(thread_idx.x)
    var fnum = i2f(Int(n))
    if Int(init) != 0:
        var j = tid
        while j < dd:
            st(u, j, ld(sums, j) / fnum)
            j += SF_TPB
        return
    var c = Float32(1) / (nu * fnum)
    var need = Int(sel.unsafe_load(1))
    var m_eq = Int(sel.unsafe_load(2))
    var m_less = Int(r_rank) - need
    var w_tie = (Float32(1) - i2f(m_less) * c) / i2f(max(m_eq, 1))
    var uu = Float32(0)
    var us = Float32(0)
    var ss = Float32(0)
    var gg = Float32(0)
    var dm = Float32(0)
    var j = tid
    while j < dd:
        var sj = c * ld(sums, j) + w_tie * ld(sums, dd + j)
        st(s, j, sj)
        var uj = ld(u, j)
        var df = uj - sj
        uu += uj * uj
        us += uj * sj
        ss += sj * sj
        gg += uj * df
        dm += df * df
        j += SF_TPB
    var sh = stack_allocation[5 * SF_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    sh[tid] = uu
    sh[SF_TPB + tid] = us
    sh[2 * SF_TPB + tid] = ss
    sh[3 * SF_TPB + tid] = gg
    sh[4 * SF_TPB + tid] = dm
    barrier()
    var h = SF_TPB // 2
    while h > 0:
        if tid < h:
            for k in range(5):
                sh[k * SF_TPB + tid] = sh[k * SF_TPB + tid] + sh[k * SF_TPB + tid + h]
        barrier()
        h //= 2
    var gamma = Float32(0)
    var go = False
    var t_uu = sh[0]
    var t_us = sh[SF_TPB]
    var t_ss = sh[2 * SF_TPB]
    var t_g = sh[3 * SF_TPB]
    var t_dm = sh[4 * SF_TPB]
    var it = Int(ld(stt, 1))
    # stop on either certificate: the gap g (the primal at t u) or |u|^2 / 2
    # (the primal (0, 0): J(0, 0) - J* <= nu |u|^2 / 2 since the dual at u is
    # nu - nu |u|^2 / 2), relative to the hull's scale
    if not (min(t_g, Float32(0.5) * t_uu) <= SF_EPS * max(t_uu, t_ss) or it >= Int(cap) or t_dm <= Float32(0)):
        gamma = min(Float32(1), max(Float32(0), t_g / t_dm))
        go = True
    barrier()
    if tid == 0:
        st(stt, 2, t_uu)
        st(stt, 3, t_us)
        st(stt, 4, t_g)
        st(stt, 5, _unkey(bitcast[DType.uint32](sel.unsafe_load(0))))
        st(stt, 6, gamma)
        st(stt, 7, t_ss)
        if go:
            st(stt, 1, i2f(it + 1))
        else:
            st(stt, 0, Float32(1))
    if go:
        var jj = tid
        while jj < dd:
            var uj = ld(u, jj)
            st(u, jj, uj + gamma * (ld(s, jj) - uj))
            jj += SF_TPB


def sf_res_kernel(u: FP, stt: FP, res: FP, d: Int32):
    """coef = t u, offset = t tau, t = max(0, <u,s>) / |u|^2 (the primal's
    best point on the ray); then the iterations and status 0."""
    var dd = Int(d)
    var q = Int(block_idx.x) * SF_TPB + Int(thread_idx.x)
    var uu = ld(stt, 2)
    var us = ld(stt, 3)
    var t = us / uu if (us > Float32(0) and uu > Float32(0)) else Float32(0)
    if q < dd:
        st(res, q, t * ld(u, q))
    elif q == dd:
        st(res, q, t * ld(stt, 5))
    elif q == dd + 1:
        st(res, q, ld(stt, 1))
    elif q == dd + 2:
        st(res, q, Float32(0))


def sgdoc_fw_fit(mut ctx: DeviceContext, x: FP, n_x: Int, n: Int, d: Int, nu: Float32, max_iter: Int,
                 n_out: Int, res: FP) raises:
    """SGDOneClassSVM (fit_intercept, no sample weights) by Frank-Wolfe on
    the dual (module notes). x: host rows (n x d). res: coef (d), offset,
    iterations, status, as the per-sample fit's result words."""
    var r_rank = Int(Float64(nu) * Float64(n) - Float64(n) * 1e-7)
    if Float64(r_rank) < Float64(nu) * Float64(n) - Float64(n) * 1e-7:
        r_rank += 1
    r_rank = max(1, min(n, r_rank))
    var cap = max(SF_ITERS_PER_EPOCH, SF_ITERS_PER_EPOCH * max_iter)
    var gl = 32 if d >= 32 else 1
    var score_blocks = (n + SF_TPB // gl - 1) // (SF_TPB // gl)
    var sel_blocks = (n + SF_SEL_ROWS - 1) // SF_SEL_ROWS
    var col_blocks = (n + SF_COL_ROWS - 1) // SF_COL_ROWS
    var fold_blocks = (2 * d + SF_TPB - 1) // SF_TPB
    var res_blocks = (d + 3 + SF_TPB - 1) // SF_TPB
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dsc = ctx.enqueue_create_buffer[DType.float32](n)
    var dsp = ctx.enqueue_create_buffer[DType.int32](sel_blocks * 16)
    var dsel = ctx.enqueue_create_buffer[DType.int32](4)
    var dcp = ctx.enqueue_create_buffer[DType.float32](col_blocks * 2 * d)
    var dsum = ctx.enqueue_create_buffer[DType.float32](2 * d)
    var du = ctx.enqueue_create_buffer[DType.float32](d)
    var ds = ctx.enqueue_create_buffer[DType.float32](d)
    var dst = ctx.enqueue_create_buffer[DType.float32](SF_ST)
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var hst = List[Float32](length=SF_ST, fill=Float32(0))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    dst.enqueue_fill(Float32(0))
    dsel.enqueue_fill(Int32(0))
    dres.enqueue_fill(Float32(0))
    # u0 = the column mean
    ctx.enqueue_function[sf_colsum_kernel](
        dx.unsafe_ptr(), Int32(n), Int32(d), dsc.unsafe_ptr(), dsel.unsafe_ptr(), dcp.unsafe_ptr(), Int32(1),
        dst.unsafe_ptr(), grid_dim=col_blocks, block_dim=SF_TPB)
    ctx.enqueue_function[sf_fold_kernel](
        dcp.unsafe_ptr(), Int32(col_blocks), Int32(d), dsum.unsafe_ptr(), Int32(1), dst.unsafe_ptr(),
        grid_dim=fold_blocks, block_dim=SF_TPB)
    ctx.enqueue_function[sf_step_kernel](  # small-launch(d: weights): one block strides the d weights; n only scales the mean and the cap 1/(nu n)
        dsum.unsafe_ptr(), Int32(d), Int32(n), nu, Int32(r_rank), du.unsafe_ptr(), ds.unsafe_ptr(),
        dsel.unsafe_ptr(), dst.unsafe_ptr(), Int32(cap), Int32(1), grid_dim=1, block_dim=SF_TPB)
    var evals = 0
    while evals <= cap:
        for _ in range(SF_BATCH):
            ctx.enqueue_function[sf_score_kernel](
                dx.unsafe_ptr(), Int32(n), Int32(d), du.unsafe_ptr(), dsc.unsafe_ptr(), dst.unsafe_ptr(),
                Int32(gl), grid_dim=score_blocks, block_dim=SF_TPB)
            var shift = 28
            while shift >= 0:
                var first = Int32(1) if shift == 28 else Int32(0)
                ctx.enqueue_function[sf_hist_kernel](
                    dsc.unsafe_ptr(), Int32(n), dsel.unsafe_ptr(), dsp.unsafe_ptr(), Int32(shift), first,
                    dst.unsafe_ptr(), grid_dim=sel_blocks, block_dim=SF_TPB)
                ctx.enqueue_function[sf_pick_kernel](
                    dsp.unsafe_ptr(), Int32(sel_blocks), dsel.unsafe_ptr(), first, Int32(r_rank),
                    dst.unsafe_ptr(), grid_dim=1, block_dim=SF_TPB)
                shift -= 4
            ctx.enqueue_function[sf_colsum_kernel](
                dx.unsafe_ptr(), Int32(n), Int32(d), dsc.unsafe_ptr(), dsel.unsafe_ptr(), dcp.unsafe_ptr(),
                Int32(0), dst.unsafe_ptr(), grid_dim=col_blocks, block_dim=SF_TPB)
            ctx.enqueue_function[sf_fold_kernel](
                dcp.unsafe_ptr(), Int32(col_blocks), Int32(d), dsum.unsafe_ptr(), Int32(0), dst.unsafe_ptr(),
                grid_dim=fold_blocks, block_dim=SF_TPB)
            ctx.enqueue_function[sf_step_kernel](  # small-launch(d: weights): one block strides the d weights; n only scales the mean and the cap 1/(nu n)
                dsum.unsafe_ptr(), Int32(d), Int32(n), nu, Int32(r_rank), du.unsafe_ptr(), ds.unsafe_ptr(),
                dsel.unsafe_ptr(), dst.unsafe_ptr(), Int32(cap), Int32(0), grid_dim=1, block_dim=SF_TPB)
            evals += 1
        ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dst)
        ctx.synchronize()
        if hst[0] != Float32(0):
            break
    ctx.enqueue_function[sf_res_kernel](
        du.unsafe_ptr(), dst.unsafe_ptr(), dres.unsafe_ptr(), Int32(d), grid_dim=res_blocks, block_dim=SF_TPB)
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    _ = hst^
    _ = dx^
    _ = dsc^
    _ = dsp^
    _ = dsel^
    _ = dcp^
    _ = dsum^
    _ = du^
    _ = ds^
    _ = dst^
    _ = dres^
