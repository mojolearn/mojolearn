# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-ldaqda, -D MOJOLEARN_LDAQDA_RR_EIGH): the
prep lane's `eigh` stage (op 18, x_prep/eigh.mojo `eigh_unit`) as the
round-robin Jacobi on the whole GPU.

`eigh_unit` is ONE THREAD per matrix running the cyclic Jacobi: at LDA's /
QDA's d = 220 (Istella) that is 24,090 serial rotations a sweep, each
walking 2 x 220 cells, on one GPU thread while the rest of the GPU idles
(lda-clf Istella 19.8 s, qda 15.9 s on the M3 board). Here the columns play
x_decomp/rr.mojo's circle-method tournament: m - 1 rounds a sweep, the m / 2
disjoint rotations of a round at once, every 2 x 2 block of J^T A J and
every (row, pair) of V J on its own thread (`rr_cs`, `rr_block`, `rr_vrow`,
the cells of x_decomp/jacobi_par.mojo), for every matrix of the batch in
the same launch.

ONE launch a round: A ping-pongs between the arena and a scratch copy, so
every thread takes the (c, s) it needs from the round's source (`rr_cs`,
the same words as x_decomp's separate cs launch) and writes its cells of
the destination; V is updated in place (each cell one reader and writer).

Convergence: x_decomp's test (off-diagonal squares against tol^2 ||A||_F^2,
`rr_off_fold`'s order) before every sweep, decided ON THE DEVICE per matrix
into a done mark; every later kernel of a done matrix returns at once. No
host step: the host enqueues a fixed budget of RRE_SWEEPS sweeps and never
reads the marks (a converged matrix's remaining launches return at once).
The test also leaves the diagonal of its source in dg, so the tail reads
the eigenvalues from dg, wherever the matrix stopped. A matrix that has not
converged in RRE_SWEEPS stops there, as `eigh_unit` stops at its cap.

The tail is `eigh_unit`'s contract: eigenvalues DESCENDING (index order on
a tie, `spectrum_rank_desc`), each vector's largest-magnitude component
(first on a tie) positive, EVAL[t m + r], EVEC[t m m + c m + r]. FAST only:
the round-robin order is not the cyclic order, so the words differ from
`eigh_unit`'s within float32 round-off; IDENTICAL never compiles a call.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_div, identical_mul
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.spectrum_order_device import spectrum_rank_desc
from x_decomp.rr import RR_OFF_TPB, pj_first, pj_second, rr_add, rr_converged, rr_cs, rr_row_off, rr_sub
from x_prep.common import FP

comptime RRE_TPB = 256
"""Launch width of the per-cell kernels."""
comptime RRE_SWEEPS = 32
"""Sweep budget (enqueued whole; a converged matrix's launches return at once)."""
comptime RRE_SYNC_ROUNDS = 512
"""Rounds enqueued between two waits (x_decomp/device.mojo PJ_SYNC_ROUNDS;
a wait only, nothing is read)."""
comptime RRE_STATE = 2
"""State words a matrix: [done, tests]."""


@always_inline
def rre_m(n: Int) -> Int:
    return n + (n % 2)


@always_inline
def rre_nb(n: Int) -> Int:
    return max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)


@always_inline
def rre_words(n: Int, batch: Int) -> Int:
    """Scratch words for `rr_eigh_into`: A's ping-pong copy, V, partials,
    diagonal, state."""
    return batch * (2 * n * n + 2 * rre_nb(n) + n + RRE_STATE)


@always_inline
def _blocks(t: Int) -> Int:
    return max((t + RRE_TPB - 1) // RRE_TPB, 1)


def rre_init_kernel(v: FP, stt: FP, n_in: Int32, batch_in: Int32):
    """V = I for every matrix; state = [0, 0]."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch = Int(batch_in)
    if t < batch * n * n:
        var c = t % (n * n)
        var i = c // n
        v.unsafe_store(t, Float32(1.0) if i * n + i == c else Float32(0.0))
    if t < batch:
        stt.unsafe_store(t * RRE_STATE, Float32(0.0))
        stt.unsafe_store(t * RRE_STATE + 1, Float32(0.0))


def rre_part_kernel(a0: FP, astride: Int32, part: FP, dg: FP, stt: FP, n_in: Int32):
    """Block g = b nb + blk: rows blk RR_OFF_TPB .. of matrix b (at a0 + b
    astride), the pairwise tree of their off-diagonal squares and a_kk^2
    into part[b][2 blk ..], the diagonal into dg[b n + k]
    (`eigh_par_off_part_kernel`)."""
    var n = Int(n_in)
    var nb = rre_nb(n)
    var g = Int(block_idx.x)
    var b = g // nb
    var blk = g - b * nb
    if stt.unsafe_load(b * RRE_STATE) != Float32(0.0):
        return
    var a = a0 + b * Int(astride)
    var tid = Int(thread_idx.x)
    var k = blk * RR_OFF_TPB + tid
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var o = SIMD[DType.float32, 2](0.0, 0.0)
    if k < n:
        o = rr_row_off(a, n, k)
        dg.unsafe_store(b * n + k, ftz(a.unsafe_load(k * n + k)))
    so[tid] = o[0]
    sd[tid] = o[1]
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            so[tid] = ftz(so[tid] + so[tid + w])
            sd[tid] = ftz(sd[tid] + sd[tid + w])
        barrier()
        w = w // 2
    if tid == 0:
        part.unsafe_store(b * 2 * nb + 2 * blk, so[0])
        part.unsafe_store(b * 2 * nb + 2 * blk + 1, sd[0])


def rre_fold_kernel(part: FP, stt: FP, n_in: Int32, sweeps_in: Int32, tol: Float32):
    """Block b: the tree past matrix b's blocks (`eigh_par_off_fold_kernel`),
    then the test: converged (rr_converged) or out of sweeps marks it done."""
    var n = Int(n_in)
    var nb = rre_nb(n)
    var b = Int(block_idx.x)
    var s0 = b * RRE_STATE
    if stt.unsafe_load(s0) != Float32(0.0):
        return
    var tid = Int(thread_idx.x)
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ao = Float32(0.0)
    var ad = Float32(0.0)
    var j = tid
    while j < nb:
        ao = ftz(ao + part.unsafe_load(b * 2 * nb + 2 * j))
        ad = ftz(ad + part.unsafe_load(b * 2 * nb + 2 * j + 1))
        j += RR_OFF_TPB
    so[tid] = ao
    sd[tid] = ad
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            so[tid] = ftz(so[tid] + so[tid + w])
            sd[tid] = ftz(sd[tid] + sd[tid + w])
        barrier()
        w = w // 2
    if tid == 0:
        var tests = Int(stt.unsafe_load(s0 + 1)) + 1
        stt.unsafe_store(s0 + 1, Float32(tests))
        if rr_converged(so[0], sd[0], tol) or tests > Int(sweeps_in):
            stt.unsafe_store(s0, Float32(1.0))


@always_inline
def _pair(r: Int, b: Int, m: Int) -> Tuple[Int, Int]:
    var x0 = pj_first(r, b, m)
    var x1 = pj_second(r, b, m)
    return (min(x0, x1), max(x0, x1))


def rre_round_kernel(src0: FP, sstride: Int32, dst0: FP, dstride: Int32, v: FP, stt: FP, n_in: Int32, rd: Int32,
                     batch_in: Int32):
    """Thread b U + u, U = h h + n h, round rd of matrix b: units below h h
    the 2 x 2 blocks (i, j), i <= j, of J^T A J from src into dst (and the
    mirror; the pair's own block in closed form, a bye's diagonal copied),
    the rest V J by (row, pair) in place: x_decomp/rr.mojo's `rr_block` /
    `rr_vrow` with each (c, s) taken by `rr_cs` from src."""
    var n = Int(n_in)
    var m = rre_m(n)
    var h = m // 2
    var units = h * h + n * h
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(batch_in) * units:
        return
    var b = t // units
    var u = t - b * units
    if stt.unsafe_load(b * RRE_STATE) != Float32(0.0):
        return
    var r = Int(rd)
    var src = src0 + b * Int(sstride)
    if u >= h * h:
        var x = u - h * h
        var k = x // h
        var j = x - k * h
        var pq = _pair(r, j, m)
        if pq[1] < n:
            var cs = rr_cs(src, n, m, r, j)
            var vb = v + b * n * n
            var vkp = vb.unsafe_load(k * n + pq[0])
            var vkq = vb.unsafe_load(k * n + pq[1])
            vb.unsafe_store(k * n + pq[0], rr_sub(cs[0], vkp, cs[1], vkq))
            vb.unsafe_store(k * n + pq[1], rr_add(cs[1], vkp, cs[0], vkq))
        return
    var i = u // h
    var j = u - i * h
    if i > j:
        return
    var dst = dst0 + b * Int(dstride)
    var ip = _pair(r, i, m)
    var pi = ip[0]
    var qi = ip[1]
    var csi = rr_cs(src, n, m, r, i)
    var ci = csi[0]
    var si = csi[1]
    if i == j:
        if qi < n:
            var app = src.unsafe_load(pi * n + pi)
            var aqq = src.unsafe_load(qi * n + qi)
            var apq = src.unsafe_load(pi * n + qi)
            var tt = ftz(identical_div(si, ci))
            var dlt = ftz(identical_mul(tt, apq))
            dst.unsafe_store(pi * n + pi, ftz(app - dlt))
            dst.unsafe_store(qi * n + qi, ftz(aqq + dlt))
            dst.unsafe_store(pi * n + qi, Float32(0.0))
            dst.unsafe_store(qi * n + pi, Float32(0.0))
        else:
            dst.unsafe_store(pi * n + pi, src.unsafe_load(pi * n + pi))
        return
    var jp = _pair(r, j, m)
    var pj = jp[0]
    var qj = jp[1]
    var csj = rr_cs(src, n, m, r, j)
    var cj = csj[0]
    var sj = csj[1]
    var vi = qi < n
    var vj = qj < n
    var b00 = src.unsafe_load(pi * n + pj)
    var b01 = Float32(0.0)
    var b10 = Float32(0.0)
    var b11 = Float32(0.0)
    if vj:
        b01 = src.unsafe_load(pi * n + qj)
    if vi:
        b10 = src.unsafe_load(qi * n + pj)
    if vi and vj:
        b11 = src.unsafe_load(qi * n + qj)
    var t00 = rr_sub(cj, b00, sj, b01)
    var t01 = rr_add(sj, b00, cj, b01)
    var t10 = rr_sub(cj, b10, sj, b11)
    var t11 = rr_add(sj, b10, cj, b11)
    var n00 = rr_sub(ci, t00, si, t10)
    var n01 = rr_sub(ci, t01, si, t11)
    var n10 = rr_add(si, t00, ci, t10)
    var n11 = rr_add(si, t01, ci, t11)
    dst.unsafe_store(pi * n + pj, n00)
    dst.unsafe_store(pj * n + pi, n00)
    if vj:
        dst.unsafe_store(pi * n + qj, n01)
        dst.unsafe_store(qj * n + pi, n01)
    if vi:
        dst.unsafe_store(qi * n + pj, n10)
        dst.unsafe_store(pj * n + qi, n10)
    if vi and vj:
        dst.unsafe_store(qi * n + qj, n11)
        dst.unsafe_store(qj * n + qi, n11)


def rre_sign_kernel(v: FP, n_in: Int32, batch_in: Int32):
    """Thread b n + col: column col of V made positive at its largest-magnitude
    component (first on a tie; comparisons only)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(batch_in) * n:
        return
    var b = t // n
    var col = t - b * n
    var vb = v + b * n * n
    var biggest = Float32(0.0)
    var first = 0
    for r in range(n):
        var mg = abs(vb.unsafe_load(r * n + col))
        if mg > biggest:
            biggest = mg
            first = r
    if vb.unsafe_load(first * n + col) < Float32(0.0):
        for r in range(n):
            vb.unsafe_store(r * n + col, -vb.unsafe_load(r * n + col))


def rre_order_kernel(f: FP, dg: FP, v: FP, w_off: Int32, v_off: Int32, n_in: Int32, batch_in: Int32):
    """Thread b n + i: eigenpair i of matrix b to its descending position r
    (`spectrum_rank_desc`: index order on a tie): EVAL[b n + r] and column r
    of EVEC[b n n ..]."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(batch_in) * n:
        return
    var b = t // n
    var i = t - b * n
    var r = spectrum_rank_desc(dg + b * n, n, i)
    f.unsafe_store(Int(w_off) + b * n + r, dg.unsafe_load(b * n + i))
    var vb = v + b * n * n
    var eo = Int(v_off) + b * n * n
    for c in range(n):
        f.unsafe_store(eo + c * n + r, ftz(vb.unsafe_load(c * n + i)))


def rr_eigh_into(mut ctx: DeviceContext, f: FP, a_off: Int, n: Int, astride: Int, batch: Int, w_off: Int,
                 v_off: Int, scratch: FP) raises:
    """`eigh_unit`'s stage q = [A, n, astride, EVAL, EVEC] for `batch`
    matrices, on the arena f (device): A destroyed, EVAL / EVEC as
    `eigh_unit` writes them. scratch: `rre_words(n, batch)` device words.
    Enqueues (a wait every RRE_SYNC_ROUNDS rounds, nothing read)."""
    if n <= 0 or batch <= 0:
        return
    var m = rre_m(n)
    var h = m // 2
    var nb = rre_nb(n)
    var a1 = scratch
    var v = a1 + batch * n * n
    var part = v + batch * n * n
    var dg = part + batch * 2 * nb
    var stt = dg + batch * n
    var a0 = f + a_off
    ctx.enqueue_function[rre_init_kernel](v, stt, Int32(n), Int32(batch), grid_dim=_blocks(batch * n * n),
                                          block_dim=RRE_TPB)
    var units = h * h + n * h
    var rounds = 0
    for sweep in range(RRE_SWEEPS + 1):
        # every matrix not yet done sits in the same buffer: rounds so far mod 2
        var cur = a0 if rounds % 2 == 0 else a1
        var cst = astride if rounds % 2 == 0 else n * n
        ctx.enqueue_function[rre_part_kernel](cur, Int32(cst), part, dg, stt, Int32(n), grid_dim=batch * nb,
                                              block_dim=RR_OFF_TPB)
        ctx.enqueue_function[rre_fold_kernel](part, stt, Int32(n), Int32(RRE_SWEEPS), Float32(JACOBI_TOL),
                                              grid_dim=batch, block_dim=RR_OFF_TPB)
        if sweep == RRE_SWEEPS:
            break
        for rd in range(m - 1):
            var even = rounds % 2 == 0
            var sp = FP(unsafe_from_address=Int(a0) if even else Int(a1))
            var dp = FP(unsafe_from_address=Int(a1) if even else Int(a0))
            ctx.enqueue_function[rre_round_kernel](
                sp, Int32(astride if even else n * n), dp, Int32(n * n if even else astride), v, stt, Int32(n),
                Int32(rd), Int32(batch), grid_dim=_blocks(batch * units), block_dim=RRE_TPB,
            )
            rounds += 1
            if rounds % RRE_SYNC_ROUNDS == 0:
                ctx.synchronize()
    ctx.enqueue_function[rre_sign_kernel](v, Int32(n), Int32(batch), grid_dim=_blocks(batch * n),
                                          block_dim=RRE_TPB)
    ctx.enqueue_function[rre_order_kernel](f, dg, v, Int32(w_off), Int32(v_off), Int32(n), Int32(batch),
                                           grid_dim=_blocks(batch * n), block_dim=RRE_TPB)
