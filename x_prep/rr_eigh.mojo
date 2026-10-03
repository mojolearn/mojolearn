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

Convergence: x_decomp's test (off-diagonal squares against tol^2 ||A||_F^2,
`rr_off_fold`'s order) before every sweep, decided ON THE DEVICE per matrix
into a done mark; every later kernel of a done matrix returns at once, so
the host enqueues sweeps ahead and reads the marks only every
RRE_POLL_SWEEPS sweeps (no per-sweep wait). A matrix that has not converged
in RR_EIGH_SWEEPS stops there, as `eigh_unit` stops at its sweep cap.

The tail is `eigh_unit`'s contract: eigenvalues DESCENDING (index order on
a tie, `spectrum_rank_desc`), each vector's largest-magnitude component
(first on a tie) positive, EVAL[t m + r], EVEC[t m m + c m + r]. FAST only:
the round-robin order is not the cyclic order, so the words differ from
`eigh_unit`'s within float32 round-off; IDENTICAL never compiles a call.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.spectrum_order_device import spectrum_rank_desc
from x_decomp.rr import RR_EIGH_SWEEPS, RR_OFF_TPB, rr_block, rr_converged, rr_cs, rr_row_off, rr_vrow
from x_prep.common import FP

comptime RRE_TPB = 256
"""Launch width of the per-cell kernels."""
comptime RRE_POLL_SWEEPS = 2
"""Sweeps enqueued between two reads of the done marks."""
comptime RRE_STATE = 4
"""State words a matrix: [done, fro_in, tests, converged]."""


@always_inline
def rre_m(n: Int) -> Int:
    return n + (n % 2)


@always_inline
def rre_nb(n: Int) -> Int:
    return max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)


@always_inline
def rre_words(n: Int, batch: Int) -> Int:
    """Scratch words for `rr_eigh_into`: V, cs, partials, diagonal, state."""
    var m = rre_m(n)
    return batch * (n * n + m + 2 * rre_nb(n) + n + RRE_STATE)


@always_inline
def _blocks(t: Int) -> Int:
    return max((t + RRE_TPB - 1) // RRE_TPB, 1)


def rre_init_kernel(v: FP, stt: FP, n_in: Int32, batch_in: Int32):
    """V = I for every matrix; state = [0, -1, 0, 0]."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch = Int(batch_in)
    if t < batch * n * n:
        var c = t % (n * n)
        var i = c // n
        v.unsafe_store(t, Float32(1.0) if i * n + i == c else Float32(0.0))
    if t < batch:
        stt.unsafe_store(t * RRE_STATE, Float32(0.0))
        stt.unsafe_store(t * RRE_STATE + 1, Float32(-1.0))
        stt.unsafe_store(t * RRE_STATE + 2, Float32(0.0))
        stt.unsafe_store(t * RRE_STATE + 3, Float32(0.0))


def rre_part_kernel(f: FP, a_off: Int32, astride: Int32, part: FP, dg: FP, stt: FP, n_in: Int32):
    """Block g = b nb + blk: rows blk RR_OFF_TPB .. of matrix b, the
    pairwise tree of their off-diagonal squares and a_kk^2 into part[b][2
    blk ..], the diagonal into dg[b n + k] (`eigh_par_off_part_kernel`)."""
    var n = Int(n_in)
    var nb = rre_nb(n)
    var g = Int(block_idx.x)
    var b = g // nb
    var blk = g - b * nb
    if stt.unsafe_load(b * RRE_STATE) != Float32(0.0):
        return
    var a = f + (Int(a_off) + b * Int(astride))
    var tid = Int(thread_idx.x)
    var k = blk * RR_OFF_TPB + tid
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var o = SIMD[DType.float32, 2](0.0, 0.0)
    if k < n:
        o = rr_row_off(a, n, k)
        dg.unsafe_store(b * n + k, a.unsafe_load(k * n + k))
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
        var off = so[0]
        var dd = sd[0]
        var fro = ftz(off + dd)
        if stt.unsafe_load(s0 + 1) < Float32(0.0):
            stt.unsafe_store(s0 + 1, fro)
        var tests = stt.unsafe_load(s0 + 2) + Float32(1.0)
        stt.unsafe_store(s0 + 2, tests)
        if rr_converged(off, dd, tol):
            stt.unsafe_store(s0 + 3, Float32(1.0))
            stt.unsafe_store(s0, Float32(1.0))
        elif Int(tests) > Int(sweeps_in):
            stt.unsafe_store(s0, Float32(1.0))


def rre_cs_kernel(f: FP, a_off: Int32, astride: Int32, cs: FP, stt: FP, n_in: Int32, rd: Int32, batch_in: Int32):
    """Thread b h + p: pair p of round rd of matrix b, its (c, s) (`rr_cs`)."""
    var n = Int(n_in)
    var m = rre_m(n)
    var h = m // 2
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(batch_in) * h:
        return
    var b = t // h
    var pp = t - b * h
    if stt.unsafe_load(b * RRE_STATE) != Float32(0.0):
        return
    var a = f + (Int(a_off) + b * Int(astride))
    var got = rr_cs(a, n, m, Int(rd), pp)
    cs.unsafe_store(b * m + 2 * pp, got[0])
    cs.unsafe_store(b * m + 2 * pp + 1, got[1])


def rre_update_kernel(f: FP, a_off: Int32, astride: Int32, v: FP, cs: FP, stt: FP, n_in: Int32, rd: Int32,
                      batch_in: Int32):
    """Thread b U + u, U = h h + n h: J^T A J by 2 x 2 blocks (`rr_block`),
    then V J by (row, pair) (`rr_vrow`), matrix b, round rd."""
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
    var a = f + (Int(a_off) + b * Int(astride))
    var c = cs + b * m
    if u < h * h:
        var i = u // h
        var j = u - i * h
        if i <= j:
            rr_block(a, c, n, m, Int(rd), i, j)
    else:
        var x = u - h * h
        var k = x // h
        rr_vrow(v + b * n * n, c, n, m, Int(rd), k, x - k * h)


def rre_sign_kernel(f: FP, a_off: Int32, astride: Int32, v: FP, dg: FP, n_in: Int32, batch_in: Int32):
    """Thread b n + col: column col of V made positive at its largest-magnitude
    component (first on a tie; comparisons only); dg = the diagonal of A."""
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
    var a = f + (Int(a_off) + b * Int(astride))
    dg.unsafe_store(b * n + col, ftz(a.unsafe_load(col * n + col)))


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
                 v_off: Int, scratch: FP, mut host_state: List[Float32],
                 mut state_buf: DeviceBuffer[DType.float32]) raises:
    """`eigh_unit`'s stage q = [A, n, astride, EVAL, EVEC] for `batch`
    matrices, on the arena f (device): A destroyed, EVAL / EVEC as
    `eigh_unit` writes them. scratch: `rre_words(n, batch)` words minus the
    state (kept in state_buf, batch * RRE_STATE words, read back every
    RRE_POLL_SWEEPS sweeps into host_state)."""
    if n <= 0 or batch <= 0:
        return
    var m = rre_m(n)
    var h = m // 2
    var nb = rre_nb(n)
    var v = scratch
    var cs = v + batch * n * n
    var part = cs + batch * m
    var dg = part + batch * 2 * nb
    var stt = FP(unsafe_from_address=Int(state_buf.unsafe_ptr()))
    ctx.enqueue_function[rre_init_kernel](v, stt, Int32(n), Int32(batch), grid_dim=_blocks(batch * n * n),
                                          block_dim=RRE_TPB)
    var units = h * h + n * h
    for sweep in range(RR_EIGH_SWEEPS + 1):
        ctx.enqueue_function[rre_part_kernel](f, Int32(a_off), Int32(astride), part, dg, stt, Int32(n),
                                              grid_dim=batch * nb, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[rre_fold_kernel](part, stt, Int32(n), Int32(RR_EIGH_SWEEPS), Float32(JACOBI_TOL),
                                              grid_dim=batch, block_dim=RR_OFF_TPB)
        if sweep % RRE_POLL_SWEEPS == RRE_POLL_SWEEPS - 1 or sweep == RR_EIGH_SWEEPS:
            ctx.enqueue_copy(dst_ptr=host_state.unsafe_ptr(),
                             src_buf=state_buf.create_sub_buffer[DType.float32](0, batch * RRE_STATE))
            ctx.synchronize()
            var all_done = True
            for b in range(batch):
                if host_state[b * RRE_STATE] == Float32(0.0):
                    all_done = False
            if all_done:
                break
        if sweep == RR_EIGH_SWEEPS:
            break
        for rd in range(m - 1):
            ctx.enqueue_function[rre_cs_kernel](f, Int32(a_off), Int32(astride), cs, stt, Int32(n), Int32(rd),
                                                Int32(batch), grid_dim=_blocks(batch * h), block_dim=RRE_TPB)
            ctx.enqueue_function[rre_update_kernel](f, Int32(a_off), Int32(astride), v, cs, stt, Int32(n),
                                                    Int32(rd), Int32(batch), grid_dim=_blocks(batch * units),
                                                    block_dim=RRE_TPB)
    ctx.enqueue_function[rre_sign_kernel](f, Int32(a_off), Int32(astride), v, dg, Int32(n), Int32(batch),
                                          grid_dim=_blocks(batch * n), block_dim=RRE_TPB)
    ctx.enqueue_function[rre_order_kernel](f, dg, v, Int32(w_off), Int32(v_off), Int32(n), Int32(batch),
                                           grid_dim=_blocks(batch * n), block_dim=RRE_TPB)
