# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device kernels of lane xd-rr-one-block (2026-10-11; switches and cost
rule in x_decomp/xd_lean.mojo). NOT COMPILED, NOT VERIFIED, NOT MEASURED.

C3 `rr_eigh_one_block_d_kernel` (MOJOLEARN_IDN_RR_ONE_BLOCK_D): the whole
round-robin two-sided Jacobi of `_eig_rr_device` in one launch of one block.
Rebuilt from the deleted MOJOLEARN_IDN_PCA_RR_ONE_BLOCK kernel (recoverable
at main 0a7b206f1, x_decomp/rr_one_block.mojo), which REFUSED at its launch
on NVIDIA (nv2 v1030) with RR_ONE_TPB = 1024 threads. The cause read from
the code: a 1024-thread block may use at most 64 registers a thread (64 K
registers an SM on sm_89), and the fused kernel (rr_block's 2 x 2 update,
rr_cs, the test trees, the gate) needs more, so the launch is refused for
resources; nothing in it is a cross-block hazard. Here the block is
RR_ONE_D_TPB = 256 = RR_OFF_TPB threads (the register cap is then 255, and
the test's part tree is then exactly `eigh_par_off_part_kernel`'s block
tree, one block b a pass, with no thread-group mapping).

BITS: none. A round's cells have one writer and no other reader
(x_decomp/rr.mojo), so strided loops in one block store the words the grid
stored; `dev_barrier` (orders device memory) stands where the launch
boundaries were; the test folds in `eigh_par_off_part_kernel` then
`eigh_par_off_fold_kernel`'s order; the gate is `rr_gate_state`. The host
column (`host_eigh_rr`) is unchanged.

C4 c `rs_round_gated_kernel` + `rs_sweep_gate_kernel`
(MOJOLEARN_IDN_SVD_TALL_LEAN): `rs_round_kernel`'s statements unchanged
behind a sticky device convergence word, set after the first sweep whose
rotation flags are all clear. No bit moves.

C4 b `orth_lean_test_kernel`: the skip test of the orth's second pass.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.jacobi2 import dev_barrier
from x_decomp.rr import RR_OFF_TPB, rr_block, rr_cs, rr_gate_state, rr_row_off, rr_vrow
from x_decomp.rr_svd import RS_TPB, rs_apply_lane, rs_decide, rs_lane3, rs_pair
from x_decomp.xd_lean import RR_ONE_D_TPB, orth_lean_violates


def rr_eigh_one_block_d_kernel(
    a: F32Ptr,
    v: F32Ptr,
    cs: F32Ptr,
    doff: F32Ptr,
    part: F32Ptr,
    fold: F32Ptr,
    state: F32Ptr,
    n_in: Int32,
    sweeps_in: Int32,
    tol: Float32,
):
    """The solve on `a` (n x n, in place) and `v` (n x n, already the
    identity); the decision in `state` (PCA_RR_STATE words initialized as
    `_eig_rr_device` does: zeros, state[2] = -1). Scratch as the grid
    route's: cs 2 h, doff 3 n, part 3 nb, fold 3. Launch grid 1, block
    RR_ONE_D_TPB (== RR_OFF_TPB)."""
    var n = Int(n_in)
    var sweeps = Int(sweeps_in)
    var m = n + (n % 2)
    var h = m // 2
    var nb = (n + RR_OFF_TPB - 1) // RR_OFF_TPB
    var tid = Int(thread_idx.x)
    var so = stack_allocation[RR_ONE_D_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_ONE_D_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sm = stack_allocation[RR_ONE_D_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stop = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for sweep in range(sweeps + 1):
        # ---- the test, part: `eigh_par_off_part_kernel`'s block b on pass b
        for b in range(nb):
            var k = b * RR_OFF_TPB + tid
            var o = SIMD[DType.float32, 2](0.0, 0.0)
            if k < n:
                o = rr_row_off(a, n, k)
                doff.unsafe_store(2 * n + k, a.unsafe_load(k * n + k))
            so[tid] = o[0]
            sd[tid] = o[1]
            dev_barrier()
            var w = RR_OFF_TPB // 2
            while w > 0:
                if tid < w:
                    so[tid] = ftz(so[tid] + so[tid + w])
                    sd[tid] = ftz(sd[tid] + sd[tid + w])
                dev_barrier()
                w = w // 2
            if tid == 0:
                part.unsafe_store(3 * b, so[0])
                part.unsafe_store(3 * b + 1, sd[0])
                part.unsafe_store(3 * b + 2, Float32(0.0))
            dev_barrier()
        # ---- the test, fold: `eigh_par_off_fold_kernel`'s statements
        var ao = Float32(0.0)
        var ad = Float32(0.0)
        var am = Float32(0.0)
        var bb = tid
        while bb < nb:
            ao = ftz(ao + part.unsafe_load(3 * bb))
            ad = ftz(ad + part.unsafe_load(3 * bb + 1))
            am = min(am, part.unsafe_load(3 * bb + 2))
            bb += RR_OFF_TPB
        so[tid] = ao
        sd[tid] = ad
        sm[tid] = am
        dev_barrier()
        var w2 = RR_OFF_TPB // 2
        while w2 > 0:
            if tid < w2:
                so[tid] = ftz(so[tid] + so[tid + w2])
                sd[tid] = ftz(sd[tid] + sd[tid + w2])
                sm[tid] = min(sm[tid], sm[tid + w2])
            dev_barrier()
            w2 = w2 // 2
        # ---- the gate (`pca_rr_gate_kernel`'s statements) and the stop
        if tid == 0:
            fold.unsafe_store(0, so[0])
            fold.unsafe_store(1, sd[0])
            fold.unsafe_store(2, sm[0])
            rr_gate_state(so[0], sd[0], sm[0], state, tol)
            var done = state.unsafe_load(0) == Float32(1.0) or state.unsafe_load(4) < Float32(0.0)
            stop[0] = Float32(1.0) if done else Float32(0.0)
        dev_barrier()
        # block-uniform: every thread reads the one shared word after the barrier
        if stop[0] != Float32(0.0) or sweep == sweeps:
            break
        # ---- the sweep's rounds: (c, s), barrier, blocks + V rows, barrier
        for rd in range(m - 1):
            var p = tid
            while p < h:
                var got = rr_cs(a, n, m, rd, p)
                cs.unsafe_store(2 * p, got[0])
                cs.unsafe_store(2 * p + 1, got[1])
                p += RR_ONE_D_TPB
            dev_barrier()
            var u = tid
            while u < h * h + n * h:
                if u < h * h:
                    var i = u // h
                    var j = u - i * h
                    if i <= j:
                        rr_block(a, cs, n, m, rd, i, j)
                else:
                    var x = u - h * h
                    var kk = x // h
                    rr_vrow(v, cs, n, m, rd, kk, x - kk * h)
                u += RR_ONE_D_TPB
            dev_barrier()


def rs_round_gated_kernel(
    rt: F32Ptr, vt: F32Ptr, flag: F32Ptr, done: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32, tol: Float32
):
    """`rs_round_kernel` (x_decomp/rr_svd_device.mojo) unchanged behind the
    sticky word done[0] (`rs_sweep_gate_kernel`): nothing once converged.
    The load of done[0] is block-uniform, so the early return precedes
    every barrier for all threads alike."""
    if done.unsafe_load(0) != Float32(0.0):
        return
    var n = Int(n_in)
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var pq = rs_pair(Int(round_in), b, Int(m_in))
    if pq[1] >= n:
        return
    var s0 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s1 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s2 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var l = rs_lane3(rt, n, pq[0], pq[1], t)
    s0[t] = l[0]
    s1[t] = l[1]
    s2[t] = l[2]
    barrier()
    var w = RS_TPB // 2
    while w > 0:
        if t < w:
            s0[t] = ftz(s0[t] + s0[t + w])
            s1[t] = ftz(s1[t] + s1[t + w])
            s2[t] = ftz(s2[t] + s2[t + w])
        barrier()
        w = w // 2
    var d = rs_decide(s0[0], s1[0], s2[0], tol)
    if d[0] > Float32(0.0):
        rs_apply_lane(rt, vt, n, pq[0], pq[1], d[1], d[2], t)
        if t == 0:
            flag.unsafe_store(b, Float32(1.0))


def rs_sweep_gate_kernel(first: I32Ptr, done: F32Ptr, no_flag: Int32):
    """After a sweep's `xd_flag_first_kernel`: while not converged, done[1]
    += 1 (sweeps run) and done[0] = 1 when the sweep rotated nothing
    (first[0] still `no_flag`). A gated sweep (done[0] already 1) leaves
    both. One thread."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        if done.unsafe_load(0) == Float32(0.0):
            done.unsafe_store(1, done.unsafe_load(1) + Float32(1.0))
            if first.unsafe_load(0) == no_flag:
                done.unsafe_store(0, Float32(1.0))


def orth_lean_test_kernel(r: F32Ptr, out_flag: F32Ptr, l_in: Int32):
    """out_flag[0] = 1 when some cell (i, j), i < j, of the guarded R
    violates `orth_lean_violates` (the caller fills 0). Every writer stores
    the same word, so the result does not depend on their order. Thread per
    (i, j) of the l x l grid."""
    var l = Int(l_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < l * l:
        var i = t // l
        var j = t - i * l
        if i < j and orth_lean_violates(r, l, i, j):
            out_flag.unsafe_store(0, Float32(1.0))
        elif l == 1 and t == 0 and r.unsafe_load(0) == Float32(0.0):
            out_flag.unsafe_store(0, Float32(1.0))
