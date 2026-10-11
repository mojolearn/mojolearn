# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device kernels of lane xd-rr-one-block (2026-10-11; switches and cost
rule in x_decomp/xd_lean.mojo). NOT COMPILED, NOT VERIFIED, NOT MEASURED.

C3 `rr_eigh_one_block_d_kernel` (MOJOLEARN_IDN_RR_ONE_BLOCK_D) deleted 2026-10-11 (loser).

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
from x_decomp.xd_lean import orth_lean_violates


# TOMBSTONE: MOJOLEARN_IDN_RR_ONE_BLOCK_D (loser) deleted 2026-10-11 by SHA_RR; code recoverable at SHA_RR^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_ONE_BLOCK_D.patch


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
