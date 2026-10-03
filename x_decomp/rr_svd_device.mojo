# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/rr_svd.mojo's round on the device: one block a pair, RS_TPB
lanes folding its three sums in shared memory, then every lane rotating its
rows. The driver is `DevExec.svd` (x_decomp/device.mojo)."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_sqrt
from x_decomp.cells import F32Ptr
from x_decomp.rr_svd import RS_TPB, rs_apply_lane, rs_decide, rs_lane3, rs_lane_sq, rs_pair


def rs_round_kernel(rt: F32Ptr, vt: F32Ptr, flag: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32, tol: Float32):
    """Pair block_idx.x of round `round_in`; flag[pair] = 1 when it rotated
    (a sweep's flags are cleared before its first round)."""
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


def rs_norm_kernel(rt: F32Ptr, s: F32Ptr, n_in: Int32):
    """s[j] = ||row j of R^T|| (block j, the pair fold's lanes and tree)."""
    var n = Int(n_in)
    var j = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var s0 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    s0[t] = rs_lane_sq(rt, n, j, t)
    barrier()
    var w = RS_TPB // 2
    while w > 0:
        if t < w:
            s0[t] = ftz(s0[t] + s0[t + w])
        barrier()
        w = w // 2
    if t == 0:
        s.unsafe_store(j, ftz(identical_sqrt(s0[0])))
