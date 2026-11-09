# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/rr_svd.mojo's round on the device: one block a pair, RS_TPB
lanes folding its three sums in shared memory, then every lane rotating its
rows. The driver is `DevExec.svd` (x_decomp/device.mojo); the one-block
form below is the executor op's (x_decomp/svd_ops.mojo)."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_sqrt
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.jacobi2 import dev_barrier
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


# ---- the round-robin one-sided Jacobi as ONE block a problem (lane
# n9b-device-svd-qr, 2026-11-09): the executor SVD op's solve
# (x_decomp/svd_ops.mojo). The same pairs, rounds, sums, tree, decisions and
# rotations as `rs_round_kernel` + `rs_norm_kernel` (and `host_rr_svd`), so
# the same words; the sweeps, the convergence test and the norms run on the
# device with no host read (the round launches read one word home per
# XD_SVD_GROUP sweeps). RS_SLOT_LANES threads take one pair: thread r of a
# slot owns the RS_TPB-lane fold's virtual lanes r, r + RS_SLOT_LANES, ...
# (RS_VLANES of them), so the tree's levels w >= RS_SLOT_LANES add two lanes
# of one thread and the levels below go through shared memory: the 256-lane
# tree's additions in its order. RS_TPB / RS_SLOT_LANES pairs of a round run
# at once (the pairs of a round are disjoint, so their order does not matter).

comptime RS_SLOT_LANES = 32
"""Threads per pair slot of `rs_block_gated_kernel`."""
comptime RS_VLANES = RS_TPB // RS_SLOT_LANES
"""Virtual fold lanes per thread (8)."""
comptime RS_SLOTS = RS_TPB // RS_SLOT_LANES
"""Pairs in flight per block (8)."""


@always_inline
def _rs_tree8(v: SIMD[DType.float32, 8]) -> Float32:
    """A thread's RS_VLANES virtual lanes through the 256-lane tree's levels
    w = 128, 64, 32 (lane r + 32 j pairs with lane r + 32 (j + w / 32))."""
    var a0 = ftz(v[0] + v[4])
    var a1 = ftz(v[1] + v[5])
    var a2 = ftz(v[2] + v[6])
    var a3 = ftz(v[3] + v[7])
    var b0 = ftz(a0 + a2)
    var b1 = ftz(a1 + a3)
    return ftz(b0 + b1)


def rs_block_gated_kernel(
    gate: I32Ptr, rt_all: F32Ptr, vt_all: F32Ptr, s_all: F32Ptr, marks: F32Ptr, n_in: Int32, sweeps_in: Int32,
    tol: Float32,
):
    """Problem block_idx.x: rt (n x n, R^T, consumed) rotated until a sweep
    rotates nothing or `sweeps_in` sweeps ran, vt = V^T (set to I here),
    s = the rows' norms; marks[2 b] = 1 converged / 0 not, marks[2 b + 1] =
    the sweeps run. Every block returns at once (before any barrier) when
    gate[0] != 0."""
    comptime assert RS_TPB == 256 and RS_SLOT_LANES == 32, "the tree split assumes 256 lanes in slots of 32"
    if gate.unsafe_load(0) != Int32(0):
        return
    var n = Int(n_in)
    var sweeps = Int(sweeps_in)
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var rt = rt_all + bid * n * n
    var vt = vt_all + bid * n * n
    var s = s_all + bid * n
    var g = tid // RS_SLOT_LANES
    var r = tid - g * RS_SLOT_LANES
    var base = g * RS_SLOT_LANES
    var s0 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s1 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s2 = stack_allocation[RS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = tid
    while t < n * n:
        var i = t // n
        vt.unsafe_store(t, Float32(1.0) if i * n + i == t else Float32(0.0))
        t += RS_TPB
    dev_barrier()
    var m = n + (n % 2)
    var h = m // 2
    var converged = n < 2
    var executed = 0
    while not converged and executed < sweeps:
        executed += 1
        if tid == 0:
            flag[0] = Int32(0)
        dev_barrier()
        for rd in range(m - 1):
            var b0 = 0
            while b0 < h:
                var pb = b0 + g
                var p = 0
                var q = n
                if pb < h:
                    var pq = rs_pair(rd, pb, m)
                    p = pq[0]
                    q = pq[1]
                var live = q < n
                var vp = SIMD[DType.float32, 8](0.0)
                var vq = SIMD[DType.float32, 8](0.0)
                var vx = SIMD[DType.float32, 8](0.0)
                if live:
                    comptime for j in range(RS_VLANES):
                        var l = rs_lane3(rt, n, p, q, r + j * RS_SLOT_LANES)
                        vp[j] = l[0]
                        vq[j] = l[1]
                        vx[j] = l[2]
                s0[tid] = _rs_tree8(vp)
                s1[tid] = _rs_tree8(vq)
                s2[tid] = _rs_tree8(vx)
                dev_barrier()
                var w = RS_SLOT_LANES // 2
                while w > 0:
                    if r < w:
                        s0[tid] = ftz(s0[tid] + s0[tid + w])
                        s1[tid] = ftz(s1[tid] + s1[tid + w])
                        s2[tid] = ftz(s2[tid] + s2[tid + w])
                    dev_barrier()
                    w = w // 2
                if live:
                    var d = rs_decide(s0[base], s1[base], s2[base], tol)
                    if d[0] > Float32(0.0):
                        comptime for j in range(RS_VLANES):
                            rs_apply_lane(rt, vt, n, p, q, d[1], d[2], r + j * RS_SLOT_LANES)
                        if r == 0:
                            flag[0] = Int32(1)
                dev_barrier()
                b0 += RS_SLOTS
        converged = flag[0] == Int32(0)
        dev_barrier()
    # the norms, RS_SLOTS columns at once, the same lanes and tree
    var j0 = 0
    while j0 < n:
        var jc = j0 + g
        var vs = SIMD[DType.float32, 8](0.0)
        if jc < n:
            comptime for j in range(RS_VLANES):
                vs[j] = rs_lane_sq(rt, n, jc, r + j * RS_SLOT_LANES)
        s0[tid] = _rs_tree8(vs)
        dev_barrier()
        var w = RS_SLOT_LANES // 2
        while w > 0:
            if r < w:
                s0[tid] = ftz(s0[tid] + s0[tid + w])
            dev_barrier()
            w = w // 2
        if jc < n and r == 0:
            s.unsafe_store(jc, ftz(identical_sqrt(s0[base])))
        dev_barrier()
        j0 += RS_SLOTS
    if tid == 0:
        marks.unsafe_store(2 * bid, Float32(1.0) if converged else Float32(0.0))
        marks.unsafe_store(2 * bid + 1, Float32(executed))
