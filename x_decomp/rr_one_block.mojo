# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The whole round-robin Jacobi solve in ONE launch of ONE block (lane fg-pca
P1 + P1b, 2026-10-09; switch `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK`, default OFF,
read in decomposition/pca_rr_switch.mojo). NOT COMPILED, NOT VERIFIED, NOT
MEASURED.

What it replaces (PCA / TruncatedSVD `_eig_rr_device`, and the same shape of
loop in x_decomp `DevExec._eigh_par_on`): per sweep three test launches
(`eigh_par_off_part_kernel`, `eigh_par_off_fold_kernel`,
`pca_rr_gate_kernel`), a host read of the six state words, then m - 1
rounds of two launches each (`pca_rr_cs_kernel`, `pca_rr_update_kernel`):
2 (m - 1) + 3 launches and one wait a sweep.

Here one block of RR_ONE_TPB threads runs every sweep: the test (the block
trees of RR_OFF_TPB rows, RR_ONE_TPB / RR_OFF_TPB virtual blocks at a time,
each group of RR_OFF_TPB threads running exactly the part kernel's tree over
its own slots; then the fold past the blocks on the first group, exactly the
fold kernel's; then `rr_gate_state` on thread 0), then each round's (c, s)
(`rr_cs`, one thread a pair: P1b's fused cs) and, behind a barrier, the
round's 2 x 2 blocks (`rr_block`) and V rows (`rr_vrow`), behind another.
The stop is the device's own verdict (state[0] converged or state[4] a test
block that did not run), the PCA_RR_FLAG_TEST rule, decided without a host
read.

BITS: none. Every cell is the multi-launch solver's function on the same
words in the same order: a round's cells have one writer and no other
reader (x_decomp/rr.mojo), so running them in strided loops of one block
instead of across a grid stores the same words, and the barrier between
the (c, s) pass and the update pass is the launch boundary it replaces. The
test folds in `rr_off_fold`'s order (same block trees, same fold, same
gate statements). The host column (`host_eigh_rr`) is unchanged.

COST RULE (`rr_one_block_applies`, from size and launch cost, no board
width): a round costs the multi-launch solver two launches (about 5 us each
on NVIDIA, 10-15 us on AMD, plus the tiny kernels' own time); in one block it
costs ceil((h h + n h) / RR_ONE_TPB) strided cell steps, each a handful of
L1/L2-resident loads and fmas with RR_ONE_TPB / 32 warps hiding the latency.
RR_ONE_BLOCK_STEPS (default 64 steps a thread a round, override
`-D MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS=<k>`) is the point where one block's
round is estimated to stop beating two launches; past it the multi-launch
solver (whose rounds use the whole GPU) keeps the solve. The sweep's test
adds ceil(n / RR_OFF_TPB) row passes, small beside the rounds.
"""
from std.gpu import thread_idx
from std.memory import stack_allocation
from std.sys.defines import get_defined_int
from max.gpu.memory import AddressSpace

from checks.numerics import ftz
from x_decomp.cells import F32Ptr
from x_decomp.jacobi2 import dev_barrier
from x_decomp.rr import RR_OFF_TPB, rr_block, rr_cs, rr_gate_state, rr_row_off, rr_vrow

#: threads of the one block (the most every column launches: 1024 on NVIDIA,
#: AMD and Apple); a multiple of RR_OFF_TPB so the test's virtual blocks map
#: onto whole thread groups
comptime RR_ONE_TPB = 1024
comptime RR_ONE_GROUPS = RR_ONE_TPB // RR_OFF_TPB
#: strided cell steps a thread a round at which the one block is estimated to
#: stop beating two launches a round (module header)
comptime RR_ONE_BLOCK_STEPS = get_defined_int["MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS", 64]()


def rr_one_block_applies(n: Int) -> Bool:
    """Whether the one-block solve is the cheaper schedule at width n (the
    module header's cost rule): the round's h h + n h cells in at most
    RR_ONE_BLOCK_STEPS strided steps of RR_ONE_TPB threads."""
    var m = n + (n % 2)
    var h = m // 2
    return n >= 2 and h * h + n * h <= RR_ONE_BLOCK_STEPS * RR_ONE_TPB


def rr_eigh_one_block_kernel(
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
    identity), the decision in `state` (PCA_RR_STATE words, already
    initialized as `_eig_rr_device` does: zeros, state[2] = -1). Scratch as
    the multi-launch solver's: cs 2 h, doff 3 n (doff[2 n + k] = a_kk at the
    last test, as `eigh_par_off_part_kernel` leaves it), part 3 nb, fold 3.
    Launch grid 1, block RR_ONE_TPB."""
    var n = Int(n_in)
    var sweeps = Int(sweeps_in)
    var m = n + (n % 2)
    var h = m // 2
    var nb = (n + RR_OFF_TPB - 1) // RR_OFF_TPB
    var tid = Int(thread_idx.x)
    var grp = tid // RR_OFF_TPB
    var lt = tid - grp * RR_OFF_TPB
    var so = stack_allocation[RR_ONE_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_ONE_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sm = stack_allocation[RR_ONE_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stop = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for sweep in range(sweeps + 1):
        # ---- the test, part: `eigh_par_off_part_kernel` block b on group
        # b % RR_ONE_GROUPS, pass b / RR_ONE_GROUPS; each group's tree runs on
        # its own RR_OFF_TPB slots (lt + w < RR_OFF_TPB stays in the group)
        var passes = (nb + RR_ONE_GROUPS - 1) // RR_ONE_GROUPS
        for ps in range(passes):
            var b = ps * RR_ONE_GROUPS + grp
            var k = b * RR_OFF_TPB + lt
            var o = SIMD[DType.float32, 2](0.0, 0.0)
            if b < nb and k < n:
                o = rr_row_off(a, n, k)
                doff.unsafe_store(2 * n + k, a.unsafe_load(k * n + k))
            so[tid] = o[0]
            sd[tid] = o[1]
            dev_barrier()
            var w = RR_OFF_TPB // 2
            while w > 0:
                if lt < w:
                    so[tid] = ftz(so[tid] + so[tid + w])
                    sd[tid] = ftz(sd[tid] + sd[tid + w])
                dev_barrier()
                w = w // 2
            if lt == 0 and b < nb:
                part.unsafe_store(3 * b, so[tid])
                part.unsafe_store(3 * b + 1, sd[tid])
                part.unsafe_store(3 * b + 2, Float32(0.0))
            dev_barrier()
        # ---- the test, fold: `eigh_par_off_fold_kernel` on group 0
        var ao = Float32(0.0)
        var ad = Float32(0.0)
        var am = Float32(0.0)
        if tid < RR_OFF_TPB:
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
        if stop[0] != Float32(0.0):
            break
        if sweep == sweeps:
            break
        # ---- the sweep's rounds: (c, s), barrier, blocks + V rows, barrier
        for rd in range(m - 1):
            var p = tid
            while p < h:
                var got = rr_cs(a, n, m, rd, p)
                cs.unsafe_store(2 * p, got[0])
                cs.unsafe_store(2 * p + 1, got[1])
                p += RR_ONE_TPB
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
                u += RR_ONE_TPB
            dev_barrier()
