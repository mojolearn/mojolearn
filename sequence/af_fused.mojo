# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ADAFACTOR'S VECTOR STEP IN TWO LAUNCHES PLUS THE APPLY (lane neural-io-2,
2026-10-09; -D MOJOLEARN_IDN_AF_VEC_FUSED, default OFF; the reasoning and the
item bodies the host column runs are in sequence/adafactor.mojo, "IDN_AF_VEC_FUSED").

`af_vec_fused_kernel`, one block of AF_VF_TPB threads per AF_NORM_BLOCK
values: every thread runs `op_af_vec` on a strided share of the block's cells
(the variance lerp and the update, element-wise), the block waits, then cell 0
(threads 0-31) folds p's squares and cell 2 (threads 64-95, another warp, and
another wave on a 64-lane CDNA part) folds u's squares, each with
`coop_sumsq`: the one-thread `sumsq_fold` chain with the loads spread over the
32 lanes (sequence/coop.mojo), exactly the chain `OP_AF_BLK_SUMSQ` runs on
this column today. Lane 0 of each stores the block's partial.

`af_vec_finish_kernel`, one block: thread 0 adds p's partials ascending and
runs the alpha tail, thread 64 (another warp / wave) adds u's partials
ascending at the same time; after the wait thread 0 runs the denom tail. The
two tails are `op_af_alpha` / `op_af_denom`'s, in the same order on the same
values as the two one-thread launches they replace."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from sequence.ops import FP, Args, st
from sequence.adafactor import AF_NORM_BLOCK, af_fold_parts, af_vfin_alpha, af_vfin_denom, op_af_vec
from sequence.coop import COOP_W, coop_sumsq

#: threads per block: four 32-lane cells (two waves on a 64-lane part)
comptime AF_VF_TPB = 128


def af_vec_fused_kernel(
    p: FP, g: FP, s1: FP, u: FP, pp: FP, pu: FP, n_in: Int32, w: Float32, eps1sq: Float32,
):
    """Block b: `op_af_vfuse(b, ...)` (sequence/adafactor.mojo), the cells
    spread over the block and the two block chains on two cells."""
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var lo = blk * AF_NORM_BLOCK
    var cnt = min(AF_NORM_BLOCK, n - lo)
    var v = Args()
    v.p0 = g
    v.p1 = s1
    v.p2 = u
    v.f0 = w
    v.f1 = eps1sq
    var i = tid
    while i < cnt:
        op_af_vec(lo + i, v)
        i += AF_VF_TPB
    # every u word of the block stored before its chain reads it
    barrier()
    var cell = tid // COOP_W
    var lane = tid - cell * COOP_W
    if cell == 0:
        var ss = coop_sumsq(p, lo, cnt, lane)
        if lane == 0:
            st(pp, blk, ss)
    elif cell == 2:
        var su = coop_sumsq(u, lo, cnt, lane)
        if lane == 0:
            st(pu, blk, su)


def af_vec_finish_kernel(
    pp: FP, sc: FP, pu: FP, n_in: Int32, nb_in: Int32, eps2: Float32, rho: Float32, d: Float32,
):
    """One block: `op_af_vfin` with its two partial folds on two warps."""
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()
    var a = Args()
    a.p0 = pp
    a.p1 = sc
    a.p2 = pu
    a.i0 = Int(n_in)
    a.i1 = Int(nb_in)
    a.f0 = eps2
    a.f1 = rho
    a.f2 = d
    if tid == 0:
        af_vfin_alpha(a, af_fold_parts(pp, a.i1))
    elif tid == 64:
        sh[0] = af_fold_parts(pu, a.i1)
    barrier()
    if tid == 0:
        af_vfin_denom(a, sh[0])
