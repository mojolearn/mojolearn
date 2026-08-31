# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The within-block float fold, with ONE shape on every vendor.

DEVIATION 504 (IDENTITY_PATHS row 20). Reached only under
`NUMERIC_IDENTICAL`; the FAST arm IS the library call.

NOT A PORT. cuVS, cuML and RAFT each ship one GPU backend and reduce with
whatever CUB gives them; the question this file answers -- *does the fold
combine the same partials in the same order on Metal, CUDA and HIP* -- only
exists because we ship three backends from one source.

WHY A LIBRARY `block.sum` IS NOT ENOUGH
---------------------------------------
`max.gpu.primitives.block.sum` is correct, tuned, and NOT machine
independent: its internal cross-lane stage folds at the HARDWARE'S warp
width, which is 32 on Apple and NVIDIA and **64 on AMD's CDNA wavefront**.
Float addition is not associative, so a 64-wide fold and a 32-wide fold of
the same 128 values are two different sums of the same multiset, and they
differ in the last bits. That is IDENTITY_PATHS row 8's AMD residue, found
by the GBDT lane on `pointwise_targets.mojo` and closed there at 14 of 16
producer sites with a fold of exactly this shape (DEVIATION 251).

The unsupervised sections had the same residue at every `block.sum` they
call -- `core/row_norms.mojo`, `cluster/mojo_only/reduce_by_key.mojo`,
`cluster/mojo_only/plus_plus.mojo` -- and no equivalent. This file is that
equivalent, and rows 19-24 of the ledger route through it.

WHY IT IS A SECOND COPY, WHICH IS A COST AND IS DELIBERATE
-----------------------------------------------------------
`gbdt/targets/kernel/pointwise_targets.mojo:68` already has
`pinned_block_sum`, spelled the same way and folding the same shape.
Importing THAT one here would make the identity claim ride on one
implementation, which is what this repository normally insists on
(`core/identity_trace.mojo`: "a second hash function in one repository is a
second thing to get wrong").

It is duplicated anyway because that file is a CatBoost-ported device-kernel
module owned by another lane, importing it drags its whole compile into
`cluster/` and `neighbors/`, and cross-lane edits to hot files are how two
sessions collide. The duplication is therefore a LANE boundary, not a
judgement that two folds are fine.

**The debt is named, not hidden:** the two must stay the same shape, and
`check_pinned_fold_shape` in `cluster/mojo_only/kmeans_identity_check.mojo`
gates the property that matters -- that this fold is a pure function of the
value vector and NOT of the lane width -- by folding the same inputs at
several block widths and requiring the halving tree to agree with a host
computation done in the same order. When the GBDT lane next touches
`pointwise_targets.mojo`, the merge is one import: delete its copy, import
this one, run both suites.

THE CONTRACT, and it is the same one `block.sum` already carries
-----------------------------------------------------------------
- EVERY thread of the block must call this. Threads with no data pass 0.0.
- Only thread 0's return value is meaningful.
- `block_size` must be a power of two, so the halving fold is exact.
- The trailing `barrier()` protects the shared slab so back-to-back calls
  in one kernel are safe.

Under `NUMERIC_FAST` this is the library call, bit for bit, and compiles to
exactly what was there before.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


@always_inline
def pinned_block_sum[block_size: Int](value: Float32) -> Float32:
    """`block.sum` under FAST; a lane-width-independent halving tree under
    IDENTICAL.

    The IDENTICAL arm writes every thread's value into a `block_size` slab
    of threadgroup memory and folds it `red[t] += red[t + step]` for
    `step = block_size/2, ..., 1`. No warp primitive appears anywhere in
    it, so nothing in the fold can consult the hardware's lane width, and
    the sequence of additions is a pure function of `block_size` -- the
    same on Metal, PTX and AMDGPU.

    It is NOT the same sum as the library call: a halving tree and CUB's
    warp-then-block shape combine different partials. IDENTICAL bits
    therefore differ from FAST bits on Apple BY DESIGN, exactly as
    `identical_mul_add` does, and what is purchased is that IDENTICAL's
    bits are the same bits everywhere.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var tid = Int(thread_idx.x)
        var red = stack_allocation[
            block_size,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        red[tid] = value
        barrier()
        var step = block_size // 2
        while step > 0:
            if tid < step:
                red[tid] = red[tid] + red[tid + step]
            barrier()
            step //= 2
        var total = red[0]
        barrier()
        return total
    else:
        return block_sum[block_size=block_size](value)


# ===========================================================================
# THE SELECTIONS (DEVIATION 528, IDENTITY_PATHS row 30's compile half)
# ===========================================================================
#
# `pinned_block_sum` above exists because a SUM's fold shape changes its
# bits. A MAX or a MIN is a selection over a total order, exactly associative
# and commutative, so no fold shape can move it and by that reasoning these
# two should not need to exist at all.
#
# THEY EXIST FOR A DIFFERENT REASON, AND IT IS NOT NUMERIC: the library
# primitive REFUSES TO COMPILE at these block sizes on a 64-wide wavefront.
#
#     max/mojo/max/gpu/primitives/block.mojo:186:
#     constraint failed: Block size must be a greater than warp size
#
# Measured on an MI325X, 2026-08-23, by the E2 lane's AMD leg:
# `bindings/build_estimators.sh` FAILED there, and the two call sites were
# `block_max`/`block_min` at `SIGNFLIP_TPB = 32` in
# `decomposition/ported/linalg/detail/pca.mojo`. So **PCA and truncated SVD
# did not build on AMD at all**, in either mode, while every gate on this
# side was green -- a whole-section failure that one M4 cannot see and that
# no amount of bit-comparison would have found, because there were no bits.
#
# 32 is not an arbitrary choice there either: the sign rule's tie-break is
# stated over lanes and the fixtures plant ties both across and within a
# 32-lane group, so widening the block to satisfy the constraint would move
# what the check is checking. Replacing the primitive is the smaller change.
#
# NO MODE GATE, deliberately, and this is the difference from
# `pinned_block_sum`. A halving selection returns exactly what CUB's shape
# returns, so there is no FAST arm to preserve and no IDENTICAL arm to buy
# anything: swapping these in moves ZERO bits on every column, which
# `check_sign_flip_matches_host_rule` asserts against a fold-free host scan.
# A mode gate here would imply a difference that does not exist.
#
# THE ±0.0 AND NaN CAVEAT, because a selection is only exactly commutative
# away from them (IDENTITY_PATHS row 13). `max(+0.0, -0.0)` may return
# either, so WHICH zero survives is a fold-order question after all, and a
# NaN operand makes `>` false in both directions. Neither reaches these at
# their one call site: pass 1 folds `abs(...)`, and `abs(-0.0)` is `+0.0`, so
# no negative zero is ever compared; and the per-thread partial is seeded
# `+0.0` and only updated on a strict `>`, so a NaN never enters it. A future
# caller that cannot say the same about its inputs must state why before
# using these.


@always_inline
def pinned_block_max[block_size: Int](value: Float32) -> Float32:
    """Block-wide max through threadgroup memory, no lane primitive.

    Same contract as `pinned_block_sum`: EVERY thread of the block calls
    it, threads with no data pass the identity (`+0.0` for a fold over
    magnitudes), only thread 0's return is meaningful, `block_size` is a
    power of two, and the trailing `barrier()` protects the slab so
    back-to-back calls in one kernel are safe.

    Unlike the sum, this is bit-for-bit the library's answer on any width;
    see the block comment above for why it exists anyway.
    """
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var other = red[tid + step]
            if other > red[tid]:
                red[tid] = other
        barrier()
        step //= 2
    var total = red[0]
    barrier()
    return total


@always_inline
def pinned_block_min[block_size: Int](value: Float32) -> Float32:
    """Block-wide min through threadgroup memory, no lane primitive.

    The twin of `pinned_block_max`; same contract, and the identity a
    dataless thread passes is whatever sentinel the caller uses as its "no
    candidate" value (at the sign-flip call site that is `Float32(n)`).
    """
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var other = red[tid + step]
            if other < red[tid]:
                red[tid] = other
        barrier()
        step //= 2
    var total = red[0]
    barrier()
    return total
