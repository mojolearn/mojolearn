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
