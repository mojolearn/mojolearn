"""The metrics' float sums over `n`, with ONE order on every vendor and on
every LAUNCH.

DEVIATION 653 (metrics lane, 2026-08-23). NOT A PORT. RAFT's metric sums are
`thrust::reduce` (`scores.cuh:69-70`, the batched silhouette's `:265`) and
`mapThenSumReduce` (`kl_divergence.cuh:62`, `entropy.cuh:133`,
`silhouette_score.cuh:294`) -- a CUB block fold whose partials land in ONE
float slot through `atomicAdd` (`raft/linalg/detail/map_then_reduce.cuh:
33-38`). Two runs of that on one GPU are entitled to two last bits; two
vendors certainly are. RAFT ships one backend and accepts it. We ship three
from one source, so the sum is REPLACED under `NUMERIC_IDENTICAL`, not
configured (the `mojo_only/numerics.mojo` header: "IDENTICAL has to REPLACE
the accumulator").

WHAT IS PINNED, AND WHAT IT IS A FUNCTION OF
--------------------------------------------
`virtual_block_sum` folds `PINNED_SUM_W` values as a halving tree over a
threadgroup slab of exactly that width -- the SAME tree `core/pinned_reduce.
mojo::pinned_block_sum` folds at `block_size == PINNED_SUM_W` (gated by
`check_virtual_sum_equals_pinned_block_sum`) -- but the slab width is a
CONSTANT of this file, not the launch's block size. A 64-thread block and a
256-thread block fill the same 256 slots and perform the same additions in
the same order; the threads only differ in how many slots each one serves.
That is what makes the headline launch-invariance gate (two block sizes, two
grid shapes) provable for a REDUCTION, where `pinned_block_sum[block_size]`
alone would make the fold a function of the launch.

The values of one chunk are the `PINNED_SUM_W` consecutive indices
`[chunk * W, (chunk + 1) * W)`, positions past `n` hold `+0.0`. Each chunk's
total is written to `partials[chunk]`, and the host folds the partials
ASCENDING (`host_fold_partials`). So the whole sum is a pure function of `n`
and the values: `ceil(n / W)` chunks, a fixed tree per chunk, a fixed serial
fold over chunks. Grid shape, block size, how many chunks one physical block
serves: none of it reaches an addition.

`ftz` is applied to every stored partial (row 10): a denormal partial is
flushed on Apple by the hardware and on a denormal-honoring vendor by the
call, so the stored bits agree. Apple FAST/IDENTICAL bits at those seams do
not move.

THE FAST ARM IS THE LIBRARY'S SHAPE. Under `NUMERIC_FAST` each thread sums
the slots it serves serially and the block folds with `block.sum` (CUB's
warp-then-block shape, the same primitive `pinned_block_sum`'s FAST arm is),
and the partials are still written per chunk and folded on the host -- the
repository does not use a float `atomicAdd` anywhere (reduce_by_key.mojo's
header says why), so the one place this differs from RAFT's FAST shape is
that the partials are added in a fixed order instead of an arrival order.
FAST bits are a REPORT, never an assertion.

THE HOST MODEL (`host_tree_sum`) performs the identical sequence of Float32
additions through the same `ftz`, so the device is gated BIT FOR BIT against
it under IDENTICAL. It is the oracle every Group B and C check reads first.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz


#: NUMERIC: the slab width. A different W is a different tree.
comptime PINNED_SUM_W = 256

#: SCHEDULING: the launch width the drivers use by default. The checks launch
#: at 64 and 256 to prove the tree does not see it.
comptime PINNED_SUM_TPB = 256


def chunk_count(n: Int) -> Int:
    """How many `PINNED_SUM_W` chunks cover `n` values. Pure function of n."""
    return (n + PINNED_SUM_W - 1) // PINNED_SUM_W


@always_inline
def linear_block_id() -> Int:
    """The chunk a physical block serves first, from a 1-D or 2-D grid. The
    grid's SHAPE is scheduling (row x, row y); the chunk INDEX it resolves
    to is what the tree sees, and that index is the same whichever shape
    carried it."""
    return Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)


@always_inline
def physical_block_count() -> Int:
    return Int(grid_dim.x) * Int(grid_dim.y)


@always_inline
def virtual_block_sum[
    block_size: Int
](vals: SIMD[DType.float32, PINNED_SUM_W // block_size]) -> Float32:
    """Fold one chunk of `PINNED_SUM_W` values. EVERY thread of the block
    calls this; thread `t` supplies the values of slots `t + r * block_size`
    for `r` in `[0, W / block_size)` in `vals[r]` (`+0.0` where there is no
    value). Only thread 0's return is meaningful. `block_size` must divide
    `PINNED_SUM_W` and be a power of two. The trailing `barrier()` makes
    back-to-back calls in one kernel safe.

    IDENTICAL: the slab tree described in the header. FAST: serial per
    thread, then `block.sum`."""
    comptime R = PINNED_SUM_W // block_size
    comptime assert R * block_size == PINNED_SUM_W, (
        "virtual_block_sum: block_size must divide PINNED_SUM_W"
    )
    var tid = Int(thread_idx.x)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var slab = stack_allocation[
            PINNED_SUM_W,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        comptime for r in range(R):
            slab[unsafe_offset = tid + r * block_size] = ftz(vals[r])
        barrier()
        var step = PINNED_SUM_W // 2
        while step > 0:
            var t = tid
            while t < step:
                slab[unsafe_offset = t] = ftz(slab[unsafe_offset = t] + slab[unsafe_offset = t + step])
                t += block_size
            barrier()
            step //= 2
        var total = slab[unsafe_offset = 0]
        barrier()
        return total
    else:
        var mine = Float32(0.0)
        comptime for r in range(R):
            mine += vals[r]
        return block_sum[block_size=block_size](mine)


def host_tree_sum(values: List[Float32], n: Int) -> Float32:
    """The host model of `virtual_block_sum` over every chunk plus
    `host_fold_partials`: the same additions in the same order, through
    the same `ftz`. `values[0:n]` are the terms; the caller has already
    applied the per-term map through the same helpers the kernel uses."""
    var partials = List[Float32]()
    var chunks = chunk_count(n)
    var slab = List[Float32]()
    for _ in range(PINNED_SUM_W):
        slab.append(Float32(0.0))
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            slab[t] = ftz(values[i]) if i < n else Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        partials.append(slab[0])
    return host_fold_partials(partials, chunks)


def host_fold_partials(partials: List[Float32], chunks: Int) -> Float32:
    """The second stage: the chunk totals, ascending, serially, each
    partial stored through `ftz`. Starts from `+0.0` so a single chunk's
    total passes through one addition like every other (and so `-0.0`
    cannot survive as the sum of nothing)."""
    var acc = Float32(0.0)
    for c in range(chunks):
        acc = ftz(acc + partials[c])
    return acc


def sabotage_rotated_host_tree_sum(
    values: List[Float32], n: Int, rotate_by: Int
) -> Float32:
    """FOR THE SABOTAGE GATE ONLY. The same tree with the fold START rotated
    by `rotate_by` slots inside each chunk -- the addition that a different
    block count, warp width or arrival order would perform. It is the
    wrong answer the checks must be able to tell from the right one; see
    the README's sabotage table."""
    var partials = List[Float32]()
    var chunks = chunk_count(n)
    var slab = List[Float32]()
    for _ in range(PINNED_SUM_W):
        slab.append(Float32(0.0))
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + ((t + rotate_by) % PINNED_SUM_W)
            slab[t] = ftz(values[i]) if i < n else Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        partials.append(slab[0])
    return host_fold_partials(partials, chunks)
