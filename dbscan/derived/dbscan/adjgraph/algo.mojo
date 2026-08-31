# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Boolean adjacency to CSR.

PORT OF `cuml/cpp/src/dbscan/adjgraph/algo.cuh` at cuML `00094f7`.
Partial. Do not improve.

Their `launcher` is two calls and both are here:

    thrust::exclusive_scan(policy, dev_vd, dev_vd + batch_size, dev_ex_scan);
    raft::sparse::convert::adj_to_csr(handle, adj, data.ex_scan, num_rows,
                                      num_cols, row_counters, data.adj_graph);

**THE CSR IS BUILT FROM EVERY ROW, CORE OR NOT.** This was first written with
a core-point filter in both the scan and the compaction, which was an
invention: their `thrust::exclusive_scan` runs over the whole `vd` array with
no mask, and `adj_to_csr` emits every non-zero.

The core restriction is real but it lives one step later, in `weak_cc`'s
`filter_op` (`dbscan/runner.cuh:384`), which decides which vertices may
PROPAGATE a label. Putting it here instead gives a graph missing the
border-point edges that the labeler still needs to see in order to attach
borders to clusters. Same idea, wrong place, and the fix was to read their
file instead of describing it.

**Their compaction is deliberately UNORDERED**, and their own docstring says
so: "High performance comes at the cost of non-deterministic output: the
column indices are not guaranteed to be stored in order"
(`raft/sparse/convert/detail/adj_to_csr.cuh`). Multiple blocks cooperate on
one row through an atomic counter. That is the THIRD documented source of
run-to-run nondeterminism found in these upstreams, after CatBoost's float
atomics and RAFT's radix-select tie handling. It does not change DBSCAN's
answer, because label propagation converges to the same fixed point whatever
order the edges are visited in.

Ours is now theirs: a shared per-row cursor bumped with an atomic, chunked
16-bool loads, and the same unordered output. What is still not theirs is the
warp aggregation of that atomic and the multi-block-per-row grid, both priced
in deviation 34.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime SCAN_TPB = 256
# Elements one block of the first scan pass owns. `SCAN_TPB` threads times
# eight elements each, which is the shape `cub::BlockScan` is used in
# throughout these upstreams.
comptime SCAN_ITEMS_PER_TH = 8
comptime SCAN_ELEMS_PER_BLOCK = SCAN_TPB * SCAN_ITEMS_PER_TH


def scan_blocks_needed(n: Int) -> Int:
    """Blocks the first scan pass launches, and the length `block_sums` needs
    minus one."""
    return (n + SCAN_ELEMS_PER_BLOCK - 1) // SCAN_ELEMS_PER_BLOCK


# ---------------------------------------------------------------------------
# DEVIATION BLOCK 32: SCAN-THEN-PROPAGATE, NOT SINGLE-PASS DECOUPLED LOOKBACK
#
# THEIRS: `thrust::exclusive_scan` (`adjgraph/algo.cuh:65`). Thrust dispatches
#   this to `cub::DeviceScan`, whose kernel is a SINGLE pass over the data:
#   each tile computes its local scan, publishes its aggregate, and then
#   inspects its predecessors' published aggregates (decoupled lookback) to
#   resolve its own prefix without a second read of the input.
# OURS: three launches -- local scan, scan of the per-block totals, add back.
#   Two full passes over `ex_scan` instead of one.
# REASON: their kernel is READABLE and therefore a port candidate, not a
#   substitution candidate; what is not portable is the mechanism decoupled
#   lookback rests on. It needs a forward-progress guarantee between blocks
#   (a block spins on a flag written by a lower-numbered block) plus a memory
#   fence with release/acquire ordering across threadgroups. Metal makes no
#   such scheduling promise. (Device-scope release/acquire itself IS
#   expressible -- `Atomic.store[RELEASE]` / `Atomic.load[ACQUIRE]` legalize
#   on the Apple target and `neighbors/mutex_probe_main.mojo` established
#   them sound on 2026-08-19 -- but only under the co-residency that
#   `launchConfigGenerator`-capped grids guarantee. Lookback's grid scales
#   with the DATA, not the device, so a block can spin on a predecessor the
#   scheduler has not started, and a literal port can deadlock rather than
#   run slowly.) Scan-then-propagate is the same algorithm with the
#   inter-block communication moved into a second launch, which the driver
#   orders for us. Cost: one extra read and write of `ex_scan`, `batch_size`
#   elements, against a kernel that is `batch_size * N` bytes of adjacency in
#   the same fit -- under a thousandth of the traffic either way.
#
#   `nn.cumsum` is NOT the answer and was never available: it carries neither
#   a `ctx: DeviceContext` nor a `target` parameter (installed signature
#   `cumsum[dtype, exclusive, reverse, *, axis](output, input)`, the same
#   shape as the CPU-only `max.algorithm.reduction.cumsum` and unlike
#   `nn.argsort`/`nn.topk`/`nn.gather`, which take both), so there is no way
#   to enqueue it. `VENDOR_LIBS.md` already said so; this re-confirms it.
#
#   What this replaced was not a device scan at all: `exclusive_scan_kernel`
#   was launched with `grid_dim = (1, 1, 1)`, so a single threadgroup walked
#   the whole array. At 200,000 rows that is one core of the M4 doing 200,000
#   serial adds while the other nine idle, once per batch, twice per fit.
# ---------------------------------------------------------------------------


def exclusive_scan_pass1_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """Each block exclusive-scans its own `SCAN_ELEMS_PER_BLOCK` slice.

    The per-thread block scan is `max.gpu.primitives.block.prefix_sum` with
    `exclusive=True`, which is `cub::BlockScan::ExclusiveSum`'s counterpart.
    Every block writes its slice total to `block_sums[blockIdx.x]`; pass 2
    scans those and pass 3 adds them back.
    """
    var n_rows = Int(n_rows_in)
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * SCAN_ELEMS_PER_BLOCK

    var begin = base + tid * SCAN_ITEMS_PER_TH
    var end = min(begin + SCAN_ITEMS_PER_TH, n_rows)
    if begin > n_rows:
        begin = n_rows
    if end < begin:
        end = begin

    var s = Int32(0)
    var i = begin
    while i < end:
        s += vd.unsafe_load(i)
        i += 1

    var offset = prefix_sum[block_size=SCAN_TPB, exclusive=True](s)

    var running = offset
    i = begin
    while i < end:
        ex_scan.unsafe_store(i, running)
        running += vd.unsafe_load(i)
        i += 1

    if tid == SCAN_TPB - 1:
        block_sums.unsafe_store(Int(block_idx.x), offset + s)


def exclusive_scan_pass2_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_blocks_in: Int32,
):
    """One block, exclusive-scanning ONE ENTRY PER BLOCK of pass 1.

    This is the only serial step left and it is `n / 2048` long. The chunk
    per thread is computed from `n_blocks`, not fixed: the first version of
    the scan in this file gave each thread a FIXED 64 rows against 256
    threads, which capped it at 16,384 elements, and past that it simply
    stopped scanning and returned a wrong `nnz` and a truncated CSR without
    raising. Found by audit, not by a test, because every fixture in this
    repository was smaller than the cap.

    `block_sums[n_blocks]` ends as the grand total, which pass 3 writes to
    `ex_scan[n_rows]`.
    """
    var n_blocks = Int(n_blocks_in)
    var tid = Int(thread_idx.x)

    var chunk = (n_blocks + SCAN_TPB - 1) // SCAN_TPB
    var begin = tid * chunk
    var end = min(begin + chunk, n_blocks)
    if begin > n_blocks:
        begin = n_blocks
    if end < begin:
        end = begin

    var s = Int32(0)
    var i = begin
    while i < end:
        s += block_sums.unsafe_load(i)
        i += 1

    var offset = prefix_sum[block_size=SCAN_TPB, exclusive=True](s)
    barrier()

    var running = offset
    i = begin
    while i < end:
        var v = block_sums.unsafe_load(i)
        block_sums.unsafe_store(i, running)
        running += v
        i += 1

    if tid == SCAN_TPB - 1:
        block_sums.unsafe_store(n_blocks, offset + s)


def exclusive_scan_pass3_kernel(
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_blocks_in: Int32,
):
    """Add each block's offset back, and plant the total at `ex_scan[n]`.

    `ex_scan[n_rows]` is OURS and not theirs. Their `weak_cc_label_device`
    calls `get_stop_idx(tid, batch_size, nnz, row_ind)` to find a row's end,
    branching on the last row and falling back to `nnz`; their own TODO one
    line above it (`raft/sparse/detail/csr.cuh:72`) says "add one element to
    row_ind and avoid get_stop_idx". That is what this does, so the CSR here
    has `batch_size + 1` row offsets and every consumer reads `row_ind[i+1]`
    with no special case.
    """
    var n_rows = Int(n_rows_in)
    var add = block_sums.unsafe_load(Int(block_idx.x))
    var i = (
        Int(block_idx.x) * SCAN_ELEMS_PER_BLOCK + Int(thread_idx.x)
    )
    var stop = min(
        (Int(block_idx.x) + 1) * SCAN_ELEMS_PER_BLOCK, n_rows
    )
    while i < stop:
        ex_scan.unsafe_store(i, ex_scan.unsafe_load(i) + add)
        i += SCAN_TPB

    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        ex_scan.unsafe_store(n_rows, block_sums.unsafe_load(Int(n_blocks_in)))


def exclusive_scan(
    ctx: DeviceContext,
    mut ex_scan: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    n_rows: Int,
) raises:
    """`thrust::exclusive_scan(policy, vd, vd + n_rows, ex_scan)`, device-wide.

    `block_sums` must hold `scan_blocks_needed(n_rows) + 1` entries.
    """
    var n_blocks = scan_blocks_needed(n_rows)
    ctx.enqueue_function[exclusive_scan_pass1_kernel](
        ex_scan.unsafe_ptr(),
        vd.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[exclusive_scan_pass2_kernel](
        block_sums.unsafe_ptr(),
        Int32(n_blocks),
        grid_dim=(1, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[exclusive_scan_pass3_kernel](
        ex_scan.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_blocks),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )


# `adj_to_csr_kernel`'s block size (`adj_to_csr.cuh:35`).
comptime ADJ_TO_CSR_TPB = 512
# `const int chunk_size = 16;` (`adj_to_csr.cuh:75`), the width of their
# `raft::TxN_t<bool, chunk_size>` vector load.
comptime ADJ_CHUNK_SIZE = 16


def compact_adjacency_kernel(
    col_ind: MutPointer[Int32, MutAnyOrigin],
    adj: MutPointer[UInt8, MutAnyOrigin],
    ex_scan: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`adj_to_csr_kernel` (`raft/sparse/convert/detail/adj_to_csr.cuh:67`).

    One block per row, `ADJ_TO_CSR_TPB` threads, a shared per-row cursor
    incremented atomically, and the row read `ADJ_CHUNK_SIZE` bools at a time.

    **THE OUTPUT COLUMN ORDER IS NOT SORTED, AND THAT IS THEIRS.** Their own
    docstring: "High performance comes at the cost of non-deterministic
    output: the column indices are not guaranteed to be stored in order."
    Nothing downstream cares: `weak_cc` takes a min over each row's
    neighbours and a min over a set does not depend on visitation order.

    See deviation 34 for what is not literal.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    if row >= Int(n_rows_in):
        return
    var tid = Int(thread_idx.x)
    var row_base = Int(ex_scan.unsafe_load(row))
    var row_off = row * n_cols

    var row_count = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        row_count.unsafe_store(0, Int32(0))
    barrier()

    # Their chunked grid-stride loop, `adj_to_csr.cuh:94-101`. Reading
    # `ADJ_CHUNK_SIZE` bools per iteration is the point: the adjacency is
    # `batch_size x N` BYTES and this kernel reads all of it, so a
    # one-byte-per-iteration loop issues sixteen times the loads.
    var j = ADJ_CHUNK_SIZE * tid
    while j + ADJ_CHUNK_SIZE - 1 < n_cols:
        var chunk = adj.load[width=ADJ_CHUNK_SIZE, alignment=1](row_off + j)
        for k in range(ADJ_CHUNK_SIZE):
            if chunk[k] != 0:
                var pos = Atomic.fetch_add(
                    row_count.unsafe_offset(0), Int32(1)
                )
                col_ind.unsafe_store(row_base + Int(pos), Int32(j + k))
        j += ADJ_CHUNK_SIZE * ADJ_TO_CSR_TPB

    # Their remainder, `:104-108`. Theirs peels the head instead, because it
    # aligns the chunk loads to a 16-byte boundary; ours loads unaligned and
    # therefore only has a tail.
    var j1 = n_cols % ADJ_CHUNK_SIZE
    if tid < j1:
        var jt = n_cols - j1 + tid
        if adj.unsafe_load(row_off + jt) != 0:
            var pos2 = Atomic.fetch_add(row_count.unsafe_offset(0), Int32(1))
            col_ind.unsafe_store(row_base + Int(pos2), Int32(jt))


# ---------------------------------------------------------------------------
# DEVIATION BLOCK 34: THE CSR CURSOR IS A PLAIN ATOMIC, NOT A WARP-AGGREGATED
# ONE, AND ONE BLOCK OWNS A ROW
#
# THEIRS: `atomicIncWarp` (`raft/util/device_atomics.cuh:662`) --
#   `cg::coalesced_threads()`, one `atomicAdd` of the group size by lane 0,
#   then `g.shfl(warp_res, 0) + g.thread_rank()`. Their comment: "The use of
#   atomicIncWarp is a performance optimization. It can reduce the amount of
#   atomic memory traffic by a factor of 32."
# OURS: `Atomic.fetch_add` per hit.
# REASON: `cooperative_groups::coalesced_threads()` has no Mojo counterpart.
#   The set of currently-converged lanes is not something `std.gpu.primitives`
#   exposes, and a fixed-mask reduction is not the same thing -- the whole
#   point of `coalesced_threads` is that only the lanes that took the branch
#   participate. Their own comment prices this at up to 32x the atomic
#   traffic, and it is the largest remaining gap in this file.
#
# THEIRS: `dim3 grid(blocks_per_row, grid_rows)` where `blocks_per_row` comes
#   from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` -- when there are
#   more resident blocks than rows, SEVERAL blocks cooperate on one row
#   through that shared counter (`adj_to_csr.cuh:157-166`).
# OURS: one block per row, `grid_dim = (n_points, 1, 1)`.
# REASON: MAX exposes no occupancy calculator. It costs parallelism only when
#   the row count is below the device's resident-block count, which for
#   DBSCAN means a batch smaller than a few hundred rows; `compute_batch_size`
#   picks thousands.
#
# What this replaced was a `block.prefix_sum` over each 256-column window
# with two barriers per window. That produced ASCENDING column indices, which
# is not theirs and was recorded as a deviation "worth more than their last
# increment of throughput". It was not: at `n_cols = 200,000` it ran 782
# block-wide scans and 1,564 barriers PER ROW, against their zero barriers
# and one atomic per non-zero. Nothing reads the CSR in order.
# ---------------------------------------------------------------------------


def adj_graph_run(
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut ex_scan: DeviceBuffer[DType.int32],
    mut col_ind: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    n_points: Int,
    n_cols: Int,
) raises:
    """`AdjGraph::run` (`adjgraph/runner.cuh:27`), the `algo == 1` arm.

    Their `Algo::launcher` scans `vd` over `batch_size` entries and then
    compacts `adj`, in that order and with no mask on either.
    """
    exclusive_scan(ctx, ex_scan, vd, block_sums, n_points)
    ctx.enqueue_function[compact_adjacency_kernel](
        col_ind.unsafe_ptr(),
        adj.unsafe_ptr(),
        ex_scan.unsafe_ptr(),
        Int32(n_points),
        Int32(n_cols),
        grid_dim=(n_points, 1, 1),
        block_dim=(ADJ_TO_CSR_TPB, 1, 1),
    )
