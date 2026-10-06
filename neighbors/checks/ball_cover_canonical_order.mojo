# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 551. The ball cover's CSR, in one canonical intra-row order.

NO CUVS COUNTERPART. cuVS has no such pass because cuVS makes no cross-vendor
identity claim: `registers.cuh` emits a row in whatever order its 32-lane
ballot assigns, and that is the answer.

IDENTITY_PATHS row 61 is why this file exists. The row's MEMBERS are already a
pure function of the input bits (rows 19 and 24, plus DEVIATION 550's
`identical_sqrt` on the three pruning bounds at `ball_cover/registers.mojo:280,
422,547`). THE POSITIONS ARE NOT: the chunked backward walk bounds itself with
`limit = (r_size // RBC_LANES) * RBC_LANES` (`registers.mojo:284`, `:426`,
`:551`) and `RBC_LANES` is 32 on Apple and NVIDIA and 64 on CDNA, so a
different lane width is a different chunk boundary, a different `vote` ballot,
and a different `pop_count(mask & lid_mask)` slot.

PINNING THE LANE WIDTH IS NOT AVAILABLE. DEVIATION 515 records that a 32-bit
`vote` on a 64-lane wavefront did not merely merge two queries' ballots, it
aborted the AMD compile outright:

    LLVM ERROR: Cannot select: i32 = AMDGPUISD::SETCC <i1 CopyFromReg>, 0, setne

So the move is OUTPUT CANONICALIZATION, which is how row 11 closed k-NN's ties
(DEVIATIONS 500/501): make the answer arithmetic rather than a consequence of
the launch shape.

THE KEY IS THE COLUMN INDEX ALONE, AND THAT IS A DEPARTURE FROM WHAT ROW 61
PROPOSED. Row 61 names the 64-bit composite `(twiddle_in(distance) << 32) |
index`. The high half is INERT here: a CSR row's column indices are UNIQUE --
`checks/ball_cover_check.mojo:207-213` already raises "row N lists column C
twice" -- so `index` alone is a TOTAL order and there is no tie class for a
distance half to break.

What the composite would cost is real. The fill kernel does not emit distances,
so stores would have to be added at five sites inside a body marked partial;
`nnz * 4` extra bytes would be written on the hot
kernel in BOTH modes, because a `comptime` cannot remove a store from a kernel
FAST also runs, so FAST would pay for IDENTICAL's key; four entry points and
two callers would change signature; and the resulting 64-bit key is wider than
either segmented sort in this tree accepts. The distance stored is also
`eps_dist_sq`, the SQUARED distance, so the result would be ordered by d^2.

AND THE INDEX PASS COMPOSES FORWARD RATHER THAN BLOCKING THE COMPOSITE. A
later STABLE sort keyed on `twiddle_in(distance)` over an index-ascending row
yields EXACTLY row 61's composite order. When a `radius_neighbors(
sort_results=True)` surface lands it adds that stage AT THE LAYER THAT RETURNS
DISTANCES, and inherits this one as its tiebreak.

ASCENDING COLUMN INDEX IS ALSO THE STANDARD CANONICAL CSR, not a compromise:
`scipy.sparse` carries `has_sorted_indices` for exactly this, and it is what
makes `adj_ja` comparable across two vendors by a straight byte compare, which
is what the AMD leg needs.

BOUNDED WORK FOR DENSE ROWS
---------------------------
The original rank-by-counting pass performs sum(degree**2) comparisons.
Splitting CSR batches limits total edges but cannot shorten a dense row.
The opt-in MOJOLEARN_RBC_CANON_MERGE control uses stable bottom-up merges
for rows longer than one block; MOJOLEARN_RBC_CANON_DEGREE_BUCKETS instead
compacts rows by the exact merge-level count. Both are default off pending
NVIDIA/AMD qualification. Each
key binary-searches its rank in the opposite sorted run. The left run
uses lower_bound and the right uses upper_bound, so equal keys retain
multiplicity without output collisions. This is O(nnz * log(max_degree)^2)
integer comparisons and one nnz-sized scratch, not a change to the graph.

The default retains the original one-launch rank pass. The global-merge
control retains that rank pass only when the entire batch
fits one key per thread (maximum degree <= RBC_CANON_TPB). This boundary
comes from the block's work assignment, not a benchmark shape. Its tie
rule also orders by original position, although actual RBC rows are unique.
All dispatch is vendor independent. A separate max-degree pass determines
the number of merge levels; no data or degree estimate changes an edge.

PORTABILITY
-----------
No warp or wavefront width appears in this file and no floating-point
arithmetic. `RBC_CANON_TPB` is a fixed constant, not a device query, for the
same reason `core/segmented_sort.mojo` fixes `SORT_BLOCK`: the answer must not
move with the block count. The only operations are an integer compare and an
integer increment, both exact on every backend, so this pass is cross-vendor
bit-exact BY CONSTRUCTION rather than by measurement. That is a claim about
this file only; whether the CSR it is handed is identical across vendors is
what the AMD leg owes.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx
from std.atomic import Atomic
from std.sys.compile import is_defined
from core.device_zero import enqueue_fill

from checks.numerics import PIN_CROSS_VENDOR


#: Threads per block, one block per CSR row. Fixed, not device-derived: see
#: PORTABILITY. Matches `RBC_SCAN_TPB` in `ball_cover/scan.mojo`. A pure
#: scheduling knob, so the answer cannot move with it, but it is UNMEASURED
#: and a 64/128/256/512 sweep is owed alongside the cost banner.
comptime RBC_CANON_TPB = 256


def rbc_canonical_row_order_kernel(
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    adj_ja_in: MutPointer[Int32, MutAnyOrigin],
    adj_ja_out: MutPointer[Int32, MutAnyOrigin],
):
    """One block per row. Each element's destination is the number of elements
    of the SAME row that are strictly smaller.

    The rank uses (key, original position), so duplicate keys retain their
    multiplicity and every output slot is written exactly once. Actual RBC
    neighborhoods have unique columns; the tie rule also makes the helper
    safe for repeated keys. No initialisation, atomics or barrier are needed.

    OUT OF PLACE. `adj_ja_in` and `adj_ja_out` must be distinct: ranks are read
    from the input while the output is written, and there is no ordering
    between blocks or between threads of one block.
    """
    var row = Int(block_idx.x)
    var start = Int(adj_ia.unsafe_load(row))
    var n = Int(adj_ia.unsafe_load(row + 1)) - start
    if n <= 0:
        return

    var p = Int(thread_idx.x)
    while p < n:
        var key = adj_ja_in.unsafe_load(start + p)
        var rank = 0
        for q in range(n):
            var other = adj_ja_in.unsafe_load(start + q)
            if other < key or (other == key and q < p):
                rank += 1
        adj_ja_out.unsafe_store(start + rank, key)
        p += RBC_CANON_TPB


def _rbc_max_degree_kernel(
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    n_queries: Int32,
    maximum: MutPointer[Int32, MutAnyOrigin],
):
    var row = Int(block_idx.x) * RBC_CANON_TPB + Int(thread_idx.x)
    if row < Int(n_queries):
        var degree = adj_ia[row + 1] - adj_ia[row]
        _ = Atomic[DType.int32].max(maximum, degree)


def _rbc_merge_row(
    row: Int,
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    width_in: Int32,
):
    """Merge adjacent sorted runs; stable binary-search ranks are disjoint."""
    var start = Int(adj_ia[row])
    var n = Int(adj_ia[row + 1]) - start
    var width = Int(width_in)
    var p = Int(thread_idx.x)
    while p < n:
        var pair = (p // (2 * width)) * (2 * width)
        var middle = min(pair + width, n)
        var end = min(pair + 2 * width, n)
        var left = p < middle
        var low = middle if left else pair
        var high = end if left else middle
        var other_start = low
        var own_start = pair if left else middle
        var key = src[start + p]
        while low < high:
            var mid = low + (high - low) // 2
            var other = src[start + mid]
            if other < key or (not left and other == key):
                low = mid + 1
            else:
                high = mid
        var destination = pair + (p - own_start) + (low - other_start)
        dst[start + destination] = key
        p += RBC_CANON_TPB



def _rbc_merge_rows_kernel(adj_ia: MutPointer[Int32, MutAnyOrigin], src: MutPointer[Int32, MutAnyOrigin], dst: MutPointer[Int32, MutAnyOrigin], width_in: Int32):
    _rbc_merge_row(Int(block_idx.x), adj_ia, src, dst, width_in)


def _degree_bucket(n: Int) -> Int:
    # ceil(log2(degree)): each row executes exactly the merge levels it
    # needs. Bucket boundaries follow work complexity, never board sizes.
    var value = max(n - 1, 0)
    var bucket = 0
    while value > 0:
        value >>= 1
        bucket += 1
    return bucket


def _rbc_bucket_count(ia: MutPointer[Int32, MutAnyOrigin], n: Int32, counts: MutPointer[Int32, MutAnyOrigin]):
    var row = Int(block_idx.x) * RBC_CANON_TPB + Int(thread_idx.x)
    if row < Int(n):
        var bucket = _degree_bucket(Int(ia[row + 1] - ia[row]))
        _ = Atomic[DType.int32].fetch_add(counts + bucket, Int32(1))


def _rbc_bucket_offsets(counts: MutPointer[Int32, MutAnyOrigin], offsets: MutPointer[Int32, MutAnyOrigin]):
    # Exactly 32 integer bucket counts: bounded control-plane prefix.
    var run = Int32(0)
    for b in range(32):
        offsets[b] = run
        run += counts[b]
        counts[b] = Int32(0)
    offsets[32] = run


def _rbc_bucket_scatter(ia: MutPointer[Int32, MutAnyOrigin], n: Int32, counts: MutPointer[Int32, MutAnyOrigin], offsets: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin]):
    var row = Int(block_idx.x) * RBC_CANON_TPB + Int(thread_idx.x)
    if row < Int(n):
        var bucket = _degree_bucket(Int(ia[row + 1] - ia[row]))
        var slot = Atomic[DType.int32].fetch_add(counts + bucket, Int32(1))
        rows[Int(offsets[bucket] + slot)] = Int32(row)


def _rbc_bucket_merge(ia: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], offset: Int32, src: MutPointer[Int32, MutAnyOrigin], dst: MutPointer[Int32, MutAnyOrigin], width: Int32):
    var row = Int(rows[Int(offset) + Int(block_idx.x)])
    _rbc_merge_row(row, ia, src, dst, width)


def _rbc_bucket_copy(ia: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], offset: Int32, src: MutPointer[Int32, MutAnyOrigin], dst: MutPointer[Int32, MutAnyOrigin]):
    var row = Int(rows[Int(offset) + Int(block_idx.x)])
    var p = Int(ia[row]) + Int(thread_idx.x)
    while p < Int(ia[row + 1]):
        dst[p] = src[p]
        p += RBC_CANON_TPB


def rbc_canonicalize_degree_buckets(ctx: DeviceContext, mut ia: DeviceBuffer[DType.int32], mut ja: DeviceBuffer[DType.int32], rows_count: Int, nnz: Int) raises:
    """Compact degree tasks; nnz scratch + one row descriptor per row.
    Atomic task order cannot affect output: tasks own disjoint CSR rows and
    sorting comparisons/duplicate tie rules are the existing stable merge.
    One 33-word readback replaces maximum-degree readback. No row data
    arithmetic runs on the host; counts only determine kernel launch grids.
    All asynchronous storage lives until the final completion."""
    if rows_count <= 0 or nnz <= 0:
        return
    var counts = ctx.enqueue_create_buffer[DType.int32](32)
    var offsets = ctx.enqueue_create_buffer[DType.int32](33)
    var tasks = ctx.enqueue_create_buffer[DType.int32](rows_count)
    var scratch = ctx.enqueue_create_buffer[DType.int32](nnz)
    enqueue_fill(ctx, counts, Int32(0))
    var blocks = (rows_count + RBC_CANON_TPB - 1) // RBC_CANON_TPB
    ctx.enqueue_function[_rbc_bucket_count](ia.unsafe_ptr(), Int32(rows_count), counts.unsafe_ptr(), grid_dim=(blocks,1,1), block_dim=(RBC_CANON_TPB,1,1))
    ctx.enqueue_function[_rbc_bucket_offsets](counts.unsafe_ptr(), offsets.unsafe_ptr(), grid_dim=(1,1,1), block_dim=(1,1,1))
    ctx.enqueue_function[_rbc_bucket_scatter](ia.unsafe_ptr(), Int32(rows_count), counts.unsafe_ptr(), offsets.unsafe_ptr(), tasks.unsafe_ptr(), grid_dim=(blocks,1,1), block_dim=(RBC_CANON_TPB,1,1))
    var ho = ctx.enqueue_create_host_buffer[DType.int32](33)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=offsets)
    ctx.synchronize()
    for bucket in range(1,32):
        var count = Int(ho[bucket+1] - ho[bucket])
        if count <= 0:
            continue
        var width = 1
        for level in range(bucket):
            if level % 2 == 0:
                ctx.enqueue_function[_rbc_bucket_merge](ia.unsafe_ptr(), tasks.unsafe_ptr(), ho[bucket], ja.unsafe_ptr(), scratch.unsafe_ptr(), Int32(width), grid_dim=(count,1,1), block_dim=(RBC_CANON_TPB,1,1))
            else:
                ctx.enqueue_function[_rbc_bucket_merge](ia.unsafe_ptr(), tasks.unsafe_ptr(), ho[bucket], scratch.unsafe_ptr(), ja.unsafe_ptr(), Int32(width), grid_dim=(count,1,1), block_dim=(RBC_CANON_TPB,1,1))
            width *= 2
        if bucket % 2 != 0:
            ctx.enqueue_function[_rbc_bucket_copy](ia.unsafe_ptr(), tasks.unsafe_ptr(), ho[bucket], scratch.unsafe_ptr(), ja.unsafe_ptr(), grid_dim=(count,1,1), block_dim=(RBC_CANON_TPB,1,1))
    ctx.synchronize()
    _ = counts^; _ = offsets^; _ = tasks^; _ = scratch^

def rbc_canonicalize_row_order(
    ctx: DeviceContext,
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    n_queries: Int,
    nnz: Int,
) raises:
    """Rewrite every CSR row of `adj_ja` into ascending column order.

    `adj_ia` is NOT touched. `ball_cover/scan.mojo`'s exclusive scan is an
    Int32 block scan, so the row BOUNDARIES are already cross-vendor identical
    and only the contents within a boundary move.

    NO-OP UNDER FAST AND DETERMINISTIC. `PIN_CROSS_VENDOR` is true only under
    IDENTICAL, which is the correct tier: the emission order is already a pure
    function of the build ON ONE BOX, so this is not a determinism pin, it is a
    CROSS-VENDOR pin, and a DETERMINISTIC user should not pay for it. The guard
    is `comptime`, so FAST allocates nothing and launches nothing.
    """

    @parameter
    if not PIN_CROSS_VENDOR:
        return
    if n_queries <= 0 or nnz <= 0:
        return

    # I13 new candidate remains default off. Qualification is pending: native
    # compilation is not four-column identity or NVIDIA+AMD full-operation speed.
    comptime if is_defined["MOJOLEARN_RBC_CANON_DEGREE_BUCKETS"]():
        rbc_canonicalize_degree_buckets(ctx, adj_ia, adj_ja, n_queries, nnz)
        return

    comptime if not is_defined["MOJOLEARN_RBC_CANON_MERGE"]():
        var ranked = ctx.enqueue_create_buffer[DType.int32](nnz)
        ctx.enqueue_function[rbc_canonical_row_order_kernel](
            adj_ia.unsafe_ptr(), adj_ja.unsafe_ptr(), ranked.unsafe_ptr(),
            grid_dim=(n_queries,1,1), block_dim=(RBC_CANON_TPB,1,1),
        )
        ctx.enqueue_copy(dst_buf=adj_ja.create_sub_buffer[DType.int32](0,nnz),src_buf=ranked)
        ctx.synchronize()
        _ = ranked^
        return

    var maximum = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, maximum, Int32(0))
    ctx.enqueue_function[_rbc_max_degree_kernel](
        adj_ia.unsafe_ptr(), Int32(n_queries), maximum.unsafe_ptr(),
        grid_dim=((n_queries + RBC_CANON_TPB - 1) // RBC_CANON_TPB, 1, 1),
        block_dim=(RBC_CANON_TPB, 1, 1),
    )
    var host_max = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=host_max.unsafe_ptr(), src_buf=maximum)
    ctx.synchronize()
    _ = maximum^
    var max_degree = Int(host_max.unsafe_ptr()[0])
    if max_degree <= 1:
        return
    var scratch = ctx.enqueue_create_buffer[DType.int32](nnz)
    if max_degree <= RBC_CANON_TPB:
        ctx.enqueue_function[rbc_canonical_row_order_kernel](
            adj_ia.unsafe_ptr(), adj_ja.unsafe_ptr(), scratch.unsafe_ptr(),
            grid_dim=(n_queries, 1, 1), block_dim=(RBC_CANON_TPB, 1, 1),
        )
        ctx.enqueue_copy(
            dst_buf=adj_ja.create_sub_buffer[DType.int32](0, nnz),
            src_buf=scratch,
        )
    else:
        var width = 1
        var in_original = True
        while width < max_degree:
            if in_original:
                ctx.enqueue_function[_rbc_merge_rows_kernel](
                    adj_ia.unsafe_ptr(), adj_ja.unsafe_ptr(), scratch.unsafe_ptr(),
                    Int32(width), grid_dim=(n_queries, 1, 1),
                    block_dim=(RBC_CANON_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[_rbc_merge_rows_kernel](
                    adj_ia.unsafe_ptr(), scratch.unsafe_ptr(), adj_ja.unsafe_ptr(),
                    Int32(width), grid_dim=(n_queries, 1, 1),
                    block_dim=(RBC_CANON_TPB, 1, 1),
                )
            in_original = not in_original
            width *= 2
        if not in_original:
            ctx.enqueue_copy(
                dst_buf=adj_ja.create_sub_buffer[DType.int32](0, nnz),
                src_buf=scratch,
            )
    ctx.synchronize()
    # Retain every async source until completion, including the max-degree
    # scalar. No CSR capacity beyond its live prefix is read or overwritten.
    _ = scratch^
