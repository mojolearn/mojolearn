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
`mojo_only/ball_cover_check.mojo:207-213` already raises "row N lists column C
twice" -- so `index` alone is a TOTAL order and there is no tie class for a
distance half to break.

What the composite would cost is real. The fill kernel does not emit distances,
so stores would have to be added at five sites inside a body whose banner says
"Partial. Do not improve."; `nnz * 4` extra bytes would be written on the hot
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

WHY RANK-BY-COUNTING AND NOT A SEGMENTED RADIX
----------------------------------------------
`gbdt/gpu_util/kernel/segmented_sort.mojo::launch_segmented_radix_sort` is the
only one of the tree's two segmented sorts that takes RAGGED segments
(`core/segmented_sort.mojo` computes `base = seg * seg_size` and cannot express
a CSR), and it is bit-exact on every vendor. It is still the wrong SHAPE here:

  1. it puts the segment on `grid.y`, and CUDA caps `maxGridSize[1]` at 65,535.
     DBSCAN's default `batch_size` is `n_rows` (`dbscan/ported/dbscan/
     runner.mojo:302`), so `n_segments` is 200,000 on this lane's own scaling
     fixture. `gbdt/` never hit this because its segments are leaves, bounded
     by `2^depth`.
  2. `blocks_wide` comes from the GLOBAL `max_segment_size`, so one long row
     makes every one of `n_queries` segments launch that many blocks, most of
     them returning on their first line.

Rank-by-counting is ONE launch on `grid.x`, needs no scratch but the output,
and is the construction THIS LANE ALREADY CHOSE for its index build
(`ball_cover.mojo`'s DEVIATION 3: "Rank-by-counting needs no barrier, no
shared memory and no atomics, and it is deterministic").

COST, UNMEASURED
----------------
The shape is `sum over rows of |row|^2 / RBC_CANON_TPB` per-thread compares
spread over `n_queries` blocks, so it scales with the SQUARE of the mean degree
while the query kernel it follows scales with the first power. At a mean degree
in the tens it should disappear; at a mean degree in the high hundreds it will
not. THIS BANNER MUST BE REPLACED BY NUMBERS. If it binds, the fix is a second
tier: route rows above a threshold through `launch_segmented_radix_sort`, where
the long rows are FEW so `grid.y` and `blocks_wide` are both small. That tier
is NOT built and must not be assumed.

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

from mojo_only.numerics import PIN_CROSS_VENDOR


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

    The keys are unique within a row, so the map is a bijection onto [0, n) and
    every output slot is written exactly once. No initialisation, no atomics,
    no barrier, and no tie rule to get wrong. A duplicate column would break
    that invariant by writing one slot twice and leaving another unwritten,
    which is why the gate asserts uniqueness BEFORE it asserts order.

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
            if adj_ja_in.unsafe_load(start + q) < key:
                rank += 1
        adj_ja_out.unsafe_store(start + rank, key)
        p += RBC_CANON_TPB


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

    var scratch = ctx.enqueue_create_buffer[DType.int32](nnz)
    ctx.synchronize()
    ctx.enqueue_function[rbc_canonical_row_order_kernel](
        adj_ia.unsafe_ptr(),
        adj_ja.unsafe_ptr(),
        scratch.unsafe_ptr(),
        grid_dim=(n_queries, 1, 1),
        block_dim=(RBC_CANON_TPB, 1, 1),
    )
    # `adj_ja` is allocated to a CAPACITY, which the caller may size above
    # `nnz` (the max_k arm sizes it `n_queries * max_k`). Copying an
    # nnz-sized scratch into the whole buffer fails with "not enough data
    # in src", which is how this was found. Only the live prefix moves.
    ctx.enqueue_copy(
        dst_buf=adj_ja.create_sub_buffer[DType.int32](0, nnz),
        src_buf=scratch,
    )
    ctx.synchronize()
    # [[mojo-buffer-freed-at-last-use]]: `scratch` is the copy's SOURCE and its
    # last textual use is the enqueue, which is asynchronous. Held past the
    # synchronize or the copy reads freed memory.
    _ = scratch^
