"""The eps-neighborhood query kernels of the random ball cover.

PORT OF `cuvs/src/neighbors/ball_cover/registers.cuh` at cuVS `94c2819`:
`block_rbc_kernel_eps_csr_pass` (`:576`), `block_rbc_kernel_eps_dense`
(`:458`), `block_rbc_kernel_eps_max_k` (`:859`),
`block_rbc_kernel_eps_max_k_copy` (`:983`), and both `rbc_eps_pass` host
drivers (`:1271`, `:1314`). Partial. Do not improve.

WHY THIS IS THE FILE THAT MATTERS
---------------------------------
`bench/results/FIRST_RUN_2026-08-19.md` measured our DBSCAN at 37x slower
than scikit-learn at 200,000 points, and exactly quadratic: 50x the points
cost 2400x the time while scikit-learn's kd-tree cost 93x. No kernel tuning
closes a gap between `O(n^2)` and `O(n log n)`; only an index does. This is
the index cuML itself reaches for — `cuml/cpp/src/dbscan/vertexdeg/algo.cuh`
calls `cuvs::neighbors::ball_cover::eps_nn` whenever `data.rbc_index` is
non-null, and `runner.cuh:234-242` builds it whenever `sparse_rbc_mode` is on.

THE PRUNE IS ARITHMETIC, WHICH IS WHY IT SUITS A GPU
----------------------------------------------------
Two triangle-inequality tests, both on values already in registers:

1. Skip a whole landmark `r` when `d(q, r) > eps + radius(r)`. Sound because
   every point `y` in `r`'s ball has `d(r, y) <= radius(r)`, so
   `d(q, y) >= d(q, r) - radius(r) > eps`.
2. Inside a landmark, walk its points BACKWARD (they are sorted ascending by
   `d(r, y)`) and stop the moment `d(q, r) - d(r, y) > eps`, because every
   remaining point has a smaller `d(r, y)` and therefore a larger bound.

There is no pointer chasing and no graph walk, which is the difference
between this and HNSW that `ROADMAP.md` rules out permanently.

**THE METHOD IS EXACT.** Their own header calls it "a faster exact knn in
metric spaces" (`cuvs/include/cuvs/neighbors/ball_cover.hpp:191`). Neither
test above can discard a point within `eps`, for any landmark set. That
matters more here than anywhere else in this repository: a neighborhood that
is one point short changes which points are core, and DBSCAN's whole
partition changes with it.

DEVIATION 1: THE BALLOT IS A `vote`, THE BIT WALK IS A `ctz`
-------------------------------------------------------------
Theirs is `raft::ballot` then `__brev` then `__clz`, at `:629-636`. The
`__brev` exists only so the loop can use `__clz` instead of `__ffs`; it is a
CUDA instruction-selection trick and carries no meaning. Ours is
`std.gpu.primitives.warp.vote` (probed working on this M4, see the build
evidence in the lane file) then `std.bit.count_trailing_zeros`, then
`mask &= mask - 1` in place of their `mask &= (0x7fffffff >> k_offset)`.
Both walk the set landmarks in ASCENDING index order, so the order columns
land in `adj_ja` is theirs as well as ours.

`__popc(mask & lid_mask)` and `__popc(mask)` are `std.bit.pop_count` on the
same `vote` mask, and `raft::warpReduce` is
`std.gpu.primitives.warp.sum`.

DEVIATION 2: 32 THREADS PER BLOCK, ONE QUERY PER BLOCK
-------------------------------------------------------
Theirs launches `tpb = 64` with two warps per block and
`ceildiv(n_query_rows, 2)` blocks (`:1332-1335`). Ours launches 32 threads and
`n_query_rows` blocks. Their `query_id >= n_queries` early-out is kept anyway;
it is dead at this launch shape and it costs one comparison.

The kernel is written in THEIR shape — `RBC_QPB` is their `num_warps` and the
query id is their `blockIdx.x * num_warps + threadIdx.x / WarpSize` (`:600`)
— so the only thing that differs is the value of `RBC_TPB`, and it is 32 for
two reasons, one of them measured.

**Correctness.** `vote` returns a mask over the WHOLE warp, so packing two
32-lane query groups into one 64-lane wavefront — which is what AMD would do
with `tpb = 64` — would merge two queries' ballots and silently corrupt both.
A vendor that wants their 64 must set `RBC_QPB` from its own lane width, not
from a constant.

**Cost.** Measured on an M4, 2026-08-19, one window, three repeats, min, on
the DBSCAN scaling fixture (d = 8, eps = 0.30), count-pass milliseconds:

    RBC_TPB     n=16,000   n=50,000   n=100,000   n=200,000
    32 (ours)       4.25      18.53       49.75      129.08
    64 (theirs)     6.53      20.08       49.60      130.85
    128             6.35      20.21       49.87      142.10
    256             5.40      20.22       51.23      135.56

Their 64 is not faster than our 32 anywhere on this device and is 1.54x
slower at 16,000. **This kernel is not block-shape bound**, so the deviation
costs nothing here and the portability argument decides it unopposed.

WHY `RBC_TPB` IS TYPED HERE AND NOT READ FROM THE KERNEL MATRIX. cuVS types
`int tpb = 64;` at the launch site, `registers.cuh:1330`, in this same file
as the kernels -- their block-size source IS a launcher-local constant, and
a `comptime` in the file that holds our `rbc_eps_pass_*` launchers mirrors
that file for file. The matrix row `K_LIB_BALL_COVER_EPS`
(`mojo_only/kernel_matrix.mojo:489`) is therefore deliberately UNWIRED: it
resolves to the table's fall-through 128 today, which the sweep above
measures as the WORST of the four shapes at 200,000 (142.10 ms against
129.08), so wiring it as it stands is a priced ~10% regression, and wiring
it at 32 would replace their structure with a table they do not have.
Re-derived 2026-08-19 with the two-loop max_k dispatch landed: that change
adds no launch site and moves no shape, so nothing about this conclusion
moved. A vendor measurement that wants a different value lands in that row
first, with this banner's sweep as the bar to clear.

NOT PORTED
----------
`block_rbc_kernel_eps_csr_pass_xd` (`:714`), which is this kernel with the
query row copied into a `local_x_ptr[MAX_COL_Q]` register array and used when
`index.n == 2 || index.n == 3` (`:1331`). It computes the same numbers; the
register array needs the dimension at compile time. Recorded in the lane file
as unported with that reason.
"""

from std.bit import count_trailing_zeros, pop_count
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import lane_id, shuffle_idx, vote
from std.gpu.primitives.warp import sum as warp_sum
from mojo_only.numerics import identical_sqrt  # DEVIATION 550
from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.kernel_matrix import TARGET_COLUMN, lib_lane_width_for

from neighbors.ported.neighbors.ball_cover.common import (
    RBC_FLT_MAX,
    eps_dist_sq,
)
from neighbors.ported.neighbors.ball_cover.scan import (
    RBC_SCAN_TPB,
    rbc_clamp_kernel,
    rbc_exclusive_scan_kernel,
    rbc_max_reduce_kernel,
)


#: The lane group one query is walked by. `vote`, `shuffle_idx`, `warp_sum`
#: and `pop_count(mask & lid_mask)` all range over exactly this many lanes,
#: so it is the hardware warp and nothing else.
#:
#: DEVIATION 515 (2026-08-23): READ FROM THE COLUMN, was the literal 32.
#: This is the constant the DEVIATION 2 banner above already told a future
#: vendor to change -- "A vendor that wants their 64 must set `RBC_QPB` from
#: its own lane width, not from a constant" -- and an MI300X made it
#: mandatory rather than tidy. On a 64-wide wavefront the old code did not
#: merely compute a merged ballot, IT DID NOT COMPILE:
#:
#:   LLVM ERROR: Cannot select: i32 = AMDGPUISD::SETCC <i1 CopyFromReg>, 0,
#:   setne   In function: neighbors_ported_neighbors_ba..._658cdd32df991420
#:
#: because `vote[DType.uint32]` (what stood here) asks for a 32-bit
#: ballot of a 64-lane
#: wavefront and there is no such instruction. Measured on RunPod MI300X,
#: ROCm 6.4.1, Mojo 1.0.0 (ed45d567), 2026-08-23: it aborted the whole
#: compile, so `cluster/` and `dbscan/` could not be built on AMD at all
#: even though neither of them calls this kernel -- one module, one codegen.
comptime RBC_LANES = lib_lane_width_for[TARGET_COLUMN]()

#: The BALLOT'S WIDTH, and it is a correctness type rather than a tuning
#: one: `vote` returns one bit per lane of the wavefront, so a 64-lane
#: column needs 64 bits to hold it. `count_trailing_zeros`, `pop_count` and
#: the `mask &= mask - 1` walk are all width-generic once the type is.
comptime RBC_MASK_DT = DType.uint64 if RBC_LANES == 64 else DType.uint32

#: Threads per block. Theirs is 64, two queries per block
#: (`registers.cuh:1332-1335`). See DEVIATION 2 and the measurement in it.
#:
#: PINNED TO ONE QUERY PER WAVEFRONT, which is what makes widening SAFE.
#: The DEVIATION 2 banner's correctness argument is that `vote` returns a
#: mask over the WHOLE warp, so two 32-lane query groups sharing one 64-lane
#: wavefront would merge two queries' ballots and silently corrupt both.
#: Setting this to `RBC_LANES` keeps `RBC_QPB == 1` on every column, so that
#: hazard cannot arise on any of them. The M4 measurement in the banner is
#: unaffected: on a 32-lane column this is still 32.
comptime RBC_TPB = RBC_LANES

#: Queries per block, derived. `RBC_TPB // RBC_LANES` is their
#: `num_warps = tpb / WarpSize` (`registers.cuh:1330`) and the query id is
#: their `blockIdx.x * num_warps + threadIdx.x / WarpSize` (`:600`), so the
#: kernel is written in their shape at every value of this. It is 1 here for
#: the reason DEVIATION 2 gives.
comptime RBC_QPB = RBC_TPB // RBC_LANES

#: `block_rbc_kernel_eps_max_k_copy<value_idx, 32><<<n_query_rows, 32>>>`,
#: `registers.cuh:1476-1478`. One block per ROW and 32 threads, theirs
#: exactly; it is a compaction, not a warp-cooperative walk, so it does not
#: move with `RBC_TPB`.
comptime RBC_COPY_TPB = 32


def block_rbc_kernel_eps_csr_pass(
    x_reordered: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_queries_in: Int32,
    n_cols_in: Int32,
    r: MutPointer[Float32, MutAnyOrigin],
    eps_in: Float32,
    n_landmarks_in: Int32,
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    r_radius: MutPointer[Float32, MutAnyOrigin],
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    adj_ja: MutPointer[Int32, MutAnyOrigin],
    write_pass_in: Int32,
):
    """`block_rbc_kernel_eps_csr_pass`, `registers.cuh:576-708`.

    Two passes over the same code. `write_pass = 0` counts each query's
    neighbors and writes the COUNT into `adj_ia[query_id]` — their host
    passes `vd_ptr` in that argument (`:1349`), scans it, and only then does
    `adj_ia` hold offsets. `write_pass = 1` re-walks and emits the column
    indices at those offsets.

    Recomputing every distance in pass two is theirs and is deliberate: the
    alternative is materializing an `n_queries x m` boolean, which is the
    dense kernel below and is the memory blow-up the CSR path exists to
    avoid.
    """
    var n_queries = Int(n_queries_in)
    var n_cols = Int(n_cols_in)
    var n_landmarks = Int(n_landmarks_in)
    var write_pass = write_pass_in != Int32(0)
    var eps = eps_in

    var lid = Int(lane_id())
    var lid_mask = (Scalar[RBC_MASK_DT](1) << Scalar[RBC_MASK_DT](lid)) - Scalar[RBC_MASK_DT](1)

    # `raft::shfl(blockIdx.x * num_warps + threadIdx.x / WarpSize, 0)`, `:600`.
    # Their comment: "this should help the compiler to prevent branches".
    var query_id = Int(
        shuffle_idx(
            Int32(
                Int(block_idx.x) * RBC_QPB + Int(thread_idx.x) // RBC_LANES
            ),
            UInt32(0),
        )
    )
    if query_id >= n_queries:
        return

    var column_index_offset = 0
    var ja_pos = 0

    if write_pass:
        var offset = Int(adj_ia.unsafe_load(query_id))
        # we have no neighbors to fill for this query
        if offset == Int(adj_ia.unsafe_load(query_id + 1)):
            return
        ja_pos = offset

    var x_base = n_cols * query_id

    # we omit the sqrt() in the inner distance compute
    var eps2 = eps * eps

    for cur_k0 in range(0, n_landmarks, RBC_LANES):
        # Pre-compute landmark_dist & triangularization checks for 32 lanes.
        var lane_k = cur_k0 + lid
        var lane_r_dist_sq = RBC_FLT_MAX
        var lane_check = False
        if lane_k < n_landmarks:
            lane_r_dist_sq = eps_dist_sq(
                x, x_base, r, lane_k * n_cols, n_cols
            )
            var bound = eps + r_radius.unsafe_load(lane_k)
            lane_check = lane_r_dist_sq <= bound * bound

        var lane_mask = vote[RBC_MASK_DT](lane_check)
        if lane_mask == Scalar[RBC_MASK_DT](0):
            continue

        while lane_mask != Scalar[RBC_MASK_DT](0):
            # look for next k_offset
            var k_offset = Int(count_trailing_zeros(lane_mask))
            # update lane_mask for next iteration - erase bits up to k_offset
            lane_mask &= lane_mask - Scalar[RBC_MASK_DT](1)

            var cur_k = cur_k0 + k_offset

            # The whole warp should iterate through the elements in this R
            var r_start = Int(r_indptr.unsafe_load(cur_k))
            var r_size = Int(r_indptr.unsafe_load(cur_k + 1)) - r_start

            # we have precomputed the query<->landmark distance
            var cur_r_dist = identical_sqrt(  # DEVIATION 550: the bound's sqrt, pinned under IDENTICAL (FAST = stdlib, verbatim)
                shuffle_idx(lane_r_dist_sq, UInt32(k_offset))
            )

            var limit = (r_size // RBC_LANES) * RBC_LANES
            var i = limit + lid

            # R_1nn_dists are sorted ascendingly for each landmark.
            # Iterating backwards, after pruning the first point w.r.t. the
            # triangle inequality all subsequent points can be pruned too.
            var min_warp_dist = cur_r_dist
            if limit < r_size:
                min_warp_dist = r_1nn_dists.unsafe_load(r_start + limit)

            var dist = RBC_FLT_MAX
            if i < r_size:
                dist = eps_dist_sq(
                    x, x_base, x_reordered, (r_start + i) * n_cols, n_cols
                )
            var in_range = dist <= eps2
            if write_pass:
                var mask = vote[RBC_MASK_DT](in_range)
                if in_range:
                    var index = r_1nn_cols.unsafe_load(r_start + i)
                    var row_pos = Int(pop_count(mask & lid_mask))
                    adj_ja.unsafe_store(ja_pos + row_pos, index)
                ja_pos += Int(pop_count(mask))
            else:
                if in_range:
                    column_index_offset += 1

            # abort in case subsequent points cannot possibly be in reach
            if cur_r_dist - min_warp_dist > eps:
                i = 0

            var i0 = Int(shuffle_idx(Int32(i), UInt32(0)))

            while i0 >= RBC_LANES:
                i0 -= RBC_LANES
                var min_warp_dist2 = r_1nn_dists.unsafe_load(r_start + i0)
                var dist2 = eps_dist_sq(
                    x,
                    x_base,
                    x_reordered,
                    (r_start + i0 + lid) * n_cols,
                    n_cols,
                )
                var in_range2 = dist2 <= eps2
                if write_pass:
                    var mask2 = vote[RBC_MASK_DT](in_range2)
                    if in_range2:
                        var index2 = r_1nn_cols.unsafe_load(
                            r_start + i0 + lid
                        )
                        var row_pos2 = Int(pop_count(mask2 & lid_mask))
                        adj_ja.unsafe_store(ja_pos + row_pos2, index2)
                    ja_pos += Int(pop_count(mask2))
                else:
                    if in_range2:
                        column_index_offset += 1
                if cur_r_dist - min_warp_dist2 > eps:
                    i0 = 0

    if not write_pass:
        var row_sum = warp_sum(Int32(column_index_offset))
        if lid == 0:
            adj_ia.unsafe_store(query_id, row_sum)


def block_rbc_kernel_eps_dense(
    x_reordered: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_queries_in: Int32,
    n_cols_in: Int32,
    r: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    eps_in: Float32,
    n_landmarks_in: Int32,
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    r_radius: MutPointer[Float32, MutAnyOrigin],
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
):
    """`block_rbc_kernel_eps_dense`, `registers.cuh:458-570`.

    Same walk, writing an `n_queries x m` boolean instead of a CSR. This is
    the shape `cuml/cpp/src/dbscan/vertexdeg/algo.cuh` uses when there is NO
    ball-cover index, so it is what our own `dbscan/vertexdeg` already
    produces, and it is the cheapest way to check the CSR path against
    something that shares no code with it.

    `adj` is `UInt8` and not `Bool` for the same reason the rest of this
    repository uses `UInt8`: it is the byte the DBSCAN section already reads.
    """
    var n_queries = Int(n_queries_in)
    var n_cols = Int(n_cols_in)
    var n_landmarks = Int(n_landmarks_in)
    var m = Int(m_in)
    var eps = eps_in

    var lid = Int(lane_id())

    var query_id = Int(
        shuffle_idx(
            Int32(
                Int(block_idx.x) * RBC_QPB + Int(thread_idx.x) // RBC_LANES
            ),
            UInt32(0),
        )
    )
    if query_id >= n_queries:
        return

    var column_count = 0
    var x_base = n_cols * query_id
    var adj_base = query_id * m
    var eps2 = eps * eps

    for cur_k0 in range(0, n_landmarks, RBC_LANES):
        var lane_k = cur_k0 + lid
        var lane_r_dist_sq = RBC_FLT_MAX
        var lane_check = False
        if lane_k < n_landmarks:
            lane_r_dist_sq = eps_dist_sq(
                x, x_base, r, lane_k * n_cols, n_cols
            )
            var bound = eps + r_radius.unsafe_load(lane_k)
            lane_check = lane_r_dist_sq <= bound * bound

        var lane_mask = vote[RBC_MASK_DT](lane_check)
        if lane_mask == Scalar[RBC_MASK_DT](0):
            continue

        while lane_mask != Scalar[RBC_MASK_DT](0):
            var k_offset = Int(count_trailing_zeros(lane_mask))
            lane_mask &= lane_mask - Scalar[RBC_MASK_DT](1)

            var cur_k = cur_k0 + k_offset
            var r_start = Int(r_indptr.unsafe_load(cur_k))
            var r_size = Int(r_indptr.unsafe_load(cur_k + 1)) - r_start
            var cur_r_dist = identical_sqrt(  # DEVIATION 550: the bound's sqrt, pinned under IDENTICAL (FAST = stdlib, verbatim)
                shuffle_idx(lane_r_dist_sq, UInt32(k_offset))
            )

            var limit = (r_size // RBC_LANES) * RBC_LANES
            var i = limit + lid

            var min_warp_dist = cur_r_dist
            if limit < r_size:
                min_warp_dist = r_1nn_dists.unsafe_load(r_start + limit)

            var dist = RBC_FLT_MAX
            if i < r_size:
                dist = eps_dist_sq(
                    x, x_base, x_reordered, (r_start + i) * n_cols, n_cols
                )
            if dist <= eps2:
                var index = Int(r_1nn_cols.unsafe_load(r_start + i))
                column_count += 1
                adj.unsafe_store(adj_base + index, UInt8(1))

            if cur_r_dist - min_warp_dist > eps:
                i = 0

            var i0 = Int(shuffle_idx(Int32(i), UInt32(0)))

            while i0 >= RBC_LANES:
                i0 -= RBC_LANES
                var min_warp_dist2 = r_1nn_dists.unsafe_load(r_start + i0)
                var dist2 = eps_dist_sq(
                    x,
                    x_base,
                    x_reordered,
                    (r_start + i0 + lid) * n_cols,
                    n_cols,
                )
                if dist2 <= eps2:
                    var index2 = Int(
                        r_1nn_cols.unsafe_load(r_start + i0 + lid)
                    )
                    column_count += 1
                    adj.unsafe_store(adj_base + index2, UInt8(1))
                if cur_r_dist - min_warp_dist2 > eps:
                    i0 = 0

    var row_sum = warp_sum(Int32(column_count))
    if lid == 0:
        vd.unsafe_store(query_id, row_sum)


def block_rbc_kernel_eps_max_k(
    x_reordered: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_queries_in: Int32,
    n_cols_in: Int32,
    r: MutPointer[Float32, MutAnyOrigin],
    eps_in: Float32,
    n_landmarks_in: Int32,
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    r_radius: MutPointer[Float32, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    max_k_in: Int32,
    tmp: MutPointer[Int32, MutAnyOrigin],
):
    """`block_rbc_kernel_eps_max_k`, `registers.cuh:859-980`.

    ONE pass instead of two, at the cost of an `n_queries x max_k` scratch
    buffer. `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:119-135` picks this form
    whenever the caller's `max_k` is smaller than the spare room in `ja`,
    and asserts afterwards that the returned `max_k` did not grow — that is,
    it is a bet that no row is longer than `max_k`, checked after the fact.

    The bet losing is not a wrong answer, it is a truncation: `:944-950` keeps
    counting past `max_k` so `vd` stays TRUE, and only the writes stop. The
    host then learns the real maximum and the caller re-runs.
    """
    var n_queries = Int(n_queries_in)
    var n_cols = Int(n_cols_in)
    var n_landmarks = Int(n_landmarks_in)
    var max_k = Int(max_k_in)
    var eps = eps_in

    var lid = Int(lane_id())
    var lid_mask = (Scalar[RBC_MASK_DT](1) << Scalar[RBC_MASK_DT](lid)) - Scalar[RBC_MASK_DT](1)

    var query_id = Int(
        shuffle_idx(
            Int32(
                Int(block_idx.x) * RBC_QPB + Int(thread_idx.x) // RBC_LANES
            ),
            UInt32(0),
        )
    )
    if query_id >= n_queries:
        return

    var column_count = 0
    var x_base = n_cols * query_id
    var tmp_base = query_id * max_k
    var eps2 = eps * eps

    for cur_k0 in range(0, n_landmarks, RBC_LANES):
        var lane_k = cur_k0 + lid
        var lane_r_dist_sq = RBC_FLT_MAX
        var lane_check = False
        if lane_k < n_landmarks:
            lane_r_dist_sq = eps_dist_sq(
                x, x_base, r, lane_k * n_cols, n_cols
            )
            var bound = eps + r_radius.unsafe_load(lane_k)
            lane_check = lane_r_dist_sq <= bound * bound

        var lane_mask = vote[RBC_MASK_DT](lane_check)
        if lane_mask == Scalar[RBC_MASK_DT](0):
            continue

        while lane_mask != Scalar[RBC_MASK_DT](0):
            var k_offset = Int(count_trailing_zeros(lane_mask))
            lane_mask &= lane_mask - Scalar[RBC_MASK_DT](1)

            var cur_k = cur_k0 + k_offset
            var r_start = Int(r_indptr.unsafe_load(cur_k))
            var r_size = Int(r_indptr.unsafe_load(cur_k + 1)) - r_start
            var cur_r_dist = identical_sqrt(  # DEVIATION 550: the bound's sqrt, pinned under IDENTICAL (FAST = stdlib, verbatim)
                shuffle_idx(lane_r_dist_sq, UInt32(k_offset))
            )

            var limit = (r_size // RBC_LANES) * RBC_LANES
            var i = limit + lid

            var min_warp_dist = cur_r_dist
            if limit < r_size:
                min_warp_dist = r_1nn_dists.unsafe_load(r_start + limit)

            var dist = RBC_FLT_MAX
            if i < r_size:
                dist = eps_dist_sq(
                    x, x_base, x_reordered, (r_start + i) * n_cols, n_cols
                )
            var in_range = dist <= eps2
            var mask = vote[RBC_MASK_DT](in_range)
            if in_range:
                var row_pos = column_count + Int(pop_count(mask & lid_mask))
                # we still continue to look for more hits to return valid vd
                if row_pos < max_k:
                    tmp.unsafe_store(
                        tmp_base + row_pos, r_1nn_cols.unsafe_load(r_start + i)
                    )
            column_count += Int(pop_count(mask))

            if cur_r_dist - min_warp_dist > eps:
                i = 0

            var i0 = Int(shuffle_idx(Int32(i), UInt32(0)))

            while i0 >= RBC_LANES:
                i0 -= RBC_LANES
                var min_warp_dist2 = r_1nn_dists.unsafe_load(r_start + i0)
                var dist2 = eps_dist_sq(
                    x,
                    x_base,
                    x_reordered,
                    (r_start + i0 + lid) * n_cols,
                    n_cols,
                )
                var in_range2 = dist2 <= eps2
                var mask2 = vote[RBC_MASK_DT](in_range2)
                if in_range2:
                    var row_pos2 = column_count + Int(
                        pop_count(mask2 & lid_mask)
                    )
                    if row_pos2 < max_k:
                        tmp.unsafe_store(
                            tmp_base + row_pos2,
                            r_1nn_cols.unsafe_load(r_start + i0 + lid),
                        )
                column_count += Int(pop_count(mask2))
                if cur_r_dist - min_warp_dist2 > eps:
                    i0 = 0

    if lid == 0:
        vd.unsafe_store(query_id, Int32(column_count))


def block_rbc_kernel_eps_max_k_copy(
    max_k_in: Int32,
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    tmp: MutPointer[Int32, MutAnyOrigin],
    adj_ja: MutPointer[Int32, MutAnyOrigin],
):
    """`block_rbc_kernel_eps_max_k_copy`, `registers.cuh:983-1000`.

    Compacts the `max_k`-strided scratch into the CSR that the scan of `vd`
    just laid out. One block per row.
    """
    var max_k = Int(max_k_in)
    var row_idx = Int(block_idx.x)
    var offset = row_idx * max_k
    var col_start = Int(adj_ia.unsafe_load(row_idx))
    var num_cols = Int(adj_ia.unsafe_load(row_idx + 1)) - col_start

    var i = Int(thread_idx.x)
    while i < num_cols:
        adj_ja.unsafe_store(col_start + i, tmp.unsafe_load(offset + i))
        i += RBC_COPY_TPB


# ---------------------------------------------------------------------------
# The host side: `rbc_eps_pass`, `registers.cuh:1271` and `:1314`.
# ---------------------------------------------------------------------------


def rbc_eps_pass_count(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    eps: Float32,
) raises -> Int:
    """Pass one: `rbc_eps_pass` with `max_k == nullptr` and `adj_ja == nullptr`.

    `registers.cuh:1327-1381`. Counts into `vd`, exclusive-scans `vd` into
    `adj_ia`, and copies the total into `vd[n_queries]` (`:1484-1490`). The
    total is returned because that is exactly what
    `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:149-152` reads back to the host
    to size `ja` before pass two.

    `vd` must be `n_queries + 1` long and `adj_ia` `n_queries + 1` long.
    """
    ctx.enqueue_function[block_rbc_kernel_eps_csr_pass](
        x_reordered.unsafe_ptr(),
        query.unsafe_ptr(),
        Int32(n_queries),
        Int32(n_cols),
        r.unsafe_ptr(),
        eps,
        Int32(n_landmarks),
        r_indptr.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        r_radius.unsafe_ptr(),
        vd.unsafe_ptr(),
        adj_ia.unsafe_ptr(),
        Int32(0),
        grid_dim=((n_queries + RBC_QPB - 1) // RBC_QPB, 1, 1),
        block_dim=(RBC_TPB, 1, 1),
    )
    ctx.enqueue_function[rbc_exclusive_scan_kernel](
        adj_ia.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n_queries),
        grid_dim=(1, 1, 1),
        block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    ctx.synchronize()

    var h = ctx.enqueue_create_host_buffer[DType.int32](n_queries + 1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=adj_ia)
    ctx.synchronize()
    var nnz = Int(h.unsafe_ptr().unsafe_load(n_queries))

    # `vd + n_query_rows` stores the total number of edges, `:1486-1490`.
    var t = ctx.enqueue_create_host_buffer[DType.int32](1)
    t.unsafe_ptr().unsafe_store(0, Int32(nnz))
    ctx.enqueue_copy(
        dst_buf=vd.create_sub_buffer[DType.int32](n_queries, 1),
        src_ptr=t.unsafe_ptr(),
    )
    ctx.synchronize()
    return nnz


def rbc_eps_pass_fill(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    eps: Float32,
) raises:
    """Pass two: `rbc_eps_pass` with `adj_ja != nullptr`, `:1382-1426`."""
    ctx.enqueue_function[block_rbc_kernel_eps_csr_pass](
        x_reordered.unsafe_ptr(),
        query.unsafe_ptr(),
        Int32(n_queries),
        Int32(n_cols),
        r.unsafe_ptr(),
        eps,
        Int32(n_landmarks),
        r_indptr.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        r_radius.unsafe_ptr(),
        adj_ia.unsafe_ptr(),
        adj_ja.unsafe_ptr(),
        Int32(1),
        grid_dim=((n_queries + RBC_QPB - 1) // RBC_QPB, 1, 1),
        block_dim=(RBC_TPB, 1, 1),
    )
    ctx.synchronize()


def rbc_eps_pass_dense(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    m: Int,
    eps: Float32,
) raises:
    """`rbc_eps_pass`, the dense overload, `registers.cuh:1271-1311`.

    `perform_rbc_eps_nn_query` zeroes `adj` first (`ball_cover.cuh:288-289`)
    because the kernel only ever writes `true`. That memset is done here.
    """
    ctx.enqueue_memset(adj, UInt8(0))
    ctx.enqueue_function[block_rbc_kernel_eps_dense](
        x_reordered.unsafe_ptr(),
        query.unsafe_ptr(),
        Int32(n_queries),
        Int32(n_cols),
        r.unsafe_ptr(),
        Int32(m),
        eps,
        Int32(n_landmarks),
        r_indptr.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        r_radius.unsafe_ptr(),
        adj.unsafe_ptr(),
        vd.unsafe_ptr(),
        grid_dim=((n_queries + RBC_QPB - 1) // RBC_QPB, 1, 1),
        block_dim=(RBC_TPB, 1, 1),
    )
    ctx.synchronize()


def rbc_eps_pass_max_k(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    mut tmp: DeviceBuffer[DType.int32],
    mut scratch: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    eps: Float32,
    max_k: Int,
) raises -> Int:
    """`rbc_eps_pass`, the max_k overload, `registers.cuh:1427-1482`.

    Returns their `actual_max`, the longest row seen. `tmp` must be
    `n_queries * max_k` long and `scratch` at least one element.

    Their order is exact and is copied: run the kernel, take the MAX of the
    unclamped `vd`, clamp `vd` only if that maximum overflowed `max_k`, then
    scan and compact. Clamping before taking the maximum would lose the very
    number the caller re-runs on.
    """
    ctx.enqueue_function[block_rbc_kernel_eps_max_k](
        x_reordered.unsafe_ptr(),
        query.unsafe_ptr(),
        Int32(n_queries),
        Int32(n_cols),
        r.unsafe_ptr(),
        eps,
        Int32(n_landmarks),
        r_indptr.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        r_radius.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(max_k),
        tmp.unsafe_ptr(),
        grid_dim=((n_queries + RBC_QPB - 1) // RBC_QPB, 1, 1),
        block_dim=(RBC_TPB, 1, 1),
    )
    ctx.enqueue_function[rbc_max_reduce_kernel](
        scratch.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n_queries),
        grid_dim=(1, 1, 1),
        block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    ctx.synchronize()

    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=scratch)
    ctx.synchronize()
    var actual_max = Int(h.unsafe_ptr().unsafe_load(0))

    if actual_max > max_k:
        ctx.enqueue_function[rbc_clamp_kernel](
            vd.unsafe_ptr(),
            Int32(n_queries),
            Int32(max_k),
            grid_dim=((n_queries + RBC_SCAN_TPB - 1) // RBC_SCAN_TPB, 1, 1),
            block_dim=(RBC_SCAN_TPB, 1, 1),
        )

    ctx.enqueue_function[rbc_exclusive_scan_kernel](
        adj_ia.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(n_queries),
        grid_dim=(1, 1, 1),
        block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[block_rbc_kernel_eps_max_k_copy](
        Int32(max_k),
        adj_ia.unsafe_ptr(),
        tmp.unsafe_ptr(),
        adj_ja.unsafe_ptr(),
        grid_dim=(n_queries, 1, 1),
        block_dim=(RBC_COPY_TPB, 1, 1),
    )
    ctx.synchronize()

    # `vd + n_query_rows` gets the edge total, `:1484-1490`.
    var hi = ctx.enqueue_create_host_buffer[DType.int32](n_queries + 1)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=adj_ia)
    ctx.synchronize()
    var t = ctx.enqueue_create_host_buffer[DType.int32](1)
    t.unsafe_ptr().unsafe_store(0, hi.unsafe_ptr().unsafe_load(n_queries))
    ctx.enqueue_copy(
        dst_buf=vd.create_sub_buffer[DType.int32](n_queries, 1),
        src_ptr=t.unsafe_ptr(),
    )
    ctx.synchronize()
    return actual_max
