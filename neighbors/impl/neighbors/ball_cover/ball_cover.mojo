# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Building the random ball cover index, and the eps query entry points.

PORT OF `cuvs/src/neighbors/ball_cover/ball_cover.cuh` at cuVS `94c2819`:
`sample_landmarks` (`:62`), `construct_landmark_1nn` (`:121`),
`k_closest_landmarks` (`:180`), `compute_landmark_radii` (`:212`),
`rbc_build_index` (`:330`), `perform_rbc_eps_nn_query` (`:277`, `:300`) and
`rbc_eps_nn_query` (`:533`, `:550`). Partial. Do not improve.

WHAT THE CALLER GETS AND WHAT CUML'S CALLER DOES WITH IT
---------------------------------------------------------
`cuml/cpp/src/dbscan/runner.cuh:234-242` builds one index for the whole
dataset before the batch loop, and `vertexdeg/algo.cuh:107-164` then calls
`eps_nn` once per batch with the batch's rows as the QUERY and the whole
dataset as the index. Two shapes come out of `eps_nn`, and both are here:

- the two-pass CSR (`algo.cuh:137-163`): count into `ia`/`vd`, read the total
  back to the host, size `ja`, then fill `ja`;
- the one-pass `max_k` form (`algo.cuh:122-135`), taken when the caller can
  bound the row length.

`rbc_eps_nn_query_csr` and `rbc_eps_nn_query_max_k` below are those two, and
`bench/results/LANE_ball-cover_2026-08-19.md` records exactly what the DBSCAN
lane has to call.

THE INDEX IS sqrt(m) LANDMARKS, WHICH IS THEIRS AND IS THE WHOLE BOUND
-----------------------------------------------------------------------
`cuvs/include/cuvs/neighbors/ball_cover.hpp:62` sets
`n_landmarks(raft::sqrt(X_.extent(0)))` with their comment at `:57-60`: "the
sqrt() here makes the sqrt(m)^2 a linear-time lower bound". Confirmed in
their file rather than assumed. Their footprint comment on the same lines,
`(2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)`, undercounts by exactly
`X_reordered`, which is another `n * m` and is allocated at `:68`, six lines
below that comment.

DEVIATION 1: THE LANDMARK DRAW IS FLOYD'S, ON THE HOST
-------------------------------------------------------
`sample_landmarks` (`:89-97`) calls `raft::random::sampleWithoutReplacement`
with a fixed seed and a weight vector of all ones (`:76-79`). That routine
(`raft/random/detail/rng_impl.cuh:292-325`) is three steps: one random key per
input element, a FULL device sort of all m keys — their own `///@todo` at
`:315` calls the full sort wasteful — and the first `sampledLen` indices of
the result.

Ours draws the same distribution with Floyd's algorithm on the host, in
`_floyd_sample` below: O(sqrt(m)) draws, no device sort, no m-element
allocation. With all weights equal, their three steps are exactly "a uniform
random subset of size sqrt(m)", and Floyd's is exactly that subset drawn
directly. It is not the same SUBSET, and that cannot change any answer: the
triangle-inequality prune is exact for ANY landmark set (see
`registers.mojo`), so the draw moves how much work the query does and nothing
else.

**The sort in their version is not portable to this toolchain anyway.** See
DEVIATION 3.

DEVIATION 2: THE 1-NN IS FUSED, THEIR BRUTE FORCE IS NOT
---------------------------------------------------------
`k_closest_landmarks` (`:188-200`) calls `cuvs::neighbors::brute_force::build`
and `search` at k = 1. That path materializes an `m x n_landmarks` distance
matrix. At m = 200,000 that is 200,000 x 447 float32 = 357 MB, for an argmin
that keeps two values. `rbc_landmark_1nn_kernel` below computes the same
argmin without materializing anything, which is the shape RAFT itself uses
when k = 1 — `fusedDistanceNN`, already ported in this repository at
`cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo`. It is not
tiled because it does not need to be: `n_landmarks` is sqrt(m) and the
landmark matrix is small enough to stay in cache for every query row.

**This also makes the build and the query agree arithmetically.** Their
brute_force under `L2SqrtExpanded` uses the expanded identity
`||a||^2 + ||b||^2 - 2ab`, while their query kernel's `EuclideanSqFunc` sums
the differences directly. `PORTING.md 21` already records what the expanded
identity costs in float32 — for collinear points the closest-pair distance
falls below the ulp of the norms at any scale — and here the two formulas are
being COMPARED to each other, `R_radius` from one against `cur_R_dist` from
the other. Ours sums the differences directly in both, so `R_1nn_dists` and
`R_radius` are produced by the same arithmetic the query compares them
against. Their pairing permits a boundary case where a point is inside eps by
one formula and outside by the other; ours does not.

DEVIATION 3: A COUNTING SORT BY LANDMARK, THEN AN EXACT RANK IN EACH GROUP
---------------------------------------------------------------------------
`construct_landmark_1nn` sorts every point once, globally, with
`thrust::sort_by_key` and the `NNComp` comparator (`:148-152`), and then
rebuilds the group boundaries from the sorted keys with
`raft::sparse::convert::sorted_coo_to_csr` (`:155-159`).

**`nn.argsort[target="gpu"]` CANNOT BE USED HERE, AND THE REASON IS NOT A
RULE.** It is wrong. Measured on this M4, this toolchain, 2026-08-19: it
sorts correctly at n = 256 and returns a non-monotone permutation at
n = 257 and every larger size tried (512, 1024, 1200, 4096), for uint64,
uint32 and float32 keys and for both int32 and int64 index outputs, with the
first inversion always at output position 256. The probe and its output are
in `bench/results/LANE_ball-cover_2026-08-19.md`. Substituting it would have
shipped an index whose groups are not sorted, which makes the query kernel's
early stop drop real neighbors — a wrong DBSCAN, silently.

So the ordering is produced here, in two kernels, and the output is the same
ordering theirs produces:

1. `rbc_count_landmarks_kernel` plus an exclusive scan IS
   `sorted_coo_to_csr`: their routine counts per row and scans, and counting
   does not care whether the input is sorted. This gives `R_indptr`.
2. `rbc_scatter_kernel` places each point in its landmark's slice in
   arbitrary order, and `rbc_rank_kernel` then gives each element of a slice
   its exact rank by counting the elements of the SAME slice that precede it.
   Rank-by-counting needs no barrier, no shared memory and no atomics, and
   it is deterministic: ties on distance are broken by the original point
   index, where `thrust::sort_by_key` is unstable and leaves them arbitrary.

The cost is `sum over landmarks of |group|^2`, which at sqrt(m) landmarks is
O(m^1.5) expected — the same order as the ball cover's own query bound
(`ball_cover.cuh:318-327`, "a guarantee of sqrt(n) * c^{3/2}"), so it does
not touch the asymptotics this lane exists to fix.

**AND IT IS NOT WORTH REPLACING. MEASURED, M4, 2026-08-19.** The claim this
paragraph used to carry — that the rank would dominate the build at large m —
is false, and it is false for a structural reason: the rank and the 1-nn
kernel above it are BOTH O(m^1.5), so their ratio is fixed and the rank never
catches up. Build stages, min of three, DBSCAN scaling fixture (d = 8):

    n          1-nn      count+scan+scatter    RANK    reorder+radii
    16,000     0.62 ms         0.36 ms        0.30 ms      0.25 ms
    50,000     2.33            0.43           0.58         0.35
    100,000    6.05            0.46           1.16         0.49
    200,000   16.53            0.54           2.59         0.80

The rank is 13% of the build at 200,000 and the whole build is 7% of a
DBSCAN fit, so **the rank is 0.4% of the fit** — and the 1-nn kernel above it
is 6.4x larger. The tail is mild too: the worst group is 2.3x the mean on a
uniform fixture and 4.5x on a 12-blob clustered one, which puts
`sum |group|^2` at 1.18x and 1.42x the balanced `m^1.5` and the rank kernel
at 4.4 ms and 6.4 ms at m = 200,000.

So a per-group bitonic sort, or a port of CUB's `DeviceRadixSort` (open, and
the same digit-histogram shape as the RAFT radix SELECT already at
`neighbors/gbdt/matrix/detail/select_radix.mojo`), is worth at most 1% of a
fit here. `DeviceRadixSort` is still the general device sort this repository
lacks — `nn.argsort[target="gpu"]` is wrong above 256 elements — and it
should be ported for that reason, by whoever needs it. It should not be
ported for this.

**THE ORDER IS LOAD-BEARING AND IS NOT AN OPTIMIZATION.** Ascending order
within each landmark group is what makes the query kernel's backward walk
sound: it stops at the first point that fails `d(q,r) - d(r,y) <= eps` and
declares every remaining point out of reach. Shuffle the group and that early
stop drops real neighbors. Their own comment says so at
`registers.cuh:654-656`.

DEVIATION 4: AN EMPTY LANDMARK GETS RADIUS 0 INSTEAD OF AN OUT-OF-BOUNDS READ
-----------------------------------------------------------------------------
`compute_landmark_radii` (`:224`) reads `R_1nn_dists[R_indptr[r+1] - 1]` with
no emptiness guard. A landmark whose group is empty therefore reads the
previous group's last element, and landmark 0 with an empty group reads index
-1. Their code is safe in practice because a landmark is one of the index
points and is at distance 0 from itself, so it is its own nearest landmark
unless a duplicate point with a lower landmark index steals the tie. Ours
takes radius 0 for an empty group, which is exact — a ball with no points in
it cannot contain a neighbor, so pruning it always is correct — and it cannot
read out of bounds.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from neighbors.impl.neighbors.ball_cover.common import (
    RBC_FLT_MAX,
    eps_dist_sq,
)
from neighbors.impl.neighbors.ball_cover.registers import (
    RBC_TPB,
    rbc_eps_pass_count,
    rbc_eps_pass_dense,
    rbc_eps_pass_fill,
    rbc_eps_pass_max_k,
)
from neighbors.impl.neighbors.ball_cover.scan import (
    RBC_SCAN_TPB,
    rbc_exclusive_scan_kernel,
)


comptime RBC_BUILD_TPB = 256


def rbc_n_landmarks(m: Int) -> Int:
    """`index::n_landmarks = raft::sqrt(X_.extent(0))`, `ball_cover.hpp:62`.

    Their `raft::sqrt` of an integer extent returns a double that is then
    truncated into an `int64_t`, so this is `floor(sqrt(m))`. Computed with a
    correction step rather than trusted to the float, because a landmark
    count one too large would index past `R`.
    """
    if m <= 1:
        return 1
    var s = Int(sqrt(Float64(m)))
    while (s + 1) * (s + 1) <= m:
        s += 1
    while s * s > m:
        s -= 1
    if s < 1:
        s = 1
    return s


def _floyd_sample(m: Int, n_landmarks: Int, seed: UInt64) -> List[Int32]:
    """`sample_landmarks`, `ball_cover.cuh:62-108`. See DEVIATION 1.

    Floyd's algorithm draws `n_landmarks` DISTINCT indices from `[0, m)`,
    uniformly over subsets, in `n_landmarks` steps and with no array of size
    m. Theirs draws the same distribution by keying and sorting all m.

    The generator is splitmix64 on `(seed, step)`, so the draw is a pure
    function of the seed exactly as their fixed `RngState(12345)` makes
    theirs a pure function of 12345.
    """
    var picked = List[Int32]()
    var step = UInt64(0)
    for j in range(m - n_landmarks, m):
        var z = seed + step * UInt64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> UInt64(31))
        step += UInt64(1)
        var t = Int32(Int(z % UInt64(j + 1)))
        var hit = False
        for q in range(len(picked)):
            if picked[q] == t:
                hit = True
                break
        if hit:
            picked.append(Int32(j))
        else:
            picked.append(t)
    return picked^


def rbc_copy_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    idx: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`raft::matrix::copy_rows`, used at `ball_cover.cuh:102` and `:162`.

    `dst[j] = src[idx[j]]`. Both uses are a gather of whole rows: the first
    builds `R` from the sampled landmark ids, the second builds
    `X_reordered` from `R_1nn_cols` so the query kernel's walk down a
    landmark's group is a contiguous read. Their comment at `:161` is
    "reorder X to allow aligned access".
    """
    var n_cols = Int(n_cols_in)
    var e = Int(block_idx.x) * RBC_BUILD_TPB + Int(thread_idx.x)
    var total = Int(n_rows_in) * n_cols
    if e >= total:
        return
    var row = e // n_cols
    var col = e % n_cols
    var srow = Int(idx.unsafe_load(row))
    dst.unsafe_store(e, src.unsafe_load(srow * n_cols + col))


def rbc_landmark_1nn_kernel(
    nearest: MutPointer[Int32, MutAnyOrigin],
    nearest_dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_cols_in: Int32,
    n_landmarks_in: Int32,
):
    """`k_closest_landmarks` at k = 1, fused. See DEVIATION 2.

    Writes the TRUE Euclidean distance, not the squared one. The query
    kernels compare `R_1nn_dists` and `R_radius` against unsquared quantities
    (`registers.cuh:500` and `:676`), so the root belongs here.

    Ties go to the LOWER landmark index, which is what a strict `<` gives and
    what makes a landmark its own nearest landmark except against a duplicate
    point with a smaller index. That is the case DEVIATION 4 guards.
    """
    var m = Int(m_in)
    var n_cols = Int(n_cols_in)
    var n_landmarks = Int(n_landmarks_in)
    var i = Int(block_idx.x) * RBC_BUILD_TPB + Int(thread_idx.x)
    if i >= m:
        return

    var best = RBC_FLT_MAX
    var best_k = 0
    var x_base = i * n_cols
    for k in range(n_landmarks):
        var d = eps_dist_sq(x, x_base, r, k * n_cols, n_cols)
        if d < best:
            best = d
            best_k = k

    nearest.unsafe_store(i, Int32(best_k))
    nearest_dist.unsafe_store(i, sqrt(best))


def rbc_scatter_kernel(
    slot_cols: MutPointer[Int32, MutAnyOrigin],
    slot_dists: MutPointer[Float32, MutAnyOrigin],
    cursor: MutPointer[Int32, MutAnyOrigin],
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    nearest_dist: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
):
    """Group by landmark. The counting-sort half of DEVIATION 3.

    Order WITHIN a slice is whatever the atomics gave; `rbc_rank_kernel`
    fixes that next. Splitting it in two is what removes the barrier and the
    shared-memory bound a one-pass segmented sort would carry.
    """
    var i = Int(block_idx.x) * RBC_BUILD_TPB + Int(thread_idx.x)
    if i >= Int(m_in):
        return
    var k = Int(nearest.unsafe_load(i))
    var off = Atomic.fetch_add(cursor.unsafe_offset(k), Int32(1))
    var pos = Int(r_indptr.unsafe_load(k)) + Int(off)
    slot_cols.unsafe_store(pos, Int32(i))
    slot_dists.unsafe_store(pos, nearest_dist.unsafe_load(i))


def rbc_rank_kernel(
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    slot_cols: MutPointer[Int32, MutAnyOrigin],
    slot_dists: MutPointer[Float32, MutAnyOrigin],
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    n_landmarks_in: Int32,
):
    """Sort each landmark's slice ascending by distance. One block per slice.

    `NNComp`'s order (`common.cuh:26-37`) with the tie broken by the original
    point index, which theirs leaves arbitrary because `thrust::sort_by_key`
    is not stable. Each element counts how many elements of its own slice
    precede it and writes itself at that rank. No barrier, no shared memory,
    no atomic, and the same answer every run.
    """
    var k = Int(block_idx.x)
    if k >= Int(n_landmarks_in):
        return
    var s = Int(r_indptr.unsafe_load(k))
    var e = Int(r_indptr.unsafe_load(k + 1))

    var p = s + Int(thread_idx.x)
    while p < e:
        var dp = slot_dists.unsafe_load(p)
        var ip = slot_cols.unsafe_load(p)
        var rank = 0
        for q in range(s, e):
            var dq = slot_dists.unsafe_load(q)
            if dq < dp:
                rank += 1
            elif dq == dp and slot_cols.unsafe_load(q) < ip:
                rank += 1
        r_1nn_dists.unsafe_store(s + rank, dp)
        r_1nn_cols.unsafe_store(s + rank, ip)
        p += RBC_BUILD_TPB


def rbc_count_landmarks_kernel(
    counts: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
):
    """`raft::sparse::convert::sorted_coo_to_csr`, `ball_cover.cuh:155-159`.

    Their routine counts the rows of a SORTED coo and exclusive-scans the
    counts. Counting is order-independent, so it is done here on the
    unsorted assignment array and the scan follows; the result is the same
    `R_indptr` because the sort groups by exactly this key.
    """
    var i = Int(block_idx.x) * RBC_BUILD_TPB + Int(thread_idx.x)
    if i >= Int(m_in):
        return
    _ = Atomic.fetch_add(
        counts.unsafe_offset(Int(nearest.unsafe_load(i))), Int32(1)
    )


def rbc_landmark_radii_kernel(
    r_radius: MutPointer[Float32, MutAnyOrigin],
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    n_landmarks_in: Int32,
):
    """`compute_landmark_radii`, `ball_cover.cuh:212-227`.

    The group is sorted ascending, so its LAST element is the radius. See
    DEVIATION 4 for the empty-group guard, which theirs does not have.
    """
    var k = Int(block_idx.x) * RBC_BUILD_TPB + Int(thread_idx.x)
    if k >= Int(n_landmarks_in):
        return
    var start = Int(r_indptr.unsafe_load(k))
    var end = Int(r_indptr.unsafe_load(k + 1))
    if end <= start:
        r_radius.unsafe_store(k, Float32(0.0))
        return
    r_radius.unsafe_store(k, r_1nn_dists.unsafe_load(end - 1))


def rbc_build_index(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut x_reordered: DeviceBuffer[DType.float32],
    mut landmark_ids: DeviceBuffer[DType.int32],
    mut slot_cols: DeviceBuffer[DType.int32],
    mut slot_dists: DeviceBuffer[DType.float32],
    mut nearest: DeviceBuffer[DType.int32],
    mut nearest_dist: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut counts: DeviceBuffer[DType.int32],
    m: Int,
    n_cols: Int,
    n_landmarks: Int,
    seed: UInt64 = UInt64(12345),
) raises:
    """`rbc_build_index`, `ball_cover.cuh:330-378`. Their four steps, in order.

    Buffer sizes, all device:
        x             m * n_cols        (input, row major, untouched)
        r             L * n_cols
        x_reordered   m * n_cols
        landmark_ids  L
        slot_cols     m                 (int32 scratch)
        slot_dists    m                 (float32 scratch)
        nearest       m
        nearest_dist  m
        r_indptr      L + 1
        r_1nn_cols    m
        r_1nn_dists   m
        r_radius      L
        counts        L                 (int32 scratch, reused as the cursor)

    with `L = rbc_n_landmarks(m)`. `seed` is theirs, 12345 at `:89`.

    Everything after the landmark draw is on the device, which is their split:
    `sample_landmarks` is the only step whose host side is more than a launch.
    """
    var grid_m = (m + RBC_BUILD_TPB - 1) // RBC_BUILD_TPB

    # 1. Randomly sample sqrt(n) points from X. `:347-350`. See DEVIATION 1.
    var picked = _floyd_sample(m, n_landmarks, seed)
    var hid = ctx.enqueue_create_host_buffer[DType.int32](n_landmarks)
    for j in range(n_landmarks):
        hid.unsafe_ptr().unsafe_store(j, picked[j])
    ctx.enqueue_copy(dst_buf=landmark_ids, src_ptr=hid.unsafe_ptr())
    ctx.synchronize()

    var cells_r = n_landmarks * n_cols
    ctx.enqueue_function[rbc_copy_rows_kernel](
        r.unsafe_ptr(),
        x.unsafe_ptr(),
        landmark_ids.unsafe_ptr(),
        Int32(n_landmarks),
        Int32(n_cols),
        grid_dim=((cells_r + RBC_BUILD_TPB - 1) // RBC_BUILD_TPB, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )

    # 2. knn = bfknn(X, R, k) with k = 1. `:352-362`. See DEVIATION 2.
    ctx.enqueue_function[rbc_landmark_1nn_kernel](
        nearest.unsafe_ptr(),
        nearest_dist.unsafe_ptr(),
        x.unsafe_ptr(),
        r.unsafe_ptr(),
        Int32(m),
        Int32(n_cols),
        Int32(n_landmarks),
        grid_dim=(grid_m, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )

    # 3. L_r = knn[:,0].T as CSR, each group sorted by distance. `:364-371`.
    #    See DEVIATION 3: counts and scan first (their `sorted_coo_to_csr`),
    #    then scatter into the slices, then rank inside each slice.
    ctx.enqueue_memset(counts, Int32(0))
    ctx.enqueue_function[rbc_count_landmarks_kernel](
        counts.unsafe_ptr(),
        nearest.unsafe_ptr(),
        Int32(m),
        grid_dim=(grid_m, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )
    ctx.enqueue_function[rbc_exclusive_scan_kernel](
        r_indptr.unsafe_ptr(),
        counts.unsafe_ptr(),
        Int32(n_landmarks),
        grid_dim=(1, 1, 1),
        block_dim=(RBC_SCAN_TPB, 1, 1),
    )
    # `counts` becomes the per-landmark write cursor.
    ctx.enqueue_memset(counts, Int32(0))
    ctx.enqueue_function[rbc_scatter_kernel](
        slot_cols.unsafe_ptr(),
        slot_dists.unsafe_ptr(),
        counts.unsafe_ptr(),
        r_indptr.unsafe_ptr(),
        nearest.unsafe_ptr(),
        nearest_dist.unsafe_ptr(),
        Int32(m),
        grid_dim=(grid_m, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )
    ctx.enqueue_function[rbc_rank_kernel](
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        slot_cols.unsafe_ptr(),
        slot_dists.unsafe_ptr(),
        r_indptr.unsafe_ptr(),
        Int32(n_landmarks),
        grid_dim=(n_landmarks, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )

    # "reorder X to allow aligned access", `:161-163`.
    var cells_x = m * n_cols
    ctx.enqueue_function[rbc_copy_rows_kernel](
        x_reordered.unsafe_ptr(),
        x.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        Int32(m),
        Int32(n_cols),
        grid_dim=((cells_x + RBC_BUILD_TPB - 1) // RBC_BUILD_TPB, 1, 1),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )

    # 4. radius(r) for the filter p(q, r) <= p(q, q_r) + radius(r). `:373-377`.
    ctx.enqueue_function[rbc_landmark_radii_kernel](
        r_radius.unsafe_ptr(),
        r_indptr.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        Int32(n_landmarks),
        grid_dim=(
            (n_landmarks + RBC_BUILD_TPB - 1) // RBC_BUILD_TPB,
            1,
            1,
        ),
        block_dim=(RBC_BUILD_TPB, 1, 1),
    )
    ctx.synchronize()


# ---------------------------------------------------------------------------
# The query entry points. These are the shape `cuvs::neighbors::ball_cover::
# eps_nn` presents and the shape `cuml/cpp/src/dbscan/vertexdeg/algo.cuh`
# consumes.
# ---------------------------------------------------------------------------


def rbc_eps_nn_query_count(
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
    """`eps_nn` with a null `ja`: fills `ia` and `vd`, returns the edge count.

    `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:137-153`. The caller sizes `ja`
    from the return value and then calls `rbc_eps_nn_query_fill`.
    """
    return rbc_eps_pass_count(
        ctx,
        x_reordered,
        query,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        vd,
        n_queries,
        n_cols,
        n_landmarks,
        eps,
    )


def rbc_eps_nn_query_fill(
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
    """`eps_nn` with a sized `ja`, `algo.cuh:155-162`."""
    rbc_eps_pass_fill(
        ctx,
        x_reordered,
        query,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        adj_ja,
        n_queries,
        n_cols,
        n_landmarks,
        eps,
    )


def rbc_eps_nn_query_max_k(
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
    """`eps_nn` with a host `max_k`, `algo.cuh:122-135`.

    Returns the longest row actually found. The caller's contract is theirs:
    if the return value exceeds the `max_k` passed in, the CSR is truncated
    and must be recomputed with the larger bound (`algo.cuh:135` asserts
    rather than retries).
    """
    return rbc_eps_pass_max_k(
        ctx,
        x_reordered,
        query,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        adj_ja,
        vd,
        tmp,
        scratch,
        n_queries,
        n_cols,
        n_landmarks,
        eps,
        max_k,
    )


def rbc_eps_nn_query_dense(
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
    """`rbc_eps_nn_query`, the dense overload, `ball_cover.cuh:533-547`."""
    rbc_eps_pass_dense(
        ctx,
        x_reordered,
        query,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj,
        vd,
        n_queries,
        n_cols,
        n_landmarks,
        m,
        eps,
    )
