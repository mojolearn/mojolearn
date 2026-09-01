# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-NEAREST-NEIGHBOUR query over the random ball cover.

PORT OF `cuvs/src/neighbors/ball_cover/ball_cover.cuh::rbc_knn_query`
(`:446-498`), `perform_rbc_query` (`:240-270`) and
`cuvs/src/neighbors/ball_cover/registers.cuh::block_rbc_kernel_registers`
(`:305-441`) plus `perform_post_filter_registers` (`:64-121`) and
`compute_final_dists_registers` (`:146-280`), at cuVS `94c2819`.
Reimplemented; the DEVIATIONS are numbered below and each says what moved.

WHY THIS FILE EXISTS AND WHAT IT CLOSES
----------------------------------------
`neighbors/NOT_IMPLEMENTED.tsv` carried two rows that were really one
question. The first refused `algorithm='kd_tree'`, correctly: a kd-tree eps
query is a per-query stack walk with data-dependent branching and divergent
per-lane memory access, which is the shape a GPU is worst at, and past
roughly fifteen dimensions the pruning bound stops firing and it degenerates
to a full scan, so it loses to the brute force in this same directory on
exactly the workloads it claims to help. The second recorded what a caller
asking for a tree actually wants and what was missing: `rbc_knn_query`, the
k-NN entry point over the index that was already here.

That is what this file is. **The answer to "give me a spatial index instead
of brute force" on a GPU is the ball cover, not a tree**, and it is exact,
so nothing is traded away to get it.

THE BOUNDS, AND THE PROOF THAT EACH ONE IS EXACT
-------------------------------------------------
Write `d_k(q)` for the true distance from `q` to its k-th nearest index
point, `L` for the landmark set, and `radius(l)` for the largest `d(l, y)`
over the points `y` assigned to landmark `l`. Every landmark IS an index
point (`_floyd_sample` draws them from `X`), and every point is assigned to
its NEAREST landmark (`rbc_landmark_1nn_kernel`). Those two facts are what
the bounds below rest on, together with the triangle inequality.

  D. **The k-th nearest LANDMARK distance is an upper bound on `d_k(q)`.**
     The landmarks are index points, so the k nearest of them are k index
     points, so there are at least k index points within `D := d(q, l_(k))`.
     Hence `d_k(q) <= D`. This is cuVS's `min_R_dist` (`registers.cuh:347`,
     misleadingly named -- it is the k-th, not the first).

  T. **The running k-th best is an upper bound on `d_k(q)`.** `tau` is the
     k-th smallest distance over a SUBSET of the index, so `tau >= d_k(q)`.
     The threshold used everywhere below is `thresh = min(tau, D)`, which
     is therefore also `>= d_k(q)`.

  1. **Landmark test.** Every `y` in `l`'s ball has `d(l, y) <= radius(l)`,
     so `d(q, y) >= d(q, l) - radius(l)`. If `d(q, l) - radius(l) > thresh`
     then every `y` in `l` is strictly farther than `d_k(q)` and none of
     them is in the answer. cuVS's `:363`.

  2. **Cayton's landmark test.** If `y` is a true k-NN of `q` and `l` is
     `y`'s landmark, then `d(y, l) <= d(y, l_q)` for `q`'s nearest landmark
     `l_q` (because `l` is `y`'s NEAREST landmark), and
     `d(y, l_q) <= d(y, q) + d(q, l_q) <= d_k + m` where `m = d(q, l_q)`.
     So `d(q, l) <= d(q, y) + d(y, l) <= 2 d_k + m <= 3D`, using `m <= D`
     and `d_k <= D`. Hence `d(q, l) > 3D` prunes `l` outright. This is
     cuVS's `:364` and their post-filter's `dist > 3 * closest_R_dist`
     (`:110`). **It needs D and NOT tau**: `tau >= d_k` says nothing about
     `m`, so `3 * tau` is not a bound.

  3. **In-group test.** `d(q, y) >= |d(q, l) - d(l, y)|`, both directions.
     `R_1nn_dists` holds `d(l, y)` ASCENDING inside each group, so the
     high side `d(l, y) - d(q, l) > thresh` is monotone in the walk: once
     it fires, every later element of the group also fails it and the walk
     stops. The low side `d(q, l) - d(l, y) > thresh` skips an element
     without stopping. This is the same pair of statements the eps kernels
     in `registers.mojo` use, with `thresh` in place of `eps`; the only
     difference is that `thresh` SHRINKS as the answer fills, which keeps
     the stop valid because a smaller threshold prunes more.

  A pruned point is never in the answer, so **the answer is the exact k-NN
  set and does not depend on the landmark draw, the visiting order, the
  block width or the lane width.** That last clause is the sharp difference
  from the eps path: the eps CSR's COLUMN ORDER is lane-width dependent and
  needed DEVIATION 551's canonicalization pass. A k-NN answer is a top-k
  under a total order, so any two visiting orders that see supersets of the
  answer produce the same answer, byte for byte, with no canonicalization
  and no extra kernel. See DEVIATION 558.

THE TOTAL ORDER, STATED SO THE ORACLE CAN USE THE SAME ONE
-----------------------------------------------------------
    (comparison-space distance, index)  ascending

where the comparison-space distance is `rbc_cmp_dist`: SQUARED Euclidean on
the Euclidean arm and the plain metric on the other three (DEVIATION 564 in
`common.mojo`). Ordering on the squared value rather than on its rounded
root is deliberate: `identical_sqrt` is a rounding, so two different squared
distances can share one root, and ordering on the root would make the answer
depend on which side of that rounding a pair fell. The REPORTED distance is
still the true one; only the ORDER is taken in comparison space.

DEVIATION 558: A BLOCK-LEVEL SELECTOR, NOT FAISS'S `KeyValueBlockSelect`
-------------------------------------------------------------------------
THEIRS: `registers.cuh:339` instantiates
`faiss_select::KeyValueBlockSelect<..., warp_q, thread_q, tpb>`, a
register-resident warp queue whose capacity is a template parameter, chosen
by a six-way host dispatch on `k` (`:1016-1128`).

OURS: one query per BLOCK, `RBC_KNN_TPB` threads, and the running answer in
SHARED memory with the merge done by rank-by-counting (DEVIATION 566). Three
reasons, and the first is not a preference.

  * **LANE WIDTH.** `neighbors/impl/neighbors/topk/warp_topk.mojo` and
    `impl/matrix/detail/select_warpsort.mojo` both pin 32 lanes (the second
    says so at its `WARP_LANES`, "Theirs, pinned, and wrong on AMD").
    DEVIATION 515 in `registers.mojo` records what a 32-pinned warp
    primitive did to a 64-lane wavefront: it did not merely compute the
    wrong thing, IT DID NOT COMPILE, and it took `cluster/` and `dbscan/`
    down with it because one module is one codegen. This file uses no
    `vote`, no `shuffle_idx`, no `lane_id` and no warp reduction. Only
    `barrier()` and shared memory, which are the same on all three columns.
  * **CAPACITY.** Their queue's capacity is a compile-time parameter, so
    `k` costs a kernel instantiation per bucket. Here `k` is a runtime value
    bounded by `RBC_KNN_MAX_K`, and one kernel serves every `k`.
  * **DETERMINISM.** The merge is a total order over a fixed array, so the
    answer does not depend on which thread saw which candidate.

DEVIATION 559: THE PTOLEMAIC `z` BOUND IS NOT PORTED
------------------------------------------------------
THEIRS: inside the group walk, `registers.cuh:390-395` computes

    z = (|warpKTop - warpKTopRDist| * |warpKTopRDist - cur_candidate_dist|
         - warpKTop * cur_candidate_dist) / warpKTopRDist

guarded by `warpKTopRDist == 0` and then by `isnan(z) || isinf(z) ? 0`, and
skips the distance when `z > warpKTop`. It is a second, tighter lower bound
that also needs the queue to carry a SECOND key per entry (the candidate's
landmark distance), which is the whole reason their selector is a
`KeyValue` one rather than a plain one.

OURS: the plain two-sided triangle bound of test 3 above and nothing else.
It is exact, it needs no second key, no division, and no NaN/inf repair, and
dropping it removes the only division from the inner loop. What it costs is
distances that their bound would have skipped; what it buys is a selector
half the width and a bound whose correctness is one line. The number of
distances actually computed is reported by `dist_count` so the cost is
measurable rather than assumed. If a measurement later says the `z` bound
pays for itself, it lands here with that measurement attached.

DEVIATION 560: `D` COMES FROM THE SAME SELECTOR, NOT FROM A SECOND k-NN
-------------------------------------------------------------------------
THEIRS: `k_closest_landmarks` (`ball_cover.cuh:180-200`) builds a whole
`cuvs::neighbors::brute_force` index over the landmarks and runs a k-NN
search against it, materializing `R_knn_inds` and `R_knn_dists`, both
`n_query_pts x k`, which the kernel then reads back.

OURS: stage one of the same kernel runs the landmarks through the same
block selector, reads `D` out of slot `k - 1`, and RESETS the selector.
No second index, no second kernel, no `2 * k * n_queries` allocation. This
is DEVIATION 2's argument (the fused 1-NN) applied one level up, and it is
the same reason: the only thing wanted out of that search is one number per
query.

The landmarks are NOT left in the answer. They are index points and would
otherwise be correct entries, but every one of them is also visited again in
stage two as a member of its own group (a landmark is its own nearest
landmark at distance zero), so leaving them in would duplicate indices.

DEVIATION 561: ONE PASS, NO BITSET, NO POST-FILTER KERNEL
-----------------------------------------------------------
THEIRS: three kernels. `block_rbc_kernel_registers` visits only the k
closest landmarks; `perform_post_filter_registers` then builds an
`n_queries x ceil(n_landmarks/32)` BITSET marking which of the REMAINING
landmarks still have to be checked; `compute_final_dists_registers` walks
those. The split exists because their pass one is also usable ALONE as an
approximate query (`perform_post_filtering = false`, `ball_cover.cuh:246`).

OURS: one kernel that walks every landmark surviving tests 1 and 2. There
is no approximate mode to keep separate (DEVIATION 562), so there is
nothing for the bitset to communicate between passes, and the bitset's own
allocation and its two extra launches are removed. The set of landmarks
examined is exactly theirs: their pass one takes the k closest, their post
filter takes everything else that survives the same two tests, and the
union is "everything that survives the two tests".

DEVIATION 562: NO `weight`, BECAUSE THIS QUERY IS EXACT OR IT IS NOTHING
--------------------------------------------------------------------------
THEIRS: `rbc_knn_query` takes `float weight = 1.0` and multiplies the
landmark bound by it (`registers.cuh:363`, `:110`). At `weight < 1` the
prune is no longer implied by the triangle inequality and the answer
becomes approximate with no error bound and no recall reported. Combined
with `perform_post_filtering = false` it is their approximate mode.

OURS: neither parameter exists. `ROADMAP.md` rules out approximate search;
an index that returns a different answer than brute force is a different
algorithm with a different contract, and this library does not ship one.
`prune_scale` on the kernel below looks like `weight` and is NOT it: it is
pinned to 1.0 by the shipped entry point and exists only so
`neighbors/checks/ball_cover_knn_check.mojo` can TIGHTEN the bound and
require the exhaustive comparison to fail. A gate never shown capable of
failing does not count.

DEVIATION 563: `n_landmarks < k` DEGRADES, IT DOES NOT ASSERT
---------------------------------------------------------------
THEIRS: `ASSERT(index.n_landmarks >= k, ...)` (`ball_cover.cuh:459`), and
separately `ASSERT(index.n <= 3, "only 2d and 3d vectors are supported")`
(`:458`).

OURS: neither. `n_landmarks` is `floor(sqrt(m))`, so their assert refuses
`k > sqrt(m)` -- at m = 100 that is any k above 10, which is an ordinary
request. Here `D` is simply `+inf` when fewer than k landmarks exist, tests
1 and 2 then never fire, and the query degrades to an exact full scan
driven by `tau` alone. Slower, never wrong. The dimension assert is not
copied either: it exists upstream because their kernels stage the query row
in a `local_x_ptr[MAX_COL_Q]` register array with `MAX_COL_Q == 3`
(`registers.cuh:41`), which this file does not do -- it reads the query row
from global memory like the eps kernels do. High dimensions make the cover
prune LESS (that is geometry, and it is the same reason a kd-tree fails),
which is a speed statement and not a correctness one, so it is a note in
the host entry point rather than a refusal.

DEVIATION 566: THE MERGE IS RANK-BY-COUNTING OVER THE CONCATENATION
---------------------------------------------------------------------
Each batch of `RBC_KNN_TPB` candidates is concatenated after the `k` slots
of the running answer, and every element of the resulting `k + TPB` array
computes its own rank by counting the elements that precede it under the
total order, writing itself back into the answer when that rank is below
`k`. No sort network, no shared-memory bitonic ladder, no binary search:
one pass, one temporary, four barriers.

THIS IS THE SAME TRICK `rbc_rank_kernel` ALREADY USES ONE FILE OVER, for
the same reason DEVIATION 3 gives there: it needs no barrier inside itself,
it is deterministic where a sort would be unstable, and this toolchain's
`nn.argsort[target="gpu"]` is measured wrong above 256 elements.

ITS COST IS `(k + TPB)^2 / TPB` PER MERGE AND THAT IS NOT SMALL. At k = 32
and TPB = 128 it is 320 compares per thread. What makes it affordable is
the ADMISSION GUARD ahead of it: a batch in which no candidate beats the
current `tau` skips the merge entirely, at the price of one barrier and one
shared flag, and once `tau` is tight that is almost every batch. The merges
that do happen are bounded by how often the k-th best can improve, which is
`O(k log(n/k))` in expectation and not by the number of candidates. A
bitonic merge network would take this from `(k+TPB)^2/TPB` to
`(k+TPB) log^2(k+TPB) / TPB`; it is named here as the upgrade and is NOT
done, because it is a speed change and no arm of this lane has measured
speed yet. RUN OWED: `bench/` has no ball-cover k-NN row.

DEVIATION 567: THE PRUNING THRESHOLD CARRIES A NAMED FOUR-ULP SLACK
---------------------------------------------------------------------
The three bounds above are exact in real arithmetic. In float32 they are
not: `d(q, l)`, `d(l, y)` and `thresh` are each a rounded value, and a
prune decided at the last ulp can drop a point that the exact bound keeps.
A prune that is one ulp too TIGHT returns a WRONG ANSWER rather than a slow
one, and this repository has already paid once for a bound that looked
right.

So every threshold is widened before it is compared, by
`RBC_KNN_ULP_SLACK` (2^-21, four ulp of float32) times the magnitude of the
operands it is compared against:

    thresh_relaxed = thresh + (|d(q,l)| + radius(l)) * RBC_KNN_ULP_SLACK

`radius(l)` bounds every `d(l, y)` in the group, so one relaxed threshold
per landmark dominates the per-element one and the slack is paid once per
landmark rather than once per candidate. The widening can only ADMIT a
candidate the exact bound would have pruned, never drop one, so it costs
work and cannot cost an answer. Two arms of
`ball_cover_knn_check.mojo` measure exactly that: one runs with the slack
removed and requires the answer to be UNCHANGED, and one sweeps
`prune_scale` downward and requires the answer to BREAK.

**THIS IS WHY A ONE-ULP SABOTAGE CANNOT MOVE THIS KERNEL, AND SAYING SO IS
PART OF THE GATE.** The shipped bound is already four ulp looser than the
mathematical one, so a one-ulp tightening lands inside the slack and
changes nothing. The gate therefore sweeps the tightening downward, asserts
that the answer breaks, and PRINTS the largest `prune_scale` at which it
breaks -- that number, not a one-ulp claim, is the honest measure of how
close to vacuous the pruning gate is.

NOT PORTED
----------
`rbc_all_knn_query` (`ball_cover.cuh:384-437`), which is this query with
the index as its own query set and the index built inside the call. It is
two lines of host code over what is here and it would need its own
self-neighbour policy decision (cuVS keeps the self edge; scikit-learn's
`kneighbors(X=None)` drops it); recorded in `neighbors/NOT_IMPLEMENTED.tsv`
rather than guessed at.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from neighbors.impl.neighbors.ball_cover.common import (
    RBC_FLT_MAX,
    RBC_METRIC_DEFAULT,
    rbc_cmp_dist,
    rbc_true_dist,
    rbc_validate_metric,
)


#: Threads per block, one query per block. NOT a lane width: nothing in
#: this file votes, shuffles or reduces across a warp (DEVIATION 558), so
#: this is free to be any multiple of the largest wavefront in the matrix.
#: 128 is four 32-lane warps or two 64-lane wavefronts.
comptime RBC_KNN_TPB = 128

#: The largest `k` one block can answer. Sets the shared-memory footprint
#: with `RBC_KNN_TPB`: `(2 * MAX_K + 2 * (MAX_K + TPB)) * 4` bytes for the
#: answer and the concatenation, plus `2 * TPB * 4` for the landmark cache
#: and the distance counter, which at these values is 3.6 KB and fits the
#: 32 KB Metal wall with room to spare (`fused_l2_knn.mojo` records that
#: wall). A larger `k` is refused BY NAME at the host entry, with this
#: number in the message, rather than silently truncated.
comptime RBC_KNN_MAX_K = 128

#: The empty slot. `RBC_FLT_MAX` is the file-wide "further than anything"
#: sentinel (`common.mojo`); the payload is the largest `UInt32` so that a
#: sentinel sorts after every real (distance, index) pair under the total
#: order, and so that a short answer is recognisable on the host.
comptime RBC_KNN_SENTINEL_V = UInt32(0xFFFFFFFF)

#: DEVIATION 567. 2^-21, four ulp of float32, as a relative widening of
#: every pruning threshold.
comptime RBC_KNN_ULP_SLACK = Float32(4.76837158203125e-07)


@always_inline
def rbc_knn_relax(t: Float32, mag: Float32) -> Float32:
    """DEVIATION 567: widen a threshold so float rounding can only admit.

    One multiply and one add, both `ftz`'d, so the widened threshold is the
    same bits on every column. `mag` is the largest magnitude the threshold
    will be compared against; for a landmark that is `d(q, l) + radius(l)`,
    which dominates every `|d(q,l) - d(l,y)|` in its group.
    """
    return ftz(t + ftz(mag * RBC_KNN_ULP_SLACK))


@always_inline
def rbc_knn_before(
    ka: Float32, va: UInt32, pa: Int, kb: Float32, vb: UInt32, pb: Int
) -> Bool:
    """The total order, and the ONLY place it is written.

    `(distance, index)` ascending, with the ARRAY POSITION as the final
    tie-break. The position tie-break is not cosmetic: the sentinel pair
    occupies many slots at once, so without it the ranks below would not be
    a permutation and two elements would collide on one output slot.
    Real entries never collide -- a point belongs to exactly one landmark
    group and is offered exactly once -- so the position clause only ever
    orders sentinels against each other.
    """
    if ka < kb:
        return True
    if ka > kb:
        return False
    if va < vb:
        return True
    if va > vb:
        return False
    return pa < pb


@always_inline
def rbc_knn_merge(
    s_ak: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    s_av: UnsafePointer[
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    s_ck: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    s_cv: UnsafePointer[
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    s_flag: UnsafePointer[
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    k: Int,
    tid: Int,
    key: Float32,
    val: UInt32,
):
    """Fold one block-wide batch of candidates into the sorted answer.

    DEVIATION 566. **EVERY THREAD OF THE BLOCK MUST REACH THIS CALL THE
    SAME NUMBER OF TIMES**: it contains four barriers and one block-uniform
    early return. Every call site below is inside a loop whose bounds and
    whose `break` conditions are block-uniform, which is what makes that
    contract hold; a per-lane `continue` around this call would hang the
    block.

    On entry `s_ak[0 .. k)` is sorted best-first under `rbc_knn_before`.
    On exit it is the best `k` of the union of that array and the `TPB`
    offered pairs, still sorted. A thread with nothing to offer passes
    `RBC_KNN_SENTINEL_V`.
    """
    barrier()
    if tid == 0:
        s_flag[0] = Int32(0)
    barrier()

    # THE ADMISSION GUARD, and it is what makes the quadratic merge
    # affordable (see DEVIATION 566). A batch none of whose candidates can
    # beat the current cut cannot change the answer, so the merge is
    # skipped. `<=` and not `<`: a candidate that TIES with the k-th can
    # still displace it on the index tie-break, and being too permissive
    # costs one merge where being too strict loses an answer.
    var tau = s_ak[k - 1]
    var mine = key
    var mine_v = val
    if mine_v == RBC_KNN_SENTINEL_V:
        mine = RBC_FLT_MAX
    elif mine <= tau:
        s_flag[0] = Int32(1)
    barrier()
    if s_flag[0] == Int32(0):
        return

    # The concatenation: the answer first, this batch after it. The array
    # position tie-break in `rbc_knn_before` is what makes "answer first"
    # meaningful, and it is the STABLE choice -- an incumbent sentinel
    # cannot be displaced by an offered sentinel and change the payload.
    var n = k + RBC_KNN_TPB
    if tid < k:
        s_ck[tid] = s_ak[tid]
        s_cv[tid] = s_av[tid]
    s_ck[k + tid] = mine
    s_cv[k + tid] = mine_v
    barrier()

    var p = tid
    while p < n:
        var kp = s_ck[p]
        var vp = s_cv[p]
        var rank = 0
        for q in range(n):
            if rbc_knn_before(s_ck[q], s_cv[q], q, kp, vp, p):
                rank += 1
        if rank < k:
            s_ak[rank] = kp
            s_av[rank] = vp
        p += RBC_KNN_TPB
    barrier()


def rbc_knn_kernel(
    x_reordered: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    r_radius: MutPointer[Float32, MutAnyOrigin],
    out_inds: MutPointer[Int32, MutAnyOrigin],
    out_dists: MutPointer[Float32, MutAnyOrigin],
    dist_count: MutPointer[Int32, MutAnyOrigin],
    n_queries_in: Int32,
    n_cols_in: Int32,
    n_landmarks_in: Int32,
    k_in: Int32,
    metric_in: Int32,
    metric_arg_in: Float32,
    prune_scale_in: Float32,
):
    """One query per block. `block_rbc_kernel_registers` + their post
    filter + `compute_final_dists_registers`, folded into one pass.

    `out_inds` is `n_queries x k` and holds the ORIGINAL index-point ids,
    ascending under the total order at the top of this file; `out_dists` is
    the same shape and holds TRUE distances (`rbc_true_dist` of the
    comparison-space key the order was taken on). A slot with no neighbour
    -- only reachable when `k > m` -- gets `-1` and `RBC_FLT_MAX`.

    `dist_count[query_id]` is how many candidate distances this query
    actually computed. It is cuVS's `n_dists_computed` (`:409`), which they
    increment and never read; here it is read, because it is the only
    direct evidence that the index pruned anything at all and it is what
    `check_rbc_knn_prunes_work` asserts against `n_queries * m`.

    `prune_scale_in` is 1.0 in every shipped path. See DEVIATION 562.
    """
    var n_queries = Int(n_queries_in)
    var n_cols = Int(n_cols_in)
    var n_landmarks = Int(n_landmarks_in)
    var k = Int(k_in)
    var metric = Int(metric_in)
    var metric_arg = metric_arg_in
    var prune_scale = prune_scale_in

    var query_id = Int(block_idx.x)
    if query_id >= n_queries:
        return
    var tid = Int(thread_idx.x)
    var x_base = n_cols * query_id

    var s_ak = stack_allocation[
        RBC_KNN_MAX_K,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_av = stack_allocation[
        RBC_KNN_MAX_K,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_ck = stack_allocation[
        RBC_KNN_MAX_K + RBC_KNN_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_cv = stack_allocation[
        RBC_KNN_MAX_K + RBC_KNN_TPB,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_ld = stack_allocation[
        RBC_KNN_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_cnt = stack_allocation[
        RBC_KNN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_flag = stack_allocation[
        1,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var my_dists = 0

    # -- the answer starts empty -------------------------------------------
    var t = tid
    while t < k:
        s_ak[t] = RBC_FLT_MAX
        s_av[t] = RBC_KNN_SENTINEL_V
        t += RBC_KNN_TPB
    barrier()

    # -- STAGE ONE: D, the k-th nearest LANDMARK distance ------------------
    # DEVIATION 560. The landmarks go through the same selector; only slot
    # `k - 1` is kept, and the selector is reset before stage two.
    for l0 in range(0, n_landmarks, RBC_KNN_TPB):
        var l = l0 + tid
        var key = RBC_FLT_MAX
        var val = RBC_KNN_SENTINEL_V
        if l < n_landmarks:
            key = rbc_cmp_dist(
                x, x_base, r, l * n_cols, n_cols, metric, metric_arg
            )
            val = UInt32(l)
        rbc_knn_merge(s_ak, s_av, s_ck, s_cv, s_flag, k, tid, key, val)

    # `D` in TRUE distance space: bounds 1, 2 and 3 are all statements
    # about true distances, which is the whole of DEVIATION 564.
    var d_bound = rbc_true_dist(metric, s_ak[k - 1])
    var three_d = Float32(3.0) * d_bound
    barrier()

    t = tid
    while t < k:
        s_ak[t] = RBC_FLT_MAX
        s_av[t] = RBC_KNN_SENTINEL_V
        t += RBC_KNN_TPB
    barrier()

    # -- STAGE TWO: walk the cover -----------------------------------------
    for l0 in range(0, n_landmarks, RBC_KNN_TPB):
        var lane_l = l0 + tid
        var lane_cmp = RBC_FLT_MAX
        if lane_l < n_landmarks:
            lane_cmp = rbc_cmp_dist(
                x, x_base, r, lane_l * n_cols, n_cols, metric, metric_arg
            )
        s_ld[tid] = lane_cmp
        barrier()

        var j_max = n_landmarks - l0
        if j_max > RBC_KNN_TPB:
            j_max = RBC_KNN_TPB

        for j in range(j_max):
            var lm = l0 + j
            var dl = rbc_true_dist(metric, s_ld[j])
            var radius = r_radius.unsafe_load(lm)

            # bound 2, Cayton. `D` and NOT `tau`; see the header.
            if dl > rbc_knn_relax(three_d * prune_scale, three_d):
                continue

            # bound 1. `thresh` is `min(tau, D)`, both of which are upper
            # bounds on d_k(q); `tau` is re-read here because it may have
            # tightened while the previous landmarks were walked.
            var tau = rbc_true_dist(metric, s_ak[k - 1])
            var thresh = tau
            if d_bound < thresh:
                thresh = d_bound
            var mag = ftz(dl + radius)
            if dl - radius > rbc_knn_relax(thresh * prune_scale, mag):
                continue

            var r_start = Int(r_indptr.unsafe_load(lm))
            var r_end = Int(r_indptr.unsafe_load(lm + 1))

            var i0 = r_start
            while i0 < r_end:
                # `tau` shrinks monotonically, so a threshold re-read at
                # the top of every batch keeps the high-side stop valid:
                # a later batch is compared against a threshold no larger
                # than the one that already failed to stop it.
                tau = rbc_true_dist(metric, s_ak[k - 1])
                thresh = tau
                if d_bound < thresh:
                    thresh = d_bound
                var th = rbc_knn_relax(thresh * prune_scale, mag)

                # bound 3, the high side. The group is ascending in
                # `d(l, y)`, so if the FIRST element of this batch is out
                # of reach so is every element after it, in this group.
                if r_1nn_dists.unsafe_load(i0) - dl > th:
                    break

                var i = i0 + tid
                var key = RBC_FLT_MAX
                var val = RBC_KNN_SENTINEL_V
                if i < r_end:
                    var dly = r_1nn_dists.unsafe_load(i)
                    # bound 3, BOTH sides, written as two `<=` tests rather
                    # than as the negation of two `>` tests. Same answer on
                    # every finite input, and it is the spelling that keeps
                    # the two directions separately readable next to the
                    # `>` the break above uses. The low side SKIPS rather
                    # than stops: walking ascending, it stops firing as
                    # `d(l, y)` grows, so there is nothing monotone to exit
                    # on.
                    if dly - dl <= th:
                        if dl - dly <= th:
                            key = rbc_cmp_dist(
                                x,
                                x_base,
                                x_reordered,
                                i * n_cols,
                                n_cols,
                                metric,
                                metric_arg,
                            )
                            val = UInt32(r_1nn_cols.unsafe_load(i))
                            my_dists += 1
                rbc_knn_merge(
                    s_ak, s_av, s_ck, s_cv, s_flag, k, tid, key, val
                )
                i0 += RBC_KNN_TPB
        barrier()

    # -- the answer, and the work it took ----------------------------------
    s_cnt[tid] = Int32(my_dists)
    barrier()
    if tid == 0:
        var total = Int32(0)
        for e in range(RBC_KNN_TPB):
            total += s_cnt[e]
        dist_count.unsafe_store(query_id, total)

    var o = tid
    while o < k:
        var kv = s_ak[o]
        var vv = s_av[o]
        if vv == RBC_KNN_SENTINEL_V:
            out_inds.unsafe_store(query_id * k + o, Int32(-1))
            out_dists.unsafe_store(query_id * k + o, RBC_FLT_MAX)
        else:
            out_inds.unsafe_store(query_id * k + o, Int32(Int(vv)))
            out_dists.unsafe_store(
                query_id * k + o, rbc_true_dist(metric, kv)
            )
        o += RBC_KNN_TPB


def rbc_knn_query_scaled(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut out_inds: DeviceBuffer[DType.int32],
    mut out_dists: DeviceBuffer[DType.float32],
    mut dist_count: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    k: Int,
    prune_scale: Float32,
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises:
    """`rbc_knn_query` WITH THE PRUNING BOUND SCALABLE. Gates only.

    `prune_scale < 1` tightens every bound past what the triangle
    inequality justifies and therefore returns a WRONG answer on purpose.
    It exists so `ball_cover_knn_check.mojo` can show that the exhaustive
    comparison is capable of failing; the shipped entry point below pins it
    to 1.0. See DEVIATION 562 for why this is not cuVS's `weight`.
    """
    rbc_validate_metric(metric, metric_arg)
    if k < 1:
        raise Error(
            "rbc_knn_query: k must be at least 1, got " + String(k)
        )
    if k > RBC_KNN_MAX_K:
        raise Error(
            "rbc_knn_query: k = "
            + String(k)
            + " exceeds RBC_KNN_MAX_K = "
            + String(RBC_KNN_MAX_K)
            + ", which is what one block's shared answer array holds"
            " (neighbors/impl/neighbors/ball_cover/knn.mojo). This is"
            " refused rather than truncated: a truncated k-NN looks exactly"
            " like a complete one. Use NearestNeighbors, whose selector is"
            " sized per launch, or raise RBC_KNN_MAX_K and re-measure the"
            " shared-memory footprint against the 32 KB wall."
        )
    if n_queries <= 0 or n_cols <= 0 or n_landmarks <= 0:
        raise Error(
            "rbc_knn_query: n_queries, n_cols and n_landmarks must all be"
            " positive, got "
            + String(n_queries)
            + ", "
            + String(n_cols)
            + ", "
            + String(n_landmarks)
        )
    ctx.enqueue_function[rbc_knn_kernel](
        x_reordered.unsafe_ptr(),
        query.unsafe_ptr(),
        r.unsafe_ptr(),
        r_indptr.unsafe_ptr(),
        r_1nn_cols.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(),
        r_radius.unsafe_ptr(),
        out_inds.unsafe_ptr(),
        out_dists.unsafe_ptr(),
        dist_count.unsafe_ptr(),
        Int32(n_queries),
        Int32(n_cols),
        Int32(n_landmarks),
        Int32(k),
        Int32(metric),
        metric_arg,
        prune_scale,
        grid_dim=(n_queries, 1, 1),
        block_dim=(RBC_KNN_TPB, 1, 1),
    )
    ctx.synchronize()


def rbc_knn_query(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut out_inds: DeviceBuffer[DType.int32],
    mut out_dists: DeviceBuffer[DType.float32],
    mut dist_count: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    k: Int,
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises:
    """`cuvs::neighbors::ball_cover::rbc_knn_query`, `ball_cover.cuh:446`.

    The index buffers are exactly those `rbc_build_index` filled, and the
    index MUST have been built under the same `metric` and `metric_arg`;
    `r_1nn_dists` and `r_radius` are distances in that metric and comparing
    them against a query distance in another one is the single way to make
    the prune unsound without any line being wrong.

    `out_inds` and `out_dists` are `n_queries * k`, `dist_count` is
    `n_queries`.

    A NOTE ON DIMENSION, WHICH IS A SPEED STATEMENT AND NOT A LIMIT.
    cuVS asserts `index.n <= 3` here (`:458`) and cuML restricts the same
    index to three dimensions in its Python layer
    (`nearest_neighbors.pyx:439-441`). Their assert is about their register
    staging (DEVIATION 563), but the geometry behind the restriction is
    real: as the dimension grows, `radius(l)` approaches `d(q, l)` for
    every landmark, bounds 1 and 2 stop firing, and this degrades toward a
    full scan -- the same way a kd-tree does, and for the same reason. It
    degrades to something EXACT and merely slow, so it is not refused here;
    `dist_count` reports how much pruning actually happened, and a caller
    who sees no pruning should use brute force.
    """
    rbc_knn_query_scaled(
        ctx,
        x_reordered,
        query,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        out_inds,
        out_dists,
        dist_count,
        n_queries,
        n_cols,
        n_landmarks,
        k,
        Float32(1.0),
        metric,
        metric_arg,
    )
