# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cluster stabilities, and the two reductions that decide them.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/stabilities.cuh`
(cuML `265b9da`): `compute_stabilities` (`:49-137`) and
`get_stability_scores` (`:153-200`), plus
`detail/kernels/stabilities.cuh::stabilities_functor` (`:22-49`).
Transliterated, their order, with two declared replacements.

======================================================================
THE STABILITY SUM IS A SUMMATION ORDER. (IDENTITY hazard 3, second half.)
======================================================================
Their `stabilities_functor::operator()` (`kernels/stabilities.cuh:39-44`):

    auto parent = parents[idx] - n_leaves;
    atomicAdd(&stabilities[parent], (lambdas[idx] - births[parent]) * sizes[idx]);

one thread per CONDENSED EDGE, a FLOAT `atomicAdd` into a per-cluster
cell. Every edge of one cluster lands in one accumulator in ARRIVAL
ORDER, which is not reproducible run to run on one device, let alone
across three. `IDENTITY_PATHS`' opening rule allows PIN, REPLACE or
REFUSE and nothing else; this is REPLACE.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1603. THE STABILITY SUM IS A PER-CLUSTER
SERIAL FOLD IN CONDENSED-TREE ORDER, NOT A FLOAT `atomicAdd`.
======================================================================
WHAT THEIRS DOES: the block above. One float atomic per condensed edge.

WHY IT CANNOT BE PORTED AS-IS. A float `atomicAdd` is order-dependent by
construction and the order is the scheduler's. It is IDENTITY_PATHS rows
1, 8 and 36's defect, and this lane may not reintroduce it. It is also
BANNED OUTRIGHT on this path by the lane's brief: no floating-point
atomic anywhere that reaches an output.

WHAT OURS DOES. `compute_stabilities` builds the SAME CSR index over
sorted parents their own code builds (`Utils::parent_csr`,
`stabilities.cuh:69`) and then runs ONE THREAD PER CLUSTER, walking that
cluster's contiguous segment ASCENDING and accumulating through
`ftz`/`identical_mul_add`. There is no atomic, no lane primitive, no
block fold and no cross-thread communication of any kind: cluster `c`'s
total is written by thread `c` alone.

WHY THE SEGMENT ORDER IS A TOTAL ORDER AND NOT A CONVENTION. The
condensed tree is sorted by `(parent, child)` (their `TupleComp`,
`condensed_hierarchy.cu:34-49`; DEVIATION 1611 for the spelling), and
every node of a tree has exactly one parent, so `child` is unique across
the whole array. Within a segment the edges are therefore in strictly
increasing `child` order with no ties possible. The fold order is a pure
function of the condensed tree, which is a pure function of the
dendrogram, which is a pure function of the mutual reachability bytes.

WHAT IT COSTS. Their kernel is `n_edges` threads; ours is `n_clusters`
threads with the longest segment on the critical path. NO TIMING WAS
TAKEN and none is claimed. The shape was chosen because it is the
simplest thing with no order in it, which is `pinned_distance_tile`'s
argument one directory over.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1604. THE PER-PARENT MINIMUM LAMBDA IS A
TOTAL-ORDER SCAN, NOT `cub::DeviceSegmentedReduce::Min`.
======================================================================
WHAT THEIRS DOES. `Utils::cub_segmented_reduce(lambdas,
births_parent_min.data() + 1, n_clusters - 1, offsets + 1, stream, Min)`
(`stabilities.cuh:109-114`), a CUB segmented reduction whose internal
fold shape is CUB's choice, followed by a `thrust::transform` taking
`birth < births_parent_min ? birth : births_parent_min` (`:119-126`).

WHY IT CANNOT BE PORTED AS-IS. Two reasons, and the second is the one a
reader will not guess. (a) The fold SHAPE is a library's and varies with
the segment length and the target; a min over floats is associative and
commutative EXCEPT on a `(+0.0, -0.0)` pair, where IDENTITY_PATHS row 39
measured `min(+0, -0)` as `-0.0` on all three columns but `min(-0, +0)`
as `-0.0` on NVIDIA and AMD and `+0.0` on APPLE -- so the answer depends
on which operand the fold happened to put first. (b) An EMPTY segment:
CUB's Min identity is the type's max, and their `births_parent_min[0]` is
never written at all (`:110` starts the output at `+1`) and never read
(`:126` starts the transform at `+1`), so its `rmm::device_uvector`
contents are uninitialized memory that the code is careful not to touch.

WHAT OURS DOES. A serial ascending scan of the same segment, comparing
`hierarchy/original/edge_order.mojo::weight_order_key` -- the INTEGER
total order the MST already uses in this same fit, imported and not
re-derived -- so `-0.0` orders strictly below `+0.0` on every vendor and
no hardware `min` appears. The empty-segment identity is written
EXPLICITLY as `FLT_MAX`, which is what CUB's Min identity is, so the
subsequent `min(birth, seg_min)` is a no-op exactly as theirs is; and
index 0 is skipped in the same place theirs skips it, rather than left
uninitialized.

CAN A LAMBDA BE `-0.0`? Only through `1 / distance` with a distance whose
reciprocal underflows to a signed zero, which needs an infinite distance,
which DEVIATION 1607 refuses upstream of here. The pin is inert on the
default path and is kept for the reason `hierarchy`'s key is:
`hdbscan_check.mojo` plants the value directly and the gate then has
teeth.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.original.hdbscan_sabotage import (
    HDB_SAB_NONE,
    HDB_SAB_STABILITY_DESCENDING,
)
from hdbscan.derived.hdbscan.condensed_hierarchy import CondensedHierarchy
from hdbscan.derived.hdbscan.detail.utils import utils_parent_csr
from hierarchy.original.edge_order import weight_order_key
from hierarchy.derived.cluster.detail.connectivities import FLOAT32_MAX
from original.numerics import ftz, identical_div, identical_mul, identical_mul_add


comptime STAB_TPB = 256
"""SCHEDULING. One thread per cluster and one thread per edge; neither
kernel folds across threads, so the block size cannot reach a value.
`check_hdbscan_launch_invariance` varies it."""


def births_init_kernel(
    births: MutPointer[Float32, MutAnyOrigin],
    children: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    n_leaves_in: Int32,
    n_edges_in: Int32,
):
    """`stabilities.cuh:76-80` `births_init_op`, one thread per condensed
    edge:

        auto child = children[idx];
        if (child >= n_leaves) { births[child - n_leaves] = lambdas[idx]; }

    "This is to consider the case where a child may also be a parent, in
    which case births for that parent are initialized to lambda for that
    child" (`:71-73`).

    NOT A RACE, and it is worth saying why rather than trusting it: every
    node of a tree has exactly one parent, so each cluster appears as a
    CHILD at most once across the whole array and each `births` cell is
    written by at most one thread. The store is therefore order-free
    without an atomic.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_edges_in):
        return
    var child = Int(children.unsafe_load(idx))
    if child >= Int(n_leaves_in):
        births.unsafe_store(child - Int(n_leaves_in), lambdas.unsafe_load(idx))


def cluster_stability_kernel(
    stabilities: MutPointer[Float32, MutAnyOrigin],
    births: MutPointer[Float32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    sizes: MutPointer[Int32, MutAnyOrigin],
    n_clusters_in: Int32,
    sabotage: Int32,
):
    """DEVIATIONS 1603 and 1604, one thread per cluster.

    Three of their steps in one kernel, in their order:

      `:109-114`  the per-parent minimum lambda over the segment
      `:117-126`  `births[c] = min(births[c], births_parent_min[c])`,
                  for `c >= 1` only
      `:131-136`  `stability[c] = sum over segment of
                  (lambda - births[c]) * size`

    They are fused because each thread's three steps read and write only
    ITS OWN cluster's cells, so no barrier and no second launch is needed
    -- and because splitting them would put `births` in device memory
    between two kernels for no reason a reader could act on.
    """
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_clusters_in):
        return
    var lo = Int(indptr.unsafe_load(c))
    var hi = Int(indptr.unsafe_load(c + 1))

    # `:109-114` the segmented Min, DEVIATION 1604. FLT_MAX is CUB's Min
    # identity and therefore the empty-segment answer.
    var seg_min = FLOAT32_MAX
    for i in range(lo, hi):
        var lam = lambdas.unsafe_load(i)
        if weight_order_key(lam) < weight_order_key(seg_min):
            seg_min = lam

    # `:117-126` their transform runs over indices 1 .. n_clusters-1, so
    # cluster 0 (the root) keeps the 0.0 the fill gave it.
    var birth = births.unsafe_load(c)
    if c > 0:
        if weight_order_key(seg_min) < weight_order_key(birth):
            birth = seg_min
        births.unsafe_store(c, birth)

    # `:131-136` the stability sum, DEVIATION 1603. Ascending through the
    # segment; the descending arm is the sabotage.
    var acc = Float32(0.0)
    if sabotage == HDB_SAB_STABILITY_DESCENDING:
        for t in range(hi - lo):
            var i = hi - 1 - t
            var term = ftz(lambdas.unsafe_load(i) - birth)
            acc = ftz(
                identical_mul_add(
                    term, Float32(Int(sizes.unsafe_load(i))), acc
                )
            )
    else:
        for i in range(lo, hi):
            var term = ftz(lambdas.unsafe_load(i) - birth)
            acc = ftz(
                identical_mul_add(
                    term, Float32(Int(sizes.unsafe_load(i))), acc
                )
            )
    stabilities.unsafe_store(c, acc)


def compute_stabilities(
    ctx: DeviceContext,
    tree: CondensedHierarchy,
    mut stabilities: DeviceBuffer[DType.float32],
    stab_tpb: Int = STAB_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`stabilities.cuh:49-137`. `stabilities` is `n_clusters` long."""
    var n_clusters = tree.n_clusters
    var n_edges = tree.n_edges
    if n_clusters < 1:
        raise Error(
            "hdbscan.compute_stabilities: n_clusters=" + String(n_clusters)
            + " < 1; the condensed tree has no cluster to score"
        )

    # `:65-69` sorted_parents + Utils::parent_csr. Ours reads the tree's
    # own parents without copying (see `utils_parent_csr`'s docstring).
    var indptr_h = utils_parent_csr(tree)

    var d_indptr = ctx.enqueue_create_buffer[DType.int32](n_clusters + 1)
    var d_children = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var d_lambdas = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var d_sizes = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var births = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.synchronize()
    var h_children = tree.children.copy()
    var h_lambdas = tree.lambdas.copy()
    var h_sizes = tree.sizes.copy()
    var h_indptr = indptr_h.copy()
    ctx.enqueue_copy(dst_buf=d_indptr, src_ptr=h_indptr.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_children, src_ptr=h_children.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_lambdas, src_ptr=h_lambdas.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_sizes, src_ptr=h_sizes.unsafe_ptr())
    ctx.synchronize()

    # `:74-75` thrust::fill(births, 0.0f)
    ctx.enqueue_memset(births, Float32(0.0))
    ctx.enqueue_function[births_init_kernel](
        births.unsafe_ptr(),
        d_children.unsafe_ptr(),
        d_lambdas.unsafe_ptr(),
        Int32(tree.n_leaves),
        Int32(n_edges),
        grid_dim=((n_edges + stab_tpb - 1) // stab_tpb if n_edges > 0 else 1, 1, 1),
        block_dim=(stab_tpb, 1, 1),
    )
    # `:128` thrust::fill(stabilities, 0.0f). Kept even though every cell
    # is written below, because a cluster with an EMPTY segment must come
    # out 0.0 rather than whatever the allocation held -- which is what
    # their fill is for.
    ctx.enqueue_memset(stabilities, Float32(0.0))
    ctx.enqueue_function[cluster_stability_kernel](
        stabilities.unsafe_ptr(),
        births.unsafe_ptr(),
        d_indptr.unsafe_ptr(),
        d_lambdas.unsafe_ptr(),
        d_sizes.unsafe_ptr(),
        Int32(n_clusters),
        sabotage,
        grid_dim=((n_clusters + stab_tpb - 1) // stab_tpb, 1, 1),
        block_dim=(stab_tpb, 1, 1),
    )
    ctx.synchronize()
    _ = d_indptr^
    _ = d_children^
    _ = d_lambdas^
    _ = d_sizes^
    _ = births^
    _ = h_children^
    _ = h_lambdas^
    _ = h_sizes^
    _ = h_indptr^


def max_lambda_of(tree: CondensedHierarchy) raises -> Float32:
    """`runner.h:208-210`:

        value_t max_lambda = *(thrust::max_element(exec_policy, lambdas_ptr,
                                lambdas_ptr + condensed_tree.get_n_edges()));

    A max over floats, so IDENTITY_PATHS row 39 applies: taken on the
    `weight_order_key` INTEGER order rather than a hardware `max`, for the
    same reason DEVIATION 1604's min is. `thrust::max_element` returns the
    FIRST maximal element; with a total order there is only one maximal
    VALUE, and the value is all the caller reads.
    """
    if tree.n_edges < 1:
        raise Error(
            "hdbscan.max_lambda_of: the condensed tree has no edges; their"
            " thrust::max_element at runner.h:209 dereferences an empty"
            " range here"
        )
    var best = tree.lambdas[0]
    for i in range(1, tree.n_edges):
        if weight_order_key(tree.lambdas[i]) > weight_order_key(best):
            best = tree.lambdas[i]
    return best


def get_stability_scores(
    labels: List[Int32],
    stability: List[Float32],
    n_condensed_clusters: Int,
    max_lambda: Float32,
    n_leaves: Int,
    label_map: List[Int32],
    n_selected: Int,
) raises -> List[Float32]:
    """`stabilities.cuh:153-200`, on the host.

    WHERE IT RUNS. Theirs is two `thrust::for_each`es on the device: an
    INTEGER `atomicAdd` per point into `cluster_sizes` (`:173-175`, exact
    and order-free) and a per-cluster elementwise epilogue (`:183-199`).
    Ours is the same two passes serially on the host, because `labels` is
    already a host array by the time this is called (`do_labelling_on_host`
    produces it, `extract.cuh:89-167`, and it is theirs that puts it
    there). No fold order changes: a count is exact and each output cell
    is written by one pass over one cluster.

    THE EPILOGUE, unchanged (`:191-198`):

        if (out_cluster >= 0) {
          bool expr = max_lambda == FLT_MAX || max_lambda == 0.0 || size == 0;
          result[out_cluster] = expr ? 1.0f : stability[c] / (size * max_lambda);
        }

    with the product and the quotient through `identical_mul` /
    `identical_div` (rows 9 and 49's seams). The three-way guard is theirs
    and is what keeps a `0/0` out of the result.
    """
    # `:167-175` populate cluster sizes
    var cluster_sizes = List[Int](capacity=n_condensed_clusters)
    for _ in range(n_condensed_clusters):
        cluster_sizes.append(0)
    for i in range(n_leaves):
        var v = Int(labels[i])
        if v > -1:
            if v >= n_condensed_clusters:
                raise Error(
                    "hdbscan.get_stability_scores: label " + String(v)
                    + " at point " + String(i) + " is outside [0, "
                    + String(n_condensed_clusters) + "); their atomicAdd at"
                    " stabilities.cuh:174 would write past cluster_sizes"
                )
            cluster_sizes[v] += 1

    var result = List[Float32](capacity=n_selected)
    for _ in range(n_selected):
        result.append(Float32(0.0))
    # `:181-199`
    for c in range(n_condensed_clusters):
        var out_cluster = Int(label_map[c])
        if out_cluster < 0:
            continue
        var size = cluster_sizes[c]
        var expr = (
            max_lambda == FLOAT32_MAX
            or max_lambda == Float32(0.0)
            or size == 0
        )
        if expr:
            result[out_cluster] = Float32(1.0)
        else:
            result[out_cluster] = identical_div(
                stability[c], identical_mul(Float32(size), max_lambda)
            )
    return result^
