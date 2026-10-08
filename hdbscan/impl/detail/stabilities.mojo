# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cluster stabilities, and the two reductions that decide them.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/stabilities.cuh`
(cuML `265b9da`): `compute_stabilities` (`:49-137`) and
`get_stability_scores` (`:153-200`), plus
`detail/kernels/stabilities.cuh::stabilities_functor` (`:22-49`).
Steps run in the reference order, with two declared replacements.

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

WHY IT CANNOT BE IMPLEMENTED AS-IS. A float `atomicAdd` is order-dependent by
construction and the order is the scheduler's. It is IDENTITY_PATHS rows
1, 8 and 36's defect, and this lane may not reintroduce it. It is also
BANNED OUTRIGHT on this path by the lane's brief: no floating-point
atomic anywhere that reaches an output.

WHAT OURS DOES. `compute_stabilities` reads the SAME CSR index over
sorted parents their own code builds (`Utils::parent_csr`,
`stabilities.cuh:69`; the device tree's `indptr`) and runs ONE BLOCK PER
CLUSTER of `STAB_FOLD` threads: thread t folds the segment positions t,
t + STAB_FOLD, ... ascending through `ftz`/`identical_mul_add`, and the
`STAB_FOLD` partials are combined by a fixed pairwise tree (stride 128,
64, ..., 1). No atomic and no order the scheduler picks: the fold shape is
a function of the segment length alone, the same on every vendor and in
the host column (`stability_fold_host`, which `hdbh_stabilities` calls).
Until 2026-10-03 this was one thread per cluster walking its whole segment,
a serial chain as long as the root's leaf count; the blocked fold replaced
it on every column at once (lane cgr2-hdbscan), so the bits moved
everywhere together.

WHY THE SEGMENT ORDER IS A TOTAL ORDER AND NOT A CONVENTION. The
condensed tree is sorted by `(parent, child)` (their `TupleComp`,
`condensed_hierarchy.cu:34-49`; DEVIATION 1611 for the spelling), and
every node of a tree has exactly one parent, so `child` is unique across
the whole array. Within a segment the edges are therefore in strictly
increasing `child` order with no ties possible. The fold order is a pure
function of the condensed tree, which is a pure function of the
dendrogram, which is a pure function of the mutual reachability bytes.

WHAT IT COSTS. Their kernel is `n_edges` threads; ours is `n_clusters`
blocks. NO TIMING WAS TAKEN and none is claimed.
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

WHY IT CANNOT BE IMPLEMENTED AS-IS. Two reasons, and the second is the one a
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
`hierarchy/checks/edge_order.mojo::weight_order_key` -- the INTEGER
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

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from hdbscan.checks.hdbscan_sabotage import (
    HDB_SAB_NONE,
    HDB_SAB_STABILITY_DESCENDING,
)
from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hdbscan.impl.detail.tree_device import DeviceTree, td_download_f32
from hierarchy.checks.edge_order import (
    WEIGHT_KEY_NAN,
    weight_order_key,
    weight_order_unkey,
)
from hierarchy.impl.cluster.detail.connectivities import FLOAT32_MAX
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from hdbscan.impl.detail.stability_fold import (
    STAB_FOLD,
    stability_fold_host,
    stability_order_key,
    stability_order_key_bits,
    stability_order_unkey_bits,
)


comptime STAB_TPB = 256
"""SCHEDULING for the per-edge and per-cluster launches. The stability
fold's block is `STAB_FOLD`, which is part of the order and is not a
tuning knob."""



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
    """DEVIATIONS 1603 and 1604, one BLOCK per cluster (`STAB_FOLD`).

    Their three steps in their order, each over the cluster's segment:

      `:109-114`  the per-parent minimum lambda: each thread's strided
                  minimum KEY, then a tree minimum over the keys (an
                  integer minimum, so the grouping cannot move it)
      `:117-126`  `births[c] = min(births[c], births_parent_min[c])`, for
                  `c >= 1` only, by thread 0, selected as UInt32 bits
      `:131-136`  `stability[c] = sum over segment of (lambda - births[c])
                  * size`: each thread's strided serial fold through
                  `ftz`/`identical_mul_add`, then the fixed pairwise tree
                  `p[t] = ftz(identical_mul_add(1, p[t + s], p[t]))`

    No float is selected by a compare and no float is compared (the gfx942
    instruction-selector crash of 2026-09-14 was a float selected under an
    integer key test); the minimum travels as its key and is turned back
    into its bits once.
    """
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= Int(n_clusters_in):
        return
    var keys = stack_allocation[
        STAB_FOLD, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var part = stack_allocation[
        STAB_FOLD, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var bsh = stack_allocation[
        1, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var lo = Int(indptr.unsafe_load(c))
    var hi = Int(indptr.unsafe_load(c + 1))
    var n_seg = hi - lo

    # `:109-114` the segmented Min, DEVIATION 1604. FLT_MAX is CUB's Min
    # identity and therefore the empty-segment answer.
    var seg_key = stability_order_key_bits(bitcast[DType.uint32](FLOAT32_MAX))
    var k = t
    while k < n_seg:
        var lam_key = stability_order_key_bits(
            bitcast[DType.uint32](lambdas.unsafe_load(lo + k))
        )
        if lam_key < seg_key:
            seg_key = lam_key
        k += STAB_FOLD
    keys[t] = seg_key
    barrier()
    var s = STAB_FOLD // 2
    while s > 0:
        if t < s:
            var o = keys[t + s]
            if o < keys[t]:
                keys[t] = o
        barrier()
        s //= 2

    # `:117-126` their transform runs over indices 1 .. n_clusters-1, so
    # cluster 0 (the root) keeps the 0.0 the fill gave it.
    if t == 0:
        var birth_bits = bitcast[DType.uint32](births.unsafe_load(c))
        if c > 0:
            var mk = keys[0]
            if mk < stability_order_key_bits(birth_bits):
                birth_bits = stability_order_unkey_bits(mk)
            births.unsafe_store(c, bitcast[DType.float32](birth_bits))
        bsh[0] = birth_bits
    barrier()
    var birth = bitcast[DType.float32](bsh[0])

    # `:131-136` the stability sum, DEVIATION 1603. The descending arm (the
    # sabotage) reads the segment back to front at every position.
    var descending = sabotage == HDB_SAB_STABILITY_DESCENDING
    var acc = Float32(0.0)
    k = t
    while k < n_seg:
        var i = lo + k
        if descending:
            i = hi - 1 - k
        var term = ftz(lambdas.unsafe_load(i) - birth)
        var size_f = sizes.unsafe_load(i).cast[DType.float32]()
        acc = ftz(identical_mul_add(term, size_f, acc))
        k += STAB_FOLD
    part[t] = acc
    barrier()
    s = STAB_FOLD // 2
    while s > 0:
        if t < s:
            part[t] = ftz(identical_mul_add(Float32(1.0), part[t + s], part[t]))
        barrier()
        s //= 2
    if t == 0:
        stabilities.unsafe_store(c, part[0])


def compute_stabilities(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut stabilities: DeviceBuffer[DType.float32],
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`stabilities.cuh:49-137` over the device tree (its parent CSR is
    `tree.indptr`). `stabilities` is `n_clusters` long."""
    var n_clusters = tree.n_clusters
    var n_edges = tree.n_edges
    if n_clusters < 1:
        raise Error(
            "hdbscan.compute_stabilities: n_clusters=" + String(n_clusters)
            + " < 1; the condensed tree has no cluster to score"
        )
    var births = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    # `:74-75` thrust::fill(births, 0.0f)
    ctx.enqueue_memset(births, Float32(0.0))
    ctx.enqueue_function[births_init_kernel](
        births.unsafe_ptr(),
        tree.children.unsafe_ptr(),
        tree.lambdas.unsafe_ptr(),
        Int32(tree.n_leaves),
        Int32(n_edges),
        grid_dim=((n_edges + STAB_TPB - 1) // STAB_TPB if n_edges > 0 else 1, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    # `:128` thrust::fill(stabilities, 0.0f)
    ctx.enqueue_memset(stabilities, Float32(0.0))
    ctx.enqueue_function[cluster_stability_kernel](
        stabilities.unsafe_ptr(),
        births.unsafe_ptr(),
        tree.indptr.unsafe_ptr(),
        tree.lambdas.unsafe_ptr(),
        tree.sizes.unsafe_ptr(),
        Int32(n_clusters),
        sabotage,
        grid_dim=(n_clusters, 1, 1),
        block_dim=(STAB_FOLD, 1, 1),
    )
    ctx.synchronize()
    _ = births^


def max_lambda_key_kernel(
    lambdas: MutPointer[Float32, MutAnyOrigin],
    out_key: MutPointer[Int32, MutAnyOrigin],
    n_edges: Int32,
):
    """`runner.h:208-210`'s max_element, as an integer `Atomic.max` on
    `weight_order_key` (order-free)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_edges):
        return
    _ = Atomic.max(out_key, weight_order_key(lambdas.unsafe_load(i)))


def cluster_count_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_leaves: Int32,
):
    """`stabilities.cuh:167-175`: an integer atomicAdd per labelled point."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_leaves):
        return
    var v = Int(labels.unsafe_load(i))
    if v > -1:
        _ = Atomic.fetch_add(counts.unsafe_offset(v), Int32(1))


def stability_score_kernel(
    stability: MutPointer[Float32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    label_map: MutPointer[Int32, MutAnyOrigin],
    max_key: MutPointer[Int32, MutAnyOrigin],
    result: MutPointer[Float32, MutAnyOrigin],
    n_clusters: Int32,
):
    """`stabilities.cuh:181-199`, one thread per condensed cluster."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_clusters):
        return
    var out_cluster = Int(label_map.unsafe_load(c))
    if out_cluster < 0:
        return
    var max_lambda = weight_order_unkey(max_key.unsafe_load(0))
    var size = Int(counts.unsafe_load(c))
    var expr = (
        max_lambda == FLOAT32_MAX or max_lambda == Float32(0.0) or size == 0
    )
    if expr:
        result.unsafe_store(out_cluster, Float32(1.0))
    else:
        result.unsafe_store(
            out_cluster,
            identical_div(
                stability.unsafe_load(c),
                identical_mul(Float32(size), max_lambda),
            ),
        )


def get_stability_scores_device(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut labels: DeviceBuffer[DType.int32],
    mut stability: DeviceBuffer[DType.float32],
    mut label_map: DeviceBuffer[DType.int32],
    n_selected: Int,
) raises -> List[Float32]:
    """`runner.h:208-219` (max_lambda, then `get_stability_scores`) on the
    device (the host form had no caller and was removed, cpu3-neighbors).
    Returns the `n_selected` scores."""
    if tree.n_edges < 1:
        raise Error(
            "hdbscan.max_lambda_of: the condensed tree has no edges; their"
            " thrust::max_element at runner.h:209 dereferences an empty"
            " range here"
        )
    var n_clusters = tree.n_clusters
    var n_leaves = tree.n_leaves
    var max_key = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(max_key, Int32(-2147483647 - 1))
    ctx.enqueue_function[max_lambda_key_kernel](
        tree.lambdas.unsafe_ptr(), max_key.unsafe_ptr(), Int32(tree.n_edges),
        grid_dim=((tree.n_edges + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    var counts = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(counts, Int32(0))
    ctx.enqueue_function[cluster_count_kernel](
        labels.unsafe_ptr(), counts.unsafe_ptr(), Int32(n_leaves),
        grid_dim=((n_leaves + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    var result = ctx.enqueue_create_buffer[DType.float32](max(n_selected, 1))
    ctx.enqueue_memset(result, Float32(0.0))
    ctx.enqueue_function[stability_score_kernel](
        stability.unsafe_ptr(), counts.unsafe_ptr(), label_map.unsafe_ptr(),
        max_key.unsafe_ptr(), result.unsafe_ptr(), Int32(n_clusters),
        grid_dim=((n_clusters + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    var out = td_download_f32(ctx, result, n_selected)
    _ = max_key^
    _ = counts^
    _ = result^
    return out^


# cpu3-neighbors (2026-10-04): `max_lambda_of` and `get_stability_scores`, the
# host forms of `runner.h:208-219`, had no caller left (the device fit uses
# `get_stability_scores_device` above, which carries their statements) and
# were removed from this GPU module.
