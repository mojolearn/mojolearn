# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mutual reachability graph as a DENSE matrix, and why it is dense.

NOT A PORT of a file; a port of ONE FUNCTOR (`ReachabilityPostProcess`,
`cuvs cpp/src/neighbors/detail/reachability.cuh:126-135`) applied to the
connectivity `hierarchy/` already builds and gates. The DEVIATION BLOCK
below is the whole justification and it is the first thing to read in
this lane.

======================================================================
DEVIATION BLOCK -- DEVIATION 1600. THE MUTUAL REACHABILITY GRAPH IS THE
COMPLETE PAIRWISE ONE, NOT THEIR SPARSE k-NN COO.
======================================================================

WHAT THEIRS DOES. `mutual_reachability_graph`
(`cuvs .../neighbors/detail/reachability.cuh:190-256`) builds a SPARSE
graph in four steps: a `min_samples`-nearest-neighbour search; the core
distances sliced out of it; a SECOND k-NN search
(`mutual_reachability_knn_l2`, `:151-188`) that runs
`tiled_brute_force_knn` with the `ReachabilityPostProcess` epilogue so
the selection happens IN mutual reachability space; then
`raft::sparse::linalg::symmetrize` + `sorted_coo_to_csr` into a COO/CSR
of at most `2 * min_samples * m` edges. `build_mr_linkage`
(`cuvs cpp/src/cluster/detail/single_linkage.cuh:50-118`) hands that CSR
to `build_sorted_mst` together with a
`MutualReachabilityFixConnectivitiesRedOp`.

WHY THAT PATH CANNOT BE PORTED IN THIS RUNG, AND IT IS TWO WALLS RATHER
THAN ONE.

(a) THE GRAPH IS DISCONNECTED BY CONSTRUCTION AND THE FIX-UP IS NOT
    PORTED. A symmetrized k-NN graph over well-separated clusters has one
    component per cluster -- that is the case HDBSCAN exists for -- so
    Boruvka returns a FOREST and `build_sorted_mst` enters its fix-up
    loop, which calls `connect_knn_graph` / `cross_component_nn` /
    `merge_msts`. `hierarchy/DERIVATION_MAP.tsv` and `hierarchy/NOT_IMPLEMENTED.tsv`
    record all three as NOT PORTED, and
    `hierarchy/impl/cluster/detail/mst.mojo::connect_knn_graph` RAISES
    BY NAME. A rung-1 HDBSCAN over the sparse graph would therefore
    refuse on its own headline fixture. It still refuses: the sparse arm
    is REFUSED BY NAME in `impl/hdbscan/detail/reachability.mojo`
    rather than half-written.

(b) THE EPILOGUE HOOK DOES NOT EXIST IN THIS TREE.
    `mutual_reachability_knn_l2` needs `tiled_brute_force_knn`'s
    `DistanceEpilogue` template parameter, and
    `neighbors/impl/neighbors/detail/knn_brute_force.mojo` carries no
    such parameter (`neighbors/NOT_IMPLEMENTED.tsv` records the epilogue
    template as not ported). Adding one is the NEIGHBORS lane's call, not
    this lane's, and it is named in this lane's README under WHAT THE
    ORCHESTRATOR MUST WIRE.

    THE EPILOGUE IS NOT A DECORATION. `max(core[col], max(core[row],
    alpha*d))` is not monotone in `d` alone, because `core[col]` varies
    with the candidate, so the mutual-reachability k nearest neighbours
    are NOT the plain k nearest neighbours reordered. There is no way to
    get their sparse graph out of a plain k-NN result, which is why this
    is a wall rather than an inconvenience.

WHAT OURS DOES. The connectivity is `Linkage::PAIRWISE`
(`cuvs cpp/src/cluster/detail/connectivities.cuh:110-204`), which cuVS
ships, which `hierarchy/impl/cluster/detail/connectivities.mojo` has
already ported and gated bit for bit, and which yields a COMPLETE graph
whose MST is connected on the first Boruvka call -- so the fix-up loop's
body is never entered and (a) does not arise. Their functor is then
applied to every cell of that matrix by the kernel below. `alpha` and the
self-loop `FLT_MAX` are theirs, unchanged.

WHAT IT COSTS IN FIDELITY, STATED RATHER THAN GLOSSED. The MST of the
COMPLETE mutual-reachability graph is the reference answer -- it is what
scikit-learn-contrib's `_hdbscan_generic` computes and it is what their
sparse path plus `connect_knn_graph` is an ACCELERATION of, not an
alternative to. So on a fixture where their fix-up would have found the
same edges the two agree; where it would not, ours is the exact one. The
price is memory and work: `m * m` cells rather than `2 * min_samples * m`,
which is why `PAIRWISE_MAX_ROWS` (46340) is the hard bound this lane
inherits from `hierarchy/` and refuses past.

NOTHING IS MEASURED HERE. No timing was taken, and the sentence above
about cost is an arithmetic statement about array sizes, not a
benchmark.
======================================================================

======================================================================
DEVIATION BLOCK -- DEVIATION 1601. THE THREE-WAY MAX IS A TOTAL-ORDER
SELECTION, NOT A HARDWARE `max`.
======================================================================

WHAT THEIRS DOES. `return max(core_dists[col], max(core_dists[row],
alpha * value));` (`reachability.cuh:129`), the CUDA `max` on floats.

WHY IT CANNOT BE PORTED AS-IS. IDENTITY_PATHS row 39, MEASURED on all
three columns on 2026-08-23: `max(+0.0, -0.0)` is `-0.0` on Apple (the
SECOND operand) and `+0.0` on NVIDIA and AMD (IEEE-2019 `maximum`). A
three-way max is a two-level fold, so the answer to a `(+0, -0)` pair
lands in the graph as a WEIGHT, and a weight is what the MST's total
order reads. One bit there moves an edge, a dendrogram row, a condensed
cluster and a label. Row 39 also records that a NaN's PAYLOAD is the
vendor's (Apple `0x7fc00000`, NVIDIA `0x7fffffff`, AMD `0xffc00000`), and
a hardware `max` propagates or absorbs a NaN at the vendor's discretion
too.

WHAT OURS DOES. `checks/hdbscan_sabotage.mojo::mr_max3` calls
`numerics.identical_fmax`, which under IDENTICAL is `portable_fmaxf`:
operands flushed, NaN canonicalized to `0x7fc00000` BEFORE the compare,
and the selection made on `_total_order_key`, an INTEGER map whose order
is the float order with `-0.0` strictly below `+0.0`. No float compare
and no hardware max instruction appear on the path. Under FAST it is the
stdlib `max` and this lane makes no cross-vendor claim.

IS THE HAZARD REACHABLE ON THE DEFAULT PATH? Not through the distances:
both distance arms clamp with `if dist <= 0.0: dist = 0.0`, an IEEE
compare that maps `-0.0` and every negative residue to `+0.0` on every
vendor (`hierarchy/README.md`'s row-39 audit proves that for the same two
files), and `identical_sqrt(+0.0)` is `+0.0`. Core distances are read out
of those same cells, so they are `+0.0` or positive too. The pin is
therefore INERT on the default path AND THAT IS NOT A REASON TO DROP IT:
the same was true of `hierarchy`'s key until a caller fed
`build_sorted_mst` a graph directly. `hdbscan_check.mojo` plants `-0.0`
into a core-distance array and into a distance cell and drives this
kernel with them, which is the only way the gate has teeth.
======================================================================
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.checks.hdbscan_sabotage import (
    HDB_SAB_NONE,
    HDB_SAB_SKIP_GUARDS,
    mr_max3,
    mr_scale,
)
from hierarchy.impl.cluster.detail.connectivities import FLOAT32_MAX


comptime MR_TPB = 256
"""The mutual-reachability transform's block size. SCHEDULING, not
numeric: one thread per cell and no fold, so nothing in the arithmetic
can see it. `check_hdbscan_launch_invariance` varies it and requires the
bytes to stand still."""


def mutual_reachability_dense_kernel(
    mr: MutPointer[Float32, MutAnyOrigin],
    dists: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    inv_alpha: Float32,
    sabotage: Int32,
):
    """One thread per cell of the `m x m` matrix.

    `mr[i][j] = max(core[j], max(core[i], (1/alpha) * d[i][j]))` for
    `i != j`, and `FLT_MAX` on the diagonal -- their self-loop transform
    (`reachability.cuh:243-255`, `row == col ? numeric_limits::max() :
    val`), applied here rather than after because `hierarchy`'s
    `pairwise_distances` has already put `FLT_MAX` on the diagonal and
    `max(c_i, c_i, FLT_MAX)` would return it anyway. Writing it
    explicitly means the diagonal's bits do not depend on the max at all.

    NO FOLD, NO ATOMIC, NO LANE PRIMITIVE. Every cell is written by
    exactly one thread from three loads, so the grid shape, the block
    size and the landing order cannot reach the answer.
    """
    var m = Int(m_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m * m:
        return
    var row = idx // m
    var col = idx % m
    if row == col:
        mr.unsafe_store(idx, FLOAT32_MAX)
        return
    var scaled = mr_scale(inv_alpha, dists.unsafe_load(idx))
    var v = mr_max3(
        core.unsafe_load(row), core.unsafe_load(col), scaled, sabotage
    )
    mr.unsafe_store(idx, v)


def mutual_reachability_dense(
    ctx: DeviceContext,
    mut mr: DeviceBuffer[DType.float32],
    mut dists: DeviceBuffer[DType.float32],
    mut core: DeviceBuffer[DType.float32],
    m: Int,
    inv_alpha: Float32,
    mr_tpb: Int = MR_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """Launch the transform over the whole matrix. `mr` and `dists` may
    be the same buffer only through a sub-buffer view; the caller here
    always passes two, because the check compares the two stages."""
    var nnz = m * m
    var blocks = (nnz + mr_tpb - 1) // mr_tpb if nnz > 0 else 1
    ctx.enqueue_function[mutual_reachability_dense_kernel](
        mr.unsafe_ptr(),
        dists.unsafe_ptr(),
        core.unsafe_ptr(),
        Int32(m),
        inv_alpha,
        sabotage,
        grid_dim=(blocks, 1, 1),
        block_dim=(mr_tpb, 1, 1),
    )
    ctx.synchronize()


# ======================================================================
# DEVIATION BLOCK -- DEVIATION 1607. A NON-FINITE VALUE IS REFUSED BY
# NAME BEFORE IT CAN REACH A RECORDED STAGE.
# ======================================================================
#
# WHAT THEIRS DOES. Nothing. `_fit_hdbscan` (`cuml .../hdbscan/runner.h:
# 152-234`) validates `min_samples <= m` and no more; a NaN coordinate
# becomes a NaN distance, a NaN core distance, a NaN mutual reachability
# (their `max` with a NaN operand is the vendor's answer), a NaN MST
# weight, a NaN `delta`, and `1.0 / NaN` is a NaN lambda that
# `stabilities_functor` then `atomicAdd`s. `probabilities_functor`
# (`kernels/membership.cuh:44`) even tests `isnan(child_lambda)`
# explicitly, so upstream KNOWS a NaN gets that far.
#
# WHY OURS CANNOT. A computed NaN carries the VENDOR'S payload
# (IDENTITY_PATHS row 39, measured: Apple 0x7fc00000, NVIDIA 0x7fffffff,
# AMD 0xffc00000) and every array named below is a RECORDED CARD STAGE in
# `hdbscan_main.mojo`. A card that differs only in a NaN payload reports a
# cross-vendor divergence that is not one, and a card that AGREES because
# both sides wrote the same payload has certified nothing about the
# arithmetic. So the value never gets in.
#
# THREE GUARDS, at the three seams where a non-finite can first appear:
#
#   1. the distance matrix -- ALREADY GUARDED, and not by this lane:
#      `hierarchy/checks/nan_guard.mojo::refuse_nan_distances`
#      (DEVIATION 623) is called from inside `pairwise_distances`, which
#      is the function this lane reuses. Nothing is added there.
#   2. the CORE DISTANCES, `refuse_nonfinite` below. They come out of the
#      k-NN result, which is a different code path from the dense matrix
#      (`neighbors/`), so a guard on the matrix does not cover them.
#   3. the LAMBDAS, `refuse_nonfinite` below on the host. `lambda =
#      1/delta` is `FLT_MAX` at `delta == 0` BY THEIR OWN RULE
#      (`condense.cuh:149`), which is finite and is kept; what is refused
#      is a NaN or an infinity arising any other way.
#
# `+inf` IS REFUSED HERE AND IS NOT IN `hierarchy`. That is deliberate and
# it is a difference worth naming: `hierarchy` keeps `+inf` because its
# bit pattern is the same on every vendor and it orders cleanly. Here an
# infinite core distance would make `mr` infinite for every pair touching
# that point, `1/inf` is `+0.0`, and a `+0.0` lambda is indistinguishable
# from `1/FLT_MAX`'s subnormal underflow at the stability seam -- two
# different causes collapsing onto one value inside a certified stage.
# Refused with the count and the first offending index.
# ======================================================================


comptime GUARD_TPB = 256


def count_nonfinite_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    count: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`count[0] += 1` for every cell that is NaN or infinite. An INTEGER
    atomic add, so the total is order-free on every vendor -- the same
    argument `hierarchy/checks/nan_guard.mojo` makes for its count."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var v = data.unsafe_load(i)
    var bad = False
    if v != v:
        bad = True
    elif v > FLOAT32_MAX:
        bad = True
    elif v < -FLOAT32_MAX:
        bad = True
    if bad:
        _ = Atomic.fetch_add(count.unsafe_offset(0), Int32(1))


def count_nonfinite_cells(
    ctx: DeviceContext, mut data: DeviceBuffer[DType.float32], n: Int
) raises -> Int:
    """How many of the first `n` cells are NaN or infinite. Synchronizes."""
    var count = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(count, Int32(0))
    var blocks = (n + GUARD_TPB - 1) // GUARD_TPB if n > 0 else 1
    ctx.enqueue_function[count_nonfinite_kernel](
        data.unsafe_ptr(),
        count.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(GUARD_TPB, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=count)
    ctx.synchronize()
    var n_bad = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    _ = count^
    return n_bad


def refuse_nonfinite_device(
    ctx: DeviceContext,
    mut data: DeviceBuffer[DType.float32],
    n: Int,
    where: String,
    what: String,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """DEVIATION 1607 on a device buffer."""
    if sabotage == HDB_SAB_SKIP_GUARDS:
        return
    var n_bad = count_nonfinite_cells(ctx, data, n)
    if n_bad != 0:
        raise Error(
            where + ": " + String(n_bad) + " of " + String(n) + " " + what
            + " are NaN or infinite; refused by name (DEVIATION 1607,"
            " IDENTITY_PATHS row 39). A computed NaN's payload is the"
            " vendor's (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD"
            " 0xffc00000) and this array is a recorded card stage, so its"
            " bits cannot be allowed to name the vendor instead of the"
            " arithmetic. To close this refusal, give the caller a finite"
            " input; there is no canonicalization that would be honest"
            " here, because a NaN of OUR choosing inside a dendrogram"
            " distance is a number nobody computed"
        )


def refuse_nonfinite_host(
    values: List[Float32],
    where: String,
    what: String,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """DEVIATION 1607 on a host list. Reports the FIRST offending index as
    well as the count, because the host arrays this guards (`delta`, the
    lambdas) are indexed by merge row and the row number is the
    diagnosis."""
    if sabotage == HDB_SAB_SKIP_GUARDS:
        return
    var n_bad = 0
    var first = -1
    for i in range(len(values)):
        var v = values[i]
        var bad = False
        if v != v:
            bad = True
        elif v > FLOAT32_MAX:
            bad = True
        elif v < -FLOAT32_MAX:
            bad = True
        if bad:
            n_bad += 1
            if first < 0:
                first = i
    if n_bad != 0:
        raise Error(
            where + ": " + String(n_bad) + " of " + String(len(values)) + " "
            + what + " are NaN or infinite, first at index " + String(first)
            + "; refused by name (DEVIATION 1607, IDENTITY_PATHS row 39)."
            " A computed NaN's payload is the vendor's and this array is a"
            " recorded card stage. Note that lambda = FLT_MAX at delta == 0"
            " is THEIR rule (condense.cuh:149) and is finite, so it is not"
            " what this refusal is about"
        )
