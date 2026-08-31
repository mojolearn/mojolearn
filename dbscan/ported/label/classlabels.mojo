# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`make_monotonic`: renumber labels onto a monotonically increasing set.

PORT OF `raft/label/detail/classlabels.cuh::make_monotonic` and
`map_label_kernel` at RAFT `661a3b8`. Replaced in its unique-value step; see
deviation 33. Do not improve.

WHY THIS IS NOT OPTIONAL, WHICH `dbscan/UNPORTED.tsv` GOT WRONG
---------------------------------------------------------------
The old entry excused skipping it with "labels are arbitrary up to
permutation, so the check compares the PARTITION". That is a true statement
about the CHECK and a false one about the API. `cuml/cpp/src/dbscan/runner.cuh:412`
runs `final_relabel` and then `relabelForSkl` on every fit, so a cuML user
gets `0..k-1` and `-1`. Anyone comparing our output with scikit-learn's or
cuML's got different numbers for the same clustering.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from dbscan.ported.dbscan.adjgraph.algo import exclusive_scan
from dbscan.ported.sparse.detail.csr import MAX_LABEL


comptime LABEL_TPB = 256


# ---------------------------------------------------------------------------
# DEVIATION BLOCK 33: THE UNIQUE-LABEL SET IS BUCKETED, NOT SORTED
#
# THEIRS: `make_monotonic` calls `getUniquelabels`
#   (`classlabels.cuh:50`), which is `cub::DeviceRadixSort::SortKeys`
#   followed by `cub::DeviceSelect::Unique` over all N labels, and then
#   `map_label_kernel` (`:122`), which for EVERY element linearly scans the
#   unique array until it finds a match -- O(N * n_clusters).
# OURS: a mark-and-scan. `weak_cc` labels are `min(vertex index) + 1` of the
#   component, so every non-noise label is an integer in `1..N`. Flag
#   `seen[l - 1] = 1`, exclusive-scan the flags, and `l -> seen_scan[l-1] + 1`
#   is the rank of `l` among the distinct labels in ascending order.
# REASON: it is the SAME FUNCTION, not an approximation. Their sorted-unique
#   array is the distinct labels in ascending order and `map_label_kernel`
#   assigns `i + 1` for position `i`; an exclusive scan of the occupancy
#   flags computes that same position without materializing the array. It is
#   O(N) instead of O(N * n_clusters), and it needs neither a device sort nor
#   a device stream-compaction, which is the only reason a port of their
#   literal route would have had to reach for a substitute at all.
#   `MAX_LABEL` is excluded exactly as their `filter_op` excludes it (the
#   caller passes `val == MAX_LABEL`); since `MAX_LABEL` sorts last in their
#   version it never shifted any rank there either, so the two agree.
#   The precondition -- labels are in `1..N` or `MAX_LABEL` -- is guaranteed
#   by `weak_cc_init_all_kernel` and preserved by `merge_labels`, both of
#   which only ever move a label DOWN to another vertex's index + 1.
# ---------------------------------------------------------------------------


def mark_labels_kernel(
    seen: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Occupancy flags of the distinct non-noise labels. See deviation 33."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        var l = labels.unsafe_load(tid)
        if l != MAX_LABEL:
            seen.unsafe_store(Int(l) - 1, Int32(1))


def map_label_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`map_label_kernel` with `zero_based = false`: `out = i + 1`.

    Filtered elements are LEFT UNWRITTEN, which is theirs: their kernel only
    assigns inside `if (!filter_op(in[tid]))`, and because `out` and `in` are
    the same array for `final_relabel`, a noise point keeps `MAX_LABEL`.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        var l = labels.unsafe_load(tid)
        if l != MAX_LABEL:
            labels.unsafe_store(tid, rank.unsafe_load(Int(l) - 1) + Int32(1))


def make_monotonic(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut seen: DeviceBuffer[DType.int32],
    mut rank: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    n_rows: Int,
) raises:
    """`make_monotonic(db_cluster, db_cluster, N, stream, filter_op)`.

    `seen` and `rank` are `n_rows` and `n_rows + 1` scratch arrays; `rank` is
    the exclusive scan of `seen`.
    """
    var blocks = (n_rows + LABEL_TPB - 1) // LABEL_TPB
    ctx.enqueue_memset(seen, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[mark_labels_kernel](
        seen.unsafe_ptr(),
        labels.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(blocks, 1, 1),
        block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.synchronize()
    exclusive_scan(ctx, rank, seen, block_sums, n_rows)
    ctx.synchronize()
    ctx.enqueue_function[map_label_kernel](
        labels.unsafe_ptr(),
        rank.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(blocks, 1, 1),
        block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.synchronize()
