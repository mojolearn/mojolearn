# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Merge two labellings in place, according to a core-point mask.

PORT OF `raft/label/detail/merge_labels.cuh` at RAFT `661a3b8`
(`propagate_label_kernel`, `reassign_label_kernel`, `merge_labels`).
Transliterated. Do not improve.

WHY DBSCAN NEEDS THIS AT ALL
----------------------------
`weak_cc_batched` labels the sub-graph of ONE batch and overwrites the whole
label array doing it (`raft/sparse/detail/csr.cuh:110-117`, "it is the
responsibility of the caller to combine the results from different batches").
cuML's runner therefore keeps batch 0's labelling in `labels`, puts every
later batch's in `labels_temp`, and merges (`runner.cuh:398`).

Their comment at `runner.cuh:392-395` is the reason this cannot be shortcut:

    Using the labelling from the previous batches as initial value for
    weak_cc_batched and skipping the merge step would lead to incorrect
    results as described in #3094.

This port previously had no merge and no per-batch labelling. It kept ONE
`weak_cc` over a CSR built from every row of the dataset, which is correct
but is the thing batching exists to prevent: the global CSR is as large as
the whole adjacency in sparse form, so the memory the batched adjacency
saved came straight back in `col_ind`.

THE LABEL EQUIVALENCE GRAPH IS IMPLICIT
---------------------------------------
Their own note at the top of the file: the graph
`E = {(A[i], B[i]) | M[i] = 1}` is never built. `R` is a relabelling map over
`1..N`, initialized to the identity, and `propagate_label_kernel` runs
`atomicMin` on both endpoints until nothing moves. Then
`reassign_label_kernel` applies `label l -> R[l-1] + 1` to both inputs and
takes the smaller.

`R[min(ra, rb)]` rather than `min(ra, rb)` on line 54 is theirs and their
comment says why: "min(ra, rb) would be sufficient but this speeds up
convergence". Copied, including the extra indirection.
"""

from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    PIN_DETERMINISM,
    numeric_mode_name,
)
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from dbscan.derived.sparse.detail.csr import MAX_LABEL


comptime MERGE_TPB = 256


def range_kernel(r: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    """`raft::linalg::range(R, N, stream)` (`merge_labels.cuh:129`)."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        r.unsafe_store(tid, Int32(tid))


def propagate_label_kernel(
    labels_a: MutPointer[Int32, MutAnyOrigin],
    labels_b: MutPointer[Int32, MutAnyOrigin],
    r: MutPointer[Int32, MutAnyOrigin],
    mask: MutPointer[UInt8, MutAnyOrigin],
    m: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`propagate_label_kernel`, copied.

    Only masked (core) points may connect a group of `labels_a` to a group of
    `labels_b`, which is the same rule as `weak_cc`'s `filter_op` one level
    down: a border point may be labelled by both labellings without welding
    them together.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        if mask.unsafe_load(tid) != 0:
            # Note: labels are from 1 to N
            var la = Int(labels_a.unsafe_load(tid)) - 1
            var lb = Int(labels_b.unsafe_load(tid)) - 1
            var ra = r.unsafe_load(la)
            var rb = r.unsafe_load(lb)
            if ra != rb:
                m.unsafe_store(0, Int32(1))
                # min(ra, rb) would be sufficient but this speeds up
                # convergence
                var lo = ra
                if rb < ra:
                    lo = rb
                var rmin = r.unsafe_load(Int(lo))
                _ = Atomic.min(r.unsafe_offset(la), rmin)
                _ = Atomic.min(r.unsafe_offset(lb), rmin)


def reassign_label_kernel(
    labels_a: MutPointer[Int32, MutAnyOrigin],
    labels_b: MutPointer[Int32, MutAnyOrigin],
    r: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`reassign_label_kernel`, copied. `MAX_LABEL` stays `MAX_LABEL`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        # Note: labels are from 1 to N
        var la = labels_a.unsafe_load(tid)
        var lb = labels_b.unsafe_load(tid)
        var ra = MAX_LABEL
        if la != MAX_LABEL:
            ra = r.unsafe_load(Int(la) - 1) + Int32(1)
        var rb = MAX_LABEL
        if lb != MAX_LABEL:
            rb = r.unsafe_load(Int(lb) - 1) + Int32(1)
        var lo = ra
        if rb < ra:
            lo = rb
        labels_a.unsafe_store(tid, lo)


def merge_labels(
    ctx: DeviceContext,
    mut labels_a: DeviceBuffer[DType.int32],
    mut labels_b: DeviceBuffer[DType.int32],
    mut mask: DeviceBuffer[DType.uint8],
    mut work_buffer: DeviceBuffer[DType.int32],
    mut d_m: DeviceBuffer[DType.int32],
    mut h_m: HostBuffer[DType.int32],
    n_rows: Int,
    max_iterations: Int,
) raises:
    """`merge_labels` (`merge_labels.cuh:115`): init R, propagate, reassign.

    `work_buffer` is their `R`, `n_rows` long, which is exactly the buffer
    `runner.cuh:222` carves out of the workspace for it.
    """
    var blocks = (n_rows + MERGE_TPB - 1) // MERGE_TPB
    ctx.enqueue_function[range_kernel](
        work_buffer.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(blocks, 1, 1),
        block_dim=(MERGE_TPB, 1, 1),
    )
    ctx.synchronize()

    # Step 1: compute connected components in the label equivalence graph.
    #
    # The same cap and the same refusal as `weak_cc_batched` (DEVIATION
    # 507): `propagate_label_kernel` reaches a UNIQUE fixed point whatever
    # order its `atomicMin`s land in, and a run cut off at `max_iterations`
    # has not reached it. Under IDENTICAL that is refused; under FAST it is
    # upstream's silent truncation.
    var merged = False
    for _it in range(max_iterations):
        h_m.unsafe_ptr().unsafe_store(0, Int32(0))
        ctx.enqueue_copy(dst_buf=d_m, src_ptr=h_m.unsafe_ptr())
        ctx.enqueue_function[propagate_label_kernel](
            labels_a.unsafe_ptr(),
            labels_b.unsafe_ptr(),
            work_buffer.unsafe_ptr(),
            mask.unsafe_ptr(),
            d_m.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=(blocks, 1, 1),
            block_dim=(MERGE_TPB, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=h_m.unsafe_ptr(), src_buf=d_m)
        ctx.synchronize()
        if h_m.unsafe_ptr().unsafe_load(0) == Int32(0):
            merged = True
            break
    comptime if PIN_DETERMINISM:
        # DEVIATION 507, and `PIN_DETERMINISM` since 2026-08-29:
        # `propagate_label_kernel` reaches a UNIQUE fixed point whatever
        # order its `atomicMin`s land in, so a run CUT OFF at the cap is
        # holding the atomic order rather than the graph -- which two
        # runs on one device need not agree about. Keyed to the top tier
        # only, the middle tier returned it.
        if not merged:
            raise Error(
                "merge_labels: the label equivalence graph did not settle"
                " in "
                + String(max_iterations)
                + " passes. Under "
                + numeric_mode_name()
                + " a truncated merge is"
                " refused rather than returned; raise max_iterations."
            )

    # Step 2: re-assign minimum equivalent label
    ctx.enqueue_function[reassign_label_kernel](
        labels_a.unsafe_ptr(),
        labels_b.unsafe_ptr(),
        work_buffer.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(blocks, 1, 1),
        block_dim=(MERGE_TPB, 1, 1),
    )
    ctx.synchronize()
