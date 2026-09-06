# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`MergeLabels::run`, which is one call, kept as its own file like theirs.

PORT OF `cuml/cpp/src/dbscan/mergelabels/runner.cuh` at cuML `00094f7`.
Transliterated.

Their whole file is:

    raft::label::merge_labels<Index_, TPB_X>(
        labels_a, labels_b, mask, work_buffer, m, N, stream);

`mergelabels/tree_reduction.cuh` is the multi-GPU sibling and is out of
scope: it is `comm.isend` / `comm.irecv` over a binary tree of ranks, and it
calls this same `run` on the receiving side.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from dbscan.impl.label.merge_labels import merge_labels


def merge_labels_run(
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
    """`MergeLabels::run`."""
    merge_labels(
        ctx,
        labels_a,
        labels_b,
        mask,
        work_buffer,
        d_m,
        h_m,
        n_rows,
        max_iterations,
    )
