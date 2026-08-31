# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`core[i] = vd[i] >= min_pts`, which is the whole of their file.

PORT OF `cuml/cpp/src/dbscan/corepoints/compute.cuh` at cuML `00094f7`.
Transliterated.

Kept as its own file, mirroring theirs, even though it is four lines. The
mirror is the point: `ls` answers "did we port this", and a reviewer can put
this beside `compute.cuh`. Collapsing it into the caller would save nothing
and would lose that.

`corepoints/exchange.cuh` is the multi-GPU allgather of this mask and is out
of scope.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext


comptime CORE_TPB = 256


def core_points_kernel(
    core: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    start_vertex_id_in: Int32,
    batch_size_in: Int32,
    min_pts_in: Int32,
):
    """`mask[idx + start_vertex_id] = (Index_)vd[idx] >= min_pts;`

    `vd` is indexed by the BATCH and `mask` by the DATASET, which is theirs
    (`compute.cuh:50`) and is what lets the mask accumulate across batches
    while `vd` is overwritten by each one.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(batch_size_in):
        core.unsafe_store(
            i + Int(start_vertex_id_in),
            UInt8(1) if vd.unsafe_load(i) >= min_pts_in else UInt8(0),
        )


def core_points_compute(
    ctx: DeviceContext,
    mut vd: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    min_pts: Int,
    start_vertex_id: Int,
    batch_size: Int,
) raises:
    """`CorePoints::compute`."""
    ctx.enqueue_function[core_points_kernel](
        core.unsafe_ptr(),
        vd.unsafe_ptr(),
        Int32(start_vertex_id),
        Int32(batch_size),
        Int32(min_pts),
        grid_dim=((batch_size + CORE_TPB - 1) // CORE_TPB, 1, 1),
        block_dim=(CORE_TPB, 1, 1),
    )
