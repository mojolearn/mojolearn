# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core[i] = vd[i] >= min_pts`, which is the whole of their file.

PORT OF `cuml/cpp/src/dbscan/corepoints/compute.cuh` at cuML `00094f7`.
Transliterated.

Kept as its own file, mirroring theirs, even though it is four lines. The
mirror is the point: `ls` answers "did we port this", and a reviewer can put
this beside `compute.cuh`. Collapsing it into the caller would save nothing
and would lose that.

`corepoints/exchange.cuh` is the multi-GPU allgather of this mask and is out
of scope.

THE WEIGHTED OVERLOAD IS THE SAME FILE, BECAUSE IT IS THE SAME FUNCTION
-----------------------------------------------------------------------
`compute` is templated on the VALUE type (`compute.cuh:38`,
`template <typename Values_ = int, typename Index_ = int>`), and
`runner.cuh:300-306` instantiates it twice: `Values_ = Index_` over `vd` on
the unweighted path and `Values_ = Type_f` over `wght_sum` on the weighted
one. Mojo has no template here, so it is two functions in the file their one
template lives in.

`core_points_compute_weighted` is where `sample_weight` actually CHANGES THE
ANSWER, and it is the only place it does. Everything before it -- the
neighborhood, the adjacency, the CSR -- and everything after it -- the
propagation, the merge, the relabel -- is byte-for-byte the unweighted path.

WHY THE FLOAT IS COMPARED DIRECTLY AND THEIR CAST IS NOT REPRODUCED.
`compute.cuh:50` reads `mask[...] = (Index_)vd[idx] >= min_pts`, so on the
weighted instantiation a Float32 sum is truncated to Int32 before the
compare. That is not a bug of theirs and it is not ported as one: for
`min_pts >= 1` and a non-negative sum, `trunc(w) >= p` and `w >= p` are the
same predicate, because `trunc` is monotone and `trunc(p) == p` for integer
`p`. Comparing the float directly is the same answer and does not depend on
that argument holding, which matters because scikit-learn admits NEGATIVE
weights (`_dbscan.py:414-415`) and `trunc` rounds toward zero rather than
down.
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


def core_points_weighted_kernel(
    core: MutPointer[UInt8, MutAnyOrigin],
    wght_sum: MutPointer[Float32, MutAnyOrigin],
    start_vertex_id_in: Int32,
    batch_size_in: Int32,
    min_pts_in: Int32,
):
    """`compute<Type_f, Index_>`: the SUM OF WEIGHTS reaches `min_pts`.

    Their `mask[idx + start_vertex_id] = (Index_)vd[idx] >= min_pts` with
    `vd` bound to `wght_sum` (`runner.cuh:301-302`). Same indexing as the
    unweighted kernel: `wght_sum` by the BATCH, `core` by the DATASET.

    No `ftz` on the compare. The value was produced and flushed by
    `weighted_vertex_deg_*_kernel`; a comparison performs no arithmetic and
    cannot make a denormal, and flushing again here would be a second
    opinion about a value that is already pinned.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(batch_size_in):
        core.unsafe_store(
            i + Int(start_vertex_id_in),
            UInt8(1) if wght_sum.unsafe_load(i) >= Float32(min_pts_in) else UInt8(0),
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


def core_points_compute_weighted(
    ctx: DeviceContext,
    mut wght_sum: DeviceBuffer[DType.float32],
    mut core: DeviceBuffer[DType.uint8],
    min_pts: Int,
    start_vertex_id: Int,
    batch_size: Int,
) raises:
    """`CorePoints::compute<Type_f, Index_>`, `runner.cuh:301-302`."""
    ctx.enqueue_function[core_points_weighted_kernel](
        core.unsafe_ptr(),
        wght_sum.unsafe_ptr(),
        Int32(start_vertex_id),
        Int32(batch_size),
        Int32(min_pts),
        grid_dim=((batch_size + CORE_TPB - 1) // CORE_TPB, 1, 1),
        block_dim=(CORE_TPB, 1, 1),
    )
