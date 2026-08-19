"""`core[i] = vd[i] >= min_pts`, which is the whole of their file.

PORT OF `cuml/cpp/src/dbscan/corepoints/compute.cuh` at cuML `7e29955c`.
Transliterated.

Kept as its own file, mirroring theirs, even though it is four lines. The
mirror is the point: `ls` answers "did we port this", and a reviewer can put
this beside `compute.cuh`. Collapsing it into the caller would save nothing
and would lose that.
"""

from std.gpu import block_dim, block_idx, thread_idx


def core_points_kernel(
    core: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    min_pts_in: Int32,
):
    """`mask[idx + start_vertex_id] = (Index_)vd[idx] >= min_pts;`"""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_rows_in):
        core.unsafe_store(
            i, UInt8(1) if vd.unsafe_load(i) >= min_pts_in else UInt8(0)
        )
