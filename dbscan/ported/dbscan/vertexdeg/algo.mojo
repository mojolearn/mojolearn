"""Epsilon neighborhood: the boolean adjacency and the vertex degree.

PORT OF `cuml/cpp/src/dbscan/vertexdeg/algo.cuh` at cuML `7e29955c`, and of
the `corepoints/compute.cuh` step that follows it. Partial. Do not improve.

cuML delegates the neighborhood itself to
`cuvs::neighbors::epsilon_neighborhood` or to a ball-cover index, which are
two more upstreams than this section needs. What is ported is the SHAPE
their runner depends on, which is the same in either case:

    adj[i][j] = dist(i, j) <= eps      (a boolean matrix)
    vd[i]     = number of j with adj[i][j]

and then, in `corepoints/compute.cuh`:

    core[i] = vd[i] >= min_pts

**`eps` is squared once on the host and never per pair.** DBSCAN's radius is
a distance and our distances are squared, and squaring the threshold instead
of rooting a million distances is theirs and is the obvious thing to keep.

A point is its own neighbor here, exactly as in scikit-learn: the distance to
itself is zero, which is `<= eps`, so `vd` includes the point and `min_pts`
counts it. Getting that wrong shifts every core-point decision by one and
produces a plausible clustering that disagrees with sklearn everywhere.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime VD_TPB = 128


def eps_neighborhood_kernel(
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    dist: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    eps_sq_in: Float32,
):
    """One block per row of the distance tile. Writes `adj` and counts it.

    `dist` is the already-expanded distance tile from `core/gemm.mojo` plus
    `core/expand_distances.mojo`, which is the same pair of kernels k-means
    and k-NN use. DBSCAN differs from k-NN only in what it does with that
    tile: a radius test instead of a top-k.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var count = Int32(0)
    var j = tid
    while j < n_cols:
        var inside = dist.unsafe_load(row * n_cols + j) <= eps_sq_in
        adj.unsafe_store(row * n_cols + j, UInt8(1) if inside else UInt8(0))
        if inside:
            count += 1
        j += VD_TPB

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=VD_TPB](count)
    if tid == 0:
        vd.unsafe_store(row, s0)
