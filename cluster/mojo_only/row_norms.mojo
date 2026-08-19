"""Squared L2 norm of every row, which the expanded identity needs twice.

NOT A PORT of cuVS. Their call is
`raft::linalg::norm<L2Norm, Apply::ALONG_ROWS>` (`detail/kmeans.cuh:770`,
`minClusterDistanceCompute.cu:44`), and RAFT is a separate library whose
primitives this tree does not mirror file for file. Only the CALL SITES and
their semantics are theirs, and those are copied exactly:

- X's norms are computed ONCE per fit, before the iteration loop, and reused
  by every Lloyd iteration and by k-means++ (`detail/kmeans.cuh:786-790`).
- Centroid norms are recomputed EVERY assignment, because the centroids move
  (`minClusterDistanceCompute.cu:43-49`).
- For L2 the norm is left SQUARED. For cosine it is passed through `sqrt`,
  because the cosine branch of the reduction divides by `||x|| ||y||` rather
  than subtracting. Getting that backward gives a plausible, wrong answer on
  every row, which is why the flag is a parameter and not a comment.

**This reduction IS numeric**, unlike the argmin it feeds. It is a float sum
over the feature axis, so the block size changes the summation order and
therefore the last bits. It is listed in the `IDENTICAL` column's scope.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime NORM_TPB = 128


def row_norm_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    take_sqrt_in: Int32,
):
    """One block per row, block sum of squares, optional square root."""
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var col = tid
    while col < n_cols:
        var v = a.unsafe_load(row * n_cols + col)
        acc += v * v
        col += NORM_TPB

    var s = stack_allocation[
        NORM_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s[tid] = acc
    barrier()

    var stride = NORM_TPB // 2
    while stride > 0:
        if tid < stride:
            s[tid] = s[tid] + s[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        var total = s[0]
        if take_sqrt_in != 0:
            if total <= Float32(0.0):
                total = Float32(0.0)
            total = sqrt(total)
        out_norm.unsafe_store(row, total)
