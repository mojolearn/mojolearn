"""Turn a GEMM output into distances, in place.

NOT A PORT of a file. In cuVS this is an EPILOGUE fused into the distance
call (`distance/detail/`), and in the unfused path it lives inside
`reduce_min_kernel` because the reduction consumes each element as it is
formed. Brute-force k-NN cannot do that: it needs every distance to survive
so a top-k can run over them, so the epilogue has to become its own pass.

The arithmetic is the same expanded identity and the same clamp
(`unfused_distance_nn.cuh:80-81`), including the reason the clamp exists:
GEMM round-off makes a point sitting on its own neighbor come out slightly
negative, and `sqrt` of that is NaN.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt


def expand_distances_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    is_sqrt_in: Int32,
):
    """`z[i][j] <- ||x_i||^2 + ||y_j||^2 - 2 z[i][j]`, clamped at zero."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return

    var row = idx // n_cols
    var col = idx % n_cols
    var d = (
        x_norm.unsafe_load(row)
        + y_norm.unsafe_load(col)
        - Float32(2.0) * z.unsafe_load(idx)
    )
    if d <= Float32(0.0):
        d = Float32(0.0)
    if is_sqrt_in != 0:
        d = sqrt(d)
    z.unsafe_store(idx, d)
