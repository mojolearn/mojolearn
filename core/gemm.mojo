"""The matrix product the unfused path calls cuBLAS for.

NOT A PORT. cuVS calls `cublasGemmEx` (`unfused_distance_nn.cuh:205`) and
cuBLAS is a closed NVIDIA library with no source to transliterate, so there
is nothing here to copy and this file lives in `mojo_only/` for the same
reason `gpu_util/copy.mojo` does on the boosting side.

**Read their call before reading this file, because one argument of it is the
most interesting thing found in the whole cuVS k-means source:**

    computeType = CUBLAS_COMPUTE_32F_FAST_TF32;   // unfused_distance_nn.cuh:196

For `float` input, cuVS's default distance GEMM runs in **TF32**, which keeps
8 exponent bits and **10 mantissa bits** where float32 has 23. So cuVS's
float32 k-means, at its shipped defaults, does not compute float32 distances
on NVIDIA. It computes 10-bit-mantissa products and accumulates them in
float32.

That is the same shape of finding as XGBoost's CPU float64 against its GPU
Int64 fixed point, and it is stronger, because here the two number systems
are not even CPU versus GPU: they are NVIDIA versus everyone. A Metal or a
CPU implementation of the identical algorithm CANNOT reproduce cuVS's float32
answer bit for bit, and the reason is a compute-type argument three layers
below their public API, documented nowhere in their k-means page. See the
`bitwise-gbdt` tree; this is a second incumbent with a device-dependent
number system.

**Consequence for our numerics table.** The GEMM is NUMERIC, unambiguously.
Tile shape changes the summation order over `k`, so a block-size row here
moves bits. That is exactly the classification mistake
`mojo_only/numerics.mojo` warns about, and it is why `TILE` below is a fixed
constant and not something the capability matrix picks per backend.

The kernel is the ordinary shared-memory tiled product. It is not trying to
be cuBLAS; it is trying to be a correct and honest baseline so that the first
measurement is of the ALGORITHM. When the algorithm is validated and the
number exists, replacing this with a real tiled/vectorized product is the
first optimization, and it is the one place where per-backend paths are
obviously worth having.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


# Fixed, and fixed deliberately: see the docstring. Changing it changes the
# summation order over the feature axis and therefore the answer.
comptime GEMM_TILE = 16


def gemm_nt_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] . y[n x k]^T`, all row major.

    This is the shape k-means wants and the shape their cuBLAS call is
    configured for: X is `n_samples x n_features`, centroids are
    `n_clusters x n_features`, and the product wanted is every sample against
    every centroid, so the second operand is transposed and neither matrix
    has to be materialized in another layout.

    Both operands are read through shared tiles so each element of X is
    loaded once per centroid TILE rather than once per centroid.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)

    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var row = Int(block_idx.y) * GEMM_TILE + ty
    var col = Int(block_idx.x) * GEMM_TILE + tx

    var s_x = stack_allocation[
        GEMM_TILE * GEMM_TILE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_y = stack_allocation[
        GEMM_TILE * GEMM_TILE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var acc = Float32(0.0)

    var tile = 0
    while tile * GEMM_TILE < k:
        var kx = tile * GEMM_TILE + tx
        var ky = tile * GEMM_TILE + tx

        # X tile: row of X, feature kx.
        if row < m and kx < k:
            s_x[ty * GEMM_TILE + tx] = x.unsafe_load(row * k + kx)
        else:
            s_x[ty * GEMM_TILE + tx] = Float32(0.0)

        # Y tile: centroid (block_idx.x * TILE + ty), feature ky. Loaded
        # transposed into shared so the inner loop reads it by feature.
        var y_row = Int(block_idx.x) * GEMM_TILE + ty
        if y_row < n and ky < k:
            s_y[ty * GEMM_TILE + tx] = y.unsafe_load(y_row * k + ky)
        else:
            s_y[ty * GEMM_TILE + tx] = Float32(0.0)

        barrier()

        for i in range(GEMM_TILE):
            acc += s_x[ty * GEMM_TILE + i] * s_y[tx * GEMM_TILE + i]

        barrier()
        tile += 1

    if row < m and col < n:
        z.unsafe_store(row * n + col, acc)
