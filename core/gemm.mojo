"""The matrix product. One call to MAX's tuned matmul, plus RAFT's policy.

WHAT IS LEFT IN THIS FILE, AND WHY THE KERNEL IS NOT
-----------------------------------------------------
Where cuVS and cuML call cuBLAS for a STANDALONE matrix product, they call a
library with no source, so there is nothing to port and `linalg.matmul` is the
faithful mirror. That is the Gram product PCA, truncated SVD and OLS need, and
those three go through `gemm_tn` here.

**It is not the rule for a distance step.** `VENDOR_LIBS.md` opens with why:
their dispatch for pairwise distance under an argmin or a top-k does not call
cuBLAS at all, it calls a FUSED kernel that never materializes the distance
matrix, and a device-wide matmul cannot be fused into anything. Callers whose
product is an intermediate inside a reduction belong on the fused kernel
(`cluster/ported/distance/fused_distance_nn/simt_kernel.mojo`,
`neighbors/.../fused_l2_knn.mojo`), not here.

A STANDALONE register-tiled contraction used to sit here as well, a port of
`raft/linalg/contractions.cuh::KernelPolicy` and the load/accumulate structure
of `raft/linalg/detail/contractions.cuh` at RAFT `9aa17e5`, unfused and
writing its product to memory. It is DELETED. Nothing called it once OLS's
step 6 went to `gemv_gpu`, it was measured at about 15 GFLOP/s against
`linalg.matmul`'s 248 at 1M x 128, and the FUSED instantiation of the same
policy, which is the one RAFT's dispatch actually runs, lives in
`simt_kernel.mojo` and is untouched.

If a later round wants the standalone contraction back, take it from
`simt_kernel.mojo` or from git history rather than rewriting it.

THEIR POLICY IS STILL HERE, BECAUSE THE FUSED KERNEL NEEDS IT
--------------------------------------------------------------
`KernelPolicy<float, Veclen=4, Kblk=32, AccRowsPerTh=4, AccColsPerTh=4,
AccThRows=16, AccThCols=16>`, RAFT's `Policy4x4<float>` and the one their float
distance kernels instantiate:

    Nthreads = AccThRows * AccThCols    = 256
    Mblk     = AccRowsPerTh * AccThRows = 64
    Nblk     = AccColsPerTh * AccThCols = 64
    SmemStride = Kblk + Veclen          = 36   (padding, not a rounding)

These constants stay because `cluster/ported/distance/fused_distance_nn/
simt_kernel.mojo` is instantiated at this policy and its callers compute their
launch geometry from it. That kernel is NOT a substitution candidate under any
version of the rule: RAFT fuses the argmin epilogue into the contraction and
never writes the distance tile, which is a thing no BLAS call can do and is
the whole point of `fusedDistanceNN`.

`SmemStride = Kblk + Veclen` is theirs and is not arbitrary: the padding
staggers each row's start so that threads reading down a column of shared
memory do not all land in the same bank.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext


# `KernelPolicy<float, 4, 32, 4, 4, 16, 16>`, RAFT's Policy4x4<float>.
comptime GEMM_VECLEN = 4
comptime GEMM_KBLK = 32
comptime GEMM_ACC_ROWS_PER_TH = 4
comptime GEMM_ACC_COLS_PER_TH = 4
comptime GEMM_ACC_TH_ROWS = 16
comptime GEMM_ACC_TH_COLS = 16

comptime GEMM_THREADS = GEMM_ACC_TH_ROWS * GEMM_ACC_TH_COLS
comptime GEMM_MBLK = GEMM_ACC_ROWS_PER_TH * GEMM_ACC_TH_ROWS
comptime GEMM_NBLK = GEMM_ACC_COLS_PER_TH * GEMM_ACC_TH_COLS
comptime GEMM_SMEM_STRIDE = GEMM_KBLK + GEMM_VECLEN
comptime GEMM_SMEM_PAGE_X = GEMM_SMEM_STRIDE * GEMM_MBLK
comptime GEMM_SMEM_PAGE_Y = GEMM_SMEM_STRIDE * GEMM_NBLK


def gemm_nt(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[m x k] . y[n x k]^T` through MAX's tuned matmul.

    `transpose_b=True` is exactly the shape every algorithm here wants: rows
    against centroids, index points or candidates, with neither operand ever
    materialized in another layout.

    **`n == 1` GOES TO GEMV, BECAUSE `transpose_b=True` DOES NOT WRITE THERE.**
    Measured 2026-08-19 through this wrapper (`mojo_only/
    vendor_correctness_check.mojo`): m=64, n=1, k=32, output poisoned before
    the call, **63 of the 64 rows still held the poison afterwards**. Nothing
    was written wrong; 63 rows were not written at all, so a caller reusing a
    buffer reads whatever was in it last time. That is worse than zeros, and
    `VENDOR_LIBRARIES.md` used to say "returns zeros for some outputs", which
    is why this was mistaken for a benign edge case for as long as it was.

    The fault is `transpose_b=True`, not `n == 1`: the same product with
    `transpose_b=False` is correct at the identical shape, which is what the
    sabotage in that check established.

    `n == 1` is not exotic. It is a one-column design matrix in `lstsq.mojo`
    and `pca.mojo`, a single index point in `knn_brute_force.mojo`, and in
    `min_cluster_distance_compute.mojo:197` it is any run where
    `n_clusters % centroid_batch == 1` -- a REMAINDER, so ordinary cluster
    counts reach it.

    Routing to `gemv_n` needs no reshaping and is what RAFT does for the same
    degenerate case: at `n == 1` this product IS a matrix-vector product, `y`
    is already a contiguous k-vector whether it is read as `(1, k)` or
    `(k, 1)`, and `z` is already a contiguous m-vector. `gemv_n` wraps
    `gemv_gpu`, which tests correct at every m from 1 to 100,003.
    """
    if n == 1:
        gemv_n(ctx, z, x, y, m, k)
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_tn(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T . x[k x n]`, ON MAX'S TUNED MATMUL.

    **This route was abandoned twice and both times for a bad reason.**
    First because `matmul` refuses `transpose_a`; then because
    `linalg.transpose` compiles and signals on device buffers. Neither of
    those blocks the actual identity:

        Xt = transpose(X)      Xt . Xt^T == X^T X

    which is the N-T shape MAX does support. The only missing piece was a
    device transpose, and `core/column_stats.mojo::transpose_kernel` is
    twenty lines.

    WHY IT MATTERS, MEASURED. Scaling the benchmark to 1M x 128 put k-NN at
    2.70x over scikit-learn while PCA fell to 0.12x and OLS to 1.64x. k-NN
    goes through MAX's matmul; PCA and OLS went through our hand-written
    contraction. Same machine, same round: about 248 GFLOP/s on the vendor
    kernel against about 15 on ours. Register tiling and split-K narrowed
    that gap and did not close it, and the honest reading is that a tuned
    matmul is not something to reimplement when one ships.

    `transpose_a` is STILL unsupported and this does not use it. The transpose
    is two extra passes over `k x m` floats against a product that is
    `O(k * m * n)`, and it buys the tuned kernel.
    """
    from core.column_stats import TRANSPOSE_TILE, transpose_kernel

    # Xt and Xt2 are two copies because `matmul` refuses one buffer as two
    # mutable arguments (PORTING.md 24), and `Xt . Xt^T` names it twice.
    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            (k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    ctx.enqueue_function[transpose_kernel](
        xt2.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            (k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    ctx.synchronize()
    gemm_nt(ctx, z, xt, xt2, m, n, k)
    ctx.synchronize()


# `gemm_tn` IS ON THE VENDOR PATH. `transpose_a` IS THE THING THAT IS NOT.
#
# `raft::stats::cov`, `lstsqEig`'s first step and cuML's `tsvd_fit` all ask
# cuBLAS for `CUBLAS_OP_T, CUBLAS_OP_N`: the Gram shape `A^T A`, contracting
# down the ROW axis. Handing that shape to MAX's matmul directly fails to
# compile, and the constraint is still live:
#
#     max/kernels/src/linalg/matmul/__init__.mojo:110:9:
#     note: constraint failed: transpose_a not yet supported
#
# What does NOT follow, and was believed here for two rounds, is that the Gram
# shape therefore needs a hand-written kernel. `transpose(X) . transpose(X)^T`
# is the same matrix and is the N-T shape, so one twenty-line transpose puts
# the T-N users on the tuned matmul too. `gemm_tn` above does exactly that,
# and the ported column-major contraction it replaced is deleted.
#
# The N-T route is still the one to prefer where a caller can choose it: it
# skips the transpose entirely.

from linalg.gemv import gemv_gpu


def gemv_n(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m] = x[m x k] . y[k]`, on MAX's tuned GEMV.

    `raft::linalg::gemv` is what RAFT calls for `w <- covA Ab`, and
    `linalg.gemv.gemv` is HOST-ONLY (no ctx, no target). `gemv_gpu` is the
    GPU sibling. See VENDOR_LIBRARIES.md.

    Orientation read from MAX's `gemv.mojo`, not guessed: `GemmShape.get`
    takes m and n from `c` and k from `a`, ignoring `b` entirely, so
    `c = (m, 1)`, `a = (m, k)`, `b = (k, 1)`. `transpose_b` must be False;
    its True arm swaps a and b and at n=1 would write one output instead of
    m. `b` is `(k, 1)` and not `(1, k)` because the MATMUL_NAIVE fallback in
    the same dispatcher indexes B as `(k, n)`.

    REACH PROVED BY SABOTAGE, 2026-08-19. A green OLS run does NOT show this
    is reached: `w <- inv Ab` is 8x8 in the checks, and the scale-invariance
    check in particular would pass under any wrong-but-linear kernel. So the
    orientation was flipped to `transpose_b=True` and the run failed with
    `coefficient 1 = 0.0` — only the first output written, exactly what that
    arm predicts at n=1, since it swaps a and b and passes `(n, m, k)`.
    Reverted; that failure is the evidence.
    """
    var tz = TileTensor(z, row_major(m, Int(1)))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(k, Int(1)))
    gemv_gpu[transpose_b=False](tz, tx, ty, ctx)
