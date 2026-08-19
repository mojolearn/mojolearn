"""The matrix product, register-tiled, ported from RAFT's contraction policy.

PORT OF `raft/linalg/contractions.cuh::KernelPolicy` and the load/accumulate
structure of `raft/linalg/detail/contractions.cuh` at RAFT `9aa17e5`.
Partial. Do not improve.

WHY THIS FILE WAS REWRITTEN, WHICH IS THE MORE USEFUL PART OF THIS DOCSTRING
----------------------------------------------------------------------------
The first version was a plain 16x16 tile with ONE output element per thread,
written from scratch. The reasoning was: cuVS and cuML call cuBLAS, cuBLAS is
closed, therefore there is nothing to port, therefore write the simplest
honest thing and optimize after the first measurement.

**The first half of that was right and the conclusion was wrong.** cuBLAS has
no source, but RAFT does not depend on cuBLAS for its distance kernels. It
ships `linalg/contractions.cuh`, a register-tiled double-buffered contraction
that every pairwise-distance kernel in RAFT and cuVS is built on, and it
contains no warp intrinsics and no tensor-core instructions at all. It was
portable and it was the thing to port.

The measurement that exposed this is `bench/results/FIRST_RUN_2026-08-19.md`:
we lost k-means, PCA and DBSCAN to scikit-learn on Apple Accelerate, and all
three are dominated by this file.

THEIR POLICY, COPIED
--------------------
`KernelPolicy<float, Veclen=4, Kblk=32, AccRowsPerTh=4, AccColsPerTh=4,
AccThRows=16, AccThCols=16>`, which is RAFT's `Policy4x4<float>` and the one
their float distance kernels instantiate:

    Nthreads = AccThRows * AccThCols   = 256
    Mblk     = AccRowsPerTh * AccThRows = 64
    Nblk     = AccColsPerTh * AccThCols = 64
    SmemStride = Kblk + Veclen         = 36   (padding, not a rounding)

**The whole idea is `AccRowsPerTh x AccColsPerTh`.** Each thread computes a
4x4 block of the output instead of one element, so the four X values and four
Y values it holds in registers each get used four times. That cuts shared
memory traffic per output by 4x and raises arithmetic intensity by the same
factor. It is why 256 threads now cover a 64x64 output tile where before they
covered 16x16.

`SmemStride = Kblk + Veclen` is theirs and is not arbitrary: the padding
staggers each row's start so that threads reading down a column of shared
memory do not all land in the same bank.

NOT PORTED YET, and named so it does not get forgotten
------------------------------------------------------
**Double buffering.** Theirs keeps TWO shared-memory pages (`SmemPage`,
`pageWr`, `pageRd`) and issues the global loads for tile `t+1` into registers
while computing tile `t`, so the load latency hides behind the arithmetic.
Ours loads, barriers, computes, barriers. That is the single largest
remaining gap in this kernel.

It is deferred rather than skipped because two pages at this policy is 32 KB
of threadgroup memory, which is exactly Metal's limit (`PORTING.md 1`), so it
needs either a smaller `Kblk` or a check of what Metal actually permits. That
is a real decision and it deserves its own measurement.

**Vectorized loads.** `Veclen = 4` means their loads move four floats at a
time. Ours move one. `Veclen` still sets `SmemStride` here, so the padding is
theirs even though the load width is not yet.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


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

# Kept so call sites that computed a grid from a single tile size still read
# correctly. The output tile is MBLK x NBLK now, not TILE x TILE.
comptime GEMM_TILE = GEMM_MBLK


def gemm_nt_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] . y[n x k]^T`, all row major.

    Launch with `block_dim = (GEMM_THREADS, 1, 1)` and
    `grid_dim = (ceil(n / GEMM_NBLK), ceil(m / GEMM_MBLK), 1)`.

    This is the shape every algorithm here wants: X is `rows x features`, the
    second operand is `centroids`, `index points` or `candidates`, and the
    product is everything against everything, so the second operand is
    transposed and neither matrix is ever materialized in another layout.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)

    var tid = Int(thread_idx.x)
    var tr = tid // GEMM_ACC_TH_COLS
    var tc = tid % GEMM_ACC_TH_COLS

    var m0 = Int(block_idx.y) * GEMM_MBLK
    var n0 = Int(block_idx.x) * GEMM_NBLK

    var sx = stack_allocation[
        GEMM_SMEM_PAGE_X,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sy = stack_allocation[
        GEMM_SMEM_PAGE_Y,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # SIMD values, NOT `stack_allocation`. Their `regx[][]` and `acc[][]` are
    # plain C arrays that nvcc keeps in registers; `stack_allocation` without
    # an address space is thread-local MEMORY, so the first attempt turned
    # every accumulator access into a load and a store and made the kernel
    # SLOWER than the naive one it replaced. See PORTING.md 26.
    var acc0 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc1 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc2 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc3 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)

    var kt = 0
    while kt < k:
        # --- ldgXY + stsXY. `LdgPerThX = Mblk * LdgThRow / Nthreads` loads
        # per thread; expressed here as a flat strided sweep of the tile.
        var e = tid
        while e < GEMM_MBLK * GEMM_KBLK:
            var r = e // GEMM_KBLK
            var c = e % GEMM_KBLK
            var gr = m0 + r
            var gc = kt + c
            var v = Float32(0.0)
            if gr < m and gc < k:
                v = x.unsafe_load(gr * k + gc)
            sx[r * GEMM_SMEM_STRIDE + c] = v
            e += GEMM_THREADS

        e = tid
        while e < GEMM_NBLK * GEMM_KBLK:
            var r2 = e // GEMM_KBLK
            var c2 = e % GEMM_KBLK
            var gr2 = n0 + r2
            var gc2 = kt + c2
            var v2 = Float32(0.0)
            if gr2 < n and gc2 < k:
                v2 = y.unsafe_load(gr2 * k + gc2)
            sy[r2 * GEMM_SMEM_STRIDE + c2] = v2
            e += GEMM_THREADS
        barrier()

        # --- ldsXY + accumulate. The four-by-four block is the point: each
        # register value is reused AccColsPerTh (or AccRowsPerTh) times.
        var xb = tr * GEMM_ACC_ROWS_PER_TH
        var yb = tc * GEMM_ACC_COLS_PER_TH
        for kk in range(GEMM_KBLK):
            var regy = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
            for j in range(GEMM_ACC_COLS_PER_TH):
                regy[j] = sy[(yb + j) * GEMM_SMEM_STRIDE + kk]
            acc0 += sx[(xb + 0) * GEMM_SMEM_STRIDE + kk] * regy
            acc1 += sx[(xb + 1) * GEMM_SMEM_STRIDE + kk] * regy
            acc2 += sx[(xb + 2) * GEMM_SMEM_STRIDE + kk] * regy
            acc3 += sx[(xb + 3) * GEMM_SMEM_STRIDE + kk] * regy

        barrier()
        kt += GEMM_KBLK

    for i in range(GEMM_ACC_ROWS_PER_TH):
        var gr3 = m0 + tr * GEMM_ACC_ROWS_PER_TH + i
        if gr3 >= m:
            continue
        for j in range(GEMM_ACC_COLS_PER_TH):
            var gc3 = n0 + tc * GEMM_ACC_COLS_PER_TH + j
            if gc3 < n:
                var v3 = acc0[j]
                if i == 1:
                    v3 = acc1[j]
                elif i == 2:
                    v3 = acc2[j]
                elif i == 3:
                    v3 = acc3[j]
                z.unsafe_store(gr3 * n + gc3, v3)


# ---------------------------------------------------------------------------
# The vendor-library path. Prefer this.
#
# cuVS and cuML call cuBLAS for their matrix products. cuBLAS has no source to
# port, and the first version of this file drew the wrong conclusion from that
# and hand-wrote a kernel. **The faithful mirror of "they call a tuned vendor
# BLAS" is to call OURS**, and MAX ships one: `linalg.matmul`, targeting the
# GPU, with `transpose_a` and `transpose_b` and an `elementwise_lambda_fn`
# epilogue hook.
#
# `gemm_nt_kernel` above stays as the ported RAFT contraction. It is the
# reference these wrappers are checked against, and it is what runs if a
# backend ever lacks a tuned matmul.
# ---------------------------------------------------------------------------

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext


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
    """
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
    """`z[m x n] = x[k x m]^T . x[k x n]`, the Gram shape.

    **STILL OFF THE VENDOR PATH, AND NOW ON A REGISTER-TILED CONTRACTION
    RATHER THAN A NAIVE ONE.** The vendor route was built, compiled, and
    CRASHED, and that is worth more than the attempt.

    The identity was right: MAX refuses `transpose_a`, but
    `Xt . Xt^T == X^T X`, so one transpose turns the unsupported T-N shape
    into the N-T shape MAX does support. It compiled. It then died inside

        linalg::transpose::_copy_with_strides[...] rank=2, dtype=f32

    with a signal, because that is a HOST strided-copy path being handed
    DEVICE pointers. `linalg.transpose` takes an `Optional[DeviceContext]`
    and accepting one is not the same as dispatching on it.

    So `transpose` joins `matmul`'s `transpose_a` and `matmul` at `n = 1` on
    the list of vendor calls that exist, compile, and do not do the job here.
    All three are in `VENDOR_LIBRARIES.md` with what was tried.

    THE TRANSPOSE IS NO LONGER THE PLAN, AND THAT IS THE NEWS
    ---------------------------------------------------------
    The other branch was to register-tile `covariance_kernel` directly, and
    it turned out not to need a transpose at all: RAFT already ships the
    column-major arm of its contraction, and a row-major `X` read as a
    column-major `X^T` puts this product in exactly that arm. That is done.
    `xt` and `xt2` are now dead weight in this signature; they are kept only
    so the three call sites do not have to change in the same commit, and
    they are the next thing to delete.

    THE GEOMETRY LIVES HERE, AND ONLY HERE
    ---------------------------------------
    Nothing outside this function should compute a grid for
    `covariance_kernel`. The launch is

        block_dim = (COV_THREADS, 1, 1)                      = (256, 1, 1)
        grid_dim  = (ceil(n / COV_NBLK), ceil(m / COV_MBLK), splits)

    and `splits > 1` means `covariance_kernel` writes `splits` partial
    matrices instead of one and `covariance_reduce_kernel` sums them.

    WHY SPLIT-K, spelled out because it is the part that is not a port
    ------------------------------------------------------------------
    The output is `m x n = n_cols x n_cols` and `k = n_rows`. Register tiling
    grows the output tile from 16x16 to 64x64, and at `n_cols = 32`, which is
    what `bench/bench_main.mojo` runs, that is ONE thread block for a 200,000
    row contraction where the naive kernel at least had four. The tiling would
    have been a regression on its own. Partitioning the ROW axis across
    `grid_dim.z` is where the parallelism comes from, and it is what cuBLAS
    does for a T-N with tiny `m`, `n` and enormous `k`.

    The split count is chosen to fill the grid and then capped so a split is
    never shorter than `COV_SPLIT_MIN_ROWS` rows, below which the fixed cost
    of a block outweighs the work in it. Any split count is CORRECT: the
    kernel re-derives its own row range from `grid_dim.z`, so this policy can
    be tuned without touching the kernel.
    """
    from core.column_stats import (
        COV_MBLK,
        COV_NBLK,
        COV_SPLIT_MIN_ROWS,
        COV_TARGET_BLOCKS,
        COV_THREADS,
        covariance_kernel,
        covariance_reduce_kernel,
    )

    var tiles_x = (n + COV_NBLK - 1) // COV_NBLK
    var tiles_y = (m + COV_MBLK - 1) // COV_MBLK
    var tiles = tiles_x * tiles_y

    var splits = (COV_TARGET_BLOCKS + tiles - 1) // tiles
    var cap = k // COV_SPLIT_MIN_ROWS
    if splits > cap:
        splits = cap
    if splits < 1:
        splits = 1

    if splits == 1:
        ctx.enqueue_function[covariance_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(k),
            Int32(m),
            Float32(1.0),
            grid_dim=(tiles_x, tiles_y, 1),
            block_dim=(COV_THREADS, 1, 1),
        )
        return

    var cells = m * n
    var partials = ctx.enqueue_create_buffer[DType.float32](splits * cells)
    ctx.enqueue_function[covariance_kernel](
        partials.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        Float32(1.0),
        grid_dim=(tiles_x, tiles_y, splits),
        block_dim=(COV_THREADS, 1, 1),
    )
    ctx.enqueue_function[covariance_reduce_kernel](
        z.unsafe_ptr(),
        partials.unsafe_ptr(),
        Int32(cells),
        Int32(splits),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    # `partials` dies at the end of this scope and the reduction must be done
    # with it before that happens. One synchronize per fit, not per iteration.
    ctx.synchronize()


# WHY `gemm_tn` IS NOT ON THE VENDOR PATH, AND IT IS A HARD LIMIT
#
# `raft::stats::cov`, `lstsqEig`'s first step and cuML's `tsvd_fit` all ask
# cuBLAS for `CUBLAS_OP_T, CUBLAS_OP_N`: the Gram shape `A^T A`, contracting
# down the ROW axis. Routing those through MAX's matmul fails to compile:
#
#     max/kernels/src/linalg/matmul/__init__.mojo:110:9:
#     note: constraint failed: transpose_a not yet supported
#
# So the vendor path covers the N-T shape every distance computation wants
# and does NOT cover the T-N shape every covariance wants.
# `core/column_stats.mojo::covariance_kernel`, the hand-ported contraction,
# stays as the only implementation of that shape.
#
# That limit is unchanged. What changed is what sits behind it: that kernel
# is no longer the naive 16x16 one-element-per-thread tile that left PCA and
# OLS flat for six benchmark rounds. It is now RAFT's COLUMN-major
# contraction (`ColKernelPolicy`, the `isRowMajor == false` arm of
# `raft/linalg/detail/contractions.cuh`), which is the same policy this file
# ported for the N-T shape and which fits the T-N shape without a transpose
# at all. The transpose-plus-vendor-matmul route is therefore ABANDONED
# rather than pending: there is nothing left for it to buy.
#
# `VENDOR_LIBRARIES.md` still lists that transpose as the follow-up and now
# says something false. It is outside the scope of this change and has to be
# corrected before this lands.
