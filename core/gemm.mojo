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

    var acc = stack_allocation[
        GEMM_ACC_ROWS_PER_TH * GEMM_ACC_COLS_PER_TH,
        Scalar[DType.float32],
    ]()
    for i in range(GEMM_ACC_ROWS_PER_TH * GEMM_ACC_COLS_PER_TH):
        acc[i] = Float32(0.0)

    var regx = stack_allocation[
        GEMM_ACC_ROWS_PER_TH, Scalar[DType.float32]
    ]()
    var regy = stack_allocation[
        GEMM_ACC_COLS_PER_TH, Scalar[DType.float32]
    ]()

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
        for kk in range(GEMM_KBLK):
            for i in range(GEMM_ACC_ROWS_PER_TH):
                regx[i] = sx[
                    (tr * GEMM_ACC_ROWS_PER_TH + i) * GEMM_SMEM_STRIDE + kk
                ]
            for j in range(GEMM_ACC_COLS_PER_TH):
                regy[j] = sy[
                    (tc * GEMM_ACC_COLS_PER_TH + j) * GEMM_SMEM_STRIDE + kk
                ]
            for i in range(GEMM_ACC_ROWS_PER_TH):
                for j in range(GEMM_ACC_COLS_PER_TH):
                    acc[i * GEMM_ACC_COLS_PER_TH + j] += regx[i] * regy[j]

        barrier()
        kt += GEMM_KBLK

    for i in range(GEMM_ACC_ROWS_PER_TH):
        var gr3 = m0 + tr * GEMM_ACC_ROWS_PER_TH + i
        if gr3 >= m:
            continue
        for j in range(GEMM_ACC_COLS_PER_TH):
            var gc3 = n0 + tc * GEMM_ACC_COLS_PER_TH + j
            if gc3 < n:
                z.unsafe_store(
                    gr3 * n + gc3, acc[i * GEMM_ACC_COLS_PER_TH + j]
                )
