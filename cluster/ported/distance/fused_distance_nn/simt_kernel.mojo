"""Distance and argmin FUSED, so the distance matrix is never written.

PORT OF `cuvs/src/distance/detail/fused_distance_nn/simt_kernel.cuh` at cuVS
`2140532c`, built on their `linalg/contractions.cuh` policy. Partial.
Do not improve.

WHY THIS EXISTS AND THE UNFUSED PATH WAS NOT ENOUGH
---------------------------------------------------
`unfused_distance_nn.mojo` computes the whole `m x n` distance tile into
memory and then reduces it. This kernel never writes it. Each thread holds a
running `(value, key)` minimum for each of the `AccRowsPerTh` rows it owns,
in REGISTERS, and updates it as each column tile's accumulator is completed:

    KVPair val[P::AccRowsPerTh];        // simt_kernel.cuh:78

That is the design, not a tuning of it. At k-means' shape, 200,000 rows and
16 clusters, the unfused path writes 3.2 million floats to global memory and
reads them straight back, every single Lloyd iteration, to extract one
minimum per row. This kernel does the same arithmetic and moves none of it.

**I ported the unfused arm first and justified it by their own selector
preferring unfused on Blackwell, which is true and was not the point.** The
fused arm is what runs on the hardware most people have, and its CUTLASS
version is unportable but THIS one is not: `simt_kernel.cuh` is the SIMT
path, plain CUDA cores, no tensor-core instructions anywhere.

THE ONE THING THAT DID NOT PORT, AND WHAT REPLACED IT
-----------------------------------------------------
Their cross-block merge, `updateReducedVal`, serializes per-row updates with
`raft::laneId()`, a mutex array and `atomicCAS` (`simt_kernel.cuh:25-47`).
Their own comment calls it a workaround for pre-Volta hang behavior and says
a 64-bit `atomicCAS` would eliminate it. Mojo 1.0 has no lane primitives.

Replaced by making the column axis a grid-stride loop INSIDE the block and
launching one block row-tile: every row is then owned by exactly one block,
so there is no cross-block reduction and no mutex at all. Their kernel
grid-strides both axes and therefore needs one. This is a `replaced` row in
`PORTED_MAP.tsv`, and it costs parallelism when `n` is large and `m` is
small, which is the opposite of every shape in this repository.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.gemm import (
    GEMM_ACC_COLS_PER_TH,
    GEMM_ACC_ROWS_PER_TH,
    GEMM_ACC_TH_COLS,
    GEMM_ACC_TH_ROWS,
    GEMM_KBLK,
    GEMM_MBLK,
    GEMM_NBLK,
    GEMM_SMEM_PAGE_X,
    GEMM_SMEM_PAGE_Y,
    GEMM_SMEM_STRIDE,
    GEMM_THREADS,
)


comptime FUSED_MAX = Float32(3.4028234663852886e38)


def fused_distance_nn_kernel(
    out_key: MutPointer[UInt32, MutAnyOrigin],
    out_value: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    xn: MutPointer[Float32, MutAnyOrigin],
    yn: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    is_sqrt_in: Int32,
):
    """`fusedDistanceNNkernel` with the L2-expanded op and a min reduce.

    One block per row tile, grid-striding the column axis internally.
    Launch `grid_dim = (1, ceil(m / GEMM_MBLK), 1)`,
    `block_dim = (GEMM_THREADS, 1, 1)`.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)

    var tid = Int(thread_idx.x)
    var tr = tid // GEMM_ACC_TH_COLS
    var tc = tid % GEMM_ACC_TH_COLS
    var m0 = Int(block_idx.y) * GEMM_MBLK

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
    # The cross-thread merge at the end: 64 rows x 16 column-threads.
    var s_val = stack_allocation[
        GEMM_MBLK * GEMM_ACC_TH_COLS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_key = stack_allocation[
        GEMM_MBLK * GEMM_ACC_TH_COLS,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()

    # `KVPair val[P::AccRowsPerTh]`, in registers. This is the whole point.
    var val0 = FUSED_MAX
    var val1 = FUSED_MAX
    var val2 = FUSED_MAX
    var val3 = FUSED_MAX
    var key0 = UInt32(0xFFFFFFFF)
    var key1 = UInt32(0xFFFFFFFF)
    var key2 = UInt32(0xFFFFFFFF)
    var key3 = UInt32(0xFFFFFFFF)

    var n0 = 0
    while n0 < n:
        var acc0 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc1 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc2 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc3 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)

        var kt = 0
        while kt < k:
            var e = tid
            while e < GEMM_MBLK * GEMM_KBLK:
                var r = e // GEMM_KBLK
                var c = e % GEMM_KBLK
                var v = Float32(0.0)
                if m0 + r < m and kt + c < k:
                    v = x.unsafe_load((m0 + r) * k + kt + c)
                sx[r * GEMM_SMEM_STRIDE + c] = v
                e += GEMM_THREADS
            e = tid
            while e < GEMM_NBLK * GEMM_KBLK:
                var r2 = e // GEMM_KBLK
                var c2 = e % GEMM_KBLK
                var v2 = Float32(0.0)
                if n0 + r2 < n and kt + c2 < k:
                    v2 = y.unsafe_load((n0 + r2) * k + kt + c2)
                sy[r2 * GEMM_SMEM_STRIDE + c2] = v2
                e += GEMM_THREADS
            barrier()

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

        # --- THE EPILOGUE, in registers. `l2_exp_distance_op` then the
        # min reduce, applied to this column tile before it is discarded.
        for j in range(GEMM_ACC_COLS_PER_TH):
            var col = n0 + tc * GEMM_ACC_COLS_PER_TH + j
            if col >= n:
                continue
            var ynv = yn.unsafe_load(col)

            for i in range(GEMM_ACC_ROWS_PER_TH):
                var row = m0 + tr * GEMM_ACC_ROWS_PER_TH + i
                if row >= m:
                    continue
                var dot = acc0[j]
                if i == 1:
                    dot = acc1[j]
                elif i == 2:
                    dot = acc2[j]
                elif i == 3:
                    dot = acc3[j]
                var d = xn.unsafe_load(row) + ynv - Float32(2.0) * dot
                if d <= Float32(0.0):
                    d = Float32(0.0)

                # Strict `<`, so the lowest column wins a tie, matching
                # `unfused_distance_nn.mojo` and their Reducer's total order.
                if i == 0:
                    if d < val0:
                        val0 = d
                        key0 = UInt32(col)
                elif i == 1:
                    if d < val1:
                        val1 = d
                        key1 = UInt32(col)
                elif i == 2:
                    if d < val2:
                        val2 = d
                        key2 = UInt32(col)
                else:
                    if d < val3:
                        val3 = d
                        key3 = UInt32(col)
        n0 += GEMM_NBLK

    # --- merge the 16 column-threads that share each row -----------------
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var local_row = tr * GEMM_ACC_ROWS_PER_TH + i
        var v = val0
        var kk2 = key0
        if i == 1:
            v = val1
            kk2 = key1
        elif i == 2:
            v = val2
            kk2 = key2
        elif i == 3:
            v = val3
            kk2 = key3
        s_val[local_row * GEMM_ACC_TH_COLS + tc] = v
        s_key[local_row * GEMM_ACC_TH_COLS + tc] = kk2
    barrier()

    var lr = tid
    while lr < GEMM_MBLK:
        var row2 = m0 + lr
        if row2 < m:
            var best_v = s_val[lr * GEMM_ACC_TH_COLS + 0]
            var best_k = s_key[lr * GEMM_ACC_TH_COLS + 0]
            for t in range(1, GEMM_ACC_TH_COLS):
                var cv = s_val[lr * GEMM_ACC_TH_COLS + t]
                var ck = s_key[lr * GEMM_ACC_TH_COLS + t]
                if cv < best_v or (cv == best_v and ck < best_k):
                    best_v = cv
                    best_k = ck
            if is_sqrt_in != 0:
                if best_v <= Float32(0.0):
                    best_v = Float32(0.0)
                best_v = sqrt(best_v)
            out_value.unsafe_store(row2, best_v)
            out_key.unsafe_store(row2, best_k)
        lr += GEMM_THREADS
