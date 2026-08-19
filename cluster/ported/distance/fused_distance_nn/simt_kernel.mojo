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

THE CROSS-THREAD MERGE IS NOW THEIRS. THE CROSS-BLOCK ONE STILL IS NOT
-----------------------------------------------------------------------
`rowEpilog_lambda` does two separate things and only the second one is
unportable. Keeping them apart is the whole content of this section.

1. **The intra-warp merge, `simt_kernel.cuh:119-130`.** The `P::AccThCols`
   threads that share a row hold partial `(value, key)` minima, and they are
   combined with `raft::shfl` on the key AND on the value. **That is now
   ported.** It used to be a shared-memory transpose plus a serial 16-way
   scan, justified by a claim that Mojo has no lane primitives. **The claim
   was false** (`PORTING.md` 2): `std.gpu.primitives.warp` has the shuffles,
   and `block.min` was never the answer here because it reduces VALUES ONLY
   and this reduction has to carry the key that achieved the minimum.

   One detail is not literal. Their `raft::shfl(x, lid + j, P::AccThCols)`
   ROTATES within a width-`AccThCols` subgroup, relying on CUDA's `width`
   argument to apply the modulo; their own comment at `simt_kernel.cuh:123`
   says so ("the shfl op applies the modulo internally"). Mojo's
   `shuffle_idx` has no width parameter, so the same coverage is obtained
   with `shuffle_xor`, which Modular's own GPU guide names as THE reduction
   primitive. XOR with an offset below
   `AccThCols` flips only the low lane bits, so it stays inside the same
   aligned `AccThCols` group their rotate stays inside, and both leave EVERY
   lane of the group holding the group's winner. The results are identical,
   not merely equivalent, because the reducer is a min over a TOTAL order and
   is therefore associative, commutative and idempotent: no reduction shape
   can change the answer.

   The total order is the one part of this that is OURS. Their fused
   comparator `KVPMinReduceImpl`
   (`detail/fused_distance_nn/helper_structs.cuh:28-33`) compares the VALUE
   only and keeps the shuffled partner on a tie, so their fused answer is
   shuffle-shape dependent; this tree uses the tie-break from their UNFUSED
   `Reducer` (`unfused_distance_nn.cuh:44-49`) in both arms so the two can be
   diffed against each other. That was already true of the shared-memory
   merge this replaced, and it is load-bearing rather than tidiness
   (`PORTING.md` 14).

   Two things this relies on, both true here: `GEMM_ACC_TH_COLS` is a power of
   two no larger than the lane width, and `tc = tid % GEMM_ACC_TH_COLS`, so
   the threads sharing a row are CONTIGUOUS and aligned and never straddle a
   warp boundary. Change either and the shuffle silently reduces the wrong
   set.

2. **The cross-block merge, `updateReducedVal` (`simt_kernel.cuh:25-47`).**
   Still replaced, and for a reason that has nothing to do with Mojo. It
   serializes per-row global updates with a mutex array and `atomicCAS`;
   their own comment calls it a workaround for pre-Volta hang behavior and
   says a 64-bit `atomicCAS` would eliminate it. We make the column axis a
   grid-stride loop INSIDE the block and launch one block per row tile, so
   every row is owned by exactly one block and there is nothing to serialize.
   Their kernel grid-strides both axes and therefore needs one. This is the
   `replaced` row in `PORTED_MAP.tsv`, and it costs parallelism when `n` is
   large and `m` is small, which is the opposite of every shape in this
   repository.

   What survives from `updateReducedVal` is its LANE STRUCTURE: the lanes it
   lets write are `lid == j * P::AccThCols`, the first lane of each group,
   each writing its own `P::AccRowsPerTh` rows. That is `tc == 0` below.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives.warp import shuffle_xor
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

    The row epilogue is a warp shuffle, so `block_dim` is not free: it must be
    `GEMM_THREADS`, and `GEMM_ACC_TH_COLS` must stay a power of two no larger
    than the hardware lane width. See the module docstring, part 1.
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
    # The cross-thread merge at the end needs NO shared memory: it is a warp
    # shuffle, `simt_kernel.cuh:125-126`. The 64 x 16 pair of scratch arrays
    # that used to sit here cost 8 KB of the 32 KB Apple allows a threadgroup,
    # which the two GEMM pages already spend 18 KB of.

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

    # --- `rowEpilog_lambda`, `simt_kernel.cuh:119-130` -------------------
    # Merge the GEMM_ACC_TH_COLS column-threads that share each row, one
    # butterfly pass per accumulator row, key and value shuffled together.
    # The comparator is NOT literally theirs, and this is the one place in
    # this kernel where that is deliberate. `KVPMinReduceImpl`
    # (`detail/fused_distance_nn/helper_structs.cuh:28-33`) is
    # `b.value < a.value ? b : a`, VALUE ONLY, so on a tie their shuffled
    # partner wins and the answer depends on the shuffle shape. This tree
    # uses the total order from their UNFUSED `Reducer`
    # (`unfused_distance_nn.cuh:44-49`) in both kernels instead:
    #     (a.value < b.value) || (a.value == b.value && a.key < b.key)
    # That is what makes the fused and unfused arms diffable against each
    # other, and it is what makes any reduction shape return the same pair.
    # It was already the rule in the shared-memory merge this replaced.
    #
    # Every thread reaches every shuffle: the loop bound is a comptime
    # constant and nothing here is conditional. That is required, because a
    # lane that skips a full-mask shuffle hangs the lanes that reach it.
    var lane_offset = 1
    while lane_offset < GEMM_ACC_TH_COLS:
        var o0v = shuffle_xor(val0, UInt32(lane_offset))
        var o0k = shuffle_xor(key0, UInt32(lane_offset))
        if o0v < val0 or (o0v == val0 and o0k < key0):
            val0 = o0v
            key0 = o0k

        var o1v = shuffle_xor(val1, UInt32(lane_offset))
        var o1k = shuffle_xor(key1, UInt32(lane_offset))
        if o1v < val1 or (o1v == val1 and o1k < key1):
            val1 = o1v
            key1 = o1k

        var o2v = shuffle_xor(val2, UInt32(lane_offset))
        var o2k = shuffle_xor(key2, UInt32(lane_offset))
        if o2v < val2 or (o2v == val2 and o2k < key2):
            val2 = o2v
            key2 = o2k

        var o3v = shuffle_xor(val3, UInt32(lane_offset))
        var o3k = shuffle_xor(key3, UInt32(lane_offset))
        if o3v < val3 or (o3v == val3 and o3k < key3):
            val3 = o3v
            key3 = o3k

        lane_offset *= 2

    # `updateReducedVal`'s lane structure without its mutex: the first lane
    # of each column group writes the `GEMM_ACC_ROWS_PER_TH` rows that group
    # owns. Every lane holds the winner after the butterfly, so `tc == 0` is
    # a choice of writer and not a reduction step.
    if tc == 0:
        for i in range(GEMM_ACC_ROWS_PER_TH):
            var row2 = m0 + tr * GEMM_ACC_ROWS_PER_TH + i
            if row2 < m:
                var best_v = val0
                var best_k = key0
                if i == 1:
                    best_v = val1
                    best_k = key1
                elif i == 2:
                    best_v = val2
                    best_k = key2
                elif i == 3:
                    best_v = val3
                    best_k = key3
                if is_sqrt_in != 0:
                    if best_v <= Float32(0.0):
                        best_v = Float32(0.0)
                    best_v = sqrt(best_v)
                out_value.unsafe_store(row2, best_v)
                out_key.unsafe_store(row2, best_k)
