# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""UNCOMPILED experiments: full-fp32 Apple 8x8 MMA, direct or shared tiles.

Explicit entry only. No timing, allocation, waits, or production dispatch.
The fragment ABI is documented in upstream Modular's Apple matmul_8x8.
"""
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx, lane_id, warp_id
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


# Four simdgroups arranged 2x2; each owns a disjoint register rectangle.
def apple_mma_kernel[
    BM: Int, BN: Int, BK: Int, STAGED: Bool, PAD: Int, TRANSPOSE_B: Bool,
](
    out: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, n_in: Int32, k_in: Int32,
):
    comptime assert BM % 16 == 0 and BN % 16 == 0
    comptime assert BK > 0 and BK % 8 == 0
    comptime assert PAD >= 0
    comptime RM = BM // 16
    comptime RN = BN // 16
    comptime STRIDE = BK + PAD
    comptime PAGE_A = BM * STRIDE
    comptime PAGE_B = BN * STRIDE
    comptime assert not STAGED or (PAGE_A + PAGE_B) * 4 <= 32768
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var sg = Int(warp_id())
    # Two adjacent columns in one fragment row belong to this lane.
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var bm = Int(block_idx.y) * BM
    var bn = Int(block_idx.x) * BN
    var sr = (sg // 2) * (BM // 2)
    var sc = (sg % 2) * (BN // 2)
    var acc = Array[SIMD[DType.float32, 2], RM * RN](fill=SIMD[DType.float32, 2](0))
    # Direct variants reserve one float, not a full shared tile.
    comptime SHARED_FLOATS = PAGE_A + PAGE_B if STAGED else 1
    var tile = stack_allocation[
        SHARED_FLOATS, Float32, address_space=AddressSpace.SHARED,
    ]()
    for kb in range(0, k, BK):
        comptime if STAGED:
            # Contiguous global reads for A and NT B. NN B is staged in its
            # physical row order then transposed on write to shared memory.
            for index in range(tid, BM * BK, 128):
                var row = index // BK
                var kk = index % BK
                var value = Float32(0)
                if bm + row < m and kb + kk < k:
                    value = a.unsafe_load((bm + row) * k + kb + kk)
                tile.unsafe_store(row * STRIDE + kk, value)
            for index in range(tid, BN * BK, 128):
                comptime if TRANSPOSE_B:
                    var col = index // BK
                    var kk = index % BK
                    var value = Float32(0)
                    if bn + col < n and kb + kk < k:
                        value = b.unsafe_load((bn + col) * k + kb + kk)
                    tile.unsafe_store(PAGE_A + col * STRIDE + kk, value)
                else:
                    var kk = index // BN
                    var col = index % BN
                    var value = Float32(0)
                    if bn + col < n and kb + kk < k:
                        value = b.unsafe_load((kb + kk) * n + bn + col)
                    tile.unsafe_store(PAGE_A + col * STRIDE + kk, value)
            barrier()
        for ks in range(0, BK, 8):
            var af = Array[SIMD[DType.float32, 2], RM](uninitialized=True)
            var bf = Array[SIMD[DType.float32, 2], RN](uninitialized=True)
            comptime for mi in range(RM):
                var row = sr + mi * 8 + fr
                var fragment = SIMD[DType.float32, 2](0)
                comptime for s in range(2):
                    comptime if STAGED:
                        fragment[s] = tile.unsafe_load(row * STRIDE + ks + fc + s)
                    else:
                        if bm + row < m and kb + ks + fc + s < k:
                            fragment[s] = a.unsafe_load((bm + row) * k + kb + ks + fc + s)
                af[mi] = fragment
            comptime for ni in range(RN):
                var fragment = SIMD[DType.float32, 2](0)
                comptime for s in range(2):
                    var col = sc + ni * 8 + fc + s
                    comptime if STAGED:
                        fragment[s] = tile.unsafe_load(PAGE_A + col * STRIDE + ks + fr)
                    else:
                        if bn + col < n and kb + ks + fr < k:
                            comptime if TRANSPOSE_B:
                                fragment[s] = b.unsafe_load((bn + col) * k + kb + ks + fr)
                            else:
                                fragment[s] = b.unsafe_load((kb + ks + fr) * n + bn + col)
                bf[ni] = fragment
            comptime for mi in range(RM):
                comptime for ni in range(RN):
                    var previous = acc[mi * RN + ni]
                    _mma_apple_8x8(acc[mi * RN + ni], af[mi], bf[ni], previous)
        comptime if STAGED:
            # Every simdgroup finishes all reads before any thread refills.
            barrier()
    comptime for mi in range(RM):
        comptime for ni in range(RN):
            var fragment = acc[mi * RN + ni]
            comptime for s in range(2):
                var row = bm + sr + mi * 8 + fr
                var col = bn + sc + ni * 8 + fc + s
                if row < m and col < n:
                    out.unsafe_store(row * n + col, fragment[s])


def apple_gemm_experiment[
    BM: Int = 64, BN: Int = 64, BK: Int = 32,
    STAGED: Bool = True, PAD: Int = 0, TRANSPOSE_B: Bool = True,
](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int,
) raises:
    """Explicit resident-buffer entry: C=A B^T (or A B for TRANSPOSE_B=False).

    Buffers must be contiguous; output must not alias either input. Inputs
    may alias one another for Gram products. Caller keeps buffers alive.
    Raises outside Apple FAST. K tails are zero-filled, including K=0.
    """
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_FAST:
        raise Error("Apple MMA experiments require NUMERIC_FAST")
    else:
        comptime if not has_apple_gpu_accelerator():
            raise Error("Apple MMA experiments require an Apple GPU target")
        else:
            if m < 0 or n < 0 or k < 0:
                raise Error("Apple MMA experiments require nonnegative extents")
            if m > 2147483647 or n > 2147483647 or k > 2147483647:
                raise Error("Apple MMA experiment extent exceeds Int32")
            if len(out) < m * n or len(a) < m * k or len(b) < n * k:
                raise Error("Apple MMA experiment buffer is smaller than its shape")
            if m == 0 or n == 0:
                return
            ctx.enqueue_function[apple_mma_kernel[BM, BN, BK, STAGED, PAD, TRANSPOSE_B]](
                out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
                Int32(m), Int32(n), Int32(k),
                grid_dim=((n + BN - 1) // BN, (m + BM - 1) // BM, 1),
                block_dim=(128, 1, 1),
            )
