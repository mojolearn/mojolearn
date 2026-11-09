# SPDX-License-Identifier: Apache-2.0
"""The scoped Apple kernel's shared lower-triangle subtract (F02) for
MOJOLEARN_CHOL_FAST_SHARED_SUB (Apple FAST, default OFF, opt-in).

Restored 2026-11-09 by lane r1-restore-measured-winners (AGENTS.md "A measured
winner is never deleted on reasoning"; Apple FAST work is paused, not deleted):
the bodies below are f59d913b0b's experiments/apple_fast/gemm/scoped_dispatch.mojo
(`SharedSubAudit`, `shared_sub_record`, `shared_sub_count`, `scoped_kernel`),
copied verbatim with `git show`. The rest of that file (the scoped G1/G2
adapter family: MOJOLEARN_SCOPED_GEMM_G1_TALL, _G1_DENSE, _G1_GRAM, _G2_NARROW,
_SPLIT, _PCA, _AUDIT, MOJOLEARN_LEGACY_NARROW_SCOPED_GEMM) stays deleted:
never measured (docs/TOMBSTONES.md, lane k1). Its original header follows.

Opt-in strided G1/G2 adapter. No allocation, sync or host numerical work.
Fragment math copied from compiled catalog fa39073607e3b19c4fbfe063a5545a63318f13c2.
"""
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from max.gpu.host import DeviceContext
from std.gpu import block_idx, lane_id, warp_id
from std.atomic import Atomic
from std.ffi import _Global

comptime FPtr = MutPointer[Float32, MutAnyOrigin]


struct SharedSubAudit(Defaultable, Movable):
    var lu: Int
    var chol: Int
    def __init__(out self):
        self.lu = 0
        self.chol = 0
comptime SUB_STATE = _Global[StorageType=SharedSubAudit, name="SharedSubtractAuditV1", init_fn=SharedSubAudit.__init__]

def shared_sub_record(cholesky: Bool) raises:
    if cholesky:
        SUB_STATE.get_or_create_ptr()[].chol += 1
    else:
        SUB_STATE.get_or_create_ptr()[].lu += 1

def shared_sub_count(cholesky: Bool) raises -> Int:
    return SUB_STATE.get_or_create_ptr()[].chol if cholesky else SUB_STATE.get_or_create_ptr()[].lu


def scoped_kernel[BM: Int, BN: Int, SPLIT: Bool, SUBTRACT: Bool = False, LOWER: Bool = False](
    dst: FPtr, a: FPtr, b: FPtr,
    m_in: Int32, n_in: Int32, k_in: Int32,
    a_si_in: Int32, a_sp_in: Int32, b_sp_in: Int32, b_sj_in: Int32, per_in: Int32,
    dst_stride_in: Int32 = Int32(0),
):
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var per = Int(per_in)
    comptime assert not (SPLIT and SUBTRACT), "subtract requires an unsplit accumulation"
    comptime assert BM % 16 == 0 and BN % 16 == 0
    comptime RM = BM // 16
    comptime RN = BN // 16
    var lane = Int(lane_id())
    var sg = Int(warp_id())
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var cols = (n + BN - 1) // BN
    var bm = (Int(block_idx.x) // cols) * BM
    var bn = (Int(block_idx.x) % cols) * BN
    comptime if LOWER:
        if bn >= bm + BM:
            return  # uniform whole-tile rejection, no partial warp exit
    var sr = (sg // 2) * (BM // 2)
    var sc = (sg % 2) * (BN // 2)
    var first = 0
    var last = k
    comptime if SPLIT:
        first = Int(block_idx.y) * per
        last = min(k, first + per)
    var acc = InlineArray[SIMD[DType.float32, 2], RM * RN](fill=SIMD[DType.float32, 2](0))
    for kb in range(first, last, 16):
        for ks in range(0, 16, 8):
            var af = InlineArray[SIMD[DType.float32, 2], RM](fill=SIMD[DType.float32, 2](0))
            var bf = InlineArray[SIMD[DType.float32, 2], RN](fill=SIMD[DType.float32, 2](0))
            comptime for mi in range(RM):
                var row = bm + sr + mi * 8 + fr
                var fragment = SIMD[DType.float32, 2](0)
                comptime for s in range(2):
                    var kk = kb + ks + fc + s
                    if row < m and kk < last:
                        fragment[s] = a.unsafe_load(row * a_si + kk * a_sp)
                af[mi] = fragment
            comptime for ni in range(RN):
                var fragment = SIMD[DType.float32, 2](0)
                comptime for s in range(2):
                    var col = bn + sc + ni * 8 + fc + s
                    var kk = kb + ks + fr
                    if col < n and kk < last:
                        fragment[s] = b.unsafe_load(kk * b_sp + col * b_sj)
                bf[ni] = fragment
            comptime for mi in range(RM):
                comptime for ni in range(RN):
                    var previous = acc[mi * RN + ni]
                    _mma_apple_8x8(acc[mi * RN + ni], af[mi], bf[ni], previous)
    comptime for mi in range(RM):
        comptime for ni in range(RN):
            var fragment = acc[mi * RN + ni]
            comptime for s in range(2):
                var row = bm + sr + mi * 8 + fr
                var col = bn + sc + ni * 8 + fc + s
                if row < m and col < n and (not LOWER or col <= row):
                    comptime if SPLIT:
                        _ = Atomic.fetch_add(dst.unsafe_offset(row * n + col), fragment[s])
                    elif SUBTRACT:
                        var cell = row * Int(dst_stride_in) + col
                        dst.unsafe_store(cell, dst.unsafe_load(cell) - fragment[s])
                    else:
                        dst.unsafe_store(row * n + col, fragment[s])


