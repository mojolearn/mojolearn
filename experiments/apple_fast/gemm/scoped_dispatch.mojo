# SPDX-License-Identifier: Apache-2.0
"""Opt-in strided G1/G2 adapter. No allocation, sync or host numerical work.

Fragment math copied from compiled catalog fa39073607e3b19c4fbfe063a5545a63318f13c2.
Strides and caller-owned split plans are new and require independent quality.
Scope selection uses operation metadata only. No production default changes.
"""
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from max.gpu.host import DeviceContext
from std.gpu import block_idx, lane_id, warp_id
from std.atomic import Atomic
from std.ffi import _Global
from std.sys.compile import is_defined
from gemm.afn_apple_fast import AFN_GEMM_APPLE

comptime TALL = is_defined["MOJOLEARN_SCOPED_GEMM_G1_TALL"]()
comptime DENSE = is_defined["MOJOLEARN_SCOPED_GEMM_G1_DENSE"]()
comptime GRAM = is_defined["MOJOLEARN_SCOPED_GEMM_G1_GRAM"]()
comptime NARROW = is_defined["MOJOLEARN_SCOPED_GEMM_G2_NARROW"]()
comptime SPLITS = is_defined["MOJOLEARN_SCOPED_GEMM_SPLIT"]()
comptime PCA = is_defined["MOJOLEARN_SCOPED_GEMM_PCA"]()
comptime AUDIT = is_defined["MOJOLEARN_SCOPED_GEMM_AUDIT"]()
comptime ENABLED = AFN_GEMM_APPLE and (TALL or DENSE or GRAM or NARROW or AUDIT)
comptime FPtr = MutPointer[Float32, MutAnyOrigin]

struct ScopeAudit(Defaultable, Movable):
    var counts: InlineArray[Int, 9]
    var last: InlineArray[Int, 15]
    def __init__(out self):
        self.counts = InlineArray[Int, 9](fill=0)
        self.last = InlineArray[Int, 15](fill=0)

comptime STATE = _Global[StorageType=ScopeAudit, name="ScopedGemmAuditV1", init_fn=ScopeAudit.__init__]

def scoped_count(route: Int, arm: Int) raises -> Int:
    if route < 0 or route > 2 or arm < 0 or arm > 2:
        raise Error("invalid scoped GEMM count")
    return STATE.get_or_create_ptr()[].counts[route * 3 + arm]

def scoped_last(index: Int) raises -> Int:
    if index < 0 or index >= 15:
        raise Error("invalid scoped GEMM metadata index")
    return STATE.get_or_create_ptr()[].last[index]


def scoped_kernel[BM: Int, BN: Int, SPLIT: Bool](
    dst: FPtr, a: FPtr, b: FPtr,
    m: Int, n: Int, k: Int, a_si: Int, a_sp: Int, b_sp: Int, b_sj: Int, per: Int,
):
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
                if row < m and col < n:
                    comptime if SPLIT:
                        _ = Atomic.fetch_add(dst.unsafe_offset(row * n + col), fragment[s])
                    else:
                        dst.unsafe_store(row * n + col, fragment[s])


def try_scoped_gemm[SPLIT: Bool, ROUTE: Int](
    ctx: DeviceContext, dst: FPtr, a: FPtr, b: FPtr,
    m: Int, n: Int, k: Int, a_si: Int, a_sp: Int, b_sp: Int, b_sj: Int,
    splits: Int, per: Int,
) raises -> Bool:
    comptime assert ROUTE >= 0 and ROUTE < 3
    comptime if not ENABLED:
        return False
    else:
        var nn = a_si == k and a_sp == 1 and b_sp == n and b_sj == 1
        var nt = a_si == k and a_sp == 1 and b_sp == 1 and b_sj == k
        var tn = a_si == 1 and a_sp == m and b_sp == n and b_sj == 1
        var aliased_inputs = a == b
        var arm = 0
        # Conservative experimental windows; no speed claim between measured points.
        if TALL and nn and m >= 4096 and n >= 32 and n <= 128 and k >= 128 and k <= 512:
            arm = 1
        if DENSE and nn and m >= 1024 and m <= 8192 and n >= 256 and n <= 768 and k >= 256 and k <= 768 and m >= 2 * n:
            arm = 1
        if GRAM and aliased_inputs and m == n and m >= 129 and m <= 1024 and k >= 128 and (nt or tn):
            arm = 1
        if NARROW and nt and m >= 4096 and n >= 2 and n <= 16 and k >= 128 and k <= 512:
            arm = 2
        if m <= 0 or n <= 1 or k <= 0 or dst == a or dst == b:
            arm = 0
        if max(m * n, max(m * k, n * k)) > 2147483647:
            arm = 0
        if splits < 1 or per < 1 or (SPLIT and per % 16 != 0):
            arm = 0
        comptime if SPLIT and not SPLITS:
            arm = 0
        comptime if ROUTE == 2 and not PCA:
            arm = 0
        comptime if AUDIT:
            var state = STATE.get_or_create_ptr()
            state[].counts[ROUTE * 3 + arm] += 1
            state[].last[0] = ROUTE
            state[].last[1] = arm
            state[].last[2] = m
            state[].last[3] = n
            state[].last[4] = k
            state[].last[5] = splits
            state[].last[6] = per
            state[].last[7] = Int(SPLIT)
            state[].last[8] = a_si
            state[].last[9] = a_sp
            state[].last[10] = b_sp
            state[].last[11] = b_sj
            state[].last[12] = Int(aliased_inputs)
            state[].last[13] = Int(nn)
            state[].last[14] = Int(nt)
        if arm == 1:
            ctx.enqueue_function[scoped_kernel[64, 64, SPLIT]](
                dst, a, b, m, n, k, a_si, a_sp, b_sp, b_sj, per,
                grid_dim=(((m + 63) // 64) * ((n + 63) // 64), splits, 1), block_dim=(128, 1, 1),
            )
        elif arm == 2:
            ctx.enqueue_function[scoped_kernel[32, 32, SPLIT]](
                dst, a, b, m, n, k, a_si, a_sp, b_sp, b_sj, per,
                grid_dim=(((m + 31) // 32) * ((n + 31) // 32), splits, 1), block_dim=(128, 1, 1),
            )
        return arm != 0
