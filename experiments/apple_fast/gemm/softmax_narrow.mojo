# SPDX-License-Identifier: Apache-2.0
"""Opt-in G2 for the actual multiclass linear_fwd only. UNVALIDATED.

No change to GEMV, transpose, bias, optimizer, precision or buffer lifecycle.
Shape window is a hypothesis around resident G2 32768x8x220, not a winner.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import _Global
from std.sys.compile import is_defined
from gemm.afn_apple_fast import AFN_GEMM_APPLE
from core.gemm import gemm_nt
from experiments.apple_fast.gemm.scoped_dispatch import scoped_kernel

comptime SOFTMAX_G2 = AFN_GEMM_APPLE and is_defined["MOJOLEARN_SOFTMAX_FAST_G2_NARROW"]()
comptime SOFTMAX_AUDIT = AFN_GEMM_APPLE and is_defined["MOJOLEARN_SOFTMAX_G2_AUDIT"]()

struct SoftmaxAudit(Defaultable, Movable):
    var counts: InlineArray[Int, 3]
    var last: InlineArray[Int, 5]
    def __init__(out self):
        self.counts = InlineArray[Int, 3](fill=0)
        self.last = InlineArray[Int, 5](fill=0)

comptime STATE = _Global[StorageType=SoftmaxAudit, name="SoftmaxG2AuditV1", init_fn=SoftmaxAudit.__init__]

def softmax_count(index: Int) raises -> Int:
    if index < 0 or index >= 3:
        raise Error("invalid softmax count")
    return STATE.get_or_create_ptr()[].counts[index]

def softmax_last(index: Int) raises -> Int:
    if index < 0 or index >= 5:
        raise Error("invalid softmax metadata")
    return STATE.get_or_create_ptr()[].last[index]

def softmax_reset() raises:
    var state = STATE.get_or_create_ptr()
    state[].counts = InlineArray[Int, 3](fill=0)
    state[].last = InlineArray[Int, 5](fill=0)


def softmax_gemm_nt(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32],
                   mut x: DeviceBuffer[DType.float32], mut weights: DeviceBuffer[DType.float32],
                   m: Int, n: Int, k: Int) raises:
    comptime if SOFTMAX_G2 or SOFTMAX_AUDIT:
        var eligible = m >= 4096 and n >= 2 and n <= 16 and k >= 128 and k <= 512
        eligible = eligible and max(m * n, max(m * k, n * k)) <= 2147483647
        eligible = eligible and dst.unsafe_ptr() != x.unsafe_ptr() and dst.unsafe_ptr() != weights.unsafe_ptr()
        eligible = eligible and len(dst) >= m * n and len(x) >= m * k and len(weights) >= n * k
        var selected = SOFTMAX_G2 and eligible
        comptime if SOFTMAX_AUDIT:
            var state = STATE.get_or_create_ptr()
            state[].counts[0] += Int(eligible)
            state[].counts[1] += Int(selected)
            state[].counts[2] += Int(not selected)
            state[].last[0], state[].last[1], state[].last[2] = m, n, k
            state[].last[3], state[].last[4] = Int(eligible), Int(selected)
        if selected:
            ctx.enqueue_function[scoped_kernel[32, 32, False]](
                dst.unsafe_ptr(), x.unsafe_ptr(), weights.unsafe_ptr(),
                Int32(m), Int32(n), Int32(k), Int32(k), Int32(1), Int32(1), Int32(k), Int32(k),
                grid_dim=(((m + 31) // 32) * ((n + 31) // 32), 1, 1), block_dim=(128, 1, 1),
            )
            return
    gemm_nt(ctx, dst, x, weights, m, n, k)
