# SPDX-License-Identifier: Apache-2.0
"""Opt-in G2 for the actual multiclass linear_fwd only. UNVALIDATED.

No change to GEMV, transpose, bias, optimizer, precision or buffer lifecycle.
No shape window (2026-10-04): every contiguous NT call the kernel can run
correctly is eligible. Only kernel limits remain (see softmax_gemm_nt).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import _Global
from std.sys.compile import is_defined
from gemm.afn_apple_fast import AFN_GEMM_APPLE
from core.gemm import gemm_nt
from experiments.apple_fast.gemm.scoped_dispatch import scoped_kernel

# SOURCE-READY / UNBUILT, 2026-10-04, source4bfc1424a: default OFF.
# No actual softmax quality or timing result exists for this algorithm flag.
# Resident G2 32768x8x220 1.203750->0.865625ms is a matrix probe only;
# scoped-r2-all-q-v1 PASS covers the shared kernel, not this optimizer caller.
# Require M2 A/B compile, actual fit/predict reach and all zero-regression
# gates in docs/apple-fast/ab/softmax-g2-narrow.md before timing/promotion.
# NEVER RUN — PENDING VALIDATION. New candidate remains opt-in/default OFF.
comptime SOFTMAX_G2 = AFN_GEMM_APPLE and is_defined["MOJOLEARN_SOFTMAX_FAST_G2_NARROW"]()
comptime SOFTMAX_AUDIT = AFN_GEMM_APPLE and is_defined["MOJOLEARN_SOFTMAX_G2_AUDIT"]()
# LEGACY, default OFF: the old window admitted only M>=4096, C 2..16, D 128..512,
# which brackets the board (istella-like 220 features, 8 classes). Removed as
# benchmark-tuned on 2026-10-04; the window-free replacement is UNMEASURED.
comptime SOFTMAX_LEGACY_WINDOW = is_defined["MOJOLEARN_LEGACY_NARROW_SOFTMAX_G2"]()

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
        # Kernel limits only. scoped_kernel[32, 32, False] bounds-checks every
        # row, column and K index, so any M, C, D >= 1 is correct (one launch
        # replaces one launch, so no launch-amortization floor is needed).
        # Its shape arguments and strides are Int32, so every operand extent
        # must fit in Int32.
        var eligible = m >= 1 and n >= 1 and k >= 1
        comptime if SOFTMAX_LEGACY_WINDOW:
            eligible = eligible and m >= 4096 and n >= 2 and n <= 16 and k >= 128 and k <= 512
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
                Int32(m), Int32(n), Int32(k), Int32(k), Int32(1), Int32(1), Int32(k), Int32(k), Int32(n),
                grid_dim=(((m + 31) // 32) * ((n + 31) // 32), 1, 1), block_dim=(128, 1, 1),
            )
            return
    gemm_nt(ctx, dst, x, weights, m, n, k)
