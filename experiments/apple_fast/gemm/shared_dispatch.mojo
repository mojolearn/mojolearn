# SPDX-License-Identifier: Apache-2.0
"""Default-OFF shared adapter for the two unfused FAST Apple SDK entrances.

No buffers allocated, no synchronization, no arithmetic on the host. The
probe-only runtime selector/counters require serial calls and never exist
in a normal production build. Route IDs: core NT=0, core Gram=1,
vendor NN=2, vendor NT=3. Unsupported/degenerate products retain callers.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.ffi import _Global
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from experiments.apple_fast.gemm.mma import apple_gemm_experiment

comptime G1 = is_defined["MOJOLEARN_APPLE_FAST_SHARED_GEMM_G1"]()
comptime G5 = is_defined["MOJOLEARN_APPLE_FAST_SHARED_GEMM_G5"]()
comptime AUDIT = is_defined["MOJOLEARN_APPLE_FAST_SHARED_GEMM_AUDIT"]()
comptime ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]() and TARGET_COLUMN == COLUMN_APPLE and (G1 or G5 or AUDIT)

struct AuditState(Defaultable, Movable):
    var arm: Int
    var counts: InlineArray[Int, 16]
    def __init__(out self):
        self.arm = 0
        self.counts = InlineArray[Int, 16](fill=0)

comptime STATE = _Global[StorageType=AuditState, name="AppleSharedGemmAudit", init_fn=AuditState.__init__]

def select_audit_arm(arm: Int) raises:
    comptime if not AUDIT:
        raise Error("shared GEMM audit selector not compiled")
    if arm != 0 and arm != 1 and arm != 5:
        raise Error("invalid shared GEMM audit arm")
    STATE.get_or_create_ptr()[].arm = arm

def audit_count(route: Int, column: Int) raises -> Int:
    if route < 0 or route > 3 or column < 0 or column > 3:
        raise Error("invalid shared GEMM counter")
    return STATE.get_or_create_ptr()[].counts[route * 4 + column]

def try_shared_gemm[TRANSPOSE_B: Bool, ROUTE: Int](
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int,
) raises -> Bool:
    comptime assert not (G1 and G5), "select exactly one shared GEMM candidate"
    comptime assert not (AUDIT and (G1 or G5)), "audit selector is isolated from production selection"
    comptime assert ROUTE >= 0 and ROUTE < 4
    comptime if not ENABLED:
        return False
    else:
        # n=1 retains existing GEMV; K=0 never enters a TileTensor here.
        var eligible = m > 0 and n > 1 and k > 0
        eligible = eligible and max(m, max(n, k)) <= 2147483647
        eligible = eligible and max(m * n, max(m * k, n * k)) <= 2147483647
        var arm = 5 if G5 else 1
        comptime if AUDIT:
            arm = STATE.get_or_create_ptr()[].arm
        if not eligible or arm == 0:
            comptime if AUDIT:
                STATE.get_or_create_ptr()[].counts[ROUTE * 4] += 1
            return False
        # Inputs may alias each other; no change to caller output ownership.
        if arm == 1:
            apple_gemm_experiment[64, 64, 16, False, 0, TRANSPOSE_B](ctx, dst, a, b, m, n, k)
        else:
            apple_gemm_experiment[64, 64, 16, True, 0, TRANSPOSE_B](ctx, dst, a, b, m, n, k)
        comptime if AUDIT:
            STATE.get_or_create_ptr()[].counts[ROUTE * 4 + (1 if arm == 1 else 2)] += 1
            STATE.get_or_create_ptr()[].counts[ROUTE * 4 + 3] += 1
        return True
