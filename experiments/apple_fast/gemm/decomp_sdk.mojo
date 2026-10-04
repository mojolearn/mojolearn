# SPDX-License-Identifier: Apache-2.0
"""Default-OFF SDK control for caller-owned contiguous decomp buffers.

No allocations, copies, synchronization, fusion or transpose materialization.
The caller selects this only when its original AFN plan is non-split.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from std.sys.compile import is_defined
from std.ffi import _Global
from gemm.afn_apple_fast import AFN_GEMM_APPLE

comptime SDK_ON = AFN_GEMM_APPLE and is_defined["MOJOLEARN_DECOMP_SDK_NNNT"]()
comptime SDK_AUDIT = AFN_GEMM_APPLE and is_defined["MOJOLEARN_DECOMP_SDK_AUDIT"]()
comptime SDK_INTERPOSE = SDK_ON or SDK_AUDIT

struct SdkAudit(Defaultable, Movable):
    var counts: InlineArray[Int, 3]
    var metadata: InlineArray[Int, 8]
    def __init__(out self):
        self.counts = InlineArray[Int, 3](fill=0)
        self.metadata = InlineArray[Int, 8](fill=0)

comptime STATE = _Global[StorageType=SdkAudit, name="DecompSdkAuditV1", init_fn=SdkAudit.__init__]

def sdk_count(index: Int) raises -> Int:
    if index < 0 or index >= 3:
        raise Error("SDK audit count index")
    return STATE.get_or_create_ptr()[].counts[index]

def sdk_metadata(index: Int) raises -> Int:
    if index < 0 or index >= 8:
        raise Error("SDK audit metadata index")
    return STATE.get_or_create_ptr()[].metadata[index]

def try_decomp_sdk(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    m: Int, k: Int, n: Int, ta: Bool, tb: Bool, split_plan: Bool,
) raises -> Bool:
    comptime if not SDK_INTERPOSE:
        return False
    else:
        var eligible = m > 0 and n > 1 and k > 0 and not ta and not split_plan
        eligible = eligible and len(a) >= m * k and len(b) >= k * n and len(dst) >= m * n
        eligible = eligible and dst.unsafe_ptr() != a.unsafe_ptr() and dst.unsafe_ptr() != b.unsafe_ptr()
        var selected = SDK_ON and eligible
        comptime if SDK_AUDIT:
            var state = STATE.get_or_create_ptr()
            state[].counts[0 if not selected else (2 if tb else 1)] += 1
            state[].metadata[0] = m
            state[].metadata[1] = n
            state[].metadata[2] = k
            state[].metadata[3] = Int(ta)
            state[].metadata[4] = Int(tb)
            state[].metadata[5] = Int(split_plan)
            state[].metadata[6] = Int(eligible)
            state[].metadata[7] = Int(selected)
        if not selected:
            return False
        var tc = TileTensor(dst, row_major(m, n))
        var tx = TileTensor(a, row_major(m, k))
        if tb:
            var ty = TileTensor(b, row_major(n, k))
            matmul[transpose_b=True, target="gpu"](tc, tx, ty, ctx)
        else:
            var ty = TileTensor(b, row_major(k, n))
            matmul[transpose_b=False, target="gpu"](tc, tx, ty, ctx)
        return True
