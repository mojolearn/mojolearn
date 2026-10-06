# SPDX-License-Identifier: Apache-2.0
"""Session-owned bounded scratch. Stream ownership is the caller's contract.

Calls above the declared retained budget use a temporary allocation and wait
for its last consumer before release. This intentionally charges the wait to
the call instead of disguising it as asynchronous lifetime management.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import GemmWorkspace, identical_gemm_workspace_max_floats

# I02 experiment: NEVER RUN — PENDING MEASUREMENT; incumbent defaults retained.
# I02 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit caller-owned experiment workspace; no default dispatcher admission.
struct BoundedGemmWorkspace(Movable):
    var workspace: GemmWorkspace
    var max_retained_floats: Int
    var oversized_calls: Int

    def __init__(out self, ctx: DeviceContext, max_retained_floats: Int) raises:
        if max_retained_floats < 1:
            raise Error("retained scratch budget must be positive")
        self.workspace = GemmWorkspace(ctx)
        self.max_retained_floats = max_retained_floats
        self.oversized_calls = 0

    def run[allow_vendor: Bool = True](mut self, ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32], m: Int,n: Int,k: Int,op: Int) raises:
        if identical_gemm_workspace_max_floats(m,n,k) > self.max_retained_floats:
            self.oversized_calls += 1
            var temporary = GemmWorkspace(ctx)
            temporary.run[allow_vendor](ctx,c,a,b,m,n,k,op)
            ctx.synchronize()
            _ = temporary^
            return
        self.workspace.run[allow_vendor](ctx,c,a,b,m,n,k,op)

    def close(mut self, ctx: DeviceContext) raises:
        # Explicit completion protects outstanding folds before releasing the
        # session's high-water buffer. Safe even after a failed caller operation.
        ctx.synchronize()
        self.workspace = GemmWorkspace(ctx)
