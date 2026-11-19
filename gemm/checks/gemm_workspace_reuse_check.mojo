# SPDX-License-Identifier: Apache-2.0
"""Two queued products share scratch; grow then shrink it, and test small-buffer fallback."""
from max.gpu.host import DeviceContext
from gemm.checks.gemm_identical import (
    GemmWorkspace, GEMM_REUSE_GROUP_WS, GEMM_KPACK_RPT, GEMM_KPACK_CPT,
    GEMM_KPACK_KS, GEMM_KPACK_FS, GEMM_KPACK_PAD, GEMM_KPACK_ALIGN,
    TUNED_TC, _kpack_run, gemm_default_ksplit_leaves,
    identical_gemm_shipped_into, identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_step_arms import (
    gemm_step_fill, gemm_step_poison, gemm_step_readback,
    gemm_step_compare, gemm_step_digest,
)


def run_checks() raises:
    comptime assert GEMM_REUSE_GROUP_WS, "this check must exercise reusable grouped scratch"
    var ctx = DeviceContext()
    var ws = GemmWorkspace(ctx)
    var prior = 1
    for shape in range(3):
        var m = 512
        var n = 512
        var k = 769
        if shape == 1:
            m = 513
            n = 769
            k = 1537
        elif shape == 2:
            # The shrink shape (fewer leaf groups than shape 0). It was
            # k = 257 until 2026-11-19: since c98e7d4a6a (2026-10-01) the
            # NVIDIA column steps a short chain (k <= GEMM_TILE_SHORT_K, 512)
            # with under 192 blocks of 128x128 down to the 64x64 tile, all
            # leaves in one launch and no group scratch, so the shipped
            # dispatch never asked for grouped scratch there and the check
            # stopped as blind on its first NVIDIA run. A k between the
            # short-chain and the long-chain (4096) step-down gates keeps the
            # 128x128 plan and its grouped scratch, like shapes 0 and 1.
            k = 641
        var mn = m*n
        var gl = gemm_default_ksplit_leaves(m,n,k)
        var required = identical_gemm_workspace_max_floats(m,n,k)
        if gl <= 0 or required <= 1:
            raise Error("blind shape: grouped workspace not selected (shape="
                        + String(shape) + " " + String(m) + "x" + String(n) + "x" + String(k)
                        + " group_leaves=" + String(gl) + " required=" + String(required) + ")")
        var a = ctx.enqueue_create_buffer[DType.float32](m*k)
        var b = ctx.enqueue_create_buffer[DType.float32](k*n)
        var c0 = ctx.enqueue_create_buffer[DType.float32](mn)
        var c1 = ctx.enqueue_create_buffer[DType.float32](mn)
        var ref0 = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var ref1 = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var got0 = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var got1 = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var tiny = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.synchronize()
        gemm_step_fill(ctx,a,m*k,31+shape,False)
        gemm_step_fill(ctx,b,k*n,43+shape,False)
        # Independent legacy reference, unchanged allocating group runner.
        for op in range(2):
            _kpack_run[GEMM_KPACK_RPT,GEMM_KPACK_CPT,TUNED_TC,GEMM_KPACK_KS,
                GEMM_KPACK_FS,False,GEMM_KPACK_PAD,GEMM_KPACK_ALIGN,0,True,True](
                ctx,c0,a,b,m,n,k,op,gl)
            if op == 0:
                gemm_step_readback(ctx,c0,ref0)
            else:
                gemm_step_readback(ctx,c0,ref1)
        gemm_step_poison(ctx,c0,got0,mn)
        gemm_step_poison(ctx,c1,got1,mn)
        # No host fence between the two calls. Different outputs and op kinds
        # make reusing scratch before its fold observable in the first result.
        ws.run(ctx,c0,a,b,m,n,k,0)
        ws.run(ctx,c1,a,b,m,n,k,1)
        gemm_step_readback(ctx,c0,got0)
        gemm_step_readback(ctx,c1,got1)
        var cmp0 = gemm_step_compare(got0,ref0,mn)
        var cmp1 = gemm_step_compare(got1,ref1,mn)
        print("WORKSPACE shape=" + String(shape) + " required=" + String(required)
              + " retained=" + String(len(ws.buffer)) + " group_leaves=" + String(gl))
        if cmp0[0] != 0 or cmp0[1] != 0 or cmp1[0] != 0 or cmp1[1] != 0:
            raise Error("workspace output mismatch: first=" + String(cmp0[2])
                        + " poison=" + String(cmp0[1]))
        print("MATCH shape=" + String(shape) + " op=0 reference=" + hex(gemm_step_digest(ref0,mn))
              + " reused=" + hex(gemm_step_digest(got0,mn)))
        print("MATCH shape=" + String(shape) + " op=1 reference=" + hex(gemm_step_digest(ref1,mn))
              + " reused=" + hex(gemm_step_digest(got1,mn)))
        if len(ws.buffer) < required or len(ws.buffer) < prior:
            raise Error("workspace growth/shrink capacity")
        prior = len(ws.buffer)
        gemm_step_poison(ctx,c0,got0,mn)
        identical_gemm_shipped_into(ctx,c0,a,b,tiny,m,n,k,0)
        gemm_step_readback(ctx,c0,got0)
        var fallback = gemm_step_compare(got0,ref0,mn)
        if fallback[0] != 0 or fallback[1] != 0:
            raise Error("small workspace fallback mismatch")
        print("MATCH small-workspace shape=" + String(shape) + " reference=" + hex(gemm_step_digest(ref0,mn))
              + " fallback=" + hex(gemm_step_digest(got0,mn)))
    ctx.synchronize()
    _ = ws
    print("WORKSPACE_REUSE_PASS")
