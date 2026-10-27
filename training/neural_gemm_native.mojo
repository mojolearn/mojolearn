# SPDX-License-Identifier: Apache-2.0
# Shared native owner; public Python door and provider use identical transport.
from core.neural_context import neural_ctx
from core.device_scan import device_first_nonfinite
from core.hostptr import f32_ptr
from gemm.neural_dispatch import identical_gemm_into,identical_gemm_workspace_max_floats
from gemm.host_transport import GemmHostLease,ROLE_A,ROLE_B,ROLE_C,ROLE_WS,gemm_up_f32,gemm_down_f32


def neural_gemm_native(aa: Int, ba: Int, ca: Int, shape: Tuple[Int, Int, Int, Int]) raises -> Int:
    var ap = f32_ptr(aa)
    var bp = f32_ptr(ba)
    var cp = f32_ptr(ca)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var ctx = neural_ctx["MojoNeuralTrainingContextIdentical"]()
    var lease = GemmHostLease()
    var a = lease.f32(ctx,ROLE_A,m*k)
    var b = lease.f32(ctx,ROLE_B,n*k)
    var c = lease.f32(ctx,ROLE_C,m*n)
    var ws = lease.f32(ctx,ROLE_WS,identical_gemm_workspace_max_floats(m,n,k))
    try:
        gemm_up_f32(ctx,lease,a,ap,m*k)
        gemm_up_f32(ctx,lease,b,bp,n*k)
        if device_first_nonfinite(ctx,a,m*k)>=0 or device_first_nonfinite(ctx,b,n*k)>=0:
            raise Error("SmallMLPTrainer GEMM refuses nonfinite inputs")
        identical_gemm_into(ctx,c,a,b,ws,m,n,k,shape[3])
        if device_first_nonfinite(ctx,c,m*n)>=0:
            raise Error("RuntimeError: SmallMLPTrainer GEMM returned a nonfinite result")
        gemm_down_f32(ctx,lease,c,cp,m*n)
    except error:
        ctx.synchronize()
        lease.release()
        raise error
    lease.release()
    _ = a
    _ = b
    _ = c
    _ = ws
    return m*n
