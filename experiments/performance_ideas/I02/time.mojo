# SPDX-License-Identifier: Apache-2.0
"""Full GEMM call timing including fresh versus retained scratch ownership."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_fill,gemm_step_env_int
from gemm.checks.gemm_identical import GemmWorkspace,identical_gemm_workspace_max_floats
from gemm.experiments.bounded_workspace import BoundedGemmWorkspace

def main() raises:
    var ctx=DeviceContext()
    var m=gemm_step_env_int("AB_M",1024);var n=gemm_step_env_int("AB_N",1024);var k=gemm_step_env_int("AB_K",2048)
    var a=ctx.enqueue_create_buffer[DType.float32](m*k);var b=ctx.enqueue_create_buffer[DType.float32](n*k);var out=ctx.enqueue_create_buffer[DType.float32](m*n)
    gemm_step_fill(ctx,a,m*k,17,False);gemm_step_fill(ctx,b,n*k,29,False);ctx.synchronize()
    var retained=BoundedGemmWorkspace(ctx,identical_gemm_workspace_max_floats(m,n,k))
    for arm in range(2):
        for phase in range(2):
            var start=perf_counter_ns()
            if arm==0:
                var fresh=GemmWorkspace(ctx)
                fresh.run[False](ctx,out,a,b,m,n,k,0)
                ctx.synchronize()
                _=fresh^
            else:
                retained.run[False](ctx,out,a,b,m,n,k,0)
                ctx.synchronize()
            print("MEASURE id=I02 arm="+String(arm)+" phase="+String(phase)+" m="+String(m)+" n="+String(n)+" k="+String(k)+" elapsed_ns="+String(perf_counter_ns()-start))
    retained.close(ctx)
