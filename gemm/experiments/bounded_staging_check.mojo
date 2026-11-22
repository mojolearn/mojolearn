# SPDX-License-Identifier: Apache-2.0
"""Exact staging depth controls, leaf/tile tails and all orientations."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_NVIDIA,COLUMN_AMD
from gemm.checks.gemm_identical import PLAN_FLAT,identical_gemm_with_plan
from gemm.checks.gemm_step_arms import gemm_step_fill,gemm_step_poison,gemm_step_readback,gemm_step_compare,gemm_step_digest
from gemm.experiments.bounded_staging import bounded_staging_gemm


def run_checks() raises:
    var ctx=DeviceContext()
    for fixture in range(3):
        var m=17;var n=19;var k=129
        if fixture==1:m=65;n=67;k=513
        if fixture==2:m=33;n=1;k=1025
        var a=ctx.enqueue_create_buffer[DType.float32](m*k)
        var b=ctx.enqueue_create_buffer[DType.float32](n*k)
        var control=ctx.enqueue_create_buffer[DType.float32](m*n)
        var candidate=ctx.enqueue_create_buffer[DType.float32](m*n)
        var ws=ctx.enqueue_create_buffer[DType.float32](1)
        var expected=ctx.enqueue_create_host_buffer[DType.float32](m*n)
        var actual=ctx.enqueue_create_host_buffer[DType.float32](m*n)
        ctx.synchronize()
        gemm_step_fill(ctx,a,m*k,17+fixture,False)
        gemm_step_fill(ctx,b,n*k,31+fixture,False)
        for op in range(3):
            identical_gemm_with_plan(ctx,control,a,b,ws,m,n,k,op,PLAN_FLAT)
            gemm_step_readback(ctx,control,expected)
            for arm in range(3):
                gemm_step_poison(ctx,candidate,actual,m*n)
                var start=Int(0)
                comptime if TARGET_COLUMN==COLUMN_NVIDIA or TARGET_COLUMN==COLUMN_AMD:start=perf_counter_ns()
                var depth=1
                if arm==0:bounded_staging_gemm[1](ctx,candidate,a,b,m,n,k,op)
                elif arm==1:
                    depth=2
                    bounded_staging_gemm[2](ctx,candidate,a,b,m,n,k,op)
                else:
                    depth=4
                    bounded_staging_gemm[4](ctx,candidate,a,b,m,n,k,op)
                ctx.synchronize()
                var elapsed=Int(0)
                comptime if TARGET_COLUMN==COLUMN_NVIDIA or TARGET_COLUMN==COLUMN_AMD:elapsed=perf_counter_ns()-start
                gemm_step_readback(ctx,candidate,actual)
                var cmp=gemm_step_compare(actual,expected,m*n)
                if cmp[0]!=0 or cmp[1]!=0:raise Error("bounded staging changed output or left poison")
                print("A02_BOUNDED_STAGING_PASS fixture="+String(fixture)+" op="+String(op)+" depth="+String(depth)+" shared_bytes="+String(depth*128*2*4)+" nv_amd_completion_ns="+String(elapsed)+" digest="+hex(gemm_step_digest(actual,m*n)))


def main() raises:
    run_checks()
