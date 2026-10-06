# SPDX-License-Identifier: Apache-2.0
"""Measurement-only paired GEMM adapters; inherited identity is not rerun.

Each actual arm is warmed in this DeviceContext immediately before one score.
Upload and fixture generation are excluded; launch and device completion are
included. These are component measurements, not full public caller claims.
"""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_fill,gemm_step_env_int
from gemm.checks.gemm_identical import identical_gemm_flat_kernel,contract_partition,gemm_operand_strides
from gemm.experiments.grouped_jobs import grouped_gemm
from gemm.experiments.async_operand_pipeline import pipeline_gemm
from gemm.experiments.rounded_epilogue import gemm_bias

def main() raises:
    var ctx=DeviceContext()
    var m=gemm_step_env_int("AB_M",257)
    var n=gemm_step_env_int("AB_N",259)
    var k=gemm_step_env_int("AB_K",1025)
    var jobs=gemm_step_env_int("AB_JOBS",3)
    var kind=gemm_step_env_int("AB_KIND",0)
    if m<1 or n<1 or k<1 or jobs<1:
        raise Error("positive geometry required")
    var a=ctx.enqueue_create_buffer[DType.float32](m*k)
    var b=ctx.enqueue_create_buffer[DType.float32](jobs*n*k)
    var bias=ctx.enqueue_create_buffer[DType.float32](n)
    var out=ctx.enqueue_create_buffer[DType.float32](jobs*m*n)
    gemm_step_fill(ctx,a,m*k,17,False)
    gemm_step_fill(ctx,b,jobs*n*k,29,False)
    gemm_step_fill(ctx,bias,n,41,False)
    ctx.synchronize()
    var versions=3 if kind==3 else 1
    for version in range(versions):
        if kind==3:
            gemm_step_fill(ctx,b,jobs*n*k,29+version*17,False)
            ctx.synchronize()
        for op in range(3):
            var baseline_ns=Int(0)
            var candidate_ns=Int(0)
            for arm in range(2):
                for sample in range(2):
                    var start=perf_counter_ns()
                    if kind==0:
                        if arm==0:
                            gemm_bias[False](ctx,out,a,b,bias,m,n,k,op)
                        else:
                            gemm_bias[True](ctx,out,a,b,bias,m,n,k,op)
                    elif kind==1:
                        if arm==0:
                            pipeline_gemm[False](ctx,out,a,b,m,n,k,op)
                        else:
                            pipeline_gemm[True](ctx,out,a,b,m,n,k,op)
                    else:
                        if arm==0:
                            var part=contract_partition(k)
                            var st=gemm_operand_strides(op,m,n,k)
                            for job in range(jobs):
                                ctx.enqueue_function[identical_gemm_flat_kernel](out.unsafe_ptr()+job*m*n,a.unsafe_ptr(),b.unsafe_ptr()+job*n*k,Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
                        else:
                            grouped_gemm(out,a,b,ctx,m,n,k,op,jobs)
                    ctx.synchronize()
                    var elapsed=perf_counter_ns()-start
                    if sample==1:
                        if arm==0:
                            baseline_ns=elapsed
                        else:
                            candidate_ns=elapsed
            print("PAIRED kind="+String(kind)+" op="+String(op)+" version="+String(version)+" m="+String(m)+" n="+String(n)+" k="+String(k)+" jobs="+String(jobs)+" baseline_completion_ns="+String(baseline_ns)+" candidate_completion_ns="+String(candidate_ns))
