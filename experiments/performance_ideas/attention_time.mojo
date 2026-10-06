# SPDX-License-Identifier: Apache-2.0
"""Actual production fused attention forward/backward timing, no identity replay."""
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from gemm.checks.gemm_step_arms import gemm_step_env_int
from transformer.impl.llama.fused_attention import FUSED_RAN,fused_attention_arm_parse,fused_forward_launch_estash_ran,fused_backward_launch_estash_ran

def main() raises:
    var ctx=DeviceContext()
    var l=gemm_step_env_int("AB_LENGTH",1024)
    var h=gemm_step_env_int("AB_HEADS",12)
    var nkv=gemm_step_env_int("AB_KV_HEADS",4)
    var hd=64;var r=h*l
    var q=ctx.enqueue_create_buffer[DType.float32](r*hd);var k=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd);var v=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd)
    var dy=ctx.enqueue_create_buffer[DType.float32](r*hd);var dq=ctx.enqueue_create_buffer[DType.float32](r*hd);var dk=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd);var dv=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd)
    var out=ctx.enqueue_create_buffer[DType.float32](r*hd);var rm=ctx.enqueue_create_buffer[DType.float32](r);var rz=ctx.enqueue_create_buffer[DType.float32](r);var rzd=ctx.enqueue_create_buffer[DType.float32](r)
    var kept=ctx.enqueue_create_buffer[DType.float32](1);var kept_cells=0;var ran=-1
    ctx.enqueue_memset(q,Float32(.03125));ctx.enqueue_memset(k,Float32(-.0625));ctx.enqueue_memset(v,Float32(.125));ctx.enqueue_memset(dy,Float32(.015625));ctx.synchronize()
    var arm=fused_attention_arm_parse(String("stash_tiled_fgrid_r32_qres_pf"))
    comptime if is_defined["MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE"]():
        arm=fused_attention_arm_parse(String("stash_tiled_fgrid_r32_qres_pf_kvgrid_r32"))
    # I07 measures the existing retained-estash lifetime against its existing
    # recompute macro. The original no-estash word ran the same path twice.
    # Keep I06's selector unchanged; only affected I07 driver builds opt in.
    comptime if is_defined["MOJOLEARN_MEASURE_I07_LIFETIME"]():
        arm=fused_attention_arm_parse(String("stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32"))
    for phase in range(2):
        var start=perf_counter_ns()
        var sf=fused_forward_launch_estash_ran(ctx,out,rm,rz,q,k,v,kept,1,l,h,nkv,hd,l,0,0,0,Float32(.125),arm,ran,kept_cells)
        var sb=fused_backward_launch_estash_ran(ctx,rzd,dq,dk,dv,q,dy,k,v,rm,rz,kept,kept_cells,1,l,h,nkv,hd,l,0,0,0,Float32(.125),arm,ran)
        ctx.synchronize()
        var elapsed=perf_counter_ns()-start
        if sf!=FUSED_RAN or sb!=FUSED_RAN:raise Error("production attention operation refused")
        # Execution admission only: refuse a timing whose intended lifetime
        # did not run. No numerical comparison or identity check is performed.
        comptime if is_defined["MOJOLEARN_MEASURE_I07_LIFETIME"]():
            comptime if is_defined["MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD"]():
                if kept_cells!=0:raise Error("I07 recompute timing retained unexpected exp storage")
            else:
                if kept_cells!=h*l*l:raise Error("I07 retained timing did not retain its declared exp storage")
        print("MEASURE phase="+String(phase)+" length="+String(l)+" heads="+String(h)+" kv_heads="+String(nkv)+" ran_arm="+String(ran)+" kept_cells="+String(kept_cells)+" elapsed_ns="+String(elapsed))
