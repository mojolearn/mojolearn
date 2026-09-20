# SPDX-License-Identifier: Apache-2.0
"""Measured resident-memory/time witness for exact recomputing v2 backward."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_backward
from transformer.impl.llama.fused_attention import ATTN_ARM_DEFAULT,FUSED_RAN,fused_forward_launch_estash_ran,fused_backward_launch_estash_ran
from transformer.impl.llama.modeling_llama import _download

def main() raises:
    comptime H=12; comptime L=2048; comptime HD=64; comptime R=H*L
    var ctx=DeviceContext()
    var q=ctx.enqueue_create_buffer[DType.float32](R*HD);var k=ctx.enqueue_create_buffer[DType.float32](H*L*HD);var v=ctx.enqueue_create_buffer[DType.float32](H*L*HD)
    var dy=ctx.enqueue_create_buffer[DType.float32](R*HD);var dq=ctx.enqueue_create_buffer[DType.float32](R*HD);var dk=ctx.enqueue_create_buffer[DType.float32](H*L*HD);var dv=ctx.enqueue_create_buffer[DType.float32](H*L*HD)
    var rm=ctx.enqueue_create_buffer[DType.float32](R);var rz=ctx.enqueue_create_buffer[DType.float32](R);var rzd=ctx.enqueue_create_buffer[DType.float32](R)
    var lo=ctx.enqueue_create_buffer[DType.int32](R);var hi=ctx.enqueue_create_buffer[DType.int32](R);var hlo=ctx.enqueue_create_host_buffer[DType.int32](R);var hhi=ctx.enqueue_create_host_buffer[DType.int32](R)
    for r in range(R): hlo[r]=0;hhi[r]=Int32((r%L)+1)
    ctx.enqueue_memset(q,Float32(.03125));ctx.enqueue_memset(k,Float32(-.0625));ctx.enqueue_memset(v,Float32(.125));ctx.enqueue_memset(dy,Float32(.015625));ctx.enqueue_copy(dst_buf=lo,src_buf=hlo);ctx.enqueue_copy(dst_buf=hi,src_buf=hhi);ctx.synchronize()
    var resident=(2*R*HD+4*H*L*HD+5*R)*4
    for rep in range(3):
        var t=perf_counter_ns();enqueue_attention_v2_backward(ctx,q,k,v,dy,lo,hi,dq,dk,dv,rm,rz,rzd,R,L,HD,HD,L,Float32(.125));ctx.synchronize()
        print("attention_v2_backward B1 H12 L2048 HD64 rep",rep,"ms",Float64(perf_counter_ns()-t)/1e6,"resident_bytes",resident,"quadratic_saved_bytes",2*R*L*4)
    var ctxv=ctx.enqueue_create_buffer[DType.float32](R*HD);var kept=ctx.enqueue_create_buffer[DType.float32](1);var kept_cells=0;var ran=-1
    var sf=fused_forward_launch_estash_ran(ctx,ctxv,rm,rz,q,k,v,kept,1,L,H,H,HD,L,0,0,0,Float32(.125),ATTN_ARM_DEFAULT,ran,kept_cells);ctx.synchronize()
    if sf!=FUSED_RAN:raise Error("production v1 forward refused")
    for rep in range(3):
        var t=perf_counter_ns();var sb=fused_backward_launch_estash_ran(ctx,rzd,dq,dk,dv,q,dy,k,v,rm,rz,kept,kept_cells,1,L,H,H,HD,L,0,0,0,Float32(.125),ATTN_ARM_DEFAULT,ran);ctx.synchronize()
        if sb!=FUSED_RAN:raise Error("production v1 backward refused")
        print("attention_v1_backward B1 H12 L2048 HD64 rep",rep,"ms",Float64(perf_counter_ns()-t)/1e6,"base_resident_bytes",resident+R*HD*4,"kept_cells",kept_cells,"extra_kept_bytes",kept_cells*4,"ran_arm",ran)
    var stash_dq=_download(ctx,dq,R*HD);var stash_dk=_download(ctx,dk,H*L*HD);var stash_dv=_download(ctx,dv,H*L*HD)
    for rep in range(3):
        var t=perf_counter_ns();var sb=fused_backward_launch_estash_ran(ctx,rzd,dq,dk,dv,q,dy,k,v,rm,rz,kept,0,1,L,H,H,HD,L,0,0,0,Float32(.125),ATTN_ARM_DEFAULT,ran);ctx.synchronize()
        if sb!=FUSED_RAN:raise Error("production v1 recompute backward refused")
        print("attention_v1_recompute_backward B1 H12 L2048 HD64 rep",rep,"ms",Float64(perf_counter_ns()-t)/1e6,"resident_bytes",resident+R*HD*4,"stash_saved_bytes",kept_cells*4,"ran_arm",ran)
    var rdq=_download(ctx,dq,R*HD);var rdk=_download(ctx,dk,H*L*HD);var rdv=_download(ctx,dv,H*L*HD);var bad=0
    for i in range(R*HD):
        if rdq[i]!=stash_dq[i]:bad+=1
    for i in range(H*L*HD):
        if rdk[i]!=stash_dk[i] or rdv[i]!=stash_dv[i]:bad+=1
    print("attention_v1_recompute exact_bad",bad)
    if bad!=0:raise Error("v1 recompute moved gradients")
    _=q^;_=k^;_=v^;_=dy^;_=dq^;_=dk^;_=dv^;_=rm^;_=rz^;_=rzd^;_=ctxv^;_=kept^;_=lo^;_=hi^;_=hlo^;_=hhi^;_=stash_dq^;_=stash_dk^;_=stash_dv^;_=rdq^;_=rdk^;_=rdv^
