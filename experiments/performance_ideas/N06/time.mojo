# SPDX-License-Identifier: Apache-2.0
# N06 2026-10-06 L40S component WIN: 12.389 vs 17.198 ms (1.388x).
# H=4,L=512,HD=64; one same-process warmup/score; full caller promotion owed.
"""Measured resident-memory/time witness for exact recomputing v2 backward."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_backward
from experiments.performance_ideas.N06.compact_grad import compact_backward
from transformer.impl.llama.fused_attention import ATTN_ARM_DEFAULT,FUSED_RAN,fused_forward_launch_estash_ran,fused_backward_launch_estash_ran
from transformer.impl.llama.modeling_llama import _download

def main() raises:
    comptime H=4; comptime L=512; comptime HD=64; comptime R=H*L
    var ctx=DeviceContext()
    var q=ctx.enqueue_create_buffer[DType.float32](R*HD);var k=ctx.enqueue_create_buffer[DType.float32](H*L*HD);var v=ctx.enqueue_create_buffer[DType.float32](H*L*HD)
    var dy=ctx.enqueue_create_buffer[DType.float32](R*HD);var dq=ctx.enqueue_create_buffer[DType.float32](R*HD);var dk=ctx.enqueue_create_buffer[DType.float32](H*L*HD);var dv=ctx.enqueue_create_buffer[DType.float32](H*L*HD)
    var rm=ctx.enqueue_create_buffer[DType.float32](R);var rz=ctx.enqueue_create_buffer[DType.float32](R);var rzd=ctx.enqueue_create_buffer[DType.float32](R)
    var lo=ctx.enqueue_create_buffer[DType.int32](R);var hi=ctx.enqueue_create_buffer[DType.int32](R);var hlo=ctx.enqueue_create_host_buffer[DType.int32](R);var hhi=ctx.enqueue_create_host_buffer[DType.int32](R)
    for r in range(R): hlo[r]=0;hhi[r]=Int32((r%L)+1)
    ctx.enqueue_memset(q,Float32(.03125));ctx.enqueue_memset(k,Float32(-.0625));ctx.enqueue_memset(v,Float32(.125));ctx.enqueue_memset(dy,Float32(.015625));ctx.enqueue_copy(dst_buf=lo,src_buf=hlo);ctx.enqueue_copy(dst_buf=hi,src_buf=hhi);ctx.synchronize()
    var resident=(2*R*HD+4*H*L*HD+5*R)*4
    for arm in range(2):
        for phase in range(2):
            var start=perf_counter_ns()
            if arm==0:
                enqueue_attention_v2_backward(ctx,q,k,v,dy,lo,hi,dq,dk,dv,rm,rz,rzd,R,L,HD,HD,L,Float32(.125))
            else:
                compact_backward(ctx,q,k,v,dy,lo,hi,dq,dk,dv,rm,rz,rzd,R,L,HD,HD,L,Float32(.125))
            ctx.synchronize()
            print("MEASURE id=N06 arm="+String(arm)+" phase="+String(phase)+" length="+String(L)+" heads="+String(H)+" elapsed_ns="+String(perf_counter_ns()-start))
