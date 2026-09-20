# SPDX-License-Identifier: Apache-2.0
"""Measured resident-memory/time witness for exact recomputing v2 backward."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_backward

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
    _=q^;_=k^;_=v^;_=dy^;_=dq^;_=dk^;_=dv^;_=rm^;_=rz^;_=rzd^;_=lo^;_=hi^;_=hlo^;_=hhi^
