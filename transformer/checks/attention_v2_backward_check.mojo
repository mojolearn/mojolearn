# SPDX-License-Identifier: Apache-2.0
"""Exact Apple gate for opt-in attention-v2 recomputing backward."""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_backward

def main() raises:
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
    comptime R=5; comptime K=4; comptime HD=3; comptime W=2
    var ctx=DeviceContext()
    var hq=ctx.enqueue_create_host_buffer[DType.float32](R*HD)
    var hk=ctx.enqueue_create_host_buffer[DType.float32](K*HD)
    var hv=ctx.enqueue_create_host_buffer[DType.float32](K*W)
    var hdy=ctx.enqueue_create_host_buffer[DType.float32](R*W)
    for i in range(R):
        for d in range(HD): hq[i*HD+d]=Float32((i*5+d*3)%11-5)/Float32(4)
        for d in range(W): hdy[i*W+d]=Float32((i*7+d*2)%13-6)/Float32(8)
    for j in range(K):
        for d in range(HD): hk[j*HD+d]=Float32((j*3+d*7)%17-8)/Float32(8)
        for d in range(W): hv[j*W+d]=Float32((j*11+d*5)%19-9)/Float32(8)
    var hlo=ctx.enqueue_create_host_buffer[DType.int32](R); var hhi=ctx.enqueue_create_host_buffer[DType.int32](R)
    hlo[0]=0;hhi[0]=1; hlo[1]=0;hhi[1]=2; hlo[2]=1;hhi[2]=3; hlo[3]=0;hhi[3]=4; hlo[4]=2;hhi[4]=4
    var q=ctx.enqueue_create_buffer[DType.float32](R*HD);var k=ctx.enqueue_create_buffer[DType.float32](K*HD)
    var v=ctx.enqueue_create_buffer[DType.float32](K*W);var dy=ctx.enqueue_create_buffer[DType.float32](R*W)
    var lo=ctx.enqueue_create_buffer[DType.int32](R);var hi=ctx.enqueue_create_buffer[DType.int32](R)
    ctx.enqueue_copy(dst_buf=q,src_buf=hq);ctx.enqueue_copy(dst_buf=k,src_buf=hk);ctx.enqueue_copy(dst_buf=v,src_buf=hv);ctx.enqueue_copy(dst_buf=dy,src_buf=hdy);ctx.enqueue_copy(dst_buf=lo,src_buf=hlo);ctx.enqueue_copy(dst_buf=hi,src_buf=hhi)
    var dq=ctx.enqueue_create_buffer[DType.float32](R*HD);var dk=ctx.enqueue_create_buffer[DType.float32](K*HD);var dv=ctx.enqueue_create_buffer[DType.float32](K*W)
    var rm=ctx.enqueue_create_buffer[DType.float32](R);var rz=ctx.enqueue_create_buffer[DType.float32](R);var rzd=ctx.enqueue_create_buffer[DType.float32](R)
    enqueue_attention_v2_backward(ctx,q,k,v,dy,lo,hi,dq,dk,dv,rm,rz,rzd,R,K,W,HD,R,Float32(.5))
    var hdq=ctx.enqueue_create_host_buffer[DType.float32](R*HD);var hdk=ctx.enqueue_create_host_buffer[DType.float32](K*HD);var hdv=ctx.enqueue_create_host_buffer[DType.float32](K*W)
    ctx.enqueue_copy(dst_buf=hdq,src_buf=dq);ctx.enqueue_copy(dst_buf=hdk,src_buf=dk);ctx.enqueue_copy(dst_buf=hdv,src_buf=dv);ctx.synchronize()
    var wdq:List[UInt32]=[0,0,0,1018716603,1018716605,3184991410,1027576872,1027576874,1027576871,1008299425,1008299442,3183373642,3165120610,3165120612,3165120608]
    var wdk:List[UInt32]=[1011173012,3180167811,1014158684,3190710099,1043027173,3142409292,1045854202,3189077410,3154373010,3176248385,1027835904,3144569460]
    var wdv:List[UInt32]=[3206445308,3193367915,3190352503,1044141200,3205038722,3194624110,3191416720,999696688]
    var bad=0
    for i in range(R*HD):
        if bitcast[DType.uint32](hdq[i])!=wdq[i]: print("dq",i,bitcast[DType.uint32](hdq[i]),wdq[i]);bad+=1
    for i in range(K*HD):
        if bitcast[DType.uint32](hdk[i])!=wdk[i]: print("dk",i,bitcast[DType.uint32](hdk[i]),wdk[i]);bad+=1
    for i in range(K*W):
        if bitcast[DType.uint32](hdv[i])!=wdv[i]: print("dv",i,bitcast[DType.uint32](hdv[i]),wdv[i]);bad+=1
    if bad: raise Error("attention-v2 backward mismatches "+String(bad))
    print("attention_v2_backward_check PASS: dQ dK dV exact")
    _=q^;_=k^;_=v^;_=dy^;_=lo^;_=hi^;_=dq^;_=dk^;_=dv^;_=rm^;_=rz^;_=rzd^;_=hq^;_=hk^;_=hv^;_=hdy^;_=hlo^;_=hhi^;_=hdq^;_=hdk^;_=hdv^
