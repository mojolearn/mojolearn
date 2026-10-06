# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from max.gpu.host import DeviceContext
from x_prep.ragged_quantile import enqueue_ragged_quantiles
from x_prep.prims import quantile_unit
from core.stable_radix_sort import stable_radix_counts_len,stable_radix_bsum_len

def check_quantile_caller(ctx: DeviceContext) raises:
    var offsets: List[Int32]=[0,0,1,34,291,291,804]
    var total=804
    var segments=len(offsets)-1
    var fractions: List[Float32]=[0,0.25,0.5,0.75,1]
    var nq=len(fractions)
    var host=ctx.enqueue_create_host_buffer[DType.float32](total)
    for i in range(total):
        host[i]=Float32((i*19)%97-48)*Float32(0.0625)
    var src=ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_buf=src,src_buf=host)
    var q=ctx.enqueue_create_buffer[DType.float32](nq)
    ctx.enqueue_copy(dst_buf=q,src_ptr=fractions.unsafe_ptr())
    var arena=ctx.enqueue_create_buffer[DType.float32](total+nq+segments*nq)
    var permutation=ctx.enqueue_create_buffer[DType.uint32](total)
    var descriptor=ctx.enqueue_create_buffer[DType.int32](segments*7)
    var keys=ctx.enqueue_create_buffer[DType.uint32](513)
    var positions=ctx.enqueue_create_buffer[DType.uint32](513)
    var tk=ctx.enqueue_create_buffer[DType.uint32](513)
    var tv=ctx.enqueue_create_buffer[DType.uint32](513)
    var counts=ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(513))
    var bsum=ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(513))
    enqueue_ragged_quantiles(ctx,src,arena,permutation,offsets,q,nq,descriptor,keys,positions,tk,tv,counts,bsum)
    var actual=ctx.enqueue_create_host_buffer[DType.float32](total+nq+segments*nq)
    ctx.enqueue_copy(dst_buf=actual,src_buf=arena)
    ctx.synchronize()
    var expected=List[Float32](length=total+nq+segments*nq,fill=Float32(0))
    for s in range(segments):
        var base=Int(offsets[s]); var end=Int(offsets[s+1])
        for i in range(base,end):
            var value=host[i]
            var pos=i
            while pos>base and expected[pos-1]>value:
                expected[pos]=expected[pos-1]
                pos-=1
            expected[pos]=value
    for i in range(nq):
        expected[total+i]=fractions[i]
    for s in range(segments):
        var params: List[Int32]=[offsets[s],offsets[s+1]-offsets[s],1,Int32(total),Int32(nq),Int32(total+nq+s*nq),-1]
        for j in range(nq):
            quantile_unit(j,expected.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),params.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
            var index=total+nq+s*nq+j
            if bitcast[DType.uint32](actual[index])!=bitcast[DType.uint32](expected[index]):
                raise Error("I19 actual resident preparation quantile differs")
    _ = src^; _ = q^; _ = arena^; _ = permutation^; _ = descriptor^; _ = keys^; _ = positions^; _ = tk^; _ = tv^; _ = counts^; _ = bsum^
    print("I19 QUANTILE_PASS segments=",segments,"fractions=",nq,"empty repeated_keys tails existing_pinned_interpolation")
