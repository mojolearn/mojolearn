# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from max.gpu.host import DeviceContext
from x_prep.ragged_categories import enqueue_ragged_categories
from x_prep.prims import unique_cols_unit,lookup_unit
from x_prep.common import canon,word_order
from checks.numerics import ftz
from core.stable_radix_sort import stable_radix_counts_len,stable_radix_bsum_len

def check_categories(ctx: DeviceContext) raises:
    var offsets: List[Int32]=[0,0,1,34,291,291,804]
    var total=804; var segments=len(offsets)-1
    var raw: List[UInt32]=[0,0x80000000,0x7fc00001,0x7fffffff,0xffc00011,1,0x80000001,0x7f800000,0xff800000,0x3f800000,0xbf800000]
    var values=List[Float32]()
    for i in range(total):
        values.append(bitcast[DType.float32](raw[(i*19)%len(raw)]))
    var expected=List[Float32](length=total*4+segments,fill=Float32(0))
    for s in range(segments):
        var base=Int(offsets[s]); var end=Int(offsets[s+1])
        for i in range(base,end):
            var v=canon(ftz(values[i])); var pos=i
            while pos>base and word_order(bitcast[DType.uint32](expected[pos-1]))>word_order(bitcast[DType.uint32](v)):
                expected[pos]=expected[pos-1]; pos-=1
            expected[pos]=v
            expected[total+i]=values[i]
        var uq: List[Int32]=[Int32(base),Int32(end-base),1,Int32(total*2+base),Int32(total*4+s)]
        unique_cols_unit(0,expected.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),uq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
        var lq: List[Int32]=[Int32(total+base),Int32(end-base),1,Int32(total*2+base),Int32(end-base),Int32(total*4+s),Int32(total*3+base)]
        for i in range(end-base):
            lookup_unit(i,expected.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),lq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
    var src=ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_buf=src,src_ptr=values.unsafe_ptr())
    var arena=ctx.enqueue_create_buffer[DType.float32](total*4+segments)
    arena.enqueue_fill(Float32(0))
    var permutation=ctx.enqueue_create_buffer[DType.uint32](total)
    var descriptor=ctx.enqueue_create_buffer[DType.int32](segments*12)
    var keys=ctx.enqueue_create_buffer[DType.uint32](513)
    var positions=ctx.enqueue_create_buffer[DType.uint32](513)
    var tk=ctx.enqueue_create_buffer[DType.uint32](513)
    var tv=ctx.enqueue_create_buffer[DType.uint32](513)
    var counts=ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(513))
    var bsum=ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(513))
    enqueue_ragged_categories(ctx,src,arena,permutation,offsets,descriptor,keys,positions,tk,tv,counts,bsum)
    var actual=ctx.enqueue_create_host_buffer[DType.float32](total*4+segments)
    ctx.enqueue_copy(dst_buf=actual,src_buf=arena); ctx.synchronize()
    for s in range(segments):
        var base=Int(offsets[s]); var end=Int(offsets[s+1]); var count=Int(expected[total*4+s])
        if actual[total*4+s]!=expected[total*4+s]:
            raise Error("I19 dictionary cardinality differs")
        for i in range(count):
            if bitcast[DType.uint32](actual[total*2+base+i])!=bitcast[DType.uint32](expected[total*2+base+i]):
                raise Error("I19 canonical dictionary words differ")
        for i in range(base,end):
            if bitcast[DType.uint32](actual[total*3+i])!=bitcast[DType.uint32](expected[total*3+i]):
                raise Error("I19 ordinal code differs")
    _ = src^; _ = arena^; _ = permutation^; _ = descriptor^; _ = keys^; _ = positions^; _ = tk^; _ = tv^; _ = counts^; _ = bsum^
    print("I19 ENCODER_PASS dictionary_codes empty signedzero NaN_payload FTZ infinity")
