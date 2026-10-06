# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import ftz
from x_prep.common import canon
from x_prep.dradix import radix_key
from x_prep.ragged_select import enqueue_ragged_select,enqueue_bootstrap_order_statistics

def check_select(ctx: DeviceContext,categories: Bool,bootstrap: Bool) raises:
    var offsets: List[Int32]=[0,0,1,34,291,291,804]
    var total=804
    var raw: List[UInt32]=[0,0x80000000,0x7fc00001,0x7fffffff,0xffc00011,1,0x80000001,0x7f800000,0xff800000,0x3f800000,0xbf800000]
    var input=List[Float32]()
    var indices=List[Int32]()
    var gathered=List[Float32]()
    for i in range(total):
        input.append(bitcast[DType.float32](raw[(i*19)%len(raw)]))
        indices.append(Int32((i*17+3)%total))
    for i in range(total):
        gathered.append(input[Int(indices[i])] if bootstrap else input[i])
    var segments=List[Int32](); var ranks=List[Int32](); var expected_values=List[Float32](); var expected_positions=List[Int32]()
    for s in range(len(offsets)-1):
        var base=Int(offsets[s]); var end=Int(offsets[s+1]); var n=end-base
        if n==0:
            continue
        var ordered=List[Int32]()
        for i in range(base,end):
            var v=ftz(gathered[i]); v=canon(v) if categories else v
            var key=radix_key(bitcast[DType.uint32](v)); var at=len(ordered)
            ordered.append(Int32(i))
            while at>0:
                var prev=ftz(gathered[Int(ordered[at-1])]); prev=canon(prev) if categories else prev
                if radix_key(bitcast[DType.uint32](prev))<=key:
                    break
                ordered[at]=ordered[at-1]; at-=1
            ordered[at]=Int32(i)
        for rank in [0,n//4,n//2,n-1]:
            var pos=ordered[rank]; var v=ftz(gathered[Int(pos)]); v=canon(v) if categories else v
            segments.append(Int32(s)); ranks.append(Int32(rank)); expected_values.append(v); expected_positions.append(pos)
    var tasks=len(ranks)
    var src=ctx.enqueue_create_buffer[DType.float32](total)
    var samples=ctx.enqueue_create_buffer[DType.float32](total)
    var di=ctx.enqueue_create_buffer[DType.int32](total)
    var invalid=ctx.enqueue_create_buffer[DType.int32](total)
    ctx.enqueue_copy(dst_buf=src,src_ptr=input.unsafe_ptr()); ctx.enqueue_copy(dst_buf=di,src_ptr=indices.unsafe_ptr())
    var values=ctx.enqueue_create_buffer[DType.float32](tasks)
    var positions=ctx.enqueue_create_buffer[DType.int32](tasks)
    var descriptors=ctx.enqueue_create_buffer[DType.int32](tasks*3)
    for repeat in range(2):
        if bootstrap:
            enqueue_bootstrap_order_statistics(ctx,src,di,samples,invalid,offsets,segments,ranks,values,positions,descriptors)
        else:
            enqueue_ragged_select(ctx,src,offsets,segments,ranks,categories,values,positions,descriptors)
        var hv=ctx.enqueue_create_host_buffer[DType.float32](tasks)
        var hp=ctx.enqueue_create_host_buffer[DType.int32](tasks)
        var hi=ctx.enqueue_create_host_buffer[DType.int32](total)
        ctx.enqueue_copy(dst_buf=hv,src_buf=values); ctx.enqueue_copy(dst_buf=hp,src_buf=positions)
        if bootstrap:
            ctx.enqueue_copy(dst_buf=hi,src_buf=invalid)
        ctx.synchronize()
        for j in range(tasks):
            if bitcast[DType.uint32](hv[j])!=bitcast[DType.uint32](expected_values[j]) or hp[j]!=expected_positions[j]:
                raise Error("I19 selection-only word/stable-position differs")
        if bootstrap:
            for j in range(total):
                if hi[j]!=0:
                    raise Error("I19 valid bootstrap index refused")
    var refused=False
    try:
        var bad_ranks=ranks
        bad_ranks[0]=Int32(-1)
        enqueue_ragged_select(ctx,src,offsets,segments,bad_ranks,categories,values,positions,descriptors)
    except:
        refused=True
    if not refused:
        raise Error("I19 invalid requested rank admitted")
    if bootstrap:
        indices[0]=Int32(-1); indices[total-1]=Int32(total)
        ctx.enqueue_copy(dst_buf=di,src_ptr=indices.unsafe_ptr())
        enqueue_bootstrap_order_statistics(ctx,src,di,samples,invalid,offsets,segments,ranks,values,positions,descriptors)
        var hi=ctx.enqueue_create_host_buffer[DType.int32](total)
        ctx.enqueue_copy(dst_buf=hi,src_buf=invalid); ctx.synchronize()
        for j in range(total):
            if hi[j]!=Int32(1 if j==0 or j==total-1 else 0):
                raise Error("I19 bootstrap safe-gather refusal differs")
    _ = src^; _ = samples^; _ = di^; _ = invalid^; _ = values^; _ = positions^; _ = descriptors^
    print("I19 SELECT_PASS categories",categories,"bootstrap",bootstrap,"tasks",tasks,"no_sorted_output_or_permutation stable_positions refusal")

def main() raises:
    var ctx=DeviceContext()
    check_select(ctx,False,False)
    check_select(ctx,True,False)
    check_select(ctx,False,True)
