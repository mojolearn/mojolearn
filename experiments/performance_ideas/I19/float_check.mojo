# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import ftz
from x_prep.common import canon,word_order
from experiments.performance_ideas.I19.ragged_float import enqueue_ragged_float_sort
from core.stable_radix_sort import stable_radix_counts_len,stable_radix_bsum_len

def check_float_ragged(ctx: DeviceContext) raises:
    var planted: List[UInt32] = [0xff800000,0xbf800000,0x80000001,0x80000000,0,1,0x3f800000,0x7f800000,0x7fc00000,0x7f800001,0x7fc00001,0xff800001,0xffc00000,0xbf800000,0]
    var offsets = List[Int32]()
    offsets.append(Int32(0))
    for length in [0,1,33,257,0,513,3]:
        offsets.append(offsets[len(offsets)-1]+Int32(length))
    var total = Int(offsets[len(offsets)-1])
    var src = ctx.enqueue_create_buffer[DType.float32](total)
    var dst = ctx.enqueue_create_buffer[DType.float32](total+17)
    var permutation = ctx.enqueue_create_buffer[DType.uint32](total+17)
    var host = ctx.enqueue_create_host_buffer[DType.float32](total)
    for i in range(total):
        host[i] = bitcast[DType.float32](planted[(i*17+i//13)%len(planted)])
    ctx.enqueue_copy(dst_buf=src,src_buf=host)
    var keys = ctx.enqueue_create_buffer[DType.uint32](513)
    var positions = ctx.enqueue_create_buffer[DType.uint32](513)
    var temp_keys = ctx.enqueue_create_buffer[DType.uint32](513)
    var temp_positions = ctx.enqueue_create_buffer[DType.uint32](513)
    var counts = ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(513))
    var bsum = ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(513))
    for categories in [False,True,False]:
        dst.enqueue_fill(bitcast[DType.float32](UInt32(0xdeadbeef)))
        permutation.enqueue_fill(UInt32(0xcafebabe))
        enqueue_ragged_float_sort(ctx,src,dst,permutation,offsets,categories,keys,positions,temp_keys,temp_positions,counts,bsum)
        var actual = ctx.enqueue_create_host_buffer[DType.float32](total+17)
        var order = ctx.enqueue_create_host_buffer[DType.uint32](total+17)
        ctx.enqueue_copy(dst_buf=actual,src_buf=dst)
        ctx.enqueue_copy(dst_buf=order,src_buf=permutation)
        ctx.synchronize()
        for s in range(len(offsets)-1):
            var base = Int(offsets[s])
            var end = Int(offsets[s+1])
            var expected = List[Int]()
            # Independent canonical word comparator, stable insertion order.
            for i in range(base,end):
                var value = ftz(host[i])
                if categories:
                    value=canon(value)
                var key = word_order(bitcast[DType.uint32](value))
                var pos = len(expected)
                expected.append(i)
                while pos>0:
                    var other = ftz(host[expected[pos-1]])
                    if categories:
                        other=canon(other)
                    if word_order(bitcast[DType.uint32](other))<=key:
                        break
                    expected[pos]=expected[pos-1]
                    pos-=1
                expected[pos]=i
            for i in range(len(expected)):
                var value = ftz(host[expected[i]])
                if categories:
                    value=canon(value)
                if order[base+i]!=UInt32(expected[i]) or bitcast[DType.uint32](actual[base+i])!=bitcast[DType.uint32](value):
                    raise Error("I19 floating public policy/permutation moved")
        for i in range(total,total+17):
            if bitcast[DType.uint32](actual[i])!=UInt32(0xdeadbeef) or order[i]!=UInt32(0xcafebabe):
                raise Error("I19 ragged sort touched output capacity tail")
    _ = src^; _ = dst^; _ = permutation^; _ = keys^; _ = positions^; _ = temp_keys^; _ = temp_positions^; _ = counts^; _ = bsum^
    print("I19 PASS ragged empty/tails signed_zero subnormal infinities NaN_payloads category_canonicalization")
