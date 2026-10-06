# SPDX-License-Identifier: Apache-2.0
"""Resident ragged dictionary and ordinal encoder preparation.

Source sorting is an explicit default-off I19 candidate. The dictionary
and lookup execute the existing pinned category primitives. Canonical NaN,
signed-zero and FTZ policy are unchanged. Compile/device qualification is
pending; no estimator default changes. Caller owns all bounded storage.
"""
from std.gpu import block_idx,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from x_prep.prims import unique_cols_unit,lookup_unit
from experiments.performance_ideas.I19.ragged_float import enqueue_ragged_float_sort

def _unique(arena: MutPointer[Float32,MutAnyOrigin],descriptors: MutPointer[Int32,MutAnyOrigin],segments: Int32):
    var s=Int(block_idx.x)*128+Int(thread_idx.x)
    if s<Int(segments):
        unique_cols_unit(0,arena,descriptors.unsafe_offset(s*12))

def _lookup(arena: MutPointer[Float32,MutAnyOrigin],descriptors: MutPointer[Int32,MutAnyOrigin],s: Int32,n: Int32):
    var i=Int(block_idx.x)*128+Int(thread_idx.x)
    if i<Int(n):
        lookup_unit(i,arena,descriptors.unsafe_offset(Int(s)*12+5))

def enqueue_ragged_categories(ctx: DeviceContext,mut src: DeviceBuffer[DType.float32],
    mut arena: DeviceBuffer[DType.float32],mut permutation: DeviceBuffer[DType.uint32],
    offsets: List[Int32],mut descriptors: DeviceBuffer[DType.int32],
    mut keys: DeviceBuffer[DType.uint32],mut positions: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],mut temp_positions: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],mut bsum: DeviceBuffer[DType.int32]) raises:
    if len(offsets)<2:
        raise Error("ragged categories: missing offsets")
    var segments=len(offsets)-1
    var total=Int(offsets[segments])
    if total<0 or total>(2147483647-segments)//4 or len(arena)<total*4+segments or len(descriptors)<segments*12:
        raise Error("ragged categories: descriptor/arena index bound")
    enqueue_ragged_float_sort(ctx,src,arena,permutation,offsets,True,keys,positions,temp_keys,temp_positions,counts,bsum)
    var original=arena.create_sub_buffer[DType.float32](total,total)
    var original_src=src.create_sub_buffer[DType.float32](0,total)
    ctx.enqueue_copy(dst_buf=original,src_buf=original_src)
    _ = original^; _ = original_src^
    var host=ctx.enqueue_create_host_buffer[DType.int32](segments*12)
    for s in range(segments):  # small-loop(segments: shape metadata only)
        var base=offsets[s]; var n=offsets[s+1]-base
        host[s*12]=base; host[s*12+1]=n; host[s*12+2]=Int32(1)
        host[s*12+3]=Int32(total*2)+base; host[s*12+4]=Int32(total*4+s)
        host[s*12+5]=Int32(total)+base; host[s*12+6]=n; host[s*12+7]=Int32(1)
        host[s*12+8]=Int32(total*2)+base; host[s*12+9]=n
        host[s*12+10]=Int32(total*4+s); host[s*12+11]=Int32(total*3)+base
    ctx.enqueue_copy(dst_buf=descriptors,src_buf=host)
    ctx.enqueue_function[_unique](arena.unsafe_ptr(),descriptors.unsafe_ptr(),Int32(segments),grid_dim=((segments+127)//128,1,1),block_dim=(128,1,1))
    for s in range(segments):  # small-loop(segments: launch descriptors only)
        var n=Int(offsets[s+1]-offsets[s])
        if n>0:
            ctx.enqueue_function[_lookup](arena.unsafe_ptr(),descriptors.unsafe_ptr(),Int32(s),Int32(n),grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
    ctx.synchronize()  # descriptor lifetime and full-operation boundary counted
    _ = host^
