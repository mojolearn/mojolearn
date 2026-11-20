# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded order statistics without sorted/permutation materialization.

The rank-count control is the only route (the I19 radix arm is deleted). Unsigned key decisions and stable original-position
secondary keys are exact integers. No estimator/default profile changes.
"""
from max.gpu import block_idx,thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from x_prep.common import canon
from x_prep.dradix import radix_key,radix_word

# TOMBSTONE: MOJOLEARN_IDN_RAGGED_RADIX_SELECT (opt-in arm, never run) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_RADIX_SELECT.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella, EXP:2438; every IDENTICAL default kept) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-36-neural-series-linear.patch; record in docs/TOMBSTONES.md.
comptime SELECT_TPB=128

@always_inline
def _select_key(v: Float32,categories: Int32) -> UInt32:
    var value=ftz(v)
    if categories!=0:
        value=canon(value)
    return radix_key(bitcast[DType.uint32](value))

def _select_kernel(src: MutPointer[Float32,MutAnyOrigin],desc: MutPointer[Int32,MutAnyOrigin],values: MutPointer[Float32,MutAnyOrigin],positions: MutPointer[Int32,MutAnyOrigin],categories: Int32):
    var task=Int(block_idx.x); var t=Int(thread_idx.x)
    var base=Int(desc[task*3]); var n=Int(desc[task*3+1]); var r=Int(desc[task*3+2])
    # TOMBSTONE: MOJOLEARN_IDN_RAGGED_RADIX_SELECT (opt-in arm, never run) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_RADIX_SELECT.patch; record in docs/TOMBSTONES.md.
    if t==0:
        for i in range(n):
            var key=_select_key(src[base+i],categories); var rank=0
            for j in range(n):
                var other=_select_key(src[base+j],categories)
                if other<key or (other==key and j<i):
                    rank+=1
            if rank==r:
                values[task]=bitcast[DType.float32](radix_word(key))
                positions[task]=Int32(base+i)
                break

def enqueue_ragged_select(ctx: DeviceContext,mut src: DeviceBuffer[DType.float32],
    offsets: List[Int32],segments: List[Int32],ranks: List[Int32],categories: Bool,
    mut values: DeviceBuffer[DType.float32],mut positions: DeviceBuffer[DType.int32],
    mut descriptors: DeviceBuffer[DType.int32]) raises:
    if len(offsets)<2 or offsets[0]!=0 or len(segments)!=len(ranks) or len(segments)>2147483647//3:
        raise Error("ragged select: invalid metadata shape")
    for s in range(len(offsets)-1):  # small-loop(segments: shape metadata)
        if offsets[s]<0 or offsets[s+1]<offsets[s]:
            raise Error("ragged select: offsets not monotone")
    var tasks=len(segments)
    if Int(offsets[len(offsets)-1])>len(src) or len(values)<tasks or len(positions)<tasks or len(descriptors)<tasks*3:
        raise Error("ragged select: caller storage too short")
    if tasks==0:
        return
    var host=ctx.enqueue_create_host_buffer[DType.int32](tasks*3)
    for task in range(tasks):  # small-loop(tasks: requested order-statistic metadata)
        var s=Int(segments[task])
        if s<0 or s>=len(offsets)-1:
            raise Error("ragged select: invalid segment")
        var n=offsets[s+1]-offsets[s]
        if n<=0 or n>1048576 or ranks[task]<0 or ranks[task]>=n:
            raise Error("ragged select: invalid rank/capacity")
        if n>4096:
            raise Error("ragged select: rank control exceeds bounded small rows")
        host[task*3]=offsets[s]; host[task*3+1]=n; host[task*3+2]=ranks[task]
    ctx.enqueue_copy(dst_buf=descriptors,src_buf=host)
    ctx.enqueue_function[_select_kernel](src.unsafe_ptr(),descriptors.unsafe_ptr(),values.unsafe_ptr(),positions.unsafe_ptr(),Int32(1 if categories else 0),grid_dim=(tasks,1,1),block_dim=(SELECT_TPB,1,1))
    ctx.synchronize()  # host descriptor lifetime; included in full caller cost
    _ = host^

def _bootstrap_gather(src: MutPointer[Float32,MutAnyOrigin],indices: MutPointer[Int32,MutAnyOrigin],samples: MutPointer[Float32,MutAnyOrigin],invalid: MutPointer[Int32,MutAnyOrigin],n: Int32,total: Int32):
    var i=Int(block_idx.x)*SELECT_TPB+Int(thread_idx.x)
    if i<Int(total):
        var index=indices[i]
        var valid=index>=0 and index<n
        samples[i]=src[Int(index)] if valid else Float32(0)
        invalid[i]=Int32(0 if valid else 1)

def enqueue_bootstrap_order_statistics(ctx: DeviceContext,mut src: DeviceBuffer[DType.float32],
    mut indices: DeviceBuffer[DType.int32],mut samples: DeviceBuffer[DType.float32],
    mut invalid: DeviceBuffer[DType.int32],offsets: List[Int32],segments: List[Int32],
    ranks: List[Int32],mut values: DeviceBuffer[DType.float32],mut positions: DeviceBuffer[DType.int32],
    mut descriptors: DeviceBuffer[DType.int32]) raises:
    """Indices already selected by the caller's RNG contract; never resampled.

    `invalid[0:total]` is mandatory refusal evidence. Any nonzero word means
    the caller must reject the result. Device gather checks before load.
    Scratch samples are caller-owned; no full sorted array/permutation.
    """
    if len(offsets)<2:
        raise Error("bootstrap order statistics: missing offsets")
    var total=Int(offsets[len(offsets)-1])
    if len(src)>2147483647 or total<0 or len(indices)<total or len(samples)<total or len(invalid)<total:
        raise Error("bootstrap order statistics: storage/index bounds")
    if total>0:
        ctx.enqueue_function[_bootstrap_gather](src.unsafe_ptr(),indices.unsafe_ptr(),samples.unsafe_ptr(),invalid.unsafe_ptr(),Int32(len(src)),Int32(total),grid_dim=((total+SELECT_TPB-1)//SELECT_TPB,1,1),block_dim=(SELECT_TPB,1,1))
    enqueue_ragged_select(ctx,samples,offsets,segments,ranks,False,values,positions,descriptors)
