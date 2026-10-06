# SPDX-License-Identifier: Apache-2.0
"""Resident ragged quantile caller for stable radix preparation.

Sorts every admitted segment, then executes the existing quantile_unit's
pinned interpolation. Fractions remain device resident. Source values and
output storage are distinct; only metadata descriptors come from host.
No estimator default or numerical profile changes.
"""
from std.gpu import block_idx,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from x_prep.prims import quantile_unit
from experiments.performance_ideas.I19.ragged_float import enqueue_ragged_float_sort

def _ragged_quantile_kernel(arena: MutPointer[Float32,MutAnyOrigin],params: MutPointer[Int32,MutAnyOrigin],nq_in: Int32,tasks_in: Int32):
    var task=Int(block_idx.x)*128+Int(thread_idx.x)
    if task<Int(tasks_in):
        var segment=task//Int(nq_in)
        quantile_unit(task%Int(nq_in),arena,params.unsafe_offset(segment*7))

def enqueue_ragged_quantiles(ctx: DeviceContext,mut src: DeviceBuffer[DType.float32],
    mut arena: DeviceBuffer[DType.float32],mut permutation: DeviceBuffer[DType.uint32],
    offsets: List[Int32],fractions: DeviceBuffer[DType.float32],nq: Int,
    mut descriptors: DeviceBuffer[DType.int32],mut keys: DeviceBuffer[DType.uint32],
    mut positions: DeviceBuffer[DType.uint32],mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_positions: DeviceBuffer[DType.uint32],mut counts: DeviceBuffer[DType.int32],
    mut bsum: DeviceBuffer[DType.int32]) raises:
    if len(offsets)<2 or nq<=0 or nq>len(fractions):
        raise Error("ragged quantile: invalid fraction/segment shape")
    var segments=len(offsets)-1
    var total=Int(offsets[len(offsets)-1])
    if total<0 or segments>2147483647//nq or total>2147483647-nq-segments*nq:
        raise Error("ragged quantile: device descriptor index bound exceeded")
    var output=total+nq
    if len(arena)<output+segments*nq or len(descriptors)<segments*7:
        raise Error("ragged quantile: caller arena or descriptors too short")
    enqueue_ragged_float_sort(ctx,src,arena,permutation,offsets,False,keys,positions,temp_keys,temp_positions,counts,bsum)
    var qdest=arena.create_sub_buffer[DType.float32](total,nq)
    var qsrc=fractions.create_sub_buffer[DType.float32](0,nq)
    ctx.enqueue_copy(dst_buf=qdest,src_buf=qsrc)
    _ = qdest^; _ = qsrc^
    var host=ctx.enqueue_create_host_buffer[DType.int32](segments*7)
    for segment in range(segments):  # small-loop(segments: quantile descriptors): shapes only, no data arithmetic
        host[segment*7]=offsets[segment]
        host[segment*7+1]=offsets[segment+1]-offsets[segment]
        host[segment*7+2]=Int32(1)
        host[segment*7+3]=Int32(total)
        host[segment*7+4]=Int32(nq)
        host[segment*7+5]=Int32(output+segment*nq)
        host[segment*7+6]=Int32(-1)
    ctx.enqueue_copy(dst_buf=descriptors,src_buf=host)
    ctx.enqueue_function[_ragged_quantile_kernel](arena.unsafe_ptr(),descriptors.unsafe_ptr(),Int32(nq),Int32(segments*nq),grid_dim=((segments*nq+127)//128,1,1),block_dim=(128,1,1))
    # Host descriptor storage must survive upload. It drains the actual
    # caller here; timing must count this admission/consumer boundary.
    ctx.synchronize()
    _ = host^
