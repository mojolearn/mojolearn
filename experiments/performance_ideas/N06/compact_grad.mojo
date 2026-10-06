# SPDX-License-Identifier: Apache-2.0
"""Four-feature gradient tiles with register accumulation in query order.

The rejected cooperative form launched one feature-wide block per key and
paid query barriers. Here many independent key/feature tiles share each
block, with four cells per thread and no query barriers. Score recomputation
is explicit and must be included in the full backward comparison. Every
cell preserves the production ascending-query chain and pinned operations.
"""
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from transformer.impl.llama.attention_v2 import (
    _v2_score,_v2_dyv,_v2_mul,attention_v2_backward_prepare_kernel,attention_v2_dq_kernel,
)
from checks.numerics import identical_mul_add,identical_div,portable_expf


def feature_grad_kernel[DK: Bool,TILE: Int](queries: MutPointer[Float32,MutAnyOrigin],
    key_vectors: MutPointer[Float32,MutAnyOrigin],values: MutPointer[Float32,MutAnyOrigin],
    dy: MutPointer[Float32,MutAnyOrigin],visible_lo: MutPointer[Int32,MutAnyOrigin],
    visible_hi: MutPointer[Int32,MutAnyOrigin],row_max: MutPointer[Float32,MutAnyOrigin],
    denominator: MutPointer[Float32,MutAnyOrigin],row_zdot: MutPointer[Float32,MutAnyOrigin],
    output: MutPointer[Float32,MutAnyOrigin],rows_in: Int32,keys_in: Int32,width_in: Int32,
    head_dim_in: Int32,qpg_in: Int32,scale: Float32):
    var rows=Int(rows_in);var keys=Int(keys_in);var width=Int(width_in);var hd=Int(head_dim_in);var qpg=Int(qpg_in)
    var dims=width
    comptime if DK:
        dims=hd
    var tiles=(dims+TILE-1)//TILE
    var idx=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var groups=(rows+qpg-1)//qpg
    if idx>=groups*keys*tiles:
        return
    var owner=idx//tiles
    var base=(idx%tiles)*TILE
    var key=owner%keys
    var group=owner//keys
    var acc=SIMD[DType.float32,TILE](0.0)
    for row in range(group*qpg,min((group+1)*qpg,rows)):
        if key>=Int(visible_lo.unsafe_load(row)) and key<Int(visible_hi.unsafe_load(row)):
            var p=identical_div(portable_expf(_v2_score(queries,key_vectors,row,group,key,keys,hd,scale)-row_max.unsafe_load(row)),denominator.unsafe_load(row))
            var coeff=p
            comptime if DK:
                coeff=_v2_mul(_v2_mul(p,_v2_dyv(values,dy,row,group,key,keys,width)-row_zdot.unsafe_load(row)),scale)
            comptime for d in range(TILE):
                if base+d<dims:
                    var value=Float32(0.0)
                    comptime if DK:
                        value=queries.unsafe_load(row*hd+base+d)
                    else:
                        value=dy.unsafe_load(row*width+base+d)
                    acc[d]=identical_mul_add(coeff,value,acc[d])
    comptime for d in range(TILE):
        if base+d<dims:
            output.unsafe_store(owner*dims+base+d,acc[d])


# N06 2026-10-06 scoped WIN, source cbcc8dcd3303: compact vs original
# attention-v2 backward, H=4/L=512/HD=64. AMD 50.013095 -> 35.579508 ms
# (candidate/baseline 0.711); NVIDIA L40S 17.198174 -> 12.389213 ms (0.720).
# The explicit candidate launches feature_grad_kernel[True/False,4]; this is
# distinct on both vendors. One excluded same-context warmup and one score, rc=0.
# Existing identity evidence reused; no new identity/quality claim. This synthetic
# backward component leaves full training/GQA/resource qualification pending;
# incumbent defaults retained. Evidence: measurements/20261006/index.json.
def compact_backward(ctx: DeviceContext,mut queries: DeviceBuffer[DType.float32],
    mut key_vectors: DeviceBuffer[DType.float32],mut values: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],mut visible_lo: DeviceBuffer[DType.int32],
    mut visible_hi: DeviceBuffer[DType.int32],mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32],mut dv: DeviceBuffer[DType.float32],
    mut row_max: DeviceBuffer[DType.float32],mut denominator: DeviceBuffer[DType.float32],
    mut row_zdot: DeviceBuffer[DType.float32],rows: Int,keys: Int,width: Int,
    head_dim: Int,queries_per_group: Int,scale: Float32) raises:
    if rows<1 or keys<1 or width<1 or head_dim<1 or queries_per_group<1:
        raise Error("positive backward geometry required")
    var groups=(rows+queries_per_group-1)//queries_per_group
    ctx.enqueue_function[attention_v2_backward_prepare_kernel](queries.unsafe_ptr(),key_vectors.unsafe_ptr(),values.unsafe_ptr(),dy.unsafe_ptr(),visible_lo.unsafe_ptr(),visible_hi.unsafe_ptr(),row_max.unsafe_ptr(),denominator.unsafe_ptr(),row_zdot.unsafe_ptr(),Int32(rows),Int32(keys),Int32(width),Int32(head_dim),Int32(queries_per_group),scale,grid_dim=((rows+63)//64,1,1),block_dim=(64,1,1))
    ctx.enqueue_function[attention_v2_dq_kernel](queries.unsafe_ptr(),key_vectors.unsafe_ptr(),values.unsafe_ptr(),dy.unsafe_ptr(),visible_lo.unsafe_ptr(),visible_hi.unsafe_ptr(),row_max.unsafe_ptr(),denominator.unsafe_ptr(),row_zdot.unsafe_ptr(),dq.unsafe_ptr(),Int32(rows),Int32(keys),Int32(width),Int32(head_dim),Int32(queries_per_group),scale,grid_dim=((rows+63)//64,1,1),block_dim=(64,1,1))
    ctx.enqueue_function[feature_grad_kernel[True,4]](queries,key_vectors,values,dy,visible_lo,visible_hi,row_max,denominator,row_zdot,dk,Int32(rows),Int32(keys),Int32(width),Int32(head_dim),Int32(queries_per_group),scale,grid_dim=((groups*keys*((head_dim+3)//4)+127)//128,1,1),block_dim=(128,1,1))
    ctx.enqueue_function[feature_grad_kernel[False,4]](queries,key_vectors,values,dy,visible_lo,visible_hi,row_max,denominator,row_zdot,dv,Int32(rows),Int32(keys),Int32(width),Int32(head_dim),Int32(queries_per_group),scale,grid_dim=((groups*keys*((width+3)//4)+127)//128,1,1),block_dim=(128,1,1))
