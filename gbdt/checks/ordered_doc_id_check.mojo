# SPDX-License-Identifier: Apache-2.0
"""F12 pending device qualification: actual PointHist8 with document-ID keys.

Off-grid stats, reverse nonidentity map, scalar and vector loop variants,
unaligned head/tail and empty input. Independent host quantization oracle.
"""
from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from gbdt.methods.kernel.compute_point_hist2_loop import (
    compute_histogram,
    compute_histogram_2,
    compute_histogram_4,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_SMEM_FLOATS,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_8bit import (
    PW8_MAX_FOLD_COUNT,
    PointHist8,
)

from gbdt.host.gbdt_oracle import _hist2_dither, _hist2_quantize
from gbdt.methods.ordered_fast_switches import ORD_DOC_ID_STORAGE

comptime N_ROWS=521
comptime OUT_INTS=4*PW8_MAX_FOLD_COUNT*2

def hist8_kernel[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    scale: Float32,
    out_buf: MutPointer[Int32, MutAnyOrigin],
    keys: MutPointer[UInt32,MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_SMEM_FLOATS, Int32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist8(smem, scale)

    comptime if variant == 1:
        compute_histogram[PW_HIST2_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex, dither_ids=keys,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex, dither_ids=keys,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex, dither_ids=keys,
        )
    else:
        compute_histogram_4[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex, dither_ids=keys,
        )

    barrier()
    var t = Int(thread_idx.x)
    for k in range(OUT_INTS // PW_HIST2_BLOCK):
        var at = t + k * PW_HIST2_BLOCK
        if at < OUT_INTS:
            out_buf.unsafe_store(at, smem.unsafe_load(at))


def main() raises:
    var ctx=DeviceContext()
    var keys=List[UInt32]();var idx=List[UInt32]();var ci=List[UInt32]()
    var target=List[Float32]();var weight=List[Float32]()
    for row in range(N_ROWS):
        keys.append(UInt32(N_ROWS-1-row));idx.append(UInt32(row))
        ci.append(UInt32((row%251)<<24)|UInt32(((row*3)%251)<<16)|UInt32(((row*7)%251)<<8)|UInt32((row*11)%251))
        target.append(Float32((row*17)%97)*Float32(.137)+Float32(.3141592))
        weight.append(Float32((row*23)%101)*Float32(.071)+Float32(.2718281))
    var dk=ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var di=ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var dc=ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var dt=ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var dw=ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var dout=ctx.enqueue_create_buffer[DType.int32](OUT_INTS)
    var out=ctx.enqueue_create_host_buffer[DType.int32](OUT_INTS)
    ctx.enqueue_copy(dst_buf=dk,src_ptr=keys.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=di,src_ptr=idx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc,src_ptr=ci.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dt,src_ptr=target.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dw,src_ptr=weight.unsafe_ptr())
    var scale=Float32(3.7)
    var lengths:List[Int]=[0,1,31,127,257,509]
    for length in lengths:
        var want=List[Int64]()
        for _ in range(OUT_INTS):want.append(Int64(0))
        for row in range(5,5+length):
            var key=Int(keys[row]) if ORD_DOC_ID_STORAGE else row
            var u=_hist2_dither(key)
            for feature in range(4):
                var bin=Int((ci[row]>>UInt32(24-8*feature))&UInt32(255))
                var cell=2*(PW8_MAX_FOLD_COUNT*feature+bin)
                want[cell]+=Int64(_hist2_quantize(weight[row],scale,u))
                want[cell+1]+=Int64(_hist2_quantize(target[row],scale,u))
        comptime for vi in range(4):
            comptime variant=(1,4,12,14)[vi]
            ctx.enqueue_memset(dout,Int32(0))
            ctx.enqueue_function[hist8_kernel[variant]](
                di.unsafe_ptr(),Int32(5),Int32(length),dt.unsafe_ptr(),dw.unsafe_ptr(),dc.unsafe_ptr(),
                scale,dout.unsafe_ptr(),dk.unsafe_ptr(),grid_dim=1,block_dim=PW_HIST2_BLOCK)
            ctx.enqueue_copy(dst_buf=out,src_buf=dout)
            ctx.synchronize()
            for cell in range(OUT_INTS):
                if Int64(out[cell])!=want[cell]:raise Error("document-ID histogram oracle mismatch")
    print("ORDERED_DOC_ID_CHECK PASS mapped=",ORD_DOC_ID_STORAGE," cells=",OUT_INTS," cases=24")
