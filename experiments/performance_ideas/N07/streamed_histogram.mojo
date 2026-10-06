# SPDX-License-Identifier: Apache-2.0
"""Bounded feature chunks and shared integer histogram replicas.

Input weights are the caller's existing exact Int32 accumulator addends;
this adapter performs no float quantization. One row tile owns partials,
then an exact Int64 merge runs per histogram cell in ascending tile order.
The caller proves 256*max_abs<=Int32.max; the device also rejects addends or
bin IDs outside that contract. No float atomics or unsupported global queue.
"""
from std.atomic import Atomic,Ordering
from std.gpu import block_idx,thread_idx,block_dim
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext,DeviceBuffer


@always_inline
def _histogram_tile_body[CHUNK: Int,BINS: Int,REPLICAS: Int](
    bins: MutPointer[Int32,MutAnyOrigin],weights: MutPointer[Int32,MutAnyOrigin],
    partials: MutPointer[Int64,MutAnyOrigin],status: MutPointer[Int32,MutAnyOrigin],
    features: Int32,max_abs: Int32,fg: Int,tile: Int,row_begin: Int,row_end: Int):
    comptime assert REPLICAS==1 or REPLICAS==4
    comptime assert CHUNK*BINS*REPLICAS*4<=16384, "bounded shared replication"
    var hist=stack_allocation[CHUNK*BINS*REPLICAS,Int32,address_space=AddressSpace.SHARED]()
    var tid=Int(thread_idx.x)
    for cell in range(tid,CHUNK*BINS*REPLICAS,128):
        hist[cell]=Int32(0)
    if tid==0:
        status.unsafe_store(tile*((Int(features)+CHUNK-1)//CHUNK)+fg,Int32(0))
    barrier()
    var replica=0
    comptime if REPLICAS==4:
        replica=tid//32
    for row in range(row_begin+tid,row_end,128):
        var weight=weights.unsafe_load(row)
        if Int64(weight)>Int64(max_abs) or Int64(weight)<-Int64(max_abs):
            _=Atomic.fetch_add(status.unsafe_offset(tile*((Int(features)+CHUNK-1)//CHUNK)+fg),Int32(1))
            continue
        comptime for f in range(CHUNK):
            var feature=fg*CHUNK+f
            if feature<Int(features):
                var bin=Int(bins.unsafe_load(row*Int(features)+feature))
                if bin<0 or bin>=BINS:
                    _=Atomic.fetch_add(status.unsafe_offset(tile*((Int(features)+CHUNK-1)//CHUNK)+fg),Int32(1))
                else:
                    _=Atomic.fetch_add[ordering=Ordering.RELAXED](hist.unsafe_offset((replica*CHUNK+f)*BINS+bin),weight)
    barrier()
    for cell in range(tid,CHUNK*BINS,128):
        var feature=fg*CHUNK+cell//BINS
        if feature<Int(features):
            var total=Int64(0)
            comptime for replica_id in range(REPLICAS):
                total+=Int64(hist[replica_id*CHUNK*BINS+cell])
            partials.unsafe_store(tile*Int(features)*BINS+feature*BINS+cell%BINS,total)


def histogram_kernel[CHUNK: Int,BINS: Int,REPLICAS: Int](
    bins: MutPointer[Int32,MutAnyOrigin],weights: MutPointer[Int32,MutAnyOrigin],
    partials: MutPointer[Int64,MutAnyOrigin],status: MutPointer[Int32,MutAnyOrigin],
    rows: Int32,features: Int32,max_abs: Int32):
    var tile=Int(block_idx.y)
    _histogram_tile_body[CHUNK,BINS,REPLICAS](bins,weights,partials,status,features,max_abs,
        Int(block_idx.x),tile,tile*256,min(tile*256+256,Int(rows)))

def merge_kernel(partials: MutPointer[Int64,MutAnyOrigin],output: MutPointer[Int64,MutAnyOrigin],cells: Int32,tiles: Int32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<Int(cells):
        var total=Int64(0)
        for tile in range(Int(tiles)):
            total+=partials.unsafe_load(tile*Int(cells)+cell)
        output.unsafe_store(cell,total)


# N07 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit bounded integer adapter; production fixed-point weighted route is separately guarded.
def streamed_histogram[CHUNK: Int,BINS: Int,REPLICAS: Int](ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.int32],mut weights: DeviceBuffer[DType.int32],
    mut partials: DeviceBuffer[DType.int64],mut output: DeviceBuffer[DType.int64],
    mut status: DeviceBuffer[DType.int32],rows: Int,features: Int,max_abs: Int) raises:
    if rows<1 or rows>2147483647 or features<1 or features>2147483647//BINS or max_abs<0 or max_abs>2147483647//256:
        raise Error("histogram signed integer bound is not proven")
    var tiles=(rows+255)//256
    var featuregroups=(features+CHUNK-1)//CHUNK
    if len(bins)<rows*features or len(weights)<rows or len(partials)<tiles*features*BINS or len(output)<features*BINS or len(status)<tiles*featuregroups:
        raise Error("histogram buffers too small")
    ctx.enqueue_function[histogram_kernel[CHUNK,BINS,REPLICAS]](bins,weights,partials,status,Int32(rows),Int32(features),Int32(max_abs),grid_dim=(featuregroups,tiles,1),block_dim=(128,1,1))
    ctx.enqueue_function[merge_kernel](partials,output,Int32(features*BINS),Int32(tiles),grid_dim=((features*BINS+127)//128,1,1),block_dim=(128,1,1))
