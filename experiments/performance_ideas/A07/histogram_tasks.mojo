# SPDX-License-Identifier: Apache-2.0
"""Bounded independent histogram tasks built from actual node row spans.

Device counts plus scan compact a skewed forest frontier into <=256-row
tasks, preserving feature/bin output ownership and exact integer addends.
Static control assigns one block per node and walks the same 256-row
subtiles sequentially. Candidate assigns one block per task. Both write
the same task-indexed partials and merge each node's tasks in ascending
order; no floating atomics, host data routing, or persistent global queue.
Admission to a real forest caller still requires its exact integer bounds.
"""
from std.gpu import block_idx,thread_idx,block_dim
from std.atomic import Atomic
from max.gpu.host import DeviceContext,DeviceBuffer
from max.gpu.sync import barrier
from neighbors.impl.ball_cover.scan import rbc_exclusive_scan_launch
from experiments.performance_ideas.N07.streamed_histogram import _histogram_tile_body


def counts_kernel(offsets: MutPointer[Int32,MutAnyOrigin],counts: MutPointer[Int32,MutAnyOrigin],
    status: MutPointer[Int32,MutAnyOrigin],nodes: Int32,rows: Int32):
    var node=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if node<Int(nodes):
        var begin=Int(offsets.unsafe_load(node));var end=Int(offsets.unsafe_load(node+1))
        var invalid=begin<0 or end<begin or end>Int(rows)
        if node==0 and begin!=0:invalid=True
        if node==Int(nodes)-1 and end!=Int(rows):invalid=True
        status.unsafe_store(node,Int32(1) if invalid else Int32(0))
        if invalid:_=Atomic.fetch_add(status.unsafe_offset(Int(nodes)),Int32(1))
        counts.unsafe_store(node,Int32(0) if invalid else Int32((end-begin+255)//256))


def sanitize_counts_kernel(counts: MutPointer[Int32,MutAnyOrigin],status: MutPointer[Int32,MutAnyOrigin],nodes: Int32):
    var node=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    # A malformed offset vector can describe overlapping spans. Refuse
    # all tasks before scan, keeping its capacity proof and Int32 sum valid.
    if node<Int(nodes) and status.unsafe_load(Int(nodes))!=0:counts.unsafe_store(node,Int32(0))


def descriptors_kernel(offsets: MutPointer[Int32,MutAnyOrigin],prefix: MutPointer[Int32,MutAnyOrigin],
    begin_out: MutPointer[Int32,MutAnyOrigin],end_out: MutPointer[Int32,MutAnyOrigin],
    nodes: Int32,capacity: Int32):
    var task=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if task>=Int(capacity) or task>=Int(prefix.unsafe_load(Int(nodes))):return
    # Repeated prefix offsets represent empty nodes. Upper_bound selects
    # the unique nonempty span owning this actual task without host reads.
    var lo=0;var hi=Int(nodes)
    while lo<hi:
        var mid=(lo+hi)//2
        if Int(prefix.unsafe_load(mid+1))<=task:lo=mid+1
        else:hi=mid
    var begin=Int(offsets.unsafe_load(lo))+256*(task-Int(prefix.unsafe_load(lo)))
    begin_out.unsafe_store(task,Int32(begin))
    end_out.unsafe_store(task,Int32(min(begin+256,Int(offsets.unsafe_load(lo+1)))))


def tasked_kernel[CHUNK: Int,BINS: Int,REPLICAS: Int,TASKED: Bool](
    bins: MutPointer[Int32,MutAnyOrigin],weights: MutPointer[Int32,MutAnyOrigin],
    partials: MutPointer[Int64,MutAnyOrigin],status: MutPointer[Int32,MutAnyOrigin],
    prefix: MutPointer[Int32,MutAnyOrigin],begins: MutPointer[Int32,MutAnyOrigin],ends: MutPointer[Int32,MutAnyOrigin],
    nodes: Int32,features: Int32,max_abs: Int32):
    var fg=Int(block_idx.x);var owner=Int(block_idx.y)
    var first=owner;var last=owner+1
    comptime if TASKED:
        if owner>=Int(prefix.unsafe_load(Int(nodes))):return
    else:
        first=Int(prefix.unsafe_load(owner));last=Int(prefix.unsafe_load(owner+1))
    for task in range(first,last):
        _histogram_tile_body[CHUNK,BINS,REPLICAS](bins,weights,partials,status,features,max_abs,
            fg,task,Int(begins.unsafe_load(task)),Int(ends.unsafe_load(task)))
        # Control reuses the same shared histogram for its next subtile.
        # All lanes must finish reads before any lane zeroes that storage.
        barrier()


def node_merge_kernel[BINS: Int](partials: MutPointer[Int64,MutAnyOrigin],output: MutPointer[Int64,MutAnyOrigin],
    prefix: MutPointer[Int32,MutAnyOrigin],nodes: Int32,features: Int32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var cells=Int(features)*BINS
    if cell>=Int(nodes)*cells:return
    var node=cell//cells;var offset=cell%cells;var total=Int64(0)
    for task in range(Int(prefix.unsafe_load(node)),Int(prefix.unsafe_load(node+1))):
        total+=partials.unsafe_load(task*cells+offset)
    output.unsafe_store(cell,total)


# A07 experiment: NEVER RUN — PENDING VALIDATION; incumbent defaults retained.
# A07 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit integer node-task adapter; the production workload map has a separate opt-in.
def histogram_tasks[CHUNK: Int,BINS: Int,REPLICAS: Int,TASKED: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32],mut bins: DeviceBuffer[DType.int32],mut weights: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],mut prefix: DeviceBuffer[DType.int32],mut begins: DeviceBuffer[DType.int32],
    mut ends: DeviceBuffer[DType.int32],mut partials: DeviceBuffer[DType.int64],mut output: DeviceBuffer[DType.int64],
    mut status: DeviceBuffer[DType.int32],mut node_status: DeviceBuffer[DType.int32],
    nodes: Int,rows: Int,features: Int,max_abs: Int) raises:
    if nodes<1 or rows<1 or rows>2147483647 or nodes>2147483647-(rows+255)//256 or features<1 or features>2147483647//BINS or max_abs<0 or max_abs>2147483647//256:
        raise Error("histogram task integer/resource bound not proven")
    var capacity=(rows+255)//256+nodes
    var groups=(features+CHUNK-1)//CHUNK
    if len(offsets)<nodes+1 or len(bins)<rows*features or len(weights)<rows or len(counts)<nodes or len(prefix)<nodes+1 or len(begins)<capacity or len(ends)<capacity or len(partials)<capacity*features*BINS or len(output)<nodes*features*BINS or len(status)<capacity*groups or len(node_status)<nodes+1:
        raise Error("histogram task buffers too small")
    ctx.enqueue_memset(node_status,Int32(0))
    ctx.enqueue_function[counts_kernel](offsets,counts,node_status,Int32(nodes),Int32(rows),grid_dim=((nodes+127)//128,1,1),block_dim=(128,1,1))
    ctx.enqueue_function[sanitize_counts_kernel](counts,node_status,Int32(nodes),grid_dim=((nodes+127)//128,1,1),block_dim=(128,1,1))
    rbc_exclusive_scan_launch(ctx,prefix,counts,nodes)
    ctx.enqueue_function[descriptors_kernel](offsets,prefix,begins,ends,Int32(nodes),Int32(capacity),grid_dim=((capacity+127)//128,1,1),block_dim=(128,1,1))
    var owners=capacity if TASKED else nodes
    ctx.enqueue_function[tasked_kernel[CHUNK,BINS,REPLICAS,TASKED]](bins,weights,partials,status,prefix,begins,ends,
        Int32(nodes),Int32(features),Int32(max_abs),grid_dim=(groups,owners,1),block_dim=(128,1,1))
    ctx.enqueue_function[node_merge_kernel[BINS]](partials,output,prefix,Int32(nodes),Int32(features),grid_dim=((nodes*features*BINS+127)//128,1,1),block_dim=(128,1,1))
