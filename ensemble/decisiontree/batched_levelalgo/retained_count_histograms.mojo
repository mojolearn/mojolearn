# SPDX-License-Identifier: Apache-2.0
"""Bounded I18 forest caller for exact unweighted classification counts.

The cache keys physical columns, node indices and sampled-row intervals.
Only full-feature rounds are admitted. A right child uses parent-left only
when the exact parent interval and its retained column are available and its
left sibling was histogrammed (previously or in the current phase). Otherwise
the original row loop computes it. Integer counts have no rounding seam.
The byte budget includes metadata, validity and routing arrays. High node
indices simply stop caching. All data work and routing execute on device.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.sys.compile import is_defined
from ensemble.decisiontree.batched_levelalgo.kernels.level_loop_kernels import LOOP_H_CUR, LOOP_HDR_WORDS
from std.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from core.launch_clock import log_launch_ctx
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import NodeWorkItem, WorkloadInfo
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import HistogramArgs, _histogram_inner_loop, _histogram_inner_loop_binned
from ensemble.decisiontree.batched_levelalgo.objectives import ObjectiveLike


struct _RetainedAudit(Defaultable,Movable):
    var reused: Int64
    def __init__(out self):
        self.reused=Int64(0)

comptime _RETAINED_AUDIT=_Global[StorageType=_RetainedAudit,name="MojolearnRetainedHistogramAudit",init_fn=_RetainedAudit.__init__]

def retained_histogram_reused() raises -> Int:
    comptime if is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT"]():
        ref audit=_RETAINED_AUDIT.get_or_create_ptr()[]
        return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.reused)))
    return 0

@always_inline
def live_nodes(n: Int32,live: MutPointer[Int32,MutAnyOrigin]) -> Int:
    return min(Int(n),max(0,Int(live[LOOP_H_CUR])))


def remember_intervals(items: MutPointer[NodeWorkItem, MutAnyOrigin], starts: MutPointer[Int64, MutAnyOrigin], counts: MutPointer[Int32, MutAnyOrigin], known: MutPointer[UInt8, MutAnyOrigin], n: Int32, capacity: Int32, live: MutPointer[Int32,MutAnyOrigin]):
    var i = Int(block_idx.x)*128+Int(thread_idx.x)
    if i < live_nodes(n,live):
        ref item = items[i]
        if 0 <= item.idx < Int(capacity):
            starts[item.idx] = Int64(item.instances.begin)
            counts[item.idx] = Int32(item.instances.count)
            known[item.idx] = UInt8(1)


def route_siblings(items: MutPointer[NodeWorkItem, MutAnyOrigin], starts: MutPointer[Int64, MutAnyOrigin], counts: MutPointer[Int32, MutAnyOrigin], known: MutPointer[UInt8, MutAnyOrigin], parents: MutPointer[Int32, MutAnyOrigin], n: Int32, capacity: Int32, live: MutPointer[Int32,MutAnyOrigin]):
    var i = Int(block_idx.x)*128+Int(thread_idx.x)
    if i >= live_nodes(n,live):
        return
    var node = items[i].idx
    if node < 0 or node >= Int(capacity):
        return
    var parent = -1
    var left = node-1
    if node > 0 and node%2 == 0 and known[left] != 0:
        # NodeQueue appends a consecutive left/right pair for every split.
        # Row partition is [parent.begin,left.count],[left.end,right.count].
        if starts[left]+Int64(counts[left]) == starts[node]:
            var total = Int(counts[left])+Int(counts[node])
            for p in range(left):
                if known[p] != 0 and starts[p] == starts[left] and Int(counts[p]) == total:
                    parent = p
                    break
    parents[node] = Int32(parent)


def route_columns(items: MutPointer[NodeWorkItem, MutAnyOrigin], samples: MutPointer[Int32, MutAnyOrigin], valid: MutPointer[UInt8, MutAnyOrigin], parents: MutPointer[Int32, MutAnyOrigin], skip: MutPointer[UInt8, MutAnyOrigin], n: Int32, columns: Int32, capacity: Int32, col_start: Int32, width: Int32, live: MutPointer[Int32,MutAnyOrigin]):
    var task = Int(block_idx.x)*128+Int(thread_idx.x)
    if task >= live_nodes(n,live)*Int(width):
        return
    var batch_node = task//Int(width)
    var node = items[batch_node].idx
    if node < 0 or node >= Int(capacity):
        return
    var col = Int(samples[batch_node*Int(columns)+Int(col_start)+task%Int(width)])
    var parent = Int(parents[node])
    var reuse = parent >= 0
    if reuse:
        reuse = valid[parent*Int(columns)+col] != 0
    if reuse:
        var have_left = valid[(node-1)*Int(columns)+col] != 0
        if not have_left:
            # Each node samples a different permutation, even for the full
            # feature set. Only a left column computed in THIS pass (or
            # already retained) can supply this right column.
            for j in range(live_nodes(n,live)):
                if items[j].idx == node-1:
                    for c in range(Int(width)):
                        if Int(samples[j*Int(columns)+Int(col_start)+c]) == col:
                            have_left = True
                    break
        reuse = have_left
    skip[node*Int(columns)+col] = UInt8(1) if reuse else UInt8(0)


def count_rows[O: ObjectiveLike, BINNED: Bool, SAMPLED: Bool](hist: MutPointer[O.BinT, MutAnyOrigin], items: MutPointer[NodeWorkItem, MutAnyOrigin], workloads: MutPointer[WorkloadInfo, MutAnyOrigin], samples: MutPointer[Int32, MutAnyOrigin], args: MutPointer[HistogramArgs[O], MutAnyOrigin], skip: MutPointer[UInt8, MutAnyOrigin], col_start: Int32, max_bins: Int32, columns: Int32, capacity: Int32, n: Int32, live: MutPointer[Int32,MutAnyOrigin]):
    comptime assert O.BinT.is_classification and not O.BinT.weighted
    ref data = args[].dataset
    ref workload = workloads[Int(block_idx.x)]
    var batch_node = Int(workload.nodeid)
    if batch_node<0 or batch_node>=live_nodes(n,live):
        return
    ref item = items[batch_node]
    var sampled_col = Int(col_start)+Int(block_idx.y)
    var physical_col = samples[batch_node*Int(data.n_sampled_cols)+sampled_col]
    if 0 <= item.idx < Int(capacity):
        if skip[item.idx*Int(columns)+Int(physical_col)] != 0:
            return
    var bins = args[].quantiles.n_bins_array[Int(physical_col)]
    var classes = args[].objective.NumClasses()
    var output = hist.unsafe_offset((batch_node*Int(grid_dim.y)+Int(block_idx.y))*Int(max_bins)*Int(classes))
    var tid = Int(thread_idx.x)+Int(workload.offset_blockid)*Int(block_dim.x)
    var stride = Int(block_dim.x)*Int(workload.num_blocks)
    comptime if BINNED:
        _histogram_inner_loop_binned[sampled_labels=SAMPLED](args[].objective, data, output, physical_col, bins, item.instances.begin, item.instances.begin+item.instances.count, tid, stride)
    else:
        _histogram_inner_loop[sampled_labels=SAMPLED](args[].objective, data, output, args[].quantiles.quantiles_array.unsafe_offset(Int(max_bins)*Int(physical_col)), physical_col, bins, item.instances.begin, item.instances.begin+item.instances.count, tid, stride)


def retain_computed[O: ObjectiveLike](hist: MutPointer[O.BinT, MutAnyOrigin], items: MutPointer[NodeWorkItem, MutAnyOrigin], samples: MutPointer[Int32, MutAnyOrigin], cache: MutPointer[UInt32, MutAnyOrigin], valid: MutPointer[UInt8, MutAnyOrigin], skip: MutPointer[UInt8, MutAnyOrigin], n: Int32, col_start: Int32, width: Int32, columns: Int32, slots: Int32, capacity: Int32, live: MutPointer[Int32,MutAnyOrigin]):
    var cell = Int(block_idx.x)*128+Int(thread_idx.x)
    if cell >= live_nodes(n,live)*Int(width)*Int(slots):
        return
    var task = cell//Int(slots)
    var node = items[task//Int(width)].idx
    if node < 0 or node >= Int(capacity):
        return
    var col = Int(samples[(task//Int(width))*Int(columns)+Int(col_start)+task%Int(width)])
    var key = node*Int(columns)+col
    if skip[key] == 0:
        cache[key*Int(slots)+cell%Int(slots)] = hist.unsafe_bitcast[UInt32]()[cell]
        if cell%Int(slots) == 0:
            valid[key] = UInt8(1)


def subtract_retained[O: ObjectiveLike](hist: MutPointer[O.BinT, MutAnyOrigin], items: MutPointer[NodeWorkItem, MutAnyOrigin], samples: MutPointer[Int32, MutAnyOrigin], cache: MutPointer[UInt32, MutAnyOrigin], valid: MutPointer[UInt8, MutAnyOrigin], parents: MutPointer[Int32, MutAnyOrigin], skip: MutPointer[UInt8, MutAnyOrigin], n: Int32, col_start: Int32, width: Int32, columns: Int32, slots: Int32, capacity: Int32, live: MutPointer[Int32,MutAnyOrigin], reused: MutPointer[UInt32,MutAnyOrigin]):
    var cell = Int(block_idx.x)*128+Int(thread_idx.x)
    if cell >= live_nodes(n,live)*Int(width)*Int(slots):
        return
    var task = cell//Int(slots)
    var node = items[task//Int(width)].idx
    if node < 0 or node >= Int(capacity):
        return
    var col = Int(samples[(task//Int(width))*Int(columns)+Int(col_start)+task%Int(width)])
    var key = node*Int(columns)+col
    if skip[key] != 0:
        var offset = cell%Int(slots)
        var p = Int(parents[node])
        # Admission proves each child is a disjoint subset of this parent;
        # sampled rows are Int32-bounded, so neither count nor subtraction
        # can overflow. I18's independent operator exercises refusal gates.
        var value = cache[(p*Int(columns)+col)*Int(slots)+offset]-cache[((node-1)*Int(columns)+col)*Int(slots)+offset]
        hist.unsafe_bitcast[UInt32]()[cell] = value
        cache[key*Int(slots)+offset] = value
        if offset == 0:
            valid[key] = UInt8(1)
            comptime if is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT"]():
                _ = Atomic.fetch_add(reused,UInt32(1))


struct RetainedCountHistograms(Movable):
    var capacity: Int
    var columns: Int
    var slots: Int
    var cache: DeviceBuffer[DType.uint32]
    var starts: DeviceBuffer[DType.int64]
    var counts: DeviceBuffer[DType.int32]
    var known: DeviceBuffer[DType.uint8]
    var valid: DeviceBuffer[DType.uint8]
    var parents: DeviceBuffer[DType.int32]
    var skip: DeviceBuffer[DType.uint8]
    var live: MutPointer[Int32,MutUntrackedOrigin]
    var exact_header: DeviceBuffer[DType.int32]
    var reused: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, capacity: Int, columns: Int, slots: Int) raises:
        self.exact_header=ctx.enqueue_create_buffer[DType.int32](LOOP_HDR_WORDS)
        self.live=self.exact_header.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        self.reused=ctx.enqueue_create_buffer[DType.uint32](1)
        self.capacity=capacity; self.columns=columns; self.slots=slots
        self.cache=ctx.enqueue_create_buffer[DType.uint32](capacity*columns*slots)
        self.starts=ctx.enqueue_create_buffer[DType.int64](capacity)
        self.counts=ctx.enqueue_create_buffer[DType.int32](capacity)
        self.known=ctx.enqueue_create_buffer[DType.uint8](capacity)
        self.valid=ctx.enqueue_create_buffer[DType.uint8](capacity*columns)
        self.parents=ctx.enqueue_create_buffer[DType.int32](capacity)
        self.skip=ctx.enqueue_create_buffer[DType.uint8](capacity*columns)

    def reset(mut self, ctx: DeviceContext) raises:
        self.reused.enqueue_fill(UInt32(0))
        self.known.enqueue_fill(UInt8(0))
        self.valid.enqueue_fill(UInt8(0))

    def prepare(mut self,ctx: DeviceContext,items: MutPointer[NodeWorkItem,MutAnyOrigin],n: Int) raises:
        self.exact_header.enqueue_fill(Int32(n))
        self.prepare_device(ctx,items,n,self.exact_header.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())

    def prepare_device(mut self,ctx: DeviceContext,items: MutPointer[NodeWorkItem,MutAnyOrigin],n: Int,live: MutPointer[Int32,MutAnyOrigin]) raises:
        self.live=live.unsafe_origin_cast[MutUntrackedOrigin]()
        log_launch_ctx(ctx,"retained_hist_intervals")
        ctx.enqueue_function[remember_intervals](items,self.starts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.counts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.known.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(n),Int32(self.capacity),self.live.unsafe_origin_cast[MutAnyOrigin](),grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
        log_launch_ctx(ctx,"retained_hist_parents")
        ctx.enqueue_function[route_siblings](items,self.starts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.counts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.known.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.parents.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(n),Int32(self.capacity),self.live.unsafe_origin_cast[MutAnyOrigin](),grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))

    def enqueue[O: ObjectiveLike, BINNED: Bool, SAMPLED: Bool](mut self, ctx: DeviceContext, hist: MutPointer[O.BinT, MutAnyOrigin], items: MutPointer[NodeWorkItem, MutAnyOrigin], workloads: MutPointer[WorkloadInfo, MutAnyOrigin], samples: MutPointer[Int32, MutAnyOrigin], args: MutPointer[HistogramArgs[O], MutAnyOrigin], col: Int, bins: Int, blocks: Int, width: Int, n: Int) raises:
        log_launch_ctx(ctx,"retained_hist_column_routes")
        ctx.enqueue_function[route_columns](items,samples,self.valid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.parents.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.skip.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(n),Int32(self.columns),Int32(self.capacity),Int32(col),Int32(width),self.live.unsafe_origin_cast[MutAnyOrigin](),grid_dim=((n*width+127)//128,1,1),block_dim=(128,1,1))
        log_launch_ctx(ctx,"retained_hist_compute")
        ctx.enqueue_function[count_rows[O,BINNED,SAMPLED]](hist,items,workloads,samples,args,self.skip.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(col),Int32(bins),Int32(self.columns),Int32(self.capacity),Int32(n),self.live.unsafe_origin_cast[MutAnyOrigin](),grid_dim=(blocks,width,1),block_dim=(128,1,1))
        var cells=n*width*self.slots
        log_launch_ctx(ctx,"retained_hist_store")
        ctx.enqueue_function[retain_computed[O]](hist,items,samples,self.cache.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.valid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.skip.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(n),Int32(col),Int32(width),Int32(self.columns),Int32(self.slots),Int32(self.capacity),self.live.unsafe_origin_cast[MutAnyOrigin](),grid_dim=((cells+127)//128,1,1),block_dim=(128,1,1))
        log_launch_ctx(ctx,"retained_hist_subtract")
        ctx.enqueue_function[subtract_retained[O]](hist,items,samples,self.cache.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.valid.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.parents.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),self.skip.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),Int32(n),Int32(col),Int32(width),Int32(self.columns),Int32(self.slots),Int32(self.capacity),self.live.unsafe_origin_cast[MutAnyOrigin](),self.reused.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),grid_dim=((cells+127)//128,1,1),block_dim=(128,1,1))

    def publish_audit(mut self,ctx: DeviceContext) raises:
        comptime if is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT"]():
            ref audit=_RETAINED_AUDIT.get_or_create_ptr()[]
            with self.reused.map_to_host() as values:
                _ = Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.reused),Int64(values[0]))
