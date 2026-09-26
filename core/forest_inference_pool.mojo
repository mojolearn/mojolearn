# SPDX-License-Identifier: Apache-2.0
"""Resident whole-grove RF/ET model ownership with the original fixed32 fold.

A logical grove owns trees g,g+32,... independently of device count. Devices
store complete groves. Only their unaveraged totals travel to the root, whose
16,8,4,2,1 fold and single global-tree division are the original GPU contract.
This is model capacity pooling, not equivalence to legacy host prediction.
"""
from std.os import getenv
from std.sys.compile import is_defined
from std.memory import bitcast, stack_allocation
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div
from core.forest_inference import forest_add, reached_leaf, require_finite, FOREST_PACKED_NODES


def forest_device_count() raises -> Int:
    var count = Int(getenv("MOJOLEARN_FOREST_DEVICE_COUNT", "1"))
    if count < 1 or count > 64:
        raise Error("forest pool requires 1..64 devices")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("forest pool requires IDENTICAL")
    return count


def forest_owned_groves_kernel[RF_INPUT: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], starts: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    totals: MutPointer[Float32, MutAnyOrigin], items_in: Int32,
    first_item_in: Int32, features_in: Int32, outputs_in: Int32,
):
    var tid = Int(thread_idx.x)
    var grove = tid % 32
    var item = Int(block_idx.x) * 4 + tid // 32
    var outputs = Int(outputs_in)
    if item < Int(items_in):
        var source_item = Int(first_item_in) + item
        var total = Float32(0.0)
        var first = Int(starts.unsafe_load(grove))
        var count = Int(counts.unsafe_load(grove))
        for local in range(first, first + count):
            var node = reached_leaf[RF_INPUT, PACKED](offsets, columns,
                thresholds, left, x, local, source_item // outputs, Int(features_in))
            total = forest_add(total, leaves.unsafe_load(node * outputs + source_item % outputs))
        totals.unsafe_store(item * 32 + grove, total)


def forest_finish_groves_kernel(
    totals: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    items_in: Int32, trees_in: Int32,
):
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var item = Int(block_idx.x) * 4 + tid // 32
    var value = Float32(0.0)
    if item < Int(items_in):
        value = totals.unsafe_load(item * 32 + lane)
    var sums = stack_allocation[128, Float32, address_space=AddressSpace.SHARED]()
    sums[unsafe_offset=tid] = value
    barrier()
    var step = 16
    while step > 0:
        if lane < step:
            sums[unsafe_offset=tid] = forest_add(sums[unsafe_offset=tid], sums[unsafe_offset=tid+step])
        barrier()
        step //= 2
    if lane == 0 and item < Int(items_in):
        output.unsafe_store(item, ftz(identical_div(ftz(sums[unsafe_offset=tid]), Float32(trees_in))))


struct ForestGroveOwner(Movable):
    var ctx: Optional[DeviceContext]
    var offsets: Optional[DeviceBuffer[DType.int32]]
    var columns: Optional[DeviceBuffer[DType.int32]]
    var thresholds: Optional[DeviceBuffer[DType.float32]]
    var left: Optional[DeviceBuffer[DType.int32]]
    var leaves: Optional[DeviceBuffer[DType.float32]]
    var starts: Optional[DeviceBuffer[DType.int32]]
    var counts: Optional[DeviceBuffer[DType.int32]]
    var groves: List[Int]
    var trees: Int
    var nodes: Int

    def __init__(out self, rank: Int, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        starts: List[Int32], counts: List[Int32], outputs: Int,
        owned: List[Int]) raises:
        self.ctx = Optional[DeviceContext]()
        self.offsets = Optional[DeviceBuffer[DType.int32]]()
        self.columns = Optional[DeviceBuffer[DType.int32]]()
        self.thresholds = Optional[DeviceBuffer[DType.float32]]()
        self.left = Optional[DeviceBuffer[DType.int32]]()
        self.leaves = Optional[DeviceBuffer[DType.float32]]()
        self.starts = Optional[DeviceBuffer[DType.int32]]()
        self.counts = Optional[DeviceBuffer[DType.int32]]()
        self.groves = owned.copy()
        self.trees = len(offsets) - 1
        self.nodes = len(columns)
        var packed_nodes = List[Int32]()
        var compact_leaves = List[Float32]()
        comptime if FOREST_PACKED_NODES:
            for node in range(len(columns)):
                var payload = bitcast[DType.int32](thresholds[node])
                if left[node] == -1:
                    payload = Int32(len(compact_leaves) // outputs)
                    for c in range(outputs):
                        compact_leaves.append(leaves[node * outputs + c])
                packed_nodes.append(payload)
                packed_nodes.append(left[node])
                packed_nodes.append(columns[node])
                packed_nodes.append(0)
        self.ctx = DeviceContext(device_id=rank)
        try:
            self.offsets = self.ctx.value().enqueue_create_buffer[DType.int32](len(offsets))
            self.starts = self.ctx.value().enqueue_create_buffer[DType.int32](32)
            self.counts = self.ctx.value().enqueue_create_buffer[DType.int32](32)
            self.ctx.value().enqueue_copy(dst_buf=self.offsets.value(), src_ptr=offsets.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.starts.value(), src_ptr=starts.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.counts.value(), src_ptr=counts.unsafe_ptr())
            comptime if FOREST_PACKED_NODES:
                self.columns = self.ctx.value().enqueue_create_buffer[DType.int32](len(packed_nodes))
                self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](1)
                self.left = self.ctx.value().enqueue_create_buffer[DType.int32](1)
                self.leaves = self.ctx.value().enqueue_create_buffer[DType.float32](len(compact_leaves))
                self.ctx.value().enqueue_copy(dst_buf=self.columns.value(), src_ptr=packed_nodes.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.leaves.value(), src_ptr=compact_leaves.unsafe_ptr())
            else:
                self.columns = self.ctx.value().enqueue_create_buffer[DType.int32](len(columns))
                self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](len(thresholds))
                self.left = self.ctx.value().enqueue_create_buffer[DType.int32](len(left))
                self.leaves = self.ctx.value().enqueue_create_buffer[DType.float32](len(leaves))
                self.ctx.value().enqueue_copy(dst_buf=self.columns.value(), src_ptr=columns.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.thresholds.value(), src_ptr=thresholds.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.left.value(), src_ptr=left.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.leaves.value(), src_ptr=leaves.unsafe_ptr())
            self.ctx.value().synchronize()
            _ = len(offsets)
            _ = len(starts)
            _ = len(counts)
            _ = len(columns)
            _ = len(thresholds)
            _ = len(left)
            _ = len(leaves)
        except e:
            self.ctx.value().synchronize()
            _ = len(offsets)
            _ = len(starts)
            _ = len(counts)
            _ = len(columns)
            _ = len(thresholds)
            _ = len(left)
            _ = len(leaves)
            _ = packed_nodes^
            _ = compact_leaves^
            raise e
        _ = packed_nodes^
        _ = compact_leaves^

    def __deinit__(deinit self):
        _ = self.counts^
        _ = self.starts^
        _ = self.leaves^
        _ = self.left^
        _ = self.thresholds^
        _ = self.columns^
        _ = self.offsets^
        # DEVIATION 3010 (DEVIATION 2520's drain): the frees enqueued by the
        # releases above must complete before the context is destroyed, or
        # the runtime allocator's lock is left held and the next context's
        # first allocation never returns. Host-side drain; no output bit.
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def contribution[RF_INPUT: Bool](mut self,
        x: MutPointer[Float32, MutAnyOrigin], first_item: Int, items: Int,
        features: Int, outputs: Int,
    ) raises -> List[Float32]:
        var first_row = first_item // outputs
        var rows = (first_item + items + outputs - 1) // outputs - first_row
        var relative_item = first_item - first_row * outputs
        ref ctx = self.ctx.value()
        var dx = ctx.enqueue_create_buffer[DType.float32](rows * features)
        var dt = ctx.enqueue_create_buffer[DType.float32](items * 32)
        var host = ctx.enqueue_create_host_buffer[DType.float32](items * 32)
        try:
            ctx.enqueue_copy(dst_buf=dx, src_ptr=x.unsafe_offset(first_row * features))
            ctx.enqueue_function[forest_owned_groves_kernel[RF_INPUT, FOREST_PACKED_NODES]](
                self.offsets.value().unsafe_ptr(), self.columns.value().unsafe_ptr(),
                self.thresholds.value().unsafe_ptr(), self.left.value().unsafe_ptr(),
                self.leaves.value().unsafe_ptr(), self.starts.value().unsafe_ptr(),
                self.counts.value().unsafe_ptr(), dx.unsafe_ptr(), dt.unsafe_ptr(),
                Int32(items), Int32(relative_item), Int32(features), Int32(outputs),
                grid_dim=(items + 3) // 4, block_dim=128)
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dt)
            ctx.synchronize()
        except e:
            ctx.synchronize()
            _ = dx^
            _ = dt^
            _ = host^
            raise e
        var result = List[Float32](capacity=items * 32)
        for i in range(items * 32):
            result.append(host.unsafe_ptr()[i])
        _ = dx^
        _ = dt^
        _ = host^
        return result^


struct PooledForest(Movable):
    var owners: List[ForestGroveOwner]
    var features: Int
    var outputs: Int
    var trees: Int

    def __init__(out self, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        features: Int, outputs: Int, requested: Int) raises:
        self.owners = List[ForestGroveOwner]()
        self.features = features
        self.outputs = outputs
        self.trees = len(offsets) - 1
        var active = min(requested, min(32, self.trees))
        var assignments = List[Int](length=32, fill=-1)
        var loads = List[Int](length=active, fill=0)
        for grove in range(min(32, self.trees)):
            var owner = 0
            for rank in range(1, active):
                if loads[rank] < loads[owner]:
                    owner = rank
            assignments[grove] = owner
            for tree in range(grove, self.trees, 32):
                var first = Int(offsets[tree])
                var last = Int(offsets[tree+1])
                var size = (last - first) * (12 + 4 * outputs) + 4
                comptime if FOREST_PACKED_NODES:
                    size = (last - first) * 16 + 4
                    for node in range(first, last):
                        if left[node] == -1:
                            size += 4 * outputs
                loads[owner] += size
        for rank in range(active):
            var loff = List[Int32]()
            loff.append(0)
            var col = List[Int32]()
            var thr = List[Float32]()
            var child = List[Int32]()
            var vals = List[Float32]()
            var starts = List[Int32](length=32, fill=Int32(0))
            var counts = List[Int32](length=32, fill=Int32(0))
            var groves = List[Int]()
            for grove in range(32):
                if assignments[grove] != rank:
                    continue
                groves.append(grove)
                starts[grove] = Int32(len(loff) - 1)
                for tree in range(grove, self.trees, 32):
                    counts[grove] += 1
                    for node in range(Int(offsets[tree]), Int(offsets[tree+1])):
                        var leaf_base = node * outputs
                        comptime if is_defined["MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE"]():
                            # Check-only arm: later owners read their leaf
                            # values one node early. The inner loop still runs
                            # `outputs` times, so len(vals) and every buffer
                            # sized from it are unchanged, and the node
                            # topology (`left`) is untouched so the packed
                            # leaf count cannot move either. INERT AT ONE
                            # DEVICE: grove 0 always lands on owner 0, so a
                            # rank above 0 owns only trees above 0 and
                            # `node > 0` holds; the `node > 0` test states
                            # that rather than assuming it.
                            if rank > 0 and node > 0:
                                leaf_base = (node - 1) * outputs
                        col.append(columns[node])
                        thr.append(thresholds[node])
                        child.append(left[node])
                        for c in range(outputs):
                            vals.append(leaves[leaf_base + c])
                    loff.append(Int32(len(col)))
            self.owners.append(ForestGroveOwner(rank, loff, col, thr, child,
                vals, starts, counts, outputs, groves))

    def collect[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        rows: Int,
    ) raises -> List[Float32]:
        return self.collect_items[RF_INPUT](x, 0, rows * self.outputs)

    def collect_items[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        first_item: Int, items: Int,
    ) raises -> List[Float32]:
        var totals = List[Float32](length=items * 32, fill=Float32(0.0))
        for rank in range(len(self.owners)):
            ref owner = self.owners[rank]
            var local = owner.contribution[RF_INPUT](x, first_item, items, self.features, self.outputs)
            for item in range(items):
                for g in range(len(owner.groves)):
                    var grove = owner.groves[g]
                    totals[item * 32 + grove] = local[item * 32 + grove]
            comptime if is_defined["MOJOLEARN_FOREST_POOL_FAULT"]():
                if Int(getenv("MOJOLEARN_FOREST_FAIL_OWNER", "-1")) == rank:
                    raise Error("injected forest owner failure after contribution")
        return totals^

    def predict_into[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        output: MutPointer[Float32, MutAnyOrigin], rows: Int,
    ) raises:
        var staged = List[Float32](length=rows * self.outputs, fill=Float32(0.0))
        # Bound both rows and output cells: even wide vector leaves never
        # create more than 4096*32 canonical reduction cells on any device.
        var tile = min(4096, 64 * self.outputs)
        for first in range(0, rows * self.outputs, tile):
            var items = min(tile, rows * self.outputs - first)
            var totals = self.collect_items[RF_INPUT](x, first, items)
            ref ctx = self.owners[0].ctx.value()
            var dt = ctx.enqueue_create_buffer[DType.float32](len(totals))
            var dout = ctx.enqueue_create_buffer[DType.float32](items)
            var host = ctx.enqueue_create_host_buffer[DType.float32](items)
            try:
                ctx.enqueue_copy(dst_buf=dt, src_ptr=totals.unsafe_ptr())
                ctx.enqueue_function[forest_finish_groves_kernel](dt.unsafe_ptr(),
                    dout.unsafe_ptr(), Int32(items), Int32(self.trees),
                    grid_dim=(items + 3) // 4, block_dim=128)
                ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dout)
                ctx.synchronize()
            except e:
                ctx.synchronize()
                _ = totals^
                _ = dt^
                _ = dout^
                _ = host^
                raise e
            _ = totals^
            for i in range(items):
                staged[first + i] = host.unsafe_ptr()[i]
            _ = dt^
            _ = dout^
            _ = host^
        require_finite(staged)
        for i in range(rows * self.outputs):
            output[i] = staged[i]
