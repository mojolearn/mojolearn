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
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div
from std.atomic import Atomic
from core.device_fold import device_exclusive_scan_total
from core.multi_gpu import transfer_bytes
from core.forest_inference import (
    forest_add, reached_leaf, FOREST_PACKED_NODES, FOREST_FINITE_TPB, device_ptr_all_finite,
    device_all_finite, launch_forest_argmax, forest_pack_resident, forest_validate_device,
    _forest_tree_of,
)


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


# Grove partition and per-owner gather on the device (lane cpu4-misc). The
# host no longer walks trees or nodes to size groves or to build each
# owner's arrays: the full flat model is uploaded to an owner's device once,
# the grove sizes come from one atomic count kernel (exact Int32 adds), and
# each owner gathers its groves' trees into compact buffers with a tree map
# kernel, an exclusive scan of the tree sizes and a node gather kernel. The
# owner layout is the old host one (owned groves ascending, each grove's
# trees g, g+32, ... in order, node order and tree-local children kept), so
# the per-grove totals, and the root fold, keep their bits.


def forest_grove_sizes_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    trees_in: Int32, nodes_in: Int32, sizes: MutPointer[Int32, MutAnyOrigin],
):
    """sizes[g] += nodes of grove g, sizes[32 + g] += leaves of grove g
    (one thread per node, integer atomics: order-free and exact)."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(nodes_in):
        return
    var t = _forest_tree_of(offsets, Int(trees_in), i)
    var g = t % 32
    _ = Atomic.fetch_add(sizes.unsafe_offset(g), Int32(1))
    if left.unsafe_load(i) == -1:
        _ = Atomic.fetch_add(sizes.unsafe_offset(32 + g), Int32(1))


def forest_owner_tree_map_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], starts: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin], local_trees_in: Int32,
    tmap: MutPointer[Int32, MutAnyOrigin], tsize: MutPointer[Int32, MutAnyOrigin],
):
    """Local tree t -> global tree, and its node count (scanned into the
    owner's offsets afterwards). Groves an owner does not hold have count 0."""
    var t = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if t >= Int(local_trees_in):
        return
    var tree = 0
    for g in range(32):
        var first = Int(starts.unsafe_load(g))
        var count = Int(counts.unsafe_load(g))
        if count > 0 and t >= first and t < first + count:
            tree = g + 32 * (t - first)
    tmap.unsafe_store(t, Int32(tree))
    tsize.unsafe_store(t, offsets.unsafe_load(tree + 1) - offsets.unsafe_load(tree))


def forest_owner_gather_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], tmap: MutPointer[Int32, MutAnyOrigin],
    loff: MutPointer[Int32, MutAnyOrigin], local_trees_in: Int32, local_nodes_in: Int32,
    outputs_in: Int32, rank_in: Int32,
    out_col: MutPointer[Int32, MutAnyOrigin], out_thr: MutPointer[Float32, MutAnyOrigin],
    out_left: MutPointer[Int32, MutAnyOrigin], out_leaf: MutPointer[Float32, MutAnyOrigin],
):
    """One thread per owner-local node: copy its fields and leaf vector from
    the global model (bytes only, no arithmetic on values)."""
    var j = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if j >= Int(local_nodes_in):
        return
    var outputs = Int(outputs_in)
    var t = _forest_tree_of(loff, Int(local_trees_in), j)
    var tree = Int(tmap.unsafe_load(t))
    var node = Int(offsets.unsafe_load(tree)) + j - Int(loff.unsafe_load(t))
    var leaf_base = node * outputs
    comptime if is_defined["MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE"]():
        # Check-only arm: later owners read their leaf values one node
        # early. Buffer sizes and the node topology (`left`) are unchanged,
        # so the packed leaf count cannot move either. INERT AT ONE DEVICE:
        # grove 0 always lands on owner 0, so a rank above 0 owns only
        # trees above 0 and `node > 0` holds; the test states that.
        if Int(rank_in) > 0 and node > 0:
            leaf_base = (node - 1) * outputs
    out_col.unsafe_store(j, columns.unsafe_load(node))
    out_thr.unsafe_store(j, thresholds.unsafe_load(node))
    out_left.unsafe_store(j, left.unsafe_load(node))
    for c in range(outputs):
        out_leaf.unsafe_store(j * outputs + c, leaves.unsafe_load(leaf_base + c))


def forest_merge_owner_groves_kernel(
    totals: MutPointer[Float32, MutAnyOrigin], local: MutPointer[Float32, MutAnyOrigin],
    mask_in: UInt32, cells_in: Int32,
):
    """Cross-device merge of one owner's grove totals into the root's
    item x 32 table: a cell is taken when its grove is the owner's (a byte
    copy; every grove has exactly one owner, so no sum and no order)."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(cells_in):
        return
    if ((mask_in >> UInt32(i % 32)) & 1) != 0:
        totals.unsafe_store(i, local.unsafe_load(i))


struct _ForestFlat(Movable):
    """The whole flat model uploaded to one owner's device, for the grove
    gather (dropped once the owner's compact copy exists)."""
    var off: DeviceBuffer[DType.int32]
    var col: DeviceBuffer[DType.int32]
    var thr: DeviceBuffer[DType.float32]
    var left: DeviceBuffer[DType.int32]
    var leaf: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32]) raises:
        self.off = ctx.enqueue_create_buffer[DType.int32](len(offsets))
        self.col = ctx.enqueue_create_buffer[DType.int32](max(len(columns), 1))
        self.thr = ctx.enqueue_create_buffer[DType.float32](max(len(thresholds), 1))
        self.left = ctx.enqueue_create_buffer[DType.int32](max(len(left), 1))
        self.leaf = ctx.enqueue_create_buffer[DType.float32](max(len(leaves), 1))
        ctx.enqueue_copy(dst_buf=self.off, src_ptr=offsets.unsafe_ptr())
        if len(columns) > 0:
            ctx.enqueue_copy(dst_buf=self.col, src_ptr=columns.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=self.thr, src_ptr=thresholds.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=self.left, src_ptr=left.unsafe_ptr())
        if len(leaves) > 0:
            ctx.enqueue_copy(dst_buf=self.leaf, src_ptr=leaves.unsafe_ptr())
        # the host lists are read by the time the constructor returns
        ctx.synchronize()
        _ = len(offsets)
        _ = len(columns)
        _ = len(thresholds)
        _ = len(left)
        _ = len(leaves)


struct _GrovePlan(Movable):
    """One owner's grove plan: which of the 32 groves it holds, where each
    starts in its local tree list and how many trees it has (closed form:
    grove g holds trees g, g+32, ..., so (trees - g + 31) // 32 of them)."""
    var starts: List[Int32]
    var counts: List[Int32]
    var groves: List[Int]
    var mask: UInt32
    var trees: Int

    def __init__(out self, rank: Int, assignments: List[Int], trees: Int):
        self.starts = List[Int32](length=32, fill=Int32(0))
        self.counts = List[Int32](length=32, fill=Int32(0))
        self.groves = List[Int]()
        self.mask = UInt32(0)
        self.trees = 0
        for grove in range(min(32, trees)):  # small-loop(32: groves): one plan entry per grove, closed-form tree counts, no model data
            if assignments[grove] != rank:
                continue
            var count = (trees - grove + 31) // 32
            self.groves.append(grove)
            self.starts[grove] = Int32(self.trees)
            self.counts[grove] = Int32(count)
            self.mask |= UInt32(1) << UInt32(grove)
            self.trees += count


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
    var mask: UInt32
    var trees: Int
    var nodes: Int

    def __init__(out self, rank: Int) raises:
        """An empty owner on device `rank`; `load_groves` fills it."""
        self.ctx = Optional[DeviceContext]()
        self.offsets = Optional[DeviceBuffer[DType.int32]]()
        self.columns = Optional[DeviceBuffer[DType.int32]]()
        self.thresholds = Optional[DeviceBuffer[DType.float32]]()
        self.left = Optional[DeviceBuffer[DType.int32]]()
        self.leaves = Optional[DeviceBuffer[DType.float32]]()
        self.starts = Optional[DeviceBuffer[DType.int32]]()
        self.counts = Optional[DeviceBuffer[DType.int32]]()
        self.groves = List[Int]()
        self.mask = UInt32(0)
        self.trees = 0
        self.nodes = 0
        self.ctx = DeviceContext(device_id=rank)

    def load_groves(mut self, mut flat: _ForestFlat, plan: _GrovePlan, rank: Int,
        outputs: Int, features: Int) raises:
        """Gather the planned groves out of the full model resident on this
        owner's device (`flat`) into compact buffers, then validate (and,
        packed, pack) them there (`forest_pack_resident`). No host loop over
        trees or nodes; one word, the owner's node count, comes back."""
        self.groves = plan.groves.copy()
        self.mask = plan.mask
        self.trees = plan.trees
        var t = plan.trees
        var tmap = self.ctx.value().enqueue_create_buffer[DType.int32](max(t, 1))
        var loff = self.ctx.value().enqueue_create_buffer[DType.int32](t + 1)
        try:
            self.starts = self.ctx.value().enqueue_create_buffer[DType.int32](32)
            self.counts = self.ctx.value().enqueue_create_buffer[DType.int32](32)
            self.ctx.value().enqueue_copy(dst_buf=self.starts.value(), src_ptr=plan.starts.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.counts.value(), src_ptr=plan.counts.unsafe_ptr())
            if t > 0:
                self.ctx.value().enqueue_function[forest_owner_tree_map_kernel](
                    flat.off.unsafe_ptr(), self.starts.value().unsafe_ptr(),
                    self.counts.value().unsafe_ptr(), Int32(t),
                    tmap.unsafe_ptr(), loff.unsafe_ptr(),
                    grid_dim=(t + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB,
                    block_dim=FOREST_FINITE_TPB,
                )
            device_exclusive_scan_total(self.ctx.value(), loff, t)
            var h = self.ctx.value().enqueue_create_host_buffer[DType.int32](1)
            var tail = loff.create_sub_buffer[DType.int32](t, 1)
            self.ctx.value().enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=tail)
            self.ctx.value().synchronize()
            var n = Int(h.unsafe_ptr().unsafe_load(0))
            _ = tail^
            _ = h^
            self.nodes = n
            var gcol = self.ctx.value().enqueue_create_buffer[DType.int32](max(n, 1))
            var gthr = self.ctx.value().enqueue_create_buffer[DType.float32](max(n, 1))
            var gleft = self.ctx.value().enqueue_create_buffer[DType.int32](max(n, 1))
            var gleaf = self.ctx.value().enqueue_create_buffer[DType.float32](max(n * outputs, 1))
            if n > 0:
                self.ctx.value().enqueue_function[forest_owner_gather_kernel](
                    flat.off.unsafe_ptr(), flat.col.unsafe_ptr(), flat.thr.unsafe_ptr(),
                    flat.left.unsafe_ptr(), flat.leaf.unsafe_ptr(), tmap.unsafe_ptr(),
                    loff.unsafe_ptr(), Int32(t), Int32(n), Int32(outputs), Int32(rank),
                    gcol.unsafe_ptr(), gthr.unsafe_ptr(), gleft.unsafe_ptr(), gleaf.unsafe_ptr(),
                    grid_dim=(n + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB,
                    block_dim=FOREST_FINITE_TPB,
                )
            comptime if FOREST_PACKED_NODES:
                forest_pack_resident(
                    self.ctx.value(), loff, gcol, gthr, gleft, gleaf, t, n,
                    features, outputs, self.columns, self.leaves,
                )
                self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](1)
                self.left = self.ctx.value().enqueue_create_buffer[DType.int32](1)
                self.ctx.value().synchronize()
                _ = gcol^
                _ = gthr^
                _ = gleft^
                _ = gleaf^
            else:
                # each owner re-checks its own part on its device
                forest_validate_device(
                    self.ctx.value(), loff, gcol, gthr, gleft, gleaf, t, n, features, outputs,
                )
                self.columns = gcol^
                self.thresholds = gthr^
                self.left = gleft^
                self.leaves = gleaf^
            self.ctx.value().synchronize()
        except e:
            self.ctx.value().synchronize()
            _ = tmap^
            _ = loff^
            raise e
        self.offsets = loff^
        _ = tmap^

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
        features: Int, outputs: Int, scan_input: Bool = False,
    ) raises -> DeviceBuffer[DType.float32]:
        """This owner's item x 32 grove totals, resident on its own device
        (groves it does not hold are +0.0; the root merges by grove mask)."""
        var first_row = first_item // outputs
        var rows = (first_item + items + outputs - 1) // outputs - first_row
        var relative_item = first_item - first_row * outputs
        ref ctx = self.ctx.value()
        var dx = ctx.enqueue_create_buffer[DType.float32](rows * features)
        var dt = ctx.enqueue_create_buffer[DType.float32](items * 32)
        try:
            ctx.enqueue_copy(dst_buf=dx, src_ptr=x.unsafe_offset(first_row * features))
            if scan_input and not device_ptr_all_finite(
                    ctx, dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows * features):
                raise Error("resident forest requires finite Float32 values")
            ctx.enqueue_function[forest_owned_groves_kernel[RF_INPUT, FOREST_PACKED_NODES]](
                self.offsets.value().unsafe_ptr(), self.columns.value().unsafe_ptr(),
                self.thresholds.value().unsafe_ptr(), self.left.value().unsafe_ptr(),
                self.leaves.value().unsafe_ptr(), self.starts.value().unsafe_ptr(),
                self.counts.value().unsafe_ptr(), dx.unsafe_ptr(), dt.unsafe_ptr(),
                Int32(items), Int32(relative_item), Int32(features), Int32(outputs),
                grid_dim=(items + 3) // 4, block_dim=128)
            ctx.synchronize()
        except e:
            ctx.synchronize()
            _ = dx^
            _ = dt^
            raise e
        _ = dx^
        return dt^


struct PooledForest(Movable):
    var owners: List[ForestGroveOwner]
    var features: Int
    var outputs: Int
    var trees: Int

    def __init__(out self, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        features: Int, outputs: Int, requested: Int) raises:
        """Grove partition and per-owner gather on the devices (lane
        cpu4-misc). Owner 0 receives the whole model, validates it there and
        counts each grove's nodes and leaves with one atomic kernel; the host
        reads back those 64 words and balances the 32 groves over the owners
        with the original greedy rule and size formula. Each owner then
        gathers its groves on its own device."""
        self.owners = List[ForestGroveOwner]()
        self.features = features
        self.outputs = outputs
        self.trees = len(offsets) - 1
        var nodes = len(columns)
        var groves = min(32, self.trees)
        var active = min(requested, groves)
        for rank in range(active):  # small-loop(active: owner devices, at most 32): one context per owner device
            self.owners.append(ForestGroveOwner(rank))
        var flat0 = _ForestFlat(self.owners[0].ctx.value(), offsets, columns, thresholds, left, leaves)
        var dsz = self.owners[0].ctx.value().enqueue_create_buffer[DType.int32](64)
        var hsz = self.owners[0].ctx.value().enqueue_create_host_buffer[DType.int32](64)
        try:
            # the whole model is validated on the device before it is
            # partitioned by grove (each owner re-checks its own part)
            forest_validate_device(
                self.owners[0].ctx.value(), flat0.off, flat0.col, flat0.thr, flat0.left,
                flat0.leaf, self.trees, nodes, features, outputs,
            )
            self.owners[0].ctx.value().enqueue_memset(dsz, Int32(0))
            if nodes > 0:
                self.owners[0].ctx.value().enqueue_function[forest_grove_sizes_kernel](
                    flat0.off.unsafe_ptr(), flat0.left.unsafe_ptr(), Int32(self.trees),
                    Int32(nodes), dsz.unsafe_ptr(),
                    grid_dim=(nodes + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB,
                    block_dim=FOREST_FINITE_TPB,
                )
            self.owners[0].ctx.value().enqueue_copy(dst_ptr=hsz.unsafe_ptr(), src_buf=dsz)
            self.owners[0].ctx.value().synchronize()
        except e:
            self.owners[0].ctx.value().synchronize()
            _ = dsz^
            _ = hsz^
            _ = flat0^
            raise e
        var assignments = List[Int](length=32, fill=-1)
        var loads = List[Int](length=active, fill=0)
        for grove in range(groves):  # small-loop(groves: at most 32 groves): greedy owner choice from the 64 per-grove counts read back
            var owner = 0
            for rank in range(1, active):  # small-loop(active: owner devices, at most 32): the least-loaded owner
                if loads[rank] < loads[owner]:
                    owner = rank
            assignments[grove] = owner
            var gtrees = (self.trees - grove + 31) // 32
            var gnodes = Int(hsz.unsafe_ptr().unsafe_load(grove))
            # each tree: its nodes' words plus one offset word
            var size = gnodes * (12 + 4 * outputs) + 4 * gtrees
            comptime if FOREST_PACKED_NODES:
                var gleaves = Int(hsz.unsafe_ptr().unsafe_load(32 + grove))
                size = gnodes * 16 + 4 * gtrees + 4 * outputs * gleaves
            loads[owner] += size
        _ = dsz^
        _ = hsz^
        var plan0 = _GrovePlan(0, assignments, self.trees)
        self.owners[0].load_groves(flat0, plan0, 0, outputs, features)
        self.owners[0].ctx.value().synchronize()
        _ = flat0^
        for rank in range(1, active):  # small-loop(active: owner devices, at most 32): one upload and one gather per owner device
            var plan = _GrovePlan(rank, assignments, self.trees)
            var flat = _ForestFlat(self.owners[rank].ctx.value(), offsets, columns, thresholds, left, leaves)
            self.owners[rank].load_groves(flat, plan, rank, outputs, features)
            self.owners[rank].ctx.value().synchronize()
            _ = flat^

    def collect[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        rows: Int,
    ) raises -> List[Float32]:
        return self.collect_items[RF_INPUT](x, 0, rows * self.outputs)

    def collect_items[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        first_item: Int, items: Int,
    ) raises -> List[Float32]:
        """The unaveraged item x 32 grove totals as a host list (for the
        pool checks; predict keeps them resident)."""
        var dt = self.collect_items_resident[RF_INPUT](x, first_item, items)
        var totals = List[Float32](length=items * 32, fill=Float32(0.0))
        ref ctx = self.owners[0].ctx.value()
        try:
            ctx.enqueue_copy(dst_ptr=totals.unsafe_ptr(), src_buf=dt)
            ctx.synchronize()
        except e:
            ctx.synchronize()
            _ = dt^
            raise e
        _ = dt^
        return totals^

    def collect_items_resident[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        first_item: Int, items: Int,
    ) raises -> DeviceBuffer[DType.float32]:
        """The item x 32 grove totals on owner 0's device: each owner's
        totals cross to it (`transfer_bytes`) and a mask kernel takes that
        owner's grove columns (the host merge loop is gone, lane cpu4-misc)."""
        var cells = items * 32
        var totals = self.owners[0].ctx.value().enqueue_create_buffer[DType.float32](cells)
        var grid = (cells + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB
        try:
            self.owners[0].ctx.value().enqueue_memset(totals, Float32(0.0))
            for rank in range(len(self.owners)):  # small-loop(owners: devices, at most 32): one contribution, transfer and merge launch per owner device
                var local = self.owners[rank].contribution[RF_INPUT](
                    x, first_item, items, self.features, self.outputs, rank == 0)
                var mask = self.owners[rank].mask
                if rank == 0:
                    self.owners[0].ctx.value().enqueue_function[forest_merge_owner_groves_kernel](
                        totals.unsafe_ptr(), local.unsafe_ptr(), mask, Int32(cells),
                        grid_dim=grid, block_dim=FOREST_FINITE_TPB)
                    self.owners[0].ctx.value().synchronize()
                else:
                    var staged = self.owners[0].ctx.value().enqueue_create_buffer[DType.float32](cells)
                    self.owners[0].ctx.value().synchronize()
                    transfer_bytes(self.owners[rank].ctx.value(), self.owners[0].ctx.value(),
                        local, staged, cells, True)
                    self.owners[0].ctx.value().enqueue_function[forest_merge_owner_groves_kernel](
                        totals.unsafe_ptr(), staged.unsafe_ptr(), mask, Int32(cells),
                        grid_dim=grid, block_dim=FOREST_FINITE_TPB)
                    self.owners[0].ctx.value().synchronize()
                    _ = staged^
                _ = local^
                comptime if is_defined["MOJOLEARN_FOREST_POOL_FAULT"]():
                    if Int(getenv("MOJOLEARN_FOREST_FAIL_OWNER", "-1")) == rank:
                        raise Error("injected forest owner failure after contribution")
        except e:
            self.owners[0].ctx.value().synchronize()
            _ = totals^
            raise e
        return totals^

    def _finish_resident[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        rows: Int,
    ) raises -> DeviceBuffer[DType.float32]:
        """Every row's finished output, resident on owner 0's device. The
        owners scan their input rows on the device (`contribution`), their
        grove totals merge on owner 0's device, and the finished outputs are
        scanned where they land: no host scan, no host merge and no host
        staging list (cpu-gpu-cleanup w2-core-scope, lane cpu4-misc)."""
        var dout = self.owners[0].ctx.value().enqueue_create_buffer[DType.float32](rows * self.outputs)
        # Bound both rows and output cells: even wide vector leaves never
        # create more than 4096*32 canonical reduction cells on any device.
        var tile = min(4096, 64 * self.outputs)
        for first in range(0, rows * self.outputs, tile):
            var items = min(tile, rows * self.outputs - first)
            var dt = self.collect_items_resident[RF_INPUT](x, first, items)
            ref ctx = self.owners[0].ctx.value()
            try:
                ctx.enqueue_function[forest_finish_groves_kernel](dt.unsafe_ptr(),
                    dout.unsafe_ptr() + first, Int32(items), Int32(self.trees),
                    grid_dim=(items + 3) // 4, block_dim=128)
                ctx.synchronize()
            except e:
                ctx.synchronize()
                _ = dt^
                _ = dout^
                raise e
            _ = dt^
        if not device_all_finite(self.owners[0].ctx.value(), dout, rows * self.outputs):
            _ = dout^
            raise Error("forest inference prototype requires finite Float32 values")
        return dout^

    def predict_into[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        output: MutPointer[Float32, MutAnyOrigin], rows: Int,
    ) raises:
        if rows <= 0:
            return
        var dout = self._finish_resident[RF_INPUT](x, rows)
        ref ctx = self.owners[0].ctx.value()
        try:
            ctx.enqueue_copy(dst_ptr=output, src_buf=dout)
            ctx.synchronize()
        except e:
            ctx.synchronize()
            _ = dout^
            raise e
        _ = dout^

    def predict_labels_into[RF_INPUT: Bool](mut self, x: MutPointer[Float32, MutAnyOrigin],
        output: MutPointer[Int32, MutAnyOrigin], rows: Int,
    ) raises:
        """Class codes: the row argmax runs on owner 0's device over the
        resident votes (the host argmax loop is gone)."""
        if rows <= 0:
            return
        var dout = self._finish_resident[RF_INPUT](x, rows)
        ref ctx = self.owners[0].ctx.value()
        var dlab = ctx.enqueue_create_buffer[DType.int32](rows)
        try:
            launch_forest_argmax(ctx, dout, dlab, rows, self.outputs)
            ctx.enqueue_copy(dst_ptr=output, src_buf=dlab)
            ctx.synchronize()
        except e:
            ctx.synchronize()
            _ = dout^
            _ = dlab^
            raise e
        _ = dout^
        _ = dlab^
