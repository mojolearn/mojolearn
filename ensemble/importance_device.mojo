# SPDX-License-Identifier: Apache-2.0
"""T15 requested RF impurity importance from live finished-tree statistics.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
The fold is the incumbent: ascending nodes within each tree, ascending features
for each normalization, ascending trees, then ascending features for the forest
normalization. Integer soft binary64 preserves explicit multiply/add/divide
seams on all devices. No floating atomic or completion-order fold is used.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from ensemble.flatnode import SparseTreeNode
from checks.soft_f64 import (SF64_INF, SF64_ONE, sf64_add, sf64_div,
    sf64_from_f32, sf64_from_int, sf64_gt, sf64_mul, sf64_to_f32)


def importance_tree_kernel[dtype: DType](
    nodes: MutPointer[SparseTreeNode[dtype],MutAnyOrigin],
    finite: MutPointer[UInt64,MutAnyOrigin],
    infinite: MutPointer[UInt64,MutAnyOrigin],
    n_nodes: Int32, n_features: Int32, tree_id: Int32,
):
    # One invocation per completed tree. Serial node order is the incumbent
    # numerical graph; concurrent trees write disjoint feature vectors.
    var nf = Int(n_features)
    var base = Int(tree_id)*nf
    for c in range(nf):
        finite[unsafe_offset=base+c] = UInt64(0)
        infinite[unsafe_offset=base+c] = UInt64(0)
    if n_nodes <= 0 or nodes[unsafe_offset=0].InstanceCount() <= 0:
        return
    var has_infinite = False
    for j in range(Int(n_nodes)):
        ref nd = nodes[unsafe_offset=j]
        if nd.IsLeaf():
            continue
        var c = Int(nd.ColumnId())
        var contribution = sf64_mul(sf64_from_f32(nd.BestMetric().cast[DType.float32]()),sf64_from_int(Int(nd.InstanceCount())))
        if contribution == SF64_INF:
            infinite[unsafe_offset=base+c] = sf64_add(infinite[unsafe_offset=base+c],SF64_ONE)
            has_infinite = True
        elif (contribution & SF64_INF) != SF64_INF and sf64_gt(contribution,UInt64(0)):
            finite[unsafe_offset=base+c] = sf64_add(finite[unsafe_offset=base+c],contribution)
    var total = UInt64(0)
    for c in range(nf):
        var value = infinite[unsafe_offset=base+c] if has_infinite else finite[unsafe_offset=base+c]
        total = sf64_add(total,value)
    for c in range(nf):
        var value = infinite[unsafe_offset=base+c] if has_infinite else finite[unsafe_offset=base+c]
        finite[unsafe_offset=base+c] = sf64_div(value,total) if sf64_gt(total,UInt64(0)) else UInt64(0)


def importance_finish_kernel(
    trees: MutPointer[UInt64,MutAnyOrigin],
    output: MutPointer[Float32,MutAnyOrigin],
    accumulated: MutPointer[UInt64,MutAnyOrigin],
    n_trees: Int32, n_features: Int32,
):
    var nf = Int(n_features)
    var total = UInt64(0)
    for c in range(nf):
        var value = UInt64(0)
        for t in range(Int(n_trees)):
            value = sf64_add(value,trees[unsafe_offset=t*nf+c])
        accumulated[unsafe_offset=c] = value
        total = sf64_add(total,value)
    for c in range(nf):
        output[unsafe_offset=c] = sf64_to_f32(sf64_div(accumulated[unsafe_offset=c],total)) if sf64_gt(total,UInt64(0)) else Float32(0)


struct ImportanceStore(Movable):
    var enabled: Bool
    var n_trees: Int
    var n_features: Int
    var finite: DeviceBuffer[DType.uint64]
    var infinite: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext, trees: Int, features: Int, enabled: Bool) raises:
        self.enabled = enabled
        self.n_trees = trees
        self.n_features = features
        self.finite = ctx.enqueue_create_buffer[DType.uint64](trees*features if enabled else 1)
        self.infinite = ctx.enqueue_create_buffer[DType.uint64](trees*features if enabled else 1)

    def append[dtype: DType](mut self, ctx: DeviceContext,
        nodes: MutPointer[SparseTreeNode[dtype],MutUntrackedOrigin], count: Int, tree: Int) raises:
        if not self.enabled:
            return
        ctx.enqueue_function[importance_tree_kernel[dtype]](
            nodes.unsafe_origin_cast[MutAnyOrigin](),self.finite.unsafe_ptr(),self.infinite.unsafe_ptr(),
            Int32(count),Int32(self.n_features),Int32(tree),grid_dim=1,block_dim=1)

    def finish(mut self, ctx: DeviceContext) raises -> List[Float32]:
        var output = ctx.enqueue_create_buffer[DType.float32](self.n_features)
        var accumulated = ctx.enqueue_create_buffer[DType.uint64](self.n_features)
        var host = ctx.enqueue_create_host_buffer[DType.float32](self.n_features)
        ctx.enqueue_function[importance_finish_kernel](self.finite.unsafe_ptr(),output.unsafe_ptr(),
            accumulated.unsafe_ptr(),Int32(self.n_trees),Int32(self.n_features),grid_dim=1,block_dim=1)
        ctx.enqueue_copy(dst_buf=host,src_buf=output)
        ctx.synchronize()
        var result = List[Float32](capacity=self.n_features)
        for c in range(self.n_features):
            result.append(host.unsafe_ptr()[c])
        _ = accumulated^
        _ = output^
        _ = host^
        return result^
