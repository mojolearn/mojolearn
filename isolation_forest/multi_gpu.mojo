# SPDX-License-Identifier: Apache-2.0
"""Multi-device IsolationForest fit: whole-tree shards over a `DevicePlan`.

The single-device fit (`isolation_forest/impl/isolation_forest.mojo`,
`IsolationForest.fit`) holds no device-count branch. `fit_dispatch` takes the
plan the binding entry read once; with `plan.single()` it calls that fit
unchanged (same arguments, same order, so the same bits). Otherwise each
owner builds a contiguous range of whole trees with the unchanged
single-device fit on its own device, seeded at the range's first global tree
id, and the root model keeps the owners' buffers (`IsolationForestModel.
shards`). No floating-point value from one owner is combined with another's
at fit time; scoring carries the path-length accumulator through global tree
order (`_pooled_path_lengths` in the impl file).
"""
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import DevicePlan
from core.host_parallel import host_parallelize
from core.identity_trace import IdentityTrace
from std.sys import is_defined
from isolation_forest.impl.isolation_forest import (
    IF_C50_PACKED,
    IFLaunchKnobs,
    IFModelShard,
    IF_params,
    IsolationForest,
    IsolationForestModel,
)


def fit_dispatch(
    plan: DevicePlan,
    forest: IsolationForest,
    ctx: DeviceContext,
    input_colmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    mut model: IsolationForestModel,
    mut trace: IdentityTrace,
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
    src_addr: Int = 0,
    global_tree_start: Int = 0,
    colmajor_addr: Int = 0,
) raises:
    """`forest.fit(...)` on one device, or whole-tree shards over `plan`.

    One device (`plan.single()`, the old `setting == "" or setting == "1"`
    test) runs `IsolationForest.fit` unchanged. Otherwise the root model gets
    the same prologue (checks, sizes, scalar fields) as a one-device fit, then
    the trees are built by `_fit_tree_shards`. The refusals and their order
    are the ones the old in-file branch had.
    """
    if plan.single():
        forest.fit(ctx, input_colmajor, n_rows, n_cols, model, trace, knobs,
                   src_addr, global_tree_start, colmajor_addr)
        return
    var shape = forest.fit_prologue(
        input_colmajor, n_rows, n_cols, model, src_addr, colmajor_addr
    )
    _ = shape^
    var count = Int(plan.setting)
    if count < 1 or count > 64:
        raise Error("parallel IsolationForest requires 1..64 devices")
    if trace.enabled:
        raise Error("parallel IsolationForest does not export diagnostic traces")
    comptime if IF_C50_PACKED:
        raise Error("C50 IF packed model requires one device; pooled packed models remain pending")
    _fit_tree_shards(plan, ctx, input_colmajor, src_addr, n_rows, n_cols,
                     forest.params, model, knobs, count)


def fit_params_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    mut forest: IsolationForestModel,
    input_colmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    params: IF_params,
    mut trace: IdentityTrace,
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
    src_addr: Int = 0,
) raises:
    """The impl file's module-level `fit(ctx, forest, ...)` routed through
    `fit_dispatch`: build `IsolationForest(params)` and fit `forest`."""
    var if_model = IsolationForest(params)
    fit_dispatch(plan, if_model, ctx, input_colmajor, n_rows, n_cols, forest,
                 trace, knobs, src_addr)


def _fit_tree_shards(plan: DevicePlan, ctx: DeviceContext,
    input_colmajor: List[Float32], src_addr: Int, rows: Int, columns: Int,
    params: IF_params, mut model: IsolationForestModel, knobs: IFLaunchKnobs,
    count: Int,
) raises:
    """Build canonical whole-tree ranges and retain each owner's model buffers.

    Training data remains replicated. No complete forest is allocated on the
    root; only successful owners are published after every worker joins.

    THE NEGATIVE CONTROL is `-D MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE=1`
    below. Unlike the other `par-*` arms it perturbs an RNG STREAM rather than a
    row offset, because `first` is a curand subsequence base and nothing else:
    every buffer is indexed by the LOCAL tree id. The shard's trees are then
    entirely different rather than slightly shifted, so expect a large delta.
    Owed a two-device column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding
    restates this driver, so no CPU column can watch it fire.
    """
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel IsolationForest requires IDENTICAL")
    var active = min(count, params.n_estimators)
    ctx.synchronize()
    var shards = List[IFModelShard]()
    for rank in range(active):
        var first = params.n_estimators * rank // active
        var width = params.n_estimators * (rank + 1) // active - first
        var device = DeviceContext(device_id=plan.device(rank))
        var feature_indices = device.enqueue_create_buffer[DType.int32](1)
        var feature = device.enqueue_create_buffer[DType.int32](1)
        var threshold = device.enqueue_create_buffer[DType.float32](1)
        var left = device.enqueue_create_buffer[DType.int32](1)
        var right = device.enqueue_create_buffer[DType.int32](1)
        var offsets = device.enqueue_create_buffer[DType.int32](1)
        var nodes = device.enqueue_create_buffer[DType.int32](1)
        var depths = device.enqueue_create_buffer[DType.int32](1)
        shards.append(IFModelShard(device^, feature_indices^, feature^,
            threshold^, left^, right^, offsets^, nodes^, depths^, first, width,
            List[Int32](), List[Int32]()))
    var failures = List[Int](length=active, fill=0)
    var sp = rebind[MutPointer[IFModelShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    var xp = rebind[MutPointer[Float32, MutUntrackedOrigin]](input_colmajor.unsafe_ptr())
    var size = len(input_colmajor)
    var config = params.copy()
    def task(rank: Int) {imm sp, imm fp, imm xp, imm size, imm src_addr,
                         imm rows, imm columns, imm knobs, imm config}:
        try:
            ref shard = sp[rank]
            # cpu4-forest: the owner uploads the shared column-major block
            # from its address (no per-owner host copy)
            var data = List[Float32]()
            var cm_addr = 0
            if src_addr == 0:
                cm_addr = Int(xp)
            var local_config = config.copy()
            local_config.n_estimators = shard.count
            var forest = IsolationForest(local_config)
            var local = IsolationForestModel(shard.ctx)
            var trace = IdentityTrace.disabled()
            var tree_base = shard.first
            comptime if is_defined["MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE"]():
                # Check-only arm: later owners seed their trees one RNG stream
                # early. `shard.first` is the curand subsequence base and has
                # exactly one consumer -- this call -- so no allocation, no
                # length and no validation can see it; `shard.count` still
                # sizes every buffer and every published tree. INERT AT ONE
                # DEVICE (`active` is 1, only rank 0 runs), which is what makes
                # a moved `par-iforest` cell attributable to the define rather
                # than to the second device.
                if rank > 0:
                    tree_base = shard.first - 1
            forest.fit(shard.ctx, data, rows, columns, local, trace,
                       knobs, src_addr, tree_base, cm_addr)
            swap(shard.global_feature_indices, local.global_feature_indices)
            swap(shard.node_feature, local.node_feature)
            swap(shard.node_threshold, local.node_threshold)
            swap(shard.node_left, local.node_left)
            swap(shard.node_right, local.node_right)
            swap(shard.global_tree_offsets, local.global_tree_offsets)
            swap(shard.global_tree_n_nodes, local.global_tree_n_nodes)
            swap(shard.global_tree_max_depth, local.global_tree_max_depth)
            swap(shard.tree_n_nodes_host, local.tree_n_nodes_host)
            swap(shard.tree_max_depth_host, local.tree_max_depth_host)
            _ = local^
            shard.ctx.synchronize()
        except:
            fp[rank] = 1
    if active == 1:
        task(0)
    else:
        host_parallelize(task, active)
    for rank in range(active):  # small-loop(active: tree-shard owners): one entry per device owner, per-tree metadata words
        if failures[rank] != 0:
            raise Error("IsolationForest tree shard failed: " + String(rank))
    var node_counts = List[Int32]()
    var max_depths = List[Int32]()
    for rank in range(active):  # small-loop(active: tree-shard owners): one entry per device owner, per-tree metadata words
        for tree in range(shards[rank].count):  # small-loop(shards[rank].count: trees of one owner): two metadata words per tree, no rows
            node_counts.append(shards[rank].tree_n_nodes_host[tree])
            max_depths.append(shards[rank].tree_max_depth_host[tree])
    model.tree_n_nodes_host = node_counts^
    model.tree_max_depth_host = max_depths^
    # A model previously fitted on one device must also release its old full
    # root arrays. Allocate every placeholder before replacing any storage.
    var empty_root = IsolationForestModel(ctx)
    swap(model.global_feature_indices, empty_root.global_feature_indices)
    swap(model.node_feature, empty_root.node_feature)
    swap(model.node_threshold, empty_root.node_threshold)
    swap(model.node_left, empty_root.node_left)
    swap(model.node_right, empty_root.node_right)
    swap(model.global_tree_offsets, empty_root.global_tree_offsets)
    swap(model.global_tree_n_nodes, empty_root.global_tree_n_nodes)
    swap(model.global_tree_max_depth, empty_root.global_tree_max_depth)
    model.shards = shards^
    _ = empty_root^
    model.fitted = True
