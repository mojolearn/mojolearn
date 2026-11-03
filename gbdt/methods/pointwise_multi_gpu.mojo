# SPDX-License-Identifier: Apache-2.0
"""Whole packed feature groups; original pointwise document and sibling folds.

THE NEGATIVE CONTROL. `-D MOJOLEARN_GBDT_PARALLEL_SABOTAGE=1` makes every owner
above rank 0 read its feature descriptors one packed GROUP early. `width`,
`bin_first`, `bin_count` and the bin-range copy-back are all still derived from
the true `begin`, so every allocation and every packed-word lane position keeps
its size and only the features a shard histograms change. `count` is
`min(requested, groups)`, so `begin >= group` whenever `rank >= 1` and the
shifted offset cannot go negative. It is a `comptime if`, so no production bit
can move, and it is INERT AT ONE DEVICE, which is what makes a moved
`par-boosting-pointwise` cell attributable to the define rather than to the
second device. The same define arms the greedy searcher's feature-histogram
shards (`gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`), which
is the partition the other `gbdt_fit` lanes take. Owed a two-device column
(`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.

THE GBDT DISPATCHERS (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md). Every
multi-device decision of the GBDT family lives here and takes a `DevicePlan`
value that the binding entry read once; nothing below the entry reads the
thread plan. `gbdt_hist_dispatch` (greedy searcher) and
`pointwise_hist_dispatch` (pointwise scores calcer) call the unchanged
one-device launch when `plan.single()` (setting "" or "1", the exact test the
old in-file branches made), so single-device bits cannot move; otherwise they
run the feature-group shard plans below.
"""
from core.device_plan import DevicePlan
from max.gpu.host import DeviceContext, DeviceBuffer
from core.host_parallel import host_parallelize
from core.device_zero import enqueue_fill
from std.os import getenv
from std.sys.compile import is_defined
from core.multi_gpu import peer_clone, copy_columns_kernel, copy_scalar_kernel
from checks.numerics import NUMERIC_IDENTICAL
from gbdt.methods.pointwise_kernels import compute_hist2, compute_hist2_dev, FoldsHistogram, HIST_BUILD_MODE, PW_PRIVATE_DOC_SLOTS
from gbdt.gpu_data.grid_policy import POLICY_BINARY, POLICY_HALF_BYTE
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    BUILD_MODE as GREEDY_HIST_BUILD_MODE,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import HIST2_SMEM_MODE
# Module cycle on purpose: greedy_search_helper imports `gbdt_hist_dispatch`
# from here, and the dispatcher calls back into its one-device launcher.
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    DeviceBlock,
    OneByteWidthPlan,
    feature_groups_for,
    launch_histograms_for_blocks,
    replication_for,
)


@fieldwise_init
struct PointwiseShard(Movable):
    var ctx: DeviceContext
    var offset: DeviceBuffer[DType.uint32]
    var first: DeviceBuffer[DType.uint32]
    var folds: DeviceBuffer[DType.uint32]
    var one_hot: DeviceBuffer[DType.uint8]
    var cindex: DeviceBuffer[DType.uint32]
    var target: DeviceBuffer[DType.float32]
    var weight: DeviceBuffer[DType.float32]
    var docs: DeviceBuffer[DType.uint32]
    var parts: DeviceBuffer[DType.uint32]
    var hist: DeviceBuffer[DType.float32]
    var features: Int
    var bin_first: Int
    var bin_count: Int

    def __deinit__(deinit self):
        _ = self.hist^
        _ = self.parts^
        _ = self.docs^
        _ = self.weight^
        _ = self.target^
        _ = self.cindex^
        _ = self.one_hot^
        _ = self.folds^
        _ = self.first^
        _ = self.offset^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def gbdt_plan_shard_device(plan: DevicePlan, rank: Int) raises -> Int:
    """The device a GBDT feature shard of `rank` runs on: the plan's device
    of `rank`, or of rank 0 for every rank under
    `MOJOLEARN_GBDT_SHARD_ONE_DEVICE=1` (the logical-shard diagnostic of
    `core.multi_gpu.gbdt_shard_device_id`, here from the plan value instead
    of the thread plan)."""
    if String(getenv("MOJOLEARN_GBDT_SHARD_ONE_DEVICE")) == "1":
        return plan.device(0)
    return plan.device(rank)


def pointwise_device_count(plan: DevicePlan) raises -> Int:
    # One device: the same "" / "1" test as `plan.single()`. The old read
    # defaulted an unset count to "1", so every one-device fit lands here.
    if plan.single():
        return 1
    var count = Int(plan.setting)
    if count < 1 or count > 64:
        raise Error("pointwise device count must be in 1..64")
    if count > 1:
        if HIST_BUILD_MODE != NUMERIC_IDENTICAL:
            raise Error("parallel pointwise requires IDENTICAL mode")
        comptime if PW_PRIVATE_DOC_SLOTS:
            raise Error("parallel pointwise requires the original whole-document arm")
    return count


def pointwise_feature_shards(ctx: DeviceContext, policy: Int,
    mut offset: DeviceBuffer[DType.uint32],
    mut first: DeviceBuffer[DType.uint32],
    mut folds: DeviceBuffer[DType.uint32],
    mut one_hot: DeviceBuffer[DType.uint8],
    mut cindex: DeviceBuffer[DType.uint32],
    mut target: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut docs: DeviceBuffer[DType.uint32],
    mut parts: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    features: Int, bins: Int, rows: Int, part_count: Int, fold_count: Int,
    full_pass: Bool, folds_hist: FoldsHistogram, sm_count: Int,
    fixed_scale: Float32, requested: Int, plan: DevicePlan,
) raises:
    # Preserve packed-word lane positions by splitting only at 32/8/4 features.
    var group = 32 if policy == 0 else (8 if policy == 1 else 4)
    var groups = (features+group-1)//group
    var count = min(requested, groups)
    ctx.synchronize()
    var hf = ctx.enqueue_create_host_buffer[DType.uint32](features)
    var hn = ctx.enqueue_create_host_buffer[DType.uint32](features)
    ctx.enqueue_copy(dst_ptr=hf.unsafe_ptr(), src_buf=first)
    ctx.enqueue_copy(dst_ptr=hn.unsafe_ptr(), src_buf=folds)
    ctx.synchronize()
    var shards = List[PointwiseShard]()
    for rank in range(count):
        var begin = groups*rank//count*group
        var end = min(features, groups*(rank+1)//count*group)
        var width = end-begin
        var bin_first = Int(hf.unsafe_ptr()[begin])
        var bin_end = Int(hf.unsafe_ptr()[end-1])+Int(hn.unsafe_ptr()[end-1])
        var source = begin
        comptime if is_defined["MOJOLEARN_GBDT_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their feature descriptors one
            # packed group early. Bin ranges and widths keep the true `begin`.
            if rank > 0:
                source = begin - group
        var device = DeviceContext(device_id=gbdt_plan_shard_device(plan, rank))
        var v_offset = offset.create_sub_buffer[DType.uint32](source,width)
        var d_offset = peer_clone(ctx,device,v_offset)
        var v_first = first.create_sub_buffer[DType.uint32](source,width)
        var d_first = peer_clone(ctx,device,v_first)
        var v_folds = folds.create_sub_buffer[DType.uint32](source,width)
        var d_folds = peer_clone(ctx,device,v_folds)
        var v_one_hot = one_hot.create_sub_buffer[DType.uint8](source,width)
        var d_one_hot = peer_clone(ctx,device,v_one_hot)
        var d_cindex = peer_clone(ctx,device,cindex)
        var d_target = peer_clone(ctx,device,target)
        var d_weight = peer_clone(ctx,device,weight)
        var d_docs = peer_clone(ctx,device,docs)
        var d_parts = peer_clone(ctx,device,parts)
        var d_hist = peer_clone(ctx,device,hist)
        shards.append(PointwiseShard(device^, d_offset^, d_first^, d_folds^, d_one_hot^, d_cindex^, d_target^, d_weight^, d_docs^, d_parts^, d_hist^, width, bin_first, bin_end-bin_first))
    var failed = List[Int](length=count, fill=0)
    var sp = rebind[MutPointer[PointwiseShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failed.unsafe_ptr())
    # This copy preserves the original dispatch counts. The admitted arm has
    # multiplier one; changing the group count cannot change a document fold.
    var fh = folds_hist.copy()
    def task(rank: Int) {imm sp, imm fp, imm policy, imm bins, imm rows,
                         imm part_count, imm fold_count, imm full_pass,
                         imm fh, imm sm_count, imm fixed_scale}:
        try:
            ref s = sp[rank]
            compute_hist2(s.ctx,policy,s.offset.unsafe_ptr(),s.first.unsafe_ptr(),
                s.folds.unsafe_ptr(),s.one_hot.unsafe_ptr(),s.features,s.bin_first,s.bin_count,
                s.cindex.unsafe_ptr(),s.target.unsafe_ptr(),s.weight.unsafe_ptr(),
                s.docs.unsafe_ptr(),rows,s.parts.unsafe_ptr(),part_count,fold_count,
                s.hist.unsafe_ptr(),bins,full_pass,fh.copy(),sm_count,fixed_scale)
            s.ctx.synchronize()
        except:
            fp[rank] = 1
    host_parallelize(task,count)
    for rank in range(count):  # small-loop(count: devices, at most eight): per-device failure check that only raises
        if failed[rank] != 0:
            raise Error("pointwise feature shard failed: " + String(rank))
    # Each bin is an interleaved (weight, target) pair, not two bin planes.
    # Copy only owned pairs, including untouched parents needed next depth.
    # No cross-shard floating-point reduction is introduced.
    for rank in range(count):
        ref s = shards[rank]
        for plane in range(len(hist)//(2*bins)):
            var start = 2*(plane*bins+s.bin_first)
            var source = s.hist.create_sub_buffer[DType.float32](start,2*s.bin_count)
            var dest = hist.create_sub_buffer[DType.float32](start,2*s.bin_count)
            source.enqueue_copy_to(dest)
            s.ctx.synchronize()
    _ = shards^
    ctx.synchronize()


def pointwise_is_multi(plan: DevicePlan) raises -> Bool:
    """True when the pointwise histograms run as feature shards. Same
    validation (and the same raise on a bad count) as the old
    `pointwise_device_count() > 1` test at the calcer's submit sites."""
    return pointwise_device_count(plan) > 1


def pointwise_hist_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    policy: Int,
    mut offset: DeviceBuffer[DType.uint32],
    mut first: DeviceBuffer[DType.uint32],
    mut folds: DeviceBuffer[DType.uint32],
    mut one_hot: DeviceBuffer[DType.uint8],
    mut cindex: DeviceBuffer[DType.uint32],
    mut target: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut docs: DeviceBuffer[DType.uint32],
    mut parts: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    features: Int, bins: Int, rows: Int, part_count: Int, fold_count: Int,
    full_pass: Bool, folds_hist: FoldsHistogram, sm_count: Int,
    fixed_scale: Float32,
    scale_word: MutPointer[Float32, MutAnyOrigin],
    dither_ids: Optional[MutPointer[UInt32, MutAnyOrigin]] = None,
    fold_part_sizes: List[Int] = List[Int](),
) raises -> Bool:
    """The pointwise histogram of one policy: feature shards when the plan
    has more than one device (returns True), else the one-device
    `compute_hist2_dev` with exactly the arguments, in the order, that
    `PolicyScoreHelper._submit` passed it before (returns False).

    `fixed_scale` is the host scale the shards take; `scale_word` the device
    word the one-device launch reads. `fold_part_sizes` is one device only.
    """
    var devices = pointwise_device_count(plan)
    if devices > 1:
        pointwise_feature_shards(ctx, policy, offset, first, folds, one_hot,
            cindex, target, weight, docs, parts, hist, features, bins, rows,
            part_count, fold_count, full_pass, folds_hist.copy(), sm_count,
            fixed_scale, devices, plan)
        return True
    # TWO SEPARATE BUFFERS, and that is not a convenience: the kernels
    # take `target` and `weight` on independent origins, and Mojo
    # refuses two views of one buffer at a launch. DEVIATION 97.2.
    var weight_p = weight.unsafe_ptr()
    var target_p = target.unsafe_ptr()
    var parts_p = parts.unsafe_ptr()
    compute_hist2_dev(
        ctx,
        policy,
        offset.unsafe_ptr(),
        first.unsafe_ptr(),
        folds.unsafe_ptr(),
        one_hot.unsafe_ptr(),
        features,
        0,
        bins,
        cindex.unsafe_ptr(),
        # PLANE 1 is the weighted target and PLANE 0 the weight
        # (DEVIATION 97.3); see `PolicyScoreHelper._submit`.
        target_p,
        weight_p,
        docs.unsafe_ptr(),
        rows,
        parts_p,
        part_count,
        fold_count,
        hist.unsafe_ptr(),
        bins,
        full_pass,
        folds_hist.copy(),
        sm_count,
        scale_word,
        dither_ids=dither_ids,
        fold_part_sizes=fold_part_sizes,
    )
    return False


def gbdt_hist_dispatch[
    hist2_smem_mode: Int = HIST2_SMEM_MODE, ridx_stats: Bool = False,
    level_quant: Bool = False, group_width: Bool = False,
](
    plan: DevicePlan,
    ctx: DeviceContext,
    mut blocks: List[DeviceBlock],
    depth: Int,
    n_live: Int,
    n_rows: Int,
    stat_count: Int,
    max_leaves: Int,
    sm_count: Int,
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut dense_ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc_i32: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    skip_bridge: Bool = False,
    qstats: Optional[DeviceBuffer[DType.int32]] = None,
    width_plans: List[OneByteWidthPlan] = List[OneByteWidthPlan](),
    replicas_override: Int = 0,
    nonsymmetric_work: Bool = False,
    symmetric_policy: Bool = False,
) raises:
    """The greedy searcher's histograms under a device plan.

    `plan.single()` (setting "" or "1", the exact test of the branch that
    sat at the head of `launch_histograms_for_blocks`): the one-device
    `launch_histograms_for_blocks` with every argument passed through, so no
    one-device bit can move. Otherwise the feature-group shards, with the
    same checks and the same arguments the old branch gave them (the shard
    path never read `qstats`, `width_plans`, `replicas_override`,
    `nonsymmetric_work` or `symmetric_policy`, and still does not).
    """
    if not plan.single():
        var devices = Int(plan.setting)
        if devices < 1 or devices > 64 or GREEDY_HIST_BUILD_MODE != NUMERIC_IDENTICAL:
            raise Error("GBDT multi-GPU requires IDENTICAL and 1..64 devices")
        if skip_bridge:
            raise Error("GBDT multi-GPU does not expose the per-block scratch probe")
        launch_feature_shards[hist2_smem_mode, ridx_stats](ctx, blocks,
            depth, n_live, n_rows, stat_count, max_leaves, sm_count,
            fixed_scale, cindex, row_index, stats, p_off, p_sz, ids,
            dense_ids, hist, hist_cells_per_leaf, devices, plan)
        return
    launch_histograms_for_blocks[
        hist2_smem_mode, ridx_stats, level_quant, group_width
    ](
        ctx, blocks, depth, n_live, n_rows, stat_count, max_leaves,
        sm_count, fixed_scale, cindex, row_index, stats, p_off, p_sz, ids,
        dense_ids, hist, acc_i32, block_hist, hist_cells_per_leaf,
        skip_bridge=skip_bridge,
        qstats=qstats,
        width_plans=width_plans,
        replicas_override=replicas_override,
        nonsymmetric_work=nonsymmetric_work,
        symmetric_policy=symmetric_policy,
    )


# ---------------------------------------------------------------------------
# The greedy searcher's feature-group shards, moved unchanged from
# gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo.
# ---------------------------------------------------------------------------

@fieldwise_init
struct FeatureHistogramShard(Movable):
    """One independently owned feature slice; buffers die before the context."""
    var ctx: DeviceContext
    var blocks: List[DeviceBlock]
    var inputs: List[DeviceBuffer[DType.uint32]]
    var stats: DeviceBuffer[DType.float32]
    var scale: DeviceBuffer[DType.float32]
    var hist: DeviceBuffer[DType.float32]
    var acc: DeviceBuffer[DType.int32]
    var scratch: DeviceBuffer[DType.float32]
    var first: Int
    var folds: Int
    var replicas: Int

    def __deinit__(deinit self):
        _ = self.blocks^
        _ = self.inputs^
        _ = self.stats^
        _ = self.scale^
        _ = self.hist^
        _ = self.acc^
        _ = self.scratch^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def _feature_descriptor(ctx: DeviceContext, values: List[UInt32]) raises -> DeviceBuffer[DType.uint32]:
    var out = ctx.enqueue_create_buffer[DType.uint32](len(values))
    var host = ctx.enqueue_create_host_buffer[DType.uint32](len(values))
    for i in range(len(values)):  # small-loop(values: shard descriptor words): launch argument staging only
        host.unsafe_ptr()[i] = values[i]
    ctx.enqueue_copy(dst_buf=out, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return out^


def launch_feature_shards[hist2_smem_mode: Int, ridx_stats: Bool](
    ctx: DeviceContext, mut blocks: List[DeviceBlock], depth: Int,
    n_live: Int, n_rows: Int, stat_count: Int, max_leaves: Int, sm_count: Int,
    fixed_scale: MutPointer[Float32, MutAnyOrigin],
    mut cindex: DeviceBuffer[DType.uint32], mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32], mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32], mut ids: DeviceBuffer[DType.uint32],
    mut dense_ids: DeviceBuffer[DType.uint32], mut hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int, devices: Int, plan: DevicePlan,
) raises:
    """Partition whole packed feature groups; retain each row reduction.

    Reached only from `gbdt_hist_dispatch` when the plan has more than one
    device; moved here unchanged from `greedy_search_helper.mojo` except that
    each rank's device comes from the plan value (`gbdt_plan_shard_device`).

    A shard owns only its compressed feature columns and histogram columns.
    Row metadata and statistics are replicated. The existing driver still owns
    the full root histogram and compressed index: this is not yet an out-of-core
    fit. Original global row replication is retained even for the float-partial
    binary/half-byte kernels. The root receives disjoint columns, never sums of
    separately rounded histograms. Prefix scans and split selection stay global.

    THE NEGATIVE CONTROL. `-D MOJOLEARN_GBDT_PARALLEL_SABOTAGE=1` makes every
    owner above rank 0 read its compressed feature columns one GROUP early.
    The group count `(end_group - first_group)`, the bin range `first`, the
    fold count, every descriptor and the copy-back all keep the true
    `first_group`, so no allocation, length or validation changes; only the
    packed columns a shard histograms do. `active` is `min(devices, groups)`,
    so `first_group >= 1` whenever `rank >= 1`. It is a `comptime if`, so no
    production bit can move, and it is INERT AT ONE DEVICE, which is what makes
    a moved `par-boosting`, `par-boosting-clf`, `par-boosting-reg`,
    `par-ordered`, `par-ordered-rmse`, `par-border-types` or `par-feature-freq`
    cell attributable to the define rather than to the second device. Owed a
    two-device column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates
    this driver, so no CPU column can watch it fire.
    """
    ctx.synchronize()
    var scale = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[copy_scalar_kernel](fixed_scale, scale.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
    ctx.synchronize()
    var first_bin = 0
    for b in range(len(blocks)):
        ref block = blocks[b]
        var group_size = 4
        if block.policy == POLICY_BINARY:
            group_size = 32
        elif block.policy == POLICY_HALF_BYTE:
            group_size = 8
        var groups = feature_groups_for(block.policy, block.n_features)
        var active = min(devices, groups)
        var host_folds = ctx.enqueue_create_host_buffer[DType.uint32](block.n_features)
        ctx.enqueue_copy(dst_buf=host_folds, src_buf=block.folds)
        ctx.synchronize()
        var prefix = List[Int](length=block.n_features + 1, fill=0)
        for f in range(block.n_features):  # small-loop(n_features: block fold counts): shard bin ranges are launch shape
            prefix[f + 1] = prefix[f] + Int(host_folds.unsafe_ptr()[f])
        var shards = List[FeatureHistogramShard]()
        for rank in range(active):
            var first_group = groups * rank // active
            var end_group = groups * (rank + 1) // active
            var first_feature = first_group * group_size
            var end_feature = min(block.n_features, end_group * group_size)
            var nf = end_feature - first_feature
            var folds = prefix[end_feature] - prefix[first_feature]
            var first = first_bin + prefix[first_feature]
            var cells = max_leaves * stat_count * folds
            var device = DeviceContext(device_id=gbdt_plan_shard_device(plan, rank))
            var desc_folds = List[UInt32]()
            var desc_offsets = List[UInt32]()
            var desc_groups = List[UInt32](length=nf, fill=0)
            var desc_sizes = List[UInt32](length=nf, fill=UInt32(folds))
            for f in range(first_feature, end_feature):  # small-loop(end_feature: shard feature descriptors): launch argument lists per shard
                desc_folds.append(host_folds.unsafe_ptr()[f])
                desc_offsets.append(UInt32(prefix[f] - prefix[first_feature]))
            var local_blocks = List[DeviceBlock]()
            # Keep the parent's bit-width ladder, including a partial last group.
            local_blocks.append(DeviceBlock(block.policy, nf, 0, folds, block.max_folds,
                _feature_descriptor(device, desc_folds), _feature_descriptor(device, desc_offsets),
                _feature_descriptor(device, desc_groups), _feature_descriptor(device, desc_sizes),
                block.replication_groups))
            var inputs = List[DeviceBuffer[DType.uint32]]()
            var source_group = first_group
            comptime if is_defined["MOJOLEARN_GBDT_PARALLEL_SABOTAGE"]():
                # Check-only arm: later owners read one packed group early.
                if rank > 0:
                    source_group = first_group - 1
            var columns = cindex.create_sub_buffer[DType.uint32](
                (block.first_column + source_group) * n_rows, (end_group - first_group) * n_rows)
            inputs.append(peer_clone(ctx, device, columns))
            inputs.append(peer_clone(ctx, device, row_index))
            inputs.append(peer_clone(ctx, device, p_off))
            inputs.append(peer_clone(ctx, device, p_sz))
            inputs.append(peer_clone(ctx, device, ids))
            inputs.append(peer_clone(ctx, device, dense_ids))
            var local_stats = peer_clone(ctx, device, stats)
            var local_scale = peer_clone(ctx, device, scale)
            var packed = ctx.enqueue_create_buffer[DType.float32](cells)
            ctx.enqueue_function[copy_columns_kernel[False]](hist.unsafe_ptr(), packed.unsafe_ptr(),
                Int32(hist_cells_per_leaf), Int32(first), Int32(folds), Int32(cells),
                grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1))
            ctx.synchronize()
            var local_hist = peer_clone(ctx, device, packed)
            _ = packed^
            var acc = device.enqueue_create_buffer[DType.int32](cells)
            var scratch = device.enqueue_create_buffer[DType.float32](cells)
            enqueue_fill(device, acc, Int32(0))
            enqueue_fill(device, scratch, Float32(0))
            device.synchronize()
            var replicas = replication_for(block.replication_groups, n_live, stat_count, sm_count, gather=(depth > 0))
            shards.append(FeatureHistogramShard(device^, local_blocks^, inputs^, local_stats^,
                local_scale^, local_hist^, acc^, scratch^, first, folds, replicas))
        # Each context queues its own original kernels. No source copies remain
        # outstanding here, and the launches need no host readback before join.
        for rank in range(active):
            ref shard = shards[rank]
            var ci = shard.inputs[0].create_sub_buffer[DType.uint32](0, len(shard.inputs[0]))
            var ri = shard.inputs[1].create_sub_buffer[DType.uint32](0, len(shard.inputs[1]))
            var po = shard.inputs[2].create_sub_buffer[DType.uint32](0, len(shard.inputs[2]))
            var ps = shard.inputs[3].create_sub_buffer[DType.uint32](0, len(shard.inputs[3]))
            var li = shard.inputs[4].create_sub_buffer[DType.uint32](0, len(shard.inputs[4]))
            var di = shard.inputs[5].create_sub_buffer[DType.uint32](0, len(shard.inputs[5]))
            launch_histograms_for_blocks[hist2_smem_mode, ridx_stats, False, False](
                shard.ctx, shard.blocks, depth, n_live, n_rows, stat_count, max_leaves, sm_count,
                rebind[MutPointer[Float32, MutAnyOrigin]](shard.scale.unsafe_ptr()), ci, ri, shard.stats,
                po, ps, li, di,
                shard.hist, shard.acc, shard.scratch, shard.folds,
                replicas_override=shard.replicas)
        for rank in range(active):
            ref shard = shards[rank]
            var packed = peer_clone(shard.ctx, ctx, shard.hist)
            var cells = len(packed)
            ctx.enqueue_function[copy_columns_kernel[True]](hist.unsafe_ptr(), packed.unsafe_ptr(),
                Int32(hist_cells_per_leaf), Int32(shard.first), Int32(shard.folds), Int32(cells),
                grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1))
            ctx.synchronize()
            _ = packed^
        _ = shards^
        _ = host_folds^
        first_bin += block.total_folds
    _ = scale^
