# SPDX-License-Identifier: Apache-2.0
"""Whole packed feature groups; original pointwise document and sibling folds."""
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import sync_parallelize
from std.os import getenv
from core.multi_gpu import peer_clone, gbdt_shard_device_id
from checks.numerics import NUMERIC_IDENTICAL
from gbdt.methods.pointwise_kernels import compute_hist2, FoldsHistogram, HIST_BUILD_MODE, PW_PRIVATE_DOC_SLOTS


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


def pointwise_device_count() raises -> Int:
    var count = Int(getenv("MOJOLEARN_GBDT_DEVICE_COUNT", "1"))
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
    fixed_scale: Float32, requested: Int,
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
        var device = DeviceContext(device_id=gbdt_shard_device_id(rank))
        var v_offset = offset.create_sub_buffer[DType.uint32](begin,width)
        var d_offset = peer_clone(ctx,device,v_offset)
        var v_first = first.create_sub_buffer[DType.uint32](begin,width)
        var d_first = peer_clone(ctx,device,v_first)
        var v_folds = folds.create_sub_buffer[DType.uint32](begin,width)
        var d_folds = peer_clone(ctx,device,v_folds)
        var v_one_hot = one_hot.create_sub_buffer[DType.uint8](begin,width)
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
    sync_parallelize(task,count)
    for rank in range(count):
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
