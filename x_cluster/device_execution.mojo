# SPDX-License-Identifier: Apache-2.0
"""One native execution owner shared by typed requests and legacy boundaries.

All promoted and optional pointer routes retain their original order and
defaults. Python admission and output packing stay outside this function.
"""
from core.hostptr import f32_ptr, read_f32
from core.device_scan import device_first_nonfinite
from x_cluster.device_ops import DeviceOps
from x_cluster.bisect import IDN_BISECT_DEVICE_CENTER
from x_cluster.bisect_fast import BISECT_FAST_ZEROCOPY, bisect_entry_ptr
from x_cluster.entries import (
    ENTRY_BISECT, ENTRY_MINIBATCH, ENTRY_MEANSHIFT, ENTRY_OPTICS, ENTRY_AFFINITY, bisect_entry, run_entry,
    meanshift_entry, optics_entry, affinity_entry, ENTRY_BGMM, ENTRY_BGMM_SCORE, bgmm_entry, bgmm_score_entry,
)
from x_cluster.minibatch_ptr import (
    MBK_DEVSCAN_ANY, MBK_IDN_RESIDENT, MBK_NONFINITE_MSG, MBK_ZEROCOPY,
    minibatch_entry_ptr, minibatch_entry_ptr_idn,
)
from x_cluster.out import ClusterOut


def cluster_fold_entry(which: Int) -> Bool:
    """The entries whose X is read only as a float slot of the ops (lane vdf-scope-cluster6, with BisectingKMeans
    and MiniBatchKMeans before it): MeanShift (the bandwidth estimate, the seeds as device copies), OPTICS and
    AffinityPropagation (its exemplar rows gathered on the device); BayesianGaussianMixture's fit and score (lane
    vdf-scope-cluster7: its k-means and k-means++ initialisations read the one slot, `kmeans_slot` /
    `greedy_kmeans_pp_indices_slot`). The agglomerative, nearest, distances and spectral-assign entries keep
    host X."""
    return (which == ENTRY_MEANSHIFT or which == ENTRY_OPTICS or which == ENTRY_AFFINITY or which == ENTRY_BGMM
            or which == ENTRY_BGMM_SCORE)


def _fold_execute(which: Int, source: Int, count: Int, auxiliary: Int, auxiliary_count: Int,
                  ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
    """THE FOLD X WORD (core/fold_x_device.mojo): a negative `source` names a model-selection fold's x_prep store
    slot (core/member_fold_x.mojo). X goes into a float slot of the ops by one device copy (`put_addr`) and the
    entry reads that slot; the auxiliary block (seeds, preferences) stays host words. Same words, same kernels,
    same order as the host List's upload: no bit moves."""
    if not cluster_fold_entry(which):
        raise Error("x_cluster: entry " + String(which) + " reads host X words (no fold X word)")
    var a = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
    var ops = DeviceOps()
    var xs = ops.put_addr(source, count)
    var holder = List[Float32](length=1, fill=Float32(0))
    var result: ClusterOut
    if which == ENTRY_MEANSHIFT:
        result = meanshift_entry(ops, holder, a, ints, floats, xs)
    elif which == ENTRY_OPTICS:
        result = optics_entry(ops, holder, ints, floats, xs)
    elif which == ENTRY_BGMM:
        result = bgmm_entry(ops, holder, a, ints, floats, xs)
    elif which == ENTRY_BGMM_SCORE:
        result = bgmm_score_entry(ops, holder, a, ints, xs)
    else:
        result = affinity_entry(ops, holder, a, ints, floats, xs)
    _ = ops^
    _ = holder^
    _ = a^
    return result^


def cluster_device_execute(which: Int, source: Int, count: Int, auxiliary: Int,
                           auxiliary_count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
    # Same route sequence as the original x_cluster_call binding. Keep tuned
    # kernels behind their existing switches, including both rollback paths.
    comptime if MBK_ZEROCOPY:
        if which == ENTRY_MINIBATCH and count > 0:
            var a0 = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
            var xp = f32_ptr(source)
            var res0 = ClusterOut()
            var took = False
            var ops0 = DeviceOps()
            took = minibatch_entry_ptr(ops0, xp, count, a0, ints, floats, res0)
            _ = ops0^
            _ = a0^
            if took:
                return res0^
    comptime if BISECT_FAST_ZEROCOPY:
        if which == ENTRY_BISECT and count > 0:
            var xp1 = f32_ptr(source)
            var res1 = ClusterOut()
            var took1 = False
            var ops1 = DeviceOps()
            took1 = bisect_entry_ptr(ops1, xp1, count, auxiliary_count > 0, ints, floats, res1)
            _ = ops1^
            if took1:
                return res1^
    comptime if MBK_IDN_RESIDENT:
        if which == ENTRY_MINIBATCH and count > 0:
            var a2 = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
            var xp2 = f32_ptr(source)
            var res2 = ClusterOut()
            var took2 = False
            var ops2 = DeviceOps()
            took2 = minibatch_entry_ptr_idn(ops2, xp2, count, a2, ints, floats, res2)
            _ = ops2^
            _ = a2^
            if took2:
                return res2^
    comptime if IDN_BISECT_DEVICE_CENTER:
        # fx-board-preprocess (2026-11-19): BisectingKMeans.fit under
        # IDENTICAL reads X only as its device slot (bisect_fit centers,
        # gathers and scores on the device), so X goes up straight from the
        # caller's words (`put_addr`, synchronised before it returns; the
        # door keeps X alive through the call) and is not copied into a host
        # List first. That copy was one serial memcpy of all n * d words into
        # new memory (a page fault per 4 KiB) made only to be uploaded: on
        # the L40S board job of d3549c756b a 1,000,000 x 220 fit took 291,129
        # faults and 529 ms of system time in a 1,009 ms call. The same
        # words reach the same kernels in the same order: no bit moves.
        # bisect_predict_entry took the same step on 2026-11-17 (lane
        # fb2-neighbors). FAST keeps its own routes above and below.
        if which == ENTRY_BISECT and count > 0:
            var aw = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
            var opsb = DeviceOps()
            var xsb = opsb.put_addr(source, count)
            var holder = List[Float32](length=1, fill=Float32(0))
            var resb = bisect_entry(opsb, holder, aw, ints, floats, xsb)
            _ = opsb^
            _ = holder^
            _ = aw^
            return resb^
    if source < 0 and count > 0:
        return _fold_execute(which, source, count, auxiliary, auxiliary_count, ints, floats)
    var x = read_f32(source, count) if count > 0 else List[Float32]()
    var a = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
    var scan_dev = False
    comptime if MBK_DEVSCAN_ANY:
        scan_dev = which == ENTRY_MINIBATCH and len(ints) > 12 and ints[12] != 0 and count > 0
    var result = ClusterOut()
    var bad = False
    var ops = DeviceOps()
    if scan_dev:
        var xd = ops.ctx.enqueue_create_buffer[DType.float32](count)
        ops.ctx.enqueue_copy(dst_buf=xd, src_ptr=x.unsafe_ptr())
        bad = device_first_nonfinite(ops.ctx, xd, count) >= 0
        _ = xd^
    if not bad:
        result = run_entry(ops, which, x, a, ints, floats)
    _ = ops^
    _ = x^
    _ = a^
    if bad:
        raise Error(MBK_NONFINITE_MSG)
    return result^
