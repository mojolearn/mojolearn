# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for the POINTWISE TREE POOL's reset half.

The pooled path reuses `TOptimizationSubsets` and the calcer across trees
(PREP_BILL step 21: their per-tree reconstruction is ~17-26 ms/tree of
pure allocation and constructor drains). The pool's contract is
CONSTRUCTOR POSTCONDITIONS: after a reset, the struct must be
indistinguishable from a freshly built one over the same source. This
file holds that contract BIT-EXACTLY where it is reachable today:

P1  `reset_subsets(pooled, src_b)` -- after the pooled struct ran two real
    `split_subsets` levels over a DIFFERENT target -- leaves every live
    region bit-identical to `create_subsets(src_b)` built fresh: bins,
    indices, part_ids, both gathered stat planes, the root partition
    record, the root partition stats, and the three counters.

    TEETH, per [[uniform-test-data-hides-permutation]]: before the reset,
    every compared region that the two splits or the target swap SHOULD
    have moved is REQUIRED to differ from the reference. A comparison
    that passes on state which never diverged verifies nothing; this one
    is shown its own discrimination first. (`part_ids` is the exception
    and is compared for equality only: nothing in a tree writes it, so it
    has no dirty state to discriminate.)

P2  `ComputeHistogramsHelper.reset()` restores the constructor's fields,
    and the PLAN SEQUENCE after a reset is identical to a fresh helper's
    -- including the `was_from_scratch` flags, which are what decide the
    full-pass/partial-pass kernel dispatch.

RESIDUAL, recorded rather than waved off ([[reached-but-inert]]):
`PolicyScoreHelper.reset_for_tree`'s `d_hist` memset has no caller until
the searcher integration lands (blocked on the feature lane's in-flight
edit to `oblivious_tree_doc_parallel_structure_searcher.mojo`). Its gate
is the pooled-vs-fresh bit-identity of whole fits once wired; until then
the memset rests on the constructor's own memset being the semantics it
restates, not on a run.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_u32
from gbdt.methods.histograms_helper import ComputeHistogramsHelper
from gbdt.methods.pointwise_optimization_subsets import (
    PARTITION_RECORD,
    PARTITION_STAT_STRIDE,
    TL2Target,
    create_subsets,
    reset_subsets,
    split_subsets,
)

#: Prime, so no block size divides it (same reasoning as
#: `pointwise_subsets_check.mojo`'s fixture).
comptime N_DOCS = 2003
comptime MAX_DEPTH = 5

#: Column 0 carries feature A at bits [0,4); column 1 carries feature B at
#: bits [8,12). Same packing as the subsets check's fixture.
comptime FEAT_A_OFFSET = 0
comptime FEAT_A_SHIFT = UInt32(0)
comptime FEAT_A_MASK = UInt32(0xF)
comptime FEAT_B_OFFSET = N_DOCS
comptime FEAT_B_SHIFT = UInt32(8)
comptime FEAT_B_MASK = UInt32(0xF)


def _mix(i: Int, salt: Int) -> UInt64:
    """A splitmix step: hashed values, never uniform ones, so a permutation
    defect cannot hide behind identical cells."""
    var z = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt) * UInt64(
        0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _val_a(doc: Int) -> Int:
    return Int(_mix(doc, 1) & 15)


def _val_b(doc: Int) -> Int:
    return Int(_mix(doc, 2) & 15)


def _weight(doc: Int, salt: Int) -> Float32:
    return Float32(Int(_mix(doc, salt) % 97) + 1)


def _target(doc: Int, salt: Int) -> Float32:
    return Float32(Int(_mix(doc, salt + 1) % 89) + 3)


def _make_target(
    ctx: DeviceContext, salt: Int
) raises -> TL2Target:
    """One TL2Target with salt-hashed planes, uploaded and drained."""
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](N_DOCS)
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](N_DOCS)
    for d in range(N_DOCS):
        h_w.unsafe_ptr().unsafe_store(d, _weight(d, salt))
        h_t.unsafe_ptr().unsafe_store(d, _target(d, salt))
    var d_w = ctx.enqueue_create_buffer[DType.float32](N_DOCS)
    var d_t = ctx.enqueue_create_buffer[DType.float32](N_DOCS)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.synchronize()
    _ = h_w[0]
    _ = h_t[0]
    return TL2Target(d_w^, d_t^, N_DOCS)


def _read_u32(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.uint32],
    count: Int,
) raises -> List[UInt32]:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](count)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    var out = List[UInt32]()
    for i in range(count):
        out.append(h[i])
    return out^


def _read_f32(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    count: Int,
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(count):
        out.append(h[i])
    return out^


def _diff_u32(a: List[UInt32], b: List[UInt32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def _diff_f32(a: List[Float32], b: List[Float32]) -> Int:
    """Bit comparison, not value comparison: the contract is
    indistinguishability, and -0.0 == 0.0 would hide a sign."""
    var n = 0
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            n += 1
    return n


def check_subsets_reset() raises:
    """P1: the pooled reset against a fresh create, bit for bit."""
    var ctx = DeviceContext()
    var failures = 0

    # ---- the compressed index, two columns, shared by both arms ---------
    var h_cindex = ctx.enqueue_create_host_buffer[DType.uint32](2 * N_DOCS)
    for d in range(N_DOCS):
        h_cindex.unsafe_ptr().unsafe_store(d, UInt32(_val_a(d)))
        h_cindex.unsafe_ptr().unsafe_store(
            N_DOCS + d, UInt32(_val_b(d) << 8)
        )
    var d_cindex = ctx.enqueue_create_buffer[DType.uint32](2 * N_DOCS)
    ctx.enqueue_copy(dst_buf=d_cindex, src_ptr=h_cindex.unsafe_ptr())
    ctx.synchronize()
    _ = h_cindex[0]

    # ---- REFERENCE: a fresh build over target B -------------------------
    var src_b_ref = _make_target(ctx, 21)
    var fresh = create_subsets(ctx, MAX_DEPTH, src_b_ref)
    ctx.synchronize()

    var ref_bins = _read_u32(ctx, fresh.bins, N_DOCS)
    var ref_idx = _read_u32(ctx, fresh.indices, N_DOCS)
    var ref_pids = _read_u32(ctx, fresh.part_ids, fresh.max_part_count)
    var ref_gw = _read_f32(ctx, fresh.gathered_weight, N_DOCS)
    var ref_gt = _read_f32(ctx, fresh.gathered_target, N_DOCS)
    var ref_root = _read_u32(ctx, fresh.partitions, PARTITION_RECORD)
    var ref_stats = _read_f32(
        ctx, fresh.partition_stats, PARTITION_STAT_STRIDE
    )

    # ---- POOLED: build over target A, dirty it with two real splits -----
    var src_a = _make_target(ctx, 11)
    var pooled = create_subsets(ctx, MAX_DEPTH, src_a)
    var d_obs = ctx.enqueue_create_buffer[DType.uint32](N_DOCS)
    var d_docs = ctx.enqueue_create_buffer[DType.uint32](N_DOCS)
    launch_make_sequence(ctx, UInt32(0), d_obs, N_DOCS)
    ctx.synchronize()

    # split 1: feature A > bin 6; split 2: feature B > bin 5. Real splits
    # over the real kernels -- bins, indices, partitions, stats and depth
    # all move.
    launch_gather_with_mask_u32(
        ctx, d_docs, d_obs, pooled.indices, N_DOCS, UInt32(0xFFFFFFFF)
    )
    split_subsets(
        ctx, src_a, d_cindex, d_docs, UInt32(FEAT_A_OFFSET), FEAT_A_MASK,
        FEAT_A_SHIFT, False, UInt32(6), pooled,
    )
    ctx.synchronize()
    launch_gather_with_mask_u32(
        ctx, d_docs, d_obs, pooled.indices, N_DOCS, UInt32(0xFFFFFFFF)
    )
    split_subsets(
        ctx, src_a, d_cindex, d_docs, UInt32(FEAT_B_OFFSET), FEAT_B_MASK,
        FEAT_B_SHIFT, False, UInt32(5), pooled,
    )
    ctx.synchronize()

    # ---- TEETH: the dirty state must actually differ --------------------
    var dirty_bins = _read_u32(ctx, pooled.bins, N_DOCS)
    var dirty_idx = _read_u32(ctx, pooled.indices, N_DOCS)
    var dirty_gw = _read_f32(ctx, pooled.gathered_weight, N_DOCS)
    var dirty_gt = _read_f32(ctx, pooled.gathered_target, N_DOCS)
    var dirty_root = _read_u32(ctx, pooled.partitions, PARTITION_RECORD)
    var dirty_stats = _read_f32(
        ctx, pooled.partition_stats, PARTITION_STAT_STRIDE
    )
    var teeth_ok = True
    if _diff_u32(dirty_bins, ref_bins) == 0:
        print("FAIL P1 teeth: two splits left bins identical to fresh")
        teeth_ok = False
    if _diff_u32(dirty_idx, ref_idx) == 0:
        print("FAIL P1 teeth: two splits left indices unpermuted")
        teeth_ok = False
    if _diff_f32(dirty_gw, ref_gw) == 0 or _diff_f32(dirty_gt, ref_gt) == 0:
        print(
            "FAIL P1 teeth: target A's gathered planes match target B's"
            " -- the two sources do not discriminate"
        )
        teeth_ok = False
    if _diff_u32(dirty_root, ref_root) == 0:
        print("FAIL P1 teeth: the root partition record never moved")
        teeth_ok = False
    if _diff_f32(dirty_stats, ref_stats) == 0:
        print("FAIL P1 teeth: the root partition stats never moved")
        teeth_ok = False
    if Int(pooled.current_depth) != 2:
        print(
            "FAIL P1 teeth: two splits left current_depth at",
            pooled.current_depth,
        )
        teeth_ok = False
    if not teeth_ok:
        failures += 1
    else:
        print(
            "  ok   P1 teeth -- every mutable region diverged before the"
            " reset (bins", _diff_u32(dirty_bins, ref_bins), "cells,"
            " indices", _diff_u32(dirty_idx, ref_idx), "cells)"
        )

    # ---- THE RESET, over a second identical upload of target B ----------
    var src_b = _make_target(ctx, 21)
    reset_subsets(ctx, pooled, src_b)
    ctx.synchronize()

    var got_bins = _read_u32(ctx, pooled.bins, N_DOCS)
    var got_idx = _read_u32(ctx, pooled.indices, N_DOCS)
    var got_pids = _read_u32(ctx, pooled.part_ids, pooled.max_part_count)
    var got_gw = _read_f32(ctx, pooled.gathered_weight, N_DOCS)
    var got_gt = _read_f32(ctx, pooled.gathered_target, N_DOCS)
    var got_root = _read_u32(ctx, pooled.partitions, PARTITION_RECORD)
    var got_stats = _read_f32(
        ctx, pooled.partition_stats, PARTITION_STAT_STRIDE
    )

    var bad = 0
    bad += _diff_u32(got_bins, ref_bins)
    bad += _diff_u32(got_idx, ref_idx)
    bad += _diff_u32(got_pids, ref_pids)
    bad += _diff_f32(got_gw, ref_gw)
    bad += _diff_f32(got_gt, ref_gt)
    bad += _diff_u32(got_root, ref_root)
    bad += _diff_f32(got_stats, ref_stats)
    if Int(pooled.current_depth) != 0:
        print(
            "FAIL P1: current_depth is", pooled.current_depth,
            "after reset, want 0"
        )
        failures += 1
    if Int(pooled.fold_count) != 0 or Int(pooled.fold_bits) != 0:
        print(
            "FAIL P1: fold counters are", pooled.fold_count,
            pooled.fold_bits, "after reset, want 0 0"
        )
        failures += 1
    if bad != 0:
        print(
            "FAIL P1: --", bad, "cells differ between the pooled reset and"
            " a fresh create over the same source"
        )
        failures += 1
    else:
        print(
            "  ok   P1 -- reset_subsets is bit-identical to create_subsets"
            " across", 3 * N_DOCS + len(ref_pids) + PARTITION_RECORD,
            "u32 and", 2 * N_DOCS + PARTITION_STAT_STRIDE, "f32 cells"
        )

    if failures != 0:
        raise Error(String(failures) + " P1 gate(s) failed")


def check_hist_helper_reset() raises:
    """P2: the state machine after reset() replays a fresh helper's plan
    sequence exactly, `was_from_scratch` included."""
    var failures = 0

    var used = ComputeHistogramsHelper(0, 1, MAX_DEPTH)
    # a full tree's worth of plans, so current_bit ends at MAX_DEPTH - 1
    for d in range(MAX_DEPTH):
        _ = used.plan(d)
        used.clear_from_scratch()
    if used.current_bit != MAX_DEPTH - 1 or used.build_from_scratch:
        print(
            "FAIL P2 teeth: after a tree the helper is not dirty --"
            " current_bit", used.current_bit, "build_from_scratch",
            used.build_from_scratch,
        )
        failures += 1

    used.reset()
    if used.current_bit != -1 or not used.build_from_scratch:
        print(
            "FAIL P2: reset left current_bit", used.current_bit,
            "build_from_scratch", used.build_from_scratch,
            "-- the constructor's state is (-1, True)"
        )
        failures += 1

    var fresh = ComputeHistogramsHelper(0, 1, MAX_DEPTH)
    for d in range(MAX_DEPTH):
        var pu = used.plan(d)
        var pf = fresh.plan(d)
        if (
            pu.current_bit != pf.current_bit
            or pu.build_from_scratch != pf.build_from_scratch
            or pu.part_count != pf.part_count
        ):
            print(
                "FAIL P2: plan sequences diverge at depth", d, "-- reset (",
                pu.current_bit, pu.build_from_scratch, pu.part_count,
                ") vs fresh (", pf.current_bit, pf.build_from_scratch,
                pf.part_count, ")"
            )
            failures += 1
        used.clear_from_scratch()
        fresh.clear_from_scratch()

    if failures != 0:
        raise Error(String(failures) + " P2 gate(s) failed")
    print(
        "  ok   P2 -- reset() replays a fresh helper's plan sequence over",
        MAX_DEPTH, "levels, was_from_scratch flags included"
    )


def main() raises:
    check_subsets_reset()
    check_hist_helper_reset()
    print("pointwise pool resets: P1-P2 pass")
