# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The early-stop ROLLBACK of `run_tree_layout`, reached on purpose.

Every oracle fixture grows every tree to full depth (verified 2026-08-20:
12 trees x 4 splits in all three fixtures), so the one-level rollback the
device-side split resolution introduced (`kernel/split_resolve.mojo`,
DEVIATION BLOCK) is reached by NO other gate. A path nobody reaches is a
path nobody has checked -- this repository has been burned by exactly that
twice -- so this check builds two datasets whose stops are EXACT:

* ROOT STOP: every feature constant (all bins 0) AND the gradient +1/-1
  alternating, so every candidate leaf sum is EXACTLY zero and every
  score is exactly 0.0 (integer cancellation; float arithmetic is exact
  here) -- `best_score > 0` fails at depth 0. The tree must come back
  with NO splits, ONE leaf, and the root partition restored to all rows
  after the speculative level-0 split rolled back. (A constant-bin tree
  with a NON-cancelling gradient does NOT stop at the root: the
  full/empty split scores the parent's own positive score and the tree
  stops one level later through the REPEAT rule instead -- measured
  while building this check, and that behavior is unchanged from the
  host-gate code.)

* MID-TREE STOP: feature 0 carries exactly two bins (0 and 10, alternating
  rows), y equals the bin indicator, every other feature constant. Level 0
  separates the two groups exactly (any split bin 0..9 gives the same
  partition; the tie rule takes the smallest, feature 0 bin 0). Both
  children are then PURE, every level-1 gain is exactly 0.0, and the stop
  rolls level 1 back. The tree must come back with ONE split on feature 0,
  TWO leaves of exactly n/2 rows, and leaf values whose magnitudes match
  the hand-computed `|sum_der| / (size + l2)` -- which they can only do if
  the rolled-back partitions are the depth-1 partitions, since the tail
  computes leaf stats FROM the partitions after the rollback restored
  them.

Run twice, the mid-stop case must be bit-identical: the rollback path is
host arithmetic plus one upload, and nothing in it may depend on timing.
"""

from max.gpu.host import DeviceContext

from checks.hist2_check import build_cindex
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
    run_tree_layout,
)
from gbdt.models.oblivious_model import TBinarySplit

comptime ES_ROWS = 2048
comptime ES_FEATURES = 4
comptime ES_FOLDS = 20
comptime ES_L2 = Float32(3.0)


def _fit(
    ctx: DeviceContext,
    bins: List[List[Int]],
    grad: List[Float32],
    mut out_splits: List[TBinarySplit],
    mut out_leaf_values: List[Float32],
) raises -> List[Int]:
    var folds = List[Int]()
    for _ in range(ES_FEATURES):
        folds.append(ES_FOLDS)
    var lay = build_layout(folds)
    var cindex = build_cindex(ctx, lay, bins, ES_ROWS)

    var stats = ctx.enqueue_create_buffer[DType.float32](2 * ES_ROWS)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * ES_ROWS)
    var gmag = Float64(0.0)
    for r in range(ES_ROWS):
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(ES_ROWS + r, grad[r])
        var a = Float64(grad[r])
        if a < 0.0:
            a = -a
        gmag += a
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](ES_ROWS)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](ES_ROWS)
    for r in range(ES_ROWS):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var cursor = ctx.enqueue_create_buffer[DType.float32](ES_ROWS)
    ctx.enqueue_memset(cursor, Float32(0.0))
    ctx.synchronize()

    var _lo = List[Int]()
    var _ws0 = List[TTreeWorkspace]()
    var sizes = run_tree_layout(
        ctx, ES_ROWS, folds, 3, cindex, stats, row_index, cursor,
        Float32(ES_ROWS), Float32(gmag), out_splits, out_leaf_values,
        _lo,
        _ws0,
        apply_to_cursor=True, l2_leaf_reg=ES_L2,
    )
    return sizes^


def check_early_stop_rollback() raises:
    print("early-stop rollback (split_resolve deviation):")
    var ctx = DeviceContext()

    # ---- ROOT STOP -------------------------------------------------------
    var bins0 = List[List[Int]]()
    for _ in range(ES_FEATURES):
        var col = List[Int]()
        for _ in range(ES_ROWS):
            col.append(0)
        bins0.append(col^)
    var g0 = List[Float32]()
    for r in range(ES_ROWS):
        # +1/-1 alternating: every candidate sum cancels EXACTLY, so every
        # score is 0.0 and the root gate fires with no float ambiguity.
        g0.append(Float32(1.0) if (r & 1) == 0 else Float32(-1.0))
    var sp0 = List[TBinarySplit]()
    var lv0 = List[Float32]()
    var sz0 = _fit(ctx, bins0, g0, sp0, lv0)
    if len(sp0) != 0:
        raise Error("root stop: expected 0 splits, got " + String(len(sp0)))
    if len(sz0) != 1 or sz0[0] != ES_ROWS:
        raise Error("root stop: the rolled-back root partition does not"
                    " hold all rows")
    print("  root stop: 0 splits, one leaf of", sz0[0],
          "rows -- the depth-0 rollback restored the root partition")

    # ---- MID-TREE STOP ----------------------------------------------------
    var bins1 = List[List[Int]]()
    for f in range(ES_FEATURES):
        var col = List[Int]()
        for r in range(ES_ROWS):
            if f == 0 and (r & 1) == 1:
                col.append(10)
            else:
                col.append(0)
        bins1.append(col^)
    var g1 = List[Float32]()
    for r in range(ES_ROWS):
        g1.append(Float32(1.0) if (r & 1) == 1 else Float32(0.0))
    var sp1 = List[TBinarySplit]()
    var lv1 = List[Float32]()
    var sz1 = _fit(ctx, bins1, g1, sp1, lv1)
    if len(sp1) != 1:
        raise Error("mid stop: expected exactly 1 split, got "
                    + String(len(sp1)))
    if sp1[0].feature_id != Int32(0):
        raise Error("mid stop: split is not on the informative feature")
    if len(sz1) != 2:
        raise Error("mid stop: expected 2 leaves, got " + String(len(sz1)))
    if sz1[0] + sz1[1] != ES_ROWS:
        raise Error("mid stop: leaf sizes do not sum to the row count")
    if sz1[0] != ES_ROWS // 2 or sz1[1] != ES_ROWS // 2:
        raise Error("mid stop: the rolled-back partitions are not the"
                    " depth-1 halves: " + String(sz1[0]) + " + "
                    + String(sz1[1]))
    # leaf values FROM the restored partitions: one leaf sums 0, the other
    # sums n/2, magnitudes |sum| / (size + l2) in some order.
    if len(lv1) != 2:
        raise Error("mid stop: expected 2 leaf values")
    var half = Float32(ES_ROWS // 2)
    var expect_hot = half / (half + ES_L2)
    var a = lv1[0]
    if a < 0:
        a = -a
    var b = lv1[1]
    if b < 0:
        b = -b
    var hot = a
    var cold = b
    if b > a:
        hot = b
        cold = a
    if cold > Float32(1e-6) or (
        hot - expect_hot > Float32(1e-4)
        or expect_hot - hot > Float32(1e-4)
    ):
        raise Error("mid stop: leaf values do not match the restored"
                    " partitions: got magnitudes " + String(cold) + " / "
                    + String(hot) + ", want 0 / " + String(expect_hot))
    print("  mid stop: 1 split on feature 0, leaves", sz1[0], "/",
          sz1[1], ", leaf-value magnitudes", cold, "/", hot,
          "match the restored depth-1 partitions")

    # determinism of the rollback path
    var sp2 = List[TBinarySplit]()
    var lv2 = List[Float32]()
    var sz2 = _fit(ctx, bins1, g1, sp2, lv2)
    if (
        len(sp2) != len(sp1)
        or lv2[0] != lv1[0]
        or lv2[1] != lv1[1]
        or sz2[0] != sz1[0]
    ):
        raise Error("mid stop: two runs through the rollback differ")
    print("  rerun bit-identical: the rollback is deterministic")


def main() raises:
    # STANDALONE DRIVER, the same call `probe_main.mojo` makes.
    check_early_stop_rollback()
