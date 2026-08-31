# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUNG 1's GATE: two independent searchers, one fixture, the same tree.

`NEXT_TWO.md` promised this from the start -- "its histograms must agree with
the greedy-subsets histograms this repo already has, on the same rows and the
same compressed index. That is a differential with an existing correct
implementation on the other side, which is the strongest gate available."

This is that, one level up: not the histograms but the TREE. Both searchers
grow an oblivious tree from the same compressed index and the same weak
target, and the splits must agree feature for feature and bin for bin.

## Why the two are genuinely independent

They share the compressed index and nothing else. Every layer between it and
the split is a different file:

    greedy_subsets_searcher/          methods/kernel/pointwise_hist2_*
      kernel/hist_{binary,half_byte,    + methods/kernel/pointwise_scores
      one_byte,2_one_byte_*}            + methods/pointwise_kernels
      + kernel/compute_scores           + methods/pointwise_optimization_subsets
      + greedy_search_helper            + methods/pointwise_scores_calcer

Different histogram layouts (stat-major against stat-minor), different
collision schemes, different reduce shapes, OPPOSITE SIGN CONVENTIONS
(`PORTING.md` 94a), a fixed-point accumulator on one side at 8 bits and none
on the other. Agreement between them is not a tautology; it is two ports of
one algorithm arriving at one answer.

## What agreement does and does not prove

It does NOT prove either matches CatBoost -- both could share a misreading.
`tools/catboost_oracle.py` is what compares against their own output, and
`PORTING.md` 91 F records that it runs their CPU learner.

It DOES prove that the ~9,000 lines landed for the pointwise family compute
the same splits as an implementation that has been gated against CatBoost's
dumped decisions 48/48. A defect in either would have to be mirrored exactly
in the other to survive this.

## The fixture

Three features carry signal at different depths, so a searcher that finds
only the first split still fails. Bins are hashed per (row, feature) and the
fold counts span all three policies -- binary, half-byte and one-byte in the
5-, 6- and 7-bit ranges -- so both families dispatch across every kernel they
have. Weights are 1 and gradients are small integers, so every sum is exact
in float32 and a disagreement is a real disagreement.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
    run_tree_layout,
)
from gbdt.methods.oblivious_tree_doc_parallel_structure_searcher import (
    PointwiseTreeWorkspace,
    fit_oblivious_tree_structure,
)
from gbdt.models.oblivious_model import TBinarySplit
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE

comptime N_ROWS = 4000
comptime MAX_DEPTH = 4


def main() raises:
    var ctx = DeviceContext()

    # binary, half-byte, and one-byte across the 5/6/7-bit ranges
    # SIX one-byte features on purpose: 4 fit in one cindex word, so
    # features 5 and 6 of the policy live in the NEXT column. A fixture
    # whose signal all sits in the first group cannot see a feature_offset
    # computed from the policy's first column instead of the feature's own
    # -- which is exactly the bug this gate missed once.
    var folds: List[Int] = [1, 1, 12, 9, 20, 32, 48, 100, 64, 127]
    var n_features = len(folds)
    var lay = build_layout(folds)

    # ---- bins, hashed per (row, feature) -----------------------------
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins8 = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var host_bins = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        for r in range(N_ROWS):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var b = Int(x % UInt32(folds[f] + 1))
            col.append(b)
            hb.unsafe_ptr().unsafe_store(r, UInt8(b))
        host_bins.append(col^)
        ctx.enqueue_copy(dst_buf=bins8, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS),
            cf.mask,
            cf.shift,
            bins8.unsafe_ptr(),
            Int32(N_ROWS),
            cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # ---- a weak target with structure at three depths ----------------
    # feature 7 dominates, then feature 2, then feature 5 -- so a searcher
    # that finds only the first split still fails
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * N_ROWS)
    var tw = Float64(0.0)
    var tg = Float64(0.0)
    for r in range(N_ROWS):
        var g = Float64(0.0)
        # feature 9 is the SIXTH one-byte feature and lives in the second
        # cindex column of its policy; it carries the strongest signal
        if host_bins[9][r] > 60:
            g += 8.0
        if host_bins[2][r] > 6:
            g += 3.0
        if host_bins[8][r] > 30:
            g += 1.0
        g -= 6.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(N_ROWS + r, Float32(g))
        tw += 1.0
        tg += -g if g < 0.0 else g

    # ---- arm A: the greedy-subsets searcher --------------------------
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * N_ROWS)
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())
    var row_index = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS)
    for r in range(N_ROWS):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    var scratch = ctx.enqueue_create_buffer[DType.float32](1)
    var greedy_splits = List[TBinarySplit]()
    var leaf_values = List[Float32]()
    var leaf_offsets = List[Int]()
    var ws = List[TTreeWorkspace]()
    _ = run_tree_layout(
        ctx, N_ROWS, folds, MAX_DEPTH, cindex, stats, row_index, scratch,
        Float32(tw), Float32(tg),
        greedy_splits, leaf_values, leaf_offsets, ws,
        score_function=SCORE_FUNCTION_COSINE,
    )

    # ---- arm B: the pointwise searcher -------------------------------
    var d_w = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_g = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    ctx.enqueue_copy(
        dst_buf=d_w, src_ptr=hs.unsafe_ptr()
    )
    ctx.enqueue_copy(
        dst_buf=d_g, src_ptr=hs.unsafe_ptr().unsafe_offset(N_ROWS)
    )
    ctx.synchronize()

    var pw_pool = List[PointwiseTreeWorkspace]()
    var pw_splits = fit_oblivious_tree_structure(
        ctx, lay, N_ROWS, MAX_DEPTH, cindex, d_w^, d_g^, 10, Float32(1.0),
        SCORE_FUNCTION_COSINE, pw_pool,
    )

    # ---- compare -----------------------------------------------------
    print("greedy-subsets searcher:", len(greedy_splits), "splits")
    for i in range(len(greedy_splits)):
        print(
            "   depth", i, "feature", greedy_splits[i].feature_id,
            "bin", greedy_splits[i].bin_idx,
        )
    print("pointwise searcher     :", len(pw_splits), "splits")
    for i in range(len(pw_splits)):
        print(
            "   depth", i, "feature", pw_splits[i].feature_id,
            "bin", pw_splits[i].bin_idx,
        )

    var failures = 0
    if len(greedy_splits) != len(pw_splits):
        print(
            "FAIL: the two searchers grew trees of DIFFERENT DEPTH --",
            len(greedy_splits), "against", len(pw_splits),
        )
        failures += 1
    else:
        var differ = 0
        for i in range(len(pw_splits)):
            if (
                greedy_splits[i].feature_id != pw_splits[i].feature_id
                or greedy_splits[i].bin_idx != pw_splits[i].bin_idx
            ):
                print(
                    "   MISMATCH at depth", i, ": greedy picked feature",
                    greedy_splits[i].feature_id, "bin",
                    greedy_splits[i].bin_idx, "; pointwise picked feature",
                    pw_splits[i].feature_id, "bin", pw_splits[i].bin_idx,
                )
                differ += 1
        if differ != 0:
            print("FAIL:", differ, "of", len(pw_splits), "splits differ")
            failures += 1

    if len(pw_splits) == 0:
        print("FAIL: the pointwise searcher grew NO splits at all")
        failures += 1

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print(
        "rung 1: two independent searchers, one fixture,",
        len(pw_splits), "splits identical",
    )
