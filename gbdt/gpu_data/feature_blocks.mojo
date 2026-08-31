# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Per-policy feature descriptor arrays, ready for a histogram launch.

MIRRORS the `TFeatureInBlock` assembly in
`catboost/cuda/methods/greedy_subsets_searcher/split_properties_helper.cpp`,
which walks the compressed index's feature blocks and hands each policy's
slice to its own kernel. Their `B` in the `3B + 12` launch census IS the
number of policies present.

The kernels take four parallel arrays rather than a struct, because a Mojo
kernel parameter is a pointer or a scalar. This builds those arrays for one
policy, restricted to the features that landed in it.
"""

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout


@fieldwise_init
struct PolicyBlock(Copyable, Movable):
    """One policy's features, as the four arrays a histogram kernel takes."""

    var policy: Int
    var feature_ids: List[Int]
    """Original feature indices, so a caller can map a split back."""
    var folds: List[UInt32]
    var fold_offset: List[UInt32]
    var group_offset: List[UInt32]
    var group_size: List[UInt32]
    var first_column: Int
    """The compressed-index column this policy's block starts at. The kernel
    strides from here by `bins_line_size`, so a policy's columns must be
    CONTIGUOUS, which `build_layout` guarantees by assigning columns in this
    same policy order: one policy's columns are all opened before the next
    policy's first one."""

    def count(self) -> Int:
        return len(self.feature_ids)


def blocks_for(layout: CompressedIndexLayout, n_rows: Int) raises -> List[
    PolicyBlock
]:
    """Split a layout into one block per policy PRESENT.

    A policy with no features produces no block and therefore no launch,
    which is what keeps the launch count at `3B + 12` with `B` the number of
    policies actually used rather than the three that exist.

    `group_size` is the policy's total fold count, because that is the stride
    the writeback uses between leaves (`entriesPerLeaf = statCount *
    group.GroupSize`). `fold_offset` is the feature's slice WITHIN the
    policy's block, not within the whole histogram, for the same reason.

    **THIS ORDER IS THE FLAT BIN ORDER.** `launch_histograms_for_blocks`
    walks the blocks in the order returned here and advances
    `block_first_bin` by each block's fold count, so a feature's destination
    in the flat histogram is `sum(earlier blocks' folds) + fold_offset`.
    `build_layout` assigns `first_fold_index` by walking policies and
    features in exactly this order, which is the port of their one-builder
    invariant: their group write offset and their per-feature
    `FirstFoldIndex` both come off `BinFeaturesBuilder`
    (`compute_by_blocks_helper.cpp:189` and `:218`) inside one `AddGroup`
    pass over the groups (`:341`). Reorder either walk without the other and
    the scan, `resolve_split` and the skip mask all address the wrong cells.
    """
    var out = List[PolicyBlock]()

    for policy in range(3):
        var ids = List[Int]()
        for i in range(len(layout.policy_of)):
            if layout.policy_of[i] == policy and Int(
                layout.features[i].folds
            ) > 0:
                ids.append(i)
        if len(ids) == 0:
            continue

        var folds = List[UInt32]()
        var fold_off = List[UInt32]()
        var grp_off = List[UInt32]()
        var grp_sz = List[UInt32]()

        var total_folds = 0
        for k in range(len(ids)):
            total_folds += Int(layout.features[ids[k]].folds)

        var first_col = Int(layout.features[ids[0]].offset)
        var cursor = 0
        for k in range(len(ids)):
            ref f = layout.features[ids[k]]
            folds.append(f.folds)
            fold_off.append(UInt32(cursor))
            grp_off.append(UInt32(0))
            grp_sz.append(UInt32(total_folds))
            cursor += Int(f.folds)

        out.append(
            PolicyBlock(
                policy, ids^, folds^, fold_off^, grp_off^, grp_sz^, first_col
            )
        )

    return out^
