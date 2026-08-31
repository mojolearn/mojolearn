# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The feature descriptors the kernels read.

PORT OF `catboost/cuda/gpu_data/gpu_structures.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

    struct TCFeature {
        ui64 Offset;          // where this feature's group starts in cindex
        ui32 Mask;            // bits within the ui32
        ui32 Shift;
        ui32 FirstFoldIndex;  // where its bins start in the histogram
        ui32 Folds;
        bool OneHotFeature;
    };

**This is what "CatBoost's feature handling" reduces to at the GPU boundary.**
A feature is not a column. It is a (group column, shift, mask) triple naming
some bits inside a shared `UInt32`, plus a (first fold, fold count) pair
naming its slice of the histogram. Everything upstream, quantization, border
selection, categorical CTRs, exists to fill this struct.
"""


@fieldwise_init
struct CFeature(Copyable, ImplicitlyCopyable, Movable):
    """`TCFeature`, field for field."""

    var offset: UInt32
    """`Offset`. Which `UInt32` column of the compressed index holds this
    feature's group. Theirs is `ui64`; ours is `UInt32` because a kernel
    parameter is Int32-shaped and no dataset here reaches 4 billion columns."""

    var mask: UInt32
    """`Mask`, before shifting: `(1 << bits) - 1`."""

    var shift: UInt32
    """`Shift`, counted from the TOP of the word. See `grid_policy`."""

    var first_fold_index: UInt32
    """`FirstFoldIndex`. Where this feature's bins begin in the histogram,
    which is what `feature_fold_offset` carries into the kernels."""

    var folds: UInt32
    """`Folds`. How many bins the feature actually uses. `0` means the
    feature is absent from this block and every kernel tests it."""

    var one_hot_feature: Bool
    """`OneHotFeature`. Changes the split predicate from `>` to `==`; see
    `split_points.mojo`."""


@fieldwise_init
struct TTreeNode(Copyable, ImplicitlyCopyable, Movable):
    """`TTreeNode` (`gpu_data/gpu_structures.h:167`), field for field.

    ONE INTERNAL NODE of a NON-SYMMETRIC tree, in the flat pre-order layout
    their model builder emits (`model_builder.cpp`'s `TFlatTreeBuilder`).
    `LeftSubtree` and `RightSubtree` are SIZES, not indices: the number of
    leaves under each side. The left child's node, when it has one, is
    always at `index + 1`; the right child's is at
    `index + LeftSubtree` -- which is what their `VisitBins` walk uses
    (`non_symmetric_tree.h:79`, `:89`).

    Theirs are all `ui16`. That is not a detail to widen: it bounds a
    non-symmetric tree at 65,535 leaves per side, and `max_leaves` is
    `1 << max_depth` for Depthwise, so the bound bites only past depth 16.
    Ours are `UInt16` for the same reason and because a wider field would
    make a model file that theirs cannot read.

    `SplitTypes` is a PARALLEL vector on their structure and not a member
    here either (`non_symmetric_tree.h:122`, "used only for conversion").
    """

    var feature_id: UInt16
    """`FeatureId`, from the feature manager -- the same numbering
    `TBinarySplit.feature_id` uses."""

    var bin: UInt16
    """`Bin`, the border index within the feature."""

    var left_subtree: UInt16
    """`LeftSubtree`, the LEAF COUNT under the left child."""

    var right_subtree: UInt16
    """`RightSubtree`, the LEAF COUNT under the right child."""

    def __eq__(self, other: Self) -> Bool:
        """Their `operator==`, which ties all four members (`:177-179`)."""
        return (
            self.feature_id == other.feature_id
            and self.bin == other.bin
            and self.left_subtree == other.left_subtree
            and self.right_subtree == other.right_subtree
        )

    def __ne__(self, other: Self) -> Bool:
        """Their `operator!=`."""
        return not (self == other)
