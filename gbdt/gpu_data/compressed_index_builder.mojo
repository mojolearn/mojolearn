"""Assign features to grouping policies and lay out the compressed index.

PORT OF the layout half of
`catboost/cuda/gpu_data/compressed_index_builder.h` at CatBoost `54a8143a`.

**This is the step that makes a MIXED dataset work**, and its absence is why
this port's tree loop has only ever handled uniform binary features. Given a
fold count per feature it decides, for each one, which policy it belongs to,
which `UInt32` column it shares, which bits inside that column it owns, and
where its bins live in the histogram.

Their builder also owns the writing of the data (`WriteBinsVector` ->
`TCudaFeaturesLayoutHelper::WriteToCompressedIndex`), which the port covers
with `binarize.write_compressed_index_kernel`. What is ported here is the
LAYOUT DECISION, which is host-side arithmetic and needs no device at all.

NOT ported, and named so nobody assumes otherwise: their docParallel and
featureParallel layouts differ in how columns are striped across devices, and
this is the single-device layout. Multi-device striping is a different
function in the same header.
"""

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
    features_per_int,
    policy_for_fold_count,
    policy_mask,
    policy_shift,
)
from gbdt.gpu_data.gpu_structures import CFeature


@fieldwise_init
struct CompressedIndexLayout(Copyable, Movable):
    """What the builder decided, ready to hand to the kernels."""

    var features: List[CFeature]
    """One per input feature, in input order."""

    var policy_of: List[Int]
    """Which policy each feature landed in."""

    var columns: Int
    """How many `UInt32` columns the compressed index needs."""

    var hist_cells: Int
    """Total histogram cells across every feature, `sum(folds)`. This is the
    number `replicas_for` keys on, so the layout is where the replication
    decision gets its input."""

    def group_count(self, policy: Int) -> Int:
        """How many features landed in one policy."""
        var n = 0
        for i in range(len(self.policy_of)):
            if self.policy_of[i] == policy:
                n += 1
        return n


def build_layout(
    fold_counts: List[Int], one_hot: List[Bool] = List[Bool]()
) raises -> CompressedIndexLayout:
    """Assign policies and lay out columns and histogram slices.

    Their rule, from `grid_policy.h` plus the builder's grouping: a feature
    goes in the SMALLEST policy whose `MaxFolds` covers it, and features of
    the same policy are packed `FeaturesPerInt` to a column in order.

    ONE WALK, IN POLICY-BLOCK ORDER, AND IT HAS TO BE.
    ==================================================
    CatBoost derives a feature's `FirstFoldIndex` and its group's write
    offset into the flat histogram from the SAME builder in the SAME pass:
    `GroupBinFeatureOffsets` records `BinFeaturesBuilder.GetCurrentSize(dev)`
    at the top of `AddGroup` (`compute_by_blocks_helper.cpp:189`) and each
    feature's `FirstFoldIndex` is stamped from that same builder a few lines
    down (`:218`), with `AddGroup` called once per group in group order
    (`:341`). The two therefore agree by construction and cannot drift.

    This walk is the port of that invariant. It visits POLICIES in the order
    `feature_blocks.blocks_for` emits blocks, and features in input order
    within a policy, which is the order `launch_histograms_for_blocks`
    accumulates `block_first_bin` in. Assigning `first_fold_index` in INPUT
    order instead gives two independent walks that agree only when the input
    order happens to already be block order. Every check in this repository
    used a binary-first dataset, which IS block order, so the disagreement
    never showed. A dataset ordered one-byte-first then binary (covtype: 10
    continuous, then 44 binary) writes the binary bins near flat bin 0 while
    an input-order `first_fold_index` claims they start near 1280, and the
    scan, `resolve_split` and the skip mask all read the wrong cells.

    THE COLUMN WALK IS POLICY-ORDERED FOR THE SAME REASON. The histogram
    kernels address a feature group as `cindex_base + bins_line_size * g`
    (`hist_binary.mojo`, their `cindex += features->CompressedIndexOffset`),
    so a policy's columns must be CONTIGUOUS. A single global column cursor
    walked in input order interleaves them: 33 binary features, then a
    one-byte feature, then more binary, hands binary columns 0, 1 and 3 with
    the one-byte feature holding 2, and the second binary group then reads
    the one-byte feature's bits.

    A feature with 0 folds is CONSTANT and is given `folds = 0`, which every
    kernel already tests, rather than being dropped: dropping it would
    renumber every feature after it and break the caller's column mapping.
    It occupies no bits and no bins, so it takes no column and does not
    advance the fold cursor.
    """
    var n = len(fold_counts)

    # Pass 1: which policy each feature belongs to. A constant feature has no
    # bits at all; it is parked under `POLICY_BINARY` so `policy_of` is total,
    # and `blocks_for` drops it on the `folds > 0` test.
    var policy_of = List[Int]()
    for i in range(n):
        var folds = fold_counts[i]
        if folds < 0:
            raise Error(
                "feature " + String(i) + " has a negative fold count"
            )
        if folds == 0:
            policy_of.append(POLICY_BINARY)
        else:
            policy_of.append(policy_for_fold_count(folds))

    # `one_hot` marks CATEGORICAL features whose splits are EQUALITY
    # tests (their `TBinarizedFeature::OneHotFeature`). Empty means all
    # ordered, which is every caller before categoricals landed. The flag
    # changes NOTHING here -- policies, columns and bins are decided by
    # the fold count exactly as for an ordered feature -- it rides on the
    # CFeature so the scan skips the prefix sum, the split kernels take
    # `==`, and predict takes `==`, all of which already read it.
    if len(one_hot) != 0 and len(one_hot) != n:
        raise Error(
            "one_hot flags must be empty or one per feature: got "
            + String(len(one_hot)) + " for " + String(n)
        )

    # Pass 2: THE walk. Placeholders first so a feature can be written at its
    # input index while the walk runs in policy order.
    var features = List[CFeature]()
    for _ in range(n):
        features.append(CFeature(0, 0, 0, 0, 0, False))

    var next_column = 0
    var fold_cursor = 0

    for policy in range(3):
        var per_int = features_per_int(policy)
        var col = -1
        var used_in_col = 0

        for i in range(n):
            if policy_of[i] != policy:
                continue
            var folds = fold_counts[i]
            if folds == 0:
                # No bits, no bins. It still gets a `first_fold_index` so the
                # field is never garbage, but nothing reads it: `folds == 0`
                # short-circuits the scan, the score kernel and
                # `resolve_split`.
                features[i] = CFeature(0, 0, 0, UInt32(fold_cursor), 0, False)
                continue

            if col < 0 or used_in_col == per_int:
                col = next_column
                used_in_col = 0
                next_column += 1

            var local = used_in_col
            used_in_col = local + 1

            var is_one_hot = False
            if len(one_hot) == n:
                is_one_hot = one_hot[i]
            features[i] = CFeature(
                UInt32(col),
                policy_mask(policy),
                UInt32(policy_shift(policy, local)),
                UInt32(fold_cursor),
                UInt32(folds),
                is_one_hot,
            )
            fold_cursor += folds

    return CompressedIndexLayout(
        features^, policy_of^, next_column, fold_cursor
    )
