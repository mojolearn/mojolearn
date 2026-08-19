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

from ported.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
    features_per_int,
    policy_for_fold_count,
    policy_mask,
    policy_shift,
)
from ported.gpu_data.gpu_structures import CFeature


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


def build_layout(fold_counts: List[Int]) raises -> CompressedIndexLayout:
    """Assign policies and lay out columns and histogram slices.

    Their rule, from `grid_policy.h` plus the builder's grouping: a feature
    goes in the SMALLEST policy whose `MaxFolds` covers it, and features of
    the same policy are packed `FeaturesPerInt` to a column in order.

    Histogram slices are assigned in input order across all policies, so
    `first_fold_index` is a running total and the histogram is one flat array
    the score kernel walks without knowing about policies at all. That is
    what lets `compute_scores` be policy-blind.

    A feature with 0 folds is CONSTANT and is given `folds = 0`, which every
    kernel already tests, rather than being dropped: dropping it would
    renumber every feature after it and break the caller's column mapping.
    """
    var n = len(fold_counts)
    var features = List[CFeature]()
    var policy_of = List[Int]()

    # Column cursors, one per policy, and how many features are already in
    # the current column of each.
    var next_column = 0
    var col_of = List[Int]()
    var used_in_col = List[Int]()
    for _ in range(3):
        col_of.append(-1)
        used_in_col.append(0)

    var fold_cursor = 0

    for i in range(n):
        var folds = fold_counts[i]
        if folds < 0:
            raise Error(
                "feature " + String(i) + " has a negative fold count"
            )
        if folds == 0:
            # Constant feature: no bits, no bins, still occupies its slot.
            features.append(CFeature(0, 0, 0, UInt32(fold_cursor), 0, False))
            policy_of.append(POLICY_BINARY)
            continue

        var policy = policy_for_fold_count(folds)
        var per_int = features_per_int(policy)

        if col_of[policy] < 0 or used_in_col[policy] == per_int:
            col_of[policy] = next_column
            used_in_col[policy] = 0
            next_column += 1

        var local = used_in_col[policy]
        used_in_col[policy] = local + 1

        features.append(
            CFeature(
                UInt32(col_of[policy]),
                policy_mask(policy),
                UInt32(policy_shift(policy, local)),
                UInt32(fold_cursor),
                UInt32(folds),
                False,
            )
        )
        policy_of.append(policy)
        fold_cursor += folds

    return CompressedIndexLayout(
        features^, policy_of^, next_column, fold_cursor
    )
