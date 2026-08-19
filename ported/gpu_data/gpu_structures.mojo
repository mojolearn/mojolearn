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
struct CFeature(Copyable, Movable):
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
