"""A leaf is a contiguous range of the index array.

PORT OF `catboost/cuda/cuda_util/gpu_data/partitions.h` at CatBoost
`54a8143a`. Transliterated. Do not improve.

    struct TDataPartition {
        ui32 Offset;
        ui32 Size;
    };

**That is the entire leaf-membership representation, and it is the single
biggest structural difference from mojotrees.** There is no leaf index array
per row and no per-leaf row list. A row's leaf is decided by WHERE IT SITS in
the index array, so the histogram kernel takes `(offset, size)` and walks a
contiguous span. Membership is positional.

What gets permuted after a split is the `UInt32` index array and the stat
columns. The binned feature matrix is NEVER permuted: it is read indirectly
through `cindex[feature.offset + indices[i]]` and does not move for the life
of the fit.
"""


@fieldwise_init
struct DataPartition(Copyable, Movable):
    """`TDataPartition`. Two `ui32`, nothing else."""

    var offset: UInt32
    var size: UInt32

    @staticmethod
    def empty() -> Self:
        return Self(0, 0)


@fieldwise_init
struct FeatureInBlock(Copyable, Movable):
    """`TFeatureInBlock`, the per-feature descriptor the histogram kernel reads.

    Ported from its use sites in `hist_binary.cu` and
    `compute_hist_loop_two_stats.cuh` rather than from a header, so the field
    set is what those kernels actually touch:

    - `compressed_index_offset`: where this feature's group starts in the
      compressed index.
    - `folds`: how many folds the feature uses. `folds == 0` means the
      feature is absent from this block, and the kernels test it before
      writing anything.
    - `fold_offset_in_group`: where this feature's bins start in the flat
      per-leaf histogram the scorer reads.
    - `bin_count`: `folds`, kept separately because the writeback loop bounds
      on it.
    """

    var compressed_index_offset: UInt32
    var folds: UInt32
    var fold_offset_in_group: UInt32
    var bin_count: UInt32
