"""Sibling subtraction and the per-feature bin prefix scan.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
histogram_utils.cu` at CatBoost `54a8143a`. Transliterated. Do not improve.

Two kernels, both bucket-scaling rather than row-scaling, so neither is
where the time goes. They matter because of what they let the histogram
kernel skip.

**Subtraction** is what makes a level cost one child instead of two. The
level builds only the SMALLER child of each pair and derives the larger as
`parent - smaller`, in place, one batched kernel over ALL pairs at once
rather than a kernel per pair.

**The scan** turns each feature's bins into a running prefix along the bin
axis, once per level, in its own kernel. That is why CatBoost's score kernel
can put the BIN-FEATURE on the parallel axis and loop leaves serially inside
a thread: it never has to walk bins in order, because the walking already
happened here.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.sync import barrier


def substract_histograms_kernel(
    from_ids: MutPointer[UInt32, MutAnyOrigin],
    what_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`SubstractHistogramsImpl`, copied.

    Grid: x over bin-features, **y over PAIRS**, z over stats. One launch
    derives every larger sibling in the level.

        newVal = histogram[fromOffset] - histogram[whatOffset]
        if (statId == 0) newVal = max(newVal, 0.0f);

    The `max(., 0)` on stat 0 is theirs and is load-bearing: stat 0 is the
    weight/count plane and float cancellation can drive a derived count
    slightly negative, which would poison a later division. Do NOT lift it to
    the other stats; a gradient sum is legitimately negative.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var bin_feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(
        thread_idx.x
    )
    var from_id = Int(from_ids.unsafe_load(Int(block_idx.y)))
    var what_id = Int(what_ids.unsafe_load(Int(block_idx.y)))
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)

    if bin_feature_id < bin_feature_count:
        var from_offset = (
            from_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        var what_offset = (
            what_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        var new_val = histogram.unsafe_load(bin_feature_id + from_offset) - histogram.unsafe_load(bin_feature_id + what_offset)
        if stat_id == 0:
            new_val = max(new_val, Scalar[DType.float32](0.0))
        histogram.unsafe_store(bin_feature_id + from_offset, new_val)


def scan_histograms_kernel(
    feature_first_bin: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_count_in: Int32,
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`ScanHistogramsImpl`, restructured for a block scan.

    DEVIATION (PORTING.md 8): CatBoost scans with `cub::WarpScan<double>` and
    `cub::ShuffleIndex<32>` (`histogram_utils.cu:381`, `:413`, `:423`). Those
    are the ONLY warp shuffles in the whole oblivious path, and Mojo 1.0 has
    no warp primitives. Substituted with a serial scan by one thread per
    (feature, leaf, stat).

    This substitution is safe for identity and NOT free for speed. A prefix
    sum is order-defined, so a serial scan and a correct parallel scan agree
    exactly in exact arithmetic; in floating point they do NOT, which is why
    the scan is a NUMERIC row and the port uses one shape everywhere rather
    than a fast one per vendor.

    One thread per feature is enough because folds per feature is small (255
    at the very most, 1 for the binary features that dominate covtype), while
    features are many. It is the leaf and stat axes that fill the machine.
    """
    var feature_count = Int(feature_count_in)
    var bin_feature_count = Int(bin_feature_count_in)
    var feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var leaf_id = Int(block_idx.y)
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)

    if feature_id >= feature_count:
        return

    var folds = Int(feature_folds.unsafe_load(feature_id))
    if folds == 0:
        return

    var base = (
        leaf_id * bin_feature_count * stat_count + stat_id * bin_feature_count
    ) + Int(feature_first_bin.unsafe_load(feature_id))

    var running = Scalar[DType.float32](0.0)
    for i in range(folds):
        running += histogram.unsafe_load(base + i)
        histogram.unsafe_store(base + i, running)
