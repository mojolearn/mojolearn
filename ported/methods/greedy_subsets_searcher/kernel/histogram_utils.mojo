"""Sibling subtraction and the per-feature bin prefix scan.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
histogram_utils.cu` at CatBoost `54a8143a`. Transliterated. Do not improve.

Four kernels, all bucket-scaling rather than row-scaling, so neither is
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


def zero_histograms_kernel(
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    dst_histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`ZeroHistogramsImpl`, copied.

    Zeroes the histograms of a NAMED SET of leaves, indexed indirectly
    through `histIds`, rather than a contiguous range. That indirection is
    the point: `build_necessary_histograms` decides which leaves need a fresh
    build and which keep the parent's, so only the `Zeroes` set is cleared
    and the `PreviousPath` set is left intact.

    Grid: x over bin-features, y over the ID LIST, z over stats. So one
    launch clears every leaf that needs clearing, whatever subset that is.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var bin_feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(
        thread_idx.x
    )
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var dst_hist = Int(hist_ids.unsafe_load(Int(block_idx.y)))

    if bin_feature_id < bin_feature_count:
        var base = (
            dst_hist * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        dst_histogram.unsafe_store(base + bin_feature_id, Float32(0.0))


def copy_histograms_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    num_stats_in: Int32,
    bin_features_in_hist_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
):
    """`CopyHistogramsImpl`, copied.

    Duplicates a parent's histogram into the RIGHT child's slot before the
    level splits, which is what makes the in-place subtraction afterwards
    legal: `substract_histograms_kernel` overwrites `from` with
    `from - what`, and `from` has to already hold the parent's totals.

    So the sequence per level is copy, build the smaller child, subtract. Get
    the order wrong and the subtraction reads a stale or zeroed slot and
    produces a plausible wrong histogram rather than an obvious failure.

    Grid y is the PAIR, so one launch copies for every splitting leaf.
    """
    var num_stats = Int(num_stats_in)
    var bin_features_in_hist = Int(bin_features_in_hist_in)
    var left_leaf_id = Int(left_leaves.unsafe_load(Int(block_idx.y)))
    var right_leaf_id = Int(right_leaves.unsafe_load(Int(block_idx.y)))

    var hist_size = bin_features_in_hist * num_stats
    var src = left_leaf_id * hist_size
    var dst = right_leaf_id * hist_size

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < hist_size:
        histograms.unsafe_store(dst + i, histograms.unsafe_load(src + i))
        i += stride


def fixed_to_float_kernel(
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    fixed_scale: Float32,
):
    """Convert the fixed-point accumulator back to the float histogram.

    NO CATBOOST COUNTERPART: they accumulate partials with a float atomic and
    need no conversion. This exists because Metal has no float atomic, so
    replicated blocks sum into Int32 and the result is divided back out here.

    Also zeroes the accumulator, so the next level does not inherit it. A
    separate zeroing launch would cost a kernel for nothing.
    """
    var n = Int(n_cells_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var q = acc_i32.unsafe_load(i)
        if q != Int32(0):
            bin_sums.unsafe_store(i, Float32(Int(q)) / fixed_scale)
            acc_i32.unsafe_store(i, Int32(0))
        i += stride


def write_reduces_histograms_kernel(
    hist_block_offset_in: Int32,
    bin_features_in_block_in: Int32,
    histogram_ids: MutPointer[UInt32, MutAnyOrigin],
    block_histogram: MutPointer[Float32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    dst_histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`WriteReducesHistogramsImpl`, copied.

    **CatBoost keeps TWO histogram layouts and this is the bridge between
    them.** The absence of this kernel is why mixed-width trees in this port
    grew, conserved every row, and refused to split.

        block histogram   [leaf][stat][binFeature WITHIN THIS BLOCK]
        dst histogram     [leaf][stat][binFeature across ALL blocks]

    The histogram kernels write the first: their writeback strides by
    `entriesPerLeaf = statCount * group.GroupSize`, which is THAT BLOCK's
    leaf stride. The score kernel reads the second, whose leaf stride is
    `statCount * binFeatureCount` over every block.

    With ONE policy the two strides coincide and writing straight into the
    flat array is correct, which is why every single-policy check in this
    repository passed. With three policies each block strides by its own
    size and they land on top of each other.

    `histBlockOffset` is where this block's slice begins in the flat array,
    which is the running total of earlier blocks' bin counts.

    Their `histogramIds` indirection is kept: the destination leaf is looked
    up rather than assumed to be `blockIdx.y`, because the caller passes the
    subset of leaves being rebuilt this level.
    """
    var hist_block_offset = Int(hist_block_offset_in)
    var bin_features_in_block = Int(bin_features_in_block_in)
    var bin_feature_count = Int(bin_feature_count_in)

    var bin_feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(
        thread_idx.x
    )
    var leaf_id = Int(block_idx.y)
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var dst_id = Int(histogram_ids.unsafe_load(leaf_id))

    if bin_feature_id < bin_features_in_block:
        var src = (
            leaf_id * bin_features_in_block * stat_count
            + bin_features_in_block * stat_id
            + bin_feature_id
        )
        var val = block_histogram.unsafe_load(src)

        var dst = (
            dst_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
            + hist_block_offset
            + bin_feature_id
        )
        dst_histogram.unsafe_store(dst, val)
