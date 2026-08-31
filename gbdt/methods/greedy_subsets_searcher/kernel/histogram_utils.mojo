# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
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

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import floor
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from original.numerics import ftz


@always_inline
def hist2_dither(pos: Int) -> Float32:
    """A deterministic hash of the DOCUMENT POSITION, uniform in [0, 1).

    The dither for `hist2_quantize`. Keyed on the row's position so it is
    deterministic run to run and vendor to vendor (it depends on the data
    layout, never on the launch geometry), and so the same row quantizes
    IDENTICALLY at every tree depth, which keeps parent = child + sibling
    exact to the integer."""
    var h = UInt32(pos) * UInt32(2654435761)
    h ^= h >> 16
    h = h * UInt32(2246822519)
    h ^= h >> 13
    return Float32(Int(h >> 8)) * Float32(5.9604645e-08)  # / 2^24


@always_inline
def hist2_quantize(val: Float32, fixed_scale: Float32, u: Float32) -> Int32:
    """DETERMINISTIC DITHERED QUANTIZATION, `floor(val*scale + u(pos))` --
    and every word of that is load-bearing, because both simpler rules
    FAILED MEASURABLY:

      TRUNCATION (`Int32(val * scale)`) loses up to `1/scale` per row,
      always toward zero, so a leaf's histogram total ran ~0.16% short of
      the EXACT `partStats` total that `compute_scores` subtracts it from
      (`sumRight = partStat - sumLeft`, `compute_scores.cu:97-99`). The
      deficit surfaces as phantom mass past every bin, largest at the last
      one; measured at 300k-650k rows x 100 features, the fit locked onto
      (feature 98, bin 127) on three DIFFERENT datasets and stopped at
      depth 1.

      ROUND-TO-NEAREST is still biased whenever a stat VALUE repeats (a
      unit weight plane repeats it 800,000 times): every 1.0 became
      537/536.87, measured +52 phantom weight per 217k rows.

    `E[floor(x + U)] = x` exactly for uniform U in [0,1), for any value
    distribution, so the dithered cell error is zero-mean and grows as
    sqrt(rows), not rows. An integer-valued `val * scale` is unchanged by
    any u < 1, which is what keeps `hist2_check`'s exactness argument
    intact. The +1 worst case per row is accounted for EXACTLY by
    `choose_scale`'s `row_count` allowance (its blanket headroom covers
    callers that state none). Called ONCE per (row, stat) at load time -- computing it
    at every add cost a measured quarter of the tree.

    WRITTEN AS `floor(x)` PLUS A FRACTION COMPARE, not `floor(x + u)`: the
    one-expression form ROUNDS in the addition before `floor` sees it, so an
    integral `x` with `u` within half a Float32 spacing of 1 crossed the
    integer boundary -- 14 cells of `check-hist2`'s exact fixture moved on
    the first run. `x - floor(x)` is exact in Float32 (the operands are
    within a factor of two), the compare never manufactures a crossing, and
    an integral `x` has fraction 0, which no `u < 1` can lift to 1."""
    # IDENTITY_PATHS ROW 10: a DENORMAL `val` diverges here between
    # Metal's FTZ hardware (operand flushed, `scaled` = 0) and a
    # denormal-honoring backend (`scaled` can be a real fraction --
    # `choose_scale` can legally return 2^100-class scales when the plane
    # magnitude is tiny -- and the dither compare then increments `q` on
    # one vendor and not the other). Operand and result flushed through
    # `ftz`, aligning every backend to Metal's measured model; comptime
    # no-ops under FAST, where Apple's hardware did both flushes already.
    var scaled = ftz(ftz(val) * fixed_scale)
    var base = floor(scaled)
    var q = Int32(base)
    if ftz(scaled - base) + u >= Float32(1.0):
        q += Int32(1)
    return q


def hist2_smem_add[
    dt: DType
](
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    offset: Int,
    val: Float32,
    qval: Int32,
):
    """THE one factored shared-memory accumulation site of the hist_2 family.

    NO CATBOOST COUNTERPART for the Int32 arm. Every `AddPointsImpl` in
    `hist_2_one_byte_{5,6,7}bit.cu` lands its stat with a plain float add
    into a warp-PRIVATE slice; that is the `DType.float32` arm here,
    verbatim. The `DType.int32` arm is the `HIST_SMEM_SHARED2_I32` matrix
    row (`original/kernel_matrix.hist_smem_mode_for`): the slice is shared
    by TWO warps, so the add must be atomic, and Metal has no local float
    atomics ("Unsupported local float atomic operation", scratchpad
    `histshare_probe.mojo`), so it is a local Int32 atomic in fixed point.
    `qval` is the row's stat through `hist2_quantize`, computed once per
    (row, stat) at load time; the float arm ignores it, the Int32 arm adds
    it and ignores `val`.

    ================= DEVIATION BLOCK =================
    Theirs: `Histogram[offset] += stat;` -- private slice, plain float add.
    Ours (Apple / bit-identical column only): `Atomic.fetch_add` of the
    row's `hist2_quantize`d stat into a 2-warp-shared slice. Measured on the
    M4 (100M hashed scatter-adds, interleaved, `histshare_probe.mojo`):
    warp-private float 32 KB = 46.5 G upd/s, 2-warp-shared Int32 16 KB =
    90.2 G upd/s -- 1.94x. NVIDIA/AMD columns compile the float arm and are
    untouched. The `fixed_scale` behind `qval` must satisfy
    `original/fixed_point.mojo`'s bound (`choose_scale` of the plane's sum
    of |values|); every cell is a partial sum over a subset of all rows, so
    it stays under 2^28 - 1 plus one dither unit per row and Int32 cannot
    overflow.
    ===================================================

    Everything AROUND this call -- pass structure, bin decoding, slot
    arithmetic, sync phases -- is theirs in both modes; only where the add
    lands differs, which is why this is one function instead of a branch in
    each accumulator.
    """

    @parameter
    if dt == DType.int32:
        # DEVIATION 1898: upstream's atomicAdd is relaxed; the non-Apple Mojo
        # default is seq_cst.
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
            smem.unsafe_offset(offset), rebind[Scalar[dt]](qval)
        )
    else:
        smem[offset] = smem[offset] + rebind[Scalar[dt]](val)


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
        # IDENTITY_PATHS ROW 10: parent minus child is a cancellation, so
        # the derived sibling can land denormal; flushed at the store
        # (kernel-to-kernel seam -- the score kernel reads this cell), so
        # the histogram buffer never holds a denormal on any backend.
        # Comptime no-op under FAST.
        var new_val = ftz(
            histogram.unsafe_load(bin_feature_id + from_offset)
            - histogram.unsafe_load(bin_feature_id + what_offset)
        )
        if stat_id == 0:
            new_val = max(new_val, Scalar[DType.float32](0.0))
        histogram.unsafe_store(bin_feature_id + from_offset, new_val)


def scan_histograms_kernel(
    hist_ids: MutPointer[UInt32, MutAnyOrigin],
    feature_first_bin: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_one_hot: MutPointer[UInt8, MutAnyOrigin],
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

    ONE-HOT FEATURES ARE SKIPPED, and that is theirs, not a choice
    (`histogram_utils.cu:395-397`):

        const bool skipFeature = features->OneHotFeature || (features->Folds <= 1);
        if (!skipFeature) { ... }

    A one-hot feature's bins are EQUALITY tests, not thresholds, so the
    score kernel reads a bin's own count and a running prefix over them
    means nothing. The `Folds <= 1` half is theirs too and is free: a
    single-bin prefix sum is the identity, so running it only costs a
    read-modify-write.
    """
    var feature_count = Int(feature_count_in)
    var bin_feature_count = Int(bin_feature_count_in)
    var feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    # Their `ids` argument, `*leavesGpu` at `split_properties_helper.cpp:1265`.
    # The scan runs over the leaves that were just COMPUTED, not over leaf
    # slots 0..n. Indexing `block_idx.y` straight into the histogram was our
    # departure and it made a partial build impossible to scan correctly.
    var leaf_id = Int(hist_ids.unsafe_load(Int(block_idx.y)))
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)

    if feature_id >= feature_count:
        return

    # their `skipFeature` (`histogram_utils.cu:395`), both halves.
    var folds = Int(feature_folds.unsafe_load(feature_id))
    if feature_one_hot.unsafe_load(feature_id) != UInt8(0) or folds <= 1:
        return

    var base = (
        leaf_id * bin_feature_count * stat_count + stat_id * bin_feature_count
    ) + Int(feature_first_bin.unsafe_load(feature_id))

    var running = Scalar[DType.float32](0.0)
    for i in range(folds):
        # IDENTITY_PATHS ROW 10: the running prefix is both an
        # intermediate and a stored seam (the score kernel reads every
        # cell), flushed so a denormal crossing cannot split the vendors.
        # Comptime no-op under FAST.
        running = ftz(running + histogram.unsafe_load(base + i))
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

    **THIS IS THE FLAT HISTOGRAM'S ZERO AND ONLY THAT.** Its address
    arithmetic is `[leaf][stat][binFeature]` with a leaf stride of
    `binFeatureCount * statCount` over EVERY block. The per-block scratch has
    a leaf stride of `statCount * GroupSize` and a `deviceOffset` in front of
    it, so this kernel clears the scratch's cells only while `GroupOffset` is
    zero and `GroupSize` is the whole block, which is the single-device case
    and nothing wider. CatBoost never points it there at all: the scratch
    gets `ZeroBuffer` (`zero_buffer_kernel`,
    `split_properties_helper.cpp:1157`) and the flat histogram gets this one.
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


def zero_histogram_kernel(
    dst_hist_in: UInt32,
    bin_feature_count_in: Int32,
    dst_histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`ZeroHistogramImpl`, copied (`histogram_utils.cu:251-266`).

    The SINGULAR form of `zero_histograms_kernel`. Same addressing, same
    `WriteThrough(dst + binFeatureId, 0.0f)`, one difference: the destination
    leaf arrives BY VALUE instead of through a `histIds[blockIdx.y]` load,
    and the grid is `numBlocks.y = 1` (`histogram_utils.cu:268-284`).

    **Why they keep both.** `ZeroLeavesHistograms` dispatches on the count
    (`split_properties_helper.cpp:1375-1400`): at two leaves or fewer it
    launches this one per leaf, above two it writes a mirror id buffer and
    launches the plural once. The small case exists to avoid the host-side
    `ids.Write(leaves)` and its copy, which cost more than the extra launch
    when there are one or two ids. This is the same dispatch they use for
    `SubstractHistograms` (`:1402`) and for the whole compute path
    (`:1099-1105`).
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var bin_feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(
        thread_idx.x
    )
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var dst_hist = Int(dst_hist_in)

    if bin_feature_id < bin_feature_count:
        var base = (
            dst_hist * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        dst_histogram.unsafe_store(base + bin_feature_id, Float32(0.0))


def zero_buffer_kernel(
    buffer: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
):
    """`TZeroBuffer::Run`, copied (`split_properties_helper.cpp:34-38`).

        ui64 size = Buffer.Size() * sizeof(T);
        cudaMemsetAsync(Buffer.Get(), 0, size, stream.GetStream());

    A WHOLE-BUFFER zero, and that is the entire point of it. It is the OTHER
    zeroing CatBoost does, and it is not interchangeable with
    `zero_histograms_kernel`:

        FillBuffer(subsets.Histograms, 0.0f)   the FINAL FLAT histogram,
                                               `[leaf][stat][binFeature]`
                                               across every block; the
                                               indexed form of it is
                                               `ZeroHistogramsImpl`
                                               (`:1061`)
        ZeroBuffer(blockHistograms, streamId)  the PER-BLOCK SCRATCH, which
                                               one block owns and strides by
                                               its own `GroupSize`
                                               (`:1155-1157`)

    Line 1155 of their file is `FillBuffer(blockHistograms, 0.0f, streamId)`
    COMMENTED OUT and replaced by `ZeroBuffer` on line 1157, so this is the
    call they settled on. It is layout agnostic because it clears every byte,
    which is what the scratch needs: the histogram writeback addresses it as
    `deviceOffset + leafId * statCount * GroupSize + statId * GroupSize +
    FoldOffsetInGroup`, and `deviceOffset` is `GroupOffset * statCount *
    leafCount`.

    **`GroupOffset` and `GroupSize` are per DEVICE, not per feature group,
    and that is easy to read backwards.** They come off
    `groupOffsetsBeforeReduce[devAfterReduce]` and
    `binFeatureCountAfterReduceOnDevices[devAfterReduce]`
    (`compute_by_blocks_helper.cpp:273-274`), a running total over DEVICES of
    each device's after-reduce bin-feature count. On one device that is
    `GroupOffset = 0` and `GroupSize` = the whole block's bin-feature count,
    so `deviceOffset` vanishes and the scratch is plain
    `[leaf][stat][binInBlock]` -- which is exactly what
    `write_reduces_histograms_kernel` reads. The feature-group axis of the
    grid, `blockIdx.x / maxBlocksPerPart`, does NOT appear in this address at
    all.

    Size it with their `BlockHistogramsMapping(blockId, histCount, statCount)`
    (`compute_by_blocks_helper.h:74-78`):

        blockSize.Size() * histCount * statCount

    where `blockSize.Size()` is this block's bin-feature count.

    DEVIATION (PORTING_RULES 4): theirs is `cudaMemsetAsync`, a driver call.
    Mojo has no exposed async memset on this toolchain, so the memset is a
    grid-stride store loop. Same bytes, same value, different way of saying
    it.
    """
    var n = Int(n_cells_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        buffer.unsafe_store(i, Float32(0.0))
        i += stride


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


def copy_histograms_vec4_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    num_stats_in: Int32,
    bin_features_in_hist_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
):
    """`copy_histograms_kernel` moving 16 bytes per thread instead of 4.

    ================= DEVIATION BLOCK =================
    Their `CopyHistogramsImpl` (`histogram_utils.cu:15-34`) copies ONE
    float per thread, and this port transliterated that. On NVIDIA that
    is free: `__ldg` plus `WriteThrough` (`st.global.wt`) already move a
    full sector per warp. On this Metal box it is not -- MEASURED, at a
    depth-6 level's shape (100 features x 254 folds x 2 stats, 32 pairs):

        one float per thread   11.0 GB/s
        four per thread        65.2 GB/s

    Same bytes, same grid shape, 5.9x. The copy is a pure `memcpy` between
    two slots of one buffer, so widening the access changes nothing about
    WHICH bytes land WHERE, only how many a thread carries.

    ALIGNMENT is why this is a separate kernel rather than an edit to
    theirs: a 4-wide load needs the slot base to be 16-byte aligned, which
    holds only when `histSize` is a multiple of 4. The launcher checks and
    falls back to the scalar kernel otherwise, so no dataset silently gets
    an unaligned access.

    GPU-AGNOSTIC: a 4-wide float load is the one vector width every target
    in the matrix has (NVIDIA `float4`, AMD `float4`, Metal `float4`). No
    lane width, no shared memory, no intrinsic is assumed.
    ===================================================
    """
    var num_stats = Int(num_stats_in)
    var bin_features_in_hist = Int(bin_features_in_hist_in)
    var left_leaf_id = Int(left_leaves.unsafe_load(Int(block_idx.y)))
    var right_leaf_id = Int(right_leaves.unsafe_load(Int(block_idx.y)))

    var hist_size = bin_features_in_hist * num_stats
    var src = left_leaf_id * hist_size
    var dst = right_leaf_id * hist_size
    var n4 = hist_size >> 2

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n4:
        var off = i << 2
        (histograms + dst).store[width=4](
            off, (histograms + src).load[width=4](off)
        )
        i += stride


def substract_histograms_vec4_kernel(
    from_ids: MutPointer[UInt32, MutAnyOrigin],
    what_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`substract_histograms_kernel` at four bin-features per thread.

    ================= DEVIATION BLOCK =================
    Same measured reason as `copy_histograms_vec4_kernel`, same 4-wide
    portable width, and the same launcher-side alignment guard: the stat
    plane base is `stat_id * bin_feature_count`, so a 4-wide access is
    aligned only when `bin_feature_count % 4 == 0`.

    THE `max(., 0)` ON STAT 0 IS PRESERVED EXACTLY and is still applied
    per lane, not per vector -- it is their guard against a derived count
    going slightly negative through float cancellation
    (`SubstractHistogramsImpl`, `histogram_utils.cu`), and widening the
    access must not widen the predicate.
    ===================================================
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var from_id = Int(from_ids.unsafe_load(Int(block_idx.y)))
    var what_id = Int(what_ids.unsafe_load(Int(block_idx.y)))
    var stat_id = Int(block_idx.z)
    var stat_count = Int(grid_dim.z)
    var n4 = bin_feature_count >> 2

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n4:
        var from_offset = (
            from_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        var what_offset = (
            what_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
        )
        var off = i << 2
        var a = (histogram + from_offset).load[width=4](off)
        var b = (histogram + what_offset).load[width=4](off)
        var v = a - b
        # IDENTITY_PATHS ROW 10, per lane exactly as the `max` predicate
        # is per lane: same flushed-store seam as the scalar kernel.
        # Comptime no-op under FAST.
        comptime for e in range(4):
            v[e] = ftz(v[e])
        if stat_id == 0:
            v = max(v, SIMD[DType.float32, 4](0.0))
        (histogram + from_offset).store[width=4](off, v)


def fixed_to_float_kernel(
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Convert the fixed-point accumulator back to the float histogram.

    NO CATBOOST COUNTERPART: they accumulate partials with a float atomic and
    need no conversion. This exists for `NUMERIC_IDENTICAL`, which sums
    replicated blocks into Int32 so the order the blocks land in cannot
    change the answer; the result is divided back out here. `NUMERIC_FAST`
    takes CatBoost's float atomic and never fills the accumulator, so every
    cell reads zero and this kernel stores nothing.

    Also zeroes the accumulator, so the next level does not inherit it. A
    separate zeroing launch would cost a kernel for nothing.
    """
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var n = Int(n_cells_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var q = acc_i32.unsafe_load(i)
        if q != Int32(0):
            # IDENTITY_PATHS ROW 10: the dequantized quotient can land
            # denormal when `fixed_scale` is huge; flushed at the store so
            # the histogram never holds a denormal. Comptime no-op under
            # FAST.
            bin_sums.unsafe_store(i, ftz(Float32(Int(q)) / fixed_scale))
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


def write_reduces_from_fixed_kernel[
    read_scratch: Bool
](
    hist_block_offset_in: Int32,
    bin_features_in_block_in: Int32,
    histogram_ids: MutPointer[UInt32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    block_histogram: MutPointer[Float32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    dst_histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`WriteReducesHistogramsImpl` with `fixed_to_float_kernel` folded in.

    ================= DEVIATION BLOCK =================
    TWO parents, one pass. The IDENTICAL/shared-Int32 arms used to run

        fixed_to_float_kernel   acc_i32 -> block_hist   (ours, no CatBoost
                                counterpart: their accumulation is float)
        write_reduces_histograms_kernel   block_hist -> flat histogram
                                (theirs, `WriteReducesHistogramsImpl`)

    back to back over the SAME cells -- between them sits only their
    inter-device `ReduceScatter`, a no-op on one device
    (`reduce_scatter.h:460-462`). That is one full extra read+write of
    every histogram cell per level, PURE cell-proportional cost with no
    CatBoost counterpart, and the cell count doubles at 254 borders. This
    kernel is the writeback reading the accumulator directly: the
    conversion is the SAME expression at the same cell
    (`Float32(Int(q)) / fixed_scale`), so the flat histogram receives
    identical bits and the float block scratch is never touched.

    TWO SOURCES PER CELL, DISJOINT BY LEAF, because that is what the
    writeback kernels leave behind (`hist_2_one_byte_base.mojo`'s
    `hist2_add_to_global_memory` and the one-byte/8-bit twins): a leaf
    whose rows span MULTIPLE thread blocks drains through Int32 atomics
    into the ACCUMULATOR; a single-block leaf stores its dequantized
    float STRAIGHT into the block scratch and never touches the
    accumulator. So: a nonzero accumulator cell is converted (and
    zeroed); a zero one falls through to the block scratch, which holds
    either the single-block store or the level's memset zero. The first
    version of this kernel read the accumulator alone and zeroed every
    single-block leaf's histogram -- check-hist2's cross-family section
    failed 799 of 1600 cells on the spot, which is exactly the catch it
    exists for. What the fusion still deletes is the bridge's separate
    full read-convert-write pass and its launch; the accumulator zeroing
    rides along as before, only where the cell was nonzero.
    ===================================================
    """
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
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
        var q = acc_i32.unsafe_load(src)
        var val = Float32(0.0)
        if q != Int32(0):
            # IDENTITY_PATHS ROW 10: same flushed dequantization as
            # `fixed_to_float_kernel` -- the flat histogram is the seam
            # every scorer reads. Comptime no-op under FAST.
            val = ftz(Float32(Int(q)) / fixed_scale)
            acc_i32.unsafe_store(src, Int32(0))
        else:
            # `read_scratch` is False for a ONE-BYTE block, whose i32
            # writebacks now put single-block cells in the accumulator
            # too, so the scratch holds nothing and a zero accumulator
            # cell means a zero histogram cell. It stays True for the
            # binary and half-byte families, whose single-block stores
            # are exact floats in the scratch.
            @parameter
            if read_scratch:
                val = block_histogram.unsafe_load(src)

        var dst = (
            dst_id * bin_feature_count * stat_count
            + stat_id * bin_feature_count
            + hist_block_offset
            + bin_feature_id
        )
        dst_histogram.unsafe_store(dst, val)


def choose_scale_kernel(
    mags: MutPointer[Float32, MutAnyOrigin],
    row_count: Int32,
    scale_out: MutPointer[Float32, MutAnyOrigin],
):
    """`choose_scale` on the device, bit-for-bit, so the boosting loop
    never drains for the magnitudes (the last per-tree drain besides the
    tree's own).

    The host function's snap is `largest 2^k with mag * 2^k <= limit`,
    every operation exact -- so with `mag = S * 2^e2` read straight from
    the float's bits, the whole derivation is the integer search for the
    largest `t = e2 + k` with `S << t <= limit`. `S < 2^24 <= limit`
    guarantees `t >= 0`, and `limit < 2^30` with `S >= 1` bounds it by
    30, so the search is a short walk down from 30. The final
    `2^k` is built by exact doubling/halving, which saturates to inf
    past 2^127 and walks the subnormals below 2^-126 exactly as the
    host's `Float32(...)` cast of the f64 power does.

    One thread does everything: this is control plane, not compute.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var w = mags.unsafe_load(0)
    var g = mags.unsafe_load(1)
    var m = w
    if g > m:
        m = g
    if m == Float32(0.0):
        scale_out.unsafe_store(0, Float32(1.0))
        return
    # the host's `max((1 << 30) - 1 - row_count, SCALE_LIMIT)`
    var limit = Int64((1 << 30) - 1) - Int64(Int(row_count))
    var floor_limit = Int64((1 << 28) - 1)
    if limit < floor_limit:
        limit = floor_limit
    # |m| = S * 2^e2, exactly, from the bits
    var mbits = UInt32(m.to_bits())
    var exp_field = Int((mbits >> 23) & UInt32(0xFF))
    var mant = Int64(Int(mbits & UInt32(0x7FFFFF)))
    var s_int: Int64
    var e2: Int
    if exp_field == 0:
        s_int = mant
        e2 = -149
    else:
        s_int = mant + Int64(1 << 23)
        e2 = exp_field - 150
    var t = 30
    while (s_int << Int64(t)) > limit:
        t -= 1
    var k = t - e2
    var scale = Float32(1.0)
    if k >= 0:
        for _ in range(k):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-k):
            scale = scale * Float32(0.5)
    scale_out.unsafe_store(0, scale)
