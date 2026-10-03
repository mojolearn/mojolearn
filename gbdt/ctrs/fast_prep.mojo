# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple simple-CTR prep (lane/apple-fast-sym-ctr, 2026-10-03, family `sym-ctr`).

Every entry point here is reached from `gbdt/train.mojo` only under a
`comptime if` on one of the four flags below, each of which is
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and its
own `-D MOJOLEARN_...` define (or `-D MOJOLEARN_SYM_CTR_ALL`). IDENTICAL and
every non-Apple build compile main's CTR prep unchanged
(`ctrs/ctr_calcers.mojo`, `ctrs/ctr_bins_builder.mojo`); nothing in this file
runs for them.

What each flag changes, against the read-profile in
`docs/apple-fast/notes/sym-ctr.md`:

* `CTR_PREP_SHARED` -- ONE `CtrPrepFast` per fit: the binarized target
  uploaded once, each permutation's CTR estimation order uploaded once, the
  bin builder's and the history calcer's scratch allocated once and reused by
  every (cat feature, permutation). Main's path re-stages and re-uploads the
  order, the codes and the target per (feature, permutation) and allocates
  ~18 row-sized buffers each time. The fresh-order `ComputeCurrentBins` is
  skipped: on an order with no segment flags its end-flag extract is zero
  everywhere but the last slot, the exclusive scan of that is all zeros and
  the scatter writes zeros over the zero fill, so the fill IS its result.
* `CTR_SORT_ONCE` -- the FeatureFreq column (permutation INDEPENDENT) is read
  off permutation 0's Borders builder instead of its own identity-order
  builder: the rows are already sorted by category and a segment's LENGTH
  does not depend on the within-category order, so `ExtractMask` /
  `ScanVector` / `UpdatePartitionOffsets` / `SegmentedReduce` /
  `ComputeWeightedBinFreqCtr` over that order give the same integer counts.
  `ReadLast(Bins) + 2` is `unique_values + 1` (dense codes: every code is
  present, so the segment count is the cardinality), which removes main's
  full readback of `bins` to read one element.
* `CTR_INDEX_FUSED` -- CTR columns stay on the device: the prior divide writes
  each column into its own device buffer, `train` binarizes those buffers in
  place when it builds each permutation's compressed index, and the Borders
  grid takes its min/max from `device_minmax` (the fifteen Float64 Uniform
  borders are then the same arithmetic over the same two floats as
  `ctr_binarization.uniform_borders`).
* `CTR_ONEHOT_DEVICE` -- the dense-code pass for every declared categorical
  column on the device (`device_dense_codes`): validation, the u32 codes,
  the cardinality, the denseness check, the one-hot `maxc` and the
  apply-time CTR table counts (`code_histogram_kernel`, u32 atomics: exact)
  in two launches and two small readbacks instead of per-row host loops.

Bits: FeatureFreq counts and table counts are integers; Borders values are
the same kernels over the same sorted order; borders are the same host
arithmetic over the same min/max. FAST only, so no vendor column moves.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import memcpy
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill
from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    is_equal_up_to_prior_and_binarization,
)
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
    compute_ctr_borders,
)
from gbdt.ctrs.ctr_bins_builder import int_log2
from gbdt.ctrs.index_wrapper import CTR_INDEX_MASK
from gbdt.ctrs.kernel.ctr_calcers import (
    launch_compute_weighted_bin_freq_ctr,
    launch_extract_border_masks,
    launch_fill_binarized_targets_stats,
    launch_gather_trivial_weights,
    launch_make_means_and_scatter,
    launch_update_borders_mask,
)
from gbdt.gpu_util.kernel.partitions import launch_update_partition_offsets
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.scan import SCAN_BLOCK, launch_scan_vector_u32
from gbdt.gpu_util.kernel.segmented_reduce import launch_segmented_reduce_sum
from gbdt.gpu_util.kernel.segmented_scan import (
    SEG_SCAN_BLOCK,
    launch_segmented_scan_and_scatter_non_negative,
)
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_f32,
    launch_gather_with_mask_u32,
    launch_gather_with_mask_u8,
)


comptime _SYM_CTR_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
comptime SYM_CTR_ALL = is_defined["MOJOLEARN_SYM_CTR_ALL"]()

comptime CTR_PREP_SHARED = _SYM_CTR_FAST_APPLE and (
    is_defined["MOJOLEARN_CTR_PREP_SHARED"]() or SYM_CTR_ALL
)
comptime CTR_SORT_ONCE = _SYM_CTR_FAST_APPLE and (
    is_defined["MOJOLEARN_CTR_SORT_ONCE"]() or SYM_CTR_ALL
)
comptime CTR_INDEX_FUSED = _SYM_CTR_FAST_APPLE and (
    is_defined["MOJOLEARN_CTR_INDEX_FUSED"]() or SYM_CTR_ALL
)
comptime CTR_ONEHOT_DEVICE = _SYM_CTR_FAST_APPLE and (
    is_defined["MOJOLEARN_CTR_ONEHOT_DEVICE"]() or SYM_CTR_ALL
)

#: the permutation-DEPENDENT half runs through `CtrPrepFast` under any of
#: these three; `CTR_PREP_SHARED` alone decides whether the context lives
#: for the fit or for one (feature, permutation).
comptime CTR_FAST_PREP = CTR_PREP_SHARED or CTR_SORT_ONCE or CTR_INDEX_FUSED

#: `train`'s fast categorical walk runs under any of the four
comptime SYM_CTR_ANY = CTR_FAST_PREP or CTR_ONEHOT_DEVICE

#: the strided stripe reductions: 16 blocks x 256 threads, each thread owns
#: every `STRIPE_SLOTS`-th row, writes one slot, the host folds the slots
comptime STRIPE_BLOCKS = 16
comptime STRIPE_THREADS = 256
comptime STRIPE_SLOTS = STRIPE_BLOCKS * STRIPE_THREADS

comptime CODES_FLAG_RANGE = UInt32(1)
"""NaN, negative, +inf or at or above 2^32: `dense_category_code`'s first two
refusals."""
comptime CODES_FLAG_INTEGER = UInt32(2)
"""finite, in range, not an exact integer: its third refusal."""

comptime HIST_BLOCK = 256


# --- kernels ---------------------------------------------------------------


def dense_codes_stripe_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    codes: MutPointer[UInt32, MutAnyOrigin],
    slot_max: MutPointer[UInt32, MutAnyOrigin],
    slot_flags: MutPointer[UInt32, MutAnyOrigin],
):
    """`dense_category_code` (`models/ctr_value_table.mojo`) per row, on the
    device: the same three tests in the same order, the code written, the
    running max and the OR of the refusal flags per thread stripe."""
    var n = Int(n_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var mx = UInt32(0)
    var flags = UInt32(0)
    var i = tid
    while i < n:
        var v = src.unsafe_load(i)
        var c = UInt32(0)
        # `v >= 0` is false for NaN, so one test covers "not finite",
        # "negative" and "outside the UInt32 code range" (2^32 is exact
        # in Float32)
        if v >= Float32(0.0) and v < Float32(4294967296.0):
            c = v.cast[DType.uint32]()
            if c.cast[DType.float32]() != v:
                flags |= CODES_FLAG_INTEGER
        else:
            flags |= CODES_FLAG_RANGE
        codes.unsafe_store(i, c)
        if c > mx:
            mx = c
        i += stride
    slot_max.unsafe_store(tid, mx)
    slot_flags.unsafe_store(tid, flags)


def code_histogram_kernel(
    codes: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    target: MutPointer[UInt8, MutAnyOrigin],
    has_target: Int32,
    classes: Int32,
    counts: MutPointer[UInt32, MutAnyOrigin],
    hist: MutPointer[UInt32, MutAnyOrigin],
):
    """`build_ctr_tables`' two host passes as one launch: `++counts[code]`
    and `++hist[code * classes + target_class]`. Integer atomics, so the
    result does not depend on the arrival order."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var c = Int(codes.unsafe_load(i))
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
            counts.unsafe_offset(c), UInt32(1)
        )
        if has_target != Int32(0):
            var cls = Int(target.unsafe_load(i))
            # `build_binarized_target` writes classes below
            # `len(borders) + 1` by construction; the guard keeps a bad
            # caller from writing outside the histogram
            if cls >= Int(classes):
                return
            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                hist.unsafe_offset(c * Int(classes) + cls), UInt32(1)
            )


def minmax_stripe_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    slot_min: MutPointer[Float32, MutAnyOrigin],
    slot_max: MutPointer[Float32, MutAnyOrigin],
    slot_has: MutPointer[UInt32, MutAnyOrigin],
):
    """The min/max pass of `ctr_binarization.uniform_borders`, per thread
    stripe. A CTR column has no NaN, so `<` and `>` are the whole order; min
    and max are order-independent, so the fold order changes no bit."""
    var n = Int(n_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var lo = Float32(0.0)
    var hi = Float32(0.0)
    var has = UInt32(0)
    var i = tid
    while i < n:
        var v = src.unsafe_load(i)
        if has == UInt32(0):
            lo = v
            hi = v
            has = UInt32(1)
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        i += stride
    slot_min.unsafe_store(tid, lo)
    slot_max.unsafe_store(tid, hi)
    slot_has.unsafe_store(tid, has)


# --- the device dense-code pass (CTR_ONEHOT_DEVICE) --------------------------


@fieldwise_init
struct DeviceCodes(Movable):
    """One categorical column's dense codes on the device, with what the
    host needs to know about them."""

    var codes: DeviceBuffer[DType.uint32]
    var unique_values: Int
    var counts: List[Int]
    """rows per code (`build_ctr_tables`' `counts`)"""


def device_dense_codes(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    feature: Int,
) raises -> DeviceCodes:
    """The `cat_features` walk's host loops for one column, on the device:
    `dense_category_code` per row, the cardinality (`maxc + 1`), the
    `CB_ENSURE(uniqueValues > 1)`, the denseness check and the apply
    table's `counts`. Two launches, two small readbacks, one upload of the
    raw column."""
    if n <= 0:
        raise Error("device_dense_codes: empty column")
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    memcpy(dest=h.unsafe_ptr(), src=src, count=n)
    var x = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=x, src_ptr=h.unsafe_ptr())
    var codes = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_max = ctx.enqueue_create_buffer[DType.uint32](STRIPE_SLOTS)
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](STRIPE_SLOTS)
    ctx.enqueue_function[dense_codes_stripe_kernel](
        x.unsafe_ptr(),
        Int32(n),
        codes.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_flags.unsafe_ptr(),
        grid_dim=(STRIPE_BLOCKS, 1, 1),
        block_dim=(STRIPE_THREADS, 1, 1),
    )
    var h_max = ctx.enqueue_create_host_buffer[DType.uint32](STRIPE_SLOTS)
    var h_flags = ctx.enqueue_create_host_buffer[DType.uint32](STRIPE_SLOTS)
    ctx.enqueue_copy(dst_ptr=h_max.unsafe_ptr(), src_buf=d_max)
    ctx.enqueue_copy(dst_ptr=h_flags.unsafe_ptr(), src_buf=d_flags)
    ctx.synchronize()
    _ = h^  # past the drain (step-33 race class)
    _ = x^
    var maxc = UInt32(0)
    var flags = UInt32(0)
    for s in range(STRIPE_SLOTS):
        var m = h_max.unsafe_ptr().unsafe_load(s)
        if m > maxc:
            maxc = m
        flags |= h_flags.unsafe_ptr().unsafe_load(s)
    if (flags & CODES_FLAG_RANGE) != UInt32(0):
        raise Error(
            "categorical feature " + String(feature)
            + " has a value that is not finite, negative or outside the"
            " UInt32 code range (device code pass)"
        )
    if (flags & CODES_FLAG_INTEGER) != UInt32(0):
        raise Error(
            "categorical feature " + String(feature)
            + " must hold exact non-negative integer codes (device code"
            " pass)"
        )
    var unique_values = Int(maxc) + 1
    if unique_values <= 1:
        # their `CB_ENSURE(uniqueValues > 1)`
        # (`batch_binarized_ctr_calcer.cpp:150`)
        raise Error(
            "Error: useless catFeature found (feature "
            + String(feature)
            + " has one category)"
        )
    if unique_values > n:
        # more categories than rows cannot be dense; refuse before sizing
        # a histogram by the largest code
        raise Error(
            "cat_features column " + String(feature)
            + " is not densely coded: its largest code "
            + String(Int(maxc)) + " exceeds the row count"
        )
    if Int(maxc) > Int(CTR_INDEX_MASK):
        raise Error(
            "categorical feature " + String(feature)
            + " has a code above the 0x3FFFFFFF index mask"
        )

    var d_counts = ctx.enqueue_create_buffer[DType.uint32](unique_values)
    enqueue_fill(ctx, d_counts, UInt32(0))
    var d_dummy_hist = ctx.enqueue_create_buffer[DType.uint32](1)
    var d_dummy_target = ctx.enqueue_create_buffer[DType.uint8](1)
    var blocks = (n + HIST_BLOCK - 1) // HIST_BLOCK
    ctx.enqueue_function[code_histogram_kernel](
        codes.unsafe_ptr(),
        Int32(n),
        d_dummy_target.unsafe_ptr(),
        Int32(0),
        Int32(1),
        d_counts.unsafe_ptr(),
        d_dummy_hist.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(HIST_BLOCK, 1, 1),
    )
    var h_counts = ctx.enqueue_create_host_buffer[DType.uint32](unique_values)
    ctx.enqueue_copy(dst_ptr=h_counts.unsafe_ptr(), src_buf=d_counts)
    ctx.synchronize()
    var counts = List[Int](capacity=unique_values)
    for c in range(unique_values):
        var k = Int(h_counts.unsafe_ptr().unsafe_load(c))
        if k == 0:
            raise Error(
                "cat_features column " + String(feature)
                + " is not densely coded: category " + String(c)
                + " is absent from 0.." + String(unique_values - 1)
            )
        counts.append(k)
    _ = d_dummy_hist^  # past the drain
    _ = d_dummy_target^
    return DeviceCodes(codes^, unique_values, counts^)


def device_target_histogram(
    ctx: DeviceContext,
    mut codes: DeviceBuffer[DType.uint32],
    n: Int,
    unique_values: Int,
    mut target: DeviceBuffer[DType.uint8],
    target_classes_count: Int,
) raises -> List[Int]:
    """`build_ctr_tables`' Borders arm (`++hist[code * classes + cls]`) as
    one launch of u32 atomics (exact, order independent) and one readback
    of `unique_values * classes` counts."""
    if target_classes_count < 1:
        raise Error("device_target_histogram: a target needs >= 1 class")
    var hist_len = unique_values * target_classes_count
    var d_counts = ctx.enqueue_create_buffer[DType.uint32](unique_values)
    enqueue_fill(ctx, d_counts, UInt32(0))
    var d_hist = ctx.enqueue_create_buffer[DType.uint32](hist_len)
    enqueue_fill(ctx, d_hist, UInt32(0))
    var blocks = (n + HIST_BLOCK - 1) // HIST_BLOCK
    ctx.enqueue_function[code_histogram_kernel](
        codes.unsafe_ptr(),
        Int32(n),
        target.unsafe_ptr(),
        Int32(1),
        Int32(target_classes_count),
        d_counts.unsafe_ptr(),
        d_hist.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(HIST_BLOCK, 1, 1),
    )
    var h_hist = ctx.enqueue_create_host_buffer[DType.uint32](hist_len)
    ctx.enqueue_copy(dst_ptr=h_hist.unsafe_ptr(), src_buf=d_hist)
    ctx.synchronize()
    var histogram = List[Int](capacity=hist_len)
    for k in range(hist_len):
        histogram.append(Int(h_hist.unsafe_ptr().unsafe_load(k)))
    _ = d_counts^  # past the drain
    return histogram^


def read_codes(
    ctx: DeviceContext, mut codes: DeviceBuffer[DType.uint32], n: Int
) raises -> List[UInt32]:
    """The host copy of a device code column, for a caller that still runs
    main's host drivers over it."""
    var h = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=codes)
    ctx.synchronize()
    var out = List[UInt32]()
    out.resize(n, UInt32(0))
    memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    return out^


def upload_codes(
    ctx: DeviceContext, codes: List[UInt32]
) raises -> DeviceBuffer[DType.uint32]:
    """Host codes (main's `dense_category_code` loop) onto the device once
    per feature, for the fast drivers when `CTR_ONEHOT_DEVICE` is off."""
    var n = len(codes)
    var h = ctx.enqueue_create_host_buffer[DType.uint32](n)
    memcpy(dest=h.unsafe_ptr(), src=codes.unsafe_ptr(), count=n)
    var d = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^  # past the drain (step-33 race class)
    return d^


# --- the device min/max and the Uniform grid (CTR_INDEX_FUSED) ----------------


def device_minmax(
    ctx: DeviceContext, mut col: DeviceBuffer[DType.float32], n: Int
) raises -> Tuple[Float32, Float32]:
    """min and max of a NaN-free device column: one stripe launch, one
    `STRIPE_SLOTS`-sized readback."""
    if n <= 0:
        raise Error("device_minmax: empty column")
    var d_min = ctx.enqueue_create_buffer[DType.float32](STRIPE_SLOTS)
    var d_max = ctx.enqueue_create_buffer[DType.float32](STRIPE_SLOTS)
    var d_has = ctx.enqueue_create_buffer[DType.uint32](STRIPE_SLOTS)
    ctx.enqueue_function[minmax_stripe_kernel](
        col.unsafe_ptr(),
        Int32(n),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_has.unsafe_ptr(),
        grid_dim=(STRIPE_BLOCKS, 1, 1),
        block_dim=(STRIPE_THREADS, 1, 1),
    )
    var h_min = ctx.enqueue_create_host_buffer[DType.float32](STRIPE_SLOTS)
    var h_max = ctx.enqueue_create_host_buffer[DType.float32](STRIPE_SLOTS)
    var h_has = ctx.enqueue_create_host_buffer[DType.uint32](STRIPE_SLOTS)
    ctx.enqueue_copy(dst_ptr=h_min.unsafe_ptr(), src_buf=d_min)
    ctx.enqueue_copy(dst_ptr=h_max.unsafe_ptr(), src_buf=d_max)
    ctx.enqueue_copy(dst_ptr=h_has.unsafe_ptr(), src_buf=d_has)
    ctx.synchronize()
    var lo = Float32(0.0)
    var hi = Float32(0.0)
    var seen = False
    for s in range(STRIPE_SLOTS):
        if h_has.unsafe_ptr().unsafe_load(s) == UInt32(0):
            continue
        var a = h_min.unsafe_ptr().unsafe_load(s)
        var b = h_max.unsafe_ptr().unsafe_load(s)
        if not seen:
            lo = a
            hi = b
            seen = True
        else:
            if a < lo:
                lo = a
            if b > hi:
                hi = b
    return (lo, hi)


def uniform_ctr_borders_from_minmax(
    min_value: Float32, max_value: Float32, max_borders_count: Int
) -> List[Float32]:
    """`compute_ctr_borders` for the Uniform grid, from the column's min and
    max: `uniform_borders`' arithmetic (`ctr_binarization.mojo`: Float64
    `lo + (i + 1) * (hi - lo) / (count + 1)`, narrowed to Float32, monotone
    duplicates collapsed) plus its caller's constant-feature hack (an empty
    list becomes the single border 0.5). Same expression, same two
    operands, same borders."""
    var out = List[Float32]()
    if min_value != max_value:
        var lo = Float64(min_value)
        var hi = Float64(max_value)
        for i in range(max_borders_count):
            var current_value = lo + Float64(i + 1) * (hi - lo) / Float64(
                max_borders_count + 1
            )
            var b = Float32(current_value)
            if len(out) == 0 or out[len(out) - 1] != b:
                out.append(b)
    if len(out) == 0:
        out.append(Float32(0.5))
    return out^


def ctr_borders_from_device(
    ctx: DeviceContext,
    mut col: DeviceBuffer[DType.float32],
    n: Int,
    description: TBinarizationOptions,
) raises -> List[Float32]:
    """`compute_ctr_borders` over a CTR column that lives on the device:
    the Uniform grid from `device_minmax` (no row readback); any other
    grid reads the column back and runs the host builder unchanged."""
    if description.border_selection_type == BORDER_SELECTION_UNIFORM:
        var mm = device_minmax(ctx, col, n)
        return uniform_ctr_borders_from_minmax(
            mm[0], mm[1], description.border_count
        )
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=col)
    ctx.synchronize()
    var values = List[Float32]()
    values.resize(n, Float32(0.0))
    memcpy(dest=values.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    return compute_ctr_borders(values, description)


# --- the shared prep context (CTR_PREP_SHARED, CTR_SORT_ONCE, CTR_INDEX_FUSED) --


def _read_column(
    ctx: DeviceContext,
    mut host: HostBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    n: Int,
) raises -> List[Float32]:
    """One device CTR column back to a host list (the staging buffer is the
    prep's, allocated once)."""
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
    ctx.synchronize()
    var col = List[Float32]()
    col.resize(n, Float32(0.0))
    memcpy(dest=col.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    return col^


@fieldwise_init
struct FastCtrColumns(Movable):
    """What one fast driver call produced: host columns (in config order,
    indexed by ORIGINAL row) and, under `device_out`, the same columns as
    device buffers."""

    var host: List[List[Float32]]
    var dev: List[DeviceBuffer[DType.float32]]


struct CtrPrepFast(Movable):
    """`TCtrBinBuilderGpu` + `THistoryBasedCtrCalcerGpu` +
    `TWeightedBinFreqCalcerGpu` with their uploads and scratch hoisted out of
    the per-(feature, permutation) call. Same launches, in the same order,
    on the same buffers' roles; see the file header for what is skipped and
    why it is the same bits."""

    var n: Int
    var perm_count: Int
    var target: DeviceBuffer[DType.uint8]
    var has_target: Bool
    var orders: List[DeviceBuffer[DType.uint32]]
    var order_ready: List[Bool]
    var scratch_ready: Bool
    # the bin builder's buffers (`ctr_bins_builder.h:253-259`)
    var indices: DeviceBuffer[DType.uint32]
    var bins: DeviceBuffer[DType.uint32]
    var current_bins: DeviceBuffer[DType.uint32]
    var temp_bins: DeviceBuffer[DType.uint32]
    var tmp_u32: DeviceBuffer[DType.uint32]
    var scan_block_sums: DeviceBuffer[DType.uint32]
    var sort_offsets: DeviceBuffer[DType.int32]
    var sort_block_sums: DeviceBuffer[DType.int32]
    # the history calcer's buffers (`ctr_calcers.h:265-283`)
    var gathered_weights: DeviceBuffer[DType.float32]
    var scanned_weights: DeviceBuffer[DType.float32]
    var stats: DeviceBuffer[DType.float32]
    var scanned_stats: DeviceBuffer[DType.float32]
    var gathered_sample: DeviceBuffer[DType.uint8]
    var seg_scanned: DeviceBuffer[DType.float32]
    var seg_has_flag: DeviceBuffer[DType.uint8]
    var seg_block_sums: DeviceBuffer[DType.float32]
    var seg_block_flags: DeviceBuffer[DType.uint8]
    # the freq calcer's trivial weights
    var ones: DeviceBuffer[DType.float32]
    var host_f32: HostBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, n: Int, perm_count: Int) raises:
        if n <= 0:
            raise Error("CtrPrepFast: empty dataset")
        if n > Int(CTR_INDEX_MASK):
            raise Error(
                "CtrPrepFast: " + String(n)
                + " rows do not fit the 0x3FFFFFFF index mask"
            )
        self.n = n
        self.perm_count = perm_count
        self.has_target = False
        self.scratch_ready = False
        self.target = ctx.enqueue_create_buffer[DType.uint8](1)
        self.orders = List[DeviceBuffer[DType.uint32]]()
        self.order_ready = List[Bool]()
        for _ in range(perm_count):
            self.orders.append(ctx.enqueue_create_buffer[DType.uint32](1))
            self.order_ready.append(False)
        # placeholders; `ensure_scratch` allocates the row-sized buffers
        # the first time a builder runs
        self.indices = ctx.enqueue_create_buffer[DType.uint32](1)
        self.bins = ctx.enqueue_create_buffer[DType.uint32](1)
        self.current_bins = ctx.enqueue_create_buffer[DType.uint32](1)
        self.temp_bins = ctx.enqueue_create_buffer[DType.uint32](1)
        self.tmp_u32 = ctx.enqueue_create_buffer[DType.uint32](1)
        self.scan_block_sums = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_offsets = ctx.enqueue_create_buffer[DType.int32](1)
        self.sort_block_sums = ctx.enqueue_create_buffer[DType.int32](1)
        self.gathered_weights = ctx.enqueue_create_buffer[DType.float32](1)
        self.scanned_weights = ctx.enqueue_create_buffer[DType.float32](1)
        self.stats = ctx.enqueue_create_buffer[DType.float32](1)
        self.scanned_stats = ctx.enqueue_create_buffer[DType.float32](1)
        self.gathered_sample = ctx.enqueue_create_buffer[DType.uint8](1)
        self.seg_scanned = ctx.enqueue_create_buffer[DType.float32](1)
        self.seg_has_flag = ctx.enqueue_create_buffer[DType.uint8](1)
        self.seg_block_sums = ctx.enqueue_create_buffer[DType.float32](1)
        self.seg_block_flags = ctx.enqueue_create_buffer[DType.uint8](1)
        self.ones = ctx.enqueue_create_buffer[DType.float32](1)
        self.host_f32 = ctx.enqueue_create_host_buffer[DType.float32](n)

    def ensure_scratch(mut self, ctx: DeviceContext) raises:
        if self.scratch_ready:
            return
        var n = self.n
        self.indices = ctx.enqueue_create_buffer[DType.uint32](n)
        self.bins = ctx.enqueue_create_buffer[DType.uint32](n)
        self.current_bins = ctx.enqueue_create_buffer[DType.uint32](n)
        self.temp_bins = ctx.enqueue_create_buffer[DType.uint32](n)
        self.tmp_u32 = ctx.enqueue_create_buffer[DType.uint32](n)
        self.scan_block_sums = ctx.enqueue_create_buffer[DType.uint32](
            (n + SCAN_BLOCK - 1) // SCAN_BLOCK
        )
        # 512 is `REORDER_BLOCK`, the radix pass's block
        self.sort_offsets = ctx.enqueue_create_buffer[DType.int32](n)
        self.sort_block_sums = ctx.enqueue_create_buffer[DType.int32](
            (n + 512 - 1) // 512
        )
        self.gathered_weights = ctx.enqueue_create_buffer[DType.float32](n)
        self.scanned_weights = ctx.enqueue_create_buffer[DType.float32](n)
        self.stats = ctx.enqueue_create_buffer[DType.float32](n)
        self.scanned_stats = ctx.enqueue_create_buffer[DType.float32](n)
        self.gathered_sample = ctx.enqueue_create_buffer[DType.uint8](n)
        var n_blocks = (n + SEG_SCAN_BLOCK - 1) // SEG_SCAN_BLOCK
        self.seg_scanned = ctx.enqueue_create_buffer[DType.float32](n)
        self.seg_has_flag = ctx.enqueue_create_buffer[DType.uint8](n)
        self.seg_block_sums = ctx.enqueue_create_buffer[DType.float32](
            n_blocks
        )
        self.seg_block_flags = ctx.enqueue_create_buffer[DType.uint8](
            n_blocks
        )
        self.ones = ctx.enqueue_create_buffer[DType.float32](n)
        enqueue_fill(ctx, self.ones, Float32(1.0))
        self.scratch_ready = True

    def set_target(
        mut self, ctx: DeviceContext, binarized_target: List[UInt8]
    ) raises:
        """`SetBinarizedSample`, ONCE per fit (main uploads the same bytes
        per feature and permutation)."""
        if len(binarized_target) != self.n:
            raise Error(
                "binarized target has " + String(len(binarized_target))
                + " entries for " + String(self.n) + " rows"
            )
        var h = ctx.enqueue_create_host_buffer[DType.uint8](self.n)
        memcpy(
            dest=h.unsafe_ptr(), src=binarized_target.unsafe_ptr(),
            count=self.n,
        )
        self.target = ctx.enqueue_create_buffer[DType.uint8](self.n)
        ctx.enqueue_copy(dst_buf=self.target, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        _ = h^  # past the drain (step-33 race class)
        self.has_target = True

    def ensure_order(
        mut self, ctx: DeviceContext, p: Int, order: List[UInt32]
    ) raises:
        """`ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)`
        (`doc_parallel_dataset_builder.cpp:255`), ONCE per permutation. The
        order carries row ids below `n`, and `n` fits the index mask (checked
        at construction), so the per-entry flag-bit check of
        `TCtrBinBuilderGpu.__init__` is implied."""
        if p < 0 or p >= self.perm_count:
            raise Error("CtrPrepFast: permutation " + String(p) + " out of range")
        if self.order_ready[p]:
            return
        if len(order) != self.n:
            raise Error(
                "ctr estimation order has " + String(len(order))
                + " entries for " + String(self.n) + " rows"
            )
        var h = ctx.enqueue_create_host_buffer[DType.uint32](self.n)
        memcpy(dest=h.unsafe_ptr(), src=order.unsafe_ptr(), count=self.n)
        self.orders[p] = ctx.enqueue_create_buffer[DType.uint32](self.n)
        ctx.enqueue_copy(dst_buf=self.orders[p], src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        _ = h^  # past the drain (step-33 race class)
        self.order_ready[p] = True

    def build_bins(
        mut self,
        ctx: DeviceContext,
        p: Int,
        mut codes: DeviceBuffer[DType.uint32],
        unique_values: Int,
    ) raises:
        """`TCtrBinBuilderGpu(order)` + `add_cat_feature_bins(codes)` on the
        shared scratch, for permutation `p`'s order:

            indices <- order                      (device copy)
            CurrentBins <- 0                      (= ComputeCurrentBins on a
                                                   flag-free order)
            GatherWithMask(Bins, codes, Indices, Mask)
            ReorderBins(Bins, Indices, 0, IntLog2(uniqueValues), Tmp, TempBins)
            UpdateBordersMask(Bins, CurrentBins, Indices)
        """
        if not self.order_ready[p]:
            raise Error("CtrPrepFast: order " + String(p) + " not uploaded")
        if unique_values <= 1:
            raise Error("Error: useless catFeature found")
        self.ensure_scratch(ctx)
        var n = self.n
        ctx.enqueue_copy(dst_buf=self.indices, src_buf=self.orders[p])
        enqueue_fill(ctx, self.current_bins, UInt32(0))
        launch_gather_with_mask_u32(
            ctx, self.bins, codes, self.indices, n, CTR_INDEX_MASK
        )
        launch_radix_sort_bins(
            ctx,
            n,
            0,
            int_log2(unique_values),
            self.bins,
            self.indices,
            self.tmp_u32,
            self.temp_bins,
            self.sort_offsets,
            self.sort_block_sums,
        )
        launch_update_borders_mask(
            ctx, self.bins, self.current_bins, self.indices, n
        )

    def borders_ctrs(
        mut self,
        ctx: DeviceContext,
        group: List[TCtrConfig],
        device_out: Bool,
    ) raises -> FastCtrColumns:
        """`THistoryBasedCtrCalcerGpu.reset` + `visit_cat_feature_ctr` for one
        equal-up-to-prior group over the bins `build_bins` just built: the
        same seven launches, the prior divide written per config into its
        own buffer (`device_out`) or into `stats` and read back."""
        if len(group) == 0:
            raise Error("borders_ctrs called with no configs")
        if not self.has_target:
            raise Error(
                "borders_ctrs needs the binarized target (set_target)"
            )
        ref reference = group[0]
        if reference.ctr_type != CTR_BORDERS and (
            reference.ctr_type != CTR_BUCKETS
        ):
            raise Error("borders_ctrs takes Borders or Buckets configs only")
        var n = self.n
        # `Reset`: trivial weights, negated at a segment start, then the
        # exclusive segmented scan -> the denominators per ORIGINAL row
        launch_gather_trivial_weights(
            ctx, self.indices, n, UInt32(n), True, self.gathered_weights
        )
        launch_segmented_scan_and_scatter_non_negative(
            ctx, n, False, self.gathered_weights, self.indices,
            self.scanned_weights, self.seg_scanned, self.seg_has_flag,
            self.seg_block_sums, self.seg_block_flags,
        )
        # `GetGatheredBinSample`, `FillBinarizedTargetsStats`, the scan
        launch_gather_with_mask_u8(
            ctx, self.gathered_sample, self.target, self.indices, n,
            CTR_INDEX_MASK,
        )
        launch_fill_binarized_targets_stats(
            ctx, self.gathered_sample, self.gathered_weights, n, self.stats,
            UInt32(reference.param_id), reference.ctr_type == CTR_BORDERS,
        )
        launch_segmented_scan_and_scatter_non_negative(
            ctx, n, False, self.stats, self.indices, self.scanned_stats,
            self.seg_scanned, self.seg_has_flag, self.seg_block_sums,
            self.seg_block_flags,
        )
        var out = FastCtrColumns(List[List[Float32]](), List[DeviceBuffer[DType.float32]]())
        for c in range(len(group)):
            ref config = group[c]
            if not is_equal_up_to_prior_and_binarization(config, reference):
                raise Error(
                    "borders_ctrs: every config in one call must be equal"
                    " up to prior and binarization"
                )
            if device_out:
                var dst = ctx.enqueue_create_buffer[DType.float32](n)
                launch_make_means_and_scatter(
                    ctx, self.scanned_stats, self.scanned_weights, n,
                    config.numerator_shift(), config.denumerator_shift(),
                    self.indices, False, CTR_INDEX_MASK, dst,
                )
                out.dev.append(dst^)
            else:
                launch_make_means_and_scatter(
                    ctx, self.scanned_stats, self.scanned_weights, n,
                    config.numerator_shift(), config.denumerator_shift(),
                    self.indices, False, CTR_INDEX_MASK, self.stats,
                )
                out.host.append(_read_column(ctx, self.host_f32, self.stats, self.n))
        return out^

    def freq_ctrs(
        mut self,
        ctx: DeviceContext,
        unique_values: Int,
        group: List[TCtrConfig],
        device_out: Bool,
    ) raises -> FastCtrColumns:
        """`TWeightedBinFreqCalcerGpu.visit_equal_up_to_prior_freq_ctrs` over
        the bins `build_bins` just built (CTR_SORT_ONCE): the segment count is
        `unique_values` (dense codes), so `ReadLast(Bins) + 2` is
        `unique_values + 1` and the `bins` readback goes. The host column is
        always produced (the MinEntropy grid is a host build); `device_out`
        adds the device copy for the fused compressed index."""
        if len(group) == 0:
            raise Error("freq_ctrs called with no configs")
        var n = self.n
        launch_extract_border_masks(ctx, self.indices, self.tmp_u32, n, False)
        launch_scan_vector_u32(
            ctx, n, False, self.tmp_u32, self.temp_bins, self.scan_block_sums
        )
        var bin_count_with_fake = unique_values + 1
        var segment_starts = ctx.enqueue_create_buffer[DType.uint32](
            bin_count_with_fake
        )
        launch_update_partition_offsets(
            ctx, segment_starts, bin_count_with_fake, self.temp_bins, n
        )
        var bin_weights = ctx.enqueue_create_buffer[DType.float32](
            bin_count_with_fake - 1
        )
        launch_gather_with_mask_f32(
            ctx, self.gathered_weights, self.ones, self.indices, n,
            CTR_INDEX_MASK,
        )
        launch_segmented_reduce_sum(
            ctx, self.gathered_weights, segment_starts, bin_weights,
            bin_count_with_fake - 1,
        )
        var out = FastCtrColumns(List[List[Float32]](), List[DeviceBuffer[DType.float32]]())
        for c in range(len(group)):
            ref config = group[c]
            if config.ctr_type != CTR_FEATURE_FREQ:
                raise Error(
                    "freq_ctrs takes FeatureFreq configs only; got "
                    + String(config.ctr_type)
                )
            if device_out:
                var dst = ctx.enqueue_create_buffer[DType.float32](n)
                launch_compute_weighted_bin_freq_ctr(
                    ctx, self.indices, True, self.temp_bins, bin_weights,
                    Float32(n), config.numerator_shift(),
                    config.denumerator_shift(), dst, n,
                )
                out.host.append(_read_column(ctx, self.host_f32, dst, self.n))
                out.dev.append(dst^)
            else:
                launch_compute_weighted_bin_freq_ctr(
                    ctx, self.indices, True, self.temp_bins, bin_weights,
                    Float32(n), config.numerator_shift(),
                    config.denumerator_shift(), self.stats, n,
                )
                out.host.append(_read_column(ctx, self.host_f32, self.stats, self.n))
        _ = segment_starts^  # past the drains (step-33 race class)
        _ = bin_weights^
        return out^


def group_configs(configs: List[TCtrConfig]) raises -> List[List[Int]]:
    """`CreateEqualUpToPriorAndBinarizationCtrsGroupping` (`ctr.h:60-68`):
    config slots bucketed by (Type, ParamId), in first-seen order."""
    var done = List[Bool]()
    for _ in range(len(configs)):
        done.append(False)
    var groups = List[List[Int]]()
    for i in range(len(configs)):
        if done[i]:
            continue
        var slots = List[Int]()
        for j in range(i, len(configs)):
            if not done[j] and is_equal_up_to_prior_and_binarization(
                configs[j], configs[i]
            ):
                done[j] = True
                slots.append(j)
        groups.append(slots^)
    return groups^


def fast_dependent_ctrs(
    ctx: DeviceContext,
    mut prep: CtrPrepFast,
    p: Int,
    mut codes: DeviceBuffer[DType.uint32],
    unique_values: Int,
    configs: List[TCtrConfig],
    device_out: Bool,
) raises -> FastCtrColumns:
    """`compute_simple_ctrs_gpu` for permutation `p` on the shared prep:
    `build_bins`, then every equal-up-to-prior group through `borders_ctrs`.
    Columns come back in config order."""
    for i in range(len(configs)):
        if configs[i].ctr_type == CTR_FEATURE_FREQ:
            raise Error(
                "fast_dependent_ctrs is the permutation-DEPENDENT writer;"
                " FeatureFreq belongs to fast_freq_ctrs"
            )
    prep.build_bins(ctx, p, codes, unique_values)
    var out = FastCtrColumns(List[List[Float32]](), List[DeviceBuffer[DType.float32]]())
    for _ in range(len(configs)):
        out.host.append(List[Float32]())
    var dev_slot = List[Int]()
    for _ in range(len(configs)):
        dev_slot.append(-1)
    var groups = group_configs(configs)
    var pending = List[DeviceBuffer[DType.float32]]()
    for g in range(len(groups)):
        var group = List[TCtrConfig]()
        for k in range(len(groups[g])):
            group.append(configs[groups[g][k]])
        var cols = prep.borders_ctrs(ctx, group, device_out)
        for k in range(len(groups[g])):
            if device_out:
                dev_slot[groups[g][k]] = len(pending)
                pending.append(cols.dev[k].copy())
            else:
                out.host[groups[g][k]] = cols.host[k].copy()
    if device_out:
        for i in range(len(configs)):
            out.dev.append(pending[dev_slot[i]].copy())
    return out^


def fast_freq_ctrs(
    ctx: DeviceContext,
    mut prep: CtrPrepFast,
    unique_values: Int,
    configs: List[TCtrConfig],
    device_out: Bool,
) raises -> FastCtrColumns:
    """`compute_simple_ctrs_device` for the FeatureFreq configs, over the
    bins the LAST `build_bins` left in the prep (CTR_SORT_ONCE). Host columns
    always; device copies under `device_out`."""
    var out = FastCtrColumns(List[List[Float32]](), List[DeviceBuffer[DType.float32]]())
    for _ in range(len(configs)):
        out.host.append(List[Float32]())
    var dev_slot = List[Int]()
    for _ in range(len(configs)):
        dev_slot.append(-1)
    var groups = group_configs(configs)
    var pending = List[DeviceBuffer[DType.float32]]()
    for g in range(len(groups)):
        var group = List[TCtrConfig]()
        for k in range(len(groups[g])):
            group.append(configs[groups[g][k]])
        var cols = prep.freq_ctrs(ctx, unique_values, group, device_out)
        for k in range(len(groups[g])):
            out.host[groups[g][k]] = cols.host[k].copy()
            if device_out:
                dev_slot[groups[g][k]] = len(pending)
                pending.append(cols.dev[k].copy())
    if device_out:
        for i in range(len(configs)):
            out.dev.append(pending[dev_slot[i]].copy())
    return out^
