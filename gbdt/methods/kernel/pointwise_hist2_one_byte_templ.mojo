"""The one-byte DRIVER: which accumulator runs, on which features, and where
the result goes.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_one_byte_templ.cuh` at
CatBoost `54a8143a`. Transliterated. Do not improve.

The six accumulators know how to add a point. This file is everything else:

    1. slice the feature array          `feature += (blockIdx.x / M) * 4`
    2. pick the partition and the
       histogram slot                   `ShiftPartAndBinSumsPtr`
    3. REFUSE if this block's features
       do not belong to this bit width  `GetMaxBinCount` + the bounds
    4. choose the load width            `TLoadEntriesTrait`
    5. run the loop
    6. write 4 x folds x 2 floats out   atomicAdd or a plain store

STEP 3 IS THE ONE THAT LOOKS WRONG AND IS NOT. Every bit width's kernel is
launched over EVERY non-binary feature block, and each one returns
immediately unless the block's widest feature falls in its range
(`:180-184`):

    upperBound = 1 << BITS
    lowerBound = BITS > 5 ? upperBound / 2 : 15
    if (maxBinCount <= lowerBound || maxBinCount > upperBound) return;

so the four kernels partition the blocks between them at RUNTIME, on data the
host does not consult. Note `lowerBound` is 15 and not 16 at 5 bits, so a
feature with exactly 16 folds is handled by the 5-bit kernel; and note that a
block is claimed by its WIDEST feature, so four features of 3, 5, 40 and 200
folds all go to the 8-bit kernel together.

DEVIATION (arch): their `TLoadEntriesTrait` and `TUnrollsTrait` are
`__CUDA_ARCH__` ladders. This port takes the modern arm of each, the same
choice `hist_2_one_byte_base.mojo` records for the other family:

    5, 6, 7 bit     FourElements, outer unroll 1, both passes
    8 bit           TwoElements on a full pass, OneElement on a partial one,
                    which is theirs at every arch and not an arch choice

Scheduling, not numeric: the same points are added in the same per-lane
order, and `mojo_only/pointwise_hist2_check.mojo` A5 holds all four widths
identical across all four load widths.

DEVIATION 93's consequence, and it is why this file has a `comptime if`
CatBoost does not need: the 8-bit accumulator holds Int32 fixed point, so its
scratch is `Int32` and its writeback divides by the scale. The other three
hold float and write straight through. Everything else -- the slicing, the
bounds, the loop choice, the guard -- is shared, which is the point of
keeping them in one function rather than two.

`WriteThrough` (`:141`) is `__stwt`, a store with a cache-eviction hint. It
becomes a plain store here: Mojo exposes no non-temporal store, and the hint
changes no value.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import abs
from std.memory import stack_allocation
from std.atomic import Atomic, Ordering
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    pointwise_one_byte_fixed_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from gbdt.methods.kernel.compute_point_hist2_loop import (
    compute_histogram,
    compute_histogram_2,
    compute_histogram_4,
)
from gbdt.methods.kernel.split_properties_helpers import (
    shift_part_and_bin_sums_ptr,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_FLOAT_BLOCK,
    PW_HIST2_FLOAT_SMEM_FLOATS,
    PW_HIST2_SMEM_FLOATS,
    PointHist5,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_6bit import PointHist6
from gbdt.methods.kernel.pointwise_hist2_one_byte_7bit import PointHist7
from gbdt.methods.kernel.pointwise_hist2_one_byte_8bit import PointHist8


#: `abs(val) > 1e-20f` (`:139`). Not an epsilon on a comparison -- it is a
#: SKIP on a write, so a cell that rounds to nothing costs no global traffic.
comptime PW_WRITE_EPS = Float32(1e-20)

#: Features per block for the non-binary kernels (`:165`).
comptime PW_NB_FEATURES_PER_BLOCK = 4


@always_inline
def pass_inner_bits[bits: Int]() -> Int:
    """`TDeclarePassInnerOuterBitsTrait<BITS - 5>::Inner()`.

        5 -> <0> 0    6 -> <1> 1    7 -> <2> 2    8 -> <3> 1
    """
    comptime if bits == 5:
        return 0
    elif bits == 6:
        return 1
    elif bits == 7:
        return 2
    else:
        return 1


@always_inline
def pass_outer_bits[bits: Int]() -> Int:
    """`TDeclarePassInnerOuterBitsTrait<BITS - 5>::Outer()`.

        5 -> 0    6 -> 0    7 -> 0    8 -> 2

    The 8-bit entry is the odd one and it is what makes that accumulator a
    different design: a non-zero OUTER count quadruples the warp slice, so
    fewer slices fit and warps have to share them. See
    `pointwise_hist2_one_byte_8bit.mojo`.
    """
    comptime if bits == 8:
        return 2
    else:
        return 0


@always_inline
def pw_max_fold_count[bits: Int]() -> Int:
    """`1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT)` (`:124`).

        5 -> 32    6 -> 64    7 -> 128    8 -> 256

    which is also the number of folds each accumulator's `Reduce` lays out,
    so the writeback below and the accumulators cannot disagree without one
    of them changing this function.
    """
    return 1 << (5 + pass_inner_bits[bits]() + pass_outer_bits[bits]())


@always_inline
def get_max_bin_count[
    origin: MutOrigin, //
](
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    f_count: Int,
    smem: MutPointer[UInt32, origin, address_space = AddressSpace.SHARED],
) -> UInt32:
    """`GetMaxBinCount` (`split_properties_helpers.cuh:25-45`), copied.

    HELD BACK FROM `split_properties_helpers.mojo` UNTIL NOW, deliberately,
    because it reduces over exactly FOUR shared slots while every thread in
    the block writes one -- so it is only correct when the caller puts at
    most four features in a block, and reading it without its call site
    tells you nothing about that. `ComputeSplitPropertiesNBImpl` is the call
    site: `fCount = min(fCount - (blockIdx.x / M) * 4, 4)`.

    Their reduction is a two-step max over indices 0..3:

        if (threadIdx.x < 2) smem[tid] = max(smem[tid], smem[tid + 2]);
        if (threadIdx.x < 1) smem[tid] = max(smem[tid], smem[tid + 1]);

    Every `__syncthreads()` is OUTSIDE the `if`, which is what keeps it
    legal under a threadgroup barrier (PORTING.md 11).

    It borrows the histogram scratch, which is why it must run and finish
    BEFORE the accumulator is constructed -- the constructor zeroes that
    same memory.
    """
    var tid = Int(thread_idx.x)
    var bin_count = UInt32(0)
    if tid < f_count:
        bin_count = feature_folds.unsafe_load(tid)
    smem.unsafe_store(tid, bin_count)
    barrier()

    if tid < 2:
        var a = smem.unsafe_load(tid)
        var b = smem.unsafe_load(tid + 2)
        smem.unsafe_store(tid, a if a > b else b)
    barrier()
    if tid < 1:
        var a = smem.unsafe_load(tid)
        var b = smem.unsafe_load(tid + 1)
        smem.unsafe_store(tid, a if a > b else b)
    barrier()
    var result = smem.unsafe_load(0)
    barrier()
    return result


@always_inline
def pw_bounds[bits: Int]() -> Tuple[Int, Int]:
    """`lowerBound` and `upperBound` (`:180-181`).

        upperBound = 1 << BITS
        lowerBound = BITS > 5 ? upperBound / 2 : 15

    NOTE THE 15. At 5 bits the lower bound is 15 and not 16, so a feature
    with exactly 16 folds belongs to the 5-bit kernel -- the one whose
    accumulator holds 32 bins. Writing `upperBound / 2` uniformly would move
    16-fold features to the 6-bit kernel and leave nothing to handle them at
    5, because both tests are strict on one side.
    """
    comptime upper = 1 << bits
    # the Apple/IDENTICAL routing row widens the 8-bit kernel to accept
    # every one-byte width; see `pointwise_one_byte_fixed_for`
    comptime if bits == 8 and pointwise_one_byte_fixed_for[
        TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    ]():
        return (15, upper)
    comptime lower = (upper // 2) if bits > 5 else 15
    return (lower, upper)


def compute_split_properties_nb_kernel[
    bits: Int, full_pass: Bool, m: Int
](
    # `TCFeature*`, flattened to parallel arrays (PORTING.md 9 rule 2).
    # `Mask` and `Shift` are absent because this family never reads them:
    # a one-byte feature IS a byte of the word `Offset` selects.
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_first_fold_index: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    f_count_in: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    # `TDataPartition*` as `{Offset, Size}` pairs of UInt32
    partition: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    total_feature_count_in: Int32,
    fixed_scale: Float32,
):
    """`ComputeSplitPropertiesNBImpl` (`:153-187`) with
    `ComputeSplitPropertiesPass` (`:64-147`) inlined into it.

    THEIRS ARE TWO FUNCTIONS AND THIS IS ONE, which is a deviation of shape
    and not of content: their `ComputeSplitPropertiesPass` exists so that
    the `DECLARE_PASS` macro can pass `TDeclarePassInnerOuterBitsTrait`'s
    two values as template arguments. Here `bits` already carries that, and
    splitting would mean handing a shared-memory pointer and an accumulator
    across a boundary for no gain.
    """
    comptime max_fold_count = pw_max_fold_count[bits]()
    comptime bounds = pw_bounds[bits]()
    comptime is_fixed = bits == 8

    var tid = Int(thread_idx.x)
    var total_feature_count = Int(total_feature_count_in)

    # --- their `ShiftPartAndBinSumsPtr` (`:167`) ---------------------
    var shifted = shift_part_and_bin_sums_ptr(
        partition,
        UInt32(grid_dim.z),
        UInt32(block_idx.y),
        UInt32(block_idx.z),
        UInt32(grid_dim.y),
        UInt32(total_feature_count),
        full_pass,
        2,
    )
    var part = Int(shifted.partition_offset)
    var bin_sums_base = Int(shifted.bin_sums_offset)

    # --- their feature slicing (`:169-171`) --------------------------
    var f_base = (Int(block_idx.x) // m) * PW_NB_FEATURES_PER_BLOCK
    var f_count = Int(f_count_in) - f_base
    if f_count > PW_NB_FEATURES_PER_BLOCK:
        f_count = PW_NB_FEATURES_PER_BLOCK
    if f_count <= 0:
        return
    var cindex_base = Int(feature_offset.unsafe_load(f_base))

    # ONE allocation, read through a `bitcast` at each use. The 8-bit
    # accumulator holds Int32 and the other three hold Float32 (DEVIATION
    # 93); both are 4 bytes, so within one bit width only the
    # interpretation changes. THE SIZE AND BLOCK ARE PER BIT WIDTH: the
    # 8-bit accumulator takes the route-keyed geometry
    # (`pw_hist2_block_size_for`), the float turn-taking accumulators
    # ALWAYS take their dispatch's -- their slice offsets do not wrap, so
    # only the 8-bit accumulator may ever run at a block its own row
    # widens. The two currently resolve to the same value (the doubled
    # fixed-route block was measured a no-op and reverted; the negative
    # is on the row), and this split is what makes any future divergence
    # land here instead of in a kernel.
    comptime nb_smem = PW_HIST2_SMEM_FLOATS if bits == 8 else (
        PW_HIST2_FLOAT_SMEM_FLOATS
    )
    comptime nb_block = PW_HIST2_BLOCK if bits == 8 else (
        PW_HIST2_FLOAT_BLOCK
    )
    var smem = stack_allocation[
        nb_smem,
        Float32,
        address_space = AddressSpace.SHARED,
    ]()

    # --- their `GetMaxBinCount` on the histogram scratch (`:174`) -----
    # BEFORE the accumulator is built: its constructor zeroes this memory.
    var max_bin_count = get_max_bin_count(
        feature_folds.unsafe_offset(f_base),
        f_count,
        smem.unsafe_bitcast[UInt32](),
    )
    barrier()

    # --- their runtime partition of blocks between bit widths (`:183`) -
    if (
        Int(max_bin_count) <= bounds[0]
        or Int(max_bin_count) > bounds[1]
    ):
        return

    var part_size = Int(partition.unsafe_load(2 * part + 1))
    var part_offset = Int(partition.unsafe_load(2 * part))
    if part_size == 0:
        return

    var ci = cindex.unsafe_offset(cindex_base)

    comptime if is_fixed:
        var hist = PointHist8(smem.unsafe_bitcast[Int32](), fixed_scale)
        comptime if full_pass:
            # `TLoadEntriesTrait<3, true>`: TwoElements
            compute_histogram_2[nb_block, 1, 1, m](
                hist, indices, UInt32(part_offset), UInt32(part_size),
                target, weight, ci,
            )
        else:
            # `TLoadEntriesTrait<3, false>`: OneElement, outer unroll 2
            compute_histogram[nb_block, 2, 1, 1, m](
                hist, indices, UInt32(part_offset), UInt32(part_size),
                target, weight, ci,
            )
    elif bits == 5:
        var hist = PointHist5(smem.unsafe_bitcast[Float32]())
        compute_histogram_4[nb_block, 1, 1, m](
            hist, indices, UInt32(part_offset), UInt32(part_size),
            target, weight, ci,
        )
    elif bits == 6:
        var hist = PointHist6(smem.unsafe_bitcast[Float32]())
        compute_histogram_4[nb_block, 1, 1, m](
            hist, indices, UInt32(part_offset), UInt32(part_size),
            target, weight, ci,
        )
    else:
        var hist = PointHist7(smem.unsafe_bitcast[Float32]())
        compute_histogram_4[nb_block, 1, 1, m](
            hist, indices, UInt32(part_offset), UInt32(part_size),
            target, weight, ci,
        )

    barrier()

    # --- their writeback (`:122-146`) --------------------------------
    # `fid = threadIdx.x / 64` and `fold` strides by 32 from
    # `(threadIdx.x / 2) & 31`, so the 256 participating threads cover
    # 4 features x 32 folds at a time and loop for wider ones.
    var fid = tid // 64
    var w = tid & 1
    var feature_folds_here = 0
    if fid < f_count:
        feature_folds_here = Int(
            feature_folds.unsafe_load(f_base + fid)
        )
    var feature_offset_in_smem = fid * max_fold_count * 2 + w

    var fold = (tid // 2) & 31
    while fold < feature_folds_here:
        if fid < f_count:
            var val: Float32
            comptime if is_fixed:
                # DEVIATION 93: this accumulator's cells are fixed point,
                # so the scale comes out HERE and not inside `Reduce` --
                # dividing before the sibling subtraction would round twice
                # and break `parent == child + sibling`.
                val = Float32(
                    Int(
                        smem.unsafe_bitcast[Int32]().unsafe_load(
                            feature_offset_in_smem + 2 * fold
                        )
                    )
                ) / fixed_scale
            else:
                val = smem.unsafe_bitcast[Float32]().unsafe_load(
                    feature_offset_in_smem + 2 * fold
                )

            if abs(val) > PW_WRITE_EPS:
                var at = (
                    bin_sums_base
                    + (
                        Int(
                            feature_first_fold_index.unsafe_load(
                                f_base + fid
                            )
                        )
                        + fold
                    )
                    * 2
                    + w
                )
                comptime if m > 1:
                    # several document blocks per feature, so the writes
                    # collide; theirs is a global float atomicAdd
                    # DEVIATION 1898: upstream's atomicAdd is relaxed; the non-
                    # Apple Mojo default is seq_cst.
                    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                        bin_sums.unsafe_offset(at), val
                    )
                else:
                    # `WriteThrough`, a store with an eviction hint that
                    # Mojo does not expose. A plain store moves the same
                    # value.
                    bin_sums.unsafe_store(at, val)
        fold += 32
