"""The BINARY driver: 32 one-bit features per block.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_binary.cu` at CatBoost
`54a8143a`. Transliterated. Do not improve.

`ComputeSplitPropertiesBImpl`. It builds its histogram with
`TPointHistHalfByte` -- the same accumulator the half-byte kernel uses, not a
binary-specific one -- and the only thing that makes it a binary kernel is
the WRITEBACK.

WHY THAT WORKS, and it is the neatest trick in this family: thirty-two
one-bit features pack into eight nibbles, four features to a nibble. The
accumulator builds a 16-bin histogram over each nibble's VALUE, knowing
nothing about bits. A feature's "bit is 0" side is then recovered by adding
the eight nibble values whose bit is clear -- `pw_hb_binary_sum`. So one
accumulator serves both kernels and the binary case costs a sum instead of a
lookup.

A BINARY FEATURE HAS ONE BORDER, so the writeback lands ONE value per
(feature, stat) at `FirstFoldIndex * 2 + w` with no `+ fold` term. That is
the visible difference from the half-byte kernel's writeback, and the reason
these are two files here as they are upstream.

NO BIT-WIDTH DISPATCH. Unlike the one-byte kernels this one has no
`GetMaxBinCount` bounds test: every feature it is given is binary by
construction, and the host decides which features reach it.

DEVIATION (arch): their `use64bitLoad = IsFullPass` above compute
capability 3.5, and this port takes that modern arm -- `ComputeHistogram2`
on a full pass, the scalar loop with inner and outer unroll 1 on a partial
one. Scheduling, not numeric.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import abs
from std.memory import stack_allocation
from std.atomic import Atomic, Ordering
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.kernel.compute_point_hist2_loop import (
    compute_histogram,
    compute_histogram_2,
)
from gbdt.methods.kernel.split_properties_helpers import (
    shift_part_and_bin_sums_ptr,
)
from gbdt.methods.kernel.pointwise_hist2_half_byte_template import (
    PW_HB_BLOCK,
    PW_HB_SMEM_FLOATS,
    PointHistHalfByte,
    pw_hb_binary_sum,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_templ import PW_WRITE_EPS


#: `feature += (blockIdx.x / M) * 32` (`:31`).
comptime PW_B_FEATURES_PER_BLOCK = 32


def compute_split_properties_b_kernel[
    full_pass: Bool, m: Int
](
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_first_fold_index: MutPointer[UInt32, MutAnyOrigin],
    f_count_in: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    partition: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    total_feature_count_in: Int32,
):
    """`ComputeSplitPropertiesBImpl` (`:23-94`), copied.

    `feature_folds` is absent from the signature and that is not an
    omission: a binary feature has one fold, so their writeback never reads
    `Folds` here (compare the half-byte kernel, which bounds its fold loop
    on it).
    """
    var tid = Int(thread_idx.x)
    var total_feature_count = Int(total_feature_count_in)

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

    var f_base = (Int(block_idx.x) // m) * PW_B_FEATURES_PER_BLOCK
    var f_count = Int(f_count_in) - f_base
    if f_count > PW_B_FEATURES_PER_BLOCK:
        f_count = PW_B_FEATURES_PER_BLOCK
    if f_count <= 0:
        return
    var cindex_base = Int(feature_offset.unsafe_load(f_base))

    var smem = stack_allocation[
        PW_HB_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()

    var part_size = Int(partition.unsafe_load(2 * part + 1))
    var part_offset = Int(partition.unsafe_load(2 * part))
    if part_size == 0:
        return

    var ci = cindex.unsafe_offset(cindex_base)
    var hist = PointHistHalfByte(smem)

    comptime if full_pass:
        compute_histogram_2[PW_HB_BLOCK, 1, 1, m](
            hist, indices, UInt32(part_offset), UInt32(part_size),
            target, weight, ci,
        )
    else:
        compute_histogram[PW_HB_BLOCK, 1, 1, 1, m](
            hist, indices, UInt32(part_offset), UInt32(part_size),
            target, weight, ci,
        )

    # their writeback (`:69-91`). `Reduce` ends in a barrier, so theirs has
    # no extra sync here and neither does this.
    var w = tid & 1
    var fid = tid >> 1
    if fid < f_count:
        var acc = pw_hb_binary_sum(smem, fid, w)
        if abs(acc) > PW_WRITE_EPS:
            var at = (
                bin_sums_base
                + Int(feature_first_fold_index.unsafe_load(f_base + fid))
                * 2
                + w
            )
            comptime if m > 1:
                # DEVIATION 1898: upstream's atomicAdd is relaxed; the non-Apple
                # Mojo default is seq_cst.
                _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                    bin_sums.unsafe_offset(at), acc
                )
            else:
                bin_sums.unsafe_store(at, acc)
