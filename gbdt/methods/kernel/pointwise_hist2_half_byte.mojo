"""The HALF-BYTE driver: 8 features of up to 16 bins per block.

PORT OF `catboost/cuda/methods/kernel/pointwise_hist2_half_byte.cu` at
CatBoost `54a8143a`. Transliterated. Do not improve.

`ComputeSplitPropertiesHalfByteImpl`. Same accumulator as the binary kernel
(`TPointHistHalfByte`), same loop choice, and a writeback that reads the
reduced block directly instead of summing over it:

    binary      one value per feature      SUM over the 8 nibble values
                at `FirstFoldIndex * 2`    whose bit is clear
    half-byte   one value per (feature,    smem[fold * 16 + 2 * fid + w]
                fold) at `(FirstFoldIndex
                + fold) * 2`

THE FOLD GUARD AND THE EPSILON GUARD ARE REDUNDANT WITH EACH OTHER, and
that was measured rather than reasoned. CatBoost carries both (`:81`, `:83`):

    if (fid < fCount && fold < feature[fid].Folds) {
        const float result = smem[fold * 16 + 2 * fid + w];
        if (abs(result) > 1e-20) { ... }
    }

The accumulator always builds 16 bins. A feature with 5 folds owns bins 0-4,
and the histogram slot for its fold 5 belongs to the NEXT feature -- so
without the fold guard it writes over its neighbour's head. But on
well-formed data those overrun bins are EMPTY, the value is an exact zero,
and the epsilon guard suppresses the write anyway.

Measured against `mojo_only/pointwise_small_bin_driver_check.mojo`:

    fold guard removed        every gate green
    epsilon guard removed     every gate green
    BOTH removed              E1, 22 of 158 cells wrong

So each guard alone is sufficient and neither is individually observable.
Both are transcribed because both are theirs, and because the fold guard is
the one that still holds if a feature's declared `Folds` ever disagrees with
the bins its column actually contains -- which is a corruption the epsilon
guard cannot see, since those bins would be non-zero.

DEVIATION (arch): `use64BitLoad = IsFullPass` on the modern arm, as in the
binary kernel.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import abs
from std.memory import stack_allocation
from std.atomic import Atomic
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
    pw_hb_half_byte_slot,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_templ import PW_WRITE_EPS


#: `feature += (blockIdx.x / M) * 8` (`:36`).
comptime PW_HB_FEATURES_PER_BLOCK = 8


def compute_split_properties_half_byte_kernel[
    full_pass: Bool, m: Int
](
    feature_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_first_fold_index: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    f_count_in: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    partition: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    total_feature_count_in: Int32,
):
    """`ComputeSplitPropertiesHalfByteImpl` (`:23-94`), copied."""
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

    var f_base = (Int(block_idx.x) // m) * PW_HB_FEATURES_PER_BLOCK
    var f_count = Int(f_count_in) - f_base
    if f_count > PW_HB_FEATURES_PER_BLOCK:
        f_count = PW_HB_FEATURES_PER_BLOCK
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

    barrier()  # theirs, `:78`

    var fid = tid // 32
    var fold = (tid // 2) & 15
    var w = tid & 1

    if fid < f_count:
        var folds_here = Int(feature_folds.unsafe_load(f_base + fid))
        if fold < folds_here:
            var result = smem.unsafe_load(
                pw_hb_half_byte_slot(fid, fold, w)
            )
            if abs(result) > PW_WRITE_EPS:
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
                    _ = Atomic.fetch_add(
                        bin_sums.unsafe_offset(at), result
                    )
                else:
                    bin_sums.unsafe_store(at, result)
