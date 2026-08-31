# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `gbdt/methods/kernel/split_properties_helpers.mojo`.

The pointwise family's offset arithmetic, which is the first thing landing
from CatBoost's OTHER histogram family (`PORTING.md` 91 B, rung 1).

WHAT EACH GATE CAN CATCH, because a gate that cannot fail is not a gate:

  P1  the fold stripe.  `GetHistogramOffset` packs the fold axis TIGHT and
      `GetDataPartitionOffset` rounds it UP to a power of two. Conflating
      them is invisible at every power-of-two fold count and corrupts
      everywhere else, so this gate asserts they DIFFER at fold counts 3, 5,
      6 and 7 and AGREE at 1, 2, 4 and 8.
  P2  `EstimateBlockPerFeatureMultiplier` against a hand trace, including
      both of its exit conditions and the `limit`.
  P3  `ShiftPartAndBinSumsPtr`: the partial pass must point the DATA at the
      SMALLER child while pointing the HISTOGRAM at the RIGHT one. Getting
      the pair backwards still reads a real partition and still writes a
      real histogram slot, so nothing crashes and the tree quietly changes.
  P4  the scan, on device, against a host prefix over HASHED per-cell
      values. Uniform values would verify the total and nothing about
      placement -- this repository has twice shipped a check that did
      exactly that -- so every cell gets a distinct value derived from
      (part, stat, feature, bin) and every cell is compared.
  P5  the stat axis. The pointwise histogram is STAT-MINOR
      (`... + b * HIST_COUNT + h`) where the greedy-subsets one is
      stat-major. A transposed scan still scans every cell exactly once, so
      P4's totals cannot see it: P5 plants values that differ ACROSS stats
      at the same bin and checks the two stats' prefixes independently.
  P6  one-hot features are skipped, and features with <= 1 fold are
      skipped.

SABOTAGE. This repository puts the sabotage in the CHECK, not in the
library (`original/nan_mode_check.mojo` N1 is the precedent): a defect
compiled into shipped code is a defect that can ship. So each mechanism gets
a WRONG expected value computed on the host, and the gate must reject it.

  S1  the fold-stripe mechanism. P1 already IS this sabotage: it computes
      the tight packing beside the striped one and requires them to differ
      at ragged fold counts and agree at powers of two. A
      `data_partition_offset` that forgot the stripe passes the second half
      and fails the first.
  S2  the stat axis. P5 builds the TRANSPOSED expected array -- what a
      stat-major scan would have produced -- and requires the device output
      NOT to match it. A transposed scan touches every cell exactly once,
      so no total moves and P4 alone cannot see it.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.methods.kernel.split_properties_helpers import (
    PointwisePartOffsetsHelper,
    estimate_block_per_feature_multiplier,
    has_one_hot_features,
    scan_pointwise_histograms_kernel,
    shift_part_and_bin_sums_ptr,
)


def cell_value(part: Int, stat: Int, feature: Int, b: Int) -> Float32:
    """A distinct, non-uniform value per cell.

    Small integers only, so the running prefix is exact in float32 and a
    mismatch is a real mismatch rather than a rounding tail.
    """
    var h = (part * 7919 + stat * 104729 + feature * 1301 + b * 17) % 61
    return Float32(h + 1)


def main() raises:
    var failures = 0

    # ---------------------------------------------------------------- P1
    # the fold stripe: tight vs rounded up to a power of two
    var tight_only: List[Int] = [1, 2, 4, 8]
    for i in range(len(tight_only)):
        var fc = tight_only[i]
        var h = PointwisePartOffsetsHelper(UInt32(fc))
        for p in range(4):
            for f in range(fc):
                var a = h.histogram_offset(UInt32(p), UInt32(f))
                var b = h.data_partition_offset(UInt32(p), UInt32(f))
                if a != b:
                    print(
                        "P1 FAIL: at power-of-two foldCount",
                        fc,
                        "the two offsets must AGREE, got",
                        a,
                        b,
                    )
                    failures += 1

    var stripe_expect: List[Int] = [4, 8, 8, 8]
    var ragged: List[Int] = [3, 5, 6, 7]
    var saw_difference = False
    for i in range(len(ragged)):
        var fc = ragged[i]
        var h = PointwisePartOffsetsHelper(UInt32(fc))
        # the stripe itself, read off part 1 fold 0
        var stripe = Int(h.data_partition_offset(1, 0))
        if stripe != stripe_expect[i]:
            print(
                "P1 FAIL: foldStripe for foldCount",
                fc,
                "expected",
                stripe_expect[i],
                "got",
                stripe,
            )
            failures += 1
        for p in range(1, 4):
            for f in range(fc):
                var a = h.histogram_offset(UInt32(p), UInt32(f))
                var b = h.data_partition_offset(UInt32(p), UInt32(f))
                if a != b:
                    saw_difference = True
    if not saw_difference:
        print(
            "P1 FAIL: at ragged fold counts the histogram and data-partition"
            " offsets never differed -- the fold stripe is not being applied"
        )
        failures += 1

    # S1 stated as the sabotage it is: the tight packing, computed here,
    # must NOT be what `data_partition_offset` returns at a ragged count.
    var h5 = PointwisePartOffsetsHelper(5)
    var tight = UInt32(3) * 5 + 2
    if h5.data_partition_offset(3, 2) == tight:
        print(
            "P1/S1 FAIL: data_partition_offset(3,2) at foldCount 5 returned"
            " the TIGHT packing",
            tight,
            "-- the fold stripe is missing",
        )
        failures += 1

    # ---------------------------------------------------------------- P2
    # hand trace: sm=20 -> threshold 20*2*1.25 = 50, so the multiplier
    # doubles while it stays under 50 and while dsSize/multiplier > 10000
    if estimate_block_per_feature_multiplier(1, 1, 1, 1000000, 20) != 64:
        print("P2 FAIL: expected 64")
        failures += 1
    # dsSize exit: 100000/8 = 12500 > 10000, /16 = 6250 -> stops at 16
    if estimate_block_per_feature_multiplier(1, 1, 1, 100000, 20) != 16:
        print("P2 FAIL: dsSize exit, expected 16")
        failures += 1
    # occupancy exit: already 64 blocks against a threshold of 50
    if estimate_block_per_feature_multiplier(8, 8, 1, 1000000, 20) != 1:
        print("P2 FAIL: occupancy exit, expected 1")
        failures += 1
    # the z axis is capped at 8, so z=64 counts as 8: 1*1*8 = 8 < 50 -> 16,
    # 16*8 = 128 >= 50 stops... trace: m=1 (8<50) -> 2 (16<50) -> 4 (32<50)
    # -> 8 (64>=50 stop) = 8
    if estimate_block_per_feature_multiplier(1, 1, 64, 1000000, 20) != 8:
        print("P2 FAIL: z cap, expected 8")
        failures += 1
    # the limit argument
    if estimate_block_per_feature_multiplier(1, 1, 1, 1000000, 20, 4) != 4:
        print("P2 FAIL: limit, expected 4")
        failures += 1

    # ---------------------------------------------------------------- P3
    # partition sizes as a stride-2 UInt32 view of {Offset, Size}
    var n_parts = 8
    var sizes_host = List[UInt32]()
    for p in range(n_parts):
        sizes_host.append(UInt32(p * 100))  # Offset, unread
        sizes_host.append(UInt32(0))  # Size, set below
    # depth 1: gridDim.y == 2, so pairs are (0,2) and (1,3)
    # make the LEFT child smaller in pair 0 and the RIGHT smaller in pair 1
    sizes_host[0 * 2 + 1] = 10
    sizes_host[2 * 2 + 1] = 90
    sizes_host[1 * 2 + 1] = 70
    sizes_host[3 * 2 + 1] = 30

    var sizes_ptr = sizes_host.unsafe_ptr()
    var total_features = UInt32(5)
    var hist_count = UInt32(2)
    var line = total_features * hist_count

    # full pass: part 1 fold 0 at foldCount 1
    var fp = shift_part_and_bin_sums_ptr(
        sizes_ptr, 1, 1, 0, 2, total_features, True, hist_count
    )
    if fp.partition_offset != 1 or fp.bin_sums_offset != line:
        print(
            "P3 FAIL: full pass expected (1,",
            line,
            ") got (",
            fp.partition_offset,
            ",",
            fp.bin_sums_offset,
            ")",
        )
        failures += 1

    # partial pass, pair 0: left(0)=10 < right(2)=90 -> data at 0, hist at 2
    var p0 = shift_part_and_bin_sums_ptr(
        sizes_ptr, 1, 0, 0, 2, total_features, False, hist_count
    )
    if p0.partition_offset != 0 or p0.bin_sums_offset != line * 2:
        print(
            "P3 FAIL: pair 0 expected data 0 / hist",
            line * 2,
            "got",
            p0.partition_offset,
            p0.bin_sums_offset,
        )
        failures += 1

    # partial pass, pair 1: left(1)=70 > right(3)=30 -> data at 3, hist at 3
    var p1 = shift_part_and_bin_sums_ptr(
        sizes_ptr, 1, 1, 0, 2, total_features, False, hist_count
    )
    if p1.partition_offset != 3 or p1.bin_sums_offset != line * 3:
        print(
            "P3 FAIL: pair 1 expected data 3 / hist",
            line * 3,
            "got",
            p1.partition_offset,
            p1.bin_sums_offset,
        )
        failures += 1

    _ = sizes_host

    # ------------------------------------------------------- P4/P5/P6
    var ctx = DeviceContext()

    var n_features = 6
    var n_parts_scan = 3
    var n_stats = 2
    # ragged folds on purpose, and two skipped features
    var folds_host: List[UInt32] = [5, 1, 7, 3, 4, 2]
    var one_hot_host: List[UInt8] = [0, 0, 0, 1, 0, 0]
    var first_fold_host = List[UInt32]()
    var cursor = UInt32(0)
    for f in range(n_features):
        first_fold_host.append(cursor)
        cursor += folds_host[f]
    var hist_line_size = Int(cursor)

    var n_cells = n_parts_scan * hist_line_size * n_stats
    var hist_host = List[Float32]()
    for _ in range(n_cells):
        hist_host.append(0.0)
    for p in range(n_parts_scan):
        for f in range(n_features):
            for b in range(Int(folds_host[f])):
                for s in range(n_stats):
                    var at = (
                        p * hist_line_size + Int(first_fold_host[f]) + b
                    ) * n_stats + s
                    hist_host[at] = cell_value(p, s, f, b)

    var d_first = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var d_folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var d_onehot = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var d_hist = ctx.enqueue_create_buffer[DType.float32](n_cells)
    ctx.enqueue_copy(dst_buf=d_first, src_ptr=first_fold_host.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_folds, src_ptr=folds_host.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_onehot, src_ptr=one_hot_host.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=hist_host.unsafe_ptr())

    ctx.enqueue_function[scan_pointwise_histograms_kernel](
        d_first.unsafe_ptr(),
        d_folds.unsafe_ptr(),
        d_onehot.unsafe_ptr(),
        Int32(n_features),
        Int32(hist_line_size),
        Int32(n_stats),
        Int32(1),
        d_hist.unsafe_ptr(),
        grid_dim=(1, n_parts_scan, 1),
        block_dim=(64, 1, 1),
    )
    var back = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    ctx.enqueue_copy(dst_buf=back, src_buf=d_hist)
    ctx.synchronize()

    # host prefix, cell by cell, over the SAME hashed plants
    var wrong_p4 = 0
    var wrong_p5 = 0
    var wrong_p6 = 0
    for p in range(n_parts_scan):
        for f in range(n_features):
            var nf = Int(folds_host[f])
            var skipped = one_hot_host[f] != UInt8(0) or nf <= 1
            for s in range(n_stats):
                var running = Float32(0.0)
                for b in range(nf):
                    var at = (
                        p * hist_line_size + Int(first_fold_host[f]) + b
                    ) * n_stats + s
                    var plant = cell_value(p, s, f, b)
                    running += plant
                    var want = plant if skipped else running
                    var got = back[at]
                    if got != want:
                        if skipped:
                            wrong_p6 += 1
                        elif nf > 1 and b > 0:
                            # a stat transposition shows up first as the two
                            # stats' prefixes swapping, which only differs
                            # once more than one bin has accumulated
                            wrong_p5 += 1
                        else:
                            wrong_p4 += 1

    if wrong_p4 != 0:
        print("P4 FAIL:", wrong_p4, "cells wrong in the prefix")
        failures += 1
    if wrong_p5 != 0:
        print(
            "P5 FAIL:",
            wrong_p5,
            "cells wrong beyond the first bin -- the stat axis is"
            " transposed (pointwise is stat-MINOR)",
        )
        failures += 1
    # S2: what a stat-MAJOR scan would have written. It touches every cell
    # exactly once, so P4's per-cell comparison is the only thing that can
    # see it -- and only because the plants differ across stats.
    var matches_transposed = 0
    var transposed_cells = 0
    for p in range(n_parts_scan):
        for f in range(n_features):
            var nf = Int(folds_host[f])
            if one_hot_host[f] != UInt8(0) or nf <= 1:
                continue
            for s in range(n_stats):
                var running = Float32(0.0)
                for b in range(nf):
                    # the address a stat-major scan would have accumulated
                    # into, read back from where it actually landed
                    var at = (
                        p * hist_line_size + Int(first_fold_host[f])
                    ) * n_stats + s * nf + b
                    if at >= n_cells:
                        continue
                    running += cell_value(p, s, f, b)
                    transposed_cells += 1
                    if back[at] == running:
                        matches_transposed += 1
    if transposed_cells > 0 and matches_transposed == transposed_cells:
        print(
            "P5/S2 FAIL: every cell matches the STAT-MAJOR prefix, so the"
            " scan is transposed -- the pointwise histogram is stat-MINOR",
        )
        failures += 1

    if wrong_p6 != 0:
        print(
            "P6 FAIL:",
            wrong_p6,
            "one-hot or single-fold cells were modified and must not be",
        )
        failures += 1

    _ = d_first^
    _ = d_folds^
    _ = d_onehot^
    _ = d_hist^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise offsets: P1-P6 pass")
