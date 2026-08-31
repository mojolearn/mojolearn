# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for the one-byte pointwise accumulators, 5, 6 and 7 bit.

`TPointHist<0,0,B>`, `<0,1,B>` and `<0,2,B>`: four features per `UInt32`
word, accumulated into 1024-slot per-warp slices and reduced to a stat-MINOR
block `[2 * (folds * feature + fold) + stat]`.

The three are NOT one kernel with a mask. Doubling the bins halves the
private copies, which doubles the threads sharing one and forces a different
collision scheme each time:

    bits  bins  absent  inner copies  threads/copy  syncs/point  out floats
      5    32     32          4             8            8           256
      6    64     64          2            16           16           512
      7   128    128          1            32           32          1024

and each has its own bin-to-slot map:

      5   offset = f + 32 * (bin & 31)
      6   offset = f + 16 * (bin & 62) + 8 * (bin & 1)
      7   offset = f +  8 * (bin & 127)

Reading the 5-bit map onto the 6-bit layout aliases every even bin onto its
odd neighbour and **moves no total at all**, which is why every gate below is
PER CELL and not one of them sums
([[uniform-test-data-hides-permutation]]).

  A1  every (feature, fold, stat) cell equals a host tally over the same
      rows, from a hashed per-row plant.
  A2  WEIGHT LANDS IN THE EVEN SLOT AND TARGET IN THE ODD ONE. The planes
      are given disjoint magnitudes -- targets in [1, 100], weights in
      [1000, 100000] -- so a parity swap is a cell three orders of magnitude
      out, not a rounding difference. This family's convention is the
      OPPOSITE of the greedy-subsets family's, so a port that reused the
      neighbouring file's reading gets exactly this wrong.
  A3  the absent bin contributes NOTHING. Rows are planted at bin == bins in
      each feature in turn.
  A4  the four features must not bleed into each other: each gets its own
      bin distribution, so a shift in the `24 - (f << 2)` decode lands
      feature k's rows in feature k+1's cells.
  A5  all four loop entry points -- scalar n=1, scalar n=4, `uint2` and
      `uint4` -- must produce IDENTICAL output. Vector width is a LOAD
      choice, never a numeric one, and rung 2's bit-identity depends on it.

WHAT THIS FILE GATES THAT `pointwise_loop_check` CANNOT. That check gives
every thread a private tally, so it measures COVERAGE. This one uses the
real accumulators, where 8 to 32 threads share an inner copy and a barrier is
the only thing holding their writes apart -- so it measures CONTENTION. The
difference is not academic: the divergent peel loop recorded in
`PORTING.md` 92 passed every gate in that file, at every block size, in both
its broken and its fixed form, and failed here on the first run.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext

from gbdt.methods.kernel.compute_point_hist2_loop import (
    compute_histogram,
    compute_histogram_2,
    compute_histogram_4,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_FLOAT_BLOCK,
    PW_HIST2_FLOAT_SMEM_FLOATS,
    PointHist5,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_6bit import PointHist6
from gbdt.methods.kernel.pointwise_hist2_one_byte_7bit import PointHist7

comptime N_ROWS = 3000
comptime MAX_OUT = 1024


def hist_kernel_5[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_FLOAT_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist5(smem)
    comptime if variant == 1:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    barrier()
    var t = Int(thread_idx.x)
    var per = 256 // PW_HIST2_FLOAT_BLOCK
    if per < 1:
        per = 1
    for k in range(per):
        var at = t * per + k
        if at < 256:
            out_buf.unsafe_store(at, smem.unsafe_load(at))


def hist_kernel_6[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_FLOAT_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist6(smem)
    comptime if variant == 1:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    barrier()
    var t = Int(thread_idx.x)
    for k in range(512 // PW_HIST2_FLOAT_BLOCK):
        var at = t + k * PW_HIST2_FLOAT_BLOCK
        if at < 512:
            out_buf.unsafe_store(at, smem.unsafe_load(at))


def hist_kernel_7[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_FLOAT_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist7(smem)
    comptime if variant == 1:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_FLOAT_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[PW_HIST2_FLOAT_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    barrier()
    var t = Int(thread_idx.x)
    for k in range(1024 // PW_HIST2_FLOAT_BLOCK):
        var at = t + k * PW_HIST2_FLOAT_BLOCK
        if at < 1024:
            out_buf.unsafe_store(at, smem.unsafe_load(at))


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_out = ctx.enqueue_create_buffer[DType.float32](MAX_OUT)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](MAX_OUT)

    var off = 7
    var n = 2500

    var all_bits: List[Int] = [5, 6, 7]

    for bi in range(len(all_bits)):
        var bits = all_bits[bi]
        var bins = 1 << bits
        var out_floats = 4 * bins * 2

        # ---- the pool, rebuilt per bit width ------------------------
        # four DIFFERENT bin distributions so a decode shift is visible
        # (A4); every 37th row plants the ABSENT bin in a rotating
        # feature (A3)
        var indices_h = List[UInt32]()
        var target_h = List[Float32]()
        var weight_h = List[Float32]()
        var cindex_h = List[UInt32]()
        for r in range(N_ROWS):
            indices_h.append(UInt32((r * 2654435761) % N_ROWS))
            target_h.append(Float32((r * 37) % 100 + 1))
            weight_h.append(Float32(((r * 53) % 100 + 1) * 1000))
            var b0 = (r * 11) % (bins // 4)
            var b1 = (r * 13) % (bins // 2)
            var b2 = (r * 17) % bins
            var b3 = (r * 19) % 4
            if r % 37 == 0:
                var which = (r // 37) % 4
                if which == 0:
                    b0 = bins
                elif which == 1:
                    b1 = bins
                elif which == 2:
                    b2 = bins
                else:
                    b3 = bins
            cindex_h.append(
                UInt32(b0) << 24 | UInt32(b1) << 16 | UInt32(b2) << 8
                | UInt32(b3)
            )

        ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

        var want = List[Float32]()
        for _ in range(out_floats):
            want.append(0.0)
        var absent_rows = 0
        for r in range(off, off + n):
            var ci = cindex_h[Int(indices_h[r])]
            for f in range(4):
                var bin = Int((ci >> UInt32(24 - 8 * f)) & 255)
                if bin == bins:
                    absent_rows += 1
                    continue
                var fold = bin & (bins - 1)
                want[2 * (bins * f + fold) + 0] += weight_h[r]
                want[2 * (bins * f + fold) + 1] += target_h[r]

        if absent_rows == 0:
            raise Error(
                "no absent-bin rows in the window at bits "
                + String(bits)
                + " -- A3 gates nothing"
            )

        var variants: List[Int] = [1, 4, 12, 14]
        var names: List[String] = [
            String("scalar n=1"),
            String("scalar n=4"),
            String("uint2     "),
            String("uint4     "),
        ]
        var first = List[Float32]()

        for vi in range(len(variants)):
            var v = variants[vi]
            ctx.enqueue_memset(d_out, Float32(0.0))
            if bits == 5:
                if v == 1:
                    ctx.enqueue_function[hist_kernel_5[1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[hist_kernel_5[4]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 12:
                    ctx.enqueue_function[hist_kernel_5[12]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                else:
                    ctx.enqueue_function[hist_kernel_5[14]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
            elif bits == 6:
                if v == 1:
                    ctx.enqueue_function[hist_kernel_6[1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[hist_kernel_6[4]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 12:
                    ctx.enqueue_function[hist_kernel_6[12]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                else:
                    ctx.enqueue_function[hist_kernel_6[14]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
            else:
                if v == 1:
                    ctx.enqueue_function[hist_kernel_7[1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[hist_kernel_7[4]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                elif v == 12:
                    ctx.enqueue_function[hist_kernel_7[12]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
                else:
                    ctx.enqueue_function[hist_kernel_7[14]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1),
                        block_dim=(PW_HIST2_FLOAT_BLOCK, 1, 1),
                    )
            ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
            ctx.synchronize()

            var bad = 0
            var swapped = 0
            for f in range(4):
                for fold in range(bins):
                    for w in range(2):
                        var at = 2 * (bins * f + fold) + w
                        if h_out[at] != want[at]:
                            var other = 2 * (bins * f + fold) + (1 - w)
                            if h_out[at] == want[other]:
                                swapped += 1
                            if bad < 3:
                                print(
                                    "     bits", bits, names[vi], "f", f,
                                    "fold", fold, "stat", w, ": got",
                                    h_out[at], "want", want[at],
                                )
                            bad += 1
            if swapped != 0:
                print(
                    "FAIL A2: bits", bits, names[vi], "--", swapped,
                    "cells hold the OTHER stat's value. This family puts"
                    " WEIGHT in the even slot and TARGET in the odd one.",
                )
                failures += 1
            elif bad != 0:
                print(
                    "FAIL A1/A3/A4: bits", bits, names[vi], "--", bad,
                    "of", out_floats, "cells wrong",
                )
                failures += 1
            else:
                print(
                    "  ok   bits", bits, names[vi], "--", out_floats,
                    "cells exact (", absent_rows, "absent-bin rows dropped)",
                )

            if vi == 0:
                for k in range(out_floats):
                    first.append(h_out[k])
            else:
                var differ = 0
                for k in range(out_floats):
                    if h_out[k] != first[k]:
                        differ += 1
                if differ != 0:
                    print(
                        "FAIL A5: bits", bits, names[vi], "differs from"
                        " scalar n=1 in", differ, "cells -- vector width"
                        " must be a LOAD choice, never a numeric one",
                    )
                    failures += 1

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_out^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise one-byte accumulators 5/6/7: A1-A5 pass")
