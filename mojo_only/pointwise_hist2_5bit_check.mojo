"""Gate for `gbdt/methods/kernel/pointwise_hist2_one_byte_5bit.mojo`.

`TPointHist<0,0,BlockSize>`: four features per `UInt32` word, up to 32 bins
each, 32 as the absent marker, accumulated into a 1024-slot per-warp slice
and reduced to 256 floats laid out `[2 * (32 * feature + fold) + stat]`.

EVERY GATE HERE IS PER CELL. Not one of them sums the histogram, and that is
deliberate: the two mistakes this accumulator invites -- transposing the stat
axis, and swapping weight with target -- both leave every total identical.
[[uniform-test-data-hides-permutation]] is the standing record of this
repository shipping a check that could not see placement.

  A1  every (feature, fold, stat) cell equals a host tally over the same
      rows. The plant is hashed per row, so a cell that receives the right
      COUNT of wrong rows still fails.
  A2  WEIGHT LANDS IN THE EVEN SLOT AND TARGET IN THE ODD ONE. The two
      planes are given disjoint magnitudes -- targets in [1, 100], weights
      in [1000, 100000] -- so a parity swap is not a subtle numeric
      difference, it is a cell three orders of magnitude out. This family's
      convention is the OPPOSITE of the greedy-subsets family's
      (`stat1 = flag ? t : w` against `stat1 = flag ? s2 : s1`), so a port
      that reused the neighbouring file's reading gets exactly this wrong
      and changes no total.
  A3  bin 32 is ABSENT and must contribute nothing. Rows are planted with
      bin 32 in each feature in turn; their target and weight are large
      enough that a single leaked row moves its cell visibly.
  A4  the four features must not bleed into each other. Each gets a
      different bin distribution, so a shift in the `24 - (f << 2)` decode
      lands feature k's rows in feature k+1's cells.
  A5  all three loop entry points -- scalar n=1, n=4, the uint2 form and the
      uint4 form -- must produce IDENTICAL output. Vector width is a load
      choice, never a numeric one, and rung 2's bit-identity depends on it.

SABOTAGE, in the check rather than in the library. A2 and A4 are themselves
the sabotages, stated as expectations a wrong port satisfies differently.
Runs verified to move the gate are listed in the commit.
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
    PW_HIST2_BLOCK,
    PW_HIST2_SMEM_FLOATS,
    PointHist5,
)

comptime N_ROWS = 3000
comptime OUT_FLOATS = 256


def hist_kernel[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist5(smem)

    comptime if variant == 1:
        compute_histogram[PW_HIST2_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target,
            weight, cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target,
            weight, cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target,
            weight, cindex,
        )
    else:
        compute_histogram_4[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target,
            weight, cindex,
        )

    barrier()
    var t = Int(thread_idx.x)
    if t < OUT_FLOATS:
        out_buf.unsafe_store(t, smem.unsafe_load(t))


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ---- the pool ---------------------------------------------------
    # feature 0 uses bins 0..7, feature 1 uses 0..15, feature 2 uses 0..31,
    # feature 3 uses 0..3 -- four different distributions, so a decode shift
    # (A4) puts a feature's rows somewhere its own bins never reach.
    # Every 37th row plants bin 32 (ABSENT) in a rotating feature (A3).
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    var cindex_h = List[UInt32]()
    for r in range(N_ROWS):
        indices_h.append(UInt32((r * 2654435761) % N_ROWS))
        # disjoint magnitudes: a parity swap is unmistakable (A2)
        target_h.append(Float32((r * 37) % 100 + 1))
        weight_h.append(Float32(((r * 53) % 100 + 1) * 1000))
        var b0 = (r * 11) % 8
        var b1 = (r * 13) % 16
        var b2 = (r * 17) % 32
        var b3 = (r * 19) % 4
        if r % 37 == 0:
            var which = (r // 37) % 4
            if which == 0:
                b0 = 32
            elif which == 1:
                b1 = 32
            elif which == 2:
                b2 = 32
            else:
                b3 = 32
        cindex_h.append(
            UInt32(b0) << 24 | UInt32(b1) << 16 | UInt32(b2) << 8
            | UInt32(b3)
        )

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

    var d_out = ctx.enqueue_create_buffer[DType.float32](OUT_FLOATS)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](OUT_FLOATS)

    var off = 7
    var n = 2500

    # ---- host answer, per (feature, fold, stat) ----------------------
    # slot layout is theirs: 2 * (32 * f + fold) + w, w = 0 WEIGHT, 1 TARGET
    var want = List[Float32]()
    for _ in range(OUT_FLOATS):
        want.append(0.0)
    var absent_rows = 0
    for r in range(off, off + n):
        var ci = cindex_h[Int(indices_h[r])]
        for f in range(4):
            var bin = Int((ci >> UInt32(24 - 8 * f)) & 255)
            if bin == 32:
                absent_rows += 1
                continue
            var fold = bin & 31
            want[2 * (32 * f + fold) + 0] += weight_h[r]
            want[2 * (32 * f + fold) + 1] += target_h[r]

    if absent_rows == 0:
        raise Error(
            "the fixture plants no bin-32 rows in the window, so A3 gates"
            " nothing -- widen the window or the plant"
        )

    var variants: List[Int] = [1, 4, 12, 14]
    var names: List[String] = [
        String("scalar n=1"),
        String("scalar n=4"),
        String("uint2  "),
        String("uint4  "),
    ]
    var first = List[Float32]()

    for vi in range(len(variants)):
        var v = variants[vi]
        ctx.enqueue_memset(d_out, Float32(0.0))
        if v == 1:
            ctx.enqueue_function[hist_kernel[1]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        elif v == 4:
            ctx.enqueue_function[hist_kernel[4]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        elif v == 12:
            ctx.enqueue_function[hist_kernel[12]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[hist_kernel[14]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
        ctx.synchronize()

        var bad = 0
        var swapped = 0
        for f in range(4):
            for fold in range(32):
                for w in range(2):
                    var at = 2 * (32 * f + fold) + w
                    if h_out[at] != want[at]:
                        # is it the OTHER stat's value? that is A2
                        var other = 2 * (32 * f + fold) + (1 - w)
                        if h_out[at] == want[other]:
                            swapped += 1
                        if bad < 4:
                            print(
                                "   ", names[vi], "f", f, "fold", fold,
                                "stat", w, ": got", h_out[at], "want",
                                want[at],
                            )
                        bad += 1
        if swapped != 0:
            print(
                "FAIL A2:", names[vi], "--", swapped,
                "cells hold the OTHER stat's value. This family puts WEIGHT"
                " in the even slot and TARGET in the odd one, which is the"
                " opposite of the greedy-subsets family.",
            )
            failures += 1
        elif bad != 0:
            print("FAIL A1/A3/A4:", names[vi], "--", bad, "cells wrong")
            failures += 1
        else:
            print(
                "  ok  ", names[vi], "-- 256 cells exact (", absent_rows,
                "absent-bin rows correctly dropped )",
            )

        if vi == 0:
            for k in range(OUT_FLOATS):
                first.append(h_out[k])
        else:
            var differ = 0
            for k in range(OUT_FLOATS):
                if h_out[k] != first[k]:
                    differ += 1
            if differ != 0:
                print(
                    "FAIL A5:", names[vi], "differs from scalar n=1 in",
                    differ, "cells -- vector width must be a LOAD choice,"
                    " never a numeric one",
                )
                failures += 1

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_out^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise 5-bit accumulator: A1-A5 pass")
