"""Gate for `gbdt/methods/kernel/pointwise_hist2_one_byte_8bit.mojo`.

`TPointHist<2,1,BlockSize>` is not the 5/6/7 progression continued. Two
things about it have no counterpart in those files, and both are gated here:

  * it DEFERS. A run of equal bins accumulates in registers and only
    reaches shared memory when the bin changes, so `Reduce` has to flush
    four pending accumulators before it does anything else. Forget that and
    every feature silently loses its LAST run -- by more, the better sorted
    the data is, which is the worst possible failure mode because it looks
    fine on shuffled test data.
  * it holds INT32 FIXED POINT, because its design calls `atomicAdd` on
    threadgroup memory and Metal has none (DEVIATION 93). So its output is
    fixed point too and the caller divides by the scale.

GATES:

  B1  EXACTNESS. Run with `scale = 1.0` and integer-valued stats, so
      `val * scale` is an integer and `hist2_quantize`'s dither cannot move
      it (`floor(x) + [frac(x) + u >= 1]` with `frac(x) == 0` and `u < 1`).
      All 2048 cells must then equal an exact integer host tally. This
      gates the slot map, the wrap-around slice assignment, the two-fold
      stage-2 loop and the flush, with no numeric slack at all.

  B2  THE PENDING FLUSH, isolated. The fixture is built so every feature
      ends inside a LONG run of equal bins, and the check computes how much
      mass is in those trailing runs and refuses to run if it is zero.
      Deleting `Reduce`'s stage 0 removes exactly that mass.

  B3  THE DITHER IS UNBIASED. Re-run with non-integer stats and a
      non-integer scale. Two assertions: no cell may be off by more than
      the number of rows in it (the quantizer's worst case is +1 unit per
      row), and the SIGNED error summed over all cells must be small
      relative to the row count. Truncation -- the rule this replaced --
      fails the second badly and always in the same direction, which is how
      it was caught in the other family.

  B4  all four loop entry points must produce IDENTICAL output. For the
      float accumulators that holds because the loop delivers the same
      points in the same per-lane order. Here it holds for a stronger
      reason: the entry points assign DIFFERENT points to different
      threads, so the deferred runs differ, and only because integer
      addition is associative does every cell still land on the same total.
      That is the reproducibility this substitution buys and CatBoost's
      float version does not have.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from gbdt.methods.kernel.compute_point_hist2_loop import (
    compute_histogram,
    compute_histogram_2,
    compute_histogram_4,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_SMEM_FLOATS,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_8bit import (
    PW8_MAX_FOLD_COUNT,
    PointHist8,
)

comptime N_ROWS = 4000
comptime OUT_INTS = 4 * PW8_MAX_FOLD_COUNT * 2  # 2048
comptime RUN_LEN = 23  # rows per bin run; not a divisor of anything below


def hist8_kernel[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    scale: Float32,
    out_buf: MutPointer[Int32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HIST2_SMEM_FLOATS, Int32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHist8(smem, scale)

    comptime if variant == 1:
        compute_histogram[PW_HIST2_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HIST2_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[PW_HIST2_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )

    barrier()
    var t = Int(thread_idx.x)
    for k in range(OUT_INTS // PW_HIST2_BLOCK):
        var at = t + k * PW_HIST2_BLOCK
        if at < OUT_INTS:
            out_buf.unsafe_store(at, smem.unsafe_load(at))


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_out = ctx.enqueue_create_buffer[DType.int32](OUT_INTS)
    var h_out = ctx.enqueue_create_host_buffer[DType.int32](OUT_INTS)

    var off = 5
    var n = 3500

    # ---- the pool -----------------------------------------------------
    # RUNS of equal bins, so the deferred accumulator actually defers, and
    # an IDENTITY gather so a run in row order is a run in visit order --
    # a hashed gather would break every run into singletons and B2 would
    # gate nothing.
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    var cindex_h = List[UInt32]()
    for r in range(N_ROWS):
        indices_h.append(UInt32(r))
        target_h.append(Float32((r * 37) % 100 + 1))
        weight_h.append(Float32(((r * 53) % 100 + 1) * 7))
        var run = r // RUN_LEN
        var b0 = (run * 11) % 256
        var b1 = (run * 29) % 256
        var b2 = (run * 7) % 256
        var b3 = (run * 101) % 256
        cindex_h.append(
            UInt32(b0) << 24 | UInt32(b1) << 16 | UInt32(b2) << 8
            | UInt32(b3)
        )

    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

    # ---- B2's precondition: mass in the TRAILING runs -----------------
    # the window must end inside a run, and that run must carry weight
    var last_run = (off + n - 1) // RUN_LEN
    var trailing = 0
    for r in range(off, off + n):
        if r // RUN_LEN == last_run:
            trailing += 1
    if trailing == 0 or trailing == RUN_LEN:
        raise Error(
            "the window does not end mid-run, so B2 gates nothing:"
            " trailing=" + String(trailing)
        )

    var variants: List[Int] = [1, 4, 12, 14]
    var names: List[String] = [
        String("scalar n=1"),
        String("scalar n=4"),
        String("uint2     "),
        String("uint4     "),
    ]

    # ================================================================ B1
    # scale 1.0 and integer stats: the dither cannot move an integer, so
    # this is EXACT
    var scale = Float32(1.0)
    var want = List[Int64]()
    for _ in range(OUT_INTS):
        want.append(0)
    for r in range(off, off + n):
        var ci = cindex_h[Int(indices_h[r])]
        for f in range(4):
            var bin = Int((ci >> UInt32(24 - 8 * f)) & 255)
            want[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 0] += Int64(
                weight_h[r]
            )
            want[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 1] += Int64(
                target_h[r]
            )

    var first = List[Int32]()
    for vi in range(len(variants)):
        var v = variants[vi]
        ctx.enqueue_memset(d_out, Int32(0))
        if v == 1:
            ctx.enqueue_function[hist8_kernel[1]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                scale, d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        elif v == 4:
            ctx.enqueue_function[hist8_kernel[4]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                scale, d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        elif v == 12:
            ctx.enqueue_function[hist8_kernel[12]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                scale, d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[hist8_kernel[14]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                scale, d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
            )
        ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
        ctx.synchronize()

        var bad = 0
        var short_by = Int64(0)
        for k in range(OUT_INTS):
            if Int64(h_out[k]) != want[k]:
                short_by += want[k] - Int64(h_out[k])
                if bad < 3:
                    print(
                        "    ", names[vi], "cell", k, ": got", h_out[k],
                        "want", want[k],
                    )
                bad += 1
        if bad != 0:
            print(
                "FAIL B1/B2:", names[vi], "--", bad, "of", OUT_INTS,
                "cells wrong, net short by", short_by,
            )
            if short_by == 0:
                print(
                    "       net zero, so mass MOVED rather than vanished:"
                    " a permutation. Suspect the flush attributing a run"
                    " to the wrong bin, or the stage-2 slot map. A check"
                    " that summed the histogram would call this correct.",
                )
            else:
                print(
                    "       mass VANISHED. A deficit concentrated in a few"
                    " bins per feature is Reduce's missing pending flush --"
                    " every feature loses its last run.",
                )
            failures += 1
        else:
            print(
                "  ok   B1", names[vi], "--", OUT_INTS,
                "cells exact at scale 1.0 (", trailing,
                "rows in the trailing run )",
            )

        if vi == 0:
            for k in range(OUT_INTS):
                first.append(h_out[k])
        else:
            var differ = 0
            for k in range(OUT_INTS):
                if h_out[k] != first[k]:
                    differ += 1
            if differ != 0:
                print(
                    "FAIL B4:", names[vi], "differs from scalar n=1 in",
                    differ, "cells. The entry points assign different"
                    " points to different threads, so the deferred runs"
                    " differ -- but integer addition is associative and"
                    " every cell must still total the same.",
                )
                failures += 1

    # ================================================================ B3
    # non-integer stats and a non-integer scale: the dither must be
    # unbiased, which truncation is not
    var scale3 = Float32(3.7)
    var exact = List[Float64]()
    var count = List[Int64]()
    for _ in range(OUT_INTS):
        exact.append(0.0)
        count.append(0)
    for r in range(off, off + n):
        var ci = cindex_h[Int(indices_h[r])]
        # a deliberately non-integer pair
        var tv = Float64(target_h[r]) + 0.3141592
        var wv = Float64(weight_h[r]) + 0.2718281
        for f in range(4):
            var bin = Int((ci >> UInt32(24 - 8 * f)) & 255)
            exact[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 0] += wv
            exact[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 1] += tv
            count[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 0] += 1
            count[2 * (PW8_MAX_FOLD_COUNT * f + bin) + 1] += 1

    var tgt3 = List[Float32]()
    var wt3 = List[Float32]()
    for r in range(N_ROWS):
        tgt3.append(target_h[r] + Float32(0.3141592))
        wt3.append(weight_h[r] + Float32(0.2718281))
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=tgt3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=wt3.unsafe_ptr())

    ctx.enqueue_memset(d_out, Int32(0))
    ctx.enqueue_function[hist8_kernel[1]](
        d_idx.unsafe_ptr(), Int32(off), Int32(n),
        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
        scale3, d_out.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(PW_HIST2_BLOCK, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
    ctx.synchronize()

    var over_bound = 0
    var signed_err = Float64(0.0)
    var total_rows = Int64(0)
    for k in range(OUT_INTS):
        if count[k] == 0:
            continue
        var got = Float64(Int(h_out[k]))
        var expect = exact[k] * Float64(scale3)
        var err = got - expect
        signed_err += err
        total_rows += count[k]
        # `floor(x) + [frac(x) + u >= 1]` lands within one unit of `x` in
        # EITHER direction, so a cell over `count` rows is bounded by
        # +/- (count + 1). This is a coarse sanity bound and it is not the
        # gate that matters -- see the bias check below, which is what
        # separates a dither from a truncation.
        if err < -(Float64(count[k]) + 1.0) or err > Float64(count[k]) + 1.0:
            if over_bound < 3:
                print(
                    "    B3 cell", k, ": err", err, "outside +/-",
                    Float64(count[k]) + 1.0, "over", count[k], "rows",
                )
            over_bound += 1
    if over_bound != 0:
        print("FAIL B3:", over_bound, "cells outside the quantizer bound")
        failures += 1
    else:
        print("  ok   B3 -- every cell inside the +1-per-row bound")

    # zero-mean: the dither's whole justification. Truncation would put the
    # signed error at about -0.5 per row.
    # THE SHARP GATE, and it is count-independent. A dithered quantizer
    # rounds up about half the time; truncation rounds down every time.
    var neg = 0
    var nonzero = 0
    for k in range(OUT_INTS):
        if count[k] == 0:
            continue
        var err = Float64(Int(h_out[k])) - exact[k] * Float64(scale3)
        if err != 0.0:
            nonzero += 1
            if err < 0.0:
                neg += 1
    var neg_frac = Float64(neg) / Float64(nonzero)
    if neg_frac < 0.3 or neg_frac > 0.7:
        print(
            "FAIL B3:", neg, "of", nonzero,
            "non-exact cells err NEGATIVE (", neg_frac,
            "). A dither rounds up about half the time; truncation rounds"
            " down every time and lands this near 1.0.",
        )
        failures += 1
    else:
        print(
            "  ok   B3 --", neg, "of", nonzero,
            "non-exact cells negative (", neg_frac,
            "), so the rounding is two-sided",
        )

    var bias_per_row = signed_err / Float64(total_rows)
    if bias_per_row < -0.05 or bias_per_row > 0.05:
        print(
            "FAIL B3: signed error is", bias_per_row,
            "per row -- the quantizer is BIASED. Truncation gives about"
            " -0.5; a correct dither gives O(1/sqrt(rows)).",
        )
        failures += 1
    else:
        print(
            "  ok   B3 -- signed error", bias_per_row, "per row over",
            total_rows, "row-stats (dither is unbiased)",
        )

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_out^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise 8-bit accumulator: B1-B4 pass")
