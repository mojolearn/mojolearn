"""Gate for `gbdt/methods/kernel/pointwise_hist2_half_byte_template.mojo`.

ONE accumulator, `TPointHistHalfByte`, and the TWO readers CatBoost builds on
top of it. The accumulator is gated once; each reader is gated separately,
because the reader is where the two kernels differ and the accumulator is not.

  C1  HALF-BYTE. 8 features x 16 folds x 2 stats = 256 cells, each equal to
      a host tally. This gates the rotation: `shift = threadIdx.x & 14`
      gives the 16 threads sharing an inner copy 8 distinct starting
      features, and `RotateRight` pre-rotates the packed nibbles so that
      thread and data agree. Get the rotation direction or amount wrong and
      threads read the wrong feature's nibble -- which SHUFFLES mass between
      features and leaves the grand total untouched.

  C2  BINARY. 32 one-bit features, each recovered by summing the 8 nibble
      values whose bit is clear. Gated per (feature, stat) against a host
      tally over the same predicate.

  C3  THE BIT NUMBERING IS MOST-SIGNIFICANT-FIRST. `1 << (3 - (fid & 3))`,
      so feature 0 of a group is bit 3. Reading it as `1 << (fid & 3)`
      transposes the four features WITHIN each group -- a permutation, so
      the group's total is unchanged and only a per-feature comparison can
      see it. The fixture asserts up front that the four features of a group
      have distinct sums, because against a fixture where they did not this
      gate would pass either way.

  C4  scalar n=1, scalar n=4 and `uint2` must agree exactly. `uint4` is
      included too even though **CatBoost never selects it for this
      accumulator** -- it has no `AddPoint4` -- so that is a check of this
      port's filler, not of theirs.

  C5  the 512-thread floor. Not a runtime gate: the accumulator carries a
      `comptime assert` because their `Reduce` folds under
      `if (threadIdx.x < 512)` and a smaller block loses folds with no
      other symptom. Recorded here so the reason is findable from the check.
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
from gbdt.methods.kernel.pointwise_hist2_half_byte_template import (
    PW_HB_BLOCK,
    PW_HB_OUT_FLOATS,
    PW_HB_SMEM_FLOATS,
    PointHistHalfByte,
    pw_hb_binary_sum,
    pw_hb_half_byte_slot,
)

comptime N_ROWS = 4000


def hb_kernel[variant: Int, binary: Bool](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        PW_HB_SMEM_FLOATS, Float32, address_space = AddressSpace.SHARED
    ]()
    var hist = PointHistHalfByte(smem)

    comptime if variant == 1:
        compute_histogram[PW_HB_BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[PW_HB_BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[PW_HB_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[PW_HB_BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )

    barrier()
    var t = Int(thread_idx.x)

    comptime if binary:
        # `ComputeSplitPropertiesBImpl`'s writeback (`:69-91`): one value per
        # (binary feature, stat), 32 features x 2 = 64 outputs
        var w = t & 1
        var fid = t >> 1
        if fid < 32:
            out_buf.unsafe_store(t, pw_hb_binary_sum(smem, fid, w))
    else:
        # `ComputeSplitPropertiesHalfByteImpl`'s writeback (`:79-90`)
        var fid = t // 32
        var fold = (t // 2) & 15
        var w = t & 1
        if fid < 8:
            out_buf.unsafe_store(
                pw_hb_half_byte_slot(fid, fold, w),
                smem.unsafe_load(pw_hb_half_byte_slot(fid, fold, w)),
            )


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ---- the pool ----------------------------------------------------
    # eight nibbles per word, each feature a different distribution so a
    # rotation error moves mass between features visibly
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    var cindex_h = List[UInt32]()
    for r in range(N_ROWS):
        indices_h.append(UInt32((r * 2654435761) % N_ROWS))
        # SMALL AND DISJOINT. Disjoint so a stat-parity swap is obvious;
        # small so every cell sum stays well inside float32's exact-integer
        # range. The first version of this fixture used weights up to
        # 100,000, and one cell reached 1.8e8 -- past 2^24 -- so the device
        # and the host disagreed in the last bits and the check reported a
        # kernel bug that was its own rounding.
        target_h.append(Float32((r * 37) % 100 + 1))
        weight_h.append(Float32((r * 53) % 100 + 1001))
        var word = UInt32(0)
        for j in range(8):
            # `% 13`, NOT `% 16`, and the modulus is the point. A nibble
            # uniform over 0..15 sets each of its four bits exactly half the
            # time, so the four binary features packed into it get IDENTICAL
            # sums and C3's transposition becomes invisible -- the check
            # refused to run against exactly that plant. Thirteen values
            # give the four bits different marginals.
            var nib = Int((UInt32(r) * UInt32(2654435761 + j)) >> 9) % 13
            word |= UInt32(nib) << UInt32(28 - 4 * j)
        cindex_h.append(word)

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

    var d_out = ctx.enqueue_create_buffer[DType.float32](PW_HB_OUT_FLOATS)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](
        PW_HB_OUT_FLOATS
    )

    var off = 3
    var n = 3600

    var variants: List[Int] = [1, 4, 12, 14]
    var names: List[String] = [
        String("scalar n=1"),
        String("scalar n=4"),
        String("uint2     "),
        String("uint4     "),
    ]

    # ================================================================ C1
    # half-byte: feature j reads nibble j counting from the TOP
    var want_hb = List[Float32]()
    for _ in range(PW_HB_OUT_FLOATS):
        want_hb.append(0.0)
    for r in range(off, off + n):
        var ci = cindex_h[Int(indices_h[r])]
        for j in range(8):
            var nib = Int((ci >> UInt32(28 - 4 * j)) & 15)
            want_hb[pw_hb_half_byte_slot(j, nib, 0)] += weight_h[r]
            want_hb[pw_hb_half_byte_slot(j, nib, 1)] += target_h[r]

    var first_hb = List[Float32]()
    for vi in range(len(variants)):
        var v = variants[vi]
        ctx.enqueue_memset(d_out, Float32(0.0))
        if v == 1:
            ctx.enqueue_function[hb_kernel[1, False]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
            )
        elif v == 4:
            ctx.enqueue_function[hb_kernel[4, False]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
            )
        elif v == 12:
            ctx.enqueue_function[hb_kernel[12, False]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
            )
        else:
            ctx.enqueue_function[hb_kernel[14, False]](
                d_idx.unsafe_ptr(), Int32(off), Int32(n),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_out.unsafe_ptr(),
                grid_dim=(1, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
            )
        ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
        ctx.synchronize()

        var bad = 0
        for k in range(PW_HB_OUT_FLOATS):
            if h_out[k] != want_hb[k]:
                if bad < 3:
                    print(
                        "     C1", names[vi], "cell", k, ": got", h_out[k],
                        "want", want_hb[k],
                    )
                bad += 1
        if bad != 0:
            print(
                "FAIL C1:", names[vi], "--", bad, "of", PW_HB_OUT_FLOATS,
                "half-byte cells wrong. Suspects, in the order they were"
                " verified to land here: the rotation (direction or amount),"
                " the slice's innerHistStart, and Reduce stage 2 dropping"
                " the second inner copy. All three shuffle or lose mass"
                " without any single cell looking obviously absurd.",
            )
            failures += 1
        else:
            print("  ok   C1", names[vi], "-- 256 half-byte cells exact")

        if vi == 0:
            for k in range(PW_HB_OUT_FLOATS):
                first_hb.append(h_out[k])
        else:
            var differ = 0
            for k in range(PW_HB_OUT_FLOATS):
                if h_out[k] != first_hb[k]:
                    differ += 1
            if differ != 0:
                print(
                    "FAIL C4:", names[vi], "differs from scalar n=1 in",
                    differ, "cells",
                )
                failures += 1

    # ============================================================= C2/C3
    # binary: feature fid is bit (3 - (fid & 3)) of nibble (fid / 4)
    var want_b = List[Float32]()
    for _ in range(64):
        want_b.append(0.0)
    for r in range(off, off + n):
        var ci = cindex_h[Int(indices_h[r])]
        for fid in range(32):
            var g = fid // 4
            var nib = Int((ci >> UInt32(28 - 4 * g)) & 15)
            var bit = (nib >> (3 - (fid & 3))) & 1
            if bit == 0:
                want_b[2 * fid + 0] += weight_h[r]
                want_b[2 * fid + 1] += target_h[r]

    # C3's precondition: the four features of a group must be
    # DISTINGUISHABLE, or a transposed bit order passes
    var indistinct = 0
    for g in range(8):
        for a in range(4):
            for b in range(a + 1, 4):
                if want_b[2 * (4 * g + a)] == want_b[2 * (4 * g + b)]:
                    indistinct += 1
    if indistinct != 0:
        raise Error(
            "the fixture gives "
            + String(indistinct)
            + " pairs of same-group binary features identical sums, so C3"
            " would pass a transposed bit order -- change the plant"
        )

    ctx.enqueue_memset(d_out, Float32(0.0))
    ctx.enqueue_function[hb_kernel[1, True]](
        d_idx.unsafe_ptr(), Int32(off), Int32(n),
        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_ci.unsafe_ptr(),
        d_out.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
    ctx.synchronize()

    var bad_b = 0
    var transposed = 0
    for fid in range(32):
        for w in range(2):
            var at = 2 * fid + w
            if h_out[at] != want_b[at]:
                # is it another feature IN THE SAME GROUP? that is C3
                var g = fid // 4
                for other in range(4):
                    if h_out[at] == want_b[2 * (4 * g + other) + w]:
                        transposed += 1
                if bad_b < 3:
                    print(
                        "     C2 feature", fid, "stat", w, ": got",
                        h_out[at], "want", want_b[at],
                    )
                bad_b += 1
    if transposed != 0:
        print(
            "FAIL C3:", transposed,
            "cells hold ANOTHER FEATURE OF THE SAME GROUP's value -- the"
            " bit numbering is transposed. Theirs is"
            " `1 << (3 - (fid & 3))`, most-significant-first. The group"
            " total is unchanged by this, so only a per-feature check"
            " sees it.",
        )
        failures += 1
    elif bad_b != 0:
        print("FAIL C2: --", bad_b, "of 64 binary cells wrong")
        failures += 1
    else:
        print("  ok   C2/C3 -- 64 binary cells exact, 32 features distinct")

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_out^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise half-byte accumulator + both readers: C1-C4 pass")
