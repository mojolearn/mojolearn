# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo`.

The DRIVER, which is everything the accumulators are not: slice the feature
array, pick the partition and the histogram slot, refuse the blocks that
belong to another bit width, run the loop, write out.

This is the first check in the pointwise family that exercises the whole
chain end to end -- offsets, loop, accumulator, reduce, writeback -- so a
failure here can come from anywhere and the earlier checks are what narrow
it.

THE FIXTURE IS BUILT AROUND THE DISPATCH. Three feature groups of four,
whose WIDEST feature falls in a different bit width's range:

    group 0   folds  20,  25,  30,  32   max  32  -> the 5-bit kernel
    group 1   folds  90, 100, 120, 128    max 128  -> the 7-bit kernel
    group 2   folds  16,  10,  12,  14    max  16  -> the 5-bit kernel
    group 3   folds 200, 150, 256, 180    max 256  -> the 8-bit kernel

and all four bit-width kernels are launched over all four groups.

GROUP 3 EXISTS BECAUSE ITS ABSENCE WAS A HOLE. The first version of this
fixture had maxima of 32, 128 and 16, so the 8-bit kernel was launched and
returned immediately every time -- its fixed-point writeback, the one place
in the driver that divides by a scale (DEVIATION 93), ran zero times and D1
passed anyway. A gate that launches a path is not a gate that reaches it
([[mojotrees-verify-reach-not-output]]). Each
kernel must claim exactly the groups in its range and return from the rest,
so the three groups come out filled exactly once with no coordination
between the launches. That is CatBoost's design: the host does not decide
which kernel handles which block, the kernels do, at runtime, from the data.

GROUP 2 EXISTS FOR ONE LINE. `lowerBound = BITS > 5 ? upperBound / 2 : 15`
-- fifteen, not sixteen. A block whose widest feature has exactly 16 folds
belongs to the 5-bit kernel. Write `upperBound / 2` uniformly and 5-bit
rejects it (`16 <= 16`) while 6-bit also rejects it (`16 <= 32`), so the
group is silently dropped and its features read zero. Without group 2 both
spellings pass.

GATES:
  D1  every (bin-feature, stat) cell of `binSums` equals a host tally.
      `FirstFoldIndex` is per feature and non-zero for all but the first, so
      this also gates that a feature lands in its OWN slice rather than at
      the start.
  D2  the dispatch is a PARTITION, checked TWICE. Once implicitly -- D1
      runs all four kernels over all four groups, so a double claim doubles
      a cell and a missing claim zeroes one. And once explicitly: each bit
      width is run ALONE and its claim set recorded, which must be

          5 -> groups 0 and 2     6 -> nothing
          7 -> group 1            8 -> group 3

      The explicit form is what catches a kernel that never claims anything
      at all, which the implicit form cannot distinguish from a kernel whose
      range is empty in this fixture.
  D3  `M > 1`. The document axis is split `M` ways and the writes collide,
      so the driver switches from a plain store to a global atomicAdd. The
      result must equal the `M == 1` result.
  D4  the PARTIAL PASS. With `full_pass = False` the driver reads whichever
      SIBLING is smaller and files the histogram under the RIGHT one. Gated
      against a host tally over the smaller child, at the right child's
      slot.
  D5  the 8-bit path's fixed-point writeback divides by the scale. Run at
      scale 4, not 1, because at 1 the division is a no-op and deleting it
      leaves every gate green -- which was measured, not assumed. Four is a
      power of two over integer stats, so the round trip is exact and D1
      stays an exact comparison.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    pointwise_one_byte_fixed_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_FLOAT_BLOCK,
)

#: Whether this build routes every one-byte width through the 8-bit
#: fixed-point accumulator (`pointwise_one_byte_fixed_for`): its bounds
#: then widen to (15, 256] and it claims EVERY group, so any gate that
#: launches it TOGETHER with the float widths onto one buffer under an
#: atomicAdd writeback would double-count by construction.
comptime PW_ROUTED = pointwise_one_byte_fixed_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()
from gbdt.methods.kernel.pointwise_hist2_one_byte_templ import (
    compute_split_properties_nb_kernel,
)

comptime N_ROWS = 3000
comptime N_GROUPS = 4
comptime N_FEATURES = 4 * N_GROUPS


def _launch[
    ou: MutOrigin,
    of: MutOrigin,
    og: MutOrigin,
    oc: MutOrigin,
    ot: MutOrigin,
    ow: MutOrigin,
    oi: MutOrigin,
    op: MutOrigin,
    os: MutOrigin, //,
    bits: Int,
    full: Bool,
    m: Int,
](
    ctx: DeviceContext,
    p_off: MutPointer[UInt32, ou],
    p_ffi: MutPointer[UInt32, of],
    p_folds: MutPointer[UInt32, og],
    p_ci: MutPointer[UInt32, oc],
    p_tgt: MutPointer[Float32, ot],
    p_wt: MutPointer[Float32, ow],
    p_idx: MutPointer[UInt32, oi],
    p_part: MutPointer[UInt32, op],
    p_sums: MutPointer[Float32, os],
    total_bin_features: Int,
    scale: Float32,
    gx: Int,
    gy: Int,
) raises:
    """One launch of the driver. A module-level function rather than a
    nested one because a closure here cannot capture `DeviceContext`.

    THE BLOCK IS PER BIT WIDTH, exactly as `run_compute_hist2_non_binary`
    resolves it: the 8-bit width at the route-keyed block, the float
    widths at their dispatch's. Launching a float width at the fixed
    route's wider block would run threads whose slice offsets fall off
    the scratch."""
    comptime nb_block = PW_HIST2_BLOCK if bits == 8 else (
        PW_HIST2_FLOAT_BLOCK
    )
    ctx.enqueue_function[compute_split_properties_nb_kernel[bits, full, m]](
        p_off, p_ffi, p_folds, Int32(N_FEATURES), p_ci, p_tgt, p_wt,
        p_idx, p_part, p_sums, Int32(total_bin_features), scale,
        grid_dim=(gx, gy, 1), block_dim=(nb_block, 1, 1),
    )


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ---- features ---------------------------------------------------
    var folds_h: List[UInt32] = [
        20, 25, 30, 32,        # group 0 -> 5-bit
        90, 100, 120, 128,     # group 1 -> 7-bit
        16, 10, 12, 14,        # group 2 -> 5-bit, and the reason it exists
        200, 150, 256, 180,    # group 3 -> 8-bit, the fixed-point path
    ]
    var first_fold_h = List[UInt32]()
    var cursor = UInt32(0)
    for f in range(N_FEATURES):
        first_fold_h.append(cursor)
        cursor += folds_h[f]
    var total_bin_features = Int(cursor)

    # `feature->Offset`: where this feature's GROUP starts in the cindex.
    # One UInt32 column per group, N_ROWS deep.
    var offset_h = List[UInt32]()
    for f in range(N_FEATURES):
        offset_h.append(UInt32((f // 4) * N_ROWS))

    # ---- the pool ---------------------------------------------------
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    for r in range(N_ROWS):
        indices_h.append(UInt32((r * 2654435761) % N_ROWS))
        target_h.append(Float32((r * 37) % 100 + 1))
        weight_h.append(Float32((r * 53) % 100 + 1001))

    var cindex_h = List[UInt32]()
    for _ in range(N_GROUPS * N_ROWS):
        cindex_h.append(0)
    for g in range(N_GROUPS):
        for r in range(N_ROWS):
            var word = UInt32(0)
            for k in range(4):
                var f = 4 * g + k
                var nf = Int(folds_h[f])
                var bin = (r * (7 + 3 * f) + 5 * k) % nf
                word |= UInt32(bin) << UInt32(24 - 8 * k)
            cindex_h[g * N_ROWS + r] = word

    # ---- device ------------------------------------------------------
    var d_off = ctx.enqueue_create_buffer[DType.uint32](N_FEATURES)
    var d_ffi = ctx.enqueue_create_buffer[DType.uint32](N_FEATURES)
    var d_folds = ctx.enqueue_create_buffer[DType.uint32](N_FEATURES)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_GROUPS * N_ROWS)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=offset_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ffi, src_ptr=first_fold_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_folds, src_ptr=folds_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

    # ONE partition covering everything, at a non-zero offset
    var part_off = 4
    var part_n = 2900
    var parts_h: List[UInt32] = [UInt32(part_off), UInt32(part_n)]
    var d_part = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=d_part, src_ptr=parts_h.unsafe_ptr())

    var n_out = total_bin_features * 2
    var d_sums = ctx.enqueue_create_buffer[DType.float32](n_out)
    var h_sums = ctx.enqueue_create_host_buffer[DType.float32](n_out)

    # ---- host answer -------------------------------------------------
    var want = List[Float32]()
    for _ in range(n_out):
        want.append(0.0)
    for r in range(part_off, part_off + part_n):
        var row = Int(indices_h[r])
        for g in range(N_GROUPS):
            var word = cindex_h[g * N_ROWS + row]
            for k in range(4):
                var f = 4 * g + k
                var bin = Int((word >> UInt32(24 - 8 * k)) & 255)
                var at = (Int(first_fold_h[f]) + bin) * 2
                want[at + 0] += weight_h[r]
                want[at + 1] += target_h[r]

    # ==================================================== D1 / D2 / D5
    # SCALE 4.0, NOT 1.0, AND THAT IS D5. Only the 8-bit path reads this --
    # the other three accumulate in float and ignore it -- and its writeback
    # is the one place in the driver that divides by it (DEVIATION 93).
    #
    # At scale 1.0 the division is a no-op, so deleting it changes nothing:
    # that was measured, by deleting it and watching every gate stay green.
    # Four is a power of two and the stats are integers, so `val * 4` is
    # exact in float32, `hist2_quantize` cannot move it (its fraction is
    # zero and the dither is below one), and the division back is exact --
    # which keeps D1 an EXACT comparison while making the division
    # observable.
    var scale = Float32(4.0)

    # ---- D2 explicit: each width alone, and what it claims ----------
    # Under the fixed one-byte route the 8-bit kernel claims every group;
    # the float widths keep their own bounds in any build.
    var expect_8: String
    comptime if PW_ROUTED:
        expect_8 = String("0,1,2,3")
    else:
        expect_8 = String("3")
    var expect_claims: List[String] = [
        String("0,2"), String(""), String("1"), expect_8,
    ]
    var widths: List[Int] = [5, 6, 7, 8]
    for wi in range(len(widths)):
        ctx.enqueue_memset(d_sums, Float32(0.0))
        if widths[wi] == 5:
            _launch[5, True, 1](
                ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                total_bin_features, scale, N_GROUPS, 1,
            )
        elif widths[wi] == 6:
            _launch[6, True, 1](
                ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                total_bin_features, scale, N_GROUPS, 1,
            )
        elif widths[wi] == 7:
            _launch[7, True, 1](
                ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                total_bin_features, scale, N_GROUPS, 1,
            )
        else:
            _launch[8, True, 1](
                ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                total_bin_features, scale, N_GROUPS, 1,
            )
        ctx.enqueue_copy(dst_buf=h_sums, src_buf=d_sums)
        ctx.synchronize()
        var claimed = String("")
        for g in range(N_GROUPS):
            var touched = False
            for k in range(4):
                var f = 4 * g + k
                var b0 = Int(first_fold_h[f]) * 2
                for j in range(Int(folds_h[f]) * 2):
                    if h_sums[b0 + j] != 0.0:
                        touched = True
            if touched:
                if claimed.byte_length() > 0:
                    claimed += ","
                claimed += String(g)
        if claimed != expect_claims[wi]:
            print(
                "FAIL D2: the", widths[wi], "-bit kernel claimed groups [",
                claimed, "], expected [", expect_claims[wi], "]",
            )
            failures += 1
        else:
            print(
                "  ok   D2", widths[wi], "-bit claims groups [", claimed,
                "]",
            )

    ctx.enqueue_memset(d_sums, Float32(0.0))
    _launch[5, True, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                          d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                          d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                          total_bin_features, scale, N_GROUPS, 1)
    _launch[6, True, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                          d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                          d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                          total_bin_features, scale, N_GROUPS, 1)
    _launch[7, True, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                          d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                          d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                          total_bin_features, scale, N_GROUPS, 1)
    _launch[8, True, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                          d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                          d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                          total_bin_features, scale, N_GROUPS, 1)
    ctx.enqueue_copy(dst_buf=h_sums, src_buf=d_sums)
    ctx.synchronize()

    var bad = 0
    var doubled = 0
    var zeroed = 0
    for k in range(n_out):
        if h_sums[k] != want[k]:
            if h_sums[k] == want[k] * 2.0:
                doubled += 1
            elif h_sums[k] == 0.0 and want[k] != 0.0:
                zeroed += 1
            if bad < 4:
                print(
                    "     D1 cell", k, "(bin-feature", k // 2, "stat",
                    k % 2, "): got", h_sums[k], "want", want[k],
                )
            bad += 1
    if bad != 0:
        print(
            "FAIL D1/D2: --", bad, "of", n_out, "cells wrong (", doubled,
            "exactly doubled,", zeroed, "zeroed ).",
        )
        if doubled != 0:
            print(
                "       DOUBLED means two bit widths both claimed a block:"
                " the dispatch bounds overlap.",
            )
        if zeroed != 0:
            print(
                "       ZEROED means no bit width claimed a block. If it is"
                " group 2 (16 folds), the `15` in"
                " `lowerBound = BITS > 5 ? upperBound / 2 : 15` became a"
                " `16`.",
            )
        failures += 1
    else:
        print(
            "  ok   D1/D2 --", n_out,
            "cells exact; 4 kernels x 4 groups, each group claimed once",
        )

    # ================================================================ D3
    # M = 4: the document axis splits four ways and the writes collide.
    # ROUTE-AWARE: at M > 1 the writeback is an atomicAdd, so under the
    # fixed route -- where the 8-bit kernel claims every group -- adding
    # the float widths on top would double-count groups 0/1/2 by
    # construction. The shipped dispatch under the route IS the single
    # 8-bit launch, and it alone covers every group, so `want` is the
    # same value either way. (D1 above tolerates the overlap because its
    # M == 1 writeback is a plain store and both claimants store the
    # same exact cells.)
    ctx.enqueue_memset(d_sums, Float32(0.0))
    comptime if not PW_ROUTED:
        _launch[5, True, 4](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                              d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                              d_tgt.unsafe_ptr(),
                              d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                              d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                              total_bin_features, scale, N_GROUPS * 4, 1)
        _launch[6, True, 4](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                              d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                              d_tgt.unsafe_ptr(),
                              d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                              d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                              total_bin_features, scale, N_GROUPS * 4, 1)
        _launch[7, True, 4](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                              d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                              d_tgt.unsafe_ptr(),
                              d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                              d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                              total_bin_features, scale, N_GROUPS * 4, 1)
    _launch[8, True, 4](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                          d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                          d_part.unsafe_ptr(), d_sums.unsafe_ptr(),
                          total_bin_features, scale, N_GROUPS * 4, 1)
    ctx.enqueue_copy(dst_buf=h_sums, src_buf=d_sums)
    ctx.synchronize()

    var bad3 = 0
    var worst = Float32(0.0)
    for k in range(n_out):
        var d = h_sums[k] - want[k]
        if d < 0:
            d = -d
        var tol = want[k] * Float32(1e-5)
        if d > tol and d > Float32(0.01):
            if bad3 < 3:
                print(
                    "     D3 cell", k, ": got", h_sums[k], "want", want[k],
                )
            bad3 += 1
        if d > worst:
            worst = d
    if bad3 != 0:
        print(
            "FAIL D3: --", bad3, "of", n_out,
            "cells wrong at M=4. A result 4x too large means the document"
            " blocks are not partitioning; a short one means the atomic"
            " path is not taken.",
        )
        failures += 1
    else:
        print(
            "  ok   D3 -- M=4 matches M=1 (worst absolute drift", worst,
            "over four partial sums)",
        )

    # ================================================================ D4
    # partial pass: two siblings, gridDim.y = 2. Parts 0,1 are the left
    # children and 2,3 the right; the driver must READ the smaller of each
    # pair and WRITE under the right one.
    var p2: List[UInt32] = [
        UInt32(4), UInt32(600),      # part 0 : left of pair 0,  SMALLER
        UInt32(700), UInt32(1400),   # part 1 : left of pair 1,  larger
        UInt32(900), UInt32(1500),   # part 2 : right of pair 0, larger
        UInt32(2200), UInt32(700),   # part 3 : right of pair 1, SMALLER
    ]
    var d_part2 = ctx.enqueue_create_buffer[DType.uint32](8)
    ctx.enqueue_copy(dst_buf=d_part2, src_ptr=p2.unsafe_ptr())

    var hist_line = total_bin_features * 2
    var d_sums2 = ctx.enqueue_create_buffer[DType.float32](hist_line * 4)
    var h_sums2 = ctx.enqueue_create_host_buffer[DType.float32](
        hist_line * 4
    )
    ctx.enqueue_memset(d_sums2, Float32(0.0))

    _launch[5, False, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                           d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                           d_part2.unsafe_ptr(), d_sums2.unsafe_ptr(),
                           total_bin_features, scale, N_GROUPS, 2)
    _launch[6, False, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                           d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                           d_part2.unsafe_ptr(), d_sums2.unsafe_ptr(),
                           total_bin_features, scale, N_GROUPS, 2)
    _launch[7, False, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                           d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                           d_part2.unsafe_ptr(), d_sums2.unsafe_ptr(),
                           total_bin_features, scale, N_GROUPS, 2)
    _launch[8, False, 1](ctx, d_off.unsafe_ptr(), d_ffi.unsafe_ptr(),
                          d_folds.unsafe_ptr(), d_ci.unsafe_ptr(),
                          d_tgt.unsafe_ptr(),
                           d_wt.unsafe_ptr(), d_idx.unsafe_ptr(),
                           d_part2.unsafe_ptr(), d_sums2.unsafe_ptr(),
                           total_bin_features, scale, N_GROUPS, 2)
    ctx.enqueue_copy(dst_buf=h_sums2, src_buf=d_sums2)
    ctx.synchronize()

    # host: pair 0's smaller is part 0 (600 rows), filed at slot 2;
    #       pair 1's smaller is part 3 (700 rows), filed at slot 3
    var bad4 = 0
    var pairs: List[Int] = [0, 1]
    var smaller_off: List[Int] = [4, 2200]
    var smaller_n: List[Int] = [600, 700]
    for pi in range(2):
        var slot = 2 | pairs[pi]
        var w4 = List[Float32]()
        for _ in range(hist_line):
            w4.append(0.0)
        for r in range(smaller_off[pi], smaller_off[pi] + smaller_n[pi]):
            var row = Int(indices_h[r])
            for g in range(N_GROUPS):
                var word = cindex_h[g * N_ROWS + row]
                for k in range(4):
                    var f = 4 * g + k
                    var bin = Int((word >> UInt32(24 - 8 * k)) & 255)
                    var at = (Int(first_fold_h[f]) + bin) * 2
                    w4[at + 0] += weight_h[r]
                    w4[at + 1] += target_h[r]
        for k in range(hist_line):
            if h_sums2[slot * hist_line + k] != w4[k]:
                if bad4 < 3:
                    print(
                        "     D4 pair", pi, "slot", slot, "cell", k,
                        ": got", h_sums2[slot * hist_line + k], "want",
                        w4[k],
                    )
                bad4 += 1
    if bad4 != 0:
        print(
            "FAIL D4: --", bad4,
            "cells wrong on the partial pass. The driver must read"
            " whichever SIBLING is smaller and file it under the RIGHT"
            " one; pair 0's smaller child is on the left and pair 1's is"
            " on the right, so a version that always takes one side gets"
            " exactly half of this right.",
        )
        failures += 1
    else:
        print(
            "  ok   D4 -- partial pass reads the smaller sibling and files"
            " under the right one, both directions",
        )

    _ = d_off^
    _ = d_ffi^
    _ = d_folds^
    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_part^
    _ = d_part2^
    _ = d_sums^
    _ = d_sums2^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise one-byte driver: D1-D5 pass")
