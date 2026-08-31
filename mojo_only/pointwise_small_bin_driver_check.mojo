# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for the BINARY and HALF-BYTE drivers.

`gbdt/methods/kernel/pointwise_hist2_binary.mojo` and
`..._half_byte.mojo`: same accumulator, same loop choice, two writebacks.

  E1  HALF-BYTE, per (feature, fold, stat), against a host tally. Features
      are given DIFFERENT fold counts -- 16, 5, 12, 3, 9, 16, 7, 11 -- so
      the fold guard matters.

  E2  THE OVERRUN PRECONDITION, and read what it does NOT claim. The
      half-byte kernel guards its writeback twice -- `fold <
      feature[fid].Folds` and `abs(result) > 1e-20` -- and the two are
      REDUNDANT on well-formed data. A 5-fold feature's bins 5-15 are
      empty, so the fold guard suppresses a write of an exact zero that the
      epsilon guard would have suppressed anyway. Measured:

          fold guard removed      every gate green
          epsilon guard removed   every gate green
          BOTH removed            E1, 22 of 158 cells wrong

      So E2 does not gate the fold guard, and nothing here can while both
      are present. What it DOES gate is the precondition that makes the
      both-removed sabotage bite: that features overrun into non-zero
      neighbour cells at all. Without that this fixture could not tell a
      correct writeback from an unguarded one under any sabotage.

  E3  BINARY, per (feature, stat). Thirty-two one-bit features, one output
      cell each at `FirstFoldIndex * 2 + w` with no fold term, recovered by
      summing the eight nibble values whose bit is clear.

  E4  the binary and half-byte kernels must AGREE where they overlap. A
      binary feature is one bit of a nibble and a half-byte feature is the
      nibble; so for any nibble, the binary feature at bit b must equal the
      sum of the half-byte cells whose fold has bit b clear. Two independent
      paths through the same accumulator, and the strongest check available
      without a second implementation.

  E5  `M > 1` for both, against their `M == 1` results.
"""

from max.gpu.host import DeviceContext

from gbdt.methods.kernel.pointwise_hist2_half_byte_template import (
    PW_HB_BLOCK,
)
from gbdt.methods.kernel.pointwise_hist2_binary import (
    compute_split_properties_b_kernel,
)
from gbdt.methods.kernel.pointwise_hist2_half_byte import (
    compute_split_properties_half_byte_kernel,
)

comptime N_ROWS = 3000


def _launch_hb[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, o4: MutOrigin,
    o5: MutOrigin, o6: MutOrigin, o7: MutOrigin, o8: MutOrigin,
    o9: MutOrigin, //,
    full: Bool, m: Int,
](
    ctx: DeviceContext,
    p_off: MutPointer[UInt32, o1],
    p_ffi: MutPointer[UInt32, o2],
    p_folds: MutPointer[UInt32, o3],
    f_count: Int,
    p_ci: MutPointer[UInt32, o4],
    p_tgt: MutPointer[Float32, o5],
    p_wt: MutPointer[Float32, o6],
    p_idx: MutPointer[UInt32, o7],
    p_part: MutPointer[UInt32, o8],
    p_sums: MutPointer[Float32, o9],
    total_bin_features: Int,
    gx: Int,
) raises:
    ctx.enqueue_function[
        compute_split_properties_half_byte_kernel[full, m]
    ](
        p_off, p_ffi, p_folds, Int32(f_count), p_ci, p_tgt, p_wt, p_idx,
        p_part, p_sums, Int32(total_bin_features),
        grid_dim=(gx, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
    )


def _launch_b[
    o1: MutOrigin, o2: MutOrigin, o4: MutOrigin, o5: MutOrigin,
    o6: MutOrigin, o7: MutOrigin, o8: MutOrigin, o9: MutOrigin, //,
    full: Bool, m: Int,
](
    ctx: DeviceContext,
    p_off: MutPointer[UInt32, o1],
    p_ffi: MutPointer[UInt32, o2],
    f_count: Int,
    p_ci: MutPointer[UInt32, o4],
    p_tgt: MutPointer[Float32, o5],
    p_wt: MutPointer[Float32, o6],
    p_idx: MutPointer[UInt32, o7],
    p_part: MutPointer[UInt32, o8],
    p_sums: MutPointer[Float32, o9],
    total_bin_features: Int,
    gx: Int,
) raises:
    ctx.enqueue_function[compute_split_properties_b_kernel[full, m]](
        p_off, p_ffi, Int32(f_count), p_ci, p_tgt, p_wt, p_idx, p_part,
        p_sums, Int32(total_bin_features),
        grid_dim=(gx, 1, 1), block_dim=(PW_HB_BLOCK, 1, 1),
    )


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ---- the pool: one group of eight nibbles ------------------------
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    var cindex_h = List[UInt32]()
    var hb_folds: List[UInt32] = [16, 5, 12, 3, 9, 16, 7, 11]
    for r in range(N_ROWS):
        indices_h.append(UInt32((r * 2654435761) % N_ROWS))
        target_h.append(Float32((r * 37) % 100 + 1))
        weight_h.append(Float32((r * 53) % 100 + 1001))
        var word = UInt32(0)
        for j in range(8):
            # bins inside each feature's OWN fold count, so E2's
            # beyond-the-guard cells are genuinely untouched by their owner
            var nib = (r * (7 + 3 * j) + 2 * j) % Int(hb_folds[j])
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

    var part_off = 6
    var part_n = 2800
    var parts_h: List[UInt32] = [UInt32(part_off), UInt32(part_n)]
    var d_part = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=d_part, src_ptr=parts_h.unsafe_ptr())

    # ================================================================ E1
    var hb_ffi = List[UInt32]()
    var cur = UInt32(0)
    for j in range(8):
        hb_ffi.append(cur)
        cur += hb_folds[j]
    var hb_total = Int(cur)
    var hb_off = List[UInt32]()
    for _ in range(8):
        hb_off.append(0)

    var d_hoff = ctx.enqueue_create_buffer[DType.uint32](8)
    var d_hffi = ctx.enqueue_create_buffer[DType.uint32](8)
    var d_hfolds = ctx.enqueue_create_buffer[DType.uint32](8)
    ctx.enqueue_copy(dst_buf=d_hoff, src_ptr=hb_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hffi, src_ptr=hb_ffi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_hfolds, src_ptr=hb_folds.unsafe_ptr())

    var n_hb = hb_total * 2
    var d_hs = ctx.enqueue_create_buffer[DType.float32](n_hb)
    var h_hs = ctx.enqueue_create_host_buffer[DType.float32](n_hb)

    var want_hb = List[Float32]()
    for _ in range(n_hb):
        want_hb.append(0.0)
    for r in range(part_off, part_off + part_n):
        var word = cindex_h[Int(indices_h[r])]
        for j in range(8):
            var nib = Int((word >> UInt32(28 - 4 * j)) & 15)
            var at = (Int(hb_ffi[j]) + nib) * 2
            want_hb[at + 0] += weight_h[r]
            want_hb[at + 1] += target_h[r]

    ctx.enqueue_memset(d_hs, Float32(0.0))
    _launch_hb[True, 1](
        ctx, d_hoff.unsafe_ptr(), d_hffi.unsafe_ptr(),
        d_hfolds.unsafe_ptr(), 8, d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(),
        d_wt.unsafe_ptr(), d_idx.unsafe_ptr(), d_part.unsafe_ptr(),
        d_hs.unsafe_ptr(), hb_total, 1,
    )
    ctx.enqueue_copy(dst_buf=h_hs, src_buf=d_hs)
    ctx.synchronize()

    var bad1 = 0
    for k in range(n_hb):
        if h_hs[k] != want_hb[k]:
            if bad1 < 3:
                print(
                    "     E1 cell", k, ": got", h_hs[k], "want",
                    want_hb[k],
                )
            bad1 += 1
    if bad1 != 0:
        print("FAIL E1: --", bad1, "of", n_hb, "half-byte cells wrong")
        failures += 1
    else:
        print("  ok   E1 --", n_hb, "half-byte cells exact")

    # ================================================================ E2
    # cells past a feature's fold count belong to the NEXT feature. Their
    # owner never writes them, so they must equal that owner's tally --
    # which E1 already established. What E2 adds is the precondition that
    # such cells EXIST and are non-zero, so the guard is actually load
    # bearing in this fixture.
    var guarded = 0
    for j in range(7):
        var beyond = 16 - Int(hb_folds[j])
        for b in range(beyond):
            var slot = Int(hb_ffi[j]) + Int(hb_folds[j]) + b
            if slot * 2 < n_hb and want_hb[slot * 2] != 0.0:
                guarded += 1
    if guarded == 0:
        print(
            "FAIL E2: no feature's 16-bin block overruns into a NON-ZERO"
            " neighbour cell, so the fold guard is inert in this fixture"
            " and E1 would pass without it. Give a feature fewer folds.",
        )
        failures += 1
    else:
        print(
            "  ok   E2 --", guarded,
            "cells lie past a feature's folds and hold a neighbour's"
            " non-zero value (the precondition; see the docstring for what"
            " this does and does not gate)",
        )

    # ================================================================ E3
    var b_ffi = List[UInt32]()
    var b_off = List[UInt32]()
    for f in range(32):
        b_ffi.append(UInt32(f))
        b_off.append(0)
    var d_boff = ctx.enqueue_create_buffer[DType.uint32](32)
    var d_bffi = ctx.enqueue_create_buffer[DType.uint32](32)
    ctx.enqueue_copy(dst_buf=d_boff, src_ptr=b_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bffi, src_ptr=b_ffi.unsafe_ptr())

    var n_b = 32 * 2
    var d_bs = ctx.enqueue_create_buffer[DType.float32](n_b)
    var h_bs = ctx.enqueue_create_host_buffer[DType.float32](n_b)

    var want_b = List[Float32]()
    for _ in range(n_b):
        want_b.append(0.0)
    for r in range(part_off, part_off + part_n):
        var word = cindex_h[Int(indices_h[r])]
        for fid in range(32):
            var g = fid // 4
            var nib = Int((word >> UInt32(28 - 4 * g)) & 15)
            if ((nib >> (3 - (fid & 3))) & 1) == 0:
                want_b[2 * fid + 0] += weight_h[r]
                want_b[2 * fid + 1] += target_h[r]

    ctx.enqueue_memset(d_bs, Float32(0.0))
    _launch_b[True, 1](
        ctx, d_boff.unsafe_ptr(), d_bffi.unsafe_ptr(), 32,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), d_part.unsafe_ptr(), d_bs.unsafe_ptr(),
        32, 1,
    )
    ctx.enqueue_copy(dst_buf=h_bs, src_buf=d_bs)
    ctx.synchronize()

    var bad3 = 0
    for k in range(n_b):
        if h_bs[k] != want_b[k]:
            if bad3 < 3:
                print(
                    "     E3 feature", k // 2, "stat", k % 2, ": got",
                    h_bs[k], "want", want_b[k],
                )
            bad3 += 1
    if bad3 != 0:
        print("FAIL E3: --", bad3, "of", n_b, "binary cells wrong")
        failures += 1
    else:
        print("  ok   E3 --", n_b, "binary cells exact")

    # ================================================================ E4
    # the two kernels' outputs must be consistent: binary feature `fid` is
    # the sum of the half-byte cells of nibble `fid/4` whose fold has bit
    # `3 - (fid & 3)` clear. Two paths through one accumulator.
    var bad4 = 0
    for fid in range(32):
        var g = fid // 4
        var bit = 3 - (fid & 3)
        for w in range(2):
            var acc = Float32(0.0)
            for fold in range(Int(hb_folds[g])):
                if ((fold >> bit) & 1) == 0:
                    acc += h_hs[(Int(hb_ffi[g]) + fold) * 2 + w]
            var got = h_bs[2 * fid + w]
            var d = got - acc
            if d < 0:
                d = -d
            if d > Float32(0.5):
                if bad4 < 3:
                    print(
                        "     E4 feature", fid, "stat", w, ": binary",
                        got, "vs half-byte sum", acc,
                    )
                bad4 += 1
    if bad4 != 0:
        print(
            "FAIL E4: --", bad4,
            "cells where the binary and half-byte readings of the SAME"
            " accumulator disagree",
        )
        failures += 1
    else:
        print(
            "  ok   E4 -- binary and half-byte readings agree on all 64"
            " cells",
        )

    # ================================================================ E5
    ctx.enqueue_memset(d_hs, Float32(0.0))
    _launch_hb[True, 4](
        ctx, d_hoff.unsafe_ptr(), d_hffi.unsafe_ptr(),
        d_hfolds.unsafe_ptr(), 8, d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(),
        d_wt.unsafe_ptr(), d_idx.unsafe_ptr(), d_part.unsafe_ptr(),
        d_hs.unsafe_ptr(), hb_total, 4,
    )
    ctx.enqueue_copy(dst_buf=h_hs, src_buf=d_hs)
    ctx.synchronize()
    var bad5 = 0
    for k in range(n_hb):
        var d = h_hs[k] - want_hb[k]
        if d < 0:
            d = -d
        if d > Float32(0.01):
            bad5 += 1
    ctx.enqueue_memset(d_bs, Float32(0.0))
    _launch_b[True, 4](
        ctx, d_boff.unsafe_ptr(), d_bffi.unsafe_ptr(), 32,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), d_part.unsafe_ptr(), d_bs.unsafe_ptr(),
        32, 4,
    )
    ctx.enqueue_copy(dst_buf=h_bs, src_buf=d_bs)
    ctx.synchronize()
    for k in range(n_b):
        var d = h_bs[k] - want_b[k]
        if d < 0:
            d = -d
        if d > Float32(0.01):
            bad5 += 1
    if bad5 != 0:
        print(
            "FAIL E5: --", bad5,
            "cells wrong at M=4 across the two kernels",
        )
        failures += 1
    else:
        print("  ok   E5 -- both kernels match M=1 at M=4")

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_part^
    _ = d_hoff^
    _ = d_hffi^
    _ = d_hfolds^
    _ = d_hs^
    _ = d_boff^
    _ = d_bffi^
    _ = d_bs^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise small-bin drivers: E1-E5 pass")
