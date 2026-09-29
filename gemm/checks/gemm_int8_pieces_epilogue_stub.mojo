# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A STAND-IN for lane/lowbit-int15's `gemm/checks/gemm_int15_epilogue.mojo`,
with the signature the orchestrator approved for `int15_store_cell`, until
that file is on origin. The four-product kernel's FUSED form calls it at
its last step; the kernel itself states no float rule.

What it computes is the timing harness's stand-in recombination
(`bench/gemm_lowbit_price_main.mojo`, `_pieces_recombine_probe_kernel`),
cell for cell: the three sums in Int64 (`HH 2^14 + MID 2^7 + LL`), the
backend's conversion to float32, one multiply by `2^(ea[i] + eb[j])`, the
flush. It is NOT the fifteen-bit profile's pinned seam. Because it is the
harness's own stand-in, the fused launch must give the two-launch form's
digest, bit for bit, and the gate checks exactly that.

When lane/lowbit-int15's file lands, `gemm_int8_mma_tuned.mojo` imports
`int15_store_cell` from it instead, and this file goes.

Lane lane/lowbit-mma-speed, 2026-09-29.
"""

from checks.numerics import ftz, identical_mul, pow2_f32


@always_inline
def int15_store_cell(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    hh: Int32,
    mid: Int32,
    ll: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """One cell, masked to the output: the stand-in recombination of its
    three sums, scaled by the sum of the two row exponents."""
    if i >= m or j >= n:
        return
    var v = Int64(hh) * Int64(16384) + Int64(mid) * Int64(128) + Int64(ll)
    var e = Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j))
    c.unsafe_store(i * n + j, ftz(identical_mul(Float32(v), pow2_f32(e))))
