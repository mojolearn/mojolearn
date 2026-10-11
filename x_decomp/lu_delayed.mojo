# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lu-delayed-update (2026-10-11): MOJOLEARN_IDN_LU_DELAYED_UPDATE.
CANDIDATE, DEFAULT OFF, IDENTICAL only (idea B2 of
~/mojolearn-evidence/ideas-dense-linalg-20261011.md).

STRUCTURAL REASON. Main's IDENTICAL blocked LU (`launch_lu`, 32-column
panels) runs `lu_trail_rb_kernel` over the WHOLE trailing square after every
32-column panel: 256 read+write passes over the square at n = 8192 (~46 GB of
traffic), every pass a k = 32 scalar chain per cell. Here the columns are
taken in outer blocks of LDU_NB = 256 (LAPACK getrf's right-looking delayed
update, the schedule FAST+Apple's LU_FAST_MMA runs, x_decomp/lu_fast_mma.mojo):
the 32-column inner panel is UNCHANGED (`lu_fast_panel`, one launch per
column), its rank-32 update touches only the block's own remaining columns,
and the square right of and below the block takes ONE identical-GEMM
update at k = 256 (gemm/block_update.mojo `blk_gemm_sub`). The trailing
square is read and written once per 256 columns instead of per 32 (~5.7 GB),
and the bulk of the 2n^3/3 flops runs in the identical GEMM's tiled plans.
`MOJOLEARN_IDN_LU_NB128` lost 1.8x (docs/TOMBSTONES.md grid-act-16) because
it WIDENED THE PANEL to 128, quadrupling the panel's own rank-1 work and
keeping one launch per column; this keeps the panel at 32 and only delays the
trailing update, so the panel cost is main's. Expected: the trailing part at
the GEMM's rate (366 GFLOP: ~137 ms NV L40S / ~76 ms AMD MI325X at today's
identical GEMM) against the rb kernel's share; if the rb kernel already runs
above the GEMM's rate on NVIDIA this loses on NV and wins on AMD (measure:
lu-solve synthetic nv+amd, ID lines on both).

THE ORDER (device below and `lu_delayed_host` in x_decomp/host.mojo, the host column, spell it
once each; LDP = 32, LDU_NB = 256):

  for each outer block [K0, K1), K1 = min(K0 + LDU_NB, n):
    for each panel [k0, k1) of it, k1 = min(k0 + LDP, K1):
      1. the panel: `lu_fast_panel`'s statements (pivot over the current
         column, largest |ftz(a)|, ties to the LOWEST row; the swap on columns
         0 .. k1-1; d = ftz(a[k, k]); act[k] = d != 0, info at the first zero;
         l_i = div0(a[i, k], d); the panel's later columns
         a[i, j] = ftz(fma(-l_i, ftz(a[k, j]), ftz(a[i, j]))));
      2. every column j >= k1: the panel's swaps in order; for j < K1 only,
         the U rows k0+1 .. k1-1: acc = ftz(a[k, j]), kp ascending in
         [k0, k) with act[kp]: acc = ftz(fma(-a[k, kp], ftz(a[kp, j]),
         ftz(acc))) (`lfm_swaps_trsm_kernel`, = `lu_swaps_trsm_kernel`'s chain);
      3. rows k1 .. n-1 x columns k1 .. K1-1: BU(k0, k1 - k0).
    if K1 < n, for the columns j >= K1:
      4. for each 32-row chunk [c, ce) of [K0, K1): the U rows c+1 .. ce-1
         (step 2's chain over kp in [c, k)), then rows ce .. K1-1 x columns
         K1 .. n-1: BU(c, ce - c);
      5. rows K1 .. n-1 x columns K1 .. n-1: BU(K0, K1 - K0).

  BU(p0, k) on cell (i, j): G = the identical GEMM's cell over the k-vector
  pair (l_p = a[i, p0 + p], zeroed where act[p0 + p] == 0; u_p =
  a[p0 + p, j]): leaves of contract_leaf_size(k) ascending, each the chain
  g = ftz(fma(ftz(l_p), ftz(u_p), g)) from +0.0, folded by the contract's
  balanced tree (k = 256 is two 128-leaves and one add); then
  a[i, j] = ftz(ftz(a[i, j]) - ftz(G)) (gemm/block_update.mojo's statement).

BITS CHANGE against main's per-32-step trailing chain: NVIDIA, AMD and the
host column (`HostExec.lu` under the same define) move together. Pivot
choices move only where a rounding difference flips a strict compare. A
zero-pivot step's L column is masked to +0.0 in the GEMM (main skipped the
step); an inf/NaN U entry against that +0.0 gives NaN where main skipped:
the same on every column, and only on a singular input (info != 0).
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.block_update import blk_gemm_sub, blk_workspace_floats
from gemm.contract import OP_NN
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.lu_delayed_ctl import IDN_LU_DELAYED_UPDATE, LDP, LDU_NB
from x_decomp.lu_fast import LFS_TPB, LU_FAST_STEP1, lfs_blocks, lu_fast_panel
from x_decomp.lu_fast_mma import lfm_swaps_trsm_kernel

def ldu_workspace_floats(n: Int) -> Int:
    """The largest `blk_workspace_floats` over every BU call of an n x n
    factorization (each call's GEMM share is sized by its own helper)."""
    var need = 1
    var K0 = 0
    while K0 < n:
        var K1 = min(K0 + LDU_NB, n)
        var k0 = K0
        while k0 < K1:
            var k1 = min(k0 + LDP, K1)
            if K1 > k1:
                need = max(need, blk_workspace_floats(n - k1, K1 - k1, k1 - k0))
            k0 = k1
        if K1 < n:
            var c = K0
            while c < K1:
                var ce = min(c + LDP, K1)
                if K1 > ce:
                    need = max(need, blk_workspace_floats(K1 - ce, n - K1, ce - c))
                c = ce
            need = max(need, blk_workspace_floats(n - K1, n - K1, K1 - K0))
        K0 = K1
    return need


def _ldu_bu(
    ctx: DeviceContext, a: F32Ptr, act: F32Ptr, n: Int, r0: Int, c0: Int, p0: Int, m: Int, nc: Int, k: Int,
    mut ws: DeviceBuffer[DType.float32],
) raises:
    """BU(p0, k) on rows [r0, r0 + m) x columns [c0, c0 + nc) of `a`."""
    if m <= 0 or nc <= 0 or k <= 0:
        return
    # GUARD=False never reads `info`: any Int32-typed pointer serves
    var no_info = I32Ptr(unsafe_from_address=Int(a))
    blk_gemm_sub[False](
        ctx, a, n, r0, c0,
        a, n, r0, p0,
        a, n, p0, c0,
        m, nc, k, OP_NN, False, ws, no_info, act + p0, True,
    )


def _ldu_cols(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, act: F32Ptr, k0: Int, k1: Int, jlo: Int, jtri: Int, n: Int,
    swaps: Bool,
) raises:
    """Step 2 / step 4's column statements, one thread per column [jlo, n)."""
    if n <= jlo:
        return
    ctx.enqueue_function[lfm_swaps_trsm_kernel](
        a, piv, act, Int32(k0), Int32(k1), Int32(jlo), Int32(jtri), Int32(n), Int32(n), Int32(1 if swaps else 0),
        grid_dim=lfs_blocks(n - jlo), block_dim=LFS_TPB,
    )


def lu_delayed_factor(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, act: F32Ptr,
    p0: F32Ptr, p1: F32Ptr, pa: F32Ptr, pb: F32Ptr, n: Int, maxb: Int,
) raises:
    """The whole factorization in THE ORDER above. `info` initialized by the
    caller; p0/p1/pa/pb are `launch_lu`'s LU_FAST_STEP1 panel scratch. Waits
    for its own launches before returning (its GEMM workspace dies here)."""
    comptime assert not IDN_LU_DELAYED_UPDATE or LU_FAST_STEP1, (
        "IDN_LU_DELAYED_UPDATE: the device panel is LU_FAST_STEP1's lu_fast_panel (the host twin spells its condition)"
    )
    comptime assert LDU_NB % LDP == 0, "IDN_LU_DELAYED_UPDATE: the outer block is whole 32-column panels"
    var ws = ctx.enqueue_create_buffer[DType.float32](ldu_workspace_floats(n))
    var K0 = 0
    while K0 < n:
        var K1 = min(K0 + LDU_NB, n)
        var k0 = K0
        while k0 < K1:
            var k1 = min(k0 + LDP, K1)
            lu_fast_panel(ctx, a, piv, info, act, p0, p1, pa, pb, k0, k1, n, maxb)
            _ldu_cols(ctx, a, piv, act, k0, k1, k1, K1, n, True)
            _ldu_bu(ctx, a, act, n, k1, k1, k0, n - k1, K1 - k1, k1 - k0, ws)
            k0 = k1
        if K1 < n:
            var c = K0
            while c < K1:
                var ce = min(c + LDP, K1)
                _ldu_cols(ctx, a, piv, act, c, ce, K1, n, n, False)
                _ldu_bu(ctx, a, act, n, ce, K1, c, K1 - ce, n - K1, ce - c, ws)
                c = ce
            _ldu_bu(ctx, a, act, n, K1, K1, K0, n - K1, n - K1, K1 - K0, ws)
        K0 = K1
    ctx.synchronize()
    _ = ws^
