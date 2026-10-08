# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 128-wide blocked Cholesky of lane gap-linalg (2026-10-08;
docs/plans/gaps-2026-10-08.md section 3.3). CANDIDATE, default off:
`-D MOJOLEARN_IDN_CHOL_NB128` on an IDENTICAL build (NVIDIA, AMD and the
host column together).

What changes with the define: the profile's panel width (`potrf.CHOL_NB_PINNED`
and `chol_oracle.CHOL_HOST_NB_PINNED`) is 128 instead of 32, and the device
driver is `potrf._potrf_lower_blocked` instead of the 32-wide strip schedule:

  * the diagonal block factored by `panel_factor_guarded_kernel` (one block,
    columns serial, the panel's own columns' chains: the v1 statements at
    w = 128);
  * the panel solve `L21 = A21 L11^{-T}` by `trsm_panel_guarded_kernel` (one
    thread per trailing row, c ascending, k ascending: the v1 statements);
  * the trailing update `A22 -= L21 L21^T` through `gemm/block_update.mojo`
    in column blocks of CB_COLS, rows on or below the diagonal only (about
    half the square's products), every cell the identical GEMM's one-leaf
    chain at k = 128 then one subtraction (`block_update`'s header states
    it; the host column spells the same chain through `gemm_oracle_cell`
    at `contract_leaf_size(128)`);
  * no wait inside the loop: every kernel after a failed panel returns at
    once on `info`, one readback at the end (the strip route's rule), so a
    failure leaves LAPACK's partial factor exactly as the per-panel-wait
    loop does (it stops after the failing panel's factor kernel).

BITS CHANGE against v1 (DEVIATION 1630 called a wider panel "a v2 decision
rather than a free one"): a cell's trailing sum is now bracketed per 128
columns instead of per 32, with a 128-long fma chain where v1 had four
32-long chains and three extra subtractions. The three columns move
together, so the cross-vendor claim is kept; the A/B on nv and amd decides
whether the define becomes the default (then the profile string moves to
v2).

Why 128 and not 256: 128 is `CONTRACT_K_LEAF_MIN`, the widest panel whose
trailing product is still ONE leaf (a plain ascending chain, statable in
one sentence and replayed by a serial loop on the host); at 256 the GEMM
would fold two leaves per cell. Why column blocks of 512: the lower-only
update computes and discards the cells above the diagonal inside each
column block, CB_COLS / n_trail of the useful work (6% at n = 8192), and
every column block is one GEMM launch (16 a panel at n = 8192); a narrower
block wastes less and launches more. Both are size/cost reasoning, not a
board shape.
"""

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.block_update import blk_workspace_floats

#: The route (see the module note). IDENTICAL builds only; FAST keeps its own
#: Apple schedule and its honored hints.
comptime CHOL_IDN_NB128 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_CHOL_NB128"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: The panel width of the route: `CONTRACT_K_LEAF_MIN` (one GEMM leaf).
comptime CB_NB = 128
#: SCHEDULING: columns per trailing-update GEMM (bits do not depend on it:
#: the contract's batch-composition invariance).
comptime CB_COLS = 512


def chol_blocked_workspace_floats(n: Int) -> Int:
    """Floats `_potrf_lower_blocked` needs beside the matrix: the largest
    `blk_workspace_floats` over every (panel, column block) this `n` walks
    (sized by the GEMM's own helper, as `chol_workspace_floats` is: a
    workspace sized for one plan and run under another is an out-of-bounds
    write a small matrix will not show). Never below 1."""
    var need = 1
    var j0 = 0
    while j0 < n:
        var w = CB_NB
        if j0 + w > n:
            w = n - j0
        var n_trail = n - j0 - w
        var cb = 0
        while cb < n_trail:
            var cbw = CB_COLS
            if cb + cbw > n_trail:
                cbw = n_trail - cb
            var c = blk_workspace_floats(n_trail - cb, cbw, w)
            if c > need:
                need = c
            cb += cbw
        j0 += w
    return need
