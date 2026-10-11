# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Switches and host-safe cells of lane xd-rr-one-block (2026-10-11, gap
program C3 + C4). NOT COMPILED, NOT VERIFIED, NOT MEASURED: the
orchestrator compiles every switch off and each on.

`MOJOLEARN_IDN_RR_ONE_BLOCK_D` (C3) deleted 2026-10-11 (loser; docs/TOMBSTONES.md).

`MOJOLEARN_IDN_SVD_TALL_LEAN` (C4, default off), two parts on
`linalg.svd`'s tall route (python/mojolearn/_linalg_impl.py `_svd_tall`):
  (c) `DevExec._svd_on`'s one-sided Jacobi rounds sit behind a sticky
      device convergence word (`rs_round_gated_kernel`): the sweeps enqueued
      after the converging one return at once instead of rotating nothing
      over the whole grid, so a readback covers SVD_LEAN_GROUP = 8 sweeps
      (was XD_SVD_GROUP = 4) at the cost of cheap no-op launches. No bit
      moves (a sweep after convergence was already an identity).
  (b) `orth_diag`'s second sliced-QR pass (U = orth(A V / s)) is skipped
      when the first pass's guarded R is diagonal to 2^-20 relative
      (`orth_lean_skip`): the first pass already left Q orthonormal to that
      tolerance times float32 rounding, so the second pass only rounds again.
      BITS CHANGE when it skips (U only; S, V never), the same decision on
      NVIDIA, AMD and the host (`HostExec._orth_passes`): exact compares of
      the same words, no fold.
  (a) of the idea (skip the QR inside `_svd_on` because its input is a
      TSQR's R) does not apply: `_svd_tall` hands `Kit.svd` the tall A
      itself, so that QR is the tall factorization, not a redundant one.
"""
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul
from x_decomp.cells import F32Ptr

comptime _XD_LEAN_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()

# TOMBSTONE: MOJOLEARN_IDN_RR_ONE_BLOCK_D (loser) deleted 2026-10-11 by SHA_RR; code recoverable at SHA_RR^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_ONE_BLOCK_D.patch
#: C4 switch (x_decomp/device.mojo `_svd_on`, `orth_on_device_diag`;
#: x_decomp/host.mojo `_orth_passes`)
comptime IDN_SVD_TALL_LEAN = _XD_LEAN_IDN and is_defined["MOJOLEARN_IDN_SVD_TALL_LEAN"]()

#: sweeps between two convergence readbacks of the gated Jacobi (C4 c): a
#: well-conditioned R converges in ~8-12 sweeps, so one readback lands near
#: convergence and the up to 7 sweeps past it are gated no-op launches (each
#: returns on one word load); a size rule, not a shape rule.
comptime SVD_LEAN_GROUP = 8

#: C4 b: |R[i, j]| <= ORTH_LEAN_TOL min(|R[i, i]|, |R[j, j]|) for every i < j
#: means the first pass's input was orthogonal to 2^-20 (the svd's own
#: rotation tolerance), so the second pass is skipped
comptime ORTH_LEAN_TOL = Float32(9.5367431640625e-07)


def orth_lean_violates(R: F32Ptr, l: Int, i: Int, j: Int) -> Bool:
    """Cell (i, j), i < j, of the skip test on the guarded l x l R: True
    when R[i, j] is not negligible against the two pivots, or a pivot is 0
    (a guarded dependent column: the second pass then runs as before).
    Exact float32 compares of stored words: the same verdict everywhere."""
    var di = abs(ftz(R.unsafe_load(i * l + i)))
    var dj = abs(ftz(R.unsafe_load(j * l + j)))
    var lo = min(di, dj)
    if lo == Float32(0):
        return True
    var off = abs(ftz(R.unsafe_load(i * l + j)))
    return not (off <= ftz(identical_mul(ORTH_LEAN_TOL, lo)))


def orth_lean_skip(R: F32Ptr, l: Int) -> Bool:
    """The host column's verdict (`HostExec._orth_passes`): True when no
    cell violates (the device's `orth_lean_test_kernel` sets one word per
    violating cell; any order gives the same word)."""
    for i in range(l):
        for j in range(i + 1, l):
            if orth_lean_violates(R, l, i, j):
                return False
    if l == 1:
        return abs(ftz(R.unsafe_load(0))) != Float32(0)
    return l > 0
