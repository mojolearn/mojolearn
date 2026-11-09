# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-decomp (2026-10-04): FAST + Apple candidates.

(LLE_FAST_DEV_LU, the device LU of LocallyLinearEmbedding's shift-invert factor, was
deleted 2026-11-08: HOLD-quality; see the TOMBSTONE below.)

RSVD_FAST_DIRECT_IN (FAST + Apple default; `-D MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF`
rolls back): randomized_svd's
input goes up from the caller's own buffer into a pooled device matrix
(`_M._on_device` + `x_decomp_dev_upload`, the random projections' and
KernelPCA's resident pattern) instead of `_M.from_input`'s copy into a fresh
`array.array` (880 MB of fresh host pages at the board's 1,000,000 x 220)
that the first product then uploads. The host finiteness refusal
(`_host_all_finite`) is unchanged. Same words reach the device: no bit moves.

It is read through `x_decomp_w4_flags()` (bit 2 RSVD_FAST_DIRECT_IN; bit 1,
LLE_FAST_DEV_LU, was deleted 2026-11-08), registered on every GPU build so a
quality capture can prove which arm is installed.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL

# TOMBSTONE: MOJOLEARN_LLE_FAST_DEV_LU (HOLD-quality: lle trustworthiness_k15 istella 0.8851 -> 0.8615, taxi 0.8662 ->
# 0.8410 vs FAST main) deleted 2026-11-08 by ed9ba6e7a1; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_DEV_LU.patch
# TOMBSTONE: MOJOLEARN_LLE_FAST_NULL_CANON (DROPPED-quality) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: a canonical LLE answer when the null space is wider than n_components (all of N, then its nc directions of most data variance); trustworthiness down on lle taxi and istella (verdicts batch 6).
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_NULL_CANON.patch; record in docs/TOMBSTONES.md.
#: FAST Apple default, measured source e9d72edb5 (2026-10-04).
#: M3 afc_ab_def, full board size, 1 run per arm: randomized-svd istella
#: 517.3 -> 501.3 ms, taxi 200.8 -> 198.5 ms; relative_reconstruction_error
#: equal (istella 2.3594584442703e-4 -> 2.3594584442301e-4, taxi
#: 0.0271967625509 -> 0.0271967625509). ACCEPT.
#: `-D MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF` restores `_M.from_input`'s host copy.
comptime RSVD_FAST_DIRECT_IN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF"]()
)

#: R1, lane fg-pca (2026-10-09), IDENTICAL default on every GPU column;
#: rollback `-D MOJOLEARN_IDN_RSVD_DIRECT_IN_OFF`. The same route as
#: RSVD_FAST_DIRECT_IN above (bit 2, `_rsvd_direct_input`): randomized_svd's
#: input goes from the caller's own buffer straight into a pooled device
#: matrix (`x_decomp_dev_upload`), instead of `_M.from_input`'s copy of all
#: n*d floats into a fresh host `array.array` (880 MB of fresh host pages and
#: a full memcpy at 1M x 220, each extra n*d*4 bytes of host traffic at any
#: shape) that the first product then uploads. The host finiteness refusal
#: (`_host_all_finite`) runs first, as before; sparse, wide (host-transposed)
#: and host-binding inputs keep `_M.from_input`. Cost: -1 host copy of X per
#: call. Bits: none (a copy is not arithmetic; the same words reach the
#: device, and `_RES_MIN` = 1 already sent every product to the device).
comptime RSVD_IDN_DIRECT_IN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_RSVD_DIRECT_IN_OFF"]()
)


# TOMBSTONE: MOJOLEARN_LLE_FAST_DEV_LU (w4_copy_kernel, dev_lu_aux_run: its device LU) deleted 2026-11-08 by ed9ba6e7a1; code
# recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_DEV_LU.patch
