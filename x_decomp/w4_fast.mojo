# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-decomp (2026-10-04): FAST + Apple candidates. No live
code remains (every candidate was deleted; the records below stay).

(LLE_FAST_DEV_LU, the device LU of LocallyLinearEmbedding's shift-invert factor, was
deleted 2026-11-08: HOLD-quality; see the TOMBSTONE below.)

(RSVD_FAST_DIRECT_IN and RSVD_IDN_DIRECT_IN, randomized_svd's direct input, and
their x_decomp_w4_flags entry were deleted 2026-11-08: superseded; see below.)
"""


# TOMBSTONE: MOJOLEARN_LLE_FAST_DEV_LU (HOLD-quality: lle trustworthiness_k15 istella 0.8851 -> 0.8615, taxi 0.8662 ->
# 0.8410 vs FAST main) deleted 2026-11-08 by ed9ba6e7a1; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_DEV_LU.patch
# TOMBSTONE: MOJOLEARN_LLE_FAST_NULL_CANON (DROPPED-quality) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: a canonical LLE answer when the null space is wider than n_components (all of N, then its nc directions of most data variance); trustworthiness down on lle taxi and istella (verdicts batch 6).
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_NULL_CANON.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_RSVD_FAST_DIRECT_IN (KEPT FAST + Apple default, superseded) and MOJOLEARN_IDN_RSVD_DIRECT_IN (KEPT IDENTICAL default, superseded) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch. Both sent randomized_svd's input
# from the caller's buffer to the device instead of a host copy (M3 e9d72edb5: istella
# 517.3 -> 501.3 ms). randomized_svd is now one driver (x_decomp/randomized_svd_owner.mojo)
# that always reads M in place and uploads it once, so there is no other route to select.
# TOMBSTONE: MOJOLEARN_LLE_FAST_DEV_LU (w4_copy_kernel, dev_lu_aux_run: its device LU) deleted 2026-11-08 by ed9ba6e7a1; code
# recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_DEV_LU.patch
