# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL speed switches of lane fam2-cluster (2026-10-04), HDBSCAN.

Each is ON under `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` on every
vendor (their `_OFF` old paths were deleted 2026-11-15, lane
expdel2-11-cluster; docs/TOMBSTONES.md). None moves a bit: each one runs the
same kernels on the same values and removes waits, readbacks or whole
passes. The FAST Apple switches of `fast_apple.mojo` are not changed; where
a mechanism was written there first (lane af-hdbscan2), the IDENTICAL
switch below ORs into the same `comptime if`.
"""


from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (its term here: every switch below off at once) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 18255f368e.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-11-cluster.patch.
comptime _IDN_HDB_ON = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

# TOMBSTONE: MOJOLEARN_IDN_HDB_ONE_SYNC_OFF (old path: one wait per output download; EXP:1914 owed row withdrawn) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 3513712600.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_ONE_SYNC_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_ONE_SYNC = _IDN_HDB_ON
"""The extract's eight output downloads and the runner's six under one wait
each (`td_stage_*` / `td_take_*`, af-hdbscan2's HDB_ONE_SYNC route)."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF (old path: eight waits in the condense; no A/B recorded) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 23823b30db.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_CONDENSE_TWO_READS = _IDN_HDB_ON
"""The condense with two status readbacks instead of eight waits
(`tree_device.mojo::_condensed_two_reads`): the same kernels in the same
order, the refusals with the same messages and first indices."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF (old path: multi-read selection; EXP:1915 owed row withdrawn) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 7b7f05966a.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_SELECT_ONE_READ = _IDN_HDB_ON
"""Stabilities, selection, labels, scores and probabilities with no wait in
between and ONE readback (`extract.mojo::_extract_one_read`); a nonzero
`cluster_selection_epsilon` keeps the multi-read route."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF (old path: two more m * m guard passes; no A/B recorded) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at b02cf6086f.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_MR_FUSED_GUARD = _IDN_HDB_ON
"""Dense arm: the NaN count of the distance matrix (DEVIATION 623) and the
non-finite count of the mutual reachabilities (DEVIATION 1607) taken INSIDE
the mutual-reachability transform's one pass over the m * m cells and read
back together, instead of two more m * m passes with two waits each. Same
counts, same refusals, the distance refusal first."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF (old path: soft clustering downloads every intermediate; no A/B recorded) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 9f04fe0661.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_SOFT_LEAN = _IDN_HDB_ON
"""Soft clustering: the four intermediates are downloaded only when a trace
is being written (they are read by nothing else), and the five kernels run
with one wait at the end instead of one each."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF (old path: host narrowing loop over nq * k indices; no A/B recorded) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 675c1e06ae.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_PREDICT_DEVICE_CAST = _IDN_HDB_ON
"""approximate_predict: the k-NN's UInt32 indices are narrowed to Int32 by
the fit's device kernel instead of a host loop over nq * k cells."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF (old path: per-element staging and waits in approximate_predict; no A/B recorded) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 6260abe71f.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_HDB_PREDICT_LEAN = _IDN_HDB_ON
"""approximate_predict (lane fix-c1-cluster): every host staging buffer is
allocated under one wait, the training matrix, the queries and the model
arrays are staged with bulk copies instead of per-element loops (and without
`_upload_*`'s three waits each), the kernels run with no wait between them,
and the four outputs come back under one wait. The same kernels on the same
values: no bit moves. Implies `IDN_HDB_PREDICT_DEVICE_CAST`'s device cast."""

# TOMBSTONE: MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS (unmeasured opt-in threshold arm: =4096 / =16384 never timed; the default
# 46340 = PAIRWISE_MAX_ROWS is kept) deleted 2026-11-18 by lane/fx-env-dead; code recoverable at 0c65d61fed.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS.patch; record in docs/TOMBSTONES.md.
