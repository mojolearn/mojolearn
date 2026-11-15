# SPDX-License-Identifier: Apache-2.0
"""Tombstones of the deleted 2026-10-06 tree source experiments.

Every switch here was deleted on 2026-11-15 (expdel2-39-trees) and the
sweep of the same day folded every importer, so the module holds no code.
"""

# TOMBSTONE: MOJOLEARN_TREES_T01_ROWS (removed) deleted 2026-10-07 by e5ad4d891; code recoverable at e5ad4d891^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T01_REPLICAS (removed) deleted 2026-10-07 by e5ad4d891; code recoverable at e5ad4d891^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T01 (slower) deleted 2026-10-07 by e5ad4d891; code recoverable at e5ad4d891^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T01.patch; record in docs/TOMBSTONES.md.
# Every switch below is default OFF and IDENTICAL-only. The shared status above
# applies individually to every switch and its parameter/sub-arm.
# T01 (streamed exact histogram replicas, = N07) DELETED 2026-10-07: lost on
# NVIDIA and AMD. Its T01_ROWS task-size knob is covered by T02's cost rule.
# TOMBSTONE: MOJOLEARN_TREES_T02 (DROP, EXP:1581) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T02.patch
# (Sweep 2026-11-15: T02 and histogram_task_rows removed; readers folded.)
# TOMBSTONE: MOJOLEARN_TREES_T03 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T03.patch
# (Sweep 2026-11-15: T03 removed; the RF builder's retained-count fields and branches folded.)
# TOMBSTONE: MOJOLEARN_TREES_T04 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T04.patch
# (Sweep 2026-11-15: T04 removed; readers folded, clear_live_histogram_bins_kernel removed.)
# TOMBSTONE: MOJOLEARN_TREES_T05 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T05.patch
# (Sweep 2026-11-15: T05 and T05_ROW_VISITS removed; RF/ET small-node branches and kernels folded.)
# TOMBSTONE: MOJOLEARN_TREES_T05_ROW_VISITS (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T05_ROW_VISITS.patch
# TOMBSTONE: MOJOLEARN_TREES_T06 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T06.patch
# (Sweep 2026-11-15: T06 removed; cached-entropy readers and helpers folded.)
# TOMBSTONE: MOJOLEARN_TREES_T07 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T07.patch
# (Sweep 2026-11-15: T07 removed; readers folded, merge_split_candidates_parallel_kernel removed.)
# TOMBSTONE: MOJOLEARN_TREES_T08 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08.patch
# TOMBSTONE: MOJOLEARN_TREES_T08_LAYOUT (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_LAYOUT.patch (same patch as T08)
# TOMBSTONE: MOJOLEARN_TREES_T08_BITS (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_BITS.patch (same patch as T08)
# (Sweep 2026-11-15: T08, T08_LAYOUT and T08_BITS removed; readers folded.)
# TOMBSTONE: MOJOLEARN_TREES_T09 (dead code) deleted 2026-10-07 by eb8efb83a; code recoverable at eb8efb83a^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T09.patch; record in docs/TOMBSTONES.md.
# T09 (the ExtraTrees bootstrap-locality sort) was deleted by lane/grid-prune
# (2026-10-07): unreachable on the board, host sync inside the fit.
# TOMBSTONE: MOJOLEARN_TREES_RF_SAMPLE (ALWAYS_OFF_DELETE, EXP:1610) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_RF_SAMPLE.patch
# The per-vendor incumbent stays in ensemble/randomforest.mojo: NVIDIA fused
# sample + label gather, AMD two launches, Apple sorted rows from 64 columns.
# T11 device frontier levels per header drain: one int sweep
# `-D MOJOLEARN_TREES_T11_LEVELS=1|2|4|8|16` (absent = incumbent K=4; the
# older IDN K1/K2/K8 defines are deleted). Scheduling only, no bit moves. RF K1/K2/K8 already
# measured "no combined improvement" (forest-final-decisions 2026-10-05), so
# the RF grid should start at 16; ET has no such record.
# TOMBSTONE: MOJOLEARN_TREES_T11_LEVELS (DROP, EXP:1582) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T11_LEVELS.patch
# (Sweep 2026-11-15: T11 and T11_LEVELS removed; no reader.)
# TOMBSTONE: MOJOLEARN_TREES_T12 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T12.patch
# TOMBSTONE: MOJOLEARN_TREES_T12_BYTES (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T12_BYTES.patch (same patch as T12)
# (Sweep 2026-11-15: T12 and T12_BYTES removed; readers folded.)
# TOMBSTONE: MOJOLEARN_TREES_T13 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T13.patch
# TOMBSTONE: MOJOLEARN_TREES_T13_BYTES (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T13_BYTES.patch
# (Sweep 2026-11-15: T13 and T13_BYTES removed; readers folded.)
# TOMBSTONE: MOJOLEARN_TREES_T14 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T14.patch
# TOMBSTONE: MOJOLEARN_TREES_T14_EXACT (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T14_EXACT.patch (same patch as T14)
# (Sweep 2026-11-15: T14 and T14_EXACT removed; readers folded, ensemble/tree_moments.mojo (T14 moment arithmetic)
# and the ObjectiveLike.LabelMoment method removed.)
# TOMBSTONE: MOJOLEARN_TREES_T15 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T15.patch
# (Sweep 2026-11-15: T15 removed; no reader.)
# TOMBSTONE: MOJOLEARN_TREES_C48 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_C48.patch
# (Sweep 2026-11-15: C48 removed; RF and ET readers folded, publish_histogram_child_counts_kernel and the
# partition_scatter_kernel FUSED_OFFSETS parameter removed.)

