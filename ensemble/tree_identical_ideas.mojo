# SPDX-License-Identifier: Apache-2.0
"""2026-10-06 tree source experiments. A opts in; absence preserves B.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No symbol in this module is evidence of runtime reach or qualification.
The source integration ledger is experiments/trees_identical_20261006/forest/.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

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
comptime T02 = False
# TOMBSTONE: MOJOLEARN_TREES_T03 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T03.patch
comptime T03 = False
# TOMBSTONE: MOJOLEARN_TREES_T04 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T04.patch
comptime T04 = False
# TOMBSTONE: MOJOLEARN_TREES_T05 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T05.patch
comptime T05 = False
# RF T05 per-node bound in row visits per thread (see builder.mojo T05_NODE_ROWS).
# TOMBSTONE: MOJOLEARN_TREES_T05_ROW_VISITS (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T05_ROW_VISITS.patch
comptime T05_ROW_VISITS = 32
comptime T06 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T06"]()
comptime T07 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T07"]()
# TOMBSTONE: MOJOLEARN_TREES_T08 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08.patch
# TOMBSTONE: MOJOLEARN_TREES_T08_LAYOUT (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_LAYOUT.patch (same patch as T08)
# TOMBSTONE: MOJOLEARN_TREES_T08_BITS (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_BITS.patch (same patch as T08)
comptime T08 = False
comptime T08_LAYOUT = 1
comptime T08_BITS = 8
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
comptime T11 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T11_LEVELS"]()
comptime T11_LEVELS = get_defined_int["MOJOLEARN_TREES_T11_LEVELS", 4]()
# TOMBSTONE: MOJOLEARN_TREES_T12 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T12.patch
# TOMBSTONE: MOJOLEARN_TREES_T12_BYTES (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T12_BYTES.patch (same patch as T12)
comptime T12 = False
comptime T12_BYTES = 256*1024*1024
comptime T13 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T13"]()
comptime T13_BYTES = get_defined_int["MOJOLEARN_TREES_T13_BYTES", 16*1024*1024]()
# TOMBSTONE: MOJOLEARN_TREES_T14 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T14.patch
# TOMBSTONE: MOJOLEARN_TREES_T14_EXACT (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T14_EXACT.patch (same patch as T14)
# Folded constants kept for importers in other lanes' files.
comptime T14 = False
comptime T14_EXACT = False
# TOMBSTONE: MOJOLEARN_TREES_T15 (unmeasured arm) deleted 2026-11-15 by expdel2-39-trees; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T15.patch
comptime T15 = False
comptime C48 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C48"]()

@always_inline
def histogram_task_rows(histogram_bytes: Int, row_bytes: Int, reference: Int) -> Int:
    """Rows per histogram task: the reference geometry (the T02 arm is deleted)."""
    return reference
