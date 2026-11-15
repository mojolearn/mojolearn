# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Opt-in Apple FAST tree candidates: source only, no qualification evidence."""
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# Every candidate stays OFF absent its define. No performance or quality
# evidence exists: uncompiled, unverified, unmeasured. Definitions have no
# effect on other vendors or IDENTICAL; none changes estimator budgets.
comptime _APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
)
# TOMBSTONE: MOJOLEARN_AFT_N01 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 33b61e5bf8.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N01.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N01 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N01"]()
# TOMBSTONE: MOJOLEARN_AFT_N02 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at e601839e30.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N02.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N02 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N02"]()
# TOMBSTONE: MOJOLEARN_AFT_N03 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 5f55007c32.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N03.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N03 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N03"]()
# TOMBSTONE: MOJOLEARN_AFT_N04 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 13d92e2f1c.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N04.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N04 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N04"]()
# TOMBSTONE: MOJOLEARN_AFT_N05 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at eaabda5fe8.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N05.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N05 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N05"]()
# TOMBSTONE: MOJOLEARN_AFT_N06 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at a0eec817dc.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N06.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N06 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N06"]()
# TOMBSTONE: MOJOLEARN_AFT_N07 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at e3824bb991.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N07.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N07 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N07"]()
# TOMBSTONE: MOJOLEARN_AFT_N08 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 7127af55a3.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N08.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N08 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N08"]()
# TOMBSTONE: MOJOLEARN_AFT_N09 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 6a830e56b0.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N09.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N09 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N09"]()
# TOMBSTONE: MOJOLEARN_AFT_N10 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 3c44979412.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N10.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N10 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N10"]()
# TOMBSTONE: MOJOLEARN_AFT_N11 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at cb73446e0e.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N11.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N11 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N11"]()
# TOMBSTONE: MOJOLEARN_AFT_N12 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 89a685b7dc.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N12.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
comptime AFT_N12 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N12"]()
