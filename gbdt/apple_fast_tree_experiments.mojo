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
comptime AFT_N01 = False
# TOMBSTONE: MOJOLEARN_AFT_N02 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at e601839e30.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N02.patch; record in docs/TOMBSTONES.md.
comptime AFT_N02 = False
# TOMBSTONE: MOJOLEARN_AFT_N03 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 5f55007c32.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N03.patch; record in docs/TOMBSTONES.md.
comptime AFT_N03 = False
# TOMBSTONE: MOJOLEARN_AFT_N04 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 13d92e2f1c.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N04.patch; record in docs/TOMBSTONES.md.
comptime AFT_N04 = False
# TOMBSTONE: MOJOLEARN_AFT_N05 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at eaabda5fe8.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_N05.patch; record in docs/TOMBSTONES.md.
comptime AFT_N05 = False
comptime AFT_N06 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N06"]()
comptime AFT_N07 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N07"]()
comptime AFT_N08 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N08"]()
comptime AFT_N09 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N09"]()
comptime AFT_N10 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N10"]()
comptime AFT_N11 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N11"]()
comptime AFT_N12 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N12"]()
