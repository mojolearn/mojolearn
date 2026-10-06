# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Opt-in Apple FAST tree candidates: source only, no qualification evidence."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# Every candidate stays OFF absent its define. No performance or quality
# evidence exists: uncompiled, unverified, unmeasured. Definitions have no
# effect on other vendors or IDENTICAL; none changes estimator budgets.
comptime _APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
comptime AFT_N01 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N01"]()
comptime AFT_N02 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N02"]()
comptime AFT_N03 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N03"]()
comptime AFT_N04 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N04"]()
comptime AFT_N05 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N05"]()
comptime AFT_N06 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N06"]()
comptime AFT_N07 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N07"]()
comptime AFT_N08 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N08"]()
comptime AFT_N09 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N09"]()
comptime AFT_N10 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N10"]()
comptime AFT_N11 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N11"]()
comptime AFT_N12 = _APPLE_FAST and is_defined["MOJOLEARN_AFT_N12"]()
