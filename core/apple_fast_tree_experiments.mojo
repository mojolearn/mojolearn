# SPDX-License-Identifier: Apache-2.0
"""P01–P12, experiments/apple_fast_trees/IDEAS.md.

SOURCE ONLY: uncompiled, unverified, unmeasured. All candidates are OFF by
default. No quality or speed claim, including for combinations. These guards
keep IDENTICAL and non-Apple paths outside this experiment programme.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime AFT_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()

# P01: two rather than four logical 32-tree groves per threadgroup; less
# shared reduction state, more independently schedulable blocks.
# Public FAST resident snapshots default to ordered arithmetic even when
# their Python engine name is parallel_groves. Hold
# MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF in BOTH P01/P03 arms so those
# public callers reach the lane-grove kernels; this prerequisite is not a
# promoted default. See P.json for the complete A and B define lists.
comptime AFT_P01 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P01"]()
comptime AFT_GROVE_BLOCK = 64 if AFT_P01 else 128
comptime AFT_GROVES_PER_BLOCK = AFT_GROVE_BLOCK // 32

# P02/P04: two independent row/output tasks per worker. Constant register
# work budget, never a dispatch threshold fitted to a dataset's dimensions.
comptime AFT_P02 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P02"]()
comptime AFT_ORDERED_ITEMS = 2 if AFT_P02 else 1
comptime AFT_P04 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P04"]()
comptime AFT_ARGMAX_ROWS = 2 if AFT_P04 else 1

# P03: row staging gets a fixed 2 KiB threadgroup budget. Larger feature
# rows keep direct loads; coverage follows storage capacity, not board rows.
# This differs from the pre-existing 4 KiB / four-row staging experiment.
comptime AFT_P03 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P03"]()
comptime AFT_ROW_STAGE_BYTES = 2048

# P05: two independent rows reuse each packed split descriptor. Requires
# MOJOLEARN_GBDT_PREDICT_PACKED in BOTH arms; that prerequisite is not promoted.
comptime AFT_P05 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P05"]()

# P06: halve link-kernel threads per block to trade scheduling for lower
# per-block register demand in software binary64 exp/div and multiclass loops.
comptime AFT_P06 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P06"]()

# P07: four rows per tree worker extends (does not rename) the existing
# two-row candidate. The two alternatives must not be selected together.
comptime AFT_P07 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P07"]()
def _integration_require_1() -> Bool:
    comptime assert not (AFT_P07 and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "P07 and SHAP_FAST_ROW_PAIR are alternative row tiles"
    return True

comptime _INTEGRATION_REQUIRE_1 = _integration_require_1()

# P08: fewer SHAP workers per block may reduce register pressure from path
# stacks; P09 halves contribution scratch to 64 MiB at the cost of launches.
comptime AFT_P08 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P08"]()
comptime AFT_P09 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P09"]()

# P10: two independent output cells per SHAP fold worker; complete tree
# accumulation order is retained separately for each cell.
comptime AFT_P10 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P10"]()
comptime AFT_SHAP_FOLD_ITEMS = 2 if AFT_P10 else 1

# P11/P12: two-row tiles in DART residual/predict and score-add stages.
# Dropout RNG and reduction units are unchanged, including draw identities.
comptime AFT_P11 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P11"]()
comptime AFT_P12 = AFT_APPLE_FAST and is_defined["MOJOLEARN_AFT_P12"]()
comptime AFT_DART_ROWS = 2 if AFT_P11 else 1
comptime AFT_DART_ADD_ROWS = 2 if AFT_P12 else 1
