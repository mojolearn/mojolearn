# SPDX-License-Identifier: Apache-2.0
"""Native version label shared by host/GPU sessions and checkpoint owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
# TOMBSTONE: MOJOLEARN_IDN_NORM_FIXED_LANES (opt-in fixed 32-lane norm fold, EXP:2246 pending, never measured) oracle arms deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at fce9f4413c.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NORM_FIXED_LANES.patch
# RESTORED 2026-11-15 by lane/restore-run-arms-neural: RUN #6 of plan DELETED_ARMS_WORTH_RUNNING_2026-11-15 (git history)
# (the MOJOLEARN_IDN_NORM_FIXED_LANES arms only; the NN24 / IDN_NORM arms beside them stay deleted).
from transformer.experiments.norm_fixed_lanes import IDN_NORM_FIXED_LANES
from gemm.experiments.neural_profile import NEURAL_LEAF, NEURAL_CHAINS, NEURAL_PROFILE_CHANGED


def transformer_arithmetic_profile() -> String:
    var name = String("mojolearn.identical.transformer.fp32")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        name = String("mojolearn.fast.transformer.fp32")
    # ".attn-summary32-tree-v1" (NN20, MOJOLEARN_IDN_ATTN_SOFTMAX=1) deleted 2026-10-08 as a grid ge123e6f9 loser
    # (1.6-43x slower NV/AMD); recoverable at main bc10b8b56.
    name += ".attn-v1"
    # MOJOLEARN_IDN_NORM_FIXED_LANES: the fixed 32-lane row fold (norm_fixed_lanes.mojo)
    if IDN_NORM_FIXED_LANES:
        name += ".norm-lanes32-v1"
    else:
        name += ".norm-v1"
    name += ".gemm-leaf" + String(NEURAL_LEAF) + "-chains" + String(NEURAL_CHAINS)
    return name


def transformer_arithmetic_profile_changed() -> Bool:
    return NEURAL_PROFILE_CHANGED or IDN_NORM_FIXED_LANES
