# SPDX-License-Identifier: Apache-2.0
"""Native version label shared by host/GPU sessions and checkpoint owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
# TOMBSTONE: MOJOLEARN_IDN_NORM_FIXED_LANES (opt-in fixed 32-lane norm fold, EXP:2246 pending, never measured) oracle arms deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at fce9f4413c.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NORM_FIXED_LANES.patch
# TOMBSTONE: MOJOLEARN_IDN_NORM_FIXED_LANES (NOISE: A/B nvc-ab-20261124T095727Z-92fd9f NVIDIA L40S, 3 cells 0.96..0.99, all SAME) deleted 2026-11-24 by lane/ab-outcome-2 (ab2-ngemm); code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_NORM_FIXED_LANES.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
from gemm.experiments.neural_profile import NEURAL_LEAF, NEURAL_CHAINS, NEURAL_PROFILE_CHANGED


def transformer_arithmetic_profile() -> String:
    var name = String("mojolearn.identical.transformer.fp32")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        name = String("mojolearn.fast.transformer.fp32")
    # ".attn-summary32-tree-v1" (NN20, MOJOLEARN_IDN_ATTN_SOFTMAX=1) deleted 2026-10-08 as a grid ge123e6f9 loser
    # (1.6-43x slower NV/AMD); recoverable at main bc10b8b56.
    name += ".attn-v1"
    name += ".norm-v1"
    name += ".gemm-leaf" + String(NEURAL_LEAF) + "-chains" + String(NEURAL_CHAINS)
    return name


def transformer_arithmetic_profile_changed() -> Bool:
    return NEURAL_PROFILE_CHANGED
