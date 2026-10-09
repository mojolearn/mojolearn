# SPDX-License-Identifier: Apache-2.0
"""Native version label shared by host/GPU sessions and checkpoint owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from transformer.experiments.norm_profile_contract import NN24_NORM_LANES8
from gemm.experiments.neural_profile import NEURAL_LEAF, NEURAL_CHAINS, NEURAL_PROFILE_CHANGED


def transformer_arithmetic_profile() -> String:
    var name = String("mojolearn.identical.transformer.fp32")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        name = String("mojolearn.fast.transformer.fp32")
    # ".attn-summary32-tree-v1" (NN20, MOJOLEARN_IDN_ATTN_SOFTMAX=1) deleted 2026-10-08 as a grid ge123e6f9 loser
    # (1.6-43x slower NV/AMD); recoverable at main bc10b8b56.
    name += ".attn-v1"
    name += ".norm-lanes8-v1" if NN24_NORM_LANES8 else ".norm-v1"
    name += ".gemm-leaf" + String(NEURAL_LEAF) + "-chains" + String(NEURAL_CHAINS)
    return name


def transformer_arithmetic_profile_changed() -> Bool:
    return NEURAL_PROFILE_CHANGED or NN24_NORM_LANES8
