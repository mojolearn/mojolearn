# SPDX-License-Identifier: Apache-2.0
"""Native version label shared by host/GPU sessions and checkpoint owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from transformer.experiments.attention_summary_contract import NN20_BALANCED_SUMMARY_TREE
from transformer.experiments.norm_profile_contract import NN24_NORM_LANES8
from gemm.experiments.neural_profile import NEURAL_LEAF, NEURAL_CHAINS, NEURAL_PROFILE_CHANGED


def transformer_arithmetic_profile() -> String:
    var name = String("mojolearn.identical.transformer.fp32")
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        name = String("mojolearn.fast.transformer.fp32")
    name += ".attn-summary32-tree-v1" if NN20_BALANCED_SUMMARY_TREE else ".attn-v1"
    name += ".norm-lanes8-v1" if NN24_NORM_LANES8 else ".norm-v1"
    name += ".gemm-leaf" + String(NEURAL_LEAF) + "-chains" + String(NEURAL_CHAINS)
    return name


def transformer_arithmetic_profile_changed() -> Bool:
    return NEURAL_PROFILE_CHANGED or NN20_BALANCED_SUMMARY_TREE or NN24_NORM_LANES8
