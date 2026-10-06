# SPDX-License-Identifier: Apache-2.0
"""CPU-safe identifier for selected, same-version neural arithmetic graphs."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, NEURAL_CHAINS
from training.neural_ab_profile_contract import NN54_LOSS_PROFILE, NN57_NORM_PROFILE


def neural_arithmetic_suffix() -> String:
    var result = String("")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        comptime if NEURAL_PROFILE_CHANGED:
            result += ".nn-gemm-v2-l" + String(NEURAL_LEAF) + "-c" + String(NEURAL_CHAINS)
        comptime if is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"]():
            result += ".nn-attention-v2-tree"
        comptime if is_defined["MOJOLEARN_NN24_NORM_LANES8"]():
            result += ".nn-norm-v2-lanes8"
        comptime if NN54_LOSS_PROFILE:
            result += ".nn-loss-v2-leaf128"
        comptime if NN57_NORM_PROFILE:
            result += ".nn-clip-v2-leaf128"
    return result


def neural_training_profile() -> String:
    return String("mojolearn.neural-training.fp32.v1") + neural_arithmetic_suffix()
