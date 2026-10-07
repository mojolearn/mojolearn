# SPDX-License-Identifier: Apache-2.0
"""CPU-safe identifier for selected, same-version neural arithmetic graphs."""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, NEURAL_CHAINS
from training.neural_ab_profile_contract import NN54_LOSS_PROFILE, NN57_NORM_PROFILE


def neural_arithmetic_suffix() -> String:
    var result = String("")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]():
        comptime if NEURAL_PROFILE_CHANGED:
            result += ".nn-gemm-v2-l" + String(NEURAL_LEAF) + "-c" + String(NEURAL_CHAINS)
        comptime if get_defined_int["MOJOLEARN_IDN_ATTN_SOFTMAX", 0]() == 1:
            result += ".nn-attention-v2-tree"
        comptime if (get_defined_int["MOJOLEARN_IDN_NORM", 0]() == 1 or get_defined_int["MOJOLEARN_IDN_NORM", 0]() == 4):
            result += ".nn-norm-v2-lanes8"
        # State/gradient graphs also belong to the serialized arithmetic
        # version, even when a particular model does not consume that graph.
        # Keep this module CPU-safe: mirror the pure compile-time guards,
        # without importing a device scan implementation into host bindings.
        comptime if get_defined_int["MOJOLEARN_IDN_M1_SCAN", 0]() == 1:  # affine_prefix (NN34)
            result += ".nn-mamba1-v2-affine32"
        comptime if get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD", 0]() == 1:
            result += ".nn-mamba2-grad-v2-tree"
        elif get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD", 0]() == 2:
            result += ".nn-mamba2-grad-v2-leaf128"
        comptime if not is_defined["MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF"]():
            comptime if get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD", 0]() == 2:
                result += ".nn-recurrent-grad-v2-fixed128"
            elif get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD", 0]() == 1:
                result += ".nn-recurrent-grad-v2-min256"
        comptime if NN54_LOSS_PROFILE:
            result += ".nn-loss-v2-leaf128"
        comptime if NN57_NORM_PROFILE:
            result += ".nn-clip-v2-leaf128"
    return result


def neural_training_profile() -> String:
    return String("mojolearn.neural-training.fp32.v1") + neural_arithmetic_suffix()
