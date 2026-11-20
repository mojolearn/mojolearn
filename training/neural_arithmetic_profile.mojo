# SPDX-License-Identifier: Apache-2.0
"""CPU-safe identifier for selected, same-version neural arithmetic graphs."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, NEURAL_CHAINS
from training.neural_ab_profile_contract import NN57_NORM_PROFILE


def neural_arithmetic_suffix() -> String:
    var result = String("")
    # TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella term; owner another lane) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-32-neural.patch
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if NEURAL_PROFILE_CHANGED:
            result += ".nn-gemm-v2-l" + String(NEURAL_LEAF) + "-c" + String(NEURAL_CHAINS)
        # ".nn-attention-v2-tree" (MOJOLEARN_IDN_ATTN_SOFTMAX=1, summary_tree) deleted 2026-10-08: grid ge123e6f9
        # loser, 1.6-43x slower NV/AMD; recoverable at main bc10b8b56.
        # TOMBSTONE: MOJOLEARN_IDN_NORM (arms 1 and 4 profile tag; owner another lane) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_NORM.expdel2-32-neural.patch
        # State/gradient graphs also belong to the serialized arithmetic
        # version, even when a particular model does not consume that graph.
        # Keep this module CPU-safe: mirror the pure compile-time guards,
        # without importing a device scan implementation into host bindings.
        # TOMBSTONE: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix Mamba-1 scan; LOSER: mamba1-forward gaussian NV 5.40 -> 10.41 ms (1.93), AMD 3.14 -> 8.38 ms (2.67), avg 2.30, vs arm 3 persistent) deleted 2026-10-11 by 291629888; code recoverable at 291629888^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN_1.patch (neural_scan_profile.mojo, neural_mamba_scan.mojo and the host/gen twins come back with it); record in docs/TOMBSTONES.md.

        # TOMBSTONE: MOJOLEARN_IDN_M2_GRAD_FOLD (arms 1 and 2 profile tags; owner another lane) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_GRAD_FOLD.expdel2-32-neural.patch

        # TOMBSTONE: MOJOLEARN_IDN_SEQ_WGRAD (arms 1 and 2 profile tags, and the MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF guard around them; owner another lane) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD.expdel2-32-neural.patch
        comptime if NN57_NORM_PROFILE:
            result += ".nn-clip-v2-leaf128"
    return result


def neural_training_profile() -> String:
    return String("mojolearn.neural-training.fp32.v1") + neural_arithmetic_suffix()
