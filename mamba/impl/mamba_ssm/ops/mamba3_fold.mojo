# SPDX-License-Identifier: Apache-2.0
"""Execution spelling of an already-pinned Mamba3 FP32 fold step.

The optional NVIDIA spelling uses the same kernel-matrix row as GEMM and
fused attention. Inputs/results flush to signed zero; one RN FMA, unchanged
leaf order. The old software spelling remains the default for A/B gates.
"""
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from checks.kernel_matrix import TARGET_COLUMN, lib_hardware_ftz_fma_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add

comptime M3_HARDWARE_FOLD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_MAMBA3_HARDWARE_FOLD"]() and lib_hardware_ftz_fma_for[TARGET_COLUMN]()


@always_inline
def m3_fold_step(a: Float32, b: Float32, acc: Float32) -> Float32:
    comptime if M3_HARDWARE_FOLD:
        return llvm_intrinsic["llvm.nvvm.fma.rn.ftz.f", Float32, has_side_effect=False](a, b, acc)
    return ftz(identical_mul_add(ftz(a), ftz(b), acc))
