# SPDX-License-Identifier: Apache-2.0
"""Public FP32 GPU GEMM surface.

``matmul`` accepts NN, NT and TN products; both transpose flags together are
refused. It requires the process-selected IDENTICAL binding by default.
``identical=False`` uses the selected tier without an identity claim. See
``matmul.__doc__`` for the exact arithmetic profile and measured shape limits.
An import does not execute a matrix product or load this extension.
"""
from ._linalg_impl import (
    PROFILE,
    PROFILE_BF16,
    PROFILE_FAMILY,
    PROFILE_INT8,
    PROFILE_VERSION,
    dequantize_int8,
    from_bf16,
    matmul,
    matmul_bf16,
    matmul_int8,
    numeric_mode,
    profile,
    quantize_int8,
    require_identical,
    to_bf16,
)
# `Cholesky` (workstream D, 2026-09-14) binds `_mojolearn_gp`, not
# `_mojolearn_linalg`: the GP build already links every Cholesky kernel
# and a second copy would be a second binary of the same arithmetic.
from ._cholesky_impl import Cholesky

__all__ = ['matmul', 'numeric_mode', 'profile', 'require_identical',
           'PROFILE', 'PROFILE_FAMILY', 'PROFILE_VERSION', 'Cholesky',
           # the low-bit profiles (gemm/IDENTICAL_LOWBIT_CONTRACT.md)
           'matmul_bf16', 'matmul_int8', 'to_bf16', 'from_bf16',
           'quantize_int8', 'dequantize_int8', 'PROFILE_BF16', 'PROFILE_INT8']
