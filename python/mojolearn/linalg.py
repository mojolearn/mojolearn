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
    PROFILE_FAMILY,
    PROFILE_VERSION,
    matmul,
    numeric_mode,
    profile,
    require_identical,
)

__all__ = ['matmul', 'numeric_mode', 'profile', 'require_identical',
           'PROFILE', 'PROFILE_FAMILY', 'PROFILE_VERSION']
