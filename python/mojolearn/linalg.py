# SPDX-License-Identifier: Apache-2.0
"""Public FP32 linear-algebra surface: the GEMM, and the three
decompositions this tree computes.

``matmul`` accepts NN, NT and TN products; both transpose flags together are
refused. It requires the process-selected IDENTICAL binding by default.
``identical=False`` uses the selected tier without an identity claim. See
``matmul.__doc__`` for the exact arithmetic profile and measured shape limits.
An import does not execute a matrix product or load this extension.

``qr``, ``eigh`` and ``svdvals`` carry numpy's names and numpy's meanings for
the subset this tree computes; the modes it does not compute (a Q factor, an
``svd`` returning ``U``, a wide matrix) are refused BY NAME rather than
approximated. They take the HOST route on every box, a GPU box included, so
they return the same bits on a laptop and in a datacentre by construction.
"""
from ._linalg_impl import (
    PROFILE,
    PROFILE_BF16,
    PROFILE_FAMILY,
    PROFILE_INT8,
    PROFILE_VERSION,
    dequantize_int8,
    eigh,
    from_bf16,
    matmul,
    matmul_bf16,
    matmul_int8,
    numeric_mode,
    profile,
    qr,
    quantize_int8,
    require_identical,
    svdvals,
    to_bf16,
)
# `Cholesky` (workstream D, 2026-09-14) binds `_mojolearn_gp`, not
# `_mojolearn_linalg`: the GP build already links every Cholesky kernel
# and a second copy would be a second binary of the same arithmetic.
from ._cholesky_impl import Cholesky

__all__ = ['matmul', 'numeric_mode', 'profile', 'require_identical',
           # the three decompositions under their own names
           # (lane/linalg-public, 2026-09-19). numpy's names, numpy's
           # meanings, and the modes this tree does not compute refused BY
           # NAME rather than approximated: see each docstring.
           'qr', 'eigh', 'svdvals',
           'PROFILE', 'PROFILE_FAMILY', 'PROFILE_VERSION', 'Cholesky',
           # the low-bit profiles (gemm/IDENTICAL_LOWBIT_CONTRACT.md)
           'matmul_bf16', 'matmul_int8', 'to_bf16', 'from_bf16',
           'quantize_int8', 'dequantize_int8', 'PROFILE_BF16', 'PROFILE_INT8']
