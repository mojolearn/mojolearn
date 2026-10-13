# SPDX-License-Identifier: Apache-2.0
"""Float32 linear algebra: a bit-identical matrix product, low-bit products and
NumPy-named decompositions.

`matmul` computes ``op(a) @ op(b)`` in float32 under the profile
``mojolearn.identical.gemm.fp32.v1``: the same bits on Apple, NVIDIA and AMD for the same
inputs. It supports the NN, NT and TN products; asking for both transposes raises
`ValueError`. By default it requires the identical build and raises when another
build is loaded; ``identical=False`` uses whichever build is loaded, with no
identity guarantee. `numeric_mode`, `profile` and `require_identical` report and
check what this process loaded.

`matmul_bf16`, `matmul_int8` and `matmul_int15` are the low-bit profiles, with
`to_bf16`, `from_bf16`, `quantize_int8`, `dequantize_int8`, `quantize_int15` and
`dequantize_int15` to prepare and recover their operands.

`qr`, `eigh`, `svdvals` and `svd` carry NumPy's names and meanings for the modes
they support (note that `qr` defaults to ``mode='r'``); unsupported modes are refused
by name rather than approximated. They run on the device on a GPU install and on the
host binding on a CPU-only install, with the same bits. `Cholesky` factors a
symmetric positive-definite matrix and solves with the factor.

Importing this module does not load the extension or run any computation.

Examples
--------
>>> import numpy as np
>>> from mojolearn import linalg
>>> a = np.random.default_rng(0).random((256, 64), dtype=np.float32)
>>> b = np.random.default_rng(1).random((64, 32), dtype=np.float32)
>>> c = np.asarray(linalg.matmul(a, b))
"""
from ._linalg_impl import (
    PROFILE,
    PROFILE_BF16,
    PROFILE_FAMILY,
    INT15_MAX_K,
    PROFILE_INT15,
    PROFILE_INT8,
    PROFILE_VERSION,
    dequantize_int15,
    dequantize_int8,
    eigh,
    from_bf16,
    matmul,
    matmul_bf16,
    matmul_int15,
    matmul_int8,
    numeric_mode,
    profile,
    qr,
    quantize_int15,
    quantize_int8,
    require_identical,
    svd,
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
           'qr', 'eigh', 'svdvals', 'svd',
           'PROFILE', 'PROFILE_FAMILY', 'PROFILE_VERSION', 'Cholesky',
           # the low-bit profiles (gemm/IDENTICAL_LOWBIT_CONTRACT.md)
           'matmul_bf16', 'matmul_int8', 'to_bf16', 'from_bf16',
           'quantize_int8', 'dequantize_int8', 'PROFILE_BF16', 'PROFILE_INT8',
           # the fifteen-bit profile (contract section 6, lane/lowbit-int15)
           'matmul_int15', 'quantize_int15', 'dequantize_int15', 'PROFILE_INT15',
           'INT15_MAX_K']
