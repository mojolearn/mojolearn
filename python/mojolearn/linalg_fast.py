# SPDX-License-Identifier: Apache-2.0
"""Apple FAST BF16 GEMM helpers.

``has_bf16``, ``to_bf16``, ``from_bf16`` and ``matmul_bf16`` are the FAST
BF16 entries of the linalg binding. No identity claim: FAST.
"""
from ._linalg_impl import _NATIVE, _NATIVE_CONTEXT, _load

# TOMBSTONE: matmul_fused, has_fused and EPILOGUES (the fused-epilogue GEMM face) deleted 2026-11-15 by
# lane/expdel2-tooling: the AFN epilogue arm (MOJOLEARN_AFN_GEMM_EPILOGUE) was deleted the same day, so no binding
# exports gemm_fused and matmul_fused always raised.
# RESTORED 2026-11-15 (lane/restore-apple-fast-neural): the face of the restored Apple fast opt-in arm
# MOJOLEARN_AFN_GEMM_EPILOGUE (default off; it raises NotImplementedError unless the binding exports gemm_fused).
# Still listed in docs-site/api_exclude.txt (never run).

EPILOGUES = {None: 0, "bias": 1, "bias_residual": 2, "bias_silu": 3, "bias_gelu": 4}


def has_fused():
    """Whether the loaded linalg binding carries ``gemm_fused``."""
    return hasattr(_load(), "gemm_fused")


def matmul_fused(a, b, *, bias=None, residual=None, activation=None,
                 transpose_a=False, transpose_b=False, out=None, device=None):
    """``C = epi(op(a) @ op(b))`` on the Apple FAST simdgroup kernel.

    ``bias`` is ``n`` float32 values (required unless every epilogue
    argument is None, in which case this is ``matmul(identical=False)`` on
    the same kernel). ``residual`` is an ``m x n`` float32 matrix added
    after the bias. ``activation`` is None, ``"silu"`` or ``"gelu"`` (the
    exact erf GELU), applied after the bias; it cannot be combined with
    ``residual`` (one epilogue per store). ``transpose_a`` with
    ``transpose_b`` is refused as ``matmul`` refuses it.

    Parameters
    ----------
    device : {None, 'gpu', 'cpu'}, default None
        Where the computation runs. None and 'gpu' run on the GPU; 'cpu' raises ValueError.
    """
    # One native call: the argument refusals, operand layout, output and the
    # `gemm_fused` call are bindings/linalg_ops_lowbit_door.mojo `_lin_matmul_fused`.
    return _NATIVE(None, "family_lin_matmul_fused",
                   (None, a, b, bias, residual, activation, transpose_a, transpose_b, out, device), _NATIVE_CONTEXT)


# All BF16 argument handling, packing, allocations, launches, synchronization
# and consumed-output construction run in Mojo. Apple FAST BF16 is the default
# (`-D MOJOLEARN_AFN_GEMM_BF16_MMA=0`, restored 2026-11-15, turns it off). No
# identical fallback is used. A missing native export is an installation
# error; never fall back to the Python wrapper.
#
# The four entries are resolved at their FIRST access, once (module
# `__getattr__`, then stored in this module's globals so later reads are a
# plain dictionary hit), never at import (vdf2-public, failure row 10): on a
# CPU-only install the import read `fast_bf16_available` of the GPU binding's
# host facade and raised NoCpuEntry, so `import mojolearn.linalg_fast` (and
# every test module importing it) failed. The host binding answers
# `fast_bf16_available` False (bindings/_mojolearn_linalg_host.mojo: the CPU
# has no BF16 GEMM); the other three still refuse by name when first read.
_ENTRIES = {"has_bf16": "fast_bf16_available", "to_bf16": "fast_to_bf16",
            "from_bf16": "fast_from_bf16", "matmul_bf16": "fast_matmul_bf16"}


def __getattr__(name):
    entry = _ENTRIES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_load(), entry)  # glue: one binding read per entry, at first access
    globals()[name] = value
    return value


__all__ = ["matmul_fused", "has_fused", "EPILOGUES",
           "has_bf16", "to_bf16", "from_bf16", "matmul_bf16"]
