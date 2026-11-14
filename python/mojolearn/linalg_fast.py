# SPDX-License-Identifier: Apache-2.0
"""Apple FAST fused GEMM (lane/apple-fast-neural-gemm, 2026-10-03).

``matmul_fused`` is ``C = epi(op(a) @ op(b))`` with the epilogue applied in
the kernel's store: a bias, a bias and a residual, or a bias followed by
SiLU or GELU (exact erf). It exists only in a FAST linalg binding built on
an Apple GPU with ``-D MOJOLEARN_AFN_GEMM_EPILOGUE`` (or ``_ALL``); every
other binding raises ``NotImplementedError`` naming the define, so a caller
can fall back to ``matmul`` plus its own epilogue. No identity claim: FAST.

The attention, mamba and MLP lanes are the intended callers (their Mojo
blocks call ``gemm.afn_apple_fast.afn_gemm_fused_into`` directly; this
wrapper is the Python face of the same entry for tests and the quality
judge). Nothing in the tree calls it this round.
"""
from ._linalg_impl import _NATIVE, _NATIVE_CONTEXT, _load

EPILOGUES = {None: 0, "bias": 1, "bias_residual": 2, "bias_silu": 3, "bias_gelu": 4}


def has_fused():
    """Whether the loaded linalg binding carries ``gemm_fused``."""
    return hasattr(_load(), "gemm_fused")


def matmul_fused(a, b, *, bias=None, residual=None, activation=None,
                 transpose_a=False, transpose_b=False, out=None):
    """``C = epi(op(a) @ op(b))`` on the Apple FAST simdgroup kernel.

    ``bias`` is ``n`` float32 values (required unless every epilogue
    argument is None, in which case this is ``matmul(identical=False)`` on
    the same kernel). ``residual`` is an ``m x n`` float32 matrix added
    after the bias. ``activation`` is None, ``"silu"`` or ``"gelu"`` (the
    exact erf GELU), applied after the bias; it cannot be combined with
    ``residual`` (one epilogue per store). ``transpose_a`` with
    ``transpose_b`` is refused as ``matmul`` refuses it.
    """
    # One native call: the argument refusals, operand layout, output and the
    # `gemm_fused` call are bindings/linalg_ops_lowbit_door.mojo
    # `_lin_matmul_fused` (lane py-classical-linalg).
    return _NATIVE(None, "family_lin_matmul_fused",
                   (None, a, b, bias, residual, activation, transpose_a, transpose_b, out), _NATIVE_CONTEXT)


# Import-time binding only. All BF16 argument handling, packing, allocations,
# launches, synchronization and consumed-output construction run in Mojo.
# Apple FAST BF16 defaults on through MOJOLEARN_AFN_GEMM_BF16_MMA=1;
# explicit =0 disables it. No identical fallback is used. A missing native
# export is an installation error; never fall back to the Python wrapper.
_bf16_binding = _load()
has_bf16 = _bf16_binding.fast_bf16_available
to_bf16 = _bf16_binding.fast_to_bf16
from_bf16 = _bf16_binding.fast_from_bf16
matmul_bf16 = _bf16_binding.fast_matmul_bf16

__all__ = ["matmul_fused", "has_fused", "EPILOGUES",
           "has_bf16", "to_bf16", "from_bf16", "matmul_bf16"]
