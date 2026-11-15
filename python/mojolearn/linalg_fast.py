# SPDX-License-Identifier: Apache-2.0
"""Apple FAST BF16 GEMM helpers.

``has_bf16``, ``to_bf16``, ``from_bf16`` and ``matmul_bf16`` are the FAST
BF16 entries of the linalg binding. No identity claim: FAST.
"""
from ._linalg_impl import _load

# TOMBSTONE: matmul_fused, has_fused and EPILOGUES (the fused-epilogue GEMM face) deleted 2026-11-15 by
# lane/expdel2-tooling: the AFN epilogue arm (MOJOLEARN_AFN_GEMM_EPILOGUE) was deleted the same day, so no binding
# exports gemm_fused and matmul_fused always raised.


# Import-time binding only. All BF16 argument handling, packing, allocations,
# launches, synchronization and consumed-output construction run in Mojo.
# Apple FAST BF16 is the default (the =0 opt-out define was deleted
# 2026-11-15). No identical fallback is used. A missing native
# export is an installation error; never fall back to the Python wrapper.
_bf16_binding = _load()
has_bf16 = _bf16_binding.fast_bf16_available
to_bf16 = _bf16_binding.fast_to_bf16
from_bf16 = _bf16_binding.fast_from_bf16
matmul_bf16 = _bf16_binding.fast_matmul_bf16

__all__ = ["has_bf16", "to_bf16", "from_bf16", "matmul_bf16"]
