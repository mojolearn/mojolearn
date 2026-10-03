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
from ._buffer import addr, addr_ro, empty, probe
from ._linalg_impl import _OPS, _load, _operand, numeric_mode

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
    numeric_mode()
    binding = _load()
    if not hasattr(binding, "gemm_fused"):
        raise NotImplementedError(
            "mojolearn.linalg_fast.matmul_fused: the loaded linalg binding has no "
            "gemm_fused entry; it is a FAST, Apple-GPU build with "
            "-D MOJOLEARN_AFN_GEMM_EPILOGUE (gemm/afn_apple_fast.mojo). Use "
            "mojolearn.linalg.matmul(..., identical=False) and apply the epilogue "
            "yourself.")
    key = (bool(transpose_a), bool(transpose_b))
    if key not in _OPS:
        raise ValueError("mojolearn.linalg_fast.matmul_fused: transpose_a=True with "
                         "transpose_b=True is refused (three operations: NN, NT, TN)")
    op = _OPS[key]
    if residual is not None and activation is not None:
        raise ValueError("mojolearn.linalg_fast.matmul_fused: residual and activation "
                         "cannot be combined; one epilogue per store")
    if activation is None:
        epi = 0 if bias is None and residual is None else (2 if residual is not None else 1)
    else:
        if activation not in ("silu", "gelu"):
            raise ValueError("mojolearn.linalg_fast.matmul_fused: activation must be None, "
                             f"'silu' or 'gelu', got {activation!r}")
        epi = 3 if activation == "silu" else 4
    if epi != 0 and bias is None:
        raise ValueError("mojolearn.linalg_fast.matmul_fused: every epilogue needs bias")

    a_arr = _operand(a, "a")
    b_arr = _operand(b, "b")
    if transpose_a:
        k, m = a_arr.shape
    else:
        m, k = a_arr.shape
    if transpose_b:
        n, kb = b_arr.shape
    else:
        kb, n = b_arr.shape
    if k != kb:
        raise ValueError(f"mojolearn.linalg_fast.matmul_fused: contracted extents disagree, "
                         f"a gives k={k} and b gives k={kb}")
    bias_addr = 0
    resid_addr = 0
    keep = []
    if epi != 0:
        bias_arr = _operand(bias.reshape(1, -1) if hasattr(bias, "reshape") else bias, "bias")
        if probe(bias_arr).shape != (1, n):
            raise ValueError(f"mojolearn.linalg_fast.matmul_fused: bias must hold n={n} values")
        bias_addr = addr_ro(bias_arr, name="bias")
        keep.append(bias_arr)
    if epi == 2:
        res_arr = _operand(residual, "residual")
        if probe(res_arr).shape != (m, n):
            raise ValueError(f"mojolearn.linalg_fast.matmul_fused: residual must be ({m}, {n})")
        resid_addr = addr_ro(res_arr, name="residual")
        keep.append(res_arr)
    if out is None:
        out_arr = empty((m, n), "<f4")
    else:
        out_arr = out
        pb = probe(out_arr)
        if pb.shape != (m, n) or not pb.c_contiguous or pb.readonly:
            raise ValueError(f"mojolearn.linalg_fast.matmul_fused: out must be a writable "
                             f"C-contiguous float32 ({m}, {n}) buffer")
    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_linalg.mojo::gemm_fused_binding`):
    #     0 m, 1 n, 2 k, 3 op, 4 epi
    params = [int(m), int(n), int(k), int(op), int(epi)]
    binding.gemm_fused(addr(out_arr, name="out"), addr_ro(a_arr, name="a"),
                       addr_ro(b_arr, name="b"), bias_addr, resid_addr, params)
    del keep
    return out_arr


__all__ = ["matmul_fused", "has_fused", "EPILOGUES"]
