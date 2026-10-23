# SPDX-License-Identifier: Apache-2.0
"""Explicit residual-dropout layer: argument admission and native buffers only."""
from ._training_impl import _OPT_NATIVE, _OPT_CONTEXT


def residual_dropout(values, residual, *, p=0.5, seed=0, stream=0, offset=0):
    """Return ``residual + dropout(values)`` as one explicit ``identical``-mode layer.

    Inputs must have identical shapes and C-contiguous float32 storage. The
    native implementation admits finite values and p in [0,1), computes
    scale=1/(1-p), and preserves the rounded dropout intermediate before add.
    RNG coordinates are (64-bit seed, 32-bit stream, offset+flat_index).
    NN59 selects the fused schedule; OFF/ALL_OFF use two native stages.
    No existing model is implicitly changed and no validation is claimed.
    """
    return _OPT_NATIVE("residual_dropout", None, (values, residual, p, seed, stream, offset), "identical", _OPT_CONTEXT)


def residual_dropout_backward(gradient, *, p=0.5, seed=0, stream=0, offset=0):
    """Return ``(d_values, d_residual)`` for the explicit residual layer.

    Pass exactly the forward p/seed/stream/offset to regenerate its mask.
    d_values applies the same rounded keep-and-scale operation to gradient;
    d_residual preserves the upstream gradient words. This is an explicit
    backward operation, not an automatic differentiation registration.
    """
    return _OPT_NATIVE("residual_dropout_backward", None, (gradient, p, seed, stream, offset), "identical", _OPT_CONTEXT)
