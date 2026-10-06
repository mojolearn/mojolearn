# SPDX-License-Identifier: Apache-2.0
"""Explicit residual-dropout layer: argument admission and native buffers only."""
import operator

from . import _training_impl
from ._buffer import addr, addr_ro, empty
from ._bufcheck import is_native_f32, probe


def _operand(value, name):
    view = probe(value)
    if not is_native_f32(view.format) or not view.c_contiguous:
        raise ValueError(name + ' must be a C-contiguous native float32 buffer')
    count = 1
    for extent in view.shape:  # glue: shape metadata, no tensor elements
        count *= extent
    if count > (1 << 31) - 1:
        raise ValueError(name + ' exceeds the native element index range')
    return view.shape, count


def _uint(value, name, bits):
    if isinstance(value, bool):
        raise TypeError(name + ' must be an unsigned integer')
    value = operator.index(value)
    if value < 0 or value >= 1 << bits:
        raise ValueError(name + ' is outside its unsigned integer range')
    return value


def _params(count, p, seed, stream, offset):
    seed = _uint(seed, 'seed', 64)
    stream = _uint(stream, 'stream', 32)
    offset = _uint(offset, 'offset', 63)
    if offset > (1 << 63) - 1 - count:
        raise ValueError('offset plus element count exceeds the native Philox range')
    if isinstance(p, bool):
        raise TypeError('p must be a real dropout probability')
    # p admission, FP32 rounding and scale arithmetic are native Mojo.
    return [count, offset, seed & 0xffffffff, seed >> 32, stream, float(p)]


def _entry(name):
    binding = _training_impl._load('identical')
    entry = getattr(binding, name, None)
    if not callable(entry):
        raise ImportError('residual dropout requires updated training binding: missing ' + name)
    return entry


def residual_dropout(values, residual, *, p=0.5, seed=0, stream=0, offset=0):
    """Return ``residual + dropout(values)`` as one explicit IDENTICAL layer.

    Inputs must have identical shapes and C-contiguous float32 storage. The
    native implementation admits finite values and p in [0,1), computes
    scale=1/(1-p), and preserves the rounded dropout intermediate before add.
    RNG coordinates are (64-bit seed, 32-bit stream, offset+flat_index).
    NN59 selects the fused schedule; OFF/ALL_OFF use two native stages.
    No existing model is implicitly changed and no validation is claimed.
    """
    shape, count = _operand(values, 'values')
    residual_shape, residual_count = _operand(residual, 'residual')
    if residual_shape != shape or residual_count != count:
        raise ValueError('values and residual must have identical shapes')
    params = _params(count, p, seed, stream, offset)
    result = empty(shape, '<f4')
    written = _entry('residual_dropout')(
        addr_ro(values, name='values') if count else 0,
        addr_ro(residual, name='residual') if count else 0,
        addr(result, name='result') if count else 0, params)
    if written != count:
        raise RuntimeError('residual_dropout returned an incomplete output')
    return result


def residual_dropout_backward(gradient, *, p=0.5, seed=0, stream=0, offset=0):
    """Return ``(d_values, d_residual)`` for the explicit residual layer.

    Pass exactly the forward p/seed/stream/offset to regenerate its mask.
    d_values applies the same rounded keep-and-scale operation to gradient;
    d_residual preserves the upstream gradient words. This is an explicit
    backward operation, not an automatic differentiation registration.
    """
    shape, count = _operand(gradient, 'gradient')
    params = _params(count, p, seed, stream, offset)
    d_values = empty(shape, '<f4')
    d_residual = empty(shape, '<f4')
    written = _entry('residual_dropout_backward')(
        addr_ro(gradient, name='gradient') if count else 0,
        addr(d_values, name='d_values') if count else 0,
        addr(d_residual, name='d_residual') if count else 0, params)
    if written != count:
        raise RuntimeError('residual_dropout_backward returned incomplete gradients')
    return d_values, d_residual
