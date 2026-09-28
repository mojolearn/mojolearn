# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LayerNorm beside RMSNorm: `torch.nn.LayerNorm` /
`torch.nn.functional.layer_norm` over the trailing `normalized_shape`
dimensions (biased variance, eps inside the rsqrt, optional elementwise
affine), forward and backward, on the GPU (`sequence/layernorm.mojo`).
`layer_norm_forward` / `layer_norm_backward` are the functional pair beside
`rms_norm_forward` / `rms_norm_backward`; `LayerNorm` holds the weight and
bias and keeps their gradients after `backward`."""
import numpy as np

from . import _backend


def _f32(a, name):
    a = np.asarray(a)
    if a.dtype == np.float64:
        raise TypeError(f"{name}: float64 is refused; pass float32")
    return np.ascontiguousarray(a, dtype=np.float32)


def _run(x, D, weight, bias, eps, dy, numeric_mode):
    x = _f32(x, "x")
    if x.size % D or x.shape[-1] == 0:
        raise ValueError("layer_norm: the trailing dimensions must match normalized_shape")
    M = x.size // D
    w = None if weight is None else _f32(weight, "weight").reshape(-1)
    b = None if bias is None else _f32(bias, "bias").reshape(-1)
    for name, v in (("weight", w), ("bias", b)):
        if v is not None and v.size != D:
            raise ValueError(f"layer_norm: {name} must hold {D} values")
    bwd = dy is not None
    # backward never reads y: the binding skips its download (lane py-sequence)
    y = None if bwd else np.zeros(x.shape, dtype=np.float32)
    dyv = _f32(dy, "dy") if bwd else None
    if bwd and dyv.shape != x.shape:
        raise ValueError("layer_norm_backward: dy must have x's shape")
    dx = np.zeros(x.shape, dtype=np.float32)
    dw = np.zeros(D, dtype=np.float32)
    db = np.zeros(D, dtype=np.float32)
    addr = lambda a: 0 if a is None else a.ctypes.data   # noqa: E731
    _backend.binding("_mojolearn_x_sequence", numeric_mode).layer_norm(
        [x.ctypes.data, addr(w), addr(b), addr(y), addr(dyv), dx.ctypes.data, dw.ctypes.data, db.ctypes.data],
        [M, D, int(w is not None), int(b is not None), int(bwd)], [float(eps)])
    return y, dx, dw, db


def _D(normalized_shape):
    shape = (normalized_shape,) if isinstance(normalized_shape, (int, np.integer)) else tuple(normalized_shape)
    if not shape or any(int(s) < 1 for s in shape):
        raise ValueError("normalized_shape must be positive")
    return shape, int(np.prod(shape))


def layer_norm_forward(x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """`F.layer_norm(x, normalized_shape, weight, bias, eps)`; normalized_shape
    defaults to x's last dimension."""
    x = _f32(x, "x")
    _, D = _D(x.shape[-1] if normalized_shape is None else normalized_shape)
    return _run(x, D, weight, bias, eps, None, numeric_mode)[0]


def layer_norm_backward(dy, x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """(dx, dweight, dbias) of `layer_norm_forward`; dweight / dbias are None
    when there is no weight / bias."""
    x = _f32(x, "x")
    _, D = _D(x.shape[-1] if normalized_shape is None else normalized_shape)
    _, dx, dw, db = _run(x, D, weight, bias, eps, dy, numeric_mode)
    return dx, (None if weight is None else dw), (None if bias is None else db)


class LayerNorm:
    """`torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias)`:
    weight ones, bias zeros; `forward(x)` (also `self(x)`) and `backward(dy)`
    (after a forward) returning dx and setting `weight_grad` / `bias_grad`."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, numeric_mode=None):
        self.normalized_shape, self._D = _D(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight = np.ones(self.normalized_shape, np.float32) if elementwise_affine else None
        self.bias = np.zeros(self.normalized_shape, np.float32) if (elementwise_affine and bias) else None
        self.numeric_mode = numeric_mode
        self.weight_grad = self.bias_grad = None
        self._x = None

    def forward(self, x):
        x = _f32(x, "x")
        if tuple(x.shape[x.ndim - len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(f"LayerNorm: trailing shape {x.shape} does not end in {self.normalized_shape}")
        self._x = x
        return _run(x, self._D, self.weight, self.bias, self.eps, None, self.numeric_mode)[0]

    __call__ = forward

    def backward(self, dy):
        if self._x is None:
            raise RuntimeError("LayerNorm.backward: call forward first")
        _, dx, dw, db = _run(self._x, self._D, self.weight, self.bias, self.eps, dy, self.numeric_mode)
        self.weight_grad = None if self.weight is None else dw.reshape(self.normalized_shape)
        self.bias_grad = None if self.bias is None else db.reshape(self.normalized_shape)
        return dx
