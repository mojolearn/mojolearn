# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LayerNorm beside RMSNorm: `torch.nn.LayerNorm` /
`torch.nn.functional.layer_norm` over the trailing `normalized_shape`
dimensions (biased variance, eps inside the rsqrt, optional elementwise
affine), forward and backward, on the GPU (`sequence/layernorm.mojo`).
`layer_norm_forward` / `layer_norm_backward` are the functional pair beside
`rms_norm_forward` / `rms_norm_backward`; `LayerNorm` holds the weight and
bias and keeps their gradients after `backward`.

Device I/O (lane gap-neural-io, 2026-10-08, plan docs/plans/gaps-2026-10-08.md
Section 4): an x made by `mojolearn._x_sequence_device.to_device` stays on
the device: the forward returns y as a `SequenceDeviceTensor`, the backward
reads dy there (a host dy is uploaded once) and returns dx as one; only the
D-float weight, bias and their gradients cross the bus
(`sequence/layernorm_dev.mojo`, the same launches on the same values).

Lane layernorm-idn N3 (2026-10-09), two waste removals, both default ON:
  * `LayerNorm` is a device-I/O layer: `LayerNorm._device_io` is True on a
    binding with the resident entries and `LayerNorm.to_device(a)` uploads
    once (`_x_sequence_device.to_device`), so a caller (the board's layer
    path among them) that puts x on the device gets the resident entry
    `layer_norm_dev` instead of the host-array entry (which, at an M x D
    activation of 67 MB, uploads x, downloads y, then uploads x and dy and
    downloads dx again per fit, plus two page-faulted zero fills). Nothing
    changes for a host-array x. `MOJOLEARN_LN_CLASS_DEVICE_IO_OFF=1` (or
    `MOJOLEARN_SEQ_DEVICE_IO_OFF=1`, or `MOJOLEARN_IDN_ALL_OFF=1`) is the
    before arm: the class reports no device I/O and `to_device` returns
    the host array.
  * Kept stats: a resident forward stores its row mean and rstd in two
    M-float resident tensors the layer keeps, and `backward` hands them
    back so the device entry skips its forward recompute (one M x D launch
    and an M x D scratch fewer). The words are the forward's own stores:
    no bit moves. `MOJOLEARN_LN_KEEP_STATS_OFF=1` (or IDN_ALL_OFF) is the
    before arm (the backward recomputes).
  * The host-array path's y and dx outputs are `np.empty`: the binding
    downloads every element, so the +0.0 fill was two 67 MB passes of
    waste per fit (no bits: every word is overwritten). dweight / dbias
    stay zero-filled (written only when weight / bias are given)."""
import os as _os

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_norm')

from . import _backend
from ._x_sequence_device import SequenceDeviceTensor, is_seq_tensor, resident_binding, to_device as _seq_to_device

_SEQ_BINDING = "_mojolearn_x_sequence"
_ALL_OFF = _os.environ.get("MOJOLEARN_IDN_ALL_OFF", "") == "1"
#: the before arm of the class device-I/O switch (see the module docstring)
_CLASS_DEVICE_IO_OFF = (_os.environ.get("MOJOLEARN_LN_CLASS_DEVICE_IO_OFF", "") == "1"
                        or _os.environ.get("MOJOLEARN_SEQ_DEVICE_IO_OFF", "") == "1" or _ALL_OFF)
#: the before arm of the kept-stats switch: the backward recomputes the forward
_KEEP_STATS_OFF = _os.environ.get("MOJOLEARN_LN_KEEP_STATS_OFF", "") == "1" or _ALL_OFF


def _f32(a, name):
    a = np.asarray(a)
    if a.dtype == np.float64:
        raise TypeError(f"{name}: float64 is refused; pass float32")
    return np.ascontiguousarray(a, dtype=np.float32)


def _run_dev(x, D, weight, bias, eps, dy, stats=None):
    """`_run` on a resident x (and dy): y (forward) or dx (backward) as a
    `SequenceDeviceTensor`, dweight / dbias as host arrays. `stats` = (mean,
    rstd), two M-float resident tensors: a forward writes them, a backward
    reads them instead of recomputing the forward (kept stats)."""
    b = x.b
    if not callable(getattr(b, "layer_norm_dev", None)):
        raise RuntimeError("layer_norm: this binding has no device LayerNorm; pass a NumPy x")
    if x.size % D or x.shape[-1] == 0:
        raise ValueError("layer_norm: the trailing dimensions must match normalized_shape")
    M = x.size // D
    w = None if weight is None else _f32(weight, "weight").reshape(-1)
    bb = None if bias is None else _f32(bias, "bias").reshape(-1)
    for name, v in (("weight", w), ("bias", bb)):  # glue: the two named parameter arrays
        if v is not None and v.size != D:
            raise ValueError(f"layer_norm: {name} must hold {D} values")
    addr = lambda a: 0 if a is None else a.ctypes.data   # noqa: E731
    flags = [M, D, int(w is not None), int(bb is not None)]
    hs, sflag = [], []
    if stats is not None:
        if not (is_seq_tensor(stats[0]) and is_seq_tensor(stats[1]) and stats[0].b is b and stats[1].b is b
                and stats[0].size == M and stats[1].size == M):
            raise ValueError("layer_norm: stats must be two resident tensors of M floats on x's binding")
        hs, sflag = [stats[0].h, stats[1].h], [1]
    if dy is None:
        y = SequenceDeviceTensor._new(b, x.shape)
        b.layer_norm_dev([x.h, y.h, 0, 0] + hs, [addr(w), addr(bb), 0, 0], flags + [0] + sflag, [float(eps)])
        return y, None, None, None
    if is_seq_tensor(dy):
        if dy.b is not b or tuple(dy.shape) != tuple(x.shape):
            raise ValueError("layer_norm_backward: dy must have x's shape (on x's binding)")
        dyt = dy
    else:
        dyv = _f32(dy, "dy")
        if dyv.shape != tuple(x.shape):
            raise ValueError("layer_norm_backward: dy must have x's shape")
        dyt = SequenceDeviceTensor._wrap(b, dyv)
    dx = SequenceDeviceTensor._new(b, x.shape)
    dw = np.empty(D, dtype=np.float32)   # written by the binding when weight is given
    db = np.empty(D, dtype=np.float32)   # written by the binding when bias is given
    b.layer_norm_dev([x.h, 0, dyt.h, dx.h] + hs, [addr(w), addr(bb), dw.ctypes.data, db.ctypes.data],
                     flags + [1] + sflag, [float(eps)])
    return None, dx, dw, db


def _run(x, D, weight, bias, eps, dy, numeric_mode, stats=None):
    if is_seq_tensor(x):
        return _run_dev(x, D, weight, bias, eps, dy, stats)
    x = _f32(x, "x")
    if x.size % D or x.shape[-1] == 0:
        raise ValueError("layer_norm: the trailing dimensions must match normalized_shape")
    M = x.size // D
    w = None if weight is None else _f32(weight, "weight").reshape(-1)
    b = None if bias is None else _f32(bias, "bias").reshape(-1)
    for name, v in (("weight", w), ("bias", b)):  # glue: the two named parameter arrays
        if v is not None and v.size != D:
            raise ValueError(f"layer_norm: {name} must hold {D} values")
    bwd = dy is not None
    # backward never reads y: the binding skips its download (lane py-sequence).
    # y and dx are downloaded whole by the binding (every element written), so
    # they are not zero-filled (lane layernorm-idn: two M x D fills fewer per fit)
    y = None if bwd else np.empty(x.shape, dtype=np.float32)
    dyv = _f32(dy, "dy") if bwd else None
    if bwd and dyv.shape != x.shape:
        raise ValueError("layer_norm_backward: dy must have x's shape")
    dx = np.empty(x.shape, dtype=np.float32) if bwd else np.zeros(1, dtype=np.float32)
    dw = np.zeros(D, dtype=np.float32)
    db = np.zeros(D, dtype=np.float32)
    addr = lambda a: 0 if a is None else a.ctypes.data   # noqa: E731
    _backend.binding("_mojolearn_x_sequence", numeric_mode).layer_norm(
        [x.ctypes.data, addr(w), addr(b), addr(y), addr(dyv), dx.ctypes.data, dw.ctypes.data, db.ctypes.data],
        [M, D, int(w is not None), int(b is not None), int(bwd)], [float(eps)])
    return y, dx, dw, db


def _D(normalized_shape):
    shape = (normalized_shape,) if isinstance(normalized_shape, (int, np.integer)) else tuple(normalized_shape)
    if not shape or any(int(s) < 1 for s in shape):  # glue: shape dims of the argument
        raise ValueError("normalized_shape must be positive")
    return shape, int(np.prod(shape))  # glue: product of the shape dims


def layer_norm_forward(x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """`F.layer_norm(x, normalized_shape, weight, bias, eps)`; normalized_shape
    defaults to x's last dimension."""
    x = x if is_seq_tensor(x) else _f32(x, "x")
    _, D = _D(x.shape[-1] if normalized_shape is None else normalized_shape)
    return _run(x, D, weight, bias, eps, None, numeric_mode)[0]


def layer_norm_backward(dy, x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """(dx, dweight, dbias) of `layer_norm_forward`; dweight / dbias are None
    when there is no weight / bias."""
    x = x if is_seq_tensor(x) else _f32(x, "x")
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
        self._stats = None

    def _dev_binding(self):
        """The sequence binding when it has the resident LayerNorm entry, else None."""
        if _CLASS_DEVICE_IO_OFF:
            return None
        try:
            b = _backend.binding(_SEQ_BINDING, self.numeric_mode)
        except Exception:  # noqa: BLE001  (no GPU binding on this install)
            return None
        return b if resident_binding(b) and callable(getattr(b, "layer_norm_dev", None)) else None

    @property
    def _device_io(self):
        """Whether this layer takes and returns resident tensors (`to_device`)."""
        return self._dev_binding() is not None

    def to_device(self, a):
        """float32 array `a` as a resident tensor this layer's forward and
        backward keep on the device (the array itself when device I/O is off)."""
        if is_seq_tensor(a):
            return a
        a = _f32(a, "x")
        return a if self._dev_binding() is None else _seq_to_device(a, self.numeric_mode)

    def forward(self, x):
        x = x if is_seq_tensor(x) else _f32(x, "x")
        if tuple(x.shape[x.ndim - len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(f"LayerNorm: trailing shape {x.shape} does not end in {self.normalized_shape}")
        self._x = x
        self._stats = None
        if is_seq_tensor(x) and not _KEEP_STATS_OFF:
            M = x.size // self._D
            self._stats = (SequenceDeviceTensor._new(x.b, (M,)), SequenceDeviceTensor._new(x.b, (M,)))
        return _run(x, self._D, self.weight, self.bias, self.eps, None, self.numeric_mode, self._stats)[0]

    __call__ = forward

    def backward(self, dy):
        if self._x is None:
            raise RuntimeError("LayerNorm.backward: call forward first")
        _, dx, dw, db = _run(self._x, self._D, self.weight, self.bias, self.eps, dy, self.numeric_mode,
                             self._stats)
        self.weight_grad = None if self.weight is None else dw.reshape(self.normalized_shape)
        self.bias_grad = None if self.bias is None else db.reshape(self.normalized_shape)
        return dx
