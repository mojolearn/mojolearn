# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resident float32 tensors of the sequence binding (lane gap-neural-io,
2026-10-08; plan docs/plans/gaps-2026-10-08.md Section 4).

`to_device(array)` uploads once and returns a `SequenceDeviceTensor` (a
`sequence/seq_tensor.mojo` handle on the GPU binding's shared context). The
optimizers (`RMSprop`, `Adagrad`, `Adamax`, `NAdam`, `Lion`, `LAMB`,
`Adafactor`) take such tensors as params and grads and update the params
where they live; `layer_norm_forward` / `layer_norm_backward` and
`LayerNorm` take x and dy and return y and dx as tensors. Nothing crosses the
bus until `.numpy()`. The kernels are the host-array entries' launches on the
same values: the same bits. On a binding without the resident entries (the
CPU host binding, a build that lacks them) `to_device` returns the float32
array it was given, so the caller's code runs unchanged on host arrays.

Glue only: argument checks, handles and shapes; no arithmetic on data."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_device')

from . import _backend

_BINDING = "_mojolearn_x_sequence"
_ENTRIES = ("seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download")


def _size(shape):
    n = 1
    for v in shape:  # glue: a shape tuple's few ints
        n *= int(v)
    return n


def resident_binding(b):
    """Whether binding `b` has the resident tensor entries."""
    return all(callable(getattr(b, n, None)) for n in _ENTRIES)  # glue: the named binding entry points


class SequenceDeviceTensor:
    """A float32 tensor resident on the sequence binding's device context:
    a handle and a shape. Read it back with `.numpy()`; freed with the
    object (a `reshape` view keeps its base alive)."""
    __slots__ = ("b", "shape", "h", "_base", "__weakref__")
    _device_tensor = True
    dtype = "float32"

    def __init__(self, binding, shape, h, base=None):
        self.b, self.shape, self.h, self._base = binding, tuple(int(v) for v in shape), int(h), base  # glue: a shape tuple's few ints

    @classmethod
    def _new(cls, b, shape, zero=False):
        """An unwritten (or +0.0 filled) resident tensor of `shape`."""
        return cls(b, shape, int(b.seq_tensor_alloc(max(_size(shape), 1), 1 if zero else 0)))

    @classmethod
    def _wrap(cls, b, a):
        a = np.ascontiguousarray(a, np.float32)
        t = cls._new(b, a.shape)
        if a.size:
            b.seq_tensor_upload(t.h, a.ctypes.data, a.size)
        return t

    @property
    def size(self):
        return _size(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def numpy(self):
        out = np.empty(self.shape, np.float32)
        if out.size:
            self.b.seq_tensor_download(self.h, out.ctypes.data, out.size)
        return out

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype, copy=False)

    def copy(self):
        """A new resident tensor with this one's words (device to device)."""
        t = SequenceDeviceTensor._new(self.b, self.shape)
        if self.size:
            self.b.seq_tensor_copy(t.h, self.h, self.size)
        return t

    def upload(self, a):
        """Overwrite the words from host float32 array `a` of this shape."""
        a = np.ascontiguousarray(a, np.float32)
        if a.shape != self.shape:
            raise ValueError(f"mojolearn: an array of shape {a.shape} for a tensor of {self.shape}")
        if a.size:
            self.b.seq_tensor_upload(self.h, a.ctypes.data, a.size)
        return self

    def reshape(self, *shape):
        """The same resident words under another shape (no copy)."""
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = tuple(shape[0])
        shape = [int(v) for v in shape]  # glue: a shape tuple's few ints
        if shape.count(-1) == 1:
            known = _size(v for v in shape if v != -1)  # glue: a shape tuple's few ints
            shape[shape.index(-1)] = self.size // known if known else 0
        if _size(shape) != self.size:
            raise ValueError(f"mojolearn: cannot reshape {self.shape} to {tuple(shape)}")
        return SequenceDeviceTensor(self.b, shape, self.h, base=self)

    def __del__(self):
        if self._base is None:
            try:
                self.b.seq_tensor_free(self.h)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass


def is_seq_tensor(x):
    return isinstance(x, SequenceDeviceTensor)


def to_device(x, numeric_mode=None):
    """float32 array `x` as a `SequenceDeviceTensor` (uploaded once, here),
    or `x` itself as a float32 array on a binding without resident tensors."""
    if isinstance(x, SequenceDeviceTensor):
        return x
    a = np.asarray(x)
    if a.dtype != np.float32:
        raise TypeError(f"mojolearn: to_device takes float32 (got {a.dtype}); convert it yourself")
    a = np.ascontiguousarray(a)
    b = _backend.binding(_BINDING, numeric_mode)
    if not resident_binding(b):
        return a
    return SequenceDeviceTensor._wrap(b, a)


def on_binding(b, *ts):
    """Whether every one of `ts` is a resident tensor of binding `b`."""
    return all(isinstance(t, SequenceDeviceTensor) and t.b is b for t in ts)  # glue: the call's arguments


__all__ = ["SequenceDeviceTensor", "to_device"]
