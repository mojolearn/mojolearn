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

from ._training_impl import _OPT_NATIVE as _SEQ_NATIVE, _OPT_CONTEXT as _SEQ_CONTEXT

_BINDING = "_mojolearn_x_sequence"
#: the before arm of the A/B (as x_cnn's MOJOLEARN_XCNN_DEVICE_IO_OFF):
#: `to_device` returns the host array, so every call takes the host-array
#: entries again; the wave's MOJOLEARN_IDN_ALL_OFF arm turns it off too
_DEVICE_IO_OFF = (__import__("os").environ.get("MOJOLEARN_SEQ_DEVICE_IO_OFF", "") == "1"
                  or __import__("os").environ.get("MOJOLEARN_IDN_ALL_OFF", "") == "1")
_ENTRIES = ("seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download")


def _size(shape):
    return _SEQ_NATIVE("seqdev_size_of", None, (shape,), None, _SEQ_CONTEXT)


def resident_binding(b):
    """Whether binding `b` has the resident tensor entries."""
    return _SEQ_NATIVE("seqdev_resident", None, (b,), None, _SEQ_CONTEXT)


class SequenceDeviceTensor:
    """A float32 tensor resident on the sequence binding's device context:
    a handle and a shape. Read it back with `.numpy()`; freed with the
    object (a `reshape` view keeps its base alive)."""
    __slots__ = ("b", "shape", "h", "_base", "__weakref__")
    _device_tensor = True
    dtype = "float32"

    def __init__(self, binding, shape, h, base=None):
        return _SEQ_NATIVE("seqdev_construct", self, (binding, shape, h, base), None, _SEQ_CONTEXT)

    @classmethod
    def _new(cls, b, shape, zero=False):
        """An unwritten (or +0.0 filled) resident tensor of `shape`."""
        return _SEQ_NATIVE("seqdev_new", cls, (b, shape, zero), None, _SEQ_CONTEXT)

    @classmethod
    def _wrap(cls, b, a):
        return _SEQ_NATIVE("seqdev_wrap", cls, (b, a), None, _SEQ_CONTEXT)

    @property
    def size(self):
        return _SEQ_NATIVE("seqdev_size", self, (), None, _SEQ_CONTEXT)

    @property
    def ndim(self):
        return _SEQ_NATIVE("seqdev_ndim", self, (), None, _SEQ_CONTEXT)

    def numpy(self):
        return _SEQ_NATIVE("seqdev_host", self, (), None, _SEQ_CONTEXT)

    # Remaining conversion debt: dtype-specific NumPy conversion retains its
    # existing interface until all supported dtype conversions are native.
    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype, copy=False)

    def copy(self):
        """A new resident tensor with this one's words (device to device)."""
        return _SEQ_NATIVE("seqdev_copy", self, (), None, _SEQ_CONTEXT)

    def upload(self, a):
        """Overwrite the words from host float32 array `a` of this shape."""
        return _SEQ_NATIVE("seqdev_upload", self, (a,), None, _SEQ_CONTEXT)

    def reshape(self, *shape):
        """The same resident words under another shape (no copy)."""
        return _SEQ_NATIVE("seqdev_reshape", self, (shape,), None, _SEQ_CONTEXT)

    def __del__(self):
        return _SEQ_NATIVE("seqdev_finalize", self, (), None, _SEQ_CONTEXT)


def is_seq_tensor(x):
    return _SEQ_NATIVE("seqdev_is", None, (x,), None, _SEQ_CONTEXT)


def to_device(x, numeric_mode=None):
    """float32 array `x` as a `SequenceDeviceTensor` (uploaded once, here),
    or `x` itself as a float32 array on a binding without resident tensors."""
    return _SEQ_NATIVE("seqdev_to", None, (x,), numeric_mode, _SEQ_CONTEXT)


def on_binding(b, *ts):
    """Whether every one of `ts` is a resident tensor of binding `b`."""
    return _SEQ_NATIVE("seqdev_on", None, (b, ts), None, _SEQ_CONTEXT)


__all__ = ["SequenceDeviceTensor", "to_device"]

# glue: connect the native device ABI after its fixed owner layout exists
from . import _training_impl as _training
from ._native_optimizer_bridge import connect_sequence_device as _connect_sequence_device
_SEQ_NATIVE, _SEQ_CONTEXT = _connect_sequence_device(_training, __import__(__name__, fromlist=["*"]))
