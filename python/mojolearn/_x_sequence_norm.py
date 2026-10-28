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

from . import _x_sequence_device as _sequence_device  # glue: connect native tensor storage
from ._training_impl import _OPT_NATIVE as _SEQ_NATIVE, _OPT_CONTEXT as _SEQ_CONTEXT

_ALL_OFF = _os.environ.get("MOJOLEARN_IDN_ALL_OFF", "") == "1"
#: the before arm of the class device-I/O switch (see the module docstring)
_CLASS_DEVICE_IO_OFF = (_os.environ.get("MOJOLEARN_LN_CLASS_DEVICE_IO_OFF", "") == "1"
                        or _os.environ.get("MOJOLEARN_SEQ_DEVICE_IO_OFF", "") == "1" or _ALL_OFF)
#: the before arm of the kept-stats switch: the backward recomputes the forward
_KEEP_STATS_OFF = _os.environ.get("MOJOLEARN_LN_KEEP_STATS_OFF", "") == "1" or _ALL_OFF
_SEQ_CONTEXT["seqln_keep_stats_off"] = _KEEP_STATS_OFF  # glue: the native forward reads the before-arm switch


# The former private NumPy runners are superseded by the native owner.
def layer_norm_forward(x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """Layer normalization over the trailing dimensions, as `torch.nn.functional.layer_norm`.

    Each row of the trailing `normalized_shape` dimensions is normalized with its mean
    and biased variance, ``(x - mean) / sqrt(var + eps)``, then scaled by `weight` and
    shifted by `bias` when they are given.

    Parameters
    ----------
    x : ndarray of float32
        Input; float64 raises `TypeError`.
    normalized_shape : int or tuple of int or None, default None
        Trailing shape to normalize over. None uses the last dimension of `x`.
    weight : ndarray or None, default None
        Elementwise scale with `normalized_shape` values.
    bias : ndarray or None, default None
        Elementwise shift with `normalized_shape` values.
    eps : float, default 1e-5
        Added to the variance inside the square root.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    y : ndarray of float32
        Normalized output, the shape of `x`.

    Raises
    ------
    ValueError
        When the trailing dimensions of `x` do not match `normalized_shape`, or
        `weight`/`bias` have the wrong size.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> x = np.random.default_rng(0).random((32, 64), dtype=np.float32)
    >>> y = mojolearn.layer_norm_forward(x)
    """
    return _SEQ_NATIVE("seqln_forward", None, (x, normalized_shape, weight, bias, eps), numeric_mode, _SEQ_CONTEXT)

def layer_norm_backward(dy, x, normalized_shape=None, weight=None, bias=None, eps=1e-5, numeric_mode=None):
    """Gradients of `layer_norm_forward`.

    Parameters
    ----------
    dy : ndarray of float32
        Gradient of the loss with respect to the output, the shape of `x`.
    x : ndarray of float32
        The forward input.
    normalized_shape : int or tuple of int or None, default None
        Trailing shape normalized over. None uses the last dimension of `x`.
    weight : ndarray or None, default None
        The forward's elementwise scale.
    bias : ndarray or None, default None
        The forward's elementwise shift.
    eps : float, default 1e-5
        The forward's epsilon.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    dx : ndarray of float32
        Gradient with respect to `x`.
    dweight : ndarray or None
        Gradient with respect to `weight`; None when there is no weight.
    dbias : ndarray or None
        Gradient with respect to `bias`; None when there is no bias.

    Raises
    ------
    ValueError
        When `dy` does not have the shape of `x`, or shapes do not match
        `normalized_shape`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> x = np.random.default_rng(0).random((32, 64), dtype=np.float32)
    >>> dx, dw, db = mojolearn.layer_norm_backward(np.ones_like(x), x)
    """
    return _SEQ_NATIVE("seqln_backward", None, (dy, x, normalized_shape, weight, bias, eps), numeric_mode, _SEQ_CONTEXT)

class LayerNorm:
    """Layer normalization layer, as `torch.nn.LayerNorm`.

    Holds a weight (initialized to ones) and a bias (zeros) over `normalized_shape`.
    `forward(x)` (also ``layer(x)``) normalizes the trailing dimensions of `x`;
    `backward(dy)`, called after a forward, returns dx and stores the parameter
    gradients in `weight_grad` and `bias_grad`. Inputs must be float32.

    Parameters
    ----------
    normalized_shape : int or tuple of int
        Trailing shape to normalize over. Every dimension must be positive.
    eps : float, default 1e-5
        Added to the variance inside the square root.
    elementwise_affine : bool, default True
        Learn a weight (and a bias when `bias` is True).
    bias : bool, default True
        Learn a bias. Ignored when `elementwise_affine` is False.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    weight : ndarray or None
        Elementwise scale, None without `elementwise_affine`.
    bias : ndarray or None
        Elementwise shift, None without a bias.
    weight_grad : ndarray or None
        Gradient of `weight` from the last `backward`.
    bias_grad : ndarray or None
        Gradient of `bias` from the last `backward`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> x = np.random.default_rng(0).random((32, 64), dtype=np.float32)
    >>> ln = mojolearn.LayerNorm(64)
    >>> y = ln(x)
    >>> dx = ln.backward(np.ones_like(y))
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, numeric_mode=None):
        return _SEQ_NATIVE("seqln_initialize", self,
                           (normalized_shape, eps, elementwise_affine, bias), numeric_mode, _SEQ_CONTEXT)

    def _dev_binding(self):
        """The sequence binding when it has the resident LayerNorm entry, else None."""
        return _SEQ_NATIVE("seqln_device_binding", self, (), None, _SEQ_CONTEXT)

    @property
    def _device_io(self):
        """Whether this layer takes and returns resident tensors (`to_device`)."""
        return _SEQ_NATIVE("seqln_device_io", self, (), None, _SEQ_CONTEXT)

    def to_device(self, a):
        """Upload a float32 array once so `forward` and `backward` keep it on the device.

        Passing the result to `forward` keeps the output and gradients on the device and
        avoids a copy in each direction per call. When this install has no device path,
        the array itself is returned.

        Parameters
        ----------
        a : ndarray of float32
            Array to upload.

        Returns
        -------
        device tensor or ndarray
        """
        return _SEQ_NATIVE("seqln_to_device", self, (a,), None, _SEQ_CONTEXT)

    def forward(self, x):
        """Normalize `x` over the trailing `normalized_shape` dimensions.

        Parameters
        ----------
        x : ndarray of float32
            Input whose trailing shape is `normalized_shape`; otherwise `ValueError`.

        Returns
        -------
        y : ndarray of float32
            Normalized output, the shape of `x`.
        """
        return _SEQ_NATIVE("seqln_layer", self, (x,), None, _SEQ_CONTEXT)

    __call__ = forward

    def backward(self, dy):
        """Backpropagate through the last `forward`.

        Sets `weight_grad` and `bias_grad`. Raises `RuntimeError` when no forward has run.

        Parameters
        ----------
        dy : ndarray of float32
            Gradient with respect to the output, the shape of the forward input.

        Returns
        -------
        dx : ndarray of float32
            Gradient with respect to the forward input.
        """
        return _SEQ_NATIVE("seqln_layer_backward", self, (dy,), None, _SEQ_CONTEXT)


# glue: native instance storage and hooks are connected once after class definition.
_ln_mro = LayerNorm.__mro__
_ln_rows = tuple(vars(base) for base in _ln_mro)  # glue: fixed class ancestry
_ln_descriptor = next(row["__dict__"] for row in _ln_rows if "__dict__" in row)  # glue: native instance dictionary
_ln_hooks = tuple((base, name, row.get(name)) for base, row in zip(_ln_mro, _ln_rows)
                  for name in ("__getattribute__", "__getattr__", "__setattr__", "__dict__"))  # glue: fixed native storage hooks
_SEQ_CONTEXT["seqln_layout"] = (LayerNorm, _ln_mro, _ln_descriptor.__get__, _ln_hooks)
_SEQ_CONTEXT["seqln_class_device_io_off"] = _CLASS_DEVICE_IO_OFF
