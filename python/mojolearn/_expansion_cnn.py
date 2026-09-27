# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `cnn` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_cnn": "_mojolearn_x_cnn_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""
from . import _backend

__all__ = ["Conv2d", "Conv1d", "MaxPool2d", "AvgPool2d", "MaxPool1d", "AvgPool1d"]

_BINDING = "_mojolearn_x_cnn"


def _np():
    import numpy as np
    return np


def _f32(x, name):
    np = _np()
    a = np.asarray(x)
    if a.dtype != np.float32:
        raise TypeError(f"mojolearn: {name} must be float32 (got {a.dtype}); convert it yourself so the "
                        "bits that ran are bits you made")
    return np.ascontiguousarray(a)


def _pair(v, name):
    if isinstance(v, int):
        v = (v, v)
    v = tuple(int(t) for t in v)
    if len(v) != 2:
        raise ValueError(f"mojolearn: {name} must be an int or a pair")
    return v


def _kaiming_uniform(rng, shape, fan_in):
    """PyTorch's default Conv/Linear init (kaiming_uniform_(a=sqrt(5)) for the
    weight, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for the bias), drawn from
    NumPy's Generator in float64 and rounded once to float32, so every
    platform starts from the same bytes."""
    np = _np()
    bound = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


class _Layer:
    """A layer on `_mojolearn_x_cnn` (the GPU binding, or its CPU host twin
    on a CPU-only install / MOJOLEARN_VENDOR=cpu)."""
    numeric_mode = None

    def _binding(self):
        mode = self.numeric_mode or _backend.default_mode()
        self.numeric_mode_ = mode
        return _backend.binding(_BINDING, mode)


class Conv2d(_Layer):
    """2-D convolution, PyTorch `nn.Conv2d` semantics (groups=1, zero
    padding): im2col onto the pinned GEMM (mojolearn.identical.gemm.fp32.v1)
    forward and backward. `forward(x)` takes (N, C, H, W) float32;
    `backward(grad_out)` returns grad_x and sets `grad_weight_`, `grad_bias_`.
    `transform(X)` is the row form: X is (n, C*H*W) with `input_shape=(C, H, W)`."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, random_state=0, input_shape=None, numeric_mode=None):
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        self.dilation = _pair(dilation, "dilation")
        self.bias = bool(bias)
        self.random_state = random_state
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode
        np = _np()
        kh, kw = self.kernel_size
        fan_in = self.in_channels * kh * kw
        rng = np.random.default_rng(random_state)
        self.weight_ = _kaiming_uniform(rng, (self.out_channels, self.in_channels, kh, kw), fan_in)
        self.bias_ = (_kaiming_uniform(rng, (self.out_channels,), fan_in) if self.bias
                      else np.zeros(self.out_channels, np.float32))

    def set_weights(self, weight, bias=None):
        np = _np()
        w = _f32(weight, "weight")
        if w.shape != self.weight_.shape:
            raise ValueError(f"mojolearn: weight shape {w.shape}, expected {self.weight_.shape}")
        self.weight_ = w.copy()
        if bias is not None:
            self.bias_ = _f32(bias, "bias").reshape(self.out_channels).copy()
        elif not self.bias:
            self.bias_ = np.zeros(self.out_channels, np.float32)
        return self

    def _params(self, shape):
        n, c, h, w = shape
        if c != self.in_channels:
            raise ValueError(f"mojolearn: input has {c} channels, the layer {self.in_channels}")
        kh, kw = self.kernel_size
        return [n, c, h, w, self.out_channels, kh, kw, *self.stride, *self.padding, *self.dilation, 0, 0,
                1 if self.bias else 0, 0]

    def _out_shape(self, shape):
        oh, ow = self._binding().x_cnn_conv_shape(self._params(shape))
        return (shape[0], self.out_channels, int(oh), int(ow))

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: Conv2d.forward takes (N, C, H, W)")
        prm = self._params(x.shape)
        b = self._binding()
        out = np.empty(self._out_shape(x.shape), np.float32)
        b.x_cnn_conv2d_forward(x.ctypes.data, self.weight_.ctypes.data, self.bias_.ctypes.data,
                               out.ctypes.data, prm)
        self._x = x
        return out

    def backward(self, grad_out, x=None):
        np = _np()
        x = self._x if x is None else _f32(x, "x")
        g = _f32(grad_out, "grad_out")
        shape = self._out_shape(x.shape)
        if g.shape != shape:
            raise ValueError(f"mojolearn: grad_out shape {g.shape}, expected {shape}")
        dx = np.empty(x.shape, np.float32)
        dw = np.empty(self.weight_.shape, np.float32)
        db = np.empty(self.out_channels, np.float32)
        self._binding().x_cnn_conv2d_backward(x.ctypes.data, self.weight_.ctypes.data, g.ctypes.data,
                                              dx.ctypes.data, dw.ctypes.data, db.ctypes.data,
                                              self._params(x.shape))
        self.grad_weight_ = dw
        self.grad_bias_ = db if self.bias else np.zeros_like(db)
        return dx

    def _rows(self, X):
        X = _f32(X, "X")
        if X.ndim == 2:
            if self.input_shape is None:
                raise ValueError("mojolearn: row input needs input_shape=(C, H, W)")
            return X.reshape((X.shape[0],) + tuple(self.input_shape))
        return X

    def transform(self, X):
        out = self.forward(self._rows(X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        """Layers carry their weights from construction; fit only records the input shape."""
        X = self._rows(X)
        self.n_features_in_ = int(_np().prod(X.shape[1:]))
        return self


class Conv1d(Conv2d):
    """1-D convolution, PyTorch `nn.Conv1d` semantics: Conv2d over (N, C, 1, L)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, random_state=0, input_shape=None, numeric_mode=None):
        k, s, p, d = (int(v if isinstance(v, int) else v[0]) for v in (kernel_size, stride, padding, dilation))
        super().__init__(in_channels, out_channels, (1, k), (1, s), (0, p), (1, d), bias, random_state,
                         None, numeric_mode)
        self.input_shape = input_shape
        self.weight_ = self.weight_.reshape(self.out_channels, self.in_channels, 1, k)

    def set_weights(self, weight, bias=None):
        w = _f32(weight, "weight")
        return super().set_weights(w.reshape(self.out_channels, self.in_channels, 1, -1), bias)

    def forward(self, x):
        x = _f32(x, "x")
        if x.ndim != 3:
            raise ValueError("mojolearn: Conv1d.forward takes (N, C, L)")
        out = super().forward(x[:, :, None, :])
        return out[:, :, 0, :]

    def backward(self, grad_out, x=None):
        g = _f32(grad_out, "grad_out")
        xx = None if x is None else _f32(x, "x")[:, :, None, :]
        dx = super().backward(g[:, :, None, :], xx)
        self.grad_weight_ = self.grad_weight_[:, :, 0, :]
        return dx[:, :, 0, :]

    @property
    def weight1d_(self):
        return self.weight_[:, :, 0, :]


class _Pool2d(_Layer):
    """Shared pooling plumbing: (N, C, H, W) float32, floor mode."""
    _kind = None

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                 count_include_pad=True, input_shape=None, numeric_mode=None):
        if ceil_mode:
            raise NotImplementedError("mojolearn: ceil_mode=True is not implemented (NOT_IMPLEMENTED.tsv)")
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride if stride is not None else kernel_size, "stride")
        self.padding = _pair(padding, "padding")
        self.dilation = _pair(dilation, "dilation")
        self.count_include_pad = bool(count_include_pad)
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode

    def _params(self, shape):
        n, c, h, w = shape
        return [n, c, h, w, *self.kernel_size, *self.stride, *self.padding, *self.dilation, 0, 0,
                1 if self.count_include_pad else 0, 0]

    def _out_shape(self, shape):
        oh, ow = self._binding().x_cnn_pool_shape(self._params(shape))
        return (shape[0], shape[1], int(oh), int(ow))

    def _rows(self, X):
        return Conv2d._rows(self, X)

    def transform(self, X):
        out = self.forward(self._rows(X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        return self


class MaxPool2d(_Pool2d):
    """PyTorch `nn.MaxPool2d` (floor mode): the first maximum in (kh, kw)
    order wins a tie, a NaN wins. `indices_` holds the flat h*W + w of each
    winner (return_indices's values)."""

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: MaxPool2d.forward takes (N, C, H, W)")
        shape = self._out_shape(x.shape)
        out = np.empty(shape, np.float32)
        idx = np.empty(shape, np.int32)
        self._binding().x_cnn_maxpool2d_forward(x.ctypes.data, out.ctypes.data, idx.ctypes.data,
                                                self._params(x.shape))
        self._xshape, self.indices_ = x.shape, idx
        return out

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        dx = np.empty(self._xshape, np.float32)
        self._binding().x_cnn_maxpool2d_backward(g.ctypes.data, self.indices_.ctypes.data, dx.ctypes.data,
                                                 self._params(self._xshape))
        return dx


class AvgPool2d(_Pool2d):
    """PyTorch `nn.AvgPool2d` (floor mode, divisor_override=None)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None, input_shape=None, numeric_mode=None):
        if divisor_override is not None:
            raise NotImplementedError("mojolearn: divisor_override is not implemented (NOT_IMPLEMENTED.tsv)")
        super().__init__(kernel_size, stride, padding, 1, ceil_mode, count_include_pad, input_shape,
                         numeric_mode)

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: AvgPool2d.forward takes (N, C, H, W)")
        out = np.empty(self._out_shape(x.shape), np.float32)
        self._binding().x_cnn_avgpool2d_forward(x.ctypes.data, out.ctypes.data, self._params(x.shape))
        self._xshape = x.shape
        return out

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        dx = np.empty(self._xshape, np.float32)
        self._binding().x_cnn_avgpool2d_backward(g.ctypes.data, dx.ctypes.data, self._params(self._xshape))
        return dx


def _one(v):
    return int(v if isinstance(v, int) else v[0])


class _Pool1dMixin:
    def forward(self, x):
        x = _f32(x, "x")
        if x.ndim != 3:
            raise ValueError(f"mojolearn: {type(self).__name__}.forward takes (N, C, L)")
        return super().forward(x[:, :, None, :])[:, :, 0, :]

    def backward(self, grad_out):
        g = _f32(grad_out, "grad_out")
        return super().backward(g[:, :, None, :])[:, :, 0, :]


class MaxPool1d(_Pool1dMixin, MaxPool2d):
    """PyTorch `nn.MaxPool1d`: MaxPool2d over (N, C, 1, L)."""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                 input_shape=None, numeric_mode=None):
        k = _one(kernel_size)
        s = k if stride is None else _one(stride)
        super().__init__((1, k), (1, s), (0, _one(padding)), (1, _one(dilation)), ceil_mode, True,
                         input_shape, numeric_mode)


class AvgPool1d(_Pool1dMixin, AvgPool2d):
    """PyTorch `nn.AvgPool1d`: AvgPool2d over (N, C, 1, L)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 input_shape=None, numeric_mode=None):
        k = _one(kernel_size)
        s = k if stride is None else _one(stride)
        super().__init__((1, k), (1, s), (0, _one(padding)), ceil_mode, count_include_pad, None,
                         input_shape, numeric_mode)
