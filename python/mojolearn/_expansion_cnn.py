# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S PUBLIC DOOR.

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
from ._labels import unique_inverse
from . import _portable_math as _pm

__all__ = ["Conv2d", "Conv1d", "MaxPool2d", "AvgPool2d", "MaxPool1d", "AvgPool1d", "CNNClassifier", "BatchNorm2d", "BatchNorm1d",
           "Dropout2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "BasicBlock",
           "GCNConv", "SAGEConv"]

_BINDING = "_mojolearn_x_cnn"


def _np():
    from ._optional_numpy import require_numpy
    np = require_numpy('_expansion_cnn')
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


def _rng(random_state):
    """The layer's init stream (`_buffer.InitStream`: seeded draws in Mojo)."""
    from ._buffer import InitStream
    return InitStream(None if random_state is None else int(random_state))


def _uniform(rng, bound, shape):
    """A float32 array of U(-bound, bound) draws from `rng`, in Mojo."""
    np = _np()
    out = np.empty(shape, dtype=np.float32)
    rng.fill_uniform(out.ctypes.data, out.size, -float(bound), float(bound))
    return out


def _kaiming_uniform(rng, shape, fan_in):
    """PyTorch's default Conv/Linear init (kaiming_uniform_(a=sqrt(5)) for the
    weight, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for the bias): float64
    uniforms rounded once to float32, drawn in Mojo from the layer's seeded
    stream (lane cgr4-py-compute; it was NumPy's Generator), so every
    platform starts from the same bytes."""
    return _uniform(rng, 1.0 / _pm.sqrt(float(fan_in)), shape)


class _Layer:
    """A layer on `_mojolearn_x_cnn` (the GPU binding, or its CPU host twin
    on a CPU-only install / MOJOLEARN_VENDOR=cpu)."""
    numeric_mode = None

    def _binding(self):
        mode = self.numeric_mode or _backend.default_mode()
        self.numeric_mode_ = mode
        return _backend.binding(_BINDING, mode)


def _mixed(b):
    """Whether binding `b` has the mixed-residency entries (`x_cnn_*_m`,
    lane gap-neural-overhead2): the GPU binding does; a CPU install's twin
    runs the host entries, the same words."""
    return hasattr(b, "x_cnn_map2_m")


class _Dev:
    """A layer's resident device arrays (`x_cnn_res_alloc` handles) kept
    between calls: one named array per intermediate, reallocated only when
    its size changes, so a forward or backward allocates nothing (lane
    gap-neural-overhead2, 2026-10-02). Freed with the layer."""

    def __init__(self, binding):
        self.b = binding
        self.h = {}

    def get(self, name, n):
        n = max(int(n), 1)
        cur = self.h.get(name)
        if cur is not None and cur[1] == n:
            return cur[0]
        if cur is not None:
            self.b.x_cnn_res_free(cur[0])
            del self.h[name]
        h = int(self.b.x_cnn_res_alloc(n))
        self.h[name] = (h, n)
        return h

    def upload(self, name, a):
        """Handle `name` holding the 4-byte words of host array `a`."""
        h = self.get(name, a.size)
        self.b.x_cnn_res_upload(h, a.ctypes.data, a.size)
        return h

    def download(self, h, out):
        self.b.x_cnn_res_download(h, out.ctypes.data, out.size)
        return out

    def adopt(self, name, h, n):
        """Keep a handle made elsewhere (`x_cnn_csr_upload`) under `name`."""
        cur = self.h.pop(name, None)
        if cur is not None:
            self.b.x_cnn_res_free(cur[0])
        self.h[name] = (int(h), max(int(n), 1))
        return int(h)

    def free(self):
        h, self.h = self.h, {}
        for k, (a, _) in h.items():
            try:
                self.b.x_cnn_res_free(a)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass

    def __del__(self):
        self.free()


def _dev_of(layer, b):
    """`layer`'s `_Dev` on binding `b` (a new one if the binding changed)."""
    d = layer.__dict__.get("_dev")
    if d is None or d.b is not b:
        if d is not None:
            d.free()
        d = _Dev(b)
        layer._dev = d
    return d


class Conv2d(_Layer):
    """2-D convolution, PyTorch `nn.Conv2d` semantics: im2col onto the pinned
    GEMM (mojolearn.identical.gemm.fp32.v1) forward and backward. `padding`
    is an int, a pair, 'same' (stride 1) or 'valid'; `padding_mode` zeros,
    reflect, replicate or circular (an explicit pad, then an unpadded conv,
    as PyTorch does); `groups` splits channels into independent convs (one
    GEMM each; groups == in_channels is a depthwise conv). `forward(x)`
    takes (N, C, H, W) float32; `backward(grad_out)` returns grad_x and sets
    `grad_weight_`, `grad_bias_`. `transform(X)` is the row form: X is
    (n, C*H*W) with `input_shape=(C, H, W)`."""
    _PAD_MODES = {"zeros": 0, "reflect": 1, "replicate": 2, "circular": 3}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", random_state=0, input_shape=None, numeric_mode=None):
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.dilation = _pair(dilation, "dilation")
        self.groups = int(groups)
        if self.groups <= 0 or self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("mojolearn: in_channels and out_channels must be divisible by groups")
        if padding_mode not in self._PAD_MODES:
            raise ValueError(f"mojolearn: padding_mode must be one of {sorted(self._PAD_MODES)}, got {padding_mode!r}")
        self.padding_mode = padding_mode
        kh, kw = self.kernel_size
        if isinstance(padding, str):
            if padding == "valid":
                pads = (0, 0, 0, 0)
            elif padding == "same":
                if self.stride != (1, 1):
                    raise ValueError("mojolearn: padding='same' is not supported for strided convolutions")
                th, tw = self.dilation[0] * (kh - 1), self.dilation[1] * (kw - 1)
                pads = (th // 2, th - th // 2, tw // 2, tw - tw // 2)
            else:
                raise ValueError(f"mojolearn: invalid padding string {padding!r}, should be one of 'valid', 'same'")
            self.padding = padding
        else:
            ph, pw = _pair(padding, "padding")
            pads = (ph, ph, pw, pw)
            self.padding = (ph, pw)
        self._pads = pads
        # zeros with symmetric pads ride in the conv; anything else is an explicit pad first
        self._explicit = padding_mode != "zeros" or pads[0] != pads[1] or pads[2] != pads[3]
        self._conv_pad = (0, 0) if self._explicit else (pads[0], pads[2])
        self.bias = bool(bias)
        self.random_state = random_state
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode
        np = _np()
        fan_in = (self.in_channels // self.groups) * kh * kw
        rng = _rng(random_state)
        self.weight_ = _kaiming_uniform(rng, (self.out_channels, self.in_channels // self.groups, kh, kw), fan_in)
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
        """The binding block for ONE group's conv over the (padded) input `shape`."""
        n, c, h, w = shape
        kh, kw = self.kernel_size
        return [n, c // self.groups, h, w, self.out_channels // self.groups, kh, kw, *self.stride, *self._conv_pad,
                *self.dilation, 0, 0, 1 if self.bias else 0, 0]

    def _pad_params(self, shape):
        n, c, h, w = shape
        return [n, c, h, w, *self._pads, self._PAD_MODES[self.padding_mode]]

    def _padded_shape(self, shape):
        n, c, h, w = shape
        if not self._explicit:
            return shape
        return (n, c, h + self._pads[0] + self._pads[1], w + self._pads[2] + self._pads[3])

    def _out_shape(self, shape):
        oh, ow = self._binding().x_cnn_conv_shape(self._params(self._padded_shape(shape)))
        return (shape[0], self.out_channels, int(oh), int(ow))

    def _group(self, a, g, per):
        return _np().ascontiguousarray(a[:, g * per:(g + 1) * per]) if self.groups > 1 else a

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: Conv2d.forward takes (N, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"mojolearn: input has {x.shape[1]} channels, the layer {self.in_channels}")
        b = self._binding()
        xp = x
        if self._explicit:
            xp = np.empty(self._padded_shape(x.shape), np.float32)
            b.x_cnn_pad2d_forward(x.ctypes.data, xp.ctypes.data, self._pad_params(x.shape))
        prm = self._params(xp.shape)
        out = np.empty(self._out_shape(x.shape), np.float32)
        cg, og = self.in_channels // self.groups, self.out_channels // self.groups
        for g in range(self.groups):
            xg = self._group(xp, g, cg)
            wg = np.ascontiguousarray(self.weight_[g * og:(g + 1) * og])
            bg = np.ascontiguousarray(self.bias_[g * og:(g + 1) * og])
            yg = out if self.groups == 1 else np.empty((x.shape[0], og) + out.shape[2:], np.float32)
            b.x_cnn_conv2d_forward(xg.ctypes.data, wg.ctypes.data, bg.ctypes.data, yg.ctypes.data, prm)
            if self.groups > 1:
                out[:, g * og:(g + 1) * og] = yg
        self._x, self._xp = x, xp
        return out

    def _bind_input(self, x):
        """`forward`'s input checks and explicit pad, without the conv."""
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: Conv2d.forward takes (N, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"mojolearn: input has {x.shape[1]} channels, the layer {self.in_channels}")
        xp = x
        if self._explicit:
            xp = np.empty(self._padded_shape(x.shape), np.float32)
            self._binding().x_cnn_pad2d_forward(x.ctypes.data, xp.ctypes.data, self._pad_params(x.shape))
        self._x, self._xp = x, xp

    def backward(self, grad_out, x=None):
        np = _np()
        if x is not None:
            # lane gap-neural-overhead2: the backward needs x and its padded
            # form only; the forward this ran (a conv, its upload and its
            # download, the output discarded) is gone. The same xp words.
            self._bind_input(x)
        x, xp = self._x, self._xp
        g = _f32(grad_out, "grad_out")
        shape = self._out_shape(x.shape)
        if g.shape != shape:
            raise ValueError(f"mojolearn: grad_out shape {g.shape}, expected {shape}")
        b = self._binding()
        prm = self._params(xp.shape)
        cg, og = self.in_channels // self.groups, self.out_channels // self.groups
        dxp = np.empty(xp.shape, np.float32)
        dw = np.empty(self.weight_.shape, np.float32)
        db = np.empty(self.out_channels, np.float32)
        for k in range(self.groups):
            xg = self._group(xp, k, cg)
            gg = self._group(g, k, og)
            wg = np.ascontiguousarray(self.weight_[k * og:(k + 1) * og])
            dxg = dxp if self.groups == 1 else np.empty(xg.shape, np.float32)
            dwg = np.empty(wg.shape, np.float32)
            dbg = np.empty(og, np.float32)
            b.x_cnn_conv2d_backward(xg.ctypes.data, wg.ctypes.data, gg.ctypes.data, dxg.ctypes.data,
                                    dwg.ctypes.data, dbg.ctypes.data, prm)
            if self.groups > 1:
                dxp[:, k * cg:(k + 1) * cg] = dxg
            dw[k * og:(k + 1) * og] = dwg
            db[k * og:(k + 1) * og] = dbg
        dx = dxp
        if self._explicit:
            dx = np.empty(x.shape, np.float32)
            b.x_cnn_pad2d_backward(dxp.ctypes.data, dx.ctypes.data, self._pad_params(x.shape))
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
    """1-D convolution, PyTorch `nn.Conv1d` semantics: Conv2d over (N, C, 1, L)
    (padding int, 'same' or 'valid'; every padding_mode; groups)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", random_state=0, input_shape=None, numeric_mode=None):
        k, s, d = (int(v if isinstance(v, int) else v[0]) for v in (kernel_size, stride, dilation))
        pad = padding if isinstance(padding, str) else (0, int(padding if isinstance(padding, int) else padding[0]))
        super().__init__(in_channels, out_channels, (1, k), (1, s), pad, (1, d), groups, bias, padding_mode,
                         random_state, None, numeric_mode)
        self.input_shape = input_shape

    def set_weights(self, weight, bias=None):
        w = _f32(weight, "weight")
        return super().set_weights(w.reshape(self.out_channels, self.in_channels // self.groups, 1, -1), bias)

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
        self.ceil_mode = bool(ceil_mode)
        self.divisor_override = None
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
                1 if self.count_include_pad else 0, 0, 1 if self.ceil_mode else 0, int(self.divisor_override or 0)]

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
    """PyTorch `nn.MaxPool2d` (floor or ceil mode): the first maximum in (kh, kw)
    order wins a tie, a NaN wins. `indices_` holds the flat h*W + w of each
    winner (return_indices's values)."""

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the winners
    # stay on the device for the backward (they were downloaded by the
    # forward and uploaded again by the backward); `indices_` downloads them
    # the first time it is read. The same int32 words either way.

    @property
    def indices_(self):
        d = self.__dict__
        if d.get("_idx_host") is None and d.get("_idx_dev") is not None:
            np = _np()
            idx = np.empty(d["_idx_shape"], np.int32)
            self._dev.download(d["_idx_dev"], idx)
            d["_idx_host"] = idx
        return d.get("_idx_host")

    @indices_.setter
    def indices_(self, v):
        self.__dict__["_idx_host"] = v
        self.__dict__["_idx_dev"] = None

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError("mojolearn: MaxPool2d.forward takes (N, C, H, W)")
        shape = self._out_shape(x.shape)
        out = np.empty(shape, np.float32)
        b = self._binding()
        if _mixed(b):
            dev = _dev_of(self, b)
            h = dev.get("idx", int(np.prod(shape)))
            b.x_cnn_maxpool2d_forward_m([x.ctypes.data, out.ctypes.data, h], 0b100, self._params(x.shape))
            self._xshape = x.shape
            self.__dict__.update(_idx_host=None, _idx_dev=h, _idx_shape=shape)
            return out
        idx = np.empty(shape, np.int32)
        b.x_cnn_maxpool2d_forward(x.ctypes.data, out.ctypes.data, idx.ctypes.data, self._params(x.shape))
        self._xshape, self.indices_ = x.shape, idx
        return out

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        dx = np.empty(self._xshape, np.float32)
        b = self._binding()
        h = self.__dict__.get("_idx_dev")
        if h is not None and _mixed(b) and self.__dict__.get("_dev") is not None and self._dev.b is b:
            b.x_cnn_maxpool2d_backward_m([g.ctypes.data, h, dx.ctypes.data], 0b010, self._params(self._xshape))
            return dx
        b.x_cnn_maxpool2d_backward(g.ctypes.data, self.indices_.ctypes.data, dx.ctypes.data,
                                   self._params(self._xshape))
        return dx


class AvgPool2d(_Pool2d):
    """PyTorch `nn.AvgPool2d` (floor or ceil mode, count_include_pad,
    divisor_override)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None, input_shape=None, numeric_mode=None):
        if divisor_override is not None and int(divisor_override) <= 0:
            raise ValueError("mojolearn: divisor must be not zero")
        super().__init__(kernel_size, stride, padding, 1, ceil_mode, count_include_pad, input_shape,
                         numeric_mode)
        self.divisor_override = None if divisor_override is None else int(divisor_override)

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


class _ReLU(_Layer):
    def __init__(self, numeric_mode=None):
        self.numeric_mode = numeric_mode

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        out = np.empty_like(x)
        self._binding().x_cnn_relu_forward(x.ctypes.data, out.ctypes.data, [x.size])
        self._x = x
        return out

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        dx = np.empty_like(self._x)
        self._binding().x_cnn_relu_backward(self._x.ctypes.data, g.ctypes.data, dx.ctypes.data, [g.size])
        return dx


class _Linear(_Layer):
    """y = x W^T + b on the pinned GEMM (PyTorch nn.Linear, default init)."""

    def __init__(self, in_features, out_features, random_state=0, numeric_mode=None):
        np = _np()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.numeric_mode = numeric_mode
        rng = _rng(random_state)
        self.weight_ = _kaiming_uniform(rng, (self.out_features, self.in_features), self.in_features)
        self.bias_ = _kaiming_uniform(rng, (self.out_features,), self.in_features)

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        out = np.empty((x.shape[0], self.out_features), np.float32)
        self._binding().x_cnn_linear_forward(x.ctypes.data, self.weight_.ctypes.data, self.bias_.ctypes.data,
                                             out.ctypes.data, [x.shape[0], self.in_features, self.out_features])
        self._x = x
        return out

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        dx = np.empty_like(self._x)
        dw = np.empty_like(self.weight_)
        db = np.empty_like(self.bias_)
        self._binding().x_cnn_linear_backward(self._x.ctypes.data, self.weight_.ctypes.data, g.ctypes.data,
                                              dx.ctypes.data, dw.ctypes.data, db.ctypes.data,
                                              [g.shape[0], self.in_features, self.out_features])
        self.grad_weight_, self.grad_bias_ = dw, db
        return dx


def _softmax_xent(binding, logits, labels):
    """(mean loss, grad of the mean loss w.r.t. logits, proba)."""
    np = _np()
    logits = _f32(logits, "logits")
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    grad = np.empty_like(logits)
    proba = np.empty_like(logits)
    loss = binding.x_cnn_softmax_xent(logits.ctypes.data, labels.ctypes.data, grad.ctypes.data, proba.ctypes.data,
                                      list(logits.shape))
    return float(loss), grad, proba


def _sgd(binding, param, grad, buf, lr, momentum, weight_decay, dampening=0.0, nesterov=False, first=False):
    """In place: PyTorch SGD's step on `param` and its momentum buffer."""
    grad = _f32(grad, "grad")
    binding.x_cnn_sgd(param.ctypes.data, grad.ctypes.data, buf.ctypes.data, [param.size],
                      [float(lr), float(momentum), float(weight_decay), float(dampening), 1.0 if nesterov else 0.0,
                       1.0 if first else 0.0])


def _adam(binding, param, grad, mv, step, lr, betas, eps, weight_decay, decoupled):
    """In place: torch.optim.Adam / AdamW's step `step` (1-based) on `param`
    and mv = [exp_avg | exp_avg_sq]; the step's scalars in double, as torch
    computes them in Python."""
    grad = _f32(grad, "grad")
    binding.x_cnn_adam(param.ctypes.data, grad.ctypes.data, mv.ctypes.data, [param.size],
                       _adam_hyper(step, lr, betas, eps, weight_decay, decoupled))


def _adam_hyper(step, lr, betas, eps, weight_decay, decoupled):
    """adam_at's hyper block for 1-based step `step` (the scalars in double,
    as torch computes them in Python); `_adam` and the resident fit share it.
    DEVIATION 6900: torch's `beta ** step` calls the platform pow; here it is
    `_pm.powi`, correctly rounded (the platform's bits wherever its pow is),
    the same bits on every host. sqrt is correctly rounded everywhere."""
    if step != int(step):
        raise ValueError(f"Adam step must be a whole number, got {step!r}")
    return _adam_hyper_block(int(step), 1, lr, betas, eps, weight_decay, decoupled).reshape(-1).tolist()


def _adam_hyper_block(step0, nsteps, lr, betas, eps, weight_decay, decoupled):
    """`_adam_hyper` of steps step0 .. step0 + nsteps - 1 as an (nsteps, 9)
    float64 array, made in Mojo (the base binding's `adam_hyper_f64`; lane
    cgr4-py-compute): beta ** step by squaring in one fixed order, the
    correctly rounded sqrt, the same bits on every column."""
    np = _np()
    from ._buffer import _native
    out = np.empty((max(int(nsteps), 1), 9), dtype=np.float64)
    _native("adam_hyper_f64")(out.ctypes.data, int(step0), int(nsteps),
                              [float(lr), float(betas[0]), float(betas[1]), float(eps), float(weight_decay),
                               1.0 if decoupled else 0.0])
    return out[:int(nsteps)]


class _Res:
    """Resident arrays (DEVIATION 5718): handles the binding keeps between
    its `_r` entries (device addresses on the GPU binding, host allocations on
    the CPU twin), 4-byte words, zero filled; freed on exit."""

    def __init__(self, binding):
        self.b, self.handles = binding, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            self.b.x_cnn_res_free(h)
        self.handles = []

    def new(self, n):
        h = self.b.x_cnn_res_alloc(int(n))
        self.handles.append(h)
        return h

    def put(self, h, arr):
        np = _np()
        arr = np.ascontiguousarray(arr)
        assert arr.dtype in (np.float32, np.int32)
        self.b.x_cnn_res_upload(h, arr.ctypes.data, arr.size)

    def get(self, h, shape):
        np = _np()
        out = np.empty(shape, np.float32)
        self.b.x_cnn_res_download(h, out.ctypes.data, out.size)
        return out


#: lane/cnn-apple2 (2026-09-28): predict_proba runs at most this many rows
#: per forward pass on one set of resident arrays (every op is per row, so
#: the chunk never changes a word), and the trainer's per-step optimizer and
#: batch gather each take one binding call (the list forms). `_LEGACY_STEP`
#: True is the measurement arm's before side (per-parameter optimizer calls,
#: two gathers, one pass over all rows); never set in production.
_PREDICT_ROWS = 2048
_LEGACY_STEP = False
#: lane/py-misc (2026-09-28): with X resident, fit runs each epoch's steps in
#: ONE binding call (`x_cnn_fit_epoch_r`: the same entries' work in the same
#: order, looped in Mojo). False is the measurement arm's before side (the
#: Python step loop, also `MOJOLEARN_XCNN_PY_STEPS=1` for a whole-process
#: arm such as the identity harness); never cleared in production.
_EPOCH_ENTRY = __import__("os").environ.get("MOJOLEARN_XCNN_PY_STEPS", "") != "1"


class CNNClassifier(_Layer):
    """A small CNN image classifier, sklearn-shaped: for each entry of
    `conv_channels` a Conv2d (kernel_size, 'same' zero padding for odd
    kernels) -> ReLU -> MaxPool2d(pool_size) block (the pool is skipped once
    the map is smaller than the window), then one Linear layer to the
    classes; softmax cross entropy (mean); `optimizer` 'sgd' (torch.optim.SGD:
    momentum, dampening, nesterov, weight_decay), 'adam' or 'adamw'
    (torch.optim.Adam / AdamW: betas, eps, weight_decay). Every step runs on `_mojolearn_x_cnn` (the GPU, or the CPU
    host twin), IDENTICAL across columns. X is (n, C*H*W) rows with
    `input_shape=(C, H, W)`, or (n, C, H, W)."""

    def __init__(self, input_shape, conv_channels=(8,), kernel_size=3, pool_size=2, learning_rate=0.01,
                 momentum=0.9, weight_decay=0.0, batch_size=32, max_iter=10, shuffle=True, random_state=0,
                 numeric_mode=None, optimizer="sgd", dampening=0.0, nesterov=False, betas=(0.9, 0.999), eps=1e-8):
        if optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError(f"mojolearn: optimizer must be 'sgd', 'adam' or 'adamw', got {optimizer!r}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("mojolearn: Nesterov momentum requires a momentum and zero dampening")
        self.optimizer = optimizer
        self.dampening = float(dampening)
        self.nesterov = bool(nesterov)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.numeric_mode = numeric_mode

    def _images(self, X):
        X = _f32(X, "X")
        if X.ndim == 2:
            return np_reshape(X, (X.shape[0],) + self.input_shape)
        if X.shape[1:] != self.input_shape:
            raise ValueError(f"mojolearn: images of shape {X.shape[1:]}, the model {self.input_shape}")
        return X

    def _build(self, n_classes):
        seed = 0 if self.random_state is None else int(self.random_state)
        c, h, w = self.input_shape
        layers = []
        for i, oc in enumerate(self.conv_channels):
            layers.append(Conv2d(c, oc, self.kernel_size, padding=self.kernel_size // 2, random_state=seed + 101 * i,
                                 numeric_mode=self.numeric_mode))
            h = h + 2 * (self.kernel_size // 2) - self.kernel_size + 1
            w = w + 2 * (self.kernel_size // 2) - self.kernel_size + 1
            layers.append(_ReLU(self.numeric_mode))
            if h >= self.pool_size and w >= self.pool_size and self.pool_size > 1:
                layers.append(MaxPool2d(self.pool_size, numeric_mode=self.numeric_mode))
                h, w = h // self.pool_size, w // self.pool_size
            c = oc
        self._flat = c * h * w
        self.head_ = _Linear(self._flat, n_classes, random_state=seed + 997, numeric_mode=self.numeric_mode)
        self.layers_ = layers
        # (conv, pool or None) per block: each block runs as ONE binding call
        # each way (x_cnn_conv_block_*_r, DEVIATION 5717), the same kernels on
        # the same values as the three layer calls, on resident arrays
        # (DEVIATION 5718).
        self._blocks = []
        for i, layer in enumerate(layers):
            if isinstance(layer, Conv2d):
                nxt = layers[i + 2] if i + 2 < len(layers) else None
                self._blocks.append((layer, nxt if isinstance(nxt, MaxPool2d) else None))

    def _params(self):
        out = []
        for layer in self.layers_ + [self.head_]:
            if hasattr(layer, "weight_"):
                out.append((layer, "weight_", "grad_weight_"))
                out.append((layer, "bias_", "grad_bias_"))
        return out

    def _plan(self, n):
        """Per block (conv params, pool params or [], input size, output size,
        im2col size, conv output size) at batch size n, and the final shape."""
        plans = []
        shape = (n,) + self.input_shape
        for conv, pool in self._blocks:
            prm = conv._params(shape)
            cshape = conv._out_shape(shape)
            pprm = pool._params(cshape) if pool is not None else []
            oshape = pool._out_shape(cshape) if pool is not None else cshape
            ny = int(_np().prod(cshape))
            plans.append((prm, pprm, int(_np().prod(shape)), int(_np().prod(oshape)),
                          ny // conv.out_channels * conv.weight_[0].size, ny))
            shape = oshape
        return plans, shape

    def _resident(self, R, n, save=False, train=True):
        """The step's resident arrays at batch capacity n (DEVIATION 5718);
        `save` keeps each block's im2col matrix and conv output for its
        backward; `train` False (predict) skips the gradient arrays."""
        plans, shape = self._plan(n)
        k = len(self.classes_)
        a = dict(x=R.new(plans[0][2] if plans else n * self._flat), y=R.new(n),
                 logits=R.new(n * k), glog=R.new(n * k), proba=R.new(n * k))
        a["out"] = [R.new(p[3]) for p in plans]
        a["idx"] = [R.new(p[3]) for p in plans]
        a["gout"] = [R.new(p[3]) for p in plans] if (train or _LEGACY_STEP) else []
        a["saved"] = [[R.new(p[4]), R.new(p[5])] if save else [] for p in plans]
        if not plans:  # the head's input gradient, never read, kept apart from glog
            a["ghead"] = R.new(n * self._flat)
        return a

    def _forward_r(self, b, a, n):
        """Logits of the n rows in a["x"], every array on the binding's side."""
        plans, _ = self._plan(n)
        src = a["x"]
        for (conv, pool), p, out, idx, sv in zip(self._blocks, plans, a["out"], a["idx"], a["saved"]):
            b.x_cnn_conv_block_forward_r(src, self._rw[id(conv)][0], self._rw[id(conv)][1], out, idx, p[0], p[1], sv)
            src = out
        hw, hb = self._rw[id(self.head_)][:2]
        b.x_cnn_linear_forward_r(src, hw, hb, a["logits"], [n, self._flat, len(self.classes_)])
        return plans

    def fit(self, X, y):
        """Every step on the binding's resident arrays (DEVIATION 5718): the
        weights, optimizer state, activations and gradients stay on the
        device for the whole fit; each step uploads its batch and labels and
        downloads its loss. The same entries' kernels on the same values in
        the same order as the per-layer calls: the bits do not move."""
        np = _np()
        x = self._images(X)
        y = np.asarray(y)
        # sorted classes and inverse codes on the device (_labels.unique_inverse)
        cls, yi = unique_inverse(y)
        self.classes_ = np.asarray(cls).astype(y.dtype, copy=False) if y.dtype.kind in "biuf" else np.asarray(cls)
        yi = np.asarray(yi)
        yi = yi.astype(np.int32)
        self._build(len(self.classes_))
        b = self._binding()
        per = 2 if self.optimizer != "sgd" else 1
        params = self._params()
        # the epoch orders from one splitmix64 state, permuted in Mojo
        # (`epoch_order_i32`, sequence/schedule.mojo; lane cgr4-py-compute)
        from ._buffer import InitStream, _native
        order_state = np.array([InitStream(self.random_state).seed], dtype=np.uint64)
        n = x.shape[0]
        k = len(self.classes_)
        cap = min(self.batch_size, n)
        self.losses_, self.loss_curve_ = [], []
        with _Res(b) as R:
            self._rw = {}
            hp, hg, hbuf = [], [], []
            for layer, attr, _ in params:
                arr = getattr(layer, attr)
                hp.append(R.new(arr.size))
                R.put(hp[-1], arr)
                hg.append(R.new(arr.size))
                hbuf.append(R.new(per * arr.size))
            for i, (layer, attr, _) in enumerate(params):
                if attr == "weight_":
                    self._rw[id(layer)] = (hp[i], hp[i + 1], hg[i], hg[i + 1])
            a = self._resident(R, cap, save=True)
            sizes = [int(getattr(layer, attr).size) for layer, attr, _ in params]
            # the list forms (one optimizer call, one gather per step) on the
            # GPU binding only: the host twin's list forms are unmeasured
            lists = (not _LEGACY_STEP) and str(b.x_cnn_vendor()) != "cpu"
            # the whole X and its labels resident once when they fit a GiB:
            # each step then gathers its rows on the binding's side (a word
            # copy) instead of uploading them
            whole = x.nbytes <= (1 << 30)
            if whole:
                row = int(np.prod(self.input_shape))
                xall, yall = R.new(x.size), R.new(n)
                R.put(xall, x)
                R.put(yall, yi)
            step = 0
            epoch_entry = (whole and _EPOCH_ENTRY and not _LEGACY_STEP and hasattr(b, "x_cnn_fit_epoch_r"))
            if epoch_entry:
                bs = self.batch_size
                nsteps = (n + bs - 1) // bs
                m_last = n - (nsteps - 1) * bs
                hw_, hb_, hgw_, hgb_ = self._rw[id(self.head_)]
                blocks = []
                for j, (conv, _) in enumerate(self._blocks):
                    w_, b_, gw_, gb_ = self._rw[id(conv)]
                    blocks.append([w_, b_, gw_, gb_, a["out"][j], a["idx"][j], a["gout"][j]] + a["saved"][j])
                spec = dict(blocks=blocks, head=[hw_, hb_, hgw_, hgb_],
                            a=[a["x"], a["y"], a["logits"], a["glog"], a["proba"], a.get("ghead", 0)],
                            opt=[hp, hg, hbuf, sizes], data=[xall, yall, row], dims=[self._flat, k],
                            plan_full=[[list(p[0]), list(p[1])] for p in self._plan(cap)[0]],
                            plan_last=[[list(p[0]), list(p[1])] for p in self._plan(m_last)[0]])
                sgd_row = [self.learning_rate, self.momentum, self.weight_decay, self.dampening,
                           1.0 if self.nesterov else 0.0, 0.0]
            for _ in range(self.max_iter):
                order = np.empty(n, dtype=np.int32)
                _native("epoch_order_i32")(order.ctypes.data, n, 1 if self.shuffle else 0, order_state.ctypes.data)
                if epoch_entry:
                    rows = np.ascontiguousarray(order, dtype=np.int32)
                    if self.optimizer == "sgd":
                        hyper = np.tile(np.asarray(sgd_row, dtype=np.float64), (nsteps, 1))
                        if step == 0:
                            hyper[0, 5] = 1.0
                    else:
                        hyper = np.ascontiguousarray(_adam_hyper_block(
                            step + 1, nsteps, self.learning_rate, self.betas, self.eps, self.weight_decay,
                            self.optimizer == "adamw"))
                    losses = np.empty(nsteps, dtype=np.float64)
                    b.x_cnn_fit_epoch_r(spec, rows.ctypes.data, hyper.ctypes.data, losses.ctypes.data,
                                        [n, bs, 0 if self.optimizer == "sgd" else 1])
                    step += nsteps
                    epoch = losses.tolist()
                    self.losses_.extend(epoch)
                    # the same `_pm.nsum` as the step loop below (DEVIATION 6901; py-consolidated)
                    self.loss_curve_.append(_pm.nsum(epoch) / len(epoch))
                    continue
                epoch = []
                for s in range(0, n, self.batch_size):
                    idx = order[s:s + self.batch_size]
                    m = len(idx)
                    if whole:
                        rows = np.ascontiguousarray(idx, dtype=np.int32)
                        if not lists:
                            b.x_cnn_res_gather(a["x"], xall, rows.ctypes.data, [m, row])
                            b.x_cnn_res_gather(a["y"], yall, rows.ctypes.data, [m, 1])
                        else:
                            b.x_cnn_res_gather([a["x"], a["y"]], [xall, yall], rows.ctypes.data, [m, row, 1])
                    else:
                        # X above a GiB stays on the host: the batch rows into
                        # one staging block by the base binding's byte gather
                        # (lane pyglue-numeric: numpy fancy indexing), then up
                        rows64 = idx.astype(np.int64)
                        xb = np.empty((m,) + x.shape[1:], np.float32)
                        yb = np.empty(m, np.int32)
                        _native("gather_rows_bytes")(x.ctypes.data, xb.ctypes.data, rows64.ctypes.data,
                                                     n, m, x.nbytes // n)
                        _native("gather_rows_bytes")(yi.ctypes.data, yb.ctypes.data, rows64.ctypes.data,
                                                     n, m, 4)
                        R.put(a["x"], xb)
                        R.put(a["y"], yb)
                    plans = self._forward_r(b, a, m)
                    loss = float(b.x_cnn_softmax_xent_r(a["logits"], a["y"], a["glog"], a["proba"], [m, k]))
                    hw, _, hgw, hgb = self._rw[id(self.head_)]
                    last = a["out"][-1] if plans else a["x"]
                    glast = a["gout"][-1] if plans else a["ghead"]
                    b.x_cnn_linear_backward_r(last, hw, a["glog"], glast, hgw, hgb, [m, self._flat, k])
                    for j in range(len(self._blocks) - 1, -1, -1):
                        conv = self._blocks[j][0]
                        prm, pprm = plans[j][:2]
                        w_, b_, gw_, gb_ = self._rw[id(conv)]
                        src = a["out"][j - 1] if j > 0 else a["x"]
                        dx = a["gout"][j - 1] if j > 0 else 0
                        b.x_cnn_conv_block_backward_r(src, w_, b_, a["gout"][j], a["idx"][j],
                                                      [dx, gw_, gb_] + a["saved"][j], prm, pprm)
                    step += 1
                    if lists:
                        # every parameter in one binding call, the same launches in the same order
                        if self.optimizer == "sgd":
                            b.x_cnn_sgd_r(hp, hg, hbuf, sizes,
                                          [self.learning_rate, self.momentum, self.weight_decay, self.dampening,
                                           1.0 if self.nesterov else 0.0, 1.0 if step == 1 else 0.0])
                        else:
                            b.x_cnn_adam_r(hp, hg, hbuf, sizes, _adam_hyper(step, self.learning_rate, self.betas,
                                                                            self.eps, self.weight_decay,
                                                                            self.optimizer == "adamw"))
                    for (layer, attr, _), p_, g_, buf in zip(params, hp, hg, hbuf) if not lists else ():
                        size = getattr(layer, attr).size
                        if self.optimizer == "sgd":
                            b.x_cnn_sgd_r(p_, g_, buf, [size],
                                          [self.learning_rate, self.momentum, self.weight_decay, self.dampening,
                                           1.0 if self.nesterov else 0.0, 1.0 if step == 1 else 0.0])
                        else:
                            b.x_cnn_adam_r(p_, g_, buf, [size], _adam_hyper(step, self.learning_rate, self.betas,
                                                                            self.eps, self.weight_decay,
                                                                            self.optimizer == "adamw"))
                    epoch.append(loss)
                self.losses_.extend(epoch)
                # CPython 3.12+'s sum spelled out: the same bits on every Python (DEVIATION 6901)
                self.loss_curve_.append(_pm.nsum(epoch) / len(epoch))
            for (layer, attr, gattr), p_, g_, buf in zip(params, hp, hg, hbuf):
                arr = getattr(layer, attr)
                setattr(layer, attr, R.get(p_, arr.shape))
                setattr(layer, gattr, R.get(g_, arr.shape))
            self._bufs = [R.get(buf, (per * getattr(l, at).size,)) for (l, at, _), buf in zip(params, hbuf)]
            self._rw = {}
        self.n_features_in_ = int(np.prod(self.input_shape))
        return self

    def predict_proba(self, X):
        return self._run_predict(X, False)

    def _run_predict(self, X, codes):
        """predict_proba (n, k) float32, or with `codes` each row's class
        index (int32), the argmax on the device (`x_cnn_res_argmax`; lane
        pyglue-numeric: numpy's argmax on the downloaded probabilities)."""
        np = _np()
        x = self._images(X)
        n = x.shape[0]
        k = len(self.classes_)
        b = self._binding()
        # lane/cnn-apple2: at most _PREDICT_ROWS rows per pass on one set of
        # resident arrays (every op is per row: the same words)
        cap = n if _LEGACY_STEP else max(1, min(n, _PREDICT_ROWS))
        proba = np.empty((n, k), np.float32) if not codes else None
        lab = np.empty(n, np.int32) if codes else None
        with _Res(b) as R:
            self._rw = {}
            for layer in [c for c, _ in self._blocks] + [self.head_]:
                hw, hb = R.new(layer.weight_.size), R.new(layer.bias_.size)
                R.put(hw, layer.weight_)
                R.put(hb, layer.bias_)
                self._rw[id(layer)] = (hw, hb)
            a = self._resident(R, cap, train=False)
            R.put(a["y"], np.full(cap, -1, np.int32))
            for s in range(0, n, cap):
                m = min(cap, n - s)
                R.put(a["x"], np.ascontiguousarray(x[s:s + m]))
                self._forward_r(b, a, m)
                b.x_cnn_softmax_xent_r(a["logits"], a["y"], a["glog"], a["proba"], [m, k])
                if codes:
                    b.x_cnn_res_argmax(a["proba"], lab[s:s + m].ctypes.data, [m, k])
                else:
                    b.x_cnn_res_download(a["proba"], proba[s:s + m].ctypes.data, m * k)
            self._rw = {}
        return lab if codes else proba

    def predict(self, X):
        np = _np()
        codes = self._run_predict(X, True)
        cls = self.classes_
        if cls.dtype.kind in "iuf":
            # classes_[codes] by the base binding's native gather (the decode
            # half of label encoding)
            from ._buffer import _native
            table = np.ascontiguousarray(cls, dtype=np.float64 if cls.dtype.kind == "f" else np.int64)
            out = np.empty(codes.shape[0], table.dtype)
            c64 = codes.astype(np.int64)
            _native("gather_f64" if cls.dtype.kind == "f" else "gather_i64")(
                table.ctypes.data, table.shape[0], c64.ctypes.data, c64.shape[0], out.ctypes.data)
            return out.astype(cls.dtype, copy=False)
        return cls[codes]   # glue: label objects (str, bool) no native table holds

    def weights(self):
        """Every trained array, in layer order: [w0, b0, w1, b1, ...]."""
        return [getattr(l, a) for l, a, _ in self._params()]


def np_reshape(X, shape):
    return _np().ascontiguousarray(X).reshape(shape)


class BatchNorm2d(_Layer):
    """PyTorch `nn.BatchNorm2d` (affine, track_running_stats): training mode
    normalizes by the batch statistics and updates running_mean_/running_var_
    (momentum, the unbiased variance); eval mode (`.eval()`) uses the running
    statistics. Each channel's statistics are one fixed-order fold.
    `backward` returns grad_x and sets grad_weight_ (gamma), grad_bias_ (beta).
    `BatchNorm1d` is the same over (N, C) or (N, C, L)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 input_shape=None, numeric_mode=None):
        np = _np()
        self.num_features = int(num_features)
        self.eps, self.affine = float(eps), bool(affine)
        self.momentum = None if momentum is None else float(momentum)
        self.track_running_stats = bool(track_running_stats)
        self.num_batches_tracked_ = 0
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode
        self.weight_ = np.ones(self.num_features, np.float32)
        self.bias_ = np.zeros(self.num_features, np.float32)
        self.running_mean_ = np.zeros(self.num_features, np.float32) if self.track_running_stats else None
        self.running_var_ = np.ones(self.num_features, np.float32) if self.track_running_stats else None
        self.training = True

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def _nchw(self, x):
        x = _f32(x, "x")
        shape = x.shape
        if x.ndim == 2:
            x = x[:, :, None]
        n, c = x.shape[:2]
        if c != self.num_features:
            raise ValueError(f"mojolearn: input has {c} channels, the layer {self.num_features}")
        return np_reshape(x, (n, c, -1)), shape

    def _prep(self, n, c, hw):
        """The host words of one forward: (aux, running, batch_stats); the
        batch counter moves here, as it did in `forward`."""
        np = _np()
        C = self.num_features
        # PyTorch: batch statistics in training mode, and in eval mode too when
        # nothing is tracked; momentum=None is the cumulative average 1/batches.
        batch_stats = self.training or not self.track_running_stats
        factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked_ += 1
            factor = 1.0 / self.num_batches_tracked_ if self.momentum is None else self.momentum
        aux = np.zeros(2 + 7 * C, np.float32)
        aux[0], aux[1] = np.float32(self.eps), np.float32(factor)
        aux[2 + 5 * C:2 + 6 * C] = self.weight_
        aux[2 + 6 * C:2 + 7 * C] = self.bias_
        if self.track_running_stats:
            running = np.concatenate([self.running_mean_, self.running_var_]).astype(np.float32)
        else:
            running = np.concatenate([np.zeros(C, np.float32), np.ones(C, np.float32)])
        return aux, running, batch_stats

    def _finish(self, running):
        C = self.num_features
        if self.training and self.track_running_stats:
            self.running_mean_, self.running_var_ = running[:C].copy(), running[C:].copy()

    def _forward_dev(self, b, xh, yh, n, c, hw, dev_y=True):
        """The forward of resident x (`xh`) into resident y (`yh`, or a host
        array's address with `dev_y` False); the backward reads `xh` (lane
        gap-neural-overhead2)."""
        aux, running, batch_stats = self._prep(n, c, hw)
        b.x_cnn_batchnorm_forward_m([xh, yh, running.ctypes.data, aux.ctypes.data], 0b0011 if dev_y else 0b0001,
                                    [n, c, hw, 1 if batch_stats else 0])
        self._finish(running)
        self._aux, self._mode, self._x3shape = aux, batch_stats, (n, c, hw)
        return aux

    def _backward_dev(self, b, xh, gh, dxh, dev_dx=True):
        """The backward from resident g (`gh`) into resident dx (`dxh`, or a
        host array's address with `dev_dx` False); sets the gradients."""
        n, c, hw = self._x3shape
        C = self.num_features
        aux = self._aux.copy()
        b.x_cnn_batchnorm_backward_m([xh, gh, dxh, aux.ctypes.data], 0b0111 if dev_dx else 0b0011,
                                     [n, c, hw, 1 if self._mode else 0])
        self.grad_bias_ = aux[2 + 3 * C:2 + 4 * C].copy()
        self.grad_weight_ = aux[2 + 4 * C:2 + 5 * C].copy()
        if not self.affine:
            self.grad_weight_[:] = 0
            self.grad_bias_[:] = 0

    def forward(self, x):
        np = _np()
        x3, shape = self._nchw(x)
        n, c, hw = x3.shape
        b = self._binding()
        y = np.empty_like(x3)
        if _mixed(b):
            # lane gap-neural-overhead2: x is uploaded once and stays on the
            # device for the backward, which read it from the host again;
            # the backward is the gradient at the forward's input (the
            # tensor PyTorch saves)
            dev = _dev_of(self, b)
            xh = dev.upload("x", x3)
            self._forward_dev(b, xh, y.ctypes.data, n, c, hw, dev_y=False)
            self._xdev, self._x, self._shape, self._xdev_dev = xh, None, shape, dev
            return y.reshape(shape) if len(shape) != 2 else y[:, :, 0]
        aux, running, batch_stats = self._prep(n, c, hw)
        b.x_cnn_batchnorm_forward(x3.ctypes.data, y.ctypes.data, running.ctypes.data, aux.ctypes.data,
                                  [n, c, hw, 1 if batch_stats else 0])
        self._finish(running)
        self._x, self._aux, self._shape, self._mode = x3, aux, shape, batch_stats
        self._x3shape, self._xdev = x3.shape, None
        return y.reshape(shape) if len(shape) != 2 else y[:, :, 0]

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        g3 = np_reshape(g if g.ndim != 2 else g[:, :, None], self._x3shape)
        n, c, hw = self._x3shape
        C = self.num_features
        b = self._binding()
        dx = np.empty(self._x3shape, np.float32)
        xh = self.__dict__.get("_xdev")
        owner = self.__dict__.get("_xdev_dev")
        if xh is not None and owner is not None and owner.b is b and _mixed(b):
            aux = self._aux.copy()
            b.x_cnn_batchnorm_backward_m([xh, g3.ctypes.data, dx.ctypes.data, aux.ctypes.data], 0b0001,
                                         [n, c, hw, 1 if self._mode else 0])
        else:
            aux = self._aux.copy()
            b.x_cnn_batchnorm_backward(self._x.ctypes.data, g3.ctypes.data, dx.ctypes.data,
                                       aux.ctypes.data, [n, c, hw, 1 if self._mode else 0])
        self.grad_bias_ = aux[2 + 3 * C:2 + 4 * C].copy()
        self.grad_weight_ = aux[2 + 4 * C:2 + 5 * C].copy()
        if not self.affine:
            self.grad_weight_[:] = 0
            self.grad_bias_[:] = 0
        return dx.reshape(self._shape) if len(self._shape) != 2 else dx[:, :, 0]

    def _rows(self, X):
        X = _f32(X, "X")
        if X.ndim == 2 and self.input_shape is not None:
            return X.reshape((X.shape[0],) + tuple(self.input_shape))
        return X

    def transform(self, X):
        out = self.forward(self._rows(X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        """One training-mode pass over X (updates the running statistics)."""
        was = self.training
        self.training = True
        self.forward(self._rows(X))
        self.training = was
        return self


class BatchNorm1d(BatchNorm2d):
    """PyTorch `nn.BatchNorm1d` over (N, C) or (N, C, L)."""


class Dropout2d(_Layer):
    """PyTorch `nn.Dropout2d`: in training mode each (n, c) channel is zeroed
    with probability `p` and the rest scaled by 1/(1-p); eval mode is the
    identity. The mask is Philox4x32-10 of (random_state, the call count)
    at the channel index, compared as an integer: the same mask on every
    column (not torch's stream). `mask_` holds the last scale per element."""

    def __init__(self, p=0.5, random_state=0, input_shape=None, numeric_mode=None):
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"mojolearn: dropout probability has to be between 0 and 1, but got {p}")
        self.p = float(p)
        self.random_state = int(random_state or 0)
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode
        self.training = True
        self.calls_ = 0

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the mask
    # stays on the device for the backward (it was downloaded by the forward
    # and uploaded again by the backward, a full-size float array each way);
    # `mask_` downloads it the first time it is read. The same words.

    @property
    def mask_(self):
        d = self.__dict__
        if d.get("_mask_host") is None and d.get("_mask_dev") is not None:
            np = _np()
            m = np.empty(d["_mask_shape"], np.float32)
            self._dev.download(d["_mask_dev"], m)
            d["_mask_host"] = m
        return d.get("_mask_host")

    @mask_.setter
    def mask_(self, v):
        self.__dict__["_mask_host"] = v
        self.__dict__["_mask_dev"] = None

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if not self.training:
            self.mask_ = None
            return x.copy()
        if x.ndim not in (3, 4):
            raise ValueError("mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W)")
        x4 = x[None] if x.ndim == 3 else x
        n, c = x4.shape[:2]
        hw = int(np.prod(x4.shape[2:]))
        thresh = min(int(round(self.p * 2 ** 32)), 2 ** 32)
        seed_lo = self.random_state & 0x7FFFFFFF
        seed_hi = ((self.random_state >> 31) * 1000003 + self.calls_) & 0x7FFFFFFF
        self.calls_ += 1
        y = np.empty_like(x4)
        x4 = np.ascontiguousarray(x4)
        prm = [n, c, hw, seed_lo, seed_hi, thresh >> 16, thresh & 0xFFFF]
        b = self._binding()
        if _mixed(b):
            dev = _dev_of(self, b)
            h = dev.get("mask", x4.size)
            b.x_cnn_dropout2d_m([x4.ctypes.data, y.ctypes.data, h], 0b100, prm, self.p)
            self.__dict__.update(_mask_host=None, _mask_dev=h, _mask_shape=x.shape)
            return y.reshape(x.shape)
        mask = np.empty_like(x4)
        b.x_cnn_dropout2d(x4.ctypes.data, y.ctypes.data, mask.ctypes.data, prm, self.p)
        self.mask_ = mask.reshape(x.shape)
        return y.reshape(x.shape)

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        d = self.__dict__
        if d.get("_mask_host") is None and d.get("_mask_dev") is None:
            return g.copy()
        dx = np.empty_like(g)
        b = self._binding()
        h = d.get("_mask_dev")
        if h is not None and _mixed(b) and self._dev.b is b:
            if g.size != int(np.prod(d["_mask_shape"])):
                raise ValueError("mojolearn: grad_out does not match the forward's input")
            b.x_cnn_map2_m([g.ctypes.data, h, dx.ctypes.data], 0b010, [3, g.size])
            return dx
        mask = np.ascontiguousarray(self.mask_)
        b.x_cnn_mul(g.ctypes.data, mask.ctypes.data, dx.ctypes.data, [g.size])
        return dx

    def transform(self, X):
        out = self.forward(BatchNorm2d._rows(self, X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        return self


class _AdaptivePool(_Layer):
    """PyTorch adaptive pooling: output cell (i, j) reads rows
    [floor(i*H/OH), ceil((i+1)*H/OH)) and columns likewise. An output size
    that divides the input is the plain pooling kernel (kernel = stride =
    input / output; identical windows); any other size is the adaptive
    kernel."""
    _max = False

    def __init__(self, output_size=1, input_shape=None, numeric_mode=None):
        self.output_size = _pair(output_size, "output_size")
        self.input_shape = input_shape
        self.numeric_mode = numeric_mode

    def _divides(self, shape):
        return shape[2] % self.output_size[0] == 0 and shape[3] % self.output_size[1] == 0

    def forward(self, x):
        np = _np()
        x = _f32(x, "x")
        if x.ndim != 4:
            raise ValueError(f"mojolearn: {type(self).__name__}.forward takes (N, C, H, W)")
        self._xshape = x.shape
        if self._divides(x.shape):
            k = (x.shape[2] // self.output_size[0], x.shape[3] // self.output_size[1])
            self._layer = (MaxPool2d if self._max else AvgPool2d)(k, stride=k, numeric_mode=self.numeric_mode)
            out = self._layer.forward(x)
            if self._max:
                self.indices_ = self._layer.indices_
            return out
        self._layer = None
        n, c = x.shape[:2]
        out = np.empty((n, c) + self.output_size, np.float32)
        idx = np.zeros((n, c) + self.output_size, np.int32)
        self._binding().x_cnn_adaptive_pool(x.ctypes.data, out.ctypes.data, idx.ctypes.data,
                                            [*x.shape, *self.output_size, 2 if self._max else 0])
        self.indices_ = idx if self._max else None
        return out

    def backward(self, grad_out):
        np = _np()
        if self._layer is not None:
            return self._layer.backward(grad_out)
        g = _f32(grad_out, "grad_out")
        dx = np.empty(self._xshape, np.float32)
        idx = self.indices_ if self._max else np.zeros(g.shape, np.int32)
        self._binding().x_cnn_adaptive_pool(g.ctypes.data, dx.ctypes.data, idx.ctypes.data,
                                            [*self._xshape, *self.output_size, 3 if self._max else 1])
        return dx

    def transform(self, X):
        out = self.forward(BatchNorm2d._rows(self, X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        return self


class AdaptiveAvgPool2d(_AdaptivePool):
    """PyTorch `nn.AdaptiveAvgPool2d` (global average pooling is
    `AdaptiveAvgPool2d(1)`): each window summed in (h, w) order, one division."""


class AdaptiveMaxPool2d(_AdaptivePool):
    """PyTorch `nn.AdaptiveMaxPool2d` (global max pooling is
    `AdaptiveMaxPool2d(1)`): the first maximum in (h, w) order; `indices_`."""
    _max = True


def _add(binding, a, b):
    np = _np()
    a, b = _f32(a, "a"), _f32(b, "b")
    out = np.empty_like(a)
    binding.x_cnn_add(a.ctypes.data, b.ctypes.data, out.ctypes.data, [a.size])
    return out


class BasicBlock(_Layer):
    """torchvision.models.resnet.BasicBlock: conv3x3(stride) -> BN -> ReLU ->
    conv3x3 -> BN, plus the identity (or conv1x1(stride) -> BN when the
    shape changes), then ReLU. Convolutions carry no bias, as torchvision's.
    `forward` / `backward` over (N, C, H, W); `parameters()` lists
    (layer, weight attr, grad attr) for an optimizer."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, random_state=0, input_shape=None,
                 numeric_mode=None):
        s = int(random_state or 0)
        self.numeric_mode = numeric_mode
        self.input_shape = input_shape
        self.conv1 = Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False, random_state=s,
                            numeric_mode=numeric_mode)
        self.bn1 = BatchNorm2d(planes, numeric_mode=numeric_mode)
        self.relu1 = _ReLU(numeric_mode)
        self.conv2 = Conv2d(planes, planes, 3, padding=1, bias=False, random_state=s + 1, numeric_mode=numeric_mode)
        self.bn2 = BatchNorm2d(planes, numeric_mode=numeric_mode)
        self.relu2 = _ReLU(numeric_mode)
        if downsample is None and (stride != 1 or inplanes != planes):
            downsample = True
        self.downsample = None
        if downsample:
            self.downsample = [Conv2d(inplanes, planes, 1, stride=stride, bias=False, random_state=s + 2,
                                      numeric_mode=numeric_mode), BatchNorm2d(planes, numeric_mode=numeric_mode)]
        self.training = True

    def _bns(self):
        return [self.bn1, self.bn2] + ([self.downsample[1]] if self.downsample else [])

    def train(self, mode=True):
        self.training = bool(mode)
        for bn in self._bns():
            bn.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the block
    # runs on resident device arrays (`_Dev`): x goes up once, y comes down
    # once, and every intermediate (each conv, BN and ReLU output, the sum)
    # stays on the device for the backward, which uploads grad_out once and
    # downloads dx (and the small weight gradients). It used to make about
    # fourteen binding calls each way, each a host round trip of a full
    # activation. The same entries' bodies (the `_m` forms) in the same
    # order on the same words: no bit moves. The sublayers keep their
    # counters, running statistics and gradients; their own host-array
    # state (`_x`) is not filled by the block.

    def _chain_ok(self, b):
        convs = [self.conv1, self.conv2] + ([self.downsample[0]] if self.downsample else [])
        return _mixed(b) and all(c.groups == 1 and not c._explicit for c in convs)

    def _conv_dev(self, b, conv, xh, xshape, name, dev):
        oshape = conv._out_shape(xshape)
        yh = dev.get(name, int(_np().prod(oshape)))
        w = _np().ascontiguousarray(conv.weight_)
        bias = _np().ascontiguousarray(conv.bias_)
        b.x_cnn_conv2d_forward_m([xh, w.ctypes.data, bias.ctypes.data, yh], 0b1001, conv._params(xshape))
        return yh, oshape

    def _bn_dev(self, b, bn, xh, shape, name, dev):
        n, c = shape[:2]
        hw = int(_np().prod(shape[2:]))
        if c != bn.num_features:
            raise ValueError(f"mojolearn: input has {c} channels, the layer {bn.num_features}")
        yh = dev.get(name, n * c * hw)
        bn._forward_dev(b, xh, yh, n, c, hw)
        bn._xdev, bn._x, bn._shape, bn._xdev_dev = xh, None, tuple(shape), dev
        return yh

    def _forward_chain(self, b, x):
        np = _np()
        dev = _dev_of(self, b)
        if x.ndim != 4:
            raise ValueError("mojolearn: Conv2d.forward takes (N, C, H, W)")
        if x.shape[1] != self.conv1.in_channels:
            raise ValueError(f"mojolearn: input has {x.shape[1]} channels, the layer {self.conv1.in_channels}")
        xh = dev.upload("x", x)
        c1, s1 = self._conv_dev(b, self.conv1, xh, x.shape, "c1", dev)
        b1 = self._bn_dev(b, self.bn1, c1, s1, "b1", dev)
        n1 = int(np.prod(s1))
        r1 = dev.get("r1", n1)
        b.x_cnn_map2_m([b1, b1, r1], 0b111, [0, n1])
        c2, s2 = self._conv_dev(b, self.conv2, r1, s1, "c2", dev)
        b2 = self._bn_dev(b, self.bn2, c2, s2, "b2", dev)
        n2 = int(np.prod(s2))
        idh = xh
        if self.downsample:
            d1, sd = self._conv_dev(b, self.downsample[0], xh, x.shape, "d1", dev)
            idh = self._bn_dev(b, self.downsample[1], d1, sd, "d2", dev)
        elif x.size != n2:
            raise ValueError("mojolearn: the identity's shape is not the block's output shape")
        sh = dev.get("s", n2)
        b.x_cnn_map2_m([b2, idh, sh], 0b111, [2, n2])
        y = np.empty(s2, np.float32)
        b.x_cnn_map2_m([sh, sh, y.ctypes.data], 0b011, [0, n2])
        self._chain = dict(x=x.shape, s1=s1, s2=s2, xh=xh, c1=c1, b1=b1, r1=r1, c2=c2, s=sh,
                           d1=self.downsample and d1, sd=self.downsample and sd)
        self.conv1._x = self.conv1._xp = x
        return y

    def _conv_back_dev(self, b, conv, xh, xshape, gh, oshape, name, dev):
        """conv's backward from resident g into resident dx `name`; sets its
        weight gradients (the same words `Conv2d.backward` sets)."""
        np = _np()
        dxh = dev.get(name, int(np.prod(xshape)))
        w = np.ascontiguousarray(conv.weight_)
        dw = np.empty(conv.weight_.shape, np.float32)
        db = np.empty(conv.out_channels, np.float32)
        b.x_cnn_conv2d_backward_m([xh, w.ctypes.data, gh, dxh, dw.ctypes.data, db.ctypes.data], 0b001101,
                                  conv._params(xshape))
        conv.grad_weight_ = dw
        conv.grad_bias_ = db if conv.bias else np.zeros_like(db)
        return dxh

    def _backward_chain(self, b, grad_out):
        np = _np()
        dev, k = self._dev, self._chain
        g = _f32(grad_out, "grad_out")
        if g.shape != tuple(k["s2"]):
            raise ValueError(f"mojolearn: grad_out shape {g.shape}, expected {tuple(k['s2'])}")
        n2 = g.size
        gh = dev.upload("g", g)
        gs = dev.get("gs", n2)
        b.x_cnn_map2_m([k["s"], gh, gs], 0b111, [1, n2])              # relu2
        gb2 = dev.get("gb2", n2)
        self.bn2._backward_dev(b, k["c2"], gs, gb2)
        gr1 = self._conv_back_dev(b, self.conv2, k["r1"], k["s1"], gb2, k["s2"], "gr1", dev)
        n1 = int(np.prod(k["s1"]))
        gb1 = dev.get("gb1", n1)
        b.x_cnn_map2_m([k["b1"], gr1, gb1], 0b111, [1, n1])           # relu1
        gc1 = dev.get("gc1", n1)
        self.bn1._backward_dev(b, k["c1"], gb1, gc1)
        gx1 = self._conv_back_dev(b, self.conv1, k["xh"], k["x"], gc1, k["s1"], "gx1", dev)
        gi = gs
        if self.downsample:
            gd = dev.get("gd", n2)
            self.downsample[1]._backward_dev(b, k["d1"], gs, gd)
            gi = self._conv_back_dev(b, self.downsample[0], k["xh"], k["x"], gd, k["sd"], "gdx", dev)
        dx = np.empty(k["x"], np.float32)
        b.x_cnn_map2_m([gx1, gi, dx.ctypes.data], 0b011, [2, dx.size])
        return dx

    def forward(self, x):
        x = _f32(x, "x")
        b = self._binding()
        if self._chain_ok(b):
            return self._forward_chain(b, x)
        self._chain = None
        out = self.relu1.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))
        identity = x
        if self.downsample:
            identity = self.downsample[1].forward(self.downsample[0].forward(x))
        return self.relu2.forward(_add(self._binding(), out, identity))

    def backward(self, grad_out):
        b = self._binding()
        if self.__dict__.get("_chain") is not None and self._dev.b is b:
            return self._backward_chain(b, grad_out)
        g = self.relu2.backward(grad_out)
        gm = self.conv1.backward(self.bn1.backward(self.relu1.backward(self.conv2.backward(self.bn2.backward(g)))))
        gi = g
        if self.downsample:
            gi = self.downsample[0].backward(self.downsample[1].backward(g))
        return _add(self._binding(), gm, gi)

    def parameters(self):
        out = []
        layers = [self.conv1, self.bn1, self.conv2, self.bn2] + (self.downsample or [])
        for layer in layers:
            out.append((layer, "weight_", "grad_weight_"))
            if isinstance(layer, BatchNorm2d):
                out.append((layer, "bias_", "grad_bias_"))
        return out

    def transform(self, X):
        out = self.forward(BatchNorm2d._rows(self, X))
        return out.reshape(out.shape[0], -1)

    def fit(self, X, y=None):
        return self


def _gemm(binding, a, b, m, n, k, op):
    """C (m x n) = op(A) op(B) on the pinned GEMM (op 0 NN, 1 NT, 2 TN)."""
    np = _np()
    out = np.empty((m, n), np.float32)
    a, b = _f32(a, "a"), _f32(b, "b")   # keep the (possibly copied) operands alive through the call
    binding.x_cnn_gemm(a.ctypes.data, b.ctypes.data, out.ctypes.data, [m, n, k, op])
    return out


class _Graph:
    """Two CSR views of one edge list: rows = targets for the forward
    propagation, rows = sources for its transpose; entries in ascending
    column order within a row, ties in edge order. `vals_t(v)` carries
    per-entry values of the forward view onto the transposed one.

    The views are built by the binding (`x_cnn_csr_build`: two stable radix
    sorts, a gather and a row-pointer kernel on the device; the host twin on
    a CPU-only install), not by a NumPy sort of the edge list on the host
    (cpu-gpu-cleanup n-pyneural, 2026-10-02). Integer work: the same words
    on every backend."""

    def __init__(self, src, dst, n, binding):
        np = _np()
        self.n = int(n)
        self.src, self.dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
        self.nnz = len(self.src)
        s32 = np.ascontiguousarray(self.src, np.int32)
        d32 = np.ascontiguousarray(self.dst, np.int32)
        self.csr_f, self.order_f = self._build(binding, d32, s32)
        self.csr_t, self.order_t = self._build(binding, s32, d32)

    def _build(self, binding, rows, cols):
        np = _np()
        csr = np.empty(self.n + 1 + 2 * self.nnz, np.int32)
        order = np.empty(max(self.nnz, 1), np.int32)
        binding.x_cnn_csr_build(rows.ctypes.data, cols.ctypes.data, csr.ctypes.data, order.ctypes.data,
                                [self.n, self.nnz])
        return csr, order[:self.nnz]

    def vals_t(self, vals_f):
        np = _np()
        by_edge = np.empty(self.nnz, np.float32)
        by_edge[self.order_f] = vals_f
        return np.ascontiguousarray(by_edge[self.order_t])

    def spmm(self, binding, vals, h, mode, transposed=False):
        np = _np()
        h = _f32(h, "h")
        out = np.empty_like(h)
        vals = np.ascontiguousarray(vals if vals is not None and len(vals) else np.zeros(1, np.float32), np.float32)
        csr = self.csr_t if transposed else self.csr_f
        binding.x_cnn_spmm(vals.ctypes.data, h.ctypes.data, out.ctypes.data, csr.ctypes.data,
                           [self.n, h.shape[1], self.nnz, mode])
        return out


def _data_addr(a):
    """The data address of a NumPy array or a tensor (None otherwise)."""
    try:
        return a.__array_interface__["data"][0]
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return int(a.data_ptr())
    except (AttributeError, TypeError, RuntimeError):
        return None


class _GraphKey:
    """lane gap-neural-overhead2 (2026-10-02): an IDENTITY key for a layer's
    graph: the node count and, for each array, the object itself (held, so
    its id cannot be reused), its shape, dtype and data address. A forward
    on the same edge arrays reuses the CSR views, normalized values and
    their resident copies it built. It replaces lane/cnn-apple2's content
    key, a blake2b of `tobytes()` copies of edge_index (and the weights) on
    EVERY forward. Pass new arrays for new edges: an edge array edited in
    place keeps its key (a PyTorch tensor's in-place edit likewise does not
    invalidate PyG's cached=True graph)."""
    __slots__ = ("n", "refs", "sig")

    def __init__(self, n, *arrays):
        self.n = int(n)
        self.refs = arrays
        self.sig = tuple(None if a is None else (tuple(getattr(a, "shape", ())), str(getattr(a, "dtype", "")),
                                                 _data_addr(a)) for a in arrays)

    def __eq__(self, other):
        return (isinstance(other, _GraphKey) and self.n == other.n and len(self.refs) == len(other.refs)
                and all(a is b for a, b in zip(self.refs, other.refs)) and self.sig == other.sig)

    __hash__ = None


def _graph_key(n, *arrays):
    return _GraphKey(n, *arrays)


def _edges(edge_index, n):
    np = _np()
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("mojolearn: edge_index must be (2, E)")
    ei = ei.astype(np.int64)
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise ValueError("mojolearn: edge_index refers to a node that does not exist")
    return ei[0], ei[1]


def _graph_dev(layer, b, dev, g, vals_f, vals_t=None):
    """The resident CSR views of graph `g` and the propagation values (the
    forward's `vals_f`, the transposed view's `vals_t`, by default
    `g.vals_t(vals_f)`), uploaded once per graph and kept in `dev`."""
    np = _np()
    if layer.__dict__.get("_gdev_for") == (id(g), id(dev)) and layer.__dict__.get("_gdev_g") is g:
        return
    prm = [g.n, 1, g.nnz, 0]
    dev.adopt("csr_f", b.x_cnn_csr_upload(g.csr_f.ctypes.data, prm), g.csr_f.size)
    dev.adopt("csr_t", b.x_cnn_csr_upload(g.csr_t.ctypes.data, prm), g.csr_t.size)
    vf = np.ascontiguousarray(vals_f, np.float32)
    vt = g.vals_t(vf) if vals_t is None else np.ascontiguousarray(vals_t, np.float32)
    dev.upload("vals_f", vf if vf.size else np.zeros(1, np.float32))
    dev.upload("vals_t", vt if vt.size else np.zeros(1, np.float32))
    layer._gdev_for, layer._gdev_g = (id(g), id(dev)), g


class GCNConv(_Layer):
    """PyG `torch_geometric.nn.GCNConv` (flow source_to_target, cached=False):
    out = D^-1/2 (A + I) D^-1/2 (X W^T) + b, the degree over the targets'
    incoming edge weights (self loops via add_remaining_self_loops with
    fill 1, or 2 when improved). The propagation is a CSR SpMM in fixed row
    and column order, the transform the pinned GEMM. `forward(x,
    edge_index, edge_weight=None)`; `backward(grad_out)` returns grad_x and
    sets grad_weight_, grad_bias_."""

    def __init__(self, in_channels, out_channels, improved=False, add_self_loops=True, normalize=True, bias=True,
                 random_state=0, numeric_mode=None):
        np = _np()
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.improved, self.add_self_loops, self.normalize = bool(improved), bool(add_self_loops), bool(normalize)
        self.bias = bool(bias)
        self.numeric_mode = numeric_mode
        rng = _rng(random_state)
        bound = np.sqrt(6.0 / (self.in_channels + self.out_channels))  # glorot, PyG's Linear(weight_initializer='glorot')
        self.weight_ = _uniform(rng, bound, (self.out_channels, self.in_channels))
        self.bias_ = np.zeros(self.out_channels, np.float32)

    def _graph(self, n, edge_index, edge_weight):
        np = _np()
        src, dst = _edges(edge_index, n)
        w = (np.ones(len(src), np.float32) if edge_weight is None
             else _f32(edge_weight, "edge_weight").reshape(-1).copy())
        if len(w) != len(src):
            raise ValueError("mojolearn: edge_weight must have one entry per edge")
        if self.normalize and self.add_self_loops:
            fill = np.float32(2.0 if self.improved else 1.0)
            loop = src == dst
            loop_w = np.full(n, fill, np.float32)
            loop_w[src[loop]] = w[loop]            # add_remaining_self_loops keeps an existing loop's weight
            keep = ~loop
            src = np.concatenate([src[keep], np.arange(n)])
            dst = np.concatenate([dst[keep], np.arange(n)])
            w = np.concatenate([w[keep], loop_w]).astype(np.float32)
        g = _Graph(src, dst, n, self._binding())
        wf = np.ascontiguousarray(w[g.order_f])
        if self.normalize:
            vals = np.empty(g.nnz, np.float32)
            self._binding().x_cnn_gcn_norm(wf.ctypes.data, vals.ctypes.data, g.csr_f.ctypes.data,
                                           [n, 1, g.nnz, 0])
        else:
            vals = wf
        return g, vals

    def forward(self, x, edge_index, edge_weight=None):
        np = _np()
        x = _f32(x, "x")
        n = x.shape[0]
        b = self._binding()
        if _LEGACY_STEP:
            g, vals = self._graph(n, edge_index, edge_weight)
        else:
            key = (_graph_key(n, edge_index, edge_weight), self.improved, self.add_self_loops, self.normalize)
            hit = getattr(self, "_gcache", None)
            if hit is not None and hit[0] == key:
                g, vals = hit[1]
            else:
                g, vals = self._graph(n, edge_index, edge_weight)
                self._gcache = (key, (g, vals))
        if _mixed(b) and not _LEGACY_STEP:
            return self._forward_dev(b, x, g, vals)
        self._xdev = None
        h = _gemm(b, x, self.weight_, n, self.out_channels, self.in_channels, 1)
        out = g.spmm(b, vals, h, 0)
        if self.bias:
            out = _add(b, out, np.ascontiguousarray(np.broadcast_to(self.bias_, out.shape)))
        self._x, self._g, self._vals = x, g, vals
        return out

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the layer
    # runs on resident arrays: x goes up once and y comes down once; the
    # transform, the propagation (its CSR views and values resident with
    # the cached graph, checked once) and the bias add never visit the
    # host, nor does the backward's propagated gradient. It made a host
    # round trip of an (n, out) array per call (and uploaded a broadcast
    # bias of that size). The same entries' bodies on the same words; the
    # bias add is `bias_rows_at`, the same IEEE add per element as the
    # broadcast `add_at`. No bit moves.

    def _forward_dev(self, b, x, g, vals):
        np = _np()
        n, F, D = x.shape[0], self.out_channels, self.in_channels
        if x.ndim != 2 or x.shape[1] != D:
            raise ValueError(f"mojolearn: x must be (n, {D})")
        dev = _dev_of(self, b)
        _graph_dev(self, b, dev, g, vals)
        W = _f32(self.weight_, "b")
        xh = dev.upload("x", x)
        hh = dev.get("h", n * F)
        b.x_cnn_gemm_m([xh, W.ctypes.data, hh], 0b101, [n, F, D, 1])
        oh = dev.get("o", n * F)
        b.x_cnn_spmm_m([dev.h["vals_f"][0], hh, dev.h["csr_f"][0], oh], 0b1111, [n, F, g.nnz, 0])
        out = np.empty((n, F), np.float32)
        if self.bias:
            bias = _f32(self.bias_, "b").reshape(-1)
            if bias.size != F:
                raise ValueError("mojolearn: bias_ has the wrong size")
            b.x_cnn_map2_m([oh, bias.ctypes.data, out.ctypes.data], 0b001, [4, n * F, F])
        else:
            dev.download(oh, out)
        self._x, self._g, self._vals, self._xdev = x, g, vals, (dev, xh)
        return out

    def _backward_dev(self, b, G):
        np = _np()
        dev, xh = self._xdev
        g = self._g
        n, F, D = G.shape[0], self.out_channels, self.in_channels
        if G.shape != (self._x.shape[0], F):
            raise ValueError(f"mojolearn: grad_out shape {G.shape}, expected {(self._x.shape[0], F)}")
        Gh = dev.upload("G", G)
        if self.bias:
            gb = np.empty((F, 1), np.float32)
            ones = np.ones(n, np.float32)
            b.x_cnn_gemm_m([Gh, ones.ctypes.data, gb.ctypes.data], 0b001, [F, 1, n, 2])
            self.grad_bias_ = gb.reshape(-1)
        else:
            self.grad_bias_ = np.zeros(F, np.float32)
        dhh = dev.get("dh", n * F)
        b.x_cnn_spmm_m([dev.h["vals_t"][0], Gh, dev.h["csr_t"][0], dhh], 0b1111, [n, F, g.nnz, 0])
        gw = np.empty((F, D), np.float32)
        b.x_cnn_gemm_m([dhh, xh, gw.ctypes.data], 0b011, [F, D, n, 2])
        self.grad_weight_ = gw
        W = _f32(self.weight_, "b")
        dx = np.empty((n, D), np.float32)
        b.x_cnn_gemm_m([dhh, W.ctypes.data, dx.ctypes.data], 0b001, [n, D, F, 0])
        return dx

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
        xd = self.__dict__.get("_xdev")
        if xd is not None and xd[0].b is b:
            return self._backward_dev(b, G)
        self.grad_bias_ = (_gemm(b, G, np.ones(n, np.float32), self.out_channels, 1, n, 2).reshape(-1)
                           if self.bias else np.zeros(self.out_channels, np.float32))
        dh = self._g.spmm(b, self._g.vals_t(self._vals), G, 0, transposed=True)
        self.grad_weight_ = _gemm(b, dh, self._x, self.out_channels, self.in_channels, n, 2)
        return _gemm(b, dh, self.weight_, n, self.in_channels, self.out_channels, 0)


class SAGEConv(_Layer):
    """PyG `torch_geometric.nn.SAGEConv` (aggr 'mean', 'sum' or 'max',
    root_weight, normalize, project): out = lin_l(aggr_{j->i} p(x_j)) +
    lin_r(x_i), lin_l with the bias, then an optional row L2 normalization.
    p is the identity, or with project=True relu(lin(x)) for a biased
    in_channels -> in_channels Linear (PyG applies it to the source
    features only; the root term keeps the unprojected x_i).
    The mean is the fixed-order CSR sum divided by the in-degree; the max
    splits its gradient evenly among tied entries (scatter_reduce 'amax');
    an isolated node aggregates +0.0."""

    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False, root_weight=True, bias=True,
                 project=False, random_state=0, numeric_mode=None):
        np = _np()
        if aggr not in ("mean", "sum", "add", "max"):
            raise NotImplementedError(f"mojolearn: SAGEConv aggr={aggr!r} is not implemented (NOT_IMPLEMENTED.tsv)")
        self.normalize = bool(normalize)
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.aggr, self.root_weight, self.bias = aggr, bool(root_weight), bool(bias)
        self.numeric_mode = numeric_mode
        rng = _rng(random_state)
        self.lin_l = _Linear(self.in_channels, self.out_channels, random_state=rng.child_seed(),
                             numeric_mode=numeric_mode)
        if not self.bias:
            self.lin_l.bias_[:] = 0
        self.weight_r_ = _kaiming_uniform(rng, (self.out_channels, self.in_channels), self.in_channels)
        # drawn last, so project=False keeps every earlier draw
        self.project = bool(project)
        self.lin = None
        if self.project:
            self.lin = _Linear(self.in_channels, self.in_channels, random_state=rng.child_seed(),
                               numeric_mode=numeric_mode)
            self._relu_p = _ReLU(numeric_mode)

    def forward(self, x, edge_index):
        np = _np()
        x = _f32(x, "x")
        n = x.shape[0]
        b = self._binding()
        key = None if _LEGACY_STEP else _graph_key(n, edge_index)
        hit = getattr(self, "_gcache", None)
        if key is not None and hit is not None and hit[0] == key:
            g = hit[1]
        else:
            src, dst = _edges(edge_index, n)
            g = _Graph(src, dst, n, self._binding())
            if key is not None:
                self._gcache = (key, g)
        if (_mixed(b) and not _LEGACY_STEP and not self.project and not self.normalize
                and self.aggr in ("mean", "sum", "add")):
            return self._forward_dev(b, x, g)
        self._xdev = None
        xs = x
        if self.project:
            xs = self._relu_p.forward(self.lin.forward(x))
        self._xs = xs
        if self.aggr == "max":
            agg = np.empty_like(xs)
            self._max_aux = np.zeros(2 * xs.size, np.float32)
            b.x_cnn_graph_op(xs.ctypes.data, xs.ctypes.data, self._max_aux.ctypes.data, agg.ctypes.data,
                             g.csr_f.ctypes.data, [n, xs.shape[1], g.nnz, 0])
        else:
            agg = g.spmm(b, np.ones(g.nnz, np.float32), xs, 1 if self.aggr == "mean" else 0)
        out = self.lin_l.forward(agg)
        if self.root_weight:
            out = _add(b, out, _gemm(b, x, self.weight_r_, n, self.out_channels, self.in_channels, 1))
        self._x, self._g = x, g
        if self.normalize:
            # torch.nn.functional.normalize(out, p=2, dim=-1)
            self._pre = out
            y = np.empty_like(out)
            self._den = np.zeros(n, np.float32)
            self._nograph = np.zeros(n + 1, np.int32)
            b.x_cnn_graph_op(out.ctypes.data, out.ctypes.data, self._den.ctypes.data, y.ctypes.data,
                             self._nograph.ctypes.data, [n, out.shape[1], 0, 2])
            self._y = out = y
        return out

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding (aggr mean
    # or sum, no projection, no normalization) the layer runs on resident
    # arrays: x up once, y down once; the aggregation (the CSR views and
    # values resident with the cached graph), lin_l, the root transform and
    # their sum stay on the device, and the mean backward divides by the
    # in-degree read from the resident CSR offsets (spmm mode 3). The same
    # entries' bodies on the same words: no bit moves.

    def _forward_dev(self, b, x, g):
        np = _np()
        n, D, F = x.shape[0], self.in_channels, self.out_channels
        if x.ndim != 2 or x.shape[1] != D:
            raise ValueError(f"mojolearn: x must be (n, {D})")
        dev = _dev_of(self, b)
        mean = self.aggr == "mean"
        if self.__dict__.get("_gdev_g") is not g or self.__dict__.get("_gdev_for") != (id(g), id(dev)):
            ones = np.ones(g.nnz, np.float32)
            _graph_dev(self, b, dev, g, ones, ones)
        xh = dev.upload("x", x)
        aggh = dev.get("agg", n * D)
        b.x_cnn_spmm_m([dev.h["vals_f"][0], xh, dev.h["csr_f"][0], aggh], 0b1111, [n, D, g.nnz, 1 if mean else 0])
        lin = self.lin_l
        olh = dev.get("ol", n * F)
        b.x_cnn_linear_forward_m([aggh, lin.weight_.ctypes.data, lin.bias_.ctypes.data, olh], 0b1001, [n, D, F])
        out = np.empty((n, F), np.float32)
        if self.root_weight:
            Wr = _f32(self.weight_r_, "b")
            rh = dev.get("r", n * F)
            b.x_cnn_gemm_m([xh, Wr.ctypes.data, rh], 0b101, [n, F, D, 1])
            b.x_cnn_map2_m([olh, rh, out.ctypes.data], 0b011, [2, n * F])
        else:
            dev.download(olh, out)
        self._x, self._g, self._xs, self._xdev = x, g, x, (dev, xh, aggh)
        return out

    def _backward_dev(self, b, G):
        np = _np()
        dev, xh, aggh = self._xdev
        g = self._g
        n, D, F = self._x.shape[0], self.in_channels, self.out_channels
        if G.shape != (n, F):
            raise ValueError(f"mojolearn: grad_out shape {G.shape}, expected {(n, F)}")
        Gh = dev.upload("G", G)
        lin = self.lin_l
        dagg = dev.get("dagg", n * D)
        dw = np.empty_like(lin.weight_)
        db = np.empty_like(lin.bias_)
        b.x_cnn_linear_backward_m([aggh, lin.weight_.ctypes.data, Gh, dagg, dw.ctypes.data, db.ctypes.data],
                                  0b001101, [n, D, F])
        lin.grad_weight_, lin.grad_bias_ = dw, db
        if not self.bias:
            lin.grad_bias_[:] = 0
        dxs = dev.get("dxs", n * D)
        if self.aggr == "mean":
            # each target row of dagg over its in-degree, read from the
            # forward CSR offsets on the device (spmm mode 3), then the
            # transposed fold with unit values: no host degree count
            dsc = dev.get("dsc", n * D)
            b.x_cnn_spmm_m([dev.h["vals_f"][0], dagg, dev.h["csr_f"][0], dsc], 0b1111, [n, D, g.nnz, 3])
            dagg = dsc
        b.x_cnn_spmm_m([dev.h["vals_t"][0], dagg, dev.h["csr_t"][0], dxs], 0b1111, [n, D, g.nnz, 0])
        dx = np.empty((n, D), np.float32)
        if self.root_weight:
            gwr = np.empty((F, D), np.float32)
            b.x_cnn_gemm_m([Gh, xh, gwr.ctypes.data], 0b011, [F, D, n, 2])
            self.grad_weight_r_ = gwr
            Wr = _f32(self.weight_r_, "b")
            grh = dev.get("gr", n * D)
            b.x_cnn_gemm_m([Gh, Wr.ctypes.data, grh], 0b101, [n, D, F, 0])
            b.x_cnn_map2_m([grh, dxs, dx.ctypes.data], 0b011, [2, n * D])
        else:
            dev.download(dxs, dx)
        return dx

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
        xd = self.__dict__.get("_xdev")
        if xd is not None and xd[0].b is b:
            return self._backward_dev(b, G)
        g = self._g
        if self.normalize:
            G2 = np.empty_like(G)
            b.x_cnn_graph_op(self._y.ctypes.data, G.ctypes.data, self._den.ctypes.data, G2.ctypes.data,
                             self._nograph.ctypes.data, [n, G.shape[1], 0, 3])
            G = G2
        dagg = self.lin_l.backward(G)
        if not self.bias:
            self.lin_l.grad_bias_[:] = 0
        if self.aggr == "max":
            dx = np.empty_like(self._xs)
            dagg = np.ascontiguousarray(dagg)
            b.x_cnn_graph_op(self._xs.ctypes.data, dagg.ctypes.data, self._max_aux.ctypes.data, dx.ctypes.data,
                             g.csr_t.ctypes.data, [n, dagg.shape[1], g.nnz, 1])
        elif self.aggr == "mean":
            ones = np.ones(g.nnz, np.float32)
            dx = g.spmm(b, ones, g.spmm(b, ones, dagg, 3), 0, transposed=True)
        else:
            dx = g.spmm(b, np.ones(g.nnz, np.float32), dagg, 0, transposed=True)
        if self.project:
            dx = self.lin.backward(self._relu_p.backward(dx))
        if self.root_weight:
            self.grad_weight_r_ = _gemm(b, G, self._x, self.out_channels, self.in_channels, n, 2)
            dx = _add(b, _gemm(b, G, self.weight_r_, n, self.in_channels, self.out_channels, 0), dx)
        return dx
