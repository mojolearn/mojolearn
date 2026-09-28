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
        rng = np.random.default_rng(random_state)
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

    def backward(self, grad_out, x=None):
        np = _np()
        if x is not None:
            Conv2d.forward(self, x)
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
        rng = np.random.default_rng(random_state)
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
    b1, b2 = float(betas[0]), float(betas[1])
    if step != int(step):
        raise ValueError(f"Adam step must be a whole number, got {step!r}")
    bc1 = 1.0 - _pm.powi(b1, int(step))
    bc2 = 1.0 - _pm.powi(b2, int(step))
    return [lr / bc1, 1.0 - b1, b2, 1.0 - b2, float(eps), _pm.sqrt(bc2), float(weight_decay),
            1.0 if decoupled else 0.0, 1.0 - float(lr) * float(weight_decay)]


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
        self.classes_, yi = np.unique(y, return_inverse=True)
        yi = yi.astype(np.int32)
        self._build(len(self.classes_))
        b = self._binding()
        per = 2 if self.optimizer != "sgd" else 1
        params = self._params()
        rng = np.random.default_rng(self.random_state)
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
                order = rng.permutation(n) if self.shuffle else np.arange(n)
                if epoch_entry:
                    rows = np.ascontiguousarray(order, dtype=np.int32)
                    if self.optimizer == "sgd":
                        hyper = np.array([sgd_row] * nsteps, dtype=np.float64)
                        if step == 0:
                            hyper[0, 5] = 1.0
                    else:
                        hyper = np.array([_adam_hyper(step + 1 + t, self.learning_rate, self.betas, self.eps,
                                                      self.weight_decay, self.optimizer == "adamw")
                                          for t in range(nsteps)], dtype=np.float64)
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
                        R.put(a["x"], np.ascontiguousarray(x[idx]))
                        R.put(a["y"], np.ascontiguousarray(yi[idx]))
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
        np = _np()
        x = self._images(X)
        n = x.shape[0]
        k = len(self.classes_)
        b = self._binding()
        # lane/cnn-apple2: at most _PREDICT_ROWS rows per pass on one set of
        # resident arrays (every op is per row: the same words)
        cap = n if _LEGACY_STEP else max(1, min(n, _PREDICT_ROWS))
        proba = np.empty((n, k), np.float32)
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
                b.x_cnn_res_download(a["proba"], proba[s:s + m].ctypes.data, m * k)
            self._rw = {}
        return proba

    def predict(self, X):
        np = _np()
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

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

    def forward(self, x):
        np = _np()
        x3, shape = self._nchw(x)
        n, c, hw = x3.shape
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
        y = np.empty_like(x3)
        self._binding().x_cnn_batchnorm_forward(x3.ctypes.data, y.ctypes.data, running.ctypes.data, aux.ctypes.data,
                                                [n, c, hw, 1 if batch_stats else 0])
        if self.training and self.track_running_stats:
            self.running_mean_, self.running_var_ = running[:C].copy(), running[C:].copy()
        self._x, self._aux, self._shape, self._mode = x3, aux, shape, batch_stats
        return y.reshape(shape) if len(shape) != 2 else y[:, :, 0]

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        g3 = np_reshape(g if g.ndim != 2 else g[:, :, None], self._x.shape)
        n, c, hw = self._x.shape
        C = self.num_features
        aux = self._aux.copy()
        dx = np.empty_like(self._x)
        self._binding().x_cnn_batchnorm_backward(self._x.ctypes.data, g3.ctypes.data, dx.ctypes.data,
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
        mask = np.empty_like(x4)
        x4 = np.ascontiguousarray(x4)
        self._binding().x_cnn_dropout2d(x4.ctypes.data, y.ctypes.data, mask.ctypes.data,
                                        [n, c, hw, seed_lo, seed_hi, thresh >> 16, thresh & 0xFFFF], self.p)
        self.mask_ = mask.reshape(x.shape)
        return y.reshape(x.shape)

    def backward(self, grad_out):
        np = _np()
        g = _f32(grad_out, "grad_out")
        if self.mask_ is None:
            return g.copy()
        dx = np.empty_like(g)
        mask = np.ascontiguousarray(self.mask_)
        self._binding().x_cnn_mul(g.ctypes.data, mask.ctypes.data, dx.ctypes.data, [g.size])
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

    def forward(self, x):
        x = _f32(x, "x")
        out = self.relu1.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))
        identity = x
        if self.downsample:
            identity = self.downsample[1].forward(self.downsample[0].forward(x))
        return self.relu2.forward(_add(self._binding(), out, identity))

    def backward(self, grad_out):
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
    """Two CSR views of one edge list (integer work only, in NumPy): rows =
    targets for the forward propagation, rows = sources for its transpose;
    entries in ascending column order within a row. `vals_t(v)` carries
    per-entry values of the forward view onto the transposed one."""

    def __init__(self, src, dst, n):
        np = _np()
        self.n = int(n)
        self.src, self.dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
        # lane/cnn-apple2: a stable argsort of the one int64 key (row * n +
        # col, both in [0, n)) is lexsort's order exactly, at about a third
        # of its time on 1M edges (host work that dominated the forward)
        nn = max(self.n, 1)
        if _LEGACY_STEP:
            self.order_f = np.lexsort((self.src, self.dst))
            self.order_t = np.lexsort((self.dst, self.src))
        else:
            self.order_f = np.argsort(self.dst * nn + self.src, kind="stable")
            self.order_t = np.argsort(self.src * nn + self.dst, kind="stable")
        self.csr_f = self._csr(self.dst[self.order_f], self.src[self.order_f])
        self.csr_t = self._csr(self.src[self.order_t], self.dst[self.order_t])
        self.nnz = len(self.src)

    def _csr(self, rows, cols):
        np = _np()
        ptr = np.zeros(self.n + 1, np.int64)
        ptr[1:] = np.bincount(rows, minlength=self.n)[:self.n]
        return np.ascontiguousarray(np.concatenate([np.cumsum(ptr), cols, rows]).astype(np.int32))

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


def _graph_key(n, *arrays):
    """lane/cnn-apple2: a content key for a layer's graph (the node count
    and the bytes of each array): a forward on the same edges reuses the
    CSR views and normalized values it built, which are functions of these
    alone (the same words; PyG's cached=False recomputes them on the GPU,
    here the build is host NumPy and dominated the forward)."""
    import hashlib
    np = _np()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(int(n)).encode())
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.digest()


def _edges(edge_index, n):
    np = _np()
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("mojolearn: edge_index must be (2, E)")
    ei = ei.astype(np.int64)
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise ValueError("mojolearn: edge_index refers to a node that does not exist")
    return ei[0], ei[1]


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
        rng = np.random.default_rng(random_state)
        bound = np.sqrt(6.0 / (self.in_channels + self.out_channels))  # glorot, PyG's Linear(weight_initializer='glorot')
        self.weight_ = rng.uniform(-bound, bound, (self.out_channels, self.in_channels)).astype(np.float32)
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
        g = _Graph(src, dst, n)
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
        h = _gemm(b, x, self.weight_, n, self.out_channels, self.in_channels, 1)
        out = g.spmm(b, vals, h, 0)
        if self.bias:
            out = _add(b, out, np.ascontiguousarray(np.broadcast_to(self.bias_, out.shape)))
        self._x, self._g, self._vals = x, g, vals
        return out

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
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
        rng = np.random.default_rng(random_state)
        self.lin_l = _Linear(self.in_channels, self.out_channels, random_state=rng.integers(2 ** 31),
                             numeric_mode=numeric_mode)
        if not self.bias:
            self.lin_l.bias_[:] = 0
        self.weight_r_ = _kaiming_uniform(rng, (self.out_channels, self.in_channels), self.in_channels)
        # drawn last, so project=False keeps every earlier draw
        self.project = bool(project)
        self.lin = None
        if self.project:
            self.lin = _Linear(self.in_channels, self.in_channels, random_state=rng.integers(2 ** 31),
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
            g = _Graph(src, dst, n)
            if key is not None:
                self._gcache = (key, g)
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

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
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
            deg = np.bincount(g.dst, minlength=n).astype(np.float32)
            dx = g.spmm(b, np.ascontiguousarray(deg[g.dst[g.order_t]]), dagg, 2, transposed=True)
        else:
            dx = g.spmm(b, np.ones(g.nnz, np.float32), dagg, 0, transposed=True)
        if self.project:
            dx = self.lin.backward(self._relu_p.backward(dx))
        if self.root_weight:
            self.grad_weight_r_ = _gemm(b, G, self._x, self.out_channels, self.in_channels, n, 2)
            dx = _add(b, _gemm(b, G, self.weight_r_, n, self.in_channels, self.out_channels, 0), dx)
        return dx
