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
from . import _native_xcnn_bridge

# Package-import connection: the migrated layer methods are ONE call each
# into the compiled owner (bindings/xcnn_native_pack.mojo).
_XCNN, _XCNN_CONTEXT = _native_xcnn_bridge.connect()

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


class _Layer:
    """A layer on `_mojolearn_x_cnn` (the GPU binding, or its CPU host twin
    on a CPU-only install / MOJOLEARN_VENDOR=cpu)."""
    numeric_mode = None

    def _binding(self):
        mode = self.numeric_mode or _backend.default_mode()
        binding = _backend.binding(_BINDING, mode)
        if mode == "identical":
            # API-shell metadata only. The Mojo binding names the arithmetic
            # version shared by host and all GPUs; preserve it on the model
            # so normal state persistence cannot silently switch versions.
            report = getattr(binding, "x_cnn_numerical_profile", None)
            previous = self.__dict__.get("numerical_profile_")
            current = str(report()) if report is not None else None
            if previous is not None and previous != current:
                raise ValueError("mojolearn: CNN numerical profile differs from this model's recorded "
                                 "profile; use a matching binding or a fresh model for the new version")
            if current is not None:
                self.numerical_profile_ = current
        self.numeric_mode_ = mode
        return binding


# lane fam-neural (2026-10-04): the binding's `x_cnn_idn_flags` names the
# lane's IDENTICAL switches this build has on (x_cnn/device.mojo "lane
# fam-neural: `_m` forms"; each has a `-D ..._OFF` before arm and all are off
# under MOJOLEARN_IDN_ALL_OFF, in FAST, and on the CPU twin, which has no
# such entry). The glue takes a new path only when its bit is set; otherwise
# every layer runs exactly the calls it ran before.
# TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
# Tried: _F_BN_FUSE = 64, the fused BatchNorm flag bit.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.


# lane fam2-neural (2026-10-04): `x_cnn_idn2_flags` (BOTH bindings export it:
# these switches move bits, so the host column moves with the devices).
# Bit 0: the softmax mean loss is the blocked device fold (inside the
# binding; nothing for the glue to choose). Bit 1: reserved (0; the
# classifier's device epoch was deleted, lane x8-cnn-on-executor).
# Bit 2 (lane fix-n1-lm-neural): GCNConv's remaining self loops come from
# `x_cnn_gcn_loops` (a stable device sort and two gathers; the host twin on a
# CPU-only install), not host NumPy. Copies only: no bit moves.
# Bit 3: the candidate fold block 256. Bit 4 (lane/review-fixes): the binding
# was built with -D MOJOLEARN_IDN_ALL_OFF.
# Both flag words are read natively (python/mojolearn/_native_xcnn_bridge.py connects them;
# bindings/xcnn_common_door.mojo decides each route).
# TOMBSTONE: the Python layer glue `_mixed`, `_idn`, `_idn2`, `_dev_of`, `_device_io_off`, `_size`,
# `_pin_weights`, `_pinned`, `_resident`, `_out_new`, `_auto_pin`, `_is_t`, `_on_dev`, `_host`,
# `Conv2d._res_w/_backward_res/_padded_shape/_out_shape/_group/_res_ok`, `CNNClassifier._build`,
# `BatchNorm2d._res_bn/_backward_res/_backward_dev` and `BasicBlock._conv_back_dev/_backward_chain`
# (dead copies of bindings/xcnn_conv_native_door.mojo, xcnn_bn_native.mojo and xcnn_classifier_native.mojo;
# lane pyout-metrics-cnn) deleted 2026-11-11; code recoverable at 6f82d0255d.


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
        for k, (a, _) in h.items():  # glue: frees the device handle table
            try:
                self.b.x_cnn_res_free(a)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass

    def __del__(self):
        self.free()


# lane idn-cnn-resident (2026-10-04): DEVICE-RESIDENT TENSORS IN AND OUT.
# A layer handed host arrays uploads its input and downloads its output on
# every call, so a full activation crosses the bus both ways per forward and
# again per backward. `to_device(x)` uploads once and returns a
# `DeviceTensor`; Conv2d/Conv1d, the Max/Avg pools, BasicBlock, GCNConv and
# SAGEConv given one run the same `_m` entries with the resident bit set for
# the input and the output and return a `DeviceTensor` (so does `backward`
# for a device grad_out); `.numpy()` downloads. The same kernels, launches
# and operands on the same words: only where the bytes live changes, so no
# bit moves on any column. A CPU-only install (no `_m` entries) keeps the
# words in a host array behind the same object, and a layer or option with
# no resident form (groups, an explicit pad, any other layer through
# `__array__`) takes the host entry on the downloaded words.
# MOJOLEARN_XCNN_DEVICE_IO_OFF=1 is the A/B before arm (the same device
# kernels either way): `to_device` returns the array it was given and the
# graph layers upload their ones vector per call again. The wave's OFF arm
# (`-D MOJOLEARN_IDN_ALL_OFF=1` in the build and `MOJOLEARN_IDN_ALL_OFF=1` in
# the environment, tools/identical_wave_runner.py) turns it off too, so that
# arm runs the old form throughout (audit F4). lane/review-fixes: a binding
# built with the define reports it (`x_cnn_idn2_flags` bit 4), so the OFF arm
# holds even when the environment variable is not set (`_device_io_off`).
_DEVICE_IO_OFF = (__import__("os").environ.get("MOJOLEARN_XCNN_DEVICE_IO_OFF", "") == "1"
                  or __import__("os").environ.get("MOJOLEARN_IDN_ALL_OFF", "") == "1")


# lane fam2-neural (2026-10-04): PINNED LAYER WEIGHTS. A layer given a
# `DeviceTensor` still uploaded its `weight_` (and `bias_`) on every forward
# and every backward. `layer.pin_weights()` uploads them once into the
# layer's resident arrays; the `_m` entries then read them there (their
# residency bit set) until `unpin_weights()`. The host arrays stay the
# layer's weights: after changing them (in place, or `set_weights`) call
# `pin_weights()` again. The same entry on the same words: no bit moves.
# p3-cnn-sequence-tsa (2026-11-06): a pinned layer's `weight_` / `bias_` are
# read-only arrays (bindings/xcnn_conv_native_door.mojo `_pin_values`), so an
# in-place edit raises instead of running on the stale device copy.
# `x_cnn_idn_flags` bit 5 (`-D MOJOLEARN_IDN_PIN_WEIGHTS_OFF`); with it off,
# on FAST or on the CPU twin `pin_weights()` does nothing.


# lane cnn-resident-glue (2026-10-11, G3): CNN resident grads,
# `x_cnn_idn_flags` bit 7, default ON since 2026-10-11 (lane/w2-decide-neural,
# avg 0.754 over 9 layer lanes, same bits); `-D MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF`
# clears the bit (x_cnn/device.mojo
# IDN_CNN_RESIDENT_GRADS has the structural reason and the expected gain).
# With it on, a resident call keeps everything on the device: Conv2d (and
# the BasicBlock's convs) pin their weights on first use (`_auto_pin`; a
# reassigned `weight_` / `bias_` is uploaded again), dW / db and BatchNorm's
# running statistics and aux block stay in resident arrays and download on
# the first host read of `grad_weight_`, `grad_bias_`, `running_mean_`,
# `running_var_`, the adaptive pools take resident tensors, and outputs whose
# every word the entry stores skip the +0.0 fill (`_out_new`). Buffer
# ownership only: the same entries and operands, so no bit moves. In-place
# edits of a weight array (not a reassignment) need `pin_weights()` again,
# as they always did for a pinned layer.


class DeviceTensor:
    """A float32 tensor resident on the device (an `x_cnn_res_alloc` array
    and its shape). Make one with `to_device(array)`; read it back with
    `.numpy()`. Freed with the object."""
    __slots__ = ("b", "shape", "h", "_host", "_base", "__weakref__")
    _device_tensor = True

    def __init__(self, binding, shape, h=None, host=None, base=None):
        return _XCNN(self, "tensor_init", (binding, shape, h, host, base), _XCNN_CONTEXT)

    @classmethod
    def _new(cls, b, shape):
        """An unwritten resident tensor of `shape` on mixed binding `b`."""
        return _XCNN(cls, "tensor_new", (b, shape), _XCNN_CONTEXT)

    @classmethod
    def _new_out(cls, b, shape):
        """`_new` for an output whose every word the entry stores (lane
        neural-io-2, XCNN_RES_NEW_NOZERO): no +0.0 fill on the device, on a
        binding that has `x_cnn_res_alloc_nofill`."""
        return _XCNN(cls, "tensor_new_out", (b, shape), _XCNN_CONTEXT)

    @classmethod
    def _wrap(cls, b, a):
        """Host float32 array `a` as a tensor on binding `b` (one upload)."""
        return _XCNN(cls, "tensor_wrap", (b, a), _XCNN_CONTEXT)

    @property
    def size(self):
        return _XCNN(self, "tensor_size", (), _XCNN_CONTEXT)

    @property
    def ndim(self):
        return _XCNN(self, "tensor_ndim", (), _XCNN_CONTEXT)

    dtype = "float32"

    def numpy(self):
        return _XCNN(self, "tensor_numpy", (), _XCNN_CONTEXT)

    def __array__(self, dtype=None, copy=None):
        return _XCNN(self, "tensor_array", (dtype, copy), _XCNN_CONTEXT)

    def reshape(self, *shape):
        """The same resident words under another shape (no copy; the view
        keeps this tensor alive)."""
        return _XCNN(self, "tensor_reshape", (shape,), _XCNN_CONTEXT)

    def __del__(self):
        return _XCNN(self, "tensor_finalize", (), _XCNN_CONTEXT)


def to_device(x, numeric_mode=None):
    """float32 array `x` as a `DeviceTensor` (uploaded once, here). The
    layers that take one keep their output on the device too."""
    return _XCNN(None, "tensor_to", (x, numeric_mode), _XCNN_CONTEXT)


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
        _XCNN(self, "conv_init", (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, random_state, input_shape, numeric_mode), _XCNN_CONTEXT)

    def set_weights(self, weight, bias=None):
        return _XCNN(self, "conv_set_weights", (weight, bias), _XCNN_CONTEXT)

    def pin_weights(self):
        """Keep `weight_` and `bias_` on the device for `DeviceTensor`
        forwards and backwards. While they are kept there, `weight_` and
        `bias_` are read-only arrays: assign new arrays (or call
        `set_weights`) to change them, and the new values are uploaded on the
        next call. `unpin_weights()` makes them writeable again. Returns
        self."""
        return _XCNN(self, "conv_pin", (), _XCNN_CONTEXT)

    def unpin_weights(self):
        return _XCNN(self, "conv_unpin", (), _XCNN_CONTEXT)

    # lane cnn-resident-glue (G3): the weight gradients of a resident
    # backward stay in this layer's resident arrays (`_gw_dev`, `_gb_dev`)
    # until read; the host value lives under the same name in `__dict__`.
    @property
    def grad_weight_(self):
        return _XCNN(self, "conv_grad_weight", (), _XCNN_CONTEXT)

    @grad_weight_.setter
    def grad_weight_(self, v):
        return _XCNN(self, "conv_grad_weight_set", (v,), _XCNN_CONTEXT)

    @property
    def grad_bias_(self):
        return _XCNN(self, "conv_grad_bias", (), _XCNN_CONTEXT)

    @grad_bias_.setter
    def grad_bias_(self, v):
        return _XCNN(self, "conv_grad_bias_set", (v,), _XCNN_CONTEXT)


    def _params(self, shape):
        """The binding block for ONE group's conv over the (padded) input `shape`."""
        n, c, h, w = shape
        kh, kw = self.kernel_size
        return [n, c // self.groups, h, w, self.out_channels // self.groups, kh, kw, *self.stride, *self._conv_pad,
                *self.dilation, 0, 0, 1 if self.bias else 0, 0]


    _device_io = True


    def forward(self, x):
        return _XCNN(self, "conv_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out, x=None):
        return _XCNN(self, "conv_backward", (grad_out, x), _XCNN_CONTEXT)

    def _rows(self, X):
        X = _f32(X, "X")
        if X.ndim == 2:
            if self.input_shape is None:
                raise ValueError("mojolearn: row input needs input_shape=(C, H, W)")
            return X.reshape((X.shape[0],) + tuple(self.input_shape))
        return X

    def transform(self, X):
        return _XCNN(self, "conv_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        """Layers carry their weights from construction; fit only records the input shape."""
        return _XCNN(self, "conv_fit", (X,), _XCNN_CONTEXT)


class Conv1d(Conv2d):
    """1-D convolution, PyTorch `nn.Conv1d` semantics: Conv2d over (N, C, 1, L)
    (padding int, 'same' or 'valid'; every padding_mode; groups)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", random_state=0, input_shape=None, numeric_mode=None):
        _XCNN(self, "conv_init1d", (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, random_state, input_shape, numeric_mode), _XCNN_CONTEXT)

    def set_weights(self, weight, bias=None):
        return _XCNN(self, "conv_set_weights1d", (weight, bias), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "conv_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out, x=None):
        return _XCNN(self, "conv_backward", (grad_out, x), _XCNN_CONTEXT)

    @property
    def weight1d_(self):
        return _XCNN(self, "conv_weight1d", (), _XCNN_CONTEXT)


class _Pool2d(_Layer):
    """Shared pooling plumbing: (N, C, H, W) float32, floor mode."""
    _kind = None
    _device_io = True

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                 count_include_pad=True, input_shape=None, numeric_mode=None):
        _XCNN(self, "pool_init", (kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, None, input_shape, numeric_mode, False), _XCNN_CONTEXT)

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
        return _XCNN(self, "pool_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        return _XCNN(self, "pool_fit", (X, y), _XCNN_CONTEXT)


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
        return _XCNN(self, "pool_indices_get", (), _XCNN_CONTEXT)

    @indices_.setter
    def indices_(self, v):
        _XCNN(self, "pool_indices_set", (v,), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "pool_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "pool_backward", (grad_out,), _XCNN_CONTEXT)


class AvgPool2d(_Pool2d):
    """PyTorch `nn.AvgPool2d` (floor or ceil mode, count_include_pad,
    divisor_override)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None, input_shape=None, numeric_mode=None):
        _XCNN(self, "pool_init", (kernel_size, stride, padding, 1, ceil_mode, count_include_pad, divisor_override, input_shape, numeric_mode, False), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "pool_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "pool_backward", (grad_out,), _XCNN_CONTEXT)


def _one(v):
    return int(v if isinstance(v, int) else v[0])


class _Pool1dMixin:
    def forward(self, x):
        return _XCNN(self, "pool_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "pool_backward", (grad_out,), _XCNN_CONTEXT)


class MaxPool1d(_Pool1dMixin, MaxPool2d):
    """PyTorch `nn.MaxPool1d`: MaxPool2d over (N, C, 1, L)."""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
                 input_shape=None, numeric_mode=None):
        _XCNN(self, "pool_init", (kernel_size, stride, padding, dilation, ceil_mode, True, None, input_shape, numeric_mode, True), _XCNN_CONTEXT)


class AvgPool1d(_Pool1dMixin, AvgPool2d):
    """PyTorch `nn.AvgPool1d`: AvgPool2d over (N, C, 1, L)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 input_shape=None, numeric_mode=None):
        _XCNN(self, "pool_init", (kernel_size, stride, padding, 1, ceil_mode, count_include_pad, None, input_shape, numeric_mode, True), _XCNN_CONTEXT)


class _ReLU(_Layer):
    def __init__(self, numeric_mode=None):
        _XCNN(self, "relu_init", (numeric_mode,), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "relu_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "relu_backward", (grad_out,), _XCNN_CONTEXT)


class _Linear(_Layer):
    """y = x W^T + b on the pinned GEMM (PyTorch nn.Linear, default init)."""

    def __init__(self, in_features, out_features, random_state=0, numeric_mode=None):
        _XCNN(self, "linear_init", (in_features, out_features, random_state, numeric_mode), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "linear_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "linear_backward", (grad_out,), _XCNN_CONTEXT)


# integ-w3: _adam_hyper / _adam_hyper_block deleted (no caller; they called the binding's
# x_cnn_adam_hyper_d, which lane x8-cnn-on-executor deleted with the hand epoch).


class _Res:
    """Resident arrays (DEVIATION 5718): handles the binding keeps between
    its `_r` entries (device addresses on the GPU binding, host allocations on
    the CPU twin), 4-byte words, zero filled; freed on exit."""

    def __init__(self, binding):
        self.b, self.handles = binding, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:  # glue: releases the resident buffer handles
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


# TOMBSTONE: _PREDICT_ROWS, _LEGACY_STEP and _EPOCH_ENTRY (MOJOLEARN_XCNN_PY_STEPS), the measurement arms of the deleted hand CNN epoch, deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch


class CNNClassifier(_Layer):
    """A small CNN image classifier, sklearn-shaped: for each entry of
    `conv_channels` a Conv2d (kernel_size, 'same' zero padding for odd
    kernels) -> ReLU -> MaxPool2d(pool_size) block (the pool is skipped once
    the map is smaller than the window), then one Linear layer to the
    classes; softmax cross entropy (mean); `optimizer` 'sgd' (torch.optim.SGD:
    momentum, dampening, nesterov, weight_decay), 'adam' or 'adamw'
    (torch.optim.Adam / AdamW: betas, eps, weight_decay). The whole fit runs
    as one compiled program on the GPU (or the CPU). X is (n, C*H*W) rows
    with `input_shape=(C, H, W)`, or (n, C, H, W). After `fit`, `losses_`
    holds every minibatch's loss and `loss_curve_` each epoch's mean loss."""

    def __init__(self, input_shape, conv_channels=(8,), kernel_size=3, pool_size=2, learning_rate=0.01,
                 momentum=0.9, weight_decay=0.0, batch_size=32, max_iter=10, shuffle=True, random_state=0,
                 numeric_mode=None, optimizer="sgd", dampening=0.0, nesterov=False, betas=(0.9, 0.999), eps=1e-8):
        return _XCNN(self, "cnnclf_init", (input_shape, conv_channels, kernel_size, pool_size, learning_rate, momentum, weight_decay, batch_size, max_iter, shuffle, random_state, numeric_mode, optimizer, dampening, nesterov, betas, eps), _XCNN_CONTEXT)


    def fit(self, X, y):
        """Train the classifier on images X and labels y; returns self."""
        # the labels, the layers and every epoch in one native call
        # (bindings/xcnn_classifier_native_door.mojo -> the x_sequence program door)
        return _XCNN(self, "cnnclf_fit", (X, y), _XCNN_CONTEXT)

    def predict_proba(self, X):
        return _XCNN(self, "cnnclf_predict_proba", (X,), _XCNN_CONTEXT)

    def predict(self, X):
        return _XCNN(self, "cnnclf_predict", (X,), _XCNN_CONTEXT)

    def weights(self):
        """Every trained array, in layer order: [w0, b0, w1, b1, ...]."""
        return _XCNN(self, "cnnclf_weights", (), _XCNN_CONTEXT)


class BatchNorm2d(_Layer):
    """PyTorch `nn.BatchNorm2d` (affine, track_running_stats): training mode
    normalizes by the batch statistics and updates running_mean_/running_var_
    (momentum, the unbiased variance); eval mode (`.eval()`) uses the running
    statistics. Each channel's statistics are one fixed-order fold.
    `backward` returns grad_x and sets grad_weight_ (gamma), grad_bias_ (beta).
    `BatchNorm1d` is the same over (N, C) or (N, C, L)."""

    # lane gap-neural-io (2026-10-08, plan gaps-2026-10-08 Section 4): the
    # forward and backward already take resident tensors (`_forward_dev`,
    # `_backward_dev`, behind IDN_LAYER_DEV_IO); the flag tells callers (the
    # board harness among them) to hand x and dy over as `DeviceTensor`s, so
    # a fit no longer moves x, y, dy and dx over the bus. The same entries
    # on the same words: no bit moves.
    _device_io = True

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 input_shape=None, numeric_mode=None):
        return _XCNN(self, "bn_init", (num_features, eps, momentum, affine, track_running_stats, input_shape, numeric_mode), _XCNN_CONTEXT)

    def train(self, mode=True):
        return _XCNN(self, "bn_train", (mode,), _XCNN_CONTEXT)

    def eval(self):
        return _XCNN(self, "bn_train", (False,), _XCNN_CONTEXT)

    @property
    def running_mean_(self):
        return _XCNN(self, "bn_running_get", ("running_mean_",), _XCNN_CONTEXT)

    @running_mean_.setter
    def running_mean_(self, v):
        return _XCNN(self, "bn_running_set", ("running_mean_", v), _XCNN_CONTEXT)

    @property
    def running_var_(self):
        return _XCNN(self, "bn_running_get", ("running_var_",), _XCNN_CONTEXT)

    @running_var_.setter
    def running_var_(self, v):
        return _XCNN(self, "bn_running_set", ("running_var_", v), _XCNN_CONTEXT)

    @property
    def grad_weight_(self):
        return _XCNN(self, "bn_grad_get", ("grad_weight_",), _XCNN_CONTEXT)

    @grad_weight_.setter
    def grad_weight_(self, v):
        return _XCNN(self, "bn_grad_set", ("grad_weight_", v), _XCNN_CONTEXT)

    @property
    def grad_bias_(self):
        return _XCNN(self, "bn_grad_get", ("grad_bias_",), _XCNN_CONTEXT)

    @grad_bias_.setter
    def grad_bias_(self, v):
        return _XCNN(self, "bn_grad_set", ("grad_bias_", v), _XCNN_CONTEXT)


    # TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
    # Tried: BatchNorm2d._forward_dev_fused / _backward_dev_relu (the fused entries' glue).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.

    def forward(self, x):
        return _XCNN(self, "bn_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "bn_backward", (grad_out,), _XCNN_CONTEXT)

    def _rows(self, X):
        X = _f32(X, "X")
        if X.ndim == 2 and self.input_shape is not None:
            return X.reshape((X.shape[0],) + tuple(self.input_shape))
        return X

    def transform(self, X):
        return _XCNN(self, "bn_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        """One training-mode pass over X (updates the running statistics)."""
        return _XCNN(self, "bn_fit", (X,), _XCNN_CONTEXT)


class BatchNorm1d(BatchNorm2d):
    """PyTorch `nn.BatchNorm1d` over (N, C) or (N, C, L)."""


class Dropout2d(_Layer):
    """PyTorch `nn.Dropout2d`: in training mode each (n, c) channel is zeroed
    with probability `p` and the rest scaled by 1/(1-p); eval mode is the
    identity. The mask is Philox4x32-10 of (random_state, the call count)
    at the channel index, compared as an integer: the same mask on every
    column (not torch's stream). `mask_` holds the last scale per element."""

    # lane gap-neural-io (2026-10-08): resident x, y and dy already run
    # (`_forward_t`, `x_cnn_map2_m`); the flag lets callers pass them, as
    # BatchNorm2d above. No bit moves.
    _device_io = True

    def __init__(self, p=0.5, random_state=0, input_shape=None, numeric_mode=None):
        _XCNN(self, "dropout_init", (p, random_state, input_shape, numeric_mode), _XCNN_CONTEXT)

    def train(self, mode=True):
        return _XCNN(self, "dropout_train", (mode,), _XCNN_CONTEXT)

    def eval(self):
        return _XCNN(self, "dropout_train", (False,), _XCNN_CONTEXT)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the mask
    # stays on the device for the backward (it was downloaded by the forward
    # and uploaded again by the backward, a full-size float array each way);
    # `mask_` downloads it the first time it is read. The same words.

    @property
    def mask_(self):
        return _XCNN(self, "dropout_mask_get", (), _XCNN_CONTEXT)

    @mask_.setter
    def mask_(self, v):
        _XCNN(self, "dropout_mask_set", (v,), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "dropout_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "dropout_backward", (grad_out,), _XCNN_CONTEXT)

    def transform(self, X):
        return _XCNN(self, "dropout_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        return _XCNN(self, "pool_fit", (X, y), _XCNN_CONTEXT)


class _AdaptivePool(_Layer):
    """PyTorch adaptive pooling: output cell (i, j) reads rows
    [floor(i*H/OH), ceil((i+1)*H/OH)) and columns likewise. An output size
    that divides the input is the plain pooling kernel (kernel = stride =
    input / output; identical windows); any other size is the adaptive
    kernel."""
    _max = False

    def __init__(self, output_size=1, input_shape=None, numeric_mode=None):
        _XCNN(self, "adaptive_init", (output_size, input_shape, numeric_mode), _XCNN_CONTEXT)

    # lane cnn-resident-glue (G3): with CNN resident grads on (the default),
    # callers (the board harness among them) hand x and dy over as
    # `DeviceTensor`s, as they do for the plain pools, and a dividing size
    # reuses one pool layer (and its resident winner array) per kernel.
    @property
    def _device_io(self):
        return _XCNN(self, "adaptive_device_io", (), _XCNN_CONTEXT)

    @property
    def indices_(self):
        return _XCNN(self, "adaptive_indices_get", (), _XCNN_CONTEXT)

    @indices_.setter
    def indices_(self, v):
        _XCNN(self, "pool_indices_set", (v,), _XCNN_CONTEXT)

    def forward(self, x):
        return _XCNN(self, "adaptive_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "adaptive_backward", (grad_out,), _XCNN_CONTEXT)

    def transform(self, X):
        return _XCNN(self, "adaptive_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        return _XCNN(self, "pool_fit", (X, y), _XCNN_CONTEXT)


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
    _device_io = True

    def __init__(self, inplanes, planes, stride=1, downsample=None, random_state=0, input_shape=None,
                 numeric_mode=None):
        _XCNN(self, "block_init", (inplanes, planes, stride, downsample, random_state, input_shape, numeric_mode), _XCNN_CONTEXT)

    def _bns(self):
        return [self.bn1, self.bn2] + ([self.downsample[1]] if self.downsample else [])

    def train(self, mode=True):
        return _XCNN(self, "block_train", (mode,), _XCNN_CONTEXT)

    def eval(self):
        return _XCNN(self, "block_train", (False,), _XCNN_CONTEXT)

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


    def forward(self, x):
        return _XCNN(self, "block_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        return _XCNN(self, "block_backward", (grad_out,), _XCNN_CONTEXT)

    def parameters(self):
        return _XCNN(self, "block_parameters", (), _XCNN_CONTEXT)

    def transform(self, X):
        return _XCNN(self, "block_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        return _XCNN(self, "block_fit", (), _XCNN_CONTEXT)


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
                                                 _data_addr(a)) for a in arrays)  # glue: address key of the graph input arrays

    def __eq__(self, other):
        return (isinstance(other, _GraphKey) and self.n == other.n and len(self.refs) == len(other.refs)
                and all(a is b for a, b in zip(self.refs, other.refs)) and self.sig == other.sig)  # glue: compares the graph key references

    __hash__ = None


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
        _XCNN(self, "gcn_init", (in_channels, out_channels, improved, add_self_loops, normalize, bias, random_state, numeric_mode), _XCNN_CONTEXT)

    _device_io = True

    def forward(self, x, edge_index, edge_weight=None):
        # the graph, its cache and the propagation in one native call
        # (bindings/xcnn_graph_native_door.mojo)
        return _XCNN(self, "gcn_forward", (x, edge_index, edge_weight), _XCNN_CONTEXT)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding the layer
    # runs on resident arrays: x goes up once and y comes down once; the
    # transform, the propagation (its CSR views and values resident with
    # the cached graph, checked once) and the bias add never visit the
    # host, nor does the backward's propagated gradient. It made a host
    # round trip of an (n, out) array per call (and uploaded a broadcast
    # bias of that size). The same entries' bodies on the same words; the
    # bias add is `bias_rows_at`, the same IEEE add per element as the
    # broadcast `add_at`. No bit moves.

    # TOMBSTONE: the Python _backward_dev body (a dead copy of the native resident backward, 0 callers; removed)
    # deleted 2026-11-09 by lane/x9-moe-gcn-rules; code recoverable at origin/integ/pre-0867-w2 f59d913b0b.
    # Restore: git apply experiments/removed/XCNN_GRAPH_HAND_BACKWARD.patch

    def backward(self, grad_out):
        return _XCNN(self, "gcn_backward", (grad_out,), _XCNN_CONTEXT)


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
        _XCNN(self, "sage_init", (in_channels, out_channels, aggr, normalize, root_weight, bias, project, random_state, numeric_mode), _XCNN_CONTEXT)

    _device_io = True

    def forward(self, x, edge_index):
        # the graph, its cache, the aggregation and the transforms in one
        # native call (bindings/xcnn_graph_native_door.mojo)
        return _XCNN(self, "sage_forward", (x, edge_index), _XCNN_CONTEXT)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding (aggr mean
    # or sum, no projection, no normalization) the layer runs on resident
    # arrays: x up once, y down once; the aggregation (the CSR views and
    # values resident with the cached graph), lin_l, the root transform and
    # their sum stay on the device, and the mean backward divides by the
    # in-degree read from the resident CSR offsets (spmm mode 3). The same
    # entries' bodies on the same words: no bit moves.

    # TOMBSTONE: the Python _backward_dev body (a dead copy of the native resident backward, 0 callers; removed)
    # deleted 2026-11-09 by lane/x9-moe-gcn-rules; code recoverable at origin/integ/pre-0867-w2 f59d913b0b.
    # Restore: git apply experiments/removed/XCNN_GRAPH_HAND_BACKWARD.patch

    def backward(self, grad_out):
        return _XCNN(self, "sage_backward", (grad_out,), _XCNN_CONTEXT)


_native_xcnn_bridge.register(
    xcnn_bn_class=BatchNorm2d, xcnn_block_class=BasicBlock,
    xcnn_tensor_class=DeviceTensor, xcnn_dev_class=_Dev, xcnn_maxpool_class=MaxPool2d,
    xcnn_avgpool_class=AvgPool2d, xcnn_pool1d_class=_Pool1dMixin, xcnn_conv1d_class=Conv1d,
    xcnn_device_io_off=_DEVICE_IO_OFF, xcnn_graph_class=_Graph, xcnn_graph_key_class=_GraphKey,
    xcnn_conv2d_class=Conv2d, xcnn_relu_class=_ReLU, xcnn_linear_class=_Linear,
)
_native_xcnn_bridge.connect_classifier()
