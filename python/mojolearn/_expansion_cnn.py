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
from . import _native_xcnn_bridge

# Package-import connection: the migrated layer methods are ONE call each
# into the compiled owner (bindings/xcnn_native.mojo).
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


def _pair(v, name):
    if isinstance(v, int):
        v = (v, v)
    v = tuple(int(t) for t in v)  # glue: converts a kernel size argument pair
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


def _mixed(b):
    """Whether binding `b` has the mixed-residency entries (`x_cnn_*_m`,
    lane gap-neural-overhead2): the GPU binding does; a CPU install's twin
    runs the host entries, the same words."""
    return hasattr(b, "x_cnn_map2_m")


# lane fam-neural (2026-10-04): the binding's `x_cnn_idn_flags` names the
# lane's IDENTICAL switches this build has on (x_cnn/device.mojo "lane
# fam-neural: `_m` forms"; each has a `-D ..._OFF` before arm and all are off
# under MOJOLEARN_IDN_ALL_OFF, in FAST, and on the CPU twin, which has no
# such entry). The glue takes a new path only when its bit is set; otherwise
# every layer runs exactly the calls it ran before.
_F_ADAPT, _F_GRAPH, _F_PAD, _F_GROUP, _F_LAYER_IO = 1, 2, 4, 8, 16
# TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
# Tried: _F_BN_FUSE = 64, the fused BatchNorm flag bit.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.
_IDN_FLAGS = {}


def _idn(b, bit):
    """Whether mixed binding `b` has lane fam-neural switch `bit` on."""
    hit = _IDN_FLAGS.get(id(b))
    if hit is None or hit[0] is not b:
        f = getattr(b, "x_cnn_idn_flags", None)
        hit = (b, int(f()) if f is not None and _mixed(b) else 0)
        _IDN_FLAGS[id(b)] = hit
    return (hit[1] & bit) != 0


# lane fam2-neural (2026-10-04): `x_cnn_idn2_flags` (BOTH bindings export it:
# these switches move bits, so the host column moves with the devices).
# Bit 0: the softmax mean loss is the blocked device fold (inside the
# binding; nothing for the glue to choose). Bit 1: the fit epoch's row order
# and Adam step scalars come from device kernels (`x_cnn_epoch_rows`,
# `x_cnn_adam_hyper_d`, `x_cnn_fit_epoch_d`), not the CPU helpers
# `epoch_order_i32` and `adam_hyper_f64`.
# Bit 2 (lane fix-n1-lm-neural): GCNConv's remaining self loops come from
# `x_cnn_gcn_loops` (a stable device sort and two gathers; the host twin on a
# CPU-only install), not host NumPy. Copies only: no bit moves.
# Bit 3: the candidate fold block 256. Bit 4 (lane/review-fixes): the binding
# was built with -D MOJOLEARN_IDN_ALL_OFF (read by `_device_io_off`).
_F2_XENT_FOLD, _F2_EPOCH_DEV, _F2_GCN_LOOPS = 1, 2, 4
_F2_FOLD_BLOCK_256, _F2_ALL_OFF = 8, 16
_IDN2_FLAGS = {}


def _idn2(b, bit):
    """Whether binding `b` (either column) has lane fam2-neural switch `bit` on."""
    hit = _IDN2_FLAGS.get(id(b))
    if hit is None or hit[0] is not b:
        f = getattr(b, "x_cnn_idn2_flags", None)
        hit = (b, int(f()) if f is not None else 0)
        _IDN2_FLAGS[id(b)] = hit
    return (hit[1] & bit) != 0


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


def _dev_of(layer, b):
    """`layer`'s `_Dev` on binding `b` (a new one if the binding changed)."""
    d = layer.__dict__.get("_dev")
    if d is None or d.b is not b:
        if d is not None:
            d.free()
        d = _Dev(b)
        layer._dev = d
    return d


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


def _device_io_off(b):
    """Whether the resident I/O route is off for binding `b`: the env switches
    above, or a `-D MOJOLEARN_IDN_ALL_OFF` build of `b`."""
    return _DEVICE_IO_OFF or _idn2(b, _F2_ALL_OFF)


def _size(shape):
    """The element count of a shape tuple."""
    n = 1
    for v in shape:  # glue: a shape tuple's few ints
        n *= int(v)
    return n


# lane fam2-neural (2026-10-04): PINNED LAYER WEIGHTS. A layer given a
# `DeviceTensor` still uploaded its `weight_` (and `bias_`) on every forward
# and every backward. `layer.pin_weights()` uploads them once into the
# layer's resident arrays; the `_m` entries then read them there (their
# residency bit set) until `unpin_weights()`. The host arrays stay the
# layer's weights: after changing them (in place, or `set_weights`) call
# `pin_weights()` again. The same entry on the same words: no bit moves.
# `x_cnn_idn_flags` bit 5 (`-D MOJOLEARN_IDN_PIN_WEIGHTS_OFF`); with it off,
# on FAST or on the CPU twin `pin_weights()` does nothing.
_F_PIN = 32


def _pin_weights(layer, names):
    """Upload `layer`'s float32 arrays `names` once; returns the layer."""
    np = _np()
    b = layer._binding()
    layer._pins = None
    if not _idn(b, _F_PIN):
        return layer
    dev = _dev_of(layer, b)
    handles = {}
    for name in names:  # glue: one upload per named weight array
        arr = getattr(layer, name, None)
        if arr is None:
            continue
        a = np.ascontiguousarray(arr, np.float32)
        handles[name] = (dev.upload("pin_" + name, a), int(a.size))
    layer._pins = (b, handles)
    # lane cnn-resident-glue: the array objects pinned, so the G3 auto-pin
    # sees a reassigned `weight_` / `bias_` and uploads again
    layer._pin_src = tuple(getattr(layer, name, None) for name in names)  # glue: the named weight arrays
    return layer


def _pinned(layer, b, name, size):
    """The resident handle of `layer`'s pinned array `name` (holding `size`
    words) on binding `b`, or None when it is not pinned there."""
    p = layer.__dict__.get("_pins")
    if not p or p[0] is not b:
        return None
    hit = p[1].get(name)
    if hit is None or hit[1] != int(size):
        return None
    return hit[0]


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
_F_RESIDENT = 128


def _resident(b):
    """Whether binding `b` runs the G3 resident glue."""
    return _mixed(b) and _idn(b, _F_RESIDENT) and not _device_io_off(b)


def _out_new(b, shape):
    """A resident output whose every word the entry stores: unfilled under
    G3 (`DeviceTensor._new_out`), the filled `_new` otherwise."""
    return DeviceTensor._new_out(b, shape) if _resident(b) else DeviceTensor._new(b, shape)


def _auto_pin(layer, b, names):
    """G3: `layer`'s arrays `names` pinned on binding `b` (uploaded now when
    unpinned or reassigned since); whether they are pinned."""
    if not (_resident(b) and _idn(b, _F_PIN)):
        return False
    d = layer.__dict__
    p, src = d.get("_pins"), d.get("_pin_src")
    if (p and p[0] is b and src is not None and len(src) == len(names)
            and all(a is getattr(layer, n, None) for a, n in zip(src, names))):  # glue: the named weight arrays
        return True
    _pin_weights(layer, names)
    return bool(d.get("_pins"))


class DeviceTensor:
    """A float32 tensor resident on the device (an `x_cnn_res_alloc` array
    and its shape). Make one with `to_device(array)`; read it back with
    `.numpy()`. Freed with the object."""
    __slots__ = ("b", "shape", "h", "_host", "_base", "__weakref__")
    _device_tensor = True

    def __init__(self, binding, shape, h=None, host=None, base=None):
        self.b, self.shape = binding, tuple(int(v) for v in shape)  # glue: a shape tuple's few ints
        self.h, self._host, self._base = h, host, base

    @classmethod
    def _new(cls, b, shape):
        """An unwritten resident tensor of `shape` on mixed binding `b`."""
        return cls(b, shape, h=int(b.x_cnn_res_alloc(max(_size(shape), 1))))

    @classmethod
    def _new_out(cls, b, shape):
        """`_new` for an output whose every word the entry stores (lane
        neural-io-2, XCNN_RES_NEW_NOZERO): no +0.0 fill on the device, on a
        binding that has `x_cnn_res_alloc_nofill`."""
        alloc = getattr(b, "x_cnn_res_alloc_nofill", None) or b.x_cnn_res_alloc
        return cls(b, shape, h=int(alloc(max(_size(shape), 1))))

    @classmethod
    def _wrap(cls, b, a):
        """Host float32 array `a` as a tensor on binding `b` (one upload)."""
        np = _np()
        a = np.ascontiguousarray(a, np.float32)
        if not _mixed(b):
            return cls(b, a.shape, host=a)
        t = cls._new(b, a.shape)
        if a.size:
            b.x_cnn_res_upload(t.h, a.ctypes.data, a.size)
        return t

    @property
    def size(self):
        return _size(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    dtype = "float32"

    def numpy(self):
        np = _np()
        if self.h is None:
            return self._host.reshape(self.shape).copy()
        out = np.empty(self.shape, np.float32)
        if out.size:
            self.b.x_cnn_res_download(self.h, out.ctypes.data, out.size)
        return out

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype, copy=False)

    def reshape(self, *shape):
        """The same resident words under another shape (no copy; the view
        keeps this tensor alive)."""
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = tuple(shape[0])
        shape = [int(v) for v in shape]  # glue: a shape tuple's few ints
        if shape.count(-1) == 1:
            known = _size(v for v in shape if v != -1)  # glue: a shape tuple's few ints
            shape[shape.index(-1)] = self.size // known if known else 0
        if _size(shape) != self.size:
            raise ValueError(f"mojolearn: cannot reshape {self.shape} to {tuple(shape)}")
        return DeviceTensor(self.b, shape, h=self.h, host=self._host, base=self)

    def __del__(self):
        if self._base is None and self.h is not None:
            try:
                self.b.x_cnn_res_free(self.h)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass


def to_device(x, numeric_mode=None):
    """float32 array `x` as a `DeviceTensor` (uploaded once, here). The
    layers that take one keep their output on the device too."""
    if isinstance(x, DeviceTensor):
        return x
    a = _f32(x, "x")
    if _DEVICE_IO_OFF:
        return a
    b = _backend.binding(_BINDING, numeric_mode or _backend.default_mode())
    if _device_io_off(b):
        return a
    return DeviceTensor._wrap(b, a)


def _is_t(x):
    return isinstance(x, DeviceTensor)


def _on_dev(b, *ts):
    """Whether every one of `ts` is a tensor resident on mixed binding `b`."""
    return _mixed(b) and all(_is_t(t) and t.h is not None and t.b is b for t in ts)  # glue: the call's arguments


def _host(x):
    return x.numpy() if _is_t(x) else x


def _ones_dev(dev, n):
    """A resident vector of `n` ones kept with the layer (uploaded when the
    size changes), the bias gradient's GEMM operand."""
    n = max(int(n), 1)
    cur = dev.h.get("ones")
    if cur is not None and cur[1] == n:
        return cur[0]
    return dev.upload("ones", _np().ones(n, _np().float32))


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
        if self.__dict__.get("_pins"):
            self.pin_weights()  # the pinned copy follows the new weights
        return self

    def pin_weights(self):
        """Keep `weight_` and `bias_` on the device for `DeviceTensor`
        forwards and backwards (lane fam2-neural); call again after changing
        them. Returns self."""
        return _pin_weights(self, ("weight_", "bias_"))

    def unpin_weights(self):
        self._pins = None
        return self

    # lane cnn-resident-glue (G3): the weight gradients of a resident
    # backward stay in this layer's resident arrays (`_gw_dev`, `_gb_dev`)
    # until read; the host value lives under the same name in `__dict__`.
    @property
    def grad_weight_(self):
        d = self.__dict__
        pend = d.get("_gw_dev")
        if pend is not None:
            np = _np()
            w = pend[0].download(pend[1], np.empty(self.weight_.shape, np.float32))
            d["grad_weight_"] = w[:, :, 0, :] if d.get("_gw_1d") else w
            d["_gw_dev"] = None
        if "grad_weight_" not in d:
            raise AttributeError("grad_weight_")
        return d["grad_weight_"]

    @grad_weight_.setter
    def grad_weight_(self, v):
        self.__dict__.update(grad_weight_=v, _gw_dev=None, _gw_1d=False)

    @property
    def grad_bias_(self):
        d = self.__dict__
        pend = d.get("_gb_dev")
        if pend is not None:
            d["grad_bias_"] = pend[0].download(pend[1], _np().empty(self.out_channels, _np().float32))
            d["_gb_dev"] = None
        if "grad_bias_" not in d:
            raise AttributeError("grad_bias_")
        return d["grad_bias_"]

    @grad_bias_.setter
    def grad_bias_(self, v):
        self.__dict__.update(grad_bias_=v, _gb_dev=None)

    def _res_w(self, b):
        """G3: (weight handle, bias handle) of the auto-pinned weights on
        binding `b`, or (None, None) when the switch is off or not pinned."""
        if self.groups != 1 or not _auto_pin(self, b, ("weight_", "bias_")):
            return None, None
        return _pinned(self, b, "weight_", self.weight_.size), _pinned(self, b, "bias_", self.bias_.size)

    def _backward_res(self, b, xh, gh, dxh, prm):
        """G3: the groups == 1 resident backward with dW / db left in this
        layer's resident arrays; False when the switch is off here."""
        if not _resident(b) or self.groups != 1:
            return False
        np = _np()
        dev = _dev_of(self, b)
        pw, _ = self._res_w(b)
        w = None if pw is not None else np.ascontiguousarray(self.weight_, np.float32)
        gwh = dev.get("gw", self.weight_.size)
        gbh = dev.get("gb", self.out_channels)
        b.x_cnn_conv2d_backward_m([xh, pw if pw is not None else w.ctypes.data, gh, dxh, gwh, gbh],
                                  0b111111 if pw is not None else 0b111101, prm)
        self.__dict__.update(_gw_dev=(dev, gwh), _gw_1d=False)
        if self.bias:
            self.__dict__["_gb_dev"] = (dev, gbh)
        else:
            self.grad_bias_ = np.zeros(self.out_channels, np.float32)
        return True

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

    _device_io = True

    def _res_ok(self, b, *ts):
        """Whether the resident entries serve this layer on tensors `ts`.
        lane fam-neural: an explicit pad (IDN_PAD_M) and groups (IDN_GROUP_M)
        have resident forms too when the binding has them on."""
        if not _on_dev(b, *ts):
            return False
        if self.groups != 1 and not _idn(b, _F_GROUP):
            return False
        if self._explicit and not _idn(b, _F_PAD):
            return False
        return True

    def _pad_t(self, b, x):
        """The explicitly padded resident copy of resident `x` (lane
        fam-neural: `x_cnn_pad2d_m`, the host entry's launch on the words
        where they live), or `x` itself when the conv carries the pad."""
        if not self._explicit:
            return x
        xp = _out_new(b, self._padded_shape(x.shape))   # the pad stores every word
        b.x_cnn_pad2d_m([x.h, xp.h], 0b11, self._pad_params(x.shape) + [0])
        return xp

    def _group_prm(self, shape, channels, g):
        """`x_cnn_chan_copy_m`'s block for group `g` of an (N, channels, H, W) tensor."""
        per = channels // self.groups
        return [shape[0], channels, shape[2] * shape[3], per, g * per]

    def _check_x(self, x):
        if x.ndim != 4:
            raise ValueError("mojolearn: Conv2d.forward takes (N, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"mojolearn: input has {x.shape[1]} channels, the layer {self.in_channels}")

    def _backward_t(self, grad_out, x):
        """`backward` with a `DeviceTensor` grad_out or input."""
        np = _np()
        b = self._binding()
        xs = self._x if x is None else x
        if not (_is_t(xs) and self._res_ok(b, grad_out, xs)):
            dx = Conv2d.backward(self, _host(grad_out), _host(xs))
            return DeviceTensor._wrap(b, dx) if _is_t(grad_out) else dx
        self._check_x(xs)
        shape = self._out_shape(xs.shape)
        if grad_out.shape != shape:
            raise ValueError(f"mojolearn: grad_out shape {grad_out.shape}, expected {shape}")
        # lane fam-neural: the forward's padded tensor when this is its
        # backward (x not passed), else the pad of the x given
        held = self.__dict__.get("_xp")
        if self._explicit and x is None and _is_t(held) and held is not xs and _on_dev(b, held):
            xp = held
        else:
            xp = self._pad_t(b, xs)
        prm = self._params(xp.shape)
        dxp = _out_new(b, xp.shape)   # col2im stores every word
        dw = np.empty(self.weight_.shape, np.float32)
        db = np.empty(self.out_channels, np.float32)
        if self._backward_res(b, xp.h, grad_out.h, dxp.h, prm):
            dw = None   # G3: dW / db stay resident
        elif self.groups == 1:
            pw = _pinned(self, b, "weight_", self.weight_.size)
            if pw is not None:
                b.x_cnn_conv2d_backward_m([xp.h, pw, grad_out.h, dxp.h, dw.ctypes.data, db.ctypes.data],
                                          0b001111, prm)
            else:
                w = np.ascontiguousarray(self.weight_, np.float32)
                b.x_cnn_conv2d_backward_m([xp.h, w.ctypes.data, grad_out.h, dxp.h, dw.ctypes.data, db.ctypes.data],
                                          0b001101, prm)
        else:
            dev = _dev_of(self, b)
            og = self.out_channels // self.groups
            xg = dev.get("xg", xp.size // self.groups)
            gg = dev.get("gg", grad_out.size // self.groups)
            dxg = dev.get("dxg", xp.size // self.groups)
            for g in range(self.groups):  # glue: one entry per group
                wg = np.ascontiguousarray(self.weight_[g * og:(g + 1) * og], np.float32)
                dwg = np.empty(wg.shape, np.float32)
                dbg = np.empty(og, np.float32)
                b.x_cnn_chan_copy_m([xp.h, xg], 0b11, self._group_prm(xp.shape, self.in_channels, g) + [0])
                b.x_cnn_chan_copy_m([grad_out.h, gg], 0b11, self._group_prm(shape, self.out_channels, g) + [0])
                b.x_cnn_conv2d_backward_m([xg, wg.ctypes.data, gg, dxg, dwg.ctypes.data, dbg.ctypes.data],
                                          0b001101, prm)
                b.x_cnn_chan_copy_m([dxg, dxp.h], 0b11, self._group_prm(xp.shape, self.in_channels, g) + [1])
                dw[g * og:(g + 1) * og] = dwg
                db[g * og:(g + 1) * og] = dbg
        dx = dxp
        if self._explicit:
            dx = _out_new(b, xs.shape)   # the pad's backward stores every word
            b.x_cnn_pad2d_m([dxp.h, dx.h], 0b11, self._pad_params(xs.shape) + [1])
        self._x, self._xp = xs, xp
        if dw is not None:
            self.grad_weight_ = dw
            self.grad_bias_ = db if self.bias else np.zeros_like(db)
        return dx

    def forward(self, x):
        return _XCNN(self, "conv_forward", (x,), _XCNN_CONTEXT)

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

    def _group_dev(self, b):
        """cpu2-l11-neural (2026-10-04): whether a GROUPED conv on host
        arrays runs on the resident `_m` forms (one upload of each input,
        the per-group channel slices on the device, one download), instead
        of NumPy slicing every group on the host."""
        return (self.groups > 1 and _mixed(b) and _idn(b, _F_GROUP)
                and (not self._explicit or _idn(b, _F_PAD)))

    def backward(self, grad_out, x=None):
        np = _np()
        if _is_t(grad_out) or _is_t(x) or (x is None and _is_t(self.__dict__.get("_x"))):
            return self._backward_t(grad_out, x)
        if self._group_dev(self._binding()):
            b = self._binding()
            xs = _f32(self._x if x is None else x, "x")
            dx = self._backward_t(DeviceTensor._wrap(b, _f32(grad_out, "grad_out")),
                                  DeviceTensor._wrap(b, xs)).numpy()
            self._x, self._xp = xs, None
            return dx
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
        for k in range(self.groups):  # glue: one conv backward launch per channel group (groups-sized: convolution groups)
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
        return _XCNN(self, "conv_transform", (X,), _XCNN_CONTEXT)

    def fit(self, X, y=None):
        """Layers carry their weights from construction; fit only records the input shape."""
        return _XCNN(self, "conv_fit", (X,), _XCNN_CONTEXT)


def _t4(t):
    """Tensor (N, C, L) as (N, C, 1, L): the same resident words."""
    if t.ndim != 3:
        raise ValueError("mojolearn: expected (N, C, L)")
    return t.reshape(t.shape[0], t.shape[1], 1, t.shape[2])


def _t3(t):
    return t.reshape(t.shape[0], t.shape[1], t.shape[3])


class Conv1d(Conv2d):
    """1-D convolution, PyTorch `nn.Conv1d` semantics: Conv2d over (N, C, 1, L)
    (padding int, 'same' or 'valid'; every padding_mode; groups)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", random_state=0, input_shape=None, numeric_mode=None):
        k, s, d = (int(v if isinstance(v, int) else v[0]) for v in (kernel_size, stride, dilation))  # glue: unpacks three layer size arguments
        pad = padding if isinstance(padding, str) else (0, int(padding if isinstance(padding, int) else padding[0]))
        super().__init__(in_channels, out_channels, (1, k), (1, s), pad, (1, d), groups, bias, padding_mode,
                         random_state, None, numeric_mode)
        self.input_shape = input_shape

    def set_weights(self, weight, bias=None):
        w = _f32(weight, "weight")
        return super().set_weights(w.reshape(self.out_channels, self.in_channels // self.groups, 1, -1), bias)

    def forward(self, x):
        return _XCNN(self, "conv_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out, x=None):
        if _is_t(grad_out) or _is_t(x):
            g = _t4(grad_out) if _is_t(grad_out) else _f32(grad_out, "grad_out")[:, :, None, :]
            xx = x if x is None else (_t4(x) if _is_t(x) else _f32(x, "x")[:, :, None, :])
            dx = super().backward(g, xx)
            if self.__dict__.get("_gw_dev") is not None:
                self.__dict__["_gw_1d"] = True   # G3: squeezed when downloaded
            else:
                self.grad_weight_ = self.grad_weight_[:, :, 0, :]
            return _t3(dx) if _is_t(dx) else dx[:, :, 0, :]
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
        self.numeric_mode = numeric_mode

    def forward(self, x):
        np = _np()
        if _is_t(x):
            # lane fam-neural (IDN_LAYER_DEV_IO): x and the output resident
            # (`x_cnn_map2_m` kind 0, the host entry's element function)
            b = self._binding()
            if _on_dev(b, x) and _idn(b, _F_LAYER_IO) and x.size > 0:
                out = DeviceTensor._new(b, x.shape)
                b.x_cnn_map2_m([x.h, x.h, out.h], 0b111, [0, x.size])
                self._x = x
                return out
        x = _f32(x, "x")
        out = np.empty_like(x)
        self._binding().x_cnn_relu_forward(x.ctypes.data, out.ctypes.data, [x.size])
        self._x = x
        return out

    def backward(self, grad_out):
        np = _np()
        if _is_t(self._x):
            b = self._binding()
            if _on_dev(b, grad_out, self._x) and grad_out.size == self._x.size:
                dx = DeviceTensor._new(b, self._x.shape)
                b.x_cnn_map2_m([self._x.h, grad_out.h, dx.h], 0b111, [1, dx.size])
                return dx
            self._x = self._x.numpy()
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


def _adam_hyper(step, lr, betas, eps, weight_decay, decoupled, binding=None):
    """adam_at's hyper block for 1-based step `step` (the scalars in double,
    as torch computes them in Python) for the resident fit.
    DEVIATION 6900: torch's `beta ** step` calls the platform pow; here it is
    `_pm.powi`, correctly rounded (the platform's bits wherever its pow is),
    the same bits on every host. sqrt is correctly rounded everywhere."""
    if step != int(step):
        raise ValueError(f"Adam step must be a whole number, got {step!r}")
    return _adam_hyper_block(int(step), 1, lr, betas, eps, weight_decay, decoupled, binding).reshape(-1).tolist()


def _adam_hyper_block(step0, nsteps, lr, betas, eps, weight_decay, decoupled, binding=None):
    """`_adam_hyper` of steps step0 .. step0 + nsteps - 1 as an (nsteps, 9)
    float64 array, made by the binding's `x_cnn_adam_hyper_d` kernel (lane
    fam2-neural; the host `adam_hyper_f64` arm left the GPU route in lane
    cpu3-python): the same bits on every column."""
    np = _np()
    out = np.empty((max(int(nsteps), 1), 9), dtype=np.float64)
    if binding is None:
        raise ValueError("mojolearn CNN: the Adam step scalars come from the binding's kernel; pass the binding")
    # lane fam2-neural: the step scalars from the binding's own kernel
    # (`adam_hyper_at`, float32 pairs; the same words on every column). Lane
    # cpu3-python: on every build, the host `adam_hyper_f64` arm is gone.
    if int(step0) < 1:
        raise ValueError("Adam step must be at least 1")
    binding.x_cnn_adam_hyper_d(out.ctypes.data, [int(step0), int(nsteps)],
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
        self.input_shape = tuple(int(v) for v in input_shape)  # glue: converts the input_shape argument
        self.conv_channels = tuple(int(c) for c in conv_channels)  # glue: converts the conv_channels argument
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
        for i, oc in enumerate(self.conv_channels):  # glue: builds one layer per conv_channels entry
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
        for i, layer in enumerate(layers):  # glue: walks the built layer list
            if isinstance(layer, Conv2d):
                nxt = layers[i + 2] if i + 2 < len(layers) else None
                self._blocks.append((layer, nxt if isinstance(nxt, MaxPool2d) else None))

    def _params(self):
        out = []
        for layer in self.layers_ + [self.head_]:  # glue: walks the layer parameter list
            if hasattr(layer, "weight_"):
                out.append((layer, "weight_", "grad_weight_"))
                out.append((layer, "bias_", "grad_bias_"))
        return out

    def _plan(self, n):
        """Per block (conv params, pool params or [], input size, output size,
        im2col size, conv output size) at batch size n, and the final shape."""
        plans = []
        shape = (n,) + self.input_shape
        for conv, pool in self._blocks:  # glue: plans one launch per conv block
            prm = conv._params(shape)
            cshape = conv._out_shape(shape)
            pprm = pool._params(cshape) if pool is not None else []
            oshape = pool._out_shape(cshape) if pool is not None else cshape
            ny = int(_np().prod(cshape))  # glue: product of the output shape entries
            plans.append((prm, pprm, int(_np().prod(shape)), int(_np().prod(oshape)),  # glue: products of the plan shape entries
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
        a["out"] = [R.new(p[3]) for p in plans]  # glue: allocates one buffer per conv block
        a["idx"] = [R.new(p[3]) for p in plans]  # glue: allocates one buffer per conv block
        a["gout"] = [R.new(p[3]) for p in plans] if (train or _LEGACY_STEP) else []  # glue: allocates one buffer per conv block
        bounded = bool(getattr(R.b, "x_cnn_bounded_im2col", lambda: False)())
        # NI10: allocation planning only. The binding reports its compiled
        # A/B policy; Python never computes activations or tensor values.
        # Keep two zero handles for recomputed blocks so the native epoch
        # descriptor still has its required nine entries.
        tape_budget = (int(R.b.x_cnn_neural_tape_budget_bytes())
                       if hasattr(R.b, "x_cnn_neural_tape_budget_bytes") else -1)
        a["saved"] = []
        for p in plans:  # glue: allocates optional buffers per conv block
            tape_bytes = 4 * ((1 if bounded else p[4]) + p[5])
            keep = save and (tape_budget < 0 or tape_bytes <= tape_budget)
            a["saved"].append([R.new(1 if bounded else p[4]), R.new(p[5])] if keep else ([0, 0] if save else []))
            if keep and tape_budget >= 0:
                tape_budget -= tape_bytes
        if not plans:  # the head's input gradient, never read, kept apart from glog
            a["ghead"] = R.new(n * self._flat)
        return a

    def _forward_r(self, b, a, n):
        """Logits of the n rows in a["x"], every array on the binding's side."""
        plans, _ = self._plan(n)
        src = a["x"]
        for (conv, pool), p, out, idx, sv in zip(self._blocks, plans, a["out"], a["idx"], a["saved"]):  # glue: drives one device launch per conv block
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
        # the labels, the layers, the resident arrays and every epoch in one
        # native call (bindings/xcnn_classifier_native.mojo)
        return _XCNN(self, "cnnclf_fit", (X, y), _XCNN_CONTEXT)

    def predict_proba(self, X):
        return _XCNN(self, "cnnclf_predict_proba", (X,), _XCNN_CONTEXT)

    def predict(self, X):
        return _XCNN(self, "cnnclf_predict", (X,), _XCNN_CONTEXT)

    def weights(self):
        """Every trained array, in layer order: [w0, b0, w1, b1, ...]."""
        return [getattr(l, a) for l, a, _ in self._params()]  # glue: walks the layer parameter list


def np_reshape(X, shape):
    return _np().ascontiguousarray(X).reshape(shape)


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

    # lane cnn-resident-glue (G3, CNN resident grads, default on; _OFF clears it): on a
    # resident forward the running statistics stay in the resident array
    # "running" (uploaded when the host arrays were set since) and the aux
    # block (the statistics the backward reads, and its dgamma / dbeta) in
    # one of two resident arrays "aux0" / "aux1" (alternating, so the next
    # forward never overwrites the gradients of the last backward). The
    # attributes download on the first host read; the host values live under
    # the same names in `__dict__`.
    def _sync_running(self):
        d = self.__dict__
        if not d.get("_run_stale"):
            return
        d["_run_stale"] = False
        dev = d.get("_run_dev")
        if dev is not None and "running" in dev.h:
            np = _np()
            C = self.num_features
            r = dev.download(dev.h["running"][0], np.empty(2 * C, np.float32))
            d["running_mean_"], d["running_var_"] = r[:C].copy(), r[C:].copy()

    @property
    def running_mean_(self):
        self._sync_running()
        return self.__dict__.get("running_mean_")

    @running_mean_.setter
    def running_mean_(self, v):
        self._sync_running()
        self.__dict__.update(running_mean_=v, _run_dev=None)

    @property
    def running_var_(self):
        self._sync_running()
        return self.__dict__.get("running_var_")

    @running_var_.setter
    def running_var_(self, v):
        self._sync_running()
        self.__dict__.update(running_var_=v, _run_dev=None)

    def _grads_host(self):
        """Download the pending resident dgamma / dbeta (G3)."""
        d = self.__dict__
        pend = d.get("_g_dev")
        if pend is None:
            return
        d["_g_dev"] = None
        np = _np()
        C = self.num_features
        aux = pend[0].download(pend[1], np.empty(2 + 7 * C, np.float32))
        gb, gw = aux[2 + 3 * C:2 + 4 * C].copy(), aux[2 + 4 * C:2 + 5 * C].copy()
        if not self.affine:
            gw[:] = 0
            gb[:] = 0
        d["grad_weight_"], d["grad_bias_"] = gw, gb

    @property
    def grad_weight_(self):
        self._grads_host()
        if "grad_weight_" not in self.__dict__:
            raise AttributeError("grad_weight_")
        return self.__dict__["grad_weight_"]

    @grad_weight_.setter
    def grad_weight_(self, v):
        self._grads_host()
        self.__dict__["grad_weight_"] = v

    @property
    def grad_bias_(self):
        self._grads_host()
        if "grad_bias_" not in self.__dict__:
            raise AttributeError("grad_bias_")
        return self.__dict__["grad_bias_"]

    @grad_bias_.setter
    def grad_bias_(self, v):
        self._grads_host()
        self.__dict__["grad_bias_"] = v

    def _res_bn(self, b):
        return self.track_running_stats and _resident(b)

    def _backward_res(self, b, xh, gh, dxh, dev_dx, dev_g):
        """G3: the backward into the forward's resident aux; dgamma / dbeta
        stay there until read. False when that forward was not resident."""
        ad = self.__dict__.get("_aux_dev")
        if ad is None or ad[0].b is not b or not self._res_bn(b):
            return False
        n, c, hw = self._x3shape
        self.__dict__["_g_dev"] = None   # superseded by this backward (no download)
        b.x_cnn_batchnorm_backward_m([xh, gh, dxh, ad[1]], 0b1001 | (0b0010 if dev_g else 0) | (0b0100 if dev_dx else 0),
                                     [n, c, hw, 1 if self._mode else 0])
        self.__dict__["_g_dev"] = ad
        return True

    def _backward_dev(self, b, xh, gh, dxh, dev_dx=True):
        """The backward from resident g (`gh`) into resident dx (`dxh`, or a
        host array's address with `dev_dx` False); sets the gradients."""
        if self._backward_res(b, xh, gh, dxh, dev_dx, True):
            return
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

    # TOMBSTONE: MOJOLEARN_IDN_CNN_BN_FUSE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
    # Tried: BatchNorm2d._forward_dev_fused / _backward_dev_relu (the fused entries' glue).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_BN_FUSE.patch; record in docs/TOMBSTONES.md.

    def forward(self, x):
        return _XCNN(self, "bn_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        np = _np()
        if _is_t(grad_out):
            b = self._binding()
            xh0 = self.__dict__.get("_xdev")
            own0 = self.__dict__.get("_xdev_dev")
            if (_on_dev(b, grad_out) and _idn(b, _F_LAYER_IO) and xh0 is not None and own0 is not None
                    and own0.b is b and grad_out.size == _size(self._x3shape)):
                dx = DeviceTensor._new_out(b, self._shape)   # the dx pass stores every word
                self._backward_dev(b, xh0, grad_out.h, dx.h, dev_dx=True)
                return dx
        g = _f32(grad_out, "grad_out")
        g3 = np_reshape(g if g.ndim != 2 else g[:, :, None], self._x3shape)
        n, c, hw = self._x3shape
        C = self.num_features
        b = self._binding()
        dx = np.empty(self._x3shape, np.float32)
        xh = self.__dict__.get("_xdev")
        owner = self.__dict__.get("_xdev_dev")
        if (xh is not None and owner is not None and owner.b is b and _mixed(b)
                and self._backward_res(b, xh, g3.ctypes.data, dx.ctypes.data, False, False)):
            return dx.reshape(self._shape) if len(self._shape) != 2 else dx[:, :, 0]
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

    def _divides(self, shape):
        return shape[2] % self.output_size[0] == 0 and shape[3] % self.output_size[1] == 0

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
        for bn in self._bns():  # glue: walks the block BatchNorm layers
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

    def _conv_back_dev(self, b, conv, xh, xshape, gh, oshape, name, dev):
        """conv's backward from resident g into resident dx `name`; sets its
        weight gradients (the same words `Conv2d.backward` sets)."""
        np = _np()
        dxh = dev.get(name, int(np.prod(xshape)))  # glue: product of the input shape entries
        if conv._backward_res(b, xh, gh, dxh, conv._params(xshape)):
            return dxh   # G3: dW / db stay in the conv's resident arrays
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
        gt = _is_t(grad_out)
        if gt and not _on_dev(b, grad_out):
            return DeviceTensor._wrap(b, self._backward_chain(b, grad_out.numpy()))
        g = grad_out if gt else _f32(grad_out, "grad_out")
        if tuple(g.shape) != tuple(k["s2"]):
            raise ValueError(f"mojolearn: grad_out shape {g.shape}, expected {tuple(k['s2'])}")
        n2 = g.size
        gh = g.h if gt else dev.upload("g", g)
        gs = dev.get("gs", n2)
        b.x_cnn_map2_m([k["s"], gh, gs], 0b111, [1, n2])              # relu2
        gb2 = dev.get("gb2", n2)
        self.bn2._backward_dev(b, k["c2"], gs, gb2)
        gr1 = self._conv_back_dev(b, self.conv2, k["r1"], k["s1"], gb2, k["s2"], "gr1", dev)
        n1 = int(np.prod(k["s1"]))  # glue: product of the saved shape entries
        gc1 = dev.get("gc1", n1)
        gb1 = dev.get("gb1", n1)
        b.x_cnn_map2_m([k["b1"], gr1, gb1], 0b111, [1, n1])           # relu1
        self.bn1._backward_dev(b, k["c1"], gb1, gc1)
        gx1 = self._conv_back_dev(b, self.conv1, k["xh"], k["x"], gc1, k["s1"], "gx1", dev)
        gi = gs
        if self.downsample:
            gd = dev.get("gd", n2)
            self.downsample[1]._backward_dev(b, k["d1"], gs, gd)
            gi = self._conv_back_dev(b, self.downsample[0], k["xh"], k["x"], gd, k["sd"], "gdx", dev)
        if gt:
            dxt = _out_new(b, k["x"])   # the add stores every word
            b.x_cnn_map2_m([gx1, gi, dxt.h], 0b111, [2, dxt.size])
            return dxt
        dx = np.empty(k["x"], np.float32)
        b.x_cnn_map2_m([gx1, gi, dx.ctypes.data], 0b011, [2, dx.size])
        return dx

    def forward(self, x):
        return _XCNN(self, "block_forward", (x,), _XCNN_CONTEXT)

    def backward(self, grad_out):
        b = self._binding()
        if self.__dict__.get("_chain") is not None and self._dev.b is b:
            return self._backward_chain(b, grad_out)
        if _is_t(grad_out):
            return DeviceTensor._wrap(b, self.backward(grad_out.numpy()))
        g = self.relu2.backward(grad_out)
        gm = self.conv1.backward(self.bn1.backward(self.relu1.backward(self.conv2.backward(self.bn2.backward(g)))))
        gi = g
        if self.downsample:
            gi = self.downsample[0].backward(self.downsample[1].backward(g))
        return _add(self._binding(), gm, gi)

    def parameters(self):
        out = []
        layers = [self.conv1, self.bn1, self.conv2, self.bn2] + (self.downsample or [])
        for layer in layers:  # glue: walks the block layer list
            out.append((layer, "weight_", "grad_weight_"))
            if isinstance(layer, BatchNorm2d):
                out.append((layer, "bias_", "grad_bias_"))
        return out

    def transform(self, X):
        return _XCNN(self, "block_transform", (X,), _XCNN_CONTEXT)

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
        np = _np()
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.improved, self.add_self_loops, self.normalize = bool(improved), bool(add_self_loops), bool(normalize)
        self.bias = bool(bias)
        self.numeric_mode = numeric_mode
        rng = _rng(random_state)
        bound = np.sqrt(6.0 / (self.in_channels + self.out_channels))  # glorot, PyG's Linear(weight_initializer='glorot')  # glue: scalar glorot bound of two sizes
        self.weight_ = _uniform(rng, bound, (self.out_channels, self.in_channels))
        self.bias_ = np.zeros(self.out_channels, np.float32)

    _device_io = True

    def forward(self, x, edge_index, edge_weight=None):
        # the graph, its cache and the propagation in one native call
        # (bindings/xcnn_graph_native.mojo)
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

    def _backward_dev(self, b, G):
        np = _np()
        dev, xh = self._xdev
        g = self._g
        n, F, D = G.shape[0], self.out_channels, self.in_channels
        if tuple(G.shape) != (self._x.shape[0], F):
            raise ValueError(f"mojolearn: grad_out shape {G.shape}, expected {(self._x.shape[0], F)}")
        gt = _is_t(G)
        Gh = G.h if gt else dev.upload("G", G)
        if self.bias:
            gb = np.empty((F, 1), np.float32)
            if _device_io_off(b):
                ones = np.ones(n, np.float32)
                b.x_cnn_gemm_m([Gh, ones.ctypes.data, gb.ctypes.data], 0b001, [F, 1, n, 2])
            else:
                # lane idn-cnn-resident: the ones operand is resident with
                # the layer, not built and uploaded per call (the same words)
                b.x_cnn_gemm_m([Gh, _ones_dev(dev, n), gb.ctypes.data], 0b011, [F, 1, n, 2])
            self.grad_bias_ = gb.reshape(-1)
        else:
            self.grad_bias_ = np.zeros(F, np.float32)
        dhh = dev.get("dh", n * F)
        b.x_cnn_spmm_m([dev.h["vals_t"][0], Gh, dev.h["csr_t"][0], dhh], 0b1111, [n, F, g.nnz, 0])
        gw = np.empty((F, D), np.float32)
        b.x_cnn_gemm_m([dhh, xh, gw.ctypes.data], 0b011, [F, D, n, 2])
        self.grad_weight_ = gw
        W = _f32(self.weight_, "b")
        if gt:
            dxt = DeviceTensor._new(b, (n, D))
            b.x_cnn_gemm_m([dhh, W.ctypes.data, dxt.h], 0b101, [n, D, F, 0])
            return dxt
        dx = np.empty((n, D), np.float32)
        b.x_cnn_gemm_m([dhh, W.ctypes.data, dx.ctypes.data], 0b001, [n, D, F, 0])
        return dx

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        xd = self.__dict__.get("_xdev")
        if _is_t(grad_out):
            if xd is not None and xd[0].b is b and _on_dev(b, grad_out):
                return self._backward_dev(b, grad_out)
            return DeviceTensor._wrap(b, self.backward(grad_out.numpy()))
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
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

    _device_io = True

    def forward(self, x, edge_index):
        # the graph, its cache, the aggregation and the transforms in one
        # native call (bindings/xcnn_graph_native.mojo)
        return _XCNN(self, "sage_forward", (x, edge_index), _XCNN_CONTEXT)

    # lane gap-neural-overhead2 (2026-10-02): on the GPU binding (aggr mean
    # or sum, no projection, no normalization) the layer runs on resident
    # arrays: x up once, y down once; the aggregation (the CSR views and
    # values resident with the cached graph), lin_l, the root transform and
    # their sum stay on the device, and the mean backward divides by the
    # in-degree read from the resident CSR offsets (spmm mode 3). The same
    # entries' bodies on the same words: no bit moves.

    def _backward_dev(self, b, G):
        np = _np()
        dev, xh, aggh = self._xdev
        g = self._g
        n, D, F = self._x.shape[0], self.in_channels, self.out_channels
        if tuple(G.shape) != (n, F):
            raise ValueError(f"mojolearn: grad_out shape {G.shape}, expected {(n, F)}")
        gt = _is_t(G)
        Gh = G.h if gt else dev.upload("G", G)
        if self.normalize:
            # lane fam-neural: the normalization's backward on the resident y, G and denominators
            yh, denh, _ = self._ydev
            g2h = dev.get("G2", n * F)
            b.x_cnn_graph_op_m([yh, Gh, denh, g2h, 0], 0b01111, [n, F, 0, 3])
            Gh = g2h
        lin = self.lin_l
        dagg = dev.get("dagg", n * D)
        dw = np.empty_like(lin.weight_)
        db = np.empty_like(lin.bias_)
        b.x_cnn_linear_backward_m([aggh, lin.weight_.ctypes.data, Gh, dagg, dw.ctypes.data, db.ctypes.data],
                                  0b001101, [n, D, F])
        lin.grad_weight_, lin.grad_bias_ = dw, db
        if not self.bias:
            lin.grad_bias_[:] = 0
        dxt = DeviceTensor._new(b, (n, D)) if gt else None
        dxs = dxt.h if gt and not self.root_weight else dev.get("dxs", n * D)
        if self.aggr == "mean":
            # each target row of dagg over its in-degree, read from the
            # forward CSR offsets on the device (spmm mode 3), then the
            # transposed fold with unit values: no host degree count
            dsc = dev.get("dsc", n * D)
            b.x_cnn_spmm_m([dev.h["vals_f"][0], dagg, dev.h["csr_f"][0], dsc], 0b1111, [n, D, g.nnz, 3])
            dagg = dsc
        if self.aggr == "max":
            # lane fam-neural: the max backward over the transposed view (resident x, dagg, aux)
            b.x_cnn_graph_op_m([xh, dagg, dev.h["maxaux"][0], dxs, dev.h["csr_t"][0]], 0b11111, [n, D, g.nnz, 1])
        else:
            b.x_cnn_spmm_m([dev.h["vals_t"][0], dagg, dev.h["csr_t"][0], dxs], 0b1111, [n, D, g.nnz, 0])
        dx = dxt if gt else np.empty((n, D), np.float32)
        if self.root_weight:
            gwr = np.empty((F, D), np.float32)
            b.x_cnn_gemm_m([Gh, xh, gwr.ctypes.data], 0b011, [F, D, n, 2])
            self.grad_weight_r_ = gwr
            Wr = _f32(self.weight_r_, "b")
            grh = dev.get("gr", n * D)
            b.x_cnn_gemm_m([Gh, Wr.ctypes.data, grh], 0b101, [n, D, F, 0])
            if gt:
                b.x_cnn_map2_m([grh, dxs, dx.h], 0b111, [2, n * D])
            else:
                b.x_cnn_map2_m([grh, dxs, dx.ctypes.data], 0b011, [2, n * D])
        elif not gt:
            dev.download(dxs, dx)
        return dx

    def backward(self, grad_out):
        np = _np()
        b = self._binding()
        xd = self.__dict__.get("_xdev")
        if _is_t(grad_out):
            if xd is not None and xd[0].b is b and _on_dev(b, grad_out):
                return self._backward_dev(b, grad_out)
            return DeviceTensor._wrap(b, self.backward(grad_out.numpy()))
        G = _f32(grad_out, "grad_out")
        n = G.shape[0]
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


_native_xcnn_bridge.register(
    xcnn_tensor_class=DeviceTensor, xcnn_dev_class=_Dev, xcnn_maxpool_class=MaxPool2d,
    xcnn_avgpool_class=AvgPool2d, xcnn_pool1d_class=_Pool1dMixin, xcnn_conv1d_class=Conv1d,
    xcnn_device_io_off=_DEVICE_IO_OFF, xcnn_graph_class=_Graph, xcnn_graph_key_class=_GraphKey,
    xcnn_conv2d_class=Conv2d, xcnn_relu_class=_ReLU, xcnn_linear_class=_Linear, xcnn_epoch_entry=_EPOCH_ENTRY,
)
_native_xcnn_bridge.connect_classifier()
