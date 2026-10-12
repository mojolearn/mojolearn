# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MoEBlock: the sparse mixture-of-experts feed-forward block of HF
transformers' Mixtral (`MixtralSparseMoeBlock`: `MixtralTopKRouter` +
`MixtralExperts`), forward on the GPU (`sequence/moe.mojo`): top-k routing
over the softmax of the router logits with the probabilities renormalised,
each chosen expert down_proj(silu(gate) * up), the weighted sum. The
weights keep HF's layout: `router` (E, D), `gate_up_proj` (E, 2F, D),
`down_proj` (E, D, F). Routing ties go to the lower expert index.

Refused (sequence/NOT_IMPLEMENTED.tsv): the backward pass, router jitter
noise, activations other than SiLU, the auxiliary load-balancing loss,
float64.

THE WEIGHTS STAY ON THE DEVICE (lane gap-neural-overhead2, 2026-10-02). On
the GPU binding the forward reads device copies of the three weight arrays
made once (`moe_weights_put`) instead of uploading all of them on every call
(277 MB at the board's shape). The copies follow the attributes: assigning
`router`, `gate_up_proj` or `down_proj` (or `load_state_dict`) makes the
next forward copy again. So that a stale copy can never run, the layer
holds its own float32 copy of each array and makes it READ-ONLY: edit a
weight by assigning a new array, not in place.

DEVICE I/O (lane neural-io-2, 2026-10-09): an x made by
`mojolearn._x_sequence_device.to_device` stays on the device: the forward
returns y as a `SequenceDeviceTensor` and keeps the routing outputs there
too (`router_logits_`, `selected_experts_`, `routing_weights_` are read
back when first accessed), so nothing crosses the bus inside the call
(`sequence/moe_dev.mojo`, the same launches on the same values). A binding
built with -D MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF has no `moe_forward_dev`; a
resident x is then read back once and the host-array entry runs. The
host-array entry's outputs are `np.empty` (the binding stores every cell)."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_moe')

from . import _backend
from ._buffer import InitStream
from ._x_sequence_device import SequenceDeviceTensor, is_seq_tensor


_WEIGHTS = ("router", "gate_up_proj", "down_proj")


# Weights stay on the device (lane gap-neural-overhead2): the forward reads device copies made
# once by `moe_weights_put` instead of uploading the three arrays every call; any assignment of
# router / gate_up_proj / down_proj (or load_state_dict) re-copies on the next forward, and the
# read-only host copies make a stale device copy impossible. Device I/O (lane neural-io-2): a
# resident x keeps y and the routing outputs on the device (`sequence/moe_dev.mojo`); a binding
# built with -D MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF has no `moe_forward_dev` and the host entry runs.
class MoEBlock:
    """Sparse mixture-of-experts feed-forward block, as Mixtral's `MixtralSparseMoeBlock`.

    A router scores every token against `num_experts` experts; the `top_k` highest
    softmax probabilities are kept (renormalized to sum to one when `norm_topk_prob`),
    each chosen expert computes ``down_proj(silu(gate) * up)``, and the output is the
    weighted sum. Routing ties go to the lower expert index. Weights keep the Hugging
    Face layout. Forward only.

    The weight arrays are stored as read-only float32 copies: to change a weight,
    assign a new array (``block.router = new``) rather than editing it in place.

    Not supported: the backward pass, router jitter noise, activations other than
    SiLU, the auxiliary load-balancing loss, and float64 input.

    Parameters
    ----------
    hidden_size : int
        Model dimension D.
    intermediate_size : int
        Expert hidden dimension F.
    num_experts : int, default 8
        Number of experts E.
    top_k : int, default 2
        Experts per token; ``1 <= top_k <= num_experts``, otherwise `ValueError`.
    norm_topk_prob : bool, default True
        Renormalize the kept routing probabilities.
    hidden_act : str, default 'silu'
        Expert activation; anything else raises `NotImplementedError`.
    random_state : int, default 0
        Seed of the N(0, 0.02**2) weight initialization.
    numeric_mode : {'identical', 'deterministic', 'fast'} or None, default None
        Numeric mode. None uses the library default.

    Attributes
    ----------
    router : ndarray, shape (E, D)
        Router weights.
    gate_up_proj : ndarray, shape (E, 2F, D)
        Gate and up projections of every expert.
    down_proj : ndarray, shape (E, D, F)
        Down projection of every expert.
    router_logits_ : ndarray, shape (n_tokens, E)
        Router logits of the last forward.
    selected_experts_ : ndarray of int64, shape (n_tokens, top_k)
        Experts chosen for each token by the last forward.
    routing_weights_ : ndarray, shape (n_tokens, top_k)
        Routing weights of the chosen experts.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> block = mojolearn.MoEBlock(hidden_size=64, intermediate_size=128, num_experts=8, top_k=2)
    >>> x = np.random.default_rng(0).random((4, 16, 64), dtype=np.float32)
    >>> y = block(x)
    """
    #: the board and callers put x on the sequence binding's device with
    #: `mojolearn._x_sequence_device.to_device` (lane neural-io-2)
    _seq_device_io = True

    def to_device(self, a):
        """Upload a float32 array once so `forward` keeps its input and outputs on the device.

        When this install has no device path for the block, the array itself is
        returned. Non-float32 input raises `TypeError` on that path.

        Parameters
        ----------
        a : ndarray of float32
            Array to upload.

        Returns
        -------
        device tensor or ndarray
        """
        from ._x_sequence_device import to_device
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        if not callable(getattr(b, "moe_forward_dev", None)):
            a = np.asarray(a)
            if a.dtype != np.float32:
                raise TypeError(f"mojolearn: to_device takes float32 (got {a.dtype}); convert it yourself")
            return np.ascontiguousarray(a)
        return to_device(a, self.numeric_mode)

    def _routing_out(self, i):
        r = self.__dict__.get("_routing")
        if r is None:
            raise AttributeError("MoEBlock: call forward first")
        v = r[i]
        if is_seq_tensor(v):   # resident (the device entry): read back on first access
            v = v.numpy()
            r[i] = v
        return v

    @property
    def router_logits_(self):
        """Router logits of the last forward, (n_tokens, num_experts)."""
        return self._routing_out(0)

    @property
    def selected_experts_(self):
        """Experts chosen for each token by the last forward, int64 (n_tokens, top_k)."""
        r = self.__dict__.get("_routing")
        if r is not None and r[3] is None:
            r[3] = self._routing_out(1).astype(np.int64)   # glue: the picks' dtype for the caller
        if r is None:
            raise AttributeError("MoEBlock: call forward first")
        return r[3]

    @property
    def routing_weights_(self):
        """Routing weights of the chosen experts, (n_tokens, top_k)."""
        return self._routing_out(2)

    def __setattr__(self, name, value):
        if name in _WEIGHTS:
            a = np.array(value, dtype=np.float32, copy=True, order="C")
            a.flags.writeable = False
            object.__setattr__(self, name, a)
            object.__setattr__(self, "_wver", self.__dict__.get("_wver", 0) + 1)
            return
        object.__setattr__(self, name, value)

    def _weights_handle(self, b):
        """The device copies' handle on binding `b`, put again after any
        weight assignment (the old copies freed)."""
        cur = self.__dict__.get("_wdev")
        if cur is not None and cur[0] is b and cur[1] == self._wver:
            return cur[2]
        self._free_weights()
        h = int(b.moe_weights_put([self.router.ctypes.data, self.gate_up_proj.ctypes.data,
                                   self.down_proj.ctypes.data], [self.E, self.D, self.F]))
        object.__setattr__(self, "_wdev", (b, self._wver, h))
        return h

    def _free_weights(self):
        cur = self.__dict__.get("_wdev")
        object.__setattr__(self, "_wdev", None)
        if cur is not None:
            try:
                cur[0].moe_weights_free(cur[2])
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass

    def __del__(self):
        self._free_weights()

    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2, norm_topk_prob=True,
                 hidden_act="silu", random_state=0, numeric_mode=None):
        if hidden_act != "silu":
            raise NotImplementedError("MoEBlock: hidden_act other than 'silu' is not implemented")
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("MoEBlock: 1 <= top_k <= num_experts")
        self.D, self.F, self.E, self.k = int(hidden_size), int(intermediate_size), int(num_experts), int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.numeric_mode = numeric_mode
        # N(0, 0.02**2) weights drawn in Mojo (InitStream.fill_normal, the
        # base binding's counter-based normal_init_f32; lane pyglue-numeric:
        # numpy's Generator drew them in Python), one stream in this order
        stream = InitStream(random_state)
        for name, shape in (("router", (self.E, self.D)), ("gate_up_proj", (self.E, 2 * self.F, self.D)),  # glue: three named expert weight shapes
                            ("down_proj", (self.E, self.D, self.F))):
            a = np.empty(shape, np.float32)
            stream.fill_normal(a.ctypes.data, a.size, 0.0, 0.02)
            setattr(self, name, a)   # a read-only copy (__setattr__)

    def load_state_dict(self, sd):
        """Load `router`, `gate_up_proj` and `down_proj` from a dict of arrays; returns self.

        Shapes must match the block's, otherwise `ValueError`.
        """
        for name, shape in (("router", (self.E, self.D)), ("gate_up_proj", (self.E, 2 * self.F, self.D)),  # glue: the four named MoE parameter tensors
                            ("down_proj", (self.E, self.D, self.F))):
            a = np.asarray(sd[name], dtype=np.float32)
            if a.shape != shape:
                raise ValueError(f"MoEBlock: {name} has shape {a.shape}, expected {shape}")
            setattr(self, name, a)   # a read-only copy (__setattr__)
        return self

    def _check_weights(self):
        # `want`, not `shape`: the caller's `shape` holds x's shape for the final reshape
        for name, want in (("router", (self.E, self.D)), ("gate_up_proj", (self.E, 2 * self.F, self.D)),  # glue: checks three named expert weight shapes
                           ("down_proj", (self.E, self.D, self.F))):
            if getattr(self, name).shape != want:
                raise ValueError(f"MoEBlock: {name} has shape {getattr(self, name).shape}, expected {want}")

    def _forward_dev(self, x):
        """`forward` on a resident x (lane neural-io-2): y and the routing
        outputs as resident tensors; None when x's binding has no device
        MoE entry (the caller reads x back and runs the host-array entry)."""
        b = x.b
        if not (callable(getattr(b, "moe_forward_dev", None)) and hasattr(b, "moe_weights_put")):
            return None
        if x.ndim < 1 or x.shape[-1] != self.D or x.size == 0:
            raise ValueError(f"MoEBlock: x must have shape (..., {self.D})")
        T = x.size // self.D
        self._check_weights()
        h = self._weights_handle(b)
        # every cell of the four outputs is stored by the route and the combine: no fill
        y = SequenceDeviceTensor._new(b, x.shape)
        logits = SequenceDeviceTensor._new(b, (T, self.E))
        sel = SequenceDeviceTensor._new(b, (T, self.k))
        w = SequenceDeviceTensor._new(b, (T, self.k))
        b.moe_forward_dev([x.h, y.h, logits.h, sel.h, w.h],
                          [T, self.D, self.F, self.E, self.k, int(self.norm_topk_prob), h])
        object.__setattr__(self, "_routing", [logits, sel, w, None])
        return y

    def forward(self, x):
        """Apply the block to `x` of shape (..., hidden_size); also callable as ``block(x)``.

        Sets `router_logits_`, `selected_experts_` and `routing_weights_`. float64 input
        raises `TypeError`. An input from `to_device` gives an output on the device.

        Parameters
        ----------
        x : ndarray of float32
            Token activations.

        Returns
        -------
        y : ndarray of float32
            Output with the shape of `x`.
        """
        if is_seq_tensor(x):
            y = self._forward_dev(x)
            if y is not None:
                return y
            x = x.numpy()
        x = np.asarray(x)
        if x.dtype == np.float64:
            raise TypeError("MoEBlock: float64 is refused; pass float32")
        shape = x.shape
        X = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, self.D)
        T = X.shape[0]
        # the binding downloads every cell of the four outputs (lane neural-io-2: np.empty,
        # not np.zeros: no host zero pass over 4 T (D + E + 2k) bytes; the words are the same)
        y = np.empty((T, self.D), np.float32)
        logits = np.empty((T, self.E), np.float32)
        sel = np.empty((T, self.k), np.float32)
        w = np.empty((T, self.k), np.float32)
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        ip = [T, self.D, self.F, self.E, self.k, int(self.norm_topk_prob)]
        if hasattr(b, "moe_weights_put"):
            self._check_weights()
            h = self._weights_handle(b)
            b.moe_forward([X.ctypes.data, X.ctypes.data, X.ctypes.data, X.ctypes.data, y.ctypes.data,
                           logits.ctypes.data, sel.ctypes.data, w.ctypes.data], ip + [h])
        else:
            r, gu, dn = (np.ascontiguousarray(a) for a in (self.router, self.gate_up_proj, self.down_proj))  # glue: three contiguous weight buffers for the binding
            b.moe_forward([X.ctypes.data, r.ctypes.data, gu.ctypes.data, dn.ctypes.data, y.ctypes.data,
                           logits.ctypes.data, sel.ctypes.data, w.ctypes.data], ip)
        object.__setattr__(self, "_routing", [logits, sel, w, None])
        return y.reshape(shape)

    __call__ = forward
