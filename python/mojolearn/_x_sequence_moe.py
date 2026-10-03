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
weight by assigning a new array, not in place."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_moe')

from . import _backend


_WEIGHTS = ("router", "gate_up_proj", "down_proj")


class MoEBlock:
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
        rng = np.random.default_rng(random_state)
        std = np.float32(0.02)
        self.router = (rng.standard_normal((self.E, self.D)) * std).astype(np.float32)
        self.gate_up_proj = (rng.standard_normal((self.E, 2 * self.F, self.D)) * std).astype(np.float32)
        self.down_proj = (rng.standard_normal((self.E, self.D, self.F)) * std).astype(np.float32)

    def load_state_dict(self, sd):
        for name, shape in (("router", (self.E, self.D)), ("gate_up_proj", (self.E, 2 * self.F, self.D)),
                            ("down_proj", (self.E, self.D, self.F))):
            a = np.asarray(sd[name], dtype=np.float32)
            if a.shape != shape:
                raise ValueError(f"MoEBlock: {name} has shape {a.shape}, expected {shape}")
            setattr(self, name, a)   # a read-only copy (__setattr__)
        return self

    def forward(self, x):
        """y with x's shape (..., hidden_size); also sets `router_logits_`,
        `selected_experts_` (int) and `routing_weights_`."""
        x = np.asarray(x)
        if x.dtype == np.float64:
            raise TypeError("MoEBlock: float64 is refused; pass float32")
        shape = x.shape
        X = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, self.D)
        T = X.shape[0]
        y = np.zeros((T, self.D), np.float32)
        logits = np.zeros((T, self.E), np.float32)
        sel = np.zeros((T, self.k), np.float32)
        w = np.zeros((T, self.k), np.float32)
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        ip = [T, self.D, self.F, self.E, self.k, int(self.norm_topk_prob)]
        if hasattr(b, "moe_weights_put"):
            # `want`, not `shape`: `shape` holds x's shape for the final reshape
            for name, want in (("router", (self.E, self.D)), ("gate_up_proj", (self.E, 2 * self.F, self.D)),
                               ("down_proj", (self.E, self.D, self.F))):
                if getattr(self, name).shape != want:
                    raise ValueError(f"MoEBlock: {name} has shape {getattr(self, name).shape}, expected {want}")
            h = self._weights_handle(b)
            b.moe_forward([X.ctypes.data, X.ctypes.data, X.ctypes.data, X.ctypes.data, y.ctypes.data,
                           logits.ctypes.data, sel.ctypes.data, w.ctypes.data], ip + [h])
        else:
            r, gu, dn = (np.ascontiguousarray(a) for a in (self.router, self.gate_up_proj, self.down_proj))
            b.moe_forward([X.ctypes.data, r.ctypes.data, gu.ctypes.data, dn.ctypes.data, y.ctypes.data,
                           logits.ctypes.data, sel.ctypes.data, w.ctypes.data], ip)
        self.router_logits_ = logits
        self.selected_experts_ = sel.astype(np.int64)
        self.routing_weights_ = w
        return y.reshape(shape)

    __call__ = forward
