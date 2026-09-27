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
float64."""
import numpy as np

from . import _backend


class MoEBlock:
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
            a = np.ascontiguousarray(np.asarray(sd[name], dtype=np.float32))
            if a.shape != shape:
                raise ValueError(f"MoEBlock: {name} has shape {a.shape}, expected {shape}")
            setattr(self, name, a)
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
        r, gu, dn = (np.ascontiguousarray(a) for a in (self.router, self.gate_up_proj, self.down_proj))
        _backend.binding("_mojolearn_x_sequence", self.numeric_mode).moe_forward(
            [X.ctypes.data, r.ctypes.data, gu.ctypes.data, dn.ctypes.data, y.ctypes.data, logits.ctypes.data,
             sel.ctypes.data, w.ctypes.data], [T, self.D, self.F, self.E, self.k, int(self.norm_topk_prob)])
        self.router_logits_ = logits
        self.selected_experts_ = sel.astype(np.int64)
        self.routing_weights_ = w
        return y.reshape(shape)

    __call__ = forward
