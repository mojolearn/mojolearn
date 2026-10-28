# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MoEBlock: the sparse mixture-of-experts feed-forward block of HF
transformers' Mixtral (`MixtralSparseMoeBlock`: `MixtralTopKRouter` +
`MixtralExperts`), forward on the GPU (`sequence/moe.mojo`): top-k routing
over the softmax of the router logits with the probabilities renormalised,
each chosen expert down_proj(silu(gate) * up), the weighted sum. The
weights keep HF's layout: `router` (E, D), `gate_up_proj` (E, 2F, D),
`down_proj` (E, D, F). Routing ties go to the lower expert index.

Refused (sequence/NOT_IMPLEMENTED.tsv): router jitter noise, activations
other than SiLU, the auxiliary load-balancing loss, float64.

THE BACKWARD (lane ag-moe-backward, 2026-10-19): `backward(x, dy)` runs the
forward and the backward on the tape in one binding call
(`moe_backward`, autograd/route_ops.mojo): the gradients of x and the three
weights, every number from Mojo. Listed in docs-site/api_exclude.txt until
its cross-vendor check (sequence/checks/moe_backward_check.mojo) is recorded.

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
import os as _os
from ._optional_numpy import require_numpy
_numpy = require_numpy("_x_sequence_moe")  # glue: preserve the optional dependency at import
from . import _x_sequence_device as _sequence_device  # glue: connect native device storage before MoE
from ._training_impl import _OPT_NATIVE as _SEQ_NATIVE, _OPT_CONTEXT as _SEQ_CONTEXT


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
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

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
        return _SEQ_NATIVE("seqmoe_to_device", self, (a,), None, _SEQ_CONTEXT)

    def _routing_out(self, i):
        return _SEQ_NATIVE("seqmoe_routing", self, (i,), None, _SEQ_CONTEXT)

    @property
    def router_logits_(self):
        """Router logits of the last forward, (n_tokens, num_experts)."""
        return _SEQ_NATIVE("seqmoe_routing", self, (0,), None, _SEQ_CONTEXT)

    @property
    def selected_experts_(self):
        """Experts chosen for each token by the last forward, int64 (n_tokens, top_k)."""
        return _SEQ_NATIVE("seqmoe_selected", self, (), None, _SEQ_CONTEXT)

    @property
    def routing_weights_(self):
        """Routing weights of the chosen experts, (n_tokens, top_k)."""
        return _SEQ_NATIVE("seqmoe_routing", self, (2,), None, _SEQ_CONTEXT)

    def __setattr__(self, name, value):
        return _SEQ_NATIVE("seqmoe_setattr", self, (name, value), None, _SEQ_CONTEXT)

    def _weights_handle(self, b):
        """The device copies' handle on binding `b`, put again after any
        weight assignment (the old copies freed)."""
        return _SEQ_NATIVE("seqmoe_weights", self, (b,), None, _SEQ_CONTEXT)

    def _free_weights(self):
        return _SEQ_NATIVE("seqmoe_release", self, (), None, _SEQ_CONTEXT)

    def __del__(self):
        return _SEQ_NATIVE("seqmoe_release", self, (), None, _SEQ_CONTEXT)

    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2, norm_topk_prob=True,
                 hidden_act="silu", random_state=0, numeric_mode=None):
        return _SEQ_NATIVE("seqmoe_initialize", self,
                           (hidden_size, intermediate_size, num_experts, top_k, norm_topk_prob,
                            hidden_act, random_state), numeric_mode, _SEQ_CONTEXT)

    def load_state_dict(self, sd):
        """Load `router`, `gate_up_proj` and `down_proj` from a dict of arrays; returns self.

        Shapes must match the block's, otherwise `ValueError`.
        """
        return _SEQ_NATIVE("seqmoe_load", self, (sd,), None, _SEQ_CONTEXT)

    def _check_weights(self):
        # `want`, not `shape`: the caller's `shape` holds x's shape for the final reshape
        return _SEQ_NATIVE("seqmoe_check_weights", self, (), None, _SEQ_CONTEXT)

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
        return _SEQ_NATIVE("seqmoe_forward", self, (x,), None, _SEQ_CONTEXT)

    __call__ = forward

    def backward(self, x, dy):
        """Gradients of the block for input `x` and output gradient `dy`.

        Runs the forward on `x` and returns the gradient with respect to `x`; the
        weight gradients are stored as `router_grad_`, `gate_up_proj_grad_` and
        `down_proj_grad_`, and the routing attributes are set as by `forward`.

        Parameters
        ----------
        x : ndarray of float32
            Token activations, shape (..., hidden_size).
        dy : ndarray of float32
            Gradient of the loss with respect to the block output, shape of `x`.

        Returns
        -------
        dx : ndarray of float32
            Gradient with respect to `x`, shape of `x`.
        """
        return _SEQ_NATIVE("seqmoe_backward", self, (x, dy), None, _SEQ_CONTEXT)


# glue: connect the class storage layout and native OS entropy entry once.
_moe_mro = MoEBlock.__mro__
_moe_rows = tuple(vars(base) for base in _moe_mro)  # glue: fixed native owner ancestry
_moe_hooks = tuple((base, name, row.get(name)) for base, row in zip(_moe_mro, _moe_rows)
                   for name in ("__getattribute__", "__getattr__", "__setattr__", "__dict__"))  # glue: storage hook identities
_SEQ_CONTEXT["seqmoe_layout"] = (MoEBlock, _moe_mro, _moe_hooks)
_SEQ_CONTEXT["seqmoe_urandom"] = _os.urandom
