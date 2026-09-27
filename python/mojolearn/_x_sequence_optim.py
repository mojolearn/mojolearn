# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""torch.optim-shaped optimizers of the sequence expansion lane, over float32
NumPy arrays updated in place: `opt = RMSprop([w1, w2], lr=1e-2);
opt.step([g1, g2])`. One launch over the packed flat buffer per step, the
element body `sequence/ops.mojo::op_opt` (PyTorch's update rules), the same on
the GPU binding and on the CPU host binding. `lr` is read fresh at every step,
so a schedule is `opt.lr = ...` between steps (or `lr_scheduler=`).

Refused (sequence/NOT_IMPLEMENTED.tsv): parameter groups, maximize,
foreach/fused/capturable/differentiable, sparse gradients, float64."""
import numpy as np

from . import _backend
from ._x_sequence_rnn import OPTIMIZERS, optimizer_arguments


class _SeqOptimizer:
    _NAME = None

    def __init__(self, params, lr, options, numeric_mode=None):
        if isinstance(params, np.ndarray):
            params = [params]
        self.params = list(params)
        for k, p in enumerate(self.params):
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous:
                raise TypeError(f"{type(self).__name__}: params[{k}] must be a C-contiguous float32 "
                                "NumPy array (it is updated in place)")
        self.lr = float(lr)
        self.options = dict(options)
        self.numeric_mode = numeric_mode
        self._kind, self._flags, self._fp = optimizer_arguments(self._NAME, self.options)
        self.n_total = int(sum(p.size for p in self.params))
        if self.n_total < 1:
            raise ValueError(f"{type(self).__name__}: no parameters")
        self.state = [np.zeros(self.n_total, dtype=np.float32) for _ in range(3)]
        init = self._fp[5]
        if init:
            self.state[1][:] = np.float32(init)
        self.t = 0

    def _pack(self, arrays, what):
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        if len(arrays) != len(self.params):
            raise ValueError(f"{type(self).__name__}: {len(arrays)} {what}, {len(self.params)} params")
        for a, p in zip(arrays, self.params):
            if np.shape(a) != p.shape:
                raise ValueError(f"{type(self).__name__}: a {what} of shape {np.shape(a)} for a param {p.shape}")
            if np.asarray(a).dtype == np.float64:
                raise TypeError(f"{type(self).__name__}: float64 {what} are refused")
        return np.ascontiguousarray(np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays]))

    def step(self, grads):
        """One update of every param from `grads` (same shapes, same order)."""
        g = self._pack(grads, "grads")
        flat = self._pack(self.params, "params")
        self.t += 1
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        b.optimizer_step([flat.ctypes.data, g.ctypes.data] + [s.ctypes.data for s in self.state],
                         [self.n_total, self._kind, self._flags, self.t],
                         [self.lr] + [float(v) for v in self._fp[:5]])
        off = 0
        for p in self.params:
            p.ravel()[:] = flat[off:off + p.size]
            off += p.size
        return self

    def state_dict(self):
        return dict(t=self.t, lr=self.lr, state=[s.copy() for s in self.state], options=dict(self.options))

    def load_state_dict(self, sd):
        self.t = int(sd["t"])
        self.lr = float(sd["lr"])
        for dst, src in zip(self.state, sd["state"]):
            dst[:] = np.asarray(src, dtype=np.float32)
        return self


class RMSprop(_SeqOptimizer):
    """`torch.optim.RMSprop`: v = alpha v + (1 - alpha) g^2; centered
    subtracts the squared running mean of g; momentum keeps a buffer of
    g / (sqrt(v) + eps). Defaults are torch's."""
    _NAME = "rmsprop"

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
                 centered=False, numeric_mode=None):
        super().__init__(params, lr, dict(alpha=alpha, eps=eps, weight_decay=weight_decay,
                                           momentum=momentum, centered=centered), numeric_mode)

    @property
    def square_avg(self):
        return self.state[1]


class Adagrad(_SeqOptimizer):
    """`torch.optim.Adagrad`: sum += g^2; p -= clr g / (sqrt(sum) + eps),
    clr = lr / (1 + (t - 1) lr_decay). Defaults are torch's."""
    _NAME = "adagrad"

    def __init__(self, params, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                 eps=1e-10, numeric_mode=None):
        super().__init__(params, lr, dict(lr_decay=lr_decay, weight_decay=weight_decay,
                                           initial_accumulator_value=initial_accumulator_value, eps=eps),
                         numeric_mode)

    @property
    def sum(self):
        return self.state[1]


assert set(OPTIMIZERS) >= {"rmsprop", "adagrad"}
