# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""torch.optim-shaped optimizers of the sequence expansion lane, over float32
NumPy arrays updated in place: `opt = RMSprop([w1, w2], lr=1e-2);
opt.step([g1, g2])`. One launch over the packed flat buffer per step, the
element body `sequence/ops.mojo::op_opt` (PyTorch's update rules), the same on
the GPU binding and on the CPU host binding. `lr` is read fresh at every step,
so a schedule is `opt.lr = ...` between steps, or `opt.lr_schedule = <schedule>` (anything
with `lr_at(t)`, t one-based: `StepLR`, `ExponentialLR`, `OneCycleLR`), which
sets `lr` before every step.

Refused (sequence/NOT_IMPLEMENTED.tsv): parameter groups, maximize,
foreach/fused/capturable/differentiable, sparse gradients, float64."""
from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_optim')

import os

from . import _backend
from ._x_sequence_rnn import OPTIMIZERS, optimizer_arguments

#: lane gap-optimizers (2026-10-02): the state slots live on the device
#: binding's context across steps (`sequence/opt_resident.mojo`), so a step
#: moves only the parameters and gradients up and the parameters down, each
#: tensor straight into its place (no host concatenation); the same element
#: statements on the same values, the same bits. `MOJOLEARN_OPTIMIZER_RESIDENT=0`
#: keeps the per-call entries (the before arm of the A/B); the host binding
#: has no resident entries and keeps them too.
_RESIDENT_ENV = "MOJOLEARN_OPTIMIZER_RESIDENT"


def _resident_binding(b, entries):
    if os.environ.get(_RESIDENT_ENV, "1") == "0":
        return False
    return all(callable(getattr(b, n, None)) for n in entries)


_SEQ_RESIDENT = ("optimizer_resident_open", "optimizer_resident_close", "optimizer_resident_move",
                 "optimizer_resident_step")
_LAMB_RESIDENT = ("lamb_resident_open", "optimizer_resident_close", "optimizer_resident_move",
                  "lamb_resident_step")


class _ResidentState:
    """The state slots of a resident handle, mirrored on the host on access:
    `state` downloads the slots the handle keeps (once per stale read), and
    the next step uploads the host copies first (the caller may have
    written them, `load_state_dict` does)."""

    def _res_init(self, state):
        self._state = state
        self._res = None
        self._res_b = None
        self._res_used = 0
        self._host_fresh = True     # the host copies hold the newest values
        self._host_owned = True     # the next step uploads them first
        # the host copies are still the zeros (and initial values) the
        # constructor built: nobody has read or replaced them
        self._pristine = True

    @property
    def state(self):
        if self._res is not None and not self._host_fresh:
            for k, s in enumerate(self._state):
                self._res_b.optimizer_resident_move(self._res, k, s.ctypes.data, 0)
            self._host_fresh = True
        # a reader may write into the arrays it got: upload before the next step
        self._host_owned = True
        self._pristine = False
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._host_fresh = True
        self._host_owned = True
        self._pristine = False

    def _res_opened(self, b, ret, init_zero=True):
        """Keep the handle an open returned ([handle, used] or, from a
        FAST+Apple binding (OPT_ZERO_OPEN, the default there), [handle, used, 1]: the
        device slots are zero filled, so host copies that are still the
        constructor's zeros need no upload)."""
        self._res, self._res_b, self._res_used = int(ret[0]), b, int(ret[1])
        zeroed = len(ret) > 2 and int(ret[2]) == 1
        self._host_owned = not (zeroed and self._pristine and init_zero)

    def _res_upload(self):
        if self._host_owned:
            for k, s in enumerate(self._state):
                self._res_b.optimizer_resident_move(self._res, k, s.ctypes.data, 1)
            self._host_owned = False
        self._host_fresh = False

    @property
    def resident_(self):
        """True once the state lives on the device."""
        return self._res is not None

    def __del__(self):
        res, b = getattr(self, "_res", None), getattr(self, "_res_b", None)
        if res is not None and b is not None:
            try:
                b.optimizer_resident_close(res)
            except Exception:
                pass

    def _grad_list(self, grads):
        """The gradients as C-contiguous float32 arrays of the params' shapes
        (copied only when they are not already)."""
        if isinstance(grads, np.ndarray):
            grads = [grads]
        if len(grads) != len(self.params):
            raise ValueError(f"{type(self).__name__}: {len(grads)} grads, {len(self.params)} params")
        out = []
        for a, p in zip(grads, self.params):
            if np.shape(a) != p.shape:
                raise ValueError(f"{type(self).__name__}: a grads of shape {np.shape(a)} for a param {p.shape}")
            if np.asarray(a).dtype == np.float64:
                raise TypeError(f"{type(self).__name__}: float64 grads are refused")
            out.append(np.ascontiguousarray(a, dtype=np.float32))
        return out

    def _check_params(self):
        for k, p in enumerate(self.params):
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous \
                    or not p.flags.writeable:
                raise TypeError(f"{type(self).__name__}: params[{k}] must stay a writable C-contiguous "
                                "float32 NumPy array (it is updated in place)")
        if sum(p.size for p in self.params) != self.n_total:
            raise ValueError(f"{type(self).__name__}: the params were resized under the optimizer")


class _SeqOptimizer(_ResidentState):
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
        self._res_init([np.zeros(self.n_total, dtype=np.float32) for _ in range(3)])
        init = self._fp[5]
        if init:
            self._state[1][:] = np.float32(init)
        self.t = 0
        # the host scalars' running state (beta1^t, beta2^t, NAdam's mu
        # product) after step _sc_t, advanced by the binding each step, so a
        # step is O(1) instead of a replay of steps 1 .. t - 1
        self._sc = np.ones(3, dtype=np.float32)
        self._sc_t = 0

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

    def _flat_view(self, a):
        """`a` itself, flat, when it already IS the packed buffer (one
        C-contiguous float32 array of the param's shape): no copy in or out
        (lane sequence-cpu)."""
        if len(self.params) != 1:
            return None
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                return None
            a = a[0]
        if (isinstance(a, np.ndarray) and a.dtype == np.float32 and a.flags.c_contiguous
                and a.shape == self.params[0].shape):
            return a.reshape(-1)
        return None

    def _step_resident(self, b, grads):
        """The step with the state on the device (`_ResidentState`)."""
        self._check_params()
        gs = self._grad_list(grads)
        fp = [self.lr] + [float(v) for v in self._fp[:5]]
        if self._res is None:
            self._res_opened(b, b.optimizer_resident_open([self.n_total, self._kind, self._flags], fp),
                             init_zero=not self._fp[5])
        self._res_upload()
        # empty tensors hold nothing to move or update
        keep = [k for k, p in enumerate(self.params) if p.size > 0]
        ps = [self.params[k] for k in keep]
        b.optimizer_resident_step(self._res, [p.ctypes.data for p in ps] + [gs[k].ctypes.data for k in keep]
                                  + [self._sc.ctypes.data],
                                  [len(ps), self._kind, self._flags, self.t, self._sc_t]
                                  + [int(p.size) for p in ps], fp)
        del gs

    def step(self, grads):
        """One update of every param from `grads` (same shapes, same order)."""
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        if self._res is not None or _resident_binding(b, _SEQ_RESIDENT):
            self.t += 1
            if getattr(self, "lr_schedule", None) is not None:
                self.lr = float(self.lr_schedule.lr_at(self.t))
            if self._sc_t >= self.t:
                self._sc[:] = 1.0
                self._sc_t = 0
            try:
                self._step_resident(b, grads)
            except BaseException:
                self.t -= 1
                raise
            self._sc_t = self.t
            return self
        flat = self._flat_view(self.params)
        g = self._flat_view(grads)
        if flat is None or g is None or not flat.flags.writeable or np.shares_memory(flat, g):
            flat = None
            g = self._pack(grads, "grads")
        in_place = flat is not None
        if not in_place:
            flat = self._pack(self.params, "params")
        self.t += 1
        if getattr(self, "lr_schedule", None) is not None:
            self.lr = float(self.lr_schedule.lr_at(self.t))
        if self._sc_t >= self.t:
            self._sc[:] = 1.0
            self._sc_t = 0
        b.optimizer_step([flat.ctypes.data, g.ctypes.data] + [s.ctypes.data for s in self._state]
                         + [self._sc.ctypes.data],
                         [self.n_total, self._kind, self._flags, self.t, self._sc_t],
                         [self.lr] + [float(v) for v in self._fp[:5]])
        self._sc_t = self.t
        if not in_place:
            off = 0
            for p in self.params:
                p.ravel()[:] = flat[off:off + p.size]
                off += p.size
        return self

    def state_dict(self):
        return dict(t=self.t, lr=self.lr, state=[s.copy() for s in self.state], options=dict(getattr(self, "options", {})),
                    scalars=(self._sc_t, self._sc.copy()))

    def load_state_dict(self, sd):
        self.t = int(sd["t"])
        self.lr = float(sd["lr"])
        for dst, src in zip(self.state, sd["state"]):
            dst[:] = np.asarray(src, dtype=np.float32)
        # (reading `state` above marked the host copies for upload before
        # the next resident step)
        # without the scalars (an older state dict) the next step replays
        # them once from step 1: the same bits
        self._sc[:] = 1.0
        self._sc_t = 0
        sc = sd.get("scalars")
        if sc is not None and 0 <= int(sc[0]) <= self.t:
            self._sc_t = int(sc[0])
            self._sc[:] = np.asarray(sc[1], dtype=np.float32)
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


class Lion(_SeqOptimizer):
    """Lion (Chen et al. 2023, "Symbolic Discovery of Optimization
    Algorithms"; lion-pytorch's statement): p *= 1 - lr wd; p -= lr
    sign(b1 m + (1 - b1) g); m = b2 m + (1 - b2) g. Defaults are the paper's."""
    _NAME = "lion"

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, weight_decay=weight_decay), numeric_mode)

    @property
    def exp_avg(self):
        return self.state[0]


class Adamax(_SeqOptimizer):
    """`torch.optim.Adamax`: m.lerp_(g, 1 - b1); u = max(b2 u, |g| + eps);
    p -= lr / (1 - b1^t) m / u; coupled weight decay. Defaults are torch's."""
    _NAME = "adamax"

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, eps=eps, weight_decay=weight_decay), numeric_mode)


class NAdam(_SeqOptimizer):
    """`torch.optim.NAdam`: Nesterov-accelerated Adam with torch's momentum
    schedule mu_t = b1 (1 - 0.5 0.96^(t momentum_decay)), denominator
    sqrt(v / (1 - b2^t)) + eps, coupled or decoupled weight decay. Defaults
    are torch's."""
    _NAME = "nadam"

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3,
                 decoupled_weight_decay=False, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, eps=eps, weight_decay=weight_decay,
                                           momentum_decay=momentum_decay,
                                           decoupled_weight_decay=decoupled_weight_decay), numeric_mode)


class Adafactor:
    """`torch.optim.Adafactor` (PyTorch 2.5): relative step size, decoupled
    weight decay, a factored second moment (row and column means of g^2) for
    matrices and a full one for vectors, the update clipped by its RMS over
    d. Parameters are float32 NumPy arrays of 1 or 2 dimensions, updated in
    place; defaults are torch's. `sequence/adafactor.mojo`."""

    def __init__(self, params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, weight_decay=0.0,
                 maximize=False, numeric_mode=None):
        if maximize:
            raise NotImplementedError("Adafactor: maximize is not implemented")
        if isinstance(params, np.ndarray):
            params = [params]
        self.params = list(params)
        for k, p in enumerate(self.params):
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous:
                raise TypeError(f"Adafactor: params[{k}] must be a C-contiguous float32 NumPy array")
            if p.ndim not in (1, 2):
                raise NotImplementedError("Adafactor: tensors of more than 2 dimensions are not implemented")
        if beta2_decay > 0:
            raise ValueError("Adafactor: beta2_decay must be <= 0")
        if d < 1.0:
            raise ValueError(f"Adafactor: clipping threshold d must be >= 1, got {d}")
        if lr < 0 or weight_decay < 0:
            raise ValueError("Adafactor: lr and weight_decay must be >= 0")
        self.lr, self.beta2_decay, self.d, self.weight_decay = float(lr), float(beta2_decay), float(d), float(weight_decay)
        eps1, eps2 = eps
        self.eps = (float(np.finfo(np.float32).eps) if eps1 is None else float(eps1), float(eps2))
        self.numeric_mode = numeric_mode
        self.state = []
        for p in self.params:
            if p.ndim == 2:
                self.state.append(dict(row_var=np.zeros(p.shape[0], np.float32),
                                       col_var=np.zeros(p.shape[1], np.float32)))
            else:
                self.state.append(dict(variance=np.zeros(p.shape[0], np.float32)))
        self.t = 0

    def step(self, grads):
        if isinstance(grads, np.ndarray):
            grads = [grads]
        if len(grads) != len(self.params):
            raise ValueError(f"Adafactor: {len(grads)} grads, {len(self.params)} params")
        self.t += 1
        if getattr(self, "lr_schedule", None) is not None:
            self.lr = float(self.lr_schedule.lr_at(self.t))
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        for p, g, st in zip(self.params, grads, self.state):
            g = np.asarray(g)
            if g.shape != p.shape or g.dtype == np.float64:
                raise ValueError("Adafactor: a grad must be float32 of its param's shape")
            g = np.ascontiguousarray(g, dtype=np.float32)
            if p.ndim == 2:
                s1, s2 = st["row_var"], st["col_var"]
                R, C = p.shape
            else:
                s1 = s2 = st["variance"]
                R, C = p.shape[0], 0
            b.adafactor_step([p.ctypes.data, g.ctypes.data, s1.ctypes.data, s2.ctypes.data], [R, C, self.t],
                             [self.lr, self.beta2_decay, self.eps[0], self.eps[1], self.d, self.weight_decay])
        return self


class LAMB(_SeqOptimizer):
    """LAMB (You et al. 2019, "Large Batch Optimization for Deep Learning"),
    timm's `Lamb` statement: optional global gradient-norm clip
    (max_grad_norm), Adam moments with bias correction, update
    m_hat / (sqrt(v_hat) + eps) + weight_decay p, scaled per tensor by the
    trust ratio ||p|| / ||update|| (1 when either is 0; at most 1 with
    trust_clip) when weight_decay != 0 or always_adapt. Defaults are timm's."""

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, grad_averaging=True, max_grad_norm=1.0, trust_clip=False,
                 always_adapt=False, numeric_mode=None):
        if isinstance(params, np.ndarray):
            params = [params]
        self.params = list(params)
        for k, p in enumerate(self.params):
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous or p.size == 0:
                raise TypeError(f"LAMB: params[{k}] must be a non-empty C-contiguous float32 NumPy array")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("LAMB: betas must lie in [0, 1)")
        self.lr, self.betas, self.eps, self.weight_decay = float(lr), (float(b1), float(b2)), float(eps), float(weight_decay)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.flags = (int(bool(trust_clip)) | 2 * int(bool(always_adapt)) | 4 * int(bool(grad_averaging))
                      | 8 * int(bool(bias_correction)) | 16 * int(self.max_grad_norm is not None))
        self.numeric_mode = numeric_mode
        self.n_total = int(sum(p.size for p in self.params))
        self._res_init([np.zeros(self.n_total, dtype=np.float32) for _ in range(2)])
        self.t = 0
        self._sc = np.ones(2, dtype=np.float32)    # beta1^_sc_t, beta2^_sc_t
        self._sc_t = 0

    def step(self, grads):
        b = _backend.binding("_mojolearn_x_sequence", self.numeric_mode)
        resident = self._res is not None or _resident_binding(b, _LAMB_RESIDENT)
        if resident:
            self._check_params()
            gs = self._grad_list(grads)
        else:
            g = self._pack(grads, "grads")
            flat = self._pack(self.params, "params")
        self.t += 1
        if getattr(self, "lr_schedule", None) is not None:
            self.lr = float(self.lr_schedule.lr_at(self.t))
        offs = [0]
        for p in self.params:
            offs.append(offs[-1] + p.size)
        if self._sc_t >= self.t:
            self._sc[:] = 1.0
            self._sc_t = 0
        fp = [self.lr, self.betas[0], self.betas[1], self.eps, self.weight_decay,
              self.max_grad_norm if self.max_grad_norm is not None else 1.0, float(self._sc_t)]
        if resident:
            # the moments and the tensor table on the device (`_ResidentState`,
            # `sequence/opt_resident.mojo::lamb_resident_step`): each param and
            # grad straight into its place, the params straight back
            try:
                if self._res is None:
                    self._res_opened(b, b.lamb_resident_open([len(self.params)] + offs))
                self._res_upload()
                b.lamb_resident_step(self._res, [p.ctypes.data for p in self.params] + [x.ctypes.data for x in gs]
                                     + [self._sc.ctypes.data], [len(self.params), self.t, self.flags], fp)
            except BaseException:
                self.t -= 1
                raise
            del gs
        else:
            b.lamb_step(
                [flat.ctypes.data, g.ctypes.data, self._state[0].ctypes.data, self._state[1].ctypes.data,
                 self._sc.ctypes.data],
                [len(self.params), self.t, self.flags] + offs, fp)
        if self.flags & 8:
            self._sc_t = self.t
        if not resident:
            off = 0
            for p in self.params:
                p.ravel()[:] = flat[off:off + p.size]
                off += p.size
        return self


assert set(OPTIMIZERS) >= {"rmsprop", "adagrad"}
