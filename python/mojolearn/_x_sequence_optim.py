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
from ._x_sequence_device import SequenceDeviceTensor, is_seq_tensor, resident_binding

# lane neural-optimizers (2026-10-21): every public method below is ONE native
# call (sequence/opt_public.mojo, sequence/opt_public_device.mojo): Python
# checks the arguments and passes addresses, sizes read off the arrays and
# the hyperparameter words; the packing, the size sums, the step counter,
# the host scalars' reset, the resident-or-per-call route
# (MOJOLEARN_OPTIMIZER_RESIDENT, read natively), the resident open, upload
# and download and the step itself run in Mojo. The native calls keep the
# optimizer's scalar state in an int64 word buffer (`_words`, layout below)
# and the resident handles in `_handles`; the properties read them back. The
# same launches on the same values: the same bits.
_W_T, _W_SC_T, _W_HANDLE, _W_USED, _W_FRESH, _W_OWNED, _W_PRISTINE, _W_N, _W_FLAGS = range(9)
_N_WORDS = 9
_MODE_ELEMENTWISE, _MODE_LAMB, _MODE_ADAFACTOR = 0, 1, 2

# The binding entries a device-parameter optimizer needs
# (`sequence/opt_resident.mojo`); `_device_params` refuses device params on
# a binding without them.
_SEQ_DEV = ("optimizer_resident_open", "optimizer_resident_close", "optimizer_resident_move",
            "optimizer_resident_step", "optimizer_resident_step_dev")
_AF_DEV = ("adafactor_resident_open", "optimizer_resident_close", "optimizer_resident_move",
           "adafactor_resident_step", "adafactor_resident_step_dev")
_LAMB_DEV = ("lamb_resident_open", "optimizer_resident_close", "optimizer_resident_move",
             "lamb_resident_step", "lamb_resident_step_dev")
_RESIDENT_ENV = "MOJOLEARN_OPTIMIZER_RESIDENT"


def _device_params(params, who, entries):
    """None when no param is a `SequenceDeviceTensor`; their binding when
    every one is (on one binding that has `entries`); refused otherwise."""
    dev = [is_seq_tensor(p) for p in params]  # glue: per parameter tensor, not elements
    if not any(dev):
        return None
    if not all(dev):
        raise TypeError(f"{who}: params are all device tensors or all NumPy arrays, not a mix")
    b = params[0].b
    if any(p.b is not b for p in params):  # glue: per parameter tensor, not elements
        raise TypeError(f"{who}: every device param must live on one binding")
    if not (resident_binding(b) and all(callable(getattr(b, n, None)) for n in entries)):  # glue: the named binding entry points
        raise RuntimeError(f"{who}: this binding has no device-parameter step; pass NumPy arrays")
    if os.environ.get(_RESIDENT_ENV, "1") == "0":
        raise RuntimeError(f"{who}: device params need the resident state ({_RESIDENT_ENV}=0 is set)")
    return b


def _word(k, kind=int):
    """A property over state word k (written by the native calls)."""
    def get(self):
        return kind(int(self._words[k]))

    def put(self, value):
        self._words[k] = int(value)
    return property(get, put)


class _ResidentState:
    """The state of a sequence optimizer: host copies of the state slots, the
    native state words and, once the state lives on the device, the resident
    handles. A state read downloads the device slots once per stale read;
    the next step uploads the host copies first (the reader may have written
    them, `load_state_dict` does)."""

    t = _word(_W_T)
    _sc_t = _word(_W_SC_T)
    _res_used = _word(_W_USED)
    _host_fresh = _word(_W_FRESH, bool)
    _host_owned = _word(_W_OWNED, bool)
    _pristine = _word(_W_PRISTINE, bool)

    def _native_init(self, mode, sizes, n_handles=1, extra=()):
        """The state words and handles, and the native constructor half;
        returns the parameter count."""
        self._words = np.zeros(_N_WORDS, dtype=np.int64)
        self._handles = np.full(n_handles, -1, dtype=np.int64)
        self._res_b = None
        return int(self._binding().seq_optimizer_init(self._words.ctypes.data,
                                                      [mode, len(sizes)] + list(sizes) + list(extra)))

    def _binding(self):
        if getattr(self, "_dev_b", None) is not None:
            return self._dev_b
        return _backend.binding("_mojolearn_x_sequence", self.numeric_mode)

    def _state_binding(self):
        """The binding that holds the resident state, else this optimizer's."""
        return self._res_b if self._res_b is not None else self._binding()

    def _slot_addrs(self):
        """Three slot addresses per handle (0 for a slot not kept)."""
        a = [s.ctypes.data for s in self._state]  # glue: the optimizer state slots, at most three
        return a + [0] * (3 - len(a))

    @property
    def _res(self):
        """The resident handle, or None while the state is on the host."""
        return int(self._handles[0]) if int(self._words[_W_HANDLE]) >= 0 else None

    @property
    def state(self):
        self._state_binding().seq_optimizer_state(self._words.ctypes.data, self._handles.ctypes.data,
                                                  self._slot_addrs())
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._host_fresh = True
        self._host_owned = True
        self._pristine = False

    @property
    def resident_(self):
        """True once the state lives on the device."""
        return self._res is not None

    def __del__(self):
        b = getattr(self, "_res_b", None)
        if b is not None:
            try:
                b.seq_optimizer_close(self._words.ctypes.data, self._handles.ctypes.data, len(self._handles))
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
        for a, p in zip(grads, self.params):  # glue: per parameter tensor, not elements
            if np.shape(a) != p.shape:
                raise ValueError(f"{type(self).__name__}: a grads of shape {np.shape(a)} for a param {p.shape}")
            if np.asarray(a).dtype == np.float64:
                raise TypeError(f"{type(self).__name__}: float64 grads are refused")
            out.append(np.ascontiguousarray(a, dtype=np.float32))
        return out

    def _check_params(self):
        for k, p in enumerate(self.params):  # glue: per parameter tensor, not elements
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous \
                    or not p.flags.writeable:
                raise TypeError(f"{type(self).__name__}: params[{k}] must stay a writable C-contiguous "
                                "float32 NumPy array (it is updated in place)")

    def _operands(self, b, grads):
        """A step's tensor operands: (addresses or handles of the params then
        the grads, the grad kinds (0 a device handle, 1 a host address), the
        sizes, the arrays to keep alive across the call, the device flag)."""
        who = type(self).__name__
        sizes = [int(p.size) for p in self.params]  # glue: sizes per parameter tensor, not elements
        if self._dev_b is None:
            self._check_params()
            gs = self._grad_list(grads)
            return ([p.ctypes.data for p in self.params] + [g.ctypes.data for g in gs],  # glue: addresses per parameter tensor
                    [0] * len(gs), sizes, gs, 0)
        if isinstance(grads, (np.ndarray, SequenceDeviceTensor)):
            grads = [grads]
        if len(grads) != len(self.params):
            raise ValueError(f"{who}: {len(grads)} grads, {len(self.params)} params")
        hs, kinds, keep = [], [], []
        for g, p in zip(grads, self.params):  # glue: per parameter tensor, not elements
            if is_seq_tensor(g):
                if g.b is not b or tuple(g.shape) != tuple(p.shape):
                    raise ValueError(f"{who}: a device grad of shape {g.shape} for a param {p.shape} "
                                     "(or on another binding)")
                hs.append(g.h)
                kinds.append(0)
                continue
            a = np.asarray(g)
            if a.shape != tuple(p.shape):
                raise ValueError(f"{who}: a grads of shape {a.shape} for a param {p.shape}")
            if a.dtype != np.float32:
                raise TypeError(f"{who}: grads must be float32 (got {a.dtype})")
            a = np.ascontiguousarray(a)
            keep.append(a)
            hs.append(a.ctypes.data)
            kinds.append(1)
        return [p.h for p in self.params] + hs, kinds, sizes, keep, 1  # glue: handles per parameter tensor

    def _lr_for_step(self):
        """The learning rate of the coming step: the schedule's, when set."""
        sched = getattr(self, "lr_schedule", None)
        if sched is not None:
            self.lr = float(sched.lr_at(self.t + 1))
        return self.lr


class _SeqOptimizer(_ResidentState):
    _NAME = None

    def __init__(self, params, lr, options, numeric_mode=None):
        if isinstance(params, np.ndarray):
            params = [params]
        self.params = list(params)
        self._dev_b = _device_params(self.params, type(self).__name__, _SEQ_DEV)
        for k, p in enumerate(self.params):  # glue: per parameter tensor, not elements
            if self._dev_b is None and (not isinstance(p, np.ndarray) or p.dtype != np.float32
                                        or not p.flags.c_contiguous):
                raise TypeError(f"{type(self).__name__}: params[{k}] must be a C-contiguous float32 "
                                "NumPy array or a SequenceDeviceTensor (it is updated in place)")
        self.lr = float(lr)
        self.options = dict(options)
        self.numeric_mode = numeric_mode
        self._kind, self._flags, self._fp = optimizer_arguments(self._NAME, self.options)
        self.n_total = self._native_init(_MODE_ELEMENTWISE, [int(p.size) for p in self.params])  # glue: sizes per parameter tensor
        if self.n_total < 1:
            raise ValueError(f"{type(self).__name__}: no parameters")
        # the three state slots; slot 1 starts at Adagrad's initial accumulator
        self._state = [np.zeros(self.n_total, dtype=np.float32),
                       np.full(self.n_total, self._fp[5], dtype=np.float32),
                       np.zeros(self.n_total, dtype=np.float32)]
        # the host scalars' running state (beta1^t, beta2^t, NAdam's mu
        # product) after step _sc_t, advanced natively each step, so a step
        # is O(1) instead of a replay of steps 1 .. t - 1
        self._sc = np.ones(3, dtype=np.float32)

    def step(self, grads):
        """One update of every param from `grads` (same shapes, same order)."""
        b = self._binding()
        addrs, kinds, sizes, keep, dev = self._operands(b, grads)
        lr = self._lr_for_step()
        if self._res_b is None:
            self._res_b = b
        b.seq_optimizer_step(self._words.ctypes.data, self._handles.ctypes.data,
                             addrs + [s.ctypes.data for s in self._state] + [self._sc.ctypes.data],  # glue: buffer addresses of the state slots
                             [len(self.params), dev] + kinds + sizes + [self._kind, self._flags],
                             [lr] + [float(v) for v in self._fp])  # glue: the optimizer float arguments
        del keep
        return self

    def state_dict(self):
        out = [np.empty_like(s) for s in self._state]  # glue: one output container per state slot
        sc = np.empty_like(self._sc)
        self._state_binding().seq_optimizer_export(
            self._words.ctypes.data, self._handles.ctypes.data,
            [s.ctypes.data for s in self._state] + [o.ctypes.data for o in out]  # glue: state slot addresses
            + [self._sc.ctypes.data, sc.ctypes.data],
            [len(self._state), self.n_total, len(self._sc), len(out), self.t, 0, 0])
        return dict(t=self.t, lr=self.lr, state=out, options=dict(getattr(self, "options", {})),
                    scalars=(self._sc_t, sc))

    def load_state_dict(self, sd):
        t = int(sd["t"])
        self.lr = float(sd["lr"])
        srcs = [np.ascontiguousarray(np.broadcast_to(np.asarray(src, dtype=np.float32), dst.shape))
                for dst, src in zip(self._state, sd["state"])]  # glue: converts each saved state slot
        sc = sd.get("scalars")
        # without the scalars (an older state dict) the next step replays
        # them once from step 1: the same bits
        sc_t = 0 if sc is None else int(sc[0])
        sc_src = self._sc if sc is None else np.ascontiguousarray(
            np.broadcast_to(np.asarray(sc[1], dtype=np.float32), self._sc.shape))
        self._state_binding().seq_optimizer_import(
            self._words.ctypes.data, self._handles.ctypes.data,
            [s.ctypes.data for s in self._state] + [s.ctypes.data for s in srcs]  # glue: state slot addresses
            + [self._sc.ctypes.data, sc_src.ctypes.data],
            [len(self._state), self.n_total, len(self._sc), len(srcs), t, 0 if sc is None else 1, sc_t])
        return self


class RMSprop(_SeqOptimizer):
    """RMSprop optimizer, as `torch.optim.RMSprop`.

    Keeps a running average of squared gradients, ``v = alpha * v + (1 - alpha) * g**2``,
    and divides the gradient by ``sqrt(v) + eps``. With ``centered=True`` the squared
    running mean of the gradient is subtracted from `v`; with ``momentum > 0`` a
    momentum buffer accumulates the scaled gradient. Defaults are torch's.

    Parameters
    ----------
    params : ndarray or list of ndarray
        Parameters to update in place.
    lr : float, default 1e-2
        Learning rate.
    alpha : float, default 0.99
        Smoothing constant of the squared-gradient average.
    eps : float, default 1e-8
        Term added to the denominator.
    weight_decay : float, default 0.0
        L2 penalty added to the gradient.
    momentum : float, default 0.0
        Momentum factor.
    centered : bool, default False
        Normalize by the estimated gradient variance.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode of the update. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    t : int
        Number of steps taken.
    square_avg : ndarray
        Running average of squared gradients, flat over all parameters.

    Notes
    -----
    `params` must be C-contiguous float32 NumPy arrays, updated in place; anything
    else raises `TypeError`. `step(grads)`
    takes one gradient per parameter, in the same order and shapes; float64 gradients
    raise `TypeError`. `lr` is read at every step, so you can change ``opt.lr`` between
    steps, or set ``opt.lr_schedule`` to a schedule (`StepLR`, `ExponentialLR`,
    `OneCycleLR`, or any object with a one-based `lr_at(t)`), which sets `lr` before
    each step. `state_dict()` and `load_state_dict()` save and restore the step count,
    learning rate and state buffers.

    Not supported: parameter groups, `maximize`, `foreach`/`fused`/`capturable`/
    `differentiable`, sparse gradients and float64 parameters.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros((64, 8), dtype=np.float32)
    >>> opt = mojolearn.RMSprop([w], lr=1e-2)
    >>> opt = opt.step([np.ones_like(w)])
    """
    _NAME = "rmsprop"

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
                 centered=False, numeric_mode=None):
        super().__init__(params, lr, dict(alpha=alpha, eps=eps, weight_decay=weight_decay,
                                           momentum=momentum, centered=centered), numeric_mode)

    @property
    def square_avg(self):
        """Running average of squared gradients, flat over all parameters."""
        return self.state[1]


class Adagrad(_SeqOptimizer):
    """Adagrad optimizer, as `torch.optim.Adagrad`.

    Accumulates squared gradients, ``sum += g**2``, and updates
    ``p -= clr * g / (sqrt(sum) + eps)`` with ``clr = lr / (1 + (t - 1) * lr_decay)``.
    Defaults are torch's.

    Parameters
    ----------
    params : ndarray or list of ndarray
        Parameters to update in place.
    lr : float, default 1e-2
        Learning rate.
    lr_decay : float, default 0.0
        Learning-rate decay per step.
    weight_decay : float, default 0.0
        L2 penalty added to the gradient.
    initial_accumulator_value : float, default 0.0
        Starting value of the squared-gradient sum.
    eps : float, default 1e-10
        Term added to the denominator.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode of the update. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    t : int
        Number of steps taken.
    sum : ndarray
        Accumulated squared gradients, flat over all parameters.

    Notes
    -----
    `params` must be C-contiguous float32 NumPy arrays, updated in place; anything
    else raises `TypeError`. `step(grads)`
    takes one gradient per parameter, in the same order and shapes; float64 gradients
    raise `TypeError`. `lr` is read at every step, so you can change ``opt.lr`` between
    steps, or set ``opt.lr_schedule`` to a schedule (`StepLR`, `ExponentialLR`,
    `OneCycleLR`, or any object with a one-based `lr_at(t)`), which sets `lr` before
    each step. `state_dict()` and `load_state_dict()` save and restore the step count,
    learning rate and state buffers.

    Not supported: parameter groups, `maximize`, `foreach`/`fused`/`capturable`/
    `differentiable`, sparse gradients and float64 parameters.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros(128, dtype=np.float32)
    >>> opt = mojolearn.Adagrad(w, lr=0.1).step(np.ones_like(w))
    """
    _NAME = "adagrad"

    def __init__(self, params, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                 eps=1e-10, numeric_mode=None):
        super().__init__(params, lr, dict(lr_decay=lr_decay, weight_decay=weight_decay,
                                           initial_accumulator_value=initial_accumulator_value, eps=eps),
                         numeric_mode)

    @property
    def sum(self):
        """Accumulated squared gradients, flat over all parameters."""
        return self.state[1]


class Lion(_SeqOptimizer):
    """Lion optimizer (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms").

    Follows the lion-pytorch statement of the update: ``p *= 1 - lr * weight_decay``;
    ``p -= lr * sign(beta1 * m + (1 - beta1) * g)``; ``m = beta2 * m + (1 - beta2) * g``.
    Defaults are the paper's.

    Parameters
    ----------
    params : ndarray or list of ndarray
        Parameters to update in place.
    lr : float, default 1e-4
        Learning rate.
    betas : tuple of float, default (0.9, 0.99)
        Interpolation factor of the update direction and decay of the momentum.
    weight_decay : float, default 0.0
        Decoupled weight decay.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode of the update. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    t : int
        Number of steps taken.
    exp_avg : ndarray
        Momentum buffer, flat over all parameters.

    Notes
    -----
    `params` must be C-contiguous float32 NumPy arrays, updated in place; anything
    else raises `TypeError`. `step(grads)`
    takes one gradient per parameter, in the same order and shapes; float64 gradients
    raise `TypeError`. `lr` is read at every step, so you can change ``opt.lr`` between
    steps, or set ``opt.lr_schedule`` to a schedule (`StepLR`, `ExponentialLR`,
    `OneCycleLR`, or any object with a one-based `lr_at(t)`), which sets `lr` before
    each step. `state_dict()` and `load_state_dict()` save and restore the step count,
    learning rate and state buffers.

    Not supported: parameter groups, `maximize`, `foreach`/`fused`/`capturable`/
    `differentiable`, sparse gradients and float64 parameters.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros(128, dtype=np.float32)
    >>> opt = mojolearn.Lion([w], lr=1e-4).step([np.ones_like(w)])
    """
    _NAME = "lion"

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, weight_decay=weight_decay), numeric_mode)

    @property
    def exp_avg(self):
        """Momentum buffer, flat over all parameters."""
        return self.state[0]


class Adamax(_SeqOptimizer):
    """Adamax optimizer, as `torch.optim.Adamax`.

    The infinity-norm variant of Adam: ``m = lerp(m, g, 1 - beta1)``,
    ``u = max(beta2 * u, |g| + eps)``, ``p -= lr / (1 - beta1**t) * m / u``. Weight
    decay is coupled (added to the gradient). Defaults are torch's.

    Parameters
    ----------
    params : ndarray or list of ndarray
        Parameters to update in place.
    lr : float, default 2e-3
        Learning rate.
    betas : tuple of float, default (0.9, 0.999)
        Decay rates of the first moment and the infinity norm, each in [0, 1).
    eps : float, default 1e-8
        Term added to the norm.
    weight_decay : float, default 0.0
        L2 penalty added to the gradient.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode of the update. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    t : int
        Number of steps taken.

    Notes
    -----
    `params` must be C-contiguous float32 NumPy arrays, updated in place; anything
    else raises `TypeError`. `step(grads)`
    takes one gradient per parameter, in the same order and shapes; float64 gradients
    raise `TypeError`. `lr` is read at every step, so you can change ``opt.lr`` between
    steps, or set ``opt.lr_schedule`` to a schedule (`StepLR`, `ExponentialLR`,
    `OneCycleLR`, or any object with a one-based `lr_at(t)`), which sets `lr` before
    each step. `state_dict()` and `load_state_dict()` save and restore the step count,
    learning rate and state buffers.

    Not supported: parameter groups, `maximize`, `foreach`/`fused`/`capturable`/
    `differentiable`, sparse gradients and float64 parameters.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros(128, dtype=np.float32)
    >>> opt = mojolearn.Adamax([w]).step([np.ones_like(w)])
    """
    _NAME = "adamax"

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, eps=eps, weight_decay=weight_decay), numeric_mode)


class NAdam(_SeqOptimizer):
    """NAdam optimizer, as `torch.optim.NAdam`.

    Adam with Nesterov momentum and torch's momentum schedule
    ``mu_t = beta1 * (1 - 0.5 * 0.96**(t * momentum_decay))``; the denominator is
    ``sqrt(v / (1 - beta2**t)) + eps``. Weight decay is coupled by default, decoupled
    (AdamW style) with ``decoupled_weight_decay=True``. Defaults are torch's.

    Parameters
    ----------
    params : ndarray or list of ndarray
        Parameters to update in place.
    lr : float, default 2e-3
        Learning rate.
    betas : tuple of float, default (0.9, 0.999)
        Decay rates of the first and second moments.
    eps : float, default 1e-8
        Term added to the denominator.
    weight_decay : float, default 0.0
        Weight decay.
    momentum_decay : float, default 4e-3
        Momentum schedule decay. Must be >= 0, otherwise `ValueError`.
    decoupled_weight_decay : bool, default False
        Apply weight decay to the parameters directly instead of the gradient.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode of the update. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    t : int
        Number of steps taken.

    Notes
    -----
    `params` must be C-contiguous float32 NumPy arrays, updated in place; anything
    else raises `TypeError`. `step(grads)`
    takes one gradient per parameter, in the same order and shapes; float64 gradients
    raise `TypeError`. `lr` is read at every step, so you can change ``opt.lr`` between
    steps, or set ``opt.lr_schedule`` to a schedule (`StepLR`, `ExponentialLR`,
    `OneCycleLR`, or any object with a one-based `lr_at(t)`), which sets `lr` before
    each step. `state_dict()` and `load_state_dict()` save and restore the step count,
    learning rate and state buffers.

    Not supported: parameter groups, `maximize`, `foreach`/`fused`/`capturable`/
    `differentiable`, sparse gradients and float64 parameters.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros(128, dtype=np.float32)
    >>> opt = mojolearn.NAdam([w], lr=2e-3).step([np.ones_like(w)])
    """
    _NAME = "nadam"

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3,
                 decoupled_weight_decay=False, numeric_mode=None):
        super().__init__(params, lr, dict(betas=betas, eps=eps, weight_decay=weight_decay,
                                           momentum_decay=momentum_decay,
                                           decoupled_weight_decay=decoupled_weight_decay), numeric_mode)


class Adafactor(_ResidentState):
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
        self._dev_b = _device_params(self.params, "Adafactor", _AF_DEV)
        for k, p in enumerate(self.params):  # glue: per parameter tensor, not elements
            if self._dev_b is None and (not isinstance(p, np.ndarray) or p.dtype != np.float32
                                        or not p.flags.c_contiguous):
                raise TypeError(f"Adafactor: params[{k}] must be a C-contiguous float32 NumPy array "
                                "or a SequenceDeviceTensor")
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
        # one resident handle per tensor (`adafactor_resident_*`), the second
        # moment mirrored on the host on access as for the other optimizers
        self.n_total = self._native_init(_MODE_ADAFACTOR, [int(p.size) for p in self.params],  # glue: sizes per parameter tensor
                                         n_handles=len(self.params))
        self._af_state = []
        for p in self.params:  # glue: allocates the state slots per parameter tensor, not elements
            if p.ndim == 2:
                self._af_state.append(dict(row_var=np.zeros(p.shape[0], np.float32),
                                           col_var=np.zeros(p.shape[1], np.float32)))
            else:
                self._af_state.append(dict(variance=np.zeros(p.shape[0], np.float32)))

    @staticmethod
    def _af_slots(st):
        return [st["row_var"], st["col_var"]] if "row_var" in st else [st["variance"]]

    def _slot_addrs(self):
        """Three slot addresses per tensor handle (0 for a slot not kept)."""
        out = []
        for st in self._af_state:  # glue: per parameter tensor, not elements
            a = [s.ctypes.data for s in self._af_slots(st)]  # glue: per moment slot (one or two)
            out.extend(a + [0] * (3 - len(a)))
        return out

    @property
    def _af_res(self):
        """The per-tensor resident handles, or None while the state is on the host."""
        return [int(h) for h in self._handles] if int(self._words[_W_HANDLE]) >= 0 else None  # glue: per parameter tensor handle

    @property
    def state(self):
        self._state_binding().seq_optimizer_state(self._words.ctypes.data, self._handles.ctypes.data,
                                                  self._slot_addrs())
        return self._af_state

    @state.setter
    def state(self, value):
        self._af_state = value
        self._host_fresh = True
        self._host_owned = True
        self._pristine = False

    def _grad_list(self, grads):
        if isinstance(grads, np.ndarray):
            grads = [grads]
        if len(grads) != len(self.params):
            raise ValueError(f"Adafactor: {len(grads)} grads, {len(self.params)} params")
        out = []
        for p, g in zip(self.params, grads):  # glue: per parameter tensor, not elements
            g = np.asarray(g)
            if g.shape != p.shape or g.dtype == np.float64:
                raise ValueError("Adafactor: a grad must be float32 of its param's shape")
            out.append(np.ascontiguousarray(g, dtype=np.float32))
        return out

    def _check_params(self):
        for k, p in enumerate(self.params):  # glue: per parameter tensor, not elements
            if not isinstance(p, np.ndarray) or p.dtype != np.float32 or not p.flags.c_contiguous \
                    or not p.flags.writeable:
                raise TypeError(f"Adafactor: params[{k}] must stay a writable C-contiguous float32 NumPy array")

    def step(self, grads):
        b = self._binding()
        addrs, kinds, sizes, keep, dev = self._operands(b, grads)
        rows = [int(p.shape[0]) for p in self.params]  # glue: shape metadata per parameter tensor
        cols = [int(p.shape[1]) if p.ndim == 2 else 0 for p in self.params]  # glue: shape metadata per parameter tensor
        s1 = [self._af_slots(st)[0].ctypes.data for st in self._af_state]  # glue: state slot addresses per tensor
        s2 = [self._af_slots(st)[-1].ctypes.data for st in self._af_state]  # glue: state slot addresses per tensor
        lr = self._lr_for_step()
        if self._res_b is None:
            self._res_b = b
        b.seq_adafactor_step(self._words.ctypes.data, self._handles.ctypes.data, addrs + s1 + s2,
                             [len(self.params), dev] + kinds + sizes + rows + cols,
                             [lr, self.beta2_decay, self.eps[0], self.eps[1], self.d, self.weight_decay])
        del keep
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
        self._dev_b = _device_params(self.params, "LAMB", _LAMB_DEV)
        for k, p in enumerate(self.params):  # glue: per parameter tensor, not elements
            if p.size == 0 or (self._dev_b is None and (not isinstance(p, np.ndarray) or p.dtype != np.float32
                                                         or not p.flags.c_contiguous)):
                raise TypeError(f"LAMB: params[{k}] must be a non-empty C-contiguous float32 NumPy array "
                                "or SequenceDeviceTensor")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("LAMB: betas must lie in [0, 1)")
        self.lr, self.betas, self.eps, self.weight_decay = float(lr), (float(b1), float(b2)), float(eps), float(weight_decay)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.numeric_mode = numeric_mode
        self.n_total = self._native_init(
            _MODE_LAMB, [int(p.size) for p in self.params],  # glue: sizes per parameter tensor
            extra=[int(bool(trust_clip)), int(bool(always_adapt)), int(bool(grad_averaging)),
                   int(bool(bias_correction)), int(self.max_grad_norm is not None)])
        self._state = [np.zeros(self.n_total, dtype=np.float32), np.zeros(self.n_total, dtype=np.float32)]
        self._sc = np.ones(2, dtype=np.float32)    # beta1^_sc_t, beta2^_sc_t

    #: bit0 trust_clip, bit1 always_adapt, bit2 grad_averaging, bit3
    #: bias_correction, bit4 the global gradient-norm clip
    flags = _word(_W_FLAGS)

    def step(self, grads):
        b = self._binding()
        addrs, kinds, sizes, keep, dev = self._operands(b, grads)
        lr = self._lr_for_step()
        if self._res_b is None:
            self._res_b = b
        b.seq_lamb_step(self._words.ctypes.data, self._handles.ctypes.data,
                        addrs + [self._state[0].ctypes.data, self._state[1].ctypes.data, self._sc.ctypes.data],
                        [len(self.params), dev] + kinds + sizes,
                        [lr, self.betas[0], self.betas[1], self.eps, self.weight_decay,
                         self.max_grad_norm if self.max_grad_norm is not None else 1.0])
        del keep
        return self


assert set(OPTIMIZERS) >= {"rmsprop", "adagrad"}
