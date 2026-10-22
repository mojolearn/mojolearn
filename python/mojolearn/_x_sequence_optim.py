# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""torch.optim-shaped optimizers of the sequence expansion lane, over float32
NumPy arrays updated in place: `opt = RMSprop([w1, w2], lr=1e-2);
opt.step([g1, g2])`. One launch over the packed flat buffer per step, the
element body `sequence/ops.mojo::op_opt` (PyTorch's update rules), the same on
the GPU binding and on the CPU host binding. `lr` is read fresh at every step,
so a schedule is `opt.lr = ...` between steps, or `opt.lr_schedule = <schedule>` (a built-in native schedule, t one-based: `StepLR`, `ExponentialLR`, `OneCycleLR`), which
sets `lr` before every step.

Refused (sequence/NOT_IMPLEMENTED.tsv): parameter groups, maximize,
foreach/fused/capturable/differentiable, sparse gradients, float64."""
from ._training_impl import _OPT_NATIVE as _SEQ_NATIVE, _OPT_CONTEXT as _SEQ_CONTEXT


def _word(k, kind=int):
    # glue: connect native state-word metadata when the class is defined
    is_bool = kind is bool
    def get(self):
        return _SEQ_NATIVE("seq_word", self, (k, is_bool), None, _SEQ_CONTEXT)
    def put(self, value):
        return _SEQ_NATIVE("seq_word_set", self, (k, value), None, _SEQ_CONTEXT)
    return property(get, put)


class _ResidentState:
    t = _word(0)
    _sc_t = _word(1)
    _res_used = _word(3)
    _host_fresh = _word(4, bool)
    _host_owned = _word(5, bool)
    _pristine = _word(6, bool)

    def _binding(self):
        return _SEQ_NATIVE("seq_binding", self, (), None, _SEQ_CONTEXT)

    def _state_binding(self):
        return _SEQ_NATIVE("seq_state_binding", self, (), None, _SEQ_CONTEXT)

    def _slot_addrs(self):
        return _SEQ_NATIVE("seq_slots", self, (), None, _SEQ_CONTEXT)

    @property
    def _res(self):
        return _SEQ_NATIVE("seq_handle", self, (), None, _SEQ_CONTEXT)

    @property
    def state(self):
        """The current optimizer state buffers, synchronized from the device."""
        return _SEQ_NATIVE("seq_state", self, (), None, _SEQ_CONTEXT)

    @state.setter
    def state(self, value):
        return _SEQ_NATIVE("seq_state_set", self, (value,), None, _SEQ_CONTEXT)

    @property
    def resident_(self):
        """Whether the optimizer state is resident on the device."""
        return _SEQ_NATIVE("seq_resident", self, (), None, _SEQ_CONTEXT)

    def __del__(self):
        return _SEQ_NATIVE("seq_finalize", self, (), None, _SEQ_CONTEXT)

    def _lr_for_step(self):
        return _SEQ_NATIVE("seq_lr", self, (), None, _SEQ_CONTEXT)

    def step(self, grads):
        """Update every parameter from gradients of the same shapes and order."""
        return _SEQ_NATIVE("seq_step", self, (grads,), None, _SEQ_CONTEXT)


class _SeqOptimizer(_ResidentState):
    def __init__(self, params, lr, options, numeric_mode=None):
        return _SEQ_NATIVE("seq_construct", self, (params, lr, options), numeric_mode, _SEQ_CONTEXT)

    def state_dict(self):
        """Return an independent checkpoint of the step, learning rate and state."""
        return _SEQ_NATIVE("seq_export", self, (), None, _SEQ_CONTEXT)

    def load_state_dict(self, sd):
        """Restore the step, learning rate, state buffers and running scalars."""
        return _SEQ_NATIVE("seq_import", self, (sd,), None, _SEQ_CONTEXT)


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
    `OneCycleLR`, with one-based `lr_at(t)`), which sets `lr` before
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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (alpha, eps, weight_decay, momentum, centered)), numeric_mode, _SEQ_CONTEXT)

    @property
    def square_avg(self):
        """Running average of squared gradients, flat over all parameters."""
        return _SEQ_NATIVE("seq_slot", self, (1,), None, _SEQ_CONTEXT)


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
    `OneCycleLR`, with one-based `lr_at(t)`), which sets `lr` before
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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (lr_decay, weight_decay, initial_accumulator_value, eps)), numeric_mode, _SEQ_CONTEXT)

    @property
    def sum(self):
        """Accumulated squared gradients, flat over all parameters."""
        return _SEQ_NATIVE("seq_slot", self, (1,), None, _SEQ_CONTEXT)


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
    `OneCycleLR`, with one-based `lr_at(t)`), which sets `lr` before
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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (betas, weight_decay)), numeric_mode, _SEQ_CONTEXT)

    @property
    def exp_avg(self):
        """Momentum buffer, flat over all parameters."""
        return _SEQ_NATIVE("seq_slot", self, (0,), None, _SEQ_CONTEXT)


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
    `OneCycleLR`, with one-based `lr_at(t)`), which sets `lr` before
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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (betas, eps, weight_decay)), numeric_mode, _SEQ_CONTEXT)


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
    `OneCycleLR`, with one-based `lr_at(t)`), which sets `lr` before
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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (betas, eps, weight_decay, momentum_decay, decoupled_weight_decay)), numeric_mode, _SEQ_CONTEXT)


class Adafactor(_ResidentState):
    """`torch.optim.Adafactor` (PyTorch 2.5): relative step size, decoupled
    weight decay, a factored second moment (row and column means of g^2) for
    matrices and a full one for vectors, the update clipped by its RMS over
    d. Parameters are float32 NumPy arrays of 1 or 2 dimensions, updated in
    place; defaults are torch's. `sequence/adafactor.mojo`."""

    def __init__(self, params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, weight_decay=0.0,
                 maximize=False, numeric_mode=None):
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (beta2_decay, eps, d, weight_decay, maximize)), numeric_mode, _SEQ_CONTEXT)


    @property
    def _af_res(self):
        """The per-tensor resident handles, or None while the state is on the host."""
        return _SEQ_NATIVE("seq_handles", self, (), None, _SEQ_CONTEXT)


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
        return _SEQ_NATIVE("seq_construct", self, (params, lr, (bias_correction, betas, eps, weight_decay, grad_averaging, max_grad_norm, trust_clip, always_adapt)), numeric_mode, _SEQ_CONTEXT)

    #: bit0 trust_clip, bit1 always_adapt, bit2 grad_averaging, bit3
    #: bias_correction, bit4 the global gradient-norm clip
    flags = _word(8)


# glue: connect fixed native sequence owners after their classes exist
from . import _training_impl as _training, _x_sequence_device as _device
from ._native_optimizer_bridge import connect_sequence_optimizers as _connect_sequence_optimizers
_SEQ_NATIVE, _SEQ_CONTEXT = _connect_sequence_optimizers(_training, __import__(__name__, fromlist=["*"]), _device)
