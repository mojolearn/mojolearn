# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Learning-rate schedules of `torch.optim.lr_scheduler`: StepLR,
ExponentialLR and OneCycleLR, as closed forms of the step index. THE
CONTRACT (DEVIATION 5540): `lr_at(t)` is the exact rational value of the
closed form (the cosine of OneCycleLR as `_training_impl`'s rational
enclosure decides it) rounded ONCE to float32, ties to even, flushed to +0.0
below the smallest normal, so a schedule's bits are the same on every host.
`lr_at(t)` takes the ONE-BASED optimizer step, the convention of mojolearn's
schedules: `lr_at(1)` is torch's lr before its first `scheduler.step()`
(last_epoch 0), `lr_at(t)` its lr after t - 1 steps.

HOW: every value is ONE native call (sequence/lr_table.mojo `lr_spec_fill`).
OneCycleLR's steps are first evaluated in binary64 with a rigorous error
bound and kept only when both ends of that interval round to the same
float32; every other step, and StepLR / ExponentialLR's base gamma^e, takes
the exact big-rational route (sequence/lr_exact.mojo). Rounding is
monotone, so the bits equal the exact-rational evaluation's by
construction.

Pass one as `lr_schedule=` to the lane's optimizers (RMSprop, Adagrad, Lion,
Adamax, NAdam, LAMB) and recurrent estimators (LSTM/GRU/RNN), which set
`lr = schedule.lr_at(t)` before step t.

Not carried (sequence/NOT_IMPLEMENTED.tsv): OneCycleLR's momentum cycling
(cycle_momentum, base_momentum, max_momentum); per-group learning rates;
`last_epoch` resumption (evaluate `lr_at` at the step you resume from)."""
from ._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from ._training_impl import _LR_BEYOND, _LR_EXACT_ONLY, _LR_ONE_BASED, _LrTable, _lr_fill, _lr_status

import array as _array

#: `LrSpec` kind codes (sequence/lr_table.mojo)
_KIND_ONECYCLE = 10
_KIND_POW = 20


def _q(x, name):
    """`x` as a float, refused by name when it is not finite."""
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return x


# ------------------------------------------------------------------ schedules
# Every value is ONE native call (lane neural-optimizers, 2026-10-21):
# `lr_at(t)` and `lr_values(n)` are the core binding's `lr_schedule_fill`
# (sequence/lr_table.mojo `lr_spec_fill`: OneCycleLR's phases and binary64
# table with its float32 decision and exact route, the gamma^e enclosures of
# StepLR / ExponentialLR), the same values as before.
class _Sched(_LrTable):
    #: the sequence schedules' refusal messages (bindings/lr_table_helpers.mojo)
    _LR_STYLE = 1

    def _lr_refuse(self, st, t):
        if st == _LR_ONE_BASED:
            raise ValueError("schedule step t is ONE-BASED")
        _lr_status(st)

    def lrs(self, n):
        """lr_at(1) .. lr_at(n) as a list, from one native call."""
        return _OPT_NATIVE("schedule_list", self, (n,), None, _OPT_CONTEXT)


class _PowSched(_Sched):
    """base_lr gamma^E(t), E(t) the schedule's exponent: the float32 nearest
    the exact value, in Mojo (sequence/lr_exact.mojo `pow_exact`: a P-bit
    enclosure widened until both ends round alike, then the exact power)."""

    _step = 1

    def _lr_spec(self):
        return _OPT_NATIVE("schedule_spec", self, (), None, _OPT_CONTEXT)


class StepLR(_PowSched):
    """Step-decay learning-rate schedule: multiply by `gamma` every `step_size` steps.

    Computes torch's `StepLR` closed form, ``lr = base_lr * gamma ** (epoch // step_size)``,
    as a function of the step index. `lr_at(t)` takes the one-based optimizer step:
    `lr_at(1)` is the rate before the first `scheduler.step()` in torch, and `lr_at(t)`
    the rate after ``t - 1`` steps. Each value is the exact closed form rounded once to
    float32 (ties to even, values below the smallest normal flushed to +0.0), so a
    schedule gives the same bits on every machine.

    Pass a schedule as ``lr_schedule=`` to the optimizers `RMSprop`, `Adagrad`, `Lion`,
    `Adamax` and `NAdam`, and to the recurrent estimators (`LSTMClassifier`,
    `GRURegressor`, ...); they set ``lr = schedule.lr_at(t)`` before step t.

    Parameters
    ----------
    base_lr : float
        Learning rate of the first step. Must be finite and >= 0.
    step_size : int
        Number of steps between decays. Must be an integer >= 1.
    gamma : float, default 0.1
        Multiplicative decay factor. Must be finite and >= 0.

    Raises
    ------
    TypeError
        If `base_lr` or `gamma` is not a real number, or `step_size` is not an integer.
    ValueError
        If `base_lr` or `gamma` is not finite or negative, or `step_size` < 1.

    Notes
    -----
    `lr_at(t)` raises `ValueError` for ``t < 1`` and `OverflowError` when the value overflows
    float32. Per-group learning rates and `last_epoch` resumption are not supported:
    to resume, evaluate `lr_at` at the step you resume from.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.StepLR(0.1, step_size=30, gamma=0.1)
    >>> first, after_30 = sched.lr_at(1), sched.lr_at(31)
    """

    def __init__(self, base_lr, step_size, gamma=0.1):
        return _OPT_NATIVE("schedule_construct", self, (base_lr, step_size, gamma), None, _OPT_CONTEXT)


class ExponentialLR(_PowSched):
    """Exponential learning-rate schedule: multiply by `gamma` every step.

    Computes torch's `ExponentialLR` closed form, ``lr = base_lr * gamma ** epoch``, as a
    function of the step index. `lr_at(t)` takes the one-based optimizer step
    (`lr_at(1)` is `base_lr`). Each value is the exact closed form rounded once to
    float32 (ties to even, values below the smallest normal flushed to +0.0), so a
    schedule gives the same bits on every machine.

    Pass it as ``lr_schedule=`` to `RMSprop`, `Adagrad`, `Lion`, `Adamax`, `NAdam` and
    the recurrent estimators.

    Parameters
    ----------
    base_lr : float
        Learning rate of the first step. Must be finite and >= 0.
    gamma : float
        Multiplicative decay factor per step. Must be finite and >= 0.

    Raises
    ------
    TypeError
        If `base_lr` or `gamma` is not a real number.
    ValueError
        If `base_lr` or `gamma` is not finite or negative.

    Notes
    -----
    `lr_at(t)` raises `ValueError` for ``t < 1`` and `OverflowError` when the value overflows
    float32.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.ExponentialLR(0.01, gamma=0.95)
    >>> lrs = sched.lrs(100)
    """

    def __init__(self, base_lr, gamma):
        return _OPT_NATIVE("schedule_construct", self, (base_lr, 1, gamma), None, _OPT_CONTEXT)


class OneCycleLR(_Sched):
    """One-cycle learning-rate schedule, as torch's `OneCycleLR`.

    The rate rises from ``max_lr / div_factor`` to `max_lr` over the first `pct_start`
    of `total_steps`, then anneals down to ``initial_lr / final_div_factor`` (with
    ``three_phase=True``: back down to the initial rate, then to the final one), by
    cosine or linear annealing. `lr_at(t)` takes the one-based optimizer step and is
    defined up to ``total_steps + 1``; later steps raise `ValueError`, as torch does.
    Each value is the exact closed form rounded once to float32 (ties to even, values
    below the smallest normal flushed to +0.0), so a schedule gives the same bits on
    every machine.

    Pass it as ``lr_schedule=`` to `RMSprop`, `Adagrad`, `Lion`, `Adamax`, `NAdam` and
    the recurrent estimators.

    Parameters
    ----------
    max_lr : float
        Peak learning rate. Must be finite.
    total_steps : int
        Number of steps in the cycle. Must be >= 1.
    pct_start : float, default 0.3
        Fraction of the cycle spent increasing the rate, in [0, 1].
    anneal_strategy : {'cos', 'linear'}, default 'cos'
        Annealing shape.
    div_factor : float, default 25.0
        Initial rate is ``max_lr / div_factor``.
    final_div_factor : float, default 1e4
        Final rate is ``initial_lr / final_div_factor``.
    three_phase : bool, default False
        Use torch's three-phase form.

    Notes
    -----
    Momentum cycling (`cycle_momentum`, `base_momentum`, `max_momentum`), per-group
    learning rates and `last_epoch` resumption are not supported.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.OneCycleLR(max_lr=0.01, total_steps=1000)
    >>> lrs = sched.lrs(1000)
    """

    #: OneCycleLR's refusal messages, with its step past total_steps + 1
    _LR_STYLE = 2

    def __init__(self, max_lr, total_steps, pct_start=0.3, anneal_strategy="cos", div_factor=25.0,
                 final_div_factor=1e4, three_phase=False):
        return _OPT_NATIVE("schedule_construct", self, (max_lr, total_steps, pct_start, anneal_strategy, div_factor, final_div_factor, three_phase), None, _OPT_CONTEXT)

    def _lr_spec(self):
        return _OPT_NATIVE("schedule_spec", self, (), None, _OPT_CONTEXT)

    def _lr_refuse(self, st, t):
        if st == _LR_BEYOND:
            raise ValueError(f"OneCycleLR: step {t} is beyond total_steps {self.total_steps} + 1 (torch refuses it too)")
        _Sched._lr_refuse(self, st, t)

    def _lr_at_slow(self, t):
        """The exact route of one step (sequence/lr_exact.mojo
        `onecycle_exact`), one native call: the table's reference."""
        t = int(t)
        ip, fp = self._lr_spec()
        out = _array.array("f", (0.0,))
        st, _ = _lr_fill(([ip[0] + _LR_EXACT_ONLY] + ip[1:], fp), t, out)  # glue: the kind word with the exact-only flag
        if st:
            self._lr_refuse(st, t)
        return out[0]

# Import-time registration of built-in schedule identities, never a runtime hook.
from . import _training_impl as _schedule_training
from ._native_optimizer_bridge import connect_sequence as _connect_native_sequence_schedules
import sys as _schedule_sys
_connect_native_sequence_schedules(_schedule_training, _schedule_sys.modules[__name__])
