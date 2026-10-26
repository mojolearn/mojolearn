# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Neural-network training on the GPU: SGD, Adam, AdamW, global-norm
gradient clipping and cross-entropy loss, over float32 buffers.

THE ARRAYS ARE BUFFERS, NOT NUMPY (numpy-free-0.7, DEVIATIONS 2402-2406).
Every parameter, gradient, logits and targets argument is read through the
buffer protocol (`_buffer.view`): a NumPy array, an `array.array`, a
`mojolearn.Array` or any other exporter is accepted, and NumPy is not
imported anywhere on this path. Everything this module ALLOCATES -- the
optimizer moments, the per-row loss, the logits gradient -- is a
`mojolearn.Array`, on which `numpy.asarray` is zero-copy. The in-place
contract is unchanged: `step` and `clip_grad_norm_` write back into the
very buffers you passed (`_unpack_into`), which is why those buffers must
be WRITABLE and C-CONTIGUOUS (refused by name otherwise) -- a read-only or
strided buffer cannot be written back without either a segfault or a copy
that would make "in place" a lie.

PRIVATE MODULE. The names are PyTorch's, because a caller reaching for an
optimizer has `torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.AdamW`,
`torch.nn.utils.clip_grad_norm_` and `torch.nn.functional.cross_entropy` in
mind, and inventing new spellings for the same five things would cost a
reader more than it saves.

WHAT LANDED HERE AND WHAT DID NOT
---------------------------------
**NOTHING NUMERIC. This module is wiring.** The arithmetic has existed and
been gated since 2026-08-28; what did not exist was a way to call it from
Python. A paper draft said "neural training is internal, not a public API",
and that sentence was true and was the only thing wrong with the lane. The
path is `training/checks/loss.mojo` and `training/checks/optimizer.mojo`
(the implementations), `training/estimator.mojo` (host-pointer transport,
no arithmetic), `bindings/_mojolearn_training.mojo` (the CPython boundary)
and this file.

THE GATE STRENGTH IS UNEVEN AND A USER CANNOT SEE IT FROM HERE
--------------------------------------------------------------
Two things are true and both are repeated on every class below.

  * **The contracts are carded on TWO vendors, byte-identical.** The loss
    card `training-loss.identical.card` is 17 records, md5 `a87615d9`,
    byte-identical on an Apple M4 and an AMD MI325X at the 2026-08-28 legs;
    its clause (a) matched device against oracle bitwise over 24 cases and
    all 61,925 compared cells. The optimizer card
    `training-optimizer.identical.card` is 18 records, md5 `97d160b0`,
    byte-identical on the same two boxes; its clause (a) passed 33 cases and
    all 382,822 compared cells. The composed training loop reached the same
    checkpoint digest `h_all = 463245ce6c97e68d` on both boxes over eight
    steps at one seed, with all four negative controls differing as
    required.
  * **NO NVIDIA LEG HAS RUN.** Not for either card, and not for the
    checkpoint comparison. This is a TWO-VENDOR result. Two backends
    agreeing closes nothing on its own: Apple and AMD agreed bit for bit
    through 302 GBDT stages in this same repository while NVIDIA diverged at
    `tree001.winners.scores`.

Both contracts also carry SKIPPED clauses on both columns, (b), (c), (d) and
(f), and the loss lane's `training/IDENTICAL_LOSS_CONTRACT.md` and the
optimizer lane's `training/IDENTICAL_OPTIMIZER_CONTRACT.md` are where that
list lives. Neither is a substitute for reading them.

And all of that belongs to `numeric_mode="identical"`. **Under the default
FAST build the pinned helpers compile away and there is no contract at
all**: the same loops run in whatever arithmetic the vendor's compiler
chose. They train. They promise nothing about bits.

WHAT IS NOT COVERED, REFUSED BY NAME RATHER THAN HALF-BUILT
------------------------------------------------------------
**Mixed precision and distributed training.** Every entry point here refuses
a non-float32 array by name rather than casting it, and nothing here is aware
of a second process. See `_NOT_COVERED` below. Dropout, initialization and
the learning-rate schedule JOINED this surface with the Samba stack lane:
`Generator` (position-keyed Philox, `core/philox_neural.mojo`), the three
`*LR` schedules (exact rational arithmetic, no host libm) and
`accumulate_grads` (optimizer contract clause 9.2's balanced tree) are at
the bottom of this file.

THE ONE SHAPE DEPARTURE FROM TORCH, AND IT IS UNAVOIDABLE
----------------------------------------------------------
**`step()` takes the gradients as an argument.** There is no autograd in this
library, so nothing anywhere holds a `.grad` for an optimizer to read.
`opt.step(grads)` is the honest spelling and `opt.step()` would have to
invent a place for the gradients to have come from. `cross_entropy(...,
return_grad=True)` is where a gradient comes from on this surface.

WHAT IT COSTS TO CROSS THIS BOUNDARY, STATED ONCE
--------------------------------------------------
The certified entry points take ONE FLAT BUFFER per role with an `offsets`
registry, so a list of per-tensor arrays is packed into flat float32 on the
way in and unpacked on the way out. That is a copy of the parameters and a
copy of the gradients per step, in host memory, on top of the upload. Pass
ONE flat array instead of a list and the packing collapses to a borrow; the
`packed_` attribute on each optimizer reports which happened.

The device side costs more and it is `training/estimator.mojo`'s docstrings
that price it: the optimizer's non-finite refusal downloads all four buffers
every step, and the loss materializes `logits` on the host before the upload
so the refusal can run before any device work. Both are stated there rather
than hidden, and both have a cheaper device-side form that is OWED.
"""

from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
from . import _portable_math as math
from . import _numeric_profile
from . import _native_samba_bridge
from fractions import Fraction

import array
import ctypes
import operator
import os
import numbers
import struct
from ._arrays import _addr, _addr_ro

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, memory_at, zeros
from ._bufcheck import (
    base_format, dtype_name, is_integer, is_native_f32, memcopy, memzero,
    nelems, probe,
)
from ._mode import NumericModeMixin


def _round_f32(value):
    """Binary32 scalar conversion only; neural arithmetic stays in the binding."""
    return ctypes.c_float(float(value)).value

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "clip_grad_norm_",
    "cross_entropy",
    "ConstantLR",
    "WarmupLinearLR",
    "WarmupCosineLR",
    "Generator",
    "accumulate_grads",
    "accumulation_is_aligned",
]

_EXT_NAME = "_mojolearn_training"

#: `training/checks/optimizer_oracle.mojo`'s own constants. SGD, Adam with
#: COUPLED decay (folded into the gradient, so it passes through `m` and `v`
#: and is itself smoothed and normalized), AdamW with DECOUPLED decay (it
#: multiplies the parameter and the gradient is untouched). The difference
#: between the last two is an ORDER, not a coefficient (contract 7.4), and at
#: `weight_decay = 0` they are the same arithmetic.
_KIND_SGD = 0
_KIND_ADAM = 1
_KIND_ADAMW = 2

#: `training/checks/loss_oracle.mojo`'s own constants.
_REDUCTION_NONE = 0
_REDUCTION_SUM = 1
_REDUCTION_MEAN = 2
_REDUCTIONS = {
    "none": _REDUCTION_NONE,
    "sum": _REDUCTION_SUM,
    "mean": _REDUCTION_MEAN,
}

#: torch's own default, `loss_oracle.IGNORE_INDEX_DEFAULT`.
_IGNORE_INDEX_DEFAULT = -100

# ===================================================================
# THE THREE THINGS THIS SURFACE DOES NOT COVER
# ===================================================================
# Named here, raised from `_refuse_out_of_scope` below, so that a caller who
# reaches for one gets the word "not covered" and not a plausible number.
# A paper draft lists exactly these three as outside the lane and this
# module is what has to keep that sentence true.
_NOT_COVERED = {
    "mixed precision": (
        "MIXED PRECISION IS NOT COVERED. Every buffer on this surface is "
        "float32 (loss contract section 1, optimizer contract section 1) and "
        "there is no float16 or bfloat16 anywhere in the two profiles: no "
        "master-weight copy, no loss scaler, no autocast, and no gated "
        "measurement of what a half-precision accumulation would do to the "
        "bits. A float16 array is REFUSED BY NAME here rather than upcast, "
        "because upcasting it silently would answer a question about float32 "
        "and label it as an answer about float16."
    ),
    "dropout": (
        "dropout is not an OPTIMIZER option. It is an op on this surface: "
        "`Generator(seed).dropout(x, p)` draws its mask from the "
        "position-keyed Philox stream in core/philox_neural.mojo (a pure "
        "function of seed, stream id and element index, never of the launch "
        "geometry) and `Generator.dropout_backward(dy, key)` replays it. "
        "Stochastic depth and any other sampling stay uncovered."
    ),
    "distributed": (
        "DISTRIBUTED TRAINING IS NOT COVERED. Nothing here is aware of a "
        "second process or a second device: there is no all-reduce, no "
        "gradient bucket and no rank. A cross-device gradient sum is another "
        "reduction and would need its own contract clause, its own fixture "
        "and its own sabotage before it could carry the identity claim the "
        "single-device path carries."
    ),
}


def _refuse_out_of_scope(topic, where):
    raise NotImplementedError(
        "mojolearn." + where + ": " + _NOT_COVERED[topic]
    )


# ===================================================================
# THE BINDING
# ===================================================================


def _load(mode=None):
    """The `_mojolearn_training` extension, in the tier the caller asked for,
    cross-checked against the tier the binary was COMPILED in.

    `_mojolearn_training` IS listed in `_backend.py`'s `_MODULES` and
    `_build_script`, so `_backend.binding` resolves it from the right
    directory on every layout and no private loader is needed here. DEVIATION
    869 is why that matters: an extension absent from `_MODULES` is never
    re-pointed, so under `numeric_mode="identical"` a plain relative import
    resolves to the FAST binary sitting beside it and returns fast arithmetic
    under the identical label.

    The read-back below is the second half of the same discipline. It is a
    NAME lookup and not a boolean.
    """
    mode = (mode or _backend.default_mode()).strip().lower()
    mod = _backend.binding(_EXT_NAME, mode)
    compiled = _backend._CODE_MODE.get(mod.training_numeric_mode(), "unknown")
    if compiled != mode:
        raise ImportError(
            "mojolearn: %s was compiled %s but numeric_mode asked for %s -- "
            "a binary is in the wrong directory; rebuild the sets with\n    "
            "bash bindings/build_training.sh\n    "
            "MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh"
            % (_EXT_NAME, compiled, mode)
        )
    return mod


# ===================================================================
# PACKING: A LIST OF TENSORS <-> ONE FLAT BUFFER PLUS AN OFFSETS REGISTRY
# ===================================================================
# The certified entry points take one flat buffer per role and an `offsets`
# vector of `J + 1` int32, where `offsets[j] .. offsets[j+1]` is tensor `j`
# and **`j` IS the `param_id`** (optimizer contract 3.3). Its ascending order
# is the cross-tensor summation order of the global-norm clip, so THE ORDER
# YOU PASS THE TENSORS IN IS PART OF THE ANSWER. Reorder them and the clip's
# total norm is a different, equally valid, different-bits number. That is
# the reference's own semantics, not a defect.


def _is_buffer(x):
    try:
        probe(x)
    except TypeError:
        return False
    return True


def _check_dtype(arrays, name, where, *, inplace):
    """float32 or REFUSED BY NAME. Never cast. Returns the `Probe` of every
    array, in order, for the callers that size and address them.

    A silent upcast from float16 would answer a float32 question and label it
    as a half-precision answer, which is the mixed-precision refusal this
    module opens with. float64 is refused for the reason `_buffer.py` gives
    at the library boundary: there is no float64 on a Metal device and every
    kernel in this library is float32, so a float64 array here is a caller
    who believes something about the arithmetic that is not true.

    `inplace=True` (parameters and gradients, which `_unpack_into` writes
    BACK INTO) additionally requires the buffer to be C-CONTIGUOUS and
    WRITABLE, refused by name (DEVIATION 2403). The NumPy spelling copied a
    strided tensor on the way in and wrote it back element-wise on the way
    out; without NumPy a strided write-back would be a Python element loop
    over the model, which the numpy-free contract forbids on this path.
    """
    probes = []
    for i, a in enumerate(arrays):  # glue: probes each parameter tensor
        pb = probe(a)
        dn = dtype_name(a, pb)
        if base_format(pb.format) == "e" or dn in ("float16", "bfloat16"):
            _refuse_out_of_scope("mixed precision", where)
        if not is_native_f32(pb.format):
            raise TypeError(
                "mojolearn.%s: %s[%d] has dtype %s; this surface is float32 "
                "only (loss contract section 1, optimizer contract section "
                "1) and refuses rather than casts, because a cast changes "
                "which arithmetic answered and leaves the label unchanged "
                "(python/mojolearn/_training_impl.py)"
                % (where, name, i, dn)
            )
        if nelems(pb.shape) < 1:
            raise ValueError(
                "mojolearn.%s: %s[%d] is empty, shape %r; an empty tensor is "
                "REFUSED and not skipped -- the clip would fold a zero-length "
                "GEMM, which no gate in this tree has run "
                "(training/estimator.mojo)" % (where, name, i, pb.shape)
            )
        if inplace and (not pb.c_contiguous or pb.readonly):
            raise ValueError(
                "mojolearn.%s: %s[%d] must be C-contiguous and writable; it "
                "is written back IN PLACE (the torch.optim contract, the "
                "trailing underscore on clip_grad_norm_) and a strided or "
                "read-only buffer cannot be. Pass np.ascontiguousarray(...) "
                "of it, or a writable mojolearn.Array "
                "(python/mojolearn/_training_impl.py, DEVIATION 2403)"
                % (where, name, i)
            )
        probes.append(pb)
    return probes


# ===================================================================
# THE OPTIMIZERS
# ===================================================================
# Each public method checks its plain Python arguments here and makes ONE
# native call with the optimizer object and the caller's arguments as they
# are (lane optimizers-thin-glue, 2026-10-21): the door
# (bindings/optimizer_public_door.mojo) reads the parameter and gradient
# tensors through the buffer protocol, refuses wrong dtypes, layouts,
# counts and shapes by name, resolves the built-in schedule's words and the
# optimizer state's host mirrors, calls the one kernel entry
# (training/optimizer_public.mojo on the GPU binding,
# training/host/optimizer_public_host.mojo on a CPU-only install) and
# assigns its result on the optimizer. Built-in schedules are native data;
# unsupported Python callback schedules are refused before invoking any hook.

#: `optimizer_door_step`'s operations (bindings/optimizer_public_door.mojo)
_OP_STEP, _OP_STEP_DEVICE, _OP_STEP_ACCUMULATED = 0, 1, 2
#: `optimizer_door_state`'s readings
_STATE_DICT, _STATE_EXP_AVG, _STATE_EXP_AVG_SQ = 0, 1, 2
#: `optimizer_door_io`'s operations (training/optimizer_public.mojo)
_IO_QUERY, _IO_PARAMS_UP, _IO_PARAMS_DOWN, _IO_GRADS_UP, _IO_GRADS_DOWN, _IO_PARAM_DEVICE = 0, 1, 2, 3, 4, 5


class _Optimizer(NumericModeMixin):
    """The shared half of `SGD`, `Adam` and `AdamW`.

    ONE PARAMETER GROUP PER OPTIMIZER. `torch.optim` accepts a list of dicts
    so that different tensors carry different learning rates; the certified
    entry point takes ONE `OptimizerConfig` for the whole flat buffer, so a
    second group would be a second call with a second registry and a second
    clip, and the clip's coefficient is a function of every gradient in ITS
    registry (contract 3.5). Per-parameter groups are REFUSED BY NAME below
    rather than emulated by two calls, because two calls do not give the same
    clip as one.
    """

    _mojolearn_native_public = True
    __getattribute__ = object.__getattribute__
    _BINDING = _EXT_NAME
    _KIND = None

    def __init__(self, params, where, lr_schedule=None, accumulation_steps=1,
                 resident=None):
        self._where = where
        #: The optimizer state's native slot (training/optimizer_public.mojo):
        #: its handle once the first step opened it, the module that owns it,
        #: whether the slot keeps the state on the host, and whether the host
        #: mirrors `_m` / `_v` are current.
        self._res = None
        self._res_binding = None
        self._res_host = False
        self._host_fresh = True
        self._resident_wanted = resident
        # A trainer runs only under a numeric profile whose TRAINING gates have passed.
        _numeric_profile.require_training("mojolearn.%s" % where)
        #: A schedule object with `lr_at(t)` (t ONE-BASED), or None. When set,
        #: each step's learning rate is the schedule's value at that step and
        #: `lr` is overwritten with it.
        if lr_schedule is not None and not hasattr(lr_schedule, "lr_at"):
            raise TypeError(
                "mojolearn.%s: lr_schedule must have an lr_at(step) method "
                "(ConstantLR, WarmupLinearLR, WarmupCosineLR) "
                "(python/mojolearn/_training_impl.py)" % where
            )
        self.lr_schedule = lr_schedule
        #: The declared microbatch count A of optimizer contract clause 9.2.
        #: `step_accumulated` refuses a different count by name.
        a = int(accumulation_steps)
        if a < 1 or (a & (a - 1)) != 0:
            raise ValueError(
                "mojolearn.%s: accumulation_steps must be a power of two >= 1 "
                "(optimizer contract clause 9.2 condition 4), got %r "
                "(python/mojolearn/_training_impl.py)" % (where, accumulation_steps)
            )
        self.accumulation_steps = a
        # ONE native call (the tier-free core binding: an optimizer is built
        # before its numeric mode is known): `params` as the tensor list,
        # refused by name unless float32, non-empty, C-contiguous and
        # writable; `offsets` (J + 1 int32, `offsets[j] .. offsets[j + 1]` is
        # tensor j of the flat registry); `n_total`; SGD's per-tensor
        # `buf_initialized` flag (contract 7.3b, CARRIED STATE that belongs in
        # a checkpoint beside `exp_avg`); and the host mirrors `_m` / `_v` of
        # `exp_avg` and `exp_avg_sq`, allocated on first need (with the
        # optimizer under MOJOLEARN_OPT_LAZY_MOMENTS=0, the A/B).
        _buffers._native("optimizer_door_init")(self, params)  # cpu-route: the parameter registry offsets are host metadata built once with the optimizer
        #: ONE-BASED. The first step of a run is `t = 1`.
        self.t = 0
        self.packed_ = None
        #: The pre-clip global gradient norm of the most recent step, or None
        #: when that step ran with `max_norm=None`.
        self.total_norm_ = None
        #: The clamped coefficient the most recent step applied, or None.
        self.clip_coef_ = None

    # -- the state: a native slot, mirrored on the host on access -----------
    @property
    def exp_avg(self):
        """Adam's first moment (SGD's momentum buffer), a `mojolearn.Array`
        of `n_total` floats. Reading it copies the optimizer state into the
        returned array, so read it once per use, not per element; assign to
        `exp_avg` to change the state."""
        return _OPT_NATIVE("state", self, (_STATE_EXP_AVG,), None, _OPT_CONTEXT)

    @exp_avg.setter
    def exp_avg(self, value):
        return _OPT_NATIVE("state_set", self, (_STATE_EXP_AVG, value), None, _OPT_CONTEXT)

    @property
    def exp_avg_sq(self):
        """Adam's second moment; see `exp_avg`."""
        return _OPT_NATIVE("state", self, (_STATE_EXP_AVG_SQ,), None, _OPT_CONTEXT)

    @exp_avg_sq.setter
    def exp_avg_sq(self, value):
        return _OPT_NATIVE("state_set", self, (_STATE_EXP_AVG_SQ, value), None, _OPT_CONTEXT)

    @property
    def resident_(self):
        """True once the optimizer state lives on the device."""
        return _OPT_NATIVE("resident", self, (), None, _OPT_CONTEXT)

    def _binding(self):
        return _OPT_NATIVE("binding", self, (), None, _OPT_CONTEXT)

    # -- lane fam2-neural (2026-10-04): parameters and gradients resident ----
    # After `params_to_device()` the parameters live in the optimizer's
    # device slot: `step(grads)` then uploads the gradient only, and
    # `step_device()` with the gradient already on the device
    # (`grads_to_device`) moves nothing. The caller's host arrays are STALE
    # until `params_to_host()` writes them back. The step is the same device
    # call on the same buffers: the same bits.
    def _io(self, op, tensors=None, name="params", tensor=0):
        """One `optimizer_door_io` call; returns its result word."""
        return _OPT_NATIVE("raw_io", self, (op, tensors, name, tensor), None, _OPT_CONTEXT)

    def params_resident_available(self):
        """Whether `params_to_device` works on this build."""
        return _OPT_NATIVE("io", self, (_IO_QUERY, None, "params", 0), None, _OPT_CONTEXT)

    def params_to_device(self):
        """Upload the parameters into the optimizer's device handle and keep
        them there: later steps neither upload nor download them. Call it
        again after editing the host arrays. Returns self."""
        return _OPT_NATIVE("io", self, (_IO_PARAMS_UP, None, "params", 0), None, _OPT_CONTEXT)

    def params_to_host(self):
        """Download the device parameters into the caller's arrays (in
        place); they stay resident. Returns the parameter list."""
        return _OPT_NATIVE("io", self, (_IO_PARAMS_DOWN, None, "params", 0), None, _OPT_CONTEXT)

    def grads_to_device(self, grads):
        """Upload a gradient into the handle's device gradient buffer, for
        `step_device()`. Returns self."""
        return _OPT_NATIVE("io", self, (_IO_GRADS_UP, grads, "grads", 0), None, _OPT_CONTEXT)

    def grads_to_host(self, grads):
        """Download the device gradient (clipped, if the last step clipped)
        into `grads`, in place. Returns `grads`."""
        return _OPT_NATIVE("io", self, (_IO_GRADS_DOWN, grads, "grads", 0), None, _OPT_CONTEXT)

    def param_device(self, j):
        """Parameter tensor `j` as a `TrainDeviceArray` (a device copy of
        the resident parameters; `params_to_device()` first), for the next
        forward. Its shape is the host tensor's."""
        return _OPT_NATIVE("param_device", self, (j,), None, _OPT_CONTEXT)

    def step_device(self, grads=None, max_norm=None):
        """`step` on device-resident parameters (`params_to_device()` first).
        `grads` given: uploaded (one transfer), and scaled in place on the
        host when `max_norm` clips, as `step` does. `grads` None: the
        gradient already in the handle (`grads_to_device`) is used and, with
        `max_norm`, clipped there; nothing crosses the bus but the three
        info words. Returns what `step` returns."""
        return _OPT_NATIVE("step", self, (grads, max_norm, _OP_STEP_DEVICE, None), None, _OPT_CONTEXT)

    def get_params(self, deep=True):
        """Return constructor parameters for inspection or cloning."""
        return _OPT_NATIVE("get_params", self, (deep,), None, _OPT_CONTEXT)

    def set_params(self, **params):
        """Rebuild with updated constructor parameters and reset optimizer state."""
        return _OPT_NATIVE("set_params", self, (params,), None, _OPT_CONTEXT)

    def state_dict(self):
        """Everything a resumed run needs, and nothing that is not state.

        `t`, `exp_avg`, `exp_avg_sq` and `buf_initialized`. Leaving
        `buf_initialized` out is the quiet way to break a resume: SGD's first
        step after the reload would take the COPY arm again and overwrite a
        momentum buffer it should have continued, which is the failure the
        optimizer contract's clause (d) control 2 is built to catch.
        """
        return _OPT_NATIVE("state", self, (_STATE_DICT,), None, _OPT_CONTEXT)

    # DEVIATION 2404: arrays go through _buffer.as_f32_c / as_i32_c and are copied (the NumPy spelling
    # aliased a caller's float32 C-order array).
    def load_state_dict(self, state):
        """Restore the optimizer state saved by `state_dict`.

        Parameters
        ----------
        state : dict
            The dictionary returned by `state_dict`. Its arrays may be any buffer or
            nested list and are converted to float32 (int32 for ``buf_initialized``).
            The optimizer keeps its own copy, so later steps never modify ``state``.
        """
        return _OPT_NATIVE("load", self, (state,), None, _OPT_CONTEXT)

    def zero_grad(self, grads):
        """`torch.optim.Optimizer.zero_grad`, over arrays the caller owns.
        Present so that a training loop written against torch reads the same;
        it writes `+0.0` and nothing else -- all-zero bytes into each buffer,
        which must be C-contiguous and writable."""
        return _OPT_NATIVE("zero", self, (grads,), None, _OPT_CONTEXT)

    def step(self, grads, max_norm=None):
        """One step. `grads` matches `params` in count, shape and order.

        `max_norm` runs the global-norm clip BEFORE the elementwise update,
        over the whole registry, and SCALES THE CALLER'S GRADIENT ARRAYS IN
        PLACE, which is what `torch.nn.utils.clip_grad_norm_` does and why it
        carries a trailing underscore. `None` turns it off; the gradient is
        then untouched.

        **WITH CLIPPING ON, ONE PARAMETER'S UPDATE IS NOT INDEPENDENT OF THE
        REST OF THE MODEL.** The coefficient is a function of every gradient
        in the registry, by the reference's own semantics and not by a defect
        (optimizer contract 3.5).

        Returns the pre-clip total gradient norm as a float when the clip
        ran, and None when it did not.
        """
        return _OPT_NATIVE("step", self, (grads, max_norm, _OP_STEP, None), None, _OPT_CONTEXT)

    def step_accumulated(self, microbatch_grads, tokens, max_norm=None):
        """One step from `A = self.accumulation_steps` microbatch gradient
        lists, combined on device by the clause 9.2 balanced tree in
        ascending microbatch index, then `step`. `tokens` is the FULL step's
        token count T; a split that clause 9.2 does not admit is refused by
        name (no gradient is combined). Returns what `step` returns."""
        return _OPT_NATIVE("step", self, (microbatch_grads, max_norm, _OP_STEP_ACCUMULATED, tokens), None, _OPT_CONTEXT)


# Optimizer contract 7.3 (momentum 7.3, dampening 7.3a, nesterov 7.3c). `params` order is the
# param_id order, i.e. the clip's cross-tensor summation order (contract 3.3). lr is a host scalar
# computed once per step (contract 7.1). At dampening 0, c_damp is exactly 1.0, so the first-step
# copy and the recurrence agree. nesterov with momentum 0 is refused because the profile would accept
# it and silently equal the plain update. maximize: torch's -grad at the top of the step, before the
# coupled decay and the momentum buffer (training/maximize.mojo, DEVIATION 6200). One launch per
# tensor for SGD; per-parameter groups refused because two clips do not equal one (contract 3.5).
# Mixed precision: see `_NOT_COVERED` at the top of this file.
class SGD(_Optimizer):
    """Stochastic gradient descent, as `torch.optim.SGD`, over float32 buffers (NumPy arrays, `array.array` or `mojolearn.Array`; any writable buffer-protocol object).

    Supports momentum, dampening, Nesterov momentum and weight decay. Weight decay is
    coupled, as in torch: it is added to the gradient and so passes through the momentum
    buffer. Parameters are updated in place.

    Parameters
    ----------
    params : buffer or list of buffers
        float32 parameter arrays, updated in place.
    lr : float, default 1e-3
        Learning rate. torch's SGD has no default; this one is 1e-3.
    momentum : float, default 0.0
        Momentum factor.
    dampening : float, default 0.0
        Dampening of the momentum.
    weight_decay : float, default 0.0
        L2 penalty, added to the gradient.
    nesterov : bool, default False
        Nesterov momentum. Needs ``momentum > 0`` and ``dampening == 0``, otherwise
        `ValueError` (as torch).
    lr_schedule : object or None, default None
        ConstantLR, WarmupLinearLR or WarmupCosineLR; sets `lr` before every step.
        Python callback schedules raise `TypeError` before invocation.
    accumulation_steps : int, default 1
        Number of microbatches combined by `step_accumulated`; a power of two >= 1,
        otherwise `ValueError`.
    maximize : bool, default False
        Maximize the objective (the gradient's sign is flipped inside the step; the
        caller's gradient arrays are not negated). Must be a bool.
    resident : bool or None, default None
        Keep the optimizer state on the device between steps when the install supports
        it (None or True). False keeps it on the host, which suits an optimizer that is
        rebuilt every step.

    Notes
    -----
    `step(grads, max_norm=None)` takes one gradient per parameter, matching `params` in
    count, shape and order. With `max_norm` set, the global L2 gradient norm is clipped
    first: the caller's gradient arrays are scaled in place (as
    `torch.nn.utils.clip_grad_norm_`) and `step` returns the pre-clip total norm;
    without it, `step` returns None and the gradients are untouched. With clipping on,
    the order of `params` is part of the result, because it fixes the summation order of
    the norm.

    `lr` is read at every step, so you can assign ``opt.lr`` between steps, or pass an
    `lr_schedule` (`ConstantLR`, `WarmupLinearLR` or `WarmupCosineLR`). `step_accumulated(microbatch_grads, tokens)` combines
    `accumulation_steps` microbatch gradient lists before one step. `zero_grad(grads)`
    writes +0.0 into the given gradient arrays. `state_dict()` / `load_state_dict()`
    save and restore the step count and the optimizer state.

    Refused with `TypeError`: `foreach`, `fused`, `capturable`, `differentiable`,
    parameter groups, sparse gradients, float16/bfloat16 and float64 buffers, and any
    unknown keyword.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros((64, 8), dtype=np.float32)
    >>> opt = mojolearn.SGD([w], lr=1e-2, momentum=0.9)
    >>> g = np.ones_like(w)
    >>> norm = opt.step([g], max_norm=1.0)
    """

    _KIND = _KIND_SGD

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, lr_schedule=None,
                 accumulation_steps=1, maximize=False, resident=None, *, numeric_mode=None, **kwargs):
        return _OPT_NATIVE("construct", self, (params, _KIND_SGD, lr, (0.9, 0.999), 1e-8, weight_decay, momentum, dampening, nesterov, lr_schedule, accumulation_steps, maximize, resident, kwargs), numeric_mode, _OPT_CONTEXT)

    def _config(self):
        return {
            "lr": self.lr,
            # beta1, beta2 and eps are in the params list for every
            # algorithm because ONE ORDER for three algorithms is one thing
            # to keep in step instead of three. SGD's kernel never reads
            # them; these are the reference's Adam defaults so that a card
            # comparing two runs sees the same bits in the unread slots.
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "nesterov": 1 if self.nesterov else 0,
        }


# Optimizer contract 7.2, coupled weight decay; AdamW (7.4) differs by an order, not a
# coefficient. Contracts 5.2/5.3: bias correction is not float64 here, and `1 - beta2` in float32
# differs from torch's float64 route by about 1.3e-5 relative. eps is contract 4d (`sqrt(v) + eps`,
# not `sqrt(v + eps)`; they separate when v lands in 1e-20..1e-12). beta 1.0 refused by
# training/estimator.mojo (1 - beta^t would be zero). amsgrad excluded: max(v_max, v) would create
# a -0.0 vs +0.0 selection hazard (see `clip_coefficient`). Adam is one launch over the flat buffer.
class Adam(_Optimizer):
    """Adam optimizer, as `torch.optim.Adam`, over float32 buffers (NumPy arrays, `array.array` or `mojolearn.Array`; any writable buffer-protocol object).

    Weight decay is coupled, as in torch's Adam: it is added to the gradient and so
    passes through both moments. `AdamW` is the decoupled form; at ``weight_decay=0``
    the two compute the same thing. The bias corrections are computed in float32, not
    in torch's float64, so results are close to torch's but not bit-equal to them.
    Parameters are updated in place.

    Parameters
    ----------
    params : buffer or list of buffers
        float32 parameter arrays, updated in place.
    lr : float, default 1e-3
        Learning rate.
    betas : tuple of float, default (0.9, 0.999)
        Decay rates of the first and second moments, each in [0, 1); 1.0 is refused.
    eps : float, default 1e-8
        Added to ``sqrt(v)`` in the denominator (``sqrt(v) + eps``).
    weight_decay : float, default 0.0
        L2 penalty, added to the gradient.
    lr_schedule : object or None, default None
        ConstantLR, WarmupLinearLR or WarmupCosineLR; sets `lr` before every step.
        Python callback schedules raise `TypeError` before invocation.
    accumulation_steps : int, default 1
        Number of microbatches combined by `step_accumulated`; a power of two >= 1,
        otherwise `ValueError`.
    maximize : bool, default False
        Maximize the objective (the gradient's sign is flipped inside the step; the
        caller's gradient arrays are not negated). Must be a bool.
    resident : bool or None, default None
        Keep the optimizer state on the device between steps when the install supports
        it (None or True). False keeps it on the host, which suits an optimizer that is
        rebuilt every step.

    Attributes
    ----------
    t : int
        Number of steps taken.
    exp_avg : array
        First moment, flat over all parameters.
    exp_avg_sq : array
        Second moment, flat over all parameters.

    Notes
    -----
    `step(grads, max_norm=None)` takes one gradient per parameter, matching `params` in
    count, shape and order. With `max_norm` set, the global L2 gradient norm is clipped
    first: the caller's gradient arrays are scaled in place (as
    `torch.nn.utils.clip_grad_norm_`) and `step` returns the pre-clip total norm;
    without it, `step` returns None and the gradients are untouched. With clipping on,
    the order of `params` is part of the result, because it fixes the summation order of
    the norm.

    `lr` is read at every step, so you can assign ``opt.lr`` between steps, or pass an
    `lr_schedule` (`ConstantLR`, `WarmupLinearLR` or `WarmupCosineLR`). `step_accumulated(microbatch_grads, tokens)` combines
    `accumulation_steps` microbatch gradient lists before one step. `zero_grad(grads)`
    writes +0.0 into the given gradient arrays. `state_dict()` / `load_state_dict()`
    save and restore the step count and the optimizer state.

    Refused with `TypeError`: `foreach`, `fused`, `capturable`, `differentiable`,
    parameter groups, sparse gradients, float16/bfloat16 and float64 buffers, and any
    unknown keyword.
    `amsgrad` is not supported and raises `TypeError`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros((64, 8), dtype=np.float32)
    >>> opt = mojolearn.Adam([w], lr=1e-3)
    >>> opt.step([np.ones_like(w)])
    """

    _KIND = _KIND_ADAM

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, lr_schedule=None, accumulation_steps=1,
                 maximize=False, resident=None, *, numeric_mode=None, **kwargs):
        return _OPT_NATIVE("construct", self, (params, _KIND_ADAM, lr, betas, eps, weight_decay, 0.0, 0.0, False, lr_schedule, accumulation_steps, maximize, resident, kwargs), numeric_mode, _OPT_CONTEXT)

    def _config(self):
        return {
            "lr": self.lr,
            "beta1": self.betas[0],
            "beta2": self.betas[1],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            # Read only by SGD's kernel. Present so that one params list
            # serves three algorithms; the reference's SGD defaults.
            "momentum": 0.0,
            "dampening": 0.0,
            "nesterov": 0,
        }


# Optimizer contract 7.4: decoupled decay `p * (1 - lr*wd)`; the only difference from `Adam` is the
# order. The defaults are Adam's (weight_decay 0.0), not torch's AdamW 0.01.
class AdamW(Adam):
    """AdamW optimizer, as `torch.optim.AdamW`, over float32 buffers (NumPy arrays, `array.array` or `mojolearn.Array`; any writable buffer-protocol object).

    Identical to `Adam` except that weight decay is decoupled: the parameter is
    multiplied by ``1 - lr * weight_decay`` and the gradient is left untouched, so the
    decay does not pass through the moments. Parameters, attributes and refusals are
    `Adam`'s.

    Note that the default ``weight_decay`` here is 0.0, while torch's AdamW default is
    0.01; at 0.0 AdamW and Adam compute the same thing. Pass ``weight_decay=0.01`` to
    match torch's default.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> w = np.zeros((64, 8), dtype=np.float32)
    >>> opt = mojolearn.AdamW([w], lr=1e-3, weight_decay=0.01)
    >>> opt.step([np.ones_like(w)])
    """

    _KIND = _KIND_ADAMW

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, lr_schedule=None, accumulation_steps=1,
                 maximize=False, resident=None, *, numeric_mode=None, **kwargs):
        return _OPT_NATIVE("construct", self, (params, _KIND_ADAMW, lr, betas, eps, weight_decay, 0.0, 0.0, False, lr_schedule, accumulation_steps, maximize, resident, kwargs), numeric_mode, _OPT_CONTEXT)


#: torch parameter names this surface does not have, and the reason each is
#: refused. Refused BY NAME rather than swallowed by `**kwargs`, because a
#: keyword that is accepted and ignored is a caller who believes something
#: about their run that is not true.
_REFUSED_KWARGS = {
    "amsgrad": (
        "the profile EXCLUDES amsgrad and the exclusion is load bearing: "
        "`max(v_max, v)` would create the profile's first -0.0 vs +0.0 "
        "selection hazard, where the order of a compare decides which of two "
        "equal-comparing values survives (optimizer contract 8c, and "
        "clip_coefficient's docstring in "
        "training/checks/optimizer_oracle.mojo)"
    ),
    "foreach": (
        "an EXECUTION knob. Adam here is ONE launch over the whole flat "
        "buffer and SGD is one launch per tensor, because SGD carries a "
        "per-tensor flag; the choice is the profile's and not the caller's "
        "(training/checks/optimizer.mojo::identical_optimizer_step)"
    ),
    "fused": (
        "an EXECUTION knob, and in torch it is a different arithmetic as well"
    ),
    "capturable": (
        "CUDA-graph capture; nothing on this surface captures, and the step "
        "synchronizes before it returns "
        "(training/checks/optimizer.mojo::identical_optimizer_step)"
    ),
    "differentiable": (
        "there is no autograd in this library, which is also why `step` takes "
        "the gradients as an argument"
    ),
    "params_groups": (
        "one `OptimizerConfig` per optimizer. Two parameter groups would be "
        "two calls with two registries and two clips, and two clips do not "
        "equal one: the clip coefficient is a function of every gradient in "
        "ITS registry (optimizer contract 3.5)"
    ),
    "param_groups": (
        "one `OptimizerConfig` per optimizer; see `params_groups`"
    ),
    "weight": (
        "a per-class weight vector is REFUSED BY NAME (loss contract section "
        "11): a weighted mean's denominator is a SUM OF FLOATS and would need "
        "a fold, a clause, a fixture and a sabotage of its own, where `count` "
        "today is an INTEGER and therefore exact, order-free and vendor-free "
        "(contract 5.5, `ce_count` in training/checks/loss_oracle.mojo)"
    ),
    "size_average": (
        "torch's own deprecated alias for `reduction`. Refused rather than "
        "translated, because translating it silently reproduces the ambiguity "
        "torch deprecated it for"
    ),
    "reduce": (
        "torch's own deprecated alias for `reduction`; see `size_average`"
    ),
    "dropout": None,
    "device": (
        "the device is chosen by the binary that loaded, and its accelerator "
        "API is readable with `vendor_used()` (python/mojolearn/_backend.py)"
    ),
    "dtype": (
        "float32 only. MIXED PRECISION IS NOT COVERED and a non-float32 "
        "array is refused by name rather than cast"
    ),
    "distributed": None,
    "world_size": None,
    "rank": None,
}


def _maximize_flag(maximize, where):
    """`maximize` as a strict bool: torch reads it as a bool, and a string
    or a number here is a caller who meant something else."""
    if not isinstance(maximize, bool):
        raise TypeError(
            "mojolearn.%s: maximize must be True or False, got %r "
            "(python/mojolearn/_training_impl.py)" % (where, maximize)
        )
    return maximize


def _refuse_unknown(kwargs, where):
    """Every keyword this surface does not have, refused BY NAME."""
    for k in kwargs:  # glue: refuses unknown keyword arguments
        if k in ("dropout",):
            _refuse_out_of_scope("dropout", where)
        if k in ("distributed", "world_size", "rank"):
            _refuse_out_of_scope("distributed", where)
        why = _REFUSED_KWARGS.get(k)
        if why:
            raise TypeError(
                "mojolearn.%s: %s= is REFUSED BY NAME -- %s "
                "(python/mojolearn/_training_impl.py)" % (where, k, why)
            )
        raise TypeError(
            "mojolearn.%s: unexpected keyword %s=. This surface takes only "
            "the parameters in its WHAT IS HONORED table; an unknown keyword "
            "is refused rather than ignored, because a keyword that is "
            "accepted and does nothing is a caller who believes something "
            "about their run that is not true "
            "(python/mojolearn/_training_impl.py)" % (where, k)
        )


# ===================================================================
# THE GLOBAL-NORM CLIP
# ===================================================================


# Optimizer contract section 3. Two-level fold (contract 3.1): a norm per tensor, then a norm over
# those; the flat sqrt(sum of squares) is a different float32 number. Both folds go through the
# certified GEMM at m = n = 1, OP_NT (its leaf partition, balanced tree, odd-tail carry). max_norm
# <= 0 refused by training/estimator.mojo. norm_type != 2 refused: an L-inf clip is a max-reduction
# with its own -0.0 selection hazard. error_if_nonfinite=False refused: the profile always refuses a
# non-finite total (contract 8a, `refuse_nonfinite_scalar`). The certified entry synchronizes once
# per tensor to keep sub-buffer views alive across each delegated GEMM; a batched launcher is owed.
def clip_grad_norm_(grads, max_norm, norm_type=2.0, error_if_nonfinite=True,
                    numeric_mode=None, **kwargs):
    """Clip gradients by their global L2 norm, as `torch.nn.utils.clip_grad_norm_`.

    Works over float32 buffers (NumPy arrays, `array.array` or `mojolearn.Array`; any writable buffer-protocol object). Computes a norm per tensor, then the norm of those norms, and
    scales every gradient in place by ``max_norm / total`` when the total exceeds
    `max_norm`. Returns the pre-clip total norm, as torch does. The order of the tensors
    is part of the result, because it fixes the summation order.

    Parameters
    ----------
    grads : buffer or list of buffers
        float32 gradient arrays, scaled in place.
    max_norm : float
        Maximum total norm; must be > 0, otherwise `ValueError`.
    norm_type : float, default 2.0
        Only 2.0 is supported; any other value (including inf) raises
        `NotImplementedError`.
    error_if_nonfinite : bool, default True
        Only True is supported (False raises `NotImplementedError`): a non-finite
        total norm always raises.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    total_norm : float
        The total gradient norm before clipping.

    Notes
    -----
    Not supported: `foreach`, per-tensor norms in the result, float16/bfloat16 and
    float64. Each tensor costs one device synchronization, so many small tensors are
    slower than a few large ones.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> g1 = np.ones((64, 8), dtype=np.float32); g2 = np.ones(8, dtype=np.float32)
    >>> total = mojolearn.clip_grad_norm_([g1, g2], max_norm=1.0)
    """
    # One native call (lane optimizers-thin-glue): the keyword, norm_type,
    # error_if_nonfinite and max_norm refusals, the tensors read through the
    # buffer protocol, and the clip (training/optimizer_public.mojo
    # `public_clip`: the registry, the route -- the tensors straight from
    # their own memory, or the packed buffer under MOJOLEARN_CLIP_PACKED=1 or
    # several optimizer devices; the same device buffer, offsets and clip, so
    # the same bits -- and the in-place scale).
    return _OPT_NATIVE("clip", None, (grads, max_norm, norm_type, error_if_nonfinite, kwargs), numeric_mode, _OPT_CONTEXT)


# ===================================================================
# THE LOSS
# ===================================================================


# Profile mojolearn.identical.loss.ce.fp32.v1, loss contract sections 2-6. The backward reads the
# forward's `expo` and `denom` and recomputes nothing. Bad targets refused by `ce_refuse_inputs`
# (training/checks/loss_oracle.mojo). reduction='none' has no backward (loss contract section 11):
# the placement of a per-row upstream product before or after seam L16's division is an open
# decision. label_smoothing 0.0 selects a different kernel (contract 6.2(c), DEVIATION 1155).
# `ce_divisor` is the one producer of the divisor. weight refused (section 11): a weighted mean's
# denominator is a float sum needing its own fold (contract 5.5). Memory: training/estimator.mojo
# allocates shift and expo at N*V, plus weights and dlogits with return_grad, plus logp with
# smoothing; the forward is row independent through seam L11.
def cross_entropy(logits, targets, ignore_index=_IGNORE_INDEX_DEFAULT,
                  reduction="mean", label_smoothing=0.0, num_items=None,
                  return_grad=False, numeric_mode=None, **kwargs):
    """Cross-entropy loss over class indices, as `torch.nn.functional.cross_entropy`.

    `logits` is (N, V) float32 and `targets` is (N,) int32 or int64 holding a class index
    or `ignore_index`. Returns a float for ``reduction='mean'`` or ``'sum'`` and an (N,)
    float32 array for ``'none'``. With ``return_grad=True`` the forward and the gradient
    with respect to the logits come from one call: ``(loss, dlogits)``, `dlogits` of
    shape (N, V).

    Parameters
    ----------
    logits : buffer, shape (N, V)
        float32 logits, row-major.
    targets : buffer, shape (N,)
        Integer class indices; a value that is neither a valid class nor `ignore_index`
        raises an error naming its row.
    ignore_index : int, default -100
        Targets equal to this are ignored, even when it is a valid class index (as
        torch).
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Reduction over rows. ``'none'`` has no gradient: `return_grad=True` with it is
        refused.
    label_smoothing : float, default 0.0
        Label smoothing in [0, 1).
    num_items : int or None, default None
        Divisor of the mean, as Hugging Face's `num_items_in_batch`. None uses the
        number of non-ignored targets.
    return_grad : bool, default False
        Also return the gradient with respect to `logits`.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    loss : float or ndarray
        The loss; (N,) per-row values under ``reduction='none'``.
    dlogits : array, shape (N, V)
        Only with ``return_grad=True``.

    Notes
    -----
    Refused: per-class `weight`, the deprecated `size_average` and `reduce`,
    class-probability (soft) targets, float16/bfloat16 and float64. Memory use is
    several (N, V) float32 buffers; for a large vocabulary, split the rows into chunks
    (with ``reduction='none'`` the chunked result equals the unsplit one).

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> logits = np.random.default_rng(0).random((32, 10), dtype=np.float32)
    >>> targets = np.arange(32, dtype=np.int64) % 10
    >>> loss, dlogits = mojolearn.cross_entropy(logits, targets, return_grad=True)
    """
    return _OPT_NATIVE("cross_entropy", None, (logits, targets, ignore_index, reduction, label_smoothing, num_items, return_grad, kwargs), numeric_mode, _OPT_CONTEXT)


def numeric_mode_used(numeric_mode=None):
    """Return the numeric mode the training binding will run in.

    Read from the compiled binding itself, not from the requested string.

    Parameters
    ----------
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    str
    """
    return _OPT_NATIVE("training_numeric_mode", None, (), numeric_mode, _OPT_CONTEXT)


def vendor_used(numeric_mode=None):
    """Return the accelerator API of the training binding: 'metal', 'cuda' or 'hip'.

    Read from the binding's compile-time constant, not from the platform.

    Parameters
    ----------
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    str
    """
    return _OPT_NATIVE("training_vendor", None, (), numeric_mode, _OPT_CONTEXT)


# ===================================================================
# LEARNING-RATE SCHEDULES: a float32 that is a pure function of (step, config)
# ===================================================================
# NO HOST LIBM. `math.cos` differs in the last bit between platforms, so a
# cosine schedule spelled with it would train a different run on each box.
# Every schedule below is DEFINED as the float32 nearest an exact rational
# (round-half-even), the cosine through a Taylor interval enclosing it,
# decided only when both ends round alike (else more terms; a 64-term
# straddle raises rather than guesses). Since lane py-runtime round 3 the
# exact route is Mojo big rationals (sequence/lr_exact.mojo,
# sequence/bigrat.mojo): the same rational intervals, the same answers.

_F32_MIN_NORMAL_EXP = -126
_F32_MAX_EXP = 127

#: lr_exact statuses -> the exceptions the Python route raised
_LR_ERRORS = {
    1: (OverflowError, "mojolearn: schedule value overflows float32"),
    2: (ArithmeticError, "mojolearn: the exact schedule value straddles a float32 rounding "
                         "boundary within 1e-60; refusing to guess a bit"),
    3: (ValueError, "mojolearn: cosine progress must be in [0, 1]"),
}


def _lr_status(st):
    """Raise the exact route's refusal for a nonzero status."""
    st = int(st)
    if st:
        kind, msg = _LR_ERRORS[st]
        raise kind(msg)


def _f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', _round_f32(value)))[0]


# THE SCHEDULES' ONE NATIVE CALL (lane neural-optimizers, 2026-10-21). A
# schedule's value at step t is DEFINED as the float32 nearest an exact
# rational, and that definition does not change. `lr_at(t)` and
# `lr_values(n)` are each ONE call of the core binding's `lr_schedule_at` /
# `lr_schedule_series` (sequence/lr_table.mojo `lr_spec_fill`): the binary64 table with its
# rigorous error bounds, the float32 decision, the exact big-rational route
# for every step the table leaves undecided, the constant tail and, for
# OneCycleLR, the phases. MOJOLEARN_LR_TABLE=0 (or MOJOLEARN_LR_EXACT_ONLY=1)
# keeps every step on the exact route (the A/B); the answers are the same.

#: `lr_spec_fill`'s statuses beyond `_LR_ERRORS`: a step below 1, and a
#: OneCycleLR step past total_steps + 1
_LR_ONE_BASED = 4
_LR_BEYOND = 5
#: added to a schedule kind: every step on the exact route (`_lr_at_slow`)
_LR_EXACT_ONLY = 100


def _lr_table_wanted():
    return (os.environ.get("MOJOLEARN_LR_TABLE") != "0"
            and os.environ.get("MOJOLEARN_LR_EXACT_ONLY") != "1")


def _lr_native(key):
    """A core-binding entry of the schedules (bindings/lr_table_helpers.mojo)."""
    return _buffers._native(key)


def _lr_buffers(n):
    """Two zeroed float64 buffers (values, error bounds) of n steps."""
    return array.array("d", bytes(8 * n)), array.array("d", bytes(8 * n))


# TOMBSTONE: lr_schedule_fill (dead native entry: no live caller) deleted 2026-10-26 by a98f13f57; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/lr_schedule_fill.patch


class _LrTable(object):
    """`lr_at`, `lr_values` and `bits_at` of a built-in schedule, one native
    call each. A subclass provides `_lr_spec()` (its `LrSpec` words: four
    ints and four floats, sequence/lr_table.mojo) and `_lr_refuse(status,
    t)` (its refusal of step t)."""

    def _lr_refuse(self, st, t):
        if st == _LR_ONE_BASED:
            raise ValueError("mojolearn schedule: step t is ONE-BASED, got %d" % t)
        _lr_status(st)

    #: whose refusal messages `lr_at` / `lr_values` raise (bindings/lr_table_helpers.mojo)
    _LR_STYLE = 0

    def lr_at(self, t):
        """The float32 learning rate of ONE-BASED step `t`, as a float."""
        return _OPT_NATIVE("schedule_at", self, (t,), None, _OPT_CONTEXT)

    def lr_values(self, n):
        """lr_at(1) .. lr_at(n) as an array('f'), in one native call."""
        return _OPT_NATIVE("schedule_values", self, (n,), None, _OPT_CONTEXT)

    def bits_at(self, t):
        """`lr_at(t)` as its float32 bit pattern, for fixtures."""
        return _OPT_NATIVE("schedule_bits", self, (t,), None, _OPT_CONTEXT)


class _Schedule(_LrTable):
    """Shared half of the schedules. `t` is the optimizer's ONE-BASED step
    counter, so `lr_at(1)` is the first step's learning rate."""

    kind = ""
    #: `LrSpec` kind codes (sequence/lr_table.mojo)
    _KIND_CODE = {"constant": 0, "linear": 1, "cosine": 2}

    def __init__(self, peak_lr, warmup_steps=0, total_steps=None, min_lr=0.0):
        return _OPT_NATIVE("schedule_construct", self, (peak_lr, warmup_steps, total_steps, min_lr), None, _OPT_CONTEXT)

    def config(self):
        return _OPT_NATIVE("schedule_config", self, (), None, _OPT_CONTEXT)

    @staticmethod
    def from_config(cfg):
        return _OPT_NATIVE("schedule_from_config", None, (cfg,), None, _OPT_CONTEXT)

    def _lr_spec(self):
        return _OPT_NATIVE("schedule_spec", self, (), None, _OPT_CONTEXT)

    def _exact_params(self):
        return ([self._KIND_CODE[self.kind], self.warmup_steps,
                 -1 if self.total_steps is None else self.total_steps],
                [float(self.peak_lr), float(self.min_lr)], "lr_schedule_exact_block")  # cpu-route: a learning-rate schedule value is one host scalar per optimizer step (exact big-rational route)

    def _lr_at_slow(self, t):
        """The exact route of one ONE-BASED step, in Mojo big rationals
        (`lr_schedule_exact`, sequence/lr_exact.mojo)."""
        t = int(t)
        if t < 1:
            raise ValueError(
                "mojolearn schedule: step t is ONE-BASED, got %d" % t)
        ip, fp, _ = self._exact_params()
        st, v = _lr_native("lr_schedule_exact")(ip + [t], fp)  # cpu-route: a learning-rate schedule value is one host scalar per optimizer step (exact big-rational route)
        _lr_status(st)
        return float(v)


class ConstantLR(_Schedule):
    """Constant learning rate, after an optional linear warmup.

    Returns `peak_lr` at every step after `warmup_steps` steps of linear warmup starting
    from ``peak_lr / warmup_steps``. `total_steps` and `min_lr` are accepted and ignored.

    Parameters
    ----------
    peak_lr : float
        Learning rate after warmup; finite and >= 0 (rounded to float32).
    warmup_steps : int, default 0
        Steps of linear warmup; >= 0.
    total_steps : int or None, default None
        Step at which the decay reaches `min_lr`; >= max(1, warmup_steps).
    min_lr : float, default 0.0
        Final learning rate; finite and >= 0.

    Notes
    -----
    `lr_at(t)` takes the optimizer's one-based step: `lr_at(1)` is the first step's
    rate. Each value is the float32 nearest the exact expression, so a schedule gives
    the same bits on every machine. `config()` returns the schedule's settings as a
    dict, which `from_config` turns back into a schedule. Invalid arguments raise
    `ValueError`. Pass a schedule as ``lr_schedule=`` to `SGD`, `Adam` or `AdamW`.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.ConstantLR(1e-3, warmup_steps=100)
    >>> lr = sched.lr_at(1)
    """

    kind = "constant"

    def __init__(self, peak_lr, warmup_steps=0, total_steps=None,
                 min_lr=0.0):
        return _OPT_NATIVE("schedule_construct", self, (peak_lr, warmup_steps, None, 0.0), None, _OPT_CONTEXT)


class WarmupLinearLR(_Schedule):
    """Linear warmup, then linear decay to `min_lr`.

    Rises linearly over `warmup_steps`, falls in a straight line from `peak_lr` to
    `min_lr` at `total_steps`, then stays at `min_lr`.

    Parameters
    ----------
    peak_lr : float
        Learning rate after warmup; finite and >= 0 (rounded to float32).
    warmup_steps : int, default 0
        Steps of linear warmup; >= 0.
    total_steps : int or None, default None
        Step at which the decay reaches `min_lr`; >= max(1, warmup_steps).
    min_lr : float, default 0.0
        Final learning rate; finite and >= 0.

    Notes
    -----
    `lr_at(t)` takes the optimizer's one-based step: `lr_at(1)` is the first step's
    rate. Each value is the float32 nearest the exact expression, so a schedule gives
    the same bits on every machine. `config()` returns the schedule's settings as a
    dict, which `from_config` turns back into a schedule. Invalid arguments raise
    `ValueError`. Pass a schedule as ``lr_schedule=`` to `SGD`, `Adam` or `AdamW`.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.WarmupLinearLR(1e-3, warmup_steps=100, total_steps=1000)
    >>> lr = sched.lr_at(500)
    """

    kind = "linear"




# lane/neural-net-experiment (2026-09-30): the cosine schedule's value is
# DEFINED as the float32 nearest the exact rational value (the interval
# route below), and that definition does not change. The binary64 table
# (`_LrTable`, in Mojo since lane py-runtime round 2:
# bindings/lr_table_helpers.mojo) answers every step whose rigorous error
# interval rounds to one float32; the exact route decides the rest.
# MOJOLEARN_LR_EXACT_ONLY=1 restores the exact route for every step.
# The value is the float32 nearest the exact rational expression (the interval route, never
# math.cos); a binary64 evaluation with a rigorous error bound answers first whenever it provably
# rounds to that same float32 (see the section comment above this class).
class WarmupCosineLR(_Schedule):
    """Linear warmup, then cosine decay to `min_lr`.

    Rises linearly over `warmup_steps`, then follows
    ``min_lr + (peak_lr - min_lr) * (1 + cos(pi * p)) / 2`` with
    ``p = (t - warmup_steps) / (total_steps - warmup_steps)``, then stays at `min_lr`.

    Parameters
    ----------
    peak_lr : float
        Learning rate after warmup; finite and >= 0 (rounded to float32).
    warmup_steps : int, default 0
        Steps of linear warmup; >= 0.
    total_steps : int or None, default None
        Step at which the decay reaches `min_lr`; >= max(1, warmup_steps).
    min_lr : float, default 0.0
        Final learning rate; finite and >= 0.

    Notes
    -----
    `lr_at(t)` takes the optimizer's one-based step: `lr_at(1)` is the first step's
    rate. Each value is the float32 nearest the exact expression, so a schedule gives
    the same bits on every machine. `config()` returns the schedule's settings as a
    dict, which `from_config` turns back into a schedule. Invalid arguments raise
    `ValueError`. Pass a schedule as ``lr_schedule=`` to `SGD`, `Adam` or `AdamW`.

    Examples
    --------
    >>> import mojolearn
    >>> sched = mojolearn.WarmupCosineLR(3e-4, warmup_steps=200, total_steps=10000, min_lr=3e-5)
    >>> lr = sched.lr_at(5000)
    """

    kind = "cosine"


# ===================================================================
# GRADIENT ACCUMULATION (optimizer contract clause 9.2)
# ===================================================================


# `microbatch_split_is_identical(T, A)` from training/checks/optimizer_oracle.mojo, contract clause 9.2.
def accumulation_is_aligned(tokens, accumulation_steps, numeric_mode=None):
    """Whether a split into microbatches gives exactly the unsplit gradient.

    True when `accumulation_steps` microbatches over `tokens` tokens, combined by
    `accumulate_grads`, give the same weight-gradient bits as one unsplit batch.

    Parameters
    ----------
    tokens : int
        Token count of the whole step.
    accumulation_steps : int
        Number of microbatches.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    bool
    """
    return _OPT_NATIVE("aligned", None, (tokens, accumulation_steps), numeric_mode, _OPT_CONTEXT)


# v1 balanced tree, ftz(ftz(x) + ftz(y)) at every node, contract clause 9.2 condition 5.
def accumulate_grads(microbatch_grads, tokens, numeric_mode=None):
    """Combine microbatch gradients into one, on the device.

    Adds the microbatch gradients by a balanced tree in ascending microbatch order
    (not a running sum), so the result does not depend on the device. When `tokens` is
    given, a split that `accumulation_is_aligned` rejects raises `ValueError`; pass
    ``tokens=None`` to combine without that check.

    Parameters
    ----------
    microbatch_grads : list
        One entry per microbatch, each a float32 array or a list of arrays with the
        same shapes across entries.
    tokens : int or None
        Token count of the whole step, or None.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array or list of arrays
        The combined gradient, in the structure of one entry.
    """
    # One native call (lane optimizers-thin-glue): the microbatches read
    # through the buffer protocol and checked by name, strided buffers made
    # contiguous, the [A, N] stack, the clause 9.2 tree and the result arrays
    # (bindings/optimizer_public_door.mojo, training/optimizer_public.mojo
    # `public_accumulate`).
    return _OPT_NATIVE("accumulate", None, (microbatch_grads, tokens), numeric_mode, _OPT_CONTEXT)


# ===================================================================
# THE NEURAL RANDOM STREAM: Generator, initializers, dropout
# ===================================================================
# `core/philox_neural.mojo` is the arithmetic and this class is its
# bookkeeping: a seed, and a COUNTER that names the next stream id. Every
# draw consumes one stream id, so the sequence of draws is a pure function
# of (seed, counter) and the counter is CARRIED STATE that belongs in a
# checkpoint beside the optimizer moments.

_RNG_UNIFORM = 0
_RNG_NORMAL = 1
_RNG_DROPOUT_FORWARD = 2
_RNG_DROPOUT_BACKWARD = 3


def _rng_call(binding, kind, n, offset, seed, stream, a, b, inp=None):
    out = _buffers.empty(n, '<f4')
    if inp is None:
        inp = _buffers.zeros(1, '<f4')
    # addresses = [out, inp]; params = [n, offset, seed_lo, seed_hi,
    # stream_id, kind, a, b] (mirrored word for word in
    # bindings/_mojolearn_training.mojo::neural_rng_binding).
    binding.neural_rng(
        [_addr(out), _addr_ro(inp)],
        [int(n), int(offset), int(seed & 0xFFFFFFFF),
         int((seed >> 32) & 0xFFFFFFFF), int(stream), int(kind),
         float(a), float(b)],
    )
    return out


# The initializers compute their bound / std on the host in IEEE float64 with only +, -, *, / and
# sqrt (correctly rounded everywhere), round to float32 and hand it to the device; the draw is
# core/philox_neural.mojo's fixed integer-to-float mapping.
class Generator(object):
    """Seeded random stream for neural-network initialization and dropout.

    Every value drawn is a function of (seed, stream id, element index) only, so the
    same call at the same counter gives the same bits on every device. Each draw uses
    the next stream id. `state_dict()` returns ``{'seed', 'counter'}``, and
    `load_state_dict()` resumes the stream exactly.

    Parameters
    ----------
    seed : int
        Seed in [0, 2**64); otherwise `ValueError`.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Examples
    --------
    >>> import mojolearn
    >>> gen = mojolearn.Generator(0)
    >>> w = gen.kaiming_uniform((128, 64), fan_in=64)
    >>> y, key = gen.dropout(w, p=0.1)
    """

    def __init__(self, seed, numeric_mode=None):
        return _native_samba_bridge.RUNTIME.samba_native_call("generator_init", (self, seed, numeric_mode), _native_samba_bridge.CONTEXT)

    def state_dict(self):
        """Return the generator state as ``{'seed': int, 'counter': int}``."""
        return _native_samba_bridge.RUNTIME.samba_native_call("generator_snapshot", (self,), _native_samba_bridge.CONTEXT)

    def load_state_dict(self, state):
        """Restore a state from `state_dict`.

        The next-stream counter may equal ``2**32`` to represent exhaustion;
        larger or negative counters raise ``ValueError``.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call("generator_restore", (self, state), _native_samba_bridge.CONTEXT)

    def next_stream(self):
        """Consume and return the next stream id.

        Use it for operations that draw several slices of one stream, such as dropout over
        microbatches. Raises `RuntimeError` when the 32-bit counter is exhausted.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call("generator_next", (self,), _native_samba_bridge.CONTEXT)

    def _next_stream(self):
        return _native_samba_bridge.RUNTIME.samba_native_call("generator_next", (self,), _native_samba_bridge.CONTEXT)

    @staticmethod
    def _shape(shape):
        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)
        else:
            shape = tuple(int(d) for d in shape)  # glue: shape tuple of ints
        n = 1
        for d in shape:  # glue: validates each shape dimension
            if d < 1:
                raise ValueError(
                    "mojolearn.Generator: shape must be positive, got %r"
                    % (shape,))
            n *= d
        return shape, n

    def uniform(self, shape, low=0.0, high=1.0):
        """Draw float32 values uniform in [low, high).

        Parameters
        ----------
        shape : int or tuple of int
            Output shape; every dimension must be positive.
        low : float, default 0.0
            Lower bound.
        high : float, default 1.0
            Upper bound; must be >= `low`.
        """
        return _native_samba_bridge.RUNTIME.generator_draw("uniform", (self, shape, low, high), _native_samba_bridge.CONTEXT)

    def normal(self, shape, mean=0.0, std=1.0):
        """Draw float32 normal values (Box-Muller).

        Parameters
        ----------
        shape : int or tuple of int
            Output shape.
        mean : float, default 0.0
            Mean.
        std : float, default 1.0
            Standard deviation; must be >= 0.
        """
        return _native_samba_bridge.RUNTIME.generator_draw("normal", (self, shape, mean, std), _native_samba_bridge.CONTEXT)

    # None defaults are evaluated by the native owner through the same
    # portable-math ABI as before; no precomputed square-root literals.
    def kaiming_uniform(self, shape, fan_in, a=None):
        """Kaiming-uniform initialization, as `torch.nn.init.kaiming_uniform_`.

        The bound is ``sqrt(2 / (1 + a**2)) * sqrt(3 / fan_in)``. The default
        ``a = sqrt(5)`` is torch's `nn.Linear` default and gives the bound ``1 / sqrt(fan_in)``.

        Parameters
        ----------
        shape : int or tuple of int
            Output shape.
        fan_in : int
            Fan-in of the layer.
        a : float or None, default None
            Negative slope; None means sqrt(5).
        """
        return _native_samba_bridge.RUNTIME.generator_draw("kaiming_uniform", (self, shape, fan_in, a), _native_samba_bridge.CONTEXT)

    def xavier_uniform(self, shape, fan_in, fan_out, gain=1.0):
        """Xavier-uniform initialization with bound ``gain * sqrt(6 / (fan_in + fan_out))``."""
        return _native_samba_bridge.RUNTIME.generator_draw("xavier_uniform", (self, shape, fan_in, fan_out, gain), _native_samba_bridge.CONTEXT)

    def kaiming_normal(self, shape, fan_in, gain=None):
        """Kaiming-normal initialization with std ``gain / sqrt(fan_in)``; `gain` None means sqrt(2)."""
        return _native_samba_bridge.RUNTIME.generator_draw("kaiming_normal", (self, shape, fan_in, gain), _native_samba_bridge.CONTEXT)

    def xavier_normal(self, shape, fan_in, fan_out, gain=1.0):
        """Xavier-normal initialization with std ``gain * sqrt(2 / (fan_in + fan_out))``."""
        return _native_samba_bridge.RUNTIME.generator_draw("xavier_normal", (self, shape, fan_in, fan_out, gain), _native_samba_bridge.CONTEXT)

    def dropout(self, x, p, offset=0, stream=None):
        """Inverted dropout of a float32 array.

        Keeps each element whose uniform draw is ``>= p`` and scales it by ``1 / (1 - p)``.
        Returns ``(y, key)``; pass `key` to `dropout_backward` to replay the same mask.

        Parameters
        ----------
        x : array of float32
            Input.
        p : float
            Drop probability in [0, 1); otherwise `ValueError`.
        offset : int, default 0
            Element index of ``x[0]`` within the stream, so that a microbatch of a larger
            batch draws the same values it would draw unsplit.
        stream : int or None, default None
            Stream id; None takes the next one.

        Returns
        -------
        y : array of float32
        key : dict
        """
        return _native_samba_bridge.RUNTIME.generator_draw("dropout", (self, x, p, offset, stream), _native_samba_bridge.CONTEXT)

    def dropout_backward(self, dy, key):
        """Gradient of `dropout`: apply the mask and scale recorded in `key` to `dy`."""
        return _native_samba_bridge.RUNTIME.generator_draw("dropout_backward", (self, dy, key), _native_samba_bridge.CONTEXT)


# ===================================================================
# THE STACK'S OPS: embedding, RMSNorm, LM head (no arithmetic here)
# ===================================================================


def _c32(a, name, where):
    pb = probe(a)
    if not is_native_f32(pb.format):
        raise TypeError('mojolearn.%s: %s must be float32' % (where, name))
    return as_f32_c(a, ndim=None, name=name)[0]


# The gather of embedding/checks/embedding_identical.mojo.
def embedding_forward(weight, ids, numeric_mode=None):
    """Embedding lookup: ``weight[ids]``.

    Parameters
    ----------
    weight : array, shape (V, D)
        float32 embedding table.
    ids : array, shape (N,)
        Integer row indices.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array, shape (N, D)
    """
    return _OPT_NATIVE("embedding_forward", None, (weight, ids), numeric_mode, _OPT_CONTEXT)


# The run-sorted ascending fold of embedding/checks/embedding_identical.mojo, never atomics.
def embedding_backward(dy, ids, vocab, numeric_mode=None):
    """Gradient of `embedding_forward` with respect to the table.

    Returns a fresh (vocab, D) gradient from `dy`. Rows with equal ids are summed in a
    fixed ascending order (no atomics), so the result is reproducible.

    Parameters
    ----------
    dy : array, shape (N, D)
        float32 gradient of the output.
    ids : array, shape (N,)
        The forward's ids; the row counts must match, otherwise `ValueError`.
    vocab : int
        Number of table rows V.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array, shape (vocab, D)
    """
    return _OPT_NATIVE("embedding_backward", None, (dy, ids, vocab), numeric_mode, _OPT_CONTEXT)


def rms_norm_forward(x, weight, eps, numeric_mode=None):
    """RMS normalization of each row: ``weight * x / sqrt(mean(x**2) + eps)``.

    Parameters
    ----------
    x : array, shape (M, D)
        float32 input.
    weight : array, shape (D,)
        float32 scale.
    eps : float
        Added to the mean square.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array, shape (M, D)
    """
    return _OPT_NATIVE("rms_norm_forward", None, (x, weight, eps), numeric_mode, _OPT_CONTEXT)


# The forward's row sums are recomputed by the same kernel.
def rms_norm_backward(dy, x, weight, eps, numeric_mode=None):
    """Gradients ``(dx, dweight)`` of `rms_norm_forward`.

    Parameters
    ----------
    dy : array, shape (M, D)
        Gradient of the output.
    x : array, shape (M, D)
        The forward input.
    weight : array, shape (D,)
        The forward's scale.
    eps : float
        The forward's epsilon.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    dx : array, shape (M, D)
    dweight : array, shape (D,)
    """
    return _OPT_NATIVE("rms_norm_backward", None, (dy, x, weight, eps), numeric_mode, _OPT_CONTEXT)


# ===================================================================
# DEVICE ARRAYS (lane fam2-neural, 2026-10-04)
# ===================================================================
# `linear_forward`, `cross_entropy`, `linear_backward` and an optimizer step
# each upload their operands and download their results, so a loop written
# with them moves every N * V and parameter-sized array across the bus four
# or more times a step. `train_to_device(array)` uploads once and returns a
# `TrainDeviceArray` (a handle in the training binding's device pool);
# `linear_forward`, `linear_backward` and `cross_entropy` given device
# arrays keep their results on the device and return device arrays, and an
# optimizer whose parameters are resident (`params_to_device`) steps from
# device gradients (`step_device(grads)`) and hands a parameter tensor back
# as a device array (`param_device(j)`). `to_host()` downloads. The device
# calls are the host entries' own, on the same values: the same bits.
# IDENTICAL GPU builds only (training/dev_tensors.mojo); elsewhere
# `train_to_device` refuses by name. No arithmetic here.
_DEV_ENTRIES = ("train_dev_alloc", "train_dev_free", "train_dev_put", "train_dev_get",
                "linear_forward_dev", "linear_backward_dev", "ce_loss_dev")


def _dev_binding(numeric_mode, where):
    binding = _load(numeric_mode)
    if not all(callable(getattr(binding, n, None)) for n in _DEV_ENTRIES):  # glue: checks binding entry names
        raise RuntimeError(
            "mojolearn.%s: this build has no training device arrays (IDENTICAL "
            "GPU builds only; MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF turns them off)" % where)
    return binding


class TrainDeviceArray(object):
    """A float32 array resident on the device (the training binding's
    pool). Make one with `train_to_device`; read it with `to_host()`."""

    def __init__(self, binding, handle, shape, base=None):
        return _OPT_NATIVE("device_construct", self, (binding, handle, shape, base), None, _OPT_CONTEXT)

    @staticmethod
    def _new(binding, shape):
        return _OPT_NATIVE("device_new", None, (binding, shape), None, _OPT_CONTEXT)

    def reshape(self, shape):
        return _OPT_NATIVE("device_reshape", self, (shape,), None, _OPT_CONTEXT)

    def to_host(self):
        """The array as a host `mojolearn.Array` (one download)."""
        return _OPT_NATIVE("device_host", self, (), None, _OPT_CONTEXT)

def _is_dev(x):
    return isinstance(x, TrainDeviceArray)


def train_to_device(a, numeric_mode=None):
    """float32 array `a` as a `TrainDeviceArray` (uploaded once, here)."""
    return _OPT_NATIVE("device_upload", None, (a,), numeric_mode, _OPT_CONTEXT)


def _all_dev(where, **named):
    if not all(_is_dev(v) for v in named.values()):  # glue: checks argument kinds
        raise TypeError(
            "mojolearn.%s: device and host arrays are mixed (%s); pass every "
            "array through train_to_device, or none"
            % (where, ", ".join(k for k in named)))  # glue: argument names for the message


def _linear_forward_dev(a, w, numeric_mode):
    _all_dev("linear_forward", a=a, weight=w)
    binding = _dev_binding(numeric_mode, "linear_forward")
    if len(w.shape) != 2 or len(a.shape) < 1 or a.shape[-1] != w.shape[1]:
        raise ValueError("mojolearn.linear_forward: a has K=%r, weight has shape %r" % (a.shape[-1:], w.shape))
    n, k = w.shape
    m = a.size // k
    c = TrainDeviceArray._new(binding, a.shape[:-1] + (n,))
    binding.linear_forward_dev([c.h, a.h, w.h], [int(m), int(n), int(k)])
    return c


def _linear_backward_dev(dc, a, w, numeric_mode):
    _all_dev("linear_backward", dc=dc, a=a, weight=w)
    binding = _dev_binding(numeric_mode, "linear_backward")
    if len(w.shape) != 2 or len(a.shape) < 1 or a.shape[-1] != w.shape[1]:
        raise ValueError("mojolearn.linear_backward: a has K=%r, weight has shape %r" % (a.shape[-1:], w.shape))
    n, k = w.shape
    m = a.size // k
    if dc.size != m * n or dc.shape[-1] != n:
        raise ValueError("mojolearn.linear_backward: dc must be (M, N)")
    da = TrainDeviceArray._new(binding, a.shape)
    dw = TrainDeviceArray._new(binding, (n, k))
    binding.linear_backward_dev([da.h, dw.h, dc.h, a.h, w.h], [int(m), int(n), int(k)])
    return da, dw


def _cross_entropy_dev(logits, targets, ignore_index, red, num_items, return_grad, label_smoothing,
                       numeric_mode):
    """`cross_entropy` on a `TrainDeviceArray` of logits: the gradient, when
    asked for, is returned as a device array; the loss and row losses are
    host values as before."""
    binding = _dev_binding(numeric_mode, "cross_entropy")
    if len(logits.shape) != 2:
        raise ValueError("mojolearn.cross_entropy: logits must be 2-D (N, V), got shape %r" % (logits.shape,))
    n_rows, vocab = logits.shape
    if not _is_buffer(targets):
        targets = Array.from_list(targets, "<i8")
    yp = probe(targets)
    if yp.ndim != 1 or not is_integer(yp.format):
        raise TypeError("mojolearn.cross_entropy: targets must be a 1-D integer array of class indices")
    y, _ = as_i32_c(targets, ndim=1, name="targets")
    if y.shape[0] != n_rows:
        raise ValueError("mojolearn.cross_entropy: logits has %d rows and targets has %d" % (n_rows, y.shape[0]))
    loss_out = zeros((1,), "<f4")
    row_out = empty((n_rows,), "<f4")
    grad = TrainDeviceArray._new(binding, logits.shape) if return_grad else None
    # `ce_loss`'s params list, word for word (see `cross_entropy`)
    plist = [int(n_rows), int(vocab), int(ignore_index), int(red),
             int(0 if num_items is None else num_items), int(1 if return_grad else 0),
             float(label_smoothing)]
    binding.ce_loss_dev(addr(loss_out, name="loss"), addr(row_out, name="row_loss"),
                        addr_ro(y, name="targets"),
                        [grad.h if grad is not None else logits.h, logits.h], plist)
    value = row_out if red == _REDUCTION_NONE else float(loss_out[0])
    if return_grad:
        return value, grad
    return value


# Through the certified GEMM at OP_NT.
def linear_forward(a, weight, numeric_mode=None):
    """Linear layer without bias: ``a @ weight.T``, as `torch.nn.Linear` with ``bias=False``.

    Parameters
    ----------
    a : array, shape (M, K)
        float32 input.
    weight : array, shape (N, K)
        float32 weight, in torch's layout.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array, shape (M, N)
    """
    return _OPT_NATIVE("linear_forward", None, (a, weight), numeric_mode, _OPT_CONTEXT)


def _optional_samba_head(binding):
    """Resolve the fused optimization only after validating the unfused entry."""
    binding.linear_forward
    try:
        return getattr(binding, "samba_head_loss", None)
    except (AttributeError, ImportError):
        return None


def samba_head_loss(a, weight, targets, num_items, numeric_mode=None):
    """`(loss, da, dweight)` of `linear_forward(a, weight)` -> `cross_entropy
    (reduction="sum", num_items=num_items, return_grad=True)` ->
    `linear_backward`, in ONE binding call whose logits and logit gradient
    stay on the device (lane/py-lm, 2026-09-28). The same kernels on the same
    operands in the same order, so the same bits. None when the loaded
    binding predates the entry (or `MOJOLEARN_HOTPATH=python` asks for the
    three-call reference arm); the caller then runs the three calls."""
    from ._buffer import hotpath_enabled
    binding = _load(numeric_mode)
    fn = _optional_samba_head(binding)
    if fn is None or not hotpath_enabled():
        return None
    a = _c32(a, "a", "samba_head_loss")
    w = _c32(weight, "weight", "samba_head_loss")
    m, k = a.reshape((-1, a.shape[-1])).shape
    n, k2 = w.shape
    if k != k2:
        raise ValueError(
            "mojolearn.linear_forward: a has K=%d, weight has K=%d" % (k, k2))
    y, _ = as_i32_c(targets, ndim=1, name="targets")
    if y.shape[0] != m:
        raise ValueError(
            "mojolearn.cross_entropy: logits has %d rows and targets has %d "
            "(python/mojolearn/_training_impl.py)" % (m, y.shape[0]))
    loss_out = zeros((1,), "<f4")
    row_out = empty((m,), "<f4")
    da = _buffers.empty((m, k), '<f4')
    dw = _buffers.empty((n, k), '<f4')
    fn([_addr(loss_out), _addr(row_out), _addr(da), _addr(dw),
        _addr_ro(a), _addr_ro(w), _addr_ro(y)],
       [int(m), int(n), int(k), int(_IGNORE_INDEX_DEFAULT), 1,
        int(0 if num_items is None else num_items), 0.0])
    return float(loss_out[0]), da, dw


# dweight's contraction is over the M tokens, the clause 9.2 contraction.
def linear_backward(dc, a, weight, numeric_mode=None):
    """Gradients ``(da, dweight)`` of `linear_forward`.

    Parameters
    ----------
    dc : array, shape (M, N)
        Gradient of the output.
    a : array, shape (M, K)
        The forward input.
    weight : array, shape (N, K)
        The forward weight.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    da : array, shape (M, K)
    dweight : array, shape (N, K)
    """
    return _OPT_NATIVE("linear_backward", None, (dc, a, weight), numeric_mode, _OPT_CONTEXT)


# ===================================================================
# THE CHUNKED LM HEAD, V2 (lane/exposure-leftovers, 2026-09-23)
# ===================================================================
# Profile `mojolearn.identical.lm_head.chunked.fp32.v2`
# (training/CHUNKED_LM_HEAD_V2.md). Both bindings have registered
# `chunked_lm_head_v2_loss` and `chunked_lm_head_v2_train`: the GPU one over
# the device kernels in training/chunked_lm_head_v2.mojo, the training host
# binding over training/checks/chunked_lm_head_oracle.mojo, which the device
# check compares those kernels against bit for bit. Until this function
# nothing in Python reached either. No arithmetic here: shapes, dtypes and the
# buffer order both bindings document.


# v2 profile (training/CHUNKED_LM_HEAD_V2.md): per row the max and the exponential denominator visit
# vocab ids 0..vocab-1 in one serial chain, d_hidden folds vocab ids ascending, every d_weight cell
# folds rows ascending.
def chunked_lm_head_loss(hidden, weight, targets, return_grad=False,
                         numeric_mode=None):
    """Mean cross-entropy of a language-model head without materializing the logits.

    Computes the mean over rows of the cross-entropy of ``hidden @ weight.T`` against
    `targets`, and optionally its gradients, without holding the (rows, vocab) logits.
    The result does not depend on how the vocabulary is chunked internally. There is no
    bias, ignore index, label smoothing or other reduction; use `linear_forward` and
    `cross_entropy` when you need those. Non-finite inputs and out-of-range targets
    raise before any output is written.

    Parameters
    ----------
    hidden : array, shape (rows, width)
        float32 hidden states.
    weight : array, shape (vocab, width)
        float32 output embedding, ``vocab >= 2``.
    targets : array, shape (rows,)
        Integer class indices in [0, vocab).
    return_grad : bool, default False
        Also return the gradients.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    loss : float
    d_hidden : array, shape (rows, width)
        Only with ``return_grad=True``.
    d_weight : array, shape (vocab, width)
        Only with ``return_grad=True``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> hidden = rng.random((64, 32), dtype=np.float32)
    >>> weight = rng.random((1000, 32), dtype=np.float32)
    >>> targets = rng.integers(0, 1000, 64)
    >>> loss, dh, dw = mojolearn.chunked_lm_head_loss(hidden, weight, targets, return_grad=True)
    """
    return _OPT_NATIVE("chunked_lm_head_loss", None, (hidden, weight, targets, return_grad), numeric_mode, _OPT_CONTEXT)


# The native NN64 switch selects packed projection versus separate projection calls.
def mlp_inference_sessions(inputs, weight1, bias1, weight2, bias2, numeric_mode="identical"):
    """Run several independent batches of rows through one Linear-ReLU-Linear model.

    Computes ``relu(x @ weight1.T + bias1) @ weight2.T + bias2`` for every session and
    returns the logits of all sessions concatenated in the order given. Every session
    has at least one row. The model is stateless (no caches, no dropout).

    Parameters
    ----------
    inputs : list of arrays
        float32 sessions, each (rows_i, width).
    weight1 : array, shape (hidden, width)
        First layer weight, in torch's layout.
    bias1 : array, shape (hidden,)
        First layer bias.
    weight2 : array, shape (out_width, hidden)
        Second layer weight.
    bias2 : array, shape (out_width,)
        Second layer bias.
    numeric_mode : {'identical', 'fast'}, default 'identical'
        Numeric mode. ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    array, shape (sum of rows_i, out_width)

    Raises
    ------
    ValueError
        When the weight, bias and session shapes do not agree.
    """
    return _OPT_NATIVE("mlp_inference_sessions", None, (inputs, weight1, bias1, weight2, bias2), numeric_mode, _OPT_CONTEXT)


# Import-time ABI connection only; finalization never invokes the selector.
from .tokenizer import _binding as _connect_optimizer_lifecycle
_optimizer_finalize_native = _connect_optimizer_lifecycle()._module.optimizer_finalize

# Package-import ABI connection; no product operation calls the Python loader.
from ._native_optimizer_bridge import connect as _connect_native_optimizer
import sys as _optimizer_sys
_OPT_NATIVE, _OPT_CONTEXT = _connect_native_optimizer(_optimizer_sys.modules[__name__])
