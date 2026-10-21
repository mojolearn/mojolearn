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


def _as_seq(x, name, where):
    """A list of arrays from either one array or an iterable of them, the way
    `torch.nn.utils.clip_grad_norm_` accepts either. DEVIATION 2402: "an
    array" means ANY object supporting the buffer protocol -- a NumPy
    array, an `array.array`, a `mojolearn.Array` -- judged by
    `_buffer.view` and not by `isinstance(x, np.ndarray)`."""
    if _is_buffer(x):
        return [x]
    try:
        out = list(x)
    except TypeError:
        raise TypeError(
            "mojolearn.%s: %s must be a float32 array (a numpy array, an "
            "array.array or a mojolearn.Array -- anything supporting the "
            "buffer protocol) or an iterable of them, got %s"
            % (where, name, type(x).__name__)
        )
    if not out:
        raise ValueError(
            "mojolearn.%s: %s is empty; there is nothing to do and an empty "
            "parameter registry is refused rather than answered "
            "(training/estimator.mojo)" % (where, name)
        )
    for i, a in enumerate(out):  # glue: checks each parameter tensor
        if not _is_buffer(a):
            raise TypeError(
                "mojolearn.%s: %s[%d] is %s, not an array (it does not "
                "support the buffer protocol)"
                % (where, name, i, type(a).__name__)
            )
    return out


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


def _nbytes(probes):
    """The byte count of every tensor, read off its buffer (no arithmetic)."""
    return [pb.nbytes for pb in probes]  # glue: one buffer size per tensor


def _addrs(arrays, writable=True):
    """The address of every tensor's buffer, for the native call."""
    if writable:
        return [addr(a, name="tensor") for a in arrays]  # glue: one address per tensor
    return [addr_ro(a, name="tensor") for a in arrays]  # glue: one address per tensor


# ===================================================================
# THE OPTIMIZERS
# ===================================================================
# Each public method checks its arguments here and makes ONE native call
# (lane neural-optimizers, 2026-10-21): the parameter traversal (the flat
# registry buffer, the write-back), the learning rate of a built-in
# schedule, the optimizer state's allocation, the choice of route (state on
# the device or on the host, the device sign flip, the device-parameter
# step) and the update itself are training/optimizer_public.mojo (the GPU
# binding) and training/host/optimizer_public_host.mojo (a CPU-only install)
# over the existing native step; the registry's offsets are
# training/optimizer_public_common.mojo through the core binding. The
# native call writes its typed results into a result buffer and this glue
# assigns them.

#: `optimizer_public_step`'s result words (training/optimizer_public_common.mojo `PublicStepOut.write`)
_STEP_OUT = (0.0,) * 10
#: `optimizer_public_step`'s gradient modes
_GRADS_HOST, _GRADS_DEVICE, _GRADS_RESIDENT = 0, 1, 2
#: `optimizer_public_io`'s operations (training/optimizer_public.mojo)
_IO_QUERY, _IO_PARAMS_UP, _IO_PARAMS_DOWN, _IO_GRADS_UP, _IO_GRADS_DOWN, _IO_PARAM_DEVICE = 0, 1, 2, 3, 4, 5
#: the optimizer's `resident=` argument as the native word
_RESIDENT_WORD = {None: -1, False: 0, True: 1}
_NO_SCHEDULE = ([-1, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])


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
        self.params = _as_seq(params, "params", where)
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
        probes = _check_dtype(self.params, "params", where, inplace=True)
        #: `offsets[j] .. offsets[j + 1]` is tensor j of the flat registry,
        #: J + 1 int32, filled natively.
        self.offsets = zeros((len(self.params) + 1,), "<i4")
        flags = array.array("q", (0,))
        self.n_total = int(_buffers._native("optimizer_registry")(  # cpu-route: the parameter registry offsets are host metadata built once with the optimizer
            _nbytes(probes), addr(self.offsets, name="offsets"), flags.buffer_info()[0]))
        #: The host mirrors of `exp_avg` (Adam's first moment, SGD's momentum
        #: buffer) and `exp_avg_sq`, allocated on first need; the state
        #: itself is the native slot's from the first step on.
        self._m = None
        self._v = None
        self._pristine = True
        if flags[0]:
            # MOJOLEARN_OPT_LAZY_MOMENTS=0: the eager pair (the A/B)
            self._host_moments()
            self._pristine = False
        #: SGD's per-tensor `buf_initialized` flag, contract 7.3b: CARRIED
        #: STATE that belongs in a checkpoint beside `exp_avg`.
        self.buf_initialized = zeros((len(self.params),), "<i4")
        #: ONE-BASED. The first step of a run is `t = 1`.
        self.t = 0
        self.packed_ = None
        #: The pre-clip global gradient norm of the most recent step, or None
        #: when that step ran with `max_norm=None`.
        self.total_norm_ = None
        #: The clamped coefficient the most recent step applied, or None.
        self.clip_coef_ = None

    # -- the state: a native slot, mirrored on the host on access -----------
    def _host_moments(self):
        """The host mirrors, allocated (zero filled) on first need."""
        if self._m is None:
            self._m = zeros((self.n_total,), "<f4")
        if self._v is None:
            self._v = zeros((self.n_total,), "<f4")
        return self._m, self._v

    def _download(self):
        """The slot's state into the mirrors: one native call, when they are stale."""
        if self._res is not None and not self._host_fresh:
            m, v = self._host_moments()
            self._res_binding.optimizer_resident_download(
                self._res, addr(m, name="exp_avg"),
                addr(v, name="exp_avg_sq"), int(self.n_total))
            self._host_fresh = True

    def _upload_one(self, m, v):
        """An assigned moment into the slot (one native call; 0 leaves the other)."""
        if self._res is not None:
            self._res_binding.optimizer_resident_upload(
                self._res, 0 if m is None else addr(m, name="exp_avg"),
                0 if v is None else addr(v, name="exp_avg_sq"), int(self.n_total))

    @property
    def exp_avg(self):
        """Adam's first moment (SGD's momentum buffer), a `mojolearn.Array`
        of `n_total` floats. Reading it copies the optimizer state into the
        returned array, so read it once per use, not per element; assign to
        `exp_avg` to change the state."""
        self._download()
        self._pristine = False
        return self._host_moments()[0]

    @exp_avg.setter
    def exp_avg(self, value):
        self._m = value
        self._pristine = False
        self._upload_one(value, None)

    @property
    def exp_avg_sq(self):
        """Adam's second moment; see `exp_avg`."""
        self._download()
        self._pristine = False
        return self._host_moments()[1]

    @exp_avg_sq.setter
    def exp_avg_sq(self, value):
        self._v = value
        self._pristine = False
        self._upload_one(None, value)

    @property
    def resident_(self):
        """True once the optimizer state lives on the device."""
        return self._res is not None and not getattr(self, "_res_host", False)

    def _binding(self):
        return _load(getattr(self, "numeric_mode", None))

    def _state_words(self):
        """The state slot's words of a native call: handle, resident,
        pristine, and the mirrors' addresses (0 when absent)."""
        return [
            -1 if self._res is None else int(self._res),
            _RESIDENT_WORD.get(self._resident_wanted, 1),
            1 if self._pristine else 0,
            0 if self._m is None else addr(self._m, name="exp_avg"),
            0 if self._v is None else addr(self._v, name="exp_avg_sq"),
        ]

    def _keep_slot(self, handle, binding, on_host):
        """Record a slot the native call opened, with the extension module
        that owns it (the module its entries belong to: on a CPU-only install
        `binding` is the host binding's stand-in), which the native finalizer
        closes it through."""
        if self._res is None and handle >= 0:
            self._res = int(handle)
            self._res_binding = binding.optimizer_resident_close.__self__
            self._res_host = bool(on_host)

    # -- lane fam2-neural (2026-10-04): parameters and gradients resident ----
    # After `params_to_device()` the parameters live in the optimizer's
    # device slot: `step(grads)` then uploads the gradient only, and
    # `step_device()` with the gradient already on the device
    # (`grads_to_device`) moves nothing. The caller's host arrays are STALE
    # until `params_to_host()` writes them back. The step is the same device
    # call on the same buffers: the same bits.
    def _io(self, op, tensors=None, name="params", tensor=0):
        """One `optimizer_public_io` call; returns its result word."""
        binding = self._binding()
        addrs, nbytes = [], []
        if tensors is not None:
            seq = _as_seq(tensors, name, self._where)
            probes = _check_dtype(seq, name, self._where, inplace=True)
            if len(seq) != len(self.params):
                raise ValueError(
                    "mojolearn.%s: %s does not match the optimizer's registry "
                    "(%d tensors, %d floats)" % (self._where, name, len(self.params), self.n_total))
            addrs, nbytes = _addrs(seq), _nbytes(probes)
        out = array.array("d", (0.0, 0.0))
        binding.optimizer_public_io(
            [op, int(tensor), len(self.params), int(self.n_total),
             addr_ro(self.offsets, name="offsets")] + self._state_words(),
            addrs, nbytes, out.buffer_info()[0])
        self._keep_slot(int(out[0]), binding, False)
        return out[1]

    def params_resident_available(self):
        """Whether `params_to_device` works on this build."""
        return bool(self._io(_IO_QUERY))

    def params_to_device(self):
        """Upload the parameters into the optimizer's device handle and keep
        them there: later steps neither upload nor download them. Call it
        again after editing the host arrays. Returns self."""
        self._io(_IO_PARAMS_UP, self.params, "params")
        self._p_dev = True
        return self

    def params_to_host(self):
        """Download the device parameters into the caller's arrays (in
        place); they stay resident. Returns the parameter list."""
        if getattr(self, "_p_dev", False):
            self._io(_IO_PARAMS_DOWN, self.params, "params")
        return self.params

    def grads_to_device(self, grads):
        """Upload a gradient into the handle's device gradient buffer, for
        `step_device()`. Returns self."""
        self._io(_IO_GRADS_UP, grads, "grads")
        self._g_dev = True
        return self

    def grads_to_host(self, grads):
        """Download the device gradient (clipped, if the last step clipped)
        into `grads`, in place. Returns `grads`."""
        self._io(_IO_GRADS_DOWN, grads, "grads")
        return grads

    def param_device(self, j):
        """Parameter tensor `j` as a `TrainDeviceArray` (a device copy of
        the resident parameters; `params_to_device()` first), for the next
        forward. Its shape is the host tensor's."""
        if not getattr(self, "_p_dev", False):
            raise RuntimeError("mojolearn.%s.param_device: call params_to_device() first" % self._where)
        shape = probe(self.params[j]).shape
        dev = _dev_binding(getattr(self, "numeric_mode", None), "param_device")
        return TrainDeviceArray(dev, int(self._io(_IO_PARAM_DEVICE, tensor=j)), shape)

    def step_device(self, grads=None, max_norm=None):
        """`step` on device-resident parameters (`params_to_device()` first).
        `grads` given: uploaded (one transfer), and scaled in place on the
        host when `max_norm` clips, as `step` does. `grads` None: the
        gradient already in the handle (`grads_to_device`) is used and, with
        `max_norm`, clipped there; nothing crosses the bus but the three
        info words. Returns what `step` returns."""
        if not getattr(self, "_p_dev", False):
            raise RuntimeError("mojolearn.%s.step_device: call params_to_device() first" % self._where)
        return self._step(grads, max_norm, "step_device", device=True)

    def __del__(self):
        return _optimizer_finalize_native(self)

    # -- the parameters every algorithm shares --------------------------
    def _config(self):
        raise NotImplementedError

    def state_dict(self):
        """Everything a resumed run needs, and nothing that is not state.

        `t`, `exp_avg`, `exp_avg_sq` and `buf_initialized`. Leaving
        `buf_initialized` out is the quiet way to break a resume: SGD's first
        step after the reload would take the COPY arm again and overwrite a
        momentum buffer it should have continued, which is the failure the
        optimizer contract's clause (d) control 2 is built to catch.
        """
        self._download()
        self._pristine = False
        m, v = self._host_moments()
        return {
            "t": int(self.t),
            "exp_avg": m.copy(),
            "exp_avg_sq": v.copy(),
            "buf_initialized": self.buf_initialized.copy(),
        }

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
        t = int(state["t"])
        ea, c1 = as_f32_c(state["exp_avg"], ndim=1, name="exp_avg")
        es, c2 = as_f32_c(state["exp_avg_sq"], ndim=1, name="exp_avg_sq")
        bi, c3 = as_i32_c(state["buf_initialized"], ndim=1,
                          name="buf_initialized")
        ea = ea if c1 else ea.copy()
        es = es if c2 else es.copy()
        for name, a in (("exp_avg", ea), ("exp_avg_sq", es)):  # glue: checks the two state arrays
            if a.size != self.n_total:
                raise ValueError(
                    "mojolearn.%s.load_state_dict: %s holds %d floats, this "
                    "optimizer's registry is %d"
                    % (self._where, name, a.size, self.n_total)
                )
        self._m, self._v = ea, es
        self._pristine = False
        self._upload_one(ea, es)
        self._host_fresh = True
        self.buf_initialized = bi if c3 else bi.copy()
        self.t = t

    def zero_grad(self, grads):
        """`torch.optim.Optimizer.zero_grad`, over arrays the caller owns.
        Present so that a training loop written against torch reads the same;
        it writes `+0.0` and nothing else -- all-zero bytes into each buffer,
        which must be C-contiguous and writable."""
        gs = _as_seq(grads, "grads", self._where)
        probes = _check_dtype(gs, "grads", self._where, inplace=True)
        self._binding().optimizer_public_zero(_addrs(gs), _nbytes(probes))

    def _schedule_words(self):
        """The built-in schedule as its native words, or None for a caller's
        own schedule object (its `lr_at` is then called here)."""
        s = self.lr_schedule
        if s is None:
            return _NO_SCHEDULE
        spec = getattr(s, "_lr_spec", None)
        return None if spec is None else spec()

    def _step(self, grads, max_norm, where, device=False, microbatches=None, tokens=-1):
        """The glue of `step`, `step_device` and `step_accumulated`: the
        argument checks, then ONE `optimizer_public_step` call, then the
        assignments of what it returns."""
        if max_norm is not None and not (float(max_norm) > 0.0):
            raise ValueError(
                "mojolearn.%s.%s: max_norm must be > 0 or None, got %r; "
                "None is how 'no clipping' is spelled "
                "(python/mojolearn/_training_impl.py)"
                % (self._where, where, max_norm)
            )
        p_dev = bool(getattr(self, "_p_dev", False))
        p_addrs, p_bytes, g_addrs, g_bytes = [], [], [], []
        g_mode, a_count = _GRADS_HOST, 1
        keep = None
        if microbatches is not None:
            a_count = len(microbatches)
            for i, part in enumerate(microbatches):  # glue: checks each microbatch argument
                gs = _as_seq(part, "microbatch_grads[%d]" % i, self._where)
                probes = _check_dtype(gs, "microbatch_grads[%d]" % i, self._where,
                                      inplace=a_count == 1)
                self._check_grads(gs, probes, where)
                if a_count > 1:
                    # read-only inputs: strided buffers are made contiguous
                    gs = [_buffers.as_f32_c(g, ndim=None, name="microbatch")[0] for g in gs]  # glue: converts each microbatch tensor buffer
                    g_addrs += _addrs(gs, writable=False)
                    keep = (keep or []) + gs
                else:
                    g_addrs += _addrs(gs)
                g_bytes += _nbytes(probes)
        elif p_dev and grads is not None and (_is_dev(grads) or (
                isinstance(grads, (list, tuple)) and len(grads) > 0 and _is_dev(grads[0]))):
            gs = [grads] if _is_dev(grads) else list(grads)
            if len(gs) != len(self.params):
                raise ValueError("mojolearn.%s.%s: %d gradients for %d parameter tensors"
                                 % (self._where, where, len(gs), len(self.params)))
            for j, g in enumerate(gs):  # glue: checks each argument is a device array
                if not _is_dev(g):
                    raise ValueError("mojolearn.%s.%s: grads[%d] must be a device array"
                                     % (self._where, where, j))
            g_mode = _GRADS_DEVICE
            g_addrs = [g.h for g in gs]  # glue: one device handle per tensor
            g_bytes = [g.size for g in gs]  # glue: one device array size per tensor
        elif p_dev and grads is None:
            g_mode = _GRADS_RESIDENT
        else:
            gs = _as_seq(grads, "grads", self._where)
            probes = _check_dtype(gs, "grads", self._where, inplace=True)
            if p_dev and len(gs) != len(self.params):
                raise ValueError(
                    "mojolearn.%s: grads does not match the optimizer's registry "
                    "(%d tensors, %d floats)" % (self._where, len(self.params), self.n_total))
            if not p_dev:
                self._check_grads(gs, probes, where)
            g_addrs, g_bytes = _addrs(gs), _nbytes(probes)
        if not p_dev:
            # Re-read every parameter buffer's layout on EVERY step: a caller
            # can reshape or resize their own array between steps.
            pprobes = _check_dtype(self.params, "params", self._where, inplace=True)
            p_addrs, p_bytes = _addrs(self.params), _nbytes(pprobes)
        sched = self._schedule_words()
        if sched is None:
            # a caller's own schedule object: its value for this step
            self.lr = float(self.lr_schedule.lr_at(self.t + 1))  # glue: the caller's schedule object, one-based next step
            sched = _NO_SCHEDULE
        if self._resident_wanted is False:
            # `resident=False`: the state lives in these host arrays (no slot)
            self._host_moments()
        cfg = self._config()
        binding = self._binding()
        out = array.array("d", _STEP_OUT)
        binding.optimizer_public_step(
            [len(self.params), int(self.t), int(self._KIND), int(cfg["nesterov"]),
             1 if getattr(self, "maximize", False) else 0] + sched[0][:4]
            + [int(self.n_total), addr_ro(self.offsets, name="offsets"),
               addr(self.buf_initialized, name="buf_initialized"), g_mode, a_count, int(tokens)]
            + self._state_words()
            + [1 if p_dev else 0, 1 if getattr(self, "_g_dev", False) else 0, 1 if device else 0],
            [float(cfg["lr"]), float(cfg["beta1"]), float(cfg["beta2"]), float(cfg["eps"]),
             float(cfg["weight_decay"]), float(cfg["momentum"]), float(cfg["dampening"]),
             0.0 if max_norm is None else float(max_norm)] + sched[1][:4],
            p_addrs, p_bytes, g_addrs, g_bytes, out.buffer_info()[0])
        del keep
        # the assignments of the native result (`PublicStepOut`)
        self.t = int(out[0])
        self._keep_slot(int(out[1]), binding, not out[2])
        self._host_fresh = self._res is None
        if out[3]:
            self.total_norm_, self.clip_coef_ = out[4], out[5]
        else:
            self.total_norm_, self.clip_coef_ = None, None
        #: The learning rate the step just used, as a float32 value.
        self.lr_ = out[6]
        if sched is not _NO_SCHEDULE:
            self.lr = out[6]
        if out[7] >= 0:
            self.packed_ = bool(out[7])
        self._g_dev = bool(out[8])
        return self.total_norm_

    def _check_grads(self, gs, probes, where):
        """`grads` matches `params` in count and shape."""
        if len(gs) != len(self.params):
            raise ValueError(
                "mojolearn.%s.%s: %d gradients for %d parameter tensors"
                % (self._where, where, len(gs), len(self.params))
            )
        for j, (p, gp) in enumerate(zip(self.params, probes)):  # glue: checks each parameter tensor shape
            pp = probe(p)
            if pp.shape != gp.shape:
                raise ValueError(
                    "mojolearn.%s.%s: grads[%d] has shape %r, params[%d] "
                    "has %r" % (self._where, where, j, gp.shape, j, pp.shape)
                )

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
        return self._step(grads, max_norm, "step")

    def step_accumulated(self, microbatch_grads, tokens, max_norm=None):
        """One step from `A = self.accumulation_steps` microbatch gradient
        lists, combined on device by the clause 9.2 balanced tree in
        ascending microbatch index, then `step`. `tokens` is the FULL step's
        token count T; a split that clause 9.2 does not admit is refused by
        name (no gradient is combined). Returns what `step` returns."""
        parts = list(microbatch_grads)
        if len(parts) != self.accumulation_steps:
            raise ValueError(
                "mojolearn.%s.step_accumulated: %d microbatch gradient lists "
                "for accumulation_steps=%d; the count is part of the run's "
                "numerical specification (optimizer contract clause 9.2) "
                "(python/mojolearn/_training_impl.py)"
                % (self._where, len(parts), self.accumulation_steps)
            )
        t_tokens = -1 if tokens is None else int(tokens)
        if len(parts) > 1 and tokens is not None and t_tokens < 1:
            raise ValueError("mojolearn.accumulate_grads: tokens must be >= 1")
        return self._step(None, max_norm, "step_accumulated", microbatches=parts, tokens=t_tokens)


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
        A schedule with a one-based `lr_at(t)` method; it sets `lr` before every step.
        Anything without `lr_at` raises `TypeError`.
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
    `lr_schedule` (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, or any object with
    a one-based `lr_at(t)`). `step_accumulated(microbatch_grads, tokens)` combines
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
                 accumulation_steps=1, maximize=False, resident=None, **kwargs):
        _refuse_unknown(kwargs, "SGD")
        super().__init__(params, "SGD", lr_schedule, accumulation_steps,
                         resident=resident)
        if nesterov and (momentum == 0.0 or dampening != 0.0):
            raise ValueError(
                "mojolearn.SGD: nesterov=True needs momentum > 0 and "
                "dampening == 0, got momentum=%r dampening=%r. That is "
                "torch.optim.SGD's own refusal and it is repeated here "
                "because the profile would ACCEPT this combination and "
                "silently degenerate: at momentum = 0 the Nesterov reading "
                "is bit-identical to the plain one, so nothing would tell "
                "you the flag did nothing "
                "(python/mojolearn/_training_impl.py)"
                % (momentum, dampening)
            )
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.dampening = float(dampening)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.maximize = _maximize_flag(maximize, "SGD")

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
        A schedule with a one-based `lr_at(t)` method; it sets `lr` before every step.
        Anything without `lr_at` raises `TypeError`.
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
    `lr_schedule` (`ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR`, or any object with
    a one-based `lr_at(t)`). `step_accumulated(microbatch_grads, tokens)` combines
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
                 maximize=False, resident=None, **kwargs):
        _refuse_unknown(kwargs, type(self).__name__)
        super().__init__(params, type(self).__name__, lr_schedule,
                         accumulation_steps, resident=resident)
        try:
            b1, b2 = betas
        except (TypeError, ValueError):
            raise ValueError(
                "mojolearn.%s: betas must be a pair (beta1, beta2), got %r "
                "(python/mojolearn/_training_impl.py)"
                % (type(self).__name__, betas)
            )
        self.lr = float(lr)
        self.betas = (float(b1), float(b2))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.maximize = _maximize_flag(maximize, type(self).__name__)

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
                 maximize=False, resident=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, lr_schedule=lr_schedule,
                         accumulation_steps=accumulation_steps,
                         maximize=maximize, resident=resident, **kwargs)


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
    _refuse_unknown(kwargs, "clip_grad_norm_")
    if float(norm_type) != 2.0:
        raise NotImplementedError(
            "mojolearn.clip_grad_norm_: norm_type=%r is REFUSED BY NAME. The "
            "profile has ONE norm, the two-level L2 of optimizer contract "
            "3.1; an L-inf or L1 clip is a different reduction with its own "
            "clause, its own fixture and its own sabotage, and none of them "
            "exist (python/mojolearn/_training_impl.py)" % (norm_type,)
        )
    if not error_if_nonfinite:
        raise NotImplementedError(
            "mojolearn.clip_grad_norm_: error_if_nonfinite=False is REFUSED "
            "BY NAME. The profile ALWAYS refuses a non-finite total norm "
            "(optimizer contract 8a, refuse_nonfinite_scalar in "
            "training/checks/optimizer_oracle.mojo) and there is no build in "
            "which it does not, so accepting False would be accepting a "
            "promise this surface cannot keep. Skipping a step whose gradient "
            "norm is non-finite is a useful thing to do and the contract says "
            "where it belongs: OUTSIDE the pinned region, as an explicit "
            "recorded branch in your own loop "
            "(python/mojolearn/_training_impl.py)"
        )
    gs = _as_seq(grads, "grads", "clip_grad_norm_")
    probes = _check_dtype(gs, "grads", "clip_grad_norm_", inplace=True)
    mn = float(max_norm)
    if not (mn > 0.0):
        raise ValueError(
            "mojolearn.clip_grad_norm_: max_norm must be > 0, got %r; "
            "reaching this function IS the clip running, and 'no clipping' is "
            "spelled by not calling it (training/estimator.mojo)" % (max_norm,)
        )

    # One native call (lane neural-optimizers): the registry, the route
    # (the tensors straight from their own memory, or the packed buffer under
    # MOJOLEARN_CLIP_PACKED=1 or several optimizer devices; the same device
    # buffer, offsets and clip, so the same bits) and the in-place scale are
    # training/optimizer_public.mojo `public_clip`.
    out = array.array("d", (0.0, 0.0))
    _load(numeric_mode).optimizer_public_clip(_addrs(gs), _nbytes(probes), mn, out.buffer_info()[0])
    return out[0]


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
    _refuse_unknown(kwargs, "cross_entropy")
    if reduction not in _REDUCTIONS:
        raise ValueError(
            "mojolearn.cross_entropy: reduction=%r; it must be 'none', 'sum' "
            "or 'mean' (python/mojolearn/_training_impl.py)" % (reduction,)
        )
    red = _REDUCTIONS[reduction]
    if return_grad and red == _REDUCTION_NONE:
        raise NotImplementedError(
            "mojolearn.cross_entropy: reduction='none' has NO BACKWARD and "
            "return_grad=True is refused with it (loss contract section 11). "
            "A per-row upstream vector adds a product per cell whose "
            "placement, before or after seam L16's division, is a real "
            "decision with two different answers, and this lane has no caller "
            "for it (training/estimator.mojo)"
        )

    # DEVIATION 2405: `logits` and `targets` are read through the buffer
    # protocol; `logits` is refused unless it is a float32 buffer (as
    # before: a float64 array was refused, not cast) and `targets` may be
    # any integer buffer or a plain list of ints, converted to int32 the
    # way `np.ascontiguousarray(y, dtype=np.int32)` converted it.
    if _is_dev(logits):
        # lane fam2-neural: the logits are on the device and the gradient stays there
        return _cross_entropy_dev(logits, targets, ignore_index, red, num_items, return_grad,
                                  label_smoothing, numeric_mode)
    try:
        xp = probe(logits)
    except TypeError:
        raise TypeError(
            "mojolearn.cross_entropy: logits must be a float32 array (a "
            "numpy array, an array.array or a mojolearn.Array -- anything "
            "supporting the buffer protocol), got %s "
            "(python/mojolearn/_training_impl.py)" % (type(logits).__name__,)
        ) from None
    if xp.ndim != 2:
        raise ValueError(
            "mojolearn.cross_entropy: logits must be 2-D (N, V), got %d-D "
            "shape %r (python/mojolearn/_training_impl.py)"
            % (xp.ndim, xp.shape)
        )
    _check_dtype([logits], "logits", "cross_entropy", inplace=False)
    x, _ = as_f32_c(logits, ndim=2, name="logits")
    n_rows, vocab = x.shape

    if not _is_buffer(targets):
        try:
            targets = Array.from_list(targets, "<i8")
        except Exception:
            raise TypeError(
                "mojolearn.cross_entropy: targets must be an integer array "
                "of class INDICES (a numpy array, an array.array, a "
                "mojolearn.Array or a list of ints), got %s "
                "(python/mojolearn/_training_impl.py)"
                % (type(targets).__name__,)
            ) from None
    yp = probe(targets)
    if yp.ndim != 1:
        raise ValueError(
            "mojolearn.cross_entropy: targets must be 1-D (N,), got %d-D "
            "shape %r; class-PROBABILITY targets are a different forward with "
            "a different backward and are not in the contract "
            "(python/mojolearn/_training_impl.py)" % (yp.ndim, yp.shape)
        )
    if not is_integer(yp.format):
        raise TypeError(
            "mojolearn.cross_entropy: targets has dtype %s; this surface "
            "takes class INDICES only, not probabilities "
            "(python/mojolearn/_training_impl.py)"
            % (dtype_name(targets, yp),)
        )
    y, _ = as_i32_c(targets, ndim=1, name="targets")
    if y.shape[0] != n_rows:
        raise ValueError(
            "mojolearn.cross_entropy: logits has %d rows and targets has %d "
            "(python/mojolearn/_training_impl.py)" % (n_rows, y.shape[0])
        )

    eps = float(label_smoothing)
    # The [0, 1) bound and the finiteness check are `ce_refuse_inputs`'s and
    # fire from Mojo by name; this is only the shape work.

    loss_out = zeros((1,), "<f4")
    row_out = empty((n_rows,), "<f4")
    if return_grad:
        grad_out = empty((n_rows, vocab), "<f4")
    else:
        # NEVER a null address: `_f32_ptr` in the binding refuses one, and a
        # one-element placeholder is what the certified entry documents for
        # an unused output buffer.
        grad_out = zeros((1,), "<f4")

    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_training.mojo::ce_loss_binding`):
    #
    #     0  n_rows
    #     1  vocab
    #     2  ignore_index      (torch's default is -100)
    #     3  reduction         0 = none, 1 = sum, 2 = mean
    #     4  num_items         < 1 means "not supplied", the MEAN arm's own
    #                          default
    #     5  want_grad         0 = forward only, 1 = also write dlogits
    #     6  label_smoothing   (float; the contract's `eps`)
    plist = [
        int(n_rows),
        int(vocab),
        int(ignore_index),
        int(red),
        int(0 if num_items is None else num_items),
        int(1 if return_grad else 0),
        eps,
    ]

    binding = _load(numeric_mode)
    binding.ce_loss(
        addr(loss_out, name="loss"),
        addr(row_out, name="row_loss"),
        addr(grad_out, name="dlogits"),
        addr_ro(x, name="logits"),
        addr_ro(y, name="targets"),
        plist,
    )

    if red == _REDUCTION_NONE:
        value = row_out
    else:
        value = float(loss_out[0])
    if return_grad:
        return value, grad_out
    return value


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
    return _backend._CODE_MODE.get(
        _load(numeric_mode).training_numeric_mode(), "unknown")


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
    return _backend.read_vendor(_load(numeric_mode))


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
# `lr_values(n)` are each ONE call of the core binding's `lr_schedule_fill`
# (sequence/lr_table.mojo `lr_spec_fill`): the binary64 table with its
# rigorous error bounds, the float32 decision, the exact big-rational route
# for every step the table leaves undecided, the constant tail and, for
# OneCycleLR, the phases. MOJOLEARN_LR_TABLE=0 (or MOJOLEARN_LR_EXACT_ONLY=1)
# keeps every step on the exact route (the A/B); the answers are the same.

#: `lr_schedule_fill`'s statuses beyond `_LR_ERRORS`: a step below 1, and a
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


def _lr_fill(spec, t0, out):
    """ONE native call: the float32 rates of steps t0 .. t0 + len(out) - 1
    into the float32 array `out`. Returns (status, index of the failing
    step)."""
    ip, fp = spec
    res = array.array("q", (0,))
    st = _lr_native("lr_schedule_fill")(  # cpu-route: a learning-rate schedule value is one host scalar per optimizer step (exact big-rational route)
        ip + [int(t0), len(out)], fp, out.buffer_info()[0], res.buffer_info()[0])
    return int(st), res[0]


class _LrTable(object):
    """`lr_at`, `lr_values` and `bits_at` of a built-in schedule, one native
    call each. A subclass provides `_lr_spec()` (its `LrSpec` words: four
    ints and four floats, sequence/lr_table.mojo) and `_lr_refuse(status,
    t)` (its refusal of step t)."""

    def _lr_refuse(self, st, t):
        if st == _LR_ONE_BASED:
            raise ValueError("mojolearn schedule: step t is ONE-BASED, got %d" % t)
        _lr_status(st)

    def lr_at(self, t):
        """The float32 learning rate of ONE-BASED step `t`, as a float."""
        t = int(t)
        out = array.array("f", (0.0,))
        st, _ = _lr_fill(self._lr_spec(), t, out)
        if st:
            self._lr_refuse(st, t)
        return out[0]

    def lr_values(self, n):
        """lr_at(1) .. lr_at(n) as an array('f'), in one native call."""
        out = array.array("f", bytes(4 * max(int(n), 0)))  # glue: the result array of n float32 words
        st, i = _lr_fill(self._lr_spec(), 1, out)
        if st:
            self._lr_refuse(st, 1 + i)  # glue: the failing step's number for the message
        return out

    def bits_at(self, t):
        """`lr_at(t)` as its float32 bit pattern, for fixtures."""
        return memoryview(array.array("f", (self.lr_at(t),))).cast("B").cast("I")[0]


class _Schedule(_LrTable):
    """Shared half of the schedules. `t` is the optimizer's ONE-BASED step
    counter, so `lr_at(1)` is the first step's learning rate."""

    kind = ""
    #: `LrSpec` kind codes (sequence/lr_table.mojo)
    _KIND_CODE = {"constant": 0, "linear": 1, "cosine": 2}

    def __init__(self, peak_lr, warmup_steps=0, total_steps=None, min_lr=0.0):
        self.peak_lr = float(_round_f32(peak_lr))
        self.min_lr = float(_round_f32(min_lr))
        self.warmup_steps = int(warmup_steps)
        self.total_steps = None if total_steps is None else int(total_steps)
        for name, v in (("peak_lr", self.peak_lr), ("min_lr", self.min_lr)):  # glue: validates two scalar arguments
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    "mojolearn.%s: %s must be finite and >= 0, got %r"
                    % (type(self).__name__, name, v))
        if self.warmup_steps < 0:
            raise ValueError("mojolearn.%s: warmup_steps must be >= 0"
                             % type(self).__name__)
        if (self.total_steps is not None
                and self.total_steps < max(1, self.warmup_steps)):
            raise ValueError(
                "mojolearn.%s: total_steps must be >= max(1, warmup_steps), "
                "got %r" % (type(self).__name__, self.total_steps))

    def config(self):
        return {"kind": self.kind, "peak_lr": self.peak_lr,
                "min_lr": self.min_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps}

    @staticmethod
    def from_config(cfg):
        cls = {"constant": ConstantLR, "linear": WarmupLinearLR,
               "cosine": WarmupCosineLR}[cfg["kind"]]
        return cls(cfg["peak_lr"], warmup_steps=cfg["warmup_steps"],
                   total_steps=cfg["total_steps"], min_lr=cfg["min_lr"])

    def _lr_spec(self):
        return ([self._KIND_CODE[self.kind], int(self.warmup_steps),
                 -1 if self.total_steps is None else int(self.total_steps), 0],
                [float(self.peak_lr), float(self.min_lr), 0.0, 0.0])

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
        super(ConstantLR, self).__init__(peak_lr, warmup_steps, None, 0.0)


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
    binding = _load(numeric_mode)
    return bool(int(binding.accumulation_is_aligned(
        [int(tokens), int(accumulation_steps)])))


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
    parts = list(microbatch_grads)
    if not parts:
        raise ValueError("mojolearn.accumulate_grads: no microbatches")
    single = _is_buffer(parts[0])
    seqs = [_as_seq(p, "microbatch_grads[%d]" % i, "accumulate_grads")
            for i, p in enumerate(parts)]  # glue: wraps each microbatch argument
    probes = [_check_dtype(sq, "microbatch_grads[%d]" % i, "accumulate_grads", inplace=False)
              for i, sq in enumerate(seqs)]  # glue: probes each microbatch argument
    for i, sq in enumerate(seqs):  # glue: checks microbatch tensor shapes
        if len(sq) != len(seqs[0]):
            raise ValueError(
                "mojolearn.accumulate_grads: microbatch %d has %d tensors, "
                "microbatch 0 has %d" % (i, len(sq), len(seqs[0])))
        for j, (a, b) in enumerate(zip(probes[i], probes[0])):  # glue: checks microbatch tensor shapes
            if a.shape != b.shape:
                raise ValueError(
                    "mojolearn.accumulate_grads: microbatch %d tensor %d has "
                    "shape %r, microbatch 0 has %r" % (i, j, a.shape, b.shape))
    t_tokens = -1 if tokens is None else int(tokens)
    if tokens is not None and t_tokens < 1:
        raise ValueError("mojolearn.accumulate_grads: tokens must be >= 1")
    # Inputs are read-only: strided buffers are made contiguous, without the
    # writable/contiguous restriction of in-place optimizer parameters.
    contiguous = [[_buffers.as_f32_c(a, ndim=None, name='microbatch')[0] for a in sq]  # glue: converts each microbatch tensor buffer
                  for sq in seqs]  # glue: converts each microbatch tensor buffer
    result = [empty(pb.shape, "<f4") for pb in probes[0]]  # glue: allocates each result tensor
    # One native call: the [A, N] stack, the clause 9.2 tree and the result
    # tensors (training/optimizer_public.mojo `public_accumulate`).
    _load(numeric_mode).optimizer_public_accumulate(
        [addr_ro(a, name="microbatch") for sq in contiguous for a in sq],  # glue: one address per microbatch tensor
        _nbytes(probes[0]), _addrs(result), len(seqs), t_tokens)
    return result[0] if single else result


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
    w = _c32(weight, "weight", "embedding_forward")
    if w.ndim != 2:
        raise ValueError("mojolearn.embedding_forward: weight must be (V, D)")
    ids = as_i32_c(ids, ndim=None, name="ids")[0].reshape((-1,))
    n, (v, d) = int(ids.size), w.shape
    y = _buffers.empty((n, d), '<f4')
    _load(numeric_mode).embedding_forward(
        [_addr(y), _addr_ro(w), _addr_ro(ids)], [n, int(v), int(d)])
    return y


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
    dy = _c32(dy, "dy", "embedding_backward")
    if dy.ndim != 2:
        raise ValueError("mojolearn.embedding_backward: dy must be (N, D)")
    ids = as_i32_c(ids, ndim=None, name="ids")[0].reshape((-1,))
    n, d = dy.shape
    if ids.size != n:
        raise ValueError(
            "mojolearn.embedding_backward: ids and dy row counts differ")
    dw = _buffers.empty((int(vocab), d), '<f4')
    _load(numeric_mode).embedding_backward(
        [_addr(dw), _addr_ro(dy), _addr_ro(ids)],
        [int(n), int(vocab), int(d)])
    return dw


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
    x = _c32(x, "x", "rms_norm_forward")
    w = _c32(weight, "weight", "rms_norm_forward")
    m, d = x.reshape((-1, x.shape[-1])).shape
    if w.shape != (d,):
        raise ValueError("mojolearn.rms_norm_forward: weight must be (D,)")
    y = _buffers.empty((m, d), '<f4')
    _load(numeric_mode).rms_norm_forward(
        [_addr(y), _addr_ro(x), _addr_ro(w)], [int(m), int(d), float(eps)])
    return y.reshape(x.shape)


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
    x = _c32(x, "x", "rms_norm_backward")
    dy = _c32(dy, "dy", "rms_norm_backward")
    w = _c32(weight, "weight", "rms_norm_backward")
    if dy.shape != x.shape:
        raise ValueError("mojolearn.rms_norm_backward: dy and x shapes differ")
    m, d = x.reshape((-1, x.shape[-1])).shape
    dx = _buffers.empty((m, d), '<f4')
    dw = _buffers.empty((d,), '<f4')
    _load(numeric_mode).rms_norm_backward(
        [_addr(dx), _addr(dw), _addr_ro(dy), _addr_ro(x), _addr_ro(w)],
        [int(m), int(d), float(eps)])
    return dx.reshape(x.shape), dw


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
        self._b = binding
        self.h = int(handle)
        self.shape = tuple(shape)
        self.size = nelems(self.shape)
        self._base = base  # a reshaped view keeps its owner alive

    @staticmethod
    def _new(binding, shape):
        shape = tuple(shape)
        return TrainDeviceArray(binding, binding.train_dev_alloc(int(nelems(shape))), shape)

    def reshape(self, shape):
        shape = tuple(shape)
        if nelems(shape) != self.size:
            raise ValueError("mojolearn.TrainDeviceArray.reshape: %r does not hold %d floats" % (shape, self.size))
        return TrainDeviceArray(self._b, self.h, shape, base=self if self._base is None else self._base)

    def to_host(self):
        """The array as a host `mojolearn.Array` (one download)."""
        out = _buffers.empty(self.shape, '<f4')
        self._b.train_dev_get(self.h, _addr(out), int(self.size))
        return out

    def __del__(self):
        if getattr(self, "_base", None) is None and getattr(self, "_b", None) is not None:
            try:
                self._b.train_dev_free(self.h)
            except Exception:
                pass


def _is_dev(x):
    return isinstance(x, TrainDeviceArray)


def train_to_device(a, numeric_mode=None):
    """float32 array `a` as a `TrainDeviceArray` (uploaded once, here)."""
    if _is_dev(a):
        return a
    binding = _dev_binding(numeric_mode, "train_to_device")
    a = _c32(a, "a", "train_to_device")
    out = TrainDeviceArray._new(binding, a.shape)
    binding.train_dev_put(out.h, _addr_ro(a), int(out.size))
    return out


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
    if _is_dev(a) or _is_dev(weight):
        return _linear_forward_dev(a, weight, numeric_mode)
    a = _c32(a, "a", "linear_forward")
    w = _c32(weight, "weight", "linear_forward")
    m, k = a.reshape((-1, a.shape[-1])).shape
    n, k2 = w.shape
    if k != k2:
        raise ValueError(
            "mojolearn.linear_forward: a has K=%d, weight has K=%d" % (k, k2))
    c = _buffers.empty((m, n), '<f4')
    _load(numeric_mode).linear_forward(
        [_addr(c), _addr_ro(a), _addr_ro(w)], [int(m), int(n), int(k)])
    return c.reshape(a.shape[:-1] + (n,))


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
    if _is_dev(dc) or _is_dev(a) or _is_dev(weight):
        return _linear_backward_dev(dc, a, weight, numeric_mode)
    a = _c32(a, "a", "linear_backward")
    w = _c32(weight, "weight", "linear_backward")
    dc = _c32(dc, "dc", "linear_backward")
    m, k = a.reshape((-1, a.shape[-1])).shape
    n = w.shape[0]
    if dc.reshape((-1, dc.shape[-1])).shape != (m, n):
        raise ValueError("mojolearn.linear_backward: dc must be (M, N)")
    da = _buffers.empty((m, k), '<f4')
    dw = _buffers.empty((n, k), '<f4')
    _load(numeric_mode).linear_backward(
        [_addr(da), _addr(dw), _addr_ro(dc), _addr_ro(a), _addr_ro(w)],
        [int(m), int(n), int(k)])
    return da.reshape(a.shape), dw


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
    where = "training.chunked_lm_head_loss"
    h = _c32(hidden, "hidden", where)
    w = _c32(weight, "weight", where)
    if h.ndim != 2 or w.ndim != 2:
        raise ValueError(
            "mojolearn.%s: hidden must be (rows, width) and weight "
            "(vocab, width)" % where)
    rows, width = (int(d) for d in h.shape)  # glue: two shape dimensions only
    vocab, width_w = (int(d) for d in w.shape)  # glue: two shape dimensions only
    if width != width_w:
        raise ValueError(
            "mojolearn.%s: hidden has width %d, weight has width %d"
            % (where, width, width_w))
    if vocab < 2:
        raise ValueError("mojolearn.%s: vocab must be at least 2, got %d"
                         % (where, vocab))
    t = as_i32_c(targets, ndim=None, name="targets")[0].reshape((-1,))
    if int(t.size) != rows:
        raise ValueError(
            "mojolearn.%s: %d targets for %d rows" % (where, int(t.size), rows))
    loss = zeros((1,), "<f4")
    row_max = empty((rows,), "<f4")
    row_denom = empty((rows,), "<f4")
    plist = [rows, vocab, width]
    binding = _load(numeric_mode)
    if not return_grad:
        binding.chunked_lm_head_v2_loss(
            [_addr(loss), _addr(row_max), _addr(row_denom),
             _addr_ro(h), _addr_ro(w), _addr_ro(t)], plist)
        return float(loss[0])
    d_hidden = empty((rows, width), "<f4")
    d_weight = empty((vocab, width), "<f4")
    binding.chunked_lm_head_v2_train(
        [_addr(loss), _addr(row_max), _addr(row_denom), _addr(d_hidden),
         _addr(d_weight), _addr_ro(h), _addr_ro(w), _addr_ro(t)], plist)
    return float(loss[0]), d_hidden, d_weight


# The native NN64 switch selects packed projection versus separate projection calls.
def mlp_inference_sessions(inputs, weight1, bias1, weight2, bias2, numeric_mode="identical"):
    """Run several independent batches of rows through one Linear-ReLU-Linear model.

    Computes ``relu(x @ weight1.T + bias1) @ weight2.T + bias2`` for every session and
    returns the logits of all sessions concatenated in the order given. Empty sessions
    are allowed. The model is stateless (no caches, no dropout).

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
    w1 = _c32(weight1, "weight1", "mlp_inference_sessions")
    b1 = _c32(bias1, "bias1", "mlp_inference_sessions")
    w2 = _c32(weight2, "weight2", "mlp_inference_sessions")
    b2 = _c32(bias2, "bias2", "mlp_inference_sessions")
    if len(w1.shape) != 2 or len(w2.shape) != 2:
        raise ValueError("MLP session weights must be matrices")
    hidden, width = w1.shape
    out_width, hidden2 = w2.shape
    if hidden2 != hidden or b1.shape != (hidden,) or b2.shape != (out_width,):
        raise ValueError("MLP session weight/bias shapes do not agree")
    # API objects and their shape metadata only; row values stay in buffers.
    sessions = [_c32(x, "session input", "mlp_inference_sessions") for x in inputs]  # glue: converts caller session arrays
    shapes = [x.shape for x in sessions]  # glue: session shape tuples only
    native = _load(numeric_mode)
    # The binding checks every (rows, width) shape and returns the total
    # rows, so the output size is Mojo's number, not a Python sum.
    try:
        total_rows = int(native.mlp_sessions_rows(shapes, width))
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    output = _buffers.empty((total_rows, out_width), '<f4')
    addrs = [_addr_ro(x) for x in sessions]  # glue: session buffer addresses
    wrote = int(native.mlp_sessions(
        [_addr(output), _addr_ro(w1), _addr_ro(b1), _addr_ro(w2), _addr_ro(b2)],
        addrs, shapes, [width, hidden, out_width]))
    if wrote != output.size:
        raise RuntimeError("MLP sessions returned an invalid output count")
    return output


# Import-time ABI connection only; finalization never invokes the selector.
from .tokenizer import _binding as _connect_optimizer_lifecycle
_optimizer_finalize_native = _connect_optimizer_lifecycle()._module.optimizer_finalize
