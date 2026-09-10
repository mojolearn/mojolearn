# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Neural-network training on the GPU: SGD, Adam, AdamW, global-norm
gradient clipping and cross-entropy loss, over float32 buffers.

THE ARRAYS ARE BUFFERS, NOT NUMPY (numpy-free-0.7, DEVIATIONS 2402-2406).
Every parameter, gradient, logits and targets argument is read through the
buffer protocol (`_buffer.view`): a NumPy array, an `array.array`, a
`mojolearn.Array` or any other exporter is accepted, and NumPy is not
used anywhere on the OPTIMIZER, CLIP and LOSS path. It IS still
imported for the four sections added after numpy-free-0.7 was cut --
the LR schedules, gradient accumulation, the `Generator` RNG and the
embedding/rms_norm/linear layer helpers -- which have not been
converted yet (see docs/lanes/NUMPY_FREE_RESIDUAL_2026-09-10.md). Everything this module ALLOCATES -- the
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

import math
from fractions import Fraction

import numpy as np

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, zeros
from ._bufcheck import (
    base_format, dtype_name, is_integer, is_native_f32, memcopy, memzero,
    nelems, probe,
)
from ._mode import NumericModeMixin

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
    NAME lookup and not a boolean, because the middle tier reports 2 and the
    old spelling called that "fast", so a deterministic binary matched a fast
    request and the cross-check passed on the wrong arm.
    """
    mode = (mode or _backend.default_mode()).strip().lower()
    mod = _backend.binding(_EXT_NAME, mode)
    compiled = _backend._CODE_MODE.get(mod.training_numeric_mode(), "unknown")
    if compiled != mode:
        raise ImportError(
            "mojolearn: %s was compiled %s but numeric_mode asked for %s -- "
            "a binary is in the wrong directory; rebuild the sets with\n    "
            "bash bindings/build_training.sh\n    "
            "MOJOLEARN_NUMERIC_MODE=deterministic bash "
            "bindings/build_training.sh\n    "
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
    for i, a in enumerate(out):
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
    for i, a in enumerate(arrays):
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


def _offsets_for(probes):
    """`offsets[0 .. J]` as int32. `offsets[0]` is 0 and the entries are
    strictly ascending; `training/estimator.mojo::_offsets_from_ptr` refuses
    anything else BY NAME before any device work."""
    off = [0]
    for pb in probes:
        off.append(off[-1] + nelems(pb.shape))
    return Array.from_list(off, "<i4")


def _pack(arrays, probes):
    """One flat C-contiguous float32 buffer holding every tensor in order.

    Returns `(flat, packed)`. `packed` is False for the ZERO-COPY case: a
    single tensor -- already float32 and C-contiguous, `_check_dtype`
    guaranteed both -- is borrowed as-is, and the device writes the
    caller's buffer directly. Anything else is copied into a fresh
    `mojolearn.Array` by `memmove` (DEVIATION 2403), and that copy is what
    the `packed_` attribute on each optimizer reports.
    """
    if len(arrays) == 1:
        return arrays[0], False
    total = sum(nelems(pb.shape) for pb in probes)
    flat = empty((total,), "<f4")
    base = addr(flat, name="flat")
    at = 0
    for a, pb in zip(arrays, probes):
        memcopy(base + at, addr_ro(a, name="tensor"), pb.nbytes)
        at += pb.nbytes
    return flat, True


def _unpack_into(flat, arrays, probes):
    """Write a flat buffer back into the caller's arrays, IN PLACE, so that
    an optimizer step is visible on the objects the caller handed in -- which
    is what `torch.optim` does and what a trailing underscore means.

    DEVIATION 2403: a `memmove` per tensor from the flat buffer into the
    caller's own memory. The destination address comes from `_buffer.addr`
    (WRITABLE required -- it refuses a read-only exporter by name), never
    `addr_ro`, and `_check_dtype(inplace=True)` has already required
    C-contiguity, so one contiguous copy per tensor is exactly the write.
    """
    src = addr_ro(flat, name="flat")
    at = 0
    for a, pb in zip(arrays, probes):
        memcopy(addr(a, name="tensor"), src + at, pb.nbytes)
        at += pb.nbytes


# ===================================================================
# THE OPTIMIZERS
# ===================================================================


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

    def __init__(self, params, where, lr_schedule=None, accumulation_steps=1):
        self._where = where
        self.params = _as_seq(params, "params", where)
        #: A schedule object with `lr_at(t)` (t ONE-BASED), or None. When set,
        #: `step` overwrites `self.lr` with `lr_at(self.t)` before the call,
        #: so the learning rate is a pure function of (step, config).
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
        self.offsets = _offsets_for(probes)
        self.n_total = sum(nelems(pb.shape) for pb in probes)
        #: `m` is Adam's `exp_avg` and SGD's `momentum_buffer`; `v` is Adam's
        #: `exp_avg_sq` and is UNREAD BY SGD. Both are allocated at `N`
        #: whatever the algorithm, because the certified entry takes one
        #: signature for both and one state pair means a checkpoint has one
        #: shape (optimizer contract, `optimizer_step_oracle`). Both are
        #: `mojolearn.Array` (DEVIATION 2404).
        self.exp_avg = zeros((self.n_total,), "<f4")
        self.exp_avg_sq = zeros((self.n_total,), "<f4")
        #: SGD's per-tensor `buf_initialized` flag, contract 7.3b. 0 means
        #: this tensor's momentum buffer has never been written, so the first
        #: step COPIES the gradient into it instead of running the
        #: recurrence. It is CARRIED STATE and belongs in a checkpoint beside
        #: `exp_avg`; the flag flips once per TENSOR after that tensor's
        #: kernel, never per element.
        self.buf_initialized = zeros((len(self.params),), "<i4")
        #: ONE-BASED. The first step of a run is `t = 1`, and
        #: `training/estimator.mojo` refuses `t < 1` by name because at
        #: `t = 0` the bias correction `1 - beta^0` is exactly zero and the
        #: host scalars divide by it.
        self.t = 0
        self.packed_ = None
        #: The pre-clip global gradient norm of the most recent step, or None
        #: when that step ran with `max_norm=None`.
        self.total_norm_ = None
        #: The clamped coefficient the most recent step applied, or None.
        self.clip_coef_ = None

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
        return {
            "t": int(self.t),
            "exp_avg": self.exp_avg.copy(),
            "exp_avg_sq": self.exp_avg_sq.copy(),
            "buf_initialized": self.buf_initialized.copy(),
        }

    def load_state_dict(self, state):
        """DEVIATION 2404: the three arrays are taken through
        `_buffer.as_f32_c` / `as_i32_c` (any buffer or nested list; a
        float64 `exp_avg` is converted the way `np.ascontiguousarray(...,
        dtype=np.float32)` converted it) and the optimizer then OWNS a copy
        -- the NumPy spelling aliased a caller's float32 C-order array and
        this one does not, so a later `step` can never write into the
        dict you loaded from."""
        self.t = int(state["t"])
        ea, c1 = as_f32_c(state["exp_avg"], ndim=1, name="exp_avg")
        es, c2 = as_f32_c(state["exp_avg_sq"], ndim=1, name="exp_avg_sq")
        bi, c3 = as_i32_c(state["buf_initialized"], ndim=1,
                          name="buf_initialized")
        self.exp_avg = ea if c1 else ea.copy()
        self.exp_avg_sq = es if c2 else es.copy()
        self.buf_initialized = bi if c3 else bi.copy()
        if self.exp_avg.size != self.n_total:
            raise ValueError(
                "mojolearn.%s.load_state_dict: exp_avg holds %d floats, this "
                "optimizer's registry is %d"
                % (self._where, self.exp_avg.size, self.n_total)
            )

    def zero_grad(self, grads):
        """`torch.optim.Optimizer.zero_grad`, over arrays the caller owns.
        Present so that a training loop written against torch reads the same;
        it writes `+0.0` and nothing else -- as all-zero bytes through
        `memset` on the buffer's own address (DEVIATION 2406), which is
        `+0.0` in float32 and needs the buffer C-contiguous and writable."""
        gs = _as_seq(grads, "grads", self._where)
        probes = _check_dtype(gs, "grads", self._where, inplace=True)
        for g, pb in zip(gs, probes):
            memzero(addr(g, name="grads"), pb.nbytes)

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
        (optimizer contract 3.5). That is why the profile's parameter-count
        invariance clause is stated for `max_norm=None`.

        Returns the pre-clip total gradient norm as a float when the clip
        ran, and None when it did not. `+0.0` would have been the shorter
        spelling and it is the wrong one: a coefficient of 1.0 says the clip
        ran and found nothing to do, which is a different fact from the clip
        not running, and `optimizer_step_oracle` draws the same line by
        leaving its `clip.*` stages empty rather than filling them.
        """
        gs = _as_seq(grads, "grads", self._where)
        gprobes = _check_dtype(gs, "grads", self._where, inplace=True)
        # Re-read every parameter buffer's shape and layout on EVERY step
        # rather than trusting the constructor's reading: a caller can
        # reshape or resize their own array between steps, and the
        # registry's offsets would then address the wrong memory.
        pprobes = _check_dtype(self.params, "params", self._where,
                               inplace=True)
        if len(gs) != len(self.params):
            raise ValueError(
                "mojolearn.%s.step: %d gradients for %d parameter tensors"
                % (self._where, len(gs), len(self.params))
            )
        for j, (pp, gp) in enumerate(zip(pprobes, gprobes)):
            if pp.shape != gp.shape:
                raise ValueError(
                    "mojolearn.%s.step: grads[%d] has shape %r, params[%d] "
                    "has %r" % (self._where, j, gp.shape, j, pp.shape)
                )
        if sum(nelems(pb.shape) for pb in pprobes) != self.n_total:
            raise ValueError(
                "mojolearn.%s.step: the parameter tensors hold %d floats now "
                "and held %d when this optimizer was built; a parameter "
                "buffer was resized underneath its registry"
                % (self._where, sum(nelems(pb.shape) for pb in pprobes),
                   self.n_total)
            )

        if self.lr_schedule is not None:
            self.lr = float(self.lr_schedule.lr_at(self.t + 1))
        cfg = self._config()
        max_norm_f = 0.0 if max_norm is None else float(max_norm)
        if max_norm is not None and not (max_norm_f > 0.0):
            raise ValueError(
                "mojolearn.%s.step: max_norm must be > 0 or None, got %r; "
                "None is how 'no clipping' is spelled "
                "(python/mojolearn/_training_impl.py)"
                % (self._where, max_norm)
            )

        flat_p, packed_p = _pack(self.params, pprobes)
        flat_g, packed_g = _pack(gs, gprobes)
        self.packed_ = bool(packed_p or packed_g)
        info = zeros((3,), "<f4")
        self.t += 1

        # `params` is, in this exact order (mirrored word for word in
        # `bindings/_mojolearn_training.mojo::optimizer_step_binding`):
        #
        #     0   n_tensors       J; `offsets` holds J + 1 int32
        #     1   kind            0 = SGD, 1 = Adam, 2 = AdamW
        #     2   t               the step number, ONE-BASED; first step is 1
        #     3   nesterov        0 or 1
        #     4   lr              (float)
        #     5   beta1           (float; Adam and AdamW only)
        #     6   beta2           (float; Adam and AdamW only)
        #     7   eps             (float; Adam and AdamW only)
        #     8   weight_decay    (float)
        #     9   momentum        (float; SGD only)
        #     10  dampening       (float; SGD only)
        #     11  max_norm        (float; <= 0 turns the gradient-norm clip
        #                          OFF)
        #
        # A silent reorder here is a WRONG ANSWER and not a crash: swap
        # `beta1` and `beta2` and every step still returns a full buffer of
        # plausible floats. If you change this list, change the comment in
        # the binding in the same edit.
        plist = [
            int(len(self.params)),
            int(self._KIND),
            int(self.t),
            int(cfg["nesterov"]),
            float(cfg["lr"]),
            float(cfg["beta1"]),
            float(cfg["beta2"]),
            float(cfg["eps"]),
            float(cfg["weight_decay"]),
            float(cfg["momentum"]),
            float(cfg["dampening"]),
            max_norm_f,
        ]

        # `_load` and not `self._bind()`: the two resolve the same
        # module, and `_load` additionally reads the tier back OUT of
        # the binary and refuses a binary that disagrees with the tier
        # asked for. A wrong-arm measurement that is correctly labelled
        # by accident is the failure that read-back exists to prevent.
        binding = _load(getattr(self, "numeric_mode", None))
        # Every array is held in a local across the call. The Mojo side takes
        # raw addresses, borrows and retains nothing, which is only sound
        # while the owning objects are alive (`_buffer.py`). `addr` (writable
        # required) for everything the kernel writes; `addr_ro` for the
        # offsets registry, which it only reads.
        binding.optimizer_step(
            addr(flat_p, name="params"),
            addr(flat_g, name="grads"),
            addr(self.exp_avg, name="exp_avg"),
            addr(self.exp_avg_sq, name="exp_avg_sq"),
            addr_ro(self.offsets, name="offsets"),
            addr(self.buf_initialized, name="buf_initialized"),
            addr(info, name="info"),
            plist,
        )

        if packed_p:
            _unpack_into(flat_p, self.params, pprobes)
        if max_norm is not None and packed_g:
            _unpack_into(flat_g, gs, gprobes)

        if info[0] != 0.0:
            self.total_norm_ = float(info[1])
            self.clip_coef_ = float(info[2])
        else:
            self.total_norm_ = None
            self.clip_coef_ = None
        #: The learning rate the step just used, as a float32 value.
        self.lr_ = float(np.float32(cfg["lr"]))
        return self.total_norm_

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
        if self.accumulation_steps == 1:
            return self.step(parts[0], max_norm=max_norm)
        combined = accumulate_grads(
            parts, tokens, numeric_mode=getattr(self, "numeric_mode", None))
        return self.step(combined, max_norm=max_norm)


class SGD(_Optimizer):
    """`torch.optim.SGD`, on the GPU, over float32 buffers (NumPy arrays, `array.array`, `mojolearn.Array`; the buffer protocol is the boundary).

    Optimizer contract 7.3. Momentum, dampening, Nesterov and COUPLED L2
    weight decay -- coupled meaning the decay is folded into the GRADIENT and
    therefore passes through the momentum buffer, which is what
    `torch.optim.SGD` does and is a different algorithm from AdamW's
    decoupled form.

        p = SGD([w1, w2], lr=1e-2, momentum=0.9)
        p.step([g1, g2], max_norm=1.0)

    WHERE THE MEASUREMENT STOPS, AND IT IS UNEVEN
    ---------------------------------------------
    `training-optimizer.identical.card`, 18 records, md5 `97d160b0`, is
    BYTE-IDENTICAL on an Apple M4 and an AMD MI325X (2026-08-28 legs), and
    its clause (a) matched device against the host oracle BITWISE over 33
    cases and all 382,822 compared cells. **NO NVIDIA LEG HAS RUN**, for that
    card or for the composed training loop's checkpoint comparison, so this
    is a TWO-VENDOR result and not a three-vendor one. Clauses (b), (c), (d)
    and (f) are SKIPPED on both columns.

    All of that belongs to `numeric_mode="identical"`. The default FAST build
    makes no cross-vendor claim of any kind.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter accepted and ignored is a wrong answer waiting for a
    caller. Every "refused by name" names the file that raises it:

        params          honored   a float32 array, or a list of them. THE
                                  ORDER IS PART OF THE ANSWER: `j` is the
                                  `param_id` and its ascending order is the
                                  clip's cross-tensor summation order
                                  (contract 3.3). Reorder and the total norm
                                  is a different, equally valid, different-
                                  bits number.
        lr              honored   read fresh on every step, so assigning
                                  `opt.lr` between steps is how a schedule is
                                  spelled here. It is a HOST scalar computed
                                  once per step (contract 7.1) and never
                                  inside a kernel.
        momentum        honored   contract 7.3. `0.0` leaves the momentum
                                  buffer untouched and the per-tensor
                                  `buf_initialized` flag never flips.
        dampening       honored   contract 7.3a. At `dampening = 0`, which is
                                  the default, `c_damp` is exactly 1.0 and
                                  the first-step COPY and the recurrence
                                  agree; that is why the lane's own sabotage
                                  for the first step is inert at the default.
        weight_decay    honored   COUPLED, folded into the gradient
        nesterov        honored   contract 7.3c, and REFUSED BY NAME with
                                  `momentum = 0` or `dampening != 0`
                                  (python/mojolearn/_training_impl.py), which
                                  is `torch.optim.SGD`'s own refusal. The
                                  profile would accept it and silently
                                  degenerate: at `momentum = 0` the Nesterov
                                  reading is bit-identical to the plain one,
                                  so nothing would tell the caller their flag
                                  did nothing.
        max_norm        honored   on `step`, not on the constructor, because
                                  it is a property of the step and the
                                  reference spells it as a separate call
        maximize        refused   `torch.optim.SGD`'s sign flip is not in
                                  `OptimizerConfig` and has no clause, no
                                  fixture and no sabotage
                                  (python/mojolearn/_training_impl.py)
        foreach         refused   an EXECUTION knob in torch. This surface is
                                  one launch over the whole flat buffer for
                                  Adam and one per tensor for SGD, and the
                                  choice is not the caller's
        fused           refused   same, and torch's fused path is a different
                                  arithmetic in torch too
        capturable      refused   CUDA-graph capture; nothing here captures
        differentiable  refused   there is no autograd in this library
        per-parameter   refused   one `OptimizerConfig` per optimizer. Two
        option groups             groups would be two calls with two
                                  registries and two clips, and two clips do
                                  not equal one (contract 3.5)
        sparse grads    refused   dense float32 only, everywhere
        float16/bfloat16 refused  MIXED PRECISION IS NOT COVERED; see
                                  `_NOT_COVERED` at the top of this file
        float64         refused   there is no float64 on a Metal device
                                  (python/mojolearn/_buffer.py says the same
                                  at the library boundary)
    """

    _KIND = _KIND_SGD

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, lr_schedule=None,
                 accumulation_steps=1, **kwargs):
        _refuse_unknown(kwargs, "SGD")
        super().__init__(params, "SGD", lr_schedule, accumulation_steps)
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


class Adam(_Optimizer):
    """`torch.optim.Adam`, on the GPU, over float32 buffers (NumPy arrays, `array.array`, `mojolearn.Array`; the buffer protocol is the boundary).

    Optimizer contract 7.2, with COUPLED weight decay: the decay is folded
    into the GRADIENT, so it passes through `m` and `v` and is itself
    smoothed and normalized. `AdamW` is the decoupled form and the difference
    between the two is an ORDER, not a coefficient (contract 7.4).

        opt = Adam([w1, w2], lr=1e-3)
        opt.step([g1, g2])

    **`AT weight_decay = 0` ADAM AND ADAMW ARE THE SAME ARITHMETIC**, and 0
    is the reference's own default. Picking between the two classes on a
    default-configured run cannot change a bit.

    WHERE THE MEASUREMENT STOPS, AND IT IS UNEVEN
    ---------------------------------------------
    `training-optimizer.identical.card`, 18 records, md5 `97d160b0`, is
    BYTE-IDENTICAL on an Apple M4 and an AMD MI325X (2026-08-28 legs); clause
    (a) matched device against the host oracle BITWISE over 33 cases and all
    382,822 compared cells. **NO NVIDIA LEG HAS RUN**, for that card or for
    the composed training loop's checkpoint comparison. TWO VENDORS, not
    three. Clauses (b), (c), (d) and (f) are SKIPPED on both columns.

    THIS IS NOT PYTORCH'S ANSWER AND IT CANNOT BE. Contract 5.2 and 5.3 make
    that structural rather than accidental: the bias correction is not
    computed in float64 here, and `1 - beta2` in float32 differs from
    torch's float64 route by about 1.3e-5 relative, which is the third
    significant decimal of the coefficient that drives `v`. Sameness across
    vendors is what the profile buys; agreement with torch is not on offer.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY:

        params          honored   float32 array or list of them. THE ORDER IS
                                  PART OF THE ANSWER when clipping is on
                                  (contract 3.3)
        lr              honored   read fresh on every step, so a schedule is
                                  `opt.lr = ...` between steps. A HOST scalar
                                  computed once per step (7.1), never in a
                                  kernel
        betas           honored   `(beta1, beta2)`, both in [0, 1). `1.0` is
                                  REFUSED BY NAME by
                                  `training/estimator.mojo`: `1 - beta^t` is
                                  then exactly zero and the host scalars
                                  divide by it
        eps             honored   contract 4d, and it is `sqrt(v) + eps` and
                                  NOT `sqrt(v + eps)`. The two agree to the
                                  last bit on ordinary gradients and separate
                                  when `v` lands in the 1e-20 to 1e-12 band
        weight_decay    honored   COUPLED here, DECOUPLED in `AdamW`
        max_norm        honored   on `step`
        amsgrad         refused   the profile excludes it and the exclusion
                                  is LOAD BEARING (optimizer contract, and
                                  `clip_coefficient`'s docstring says why):
                                  `max(v_max, v)` would create the profile's
                                  first `-0.0` vs `+0.0` selection hazard,
                                  where the order of a compare decides which
                                  of two equal-comparing values survives.
                                  Refused by name in
                                  python/mojolearn/_training_impl.py
        maximize        refused   not in `OptimizerConfig`; no clause, no
                                  fixture, no sabotage
        foreach         refused   an EXECUTION knob; Adam here is ONE launch
        fused           refused   over the whole flat buffer and the choice
        capturable      refused   is not the caller's
        differentiable  refused   there is no autograd in this library
        per-parameter   refused   one `OptimizerConfig` per optimizer
        option groups
        sparse grads    refused   dense float32 only
        float16/bfloat16 refused  MIXED PRECISION IS NOT COVERED
        float64         refused   no float64 on a Metal device
    """

    _KIND = _KIND_ADAM

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, lr_schedule=None, accumulation_steps=1,
                 **kwargs):
        _refuse_unknown(kwargs, type(self).__name__)
        super().__init__(params, type(self).__name__, lr_schedule,
                         accumulation_steps)
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


class AdamW(Adam):
    """`torch.optim.AdamW`, on the GPU, over float32 buffers (NumPy arrays, `array.array`, `mojolearn.Array`; the buffer protocol is the boundary).

    Optimizer contract 7.4: DECOUPLED weight decay. The decay multiplies the
    PARAMETER, as `p * (1 - lr*wd)`, and the gradient is untouched, so unlike
    `Adam` it does not pass through `m` and `v`. **THE DIFFERENCE FROM `Adam`
    IS AN ORDER, NOT A COEFFICIENT**, and it is the only difference between
    the two classes.

    **AT `weight_decay = 0` THE TWO ALGORITHMS ARE THE SAME ARITHMETIC**, and
    torch's own default for `AdamW` is 0.01 while `Adam`'s is 0. A comparison
    of the two classes run at their library defaults is comparing two things
    at once.

    The measurement, the honored/refused table and every refusal are `Adam`'s
    and are not restated. The one line that differs is above.
    """

    _KIND = _KIND_ADAMW

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, lr_schedule=None, accumulation_steps=1,
                 **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, lr_schedule=lr_schedule,
                         accumulation_steps=accumulation_steps, **kwargs)


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
    "maximize": (
        "torch's sign flip is not a field of `OptimizerConfig` "
        "(training/checks/optimizer_oracle.mojo) and has no contract clause, "
        "no fixture and no sabotage. Negate your gradients yourself and leave "
        "the extra step visible in your source"
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


def _refuse_unknown(kwargs, where):
    """Every keyword this surface does not have, refused BY NAME."""
    for k in kwargs:
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


def clip_grad_norm_(grads, max_norm, norm_type=2.0, error_if_nonfinite=True,
                    numeric_mode=None, **kwargs):
    """`torch.nn.utils.clip_grad_norm_`, on the GPU, over float32 buffers (NumPy arrays, `array.array`, `mojolearn.Array`; the buffer protocol is the boundary).
    Scales the gradients IN PLACE and returns the PRE-CLIP total norm, which
    is what torch returns and why the name has a trailing underscore.

    Optimizer contract section 3.

        total = clip_grad_norm_([g1, g2, g3], max_norm=1.0)

    **THE ANSWER IS NOT `sqrt(sum of every square)`.** The reference folds
    TWICE -- a norm per tensor, then a norm over those (contract 3.1) -- and
    the flat spelling is a DIFFERENT number in float32. Both folds are
    delegated to the certified GEMM at `m = n = 1`, `OP_NT`, so they inherit
    its leaf partition, its balanced tree, its odd-tail carry and its own
    three-vendor measurement at leg 11 (`144aa5b`), which is the GEMM's
    measurement and not this lane's.

    **THE ORDER YOU PASS THE TENSORS IN IS PART OF THE ANSWER.** `j` is the
    `param_id` and its ascending order is the cross-tensor summation order
    (contract 3.3). A registry stable across runs is the only thing that
    makes the number reproducible.

    WHERE THE MEASUREMENT STOPS: the optimizer card is byte-identical on an
    Apple M4 and an AMD MI325X and **NO NVIDIA LEG HAS RUN**. See `Adam`.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY:

        grads            honored   float32 array or list of them, WRITTEN IN
                                   PLACE
        max_norm         honored   must be > 0. `<= 0` is REFUSED BY NAME by
                                   `training/estimator.mojo`: reaching this
                                   function IS the clip running, and "no
                                   clipping" is spelled by not calling it
        norm_type        honored   2.0 only. Every other value, `inf`
                                   included, is REFUSED BY NAME
                                   (python/mojolearn/_training_impl.py): the
                                   profile has one norm, the two-level L2 of
                                   contract 3.1, and an L-inf clip is a
                                   max-reduction with its own `-0.0`
                                   selection hazard, its own clause and its
                                   own sabotage, none of which exist
        error_if_nonfinite honored True only. False is REFUSED BY NAME: the
                                   profile ALWAYS refuses a non-finite total
                                   norm (contract 8a,
                                   `refuse_nonfinite_scalar`), and there is
                                   no build in which it does not, so
                                   accepting False would be accepting a
                                   promise this surface cannot keep
        foreach          refused   an EXECUTION knob; see `Adam`
        per-tensor       refused   `torch.nn.utils.clip_grad_norm_` returns
        norms                      one scalar and so does this. The
                                   per-tensor norms are computed on device
                                   and are not copied back, because no gate
                                   has ever compared them at this boundary
        float16/bfloat16 refused   MIXED PRECISION IS NOT COVERED
        float64          refused   no float64 on a Metal device

    THE COST NOBODY PRICED. The certified entry SYNCHRONIZES ONCE PER TENSOR,
    to keep its sub-buffer views alive across each delegated GEMM, so a
    registry of many small tensors pays `J` round trips per clip. That is
    stated on `identical_clip_grad_norm` itself and a batched launcher is
    OWED. You choose `J`.
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

    offsets = _offsets_for(probes)
    flat, packed = _pack(gs, probes)
    info = zeros((2,), "<f4")

    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_training.mojo::clip_grad_norm_binding`):
    #
    #     0  n_tensors    J; `offsets` holds J + 1 int32
    #     1  max_norm     (float; must be > 0, refused otherwise)
    plist = [int(len(gs)), mn]

    binding = _load(numeric_mode)
    binding.clip_grad_norm(
        addr(flat, name="grads"), addr_ro(offsets, name="offsets"),
        addr(info, name="info"), plist,
    )
    if packed:
        _unpack_into(flat, gs, probes)
    return float(info[0])


# ===================================================================
# THE LOSS
# ===================================================================


def cross_entropy(logits, targets, ignore_index=_IGNORE_INDEX_DEFAULT,
                  reduction="mean", label_smoothing=0.0, num_items=None,
                  return_grad=False, numeric_mode=None, **kwargs):
    """`torch.nn.functional.cross_entropy` over class INDICES, on the GPU.

    Profile `mojolearn.identical.loss.ce.fp32.v1`, loss contract sections 2
    through 6.

        loss = cross_entropy(logits, targets)              # a float
        loss, dlogits = cross_entropy(logits, targets, return_grad=True)
        per_row = cross_entropy(logits, targets, reduction="none")

    `logits` is `(N, V)` float32, `targets` is `(N,)` int32 or int64 holding
    a class index or `ignore_index`. Returns a float under `reduction="sum"`
    or `"mean"`, and an `(N,)` float32 array under `"none"`. With
    `return_grad=True` it returns `(loss, dlogits)` and `dlogits` is
    `(N, V)`.

    **FORWARD AND BACKWARD ARE ONE CALL AND THAT IS NOT PACKAGING.** The
    backward reads `expo` and `denom`, the buffers the forward wrote, and
    RECOMPUTES NOTHING -- a second spelling of the softmax is a second thing
    that can be wrong. Asking for the gradient afterwards would mean either
    recomputing the forward or handing its intermediates out to Python, and
    this surface does neither.

    WHERE THE MEASUREMENT STOPS, AND IT IS UNEVEN
    ---------------------------------------------
    `training-loss.identical.card`, 17 records, md5 `a87615d9`, is
    BYTE-IDENTICAL on an Apple M4 and an AMD MI325X (2026-08-28 legs); clause
    (a) matched device against the host oracle BITWISE over 24 cases and all
    61,925 compared cells, and clause (e) matched a hand-written closed form
    over 4 cases and 264 cells with no epsilon anywhere. **NO NVIDIA LEG HAS
    RUN.** TWO VENDORS, not three. Clauses (b), (c), (d) and (f) are SKIPPED
    on both columns, and without (d) the four fold arms reached through the
    GEMM are gated only by clause (a), which sees ONE execution plan. All of
    it belongs to `numeric_mode="identical"`.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY:

        logits           honored   `(N, V)` float32, row-major
        targets          honored   `(N,)` integer class indices, or
                                   `ignore_index`. A target that is neither
                                   is REFUSED BY NAME with its row
                                   (`ce_refuse_inputs` in
                                   training/checks/loss_oracle.mojo)
        ignore_index     honored   torch's default -100. **A target equal to
                                   `ignore_index` is ignored even when it is
                                   a valid class index**, which is torch's
                                   behavior and is admitted rather than
                                   refused: set it to 0 on a real vocabulary
                                   and you lose class 0 silently
        reduction        honored   'none', 'sum', 'mean'. 'none' has NO
                                   BACKWARD and `return_grad=True` is refused
                                   with it (loss contract section 11): a
                                   per-row upstream vector adds a product per
                                   cell whose placement, before or after seam
                                   L16's division, is a real decision with two
                                   different answers and this lane has no
                                   caller for it
        label_smoothing  honored   in [0, 1). **`0.0` SELECTS A DIFFERENT
                                   KERNEL rather than a bit-inert branch**
                                   (contract 6.2(c), DEVIATION 1155), so this
                                   is not a knob that quietly does nothing at
                                   its default
        num_items        honored   `fixed_cross_entropy`'s
                                   `num_items_in_batch`. `None` means "not
                                   supplied", which is the MEAN arm's own
                                   default; `ce_divisor` is the ONE producer
                                   of the divisor the forward and the backward
                                   must agree about, and it is host-side and
                                   testable with no GPU present
        return_grad      honored   the seams L14 through L16 gradient with
                                   respect to `logits`
        weight           refused   a per-class weight vector is REFUSED BY
                                   NAME (loss contract section 11): a weighted
                                   mean's denominator is a SUM OF FLOATS and
                                   would need a fold, a clause, a fixture and
                                   a sabotage of its own, where `count` today
                                   is an integer and therefore exact,
                                   order-free and vendor-free (contract 5.5)
        size_average     refused   torch's own deprecated aliases for
        reduce                     `reduction`; refused rather than
                                   translated, because translating them
                                   silently reproduces the ambiguity torch
                                   deprecated them for
        class-probability refused  `targets` is class INDICES only. Soft
        targets                    targets are a different forward with a
                                   different backward and are not in the
                                   contract
        float16/bfloat16 refused   MIXED PRECISION IS NOT COVERED
        float64          refused   no float64 on a Metal device

    WHAT IT COSTS, and at a real vocabulary this is the dominant cost of the
    call. `training/estimator.mojo` allocates `shift` and `expo` at `N * V`
    floats each, plus `weights` and `dlogits` when `return_grad` is set and a
    third `N * V` for `logp` when smoothing is on. The forward is ROW
    INDEPENDENT through seam L11 -- only the final fold runs over `N`, and it
    folds `[N]` floats and not `[N, V]` -- so a caller who cannot afford the
    residency splits the ROWS, calls per chunk and concatenates, and under
    `reduction="none"` the answer is bit-identical to the unsplit call.
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
    """The tier the training binding will run on, read back from the binary
    itself and not from the string that was passed in."""
    return _backend._CODE_MODE.get(
        _load(numeric_mode).training_numeric_mode(), "unknown")


def vendor_used(numeric_mode=None):
    """'metal', 'cuda' or 'hip': the accelerator API of the binary this
    module will call, from that binary's own compile-time constant
    (`training_vendor()`, `checks/vendor.mojo`), not from the directory it
    was loaded from and not from the platform."""
    return _backend.read_vendor(_load(numeric_mode))


# ===================================================================
# LEARNING-RATE SCHEDULES: a float32 that is a pure function of (step, config)
# ===================================================================
# NO HOST LIBM. `math.cos` differs in the last bit between platforms, so a
# cosine schedule spelled with it would train a different run on each box.
# Every schedule below is computed EXACTLY in Python integers (`Fraction`)
# and rounded ONCE to float32 with round-half-even, so the bits are the
# same on every interpreter and every OS. The cosine is a Taylor series
# evaluated on a rational interval enclosing pi; the remainder bound is
# carried as an interval and the rounding is decided only when both ends
# of the interval round to the same float32 (otherwise the precision is
# raised, and a case that cannot be decided raises rather than guesses).
# The value of `cos(pi p)` at the rational points where it is itself
# rational (p in {0, 1/3, 1/2, 2/3, 1}, Niven) is taken exactly, so the
# only exact ties float32 rounding could meet are handled by the exact
# path and the irrational cases are never ties.

_F32_MIN_NORMAL_EXP = -126
_F32_MAX_EXP = 127

#: pi to 60 decimal places, a constant string, bracketed below into a
#: rational interval [PI_LO, PI_HI] with width 1e-60.
_PI_DIGITS = "3.141592653589793238462643383279502884197169399375105820974944"
_PI_LO = Fraction(_PI_DIGITS)
_PI_HI = _PI_LO + Fraction(1, 10 ** 60)


def _f32_round(q):
    """The float32 nearest to the exact rational `q`, ties to even, flushed
    to +0.0 below the smallest normal (the identical tier's ftz), as a
    Python float holding exactly that float32 value."""
    q = Fraction(q)
    if q == 0:
        return 0.0
    sign = -1.0 if q < 0 else 1.0
    q = abs(q)
    num, den = q.numerator, q.denominator
    e = num.bit_length() - den.bit_length() - 24

    def scaled(exp):
        if exp >= 0:
            return Fraction(num, den * (1 << exp))
        return Fraction(num * (1 << (-exp)), den)

    while scaled(e) >= (1 << 24):
        e += 1
    while scaled(e) < (1 << 23):
        e -= 1
    sc = scaled(e)
    m = sc.numerator // sc.denominator
    rem = sc - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (m & 1) == 1):
        m += 1
    if m == (1 << 24):
        m = 1 << 23
        e += 1
    if e + 23 < _F32_MIN_NORMAL_EXP:
        return 0.0
    if e + 23 > _F32_MAX_EXP:
        raise OverflowError("mojolearn: schedule value overflows float32")
    return sign * float(m) * (2.0 ** e)


def _f32_bits(value):
    return int(np.asarray(np.float32(value)).view(np.uint32))


def _cos_taylor(x, terms):
    """sum_{k<terms} (-1)^k x^(2k) / (2k)!, exact rational, with the
    remainder bound x^(2 terms) / (2 terms)!."""
    total = Fraction(0)
    term = Fraction(1)
    x2 = x * x
    for k in range(terms):
        total += term
        term = -term * x2 / ((2 * k + 1) * (2 * k + 2))
    return total, abs(term)


def _cos_pi_interval(p, terms):
    """An exact rational interval [lo, hi] enclosing cos(pi * p) for a
    rational p in [0, 1]. Exact at the five rational points."""
    p = Fraction(p)
    if p < 0 or p > 1:
        raise ValueError("mojolearn: cosine progress must be in [0, 1]")
    exact = {
        Fraction(0): Fraction(1), Fraction(1, 3): Fraction(1, 2),
        Fraction(1, 2): Fraction(0), Fraction(2, 3): Fraction(-1, 2),
        Fraction(1): Fraction(-1),
    }
    if p in exact:
        return exact[p], exact[p]
    flip = False
    if p > Fraction(1, 2):
        p = 1 - p
        flip = True
    # x in [x_lo, x_hi], a subset of (0, pi/2); cos is decreasing there.
    x_lo, x_hi = _PI_LO * p, _PI_HI * p
    t_hi, r_hi = _cos_taylor(x_hi, terms)
    t_lo, r_lo = _cos_taylor(x_lo, terms)
    lo, hi = t_hi - r_hi, t_lo + r_lo
    if flip:
        lo, hi = -hi, -lo
    return lo, hi


def _decide_f32(fn):
    """`fn(terms)` returns an exact rational interval; the float32 both ends
    round to, raising the Taylor precision until they agree."""
    for terms in (24, 32, 48, 64):
        lo, hi = fn(terms)
        a, b = _f32_round(lo), _f32_round(hi)
        if a == b:
            return a
    raise ArithmeticError(
        "mojolearn: the exact schedule value straddles a float32 rounding "
        "boundary within 1e-60; refusing to guess a bit"
    )


class _Schedule(object):
    """Shared half of the schedules. `t` is the optimizer's ONE-BASED step
    counter, so `lr_at(1)` is the first step's learning rate."""

    kind = ""

    def __init__(self, peak_lr, warmup_steps=0, total_steps=None, min_lr=0.0):
        self.peak_lr = float(np.float32(peak_lr))
        self.min_lr = float(np.float32(min_lr))
        self.warmup_steps = int(warmup_steps)
        self.total_steps = None if total_steps is None else int(total_steps)
        for name, v in (("peak_lr", self.peak_lr), ("min_lr", self.min_lr)):
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

    def _progress(self, t):
        """`(p, None)` with the rational progress p in (0, 1) of the decay
        phase, or `(None, value)` with the exact value of the warmup /
        after-total phases."""
        t = int(t)
        if t < 1:
            raise ValueError(
                "mojolearn schedule: step t is ONE-BASED, got %d" % t)
        peak, lo = Fraction(self.peak_lr), Fraction(self.min_lr)
        if t <= self.warmup_steps:
            return None, peak * Fraction(t, self.warmup_steps)
        if self.total_steps is None:
            return None, peak
        if t >= self.total_steps:
            return None, lo
        span = self.total_steps - self.warmup_steps
        return Fraction(t - self.warmup_steps, span), None

    def _decay(self, p):
        raise NotImplementedError

    def lr_at(self, t):
        """The float32 learning rate of ONE-BASED step `t`, as a float."""
        p, value = self._progress(t)
        if p is None:
            return _f32_round(value)
        return self._decay(p)

    def bits_at(self, t):
        """`lr_at(t)` as its float32 bit pattern, for fixtures."""
        return _f32_bits(self.lr_at(t))


class ConstantLR(_Schedule):
    """`peak_lr` at every step, after an optional linear warmup from
    `peak_lr / warmup_steps`. `total_steps` and `min_lr` are ignored."""

    kind = "constant"

    def __init__(self, peak_lr, warmup_steps=0, total_steps=None,
                 min_lr=0.0):
        super(ConstantLR, self).__init__(peak_lr, warmup_steps, None, 0.0)


class WarmupLinearLR(_Schedule):
    """Linear warmup over `warmup_steps`, then a straight line from
    `peak_lr` down to `min_lr` at `total_steps`, then `min_lr`."""

    kind = "linear"

    def _decay(self, p):
        peak, lo = Fraction(self.peak_lr), Fraction(self.min_lr)
        return _f32_round(peak + (lo - peak) * p)


class WarmupCosineLR(_Schedule):
    """Linear warmup over `warmup_steps`, then
    `min_lr + (peak_lr - min_lr) * (1 + cos(pi p)) / 2` with
    `p = (t - warmup) / (total - warmup)`, then `min_lr`. The cosine is
    exact rational arithmetic (see the section comment), never `math.cos`.
    """

    kind = "cosine"

    def _decay(self, p):
        peak, lo = Fraction(self.peak_lr), Fraction(self.min_lr)

        def interval(terms):
            c_lo, c_hi = _cos_pi_interval(p, terms)
            a = lo + (peak - lo) * (1 + c_lo) / 2
            b = lo + (peak - lo) * (1 + c_hi) / 2
            return (a, b) if a <= b else (b, a)
        return _decide_f32(interval)


# ===================================================================
# GRADIENT ACCUMULATION (optimizer contract clause 9.2)
# ===================================================================


def accumulation_is_aligned(tokens, accumulation_steps, numeric_mode=None):
    """`microbatch_split_is_identical(T, A)` from
    `training/checks/optimizer_oracle.mojo`, contract clause 9.2: True when
    A microbatches over T tokens combine, by the balanced tree, to the
    unsplit weight-gradient bits."""
    binding = _load(numeric_mode)
    return bool(int(binding.accumulation_is_aligned(
        [int(tokens), int(accumulation_steps)])))


def accumulate_grads(microbatch_grads, tokens, numeric_mode=None):
    """Combine A microbatch gradients into one, on device, by the v1
    balanced tree in ascending microbatch index (`ftz(ftz(x) + ftz(y))` at
    every node, contract clause 9.2 condition 5; NOT a running sum).

    `microbatch_grads` is a list of A entries, each an array or a list of
    arrays of one shape set. `tokens` is the whole step's token count T
    and a split that clause 9.2 does not admit is REFUSED BY NAME; pass
    `tokens=None` to make no alignment claim (a plain pair add). Returns a
    list of arrays (or one array when the entries were arrays).
    """
    parts = list(microbatch_grads)
    if not parts:
        raise ValueError("mojolearn.accumulate_grads: no microbatches")
    single = isinstance(parts[0], np.ndarray)
    seqs = [_as_seq(p, "microbatch_grads[%d]" % i, "accumulate_grads")
            for i, p in enumerate(parts)]
    for i, sq in enumerate(seqs):
        _check_dtype(sq, "microbatch_grads[%d]" % i, "accumulate_grads")
        if len(sq) != len(seqs[0]):
            raise ValueError(
                "mojolearn.accumulate_grads: microbatch %d has %d tensors, "
                "microbatch 0 has %d" % (i, len(sq), len(seqs[0])))
        for j, (a, b) in enumerate(zip(sq, seqs[0])):
            if a.shape != b.shape:
                raise ValueError(
                    "mojolearn.accumulate_grads: microbatch %d tensor %d has "
                    "shape %r, microbatch 0 has %r" % (i, j, a.shape, b.shape))
    a_count = len(seqs)
    t_tokens = -1 if tokens is None else int(tokens)
    if tokens is not None and t_tokens < 1:
        raise ValueError("mojolearn.accumulate_grads: tokens must be >= 1")
    flats = [_pack(sq)[0] for sq in seqs]
    n = int(flats[0].size)
    stacked = np.ascontiguousarray(np.stack(flats, axis=0), dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    binding = _load(numeric_mode)
    # addresses = [out, parts]; params = [n, a, t_tokens] (mirrored word
    # for word in bindings/_mojolearn_training.mojo::accumulate_binding).
    binding.accumulate([_addr(out), _addr_ro(stacked)],
                       [n, a_count, t_tokens])
    result = [np.empty(a.shape, dtype=np.float32) for a in seqs[0]]
    _unpack_into(out, result)
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
    out = np.empty(n, dtype=np.float32)
    if inp is None:
        inp = np.zeros(1, dtype=np.float32)
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


class Generator(object):
    """A seeded, position-keyed random stream for neural training.

    Every element of every draw is a pure function of (seed, stream id,
    element index): the same call at the same counter yields the same bits
    on every vendor and at every launch geometry. `state_dict()` is
    `{"seed", "counter"}` and restoring it resumes the stream exactly.

    The initializers compute their bound / std on the host in IEEE float64
    with only +, -, *, / and sqrt (all correctly rounded on every platform),
    round it to float32 and hand it to the device; the draw itself is
    `core/philox_neural.mojo`'s fixed integer-to-float mapping.
    """

    def __init__(self, seed, numeric_mode=None):
        seed = int(seed)
        if seed < 0 or seed >= (1 << 64):
            raise ValueError("mojolearn.Generator: seed must be in [0, 2^64)")
        self.seed = seed
        self.counter = 0
        self.numeric_mode = numeric_mode

    def state_dict(self):
        return {"seed": int(self.seed), "counter": int(self.counter)}

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.counter = int(state["counter"])
        if self.counter < 0 or self.counter > 0xFFFFFFFF:
            raise ValueError(
                "mojolearn.Generator: counter must be a 32-bit unsigned")

    def next_stream(self):
        """Consume and return the next stream id (for ops that draw
        several slices of one stream, such as dropout over microbatches)."""
        return self._next_stream()

    def _next_stream(self):
        if self.counter > 0xFFFFFFFF:
            raise RuntimeError(
                "mojolearn.Generator: the stream counter is exhausted")
        s = self.counter
        self.counter += 1
        return s

    @staticmethod
    def _shape(shape):
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        else:
            shape = tuple(int(d) for d in shape)
        n = 1
        for d in shape:
            if d < 1:
                raise ValueError(
                    "mojolearn.Generator: shape must be positive, got %r"
                    % (shape,))
            n *= d
        return shape, n

    def uniform(self, shape, low=0.0, high=1.0):
        """`low + u * (high - low)` with `u` in [0, 1), one rounding."""
        shape, n = self._shape(shape)
        lo, hi = float(np.float32(low)), float(np.float32(high))
        span = float(np.float32(hi - lo))
        if not (span >= 0.0):
            raise ValueError("mojolearn.Generator.uniform: high must be >= low")
        out = _rng_call(_load(self.numeric_mode), _RNG_UNIFORM, n, 0,
                        self.seed, self._next_stream(), lo, span)
        return out.reshape(shape)

    def normal(self, shape, mean=0.0, std=1.0):
        """Box-Muller (log, sqrt, cos through the identical seams)."""
        shape, n = self._shape(shape)
        sd = float(np.float32(std))
        if not (sd >= 0.0):
            raise ValueError("mojolearn.Generator.normal: std must be >= 0")
        out = _rng_call(_load(self.numeric_mode), _RNG_NORMAL, n, 0,
                        self.seed, self._next_stream(),
                        float(np.float32(mean)), sd)
        return out.reshape(shape)

    def kaiming_uniform(self, shape, fan_in, a=math.sqrt(5.0)):
        """`torch.nn.init.kaiming_uniform_`'s bound
        `sqrt(2 / (1 + a^2)) * sqrt(3 / fan_in)`; torch's Linear default is
        `a = sqrt(5)`, which makes the bound `1 / sqrt(fan_in)`."""
        gain = math.sqrt(2.0 / (1.0 + float(a) * float(a)))
        bound = gain * math.sqrt(3.0 / float(int(fan_in)))
        return self.uniform(shape, -bound, bound)

    def xavier_uniform(self, shape, fan_in, fan_out, gain=1.0):
        bound = float(gain) * math.sqrt(
            6.0 / float(int(fan_in) + int(fan_out)))
        return self.uniform(shape, -bound, bound)

    def kaiming_normal(self, shape, fan_in, gain=math.sqrt(2.0)):
        return self.normal(
            shape, 0.0, float(gain) / math.sqrt(float(int(fan_in))))

    def xavier_normal(self, shape, fan_in, fan_out, gain=1.0):
        return self.normal(shape, 0.0, float(gain) * math.sqrt(
            2.0 / float(int(fan_in) + int(fan_out))))

    def dropout(self, x, p, offset=0, stream=None):
        """Inverted dropout: keep where the element's uniform coin is
        `>= p`, scaled by `1 / (1 - p)` (a host float32). Returns
        `(y, key)`; `key` replays the mask in `dropout_backward`. `offset`
        is the element index of `x[0]` in the stream and `stream` (default:
        the next counter value) the stream id, so a microbatch of a larger
        batch draws the same coins it would draw unsplit."""
        p = float(np.float32(p))
        if not (0.0 <= p < 1.0):
            raise ValueError("mojolearn.Generator.dropout: p must be in [0, 1)")
        x = np.ascontiguousarray(x)
        _check_dtype([x], "x", "Generator.dropout")
        scale = float(np.float32(1.0 / (1.0 - p)))
        stream = self._next_stream() if stream is None else int(stream)
        key = {"seed": self.seed, "stream": stream, "p": p, "scale": scale,
               "offset": int(offset)}
        y = _rng_call(_load(self.numeric_mode), _RNG_DROPOUT_FORWARD, x.size,
                      key["offset"], self.seed, stream, p, scale,
                      x.reshape(-1))
        return y.reshape(x.shape), key

    def dropout_backward(self, dy, key):
        dy = np.ascontiguousarray(dy)
        _check_dtype([dy], "dy", "Generator.dropout_backward")
        dx = _rng_call(_load(self.numeric_mode), _RNG_DROPOUT_BACKWARD,
                       dy.size, key["offset"], key["seed"], key["stream"],
                       key["p"], key["scale"], dy.reshape(-1))
        return dx.reshape(dy.shape)


# ===================================================================
# THE STACK'S OPS: embedding, RMSNorm, LM head (no arithmetic here)
# ===================================================================


def _c32(a, name, where):
    a = np.ascontiguousarray(a)
    _check_dtype([a], name, where)
    return a


def embedding_forward(weight, ids, numeric_mode=None):
    """`weight[ids]` for `weight (V, D)` and integer `ids (N,)`; returns
    `(N, D)`. `embedding/checks/embedding_identical.mojo`'s gather."""
    w = _c32(weight, "weight", "embedding_forward")
    if w.ndim != 2:
        raise ValueError("mojolearn.embedding_forward: weight must be (V, D)")
    ids = np.ascontiguousarray(np.asarray(ids).reshape(-1), dtype=np.int32)
    n, (v, d) = int(ids.size), w.shape
    y = np.empty((n, d), dtype=np.float32)
    _load(numeric_mode).embedding_forward(
        [_addr(y), _addr_ro(w), _addr_ro(ids)], [n, int(v), int(d)])
    return y


def embedding_backward(dy, ids, vocab, numeric_mode=None):
    """The fresh `(V, D)` gradient of `embedding_forward` from `dy (N, D)`:
    the run-sorted ascending fold, never atomics."""
    dy = _c32(dy, "dy", "embedding_backward")
    if dy.ndim != 2:
        raise ValueError("mojolearn.embedding_backward: dy must be (N, D)")
    ids = np.ascontiguousarray(np.asarray(ids).reshape(-1), dtype=np.int32)
    n, d = dy.shape
    if ids.size != n:
        raise ValueError(
            "mojolearn.embedding_backward: ids and dy row counts differ")
    dw = np.empty((int(vocab), d), dtype=np.float32)
    _load(numeric_mode).embedding_backward(
        [_addr(dw), _addr_ro(dy), _addr_ro(ids)],
        [int(n), int(vocab), int(d)])
    return dw


def rms_norm_forward(x, weight, eps, numeric_mode=None):
    """`weight * x * rsqrt(mean(x^2) + eps)` per row of `x (M, D)`."""
    x = _c32(x, "x", "rms_norm_forward")
    w = _c32(weight, "weight", "rms_norm_forward")
    m, d = x.reshape(-1, x.shape[-1]).shape
    if w.shape != (d,):
        raise ValueError("mojolearn.rms_norm_forward: weight must be (D,)")
    y = np.empty((m, d), dtype=np.float32)
    _load(numeric_mode).rms_norm_forward(
        [_addr(y), _addr_ro(x), _addr_ro(w)], [int(m), int(d), float(eps)])
    return y.reshape(x.shape)


def rms_norm_backward(dy, x, weight, eps, numeric_mode=None):
    """`(dx, dweight)` of `rms_norm_forward`; the forward's row sums are
    recomputed by the same kernel."""
    x = _c32(x, "x", "rms_norm_backward")
    dy = _c32(dy, "dy", "rms_norm_backward")
    w = _c32(weight, "weight", "rms_norm_backward")
    if dy.shape != x.shape:
        raise ValueError("mojolearn.rms_norm_backward: dy and x shapes differ")
    m, d = x.reshape(-1, x.shape[-1]).shape
    dx = np.empty((m, d), dtype=np.float32)
    dw = np.empty((d,), dtype=np.float32)
    _load(numeric_mode).rms_norm_backward(
        [_addr(dx), _addr(dw), _addr_ro(dy), _addr_ro(x), _addr_ro(w)],
        [int(m), int(d), float(eps)])
    return dx.reshape(x.shape), dw


def linear_forward(a, weight, numeric_mode=None):
    """`a (M, K) . weight (N, K)^T -> (M, N)`, torch's Linear without bias,
    through the certified GEMM at OP_NT."""
    a = _c32(a, "a", "linear_forward")
    w = _c32(weight, "weight", "linear_forward")
    m, k = a.reshape(-1, a.shape[-1]).shape
    n, k2 = w.shape
    if k != k2:
        raise ValueError(
            "mojolearn.linear_forward: a has K=%d, weight has K=%d" % (k, k2))
    c = np.empty((m, n), dtype=np.float32)
    _load(numeric_mode).linear_forward(
        [_addr(c), _addr_ro(a), _addr_ro(w)], [int(m), int(n), int(k)])
    return c.reshape(a.shape[:-1] + (n,))


def linear_backward(dc, a, weight, numeric_mode=None):
    """`(da, dweight)` of `linear_forward`. `dweight`'s contraction is over
    the M tokens, the clause 9.2 contraction."""
    a = _c32(a, "a", "linear_backward")
    w = _c32(weight, "weight", "linear_backward")
    dc = _c32(dc, "dc", "linear_backward")
    m, k = a.reshape(-1, a.shape[-1]).shape
    n = w.shape[0]
    if dc.reshape(-1, dc.shape[-1]).shape != (m, n):
        raise ValueError("mojolearn.linear_backward: dc must be (M, N)")
    da = np.empty((m, k), dtype=np.float32)
    dw = np.empty((n, k), dtype=np.float32)
    _load(numeric_mode).linear_backward(
        [_addr(da), _addr(dw), _addr_ro(dc), _addr_ro(a), _addr_ro(w)],
        [int(m), int(n), int(k)])
    return da.reshape(a.shape), dw
