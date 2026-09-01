# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Neural-network training on the GPU: SGD, Adam, AdamW, global-norm
gradient clipping and cross-entropy loss, over NumPy arrays.

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
**Mixed precision, stochastic layers such as dropout, and distributed
training.** A paper draft lists those three as not covered and that stays
true. Every entry point here refuses a non-float32 array by name rather than
casting it, there is no random-number generator anywhere on this surface, and
nothing here is aware of a second process. See `_NOT_COVERED` below, which is
where each of the three raises from.

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

import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro
from ._mode import NumericModeMixin

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "clip_grad_norm_",
    "cross_entropy",
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
        "STOCHASTIC LAYERS ARE NOT COVERED. There is no dropout, no "
        "stochastic depth and no sampling of any kind on this surface, and "
        "no random-number generator in either profile. Identity across "
        "vendors would additionally require a bit-pinned RNG whose stream is "
        "a function of the seed and not of the launch geometry, and this "
        "lane has not written one."
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


def _as_seq(x, name, where):
    """A list of arrays from either one array or an iterable of them, the way
    `torch.nn.utils.clip_grad_norm_` accepts either."""
    if isinstance(x, np.ndarray):
        return [x]
    try:
        out = list(x)
    except TypeError:
        raise TypeError(
            "mojolearn.%s: %s must be a numpy array or an iterable of them, "
            "got %s" % (where, name, type(x).__name__)
        )
    if not out:
        raise ValueError(
            "mojolearn.%s: %s is empty; there is nothing to do and an empty "
            "parameter registry is refused rather than answered "
            "(training/estimator.mojo)" % (where, name)
        )
    for i, a in enumerate(out):
        if not isinstance(a, np.ndarray):
            raise TypeError(
                "mojolearn.%s: %s[%d] is %s, not a numpy array"
                % (where, name, i, type(a).__name__)
            )
    return out


def _check_dtype(arrays, name, where):
    """float32 or REFUSED BY NAME. Never cast.

    A silent upcast from float16 would answer a float32 question and label it
    as a half-precision answer, which is the mixed-precision refusal this
    module opens with. float64 is refused for the reason `_arrays.py` gives
    at the library boundary: there is no float64 on a Metal device and every
    kernel in this library is float32, so a float64 array here is a caller
    who believes something about the arithmetic that is not true.
    """
    for i, a in enumerate(arrays):
        if a.dtype in (np.float16, getattr(np, "bfloat16", np.float16)):
            _refuse_out_of_scope("mixed precision", where)
        if a.dtype != np.float32:
            raise TypeError(
                "mojolearn.%s: %s[%d] has dtype %s; this surface is float32 "
                "only (loss contract section 1, optimizer contract section "
                "1) and refuses rather than casts, because a cast changes "
                "which arithmetic answered and leaves the label unchanged "
                "(python/mojolearn/_training_impl.py)"
                % (where, name, i, a.dtype)
            )
        if a.size < 1:
            raise ValueError(
                "mojolearn.%s: %s[%d] is empty, shape %r; an empty tensor is "
                "REFUSED and not skipped -- the clip would fold a zero-length "
                "GEMM, which no gate in this tree has run "
                "(training/estimator.mojo)" % (where, name, i, a.shape)
            )


def _offsets_for(arrays):
    """`offsets[0 .. J]` as int32. `offsets[0]` is 0 and the entries are
    strictly ascending; `training/estimator.mojo::_offsets_from_ptr` refuses
    anything else BY NAME before any device work."""
    off = np.zeros(len(arrays) + 1, dtype=np.int32)
    total = 0
    for j, a in enumerate(arrays):
        total += a.size
        off[j + 1] = total
    return off


def _pack(arrays):
    """One flat C-contiguous float32 buffer holding every tensor in order.

    Returns `(flat, packed)`. `packed` is False for the ZERO-COPY case: a
    single tensor that is already float32 and C-contiguous is borrowed, its
    `ravel()` shares the caller's memory, and the device writes the caller's
    array directly. Anything else is copied, and that copy is what the
    `packed_` attribute on each optimizer reports.
    """
    if len(arrays) == 1 and arrays[0].flags["C_CONTIGUOUS"]:
        return arrays[0].reshape(-1), False
    return np.concatenate([np.ascontiguousarray(a).reshape(-1)
                           for a in arrays]), True


def _unpack_into(flat, arrays):
    """Write a flat buffer back into the caller's arrays, in place, so that
    an optimizer step is visible on the objects the caller handed in -- which
    is what `torch.optim` does and what a trailing underscore means."""
    at = 0
    for a in arrays:
        n = a.size
        a[...] = flat[at:at + n].reshape(a.shape)
        at += n


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

    def __init__(self, params, where):
        self._where = where
        self.params = _as_seq(params, "params", where)
        _check_dtype(self.params, "params", where)
        self.offsets = _offsets_for(self.params)
        self.n_total = int(self.offsets[-1])
        #: `m` is Adam's `exp_avg` and SGD's `momentum_buffer`; `v` is Adam's
        #: `exp_avg_sq` and is UNREAD BY SGD. Both are allocated at `N`
        #: whatever the algorithm, because the certified entry takes one
        #: signature for both and one state pair means a checkpoint has one
        #: shape (optimizer contract, `optimizer_step_oracle`).
        self.exp_avg = np.zeros(self.n_total, dtype=np.float32)
        self.exp_avg_sq = np.zeros(self.n_total, dtype=np.float32)
        #: SGD's per-tensor `buf_initialized` flag, contract 7.3b. 0 means
        #: this tensor's momentum buffer has never been written, so the first
        #: step COPIES the gradient into it instead of running the
        #: recurrence. It is CARRIED STATE and belongs in a checkpoint beside
        #: `exp_avg`; the flag flips once per TENSOR after that tensor's
        #: kernel, never per element.
        self.buf_initialized = np.zeros(len(self.params), dtype=np.int32)
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
        self.t = int(state["t"])
        self.exp_avg = np.ascontiguousarray(state["exp_avg"], dtype=np.float32)
        self.exp_avg_sq = np.ascontiguousarray(
            state["exp_avg_sq"], dtype=np.float32)
        self.buf_initialized = np.ascontiguousarray(
            state["buf_initialized"], dtype=np.int32)
        if self.exp_avg.size != self.n_total:
            raise ValueError(
                "mojolearn.%s.load_state_dict: exp_avg holds %d floats, this "
                "optimizer's registry is %d"
                % (self._where, self.exp_avg.size, self.n_total)
            )

    def zero_grad(self, grads):
        """`torch.optim.Optimizer.zero_grad`, over arrays the caller owns.
        Present so that a training loop written against torch reads the same;
        it writes `+0.0` and nothing else."""
        for g in _as_seq(grads, "grads", self._where):
            g[...] = 0.0

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
        _check_dtype(gs, "grads", self._where)
        if len(gs) != len(self.params):
            raise ValueError(
                "mojolearn.%s.step: %d gradients for %d parameter tensors"
                % (self._where, len(gs), len(self.params))
            )
        for j, (p, g) in enumerate(zip(self.params, gs)):
            if p.shape != g.shape:
                raise ValueError(
                    "mojolearn.%s.step: grads[%d] has shape %r, params[%d] "
                    "has %r" % (self._where, j, g.shape, j, p.shape)
                )

        cfg = self._config()
        max_norm_f = 0.0 if max_norm is None else float(max_norm)
        if max_norm is not None and not (max_norm_f > 0.0):
            raise ValueError(
                "mojolearn.%s.step: max_norm must be > 0 or None, got %r; "
                "None is how 'no clipping' is spelled "
                "(python/mojolearn/_training_impl.py)"
                % (self._where, max_norm)
            )

        flat_p, packed_p = _pack(self.params)
        flat_g, packed_g = _pack(gs)
        self.packed_ = bool(packed_p or packed_g)
        info = np.zeros(3, dtype=np.float32)
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
        # while the owning objects are alive (`_arrays.py`).
        binding.optimizer_step(
            _addr(flat_p),
            _addr(flat_g),
            _addr(self.exp_avg),
            _addr(self.exp_avg_sq),
            _addr_ro(self.offsets),
            _addr(self.buf_initialized),
            _addr(info),
            plist,
        )

        if packed_p:
            _unpack_into(flat_p, self.params)
        if max_norm is not None and packed_g:
            _unpack_into(flat_g, gs)

        if info[0] != 0.0:
            self.total_norm_ = float(info[1])
            self.clip_coef_ = float(info[2])
        else:
            self.total_norm_ = None
            self.clip_coef_ = None
        return self.total_norm_


class SGD(_Optimizer):
    """`torch.optim.SGD`, on the GPU, over NumPy arrays.

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
                                  (python/mojolearn/_arrays.py says the same
                                  at the library boundary)
    """

    _KIND = _KIND_SGD

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, **kwargs):
        _refuse_unknown(kwargs, "SGD")
        super().__init__(params, "SGD")
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
    """`torch.optim.Adam`, on the GPU, over NumPy arrays.

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
                 weight_decay=0.0, **kwargs):
        _refuse_unknown(kwargs, type(self).__name__)
        super().__init__(params, type(self).__name__)
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
    """`torch.optim.AdamW`, on the GPU, over NumPy arrays.

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
                 weight_decay=0.01, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, **kwargs)


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
    """`torch.nn.utils.clip_grad_norm_`, on the GPU, over NumPy arrays.
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
    _check_dtype(gs, "grads", "clip_grad_norm_")
    mn = float(max_norm)
    if not (mn > 0.0):
        raise ValueError(
            "mojolearn.clip_grad_norm_: max_norm must be > 0, got %r; "
            "reaching this function IS the clip running, and 'no clipping' is "
            "spelled by not calling it (training/estimator.mojo)" % (max_norm,)
        )

    offsets = _offsets_for(gs)
    flat, packed = _pack(gs)
    info = np.zeros(2, dtype=np.float32)

    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_training.mojo::clip_grad_norm_binding`):
    #
    #     0  n_tensors    J; `offsets` holds J + 1 int32
    #     1  max_norm     (float; must be > 0, refused otherwise)
    plist = [int(len(gs)), mn]

    binding = _load(numeric_mode)
    binding.clip_grad_norm(
        _addr(flat), _addr_ro(offsets), _addr(info), plist,
    )
    if packed:
        _unpack_into(flat, gs)
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

    x = np.asarray(logits)
    if x.ndim != 2:
        raise ValueError(
            "mojolearn.cross_entropy: logits must be 2-D (N, V), got %d-D "
            "shape %r (python/mojolearn/_training_impl.py)" % (x.ndim, x.shape)
        )
    _check_dtype([x], "logits", "cross_entropy")
    x = np.ascontiguousarray(x)
    n_rows, vocab = x.shape

    y = np.asarray(targets)
    if y.ndim != 1:
        raise ValueError(
            "mojolearn.cross_entropy: targets must be 1-D (N,), got %d-D "
            "shape %r; class-PROBABILITY targets are a different forward with "
            "a different backward and are not in the contract "
            "(python/mojolearn/_training_impl.py)" % (y.ndim, y.shape)
        )
    if y.dtype.kind not in "iu":
        raise TypeError(
            "mojolearn.cross_entropy: targets has dtype %s; this surface "
            "takes class INDICES only, not probabilities "
            "(python/mojolearn/_training_impl.py)" % (y.dtype,)
        )
    y = np.ascontiguousarray(y, dtype=np.int32)
    if y.shape[0] != n_rows:
        raise ValueError(
            "mojolearn.cross_entropy: logits has %d rows and targets has %d "
            "(python/mojolearn/_training_impl.py)" % (n_rows, y.shape[0])
        )

    eps = float(label_smoothing)
    # The [0, 1) bound and the finiteness check are `ce_refuse_inputs`'s and
    # fire from Mojo by name; this is only the shape work.

    loss_out = np.zeros(1, dtype=np.float32)
    row_out = np.empty(n_rows, dtype=np.float32)
    if return_grad:
        grad_out = np.empty((n_rows, vocab), dtype=np.float32)
    else:
        # NEVER a null address: `_f32_ptr` in the binding refuses one, and a
        # one-element placeholder is what the certified entry documents for
        # an unused output buffer.
        grad_out = np.zeros(1, dtype=np.float32)

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
        _addr(loss_out),
        _addr(row_out),
        _addr(grad_out),
        _addr_ro(x),
        _addr_ro(y),
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
