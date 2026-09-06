# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the PYTHON SURFACE of the two training profiles,
`mojolearn.identical.optimizer.fp32.v1` and
`mojolearn.identical.loss.ce.fp32.v1`.

DEVIATION 1590. Written 2026-09-01, with `training/estimator.mojo`,
`bindings/_mojolearn_training.mojo` and `python/mojolearn/_training_impl.py`.

WHAT THIS CLOSES
----------------
Those three files put two gated profiles in reach of a Python caller.
Nothing in that path compared its output against anything. The build-time
smoke test in `bindings/build_training.sh` proves the kernels LAUNCH and
says so in its own comments; it asserts not one bit, and it runs only on the
FAST build, because an identical build sets `MOJOLEARN_SKIP_BUILD_GATE=1`
and skips it entirely.

The specific way a wiring path goes wrong is not a wrong kernel. It is a
swapped params slot, a wrong element count, an offsets registry off by one,
an in-place write that did not land back in the caller's array, or a state
buffer that was not carried between steps. Every one of those produces a
plausible number. `bench/gemm_card_main.mojo::_oracle_op` records exactly
that failure happening once already in a reference card, where a table
opcode reached the oracle as a different operation and every row was a
product the table does not describe; it never trapped, because a card of
plausible floats is exactly as diffable as a card of correct ones.

WHAT IT IS NOT
--------------
**It is not a second copy of `training/checks/optimizer_check.mojo` or
`training/checks/loss_check.mojo`.** Those gate the ARITHMETIC -- device
against the host oracle over 33 and 24 cases, 382,822 and 61,925 cells, with
sabotage arms that show each gate can fail. None of that is re-gated here.

**AND THE FIXTURES BELOW ARE EXACT ON PURPOSE, WHICH MEANS THEY SEPARATE NO
SPELLING FROM ANY OTHER.** Every value in the ADAM, SGD and CLIP arms is a
dyadic rational whose whole chain is exact in float32, so a fused
multiply-add and an unfused one, a reciprocal-multiply and a true divide,
`sqrt(v)+eps` and `sqrt(v+eps)`, all give the same bits. That is the loss
contract's own warning about its exact-analytic arm (12.1: being exact, it
separates no spelling from any other and run alone would pass every sabotage
in the file), and it is repeated here because a reader could otherwise take
a green from this file as evidence about the arithmetic. **IT IS EVIDENCE
ABOUT THE WIRING**: the params-list order, the offsets registry, the element
counts, the in-place writes, the carried state, and which binary answered.

WHAT THE LANE ACTUALLY MEASURED, WHICH IS NOT THIS FILE
--------------------------------------------------------
`training-loss.identical.card`, 17 records, md5 `a87615d9`, and
`training-optimizer.identical.card`, 18 records, md5 `97d160b0`, are
BYTE-IDENTICAL on an Apple M4 and an AMD MI325X at the 2026-08-28 legs. The
composed loop reached `h_all = 463245ce6c97e68d` on both boxes.
**NO NVIDIA LEG HAS RUN**, for either card or for the checkpoint. TWO
VENDORS, not three.

THE ARMS, AND WHAT EACH ONE CAN SEE
-----------------------------------
    PROVENANCE   which binary loaded, tier and vendor, read back from it
    REFUSALS     every guard on this surface made to fire, by TYPE and by
                 MESSAGE. A refusal that never fires is not a refusal
    ADAM         one step against a HAND-COMPUTED reference, bitwise:
                 `param`, `exp_avg` and `exp_avg_sq`
    ADAMW        at `weight_decay = 0` AdamW is bit-identical to Adam,
                 which is the contract's own statement and a check that the
                 `kind` slot reaches the kernel at all
    SGD          `momentum = 0` against a hand-computed `p - lr*g`, then two
                 momentum steps so BOTH arms of the per-tensor
                 `buf_initialized` flag run
    CLIP         a fixture whose two-level norm is EXACTLY 85.0, the
                 coefficient and the scaled gradient bitwise, and a
                 `max_norm` above the norm leaving the gradient untouched
    SPLIT        the same flat model regrouped into different tensors gives
                 the same Adam step bitwise. Adam is elementwise, so the
                 offsets registry may not reach it; this is the arm that
                 sees an offsets bug
    STATE        `state_dict` round trip reproduces the next step bitwise,
                 including `buf_initialized`
    LOSSGRAD     the EXACT-ANALYTIC gradient, contract 12: a uniform row at
                 a power-of-two vocabulary and a power-of-two divisor, where
                 every cell is a dyadic rational written out by hand
    LOSSVALUE    float64 sanity on the loss scalar. **NOT AN IDENTITY ARM**
                 and it says so twice: `identical_log` is not numpy's `log`
                 and this arm compares with a tolerance

HOW TO RUN IT
-------------
    # 1. build the identical extension, which is the gated one
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh

    # 2. the gate
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_training_surface

    # 3. the FAST half, in its own process, because the mode is chosen at
    #    import and cannot be changed inside one
    cd python && MOJOLEARN_TRAINING_GATE_ALLOW_FAST=1 \\
        python3 -m mojolearn.tests.test_training_surface

**UNDER `identical` THE BITWISE ARMS ARE ASSERTED. UNDER `fast` THEY ARE
REPORTS AND CANNOT FAIL THE RUN**, and the banner says so. The FAST build
makes no claim about bits, and a green from a FAST run would be a green that
means nothing -- worse, on these deliberately exact fixtures it would be a
green that LOOKS like the identical one. The refusals and the provenance are
asserted in both modes, because those are properties of the wiring and not
of the arithmetic.

ENVIRONMENT
    MOJOLEARN_NUMERIC_MODE          'identical' for the asserting run
    MOJOLEARN_TRAINING_GATE_ALLOW_FAST
                                    1 permits the FAST run, whose bitwise
                                    arms are REPORTS and check nothing
"""

import os
import sys

import numpy as np

from mojolearn import _training_impl as T

SURFACE = "mojolearn._training_impl"


# ===========================================================================
# REPORTING
# ===========================================================================
# EVERY ARM RUNS AND EVERY VERDICT IS PRINTED BEFORE THE PROCESS EXITS
# NON-ZERO. Stopping at the first failure shows one arm's opinion when the
# useful evidence is WHICH arms a given defect reaches and which it walks
# past.


class Report(object):
    def __init__(self, assert_bits):
        #: False under FAST. Every bitwise verdict is then recorded as a
        #: REPORT, printed, and unable to fail the run.
        self.assert_bits = assert_bits
        self.rows = []
        self.notes = []

    def ok(self, arm, what):
        self.rows.append(("pass", arm, what))

    def bad(self, arm, what):
        self.rows.append(("FAIL", arm, what))

    def note(self, text):
        self.notes.append(text)

    def check(self, arm, cond, what, detail=""):
        """An assertion about the WIRING. Asserted in every mode, because
        a params slot is in the wrong place or it is not, and no arithmetic
        tier has an opinion about that."""
        if cond:
            self.ok(arm, what)
        else:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
        return bool(cond)

    def bits_equal(self, arm, got, want, what):
        """The only comparison this file makes about a number.

        UNDER FAST THIS IS A REPORT AND NOT A CHECK. The FAST build makes no
        claim about bits; recording a pass there would be recording a green
        that means nothing.
        """
        gb = np.ascontiguousarray(got, dtype=np.float32)
        wb = np.ascontiguousarray(want, dtype=np.float32)
        if gb.shape != wb.shape:
            self.bad(arm, "%s -- SHAPES DIFFER, %s vs %s"
                     % (what, gb.shape, wb.shape))
            return False
        ga = gb.ravel().view(np.uint32)
        wa = wb.ravel().view(np.uint32)
        same = bool(np.array_equal(ga, wa))
        if same:
            if self.assert_bits:
                self.ok(arm, what)
            else:
                self.rows.append(("rprt", arm, what + " (FAST: no bit was "
                                  "claimed; this row proves nothing)"))
            return True
        diff = np.flatnonzero(ga != wa)
        first = int(diff[0])
        detail = ("%s -- %d of %d cells differ; first at flat index %d, "
                  "0x%08x vs 0x%08x (%r vs %r)"
                  % (what, diff.size, ga.size, first, int(ga[first]),
                     int(wa[first]), float(gb.ravel()[first]),
                     float(wb.ravel()[first])))
        if self.assert_bits:
            self.bad(arm, detail)
        else:
            self.rows.append(("rprt", arm, detail + " (FAST: expected, this "
                              "build makes no claim about bits)"))
        return False

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal. Every guard on this
        surface is a branch a passing build can contain and never take, so
        each one is made to fire BY TYPE and BY MESSAGE."""
        try:
            fn(*a, **kw)
        except exc_type as exc:
            if needle in str(exc):
                self.ok(arm, what)
                return True
            self.bad(arm, "%s -- raised %s but the message does not contain "
                     "%r: %s" % (what, exc_type.__name__, needle, exc))
            return False
        except Exception as exc:  # noqa: BLE001 - the wrong exception is data
            self.bad(arm, "%s -- raised %s, want %s: %s"
                     % (what, type(exc).__name__, exc_type.__name__, exc))
            return False
        self.bad(arm, "%s -- IS INERT: the call was ACCEPTED" % what)
        return False

    @property
    def failures(self):
        return [r for r in self.rows if r[0] == "FAIL"]

    def render(self, out):
        arm = None
        for verdict, a, what in self.rows:
            if a != arm:
                out.write("\n  %s\n" % a)
                arm = a
            out.write("    %s %s\n" % (verdict, what))
        for n in self.notes:
            out.write("\n%s\n" % n)
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


class GateAbort(Exception):
    """A condition under which no verdict can be reached at all: a missing
    binary, an import that fails. Distinguished from a FAILING check because
    the conclusion differs -- a failure says the surface is wrong, an abort
    says the gate did not run."""


f32 = np.float32


# ===========================================================================
# THE EXACT FIXTURE, AND WHY EVERY NUMBER IN IT IS THE NUMBER IT IS
# ===========================================================================
# The optimizer profile's chain per element (contract 7.2, seams O1 to O14):
#
#     ms = beta1*m ; m = fma(c1, g, ms) ; g2 = g*g ; vs = beta2*v
#     v  = fma(c2, g2, vs) ; s = sqrt(v) ; sd = s/rt_bc2 ; dn = sd + eps
#     q  = m/dn ; p_out = fma(-step_size, q, p)
#
# with the host scalars b1t = beta1^t, bc1 = 1 - b1t, step_size = lr/bc1,
# b2t = beta2^t, bc2 = 1 - b2t, rt_bc2 = sqrt(bc2), c1 = 1 - beta1,
# c2 = 1 - beta2.
#
# THE HYPERPARAMETERS ARE CHOSEN SO THAT EVERY ONE OF THOSE IS EXACT:
#
#     beta1 = 0.5   -> b1t = 0.5, bc1 = 0.5, c1 = 0.5, step_size = 2*lr
#     beta2 = 0.75  -> b2t = 0.75, bc2 = 0.25, rt_bc2 = 0.5 (a perfect
#                      square, so the square root does not round)
#     lr    = 0.5   -> step_size = 1.0
#     eps   = 2.0   -> dn = sd + 2, and the gradients below are chosen so
#                      that sd + 2 is a POWER OF TWO, which is what makes
#                      q = m/dn exact
#
# `eps = 2.0` is absurd as a hyperparameter and that is the point: the
# fixture is built to be exact, not to be realistic. A realistic eps of 1e-8
# makes `dn` a value with a full mantissa and `q` a rounded quotient, and
# then this file would be comparing against a reference it cannot compute
# without an FMA primitive Python does not portably have.
#
# WITH m = v = 0 AT t = 1: m = c1*g = g/2, v = c2*g*g = g*g/4, s = |g|/2,
# sd = |g|, dn = |g| + 2. So dn is a power of two exactly when |g| is one of
# 2, 6, 14, 30, ... and those are the only gradients this fixture uses.
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.75
ADAM_LR = 0.5
ADAM_EPS = 2.0

#: `p` and `g` per element. Two tensors, so the offsets registry is
#: exercised; Adam is ONE launch over the whole flat buffer and cannot see a
#: tensor boundary, which is what the SPLIT arm is for.
ADAM_P = [np.array([1.0, -2.0, 0.5], dtype=np.float32),
          np.array([-0.75], dtype=np.float32)]
ADAM_G = [np.array([2.0, -6.0, 14.0], dtype=np.float32),
          np.array([2.0], dtype=np.float32)]


def adam_reference():
    """The hand-computed answer for the fixture above, WRITTEN OUT rather
    than looped, so that a reader can check every line against contract 7.2
    without running anything.

        g =  2 : m =  1, v =  1, s = 1, sd = 2, dn =  4, q =  1/4
                 p =  1    - 1*(1/4)     =  0.75
        g = -6 : m = -3, v =  9, s = 3, sd = 6, dn =  8, q = -3/8
                 p = -2    - 1*(-3/8)    = -1.625
        g = 14 : m =  7, v = 49, s = 7, sd = 14, dn = 16, q =  7/16
                 p =  0.5  - 1*(7/16)    =  0.0625
        g =  2 : m =  1, v =  1, s = 1, sd = 2, dn =  4, q =  1/4
                 p = -0.75 - 1*(1/4)     = -1.0

    Every value is a dyadic rational with a short mantissa. Nothing here
    rounds, which is the whole design of the fixture and also its whole
    limitation (see this module's docstring).
    """
    p = [np.array([0.75, -1.625, 0.0625], dtype=np.float32),
         np.array([-1.0], dtype=np.float32)]
    m = np.array([1.0, -3.0, 7.0, 1.0], dtype=np.float32)
    v = np.array([1.0, 9.0, 49.0, 1.0], dtype=np.float32)
    return p, m, v


# ---------------------------------------------------------------------------
# THE CLIP FIXTURE
# ---------------------------------------------------------------------------
# The reference clips on a TWO-LEVEL norm (contract 3.1): a per-tensor L2
# norm, then the L2 norm of THAT vector. In exact arithmetic it equals the
# flat norm; in float32 it does not, because each per-tensor sqrt rounds and
# each result is squared again.
#
#     tensor 0  [3, 4]        sumsq  25   norm  5
#     tensor 1  [12, 0, 0]    sumsq 144   norm 12
#     tensor 2  [84]          sumsq 7056  norm 84
#     total sumsq 25 + 144 + 7056 = 7225,  total norm = 85 EXACTLY
#
# THREE TENSORS AND NOT TWO. The contract says a `J = 2` fixture cannot see
# the `param_id` ORDER clause at all, because reversing two elements swaps
# the two children of one tree node and `a + b` equals `b + a` bitwise; and a
# `J = 1` fixture cannot see the two-level clause, because there is no outer
# fold to have a shape. Their norms 5, 12 and 84 sit in three different
# binades, which is what the flat-norm clause asks for.
#
# 85 IS NOT AN ACCIDENT: 5^2 + 12^2 = 13^2 and 13^2 + 84^2 = 85^2, so every
# intermediate is an exact integer under 2^24 and the two square roots are
# exact.
#
# AND `CLIP_EPS` DISAPPEARS HERE, DELIBERATELY. The coefficient is
# `max_norm / (total_norm + CLIP_EPS)` with `CLIP_EPS = 0x358637BD`, which is
# 1e-6. The float32 ulp at 85 is 2^-17, about 7.6e-6, so `85 + 1e-6` rounds
# back to exactly 85 and the denominator is a clean 85.0. That is what lets
# the coefficient below be computed in numpy and compared BY BITS.
CLIP_G = [np.array([3.0, 4.0], dtype=np.float32),
          np.array([12.0, 0.0, 0.0], dtype=np.float32),
          np.array([84.0], dtype=np.float32)]
CLIP_TOTAL_NORM = np.float32(85.0)


# ---------------------------------------------------------------------------
# THE LOSS FIXTURE
# ---------------------------------------------------------------------------
# Loss contract 12, the EXACT-ANALYTIC family. A UNIFORM row at a
# power-of-two vocabulary makes every softmax weight exactly `1/V`, and a
# power-of-two divisor keeps seam L16's division exact, so the whole gradient
# is a dyadic rational a person can write down.
#
#     V = 4, uniform logits  ->  w = 1/4 exactly, at every cell
#     3 rows, one of them ignore_index  ->  count = 2
#     reduction 'mean', no num_items    ->  divisor = count = 2
#     dl[i, y] = (1/4 - 1)/2 = -0.375   dl[i, v != y] = (1/4)/2 = 0.125
#     dl[ignored, :] = +0.0, STORED and not skipped
#
# CONTRACT 12.1 SAYS WHAT THIS CANNOT DO, IN ITS OWN WORDS: being exact, it
# separates NO spelling from any other, so run alone it would pass every
# sabotage in the loss file and gate nothing about the arithmetic.
LOSS_V = 4
LOSS_LOGITS = np.full((3, LOSS_V), 0.5, dtype=np.float32)
LOSS_TARGETS = np.array([1, -100, 3], dtype=np.int32)


def loss_grad_reference():
    dl = np.full((3, LOSS_V), 0.125, dtype=np.float32)
    dl[0, 1] = np.float32(-0.375)
    dl[2, 3] = np.float32(-0.375)
    dl[1, :] = np.float32(0.0)
    return dl


# ===========================================================================
# THE ARMS
# ===========================================================================


def arm_provenance(rep):
    """Which binary answered, read back from the binary itself.

    Not from the directory it sat in and not from the environment: that is
    `_backend.py`'s whole discipline, and a gate that trusted the
    environment variable would be reporting the tier it ASKED for rather
    than the tier it got.
    """
    try:
        mode = T.numeric_mode_used()
    except Exception as exc:  # noqa: BLE001 - nothing can run without this
        raise GateAbort(
            "the training extension did not load: %s: %s\n"
            "Build it with\n"
            "    MOJOLEARN_NUMERIC_MODE=identical bash "
            "bindings/build_training.sh" % (type(exc).__name__, exc))
    vendor = T.vendor_used()
    rep.check("PROVENANCE", mode in ("fast", "deterministic", "identical"),
              "the binary reports its tier: %s" % mode)
    rep.check("PROVENANCE", vendor in ("metal", "cuda", "hip"),
              "the binary reports its accelerator API: %s" % vendor,
              "a binary that answers 'none' cannot run a kernel anywhere")
    rep.note(
        "  tier %s, vendor %s, surface %s\n"
        "  THE CARDS THIS LANE ACTUALLY MEASURED ARE TWO-VENDOR: Apple M4 and\n"
        "  AMD MI325X, byte-identical, md5 a87615d9 (loss) and 97d160b0\n"
        "  (optimizer). NO NVIDIA LEG HAS RUN, for either card or for the\n"
        "  training loop's checkpoint comparison." % (mode, vendor, SURFACE))
    return mode


def arm_refusals(rep):
    """Every guard on this surface, made to fire BY TYPE and BY MESSAGE.

    Asserted in EVERY mode. A refusal is a property of the wiring and no
    arithmetic tier has an opinion about whether it fires.
    """
    p = np.zeros(4, dtype=np.float32)

    rep.raises("REFUSALS", ValueError, "nesterov=True needs momentum",
               "SGD(nesterov=True, momentum=0) is refused by name",
               T.SGD, p, nesterov=True)
    rep.raises("REFUSALS", ValueError, "nesterov=True needs momentum",
               "SGD(nesterov=True, dampening!=0) is refused by name",
               T.SGD, p, momentum=0.9, dampening=0.1, nesterov=True)
    rep.raises("REFUSALS", TypeError, "amsgrad",
               "Adam(amsgrad=True) is refused by name",
               T.Adam, p, amsgrad=True)
    rep.raises("REFUSALS", TypeError, "maximize",
               "Adam(maximize=True) is refused by name",
               T.Adam, p, maximize=True)
    rep.raises("REFUSALS", TypeError, "foreach",
               "Adam(foreach=True) is refused by name",
               T.Adam, p, foreach=True)
    rep.raises("REFUSALS", TypeError, "unexpected keyword",
               "an unknown keyword is refused rather than ignored",
               T.Adam, p, learning_rate=1.0)

    # THE THREE THINGS A PAPER DRAFT SAYS ARE NOT COVERED. Each must refuse
    # BY NAME and say the words, or the draft's sentence stops being true.
    rep.raises("REFUSALS", NotImplementedError,
               "MIXED PRECISION IS NOT COVERED",
               "a float16 parameter is refused, not upcast",
               T.Adam, np.zeros(4, dtype=np.float16))
    rep.raises("REFUSALS", NotImplementedError,
               "STOCHASTIC LAYERS ARE NOT COVERED",
               "dropout= is refused by name",
               T.Adam, p, dropout=0.1)
    rep.raises("REFUSALS", NotImplementedError,
               "DISTRIBUTED TRAINING IS NOT COVERED",
               "distributed= is refused by name",
               T.Adam, p, distributed=True)

    rep.raises("REFUSALS", TypeError, "float32 only",
               "a float64 parameter is refused, not cast",
               T.Adam, np.zeros(4, dtype=np.float64))
    rep.raises("REFUSALS", ValueError, "REFUSED and not skipped",
               "an empty tensor is refused, not skipped",
               T.Adam, np.zeros(0, dtype=np.float32))
    rep.raises("REFUSALS", ValueError, "is empty",
               "an empty parameter registry is refused",
               T.Adam, [])

    opt = T.Adam(np.zeros(4, dtype=np.float32))
    rep.raises("REFUSALS", ValueError, "max_norm must be > 0 or None",
               "step(max_norm=0) is refused; None is how OFF is spelled",
               opt.step, np.zeros(4, dtype=np.float32), max_norm=0.0)
    rep.raises("REFUSALS", ValueError, "gradients for",
               "a gradient count that does not match params is refused",
               opt.step, [np.zeros(2, dtype=np.float32),
                          np.zeros(2, dtype=np.float32)])

    g = np.zeros(4, dtype=np.float32)
    rep.raises("REFUSALS", NotImplementedError, "norm_type",
               "clip_grad_norm_(norm_type=inf) is refused by name",
               T.clip_grad_norm_, g, 1.0, float("inf"))
    rep.raises("REFUSALS", NotImplementedError, "error_if_nonfinite=False",
               "clip_grad_norm_(error_if_nonfinite=False) is refused by name",
               T.clip_grad_norm_, g, 1.0, 2.0, False)
    rep.raises("REFUSALS", ValueError, "max_norm must be > 0",
               "clip_grad_norm_(max_norm=0) is refused by name",
               T.clip_grad_norm_, g, 0.0)

    rep.raises("REFUSALS", NotImplementedError, "NO BACKWARD",
               "cross_entropy(reduction='none', return_grad=True) is refused",
               T.cross_entropy, LOSS_LOGITS, LOSS_TARGETS,
               reduction="none", return_grad=True)
    rep.raises("REFUSALS", TypeError, "weight",
               "cross_entropy(weight=...) is refused by name",
               T.cross_entropy, LOSS_LOGITS, LOSS_TARGETS,
               weight=np.ones(LOSS_V, dtype=np.float32))
    rep.raises("REFUSALS", TypeError, "size_average",
               "cross_entropy(size_average=...) is refused by name",
               T.cross_entropy, LOSS_LOGITS, LOSS_TARGETS, size_average=True)
    rep.raises("REFUSALS", ValueError, "reduction=",
               "an unknown reduction is refused by name",
               T.cross_entropy, LOSS_LOGITS, LOSS_TARGETS, reduction="avg")
    rep.raises("REFUSALS", ValueError, "class-PROBABILITY targets",
               "2-D targets are refused by name",
               T.cross_entropy, LOSS_LOGITS,
               np.zeros((3, LOSS_V), dtype=np.int32))

    # THE MOJO-SIDE REFUSALS. These fire from `ce_refuse_inputs` in
    # `training/checks/loss_oracle.mojo`, which is the ORACLE'S OWN
    # predicate, and they cross the boundary as a generic exception -- the
    # TYPE is the binding's and the MESSAGE is the contract's, so the
    # message is what is asserted.
    bad_target = np.array([0, 9, 1], dtype=np.int32)
    rep.raises("REFUSALS", Exception, "REFUSED",
               "a target outside [0, vocab) is refused by name, with its row",
               T.cross_entropy, LOSS_LOGITS, bad_target)
    nan_logits = LOSS_LOGITS.copy()
    nan_logits[1, 2] = np.float32("nan")
    rep.raises("REFUSALS", Exception, "REFUSED",
               "a non-finite logit is refused by name before any stage",
               T.cross_entropy, nan_logits, LOSS_TARGETS)
    nan_grad = [np.zeros(4, dtype=np.float32)]
    nan_grad[0][2] = np.float32("inf")
    rep.raises("REFUSALS", Exception, "REFUSED",
               "a non-finite gradient is refused by name",
               T.Adam(np.zeros(4, dtype=np.float32)).step, nan_grad)


def arm_adam(rep):
    """One Adam step against the hand-computed reference, BITWISE, on
    `param`, `exp_avg` and `exp_avg_sq`.

    All three are checked and not just `param`. `m` and `v` are CARRIED
    STATE -- a wrong `v` is permanent, because every later step divides by
    `sqrt(v)` -- and a state buffer that is written to the wrong place, or
    not written back at all, produces a correct first step and a wrong run.
    """
    params = [a.copy() for a in ADAM_P]
    opt = T.Adam(params, lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2),
                 eps=ADAM_EPS, weight_decay=0.0)
    total = opt.step([g.copy() for g in ADAM_G])
    want_p, want_m, want_v = adam_reference()

    rep.check("ADAM", total is None,
              "step() with no max_norm reports no clip, not a coefficient "
              "of 1.0",
              "a coefficient of 1.0 says the clip RAN and found nothing")
    rep.check("ADAM", opt.t == 1, "t is ONE-BASED after the first step")
    for j in range(len(params)):
        rep.bits_equal("ADAM", params[j], want_p[j],
                       "param tensor %d matches the hand-computed step" % j)
    rep.bits_equal("ADAM", opt.exp_avg, want_m,
                   "exp_avg matches the hand-computed m")
    rep.bits_equal("ADAM", opt.exp_avg_sq, want_v,
                   "exp_avg_sq matches the hand-computed v")

    # THE GRADIENT IS NOT TOUCHED WHEN THE CLIP DOES NOT RUN. A wrapper that
    # scaled it by a coefficient of 1.0 would pass every arm above.
    #
    # ON THE ZERO-COPY PATH ON PURPOSE. A single C-contiguous float32 array
    # is BORROWED rather than packed, so the device writes the caller's
    # memory directly and a spurious write is visible here. Through a LIST
    # the gradients are packed into a scratch buffer that is never unpacked
    # when the clip is off, and this check could not fail.
    flat_g = np.concatenate([a.ravel() for a in ADAM_G])
    flat_p = np.concatenate([a.ravel() for a in ADAM_P])
    seen = flat_g.copy()
    T.Adam(flat_p, lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2),
           eps=ADAM_EPS).step(flat_g)
    rep.bits_equal("ADAM", flat_g, seen,
                   "the gradient is bitwise untouched when the clip is off, "
                   "checked on the borrowed (zero-copy) path")


def arm_adamw(rep):
    """At `weight_decay = 0` AdamW is BIT-IDENTICAL to Adam, which is the
    contract's own statement (7.4: at zero decay the two are the same
    arithmetic), and at a nonzero decay it is NOT.

    The second half is what makes the first half worth checking. If the
    `kind` slot of the params list never reached the kernel, both halves
    would be identical and the first would still pass.
    """
    def run(cls, wd):
        p = [a.copy() for a in ADAM_P]
        o = cls(p, lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2),
                eps=ADAM_EPS, weight_decay=wd)
        o.step([g.copy() for g in ADAM_G])
        return np.concatenate([a.ravel() for a in p])

    rep.bits_equal("ADAMW", run(T.AdamW, 0.0), run(T.Adam, 0.0),
                   "AdamW == Adam at weight_decay = 0, bit for bit")
    same = np.array_equal(run(T.AdamW, 0.25).view(np.uint32),
                          run(T.Adam, 0.25).view(np.uint32))
    rep.check("ADAMW", not same,
              "AdamW != Adam at weight_decay = 0.25, so `kind` reaches the "
              "kernel",
              "if these agree, the decoupled-decay arm was never selected "
              "and the arm above is vacuous")


def arm_sgd(rep):
    """SGD at `momentum = 0` against a hand-computed `p - lr*g`, then two
    momentum steps.

    TWO STEPS AND NOT ONE. The per-tensor `buf_initialized` flag (contract
    7.3b) makes step 1 COPY the gradient into the momentum buffer and step 2
    run the recurrence. One step leaves the recurrence unlaunched, and a
    wrapper that forgot to carry the flag back out of the call would pass a
    one-step gate forever.
    """
    p = [np.array([1.0, -2.0], dtype=np.float32)]
    g = [np.array([0.5, 0.25], dtype=np.float32)]
    opt = T.SGD([a.copy() for a in p], lr=0.5)
    opt.step([a.copy() for a in g])
    # p - lr*g, every value a dyadic rational: 1 - 0.25 = 0.75,
    # -2 - 0.125 = -2.125.
    want = np.array([0.75, -2.125], dtype=np.float32)
    rep.bits_equal("SGD", opt.params[0], want,
                   "momentum = 0 matches the hand-computed p - lr*g")

    mom = T.SGD([a.copy() for a in p], lr=0.5, momentum=0.5)
    rep.check("SGD", int(mom.buf_initialized[0]) == 0,
              "buf_initialized starts at 0, so step 1 takes the COPY arm")
    mom.step([a.copy() for a in g])
    rep.check("SGD", int(mom.buf_initialized[0]) == 1,
              "buf_initialized is carried BACK OUT of the call and flips "
              "once per TENSOR",
              "a flag that does not come back makes every step a first step")
    # Step 1 copies: buf = g = [0.5, 0.25], p -= lr*buf.
    rep.bits_equal("SGD", mom.exp_avg, np.array([0.5, 0.25],
                                                dtype=np.float32),
                   "step 1 COPIES the gradient into the momentum buffer")
    mom.step([a.copy() for a in g])
    # Step 2 recurs: buf = g + momentum*buf = 0.5 + 0.5*0.5 = 0.75, and
    # 0.25 + 0.5*0.25 = 0.375. Both exact.
    rep.bits_equal("SGD", mom.exp_avg, np.array([0.75, 0.375],
                                                dtype=np.float32),
                   "step 2 runs the RECURRENCE, so both arms of the flag ran")


def arm_clip(rep):
    """The two-level global norm, on a fixture whose answer is exactly 85.0.

    Three checks, and the third is the one a wrapper bug reaches: the
    returned norm is the PRE-clip one, the coefficient scales the caller's
    arrays IN PLACE, and a `max_norm` above the norm leaves every bit alone.
    """
    g = [a.copy() for a in CLIP_G]
    total = T.clip_grad_norm_(g, max_norm=8.5)
    rep.bits_equal("CLIP", np.float32(total), CLIP_TOTAL_NORM,
                   "the two-level norm of the fixture is exactly 85.0")

    # `85 + CLIP_EPS` rounds back to 85 in float32 (the ulp at 85 is 7.6e-6
    # and CLIP_EPS is 1e-6), so the coefficient is a clean float32 quotient
    # and the scale is one correctly rounded product per cell.
    coef = np.float32(8.5) / CLIP_TOTAL_NORM
    for j in range(len(g)):
        rep.bits_equal("CLIP", g[j], (coef * CLIP_G[j]).astype(np.float32),
                       "tensor %d is scaled in place by max_norm/total" % j)

    wide = [a.copy() for a in CLIP_G]
    total2 = T.clip_grad_norm_(wide, max_norm=170.0)
    rep.bits_equal("CLIP", np.float32(total2), CLIP_TOTAL_NORM,
                   "the reported norm is the PRE-clip norm, whatever "
                   "max_norm is")
    for j in range(len(wide)):
        rep.bits_equal("CLIP", wide[j], CLIP_G[j],
                       "tensor %d is bitwise untouched when the coefficient "
                       "clamps to 1.0" % j)

    # THROUGH `step` AS WELL, because that path reaches the clip by a
    # different route: the certified entry runs it internally and this
    # wrapper reads the norm back out of a buffer it owns.
    p = [np.zeros_like(a) for a in CLIP_G]
    gs = [a.copy() for a in CLIP_G]
    opt = T.SGD(p, lr=0.0)
    reported = opt.step(gs, max_norm=8.5)
    rep.bits_equal("CLIP", np.float32(reported), CLIP_TOTAL_NORM,
                   "step(max_norm=...) reports the same pre-clip norm")
    rep.check("CLIP", opt.clip_coef_ is not None and opt.clip_coef_ < 1.0,
              "step records the clamped coefficient it applied")
    for j in range(len(gs)):
        rep.bits_equal("CLIP", gs[j], (coef * CLIP_G[j]).astype(np.float32),
                       "step scales the caller's gradient tensor %d in "
                       "place" % j)


def arm_split(rep):
    """The same flat model, regrouped into different tensors, gives the same
    Adam step BIT FOR BIT.

    **THIS IS THE ARM THAT SEES AN OFFSETS BUG.** Adam is one launch over
    the whole flat buffer and nothing in its kernel reads a tensor boundary,
    so with the clip OFF the grouping may not change one bit. An offsets
    vector that is off by one, or that is built in the wrong order, changes
    nothing the other arms look at and changes the answer here.

    Run with the clip OFF on purpose: with clipping ON the grouping IS part
    of the answer, by the reference's own two-level semantics (contract 3.1),
    and this arm would be asserting the opposite of the truth.
    """
    flat_p = np.concatenate([a.ravel() for a in ADAM_P])
    flat_g = np.concatenate([a.ravel() for a in ADAM_G])

    def run(groups):
        ps, at = [], 0
        for n in groups:
            ps.append(flat_p[at:at + n].copy())
            at += n
        gs, at = [], 0
        for n in groups:
            gs.append(flat_g[at:at + n].copy())
            at += n
        o = T.Adam(ps, lr=ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2),
                   eps=ADAM_EPS)
        o.step(gs)
        return np.concatenate([a.ravel() for a in ps])

    base = run([3, 1])
    for groups in ([4], [1, 1, 1, 1], [2, 2], [1, 3]):
        rep.bits_equal("SPLIT", run(groups), base,
                       "grouping %r gives the same Adam step" % (groups,))


def arm_state(rep):
    """A `state_dict` round trip reproduces the next step BIT FOR BIT.

    Including `buf_initialized`. Leaving it out of a checkpoint is the quiet
    way to break a resume: SGD's first step after the reload takes the COPY
    arm again and overwrites a momentum buffer it should have continued,
    which is what the optimizer contract's clause (d) control 2 is built to
    catch.
    """
    p = [np.array([1.0, -2.0], dtype=np.float32)]
    g = [np.array([0.5, 0.25], dtype=np.float32)]

    a = T.SGD([x.copy() for x in p], lr=0.5, momentum=0.5)
    a.step([x.copy() for x in g])
    after1 = a.params[0].copy()
    saved = a.state_dict()
    a.step([x.copy() for x in g])
    straight = a.params[0].copy()

    b = T.SGD([after1.copy()], lr=0.5, momentum=0.5)
    b.load_state_dict(saved)
    b.step([x.copy() for x in g])
    rep.bits_equal("STATE", b.params[0], straight,
                   "a resumed optimizer takes the same second step")
    rep.check("STATE", int(saved["buf_initialized"][0]) == 1,
              "state_dict carries buf_initialized, not just m and v")

    # THE NEGATIVE HALF, without which the arm above is vacuous. A fresh
    # optimizer that did NOT load the state takes the COPY arm again and
    # must land somewhere else; if these agree, the state was never
    # consulted and the resume check proved nothing.
    c = T.SGD([after1.copy()], lr=0.5, momentum=0.5)
    c.step([x.copy() for x in g])
    rep.check("STATE",
              c.params[0].tobytes() != straight.tobytes(),
              "an optimizer that did NOT load the state lands elsewhere, so "
              "the resume arm is not vacuous")


def arm_lossgrad(rep):
    """The EXACT-ANALYTIC gradient, loss contract 12.

    Every cell is a dyadic rational written out by hand in
    `loss_grad_reference`. **Contract 12.1 says in its own words what this
    cannot do**: being exact it separates no spelling from any other, so run
    alone it would pass every sabotage in the loss file. What it CAN see is
    the wiring -- the `(N, V)` element count, the row-major order, the
    divisor reaching the backward, and the ignored row.
    """
    loss, dl = T.cross_entropy(LOSS_LOGITS, LOSS_TARGETS, return_grad=True)
    rep.check("LOSSGRAD", dl.shape == (3, LOSS_V),
              "dlogits is (N, V) = %r" % (dl.shape,))
    rep.bits_equal("LOSSGRAD", dl, loss_grad_reference(),
                   "every gradient cell matches the hand-written closed form")
    rep.bits_equal("LOSSGRAD", dl[1], np.zeros(LOSS_V, dtype=np.float32),
                   "the ignored row is +0.0, STORED and not skipped")

    per_row = T.cross_entropy(LOSS_LOGITS, LOSS_TARGETS, reduction="none")
    rep.check("LOSSGRAD", per_row.shape == (3,),
              "reduction='none' returns (N,) = %r" % (per_row.shape,))
    s = T.cross_entropy(LOSS_LOGITS, LOSS_TARGETS, reduction="sum")
    rep.bits_equal("LOSSGRAD", np.float32(s * 0.5), np.float32(loss),
                   "mean == sum / count on a fixture whose count is a power "
                   "of two",
                   )
    rep.bits_equal("LOSSGRAD", per_row[1], np.float32(0.0),
                   "an ignored row contributes +0.0 to the per-row loss")

    # label_smoothing selects a DIFFERENT KERNEL and not a bit-inert branch
    # (contract 6.2(c), DEVIATION 1155). If the params slot never arrived,
    # the two calls below would return the same number.
    #
    # ON A NON-UNIFORM ROW, and that is the whole reason this fixture is not
    # the uniform one above. On a UNIFORM row the smoothed loss is
    # MATHEMATICALLY EQUAL to the unsmoothed one -- the smoothing term is the
    # mean log-probability, which on a uniform row IS the negative log
    # likelihood -- so a uniform fixture would be asking a question whose
    # honest answer is "no difference" and could only be checked on rounding
    # noise. This is a WIRING check with a tolerance, not a bit claim.
    slope = np.arange(LOSS_V, dtype=np.float32)
    tilted = np.stack([slope, slope * np.float32(2.0),
                       slope * np.float32(-1.0)]).astype(np.float32)
    plain = T.cross_entropy(tilted, LOSS_TARGETS)
    smoothed = T.cross_entropy(tilted, LOSS_TARGETS, label_smoothing=0.25)
    rep.check("LOSSGRAD", abs(float(smoothed) - float(plain)) > 1e-3,
              "label_smoothing reaches the kernel and changes the answer "
              "(%.9g vs %.9g)" % (float(smoothed), float(plain)),
              "TOLERANCE, NOT BITS: this arm proves the params slot arrived")


def arm_lossvalue(rep):
    """float64 sanity on the loss scalar.

    **NOT AN IDENTITY ARM.** It compares with a TOLERANCE, deliberately, and
    it must: the value is `log(4)` and `identical_log` is not numpy's `log`.
    `[[mojo-log-breaks-ties]]` measured about 5e-8 of disagreement between
    the two, which is exactly the size that a bitwise comparison here would
    fail on and that a sanity check should ignore.

    **NOT AN IDENTITY ARM**, said twice on purpose, because a reader
    skimming a list of green rows cannot otherwise tell this one apart from
    the ones above it.
    """
    loss = T.cross_entropy(LOSS_LOGITS, LOSS_TARGETS)
    want = float(np.log(LOSS_V))
    rep.check("LOSSVALUE", abs(float(loss) - want) < 1e-5,
              "a uniform row's mean loss is log(V) to 1e-5 (got %.9g, want "
              "%.9g)" % (float(loss), want),
              "TOLERANCE, NOT BITS")


# ===========================================================================
# THE RUNNER
# ===========================================================================


def main(out=sys.stdout):
    out.write("test_training_surface\n")
    out.write("   profiles mojolearn.identical.optimizer.fp32.v1 and\n")
    out.write("            mojolearn.identical.loss.ce.fp32.v1\n")
    out.write("   surface  %s\n" % SURFACE)

    try:
        probe = Report(True)
        mode = arm_provenance(probe)
    except GateAbort as exc:
        out.write("\nCANNOT START: %s\n" % exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        out.write("\nCANNOT START: %s: %s\n" % (type(exc).__name__, exc))
        return 2

    if mode != "identical" and os.environ.get(
            "MOJOLEARN_TRAINING_GATE_ALLOW_FAST") != "1":
        out.write(
            "\nTHIS PROCESS LOADED THE %s BUILD, WHICH MAKES NO IDENTITY\n"
            "CLAIM OF ANY KIND, so the bitwise arms of this gate cannot be\n"
            "ASSERTED and it will not pretend they were.\n"
            "\n"
            "  MOJOLEARN_NUMERIC_MODE=identical \\\n"
            "      python3 -m mojolearn.tests.test_training_surface\n"
            "\n"
            "having built the identical extension with\n"
            "\n"
            "  MOJOLEARN_NUMERIC_MODE=identical bash "
            "bindings/build_training.sh\n"
            "\n"
            "To run the FAST half on purpose -- the refusals and the mode\n"
            "discipline, ASSERTED, plus every bitwise arm as a REPORT that\n"
            "cannot fail -- set MOJOLEARN_TRAINING_GATE_ALLOW_FAST=1.\n"
            % mode.upper())
        return 1

    rep = Report(assert_bits=(mode == "identical"))
    rep.rows.extend(probe.rows)
    rep.notes.extend(probe.notes)

    arms = (
        ("REFUSALS", arm_refusals),
        ("ADAM", arm_adam),
        ("ADAMW", arm_adamw),
        ("SGD", arm_sgd),
        ("CLIP", arm_clip),
        ("SPLIT", arm_split),
        ("STATE", arm_state),
        ("LOSSGRAD", arm_lossgrad),
        ("LOSSVALUE", arm_lossvalue),
    )
    aborted = []
    for name, fn in arms:
        try:
            fn(rep)
        except GateAbort as exc:
            aborted.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001 - report, then exit non-zero
            aborted.append((name, "%s: %s" % (type(exc).__name__, exc)))

    rep.render(out)
    for name, why in aborted:
        out.write("\n  %s ARM DID NOT RUN\n" % name)
        for line in why.splitlines():
            out.write("    %s\n" % line)

    failed = len(rep.failures) or aborted
    out.write("\n")
    if failed:
        out.write("test_training_surface: RED. %d checks failed, %d arms did "
                  "not run.\n" % (len(rep.failures), len(aborted)))
        return 1
    if mode != "identical":
        out.write(
            "test_training_surface: the FAST arms passed, AND NO BIT WAS\n"
            "CLAIMED. This run says the refusals fire, the params lists line\n"
            "up and the state is carried. It says NOTHING about\n"
            "mojolearn.identical.optimizer.fp32.v1 or\n"
            "mojolearn.identical.loss.ce.fp32.v1.\n")
        return 0
    out.write(
        "test_training_surface: GREEN. The Python surface returns the\n"
        "hand-computed Adam, SGD and clip answers bit for bit, the\n"
        "exact-analytic cross-entropy gradient bit for bit, carries its\n"
        "state across steps and a checkpoint, and refuses every parameter it\n"
        "does not honor by name.\n"
        "\n"
        "WHAT IT DOES NOT PROVE. The fixtures are EXACT and therefore\n"
        "separate no spelling from any other: this file is a gate on the\n"
        "WIRING, and the arithmetic is gated by\n"
        "training/checks/optimizer_check.mojo and\n"
        "training/checks/loss_check.mojo. Those cards are byte-identical on\n"
        "TWO vendors, Apple M4 and AMD MI325X. NO NVIDIA LEG HAS RUN.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
