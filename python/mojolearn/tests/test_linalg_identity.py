# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTITY gate for the Python surface of
`mojolearn.identical.gemm.fp32.v1`.

DEVIATIONS 950 through 959. Written 2026-08-24.

WHAT THIS CLOSES
----------------
`gemm/host_entry.mojo`, `bindings/_mojolearn_linalg.mojo` and
`python/mojolearn/_linalg_impl.py` put the profile's product in reach of a
Python caller. Nothing in that path compared its output against
`gemm/checks/gemm_oracle.mojo::gemm_oracle`.

The build-time smoke test in `bindings/build_linalg.sh` proves the kernels
launch, that all five execution plans dispatch, that the three orientations
agree to `allclose`, and that the three refusals fire. It says so in its own
comments and it asserts NOT ONE BIT. It also runs only on the FAST build --
`MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh` sets
`MOJOLEARN_SKIP_BUILD_GATE=1` and skips it entirely, because the smoke test
imports the FAST package. So on the day this file was written, the IDENTICAL
linalg extension had no Python-side check of any kind.

Two things make that dangerous rather than merely incomplete.

- **FAST and IDENTICAL coincide on Apple at both pinned seams.** Contract
  section 4.1 measured Metal fused in both modes and `ftz` is a no-op on a
  flush-to-zero backend, so a user comparing numbers on one Mac learns
  nothing about which arithmetic ran.
- **`gemm/host_entry.mojo` is new code.** A wrong element count, a wrong
  stride assumption or a transposed operand produces a plausible matrix that
  is not the profile's answer, and passes every gate that existed before this
  one. `bench/gemm_card_main.mojo::_oracle_op` records exactly that failure
  happening once already, in the reference card itself, where a table opcode
  reached the oracle as a different operation and every row of the card was
  a product the table does not describe. It never trapped, because a card of
  plausible floats is exactly as diffable as a card of correct ones.

So this gate ends in BITWISE comparisons and never in `allclose`. A tolerance
here would hide precisely the defect being hunted. The one arm that does use a
tolerance is labeled ACCURACY and is stated not to be an identity arm.

WHAT IT IS NOT
--------------
**It is not a second copy of `gemm/checks/gemm_device_check.mojo`.** That
file gates the KERNEL -- oracle agreement over the shape table, launch
invariance, batch invariance, and five sabotages that show each gate can fail,
including the `-0.0` fixture that the reference card is blind to. The
arithmetic is gated there and is not re-gated here.

This gate is about the PYTHON PATH specifically, which is the part
`gemm_device_check` cannot see at all:

    the marshalling      numpy address in, numpy bytes out
    the element counts   m*k, n*k, m*n, and which of them is which
    the strides          a non-contiguous operand, copied rather than mis-read
    the op mapping       transpose_a / transpose_b -> OP_NN / OP_NT / OP_TN
    the mode selection   which .so actually loaded, read back from the binary

THE SHAPE RULE THIS FILE IS BUILT AROUND
----------------------------------------
**A bit-level assertion at small `k` is nearly vacuous.**
`contract_leaf_size(k)` returns `k` for `k <= 128`, so the contract degenerates
to ONE leaf, the fold tree is empty (contract 7.3), and the whole product is a
plain serial ascending chain seeded `+0.0`. Genuinely different
implementations agree there. `gemm_oracle_serial`, which is the DIAGNOSTIC
reference and explicitly not the v1 answer, is bit-equal to `gemm_oracle` at
`k <= 128` and separates above it (contract 7.5).

So every bit-level assertion below includes shapes with `k > 128`, and the
self-chosen shapes are picked for their leaf count `P = ceil(k / L)` rather
than for size. `m`, `n` and `k` are pairwise distinct in every one of them,
because a square or symmetric fixture lets a transposed operand or a swapped
stride alias onto a correct answer -- the backward-GEMM lane's finding, that a
transpose error is bit-identical across vendors and structurally invisible to
an identity gate.

THE ARMS, AND WHAT EACH ONE CAN SEE
-----------------------------------
    PROVENANCE   (957) which binary loaded, read back from the binary
    REFUSALS     (958) every guard on this surface made to fire
    ORIENTATION  (953) OP_NN, OP_NT, OP_TN agree BIT for BIT, contract 3
    STRIDES      (954) non-contiguous == its contiguous copy, bitwise
    OUT          (955) out= == the allocated form, bitwise, nothing unwritten
    INVARIANCE   (956) one row alone == that row in a bigger call, bitwise
    ACCURACY     (---) float64 sanity. NOT AN IDENTITY ARM. Says so twice.
    CARD         (951) the ONLY arm with an EXTERNAL reference: every checked
                       shape's output hashed and compared against the hash
                       `tools/gemm_card.sh oracle` emitted from `gemm_oracle`
    FAST         (957) in a FAST process, the default call raises and names
                       the fix, and `identical=False` returns a number

Everything except CARD is a RELATIVE comparison: it can see an inconsistency
in the Python path and it cannot see an answer that is wrong the same way
everywhere. CARD is what makes the answer absolute, and it is why this gate
FAILS rather than skips when no card is supplied.

HOW TO RUN IT
-------------
    # 1. build both linalg extensions (the identical one is the gated one)
    bash bindings/build_linalg.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh

    # 2. emit the reference card from the NORMATIVE host oracle
    tools/gemm_card.sh oracle /tmp/gemm_oracle.card

    # 3. the gate
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        MOJOLEARN_GEMM_CARD=/tmp/gemm_oracle.card \\
        python3 -m mojolearn.tests.test_linalg_identity

    # 4. the FAST half, in its own process, because the mode is chosen at
    #    import and cannot be changed inside one
    cd python && MOJOLEARN_LINALG_GATE_ALLOW_FAST=1 \\
        python3 -m mojolearn.tests.test_linalg_identity

`gemm/PYTHON_SURFACE_GATE.md` is the maintainer's page for this file: what it
proves, what it does not, and how to regenerate the reference after a
legitimate profile change.

ENVIRONMENT
    MOJOLEARN_GEMM_CARD           the oracle card. REQUIRED under IDENTICAL.
    MOJOLEARN_GEMM_CARD_LOG       optional; the run log beside the card, read
                                  only to quote the emitter's own P values.
    MOJOLEARN_LINALG_GATE_FULL    1 removes the element budget on the card arm
    MOJOLEARN_LINALG_GATE_BUDGET  the budget in float32 elements per shape
    MOJOLEARN_LINALG_GATE_ALLOW_FAST
                                  1 permits the FAST-only run, which checks NO
                                  BITS and says so in its banner
    MOJOLEARN_LINALG_GATE_CARD_OUT
                                  where to write this gate's own card, in the
                                  same format, for `identity_trace_diff.py`
    MOJOLEARN_REPO                the checkout, when the gate is run from
                                  outside it
"""

import importlib.util
import math
import os
import sys
import tempfile

import numpy as np


# ===========================================================================
# THE SURFACE UNDER TEST
# ===========================================================================
# `mojolearn.linalg` is the name the module's own docstring uses and it is NOT
# WIRED as of 2026-08-24: `python/mojolearn/__init__.py` imports no `linalg`
# and there is no `linalg.py`, so the surface is reachable only as
# `mojolearn._linalg_impl`. `bindings/build_linalg.sh`'s smoke test already
# imports it that way and says why in the same words. Preferring the public
# name means this file needs no edit on the day the re-export lands.
try:  # pragma: no cover - one of the two branches is dead per checkout
    from mojolearn import linalg as linalg
    SURFACE = "mojolearn.linalg"
except ImportError:
    from mojolearn import _linalg_impl as linalg
    SURFACE = "mojolearn._linalg_impl"


# ===========================================================================
# THE PROFILE'S OWN ARITHMETIC ON `k`, TRANSCRIBED (contract section 6)
# ===========================================================================
# TRANSCRIBED AND NOT IMPORTED, because it lives in Mojo
# (`gemm_oracle.mojo::contract_leaf_size`) and Python cannot call it. It is
# used HERE FOR REPORTING AND FOR THE COVERAGE GUARD ONLY -- never to produce
# or to check a value. Nothing this gate asserts depends on it being right;
# what depends on it is the sentence "this run checked a shape with P > 1",
# which is the sentence that makes the run worth anything.
#
# `bench/gemm_shapes.mojo` records what a wrong copy of this rule costs: its
# first version mirrored a flat `L = 128`, and reported `P = 7813` for a shape
# whose real leaf count is 1024.
K_LEAF_MIN = 128
MAX_LEAVES = 1024


def contract_leaf_size(k):
    """`L`, contract section 6. A pure function of `k` and the two profile
    constants -- not of `m`, not of `n`, not of the machine."""
    if k <= 0:
        return 1
    if k <= K_LEAF_MIN:
        return k
    if (k + K_LEAF_MIN - 1) // K_LEAF_MIN <= MAX_LEAVES:
        return K_LEAF_MIN
    return (k + MAX_LEAVES - 1) // MAX_LEAVES


def contract_leaf_count(k):
    """`P = ceil(k / L)`. 0 when `k == 0` (contract section 8)."""
    if k <= 0:
        return 0
    return (k + contract_leaf_size(k) - 1) // contract_leaf_size(k)


def leaf_note(k):
    """One line describing what the fold does at this `k`, for the report."""
    p = contract_leaf_count(k)
    if p == 1:
        return "P=1    one leaf, NO fold addition (contract 7.3)"
    if p % 2:
        return "P=%-4d ODD, the unpaired carry clause fires (7.2 clause 3)" % p
    return "P=%-4d even, every level pairs (7.2 clause 1)" % p


# ===========================================================================
# THE FIXTURE, BIT-ASSEMBLED, IDENTICAL TO THE CARD EMITTER'S
# ===========================================================================
# A transcription of `bench/gemm_card_main.mojo::_mix` and `::_exact`, and it
# has to be exact to the bit or the CARD arm below cannot run at all. That is
# not a hope: the card carries a hash of its own inputs, and the card arm
# checks THOSE FIRST and refuses to compare a single product until they match
# (DEVIATION 952). A drift in this function is a loud, correctly attributed
# failure rather than a wrong verdict about the GEMM.
#
# NO DECIMAL CONSTANT AND NO HOST FLOAT CHAIN, for the emitter's reason: the
# numerator is below 2^21 so it is exact in a float32 mantissa and the divisor
# is a power of two, so the division is exact and no rounding happens upstream
# of the thing being gated. `bench/linalg_trace_main.mojo` learned that the
# expensive way.
_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def fixture_values(count, salt):
    """`_fill(count, salt)` from `bench/gemm_card_main.mojo`, vectorized.

    Every intermediate is an explicit `np.uint64` so that no operand can be
    promoted to float64 by a value-based casting rule and quietly change the
    low bits of the mix. `[[mojo-int-widening-sign-extends]]`'s Python
    sibling: the arithmetic is unsigned and stays unsigned.
    """
    if count == 0:
        return np.empty(0, dtype=np.float32)
    idx = np.arange(count, dtype=np.uint64) + np.uint64(1)
    z = idx * np.uint64(_GOLDEN)
    z = z + np.uint64(((salt + 1) * _MIX_A) & _MASK64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_MIX_A)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_MIX_B)
    z = z ^ (z >> np.uint64(31))
    num = (z % np.uint64(2097151)).astype(np.int64) - np.int64(1048575)
    return num.astype(np.float32) / np.float32(1048576.0)


def fixture(shape, salt):
    return fixture_values(int(np.prod(shape)), salt).reshape(shape)


# ===========================================================================
# REPORTING
# ===========================================================================
# EVERY ARM RUNS AND EVERY VERDICT IS PRINTED BEFORE THE PROCESS EXITS
# NON-ZERO. `gemm/checks/gemm_device_check.mojo` states the reason and it
# is the same one here: stopping at the first failure shows one arm's opinion
# when the useful evidence is WHICH arms a given defect reaches and which it
# walks past.


class Report(object):
    def __init__(self):
        self.rows = []
        self.notes = []

    def ok(self, arm, what):
        self.rows.append((True, arm, what))

    def bad(self, arm, what):
        self.rows.append((False, arm, what))

    def note(self, text):
        self.notes.append(text)

    def check(self, arm, cond, what, detail=""):
        if cond:
            self.ok(arm, what)
        else:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
        return bool(cond)

    def bits_equal(self, arm, got, want, what):
        """The only comparison this file makes about a product. `.tobytes()`
        is C-order whatever the array's strides are, so a non-contiguous view
        and its contiguous copy hash and compare the same -- which is the
        point of the STRIDES arm."""
        gb = np.ascontiguousarray(got).tobytes()
        wb = np.ascontiguousarray(want).tobytes()
        if gb == wb:
            self.ok(arm, what)
            return True
        ga = np.ascontiguousarray(got).ravel().view(np.uint32)
        wa = np.ascontiguousarray(want).ravel().view(np.uint32)
        if ga.shape != wa.shape:
            self.bad(arm, "%s -- SHAPES DIFFER, %s vs %s"
                     % (what, np.shape(got), np.shape(want)))
            return False
        diff = np.flatnonzero(ga != wa)
        first = int(diff[0])
        self.bad(arm, "%s -- %d of %d cells differ; first at flat index %d, "
                 "0x%08x vs 0x%08x (%r vs %r)"
                 % (what, diff.size, ga.size, first, int(ga[first]),
                    int(wa[first]),
                    float(np.ascontiguousarray(got).ravel()[first]),
                    float(np.ascontiguousarray(want).ravel()[first])))
        return False

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal. Every guard on this
        surface is a branch a passing build can contain and never take, so
        each one is made to fire by name and by message."""
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
        return [r for r in self.rows if not r[0]]

    def render(self, out):
        arm = None
        for ok, a, what in self.rows:
            if a != arm:
                out.write("\n  %s\n" % a)
                arm = a
            out.write("    %s %s\n" % ("pass" if ok else "FAIL", what))
        for n in self.notes:
            out.write("\n%s\n" % n)
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


class GateAbort(Exception):
    """A condition under which no verdict about the GEMM can be reached at
    all -- a missing reference, a card this gate cannot parse, a fixture that
    does not reproduce. Distinguished from a FAILING check because the
    conclusion is different: a failure says the surface is wrong, an abort
    says the gate did not run."""


# ===========================================================================
# THE SELF-CHOSEN SHAPES
# ===========================================================================
# `m`, `n` and `k` PAIRWISE DISTINCT in every row. A square or symmetric shape
# lets a transposed operand or a swapped stride land on a correct answer, and
# `bindings/_mojolearn_linalg.mojo` says so about its own params list: "swap
# `m` and `n` on a square shape and every call still returns a full matrix of
# plausible floats".
#
# CHOSEN BY LEAF COUNT, NOT BY SIZE. The whole set runs in well under a second
# and the point of every row is which clause of contract section 7.2 it
# evaluates.
#
# The `plan` column is PROVENANCE, NOT AN ASSERTION. `choose_gemm_plan` reads
# `m` and `n` and is allowed to (contract 6.1), the plan is EXECUTION rather
# than numerics, and Python cannot read it back. It is recorded so that a
# reader can see the rows were chosen to spread across the five dispatchable
# plans as of 2026-08-24, and so that a future plan change is a change to this
# comment rather than to a check.
#
#   name              m   n    k    P  why this row is here          plan
SELF_SHAPES = (
    # P = 3: ODD, so the last leaf is unpaired and CARRIED bit for bit
    # (contract 7.2 clause 3, the sharpest clause in the profile), and
    # 300 = 2*128 + 44 makes the last leaf RAGGED as well (section 8).
    # k = 300 is the contract's own named case, section 12.1.
    ("oddP3.40x20x300", 40, 20, 300),        # TILE_16_16_32
    # P = 5: the SMALLEST P that carries TWICE (contract 7.2.2). Levels are
    # 5 -> 3 -> 2 -> 1 and two of those transitions are odd.
    ("carry2.11x6x640", 11, 6, 640),         # SPLITK
    # P = 2: even, one pairing, no carry anywhere. The control for the two
    # rows above.
    ("evenP2.9x5x256", 9, 5, 256),           # FLAT
    # P = 1: no fold addition at all (7.3). Included because the surface must
    # handle it, NEVER as evidence on its own -- at k <= 128 the contract
    # degenerates to a serial chain that several different implementations
    # reproduce.
    ("oneleafP1.17x3x64", 17, 3, 64),        # FLAT
    # P = 3 at a wide `n`, to move the dispatch off the tiles above.
    ("wideN.8x48x384", 8, 48, 384),          # TILE_8_32_32
    # P = 3 at a tall `m` and a narrow `n`.
    ("tallM.36x9x300", 36, 9, 300),          # TILE_32_8_16
)

#: The shape the INVARIANCE arm uses, chosen so that pulling one row out of it
#: CHANGES THE EXECUTION PLAN. At (40, 20, 300) `choose_gemm_plan` returns
#: TILE_16_16_32; at (1, 20, 300) it falls through to FLAT, and at (40, 1, 300)
#: to TILE_32_8_16. So the arm is not "the same launch twice" -- it is the same
#: arithmetic under three different launch geometries, which is the property
#: contract 6.1 exists to protect and the one the serving world calls batch
#: invariance.
INVARIANCE_SHAPE = SELF_SHAPES[0]


# ===========================================================================
# ARM: PROVENANCE. DEVIATION 957.
# ===========================================================================


def arm_provenance(rep):
    """What this process is actually holding, checked against itself.

    `numeric_mode()` already cross-checks the binary's compile-time answer
    against the directory the loader chose and raises if they disagree. What
    is added here is the DIRECTORY check in the other direction, and the
    profile version, so that a stale `.so` beside a newer wrapper is caught
    before any number is believed rather than after.
    """
    arm = "PROVENANCE (957)"
    prof = linalg.profile()
    mode = linalg.numeric_mode()
    rep.check(arm, prof["profile"] == "mojolearn.identical.gemm.fp32.v1",
              "profile name is the one this gate is about",
              str(prof["profile"]))
    rep.check(arm, prof["profile_version"] == 1,
              "profile version is 1", str(prof["profile_version"]))
    rep.check(arm, prof["numeric_mode"] == mode,
              "profile() and numeric_mode() agree")
    rep.check(arm, prof["identity_claimed"] == (mode == "identical"),
              "identity_claimed tracks the mode, and nothing else")

    binary = prof["binary"]
    parent = os.path.basename(os.path.dirname(os.path.abspath(binary)))
    if mode == "identical":
        rep.check(arm, parent == "identical",
                  "the loaded binary is the one under identical/", binary)
    else:
        rep.check(arm, parent != "identical",
                  "the loaded binary is not the identical one", binary)
    rep.note("  binary  %s\n  surface %s\n  mode    %s"
             % (binary, SURFACE, mode))
    return mode


# ===========================================================================
# ARM: REFUSALS. DEVIATION 958.
# ===========================================================================


def arm_refusals(rep, extra):
    """Every guard on this surface, made to fire.

    These are the Python path's OWN policy and none of them exists in the
    kernel, so `gemm_device_check.mojo` cannot see any of them. `extra` is
    `{}` under IDENTICAL and `{"identical": False}` under FAST, because
    `matmul` calls `require_identical()` before it looks at anything else and
    a FAST process would otherwise get the mode refusal for every row.
    """
    arm = "REFUSALS (958)"
    a = fixture((6, 200), 101)
    b = fixture((200, 5), 202)

    rep.raises(arm, ValueError, "three operations",
               "DEVIATION 913, a.T @ b.T is refused by name",
               linalg.matmul, a, b, transpose_a=True, transpose_b=True,
               **extra)
    rep.raises(arm, TypeError, "float32",
               "float64 input is refused, never cast",
               linalg.matmul, a.astype(np.float64), b.astype(np.float64),
               **extra)
    rep.raises(arm, ValueError, "must be 2-D",
               "a 1-D operand is refused",
               linalg.matmul, a.ravel(), b, **extra)
    rep.raises(arm, ValueError, "no elements",
               "a zero-size operand is refused rather than answered",
               linalg.matmul, a[:0], b, **extra)
    rep.raises(arm, ValueError, "contracted extents disagree",
               "a k mismatch is caught on the host, not on the device",
               linalg.matmul, a, b[:199], **extra)

    good = np.empty((6, 5), dtype=np.float32)
    rep.raises(arm, TypeError, "out must be a numpy array",
               "out must be an ndarray",
               linalg.matmul, a, b, out=[[0.0] * 5] * 6, **extra)
    rep.raises(arm, TypeError, "float32",
               "out must be float32",
               linalg.matmul, a, b, out=good.astype(np.float64), **extra)
    rep.raises(arm, ValueError, "want (6, 5)",
               "out must have the output's shape",
               linalg.matmul, a, b, out=np.empty((5, 6), dtype=np.float32),
               **extra)
    rep.raises(arm, ValueError, "C-contiguous",
               "a non-contiguous out is refused rather than copied into",
               linalg.matmul, a, b,
               out=np.empty((6, 10), dtype=np.float32)[:, ::2], **extra)
    ro = np.empty((6, 5), dtype=np.float32)
    ro.flags.writeable = False
    rep.raises(arm, ValueError, "read-only",
               "a read-only out is refused before the device writes it",
               linalg.matmul, a, b, out=ro, **extra)


# ===========================================================================
# ARM: ORIENTATION. DEVIATION 953.
# ===========================================================================


def arm_orientation(rep):
    """OP_NN, OP_NT and OP_TN, reached through the flags, agreeing BIT for
    BIT on the same logical matrices.

    Contract section 3 makes this a REQUIREMENT of the profile rather than a
    coincidence: one numerical implementation, three addressings, and only
    `_a_at` and `_b_at` differ. `bindings/build_linalg.sh`'s smoke test checks
    the same three calls with `allclose` and says in its own comment why it
    cannot assert bits (it runs FAST). This runs IDENTICAL and asserts them.

    THIS IS THE ONLY ARM IN THE FILE THAT REACHES OP_NN AT ALL. The reference
    card is emitted over `bench/gemm_shapes.mojo`, whose twenty rows are four
    OP_TN and sixteen OP_NT and not one OP_NN (`gemm_shape_op`: `i <= 3` is
    TN, everything else is NT). So OP_NN has no external reference anywhere in
    this tree's Python path, and what stands behind it here is this agreement
    plus the ACCURACY arm.

    A wrong op code cannot hide in these shapes. `m`, `n` and `k` are pairwise
    distinct, so an addressing that belongs to another orientation reads the
    operand along the wrong extent and produces a different matrix, not a
    different last bit.
    """
    arm = "ORIENTATION (953)"
    for name, m, n, k in SELF_SHAPES:
        a = fixture((m, k), 101)
        b = fixture((k, n), 202)
        nn = linalg.matmul(a, b)
        nt = linalg.matmul(a, np.ascontiguousarray(b.T), transpose_b=True)
        tn = linalg.matmul(np.ascontiguousarray(a.T), b, transpose_a=True)
        rep.check(arm, nn.shape == (m, n) and nn.dtype == np.float32,
                  "%-18s OP_NN returns (%d, %d) float32" % (name, m, n),
                  "%s %s" % (nn.shape, nn.dtype))
        rep.bits_equal(arm, nt, nn,
                       "%-18s OP_NT == OP_NN, bitwise  [%s]"
                       % (name, leaf_note(k)))
        rep.bits_equal(arm, tn, nn,
                       "%-18s OP_TN == OP_NN, bitwise  [%s]"
                       % (name, leaf_note(k)))

    # gemv is OP_NT at n == 1 and is NOT a fourth operation (contract 0.1).
    # `core/gemm.mojo` routes it to a separate kernel for a measured
    # correctness reason (transpose_b left 63 of 64 output rows UNWRITTEN at
    # m=64, n=1, k=32 on 2026-08-19), and the contract requires the routed
    # answer to equal the OP_NT answer bit for bit. That is a KERNEL property;
    # what is checked here is that the Python path reaches it and that its
    # single column matches the same column of a wider call.
    m, n, k = 36, 9, 300
    a = fixture((m, k), 303)
    b = fixture((n, k), 404)
    wide = linalg.matmul(a, b, transpose_b=True)
    col = linalg.matmul(a, np.ascontiguousarray(b[4:5]), transpose_b=True)
    rep.check(arm, col.shape == (m, 1), "gemv returns (%d, 1)" % m,
              str(col.shape))
    rep.bits_equal(arm, col, wide[:, 4:5],
                   "gemv (n == 1) == column 4 of the n = 9 call, bitwise")


# ===========================================================================
# ARM: STRIDES. DEVIATION 954.
# ===========================================================================


def arm_strides(rep):
    """A non-contiguous operand gives BIT-IDENTICAL output to its contiguous
    copy.

    This must hold exactly, and the reason is one line: reordering float32
    values moves no bit. `_operand` accepts a non-contiguous array with
    `np.ascontiguousarray`, and contract section 2 requires contiguity because
    the device addresses row-major with no leading dimension and no stride.

    THIS IS WHERE A STRIDE ASSUMPTION BREAKS. If the copy were skipped, or
    taken in the wrong order, or if the raw address of a VIEW reached the
    device -- which is what `_addr` hands over, with no stride beside it --
    the result is a plausible matrix built from the wrong elements. The
    transposed-view row below is the sharpest of the four: `a.T` is exactly
    the array a caller reaches for when they want OP_TN, and it is a view
    whose first byte is the same first byte as `a`.
    """
    arm = "STRIDES (954)"
    m, n, k = 40, 20, 300
    a = fixture((m, k), 101)
    b = fixture((k, n), 202)
    base = linalg.matmul(a, b)

    wide_a = fixture((m, 2 * k), 505)
    view_a = wide_a[:, ::2]
    rep.check(arm, not view_a.flags["C_CONTIGUOUS"],
              "the strided operand really is non-contiguous")
    rep.bits_equal(arm,
                   linalg.matmul(view_a, b),
                   linalg.matmul(np.ascontiguousarray(view_a), b),
                   "a[:, ::2] == its contiguous copy, bitwise")

    rev_a = np.ascontiguousarray(a)[::-1]
    rep.bits_equal(arm,
                   linalg.matmul(rev_a, b),
                   linalg.matmul(np.ascontiguousarray(rev_a), b),
                   "a reversed-row view == its contiguous copy, bitwise")

    f_a = np.asfortranarray(a)
    rep.check(arm, not f_a.flags["C_CONTIGUOUS"],
              "the Fortran-order operand really is non-contiguous")
    rep.bits_equal(arm, linalg.matmul(f_a, b), base,
                   "a Fortran-order copy of `a` == the C-order call, bitwise")

    rep.bits_equal(arm,
                   linalg.matmul(a.T, b, transpose_a=True),
                   linalg.matmul(np.ascontiguousarray(a.T), b,
                                 transpose_a=True),
                   "OP_TN through the VIEW a.T == through its copy, bitwise")

    wide_b = fixture((k, 2 * n), 606)
    view_b = wide_b[:, ::2]
    rep.bits_equal(arm,
                   linalg.matmul(a, view_b),
                   linalg.matmul(a, np.ascontiguousarray(view_b)),
                   "b[:, ::2] == its contiguous copy, bitwise")


# ===========================================================================
# ARM: OUT. DEVIATION 955.
# ===========================================================================


def arm_out(rep):
    """`out=` gives bit-identical output to the allocated form, and every
    cell of it is written.

    THE OUTPUT IS POISONED WITH NaN FIRST. A cell the device never wrote would
    otherwise hold whatever the allocator handed back, which is a wrong answer
    that looks exactly like an arithmetic one. `gemm/host_entry.mojo`'s own
    header records the cost of getting this wrong on the workspace instead: a
    one-float workspace passed to a SPLITK dispatch produced right answers at
    64 x 4 and regions of `+0.0` at 64 x 64.

    NaN and not zero, because zero is a value the profile produces
    (contract section 8 requires `+0.0` to be STORED at `k == 0`) and NaN is
    not.
    """
    arm = "OUT (955)"
    for name, m, n, k in SELF_SHAPES:
        a = fixture((m, k), 101)
        b = fixture((k, n), 202)
        alloc = linalg.matmul(a, b)
        out = np.full((m, n), np.nan, dtype=np.float32)
        got = linalg.matmul(a, b, out=out)
        rep.check(arm, got is out, "%-18s out= returns the caller's array"
                  % name)
        rep.check(arm, not np.isnan(out).any(),
                  "%-18s no cell of out was left unwritten" % name,
                  "%d NaN cells survived" % int(np.isnan(out).sum()))
        rep.bits_equal(arm, out, alloc,
                       "%-18s out= == the allocated form, bitwise" % name)


# ===========================================================================
# ARM: INVARIANCE. DEVIATION 956.
# ===========================================================================


def arm_invariance(rep):
    """One row of a result is bit-identical whether it was computed alone or
    as part of a larger call.

    This is the profile's headline invariant and the reason section 6.1
    forbids the leaf size to depend on `m` or `n`: `m` is the batch dimension
    of a token GEMM, and if `L = f(k, m)` then the same row of `A` against the
    same column of `B` returns different bits depending on how many other rows
    shared the launch. `gemm_device_check.mojo` gates that at the kernel. What
    is gated HERE is that the Python surface does not break it -- by passing a
    wrong `m`, by sizing a buffer from the wrong extent, or by handing the
    device an address into the middle of an array whose stride it cannot see.

    THE ROW-ALONE CALL DISPATCHES A DIFFERENT EXECUTION PLAN. See
    INVARIANCE_SHAPE. That is what makes this more than the same launch twice.
    """
    arm = "INVARIANCE (956)"
    name, m, n, k = INVARIANCE_SHAPE
    a = fixture((m, k), 101)
    b = fixture((k, n), 202)
    full = linalg.matmul(a, b)

    rep.bits_equal(arm, linalg.matmul(a, b), full,
                   "the same call twice is the same bits (launch invariance)")

    for i in (0, 1, 17, m - 1):
        rep.bits_equal(arm, linalg.matmul(a[i:i + 1], b), full[i:i + 1],
                       "row %2d alone == row %2d of the m = %d call, bitwise"
                       % (i, i, m))

    rep.bits_equal(arm, linalg.matmul(a[8:24], b), full[8:24],
                   "rows 8..24 alone == the same rows of the m = %d call" % m)

    # BATCH COMPOSITION, not just batch size. The rows are gathered in an
    # order they do not have in `a`, so a result that depended on a row's
    # POSITION in the launch -- rather than on its contents -- fails here and
    # passes every contiguous-slice check above.
    order = [5, 31, 2]
    gathered = np.ascontiguousarray(a[order])
    got = linalg.matmul(gathered, b)
    for q, i in enumerate(order):
        rep.bits_equal(arm, got[q:q + 1], full[i:i + 1],
                       "row %2d in a REORDERED batch of 3 == row %2d of the "
                       "m = %d call" % (i, i, m))

    rep.bits_equal(arm, linalg.matmul(a, b[:, 7:8]), full[:, 7:8],
                   "column 7 alone == column 7 of the n = %d call, bitwise"
                   % n)
    rep.bits_equal(arm, linalg.matmul(a, b[:, 4:12]), full[:, 4:12],
                   "columns 4..12 alone == the same columns of the n = %d "
                   "call" % n)


# ===========================================================================
# ARM: ACCURACY. NOT AN IDENTITY ARM.
# ===========================================================================


def arm_accuracy(rep):
    """A float64 reference, at a TOLERANCE, and it proves nothing about bits.

    Stated twice because it matters. The profile's whole product is a specific
    summation order, and float64 does not reproduce that order -- contract 7.4
    is explicit that the partitioned answer is a DIFFERENT answer from the
    serial one and more accurate than it, and neither is numpy's. So this arm
    cannot see a fold defect, a leaf defect, a flush defect or a
    multiply-add defect, and it is not here to.

    What it CAN see is a gross marshalling error at a shape the reference card
    does not carry -- a transposed operand, a wrong element count, a swapped
    `m` and `n` -- and it covers OP_NN, which the card has no row for at all.
    A defect of that class moves the answer by whole magnitudes, not by ULPs.
    """
    arm = "ACCURACY (not identity)"
    for name, m, n, k in SELF_SHAPES:
        a = fixture((m, k), 101)
        b = fixture((k, n), 202)
        ref = (a.astype(np.float64) @ b.astype(np.float64))
        scale = max(1.0, float(np.max(np.abs(ref))))
        for label, got in (
            ("OP_NN", linalg.matmul(a, b)),
            ("OP_NT", linalg.matmul(a, np.ascontiguousarray(b.T),
                                    transpose_b=True)),
            ("OP_TN", linalg.matmul(np.ascontiguousarray(a.T), b,
                                    transpose_a=True)),
        ):
            err = float(np.max(np.abs(got.astype(np.float64) - ref))) / scale
            rep.check(arm, err <= 1e-3,
                      "%-18s %s within 1e-3 of a float64 reference (%.2e)"
                      % (name, label, err))


# ===========================================================================
# ARM: CARD. DEVIATIONS 951, 952, 959.
# ===========================================================================
# THE ONLY ARM WITH A REFERENCE THIS PATH DID NOT PRODUCE.
#
# WHY THE CARD AND NOT SOMETHING SIMPLER. Three mechanisms were available and
# the two rejected ones are weaker for reasons worth recording:
#
#   A TABLE OF EXPECTED BIT PATTERNS committed beside this file would be a
#   reference nobody can regenerate without trusting whoever generated it, and
#   the failure mode of a stale table is a gate that goes red after a
#   legitimate v2 and gets edited until it is green. The card is regenerated
#   by one command from the NORMATIVE oracle and carries its own provenance.
#
#   A FLOAT64 REFERENCE cannot reproduce the summation order the profile IS.
#   It is the ACCURACY arm above and it is not an identity arm.
#
# WHAT THE CARD IS. `tools/gemm_card.sh oracle` runs
# `bench/gemm_card_main.mojo` over `bench/gemm_shapes.mojo` and writes one
# record per stage: `<seq>\t<tag>\t<dtype>\t<count>\t<hash>`, FNV-1a64 over
# the little-endian bytes, three stages per shape -- `<name>.in.a`,
# `<name>.in.b`, `<name>.out`. The `.out` hash is `gemm_oracle`'s answer.
#
# HOW THIS ARM AVOIDS BECOMING A SECOND COPY OF THE SHAPE TABLE. It does not
# transcribe `m`, `n` and `k`, and it does not transcribe the host cap that
# reduces them. It SOLVES for them out of the card's own element counts:
#
#     count_a = m*k     count_b = n*k     count_out = m*n
#     => k = isqrt(count_a * count_b / count_out), m = count_a/k, n = count_b/k
#
# exact in integers for every shape, in all three orientations, because
# `A` is `m*k` and `B` is `n*k` in ALL of them (`gemm/host_entry.mojo` says
# the same thing about its own buffer sizing). What IS transcribed is the
# fixture generator, the twenty shape NAMES and the one-line op rule -- and
# the names are asserted against the card position by position, so a table
# edit, an inserted row or a skipped shape is a loud failure that names the
# table rather than a wrong verdict about the GEMM.

#: `bench/gemm_shapes.mojo`, in table order. Transcribed so that the index
#: this arm derives the SALT and the OP from is CHECKED rather than assumed.
#: A card whose stage names do not match this list, in this order, is a card
#: this gate cannot interpret, and it aborts rather than guessing.
TABLE_NAMES = (
    "gram.32x32x1M",
    "gram.32x32x64K",
    "gram.128sq.x100003",
    "ols.step1.16x16x64K",
    "pca.transform.8192x4x4",
    "pca.transform.wide.8192x64x128",
    "kmeans.dist.4096x64x64",
    "ols.predict.gemv.64Kx16",
    "llama8b.qkv.t1",
    "llama8b.qkv.t8",
    "llama8b.qkv.t512",
    "llama8b.mlp_up.t1",
    "llama8b.mlp_up.t8",
    "llama8b.mlp_up.t512",
    "llama8b.mlp_down.t1",
    "llama8b.mlp_down.t8",
    "llama8b.mlp_down.t512",
    "llama8b.lm_head.t1",
    "llama8b.lm_head.t8",
    "llama8b.lm_head.t512",
)


def table_op(idx):
    """`bench/gemm_shapes.mojo::gemm_shape_op`, one line, and the only column
    of that table this file cannot solve for out of the card."""
    return "TN" if idx <= 3 else "NT"


#: Per shape, in float32 elements over A, B and C. THE HASH IS THE COST AND
#: NOT THE GPU: the differ's FNV-1a64 is a byte-at-a-time Python loop over
#: every input and output element, so the budget is a bound on that loop. No
#: rate is quoted here because none was measured; time one run and set
#: MOJOLEARN_LINALG_GATE_BUDGET to what the box will wear.
#:
#: 4Mi leaves fifteen of the twenty rows of today's table under it, with leaf
#: counts P = 782, 512, 112, 32 and 1. What it defers is the four m = 1
#: transformer rows, whose leaf counts are already covered by their t8 and
#: t512 siblings, and the 1,000,000-k Gram row, whose leaf count is not
#: covered by anything -- see `leaf_branch` and the admission rule below,
#: which puts that row back.
#:
#: A BUDGET MAY NEVER SILENCE THE IDENTITY ARM (DEVIATION 959), and that is
#: enforced in three ways rather than intended: a branch of the leaf rule the
#: budget would leave empty gets its cheapest shape admitted anyway; the arm
#: FAILS if no checked shape has `P > 1` for an op the card carries; and every
#: deferred shape is printed by name with the command that runs it.
DEFAULT_BUDGET = 4 * 1024 * 1024


def leaf_branch(k):
    """Which BRANCH of `contract_leaf_size` this `k` takes.

    Contract section 6 is three rules and not one, and the three produce
    different leaf boundaries from the same code path:

        k <= 128                     L = k          one leaf, no fold
        ceil(k/128) <= 1024          L = 128        the flat branch
        otherwise                    L = ceil(k/1024)   the CAPPED branch

    The capped branch is reached by exactly one row of `bench/gemm_shapes.mojo`
    (`gram.32x32x1M`, k = 1,000,000, L = 977, P = 1024) and by nothing else in
    the table. `bench/gemm_shapes.mojo`'s own header records that the first
    version of that file mirrored a FLAT `L = 128` and reported `P = 7813` for
    that row against a true 1024, so this is the branch a transcription is
    most likely to get wrong and the one the fewest shapes reach.

    A budget that dropped the only row in a branch would leave the branch
    unchecked while the summary still said fifteen shapes agreed, which is the
    shape of a gate that has stopped gating.
    """
    if k <= K_LEAF_MIN:
        return "one-leaf (L = k)"
    if (k + K_LEAF_MIN - 1) // K_LEAF_MIN <= MAX_LEAVES:
        return "flat (L = 128)"
    return "capped (L = ceil(k / 1024))"


def load_differ():
    """`tools/identity_trace_diff.py`, imported by path.

    IMPORTED RATHER THAN REIMPLEMENTED. Its `parse_trace` is the definition of
    the card format, including the rules this gate would otherwise have to
    restate (seq strictly increasing by one, sixteen lowercase hex digits, the
    dtype table), and its `fnv1a64_hex` is the definition of the hash the card
    holds. A private copy of either would let this gate and the card disagree
    about the format and call it a divergence.
    """
    root = os.environ.get("MOJOLEARN_REPO")
    cands = [root] if root else []
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cands.append(here)
        here = os.path.dirname(here)
    for c in cands:
        if not c:
            continue
        path = os.path.join(c, "tools", "identity_trace_diff.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "mojolearn_identity_trace_diff", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, c
    raise GateAbort(
        "cannot find tools/identity_trace_diff.py from %s. That file is the\n"
        "definition of the card format AND of the hash the card holds, and a\n"
        "private copy of either would let this gate and the card disagree\n"
        "about the format and call it a divergence. Point MOJOLEARN_REPO at\n"
        "the checkout." % os.path.abspath(__file__))


def card_triples(differ, path):
    """The card's records, grouped one triple per shape, with `m`, `n` and `k`
    solved out of the element counts."""
    records, _, _ = differ.parse_trace(path)
    if not records:
        raise GateAbort("%s holds no records" % path)
    if len(records) % 3:
        raise GateAbort(
            "%s holds %d records, which is not three per shape. The oracle\n"
            "arm writes <name>.in.a, <name>.in.b and <name>.out for every\n"
            "shape it runs; a card that does not is not one this gate can\n"
            "interpret." % (path, len(records)))
    out = []
    for idx in range(len(records) // 3):
        ra, rb, rc = records[3 * idx:3 * idx + 3]
        if not (ra.tag.endswith(".in.a") and rb.tag.endswith(".in.b")
                and rc.tag.endswith(".out")):
            raise GateAbort(
                "%s: stages %d..%d are %r, %r, %r, want a .in.a / .in.b / "
                ".out triple" % (path, ra.seq, rc.seq, ra.tag, rb.tag, rc.tag))
        name = ra.tag[:-len(".in.a")]
        if rb.tag[:-len(".in.b")] != name or rc.tag[:-len(".out")] != name:
            raise GateAbort("%s: stages %d..%d name three different shapes"
                            % (path, ra.seq, rc.seq))
        if idx >= len(TABLE_NAMES) or name != TABLE_NAMES[idx]:
            raise GateAbort(
                "%s: shape %d is %r, and this gate expects %r there.\n"
                "\n"
                "THE INDEX IS LOAD-BEARING and that is why it is checked. The\n"
                "card carries no salt and no op, so this gate derives both\n"
                "from the shape's POSITION in bench/gemm_shapes.mojo (salt\n"
                "11 + i and 22 + i, and `i <= 3` is OP_TN). A card whose\n"
                "shapes are not that table in that order would be read with\n"
                "the wrong salt and the wrong orientation, and the fixture\n"
                "gate would then report a divergence that is this gate's\n"
                "fault and not the surface's.\n"
                "\n"
                "Either bench/gemm_shapes.mojo changed, in which case update\n"
                "TABLE_NAMES and table_op() in this file, or the run that\n"
                "emitted this card SKIPPED a shape, which its log reports as\n"
                "a SKIP line and a nonzero skip count."
                % (path, idx, name, TABLE_NAMES[idx] if idx < len(TABLE_NAMES)
                   else "<past the end of the table>"))
        ca, cb, co = ra.count, rb.count, rc.count
        if co <= 0 or (ca * cb) % co:
            raise GateAbort("%s: %s has counts a=%d b=%d out=%d, which are "
                            "not m*k, n*k and m*n" % (path, name, ca, cb, co))
        k2 = (ca * cb) // co
        k = math.isqrt(k2)
        if k <= 0 or k * k != k2 or ca % k or cb % k:
            raise GateAbort("%s: %s has counts a=%d b=%d out=%d, from which "
                            "no integer (m, n, k) follows"
                            % (path, name, ca, cb, co))
        m, n = ca // k, cb // k
        if m * n != co or m * k != ca or n * k != cb:
            raise GateAbort("%s: %s solved to m=%d n=%d k=%d, which does not "
                            "reproduce its counts" % (path, name, m, n, k))
        out.append((idx, name, table_op(idx), m, n, k, ra, rb, rc))
    return out


def arm_card(rep):
    """The absolute arm. Every checked shape's Python-path output, hashed and
    compared against the hash `gemm_oracle` produced for the same fixture.

    THE FIXTURE IS CHECKED BEFORE THE PRODUCT (DEVIATION 952), which is
    `tools/gemm_card.sh`'s own rule for its compare arm and for the same
    reason: if the inputs differ, a product divergence is a diff of the
    FIXTURES and every conclusion below it is void. So a fixture mismatch
    ABORTS this arm rather than failing a check, because the conclusion is
    "the gate did not run", not "the surface is wrong".
    """
    arm = "CARD (951)"
    path = os.environ.get("MOJOLEARN_GEMM_CARD", "").strip()
    if not path:
        raise GateAbort(
            "MOJOLEARN_GEMM_CARD is unset, so this run has NO EXTERNAL\n"
            "REFERENCE and every other arm in this file is a relative\n"
            "comparison that cannot see an answer that is wrong the same way\n"
            "everywhere. Produce the reference and point at it:\n"
            "\n"
            "    tools/gemm_card.sh oracle /tmp/gemm_oracle.card\n"
            "    MOJOLEARN_GEMM_CARD=/tmp/gemm_oracle.card ...\n"
            "\n"
            "That runs bench/gemm_card_main.mojo's ORACLE arm, which is\n"
            "gemm/checks/gemm_oracle.mojo::gemm_oracle, the NORMATIVE v1\n"
            "answer. It takes about a minute and needs no GPU.\n"
            "\n"
            "This is a FAILURE and not a skip. A silently skipped identity\n"
            "test is worse than no test.")
    differ, root = load_differ()
    triples = card_triples(differ, path)

    budget = int(os.environ.get("MOJOLEARN_LINALG_GATE_BUDGET",
                                DEFAULT_BUDGET))
    if os.environ.get("MOJOLEARN_LINALG_GATE_FULL") == "1":
        budget = 0

    # THE ADMISSION RULE, BEFORE ANYTHING RUNS. A shape is checked when it
    # fits the budget, and ALSO when it is the cheapest shape in a branch of
    # the leaf rule that nothing else in the card would cover. See
    # `leaf_branch`: the second clause is what keeps `gram.32x32x1M` in, and
    # with it the only capped-leaf row this table has.
    totals = dict((t[1], t[6].count + t[7].count + t[8].count)
                  for t in triples)
    admit = set(t[1] for t in triples
                if budget == 0 or totals[t[1]] <= budget)
    forced = {}
    for branch in sorted(set(leaf_branch(t[5]) for t in triples)):
        rows = [t for t in triples if leaf_branch(t[5]) == branch]
        if any(t[1] in admit for t in rows):
            continue
        cheapest = min(rows, key=lambda t: totals[t[1]])
        admit.add(cheapest[1])
        forced[cheapest[1]] = branch

    checked, deferred = [], []
    lines = ["  card %s\n  differ %s" % (path, os.path.join(root, "tools"))]
    lines.append("    %-32s %-3s %7s %7s %9s %6s %10s  %s"
                 % ("shape", "op", "m", "n", "k", "P", "elements",
                    "leaf-rule branch"))
    our_records = []

    for idx, name, op, m, n, k, ra, rb, rc in triples:
        total = totals[name]
        p = contract_leaf_count(k)
        if name in forced:
            mark = "   ADMITTED OVER BUDGET (only row in its branch)"
        elif name in admit:
            mark = ""
        else:
            mark = "   DEFERRED"
        lines.append("    %-32s %-3s %7d %7d %9d %6d %10d  %-26s%s"
                     % (name, op, m, n, k, p, total, leaf_branch(k), mark))
        if name not in admit:
            deferred.append((name, total, p, op))
            continue

        # The fixtures, in the layout each orientation stores. `_fill` walks a
        # flat buffer, and the oracle reads it row-major (`_a_at`, `_b_at`), so
        # the reshape is the whole of the layout.
        if op == "TN":
            a = fixture_values(ra.count, 11 + idx).reshape(k, m)
            b = fixture_values(rb.count, 22 + idx).reshape(k, n)
            kw = {"transpose_a": True}
        else:
            a = fixture_values(ra.count, 11 + idx).reshape(m, k)
            b = fixture_values(rb.count, 22 + idx).reshape(n, k)
            kw = {"transpose_b": True}

        ha = differ.fnv1a64_hex(a.tobytes())
        hb = differ.fnv1a64_hex(b.tobytes())
        if ha != ra.hash or hb != rb.hash:
            raise GateAbort(
                "FIXTURE MISMATCH at %s (shape %d of the card).\n"
                "    .in.a  card %s  this gate %s\n"
                "    .in.b  card %s  this gate %s\n"
                "\n"
                "The two sides are not looking at the same input matrices, so\n"
                "NOTHING can be concluded about the product and this arm has\n"
                "stopped rather than compare one. Three causes, in the order\n"
                "they are worth checking:\n"
                "\n"
                "  1. bench/gemm_card_main.mojo::_mix or ::_exact changed.\n"
                "     fixture_values() in this file is a transcription of\n"
                "     them and has to be exact to the bit.\n"
                "  2. the salt rule changed. This gate uses 11 + i for A and\n"
                "     22 + i for B, with `i` the shape's index in\n"
                "     bench/gemm_shapes.mojo.\n"
                "  3. the card was emitted by a build whose fixture differs\n"
                "     for some other reason. Its log carries the arm, the\n"
                "     mode and the cap.\n"
                "\n"
                "None of the three is a defect in the GEMM."
                % (name, idx, ra.hash, ha, rb.hash, hb))

        out = np.full((m, n), np.nan, dtype=np.float32)
        got = linalg.matmul(a, b, out=out, **kw)
        if np.isnan(got).any():
            rep.bad(arm, "%-32s %d cells were left UNWRITTEN by the device"
                    % (name, int(np.isnan(got).sum())))
            continue
        hc = differ.fnv1a64_hex(got.tobytes())
        rep.check(arm, hc == rc.hash,
                  "%-32s op=%s m=%d n=%d k=%d %s  ==  gemm_oracle, bitwise"
                  % (name, op, m, n, k, leaf_note(k)),
                  "card %s, this path %s" % (rc.hash, hc))
        checked.append((name, p, op))
        our_records.append((ra.tag, "f32", ra.count, ha))
        our_records.append((rb.tag, "f32", rb.count, hb))
        our_records.append((rc.tag, "f32", rc.count, hc))

    # THE GUARD ON THE BUDGET. DEVIATION 959.
    rep.check(arm, len(checked) > 0,
              "the card arm checked at least one shape",
              "every shape was deferred; raise MOJOLEARN_LINALG_GATE_BUDGET")
    ops_in_card = sorted(set(t[2] for t in triples))
    for op in ops_in_card:
        has_big = any(p > 1 and o == op for _, p, o in checked)
        card_has_big = any(contract_leaf_count(t[5]) > 1 and t[2] == op
                           for t in triples)
        if not card_has_big:
            continue
        rep.check(arm, has_big,
                  "at least one CHECKED %s shape has P > 1" % op,
                  "every P > 1 %s shape was deferred, and a check at P == 1 "
                  "cannot separate the contract's fold from a plain serial "
                  "chain (contract 7.5)" % op)
    rep.check(arm, any(p == 1 for _, p, _ in checked)
              or not any(contract_leaf_count(t[5]) == 1 for t in triples),
              "at least one CHECKED shape has P == 1 (contract 7.3)")
    checked_names = set(c[0] for c in checked)
    for branch in sorted(set(leaf_branch(t[5]) for t in triples)):
        rep.check(arm,
                  any(leaf_branch(t[5]) == branch and t[1] in checked_names
                      for t in triples),
                  "the %s branch of the leaf rule was CHECKED" % branch,
                  "every shape in this branch of contract section 6 was "
                  "deferred or failed, so this run says nothing about it")

    if deferred:
        lines.append("")
        lines.append("    %d shapes DEFERRED by the element budget of %d "
                     "float32 elements. They are not skipped quietly and they "
                     "are not counted as agreeing." % (len(deferred), budget))
        for name, total, p, op in deferred:
            lines.append("      %-32s %10d elements  P=%d %s"
                         % (name, total, p, op))
        lines.append("    Run them with MOJOLEARN_LINALG_GATE_FULL=1, which "
                     "also makes this gate's own card diffable against the "
                     "oracle card stage for stage.")

    # OUR OWN CARD, in the same format, so a failure can be taken to the
    # differ for a per-cell report instead of a hash that differs.
    out_path = os.environ.get("MOJOLEARN_LINALG_GATE_CARD_OUT") or \
        os.path.join(tempfile.gettempdir(), "mojolearn_linalg_gate.card")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# format: mojolearn-identity-trace v1\n")
        for seq, (tag, dt, count, h) in enumerate(our_records):
            fh.write("%d\t%s\t%s\t%d\t%s\n" % (seq, tag, dt, count, h))
    lines.append("")
    lines.append("    this gate's own card: %s" % out_path)
    if deferred:
        lines.append("    (it holds only the CHECKED shapes, so "
                     "identity_trace_diff.py will report the deferred ones as "
                     "unmatched tags. Use MOJOLEARN_LINALG_GATE_FULL=1 for a "
                     "card that aligns with the oracle's.)")
    else:
        lines.append("    python3 %s \\\n        %s %s"
                     % (os.path.join(root, "tools", "identity_trace_diff.py"),
                        path, out_path))
    rep.note("\n".join(lines))


# ===========================================================================
# ARM: FAST. DEVIATION 957.
# ===========================================================================


def arm_fast(rep):
    """The mode discipline, which can only be checked in a FAST process.

    `MOJOLEARN_NUMERIC_MODE` is read at import and the selected set is cached,
    so one process cannot see both modes and this arm cannot be folded into
    the run above. It is opt-in through MOJOLEARN_LINALG_GATE_ALLOW_FAST for
    one reason: a FAST run of this file checks NO BITS AT ALL, and a CI job
    that reached it by accident would report a green identity gate that never
    looked at an identity.
    """
    arm = "FAST (957)"
    a = fixture((6, 200), 101)
    b = fixture((200, 5), 202)

    rep.check(arm, linalg.profile()["identity_claimed"] is False,
              "a FAST build claims no identity")
    rep.raises(arm, RuntimeError, "FAST build",
               "DEVIATION 911, the default call REFUSES rather than returning "
               "the fast product",
               linalg.matmul, a, b)
    rep.raises(arm, RuntimeError, "MOJOLEARN_NUMERIC_MODE=identical",
               "the refusal names the environment variable that fixes it",
               linalg.matmul, a, b)
    rep.raises(arm, RuntimeError, "bindings/build_linalg.sh",
               "the refusal names the build script",
               linalg.matmul, a, b)
    rep.raises(arm, RuntimeError, "identical=False",
               "the refusal names the way to ask for the fast product on "
               "purpose",
               linalg.matmul, a, b)
    rep.raises(arm, RuntimeError, "FAST build",
               "require_identical() raises on its own",
               linalg.require_identical)

    got = linalg.matmul(a, b, identical=False)
    rep.check(arm, got.shape == (6, 5) and got.dtype == np.float32,
              "identical=False returns a (6, 5) float32 product",
              "%s %s" % (got.shape, got.dtype))
    rep.check(arm, np.isfinite(got).all(),
              "identical=False returns finite values")
    rep.bits_equal(arm, linalg.matmul(a, b, identical=False), got,
                   "the FAST product is at least repeatable in one process")


# ===========================================================================


def main(argv=None):
    out = sys.stdout
    out.write("== mojolearn.tests.test_linalg_identity ==\n")
    out.write("   profile mojolearn.identical.gemm.fp32.v1\n")
    out.write("   surface %s\n" % SURFACE)

    rep = Report()
    aborted = []
    try:
        mode = arm_provenance(rep)
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        out.write("\nCANNOT START: %s\n" % exc)
        return 2

    if mode == "identical":
        arms = (
            ("REFUSALS", lambda: arm_refusals(rep, {})),
            ("ORIENTATION", lambda: arm_orientation(rep)),
            ("STRIDES", lambda: arm_strides(rep)),
            ("OUT", lambda: arm_out(rep)),
            ("INVARIANCE", lambda: arm_invariance(rep)),
            ("ACCURACY", lambda: arm_accuracy(rep)),
            ("CARD", lambda: arm_card(rep)),
        )
    elif os.environ.get("MOJOLEARN_LINALG_GATE_ALLOW_FAST") == "1":
        arms = (
            ("FAST", lambda: arm_fast(rep)),
            ("REFUSALS", lambda: arm_refusals(rep, {"identical": False})),
        )
    else:
        out.write(
            "\nTHIS PROCESS LOADED THE FAST BUILD, WHICH MAKES NO IDENTITY\n"
            "CLAIM OF ANY KIND, so the identity arms of this gate cannot run\n"
            "and it will not pretend they did.\n"
            "\n"
            "  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GEMM_CARD=... \\\n"
            "      python3 -m mojolearn.tests.test_linalg_identity\n"
            "\n"
            "having built the identical extension with\n"
            "\n"
            "  MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh\n"
            "\n"
            "To check the FAST half on purpose -- the refusals and the mode\n"
            "discipline, which check NO BITS -- set\n"
            "MOJOLEARN_LINALG_GATE_ALLOW_FAST=1.\n")
        return 1

    for name, fn in arms:
        try:
            fn()
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
        out.write("test_linalg_identity: RED. %d checks failed, %d arms did "
                  "not run.\n" % (len(rep.failures), len(aborted)))
        return 1
    if mode != "identical":
        out.write(
            "test_linalg_identity: the FAST arms passed, AND NO BIT WAS\n"
            "CHECKED. This run says the refusals fire and the mode discipline\n"
            "holds. It says NOTHING about "
            "mojolearn.identical.gemm.fp32.v1.\n")
        return 0
    out.write(
        "test_linalg_identity: GREEN. The Python surface returns "
        "gemm_oracle's\n"
        "bits, through OP_NN, OP_NT and OP_TN, at leaf counts above and "
        "below\n"
        "the fold, from strided operands, into a caller's buffer, and one row\n"
        "at a time. gemm/PYTHON_SURFACE_GATE.md is what it does NOT prove.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
