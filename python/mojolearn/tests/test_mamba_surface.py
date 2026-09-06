# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.mamba` (Mamba1Block,
Mamba2Block, Mamba3Block).

Written 2026-09-01, the day the surface landed -- the day
`archive/evidence/mamba/FEATURE_PARITY.md`'s "PyPI surface: NONE EXISTS" row closed. The
model for this file is `test_gp_surface.py` and the house standard is
`archive/evidence/gemm/PYTHON_SURFACE_GATE.md`. The Mamba-3 arms joined later the same
day, when `Mamba3Block` did.

WHAT THIS CLOSES. `bindings/_mojolearn_mamba.mojo`'s six entry points
and `python/mojolearn/_mamba_impl.py` put the certified Mamba blocks in
reach of a Python caller. NOTHING IN THAT PATH IS COVERED BY THE LANES'
GATES: those run inside Mojo (`pixi run check-mamba-block`,
`check-mamba2`, `mamba/checks/mamba3_check.mojo`, and siblings),
against their own oracles, and stop at `mamba_block_forward` /
`mamba2_block_forward` / `mamba3_block_forward`. Everything here is
downstream of that point: an addrs list whose order is written out twice
and could be written out wrong once, state buffers marshaled up and back
per call, a buf_len (and, for Mamba-3, a pending flag) that must survive
the round trip, and dtype/shape refusals that exist only on this side.

THE EXPECTED VALUES ARE THE COMMITTED CORPUS'S, NOT OUR OWN TALLY. The
Mamba-1/2 forward arms run corpus cases (`mamba/corpus/base_b2_l4_d8`,
`mamba/corpus/mamba2/m2_base_b2_l4_d32`) and compare against their
`ref64` stages at the corpus README's calibrated tolerance (rtol 1e-5,
atol 1e-6) -- a TOLERANCE anchor, per that README: the bitwise oracle is
the lane's, and this file makes no cross-vendor claim. The corpus lives
in the repository, not the wheel, so this gate runs IN-REPO; a missing
Mamba-1/2 corpus is a FAILURE naming the path, never a skip
([[not-applicable-is-not-a-pass]]). All three families require their independent corpus cases; a missing
Mamba-3 case fails just as a missing Mamba-1/2 case does. Mamba-3 also
checks decode against prefill, split prefill against whole, and repeated
continuation through the binding on fixture-scale inputs.

WHAT IS ASSERTED AND WHAT IS REPORTED. Under `identical` the bitwise
arms -- decode == prefill per token, and split prefill (L-1 tokens then
one step) == whole prefill, and for Mamba-3 the Q = 64 chunk-crossing
resumption and the repeated Input_States continuation -- are ASSERTED;
under `fast` they are REPORTED, per [[fast-is-not-identical]]. The
corpus tolerances, the shapes, the state bookkeeping and every refusal
are asserted in every tier.

RUN LEDGER. The macOS 0.5.0 installed-wheel surface passed for all three
families and numeric modes. Current remote source-binding results belong in
the dated result directories and SUPPORT_MATRIX.md. This suite does not
expose or certify Python backward; native backward has a separate certificate.

HOW TO RUN IT
-------------
    # 1. build the extension (fast is the default tier)
    bash bindings/build_mamba.sh

    # 2. the gate, FROM THE REPOSITORY ROOT (it reads mamba/corpus/)
    cd python && python3 -m mojolearn.tests.test_mamba_surface

    # 3. the deterministic tier
    MOJOLEARN_NUMERIC_MODE=deterministic bash bindings/build_mamba.sh
    cd python && MOJOLEARN_NUMERIC_MODE=deterministic \\
        python3 -m mojolearn.tests.test_mamba_surface

    # 4. the identical tier, where the bitwise arms are asserted
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_mamba.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_mamba_surface
"""

import os
import sys

import numpy as np

from mojolearn import Mamba1Block, Mamba2Block, Mamba3Block


class Report(object):
    def __init__(self):
        self.rows = []

    def check(self, arm, cond, what, detail=""):
        self.rows.append((bool(cond), arm,
                          what + (("" if cond else " -- " + detail)
                                  if detail else "")))
        return bool(cond)

    def report_only(self, arm, cond, what):
        self.rows.append((True, arm, "[REPORT, not asserted] %s: %s"
                          % (what, "same" if cond else "MOVED")))
        return bool(cond)

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal; each one here is
        made to fire by name and by message (`test_svr_surface.py`'s
        rule). A Mojo `Error` crossing the binding is caught as
        `Exception`."""
        try:
            fn(*a, **kw)
        except exc_type as exc:
            if needle in str(exc):
                return self.check(arm, True, what)
            return self.check(arm, False, what,
                              "raised %s but the message does not contain "
                              "%r: %s" % (exc_type.__name__, needle, exc))
        except Exception as exc:  # noqa: BLE001 - the wrong exception is data
            return self.check(arm, False, what, "raised %s, want %s: %s"
                              % (type(exc).__name__, exc_type.__name__, exc))
        return self.check(arm, False, what, "IS INERT: the call was ACCEPTED")

    def bits_equal(self, arm, got, want, what, assert_it):
        same = (np.ascontiguousarray(got).tobytes()
                == np.ascontiguousarray(want).tobytes())
        if assert_it:
            return self.check(arm, same, what, "bits moved")
        return self.report_only(arm, same, what)

    def close(self, arm, got, want, what, rtol=1e-5, atol=1e-6):
        got = np.asarray(got, dtype=np.float64).ravel()
        want = np.asarray(want, dtype=np.float64).ravel()
        if got.shape != want.shape:
            return self.check(arm, False, what, "shape %s vs %s"
                              % (got.shape, want.shape))
        err = np.abs(got - want) - (atol + rtol * np.abs(want))
        worst = float(err.max()) if err.size else 0.0
        return self.check(
            arm, worst <= 0.0, what,
            "worst excess %.3e over |dump-ref| <= %g + %g*|ref| at flat "
            "index %d (got %.9g, ref %.9g)"
            % (worst, atol, rtol, int(err.argmax()),
               got[int(err.argmax())], want[int(err.argmax())]))

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
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


# ---------------------------------------------------------------------------
# The committed corpus (mamba/corpus/README.md). IN-REPO: walk up from this
# file to the repository root; a missing corpus FAILS, it never skips.
# ---------------------------------------------------------------------------

def corpus_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "mamba", "corpus")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return None


def f32(path, shape):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(shape)


def f64(path, shape):
    a = np.fromfile(path, dtype="<f8")
    return a.reshape(shape)


#: base_b2_l4_d8: B 2, L 4, d_model 8 (d_inner 16, dt_rank 1) -- the
#: "L equals d_conv" case, scan.h_all committed, small enough for a gate.
M1_CASE, M1_B, M1_L, M1_DM = "base_b2_l4_d8", 2, 4, 8
#: m2_base_b2_l4_d32: B 2, L 4, d_model 32 (d_inner 64, H 1, CD 320) --
#: the sub-chunk +SEG case; one open (padded) chunk end to end.
M2_CASE, M2_B, M2_L, M2_DM = "m2_base_b2_l4_d32", 2, 4, 32
#: Mamba-3 shapes, mirroring the check's own cases: the sub-chunk case
#: (m3_base_b2_l4_d32's shape) and the decode-cross construction
#: (contract 8d: prefill 60, decode through token 70; crosses Q = 64).
#: THE CORPUS LANDED 2026-09-02, so the forward arm below now runs the
#: committed case against its ref64 stages, exactly as Mamba-1 and -2 do;
#: the fixture-scale draws stay for the state, decode and refusal arms,
#: which compare the surface against ITSELF and need no reference.
M3_CASE = "m3_base_b2_l4_d32"
M3_B, M3_L, M3_DM = 2, 4, 32
M3_CROSS_L, M3_CROSS_SPLIT = 70, 60
M3_Q = 64  # the Mamba-3 chunk size (PART OF THE ARITHMETIC, DEV 827/783)
M3_TWO_PI_F32 = np.float32(6.2831855)  # bits 0x40C90FDB (contract s3)


def m1_weights(case_dir, dm):
    di, r = 2 * dm, (dm + 15) // 16
    return {
        "norm.weight": f32(os.path.join(case_dir, "norm.weight.f32"), (dm,)),
        "in_proj.weight": f32(
            os.path.join(case_dir, "in_proj.weight.f32"), (2 * di, dm)),
        "conv1d.weight": f32(
            os.path.join(case_dir, "conv1d.weight.f32"), (di, 1, 4)),
        "conv1d.bias": f32(os.path.join(case_dir, "conv1d.bias.f32"), (di,)),
        "x_proj.weight": f32(
            os.path.join(case_dir, "x_proj.weight.f32"), (r + 32, di)),
        "dt_proj.weight": f32(
            os.path.join(case_dir, "dt_proj.weight.f32"), (di, r)),
        "dt_proj.bias": f32(
            os.path.join(case_dir, "dt_proj.bias.f32"), (di,)),
        "A_log": f32(os.path.join(case_dir, "A_log.f32"), (di, 16)),
        "D": f32(os.path.join(case_dir, "D.f32"), (di,)),
        "out_proj.weight": f32(
            os.path.join(case_dir, "out_proj.weight.f32"), (dm, di)),
    }


def m2_weights(case_dir, dm):
    di = 2 * dm
    h = di // 64
    cd = di + 256
    dip = 2 * di + 256 + h
    return {
        "block_norm.weight": f32(
            os.path.join(case_dir, "block_norm.weight.f32"), (dm,)),
        "in_proj.weight": f32(
            os.path.join(case_dir, "in_proj.weight.f32"), (dip, dm)),
        "conv1d.weight": f32(
            os.path.join(case_dir, "conv1d.weight.f32"), (cd, 1, 4)),
        "conv1d.bias": f32(os.path.join(case_dir, "conv1d.bias.f32"), (cd,)),
        "dt_bias": f32(os.path.join(case_dir, "dt_bias.f32"), (h,)),
        "A_log": f32(os.path.join(case_dir, "A_log.f32"), (h,)),
        "D": f32(os.path.join(case_dir, "D.f32"), (h,)),
        "norm.weight": f32(os.path.join(case_dir, "norm.weight.f32"), (di,)),
        "out_proj.weight": f32(
            os.path.join(case_dir, "out_proj.weight.f32"), (dm, di)),
    }


def m3_uniform(rng, shape, lo, hi):
    """A float32 draw in [lo, hi) -- the smoke-test spelling. These are
    INPUTS, not expected values: every Mamba-3 assertion below compares
    the surface against itself through the binding (header)."""
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)


def m3_weights(rng, dm):
    """Fixture-scale weights (the ranges `mamba3_fixture.mojo`'s scale
    note documents: dt_bias in [-7, -2] so dt is small and the chunked
    recurrence actually recurses; fan-in-scaled projections; near-ones
    B/C biases, their upstream ones-init neighborhood)."""
    di = 2 * dm
    h = di // 64
    dip = 2 * di + 256 + 3 * h + 32
    s_in = float(dm) ** -0.5
    s_out = float(di) ** -0.5
    return {
        "block_norm.weight": m3_uniform(rng, (dm,), 0.5, 1.5),
        "in_proj.weight": m3_uniform(rng, (dip, dm), -s_in, s_in),
        "dt_bias": m3_uniform(rng, (h,), -7.0, -2.0),
        "B_norm.weight": m3_uniform(rng, (128,), 0.5, 1.5),
        "C_norm.weight": m3_uniform(rng, (128,), 0.5, 1.5),
        "B_bias": m3_uniform(rng, (h, 128), 0.9, 1.1),
        "C_bias": m3_uniform(rng, (h, 128), 0.9, 1.1),
        "D": m3_uniform(rng, (h,), 0.5, 1.5),
        "out_proj.weight": m3_uniform(rng, (dm, di), -s_out, s_out),
    }


def m3_weights_corpus(case_dir, dm):
    """The committed case's weights, the Mamba-3 mirror of `m2_weights`.

    The nine tensors and their shapes are the ones `m3_weights` draws,
    read from the corpus instead of a generator: `dip` is
    `2*d_inner + 256 + 3*nheads + num_rope_angles`, which is 419 at
    d_model 32, and the file sizes agree with that (in_proj.weight is
    13,408 floats = 419 x 32). A wrong shape here is caught by
    `f32`'s own count check rather than by silently reinterpreting the
    bytes, which is the mamba1 reindexing lesson."""
    di = 2 * dm
    h = di // 64
    dip = 2 * di + 256 + 3 * h + 32
    return {
        "block_norm.weight": f32(
            os.path.join(case_dir, "block_norm.weight.f32"), (dm,)),
        "in_proj.weight": f32(
            os.path.join(case_dir, "in_proj.weight.f32"), (dip, dm)),
        "dt_bias": f32(os.path.join(case_dir, "dt_bias.f32"), (h,)),
        "B_norm.weight": f32(
            os.path.join(case_dir, "B_norm.weight.f32"), (128,)),
        "C_norm.weight": f32(
            os.path.join(case_dir, "C_norm.weight.f32"), (128,)),
        "B_bias": f32(os.path.join(case_dir, "B_bias.f32"), (h, 128)),
        "C_bias": f32(os.path.join(case_dir, "C_bias.f32"), (h, 128)),
        "D": f32(os.path.join(case_dir, "D.f32"), (h,)),
        "out_proj.weight": f32(
            os.path.join(case_dir, "out_proj.weight.f32"), (dm, di)),
    }


def m3_state_bits(rep, arm, got_state, want_state, n_rows, what, assert_it):
    """The Mamba-3 carried state, compared piece by piece. The six
    buffers compare on their VALID rows only (rows [0, buf_len); rows
    beyond buf_len are not state -- the buffer update rewrites [0, r)
    and leaves stale rows above it, DEVIATION 832(i)), theta and h
    whole."""
    rep.bits_equal(arm, got_state.theta, want_state.theta,
                   what + ": theta", assert_it)
    rep.bits_equal(arm, got_state.h, want_state.h,
                   what + ": the sealed boundary h", assert_it)
    for name in ("buffer_qrot", "buffer_krot", "buffer_v", "buffer_dt",
                 "buffer_sig", "buffer_adt"):
        rep.bits_equal(arm,
                       getattr(got_state, name)[:, :n_rows],
                       getattr(want_state, name)[:, :n_rows],
                       what + ": %s rows [0, %d)" % (name, n_rows),
                       assert_it)


def main(out=sys.stdout):
    rep = Report()
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    assert_bits = mode == "identical"

    # -- REFUSALS: every guard on THIS side, made to fire by name, before
    #    any device call (none of these needs the extension built) --------
    arm = "REFUSALS (Python side, no device)"
    dm, di = M1_DM, 2 * M1_DM
    zeros1 = {
        "norm.weight": np.zeros((dm,), np.float32),
        "in_proj.weight": np.zeros((2 * di, dm), np.float32),
        "conv1d.weight": np.zeros((di, 1, 4), np.float32),
        "conv1d.bias": np.zeros((di,), np.float32),
        "x_proj.weight": np.zeros((33, di), np.float32),
        "dt_proj.weight": np.zeros((di, 1), np.float32),
        "dt_proj.bias": np.zeros((di,), np.float32),
        "A_log": np.zeros((di, 16), np.float32),
        "D": np.zeros((di,), np.float32),
        "out_proj.weight": np.zeros((dm, di), np.float32),
    }
    blk = Mamba1Block(zeros1)
    x64 = np.zeros((1, 2, dm), np.float64)
    rep.raises(arm, TypeError, "float64",
               "a float64 x is refused BY NAME (no silent downcast)",
               blk.forward, x64)
    rep.raises(arm, TypeError, "bfloat16",
               "the refusal message names bfloat16 (the SHIP LATER "
               "profile, FEATURE_PARITY.md section 7)",
               blk.forward, x64)
    rep.raises(arm, ValueError, "(B, L, d_model)",
               "a 2-D x is refused for forward (only step squeezes)",
               blk.forward, np.zeros((2, dm), np.float32))
    rep.raises(arm, ValueError, "d_model",
               "an x whose d_model is not the weights' is refused",
               blk.forward, np.zeros((1, 2, dm + 1), np.float32))
    rep.raises(arm, ValueError, "exactly one token",
               "a multi-token step is refused (mamba_simple.py:210)",
               blk.step, np.zeros((1, 2, dm), np.float32),
               blk.allocate_state(1))
    rep.raises(arm, ValueError, "state is required",
               "step without a state is refused",
               blk.step, np.zeros((1, 1, dm), np.float32), None)
    bad_w = dict(zeros1)
    bad_w["A_log"] = zeros1["A_log"].astype(np.float64)
    rep.raises(arm, TypeError, "A_log",
               "a float64 weight is refused by its own name",
               Mamba1Block, bad_w)
    bad_w = dict(zeros1)
    bad_w["in_proj.weight"] = zeros1["in_proj.weight"].T.copy()
    rep.raises(arm, ValueError, "in_proj.weight",
               "a transposed projection cannot cross as a plausible "
               "buffer (exact-shape refusal)",
               Mamba1Block, bad_w)
    del bad_w["D"]
    rep.raises(arm, ValueError, "missing",
               "a missing weight name is refused (no silent zero weight)",
               Mamba1Block, bad_w)
    st = blk.allocate_state(1)
    st.h = st.h[:, :, :8]  # wrong shape
    rep.raises(arm, ValueError, "state buffer h",
               "a wrong-shape state buffer is refused by name",
               blk.step, np.zeros((1, 1, dm), np.float32), st)
    zeros2 = {n: np.zeros(1, np.float32) for n in Mamba2Block._W_NAMES}
    zeros2["block_norm.weight"] = np.zeros((40,), np.float32)
    rep.raises(arm, ValueError, "multiple of",
               "Mamba2 d_model = 40 is refused (headdim 64 / expand 2; "
               "Mamba2Dims.of carries the same rule)",
               Mamba2Block, zeros2)
    bad3 = {n: np.zeros(1, np.float32) for n in Mamba3Block._W_NAMES}
    bad3["block_norm.weight"] = np.zeros((40,), np.float32)
    rep.raises(arm, ValueError, "multiple of",
               "Mamba3 d_model = 40 is refused (headdim 64 / expand 2; "
               "Mamba3Dims.of carries the same rule)",
               Mamba3Block, bad3)
    # The smallest legal mamba3 shape: d_model must be a multiple of 32.
    dm3, di3, h3 = 32, 64, 1
    dip3 = 2 * di3 + 256 + 3 * h3 + 32
    zeros3 = {
        "block_norm.weight": np.zeros((dm3,), np.float32),
        "in_proj.weight": np.zeros((dip3, dm3), np.float32),
        "dt_bias": np.zeros((h3,), np.float32),
        "B_norm.weight": np.zeros((128,), np.float32),
        "C_norm.weight": np.zeros((128,), np.float32),
        "B_bias": np.zeros((h3, 128), np.float32),
        "C_bias": np.zeros((h3, 128), np.float32),
        "D": np.zeros((h3,), np.float32),
        "out_proj.weight": np.zeros((dm3, di3), np.float32),
    }
    blk3z = Mamba3Block(zeros3)
    rep.raises(arm, TypeError, "float64",
               "Mamba3: a float64 x is refused BY NAME",
               blk3z.forward, np.zeros((1, 2, dm3), np.float64))
    rep.raises(arm, ValueError, "state is required",
               "Mamba3: step without a state is refused",
               blk3z.step, np.zeros((1, 1, dm3), np.float32), None)
    rep.raises(arm, ValueError, "exactly one token",
               "Mamba3: a multi-token step is refused",
               blk3z.step, np.zeros((1, 2, dm3), np.float32),
               blk3z.allocate_state(1))
    st3bad = blk3z.allocate_state(1)
    st3bad.theta = st3bad.theta[:, :, :8]  # wrong shape
    rep.raises(arm, ValueError, "state buffer theta",
               "Mamba3: a wrong-shape state buffer is refused by name",
               blk3z.step, np.zeros((1, 1, dm3), np.float32), st3bad)
    st3ok = blk3z.allocate_state(1)
    rep.raises(arm, TypeError, "theta",
               "Mamba3State.set_input_states: a float64 piece is "
               "refused by its own name (bits must arrive as made)",
               st3ok.set_input_states,
               np.zeros((1, h3, 32), np.float64), st3ok.h,
               st3ok.pending_k, st3ok.pending_v)
    bad3w = dict(zeros3)
    bad3w["in_proj.weight"] = zeros3["in_proj.weight"].T.copy()
    rep.raises(arm, ValueError, "in_proj.weight",
               "Mamba3: a transposed projection cannot cross as a "
               "plausible buffer (exact-shape refusal)",
               Mamba3Block, bad3w)

    # -- the corpus ------------------------------------------------------
    root = corpus_root()
    arm = "CORPUS"
    if not rep.check(arm, root is not None,
                     "mamba/corpus found walking up from this file",
                     "this gate runs IN-REPO; a wheel does not carry the "
                     "corpus, and a skip here would be a pass with "
                     "nothing behind it"):
        rep.render(out)
        out.write("\ntest_mamba_surface: RED (no corpus).\n")
        return 1

    # -- MAMBA-1 FORWARD vs the committed reference ----------------------
    arm = "MAMBA-1 forward (corpus %s, ref64 tolerance)" % M1_CASE
    c1 = os.path.join(root, M1_CASE)
    w1 = m1_weights(c1, M1_DM)
    x1 = f32(os.path.join(c1, "x.f32"), (M1_B, M1_L, M1_DM))
    blk1 = Mamba1Block(w1)
    st1 = blk1.allocate_state(M1_B)
    y1 = blk1.forward(x1, st1)
    rep.check(arm, y1.shape == (M1_B, M1_L, M1_DM), "y has x's shape")
    rep.close(arm, y1,
              f64(os.path.join(c1, "ref64", "block.out.f64"),
                  (M1_B, M1_L, M1_DM)),
              "block output matches ref64 block.out at the corpus "
              "tolerance (rtol 1e-5, atol 1e-6)")
    rep.close(arm, st1.h,
              f64(os.path.join(c1, "ref64", "scan.h_last.f64"),
                  (M1_B, 2 * M1_DM, 16)),
              "the carried h matches ref64 scan.h_last -- the state "
              "buffer really holds the post-call state")

    # -- MAMBA-1 DECODE == PREFILL, through the Python surface -----------
    arm = "MAMBA-1 decode (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; fast tier")
    st1d = blk1.allocate_state(M1_B)
    for t in range(M1_L):
        yt = blk1.step(x1[:, t:t + 1, :], st1d)
        rep.bits_equal(arm, yt[:, 0, :], y1[:, t, :],
                       "step token %d == prefill token %d" % (t, t),
                       assert_bits)
    rep.bits_equal(arm, st1d.h, st1.h,
                   "h after 4 steps == h after the L=4 prefill",
                   assert_bits)
    rep.bits_equal(arm, st1d.conv_window, st1.conv_window,
                   "conv window after 4 steps == prefill's",
                   assert_bits)

    # -- MAMBA-2 FORWARD vs the committed reference ----------------------
    arm = "MAMBA-2 forward (corpus %s, ref64 tolerance)" % M2_CASE
    c2 = os.path.join(root, "mamba2", M2_CASE)
    w2 = m2_weights(c2, M2_DM)
    x2 = f32(os.path.join(c2, "x.f32"), (M2_B, M2_L, M2_DM))
    blk2 = Mamba2Block(w2)
    st2 = blk2.allocate_state(M2_B)
    y2 = blk2.forward(x2, st2)
    h2 = (2 * M2_DM) // 64
    rep.check(arm, y2.shape == (M2_B, M2_L, M2_DM), "y has x's shape")
    rep.check(arm, st2.buffered_tokens == M2_L,
              "buf_len after an L=4 prefill is 4 (one open chunk)",
              "got %r" % st2.buffered_tokens)
    rep.close(arm, y2,
              f64(os.path.join(c2, "ref64", "residual.out.f64"),
                  (M2_B * M2_L, M2_DM)).reshape(M2_B, M2_L, M2_DM),
              "block output matches ref64 residual.out at the corpus "
              "tolerance")
    rep.close(arm, blk2.h_last_,
              f64(os.path.join(c2, "ref64", "ssd.h_last.f64"),
                  (M2_B, h2, 64, 128)),
              "h_last_ (the REPORT state, contract section 5) matches "
              "ref64 ssd.h_last")
    rep.check(arm, not np.any(st2.h),
              "the BOUNDARY h stays zero with no completed chunk -- "
              "h_last_ and the resumption state are different tensors")

    # -- MAMBA-2 RESUMPTION: split prefill == whole prefill ---------------
    arm = "MAMBA-2 resumption (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; fast tier")
    st2s = blk2.allocate_state(M2_B)
    y_head = blk2.forward(x2[:, :M2_L - 1, :], st2s)
    rep.check(arm, st2s.buffered_tokens == M2_L - 1,
              "buf_len after L-1 tokens is L-1",
              "got %r" % st2s.buffered_tokens)
    rep.bits_equal(arm, y_head, y2[:, :M2_L - 1, :],
                   "the L-1 prefill's rows == the whole prefill's first "
                   "L-1 rows (prefix property)", assert_bits)
    y_tail = blk2.step(x2[:, M2_L - 1:M2_L, :], st2s)
    rep.check(arm, st2s.buffered_tokens == M2_L,
              "buf_len after the step is L", "got %r" % st2s.buffered_tokens)
    rep.bits_equal(arm, y_tail[:, 0, :], y2[:, M2_L - 1, :],
                   "one decode step after L-1 == prefill token L-1 "
                   "(DEVIATION 786: decode is prefill resumption)",
                   assert_bits)
    rep.bits_equal(arm, st2s.buffer_xbc, st2.buffer_xbc,
                   "the open-chunk xBC buffer round-trips (split == whole)",
                   assert_bits)
    rep.bits_equal(arm, st2s.buffer_dtraw, st2.buffer_dtraw,
                   "the open-chunk dt buffer round-trips (split == whole)",
                   assert_bits)

    # -- MAMBA-3 FORWARD (no committed corpus yet; header) ---------------
    arm = "MAMBA-3 forward (B%d L%d d%d, fixture-scale inputs)" % (
        M3_B, M3_L, M3_DM)
    rng3 = np.random.default_rng(0x4D6D6233)  # ASCII 'Mmb3'
    w3 = m3_weights(rng3, M3_DM)
    x3 = m3_uniform(rng3, (M3_B, M3_L, M3_DM), -2.0, 2.0)
    blk3 = Mamba3Block(w3)
    st3 = blk3.allocate_state(M3_B)
    y3 = blk3.forward(x3, st3)
    h3n = (2 * M3_DM) // 64
    rep.check(arm, y3.shape == (M3_B, M3_L, M3_DM), "y has x's shape")
    rep.check(arm, bool(np.isfinite(y3).all()), "y is finite")
    rep.check(arm, st3.buffered_tokens == M3_L,
              "buf_len after an L=%d prefill is %d (one working chunk, "
              "Q = 64)" % (M3_L, M3_L), "got %r" % st3.buffered_tokens)
    rep.check(arm, st3.pending is False,
              "pending stays False on a plain prefill")
    rep.check(arm, not np.any(st3.h),
              "the SEALED boundary h stays zero with one working chunk "
              "(DEVIATION 832: h_last_ and the resumption h are "
              "different tensors)")
    rep.check(arm, bool(np.any(st3.theta)),
              "theta advanced (the serial angle recurrence ran)")
    rep.check(arm,
              bool((st3.theta >= 0.0).all()
                   and (st3.theta < M3_TWO_PI_F32).all()),
              "theta stays in [0, 2pi) (the S10 mod's invariant, "
              "surviving the round trip)")
    rep.check(arm, blk3.h_last_.shape == (M3_B, h3n, 64, 128),
              "h_last_ report shape")
    rep.check(arm, blk3.k_last_.shape == (M3_B, h3n, 128),
              "k_last_ report shape")
    rep.check(arm, blk3.v_last_.shape == (M3_B, h3n, 64),
              "v_last_ report shape")
    rep.check(arm, blk3.theta_last_.shape == (M3_B, h3n, 32),
              "theta_last_ report shape")
    # -- MAMBA-3 FORWARD vs the committed reference ----------------------
    #
    # THE DEBT THIS REPLACES. Until 2026-09-02 the block above ended in a
    # tripwire: while `mamba/corpus/mamba3` did not exist the row passed as
    # [OWED], and the moment the directory appeared without this comparison
    # being wired the row FAILED, so the debt could not rot. The corpus
    # landed, the tripwire fired on all three vendors at once, and this is
    # the arm it was holding the place for. It is the Mamba-2 arm above with
    # Mamba-3's tensors, at the same corpus tolerance.
    arm = "MAMBA-3 forward (corpus %s, ref64 tolerance)" % M3_CASE
    c3 = os.path.join(root, "mamba3", M3_CASE)
    if not os.path.isdir(c3):
        rep.check(arm, False,
                  "the committed Mamba-3 corpus case is present",
                  "expected %s; a missing corpus FAILS here, it never "
                  "skips (header)" % c3)
    else:
        w3c = m3_weights_corpus(c3, M3_DM)
        x3c = f32(os.path.join(c3, "x.f32"), (M3_B, M3_L, M3_DM))
        blk3c = Mamba3Block(w3c)
        st3c = blk3c.allocate_state(M3_B)
        y3c = blk3c.forward(x3c, st3c)
        rep.check(arm, y3c.shape == (M3_B, M3_L, M3_DM),
                  "y has x's shape")
        rep.close(arm, y3c,
                  f64(os.path.join(c3, "ref64", "residual.out.f64"),
                      (M3_B * M3_L, M3_DM)).reshape(M3_B, M3_L, M3_DM),
                  "block output matches ref64 residual.out at the corpus "
                  "tolerance")
        rep.close(arm, blk3c.h_last_,
                  f64(os.path.join(c3, "ref64", "ssd.h_last.f64"),
                      (M3_B, h3n, 64, 128)),
                  "h_last_ (the REPORT state) matches ref64 ssd.h_last")
        rep.close(arm, blk3c.k_last_,
                  f64(os.path.join(c3, "ref64", "ssd.k_last.f64"),
                      (M3_B, h3n, 128)),
                  "k_last_ matches ref64 ssd.k_last")

    # -- MAMBA-3 DECODE == PREFILL, through the Python surface -----------
    arm = "MAMBA-3 decode (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; fast tier")
    st3d = blk3.allocate_state(M3_B)
    for t in range(M3_L):
        yt = blk3.step(x3[:, t:t + 1, :], st3d)
        rep.bits_equal(arm, yt[:, 0, :], y3[:, t, :],
                       "step token %d == prefill token %d" % (t, t),
                       assert_bits)
    m3_state_bits(rep, arm, st3d, st3, M3_L,
                  "after %d steps == after the L=%d prefill"
                  % (M3_L, M3_L), assert_bits)

    # -- MAMBA-3 RESUMPTION ACROSS THE Q = 64 SEAL (the decode-cross
    #    construction, contract 8d: prefill 60, decode through 70) ------
    arm = "MAMBA-3 chunk-crossing resumption (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; fast tier")
    xc = m3_uniform(rng3, (1, M3_CROSS_L, M3_DM), -2.0, 2.0)
    st3w = blk3.allocate_state(1)
    y3w = blk3.forward(xc, st3w)
    rep.check(arm, st3w.buffered_tokens == M3_CROSS_L - M3_Q,
              "buf_len after L=%d is %d (the last working chunk's rows; "
              "the buffer never empties)" % (M3_CROSS_L,
                                             M3_CROSS_L - M3_Q),
              "got %r" % st3w.buffered_tokens)
    rep.check(arm, bool(np.any(st3w.h)),
              "the sealed boundary h is nonzero once a chunk sealed")
    st3s = blk3.allocate_state(1)
    y3h = blk3.forward(xc[:, :M3_CROSS_SPLIT, :], st3s)
    rep.check(arm, st3s.buffered_tokens == M3_CROSS_SPLIT,
              "buf_len after L=%d tokens is %d (%d < Q, still one "
              "working chunk)" % (
                  M3_CROSS_SPLIT, M3_CROSS_SPLIT, M3_CROSS_SPLIT),
              "got %r" % st3s.buffered_tokens)
    rep.bits_equal(arm, y3h, y3w[:, :M3_CROSS_SPLIT, :],
                   "the L=%d prefill's rows == the whole prefill's "
                   "first %d rows (prefix property)"
                   % (M3_CROSS_SPLIT, M3_CROSS_SPLIT), assert_bits)
    for t in range(M3_CROSS_SPLIT, M3_CROSS_L):
        yt = blk3.step(xc[:, t:t + 1, :], st3s)
        rep.bits_equal(arm, yt[:, 0, :], y3w[:, t, :],
                       "step token %d == prefill token %d (decode is "
                       "prefill resumption across the seal, DEVIATION "
                       "831)" % (t, t), assert_bits)
    rep.check(arm, st3s.buffered_tokens == st3w.buffered_tokens,
              "final buf_len: split == whole",
              "got %r vs %r" % (st3s.buffered_tokens,
                                st3w.buffered_tokens))
    m3_state_bits(rep, arm, st3s, st3w, M3_CROSS_L - M3_Q,
                  "split == whole at token %d" % M3_CROSS_L, assert_bits)

    # -- MAMBA-3 Input_States CONTINUATION (contract section 5 claim 2):
    #    the SAME continuation through the binding twice. There is no
    #    bitwise claim against an unbroken prefill BY THEOREM (DEVIATION
    #    831; the check's continuation gate says so in as many words) --
    #    the surface claims here are marshaling ones: the pieces REACH
    #    the core, the flag is consumed, and the call repeats byte for
    #    byte. ------------------------------------------------------------
    arm = "MAMBA-3 Input_States continuation (repeat bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; fast tier")
    theta0 = m3_uniform(rng3, (M3_B, h3n, 32), 0.0, 6.25)
    h0 = m3_uniform(rng3, (M3_B, h3n, 64, 128), -0.5, 0.5)
    k0 = m3_uniform(rng3, (M3_B, h3n, 128), -0.5, 0.5)
    v0 = m3_uniform(rng3, (M3_B, h3n, 64), -0.5, 0.5)
    st3a = blk3.allocate_state(M3_B)
    st3a.set_input_states(theta0, h0, k0, v0)
    rep.check(arm, st3a.pending is True,
              "set_input_states marks the pieces pending")
    y3a = blk3.forward(x3, st3a)
    rep.check(arm, st3a.pending is False,
              "pending is consumed by the call (DEVIATION 794)")
    rep.check(arm, y3a.tobytes() != y3.tobytes(),
              "the continuation REACHED the core: same x, different "
              "output than the fresh prefill "
              "[[mojotrees-verify-reach-not-output]]")
    st3b = blk3.allocate_state(M3_B)
    st3b.set_input_states(theta0, h0, k0, v0)
    y3b = blk3.forward(x3, st3b)
    rep.bits_equal(arm, y3a, y3b,
                   "the same continuation through the binding twice is "
                   "byte-identical", assert_bits)
    m3_state_bits(rep, arm, st3a, st3b, M3_L,
                  "continuation state, call 1 == call 2", assert_bits)
    # The lane's own fresh-state refusal, reached FROM PYTHON: a pending
    # continuation on a mid-sequence state goes down unjudged by the
    # wrapper (DEVIATION 794) and is refused in Mojo by name.
    st3m = blk3.allocate_state(M3_B)
    blk3.forward(x3, st3m)
    st3m.set_input_states(theta0, h0, k0, v0)
    rep.raises(arm, Exception, "fresh",
               "a pending continuation on a mid-sequence state is "
               "refused IN MOJO (set_input_states: Input_States only "
               "has upstream meaning on a fresh state)",
               blk3.forward, x3, st3m)

    rep.render(out)
    out.write("\n")
    if rep.failures:
        out.write("test_mamba_surface: RED. %d checks failed.\n"
                  % len(rep.failures))
        return 1
    if not assert_bits:
        out.write(
            "test_mamba_surface: the %s arms passed. The corpus forwards\n"
            "(Mamba-1/2/3), the\n"
            "shapes, the state bookkeeping and every refusal hold.\n"
            "THE DECODE/PREFILL BITWISE ARMS WERE REPORTED, NOT ASSERTED.\n"
            "Re-run under\n"
            "MOJOLEARN_NUMERIC_MODE=identical for the asserted form.\n"
            % mode)
        return 0
    out.write(
        "test_mamba_surface: GREEN. Mamba-1/2/3 reproduce the\n"
        "committed corpus at its calibrated tolerance through the Python\n"
        "surface; decode equals prefill bit\n"
        "for bit per token through this path for all three blocks --\n"
        "Mamba-3 across the Q=64 chunk seal included -- a split prefill\n"
        "resumes bit for bit through the explicit state, a repeated\n"
        "Input_States continuation is byte-identical and consumed, and\n"
        "every guard on the path fires by name. One box, one vendor:\n"
        "the cross-vendor statement belongs to the lanes' cards, not to\n"
        "this file.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
