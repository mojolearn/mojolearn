# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.mamba` (Mamba1Block,
Mamba2Block).

Written 2026-09-01, the day the surface landed -- the day
`mamba/FEATURE_PARITY.md`'s "PyPI surface: NONE EXISTS" row closed. The
model for this file is `test_gp_surface.py` and the house standard is
`gemm/PYTHON_SURFACE_GATE.md`.

WHAT THIS CLOSES. `bindings/_mojolearn_mamba.mojo`'s four entry points
and `python/mojolearn/_mamba_impl.py` put the certified Mamba blocks in
reach of a Python caller. NOTHING IN THAT PATH IS COVERED BY THE LANES'
GATES: those run inside Mojo (`pixi run check-mamba-block`,
`check-mamba2`, and siblings), against their own oracles, and stop at
`mamba_block_forward` / `mamba2_block_forward`. Everything here is
downstream of that point: an addrs list whose order is written out twice
and could be written out wrong once, state buffers marshaled up and back
per call, a buf_len that must survive the round trip, and dtype/shape
refusals that exist only on this side.

THE EXPECTED VALUES ARE THE COMMITTED CORPUS'S, NOT OUR OWN TALLY. The
forward arms run corpus cases (`mamba/corpus/base_b2_l4_d8`,
`mamba/corpus/mamba2/m2_base_b2_l4_d32`) and compare against their
`ref64` stages at the corpus README's calibrated tolerance (rtol 1e-5,
atol 1e-6) -- a TOLERANCE anchor, per that README: the bitwise oracle is
the lane's, and this file makes no cross-vendor claim. The corpus lives
in the repository, not the wheel, so this gate runs IN-REPO; a missing
corpus is a FAILURE naming the path, never a skip
([[not-applicable-is-not-a-pass]]).

WHAT IS ASSERTED AND WHAT IS REPORTED. Under `identical` the bitwise
arms -- decode == prefill per token, and split prefill (L-1 tokens then
one step) == whole prefill -- are ASSERTED; under `fast` they are
REPORTED, per [[fast-is-not-identical]]. The corpus tolerances, the
shapes, the state bookkeeping and every refusal are asserted in every
tier.

EVERY RUNNABLE CLAIM HERE IS UNVERIFIED, RUN OWED: neither this file nor
the binding it exercises has ever run.

HOW TO RUN IT
-------------
    # 1. build the extension (fast is the default tier)
    bash bindings/build_mamba.sh

    # 2. the gate, FROM THE REPOSITORY ROOT (it reads mamba/corpus/)
    cd python && python3 -m mojolearn.tests.test_mamba_surface

    # 3. the identical tier, where the bitwise arms are asserted
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_mamba.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_mamba_surface
"""

import os
import sys

import numpy as np

from mojolearn import Mamba1Block, Mamba2Block


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

    rep.render(out)
    out.write("\n")
    if rep.failures:
        out.write("test_mamba_surface: RED. %d checks failed.\n"
                  % len(rep.failures))
        return 1
    if not assert_bits:
        out.write(
            "test_mamba_surface: the %s arms passed. The corpus forwards,\n"
            "the shapes, the state bookkeeping and every refusal hold.\n"
            "THE BITWISE ARMS WERE REPORTED, NOT ASSERTED: this tier\n"
            "promises speed only. Re-run under\n"
            "MOJOLEARN_NUMERIC_MODE=identical for the asserted form.\n"
            % mode)
        return 0
    out.write(
        "test_mamba_surface: GREEN. Both blocks reproduce the committed\n"
        "corpus at its calibrated tolerance through the Python surface,\n"
        "decode equals prefill bit for bit per token through this path,\n"
        "a split Mamba-2 prefill resumes bit for bit through the\n"
        "three-piece state, and every guard on the path fires by name.\n"
        "One box, one vendor: the cross-vendor statement belongs to the\n"
        "lanes' cards, not to this file.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
