# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.transformer`
(TransformerBlock).

Written 2026-09-02, the day the surface landed -- the day profile
`mojolearn.identical.transformer.fp32.v1` got its first Python symbol.
The model for this file is `test_mamba_surface.py` and the house
standard is `gemm/PYTHON_SURFACE_GATE.md`.

WHAT THIS CLOSES. `bindings/_mojolearn_transformer.mojo`'s two entry
points and `python/mojolearn/_transformer_impl.py` put the certified
transformer block in reach of a Python caller. NOTHING IN THAT PATH IS
COVERED BY THE LANE'S GATES: those run inside Mojo (`pixi run
check-transformer`), against their own oracle, and stop at
`llama_decoder_layer_forward`. Everything here is downstream of that
point: an addrs list whose order is written out twice and could be
written out wrong once, a KV cache marshaled up and back per call at a
PACKED-at-used-stride layout (DEVIATION 795(ii)) that is easy to
mis-slice on exactly one side, a `cached_tokens` that must survive the
round trip, and dtype/shape refusals that exist only on this side.

THE REFERENCE ARM IS A NUMPY FLOAT64 TRANSCRIPTION OF THE CONTRACT, NOT
OF ANY EXTERNAL CODE. `transformer/corpus/` has a generator and NO case
data on disk (its own README; the lane's phase 7 row), so unlike the
mamba surface gate there is no committed reference to compare against.
The `_ref_block` below is the contract's block order (sections 2-5 of
`IDENTICAL_TRANSFORMER_CONTRACT.md`: eps 1e-6, theta 10000.0, the
additive -FLT_MAX causal mask, eager softmax, GQA as an index map),
derived from that document alone, in float64 -- a TOLERANCE anchor only;
the bitwise oracle is the lane's, and this file makes no cross-vendor
claim. The tolerance pair (rtol 1e-5, atol 1e-6) is the mamba corpus's
calibrated anchor ADOPTED AS A STARTING POINT, not calibrated here: if
the first run fails this arm on magnitude alone (an ACCURACY excess,
never a shape, state or bitwise failure), recalibrate against a torch
fp32 self-test the way `mamba/corpus/README.md` did, and record the
finding -- do not widen silently. THE DEBT ARM at the bottom keys on
`transformer/corpus/` gaining case data: the moment the corpus is
actually generated, this file must compare against it and FAILS until
that is wired, so the debt cannot rot into a silent skip
([[not-applicable-is-not-a-pass]]).

WHAT IS ASSERTED AND WHAT IS REPORTED. Under `identical` the bitwise
arms -- decode == prefill per token, split prefill (L-1 tokens then one
step) == whole prefill, the cache round-trip, and the repeat-call
determinism arm -- are ASSERTED; under `deterministic` ONLY the
repeat-call arm is asserted (same box, same bits run to run IS that
tier's promise; the cross-path decode==prefill construction is asserted
only where the lane's clause (d) is, under identical); under `fast`
every bitwise arm is REPORTED, per [[fast-is-not-identical]] (the
lane's own clause (d) FAILS under FAST by construction and that is
recorded as correct in transformer/README.md). The reference
tolerances, the shapes, the state bookkeeping and every refusal are
asserted in every tier.

RUN LEDGER. THIS FILE HAS NEVER RUN: the binding has never been
compiled. UNVERIFIED, RUN OWED, per tier, with the binding REBUILT
first.

HOW TO RUN IT
-------------
    # 1. build the extension (fast is the default tier)
    bash bindings/build_transformer.sh

    # 2. the gate, FROM THE REPOSITORY ROOT (the debt arm reads
    #    transformer/corpus/)
    cd python && python3 -m mojolearn.tests.test_transformer_surface

    # 3. the deterministic tier
    MOJOLEARN_NUMERIC_MODE=deterministic bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=deterministic \\
        python3 -m mojolearn.tests.test_transformer_surface

    # 4. the identical tier, where the bitwise arms are asserted
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_transformer_surface
"""

import os
import sys

import numpy as np

from mojolearn import TransformerBlock


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
# The float64 reference of the contract's block math. Derived from
# IDENTICAL_TRANSFORMER_CONTRACT.md sections 2-5 ALONE (the header says
# why); every constant is that document's section 3.
# ---------------------------------------------------------------------------

#: torch.finfo(torch.float32).min, the additive mask value (contract S13;
#: ADDED, not selected, and finite, not -inf -- contract 4.1).
_MASK_FILL = -3.4028234663852886e+38
_RMS_EPS = 1e-6
_ROPE_THETA = 10000.0


def _ref_rms(x, w, eps=_RMS_EPS):
    """S1-S4: variance = mean(x^2, -1); x * rsqrt(variance + eps) * w."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * w


def _ref_rope_tables(head_dim, p_max, theta=_ROPE_THETA):
    """S6-S8: inv_freq[i] = 1/theta^(2i/hd); angle = position * inv_freq
    (ABSOLUTE position, contract 5.5/7.2); the table is cat(freqs,
    freqs), so only the half table is built and indexed j % half."""
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (2.0 * i / head_dim))
    pos = np.arange(p_max, dtype=np.float64)[:, None]
    ang = pos * inv_freq[None, :]
    return np.cos(ang), np.sin(ang)  # each [p_max, half]


def _ref_rope_apply(v, cos_t, sin_t, positions):
    """S9-S10 on `v` [L, n, head_dim] at absolute `positions` [L]:
    (v * cos) + (rotate_half(v) * sin), rotate_half = cat(-x2, x1)."""
    l, n, hd = v.shape
    half = hd // 2
    c = cos_t[positions][:, None, :]  # [L, 1, half]
    s = sin_t[positions][:, None, :]
    cos_full = np.concatenate([c, c], axis=-1)
    sin_full = np.concatenate([s, s], axis=-1)
    rh = np.concatenate([-v[..., half:], v[..., :half]], axis=-1)
    return v * cos_full + rh * sin_full


def _ref_block(w, x, n_heads, n_kv, head_dim):
    """One fresh-prefill block call in float64, the contract's order
    (section 2). Returns y [B, L, d_model]. GQA is the index map
    kvh = h // n_rep (contract DEVIATION 813)."""
    x = np.asarray(x, dtype=np.float64)
    b, l, dm = x.shape
    n_rep = n_heads // n_kv
    scale = float(head_dim) ** -0.5  # contract S12/section 3
    cos_t, sin_t = _ref_rope_tables(head_dim, l)
    positions = np.arange(l)
    w64 = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    y = np.empty_like(x)
    for bb in range(b):
        h_in = x[bb]  # [L, dm]
        n1 = _ref_rms(h_in, w64["input_layernorm.weight"])
        # S5: y = x @ W^T (torch Linear, weight [out, in]).
        q = (n1 @ w64["q_proj.weight"].T).reshape(l, n_heads, head_dim)
        k = (n1 @ w64["k_proj.weight"].T).reshape(l, n_kv, head_dim)
        v = (n1 @ w64["v_proj.weight"].T).reshape(l, n_kv, head_dim)
        q = _ref_rope_apply(q, cos_t, sin_t, positions)
        k = _ref_rope_apply(k, cos_t, sin_t, positions)
        ctx = np.empty((l, n_heads, head_dim))
        for h in range(n_heads):
            kvh = h // n_rep
            scores = (q[:, h, :] @ k[:, kvh, :].T) * scale  # [L, S=L]
            # S13: the causal mask, ADDED. Query at absolute position t
            # attends keys j <= t; +0.0 where unmasked.
            mask = np.where(positions[None, :] <= positions[:, None],
                            0.0, _MASK_FILL)
            masked = scores + mask
            m = masked.max(axis=-1, keepdims=True)  # S14
            e = np.exp(masked - m)  # S15-S16
            denom = e.sum(axis=-1, keepdims=True)  # S17
            wgt = e / denom  # S18
            ctx[:, h, :] = wgt @ v[:, kvh, :]  # S19
        attn = ctx.reshape(l, n_heads * head_dim) @ w64["o_proj.weight"].T
        r1 = h_in + attn  # S22
        n2 = _ref_rms(r1, w64["post_attention_layernorm.weight"])
        g = n2 @ w64["gate_proj.weight"].T
        u = n2 @ w64["up_proj.weight"].T
        silu = g / (1.0 + np.exp(-g))  # S20, the one-division spelling
        mlp = (silu * u) @ w64["down_proj.weight"].T  # S21, S5
        y[bb] = r1 + mlp  # S23
    return y


def _uniform(rng, shape, lo, hi):
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)


def _weights(rng, dm, n_heads, n_kv, head_dim, it):
    """Fixture-scale weights: fan-in-scaled projections, near-one norm
    weights (the gate-shape neighborhood the lane's own fixtures use)."""
    qw, kw = n_heads * head_dim, n_kv * head_dim
    s_in = float(dm) ** -0.5
    s_it = float(it) ** -0.5
    return {
        "input_layernorm.weight": _uniform(rng, (dm,), 0.5, 1.5),
        "post_attention_layernorm.weight": _uniform(rng, (dm,), 0.5, 1.5),
        "q_proj.weight": _uniform(rng, (qw, dm), -s_in, s_in),
        "k_proj.weight": _uniform(rng, (kw, dm), -s_in, s_in),
        "v_proj.weight": _uniform(rng, (kw, dm), -s_in, s_in),
        "o_proj.weight": _uniform(rng, (dm, qw), -s_in, s_in),
        "gate_proj.weight": _uniform(rng, (it, dm), -s_in, s_in),
        "up_proj.weight": _uniform(rng, (it, dm), -s_in, s_in),
        "down_proj.weight": _uniform(rng, (dm, it), -s_it, s_it),
    }


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "transformer", "corpus")):
            return d
        d = os.path.dirname(d)
    return None


def _used_cache(blk, state, tokens):
    """The packed used prefix of both cache buffers at `tokens` -- the
    only region that is state (DEVIATION 795(ii): the pack stride is the
    USED length, so bytes past it are capacity, not state)."""
    n = state.batch_size * blk.n_kv_heads * tokens * blk.head_dim
    return state.k_cache[:n], state.v_cache[:n]


# The gate shape (contract section 3): d_model 32, n_heads 2, head_dim 16.
# Config A carries n_kv 1 (n_rep 2, the GQA index map live) and config B
# carries n_kv 2 (n_rep 1) with intermediate 300 -- the contract's own
# non-power-of-four down_proj shape (k = 300 gives the GEMM's P = 3 ragged
# fold), so the composition runs the fold tree the small config never
# touches.
A_DM, A_NH, A_NKV, A_HD, A_IT = 32, 2, 1, 16, 64
B_DM, B_NH, B_NKV, B_HD, B_IT = 32, 2, 2, 16, 300
L, BATCH = 4, 2
CAP = 8  # max_tokens for the carried-state arms


def main(out=sys.stdout):
    rep = Report()
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    assert_bits = mode == "identical"
    # The repeat-call arm alone is also asserted under `deterministic`:
    # same box, same bits run to run IS that tier's promise (header).
    assert_repeat = mode in ("identical", "deterministic")

    rng = np.random.default_rng(0x54726E73)  # ASCII 'Trns'
    wa = _weights(rng, A_DM, A_NH, A_NKV, A_HD, A_IT)
    xa = _uniform(rng, (BATCH, L, A_DM), -2.0, 2.0)

    # -- REFUSALS: every guard on THIS side, made to fire by name, before
    #    any device call (none of these needs the extension built) --------
    arm = "REFUSALS (Python side, no device)"
    blk = TransformerBlock(wa, n_heads=A_NH, n_kv_heads=A_NKV)
    x64 = np.zeros((1, 2, A_DM), np.float64)
    rep.raises(arm, TypeError, "float64",
               "a float64 x is refused BY NAME (no silent downcast)",
               blk.forward, x64)
    rep.raises(arm, TypeError, "bfloat16",
               "the refusal message names bfloat16 (a reduced-precision "
               "profile ships under its own version, contract section 11)",
               blk.forward, x64)
    rep.raises(arm, ValueError, "(B, L, d_model)",
               "a 2-D x is refused for forward (only step squeezes)",
               blk.forward, np.zeros((2, A_DM), np.float32))
    rep.raises(arm, ValueError, "d_model",
               "an x whose d_model is not the weights' is refused",
               blk.forward, np.zeros((1, 2, A_DM + 1), np.float32))
    rep.raises(arm, ValueError, "exactly one token",
               "a multi-token step is refused",
               blk.step, np.zeros((1, 2, A_DM), np.float32),
               blk.allocate_state(1, CAP))
    rep.raises(arm, ValueError, "state is required",
               "step without a state is refused",
               blk.step, np.zeros((1, 1, A_DM), np.float32), None)
    bad_w = dict(wa)
    bad_w["up_proj.weight"] = wa["up_proj.weight"].astype(np.float64)
    rep.raises(arm, TypeError, "up_proj.weight",
               "a float64 weight is refused by its own name",
               TransformerBlock, bad_w, n_heads=A_NH, n_kv_heads=A_NKV)
    bad_w = dict(wa)
    bad_w["q_proj.weight"] = wa["q_proj.weight"].T.copy()
    rep.raises(arm, ValueError, "q_proj.weight",
               "a transposed projection cannot cross as a plausible "
               "buffer (exact-shape refusal; the OP_NT trap's Python "
               "face)",
               TransformerBlock, bad_w, n_heads=A_NH, n_kv_heads=A_NKV)
    bad_w = dict(wa)
    del bad_w["o_proj.weight"]
    rep.raises(arm, ValueError, "missing",
               "a missing weight name is refused (no silent zero weight)",
               TransformerBlock, bad_w, n_heads=A_NH, n_kv_heads=A_NKV)
    rep.raises(arm, ValueError, "n_heads",
               "d_model = 32 with n_heads = 3 is refused "
               "(d_model == n_heads*head_dim; LlamaDims.validate carries "
               "the same rule)",
               TransformerBlock, wa, n_heads=3)
    st_bad = blk.allocate_state(1, CAP)
    st_bad.k_cache = st_bad.k_cache[: len(st_bad.k_cache) // 2]
    rep.raises(arm, ValueError, "state buffer k_cache",
               "a wrong-shape state buffer is refused by name",
               blk.step, np.zeros((1, 1, A_DM), np.float32), st_bad)
    st_wrong_b = blk.allocate_state(1, CAP)
    rep.raises(arm, ValueError, "batch_size",
               "a state allocated for another batch size is refused",
               blk.forward, xa, st_wrong_b)

    # -- REFERENCE: the float64 contract transcription, config A ---------
    arm = ("REFERENCE (contract float64, config A: n_kv %d, n_rep %d, "
           "it %d)" % (A_NKV, A_NH // A_NKV, A_IT))
    ya = blk.forward(xa)
    rep.check(arm, ya.shape == (BATCH, L, A_DM), "y has x's shape")
    rep.check(arm, bool(np.isfinite(ya).all()), "y is finite")
    rep.close(arm, ya, _ref_block(wa, xa, A_NH, A_NKV, A_HD),
              "block output matches the contract's float64 reference at "
              "the adopted anchor tolerance (rtol 1e-5, atol 1e-6; "
              "header says how to recalibrate honestly if accuracy alone "
              "fails it)")

    # -- REFERENCE, config B: n_rep 1 and the ragged down_proj ----------
    arm = ("REFERENCE (contract float64, config B: n_kv %d, n_rep %d, "
           "it %d)" % (B_NKV, B_NH // B_NKV, B_IT))
    wb = _weights(rng, B_DM, B_NH, B_NKV, B_HD, B_IT)
    xb = _uniform(rng, (BATCH, L, B_DM), -2.0, 2.0)
    blk_b = TransformerBlock(wb, n_heads=B_NH, n_kv_heads=B_NKV)
    yb = blk_b.forward(xb)
    rep.check(arm, yb.shape == (BATCH, L, B_DM), "y has x's shape")
    rep.close(arm, yb, _ref_block(wb, xb, B_NH, B_NKV, B_HD),
              "block output matches the reference with n_rep 1 and the "
              "k = 300 down_proj (the GEMM's P = 3 ragged fold inside "
              "the composition, contract section 3)")

    # -- STATE BOOKKEEPING, config A -------------------------------------
    arm = "STATE (cached_tokens and the packed cache)"
    st = blk.allocate_state(BATCH, CAP)
    y_st = blk.forward(xa, st)
    rep.check(arm, st.cached_tokens == L,
              "cached_tokens after an L=%d prefill is %d" % (L, L),
              "got %r" % st.cached_tokens)
    rep.bits_equal(arm, y_st, ya,
                   "a carried-state prefill == the stateless prefill "
                   "(the discarded cache is not an input)", assert_bits)
    rep.check(arm, st.keys().shape == (BATCH, A_NKV, L, A_HD),
              "keys() hands back the packed used region's shape")
    rep.check(arm, st.values().shape == (BATCH, A_NKV, L, A_HD),
              "values() hands back the packed used region's shape")
    rep.check(arm, bool(np.any(st.keys())),
              "the key cache is nonzero after a prefill (the state "
              "really came back)")

    # -- DECODE == PREFILL, through the Python surface -------------------
    arm = "DECODE (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; %s tier" % mode)
    std = blk.allocate_state(BATCH, CAP)
    for t in range(L):
        yt = blk.step(xa[:, t:t + 1, :], std)
        rep.bits_equal(arm, yt[:, 0, :], ya[:, t, :],
                       "step token %d == prefill token %d" % (t, t),
                       assert_bits)
    rep.check(arm, std.cached_tokens == L,
              "cached_tokens after %d steps is %d" % (L, L),
              "got %r" % std.cached_tokens)
    ks, vs = _used_cache(blk, st, L)
    kd, vd = _used_cache(blk, std, L)
    rep.bits_equal(arm, kd, ks,
                   "the packed key cache after %d steps == the "
                   "prefill's" % L, assert_bits)
    rep.bits_equal(arm, vd, vs,
                   "the packed value cache after %d steps == the "
                   "prefill's" % L, assert_bits)

    # -- RESUMPTION: split prefill == whole prefill ----------------------
    arm = "RESUMPTION (bitwise %s)" % (
        "ASSERTED" if assert_bits else "reported; %s tier" % mode)
    sts = blk.allocate_state(BATCH, CAP)
    y_head = blk.forward(xa[:, :L - 1, :], sts)
    rep.check(arm, sts.cached_tokens == L - 1,
              "cached_tokens after L-1 tokens is L-1",
              "got %r" % sts.cached_tokens)
    rep.bits_equal(arm, y_head, ya[:, :L - 1, :],
                   "the L-1 prefill's rows == the whole prefill's first "
                   "L-1 rows (prefix property)", assert_bits)
    y_tail = blk.step(xa[:, L - 1:L, :], sts)
    rep.check(arm, sts.cached_tokens == L,
              "cached_tokens after the step is L",
              "got %r" % sts.cached_tokens)
    rep.bits_equal(arm, y_tail[:, 0, :], ya[:, L - 1, :],
                   "one decode step after L-1 == prefill token L-1 "
                   "(contract 7.2: one spelling serves both paths)",
                   assert_bits)
    k1, v1 = _used_cache(blk, sts, L)
    rep.bits_equal(arm, k1, ks, "split == whole: the packed key cache",
                   assert_bits)
    rep.bits_equal(arm, v1, vs, "split == whole: the packed value cache",
                   assert_bits)

    # -- DETERMINISM: the same call twice --------------------------------
    arm = "DETERMINISM (repeat call bitwise %s)" % (
        "ASSERTED" if assert_repeat else "reported; fast tier")
    ya2 = blk.forward(xa)
    rep.bits_equal(arm, ya2, ya,
                   "the same fresh prefill through the binding twice is "
                   "byte-identical", assert_repeat)

    # -- THE LANE'S OWN REFUSALS, reached FROM PYTHON --------------------
    arm = "MOJO REFUSALS (reached from Python, refused by name)"
    st_full = blk.allocate_state(1, L)
    blk.forward(xa[:1], st_full)  # fills the cache exactly
    rep.raises(arm, Exception, "capacity",
               "a step past max_tokens is refused IN MOJO by name "
               "(the cache would grow past its capacity)",
               blk.step, xa[:1, :1, :], st_full)
    st_big = blk.allocate_state(1, 9000)
    rep.raises(arm, Exception, "ceiling",
               "max_tokens past 8192 is refused IN MOJO by name "
               "(DEVIATION 812's absolute-position ceiling)",
               blk.forward, xa[:1, :1, :], st_big)
    st_neg = blk.allocate_state(BATCH, CAP)
    st_neg.cached_tokens = -1
    rep.raises(arm, Exception, "cached_tokens",
               "cached_tokens outside [0, max_tokens] is refused BY NAME "
               "at the binding boundary (DEVIATION 795(iv): the two "
               "sides disagree about the state)",
               blk.forward, xa, st_neg)
    x_nan = xa[:1].copy()
    x_nan[0, 1, 2] = np.nan
    rep.raises(arm, Exception, "hidden_states",
               "a NaN in x is refused IN MOJO under the upstream name "
               "(llama_refuse_bad_call; payloads are vendor-shaped)",
               blk.forward, x_nan)
    # head_dim odd: sized weights for d_model 5, n_heads 1, head_dim 5;
    # the wrapper can size everything, the refusal is LlamaDims.validate's.
    w_odd = _weights(rng, 5, 1, 1, 5, 8)
    blk_odd = TransformerBlock(w_odd, n_heads=1)
    rep.raises(arm, Exception, "even",
               "an odd head_dim is refused IN MOJO by name (RoPE pairs "
               "halves; LlamaDims.validate)",
               blk_odd.forward, np.zeros((1, 2, 5), np.float32))

    # -- THE CORPUS DEBT (recorded, cannot rot -- header) ----------------
    arm = "CORPUS DEBT (transformer/corpus)"
    root = _repo_root()
    if not rep.check(arm, root is not None,
                     "transformer/corpus found walking up from this file",
                     "this gate runs IN-REPO; without the checkout the "
                     "debt arm below cannot key on the corpus landing"):
        pass
    else:
        cdir = os.path.join(root, "transformer", "corpus")
        cases = [n for n in os.listdir(cdir)
                 if os.path.isdir(os.path.join(cdir, n))
                 and n != "__pycache__"]
        if cases:
            rep.check(arm, False,
                      "transformer/corpus HOLDS CASE DATA (%s) but this "
                      "gate has no ref64 comparison wired" % sorted(cases),
                      "wire the corpus arm (mirror test_mamba_surface's "
                      "corpus arms) in the change that lands the corpus "
                      "[[fix-docs-on-discovery]]")
        else:
            rep.check(arm, True,
                      "[OWED, recorded debt -- no comparison ran] corpus "
                      "tolerance arm: transformer/corpus holds a "
                      "generator and no case data (the lane's phase 7 "
                      "row: gen_corpus.py has never been executed). The "
                      "float64 reference above is this file's own "
                      "transcription, not an independent artifact. This "
                      "row FAILS once case data appears, until the ref64 "
                      "arm is wired")

    rep.render(out)
    out.write("\n")
    if rep.failures:
        out.write("test_transformer_surface: RED. %d checks failed.\n"
                  % len(rep.failures))
        return 1
    if not assert_bits:
        out.write(
            "test_transformer_surface: the %s arms passed. The contract\n"
            "reference (both configs), the shapes, the cache bookkeeping\n"
            "and every refusal hold.\n"
            "THE CROSS-PATH BITWISE ARMS WERE %s: fast promises\n"
            "speed only (the lane's clause (d) fails under FAST by\n"
            "construction) and deterministic promises repeat-run bits\n"
            "only. Re-run under MOJOLEARN_NUMERIC_MODE=identical for the\n"
            "fully asserted form.\n"
            % (mode,
               "REPORTED, NOT ASSERTED" if mode == "fast"
               else "REPORTED (repeat-run asserted)"))
        return 0
    out.write(
        "test_transformer_surface: GREEN. The block reproduces the\n"
        "contract's float64 reference at the anchor tolerance through\n"
        "the Python surface on both GQA configs, decode equals prefill\n"
        "bit for bit per token through this path, a split prefill\n"
        "resumes bit for bit through the explicit caller-owned KV cache,\n"
        "the packed cache round-trips exactly, a repeated call is\n"
        "byte-identical, and every guard on the path fires by name --\n"
        "the Mojo-side refusals included. One box, one vendor: the\n"
        "cross-vendor statement belongs to the lane's cards, not to this\n"
        "file.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
