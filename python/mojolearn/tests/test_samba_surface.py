# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Samba training surface: schedules, clause 9.2
accumulation, the position-keyed RNG, dropout, and `SambaStack`.

    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_mamba.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \\
        python3 -m mojolearn.tests.test_samba_surface

IDENTICAL only: the surface is identical-only by the owner's order, and the
run refuses any other tier. Bitwise arms are asserted. The schedule pins
are the float32 bit patterns of the exact rational schedule, recorded from
a run and asserted thereafter (a pin that is absent is reported as OWED).
"""

import hashlib
import math
import os
import sys
import tempfile

import numpy as np

import mojolearn
from mojolearn import _training_impl as T
from mojolearn import _samba_impl as S


class Report(object):
    def __init__(self):
        self.rows = []

    def check(self, arm, cond, what, detail=""):
        self.rows.append(("pass" if cond else "FAIL", arm,
                          what + ((" -- " + detail) if (detail and not cond) else "")))
        return bool(cond)

    def report(self, arm, what):
        self.rows.append(("rprt", arm, what))

    def bits_equal(self, arm, got, want, what, assert_bits=True):
        g = np.ascontiguousarray(got, dtype=np.float32).ravel().view(np.uint32)
        w = np.ascontiguousarray(want, dtype=np.float32).ravel().view(np.uint32)
        if g.shape != w.shape:
            return self.check(arm, False, what, "shapes differ")
        diff = np.flatnonzero(g != w)
        same = diff.size == 0
        detail = "" if same else ("%d of %d cells differ, first at %d: 0x%08x vs 0x%08x"
                                  % (diff.size, g.size, int(diff[0]),
                                     int(g[diff[0]]), int(w[diff[0]])))
        if assert_bits:
            return self.check(arm, same, what, detail)
        self.report(arm, what + (": SAME BITS" if same else ": MOVED, " + detail))
        return same

    def raises(self, arm, exc_type, needle, what, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except exc_type as exc:
            return self.check(arm, needle in str(exc), what,
                              "message lacks %r: %s" % (needle, exc))
        except Exception as exc:  # noqa: BLE001
            return self.check(arm, False, what, "raised %s: %s" % (type(exc).__name__, exc))
        return self.check(arm, False, what, "IS INERT, the call was accepted")

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
        out.write("\n  %d checks, %d failed\n" % (len(self.rows), len(self.failures)))


def hashed(shape, salt, scale=1.0):
    """Deterministic non-uniform float32 test data (never np.random)."""
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.uint64)
    x = (i * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(29)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(32)
    u = (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(1 << 24)
    return ((u - 0.5) * 2.0 * scale).astype(np.float32).reshape(shape)


def hashed_ids(shape, vocab, salt):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.uint64)
    x = (i * np.uint64(0xD1B54A32D192ED03) + np.uint64(salt)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(vocab)).astype(np.int32).reshape(shape)


# Pinned float32 bit patterns of the exact rational schedules, recorded on
# the first identical run. Empty entries are OWED and reported, not
# asserted; a filled entry is asserted.
SCHEDULE_PINS = {
    "constant": {"args": (1e-3, 4, None, 0.0), "steps": 6, "bits": []},
    "linear": {"args": (1e-3, 2, 10, 1e-4), "steps": 11, "bits": []},
    "cosine": {"args": (1e-3, 2, 10, 1e-4), "steps": 11, "bits": []},
}


def arm_provenance(rep):
    mode = T.numeric_mode_used()
    rep.check("PROVENANCE", mode == "identical", "training binding is identical, got %r" % mode)
    rep.report("PROVENANCE", "vendor %s" % T.vendor_used())
    tb = getattr(mojolearn.TransformerBlock, "backward", None)
    rep.report("PROVENANCE", "TransformerBlock.backward %s"
               % ("PRESENT" if tb is not None else "ABSENT (refused by name below)"))


def arm_schedule(rep):
    arm = "SCHEDULE"
    c = T.ConstantLR(1e-3, warmup_steps=4)
    rep.check(arm, c.lr_at(4) == float(np.float32(1e-3)), "constant reaches peak at warmup end")
    rep.check(arm, c.lr_at(1) == T._f32_round(T.Fraction(c.peak_lr) / 4),
              "constant warmup step 1 is peak/4 (exact rational, rounded once)")
    rep.check(arm, c.lr_at(100) == c.lr_at(4), "constant after warmup is flat")
    rep.raises(arm, ValueError, "ONE-BASED", "step 0 is refused by name", c.lr_at, 0)
    lin = T.WarmupLinearLR(1e-3, 2, 10, 1e-4)
    rep.check(arm, lin.lr_at(10) == float(np.float32(1e-4)) and lin.lr_at(11) == lin.lr_at(10),
              "linear reaches min_lr at total_steps and holds")
    cos = T.WarmupCosineLR(1e-3, 2, 10, 1e-4)
    mid_exact = T._f32_round(T.Fraction(cos.peak_lr) / 2 + T.Fraction(cos.min_lr) / 2)
    rep.check(arm, cos.lr_at(6) == mid_exact, "cosine at p = 1/2 rounds the exact midpoint")
    # Float64 sanity against math.cos, a REPORT: the exact value differs from
    # the libm spelling by at most one float32 ulp, never asserted bitwise.
    worst = 0.0
    for t in range(3, 10):
        p = (t - 2) / 8.0
        ref = 1e-4 + (1e-3 - 1e-4) * (1 + math.cos(math.pi * p)) / 2
        worst = max(worst, abs(cos.lr_at(t) - ref) / ref)
    rep.check(arm, worst < 1e-6, "cosine within 1e-6 relative of a float64 math.cos reference (worst %.2e)" % worst)
    mono = all(cos.lr_at(t) >= cos.lr_at(t + 1) for t in range(2, 11))
    rep.check(arm, mono, "cosine is non-increasing after warmup")
    for name, pin in SCHEDULE_PINS.items():
        cls = {"constant": T.ConstantLR, "linear": T.WarmupLinearLR,
               "cosine": T.WarmupCosineLR}[name]
        sched = cls(*pin["args"])
        bits = [sched.bits_at(t) for t in range(1, pin["steps"] + 1)]
        rep.report(arm, "%s bits %s" % (name, " ".join("0x%08x" % b for b in bits)))
        if pin["bits"]:
            rep.check(arm, bits == pin["bits"], "%s lr sequence matches its pin" % name,
                      "got %s" % " ".join("0x%08x" % b for b in bits))
        else:
            rep.report(arm, "%s pin OWED (record the bits above)" % name)
    rt = T._Schedule.from_config(cos.config())
    rep.check(arm, [rt.bits_at(t) for t in range(1, 12)] == [cos.bits_at(t) for t in range(1, 12)],
              "schedule config round trip reproduces the bits")


def arm_accumulation(rep):
    arm = "ACCUM"
    rep.check(arm, T.accumulation_is_aligned(512, 2) and T.accumulation_is_aligned(512, 4),
              "T=512 aligned at A=2 and A=4 (contract table)")
    rep.check(arm, not T.accumulation_is_aligned(512, 8) and not T.accumulation_is_aligned(512, 3)
              and not T.accumulation_is_aligned(300, 2), "T=512 A=8, A=3 and T=300 A=2 are not aligned")
    m, n, k = 512, 96, 64
    a = hashed((m, k), 11)
    w = hashed((n, k), 12)
    dc = hashed((m, n), 13, 0.01)
    da_full, dw_full = T.linear_backward(dc, a, w)
    for steps in (2, 4):
        rows = m // steps
        pieces = [T.linear_backward(dc[i * rows:(i + 1) * rows], a[i * rows:(i + 1) * rows], w)[1]
                  for i in range(steps)]
        comb = T.accumulate_grads(pieces, tokens=m)
        rep.bits_equal(arm, comb, dw_full, "head weight gradient: A=%d tree == unsplit at T=512" % steps)
        serial = pieces[0]
        for pc in pieces[1:]:
            serial = T.accumulate_grads([serial, pc], tokens=None)
        rep.bits_equal(arm, serial, dw_full, "A=%d SERIAL running sum vs unsplit" % steps, assert_bits=False)
    pieces = [T.linear_backward(dc[i * 64:(i + 1) * 64], a[i * 64:(i + 1) * 64], w)[1] for i in range(8)]
    rep.raises(arm, Exception, "MISALIGNED", "A=8 at T=512 is refused by name",
               T.accumulate_grads, pieces, 512)
    rep.raises(arm, Exception, "POWER OF TWO", "A=3 is refused by name",
               T.accumulate_grads, pieces[:3], 512)
    rep.raises(arm, ValueError, "power of two", "accumulation_steps=3 refused on the optimizer",
               T.AdamW, [w.copy()], accumulation_steps=3)
    # The final RMSNorm weight gradient contracts over tokens too.
    x = hashed((m, k), 14)
    nw = (1.0 + hashed((k,), 15, 0.1)).astype(np.float32)
    dy = hashed((m, k), 16, 0.01)
    _, dnw_full = T.rms_norm_backward(dy, x, nw, 1e-5)
    pieces = [T.rms_norm_backward(dy[i * 128:(i + 1) * 128], x[i * 128:(i + 1) * 128], nw, 1e-5)[1]
              for i in range(4)]
    rep.bits_equal(arm, T.accumulate_grads(pieces, tokens=m), dnw_full,
                   "rms_norm weight gradient: A=4 tree == unsplit at T=512", assert_bits=False)
    ids = hashed_ids((m,), 40, 17)
    de_full = T.embedding_backward(dy, ids, 40)
    pieces = [T.embedding_backward(dy[i * 256:(i + 1) * 256], ids[i * 256:(i + 1) * 256], 40) for i in range(2)]
    rep.bits_equal(arm, T.accumulate_grads(pieces, tokens=m), de_full,
                   "embedding gradient: A=2 tree vs unsplit (run-sorted fold, no claim)", assert_bits=False)
    # step_accumulated equals step over the tree-combined gradient.
    p1 = hashed((k,), 18)
    p2 = p1.copy()
    o1 = T.AdamW([p1], lr=1e-2, accumulation_steps=2)
    o2 = T.AdamW([p2], lr=1e-2)
    g0, g1 = hashed((k,), 19), hashed((k,), 20)
    o1.step_accumulated([[g0], [g1]], tokens=256)
    o2.step([T.accumulate_grads([g0, g1], tokens=256)])
    rep.bits_equal(arm, p1, p2, "step_accumulated == step(accumulate_grads)")
    rep.raises(arm, ValueError, "accumulation_steps", "step_accumulated refuses a different count",
               o1.step_accumulated, [[g0]], 256)


def arm_rng(rep):
    arm = "RNG"
    g = T.Generator(12345)
    u = g.uniform((1000,))
    rep.check(arm, u.dtype == np.float32 and u.min() >= 0.0 and u.max() < 1.0, "uniform in [0, 1)")
    g2 = T.Generator(12345)
    rep.bits_equal(arm, g2.uniform((1000,)), u, "same seed, same counter, same bits")
    rep.check(arm, not np.array_equal(g.uniform((1000,)), u), "the next stream differs")
    binding = T._load()
    whole = T._rng_call(binding, T._RNG_UNIFORM, 1000, 0, 12345, 0, 0.0, 1.0)
    part = T._rng_call(binding, T._RNG_UNIFORM, 300, 700, 12345, 0, 0.0, 1.0)
    rep.bits_equal(arm, part, whole[700:], "a slice at offset 700 equals the whole stream's elements 700..")
    nrm = T.Generator(7).normal((20000,), 0.0, 1.0)
    rep.check(arm, abs(float(nrm.mean())) < 0.03 and abs(float(nrm.std()) - 1.0) < 0.03,
              "normal(0, 1) sample mean/std within 0.03 (mean %.4f std %.4f)" % (nrm.mean(), nrm.std()))
    rep.check(arm, np.isfinite(nrm).all(), "normal draws are finite")
    ku = T.Generator(3).kaiming_uniform((64, 32), fan_in=32)
    rep.check(arm, float(np.abs(ku).max()) <= 1.0 / math.sqrt(32.0) and ku.shape == (64, 32),
              "kaiming_uniform bound is 1/sqrt(fan_in)")
    x = hashed((4, 256), 21)
    gd = T.Generator(99)
    y, key = gd.dropout(x, 0.25)
    kept = y != 0.0
    frac = float(kept.mean())
    rep.check(arm, 0.65 < frac < 0.85, "dropout p=0.25 keeps about 75%% (kept %.3f)" % frac)
    scale = np.float32(1.0 / (1.0 - np.float32(0.25)))
    rep.bits_equal(arm, y[kept], (x * scale).astype(np.float32)[kept], "kept cells are x * scale bitwise")
    dy = hashed((4, 256), 22)
    dx = gd.dropout_backward(dy, key)
    rep.bits_equal(arm, dx != 0.0, kept.astype(np.float32), "backward mask equals forward mask")
    rep.bits_equal(arm, dx[kept], (dy * scale).astype(np.float32)[kept], "backward is dy * scale on kept cells")
    g3 = T.Generator(99)
    y2, _ = g3.dropout(x, 0.25)
    rep.bits_equal(arm, y2, y, "dropout replays from the same seed and counter")
    half, _ = T.Generator(99).dropout(x[2:], 0.25, offset=2 * 256)
    rep.bits_equal(arm, half, y[2:], "a microbatch at its token offset draws the unsplit coins")
    st = gd.state_dict()
    rep.check(arm, st == {"seed": 99, "counter": 1}, "state_dict is {seed, counter}")
    rep.raises(arm, ValueError, "[0, 1)", "dropout p=1 refused", gd.dropout, x, 1.0)


def _tiny_config(tie=True, layers=("mamba3", "mamba3")):
    return S.SambaConfig(vocab=64, d_model=32, layers=layers, tie_embeddings=tie)


def arm_stack(rep):
    arm = "STACK"
    cfg = _tiny_config()
    stack = S.SambaStack(cfg, generator=T.Generator(1), lr=1e-3,
                         lr_schedule=T.WarmupCosineLR(1e-3, 2, 8, 1e-4), max_norm=1.0)
    rep.check(arm, stack.n_total == sum(int(np.prod(s)) for _, s in cfg.registry()),
              "registry sizes the flat buffer (%d floats)" % stack.n_total)
    ids = hashed_ids((2, 17), 64, 31)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = stack.forward(inputs)
    rep.check(arm, logits.shape == (2, 16, 64) and np.isfinite(logits).all(), "forward gives finite (B, L, V) logits")
    loss, grads = stack.loss_and_grads(inputs, targets)
    rep.check(arm, math.isfinite(loss) and all(np.isfinite(g).all() for g in grads),
              "loss_and_grads finite (loss %.4f)" % loss)
    rep.check(arm, [g.shape for g in grads] == [stack.shapes[n] for n in stack.names],
              "gradient shapes match the registry")
    loss2, grads2 = stack.loss_and_grads(inputs, targets)
    rep.check(arm, all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(grads, grads2)),
              "loss_and_grads is repeatable bitwise in one process")
    before = stack.flat.copy()
    out = stack.train_step(inputs, targets)
    rep.check(arm, out["step"] == 1 and out["lr"] == stack.optimizer.lr_schedule.lr_at(1),
              "train_step 1 used lr_at(1) = %r" % out["lr"])
    rep.check(arm, not np.array_equal(before, stack.flat), "train_step moved the parameters")
    rep.check(arm, out["total_norm"] is not None, "clip ran (total norm %r)" % out["total_norm"])
    out = stack.train_step(inputs, targets)
    rep.check(arm, out["step"] == 2, "second train_step")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.json")
        sha1 = stack.save_checkpoint(path)
        twin = S.SambaStack.from_checkpoint(path)
        sha2 = twin.save_checkpoint(os.path.join(d, "ck2.json"))
        rep.check(arm, sha1 == sha2, "checkpoint round trip reproduces the file bytes (%s)" % sha1[:16])
        rep.check(arm, twin.optimizer.t == 2 and twin.generator.state_dict() == stack.generator.state_dict(),
                  "resumed step counter and rng state")
        a = stack.train_step(inputs, targets)
        b = twin.train_step(inputs, targets)
        rep.bits_equal(arm, twin.flat, stack.flat, "step 3 after resume equals step 3 without")
        rep.check(arm, a["loss"] == b["loss"], "resumed loss equals")
    rep.raises(arm, ValueError, "exactly one", "weights and generator both absent refused",
               S.SambaStack, cfg)
    rep.raises(arm, ValueError, "multiple of 32", "mamba3 d_model refused by name",
               S.SambaConfig, 64, 48, ("mamba3",))
    # dropout in the stack: two runs agree, and p > 0 changes the bits.
    cfg_d = S.SambaConfig(vocab=64, d_model=32, layers=("mamba3",), dropout=0.1)
    s1 = S.SambaStack(cfg_d, generator=T.Generator(5), lr=1e-3)
    s2 = S.SambaStack(cfg_d, generator=T.Generator(5), lr=1e-3)
    s1.train_step(inputs, targets)
    s2.train_step(inputs, targets)
    rep.bits_equal(arm, s1.flat, s2.flat, "dropout stack: two fresh runs agree bitwise")
    rep.check(arm, s1.generator.counter == s2.generator.counter, "dropout consumed one stream per step")


def arm_stack_accum(rep):
    arm = "STACK-ACCUM"
    cfg = _tiny_config(tie=False, layers=("mamba3",))
    ids = hashed_ids((8, 65), 64, 41)
    inputs, targets = ids[:, :-1], ids[:, 1:]   # T = 8 * 64 = 512
    base = S.SambaStack(cfg, generator=T.Generator(2), lr=1e-3)
    _, g1 = base.loss_and_grads(inputs, targets)
    for steps in (2, 4):
        st = S.SambaStack(cfg, generator=T.Generator(2), lr=1e-3, accumulation_steps=steps)
        rep.bits_equal(arm, st.flat, base.flat, "A=%d stack starts from the same bits" % steps)
        rows = 8 // steps
        count = int(targets.size)
        parts = [st.loss_and_grads(inputs[i * rows:(i + 1) * rows], targets[i * rows:(i + 1) * rows],
                                   num_items=count)[1] for i in range(steps)]
        comb = T.accumulate_grads(parts, tokens=512)
        for n, a, b in zip(st.names, comb, g1):
            assert_it = n in ("lm_head.weight", "norm_f.weight")
            rep.bits_equal(arm, a, b, "A=%d %s tree vs unsplit" % (steps, n), assert_bits=assert_it)
    st8 = S.SambaStack(cfg, generator=T.Generator(2), lr=1e-3, accumulation_steps=8)
    rep.raises(arm, ValueError, "MISALIGNED", "train_step at A=8, T=512 refused by name",
               st8.train_step, inputs, targets)
    st2 = S.SambaStack(cfg, generator=T.Generator(2), lr=1e-3, accumulation_steps=2)
    st2.train_step(inputs, targets)
    base.train_step(inputs, targets)
    rep.bits_equal(arm, st2.arrays["lm_head.weight"], base.arrays["lm_head.weight"],
                   "A=2 train_step: lm_head.weight equals the unsplit step")
    rep.bits_equal(arm, st2.flat, base.flat, "A=2 train_step: whole parameter buffer vs unsplit",
                   assert_bits=False)


def arm_attention(rep):
    arm = "ATTENTION"
    cfg = S.SambaConfig(vocab=64, d_model=32, layers=("mamba3", "attention"),
                        n_heads=2, intermediate=64)
    st = S.SambaStack(cfg, generator=T.Generator(9), lr=1e-3)
    ids = hashed_ids((2, 9), 64, 51)
    logits = st.forward(ids[:, :-1])
    rep.check(arm, logits.shape == (2, 8, 64) and np.isfinite(logits).all(), "hybrid forward is finite")
    if getattr(mojolearn.TransformerBlock, "backward", None) is None:
        rep.raises(arm, NotImplementedError, "NO backward", "attention backward refused by name",
                   st.loss_and_grads, ids[:, :-1], ids[:, 1:])
    else:
        loss, grads = st.loss_and_grads(ids[:, :-1], ids[:, 1:])
        rep.check(arm, math.isfinite(loss) and all(np.isfinite(g).all() for g in grads),
                  "hybrid loss_and_grads finite through TransformerBlock.backward")


def main(out=sys.stdout):
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").strip().lower()
    if mode != "identical":
        out.write("test_samba_surface: IDENTICAL ONLY. Set MOJOLEARN_NUMERIC_MODE=identical.\n")
        return 2
    rep = Report()
    aborted = []
    for name, fn in (("PROVENANCE", arm_provenance), ("SCHEDULE", arm_schedule),
                     ("ACCUM", arm_accumulation), ("RNG", arm_rng),
                     ("STACK", arm_stack), ("STACK-ACCUM", arm_stack_accum),
                     ("ATTENTION", arm_attention)):
        try:
            fn(rep)
        except Exception as exc:  # noqa: BLE001
            import traceback
            aborted.append((name, "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc())))
    rep.render(out)
    for name, why in aborted:
        out.write("\n  %s ARM DID NOT RUN\n" % name)
        for line in why.splitlines():
            out.write("    %s\n" % line)
    if rep.failures or aborted:
        out.write("\ntest_samba_surface: RED. %d checks failed, %d arms did not run.\n"
                  % (len(rep.failures), len(aborted)))
        return 1
    out.write("\ntest_samba_surface: GREEN.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
