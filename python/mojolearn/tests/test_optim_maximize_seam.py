# SPDX-License-Identifier: Apache-2.0
"""THE maximize= SEAM (DEVIATION 6200, IDENTITY_PATHS row 200): a
`maximize=True` step on `g` must equal, BIT FOR BIT, a `maximize=False` step
on `-g`, where `-g` is the sign bit flipped (torch's `-grads[i]`,
torch/optim/sgd.py and adam.py `_single_tensor_*`). That is the oracle, stated
in NumPy over uint32 views so it cannot share a line with the bindings'
training/maximize.mojo.

THE FIXTURE MUST SEPARATE. `-g` and `0.0 - g` are both exact and differ only at
`g = +0.0`; before anything is trusted this driver runs the plain step on both
spellings and refuses as VACUOUS (exit 2) when they agree. The fixture plants
+0.0 and -0.0 gradient cells and -0.0 parameters for that reason.

Also held: the caller's gradient is never left negated (untouched without the
clip; with the clip it equals the `maximize=False` clip of `g`), and the
refusal of a non-bool `maximize`.

Run directly (`python python/mojolearn/tests/test_optim_maximize_seam.py`;
exit 0 PASS, 1 FAIL naming the part, 2 VACUOUS) or under pytest. Listed in
tools/identity_lanes/neural.checks with the sabotage
training/checks/sabotage/maximize_6200_subtract_from_zero.patch.
"""
import sys

import numpy as np

import mojolearn as ml

T = ml.training
SHAPES = ((16, 8), (16,), (3, 16))


def _tensors(seed):
    rng = np.random.default_rng(seed)
    ps = [rng.uniform(-0.5, 0.5, s).astype(np.float32) for s in SHAPES]
    gs = [[rng.uniform(-1.0, 1.0, s).astype(np.float32) for s in SHAPES] for _ in range(3)]
    for k in range(len(SHAPES)):
        ps[k].reshape(-1)[:3] = np.float32(-0.0)
        for g in gs:
            g[k].reshape(-1)[0] = np.float32(0.0)
            g[k].reshape(-1)[1] = np.float32(-0.0)
            g[k].reshape(-1)[2] = np.float32(0.0)
    return ps, gs


def _neg_bits(a):
    """The oracle's `-g`: the sign bit flipped, over a uint32 view."""
    return (a.view(np.uint32) ^ np.uint32(0x80000000)).view(np.float32)


def _neg_subtract(a):
    """The OTHER legal spelling, `0.0 - g` (+0.0 at g = +0.0)."""
    return (np.float32(0.0) - a).astype(np.float32)


CONFIGS = (
    ("sgd-nesterov-decay", lambda p, mx: T.SGD(p, lr=1e-2, momentum=0.9, nesterov=True, weight_decay=0.01, maximize=mx)),
    ("sgd-plain", lambda p, mx: T.SGD(p, lr=1e-2, maximize=mx)),
    ("adam-decay", lambda p, mx: T.Adam(p, lr=1e-3, weight_decay=0.01, maximize=mx)),
    ("adamw", lambda p, mx: T.AdamW(p, lr=1e-3, weight_decay=0.01, maximize=mx)),
)


def _run(make, seed, maximize, spell, max_norm):
    """(state bits, the grads the caller holds after each step)."""
    ps, gs = _tensors(seed)
    opt = make(ps, maximize)
    after, norms = [], []
    for g in gs:
        feed = [x.copy() for x in g] if maximize else [spell(x) for x in g]
        norms.append(opt.step(feed, max_norm=max_norm))
        after.append([x.copy() for x in feed])
    state = [x.view(np.uint32).copy() for x in ps] + [
        np.asarray(opt.exp_avg).view(np.uint32).copy(), np.asarray(opt.exp_avg_sq).view(np.uint32).copy(),
        np.asarray(opt.buf_initialized).copy()]
    return state, after, norms


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def check():
    """[] on PASS, else the failing parts; raises SystemExit(2) when vacuous."""
    fails = []
    # THE FIXTURE MUST SEPARATE the two spellings, on the SGD copy arm.
    sep = [not _same(_run(make, 7, False, _neg_bits, None)[0], _run(make, 7, False, _neg_subtract, None)[0])
           for name, make in CONFIGS if name.startswith("sgd")]
    if not all(sep):
        print("VACUOUS: the fixture does not separate -g from 0.0 - g on the SGD arms", flush=True)
        raise SystemExit(2)
    for name, make in CONFIGS:
        for max_norm in (None, 1.0):
            for seed in (7, 11):
                got, got_after, got_n = _run(make, seed, True, None, max_norm)
                want, want_after, want_n = _run(make, seed, False, _neg_bits, max_norm)
                tag = f"{name} max_norm={max_norm} seed={seed}"
                if not _same(got, want):
                    fails.append(f"{tag}: state differs from the plain step on -g")
                if got_n != want_n:
                    fails.append(f"{tag}: pre-clip norms {got_n} vs {want_n}")
                # The caller's gradient: untouched without the clip; with it,
                # the maximize=False clip of g (the oracle's grads negated back).
                _, gs = _tensors(seed)
                for j, (ga, wa) in enumerate(zip(got_after, want_after)):
                    exp = gs[j] if max_norm is None else [_neg_bits(x) for x in wa]
                    if not _same([x.view(np.uint32) for x in ga], [x.view(np.uint32) for x in exp]):
                        fails.append(f"{tag}: step {j}: the caller's gradient came back changed")
    try:
        T.Adam(_tensors(1)[0], maximize=1)
        fails.append("Adam(maximize=1) was accepted; a non-bool must be refused")
    except TypeError:
        pass
    return fails


def test_optim_maximize_seam():
    fails = check()
    assert not fails, "\n".join(fails)


if __name__ == "__main__":
    f = check()
    print(f"maximize seam (DEVIATION 6200) on {T.vendor_used()}: " + ("PASS" if not f else "FAIL"), flush=True)
    for line in f:
        print("  " + line, flush=True)
    sys.exit(1 if f else 0)
