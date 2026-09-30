#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a SambaStack training step's time goes ABOVE the kernels
(lane/neural-net-experiment, the S16 pass). On the L40S the board's Samba
step read 131 ms while the two Mamba-3 backwards were ~67 ms and the
transformer blocks ~10 ms: the rest is the forward, the head, the
optimizer, and Python between them. This wraps the stack's phases with
wall clocks (a device wait sits inside each native call, so the walls are
inclusive) and prints them per call, sorted, with shares:

    python tools/samba_step_profile.py --calls 5

Shape: the board's samba-train-step "full" (B2 L512 d384, layers mamba3 /
attention / mamba3 / attention). Unmeasured here (no GPU).
"""
import argparse, collections, functools, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=5)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--length", type=int, default=512)
ap.add_argument("--d-model", type=int, default=384)
a = ap.parse_args()

import numpy as np
import mojolearn as ml
from mojolearn import _samba_impl as S
from mojolearn import _mamba_impl as M
from mojolearn import _training_impl as T
import bench_board_neural as B

cfg = ml.SambaConfig(256, a.d_model, ("mamba3", "attention", "mamba3", "attention"), n_heads=6,
                     intermediate=1024, norm_eps=B.SAMBA_NORM_EPS, dropout=B.SAMBA_DROPOUT)
rng = np.random.default_rng(7)
w = {n: np.ascontiguousarray(B.samba_init(rng, n, tuple(s))) for n, s in cfg.registry()}
model = ml.SambaStack(cfg, weights=w, lr=B.ADAMW["lr"], betas=B.ADAMW["betas"], eps=B.ADAMW["eps"],
                      weight_decay=B.ADAMW["weight_decay"], max_norm=None)
ids = rng.integers(0, 256, size=(a.batch, a.length + 1), dtype=np.int64)
inputs, targets = np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])

walls = collections.OrderedDict()
depth = [0]


def wrap(owner, name, label):
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def timed(*args, **kw):
        t0 = time.perf_counter()
        depth[0] += 1
        try:
            return orig(*args, **kw)
        finally:
            depth[0] -= 1
            walls.setdefault(label, []).append(time.perf_counter() - t0)
    setattr(owner, name, timed)


wrap(S.SambaStack, "_forward", "stack._forward (embedding + all blocks + norm/head fwd)")
wrap(S.SambaStack, "loss_and_grads", "stack.loss_and_grads (forward + loss + backward)")
wrap(M.Mamba3Block, "forward", "Mamba3Block.forward")
wrap(M.Mamba3Block, "backward", "Mamba3Block.backward")
tb = getattr(ml, "TransformerBlock", None)
if tb is not None:
    wrap(tb, "forward", "TransformerBlock.forward")
    wrap(tb, "backward", "TransformerBlock.backward")
for fn in ("samba_head_loss", "cross_entropy", "linear_forward", "linear_backward", "rms_norm_backward",
           "embedding_backward", "embedding_forward", "accumulate_grads"):
    if hasattr(T, fn):
        wrap(T, fn, "T." + fn)
opt = model.optimizer
wrap(type(opt), "step", "optimizer.step (AdamW)")

per_call = []
for c in range(a.calls + 1):
    walls.clear()
    t0 = time.perf_counter()
    out = model.train_step(inputs, targets)
    total = time.perf_counter() - t0
    if c == 0:
        continue  # warm-up (compiles, first allocations)
    per_call.append((total, {k: sum(v) for k, v in walls.items()}))

tot = statistics.median(t for t, _ in per_call)
print("SambaStack.train_step B=%d L=%d d_model=%d: median %.2f ms over %d calls after warm-up" % (
    a.batch, a.length, a.d_model, tot * 1e3, len(per_call)))
print("%-64s %9s %7s" % ("phase (inclusive walls, summed over the step)", "ms", "share"))
names = set()
for _, d in per_call:
    names |= set(d)
rows = [(n, statistics.median(d.get(n, 0.0) for _, d in per_call)) for n in names]
for n, v in sorted(rows, key=lambda kv: -kv[1]):
    print("%-64s %9.2f %6.1f%%" % (n, v * 1e3, 100.0 * v / tot))
covered = sum(v for n, v in rows if n.startswith(("Mamba3Block.", "TransformerBlock.", "T.", "optimizer.")))
print("%-64s %9.2f %6.1f%%" % ("leaf phases (blocks + T.* + optimizer) summed", covered * 1e3, 100.0 * covered / tot))
print("%-64s %9.2f %6.1f%%" % ("everything else (Python between the leaves)", (tot - covered) * 1e3, 100.0 * (tot - covered) / tot))
