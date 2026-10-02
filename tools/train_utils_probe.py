#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU wall and phase split of the board's training-utility cells (lane
gap-train-utils, 2026-10-02): `mojolearn.SGD` / `Adam` / `AdamW` over one
16,777,216-float parameter and 10 seed-7 gradients (the board's `optim` fit:
copy, construct, 10 steps), and `clip_grad_norm_` over 8 gradients of
2,097,152 floats (the board's clip cell). Runs on a GPU box through
`lq add <box> CMD`; never on the laptop.

    python tools/train_utils_probe.py [--reps 3] [--tag A]

Prints `PROBE <tag> <cell> min_ms=.. median_ms=..` per cell, one
`PROBE <tag> <cell> phase <name> ms=..` per step-phase timer (the sum over
the 10 steps of one extra timed run under MOJOLEARN_TRANSFORMER_TIMING=1),
and a `PROBE <tag> <cell> digest=..` of the result bits, so two arms of an
A/B (environment switches) can be compared byte for byte.
"""
import argparse
import hashlib
import os
import statistics
import sys
import tempfile
import time

import numpy as np

P, STEPS, SEED = 1 << 24, 10, 7
GRAD_TENSORS, GRAD_SIZE = 8, 2_097_152
HYPER = {
    "sgd": ("SGD", dict(lr=1e-3, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False,
                        maximize=False)),
    "adam": ("Adam", dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, maximize=False)),
    "adamw": ("AdamW", dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                            maximize=False)),
}


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _captured(fn):
    """Run fn with fd 1 redirected to a file; return (result, captured text)."""
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as f:
        os.dup2(f.fileno(), 1)
        try:
            out = fn()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        text = f.read().decode(errors="replace")
    return out, text


def _phases(text):
    acc = {}
    for line in text.splitlines():
        w = line.split()
        if len(w) == 4 and w[0] == "timing" and w[3] == "ms":
            acc[w[1]] = acc.get(w[1], 0.0) + float(w[2])
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default=os.environ.get("PROBE_TAG", "default"))
    ap.add_argument("--cells", default="sgd,adam,adamw,clip")
    a = ap.parse_args()
    import mojolearn as ml
    tag = a.tag
    rng = np.random.default_rng(SEED)
    p0 = rng.standard_normal(P).astype(np.float32)
    grads = [rng.standard_normal(P).astype(np.float32) * 0.01 for _ in range(STEPS)]
    for cell in a.cells.split(","):
        if cell == "clip":
            g0 = [np.random.default_rng(SEED + j).standard_normal(GRAD_SIZE).astype(np.float32)
                  for j in range(GRAD_TENSORS)]
            ms, norms = [], []
            for _ in range(a.reps + 1):
                G = [g.copy() for g in g0]
                t0 = time.perf_counter()
                norms.append(float(ml.clip_grad_norm_(G, 1.0)))
                ms.append((time.perf_counter() - t0) * 1e3)
            ms = ms[1:]
            print("PROBE %s clip min_ms=%.3f median_ms=%.3f norm=%r digest=%s"
                  % (tag, min(ms), statistics.median(ms), norms[-1], _digest(*G)), flush=True)
            continue
        name, hyper = HYPER[cell]
        cls = getattr(ml, name)

        def fit():
            p = p0.copy()
            opt = cls([p], **hyper)
            for gr in grads:
                opt.step([gr])
            return p, opt
        ms = []
        for _ in range(a.reps + 1):
            t0 = time.perf_counter()
            p, opt = fit()
            ms.append((time.perf_counter() - t0) * 1e3)
            del opt
        ms = ms[1:]
        print("PROBE %s %s min_ms=%.3f median_ms=%.3f digest=%s"
              % (tag, cell, min(ms), statistics.median(ms), _digest(p)), flush=True)
        os.environ["MOJOLEARN_TRANSFORMER_TIMING"] = "1"
        try:
            (p2, opt2), text = _captured(fit)
        finally:
            os.environ.pop("MOJOLEARN_TRANSFORMER_TIMING", None)
        for k, v in sorted(_phases(text).items()):
            print("PROBE %s %s phase %s ms=%.3f" % (tag, cell, k, v), flush=True)
        st = opt2.state_dict()
        print("PROBE %s %s state_digest=%s" % (tag, cell, _digest(p2, st["exp_avg"], st["exp_avg_sq"])),
              flush=True)
        del opt2


if __name__ == "__main__":
    main()
