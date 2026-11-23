#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequence lengths on Sequential's recurrent layers: what the mask costs, our arms only (lane seq-blocks-3).

Per cell (rnn, gru, lstm): Sequential(input_shape=(T, F)).<cell>(H).narrow(0, T - 1, 1).flatten().linear(2),
cross-entropy, adam, one epoch, batch 512, seed 0, on N sequences (full size: N 65536, T 16, F 8, H 64, the
recurrent board recipe's shape; --shape small for a smoke). Arms, each with the same data and seed:
  none      fit / predict without lengths (the unmasked scan or chain)
  full      lengths = T for every row: the masked route (the padded-input select, the scan's or the per-step
            cell's mask) on the same work. Quality: its parameters and predictions must be the `none` arm's
            words (no step is past a row's length), printed as `same_words`.
  varied    lengths 1 .. T (row i: i % T + 1): the packed-sequence semantics; quality printed as the held-out
            accuracy of both arms' own task (shown, never a gate)
Rows print `LENGTHS label=<label> cell=<c> arm=<a> fit_ms=<..> predict_ms=<..> same_words=<..>`.
Timing: one excluded warmup; a row whose warmup takes under 10 ms is the median of 20 consecutive calls (AGENTS.md
"Small rows"), else one call. Host probe lines first (AGENTS.md "Times are compared only on the same box").

    python tools/ops/sequence_lengths_timing.py [--shape full|small] [--cells rnn,gru,lstm] [--label lengths]
"""
import argparse
import os
import platform
import statistics
import subprocess
import time

import numpy as np


def probe():
    print("PROBE host", platform.node(), "cpu", platform.processor() or platform.machine(), "cores", os.cpu_count())
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("PROBE gpu", out or "none")
    except (OSError, subprocess.SubprocessError):
        print("PROBE gpu none")
    import mojolearn
    print("PROBE mojolearn", mojolearn.__version__, "vendor", mojolearn._backend.vendor())


def clock(fn):
    t0 = time.perf_counter()
    out = fn()
    warm = time.perf_counter() - t0
    if warm >= 0.010:
        t0 = time.perf_counter()
        out = fn()
        return (time.perf_counter() - t0) * 1e3, out
    times = []
    for _ in range(20):  # glue: the small-row median (AGENTS.md)
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3, out


def data(n, t, f, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, t, f)).astype(np.float32)
    lengths = (np.arange(n) % t + 1).astype(np.int64)
    # class: the sign of channel 0's sum over each row's valid steps
    valid = np.arange(t)[None, :] < lengths[:, None]
    y = ((x[:, :, 0] * valid).sum(axis=1) > 0).astype(np.int64)
    return x, y, lengths


def model(ml, cell, t, f, h):
    m = ml.Sequential(input_shape=(t, f), numeric_mode="identical")
    m = getattr(m, cell)(h).narrow(0, t - 1, 1).flatten().linear(2)
    return m.compile(loss="cross_entropy", optimizer="adam", lr=1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="full", choices=("full", "small"))
    ap.add_argument("--cells", default="rnn,gru,lstm")
    ap.add_argument("--label", default="lengths")
    args = ap.parse_args()
    import mojolearn as ml
    probe()
    n, t, f, h = (65536, 16, 8, 64) if args.shape == "full" else (2048, 16, 8, 16)
    x, y, varied = data(n, t, f, 0)
    xh, yh, varied_h = data(max(n // 8, 256), t, f, 1)
    full = np.full(n, t, dtype=np.int64)
    full_h = np.full(len(xh), t, dtype=np.int64)
    for cell in args.cells.split(","):  # glue: one row group per cell
        ref = None
        for arm, lens, lens_h in (("none", None, None), ("full", full, full_h), ("varied", varied, varied_h)):
            m = model(ml, cell, t, f, h)
            fit_ms, m = clock(lambda: m.fit(x, y, epochs=1, batch_size=512, shuffle=True, seed=0, lengths=lens))
            pred_ms, proba = clock(lambda: m.predict_proba(xh, lengths=lens_h))
            words = np.concatenate([np.asarray(m.params_).ravel(), np.asarray(proba).ravel()]).view(np.uint32)
            if arm == "none":
                ref = words
            same = "n/a" if arm == "varied" else str(bool(np.array_equal(words, ref)))
            acc = float((np.asarray(proba).argmax(axis=1) == yh).mean())
            print(f"LENGTHS label={args.label} cell={cell} arm={arm} N={n} T={t} F={f} H={h} "
                  f"fit_ms={fit_ms:.3f} predict_ms={pred_ms:.3f} same_words={same} heldout_acc={acc:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
