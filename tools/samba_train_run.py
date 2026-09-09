#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Train a small Samba-shaped stack for N steps on Tiny Shakespeare bytes
and record the per-step loss, the learning rate, the wall time per step
and the checkpoint sha256. IDENTICAL mode only. Two runs of this tool with
the same arguments on any vendor must print the same checkpoint sha.

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/samba_train_run.py \\
        --output /tmp/run1 --steps 64 --layers mamba3,mamba3 --d-model 64

The batch at step s (zero-based) is row b = bytes[(s*B + b)*L : +L+1] of the
corpus, a pure function of (s, b), so the data schedule needs no RNG.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

CORPUS = ROOT / "training/corpus/tinyshakespeare/input.txt"


def batch(corpus, step, b, l):
    ids = np.empty((b, l + 1), dtype=np.int32)
    for row in range(b):
        start = ((step * b + row) * l) % (len(corpus) - l - 1)
        ids[row] = np.frombuffer(corpus[start:start + l + 1], dtype=np.uint8)
    return ids[:, :-1], ids[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", default="mamba3,mamba3")
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--intermediate", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--accumulation-steps", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--untied", action="store_true")
    args = ap.parse_args()

    if os.environ.get("MOJOLEARN_NUMERIC_MODE", "").lower() != "identical":
        sys.exit("samba_train_run: MOJOLEARN_NUMERIC_MODE=identical is required")
    import mojolearn
    from mojolearn.training import Generator, WarmupCosineLR, SambaConfig, SambaStack

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    corpus = CORPUS.read_bytes()
    cfg = SambaConfig(vocab=256, d_model=args.d_model, layers=args.layers.split(","),
                      n_heads=args.n_heads, intermediate=args.intermediate,
                      tie_embeddings=not args.untied, dropout=args.dropout)
    stack = SambaStack(cfg, generator=Generator(args.seed), lr=args.lr,
                       lr_schedule=WarmupCosineLR(args.lr, args.warmup, args.steps, args.min_lr),
                       max_norm=args.max_norm, accumulation_steps=args.accumulation_steps)
    init_sha = hashlib.sha256(stack.flat.tobytes()).hexdigest()
    rows = []
    for s in range(args.steps):
        inputs, targets = batch(corpus, s, args.batch, args.seq)
        t0 = time.perf_counter()
        r = stack.train_step(inputs, targets)
        dt = time.perf_counter() - t0
        rows.append({"step": r["step"], "loss": r["loss"], "lr": r["lr"],
                     "total_norm": r["total_norm"], "seconds": dt})
        print("step %3d loss %.6f lr %.3e norm %s  %.1f ms"
              % (r["step"], r["loss"], r["lr"], r["total_norm"], dt * 1e3), flush=True)
    ck = out / "checkpoint.json"
    ck_sha = stack.save_checkpoint(ck)
    timed = [r["seconds"] for r in rows[8:]] or [r["seconds"] for r in rows]
    summary = {
        "profile": mojolearn.training.numeric_mode_used(),
        "vendor": mojolearn.training.vendor_used(),
        "config": cfg.to_dict(), "steps": args.steps, "batch": args.batch,
        "seq": args.seq, "seed": args.seed, "accumulation_steps": args.accumulation_steps,
        "n_parameters": int(stack.n_total), "init_sha256": init_sha,
        "checkpoint_sha256": ck_sha, "final_parameters_sha256":
            hashlib.sha256(stack.flat.tobytes()).hexdigest(),
        "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
        "ms_per_step_median": 1e3 * float(np.median(timed)),
        "ms_per_step_mean": 1e3 * float(np.mean(timed)),
        "timed_steps": "9..%d" % args.steps,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("checkpoint sha256 %s" % ck_sha)
    print("parameters %d  median %.1f ms/step  loss %.4f -> %.4f"
          % (stack.n_total, summary["ms_per_step_median"], rows[0]["loss"], rows[-1]["loss"]))


if __name__ == "__main__":
    main()
