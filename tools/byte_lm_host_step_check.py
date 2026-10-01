#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The byte LM host training step with its rows over host tasks against the
serial oracle calls: bits and wall.

lane/neural-pass6 (2026-09-30). `LanguageModelHostTrainer.train_step` runs the
loss's rows and the optimizer's elements over host tasks
(`training/loss_host_rows.mojo`, `training/optimizer_host_rows.mojo`) and the
serial oracle calls for both under `MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=0`. The
two must agree bit for bit on every step's loss, gradient, parameters and
both moments, because the row paths run the oracle's statements on the
oracle's operands in the oracle's per-row order, and every cross-row fold is
the same gemm v1 call. This tool runs both from one initialization over the
same batches and compares every byte, then reports the median step wall of
each and the ratio.

    python tools/byte_lm_host_step_check.py                 # the board shape, 4 steps
    python tools/byte_lm_host_step_check.py --small --steps 16
    python tools/byte_lm_host_step_check.py --layers 2 --json out.json

The board shape is `lm-host-train-step`'s (batch 1, length 512, d_model 384,
6 heads, head_dim 64, intermediate 1024, 8 layers, vocab 8192). What this
tool cannot see, the block oracles' own threaded stages (they have no serial
switch), `tools/byte_lm_cpu_train_gate.py cpu` sees against the recorded
three-vendor bytes.

Prints `BYTE_LM_HOST_STEP_CHECK PASS` or `BYTE_LM_HOST_STEP_CHECK FAIL <what>`
and exits 1 on FAIL. A FAIL is a bug in a row path, never a result.
"""
import argparse
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = "MOJOLEARN_BYTE_LM_HOST_STEP_ROWS"
ADAMW = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
BOARD = dict(batch=1, length=512, d_model=384, n_heads=6, n_kv=6, head_dim=64,
             intermediate=1024, n_layers=8, vocab_size=8192)
SMALL = dict(batch=2, length=64, d_model=64, n_heads=4, n_kv=2, head_dim=16,
             intermediate=128, n_layers=2, vocab_size=256)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _run(rows_on, shape, init, m0, v0, batches):
    """Every step's loss bits and state digests under one setting, and the walls."""
    from mojolearn._byte_lm_host import LanguageModelHostTrainer
    os.environ[ENV] = "1" if rows_on else "0"
    trainer = LanguageModelHostTrainer(init.copy(), m=m0.copy(), v=v0.copy(), shape=shape, **ADAMW)
    steps, walls = [], []
    for ids in batches:
        t0 = time.perf_counter()
        bits = trainer.train_step(ids)
        walls.append(time.perf_counter() - t0)
        steps.append(dict(loss_bits=int(bits), grad=_sha(trainer.gradient_),
                          param=_sha(trainer.parameters_), m=_sha(trainer.m_), v=_sha(trainer.v_)))
    return steps, walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--small", action="store_true", help="the small LM shape instead of the board's")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "python"))
    from mojolearn import ByteLanguageModelConfig

    fields = dict(SMALL if args.small else BOARD)
    if args.layers:
        fields["n_layers"] = args.layers
    shape = ByteLanguageModelConfig(**fields)
    n = shape.n_total
    rng = np.random.default_rng(args.seed)
    init = (rng.standard_normal(n, dtype=np.float32) * np.float32(0.02)).astype("<f4")
    m0 = (rng.standard_normal(n, dtype=np.float32) * np.float32(1e-3)).astype("<f4")
    v0 = np.abs(rng.standard_normal(n, dtype=np.float32) * np.float32(1e-4)).astype("<f4")
    batches = [rng.integers(0, shape.vocab_size, size=(shape.batch, shape.length + 1), dtype=np.int32)
               for _ in range(args.steps)]

    before = os.environ.get(ENV)
    try:
        rows, rows_walls = _run(True, shape, init, m0, v0, batches)
        serial, serial_walls = _run(False, shape, init, m0, v0, batches)
    finally:
        if before is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = before

    fails = []
    for k, (a, b) in enumerate(zip(rows, serial)):
        for key in a:
            if a[key] != b[key]:
                fails.append(f"step {k} {key}: rows {a[key]} serial {b[key]}")
    rows_med = statistics.median(rows_walls)
    serial_med = statistics.median(serial_walls)
    report = dict(shape=fields, steps=args.steps, rows_ms=rows_med * 1e3, serial_ms=serial_med * 1e3,
                  ratio=serial_med / rows_med if rows_med else None,
                  losses=[struct.unpack("<f", struct.pack("<I", s["loss_bits"]))[0] for s in rows],
                  fails=fails)
    print(f"rows {rows_med * 1e3:.1f} ms/step, serial {serial_med * 1e3:.1f} ms/step, "
          f"ratio {report['ratio']:.2f}x, losses {report['losses']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    if fails:
        print("BYTE_LM_HOST_STEP_CHECK FAIL " + "; ".join(fails[:4]))
        return 1
    print("BYTE_LM_HOST_STEP_CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
