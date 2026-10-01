#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which block stages dominate the byte LM host training step (lane
neural-pass8), the host twin of tools/mamba3_backward_timing.py.

The two block oracles print a wall per stage under
MOJOLEARN_HOST_BLOCK_TIMING=1 (`timing hblk.<stage> <ms> ms`; see
`host_tick` in core/host_lanes.mojo and the ticks in
transformer/checks/transformer_oracle.mojo and transformer_backward_oracle.mojo).
This runs `LanguageModelHostTrainer.train_step` at the board's
lm-host-train-step shape in a child process with that env set, parses the
lines, and prints the stages summed over the layers of one step, sorted by
time with their share, plus the step's wall. The ticks read the clock between
the oracles' own calls; the stage bits are the same with the env set or not.

    python tools/byte_lm_host_block_timing.py                 # the board shape, 2 steps
    python tools/byte_lm_host_block_timing.py --layers 2 --steps 3 --small

The first step warms; the table is the last step's.
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = dict(batch=1, length=512, d_model=384, n_heads=6, n_kv=6, head_dim=64,
             intermediate=1024, n_layers=8, vocab_size=8192)
SMALL = dict(batch=2, length=64, d_model=64, n_heads=4, n_kv=2, head_dim=16,
             intermediate=128, n_layers=2, vocab_size=256)
ADAMW = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def child(args):
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from mojolearn import ByteLanguageModelConfig
    from mojolearn._byte_lm_host import LanguageModelHostTrainer
    fields = dict(SMALL if args.small else BOARD)
    if args.layers:
        fields["n_layers"] = args.layers
    shape = ByteLanguageModelConfig(**fields)
    n = shape.n_total
    rng = np.random.default_rng(7)
    init = (rng.standard_normal(n, dtype=np.float32) * np.float32(0.02)).astype("<f4")
    trainer = LanguageModelHostTrainer(init, shape=shape, **ADAMW)
    for k in range(args.steps):
        ids = rng.integers(0, shape.vocab_size, size=(shape.batch, shape.length + 1), dtype=np.int32)
        print(f"step {k} begin", flush=True)
        t0 = time.perf_counter()
        trainer.train_step(ids)
        print(f"step {k} wall {(time.perf_counter() - t0) * 1e3:.3f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(args)
        return 0
    env = dict(os.environ, MOJOLEARN_HOST_BLOCK_TIMING="1")
    cmd = [sys.executable, __file__, "--child", "--steps", str(args.steps)]
    if args.layers:
        cmd += ["--layers", str(args.layers)]
    if args.small:
        cmd += ["--small"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-4000:])
        return out.returncode
    steps = []
    cur = None
    wall = None
    for line in out.stdout.splitlines():
        if line.startswith("step ") and line.endswith("begin"):
            cur = collections.OrderedDict()
        elif line.startswith("step ") and " wall " in line:
            wall = float(line.split(" wall ")[1].split()[0])
            steps.append((cur, wall))
            cur = None
        else:
            m = re.match(r"timing (hblk\.\S+) ([0-9.]+) ms", line)
            if m and cur is not None:
                cur[m.group(1)] = cur.get(m.group(1), 0.0) + float(m.group(2))
    if not steps:
        print("no timing lines; is the host binding built from a tree with host_tick?")
        return 2
    stages, wall = steps[-1]
    total = sum(stages.values())
    print(f"lm-host-train-step block stages, last of {len(steps)} steps, wall {wall:.1f} ms, "
          f"ticked {total:.1f} ms ({100 * total / wall:.0f}% of the wall)")
    for name, ms in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {name:36s} {ms:9.2f} ms  {100 * ms / total:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
