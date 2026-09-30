#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimizer with resident moments against the per-call step: bits and wall.

lane/neural-pass4 (2026-09-30). `mojolearn.AdamW` (and `Adam`, `SGD`) keep
`exp_avg` and `exp_avg_sq` on the device between steps where the training
binding offers `optimizer_resident_*`; `resident=False` (or
`MOJOLEARN_OPTIMIZER_RESIDENT=0`) keeps the per-call `optimizer_step`. Both
run the same `identical_optimizer_step` on the same buffers, so every step's
parameters, moments and flags must agree byte for byte. This tool runs both
from one initialization over the same gradients, with a `state_dict` /
`load_state_dict` round trip and a clipped step in the middle, compares
every byte after every step, and reports the median step wall of each.

    python tools/optimizer_resident_check.py                    # ~5.7 M floats (the Samba board shape's registry), 32 steps
    python tools/optimizer_resident_check.py --floats 20000000 --steps 16 --json out.json
    python tools/optimizer_resident_check.py --kind sgd

Prints `OPTIMIZER_RESIDENT_CHECK PASS` or `... FAIL <what>` and exits 1 on
FAIL; SKIP when the binding has no resident entries. A FAIL is a bug in the
resident path, never a result.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

#: tensor shapes summing to about 5.7 M floats: the Samba board registry's
#: order of magnitude (two Mamba-3 blocks, two attention blocks, d 384)
DEFAULT_SHAPES = [(1860, 384), (384, 768), (12,), (128,), (128,), (12, 128),
                  (1152, 384), (384, 384), (1024, 384), (384, 1024), (1024, 384),
                  (1860, 384), (384, 768), (1152, 384), (384, 384), (1024, 384),
                  (384, 1024), (1024, 384), (256, 384), (384,)]


def _shapes(floats):
    if floats is None:
        return DEFAULT_SHAPES
    shapes, left = [], int(floats)
    while left > 0:
        n = min(left, 1 << 20)
        shapes.append((n,))
        left -= n
    return shapes


def _make(ml, kind, params, resident):
    kw = dict(lr=0.03, weight_decay=0.01, resident=resident)
    if kind == "adamw":
        return ml.AdamW(params, betas=(0.9, 0.999), eps=1e-8, **kw)
    if kind == "adam":
        return ml.Adam(params, betas=(0.9, 0.999), eps=1e-8, **kw)
    return ml.SGD(params, momentum=0.9, **kw)


def _run(ml, kind, shapes, grads, resident, clip_at, roundtrip_at):
    rng = np.random.default_rng(1)
    params = [np.ascontiguousarray((rng.standard_normal(s) * 0.1).astype(np.float32)) for s in shapes]
    opt = _make(ml, kind, params, resident)
    records, walls = [], []
    for k, g in enumerate(grads):
        if k == roundtrip_at:
            st = opt.state_dict()
            opt = _make(ml, kind, params, resident)
            opt.load_state_dict(st)
        max_norm = 1.0 if k == clip_at else None
        t0 = time.perf_counter()
        opt.step([np.array(x, dtype=np.float32, copy=True) for x in g], max_norm=max_norm)  # a copy: the clip scales its gradients in place
        walls.append(time.perf_counter() - t0)
        st = opt.state_dict()
        records.append({
            "params": b"".join(p.tobytes() for p in params),
            "m": np.asarray(st["exp_avg"]).tobytes(),
            "v": np.asarray(st["exp_avg_sq"]).tobytes(),
            "flags": np.asarray(st["buf_initialized"]).tobytes(),
            "t": int(st["t"]).to_bytes(8, "little"),
            "norm": (b"" if opt.total_norm_ is None else np.float64(opt.total_norm_).tobytes()),
        })
    return records, walls, bool(opt.resident_)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--floats", type=int, default=None, help="registry size (default: the Samba-like shapes)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--kind", choices=("adamw", "adam", "sgd"), default="adamw")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import mojolearn as ml
    from mojolearn import _training_impl
    binding = _training_impl._load(None)
    if not _training_impl._resident_enabled(binding, None):
        print("OPTIMIZER_RESIDENT_CHECK SKIP: the training binding has no optimizer_resident entries (rebuild it) or the environment turns them off")
        return 0

    shapes = _shapes(args.floats)
    n = sum(int(np.prod(s)) for s in shapes)
    rng = np.random.default_rng(args.seed)
    grads = [[(rng.standard_normal(s) * 0.01).astype(np.float32) for s in shapes] for _ in range(args.steps)]
    clip_at = args.steps // 2
    roundtrip_at = args.steps // 3

    host, host_walls, host_res = _run(ml, args.kind, shapes, grads, False, clip_at, roundtrip_at)
    res, res_walls, res_res = _run(ml, args.kind, shapes, grads, None, clip_at, roundtrip_at)

    failures = []
    if host_res:
        failures.append("the resident=False optimizer reports resident_")
    if not res_res:
        failures.append("the default optimizer did not go resident")
    for k, (a, b) in enumerate(zip(host, res)):
        for key in a:
            if a[key] != b[key]:
                failures.append("step %d %s" % (k + 1, key))

    med_host = statistics.median(host_walls[1:] or host_walls) * 1e3
    med_res = statistics.median(res_walls[1:] or res_walls) * 1e3
    print("optimizer_resident_check kind %s floats %d tensors %d steps %d mode %s"
          % (args.kind, n, len(shapes), args.steps, ml.numeric_mode()))
    print("  per-call step (moments on the host): median %.3f ms (first %.3f ms)" % (med_host, host_walls[0] * 1e3))
    print("  resident moments:                    median %.3f ms (first %.3f ms)  ratio %.2fx"
          % (med_res, res_walls[0] * 1e3, med_host / med_res if med_res > 0 else 0.0))
    out = dict(kind=args.kind, floats=n, steps=args.steps, seed=args.seed,
               host_median_ms=med_host, resident_median_ms=med_res,
               host_walls_ms=[t * 1e3 for t in host_walls],
               resident_walls_ms=[t * 1e3 for t in res_walls], failures=failures)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if failures:
        print("OPTIMIZER_RESIDENT_CHECK FAIL " + "; ".join(failures[:12]) + (" ..." if len(failures) > 12 else ""))
        return 1
    print("OPTIMIZER_RESIDENT_CHECK PASS (%d steps, every byte equal)" % args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
