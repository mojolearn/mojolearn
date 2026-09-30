#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The small MLP's fused step against its per-operation calls: bits and wall.

lane/apple-mlp-fused (2026-09-30). `SmallMLPTrainer.train_step` runs as ONE
binding call (`mlp_train_step`) where the binding offers it, and as the
twelve calls it was under `MOJOLEARN_MLP_FUSED=0`. The two must agree bit
for bit on every step's loss, logits, gradients, input gradient and on the
state (weights, m, v, flags, step) after every step, because the fused entry
runs the same kernels on the same operands in the same order. This tool runs
both from one initialization over the same batches and compares every byte,
then reports the median step wall of each and the ratio.

    python tools/mlp_step_check.py                    # 64 steps of 256 rows
    python tools/mlp_step_check.py --steps 128 --rows 32 --json out.json

Prints `MLP_STEP_CHECK PASS` or `MLP_STEP_CHECK FAIL <what>` and exits 1 on
FAIL. A FAIL is a bug in the fused entry, never a result. Requires the
process-selected IDENTICAL (or FAST) mode and the GPU training binding; on
a binding without `mlp_train_step` it reports SKIP.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

NAMES = ("weight1", "bias1", "weight2", "bias2")
SHAPES = ((16, 8), (16,), (3, 16), (3,))
CONFIG = dict(lr=0.03, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def _bytes(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _state_bytes(state):
    parts = [_bytes(state["weights"][n]) for n in NAMES]
    opt = state["optimizer"]
    parts += [_bytes(opt["m"]), _bytes(opt["v"]), _bytes(opt["flags"]),
              int(opt["step"]).to_bytes(8, "little")]
    return b"".join(parts)


def _run(ml, w, X, y, steps, fused, input_grad_every):
    os.environ["MOJOLEARN_MLP_FUSED"] = "1" if fused else "0"
    trainer = ml.SmallMLPTrainer(*[a.copy() for a in w],
                                 data_schedule={"fixture": "tools/mlp_step_check.py"}, **CONFIG)
    records, walls = [], []
    for k in range(steps):
        want = input_grad_every > 0 and (k % input_grad_every) == 0
        t0 = time.perf_counter()
        res = trainer.train_step(X[k], y[k], return_input_grad=want)
        walls.append(time.perf_counter() - t0)
        rec = {"loss": np.float64(res["loss"]).tobytes(), "logits": _bytes(res["logits"])}
        for n in NAMES:
            rec["grad_" + n] = _bytes(res["gradients"][n])
        rec["input_grad"] = _bytes(res["input_grad"]) if want else b""
        rec["state"] = _state_bytes(trainer.state_dict())
        records.append(rec)
    # the other two entry points, once each, on the final state
    loss, grads = trainer.loss_and_grads(X[0], y[0])
    extra = {"lg_loss": np.float64(loss).tobytes(),
             "lg_grads": b"".join(_bytes(g) for g in grads),
             "predict": _bytes(trainer.predict_logits(X[1]))}
    return records, extra, walls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--input-grad-every", type=int, default=8,
                    help="ask for input_grad on every k-th step (0: never)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import mojolearn as ml
    mode = ml.numeric_mode()
    from mojolearn import _mlp_impl
    binding = _mlp_impl.SmallMLPTrainer._binding()
    if not callable(getattr(binding, "mlp_train_step", None)):
        print("MLP_STEP_CHECK SKIP: the training binding has no mlp_train_step (rebuild it)")
        return 0

    rng = np.random.default_rng(args.seed)
    w = [np.ascontiguousarray((rng.standard_normal(s) * 0.5).astype(np.float32)) for s in SHAPES]
    X = np.ascontiguousarray(rng.standard_normal((args.steps, args.rows, 8)).astype(np.float32))
    y = np.ascontiguousarray(rng.integers(0, 3, size=(args.steps, args.rows)).astype(np.int32))

    per_op, per_extra, per_walls = _run(ml, w, X, y, args.steps, False, args.input_grad_every)
    fused, fused_extra, fused_walls = _run(ml, w, X, y, args.steps, True, args.input_grad_every)

    failures = []
    for k, (a, b) in enumerate(zip(per_op, fused)):
        for key in a:
            if a[key] != b[key]:
                failures.append("step %d %s" % (k + 1, key))
    for key in per_extra:
        if per_extra[key] != fused_extra[key]:
            failures.append(key)

    losses = [np.frombuffer(r["loss"], dtype=np.float64)[0] for r in fused]
    digest = hashlib.sha256(b"".join(r["loss"] for r in fused)).hexdigest()[:16]
    med_op = statistics.median(per_walls[1:] or per_walls) * 1e3
    med_fused = statistics.median(fused_walls[1:] or fused_walls) * 1e3
    print("mlp_step_check mode %s rows %d steps %d" % (mode, args.rows, args.steps))
    print("  per-operation step: median %.3f ms (first %.3f ms)" % (med_op, per_walls[0] * 1e3))
    print("  fused step:         median %.3f ms (first %.3f ms)  ratio %.2fx"
          % (med_fused, fused_walls[0] * 1e3, med_op / med_fused if med_fused > 0 else 0.0))
    print("  losses: first %.6f last %.6f digest %s" % (losses[0], losses[-1], digest))
    out = dict(mode=mode, rows=args.rows, steps=args.steps, seed=args.seed,
               per_op_median_ms=med_op, fused_median_ms=med_fused,
               per_op_walls_ms=[t * 1e3 for t in per_walls],
               fused_walls_ms=[t * 1e3 for t in fused_walls],
               loss_digest=digest, losses=losses, failures=failures)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if failures:
        print("MLP_STEP_CHECK FAIL " + "; ".join(failures[:12])
              + (" ..." if len(failures) > 12 else ""))
        return 1
    print("MLP_STEP_CHECK PASS (%d steps, every byte equal)" % args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
