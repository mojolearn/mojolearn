#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sequence lane's optimizers with resident state against the per-call
step, and one digest per case for the cross-column comparison.

lane gap-optimizers (2026-10-02). RMSprop, Adagrad, Lion, Adamax, NAdam and
LAMB keep their state on the device binding's context between steps
(`sequence/opt_resident.mojo`); `MOJOLEARN_OPTIMIZER_RESIDENT=0` keeps the
per-call entries. Both run the same element statements (LAMB: the same
`lamb_core`) on the same values, so the params and the state must agree
byte for byte after every step. A `state_dict` / `load_state_dict` round
trip runs in the middle (the resident arm downloads, then re-uploads).

    python tools/seq_optimizer_resident_check.py              # small multi-tensor shapes
    python tools/seq_optimizer_resident_check.py --big        # plus one 16,777,216-value tensor (the board lane)
    MOJOLEARN_VENDOR=cpu python tools/seq_optimizer_resident_check.py   # the host column's digests

Prints one `SEQOPT_DIGEST <case> <sha256-16>` line per case (compare them
across NVIDIA, AMD, Metal and the host column), then
`SEQ_OPTIMIZER_RESIDENT_CHECK PASS` or `... FAIL <what>` (exit 1). On a
binding without the resident entries both arms are the per-call step.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

STEPS = 6

CASES = [
    ("rmsprop", "RMSprop", dict(lr=1e-2)),
    ("rmsprop-centered", "RMSprop", dict(lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)),
    ("adagrad", "Adagrad", dict(lr=1e-2, initial_accumulator_value=0.1, lr_decay=1e-3)),
    ("lion", "Lion", dict(lr=1e-4, weight_decay=0.1)),
    ("adamax", "Adamax", dict(lr=2e-3, weight_decay=0.05)),
    ("nadam", "NAdam", dict(lr=2e-3)),
    ("nadam-decoupled", "NAdam", dict(lr=2e-3, weight_decay=0.01, decoupled_weight_decay=True)),
    ("lamb", "LAMB", dict(lr=1e-2)),
    ("lamb-adapt", "LAMB", dict(lr=1e-2, weight_decay=0.0, always_adapt=True, trust_clip=True)),
    ("lamb-noclip", "LAMB", dict(lr=1e-2, max_grad_norm=None, bias_correction=False)),
]

SMALL = [(3000,), (64, 96), (9000,), (5,), (4096,), (4097,)]


def _run(ml, cls, hyper, shapes, resident):
    os.environ["MOJOLEARN_OPTIMIZER_RESIDENT"] = "1" if resident else "0"
    rng = np.random.default_rng(7)
    ps = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    gseq = [[(rng.standard_normal(s) * 0.5).astype(np.float32) for s in shapes] for _ in range(STEPS)]
    opt = getattr(ml, cls)(ps, **hyper)
    snaps = []
    for k, gs in enumerate(gseq):
        opt.step(gs)
        if k == STEPS // 2:
            sd = opt.state_dict()
            opt.load_state_dict(sd)
        snaps.append(b"".join(p.tobytes() for p in ps) + b"".join(s.tobytes() for s in opt.state))
    return snaps, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="add one 16,777,216-value tensor per optimizer")
    a = ap.parse_args()
    import mojolearn as ml
    shape_sets = [("small", SMALL)] + ([("big", [(1 << 24,)])] if a.big else [])
    fails = []
    for tag, shapes in shape_sets:
        for name, cls, hyper in CASES:
            on, opt_on = _run(ml, cls, hyper, shapes, True)
            off, _ = _run(ml, cls, hyper, shapes, False)
            case = "%s/%s" % (name, tag)
            for k, (x, y) in enumerate(zip(on, off)):
                if x != y:
                    fails.append("%s step %d: resident != per-call" % (case, k + 1))
                    break
            print("SEQOPT_DIGEST %s %s resident=%s" % (case, hashlib.sha256(on[-1]).hexdigest()[:16],
                                                       getattr(opt_on, "resident_", False)), flush=True)
    if fails:
        print("SEQ_OPTIMIZER_RESIDENT_CHECK FAIL " + "; ".join(fails[:5]))
        return 1
    print("SEQ_OPTIMIZER_RESIDENT_CHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
