#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Multi-GPU language model training across device counts and vendors, small.

    python3 tools/par_lm_xvendor.py run --out A.json --devices 0,1            # one box
    python3 tools/par_lm_xvendor.py run --out B.json --devices 0 --resume A.handoff.npz
    python3 tools/par_lm_xvendor.py compare A.json B.json C.json ...           # at home

EVIDENCE, NOT A VERIFIER LANE. Nothing here is imported by `mojolearn verify`.

HOW THE MULTI-GPU PATH CLAIMS BITWISE IDENTITY
(`python/mojolearn/parallel_training.py`, `training/byte_lm_parallel.mojo`).
Each step's batch is cut into K LOGICAL shards, K fixed by the caller and
independent of how many GPUs exist. Each shard's gradient is a mean cross
entropy computed by the same IDENTICAL kernels a one-GPU step uses, so a
shard's gradient bits do not depend on which GPU, or which vendor, computed
it. The shards are then combined by an ORDERED fold, never an all-reduce:
total = g0, then total = ftz(fma(1, ftz(total), ftz(g_k))) for k = 1..K-1, in
shard order. Float addition is not associative, and a collective chooses its
own tree, so the fixed left fold is the whole trick. The AdamW update is
elementwise and the same code on every device. So 1 GPU doing K shards in a
loop, 2 or 4 GPUs sharing them, and in principle GPUs of different vendors
sharing them, must all produce the same bits. Changing K changes the bits,
and that is expected: K is part of the recipe, the device count is not.

WHAT ONE `run` RECORDS, EVERY STEP, ON ONE BOX
  * the trainer under test (`--devices`, pooled optimizer) and a replicated
    one-device reference (`pool_optimizer=False`): losses, full state hash
    (parameters, m, v, flags) and summed gradient hash, held equal ON THE BOX;
  * every shard's gradient ALONE, from the pre-step state, by a one-shard
    one-device trainer, and its loss, held equal to the shard's loss inside
    the K-shard step;
  * the host's own ordered fold of those shard gradients, emulating the
    device primitive exactly, held equal to the device's summed gradient.
  With `--grads`, the shard gradients themselves go to an .npz, so `compare`
  can build MIXED-VENDOR steps: shard k's gradient taken from vendor V(k),
  every assignment, folded, and checked against every vendor's device sum.
  At `--handoff-at H` the state after step H is written as `<out>.handoff.npz`;
  `run --resume` on another box continues from those exact bytes.

`compare` exits 1 on any disagreement AND exits 1 if nothing was compared (a
check that cannot fail is not a check).
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "mojolearn.par-lm-xvendor.v1"


def _shake(label, n_bytes):
    return hashlib.shake_256(label.encode()).digest(n_bytes)


def _h(*arrays):
    d = hashlib.sha256()
    for a in arrays:
        d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()[:16]


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()


# ------------------------------------------------------------ the device fold, on the host
_EXP, _MAN, _SIGN = np.uint32(0x7F800000), np.uint32(0x007FFFFF), np.uint32(0x80000000)


def ftz(x):
    """checks/numerics.mojo `ftz`: a subnormal becomes a zero of the same sign."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    sub = ((b & _EXP) == 0) & ((b & _MAN) != 0)
    return np.where(sub, b & _SIGN, b).view(np.float32)


def ordered_fold(grads):
    """training/byte_lm_parallel.mojo: copy g[0], then
    total = ftz(fma(1, ftz(total), ftz(g[k]))). fma(1, a, b) is a + b rounded
    once, which is what an IEEE float32 add is."""
    total = np.array(grads[0], dtype=np.float32, copy=True)
    with np.errstate(over="ignore"):
        for g in grads[1:]:
            total = ftz(ftz(total) + ftz(np.asarray(g, dtype=np.float32)))
    return total


# ------------------------------------------------------------ the fixed problem
def build_problem(ml, args):
    shape = ml.ByteLanguageModelConfig(batch=args.batch, length=args.length, d_model=args.d_model,
                                       n_heads=args.heads, n_kv=args.kv, head_dim=args.d_model // args.heads,
                                       intermediate=args.ff, n_layers=args.layers, vocab_size=args.vocab)
    named = {}
    for name, shp in zip(shape.parameter_names, shape.parameter_shapes):
        n = int(np.prod(shp))
        if name.endswith("norm1_w") or name.endswith("norm2_w"):
            named[name] = np.ones(shp, dtype=np.float32)
        else:
            # dyadic values in [-1/8, 1/8): exact in float32, no RNG library in the recipe
            k = np.frombuffer(_shake(f"par-lm-xvendor:{args.seed}:{name}", 2 * n), dtype="<u2") % 2048
            named[name] = ((k.astype(np.float32) - 1024.0) / 8192.0).reshape(shp)
    count = args.steps * args.shards * args.batch * (args.length + 1)
    ids = (np.frombuffer(_shake(f"par-lm-xvendor:{args.seed}:ids", 4 * count), dtype="<u4") % args.vocab)
    ids = ids.astype(np.int32).reshape(args.steps, args.shards, args.batch, args.length + 1)
    seed = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "par-lm-xvendor", "order": "sequential"},
                                            shape=shape)
    return shape, seed.state_dict(), ids


def state_hash(state):
    return _h(*(np.asarray(state[k]) for k in ("parameters", "m", "v", "flags")))


def save_state(path, state):
    arrays = {k: np.asarray(state[k]) for k in ("parameters", "m", "v", "flags")}
    meta = {k: v for k, v in state.items() if k not in arrays}
    np.savez(path, meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8), **arrays)


def load_state(path):
    z = np.load(path)
    state = json.loads(bytes(z["meta"]).decode())
    for k in ("parameters", "m", "v", "flags"):
        state[k] = np.ascontiguousarray(z[k])
    return state


def _box():
    info = dict(host=platform.node(), machine=platform.machine(), python=platform.python_version())
    for cmd in (["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname", "--showuniqueid"]):
        try:
            info[cmd[0]] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


# ------------------------------------------------------------ run
def cmd_run(args):
    import mojolearn as ml
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par

    devices = tuple(int(x) for x in args.devices.split(","))
    shape, state0, ids = build_problem(ml, args)
    start = 0
    resumed_from = None
    if args.resume:
        state0 = load_state(args.resume)
        start = int(state0["completed_steps"])
        resumed_from = dict(path=os.path.basename(args.resume), step=start, state=state_hash(state0))
    record = dict(schema=SCHEMA, label=args.label, vendor=ml.vendor(), devices=list(devices),
                  logical_shards=args.shards, steps=args.steps, seed=args.seed, box=_box(),
                  commit=_commit(), shape=shape.to_dict(), start_state=state_hash(state0),
                  resumed_from=resumed_from, rows=[], failures=[])
    grads_out = {}

    def fail(msg):
        record["failures"].append(msg)
        print("FAIL", msg, flush=True)

    t0 = time.time()
    with Par(state0, devices=devices, logical_shards=args.shards) as T, \
         Par(state0, devices=devices[:1], logical_shards=args.shards, pool_optimizer=False) as R:
        pre = state0
        for s in range(start, args.steps):
            shards = [np.ascontiguousarray(ids[s, k]) for k in range(args.shards)]
            shard_g, shard_loss = [], []
            for k in range(args.shards):
                with Par(pre, devices=devices[:1], logical_shards=1, pool_optimizer=False) as one:
                    shard_loss.append(one.train_step([shards[k]])["losses"][0])
                    shard_g.append(np.array(one.export_gradients(), dtype=np.float32, copy=True))
            a, b = T.train_step(shards), R.train_step(shards)
            st, sr = T.state_dict(), R.state_dict()
            gt, gr = np.asarray(T.export_gradients()), np.asarray(R.export_gradients())
            host = ordered_fold(shard_g)
            row = dict(step=s + 1, losses=_bits(a["losses"]), state=state_hash(st), grad=_h(gt),
                       shard_losses=_bits(shard_loss), shard_grads=[_h(g) for g in shard_g], host_fold=_h(host))
            record["rows"].append(row)
            if _bits(a["losses"]) != _bits(b["losses"]):
                fail(f"step {s + 1}: losses differ from the one-device replica")
            if state_hash(sr) != row["state"]:
                fail(f"step {s + 1}: state differs from the one-device replica")
            if _h(gr) != row["grad"]:
                fail(f"step {s + 1}: summed gradient differs from the one-device replica")
            if row["shard_losses"] != row["losses"]:
                fail(f"step {s + 1}: a shard's loss alone differs from its loss inside the step")
            if row["host_fold"] != row["grad"]:
                fail(f"step {s + 1}: host ordered fold of shard gradients differs from the device sum")
            if args.grads:
                for k, g in enumerate(shard_g):
                    grads_out[f"s{s + 1}_k{k}"] = g
                grads_out[f"s{s + 1}_sum"] = np.array(gt, copy=True)
            if args.handoff_at and s + 1 == args.handoff_at and not args.resume:
                save_state(args.out.replace(".json", "") + ".handoff.npz", st)
                row["handoff_written"] = True
            pre = st
            print(f"step {s + 1} state {row['state']} grad {row['grad']} losses {a['losses']}", flush=True)
    record["seconds"] = round(time.time() - t0, 2)
    record["verdict"] = "PASS" if not record["failures"] and record["rows"] else "FAIL"
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    if args.grads:
        np.savez_compressed(args.grads, **grads_out)
    print(record["verdict"], args.out)
    return 0 if record["verdict"] == "PASS" else 1


def _commit():
    for p in ("commit.txt",):
        if Path(p).is_file():
            return Path(p).read_text().split()[0]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


# ------------------------------------------------------------ compare
def cmd_compare(args):
    recs = [json.loads(Path(p).read_text()) for p in args.records]
    names = [f"{r['label'] or r['vendor']}[{','.join(map(str, r['devices']))}]" for r in recs]
    bad, compared = [], 0
    for r, n in zip(recs, names):
        if r["verdict"] != "PASS":
            bad.append(f"{n}: on-box verdict {r['verdict']}: {r['failures'][:3]}")
    recipe = lambda r: (r["logical_shards"], r["seed"], json.dumps(r["shape"], sort_keys=True))
    by_step = {}
    for r, n in zip(recs, names):
        if recipe(r) != recipe(recs[0]):
            bad.append(f"{n}: a different recipe (shards, seed or shape); not comparable")
            continue
        for row in r["rows"]:
            by_step.setdefault(row["step"], []).append((n, row))
    print(f"{'step':>4}  " + "  ".join(f"{n:>22}" for n in names))
    for step in sorted(by_step):
        cells = by_step[step]
        for key in ("state", "grad", "losses", "shard_grads"):
            vals = {n: json.dumps(row[key]) for n, row in cells}
            if len(cells) > 1:
                compared += 1
                if len(set(vals.values())) != 1:
                    bad.append(f"step {step} {key}: " + ", ".join(f"{n}={v[:18]}" for n, v in vals.items()))
        print(f"{step:>4}  " + "  ".join(f"{dict(cells).get(n, {}).get('state', '-'):>22}" for n in names))
    if args.grads:
        mixed = _mixed(args.grads, recs[0]["logical_shards"])
        compared += mixed[0]
        bad += mixed[1]
    print(f"compared={compared} disagreements={len(bad)}")
    for b in bad:
        print("DISAGREE", b)
    if compared == 0:
        print("NOTHING WAS COMPARED: give two or more records of the same recipe")
        return 1
    return 1 if bad else 0


def _mixed(paths, K):
    """Every assignment of shards to runs: shard k's gradient taken from run
    V(k), folded on the host, held to every run's device sum."""
    if len(paths) < 2:
        return 0, ["--grads needs two or more .npz files for a mixed fold"]
    zs = [np.load(p) for p in paths]
    names = [Path(p).stem for p in paths]
    compared, bad = 0, []
    steps = sorted(set.intersection(*(set(int(k[1:].split("_")[0]) for k in z.files) for z in zs)))
    for step in steps:
        sums = {_h(z[f"s{step}_sum"]) for z in zs}
        for assign in itertools.product(range(len(zs)), repeat=K):
            if len(set(assign)) < 2:
                continue  # one box alone is the on-box check, already done
            folded = _h(ordered_fold([zs[b][f"s{step}_k{k}"] for k, b in enumerate(assign)]))
            compared += 1
            if sums != {folded}:
                bad.append(f"step {step} mixed {[names[b] for b in assign]}: fold {folded} vs device sums {sorted(sums)}")
    print(f"mixed-vendor folds: {compared} assignments over {len(steps)} steps")
    return compared, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--devices", default="0")
    r.add_argument("--label", default="")
    r.add_argument("--shards", type=int, default=4)
    r.add_argument("--steps", type=int, default=6)
    r.add_argument("--handoff-at", type=int, default=3)
    r.add_argument("--resume")
    r.add_argument("--grads", help="write every shard gradient to this .npz")
    r.add_argument("--seed", type=int, default=20260921)
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--length", type=int, default=32)
    r.add_argument("--d-model", type=int, default=32)
    r.add_argument("--heads", type=int, default=4)
    r.add_argument("--kv", type=int, default=2)
    r.add_argument("--ff", type=int, default=64)
    r.add_argument("--layers", type=int, default=2)
    r.add_argument("--vocab", type=int, default=512)
    c = sub.add_parser("compare")
    c.add_argument("records", nargs="+")
    c.add_argument("--grads", nargs="*")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
