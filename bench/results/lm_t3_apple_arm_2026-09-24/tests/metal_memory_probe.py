#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Metal trainer's memory at the T3 model width, at rows that fit, to size the target.

    python tools/mac_slot.py metal -- VENV/bin/python metal_memory_probe.py \
        --recipe t3_recipe.json --manifest manifest.json --rows 1x256 1x512 2x512 1x1024

One `ParallelByteLanguageModelTrainer` per (batch x length), K = 1, one
device, the T3 recipe's shape otherwise (d_model 768, 12 heads, 12 layers,
FF 2048, vocabulary 50,257; 162,147,840 parameters), weights drawn as
`lm_segment.py init` draws them. After `open` and after one `train_step` on
random ids the process's physical footprint is read with
`proc_pid_rusage(RUSAGE_INFO_V4)` (on Apple silicon a Metal buffer in shared
storage is part of the process footprint). This is a MEASUREMENT OF MEMORY
ONLY at rows far below the recipe's batch 4 x length 2048; nothing here is a
step of the run and no chain line is written. The target shape itself is
never opened: at the working set the other vendors measured it would push
this 16 GiB Mac into swap while the T3 driver and its dead-men live on it.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))


def footprint():
    """(phys_footprint, lifetime_max_phys_footprint) of this process, bytes."""
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = ctypes.create_string_buffer(512)
    if lib.proc_pid_rusage(os.getpid(), 4, buf) != 0:
        raise OSError("proc_pid_rusage failed")
    q = lambda off: int.from_bytes(buf.raw[off:off + 8], "little")  # noqa: E731
    return q(72), q(240)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--manifest", required=True, help="the token stream's manifest.json (its identity only)")
    ap.add_argument("--rows", nargs="+", required=True, help="BATCHxLENGTH")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import numpy as np
    from lm_segment import ManifestBatches, load_recipe, data_schedule, _bits_f32
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
    import mojolearn
    recipe = load_recipe(args.recipe)
    rows = []
    base = footprint()
    for spec in args.rows:
        B, L = (int(x) for x in spec.split("x"))
        shape_list = list(recipe["shape"])
        shape_list[0], shape_list[1] = B, L
        shape = Shape(*shape_list)
        batches = ManifestBatches(args.manifest, B, L)
        rng = np.random.default_rng(recipe["seed"])
        weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
        for entry in Trainer.parameter_registry(shape):
            if "norm" in entry["name"]:
                weights[entry["offset"]:entry["offset"] + entry["size"]] += np.float32(1)
        opt = recipe["optimizer"]
        t = Trainer(weights, shape=shape, data_schedule=data_schedule(recipe, batches),
                    lr=_bits_f32(int(recipe["schedule"]["table_f32_hex"][0], 16)),
                    betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
        state = t.state_dict()
        del t, weights
        before = footprint()
        ids = np.random.default_rng(1).integers(0, shape_list[8] - 1, size=(B, L + 1), dtype=np.int32)
        with Par(state, devices=(0,), logical_shards=1) as par:
            del state
            t0 = time.perf_counter()
            par.train_step([ids])
            first = time.perf_counter() - t0
            opened = footprint()
            t0 = time.perf_counter()
            par.train_step([ids])
            second = time.perf_counter() - t0
            after = footprint()
        row = dict(batch=B, length=L, tokens=B * L, n_total=shape.n_total,
                   footprint_before=before[0], footprint_after_step=after[0], lifetime_max=after[1],
                   first_step_seconds=round(first, 3), second_step_seconds=round(second, 3))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = dict(schema="mojolearn.apple-memory-probe.v1", mojolearn=mojolearn.__version__, file=mojolearn.__file__,
               process_base=base, rows=rows, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
