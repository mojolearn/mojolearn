#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On a GPU box: where the runner's per-step `hash_seconds` goes, and whether
the digests really run during the next native step.

    python box_split.py RECIPE TOKENS CKPT EXPECT_CHAIN OUT.json [--devices 0]

Run from a checkout (tools/lm_segment.py beside it) with the published wheel.
From the checkpoint it takes ONE optimizer step (as `lm_segment.py run`
does: the table's learning rate, the recipe's shards), then times, on that
post-step state:

  export_raw_s       `trainer.export_raw()`: fresh arrays, device read-back
  export_grad_s      `trainer.export_gradients()`: the same, the gradient
  old_hash_s         `_hash_arrays(v2)` + `_hash_gradient(v2)` (runner before 2026-09-25)
  alloc_s            `empty()` of the four 648.6 MB arrays alone
  snapshot_first_s   `Snapshot.read`: the same export call, allocating
  snapshot_reuse_s   `Snapshot.read` again, into the same buffers
  new_hash_s         `Digests(v2).result()`, nothing else running
  one_core_s         one sha256 over the same 2.59 GB

The old and new digests must be equal, and equal to the expected chain's
line for the step when it has one. Then it starts `Digests` on that
snapshot and takes a SECOND step with `train_step` (the native call keeps
the interpreter lock): `overlap_step_s` is that step and
`overlap_after_step_s` how long after it returned the digests were ready.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path.cwd() / "tools"))


def main():
    ap = argparse.ArgumentParser()
    for name in ("recipe", "tokens", "ckpt", "expect", "out"):
        ap.add_argument(name)
    ap.add_argument("--devices", default="0")
    args = ap.parse_args()
    import lm_segment as seg
    from mojolearn._buffer import empty
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
    recipe = seg.load_recipe(args.recipe)
    K = int(recipe["logical_shards"])
    scheme = seg.hash_scheme_of(recipe)
    table = recipe["schedule"]["table_f32_hex"]
    batches = seg.open_batches(recipe, args.tokens)
    expect = seg._chain_index(args.expect)
    state, _ = seg.load_checkpoint(args.ckpt)
    first = int(state["completed_steps"])
    devices = tuple(int(x) for x in args.devices.split(","))
    r = dict(host=platform.node(), python=platform.python_version(), cpus=os.cpu_count(), devices=list(devices),
             scheme=scheme, from_step=first)

    def timed(key, fn):
        t = time.perf_counter()
        v = fn()
        r[key] = round(time.perf_counter() - t, 3)
        return v

    with Par(state, devices=devices, logical_shards=K) as tr:
        del state

        def step(s):
            tr.set_lr(seg._bits_f32(int(table[s], 16)))
            return tr.train_step([batches.ids(s * K + k) for k in range(K)])["losses"]
        timed("step_s", lambda: step(first))
        raw = timed("export_raw_s", tr.export_raw)
        g = timed("export_grad_s", tr.export_gradients)
        old = timed("old_hash_s", lambda: (seg._hash_arrays(raw, scheme), seg._hash_gradient(g, scheme)))
        del raw, g
        n, nt = tr._shape.n_total, tr._shape.n_tensors
        junk = timed("alloc_s", lambda: [empty((n,), "<f4") for _ in range(4)] + [empty((nt,), "<i4")])
        del junk
        snap = seg.Snapshot()
        timed("snapshot_first_s", lambda: snap.read(tr))
        timed("snapshot_reuse_s", lambda: snap.read(tr))
        new = timed("new_hash_s", lambda: seg.Digests(snap.arrays, snap.gradient, scheme).result())

        def one_core():
            h = hashlib.sha256()
            for a in (snap.arrays["parameters"], snap.arrays["m"], snap.arrays["v"], snap.gradient):
                h.update(seg._bytes_of(a))
            return h.hexdigest()
        timed("one_core_s", one_core)
        want = expect.get(first + 1)
        r.update(step=first + 1, old=list(old), new=list(new), old_equals_new=old == new,
                 expected=None if want is None else [want["state_sha256"], want["gradient_sha256"]],
                 equals_expected=None if want is None else list(old) == [want["state_sha256"], want["gradient_sha256"]])
        t = time.perf_counter()
        d = seg.Digests(snap.arrays, snap.gradient, scheme)
        r["overlap_start_s"] = round(time.perf_counter() - t, 4)
        r["overlap_all_started"] = d.started_all
        timed("overlap_step_s", lambda: step(first + 1))
        back = time.perf_counter()
        again = d.result()
        r["overlap_after_step_s"] = round(time.perf_counter() - back, 3)
        r["overlap_equals"] = list(again) == list(new)
    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r))
    return 0 if r["old_equals_new"] and r["overlap_equals"] and r["equals_expected"] is not False else 1


if __name__ == "__main__":
    sys.exit(main())
