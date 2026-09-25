#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On a GPU box: what the byte-LM state and gradient exports cost, and the
sha256 of every exported array, for ONE binding (run once per binding).

    python box_export.py RECIPE TOKENS CKPT EXPECT_CHAIN OUT.json [--devices 0] [--label old]

Run from a checkout (tools/lm_segment.py beside it) with PYTHONPATH naming
the checkout's package. From the checkpoint it takes ONE optimizer step as
`lm_segment.py run` does (the table's learning rate, the recipe's K shards,
the default pooled optimizer), then on that post-step state, repeated
`REPS` times each:

  state_fresh_s     trainer.export_raw()               (allocates four arrays)
  state_direct_s    the binding call into four buffers allocated once
                    (what a reusing caller pays: the binding alone)
  state_into_s      trainer.export_raw(into=...)       (when the package has it)
  grad_fresh_s, grad_direct_s, grad_into_s             the same for the gradient
  alloc_s           allocating the four state arrays alone

and the plain sha256 of parameters, m, v, flags and the gradient from EVERY
form (they must agree within the run; across runs the old and the new
binding must agree), plus the chain digests (`_hash_arrays`,
`_hash_gradient` under the recipe's scheme) held to the expected chain's
line for the step.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path.cwd() / "tools"))
REPS = 3


def main():
    ap = argparse.ArgumentParser()
    for name in ("recipe", "tokens", "ckpt", "expect", "out"):
        ap.add_argument(name)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    import lm_segment as seg
    from mojolearn._buffer import addr, empty
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
    r = dict(label=args.label, host=platform.node(), python=platform.python_version(), cpus=os.cpu_count(),
             devices=list(devices), scheme=scheme, from_step=first, reps=REPS)
    keys = ("parameters", "m", "v", "flags")

    def sha(x):
        return hashlib.sha256(seg._bytes_of(x)).hexdigest()

    def timed(key, fn):
        t = time.perf_counter()
        v = fn()
        r.setdefault(key, []).append(round(time.perf_counter() - t, 4))
        return v

    digests = {}

    def record(form, raw=None, grad=None):
        d = digests.setdefault(form, [])
        row = {}
        if raw is not None:
            row.update({k: sha(raw[k]) for k in keys})
        if grad is not None:
            row["gradient"] = sha(grad)
        d.append(row)

    with Par(state, devices=devices, logical_shards=K) as tr:
        del state
        tr.set_lr(seg._bits_f32(int(table[first], 16)))
        timed("step_s", lambda: tr.train_step([batches.ids(first * K + k) for k in range(K)]))
        n, nt = tr._shape.n_total, tr._shape.n_tensors
        has_into = "into" in inspect.signature(tr.export_raw).parameters
        r["package_has_into"] = has_into
        for _ in range(REPS):
            raw = timed("state_fresh_s", tr.export_raw)
            record("fresh", raw=raw)
            del raw
            g = timed("grad_fresh_s", tr.export_gradients)
            record("fresh_grad", grad=g)
            del g
        bufs = timed("alloc_s", lambda: {k: empty((nt,) if k == "flags" else (n,), "<i4" if k == "flags" else "<f4")
                                          for k in keys})
        gbuf = empty((n,), "<f4")
        for _ in range(REPS):
            step = timed("state_direct_s", lambda: tr._binding.byte_lm_parallel_export(
                tr._session, [addr(bufs[k], name=k) for k in keys], 0, False))
            assert step == first + 1, step
            record("direct", raw=bufs)
            step = timed("grad_direct_s", lambda: tr._binding.byte_lm_parallel_export(
                tr._session, [addr(gbuf, name="gradients")], 0, True))
            assert step == first + 1, step
            record("direct_grad", grad=gbuf)
        if has_into:
            ibufs = {k: empty((nt,) if k == "flags" else (n,), "<i4" if k == "flags" else "<f4") for k in keys}
            ig = empty((n,), "<f4")
            for _ in range(REPS):
                out = timed("state_into_s", lambda: tr.export_raw(into=ibufs))
                assert out is ibufs
                record("into", raw=ibufs)
                out = timed("grad_into_s", lambda: tr.export_gradients(into=ig))
                assert out is ig
                record("into_grad", grad=ig)
        chain_state = seg._hash_arrays(bufs, scheme)
        chain_grad = seg._hash_gradient(gbuf, scheme)
    flat = {}
    for form, rows in digests.items():
        for row in rows:
            for k, v in row.items():
                flat.setdefault(k, set()).add(v)
    r["sha256"] = {k: sorted(v) for k, v in flat.items()}
    r["forms_agree"] = all(len(v) == 1 for v in flat.values())
    want = expect.get(first + 1)
    r.update(step=first + 1, chain_state=chain_state, chain_gradient=chain_grad,
             expected=None if want is None else [want["state_sha256"], want["gradient_sha256"]])
    r["equals_expected"] = None if want is None else [chain_state, chain_grad] == r["expected"]
    for k in [k for k in r if k.endswith("_s") and isinstance(r[k], list)]:
        r[k[:-2] + "_median_s"] = sorted(r[k])[len(r[k]) // 2]
    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r))
    return 0 if r["forms_agree"] and r["equals_expected"] is not False else 1


if __name__ == "__main__":
    sys.exit(main())
