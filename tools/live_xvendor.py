#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Live cross-vendor training on the tools/par_lm_xvendor.py recipe.

    python3 tools/live_xvendor.py coordinator --workers 2 --out rows.json [--expect COLUMN.json]
    python3 tools/live_xvendor.py worker --address HOST:PORT --shards 0,1 --name apple-m4
    python3 tools/live_xvendor.py local --workers 2 --out rows.json --expect COLUMN.json

`mojolearn.cross_vendor` does the work; this file only supplies the fixed
problem (weights and tokens from SHAKE-256, K = 4 shards) so that every step
the live group agrees on can also be held to a column recorded by one-process
runs (`--expect`, e.g. bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json).

`local` runs the coordinator and N workers in ONE process on ONE device, the
workers taking turns on it under a lock (one Metal job at a time). It tests
the protocol, the split and the fold; it is not a cross-vendor claim.
"""
import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import par_lm_xvendor as recipe  # noqa: E402


def problem(steps):
    """The par_lm_xvendor defaults: (starting state, ids[step, shard, batch, length + 1])."""
    import mojolearn as ml
    args = argparse.Namespace(seed=20260921, batch=2, length=32, d_model=32, heads=4, kv=2, ff=64,
                              layers=2, vocab=512, steps=steps, shards=4)
    _shape, state0, ids = recipe.build_problem(ml, args)
    return state0, ids


class _Locked:
    """A trainer whose device calls take turns under one lock."""

    def __init__(self, trainer, lock):
        self._t, self._lock = trainer, lock

    def __getattr__(self, name):
        attr = getattr(self._t, name)
        if not callable(attr):
            return attr

        def call(*a, **k):
            with self._lock:
                return attr(*a, **k)
        return call


def _trainer(state, device):
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
    return ParallelByteLanguageModelTrainer(state, devices=(device,), logical_shards=1, pool_optimizer=False)


def _check(rows, expect):
    if not expect:
        return 0
    want = {r["step"]: r["state"] for r in json.loads(Path(expect).read_text())["rows"]}
    bad = [r["step"] for r in rows if want.get(r["step"]) != r["state"][:16]]
    seen = sum(1 for r in rows if r["step"] in want)
    print(f"expected column {expect}: {seen} steps compared, {len(bad)} differ {bad}")
    return 1 if bad or seen == 0 else 0


def _write(out, rows, extra):
    Path(out).write_text(json.dumps(dict(schema="mojolearn.live-xvendor.v1", rows=rows, **extra), indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument("--port", type=int, default=7777)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--workers", type=int, required=True)
    c.add_argument("--steps", type=int, default=6)
    c.add_argument("--out", required=True)
    c.add_argument("--expect")
    w = sub.add_parser("worker")
    w.add_argument("--address", required=True)
    w.add_argument("--shards", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--device", type=int, default=0)
    w.add_argument("--steps", type=int, default=6)
    w.add_argument("--connect-timeout", type=float, default=1800)
    lo = sub.add_parser("local")
    lo.add_argument("--workers", type=int, default=2)
    lo.add_argument("--port", type=int, default=7791)
    lo.add_argument("--steps", type=int, default=6)
    lo.add_argument("--out", required=True)
    lo.add_argument("--expect")
    args = ap.parse_args(argv)
    from mojolearn.cross_vendor import Coordinator, Worker

    def log(row):
        print(f"step {row['step']} state {row['state'][:16]} {sorted(row['vendors'].items())}", flush=True)

    if args.cmd == "coordinator":
        rows = Coordinator(host=args.host, port=args.port, workers=args.workers, logical_shards=4,
                           steps=args.steps, on_step=log).run()
        _write(args.out, rows, dict(workers=args.workers))
        print(f"AGREED on {len(rows)} steps across {args.workers} workers")
        return _check(rows, args.expect)

    state0, ids = problem(args.steps)
    batches = lambda step, shard: ids[step, shard]  # noqa: E731
    if args.cmd == "worker":
        host, port = args.address.rsplit(":", 1)
        done = Worker(state0, shards=[int(k) for k in args.shards.split(",")], batches=batches,
                      address=(host, int(port)), name=args.name, device=args.device,
                      connect_timeout=args.connect_timeout).run()
        print(f"{args.name}: committed {done} steps")
        return 0

    # local: coordinator in a thread, workers in threads, one device under a lock
    K = 4
    split = [list(range(K))[i::args.workers] for i in range(args.workers)]
    lock = threading.Lock()
    result, errors = {}, []

    def coord():
        try:
            result["rows"] = Coordinator(host="127.0.0.1", port=args.port, workers=args.workers,
                                         logical_shards=K, steps=args.steps, on_step=log).run()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def work(i):
        try:
            Worker(trainer=_Locked(_trainer(state0, 0), lock), shards=split[i], batches=batches,
                   address=("127.0.0.1", args.port), name=f"local-{i}", connect_timeout=60).run()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=coord)] + [threading.Thread(target=work, args=(i,)) for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("FAILED:", errors)
        return 1
    _write(args.out, result["rows"], dict(workers=args.workers, local=True, split=split))
    print(f"AGREED on {len(result['rows'])} steps across {args.workers} local workers {split}")
    return _check(result["rows"], args.expect)


if __name__ == "__main__":
    sys.exit(main())
