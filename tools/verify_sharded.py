#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SHARD `verify --all` -- ON A CPU BACKEND ONLY, AND IT REFUSES ANY OTHER.

    python3 tools/verify_sharded.py                 # 4 shards, CPU backend
    python3 tools/verify_sharded.py --shards 6
    python3 tools/verify_sharded.py --weights <a previous --json-out>

READ THIS BEFORE CHANGING THE REFUSAL. On 2026-09-19 this tool was written
without it, run against the Metal backend on the shared M4, and it produced
1049 DIVERGENT cell parts across 40 lanes. NONE OF THEM WERE REAL. Three of
those lanes, re-run SOLO on the same bindings the same minute:

    gmm, gp-matern12, rbf-sampler, 4-way sharded   36 DIVERGENT parts each
    the same three, solo                            0 DIVERGENT, VERIFIED

`docs/` and this project's operating rules already said so: concurrent Metal
jobs on one M4 return NaN, constant and zero outputs, and solo reruns come
back clean -- a Metal-only cell produced under contention IS NOT EVIDENCE.
Four shards on one GPU is four concurrent Metal jobs.

AND IT BOUGHT NOTHING. Predicted critical path 323 s; measured wall 1163 s
against 1296 s serial. A 10% saving, for a run whose every divergence was an
artifact, because THE GPU IS THE SERIAL RESOURCE -- packing lanes across
processes cannot parallelise one device.

SO: this tool refuses to run unless the verifier is on the CPU route, where
the shards are genuinely independent. On a Metal or CUDA backend, one Mac
runs `verify --all` SERIALLY and 21.6 minutes is the floor on this machine.

WHEN IT DOES APPLY, THE PACKING IS THE RIGHT PACKING. A CPU run's cost is
wildly uneven -- from the 2026-09-19 measurement:

    median lane                    0.76 s
    lanes finishing under 1 s       143 of 241
    top 15 lanes                    919 s = 71% of all lane time
    gbdt-parametric-losses          200 s = 15.5%, ONE LANE

so an even split does not divide the wall clock: whichever shard draws the
200 s lane is the critical path. Heaviest lane first onto the lightest shard
-- the rule `cpu_identity_gate_check.shard_lanes` already uses, so the split
is deterministic and two runs are comparable. THE FLOOR IS THE HEAVIEST
LANE, 200 s, and no number of shards beats it.

`--weights` takes any previous `--json-out`; a lane ABSENT from its
`lane_seconds` is given the median, never zero, so a new lane is never
packed as if it were free. The comparator self-test runs ONCE, in shard 0.
A shard that exits non-zero or writes no json makes the whole run INCOMPLETE
and is named.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def lanes_and_weights(weights_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import identity_break as ib
    lanes = sorted(ib.LANES)
    weights = {}
    if weights_path:
        data = json.loads(pathlib.Path(weights_path).read_text())
        weights = {k: float(v) for k, v in (data.get("lane_seconds") or {}).items()}
    if weights:
        # A LANE WITH NO MEASUREMENT IS NOT A FREE LANE.
        fallback = statistics.median(weights.values())
        weights = {ln: weights.get(ln, fallback) for ln in lanes}
    else:
        weights = {ln: 1.0 for ln in lanes}
    return lanes, weights


def pack(lanes, weights, shards):
    """Heaviest lane first onto the lightest shard."""
    bins = [[] for _ in range(shards)]
    load = [0.0] * shards
    for lane in sorted(lanes, key=lambda ln: -weights[ln]):
        i = load.index(min(load))
        bins[i].append(lane)
        load[i] += weights[lane]
    return bins, load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--weights", default=None,
                    help="a previous --json-out; its lane_seconds pack the shards")
    ap.add_argument("--out", default=None, help="directory for the shard JSONs")
    ap.add_argument("--plan", action="store_true", help="print the split and stop")
    ap.add_argument("--cpu-only", action="store_true",
                    help="assert the verifier is on the CPU route; sharding is "
                         "refused without it (see this module's docstring)")
    args = ap.parse_args(argv)

    if args.shards > 1 and not args.cpu_only:
        print("REFUSED: sharding is valid on the CPU route only.\n"
              "  Four shards on one GPU is four CONCURRENT METAL JOBS, and this\n"
              "  tool produced 1049 phantom DIVERGENT parts that way on\n"
              "  2026-09-19; the same lanes run solo came back VERIFIED. It also\n"
              "  saved nothing -- 1163 s sharded against 1296 s serial -- because\n"
              "  the GPU is the serial resource.\n"
              "  Pass --cpu-only when the verifier is on the CPU route, or run\n"
              "  `python -m mojolearn verify --all` serially for a GPU column.",
              file=sys.stderr)
        return 2

    lanes, weights = lanes_and_weights(args.weights)
    bins, load = pack(lanes, weights, args.shards)
    unit = "s" if args.weights else " lanes"
    for i, (b, w) in enumerate(zip(bins, load)):
        print(f"# shard {i}: {len(b):3d} lanes, predicted {w:7.1f}{unit}")
    print(f"# critical path (predicted): {max(load):.1f}{unit}")
    if args.plan:
        return 0

    out = pathlib.Path(args.out or (ROOT / "bench" / "results" / "verify_sharded"
                                    / time.strftime("%Y-%m-%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    started = time.time()
    procs = []
    for i, b in enumerate(bins):
        cmd = [sys.executable, "-m", "mojolearn", "verify", "--all",
               "--lanes", ",".join(b), "--json-out", str(out / f"shard{i}.json")]
        # THE SELF-TEST IS ONE QUESTION, NOT N. Shard 0 carries it.
        if i != 0:
            cmd.append("--no-models")
        procs.append((i, subprocess.Popen(cmd, cwd=ROOT / "python", env=env,
                                          stdout=(out / f"shard{i}.log").open("w"),
                                          stderr=subprocess.STDOUT)))
    bad = []
    for i, p in procs:
        if p.wait() != 0:
            bad.append(f"shard {i} exited {p.returncode}")
    elapsed = time.time() - started

    counts, lane_seconds, verdicts = {}, {}, []
    for i, _ in procs:
        f = out / f"shard{i}.json"
        if not f.exists():
            bad.append(f"shard {i} wrote no json")
            continue
        d = json.loads(f.read_text())
        for k, v in (d.get("counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
        lane_seconds.update(d.get("lane_seconds") or {})
        verdicts.append(d.get("verdict"))

    merged = dict(counts=counts, lane_seconds=lane_seconds, shards=args.shards,
                  elapsed_s=elapsed, shard_verdicts=verdicts, problems=bad)
    (out / "merged.json").write_text(json.dumps(merged, indent=2, sort_keys=True))

    print()
    for k in sorted(counts):
        print(f"  {k:12s} {counts[k]}")
    print(f"\n  wall {elapsed:.0f}s = {elapsed/60:.1f} min over {args.shards} shards")
    print(f"  merged -> {out / 'merged.json'}")
    if bad:
        for b in bad:
            print(f"  PROBLEM: {b}")
        print("  INCOMPLETE: a shard that dies is not a shard that passed")
        return 1
    if counts.get("DIVERGENT"):
        print(f"  MISMATCH: {counts['DIVERGENT']} divergent cell parts")
        return 1
    print("  OK: no divergent cell parts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
