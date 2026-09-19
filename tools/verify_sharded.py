#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`verify --all` IN SINGLE-DIGIT MINUTES, by packing lanes instead of adding cores.

    python3 tools/verify_sharded.py                 # 4 shards, the default
    python3 tools/verify_sharded.py --shards 6
    python3 tools/verify_sharded.py --weights <a previous --json-out>

WHY THIS EXISTS, AND WHY IT IS PACKING AND NOT PARALLELISM.

`python -m mojolearn verify --all` walks its lanes SERIALLY. On 2026-09-19 it
took 1296 s, and the obvious reading -- "241 lanes, so it is slow" -- is
wrong. The run's own `lane_seconds` says:

    median lane                      0.76 s
    lanes finishing under 1 s         143 of 241
    top 15 lanes                      919 s = 71% of all lane time
    gbdt-parametric-losses            200 s = 15.5%, one lane

The library is not slow. FIFTEEN LANES ARE, ten of them gbdt. That changes
the fix: splitting 241 lanes evenly across N processes does NOT divide the
wall clock by N, because whichever shard draws the 200 s lane becomes the
critical path and the other shards sit idle. Measured against this run:

    shards   even split (naive)      heaviest-first (this tool)
         2            ~648 s                        647 s
         4            ~324 s  but in practice       323 s
         6            ~216 s   bounded by 200 s     216 s

The floor is the single heaviest lane, 200 s, and no number of shards beats
it. Four shards reach 5.4 minutes, which is the budget this was written to.

HEAVIEST FIRST ONTO THE LIGHTEST SHARD -- the same rule
`cpu_identity_gate_check.shard_lanes` uses, for the same reason, so a rerun
with the same weights gives the same split and two runs are comparable.

THE WEIGHTS ARE MEASURED, AND THEY AGE. `--weights` takes any previous
`--json-out`; its `lane_seconds` become the packing weights. Without one
every lane weighs the same and the split is merely even, which is the naive
column above. A lane absent from the weights is given the median, not zero,
so a NEW lane is never packed as if it were free.

WHAT IS NOT SHARDED. The comparator self-test runs ONCE, in shard 0, not in
every shard: it is the same question each time and it costs a full lane.
Every shard reads the same shipped reference table, so the merged counts are
the counts `verify --all` would have produced; this tool re-adds them and
re-derives the verdict, it does not re-judge any cell.

A SHARD THAT DIES IS NOT A SHARD THAT PASSED. Any non-zero exit, or a
missing json, makes the whole run INCOMPLETE and names the shard. That is
the failure mode `verify_lanes.py` documents at length, and it is refused
here by construction rather than by hoping.
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
    args = ap.parse_args(argv)

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
