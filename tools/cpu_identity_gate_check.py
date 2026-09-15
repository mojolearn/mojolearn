#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The verdict steps of .github/workflows/cpu-identity-gate.yml (the CPU
training lane, 2026-09-13; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md
section 3.4), as a tool that runs on a laptop too.

  readback   import the checkout's package on this CPU-only box and read
             every host binding under mojolearn/host/ back: vendor() must be
             cpu, and each binding's <prefix>_column() must be "cpu" (the
             kernel matrix's CPU column, the comptime assert's witness).
             Exit 2 when a named binding is not built.
  column     judge a tools/identity_break.py JSON written on a CPU-only box:
             vendor starts with cpu-, commit equals --commit, host.column is
             cpu, every named host binding is in host.families with column
             cpu, at least one cell exists, every cell of a lane in --covered
             is STABLE, and every other cell is REFUSED with the by-name
             sentence "no CPU implementation of". A JSON with no cell, or a
             cell of the wrong kind, fails: a step that was supposed to
             produce a cell and produced none is a failure, never a pass.
  run-column run tools/identity_break.py over --lanes as SHARDS in parallel
             processes and join the parts with identity_break --merge into
             --json (2026-09-15, lane/cpu-training-gate-budget). The gate's
             covered lanes outgrew one process on a hosted runner: at
             2cf5562b1 two runners hit the job's 60-minute limit (run
             34956243867). Every lane still runs every fixture and every
             repeat, in a process of its own shard; the merge refuses parts
             that ran a different machine, commit, fixture or binding bytes.
             Exit non-zero if any shard exits non-zero (a MOVED cell, a
             refusal of the lane list, a crash) or the merge refuses.

Exit 0: every check holds. 1: a check failed. 2: the tool could not run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REFUSAL = "no CPU implementation of"


def do_readback(args):
    sys.path.insert(0, os.path.abspath(args.package_root))
    try:
        import mojolearn
        from mojolearn import _backend
    except Exception as exc:
        print(f"readback: import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    vendor = mojolearn.vendor()
    print(f"readback vendor() = {vendor}")
    print(f"readback vendor_how() = {_backend.vendor_how()}")
    built = _backend.host_families_built()
    print(f"readback host_families_built = {built}")
    if vendor != "cpu":
        print("readback FAIL: this package did not take the CPU-only path", file=sys.stderr)
        return 1
    missing = [b for b in args.binding if b not in built]
    if missing:
        print(f"readback: not built: {missing}", file=sys.stderr)
        return 2
    bad = 0
    for basename in args.binding or built:
        prefix = basename[len("_mojolearn_"):]
        try:
            m = _backend.load_host_module(basename)
        except Exception as exc:
            print(f"readback FAIL {basename}: {type(exc).__name__}: {exc}", file=sys.stderr)
            bad += 1
            continue
        column = str(getattr(m, prefix + "_column")())
        detected = getattr(m, prefix + "_detected_column", None)
        detected = str(detected()) if detected is not None else "(no read-back)"
        numeric = int(getattr(m, prefix + "_numeric_mode")())
        print(f"readback {basename}: column={column} detected_column={detected} "
              f"numeric_mode={numeric} vendor={getattr(m, prefix + '_vendor')()}")
        if column != "cpu":
            print(f"readback FAIL {basename}: column {column!r} is not cpu", file=sys.stderr)
            bad += 1
    print(f"readback verdict {'OK' if not bad else 'FAIL'}")
    return 1 if bad else 0


def do_column(args):
    try:
        with open(args.json) as fh:
            j = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"column: cannot read {args.json}: {exc}", file=sys.stderr)
        return 2
    covered = set(x for x in (args.covered or "").split(",") if x)
    bindings = [x for x in (args.binding or "").split(",") if x]
    failures = []

    def need(cond, text):
        if not cond:
            failures.append(text)

    vendor = str(j.get("vendor", ""))
    need(vendor.startswith("cpu-") and len(vendor) > 4, f"vendor {vendor!r} does not start with cpu-")
    need(bool(j.get("commit")), "commit is empty")
    if args.commit:
        need(j.get("commit") == args.commit, f"commit {j.get('commit')!r} is not {args.commit!r}")
    host = j.get("host") or {}
    need(bool(host), "no host object; the run was not on a CPU-only install")
    need(host.get("column") == "cpu", f"host.column is {host.get('column')!r}, not cpu")
    need(bool(host.get("cpu_model")), "host.cpu_model is empty")
    families = host.get("families") or {}
    for b in bindings:
        need(b in families, f"host.families lacks {b}")
        need(families.get(b, {}).get("column") == "cpu", f"host.families[{b}].column is not cpu")
    cells = j.get("cells") or {}
    need(len(cells) > 0, "the JSON carries NO cell; a run that was supposed to produce cells produced none")
    need(j.get("complete", False), "the JSON is INCOMPLETE (the run was killed)")
    seen = set()
    for key, cell in cells.items():
        lane = key.split("/")[0]
        seen.add(lane)
        verdict = cell.get("verdict")
        if lane in covered:
            need(verdict == "STABLE",
                 f"{key}: covered lane reads {verdict}, not STABLE" + (f" ({cell.get('error', '')[:160]})" if verdict == "REFUSED" else ""))
        else:
            need(verdict == "REFUSED", f"{key}: uncovered lane reads {verdict}, not REFUSED; a hash from a lane with no CPU implementation is a routing bug")
            need(REFUSAL in str(cell.get("error", "")),
                 f"{key}: refused, but not by name ({str(cell.get('error', ''))[:160]!r})")
    for lane in sorted(covered - seen):
        need(False, f"covered lane {lane} has no cell in the JSON")
    print(f"column {os.path.basename(args.json)}: vendor={vendor} commit={j.get('commit')} "
          f"host.column={host.get('column')} cpu_model={host.get('cpu_model')!r} "
          f"families={sorted(families)} cells={len(cells)} lanes={sorted(seen)} covered={sorted(covered)}")
    for f in failures:
        print(f"column FAIL: {f}")
    print(f"column verdict {'OK' if not failures else 'FAIL'} ({len(failures)} failure(s))")
    return 1 if failures else 0


#: Seconds per covered lane on a hosted runner, the mean over the seven
#: runners of CPU identity gate run 34955435564 (b7e5a4287), read from the
#: log timestamps of the lane rows. A BALANCE HINT ONLY: a stale or missing
#: entry moves wall time between shards and never changes a cell. A lane not
#: listed weighs DEFAULT_LANE_SECONDS (the 60 lanes under 5 s averaged 1.5).
LANE_SECONDS = {
    "iforest-tuned": 99, "dbscan-weighted": 89, "iforest": 86, "hdbscan-leaf": 69, "hdbscan": 69,
    "kmeans-random": 67, "gbdt-symmetric": 57, "gbdt-depthwise": 53, "bootstrap": 46, "gbdt-lossguide": 43,
    "kmeans-sqrt": 41, "rf-reg-gamma-ig": 41, "kmeans": 37, "kmeans-weighted": 32, "spectral": 31,
    "kmeans-array": 25, "radius-minkowski-p3": 23, "rf-reg": 21, "agglomerative": 20, "kmeans-classic-pp": 19,
    "gbdt-rmse": 16, "metrics-classification": 16, "permutation-test": 16, "kde-cosine-minkowski": 15,
    "dbscan-brute-l1": 15, "metrics": 13, "rf-clf-balanced-parallel": 12, "knn-minkowski-p3": 12, "svr": 12,
    "rf-clf-entropy-log2-noboot": 10, "svr-linear": 10, "svc": 9, "et-reg": 9, "et-clf": 8, "rf-reg-poisson": 8,
    "cross-val": 8, "et-clf-entropy-bestfirst": 7, "arima-seasonal-c": 7, "dbscan": 7, "knn-clf": 7,
    "knn-clf-distance": 7, "kde-linear-cosine": 6, "logistic-elasticnet": 6, "logistic-l1": 6, "kde-weighted": 6,
    "logistic-multiclass": 5, "kde": 5, "et-reg-bootstrap-parallel": 5,
}
DEFAULT_LANE_SECONDS = 4


def shard_lanes(lanes, shards):
    """`lanes` split into at most `shards` lists, heaviest lane first onto
    the lightest shard; each shard keeps the caller's lane order. Every lane
    lands in exactly one shard."""
    shards = max(1, min(shards, len(lanes)))
    load = [0] * shards
    owner = {}
    for lane in sorted(lanes, key=lambda n: (-LANE_SECONDS.get(n, DEFAULT_LANE_SECONDS), lanes.index(n))):
        k = load.index(min(load))
        owner[lane] = k
        load[k] += LANE_SECONDS.get(lane, DEFAULT_LANE_SECONDS)
    out = [[n for n in lanes if owner[n] == k] for k in range(shards)]
    assert sorted(sum(out, [])) == sorted(lanes)
    return out, load


def do_run_column(args):
    lanes = [n for n in args.lanes.split(",") if n]
    if not lanes:
        print("run-column: --lanes is empty", file=sys.stderr)
        return 2
    if len(set(lanes)) != len(lanes):
        print(f"run-column: --lanes repeats a lane: {sorted(n for n in set(lanes) if lanes.count(n) > 1)}", file=sys.stderr)
        return 2
    shards, load = shard_lanes(lanes, args.shards)
    jobs = max(1, min(args.jobs, len(shards)))
    stem = args.json[:-5] if args.json.endswith(".json") else args.json
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity_break.py")
    parts = [f"{stem}.part{k}.json" for k in range(len(shards))]
    logs = [f"{stem}.part{k}.log" for k in range(len(shards))]
    for path in parts + [args.json]:
        if os.path.exists(path):
            os.remove(path)             # a part left by an earlier run must not be merged
    print(f"run-column: {len(lanes)} lanes in {len(shards)} shards, {jobs} at a time", flush=True)
    for k, shard in enumerate(shards):
        print(f"run-column: shard {k} weight {load[k]} s: {','.join(shard)}", flush=True)
    t0 = time.time()
    pending = list(range(len(shards)))
    running = {}
    codes = {}
    last_beat = t0
    while pending or running:
        while pending and len(running) < jobs:
            k = pending.pop(0)
            cmd = [sys.executable, tool, "--lanes", ",".join(shards[k]), "--json", parts[k]] + args.extra
            fh = open(logs[k], "w")
            running[k] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
        for k in list(running):
            proc, fh, started = running[k]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            codes[k] = rc
            del running[k]
            print(f"run-column: shard {k} exited {rc} after {time.time() - started:.0f} s "
                  f"({len(shards[k])} lanes, weight {load[k]} s)", flush=True)
        now = time.time()
        if running and now - last_beat >= args.heartbeat:
            last_beat = now
            for k, (proc, fh, started) in sorted(running.items()):
                try:
                    with open(logs[k]) as lf:
                        tail = [ln for ln in lf.read().splitlines() if ln.startswith("| ")][-1:]
                except OSError:
                    tail = []
                done = tail[0].split("|")[1].strip() if tail else "(no lane yet)"
                print(f"run-column: {now - t0:.0f} s, shard {k} running, last row: {done}", flush=True)
        time.sleep(0.5)
    print(f"run-column: all shards done after {time.time() - t0:.0f} s", flush=True)
    for k in range(len(shards)):
        print(f"\n=== shard {k}: {','.join(shards[k])} (exit {codes[k]}) ===", flush=True)
        with open(logs[k]) as lf:
            sys.stdout.write(lf.read())
        sys.stdout.flush()
    bad = [k for k in range(len(shards)) if codes[k] != 0]
    missing = [parts[k] for k in range(len(shards)) if not os.path.exists(parts[k])]
    for k in bad:
        print(f"run-column FAIL: shard {k} exited {codes[k]}", flush=True)
    for m in missing:
        print(f"run-column FAIL: {m} was not written", flush=True)
    if missing:
        return 1
    merged = subprocess.run([sys.executable, tool, "--merge"] + parts + ["--json", args.json])
    if merged.returncode != 0:
        print(f"run-column FAIL: identity_break --merge exited {merged.returncode}", flush=True)
        return 1
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rb = sub.add_parser("readback", help="read every host binding back on this CPU-only box")
    rb.add_argument("--package-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    rb.add_argument("--binding", action="append", default=[], metavar="BASENAME",
                    help="a host binding that must be built and read back as cpu (repeatable)")
    col = sub.add_parser("column", help="judge an identity_break JSON written on a CPU-only box")
    col.add_argument("json")
    col.add_argument("--covered", default="", help="lanes that must be STABLE, comma separated; every other lane must be REFUSED by name")
    col.add_argument("--commit", default="", help="the commit the JSON must carry")
    col.add_argument("--binding", default="", help="host bindings that must appear in host.families with column cpu, comma separated")
    rc = sub.add_parser("run-column", help="run identity_break over --lanes in parallel shards and merge the parts")
    rc.add_argument("--lanes", required=True, help="the lanes to run, comma separated")
    rc.add_argument("--json", required=True, help="the merged column; parts and logs are written beside it")
    rc.add_argument("--shards", type=int, default=os.cpu_count() or 1, help="processes the lanes are split across")
    rc.add_argument("--jobs", type=int, default=0, help="shards running at once (default: --shards)")
    rc.add_argument("--heartbeat", type=float, default=120.0, help="seconds between progress lines")
    rc.add_argument("extra", nargs="*", help="after --, arguments passed to every identity_break shard")
    args = ap.parse_args(argv)
    if args.cmd == "readback":
        return do_readback(args)
    if args.cmd == "run-column":
        if args.jobs <= 0:
            args.jobs = args.shards
        return do_run_column(args)
    return do_column(args)


if __name__ == "__main__":
    sys.exit(main())
