#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE COMMAND THAT RUNS LANES: one, the ones a change can move, or all of
them (lane/lane-selector, 2026-09-16).

SEVERALLY AND IN UNISON, THROUGH ONE PATH. A separate "full sweep" script
grows its own bugs and its own idea of what the lane set is; that is how one
afternoon produced four lane totals (176, 192, 199, 210) that each claimed to
be the count. So the selective run and the full run are the same code here,
and they differ only in which lanes the selector handed over:

    python3 tools/verify_lanes.py --lane logistic              # one lane
    python3 tools/verify_lanes.py --changed-since origin/main  # this change
    python3 tools/verify_lanes.py --all --shards 16            # everything

The lane set always comes from `tools/lane_select.py`, which reads
`identity_break.LANES` by import. No grep, no list kept by hand.

THREE PROPERTIES, IN THE ORDER THEY BITE.

  0. A REFUSED LANE IS NOT A CHECKED LANE. A shard can exit 0, write its part
     and merge cleanly with every cell reading REFUSED, which is how a stale
     host binding set produced `verdict COMPLETE` in 3 seconds on 2026-09-16.
     COMPLETE means every selected lane carries a cell that RAN.
  1. A SHARD THAT FAILS IS NOT DROPPED. A shard that exits non-zero, or whose
     part file never appeared, makes the run INCOMPLETE. The merged column is
     then written as `<out>/column.incomplete.json` and the exit code is 1.
     `<out>/column.json` exists only for a run where every shard reported.
     This is the `verify --all` failure (VERIFIED printed with most parts
     refused) and it is refused here by construction.
  2. THE SHARDS ARE THE LANE SET. The split asserts the union at split time,
     and the merge asserts again that the lanes carrying cells are exactly
     the lanes selected, counted against the registry, not against a grep.
     A lane that vanished between the two is named and fails the run.
  3. THE SPLIT IS DETERMINISTIC. `lane_select.shard` is
     `cpu_identity_gate_check.shard_lanes`, heaviest lane first onto the
     lightest shard, so the same lane set always gives the same shards and a
     rerun is comparable to the run before it.

WHERE IT RUNS. `--runner local` (the default) runs the shards as processes
here; `--jobs 1`, the default, is the Mac's one-core rule. `--runner pods`
PRINTS one `tools/runpod_cpu_leg.sh` command per shard and rents nothing,
because the full sweep belongs on rented CPU (parallel, cheap, bitwise equal
to Metal) and because renting is never something a tool should do on its own.
The tiers, and which of these to reach for, are
docs/lanes/VERIFICATION_TIERS.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

ROOT = lane_select.ROOT
#: RunPod CPU, 16 vCPU, 2026-09-16. The 8 vCPU figure in docs/RUNPOD_CPU_LEG.md
#: is $0.24/h; a 16 vCPU pod is about twice that. Printed with a plan so the
#: cost of a full sweep is a number before anyone rents anything.
POD_USD_PER_HOUR = 0.48


def _selection(args):
    sources, why = lane_select.lane_sources()
    if args.all:
        return list(sources), dict(mode="all", fallback=False), sources
    if args.lanes or args.lane:
        named = [n for n in (args.lanes or "").split(",") if n] + list(args.lane)
        unknown = [n for n in named if n not in sources]
        if unknown:
            raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}")
        order = list(sources)
        return [n for n in order if n in set(named)], dict(mode="named", fallback=False), sources
    if args.lanes_for_paths:
        sel = lane_select.select(args.lanes_for_paths, sources=sources)
    elif args.changed_since:
        paths = lane_select.changed_paths(args.changed_since)
        print(f"# {len(paths)} changed path(s) against {args.changed_since}")
        sel = lane_select.select(paths, ref=args.changed_since, sources=sources)
    else:
        raise SystemExit("REFUSING: name what to run (--all, --lane, --lanes, "
                         "--lanes-for-paths or --changed-since)")
    for path, reason in sorted(sel["reasons"].items()):
        print(f"# {path}: {reason}")
    if sel["fallback"]:
        print("# FALLING BACK TO EVERY LANE: the blast radius of the paths above could not be "
              "determined. This is a full sweep, not a narrow run.")
    return sel["lanes"], sel, sources


def _commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _identity_break_cmd(lanes, out, args):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "identity_break.py"),
           "--lanes", ",".join(lanes), "--json", out, "--repeats", str(args.repeats)]
    if args.fixtures:
        cmd += ["--fixtures", args.fixtures]
    if args.vendor:
        cmd += ["--vendor", args.vendor]
    return cmd + args.extra


def _plan(groups, load, args, out_dir):
    total = 0
    for k, group in enumerate(groups):
        part = os.path.join(out_dir, f"part{k}.json")
        if args.runner == "pods":
            inner = " ".join(_identity_break_cmd(group, f'"$LEG_OUT"/part{k}.json', args)[1:])
            print(f"\n# shard {k}: {len(group)} lanes, weight {load[k]} s")
            print(f"bash tools/runpod_cpu_leg.sh --lane {args.tag}-s{k} --rent \\\n"
                  f"  --build {args.build} --envs default \\\n"
                  f"  --cmd 'python3 {inner}'")
        else:
            print(f"# shard {k}: {len(group)} lanes, weight {load[k]} s -> {part}")
            print("  " + " ".join(_identity_break_cmd(group, part, args)))
        total += load[k]
    if args.runner == "pods":
        longest = max(load) if load else 0
        print(f"\n# {len(groups)} pods, longest shard {longest} s, serial equivalent {total} s")
        print(f"# about ${POD_USD_PER_HOUR:.2f}/h per pod, so roughly "
              f"${POD_USD_PER_HOUR * len(groups) * max(longest / 3600.0, 1 / 60.0):.2f} "
              "for the sweep at the longest shard (pods bill from create to delete)")
    print("\n# PLAN ONLY. Nothing ran and nothing was rented.")


def _run_local(groups, load, args, out_dir):
    parts = [os.path.join(out_dir, f"part{k}.json") for k in range(len(groups))]
    logs = [os.path.join(out_dir, f"part{k}.log") for k in range(len(groups))]
    for path in parts:
        if os.path.exists(path):
            os.remove(path)                # a part from an earlier run must never be merged
    jobs = max(1, min(args.jobs, len(groups)))
    t0 = time.time()
    pending, running, codes = list(range(len(groups))), {}, {}
    while pending or running:
        while pending and len(running) < jobs:
            k = pending.pop(0)
            fh = open(logs[k], "w")
            env = dict(os.environ)
            env.setdefault("PYTHONPATH", os.path.join(ROOT, "python"))
            proc = subprocess.Popen(_identity_break_cmd(groups[k], parts[k], args),
                                    stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            running[k] = (proc, fh, time.time())
            print(f"# shard {k} started: {len(groups[k])} lanes, weight {load[k]} s", flush=True)
        for k in list(running):
            proc, fh, started = running[k]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            codes[k] = rc
            del running[k]
            print(f"# shard {k} exited {rc} after {time.time() - started:.0f} s", flush=True)
        time.sleep(0.5)
    print(f"# all shards done after {time.time() - t0:.0f} s")
    return parts, codes, time.time() - t0


def _verdict(lanes, parts, codes, out_dir, elapsed):
    """COMPLETE only when every shard reported and the cells cover exactly the
    lanes selected. Anything else is INCOMPLETE and exits 1."""
    failures = []
    for k, part in enumerate(parts):
        if codes.get(k, 1) != 0:
            failures.append(f"shard {k} exited {codes.get(k)}")
        if not os.path.exists(part):
            failures.append(f"shard {k} wrote no part file ({os.path.basename(part)})")
    present = [p for p in parts if os.path.exists(p)]
    merged = os.path.join(out_dir, "column.json" if not failures else "column.incomplete.json")
    if present:
        res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "identity_break.py"),
                              "--merge"] + present + ["--json", merged], cwd=ROOT)
        if res.returncode != 0:
            failures.append(f"identity_break --merge exited {res.returncode}")
    else:
        failures.append("no part file was written at all")
    covered, verdicts = set(), {}
    if os.path.exists(merged):
        with open(merged) as fh:
            cells = json.load(fh).get("cells") or {}
        covered = {k.split("/")[0] for k in cells}
        for key, cell in cells.items():
            verdicts.setdefault(key.split("/")[0], set()).add(cell.get("verdict"))
    # A CELL THAT SAYS REFUSED IS NOT A CHECK. Measured 2026-09-16: one lane on
    # a stale host binding set exited 0, wrote its part, merged, and printed
    # `verdict COMPLETE` in 3 s with its only cell reading REFUSED. That is the
    # `verify --all` failure this file was written against, reached through the
    # one door it did not cover: every SHARD reported, so nothing looked wrong.
    # COMPLETE now means every selected lane carries a cell that actually ran.
    unchecked = sorted(n for n in lanes if n in verdicts and verdicts[n] <= {"REFUSED"})
    if unchecked:
        failures.append(f"{len(unchecked)} selected lane(s) REFUSED every cell, so they were not "
                        f"checked at all: {unchecked[:8]}{' ...' if len(unchecked) > 8 else ''}")
    missing = [n for n in lanes if n not in covered]
    extra = sorted(covered - set(lanes))
    if missing:
        failures.append(f"{len(missing)} selected lane(s) carry no cell: {missing[:8]}"
                        f"{' ...' if len(missing) > 8 else ''}")
    if extra:
        failures.append(f"cells for lanes that were not selected: {extra[:8]}")
    ran = sum(1 for n in lanes if n in verdicts and verdicts[n] - {"REFUSED"})
    print(f"\n# lanes selected {len(lanes)}, lanes with cells {len(covered)}, lanes actually "
          f"checked {ran}, shards {len(parts)}, {elapsed:.0f} s")
    for f in failures:
        print(f"# FAIL: {f}")
    if failures and merged.endswith("column.json") and os.path.exists(merged):
        # The refusal check above can only run AFTER the merge, so the name is
        # corrected here. `column.json` must never exist for an incomplete run.
        incomplete = os.path.join(out_dir, "column.incomplete.json")
        os.replace(merged, incomplete)
        merged = incomplete
    print(f"# verdict {'COMPLETE' if not failures else 'INCOMPLETE'} -> {merged}")
    if failures:
        print("# An incomplete run is NOT a pass. Rerun the shards that failed and merge again. "
              "A REFUSED lane needs its host family built, or its own GPU column; it is not "
              "evidence either way.")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="every registered lane")
    ap.add_argument("--lanes", default="", help="lanes by name, comma separated")
    ap.add_argument("--lane", action="append", default=[], help="one lane by name; repeatable")
    ap.add_argument("--changed-since", metavar="REF", help="the lanes the diff against REF can affect")
    ap.add_argument("--lanes-for-paths", nargs="+", metavar="PATH", help="the lanes these paths can affect")
    ap.add_argument("--shards", type=int, default=1, help="split into N deterministic shards (default 1)")
    ap.add_argument("--jobs", type=int, default=1, help="shards to run at once locally (default 1, the Mac rule)")
    ap.add_argument("--runner", choices=("local", "pods"), default="local",
                    help="local processes, or print one runpod_cpu_leg.sh command per shard")
    ap.add_argument("--plan", action="store_true", help="print what would run and stop")
    ap.add_argument("--out", default="", metavar="DIR", help="where parts, logs and the column go")
    ap.add_argument("--fixtures", default="", help="identity_break --fixtures (routine runs use base)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--vendor", default="")
    ap.add_argument("--tag", default="sweep", help="lane tag for the pod plan")
    ap.add_argument("--build", default="core,estimators", help="host families for the pod plan")
    ap.add_argument("extra", nargs="*", help="passed through to identity_break (after --)")
    args = ap.parse_args(argv)

    lanes, sel, _ = _selection(args)
    print(f"# {len(lanes)} of {len(lane_select.all_lanes())} lanes selected")
    if not lanes:
        print("# REFUSING: the selection is empty. An empty run is not a pass; if the change "
              "really touches no lane, say so in the lane status file rather than running this.")
        return 2
    groups, load = lane_select.shard(lanes, args.shards)
    out_dir = args.out or os.path.join(ROOT, "bench", "results", "lane_select",
                                       time.strftime("%Y-%m-%d_%H%M%S"))
    if not args.plan:
        os.makedirs(out_dir, exist_ok=True)
    manifest = dict(commit=_commit(), lanes=lanes, shards=[list(g) for g in groups], weights=load,
                    fixtures=args.fixtures, repeats=args.repeats, runner=args.runner,
                    registry_total=len(lane_select.all_lanes()), selection=sel.get("mode", "derived"),
                    fallback=bool(sel.get("fallback")))
    if args.plan:
        _plan(groups, load, args, out_dir)
        return 0
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    parts, codes, elapsed = _run_local(groups, load, args, out_dir)
    return _verdict(lanes, parts, codes, out_dir, elapsed)


if __name__ == "__main__":
    sys.exit(main())
