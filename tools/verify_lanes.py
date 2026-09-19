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

WHERE IT RUNS. Local runs default to the CPU-only route, base fixture and
core probes. Explicit --jobs enables CPU shard parallelism within shared
capacity. All children use one scheduler, per-shard limits and a shared total
deadline. --backend cuda/hip/metal requires that actual loaded backend and
serializes GPU jobs per host. Separate GPU hosts may work independently.
--runner pods only PRINTS bounded CPU commands, even without --plan; it never
rents or falls through to local execution. See docs/TEST_RUNTIME.md.

"""
import argparse
import json
import math
from pathlib import Path
import signal
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

ROOT = lane_select.ROOT

# The Apple pass. `base` is the ordinary path, `denormal` is where Apple is
# known to differ (it flushes subnormals the other vendors keep) and `odd`
# reaches every tile tail. The other six fixtures vary properties the CPU
# column already carries on all nine. Measured 2026-09-18 on the M4: train
# plus infer/save/reload is 24% of a fully probed cell, so this selection at
# one fit is about 1/25 of the two-fit, nine-fixture, every-probe column.
APPLE_PASS_FIXTURES = "base,denormal,odd"
APPLE_PASS_BUDGET = 600
#: RunPod CPU, 16 vCPU, 2026-09-16. The 8 vCPU figure in docs/RUNPOD_CPU_LEG.md
#: is $0.24/h; a 16 vCPU pod is about twice that. Printed with a plan so the
#: cost of a full sweep is a number before anyone rents anything.
POD_USD_PER_HOUR = 0.48


def _selection(args):
    if args.all or args.lanes or args.lane:
        registry = lane_select.all_lanes()
        if args.all:
            return list(registry), dict(mode="all", fallback=False), None
        named = [n for n in (args.lanes or "").split(",") if n] + list(args.lane)
        unknown = [n for n in named if n not in registry]
        if unknown:
            raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}")
        return [n for n in registry if n in set(named)], dict(mode="named", fallback=False), None
    sources, why = lane_select.lane_sources()
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
    cmd += ["--fail-on-refused", "--require-backend", args.backend]
    if args.probe_group in ("core", "rlpair"):
        cmd.append("--no-batch")
    if args.probe_group in ("core", "batch"):
        cmd.append("--no-rlpair")
    if args.resume and os.path.exists(out):
        cmd.append("--resume")
    return cmd + args.extra


def _plan(groups, load, args, out_dir):
    print(f"# backend={args.backend} jobs={args.jobs} budget={args.budget}s shard-timeout={args.timeout}s queue-timeout={args.wait_timeout}s")
    total = 0
    for k, group in enumerate(groups):
        part = os.path.join(out_dir, f"part{k}.json")
        if args.runner == "pods":
            import shlex
            inner = ["python3", "tools/verify_lanes.py", "--lanes", ",".join(group),
                     "--backend", "cpu", "--fixtures", args.fixtures, "--probe-group", args.probe_group,
                     "--repeats", str(args.repeats),
                     "--budget", str(args.budget), "--timeout", str(args.timeout),
                     "--wait-timeout", str(args.wait_timeout), "--out"]
            command = shlex.join(inner) + f' "$LEG_OUT/shard{k}"'
            print(f"\n# shard {k}: {len(group)} lanes, weight {load[k]} s")
            print("bash tools/runpod_cpu_leg.sh --lane " + shlex.quote(f"{args.tag}-s{k}") +
                  " --rent --build " + shlex.quote(args.build) + " --envs default --cmd " + shlex.quote(command))
        else:
            print(f"# shard {k}: {len(group)} lanes, weight {load[k]} s -> {part}")
            print("  # inner workload; executed through the scheduler with the shared deadline")
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
    import identity_iterate
    parts = [os.path.join(out_dir, f"part{k}.json") for k in range(len(groups))]
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    if args.backend == "metal":
        # the harness refuses more than one Metal lane unless told otherwise
        import identity_break
        env[identity_break.APPLE_FULL_DIAGNOSTIC_ENV] = "1"
    env["PYTHONPATH"] = os.path.join(ROOT, "python") + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "cpu":
        package, host = identity_iterate.cpu_package(Path(out_dir),
            args.host_dir or env.get("MOJOLEARN_HOST_DIR") or Path(ROOT) / "python/mojolearn/host")
        env["PYTHONPATH"] = str(package) + os.pathsep + env["PYTHONPATH"]
        env["MOJOLEARN_HOST_DIR"] = str(host)
    pending, running, codes = list(range(len(groups))), {}, {}
    def report():
        dest = Path(out_dir) / "run-summary.json"
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(pending=pending, running=list(running), exit_codes=codes,
            complete=False, execution_complete=len(codes) == len(groups) and all(c == 0 for c in codes.values()),
            budget=args.budget, elapsed_seconds=time.monotonic() - args.started), indent=2) + "\n")
        tmp.replace(dest)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        report()
        while pending or running:
            if time.monotonic() >= args.deadline:
                break
            while pending and len(running) < args.jobs and time.monotonic() < args.deadline:
                k = pending.pop(0)
                fh = open(os.path.join(out_dir, f"part{k}.log"), "w")
                mode = "run" if args.backend == "cpu" else args.backend
                cmd = [sys.executable, os.path.join(ROOT, "tools/mac_slot.py"),
                       "--timeout", str(args.timeout), "--wait-timeout", str(args.wait_timeout),
                       "--deadline", str(args.deadline), mode, *_identity_break_cmd(groups[k], parts[k], args)]
                try:
                    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                except BaseException:
                    fh.close()
                    pending.insert(0, k)
                    raise
                running[k] = (proc, fh)
                print(f"# shard {k} started: {len(groups[k])} lanes ({args.backend})", flush=True)
            for k, (proc, fh) in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    fh.close()
                    codes[k] = rc
                    del running[k]
            report()
            if any(c != 0 for c in codes.values()):
                break
            if pending or running:
                time.sleep(min(.05, max(0, args.deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("# interrupted; unfinished shards are not coverage", flush=True)
    finally:
        for signum in old:
            signal.signal(signum, signal.SIG_IGN)
        # Signal the scheduler, which drains its child's process group before
        # releasing the lease. Do not kill only the scheduler and orphan work.
        for proc, _ in running.values():
            if proc.poll() is None:
                proc.terminate()
        for k, (proc, fh) in list(running.items()):
            codes[k] = proc.wait()
            fh.close()
            del running[k]
        report()
        for signum, handler in old.items():
            signal.signal(signum, handler)
    return parts, codes, time.monotonic() - args.started


def _verdict(lanes, parts, codes, out_dir, elapsed, deadline=None, fixtures=None):
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
    if present and (deadline is None or time.monotonic() < deadline):
        try:
            res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "identity_break.py"),
                                  "--merge"] + present + ["--json", merged], cwd=ROOT,
                                 timeout=min(60, max(.001, deadline - time.monotonic())) if deadline else 60)
            if res.returncode != 0:
                failures.append(f"identity_break --merge exited {res.returncode}")
        except subprocess.TimeoutExpired:
            failures.append("result merge exceeded the remaining run budget")
    else:
        failures.append("no merge: no part records or total run budget exhausted")
    covered, verdicts, cells = set(), {}, {}
    if os.path.exists(merged):
        try:
            record = json.loads(Path(merged).read_text())
            if not isinstance(record, dict) or not isinstance(record.get("cells"), dict):
                raise ValueError("expected a record with a cells object")
            cells = record["cells"]
            if record.get("complete") is not True:
                failures.append("merged record is not marked complete")
        except (OSError, ValueError) as exc:
            failures.append(f"invalid merged record: {exc}")
        for key, cell in cells.items():
            if not isinstance(cell, dict) or cell.get("verdict") != "STABLE":
                failures.append(f"{key}: missing or non-STABLE training verdict")
            if not isinstance(cell, dict):
                continue
            covered.add(key.split("/")[0])
            for field, value in cell.items():
                if field.endswith("_verdict") and value not in ("STABLE", "N/A"):
                    failures.append(f"{key}: {field}={value}")
            verdicts.setdefault(key.split("/")[0], set()).add(str(cell.get("verdict")))
    if fixtures is not None:
        expected = {f"{lane}/{fixture}" for lane in lanes for fixture in fixtures}
        missing_cells, extra_cells = sorted(expected - set(cells)), sorted(set(cells) - expected)
        if missing_cells:
            failures.append(f"missing requested cells: {missing_cells}")
        if extra_cells:
            failures.append(f"unexpected cells: {extra_cells}")
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
    ran = sum(1 for n in lanes if verdicts.get(n, set()) & {"STABLE", "MOVED"})
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
    summary = Path(out_dir) / "run-summary.json"
    if summary.exists():
        result = json.loads(summary.read_text())
        result.update(complete=not failures, validation_failures=failures)
        tmp = summary.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(summary)
    return 1 if failures else 0



def validate_resume(out_dir, manifest):
    """Reject changed work before replacing manifests, logs or merged results."""
    path = Path(out_dir) / "manifest.json"
    if not path.exists():
        if any(Path(out_dir).glob("part*.json")):
            raise ValueError("cannot resume part records without their manifest")
        return
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read resume manifest: {exc}") from exc
    if not isinstance(previous, dict):
        raise ValueError("invalid resume manifest")
    fields = ("commit", "lanes", "shards", "fixtures", "repeats", "backend", "probe_group")
    changed = [name for name in fields if previous.get(name) != manifest.get(name)]
    if changed:
        raise ValueError(f"resume scope changed: {', '.join(changed)}; use a new output directory")


def main(argv=None):
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="every registered lane")
    ap.add_argument("--lanes", default="", help="lanes by name, comma separated")
    ap.add_argument("--lane", action="append", default=[], help="one lane by name; repeatable")
    ap.add_argument("--changed-since", metavar="REF", help="the lanes the diff against REF can affect")
    ap.add_argument("--lanes-for-paths", nargs="+", metavar="PATH", help="the lanes these paths can affect")
    ap.add_argument("--shards", type=int, default=1, help="split into N deterministic shards (default 1)")
    ap.add_argument("--jobs", type=int, default=1, help="CPU shards to run concurrently (default 1); GPU hosts require 1")
    ap.add_argument("--backend", choices=("cpu", "metal", "cuda", "hip"), default="cpu")
    ap.add_argument("--host-dir", help="prebuilt CPU bindings; never build or fall back to GPU")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--wait-timeout", type=float, default=60)
    ap.add_argument("--probe-group", choices=("core", "batch", "rlpair", "all"), default="core")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-selection", action="store_true", help="explicitly accept selector fallback")
    ap.add_argument("--metal-diagnostic", action="store_true")
    ap.add_argument("--cpu-pass", action="store_true",
                    help="the CPU half of a release, locally: the Apple pass's cells on the CPU route, "
                         "one shard per Mac CPU slot, nothing rented")
    ap.add_argument("--apple-pass", action="store_true",
                    help="the whole Apple check: Metal, one fit per cell, the end model only "
                         f"(train, infer, save/reload), fixtures {APPLE_PASS_FIXTURES}, "
                         f"a {APPLE_PASS_BUDGET}-second total budget")
    ap.add_argument("--runner", choices=("local", "pods"), default="local",
                    help="local processes, or print one runpod_cpu_leg.sh command per shard")
    ap.add_argument("--plan", action="store_true", help="print what would run and stop")
    ap.add_argument("--out", default="", metavar="DIR", help="where parts, logs and the column go")
    ap.add_argument("--fixtures", default=None, help="fixtures to check; default base, even with --all")
    ap.add_argument("--exhaustive", action="store_true", help="all nine fixtures, explicitly requested")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--vendor", default="")
    ap.add_argument("--tag", default="sweep", help="lane tag for the pod plan")
    ap.add_argument("--build", default="core,estimators", help="host families for the pod plan")
    ap.add_argument("extra", nargs="*", help="legacy extra arguments are rejected; use explicit scope controls")
    args = ap.parse_args(argv)
    if args.cpu_pass:
        # The CPU half of a release, on this machine: the same cells as the
        # Apple pass, spread over the Mac's CPU slots. Nothing is rented.
        if args.apple_pass or args.probe_group != "core" or args.repeats != 1 or args.exhaustive:
            ap.error("--cpu-pass is CPU, core probes, one fit; it takes --fixtures and --budget only")
        args.backend = "cpu"
        args.fixtures = args.fixtures if args.fixtures is not None else APPLE_PASS_FIXTURES
        args.budget = args.budget if args.budget is not None else APPLE_PASS_BUDGET
        args.timeout = args.wait_timeout = args.budget
        args.shards = args.jobs = int(os.environ.get("MAC_SLOTS", "5"))
    if args.apple_pass:
        # The batch and decode probes check batching logic, which the CPU and
        # NVIDIA columns carry; here Metal answers one question, whether its
        # end model is the reference's. Anything that would widen that refuses.
        if args.backend not in ("cpu", "metal") or args.probe_group != "core" or args.repeats != 1 or args.exhaustive:
            ap.error("--apple-pass is Metal, core probes, one fit; it takes --fixtures and --budget only")
        args.backend = "metal"
        args.fixtures = args.fixtures if args.fixtures is not None else APPLE_PASS_FIXTURES
        args.budget = args.budget if args.budget is not None else APPLE_PASS_BUDGET
        args.timeout = args.wait_timeout = args.budget
    args.budget = args.budget if args.budget is not None else (60 if args.backend == "metal" else 300)
    args.started, args.deadline = started, started + args.budget
    if any(not math.isfinite(v) or v <= 0 for v in (args.budget, args.timeout, args.wait_timeout)):
        ap.error("budgets and timeouts must be finite and positive")
    if args.jobs < 1 or args.shards < 1:
        ap.error("jobs and shards must be positive")
    if args.jobs > int(os.environ.get("MAC_SLOTS", "5")):
        ap.error("--jobs exceeds shared CPU capacity (MAC_SLOTS, default 5)")
    if args.backend != "cpu" and args.jobs != 1:
        ap.error("one GPU job per host; parallelize separate GPU hosts, or use --backend cpu --jobs N")
    if args.runner == "pods" and args.backend != "cpu":
        ap.error("the pod planner is CPU-only; run GPU jobs on explicitly provisioned GPU hosts")
    if args.extra:
        ap.error("unrestricted extra arguments can override scope; use the explicit scope options")
    import identity_break
    if args.exhaustive and args.fixtures is not None:
        ap.error("choose --exhaustive OR --fixtures")
    args.fixtures = ",".join(identity_break.FIXTURES) if args.exhaustive else (args.fixtures if args.fixtures is not None else "base")
    fixtures = args.fixtures.split(",")
    unknown = set(fixtures) - set(identity_break.FIXTURES)
    if unknown:
        ap.error(f"unknown fixtures: {sorted(unknown)}")
    # One fit per cell. A hash equal to the reference another box produced
    # already shows this box neither moved nor diverged; a second fit only
    # classifies a mismatch, so rerun the DIVERGENT cell, never the selection.
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    selection_modes = sum(bool(x) for x in (args.all, args.lanes or args.lane, args.changed_since, args.lanes_for_paths))
    if selection_modes != 1:
        ap.error("choose one of --all, named lanes, --changed-since, or --lanes-for-paths")

    lanes, sel, _ = _selection(args)
    if (args.apple_pass or args.cpu_pass) and not (args.lanes or args.lane):
        # A lane whose arithmetic never reaches this backend says nothing
        # here. Named lanes still refuse below; a derived selection drops
        # them, out loud.
        import lane_applicability
        skip = lane_applicability.degenerate("apple-metal" if args.apple_pass else "cpu-host")
        dropped = [n for n in lanes if n in skip]
        lanes = [n for n in lanes if n not in skip]
        if dropped:
            print(f"# leaving out {len(dropped)} lane(s) that cannot run on {args.backend}: {','.join(dropped)}")
    print(f"# {len(lanes)} of {len(lane_select.all_lanes())} lanes selected")
    print(f"# {len(fixtures)} fixture(s): {args.fixtures}; "
          f"{len(lanes) * len(fixtures)} cells, {len(lanes) * len(fixtures) * args.repeats} independent fits")
    if not lanes:
        print("# REFUSING: the selection is empty. An empty run is not a pass; if the change "
              "really touches no lane, say so in the lane status file rather than running this.")
        return 2
    if args.probe_group in ("batch", "rlpair"):
        unsupported = [n for n in lanes if
                       (not callable(identity_break.BATCH.get(n)) if args.probe_group == "batch"
                        else n not in identity_break.RLPAIR)]
        if unsupported:
            ap.error(f"no applicable {args.probe_group} probe for: {unsupported}")
    groups, load = lane_select.shard(lanes, args.shards)
    out_dir = args.out or os.path.join(ROOT, "bench", "results", "lane_select",
                                       time.strftime("%Y-%m-%d_%H%M%S"))
    if not args.plan and args.runner != "pods":
        if sel.get("fallback") and not args.full_selection:
            ap.error("selector fell back to every lane; inspect --plan or pass --full-selection")
        import lane_applicability
        column = dict(cpu="cpu-host", metal="apple-metal", cuda="nvidia-1gpu", hip="amd-1gpu")[args.backend]
        try:
            lane_applicability.check(lanes, column)
        except lane_applicability.LaneNotApplicable as exc:
            ap.error(str(exc))
        if not args.resume and any(Path(out_dir).glob("part*.json")):
            ap.error("records already exist; use --resume or a new output directory")
    manifest = dict(commit=_commit(), lanes=lanes, shards=[list(g) for g in groups], weights=load,
                    fixtures=args.fixtures, repeats=args.repeats, runner=args.runner, backend=args.backend,
                    probe_group=args.probe_group, budget=args.budget, timeout=args.timeout, jobs=args.jobs,
                    registry_total=len(lane_select.all_lanes()), selection=sel.get("mode", "derived"),
                    fallback=bool(sel.get("fallback")))
    if args.plan or args.runner == "pods":
        _plan(groups, load, args, out_dir)
        return 0
    if args.resume:
        try:
            validate_resume(out_dir, manifest)
        except ValueError as exc:
            ap.error(str(exc))
    os.makedirs(out_dir, exist_ok=True)
    for filename in ("column.json", "column.incomplete.json"):
        Path(out_dir, filename).unlink(missing_ok=True)
    manifest_path = Path(out_dir) / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    temporary.replace(manifest_path)
    try:
        parts, codes, elapsed = _run_local(groups, load, args, out_dir)
    except ValueError as exc:
        ap.error(str(exc))
    return _verdict(lanes, parts, codes, out_dir, elapsed, args.deadline, fixtures)


if __name__ == "__main__":
    sys.exit(main())
