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
# A guard against a hang, not a target: a pass that runs out keeps every
# finished cell and the next invocation resumes after it.
APPLE_PASS_BUDGET = 3600


def last_release_tag():
    """The newest `v*` tag reachable from HEAD, or "" when there is none."""
    got = subprocess.run(["git", "-C", ROOT, "describe", "--tags", "--abbrev=0", "--match", "v*"],
                         capture_output=True, text=True)
    return got.stdout.strip() if got.returncode == 0 else ""


def pass_base_dir():
    return os.environ.get("MOJOLEARN_RELEASE_CHECK_DIR") or os.path.expanduser("~/mojolearn-evidence/release-check")


def pass_out_dir(backend):
    """Where a release pass keeps its records: outside the checkout, keyed by
    commit, so rerunning the same command at the same commit resumes."""
    return os.path.join(pass_base_dir(), (_commit() or "unknown")[:12], backend)


def _git_ok(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).returncode == 0


def _record(record_dir):
    """(manifest, summary) of a finished pass directory, or None."""
    try:
        manifest = json.loads((Path(record_dir) / "manifest.json").read_text())
        summary = json.loads((Path(record_dir) / "run-summary.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        return None
    return manifest, summary


def verified_anchor(backend, fixtures, base=None, depth=0, _commit_filter=None):
    """THE LAST COMMIT THIS BACKEND WAS VERIFIED AT, from the pass records on
    this machine, or None (2026-09-21).

    A release pass checks the lanes changed since an anchor. The anchor was
    the newest `v*` tag, and that tag was v0.8.8 for four releases because
    0.8.9 through 0.8.11 were cut under `alpha-api-*` tags: the 0.8.12 Apple
    pass diffed 2,328 paths and ran every lane. The anchor is now the last
    pass that actually FINISHED on this backend, which is the thing the tag
    stood in for.

    A record counts only when ALL of these hold, and anything unreadable
    means it does not:
      * `column.json` exists and `run-summary.json` says complete with no
        validation failures;
      * it checked the same fixtures, one fit, core probes, on this backend;
      * its tree was clean where it mattered: `dirty_lanes` (the lanes its
        own uncommitted and untracked paths selected) is empty, so the bits it
        verified are the bits of its commit;
      * its own coverage is either every lane (`covers.mode == "all"`) or the
        lanes changed since a release tag, or since ANOTHER record that
        passes this same test. The chain is followed to its base, so one
        narrow pass cannot vouch for another without a full or tag-anchored
        pass underneath;
      * its commit still exists in this repository.
    Records written before these fields existed carry no `covers` and are
    never used. The first pass after this change therefore anchors on the
    tag, as before, and every pass after it on the pass before."""
    base = base or pass_base_dir()
    if depth > 64 or not os.path.isdir(base):
        return None
    best = None
    for name in os.listdir(base):
        rec_dir = os.path.join(base, name, backend)
        got = _record(rec_dir)
        if got is None:
            continue
        manifest, summary = got
        commit = manifest.get("commit") or ""
        if _commit_filter is not None and commit != _commit_filter:
            continue
        if not (os.path.isfile(os.path.join(rec_dir, "column.json")) and summary.get("complete") is True
                and not summary.get("validation_failures")):
            continue
        if (manifest.get("backend") != backend or manifest.get("fixtures") != fixtures
                or manifest.get("repeats") != 1 or manifest.get("probe_group") != "core"):
            continue
        if manifest.get("metal_shards", 1) != 1:
            # Sharded Metal is unproven (tools/metal_fanout.py); such a record
            # is evidence for its own run, never an anchor for the next.
            continue
        covers = manifest.get("covers")
        if not isinstance(covers, dict) or manifest.get("dirty_lanes") != []:
            continue
        if not (len(commit) == 40 and _git_ok("cat-file", "-e", commit + "^{commit}")):
            continue
        if covers.get("mode") == "all":
            pass
        elif covers.get("mode") == "since-tag" and isinstance(covers.get("since"), str) and \
                _git_ok("cat-file", "-e", covers["since"] + "^{commit}"):
            pass
        elif covers.get("mode") == "since-record" and isinstance(covers.get("since"), str):
            if verified_anchor(backend, fixtures, base, depth + 1, _commit_filter=covers["since"]) is None:
                continue
        else:
            continue
        when = os.path.getmtime(os.path.join(rec_dir, "run-summary.json"))
        if best is None or when > best[0]:
            best = (when, commit)
    return best[1] if best else None


def dirty_paths():
    """Uncommitted and untracked paths in the checkout right now."""
    out = set()
    for args in (["diff", "--name-only", "HEAD"], ["ls-files", "--others", "--exclude-standard"]):
        got = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
        if got.returncode != 0:
            return None
        out |= {line.strip() for line in got.stdout.splitlines() if line.strip()}
    return sorted(out)
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
            if pending and getattr(args, "metal_shards", 1) > 1:
                # ONE Metal slot for the whole group: mac_slot holds the GPU
                # lock once and tools/metal_fanout.py starts every shard under
                # it. Per-shard exit codes come back through `codes`; a shard
                # the fan-out cannot account for is recorded as failed.
                spec_path = os.path.join(out_dir, "fanout.json")
                codes_path = os.path.join(out_dir, "fanout.codes.json")
                Path(codes_path).unlink(missing_ok=True)
                spec = dict(codes=codes_path, children=[
                    dict(cmd=_identity_break_cmd(groups[k], parts[k], args),
                         log=os.path.join(out_dir, f"part{k}.log")) for k in pending])
                Path(spec_path).write_text(json.dumps(spec, indent=1) + "\n")
                cmd = [sys.executable, os.path.join(ROOT, "tools/mac_slot.py"),
                       "--timeout", str(args.timeout), "--wait-timeout", str(args.wait_timeout),
                       "--deadline", str(args.deadline), "metal",
                       sys.executable, os.path.join(ROOT, "tools/metal_fanout.py"), spec_path]
                fh = open(os.path.join(out_dir, "fanout.log"), "w")
                group = list(pending)
                pending.clear()
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                running["fanout"] = (proc, fh)
                print(f"# {len(group)} Metal shards started under one slot: "
                      f"{', '.join(str(len(groups[k])) for k in group)} lanes", flush=True)
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
                    if k == "fanout":
                        codes.update(_fanout_codes(out_dir, len(groups), rc))
                    else:
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
            rc = proc.wait()
            fh.close()
            if k == "fanout":
                codes.update(_fanout_codes(out_dir, len(groups), rc))
            else:
                codes[k] = rc
            del running[k]
        report()
        for signum, handler in old.items():
            signal.signal(signum, handler)
    return parts, codes, time.monotonic() - args.started


def _fanout_codes(out_dir, count, rc):
    """Per-shard exit codes from tools/metal_fanout.py. Missing, unreadable or
    malformed codes are failures, and a non-zero fan-out with every shard at 0
    (the slot timed out or was interrupted) fails every shard."""
    try:
        got = json.loads(Path(out_dir, "fanout.codes.json").read_text())
        codes = {k: int(got[str(k)]) for k in range(count)}
    except (OSError, ValueError, KeyError, TypeError):
        return {k: rc or 1 for k in range(count)}
    if rc != 0 and all(c == 0 for c in codes.values()):
        return {k: rc for k in range(count)}
    return codes


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
    ap.add_argument("--metal-shards", type=int, default=1, metavar="N",
                    help="Metal only, OPT-IN and unproven: split the lanes into N processes that share "
                         "the one GPU under a single Metal slot (1 to 3; default 1). Concurrent Metal "
                         "has produced bad outputs before; see docs/RELEASE_CHECKLIST.md for the "
                         "comparison that must read zero differences before this becomes a default")
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
        # MOJOLEARN_CPU_PASS_SLOTS leaves room for a concurrent Apple pass
        # (tools/release_check.py): the Metal job holds one of the MAC_SLOTS.
        args.shards = args.jobs = int(os.environ.get("MOJOLEARN_CPU_PASS_SLOTS")
                                      or os.environ.get("MAC_SLOTS", "5"))
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
    if not 1 <= args.metal_shards <= 3:
        ap.error("--metal-shards must be 1, 2 or 3")
    if args.metal_shards > 1:
        if args.backend != "metal" or args.runner != "local":
            ap.error("--metal-shards splits a LOCAL Metal run; it needs --backend metal or --apple-pass")
        if args.shards not in (1, args.metal_shards):
            ap.error("--metal-shards sets the shard count; do not also pass a different --shards")
        args.shards = args.metal_shards
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
    is_pass = args.apple_pass or args.cpu_pass
    covers = dict(mode="all") if args.all else None
    if is_pass and selection_modes == 0:
        # Only what changed since this backend was last verified. The selector
        # widens to every lane by itself when a changed path cannot be
        # attributed, so this can only ever run too much, never too little.
        # The anchor is the last FINISHED pass on this backend when one
        # qualifies (`verified_anchor`), else the newest v* tag as before.
        anchor = verified_anchor(args.backend, args.fixtures)
        if anchor:
            args.changed_since = anchor
            covers = dict(mode="since-record", since=anchor)
            print(f"# release pass: lanes changed since {anchor[:12]}, the last completed "
                  f"{args.backend} pass on this machine")
        else:
            args.changed_since = last_release_tag()
            args.all = not args.changed_since
            if args.changed_since:
                tag_commit = subprocess.run(["git", "-C", ROOT, "rev-parse", args.changed_since + "^{commit}"],
                                            capture_output=True, text=True).stdout.strip()
                covers = dict(mode="since-tag", since=tag_commit, tag=args.changed_since) if tag_commit else None
            else:
                covers = dict(mode="all")
            print(f"# release pass: lanes changed since {args.changed_since} (no completed {args.backend} "
                  f"pass on this machine qualifies as an anchor)" if args.changed_since
                  else "# release pass: no v* tag found, so every lane")
        selection_modes = 1
    if is_pass:
        args.full_selection = True
    if selection_modes != 1:
        ap.error("choose one of --all, named lanes, --changed-since, or --lanes-for-paths")

    lanes, sel, sources = _selection(args)
    if sel.get("fallback"):
        covers = dict(mode="all")
    # CAN THIS RECORD ANCHOR THE NEXT PASS? Only if the bits it checks are its
    # commit's bits: the uncommitted and untracked paths must select no lane.
    dirty_lanes = None
    if is_pass and covers is not None:
        dirty = dirty_paths()
        if dirty is not None:
            dsel = lane_select.select(dirty, ref="HEAD", sources=sources) if dirty else dict(lanes=[], fallback=False)
            dirty_lanes = ["*"] if dsel["fallback"] else list(dsel["lanes"])
        if dirty_lanes == []:
            print("# this pass, once complete, anchors the next one on this backend")
        else:
            print(f"# this pass cannot anchor the next one: uncommitted or untracked paths select "
                  f"{'every lane' if dirty_lanes == ['*'] else dirty_lanes if dirty_lanes else 'an unknown set'}")
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
    if not lanes and is_pass:
        print(f"# nothing to check on {args.backend}: the release touched no lane that runs there")
        return 0
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
    out_dir = args.out or (pass_out_dir(args.backend) if is_pass else
                           os.path.join(ROOT, "bench", "results", "lane_select", time.strftime("%Y-%m-%d_%H%M%S")))
    if is_pass and (Path(out_dir) / "manifest.json").exists():
        args.resume = True
        print(f"# resuming the records in {out_dir}")
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
                    fallback=bool(sel.get("fallback")), covers=covers, dirty_lanes=dirty_lanes,
                    metal_shards=args.metal_shards)
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
