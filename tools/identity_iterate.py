#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Run changed identity lanes in fair, bounded jobs with live progress.

The default base fixture is an iteration check, never a full release record.
Every cell still runs both fits and all its default inference/batch/state
checks. Each lane/fixture releases the GPU before the next joins the queue.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(paths, base, fixtures, lanes=()):
    import lane_select
    import identity_break
    if lanes and paths:
        raise ValueError("choose named lanes OR changed paths, not both")
    if lanes:
        unknown = set(lanes) - set(identity_break.LANES)
        if unknown:
            raise ValueError(f"unknown lanes: {sorted(unknown)}")
        # Naming one algorithm does not need the whole dependency graph.
        selection = dict(lanes=list(dict.fromkeys(lanes)), fallback=False,
                         reasons={name: "explicit lane" for name in lanes}, unattributed=[])
    else:
        selection = lane_select.select(paths if paths else lane_select.changed_paths(base), ref=base)
    unknown = set(fixtures) - set(identity_break.FIXTURES)
    if unknown:
        raise ValueError(f"unknown fixtures: {sorted(unknown)}")
    return selection


def command(python, lane, fixture, record, timeout, mode, resume):
    cmd = [python, str(ROOT / "tools/mac_slot.py"), "--timeout", str(timeout),
           "--timing-json", str(record.with_suffix(".timing.json")), mode,
           python, "-u", str(ROOT / "tools/identity_break.py"), "--lanes", lane,
           "--fixtures", fixture, "--repeats", "2", "--fail-on-refused", "--json", str(record)]
    if resume and record.exists():
        cmd.append("--resume")
    return cmd


def run_job(cmd, env):
    child = subprocess.Popen(cmd, env=env)
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
    old = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = child.wait()
        return 128 - code if code < 0 else code
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="changed source paths; otherwise diff --base plus local changes")
    ap.add_argument("--base", default="HEAD^")
    ap.add_argument("--lane", action="append", default=[], help="one algorithm lane; repeat to select several")
    ap.add_argument("--lanes", default="", help="comma-separated algorithm lanes")
    ap.add_argument("--exhaustive", action="store_true", help="all nine fixtures for the selected lanes")
    ap.add_argument("--fixtures", default=None, help="comma-separated fixture names; default base is iteration coverage")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plan", action="store_true", help="print selection without taking any slot")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout", type=float, default=900, help="maximum seconds per cell, excluding wait (default 900)")
    ap.add_argument("--mode", choices=("metal", "run"), default="metal" if sys.platform == "darwin" else "run")
    ap.add_argument("--full-selection", action="store_true", help="explicitly run a selector fallback; Apple release policy still applies")
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    import identity_break
    if args.exhaustive and args.fixtures is not None:
        ap.error("choose --exhaustive OR --fixtures")
    fixture_names = ",".join(identity_break.FIXTURES) if args.exhaustive else (args.fixtures if args.fixtures is not None else "base")
    fixtures = list(dict.fromkeys(filter(None, fixture_names.split(","))))
    if not fixtures:
        ap.error("at least one fixture is required")
    lanes = args.lane + [name for name in args.lanes.split(",") if name]
    try:
        selected = plan(args.paths, args.base, fixtures, lanes) if lanes else plan(args.paths, args.base, fixtures)
    except ValueError as exc:
        ap.error(str(exc))
    selected["fixtures"] = fixtures
    selected["repeats"] = 2
    selected["cell_count"] = len(selected["lanes"]) * len(fixtures)
    selected["fit_count"] = selected["cell_count"] * 2
    print(json.dumps(selected, indent=2), flush=True)
    if args.plan:
        return 0
    if selected["fallback"] and not args.full_selection:
        ap.error("selection fell back to all lanes; inspect the plan or explicitly pass --full-selection")
    # Splitting into processes must not bypass the harness's release guard.
    import identity_break
    if args.mode == "metal":
        refusal = identity_break.refuse_routine_apple_column(selected["lanes"], host=None)
        if refusal:
            ap.error(refusal)
    if not selected["lanes"]:
        print("No affected identity lanes. No GPU work requested.")
        return 0
    if not args.out:
        ap.error("--out is required for checkpointed execution")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    for lane in selected["lanes"]:
        for fixture in fixtures:
            record = args.out.resolve() / f"{lane}--{fixture}.json"
            if record.exists() and not args.resume:
                ap.error(f"{record} exists; use --resume or a new output directory")
            print(f"# QUEUE {lane}/{fixture}", flush=True)
            code = run_job(command(sys.executable, lane, fixture, record,
                                   args.timeout, args.mode, args.resume), env)
            if code:
                return code
    print("Selected iteration cells passed. This is not a full release qualification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
