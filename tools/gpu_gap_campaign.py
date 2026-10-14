#!/usr/bin/env python3
"""Run an explicit list of missing GPU recipes, serially on one frozen worker.

The caller provisions a bounded worker and an earlier process watchdog. This
controller never rents, extends or deletes machines. It does not select work
from timing ratios and never retries a completed recipe implicitly.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import bench_board as B
import bench_board_evidence as E

ROOT = Path(__file__).resolve().parent.parent


def selected_plan(vendor, manifest):
    if vendor not in ("nvidia", "amd"):
        raise ValueError("this campaign authorizes NVIDIA and AMD only")
    wanted = manifest[vendor]
    if not isinstance(wanted, list) or not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("supply a nonempty list of unique recipe IDs")
    recipes = {r["id"]: r for r in B.gpu_opponents_only(B.plan_races(vendor, ["identical"]), vendor)
               if r["opponents"]}
    missing = set(wanted) - set(recipes)
    if missing:
        raise ValueError("unknown GPU recipes: " + ", ".join(sorted(missing)))
    return [recipes[rid] for rid in wanted]


def command(args, race, directory):
    cmd = [args.python_env, str(ROOT / "tools/bench_board.py"),
           "--vendor", args.vendor, "--python-env", args.python_env, "--skip-install",
           "--cache", args.cache, "--rows", "full", "--rounds", "1", "--modes", "identical",
           "--gpu-opponents-only", "--retime-opponents", "--verify-data", "--no-smoke-gate",
           "--round-seconds", "240", "--arm-budget-s", "300", "--race-deadline-s", "600",
           "--families", race["family"], "--lanes", race["lane"], "--out", str(directory)]
    if race["dataset"] in B.DATASETS:
        cmd += ["--datasets", race["dataset"]]
    return cmd


def logged(cmd, log):
    with Path(log).open("ab", buffering=0) as stream:
        return subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def prepare_fixtures(args, plan, out):
    # FAIRNESS FIX (2026-10-13 audit): earlier full-plan audits staged tabular
    # data but no layer fixtures. Generate test inputs with the pinned worker
    # torch, verify every byte and retain them before running an own GPU arm.
    A = B.ALGOS
    lanes = sorted({r["lane"] for r in plan if r["family"] == "algos"
                    and A.LANES[r["lane"]]["kind"] == "layer"})
    if not lanes:
        return None
    data = Path(args.cache) / "algos-data/rows-full"
    fixtures = Path(args.cache) / "algos-data/neural-fixtures"
    commands = [
        [args.python_env, str(ROOT / "tools/bench_board_algos.py"), "prep",
         "--data", str(data), "--lanes", ",".join(lanes), "--datasets", "taxi,istella"],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "generate",
         "--out", str(fixtures), "--data", str(data), "--lanes", ",".join(lanes)],
        [args.python_env, str(ROOT / "tools/neural_fixtures.py"), "verify", "--dir", str(fixtures)],
    ]
    for cmd in commands:
        if logged(cmd, out / "fixture-preparation.log"):
            raise RuntimeError("fixture preparation failed; no measurements started")
    archive = out / "prepared-fixtures.tar.gz"
    if logged(["tar", "-czf", str(archive), "-C", str(fixtures.parent), fixtures.name], out / "fixture-retention.log"):
        raise RuntimeError("fixture archive failed")
    ref = E.upload(dict(local_path=str(archive), sha256=E.file_hash(archive), bytes=archive.stat().st_size))
    E.write(out / "fixture-retention.json", ref)
    os.environ["MOJOLEARN_NEURAL_FIXTURES"] = str(fixtures)
    return E.public_ref(ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor", choices=("nvidia", "amd"), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--python-env", default=sys.executable)
    ap.add_argument("--stop-at", type=float, required=True,
                    help="epoch after which no new recipe starts; leave retention time before provider deadline")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = selected_plan(args.vendor, json.loads(Path(args.manifest).read_text()))
    os.environ["MOJOLEARN_REPO_COMMIT"] = B.repo_commit()
    path = out / "campaign.json"
    if path.exists():
        raise ValueError("campaign directory already exists; preserve it and select remaining work in a new manifest")
    state = dict(vendor=args.vendor, source_commit=B.repo_commit(), started=time.time(),
                 stop_at=args.stop_at, expected=len(plan),
                 scope="explicit missing-evidence recipes; one excluded warmup and one scored sample; no Apple",
                 recipes={r["id"]: dict(status="pending", recipe=r) for r in plan})
    E.write(path, state)
    if args.plan_only:
        return 0
    state["fixture_artifacts"] = prepare_fixtures(args, plan, out)
    E.write(path, state)
    for race in plan:
        if time.time() + 600 >= args.stop_at:
            state["stop_reason"] = "insufficient time for another recipe and retention"
            break
        directory = out / "measured" / race["id"].replace("/", ".")
        row = state["recipes"][race["id"]]
        cmd = command(args, race, directory)
        row.update(status="running", started=time.time(), command=cmd)
        E.write(path, state)
        rc = logged(cmd, out / (race["id"].replace("/", ".") + ".log"))
        boardpath = directory / "board.json"
        board = json.loads(boardpath.read_text()) if boardpath.exists() else {}
        row.update(status="process finished", returncode=rc, finished=time.time(), board=str(boardpath),
                   preflight=(board.get("preflight") or {}).get("verdict"),
                   results={rid: {"status": r.get("status"), "audit": r.get("audit"),
                                  "evidence_failure": r.get("evidence_failure")}
                            for rid, r in board.get("races", {}).items()})
        E.write(path, state)
        if any(r.get("evidence_failure") for r in board.get("races", {}).values()):
            state["stop_reason"] = "retention failed; retry upload without rerunning completed measurements"
            break
    state["finished"] = time.time()
    E.write(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
