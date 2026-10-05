#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Score the MISSING opponents of a finished board, once, opponents only.

    <board venv python> tools/opp_only_board.py --family algos \\
        --races gamma:taxi,gamma:istella --tag opp-algos

The races are planned by tools/bench_board.py's own plan_races (the vendor's
roster, gpu_opponents_first, enforce_gpu_only) and run through its own
algos_cmd / more_cmd / classical_cmd / tree_cmd, run_logged and child_env, so
the driver, data, parameters and ceilings are the board's. The one change is
the arm list: only the planned OPPONENTS race (our arms are dropped). The
trees driver always runs `ours` (bench/speed/forest_speed_arm.py: `--arms`
filters opponents only); its ours lines are printed but are not a score.

Board root: ~/board-0834 (cache/venv, cache/{algos,more,ctd}-data/rows-full).
Our CPU never races. Output: one OPP line per opponent arm, plus the driver's
own result lines, and the raw JSON/logs under ~/mq/out/opp-<tag>/.
"""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_board():
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location("bench_board", os.path.join(HERE, "bench_board.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RESULT = {
    "algos": re.compile(r"^(ALGOS|ALGOS-REFUSED|ALGOS-INFER)\b.*"),
    "classical2": re.compile(r"^(MORE|MORE-REFUSED)\b.*"),
    "classical": re.compile(r"^(CTD|CTD-REFUSED|CTD-INFER|CTD-INFER-REFUSED)\b.*"),
    "trees": re.compile(r"^(FSPEED|FSPEED-ACC|FSPEED-REFUSED|FSPEED-FIT-VERDICT|FSPEED-INFER|FSPEED-WARMUP)\b.*"),
}


def tree_summary(log, lane, ds, opponents):
    ms, acc = {}, {}
    with open(log, errors="replace") as fh:
        for line in fh:
            m = re.match(r"^FSPEED lane=\S+ arm=(\S+) shape=\S+ round=\d+ ms=([\d.]+)", line)
            if m:
                ms.setdefault(m.group(1), []).append(float(m.group(2)))
            m = re.match(r"^FSPEED-ACC lane=\S+ arm=(\S+) (.*)", line)
            if m:
                acc.setdefault(m.group(1), []).append(m.group(2).strip()[:120])
    for arm in opponents:
        v = ms.get(arm, [])
        print("OPP family=trees lane=%s ds=%s arm=%s status=%s rounds=%d median_ms=%s all_ms=%s acc=%s"
              % (lane, ds, arm, "ok" if v else "none", len(v),
                 round(statistics.median(v), 1) if v else "none",
                 [round(x, 1) for x in v], (acc.get(arm) or ["none"])[-1]), flush=True)


def json_summary(path, fam, lane, ds, opponents):
    try:
        doc = json.load(open(path))
    except (OSError, ValueError) as exc:
        for arm in opponents:
            print("OPP family=%s lane=%s ds=%s arm=%s status=NO-JSON (%s)" % (fam, lane, ds, arm, exc))
        return
    arms = doc.get("arms") or {}
    for arm in opponents:
        a = arms.get(arm) or {}
        rounds = a.get("rounds_ms") or a.get("ms") or a.get("rounds") or []
        if isinstance(rounds, list):
            rounds = [r if isinstance(r, (int, float)) else (r or {}).get("ms") for r in rounds]
        med = a.get("median_ms")
        if med is None:
            nums = [r for r in rounds if isinstance(r, (int, float))] if isinstance(rounds, list) else []
            med = statistics.median(nums) if nums else None
        print("OPP family=%s lane=%s ds=%s arm=%s status=%s median_ms=%s quality=%s"
              % (fam, lane, ds, arm, a.get("status") or ("ok" if med is not None else "none"),
                 round(med, 3) if isinstance(med, (int, float)) else "none",
                 json.dumps((doc.get("quality") or {}).get(arm) or a.get("quality") or {}, sort_keys=True, default=str)[:300]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=("algos", "classical", "classical2", "trees"))
    p.add_argument("--races", required=True, help="lane:dataset,lane:dataset,...")
    p.add_argument("--tag", required=True)
    p.add_argument("--rounds", type=int, default=1)   # the M3 0.8.34 board ran --rounds 1
    p.add_argument("--board", default=os.path.expanduser("~/board-0834"))
    p.add_argument("--vendor", default="apple")
    # the M3 0.8.34 board's own caps (board-0834-kit/m3-0834-config.json): --round-seconds 300
    p.add_argument("--round-seconds", type=int, default=300)
    # trees: the same M3 board caps (--arm-budget-s 300 --race-deadline-s 900); an arm
    # that hits a cap is recorded as "> cap", a result
    p.add_argument("--arm-budget-s", type=int, default=300)
    p.add_argument("--race-deadline-s", type=int, default=900)
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--only-arms", default=None, help="comma list: race only these planned opponents (rerun of arms a race cap cut off)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    bb = load_board()
    cache = os.path.join(a.board, "cache")
    python = os.path.join(cache, "venv", "bin", "python")
    out = os.path.expanduser("~/mq/out/opp-%s" % a.tag)
    os.makedirs(out, exist_ok=True)
    pairs = [tuple(x.split(":", 1)) for x in a.races.split(",") if x.strip()]
    lanes = sorted({l for l, _ in pairs})
    dss = sorted({d for _, d in pairs})
    modes = ["identical"]
    planned = bb.plan_races(a.vendor, modes, families=[a.family], lanes=lanes, datasets=dss, rows=None)
    by = {(r["lane"], r["dataset"]): r for r in planned}
    missing = [pr for pr in pairs if pr not in by]
    if missing:
        print("OPP-PLAN-MISSING %s (planned: %s)" % (missing, sorted(by)), flush=True)
    arm_python = {}
    if a.family == "algos":
        for arm in bb.ALGOS.ARM_VENVS.get(a.vendor, {}):
            py = os.path.join(cache, "venv-" + arm, "bin", "python")
            if os.path.exists(py):
                arm_python[arm] = py
    ctx = {"vendor": a.vendor, "modes": modes, "python": python, "out": out, "rounds": a.rounds,
           "data_root": os.environ.get("GBM_BENCH_DATA",
                                       os.path.expanduser("~/datasets/gbm-bench")),
           "ctd_data": os.path.join(cache, "ctd-data"),
           "more_data": os.path.join(cache, "more-data"),
           "algos_data": os.path.join(cache, "algos-data"),
           "commit": bb.capture(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]) or "unknown",
           "arm_budget_s": a.arm_budget_s, "race_deadline_s": a.race_deadline_s,
           "round_seconds": a.round_seconds, "nice": 0, "ptxas": None, "infer": not a.no_infer,
           "tree_driver": os.path.join(REPO, "bench", "speed", "forest_speed_arm.py"),
           "classical_driver": os.path.join(HERE, "classical_two_datasets.py"),
           "more_driver": os.path.join(HERE, "bench_board_more.py"),
           "algos_driver": os.path.join(HERE, "bench_board_algos.py"),
           "neural_driver": os.path.join(HERE, "bench_board_neural.py"),
           "arm_python": arm_python}
    print("OPP-START tag=%s family=%s head=%s rounds=%d round_s=%d board=%s python=%s"
          % (a.tag, a.family, ctx["commit"], a.rounds, a.round_seconds, a.board, python), flush=True)
    for lane, ds in pairs:
        race = by.get((lane, ds))
        if race is None:
            continue
        opp = list(race["opponents"])
        if a.only_arms:
            opp = [x for x in opp if x in a.only_arms.split(",")]
        if not opp:
            print("OPP-NONE lane=%s ds=%s (no opponent planned for %s)" % (lane, ds, a.vendor))
            continue
        if any(bb._is_cpu_arm(x) for x in opp) and any(not bb._is_cpu_arm(x) for x in opp):
            raise SystemExit("CPU opponents beside GPU ones in %s" % race["id"])
        race = dict(race, arms=opp)       # ONLY the opponents race
        tag = "%s.%s" % (lane, ds)
        log = os.path.join(out, tag + ".log")
        extra = {}
        if a.family == "trees":
            cmd, extra = bb.tree_cmd(ctx, race)
            cmd = list(cmd) + ["--opponents-only"]   # never time ours here: our time is the board's
            ceiling = ctx["race_deadline_s"] + 900
        elif a.family == "algos":
            cmd, extra, ceiling = bb.algos_cmd(ctx, race)
        elif a.family == "classical2":
            cmd, extra, ceiling = bb.more_cmd(ctx, race)
        else:
            cmd, extra, ceiling = bb.classical_cmd(ctx, race)
        assert not any(x in ("ours", "ours-fast", "ours-cpu") for x in race["arms"])
        print("OPP-RACE %s arms=%s ceiling_s=%d" % (race["id"], ",".join(opp), ceiling), flush=True)
        if a.dry_run:
            print("OPP-CMD %s" % " ".join(cmd))
            continue
        rc = bb.run_logged(cmd, bb.child_env(ctx, extra), log, ceiling)
        print("OPP-RC %s rc=%d log=%s" % (race["id"], rc, log), flush=True)
        pat = RESULT[a.family]
        with open(log, errors="replace") as fh:
            for line in fh:
                if pat.match(line):
                    print("  " + line.rstrip()[:400])
        if a.family == "trees":
            tree_summary(log, lane, ds, opp)
        else:
            jp = {"algos": bb.algos_json_path, "classical2": bb.more_json_path,
                  "classical": bb.classical_json_path}[a.family](ctx, race)
            json_summary(jp, a.family, lane, ds, opp)
    print("OPP-DONE tag=%s out=%s" % (a.tag, out), flush=True)


if __name__ == "__main__":
    main()
