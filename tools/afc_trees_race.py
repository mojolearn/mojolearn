#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The board's trees race for ONE arm of ours, with a bench_board_algos.py
style summary line, so tools/afc_ab.sh can A/B a trees lane (AFC_FAMILY=trees).

    afc_trees_race.py race --lane gbdt-symmetric --dataset istella \
        --data ~/datasets/gbm-bench --arms ours-fast --rounds 1 --out D --work W

The command and its environment come from tools/bench_board.py's own
`tree_cmd` (bench/speed/forest_speed_arm.py --ours-only --mem,
MOJOLEARN_SPEED_SIZE=shipped, rows full, one untimed warm-up then --rounds
timed rounds); the log is read with bench_board's `parse_tree_log`.
--arms ours-fast races the FAST tier, --arms ours the IDENTICAL tier (the
driver's arm is `ours` either way; MOJOLEARN_NUMERIC_MODE picks the tier).
--data is the GBM_BENCH_DATA root (taxi/, istella/). Unlike the board, the
environment is inherited (PYTHONPATH and MOJOLEARN_BENCH_INSTALLED=0 from
afc_ab.sh keep the built tree's bindings in use).

Prints the driver's output, then one line:
    TREES lane=L dataset=D arm=A status=S median_ms=M digest=H quality={...}
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _board():
    spec = importlib.util.spec_from_file_location("afc_bench_board", os.path.join(HERE, "bench_board.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summary(lane, ds, arm, status, ms, digest, quality):
    print("TREES lane=%s dataset=%s arm=%s status=%s median_ms=%s digest=%s quality=%s"
          % (lane, ds, arm, status, ms, digest or "none",
             json.dumps(quality, sort_keys=True, separators=(",", ":"))), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("race",))
    p.add_argument("--lane", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data", required=True, help="GBM_BENCH_DATA root")
    p.add_argument("--arms", default="ours-fast", choices=("ours-fast", "ours"))
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None, help="accepted for afc_ab.sh; unused")
    p.add_argument("--vendor", default="apple", choices=("apple", "nvidia", "amd"))
    a = p.parse_args(argv)
    BB = _board()
    lanes = BB.TREE_LANES + BB.TREE_TASK_LANES
    if a.lane not in lanes:
        summary(a.lane, a.dataset, a.arms, "REFUSED(not a trees lane)", None, None, {})
        return 2
    if not BB.tree_task_datasets(a.lane, [a.dataset]):
        summary(a.lane, a.dataset, a.arms, "REFUSED(no %s task on %s)" % (a.lane, a.dataset), None, None, {})
        return 2
    mode = "fast" if a.arms == "ours-fast" else "identical"
    os.makedirs(a.out, exist_ok=True)
    ctx = {"python": sys.executable, "tree_driver": os.path.join(BB.REPO, "bench", "speed", "forest_speed_arm.py"),
           "vendor": a.vendor, "rounds": a.rounds, "arm_budget_s": 3600, "race_deadline_s": 6 * 3600,
           "infer": False}
    race = {"our_arms": {"ours": mode}, "lane": a.lane, "dataset": a.dataset, "rows": None,
            "opponents": []}
    cmd, extra = BB.tree_cmd(ctx, race)
    env = dict(os.environ)
    env.update(extra)
    env["GBM_BENCH_DATA"] = os.path.abspath(os.path.expanduser(a.data))
    env["MOJOLEARN_BOARD_VENDOR"] = a.vendor
    env["PYTHONUNBUFFERED"] = "1"
    log = os.path.join(a.out, "tree.log")
    print("TREES-CMD %s" % " ".join(cmd), flush=True)
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    with open(log, errors="replace") as fh:
        sys.stdout.write(fh.read())
    parsed = BB.parse_tree_log(log)
    rec = parsed["arms"].get("ours")
    if rec is None:
        summary(a.lane, a.dataset, a.arms, "UNKNOWN(not in log, rc=%d)" % rc, None, None, {})
        return 1
    ms = rec["ms"]
    status = BB._status(ms, rec["refused"], a.rounds)
    b = parsed["bindings"].get("ours") or {}
    if b and (b.get("compiled") != mode or b.get("resolved") != mode):
        status = "MODE-MISMATCH(requested %s, compiled %s)" % (mode, b.get("compiled"))
    med = float(statistics.median(ms)) if ms else None
    digest = rec["hashes"][-1] if rec["hashes"] else None
    summary(a.lane, a.dataset, a.arms, status, med, digest, dict(rec["acc"]))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
