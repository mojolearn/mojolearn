#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane gbdt-train-speed: one GradientBoosting TRAINING cell, timed at several
tree counts in one process, with the prediction digest of every fit.

    python3 tools/gbdt_train_probe.py fit --cell yeti --trees 10,50 --reps 3 --json out.json
    python3 tools/gbdt_train_probe.py fit --cell gbdt-lossguide --dataset taxi --trees 10,100
    python3 tools/gbdt_train_probe.py fit --cell catboost-yeti --trees 10,50     (an opponent, once)

Cells: `yeti`, `qrmse`, `pairlogit` (Istella-S LETOR with query ids, the
config of `tools/speed_gbdt_rank.py`), `catboost-yeti` (the same config on
CatBoost GPU), and the board lanes `gbdt-symmetric`, `gbdt-depthwise`,
`gbdt-lossguide` on `--dataset taxi|istella` (the config of
`tools/speed_gbdt_arm.py`, through `tools/gbdt_fairness_probe.py`'s loader).

THE CLOCK is `tools/speed_gbdt_rank.py`'s: construction + input conversion +
fit + a device drain (`cudaDeviceSynchronize`), after one untimed 1-tree
warm-up fit. Tree counts are rotated per repeat. The per-tree cost is the
slope between the smallest and largest tree count's medians, the fixed cost
the intercept.

This process is ONE ARM. An A/B is two checkouts (BEFORE and AFTER) running
this file alternately, one process per round (`tools/gbdt_train_ab.sh`); the
digests must be equal across arms. Under `nsys profile` the same command is
the per-kernel attribution. With `MOJOLEARN_STAGE_TIMES=1` it is the stage
ledger, which drains per stage and is a SPLIT, never a timing.

Every line begins `GTP ` so one log parses on the Mac.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, os.path.join(_ROOT, "python"), os.path.join(_ROOT, "bench", "speed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec              # noqa: E402
import speed_gbdt_rank as rank             # noqa: E402

RANK_LOSS = {"yeti": "YetiRank", "qrmse": "QueryRMSE", "pairlogit": "PairLogit",
             "catboost-yeti": "YetiRank"}
BOARD = ("gbdt-symmetric", "gbdt-depthwise", "gbdt-lossguide")


def build_cell(args):
    """Returns (fit(n) -> model, predict(model) -> ndarray, description)."""
    if args.cell in RANK_LOSS:
        data = spec.load_istella_rank(args.rows)
        library = "catboost" if args.cell.startswith("catboost") else "ours"
        params, _notes, fit, predict = rank.build_arm(
            library, RANK_LOSS[args.cell], "gpu", data)
        desc = "istella_rank rows=%d cols=%d queries=%d %s" % (
            data.x_train.shape[0], data.x_train.shape[1],
            np.unique(data.qid_train).size, json.dumps(params, sort_keys=True))
        return fit, predict, desc
    if args.cell not in BOARD:
        raise SystemExit("unknown cell " + args.cell)
    import gbdt_fairness_probe as fair
    data, cfg, _size = fair.load(args.dataset, args.rows or 1000000, args.cell)

    def fit(n):
        m = fair.our_estimator(cfg, data, n_estimators=n)
        m.fit(data._ours_X, data._ours_y)
        return m

    def predict(m):
        if data.task == "regression":
            return np.asarray(m.predict(data.X_test))
        return np.asarray(m.predict_proba(data.X_test))

    desc = "%s rows=%d cols=%d task=%s %s" % (
        data.tag, data.X_train.shape[0], data.X_train.shape[1], data.task,
        json.dumps({k: cfg[k] for k in sorted(cfg) if not isinstance(cfg[k], (list, dict))},
                   sort_keys=True, default=str))
    return fit, predict, desc


def cmd_fit(args):
    fit, predict, desc = build_cell(args)
    _name, drain = rank.make_drain()
    ladder = [int(v) for v in args.trees.split(",")]
    print("GTP CELL label=%s cell=%s %s" % (args.label, args.cell, desc), flush=True)
    try:
        import mojolearn
        m0 = mojolearn.GradientBoosting()
        print("GTP BUILD mode=%s vendor=%s file=%s" % (
            m0.numeric_mode_used(), m0.vendor_used(),
            getattr(m0._bind("_mojolearn_gbdt"), "__file__", "?")), flush=True)
    except Exception as e:                          # noqa: BLE001
        print("GTP BUILD unknown (%s)" % e, flush=True)
    fit(1)
    drain()
    times = {n: [] for n in ladder}
    digests = {}
    for r in range(args.reps):
        order = ladder[r % len(ladder):] + ladder[:r % len(ladder)]
        for n in order:
            t0 = time.perf_counter()
            m = fit(n)
            drain()
            ms = (time.perf_counter() - t0) * 1000.0
            times[n].append(ms)
            d = None
            if not args.no_predict:
                d = spec.hash_predictions(predict(m))
                digests.setdefault(n, set()).add(d)
            print("GTP FIT label=%s cell=%s trees=%d rep=%d ms=%.1f digest=%s"
                  % (args.label, args.cell, n, r, ms, d), flush=True)
            del m
    med = {n: statistics.median(v) for n, v in times.items()}
    lo, hi = min(ladder), max(ladder)
    out = {"label": args.label, "cell": args.cell, "dataset": args.dataset,
           "desc": desc, "times_ms": {str(n): v for n, v in times.items()},
           "median_ms": {str(n): med[n] for n in ladder},
           "digests": {str(n): sorted(x for x in v if x) for n, v in digests.items()}}
    if hi > lo:
        slope = (med[hi] - med[lo]) / float(hi - lo)
        out["per_tree_ms"] = slope
        out["fixed_ms"] = med[lo] - slope * lo
        print("GTP LINE label=%s cell=%s fixed_ms=%.1f per_tree_ms=%.3f at%d=%.1f at%d=%.1f"
              % (args.label, args.cell, out["fixed_ms"], slope, lo, med[lo], hi, med[hi]),
              flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 0


def cmd_summarize(args):
    """Medians, spreads and digest equality over the per-round JSONs of an
    A/B directory: files named `<arm>.<cell-tag>.<round>.json`."""
    rows = {}
    for path in sorted(args.files):
        d = json.load(open(path))
        arm = os.path.basename(path).split(".")[0]
        tag = ".".join(os.path.basename(path).split(".")[1:-2])
        for n, v in d["times_ms"].items():
            e = rows.setdefault((tag, int(n)), {}).setdefault(arm, {"ms": [], "dig": set()})
            e["ms"].append(statistics.median(v))
            e["dig"].update(d["digests"].get(n, []))
        if "per_tree_ms" in d:
            e = rows.setdefault((tag, -1), {}).setdefault(arm, {"ms": [], "dig": set()})
            e["ms"].append(d["per_tree_ms"])
    gate = 1.10
    for (tag, n) in sorted(rows):
        arms = rows[(tag, n)]
        what = "per_tree" if n < 0 else "trees=%d" % n
        parts = []
        for arm in sorted(arms):
            ms = arms[arm]["ms"]
            spread = max(ms) / min(ms) if min(ms) > 0 else float("inf")
            mark = "" if spread <= gate or n < 0 else " u"
            parts.append("%s median=%.2f min=%.2f max=%.2f spread=%.3f rounds=%d%s" % (
                arm, statistics.median(ms), min(ms), max(ms), spread, len(ms), mark))
        line = "GTP AB %s %s | %s" % (tag, what, " | ".join(parts))
        if "before" in arms and "after" in arms:
            b = statistics.median(arms["before"]["ms"])
            a = statistics.median(arms["after"]["ms"])
            line += " | before/after=%.3f" % (b / a)
            if n >= 0:
                same = arms["before"]["dig"] == arms["after"]["dig"] and len(arms["after"]["dig"]) == 1
                line += " digests_equal=%s %s" % (same, sorted(arms["after"]["dig"]))
        print(line)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("fit")
    q.add_argument("--cell", required=True)
    q.add_argument("--dataset", default="taxi", choices=("taxi", "istella"))
    q.add_argument("--rows", type=int, default=None)
    q.add_argument("--trees", default="10,100")
    q.add_argument("--reps", type=int, default=3)
    q.add_argument("--label", default="ours")
    q.add_argument("--no-predict", action="store_true")
    q.add_argument("--json", default=None)
    q.set_defaults(fn=cmd_fit)
    q = sub.add_parser("summarize")
    q.add_argument("files", nargs="+")
    q.set_defaults(fn=cmd_summarize)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
