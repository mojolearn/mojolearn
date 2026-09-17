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
this file alternately, one process per round (`tools/gbdt_train_body.sh ab`); the
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
    binding_file, mode_used = "?", "?"
    try:
        import mojolearn
        m0 = mojolearn.GradientBoosting()
        binding_file = os.path.abspath(getattr(m0._bind("_mojolearn_gbdt"), "__file__", "?"))
        mode_used = m0.numeric_mode_used()
        print("GTP BUILD mode=%s vendor=%s file=%s mtime=%d" % (
            mode_used, m0.vendor_used(), binding_file,
            int(os.path.getmtime(binding_file))), flush=True)
    except Exception as e:                          # noqa: BLE001
        print("GTP BUILD unknown (%s)" % e, flush=True)
    if args.expect_root and not binding_file.startswith(os.path.abspath(args.expect_root) + os.sep):
        raise SystemExit("GTP FATAL: arm %s loaded %s, not a binding under %s"
                         % (args.label, binding_file, args.expect_root))
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
           "binding_file": binding_file, "mode": mode_used,
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
    seen = {}
    for path in sorted(args.files):
        d = json.load(open(path))
        seen.setdefault(os.path.basename(path).split(".")[0], set()).add(
            "%s (%s)" % (d.get("binding_file"), d.get("mode")))
    for arm in sorted(seen):
        print("GTP AB-ARM %s loaded %s" % (arm, sorted(seen[arm])))
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


def _last_tables(path):
    """The stage tables of the LAST fit in a `MOJOLEARN_STAGE_TIMES=1` log:
    the boosting loop's (`ensemble/instruments.mojo`, seconds) and the
    searcher's (`depthwise_stage_times.mojo`, milliseconds). Returns
    {stage: ms}."""
    # a fit prints SEVERAL tables of each kind (the boosting loop's, the
    # train bracket's, the searcher's, the estimator's); the last table
    # under each heading is the last fit's. A loop table has no heading of
    # its own, so it is keyed by its first stage name.
    loop_tables, stage_tables, cur, key = {}, {}, None, None
    per_tree_tables = 0
    for line in open(path, errors="replace"):
        if line.startswith("== MOJOLEARN_STAGE_TIMES"):
            cur, key = "loop", None
            continue
        if line.startswith("[stage-times] "):
            cur = None
            body = line[len("[stage-times] "):]
            if "NOT a" in body:
                key = body.split(" -- ")[0].strip()
                if "rows=" in key:
                    # the non-symmetric searcher prints ONE TABLE PER TREE
                    # (`lossguide fit: rows=... leaves=...`): summed here
                    key = "per-tree searcher tables"
                    per_tree_tables += 1
                    stage_tables.setdefault(key, {})
                else:
                    stage_tables[key] = {}
                continue
            parts = body.strip().split("\t")
            if len(parts) == 2 and parts[1].endswith(" ms") and key in stage_tables:
                t = stage_tables[key]
                name = parts[0].strip()
                if key == "per-tree searcher tables":
                    t[name] = t.get(name, 0.0) + float(parts[1][:-3])
                else:
                    t[name] = float(parts[1][:-3])
            continue
        if cur == "loop" and line.startswith("  ") and "\t" in line:
            name, val = line.strip().split("\t")
            if val.endswith(" s"):
                if key is None:
                    key = name
                    loop_tables[key] = {}
                loop_tables[key][name] = float(val[:-2]) * 1000.0
            continue
        if cur == "loop" and not line.startswith(" "):
            cur, key = None, None
    out = {}
    for t in loop_tables.values():
        out.update({"loop." + k: v for k, v in t.items()})
    for k, t in stage_tables.items():
        out.update(t)
    out["(searcher tables summed, warm-up tree included)"] = float(per_tree_tables)
    return out


def cmd_ledger(args):
    """`ledger <trees> <arm>=<log> ...`: the last fit's stage tables side by
    side, in ms per tree, with each arm's difference from the first arm."""
    arms = [a.split("=", 1) for a in args.logs]
    tables = [(name, _last_tables(path)) for name, path in arms]
    keys = []
    for _n, t in tables:
        for k in t:
            if k not in keys:
                keys.append(k)
    print("GTP LEDGER trees=%d (ms per tree; a stage-timed run drains per stage: a SPLIT, not a timing)" % args.trees)
    print("GTP LEDGER stage | " + " | ".join(n for n, _t in tables) + " | first minus each other")
    for k in keys:
        vals = [t.get(k) for _n, t in tables]
        cells = ["%.3f" % (v / args.trees) if v is not None else "-" for v in vals]
        deltas = ["%.3f" % ((vals[0] - v) / args.trees) if (v is not None and vals[0] is not None) else "-"
                  for v in vals[1:]]
        print("GTP LEDGER %s | %s | %s" % (k, " | ".join(cells), " ".join(deltas)))
    return 0


def cmd_kernels(args):
    """Per-kernel GPU time of one nsys capture (`.sqlite`), grouped by the
    kernel's name and block size, with launch counts and the mean grid: the
    compiler's kernel names are a module prefix and a hash, so the block size
    and the launch count are what tie a row to a source kernel."""
    import sqlite3
    db = sqlite3.connect(args.sqlite)
    rows = db.execute(
        "SELECT s.value, k.blockX, COUNT(*), SUM(k.end - k.start), AVG(k.gridX), AVG(k.gridY) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
        "GROUP BY s.value, k.blockX ORDER BY 4 DESC").fetchall()
    total = sum(r[3] for r in rows)
    sync = db.execute(
        "SELECT s.value, COUNT(*), SUM(r.end - r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id = r.nameId GROUP BY s.value ORDER BY 3 DESC LIMIT 8").fetchall()
    mem = db.execute(
        "SELECT copyKind, COUNT(*), SUM(end - start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "GROUP BY copyKind").fetchall()
    print("GTP KERNELS label=%s total_gpu_kernel_ms=%.1f launches=%d" % (
        args.label, total / 1e6, sum(r[2] for r in rows)))
    for name, bx, n, ns, gx, gy in rows[:args.top]:
        print("GTP KERNEL label=%s ms=%.2f pct=%.1f launches=%d avg_us=%.1f block=%d grid=%.0fx%.0f %s" % (
            args.label, ns / 1e6, 100.0 * ns / total, n, ns / n / 1e3, bx, gx, gy, name))
    for name, n, ns in sync:
        print("GTP API label=%s ms=%.1f calls=%d %s" % (args.label, ns / 1e6, n, name))
    for kind, n, ns, b in mem:
        print("GTP MEMCPY label=%s kind=%d ms=%.1f copies=%d MiB=%.1f" % (
            args.label, kind, ns / 1e6, n, (b or 0) / 1048576.0))
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
    q.add_argument("--expect-root", default=None,
                   help="refuse unless the loaded gbdt binding lives under this tree")
    q.set_defaults(fn=cmd_fit)
    q = sub.add_parser("ledger")
    q.add_argument("trees", type=int)
    q.add_argument("logs", nargs="+")
    q.set_defaults(fn=cmd_ledger)
    q = sub.add_parser("kernels")
    q.add_argument("sqlite")
    q.add_argument("--label", default="ours")
    q.add_argument("--top", type=int, default=25)
    q.set_defaults(fn=cmd_kernels)
    q = sub.add_parser("summarize")
    q.add_argument("files", nargs="+")
    q.set_defaults(fn=cmd_summarize)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
