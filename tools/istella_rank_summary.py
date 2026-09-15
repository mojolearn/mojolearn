#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tables for the Istella-S ranking leg, from `tools/speed_gbdt_rank.py` JSONs.

    python3 tools/istella_rank_summary.py <dir with cell JSONs>

Prints markdown: one row per cell (time at 100 trees as median (min..max),
fixed cost, per tree, NDCG@5 and @10, peak GPU memory), then OUR time over
each opponent's time for every (our loss, opponent cell) pair, split into the
100-tree total, the fixed cost and the per-tree cost. A ratio above 1.0 means
our IDENTICAL arm took longer.
"""

import glob
import json
import os
import statistics
import sys


def load(folder):
    cells = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if "summary" not in rec:
            rec["_partial"] = True
        rec["_file"] = os.path.basename(path)
        cells.append(rec)
    return cells


def label(rec):
    name = {"ours": "ours IDENTICAL", "catboost": "CatBoost",
            "xgboost": "XGBoost", "lightgbm": "LightGBM"}[rec["library"]]
    dev = "GPU" if rec["device"] == "gpu" else "CPU"
    return "%s %s %s" % (name, rec["loss"], dev)


def main(folder):
    cells = load(folder)
    done = [r for r in cells if not r.get("_partial")]
    top = str(max(int(k) for r in done for k in r["summary"])) if done else "100"
    print("| cell | version | 100 trees ms, median (min..max) | fixed ms | "
          "per tree ms | NDCG@5 | NDCG@10 | NDCG@10 file-order ties | "
          "peak GPU MiB over idle | prediction hashes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in cells:
        if r.get("_partial"):
            print("| %s | %s | PARTIAL: %d fits recorded | | | | | | | |"
                  % (label(r), r.get("version"), len(r.get("fits", []))))
            continue
        s = r["summary"][top]
        q = r["quality"]
        n5 = statistics.median(x["ndcg5"] for x in q)
        n10 = statistics.median(x["ndcg10"] for x in q)
        n10f = statistics.median(x["ndcg10_fileorder"] for x in q)
        hashes = sorted(set(x["pred_sha"] for x in q))
        peak = (None if r.get("mem_peak_mib") is None
                or r.get("mem_baseline_mib") is None
                else r["mem_peak_mib"] - r["mem_baseline_mib"])
        print("| %s | %s | %.0f (%.0f..%.0f), n=%d | %.0f | %.2f | %.4f | "
              "%.4f | %.4f | %s | %d distinct of %d |"
              % (label(r), r["version"], s["median"], s["min"], s["max"],
                 s["n"], r["fixed_ms"], r["per_tree_ms"], n5, n10, n10f,
                 peak, len(hashes), len(q)))
    ours = [r for r in cells if r["library"] == "ours" and not r.get("_partial")]
    opp = [r for r in cells if r["library"] != "ours" and not r.get("_partial")]
    print()
    print("Our time over the opponent's time (above 1.0: ours took longer).")
    print()
    print("| our loss | opponent cell | 100 trees | fixed cost | per tree |")
    print("|---|---|---|---|---|")
    for o in ours:
        for p in opp:
            print("| %s | %s | %.2fx | %.2fx | %.2fx |"
                  % (o["loss"], label(p),
                     o["summary"][top]["median"] / p["summary"][top]["median"],
                     o["fixed_ms"] / p["fixed_ms"] if p["fixed_ms"] > 0
                     else float("nan"),
                     o["per_tree_ms"] / p["per_tree_ms"]))
    for r in cells:
        if r["library"] == "ours" and "ours_group_sizes_ms" in r:
            print()
            print("ours %s: group_id run lengths alone %.0f ms (inside every "
                  "timed fit), mode %s, vendor %s"
                  % (r["loss"], r["ours_group_sizes_ms"],
                     r.get("ours_mode_used"), r.get("ours_vendor_used")))
            break


if __name__ == "__main__":
    main(sys.argv[1])
