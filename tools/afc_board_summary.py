#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Summarize an Apple board.json for the classical (non-tree, non-neural)
lanes: our FAST and IDENTICAL medians, the best opponent, the ratio FAST/best
and each arm's quality. One line per race, worst FAST ratio first.

    python3 tools/afc_board_summary.py ~/board-0834/board/board.json
"""
import json
import sys

SKIP_FAMILIES = {"trees", "neural"}
TREE_WORDS = ("forest", "tree", "gbdt", "xgb", "boost", "isolation", "shap", "extra")


def short_quality(q):
    if not isinstance(q, dict):
        return str(q)[:30]
    out = []
    for k, v in sorted(q.items()):
        if isinstance(v, float):
            out.append("%s=%.4g" % (k, v))
        elif isinstance(v, (int, str)) and len(str(v)) < 16:
            out.append("%s=%s" % (k, v))
    return ",".join(out)[:60]


def main(path):
    with open(path) as fh:
        res = json.load(fh)
    rows = []
    for rid, rec in (res.get("races") or {}).items():
        fam = rec.get("family") or ""
        lane = rec.get("lane") or ""
        if fam in SKIP_FAMILIES or any(w in lane for w in TREE_WORDS):
            continue
        cells = rec.get("cells") or []
        fast = next((c for c in cells if c.get("library") == "mojolearn" and c.get("mode") == "fast"), None)
        ident = next((c for c in cells if c.get("library") == "mojolearn" and c.get("mode") == "identical"), None)
        opps = [c for c in cells if c.get("library") != "mojolearn" and c.get("status") == "ok" and c.get("median_ms")]
        best = min(opps, key=lambda c: c["median_ms"]) if opps else None

        def ms(c):
            return c["median_ms"] if c and c.get("status") == "ok" and c.get("median_ms") else None
        f, i, b = ms(fast), ms(ident), ms(best)
        ratio = f / b if f and b else None
        rows.append((ratio if ratio is not None else -1.0, fam, lane, rec.get("dataset"), f, i, b,
                     best.get("arm") if best else "-",
                     fast.get("status") if fast else "noarm",
                     short_quality((fast or {}).get("quality")),
                     short_quality((best or {}).get("quality"))))
    rows.sort(key=lambda r: -r[0])
    print("ratio_fast_best fam lane ds fast_ms ident_ms best_ms best_arm fast_status | fastQ | bestQ")
    for r in rows:
        print("%7.2f %-10s %-26s %-8s %9s %9s %9s %-18s %-8s | %s | %s" % (
            r[0], r[1], r[2], r[3],
            "%.1f" % r[4] if r[4] else "-", "%.1f" % r[5] if r[5] else "-",
            "%.1f" % r[6] if r[6] else "-", r[7], r[8], r[9], r[10]))


if __name__ == "__main__":
    main(sys.argv[1])
