#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Audit FAST quality on the M3 board: FAST vs the best opponent, FAST after vs FAST before.

    python3 tools/af_board_quality_audit.py --json BOARD.json [--board docs/apple-fast/BOARD_M3_FAST.md]
        [--out REPORT.md] [--write-board] [--date YYYY-MM-DD]

Laptop, text only. Lane worktrees leave out bench/results, so fetch the full board first:
    git show origin/main:bench/results/bench_board/m3ultra-0834/board.json > ~/mojolearn-evidence/board-quality/board-m3ultra-0834.json

Per (lane, dataset):
- FAST quality = the refresh table's "quality after (FAST)" when it has one, else the
  board JSON's ours-fast cell (0.8.34).
- Opponents = every ok opponent cell in the board JSON (any arm whose library is not
  mojolearn) plus the refresh table's "opponent quality". FAST is WORSE than the best
  opponent when it is WORSE than any of them on a shared metric (tools/af_quality.py,
  rel_tol 1e-3, abs_tol 1e-6). rel = signed relative change of the worst metric.
- After vs before = the table's "quality after (FAST)" vs "quality before (FAST)".
- UNKNOWN = shared metrics whose direction af_quality does not know.

--out writes the markdown report; --write-board puts (or replaces) the "Quality (...)"
paragraph just above the "Canonical full-board summary" paragraph of the board page.
"""
import argparse, collections, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af_quality as afq
from af_board_apply import Table, BOARD

MATERIAL = 0.01  # 1% relative: reported as a separate count; the gate itself uses af_quality's tolerances


def load_json(path):
    d = json.load(open(path))
    out = {}
    for key, r in d.get("races", {}).items():
        fast, opps = None, []
        for c in r.get("cells", []):
            if c.get("status") != "ok" or not c.get("quality"):
                continue
            if c.get("arm") == "ours-fast":
                fast = c["quality"]
            elif c.get("library") != "mojolearn" and not str(c.get("arm", "")).startswith("ours"):
                opps.append((c["arm"], c["quality"]))
        out[(r.get("lane"), r.get("dataset"))] = (fast, opps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--board", default=BOARD)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-board", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()

    lines = open(a.board).read().split("\n")
    t = Table(lines)
    js = load_json(a.json)
    table = {(t.get(r, "lane"), t.get(r, "dataset")): r for r in t.rows}
    keys = list(table) + sorted(k for k in js if k not in table and k[0])

    vs_opp, vs_before, unknown_rows = [], [], []
    unk_metrics = collections.Counter()
    n_opp_judged = n_before_judged = n_no_q = n_no_opp = 0
    for k in keys:
        row = table.get(k)
        jf, jopps = js.get(k, (None, []))
        fq, fsrc = None, None
        if row is not None and t.get(row, "quality after (FAST)") not in ("", "-"):
            fq, fsrc = t.get(row, "quality after (FAST)"), "table"
        elif jf:
            fq, fsrc = jf, "0.8.34 json"
        if fq is None or not afq.parse(fq):
            n_no_q += 1
            continue
        opps = list(jopps)
        if row is not None and t.get(row, "opponent quality") not in ("", "-"):
            opps.append(("table:" + t.get(row, "best opponent"), t.get(row, "opponent quality")))
        judged, worst = False, None
        row_unk = set()
        for arm, oq in opps:
            c = afq.compare(fq, oq)
            row_unk.update(c["unknown"])
            if c["verdict"] != afq.UNKNOWN:
                judged = True
            if c["verdict"] == afq.WORSE and (worst is None or c["worst"] < worst[2]["worst"]):
                worst = (arm, oq, c)
        if not opps:
            n_no_opp += 1
        n_opp_judged += judged
        if worst:
            vs_opp.append((k, fsrc, worst[0], worst[2]))
        if row is not None and t.get(row, "quality before (FAST)") not in ("", "-") and fsrc == "table":
            c = afq.compare(fq, t.get(row, "quality before (FAST)"))
            row_unk.update(c["unknown"])
            n_before_judged += c["verdict"] != afq.UNKNOWN
            if c["verdict"] == afq.WORSE:
                vs_before.append((k, c))
        if row_unk:
            unknown_rows.append((k, sorted(row_unk)))
            unk_metrics.update(row_unk)

    vs_opp.sort(key=lambda x: x[3]["worst"])
    vs_before.sort(key=lambda x: x[1]["worst"])
    mat_opp = sum(1 for x in vs_opp if x[3]["worst"] < -MATERIAL)
    mat_before = sum(1 for x in vs_before if x[1]["worst"] < -MATERIAL)

    def pct(v):
        return "%.2f%%" % (100 * v) if v != -float("inf") else "n/a"

    rep = ["# Board quality audit (%s)" % a.date, "",
           "Written by `tools/af_board_quality_audit.py` from `%s` and `%s`. Comparison: tools/af_quality.py, "
           "rel_tol 1e-3, abs_tol 1e-6; rel = relative change of the worst metric (negative = worse). "
           "Material = worse by more than %d%%." % (os.path.relpath(a.board, os.path.join(HERE, "..")),
                                                     os.path.basename(a.json), MATERIAL * 100), "",
           "## Counts", "",
           "- lane/dataset pairs: %d (table %d, board JSON only %d); without a parseable FAST quality: %d; "
           "without any opponent quality: %d" % (len(keys), len(table), len(keys) - len(table), n_no_q, n_no_opp),
           "- FAST vs best opponent: %d judged, **%d WORSE** (%d material)" % (n_opp_judged, len(vs_opp), mat_opp),
           "- FAST after vs FAST before: %d judged, **%d WORSE** (%d material)" % (n_before_judged, len(vs_before),
                                                                                 mat_before),
           "- rows with UNKNOWN metrics: %d (%d distinct metrics)" % (len(unknown_rows), len(unk_metrics)), "",
           "## Worst 10 quality rows (either comparison)", "",
           "| lane | dataset | against | rel | detail |", "|---|---|---|---:|---|"]
    both = [(x[3]["worst"], x[0], "opponent %s" % x[2], x[3]) for x in vs_opp] + \
           [(x[1]["worst"], x[0], "FAST before", x[1]) for x in vs_before]
    both.sort(key=lambda x: x[0])
    for w, k, who, c in both[:10]:
        rep.append("| %s | %s | %s | %s | %s |" % (k[0], k[1], who, pct(w), afq.describe(c)))
    rep += ["", "## FAST quality WORSE than an opponent (%d)" % len(vs_opp), "",
            "| lane | dataset | FAST quality from | opponent | rel | detail |", "|---|---|---|---|---:|---|"]
    for k, src, arm, c in vs_opp:
        rep.append("| %s | %s | %s | %s | %s | %s |" % (k[0], k[1], src, arm, pct(c["worst"]), afq.describe(c)))
    rep += ["", "## FAST after WORSE than FAST before (%d)" % len(vs_before), "",
            "| lane | dataset | rel | detail |", "|---|---|---:|---|"]
    for k, c in vs_before:
        rep.append("| %s | %s | %s | %s |" % (k[0], k[1], pct(c["worst"]), afq.describe(c)))
    rep += ["", "## UNKNOWN metrics (%d)" % len(unk_metrics), "",
            "Direction not known to tools/af_quality.py; never passed silently. Add them there once decided.", ""]
    for m, n in unk_metrics.most_common():
        rep.append("- `%s`: %d rows (%s)" % (m, n, ", ".join("%s %s" % k for k, ms in unknown_rows if m in ms)[:300]))
    text = "\n".join(rep) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        print("wrote %s" % a.out)
    else:
        sys.stdout.write(text)

    para = ("Quality (%s, `tools/af_board_quality_audit.py`): of %d lane/dataset pairs, FAST quality is WORSE than "
            "the best opponent on %d (%d by more than 1%%) and FAST after is WORSE than FAST before on %d (%d by more "
            "than 1%%); %d rows carry metrics of unknown direction (%d metrics), %d have no parseable FAST quality. "
            "Tolerance rel 1e-3, abs 1e-6. `tools/af_board_apply.py` refuses a row whose quality is WORSE than main "
            "or the opponent and tags every applied row Q: ..." % (
                a.date, len(keys), len(vs_opp), mat_opp, len(vs_before), mat_before, len(unknown_rows),
                len(unk_metrics), n_no_q))
    print(para)
    if a.write_board:
        qi = next((i for i, l in enumerate(lines) if l.startswith("Quality (")), None)
        if qi is not None:
            lines[qi] = para
        else:
            hi = next(i for i, l in enumerate(lines) if l.startswith("Canonical full-board summary"))
            lines[hi:hi] = [para, ""]
        open(a.board, "w").write("\n".join(lines))
        print("updated %s" % a.board)


if __name__ == "__main__":
    main()
