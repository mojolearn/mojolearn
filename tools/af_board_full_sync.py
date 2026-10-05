#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One-time migration: fold the hand-made docs/apple-fast/BOARD_M3_FAST.md into board.json,
then generate the page from board.json (tools/af_board_render.py) and report every row change.

    python3 tools/af_board_full_sync.py [--board-dir DIR] [--page PATH] [--opp-lines FILE ...]
                                        [--report PATH] [--dry-run]

Laptop, text only. Keeps DIR/board.before-fast-sync-<date>.json. Per page row:
- FAST: when the page's FAST after differs from the board's FAST cell, the cell takes the page
  time (median=min=max, rounds=1), its quality and a `source` dict: tag (the last A/B clause of
  the status, else "M3 FAST refresh"), kind, previous_median_ms / previous_quality /
  previous_hash, baseline_ms (the 0.8.34 cell, else the page's FAST before) and its quality.
- opponent: a page "(fill)" opponent (or any page opponent on a race board.json lacks) becomes
  an opponent cell (fill=True); a page opponent "-" is filled by the board's best ok opponent.
- per race: race["fast_page"] = family, status, flip note. Races board.json lacks (2026-09-29
  tree board, AFB-only lanes) go to board["fast_page"]["extra_races"].
- page prose (intro, headline and apply history, notes) goes to board["fast_page"].
--opp-lines adds OPP lines (`OPP family= lane= ds= arm= status=ok median_ms= quality={json}`)
as fill opponent cells. The report lists every row whose rendered cells differ from the old page.
"""
import argparse, copy, json, os, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af_quality as afq
import af_board_render as R
from af_board_apply import Table

EVID = os.path.expanduser("~/mojolearn-evidence")
#: Board facts with no measurement behind them (orchestrator, 2026-10-04): an opponent never timed.
OPPONENT_NOTES = {("quantile", "istella"): "sklearn-cpu: too slow to measure (216 s on taxi)"}
#: Rows the page's Quality flags keep out of the faster count ("FAST quality under review").
EXCLUDED = {("tweedie", "istella"): "FAST quality under review (r2 -24.4 vs sklearn -21.9)",
            ("linearsvr", "istella"): "FAST quality under review (r2 -0.107 vs sklearn-cpu fill -0.026)"}
PAGE_FROM = "docs/apple-fast/BOARD_M3_FAST.md (hand-made page, 2026-10-04)"


def qdict(text):
    return {k: v for k, v in afq.parse(text).items() if isinstance(v, float)}


def opp_cell(lane, ds, fam, arm, ms, qtext, fill, src):
    import bench_board as bb
    return {"arm": arm, "library": bb.arm_library(arm), "mode": "opponent", "phase": "fit",
            "device": "cpu" if arm.endswith("-cpu") or "-cpu-" in arm else "gpu", "lane": lane, "dataset": ds,
            "family": fam, "status": "ok", "median_ms": ms, "min_ms": ms, "max_ms": ms, "times_ms": [ms],
            "rounds": 1, "quality": qdict(qtext), "quality_text": qtext if qtext not in ("", "-") else None,
            "fill": fill, "source": src, "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}


def put_opp(rec, cell, log):
    """Insert an opponent cell, replacing a non-ok cell of the same arm; never an ok measured one."""
    cells = rec.setdefault("cells", [])
    for i, c in enumerate(cells):
        if c.get("arm") == cell["arm"] and c.get("phase") in (None, "fit"):
            if c.get("status") == "ok" and c.get("median_ms"):
                log.append("kept measured %s cell %s ms over page %s ms" % (c["arm"], R.fmt_ms(c["median_ms"]),
                                                                            R.fmt_ms(cell["median_ms"])))
                return False
            cell["source"] = dict(cell["source"], replaced_status=c.get("status"))
            cells[i] = cell
            return True
    cells.append(cell)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-dir", default=R.BOARD_DIR)
    ap.add_argument("--page", default=R.PAGE)
    ap.add_argument("--opp-lines", action="append", default=[])
    ap.add_argument("--report", default=os.path.join(EVID, "board-sync-report-2026-10-05.md"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import bench_board as bb

    path = os.path.join(a.board_dir, "board.json")
    board = json.load(open(path))
    old_text = open(a.page).read()
    lines = old_text.split("\n")
    t = Table(lines)
    extra = {}
    fp_top = {"updated": a.date, "extra_races": extra}
    stats = {"fast_ab": 0, "fast_refresh": 0, "fast_note": 0, "fast_same": 0, "opp_fill_cells": 0,
             "opp_from_board": 0, "extra_races": 0}
    log = []
    for r in t.rows:
        g = lambda n: t.get(r, n)
        lane, ds, fam = g("lane"), g("dataset"), g("family")
        rid, rec = R.race_key(board, lane, ds)
        if rec is not None and rec.get("family") != fam and rid in board["races"] and any(
                t.get(o, "lane") == lane and t.get(o, "dataset") == ds and t.get(o, "family") == rec.get("family")
                for o in t.rows if o is not r):
            rid, rec = None, None  # the same lane under two drivers (linearsvr taxi): a race per family
            rid2 = "%s/%s/%s/page" % (fam, lane, ds)
            if rid2 in extra:
                rid, rec = rid2, extra[rid2]
        if rec is None:
            rid = "%s/%s/%s/page" % (fam, lane, ds)
            rec = {"id": rid, "family": fam, "lane": lane, "dataset": ds, "cells": [],
                   "note": "measured outside the 0.8.34 M3 board (%s); carried from %s" % (
                       "2026-09-29 M3 tree board and AFB tree runs" if fam == "trees" else "AFB runs", PAGE_FROM)}
            extra[rid] = rec
            stats["extra_races"] += 1
        status = g("status")
        # a FLIP the page could not compute (no ratio before) is a hand note; keep it
        flip_note = "; ".join(x.strip() for x in g("flip").split(";") if x.strip() and (
            not x.strip().startswith("FLIP") or g("ratio before") == "-"))
        rec["fast_page"] = {"family": fam, "status": status}
        if flip_note:
            rec["fast_page"]["flip_note"] = flip_note
        if (lane, ds) in EXCLUDED:
            rec["fast_page"]["excluded"] = EXCLUDED[(lane, ds)]
        if (lane, ds) in OPPONENT_NOTES:
            rec["fast_page"]["opponent_note"] = OPPONENT_NOTES[(lane, ds)]
        fc = R.fast_cell(rec)
        if fc is None:
            fc = {"arm": "ours-fast", "library": "mojolearn", "mode": "fast", "phase": "fit", "device": "gpu",
                  "lane": lane, "dataset": ds, "family": fam, "status": "no FAST cell", "median_ms": None,
                  "quality": {}, "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}
            rec["cells"].append(fc)
        old_ms, old_q = R.ok_ms(fc), fc.get("quality") or {}
        after, before = R._num(g("FAST after ms")), R._num(g("FAST before ms"))
        qa, qb = g("quality after (FAST)"), g("quality before (FAST)")
        if after is not None and R.fmt_ms(old_ms) != g("FAST after ms"):
            segs = [s.strip() for s in status.split(";")]
            ab = [s for s in segs if "A/B" in s]
            if ab:
                kind, tag = "A/B", ab[-1]
                stats["fast_ab"] += 1
            elif status.strip() != "ok":
                kind, tag = "page note", status
                stats["fast_note"] += 1
            else:
                kind, tag = "refresh", "M3 FAST refresh (tools/af_board_merge.py runs listed on the page)"
                stats["fast_refresh"] += 1
            src = {"tag": tag, "kind": kind, "from": PAGE_FROM, "synced": a.date,
                   "previous_median_ms": old_ms, "previous_quality": old_q,
                   "previous_status": fc.get("status"), "previous_hash": fc.get("hash"),
                   "baseline_ms": old_ms if old_ms is not None else before,
                   "baseline_quality_text": qb if qb not in ("", "-") else (R.fmt_q(old_q) if old_ms else "-"),
                   "display_rounded": True}
            if isinstance(fc.get("source"), str):
                src["stored"] = fc["source"]
            fc.update(median_ms=after, min_ms=after, max_ms=after, times_ms=[after], rounds=1, status="ok",
                      quality=qdict(qa), quality_text=qa, hash=None, source=src)
        else:
            stats["fast_same"] += 1
            if qa not in ("", "-") and qa != R.fmt_q(old_q):
                fc["quality_text"] = qa
        # opponent
        parm, pms = g("best opponent"), R._num(g("opp ms"))
        best = R.best_opponent(rec)
        if parm != "-" and pms:
            arm = parm.replace(" (fill)", "")
            fill = parm.endswith("(fill)")
            if best and best["arm"] == arm and not (fill and not best.get("fill") and R.fmt_ms(best["median_ms"]) != g("opp ms")):
                if g("opponent quality") not in ("", "-") and g("opponent quality") != R.fmt_q(best.get("quality")):
                    best["quality_text"] = g("opponent quality")
                if R.fmt_ms(best["median_ms"]) != g("opp ms"):
                    log.append("%s %s: opponent %s board %s ms vs page %s ms (board kept)" % (
                        lane, ds, arm, R.fmt_ms(best["median_ms"]), g("opp ms")))
            elif fill or best is None:
                c = opp_cell(lane, ds, fam, arm, pms, g("opponent quality"), fill,
                             {"fill": "M3 opponent fill (tools/opp_only_board.py)" if fill else "page opponent",
                              "from": PAGE_FROM, "display_rounded": True})
                if put_opp(rec, c, log):
                    stats["opp_fill_cells"] += 1
            else:
                log.append("%s %s: page opponent %s %s ms, board best %s %s ms (board kept)" % (
                    lane, ds, parm, g("opp ms"), best["arm"], R.fmt_ms(best["median_ms"])))
        elif best is not None:
            stats["opp_from_board"] += 1
        if rid in board["races"]:
            bb.add_ratios(rec["cells"])
    # OPP lines (knn-imputer istella)
    n_opp_lines = 0
    for p in a.opp_lines:
        for ln in open(os.path.expanduser(p)):
            if not ln.startswith("OPP "):
                continue
            kv = dict(re.findall(r"(\w+)=(\{[^}]*\}|\S+)", ln))
            ms = R._num(kv.get("median_ms"))
            if kv.get("status") != "ok" or ms is None:
                continue
            rid, rec = R.race_key(board, kv["lane"], kv["ds"])
            if rec is None:
                log.append("OPP line without a race: %s %s" % (kv["lane"], kv["ds"]))
                continue
            q = json.loads(kv.get("quality") or "{}")
            c = opp_cell(kv["lane"], kv["ds"], kv.get("family"), kv["arm"], ms, None, True,
                         {"fill": "M3 opponent fill line", "from": os.path.basename(p)})
            c["quality"] = q
            if put_opp(rec, c, log):
                n_opp_lines += 1
                if rid in board["races"]:
                    bb.add_ratios(rec["cells"])
    # page prose
    hi = next(i for i, l in enumerate(lines) if l.startswith("Canonical full-board summary"))
    qi = next((i for i, l in enumerate(lines) if l.startswith("Quality (")), None)
    intro = lines[2] if lines[2].startswith("Our FAST arm") else ""
    bullets = [l for l in lines[hi + 1:t.hdr_i] if l.startswith("- ")]
    hist = ["- Hand-made headline, last before the page was generated (2026-10-04): %s" % lines[hi][len("Canonical full-board summary "):]]
    if qi is not None:
        hist.append("- Hand-made quality paragraph, last before generation: %s" % lines[qi])
    fp_top.update(intro=intro.replace("Written by `tools/af_board_merge.py`.",
                                      "First written by `tools/af_board_merge.py`; now carried in board.json."),
                  headline_history=hist + bullets,
                  notes_md=[l for l in lines[t.end:]] if t.end < len(lines) else [])
    while fp_top["notes_md"] and fp_top["notes_md"][0] == "":
        fp_top["notes_md"].pop(0)
    board["fast_page"] = fp_top
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board.setdefault("box_history", []).append({"fast_page_sync": PAGE_FROM, "date": a.date, **stats})

    new_text = R.render_page(board)
    report = diff_report(t, new_text, stats, n_opp_lines, log, lines[hi], a)
    print("\n".join(report[:22]))
    if a.dry_run:
        with open(os.path.join(EVID, "board-sync-dryrun.md"), "w") as fh:
            fh.write("\n".join(report) + "\n")
        return
    shutil.copy2(path, os.path.join(a.board_dir, "board.before-fast-sync-%s.json" % a.date.replace("-", "")))
    with open(path, "w") as fh:
        json.dump(board, fh, indent=1)
    R.write_all(a.board_dir, a.page, board)
    with open(a.report, "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("wrote %s, BOARD.md, %s; report %s" % (path, a.page, a.report))


def diff_report(t_old, new_text, stats, n_opp_lines, log, old_head, a):
    nl = new_text.split("\n")
    tn = Table(nl)
    old = {(t_old.get(r, "lane"), t_old.get(r, "dataset"), t_old.get(r, "family")): r for r in t_old.rows}
    new = {(tn.get(r, "lane"), tn.get(r, "dataset"), tn.get(r, "family")): r for r in tn.rows}
    cols = [c for c in t_old.cols if c not in ("lane", "dataset", "family")]
    changed, why_count = [], {}
    for k in sorted(set(old) & set(new)):
        diffs = [(c, t_old.get(old[k], c), tn.get(new[k], c)) for c in cols if t_old.get(old[k], c) != tn.get(new[k], c)]
        if diffs:
            changed.append((k, diffs))
            for c, o, n in diffs:
                why_count[c] = why_count.get(c, 0) + 1
    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))
    new_head = next(l for l in nl if l.startswith("Canonical full-board summary"))
    rep = ["# Board sync report (%s)" % a.date, "",
           "`tools/af_board_full_sync.py`: board.json is now the one M3 board; docs/apple-fast/BOARD_M3_FAST.md is "
           "generated from it by `tools/af_board_render.py`.", "",
           "## Headline", "", "- old (hand-made): %s" % old_head[:260], "- new (generated): %s" % new_head[:260], "",
           "## Synced into board.json", "",
           "- FAST cells from page A/B rows: %d; from page refresh rows: %d; from rows with a page note: %d; "
           "unchanged: %d" % (stats["fast_ab"], stats["fast_refresh"], stats["fast_note"], stats["fast_same"]),
           "- opponent cells added from page fills: %d; from OPP lines (knn-imputer istella): %d; page rows "
           "whose opponent \"-\" is now filled from board.json: %d" % (stats["opp_fill_cells"], n_opp_lines,
                                                                        stats["opp_from_board"]),
           "- races carried in fast_page.extra_races (not in the 0.8.34 board): %d" % stats["extra_races"], "",
           "## Row changes, generated page vs the hand-made page", "",
           "- rows on both: %d, changed: %d, unchanged: %d; rows only on the generated page: %d (board.json races "
           "the hand page left out); rows only on the old page: %d" % (
               len(set(old) & set(new)), len(changed), len(set(old) & set(new)) - len(changed), len(only_new),
               len(only_old)),
           "- changed cells by column: %s" % ", ".join("%s %d" % kv for kv in sorted(why_count.items(), key=lambda x: -x[1])),
           "", "Why columns change: FAST before = the precise 0.8.34 board cell (the page used rounded TSV values, "
           "or '-' on rows it added by hand although the board had a 0.8.34 FAST cell); opp ms / opponent = the "
           "board's best ok opponent (the page used a rounded TSV value or had '-'); ratios and flips recomputed "
           "from those; quality before = the 0.8.34 cell's quality where the page had '-'.", ""]
    rep += ["## Board kept over the page (%d)" % len(log), ""] + ["- " + x for x in log] + [""]
    rep += ["## Changed rows (%d)" % len(changed), "", "| lane | dataset | family | column: old -> new |", "|---|---|---|---|"]
    for k, diffs in changed:
        rep.append("| %s | %s | %s | %s |" % (k[0], k[1], k[2], "; ".join("%s: %s -> %s" % (c, o[:60], n[:60]) for c, o, n in diffs)
                                         .replace("|", "/")))
    rep += ["", "## Rows only on the generated page (%d)" % len(only_new), "",
            ", ".join("%s %s (%s)" % k for k in only_new), "", "## Rows only on the old page (%d)" % len(only_old), "",
            ", ".join("%s %s (%s)" % k for k in only_old) or "none"]
    return rep


if __name__ == "__main__":
    main()
