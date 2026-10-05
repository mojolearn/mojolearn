#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One-time migration: fold the hand-made docs/apple-fast/BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md
into board.json, then generate every page from board.json (tools/af_board_render.py) and report
every row change.

    python3 tools/af_board_full_sync.py [--board-dir DIR] [--fast-page PATH] [--ident-page PATH]
                                        [--docs-dir DIR] [--opp-lines FILE ...] [--report PATH] [--dry-run]

Laptop, text only. Keeps DIR/board.before-fast-sync-<date>.json. Per page row (FAST, then IDENTICAL):
- our cell: when the page's "after" differs from the board's cell of that mode, the cell takes the
  page time (median=min=max, rounds=1), its quality and a `source` dict: tag (FAST: the last A/B
  clause of the status, else "M3 FAST refresh"), kind, previous_median_ms / previous_quality /
  previous_hash, baseline_ms (the 0.8.34 cell, else the page's "before") and its quality.
- opponent: a page "(fill)" opponent (or any page opponent on a race board.json lacks) becomes
  an opponent cell (fill=True); a page opponent "-" is filled by the board's best ok opponent.
- per race: race["<mode>_page"] = family, status, flip note. Races board.json lacks (2026-09-29
  tree board, AFB-only lanes) go to board["extra_races"].
- page prose (intro, headline and apply history, notes) goes to board["<mode>_page"].
OPP_LINES (built in) and --opp-lines add OPP lines (`OPP family= lane= ds= arm= status=ok median_ms= quality={json}`)
as fill opponent cells. The report lists every row whose rendered cells differ from the old page.
"""
import argparse, json, os, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af_quality as afq
import af_board_render as R
from af_board_apply import Table

EVID = os.path.expanduser("~/mojolearn-evidence")
#: Rows the FAST page's Quality flags keep out of the faster count ("FAST quality under review").
EXCLUDED = {("tweedie", "istella"): "FAST quality under review (r2 -24.4 vs sklearn -21.9)",
            ("linearsvr", "istella"): "FAST quality under review (r2 -0.107 vs sklearn-cpu fill -0.026)"}
#: Measured M3 opponent lines folded in (from ~/mojolearn-evidence/opp_lines_m3.txt, 2026-10-04).
OPP_LINES = ['OPP family=algos lane=knn-imputer ds=istella arm=sklearn-cpu status=ok median_ms=26.045 '
             'quality={"masked_rmse": 977968.1855298833}']
#: Board facts with no measurement behind them (orchestrator, 2026-10-04): an opponent never timed.
OPPONENT_NOTES = {("quantile", "istella"): "sklearn-cpu: too slow to measure (216 s on taxi)"}


def qdict(text):
    return {k: v for k, v in afq.parse(text).items() if isinstance(v, float)}


def opp_cell(lane, ds, fam, arm, ms, qtext, fill, src):
    import bench_board as bb
    return {"arm": arm, "library": bb.arm_library(arm), "mode": "opponent", "phase": "fit",
            "device": "cpu" if arm.endswith("-cpu") or "-cpu-" in arm else "gpu", "lane": lane, "dataset": ds,
            "family": fam, "status": "ok", "median_ms": ms, "min_ms": ms, "max_ms": ms, "times_ms": [ms],
            "rounds": 1, "quality": qdict(qtext), "quality_text": qtext if qtext not in (None, "", "-") else None,
            "fill": fill, "source": src, "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}


def put_opp(rec, cell, log):
    """Insert an opponent cell, replacing a non-ok cell of the same arm; never an ok measured one."""
    cells = rec.setdefault("cells", [])
    for i, c in enumerate(cells):
        if c.get("arm") == cell["arm"] and c.get("phase") in (None, "fit"):
            if c.get("status") == "ok" and c.get("median_ms"):
                if R.fmt_ms(c["median_ms"]) != R.fmt_ms(cell["median_ms"]):
                    log.append("%s %s: kept measured %s cell %s ms over page %s ms" % (
                        rec["lane"], rec["dataset"], c["arm"], R.fmt_ms(c["median_ms"]), R.fmt_ms(cell["median_ms"])))
                return False
            cell["source"] = dict(cell["source"], replaced_status=c.get("status"))
            cells[i] = cell
            return True
    cells.append(cell)
    return True


def sync_page(board, t, mode, page_from, a, stats, log):
    import bench_board as bb
    extra = board.setdefault("extra_races", {})
    M = mode.upper()
    for r in t.rows:
        g = lambda n: t.get(r, n.replace("FAST", M))
        lane, ds, fam = g("lane"), g("dataset"), g("family")
        rid, rec = R.race_key(board, lane, ds, fam)
        if rec is not None and rid in board["races"] and rec.get("family") != fam and any(
                t.get(o, "lane") == lane and t.get(o, "dataset") == ds and t.get(o, "family") == rec.get("family")
                for o in t.rows if o is not r):
            rid, rec = None, None  # the same lane under two drivers (linearsvr taxi): a race per family
        if rec is None:
            rid = "%s/%s/%s/page" % (fam, lane, ds)
            rec = extra.get(rid)
        if rec is None:
            rec = extra[rid] = {"id": rid, "family": fam, "lane": lane, "dataset": ds, "cells": [],
                                "note": "measured outside the 0.8.34 M3 board (%s); carried from %s" % (
                                    "2026-09-29 M3 tree board and AFB tree runs" if fam == "trees" else "AFB runs",
                                    page_from)}
            stats["extra_races"] += 1
        status = g("status")
        # a FLIP the page could not compute (no ratio before) is a hand note; keep it
        flip_note = "; ".join(x.strip() for x in g("flip").split(";") if x.strip() and (
            not x.strip().startswith("FLIP") or g("ratio before") == "-"))
        meta = rec["%s_page" % mode] = {"family": fam, "status": status}
        if flip_note:
            meta["flip_note"] = flip_note
        if mode == "fast" and (lane, ds) in EXCLUDED:
            meta["excluded"] = EXCLUDED[(lane, ds)]
        if (lane, ds) in OPPONENT_NOTES:
            rec["opponent_note"] = OPPONENT_NOTES[(lane, ds)]
        fc = R.our_cell(rec, mode)
        if fc is None:
            fc = {"arm": "ours-fast" if mode == "fast" else "ours", "library": "mojolearn", "mode": mode,
                  "phase": "fit", "device": "gpu", "lane": lane, "dataset": ds, "family": fam,
                  "status": "no %s cell" % M, "median_ms": None, "quality": {},
                  "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}
            rec["cells"].append(fc)
        old_ms, old_q = R.ok_ms(fc), fc.get("quality") or {}
        after, before = R._num(g("FAST after ms")), R._num(g("FAST before ms"))
        qa, qb = g("quality after (FAST)"), g("quality before (FAST)")
        old_src = fc.get("source") if isinstance(fc.get("source"), dict) else None
        if after is not None and old_ms is None and fc.get("status") != "ok" and old_src is not None \
                and R.fmt_ms(fc.get("median_ms")) == g("FAST after ms"):
            # an ident sweep wrote the time but left the 0.8.34 REFUSED status (af_board_ident_update.py, fixed)
            old_src["previous_status"] = fc.get("status")
            fc["status"] = "ok"
            old_ms = R.ok_ms(fc)
            stats["%s_status_fixed" % mode] += 1
        if after is not None and R.fmt_ms(old_ms) != g("FAST after ms"):
            segs = [s.strip() for s in status.split(";")]
            ab = [s for s in segs if "A/B" in s]
            if ab:
                kind, tag = "A/B", ab[-1]
            elif status.strip() != "ok":
                kind, tag = "page note", status
            else:
                kind, tag = "refresh", "M3 %s refresh (tools/af_board_merge.py runs listed on the page)" % M
            stats["%s_%s" % (mode, kind.replace(" ", "_").replace("/", ""))] += 1
            base = old_ms
            if old_src and "previous_median_ms" in old_src:
                base = old_src["previous_median_ms"]
            src = {"tag": tag, "kind": kind, "from": page_from, "synced": a.date,
                   "previous_median_ms": old_ms, "previous_quality": old_q,
                   "previous_status": fc.get("status"), "previous_hash": fc.get("hash"),
                   "baseline_ms": base if base is not None else before,
                   "baseline_quality_text": qb if qb not in ("", "-") else (R.fmt_q(old_q) if old_ms else "-"),
                   "display_rounded": True}
            if old_src:
                src["history"] = [old_src]
            elif isinstance(fc.get("source"), str):
                src["stored"] = fc["source"]
            fc.update(median_ms=after, min_ms=after, max_ms=after, times_ms=[after], rounds=1, status="ok",
                      quality=qdict(qa), quality_text=qa, hash=None, source=src)
        else:
            stats["%s_same" % mode] += 1
            if qa not in ("", "-") and qa != R.fmt_q(old_q):
                fc["quality_text"] = qa
            if old_src is not None and qb not in ("", "-"):
                old_src["baseline_quality_text"] = qb
        # opponent
        parm, pms = g("best opponent"), R._num(g("opp ms"))
        best = R.best_opponent(rec)
        if parm != "-" and pms:
            arm = parm.replace(" (fill)", "")
            fill = parm.endswith("(fill)")
            if best and best["arm"] == arm:
                if g("opponent quality") not in ("", "-") and g("opponent quality") != (
                        best.get("quality_text") or R.fmt_q(best.get("quality"))):
                    best["quality_text"] = g("opponent quality")
                if R.fmt_ms(best["median_ms"]) != g("opp ms"):
                    log.append("%s %s (%s): opponent %s board %s ms vs page %s ms (board kept)" % (
                        lane, ds, mode, arm, R.fmt_ms(best["median_ms"]), g("opp ms")))
            elif fill or best is None:
                c = opp_cell(lane, ds, fam, arm, pms, g("opponent quality"), fill,
                             {"fill": "M3 opponent fill (tools/opp_only_board.py)" if fill else "page opponent",
                              "from": page_from, "display_rounded": True})
                if put_opp(rec, c, log):
                    stats["opp_fill_cells"] += 1
            else:
                log.append("%s %s (%s): page opponent %s %s ms, board best %s %s ms (board kept)" % (
                    lane, ds, mode, parm, g("opp ms"), best["arm"], R.fmt_ms(best["median_ms"])))
        elif best is not None:
            stats["%s_opp_from_board" % mode] += 1
        if rid in board["races"]:
            bb.add_ratios(rec["cells"])


def prose(lines, t, mode, date):
    M = mode.upper()
    hi = next((i for i, l in enumerate(lines) if l.startswith("Canonical full-board summary")
               or l.startswith("Summary: ")), None)
    qi = next((i for i, l in enumerate(lines) if l.startswith("Quality (")), None)
    intro = lines[2] if lines[2].startswith("Our %s arm" % M) else ""
    hist = []
    if hi is not None:
        hist.append("- Hand-made headline, last before the page was generated: %s" % lines[hi])
    if qi is not None:
        hist.append("- Hand-made quality paragraph, last before generation: %s" % lines[qi])
    bullets = [l for l in lines[hi + 1:t.hdr_i] if l.startswith("- ")] if hi is not None else []
    notes = list(lines[t.end:])
    while notes and notes[0] == "":
        notes.pop(0)
    return {"updated": date, "headline_history": hist + bullets, "notes_md": notes,
            "intro": intro.replace("Written by `tools/af_board_merge.py`.",
                                   "First written by `tools/af_board_merge.py`; now carried in board.json.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-dir", default=R.BOARD_DIR)
    ap.add_argument("--fast-page", default=os.path.join(R.DOCS, "BOARD_M3_FAST.md"))
    ap.add_argument("--ident-page", default=os.path.join(R.DOCS, "BOARD_M3_IDENTICAL.md"))
    ap.add_argument("--docs-dir", default=R.DOCS)
    ap.add_argument("--opp-lines", action="append", default=[])
    ap.add_argument("--report", default=os.path.join(EVID, "board-sync-report-2026-10-05.md"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import bench_board as bb
    from collections import Counter

    path = os.path.join(a.board_dir, "board.json")
    board = json.load(open(path))
    if board.get("fast_page"):
        sys.exit("board.json already carries fast_page: the pages are generated now; nothing to migrate")
    stats, log, olds = Counter(), [], {}
    for mode, pg in (("fast", a.fast_page), ("identical", a.ident_page)):
        lines = open(pg).read().split("\n")
        t = Table(lines)
        olds[mode] = (t, lines)
        page_from = "docs/apple-fast/%s (hand-made page, 2026-10-04)" % os.path.basename(pg)
        sync_page(board, t, mode, page_from, a, stats, log)
        board["%s_page" % mode] = prose(lines, t, mode, a.date)
    n_opp_lines = 0
    sources = [("opp_lines_m3.txt (built in)", OPP_LINES)] + [
        (p, open(os.path.expanduser(p)).read().splitlines()) for p in a.opp_lines]
    for p, plines in sources:
        for ln in plines:
            if not ln.startswith("OPP "):
                continue
            kv = dict(re.findall(r"(\w+)=(\{[^}]*\}|\S+)", ln))
            ms = R._num(kv.get("median_ms"))
            if kv.get("status") != "ok" or ms is None:
                continue
            rid, rec = R.race_key(board, kv["lane"], kv["ds"], kv.get("family"))
            if rec is None:
                log.append("OPP line without a race: %s %s" % (kv["lane"], kv["ds"]))
                continue
            c = opp_cell(kv["lane"], kv["ds"], kv.get("family"), kv["arm"], ms, None, True,
                         {"fill": "M3 opponent fill line (measured)", "from": os.path.basename(p)})
            c["quality"] = json.loads(kv.get("quality") or "{}")
            if put_opp(rec, c, log):
                n_opp_lines += 1
                if rid in board["races"]:
                    bb.add_ratios(rec["cells"])
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board.setdefault("box_history", []).append({"page_sync": "BOARD_M3_FAST.md + BOARD_M3_IDENTICAL.md",
                                                "date": a.date, **stats})
    rep = ["# Board sync report (%s)" % a.date, "",
           "`tools/af_board_full_sync.py`: board.json is now the one M3 board; BOARD.md, "
           "docs/apple-fast/BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md are generated from it by "
           "`tools/af_board_render.py` (`--check` guards them).", "",
           "## Synced into board.json", "",
           "- counts: %s" % ", ".join("%s %d" % kv for kv in sorted(stats.items())),
           "- opponent cells from OPP lines (knn-imputer istella sklearn-cpu 26.045 ms, replacing REFUSED(timeout)): %d"
           % n_opp_lines,
           "- quantile istella: opponent rendered as \"%s\", left out of eligible comparisons" %
           OPPONENT_NOTES[("quantile", "istella")],
           "- FAST excluded (quality under review, from the page's Quality flags): %s" %
           "; ".join("%s %s" % k for k in EXCLUDED), ""]
    for mode in ("fast", "identical"):
        rep += diff_report(olds[mode][0], olds[mode][1], R.render_page(board, mode), mode)
    rep += ["## Board kept over the page (%d)" % len(log), ""] + ["- " + x for x in log]
    print("\n".join(l for l in rep if l.startswith("- ")))
    if a.dry_run:
        with open(os.path.join(EVID, "board-sync-dryrun.md"), "w") as fh:
            fh.write("\n".join(rep) + "\n")
        return
    bk = os.path.join(a.board_dir, "board.before-fast-sync-%s.json" % a.date.replace("-", ""))
    if not os.path.exists(bk):
        shutil.copy2(path, bk)
    with open(path, "w") as fh:
        json.dump(board, fh, indent=1)
    R.write_all(a.board_dir, a.docs_dir, board)
    with open(a.report, "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print("wrote %s and the three pages; report %s" % (path, a.report))


def diff_report(t_old, old_lines, new_text, mode):
    M = mode.upper()
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
    old_head = next((l for l in old_lines if l.startswith("Canonical full-board summary") or l.startswith("Summary: ")), "?")
    new_head = next(l for l in nl if l.startswith("Canonical full-board summary"))
    rep = ["## %s page" % M, "", "- old headline (hand-made): %s" % old_head[:300],
           "- new headline (generated): %s" % new_head[:300],
           "- rows on both: %d, changed: %d, unchanged: %d; only on the generated page: %d (board.json races the "
           "hand page left out); only on the old page: %d" % (
               len(set(old) & set(new)), len(changed), len(set(old) & set(new)) - len(changed), len(only_new),
               len(only_old)),
           "- changed cells by column: %s" % (", ".join("%s %d" % kv for kv in sorted(why_count.items(),
                                                                                     key=lambda x: -x[1])) or "none"),
           "", "Why: before = the precise 0.8.34 board cell (the page used rounded TSV values, or '-' on rows it "
           "added by hand although the board had a 0.8.34 cell); opponent / opp ms = the board's best ok opponent "
           "(the page used a rounded TSV value or had '-'); ratios and flips recomputed from those; after = the page "
           "value printed by the board's formatter (700.3 -> 700); quality before = the 0.8.34 cell's quality "
           "where the page had '-'.", "",
           "| lane | dataset | family | column: old -> new |", "|---|---|---|---|"]
    for k, diffs in changed:
        rep.append("| %s | %s | %s | %s |" % (k[0], k[1], k[2], "; ".join(
            "%s: %s -> %s" % (c, o[:60], n[:60]) for c, o, n in diffs).replace("|", "/")))
    rep += ["", "Rows only on the generated %s page (%d): %s" % (M, len(only_new), ", ".join(
        "%s %s (%s)" % k for k in only_new) or "none"), "",
            "Rows only on the old %s page (%d): %s" % (M, len(only_old), ", ".join(
                "%s %s (%s)" % k for k in only_old) or "none"), ""]
    return rep


if __name__ == "__main__":
    main()
