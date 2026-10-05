#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apply A/B winners that became FAST defaults to the one M3 board, board.json, then
re-render both pages and check them.

    python3 tools/af_board_apply.py WINNERS.tsv [--board-dir DIR] [--page PATH] [--date YYYY-MM-DD]
                                    [--dry-run] [--allow-quality-drop REASON]

Laptop, text only. bench/results/bench_board/m3ultra-0834/board.json is the single source;
docs/apple-fast/BOARD_M3_FAST.md and BOARD.md are generated from it by
tools/af_board_render.py and never edited by hand. WINNERS.tsv has a header row with columns
    lane  dataset  new_ms  quality  source  [family]
quality is a short "metric=value" string; source names the evidence, e.g.
"VSEARCH defaults ad265a028 M3 A/B 2026-10-04".

Per row, the race's FAST cell (library mojolearn, mode fast) takes new_ms (median=min=max,
rounds=1), the quality (parsed, plus the text) and a `source` dict: tag, previous_median_ms,
previous_quality, previous_hash, baseline_ms (kept from the first change, so "FAST before" stays
the 0.8.34 value) and history (older sources). The race's fast_page status gets the source and
one Q tag. Ratios are recomputed (tools/bench_board.add_ratios). A (lane, dataset) with no race
becomes a fast_page extra race (family from the input, default "algos") with no opponent.

Quality gate (CLAUDE.md: FAST passes only if quality does not go down). Before a row is applied,
its new quality is compared (tools/af_quality.py, rel_tol 1e-3, abs_tol 1e-6) against the
current FAST quality (the before quality when there is none) and against the best opponent's
quality. A row WORSE than either is REFUSED and the reason printed; the other rows still
apply, and the exit code is 2. Pass --allow-quality-drop "REASON" to apply it anyway; the
reason goes into its status. Every applied row gets one status tag, replacing an older one:
"Q: =main >=opp" (or ">main"), "Q: < opp", "Q: < main" (only with --allow-quality-drop) or
"Q: unknown" when either side has no comparable metric.

A dated bullet listing this apply's changes goes first in board["fast_page"]["headline_history"];
the headline itself is computed by the renderer. The apply then writes board.json, re-renders
BOARD.md and the FAST page, and runs `tools/af_board_render.py --check`; a failed check exits 3.
--dry-run prints the changed rows and the new headline only.
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import af_quality as afq
import af_board_render as R

BOARD = R.PAGE  # kept for tools that import it (af_board_quality_audit.py)
Q_TAG_RE = re.compile(r"(;\s*)?Q: [^;]*")


def _num(x):
    try:
        v = float(str(x).replace(",", "").split()[0])
        return v if v == v else None
    except (TypeError, ValueError, IndexError):
        return None


def _split(line):
    return [c.strip() for c in line.strip()[1:-1].split("|")]


class Table:
    """Read-only view of a rendered FAST page table (used by af_board_quality_audit.py)."""

    def __init__(self, lines):
        self.hdr_i = next(i for i, l in enumerate(lines) if l.startswith("| lane |"))
        self.cols = _split(lines[self.hdr_i])
        self.ix = {n: i for i, n in enumerate(self.cols)}
        self.start = self.hdr_i + 2
        self.end = self.start
        while self.end < len(lines) and lines[self.end].startswith("|"):
            self.end += 1
        self.rows = [_split(l) for l in lines[self.start:self.end]]
        for r in self.rows:
            if len(r) != len(self.cols):
                sys.exit("row has %d cells, header %d: %s" % (len(r), len(self.cols), r[:2]))

    def get(self, r, name):
        return r[self.ix[name]]


def quality_gate(row, new_q):
    """(main compare, opp compare, tag) for new quality new_q against a rendered row (or None)."""
    if row is None:
        none = afq.compare(new_q, "-")
        return none, none, "Q: unknown"
    main_q = row["qa"] if row["qa"] not in ("", "-") else row["qb"]
    cm, co = afq.compare(new_q, main_q), afq.compare(new_q, row["qo"])
    vm, vo = cm["verdict"], co["verdict"]
    if vo == afq.WORSE:
        tag = "Q: < opp" + (" < main" if vm == afq.WORSE else "")
    elif vm == afq.WORSE:
        tag = "Q: < main"
    elif afq.UNKNOWN in (vm, vo):
        tag = "Q: unknown"
    else:
        tag = "Q: %smain >=opp" % (">" if vm == afq.BETTER else "=")
    return cm, co, tag


def find_row(rows, lane, ds, fam=None):
    hits = [r for r in rows if r["lane"] == lane and r["ds"] == ds]
    if fam and len(hits) > 1:
        hits = [r for r in hits if r["fam"] == fam] or hits
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv")
    ap.add_argument("--board-dir", default=R.BOARD_DIR)
    ap.add_argument("--page", default=R.PAGE)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-quality-drop", metavar="REASON", default=None,
                    help="apply rows whose quality is WORSE than main or the opponent; REASON goes into status")
    a = ap.parse_args()
    import bench_board as bb

    path = os.path.join(a.board_dir, "board.json")
    board = json.load(open(path))
    fp_top = board.setdefault("fast_page", {})
    extra = fp_top.setdefault("extra_races", {})
    tl = [l for l in open(a.tsv).read().splitlines() if l.strip() and not l.startswith("#")]
    hdr = [h.strip() for h in tl[0].split("\t")]
    for n in ("lane", "dataset", "new_ms", "quality", "source"):
        if n not in hdr:
            sys.exit("%s: header lacks column %r" % (a.tsv, n))
    wins = [dict(zip(hdr, [c.strip() for c in l.split("\t")])) for l in tl[1:]]
    if a.allow_quality_drop is not None and not a.allow_quality_drop.strip():
        sys.exit("--allow-quality-drop needs a reason")

    old_rows = R.rows_of(board)
    old_head = R.headline_counts(old_rows)
    changed, notes, refused = [], [], []
    for w in wins:
        lane, ds, new_ms, fam = w["lane"], w["dataset"], _num(w["new_ms"]), w.get("family") or None
        if new_ms is None or new_ms <= 0:
            sys.exit("bad new_ms for %s %s: %r" % (lane, ds, w["new_ms"]))
        src = w["source"]
        row = find_row(old_rows, lane, ds, fam)
        cm, co, qtag = quality_gate(row, w["quality"])
        drop = [(who, c) for who, c in (("main", cm), ("opponent", co)) if c["verdict"] == afq.WORSE]
        if drop:
            why = "; ".join("vs %s %s" % (who, afq.describe(c)) for who, c in drop)
            if a.allow_quality_drop is None:
                refused.append("%s %s: %s" % (lane, ds, why))
                print("REFUSED %s %s: quality WORSE %s. Pass --allow-quality-drop REASON to apply it anyway."
                      % (lane, ds, why), file=sys.stderr)
                continue
            print("QUALITY DROP ALLOWED %s %s: %s (reason: %s)" % (lane, ds, why, a.allow_quality_drop),
                  file=sys.stderr)
            src = "%s; quality drop allowed: %s" % (src, a.allow_quality_drop)
        if row is not None:
            rid, rec = row["rid"], row["rec"]
        else:
            rid, rec = R.race_key(board, lane, ds)
        added = rec is None
        if added:
            fam = fam or "algos"
            rid = "%s/%s/%s/page" % (fam, lane, ds)
            rec = extra[rid] = {"id": rid, "family": fam, "lane": lane, "dataset": ds, "cells": [],
                                "note": "added by tools/af_board_apply.py %s; no board race" % a.date}
        fp = rec.setdefault("fast_page", {"family": fam or rec.get("family"), "status": "ok"})
        if added:
            fp["status"] = "ok; new row (no opponent on record)"
        fc = R.fast_cell(rec)
        if fc is None:
            fc = {"arm": "ours-fast", "library": "mojolearn", "mode": "fast", "phase": "fit", "device": "gpu",
                  "lane": lane, "dataset": ds, "family": rec.get("family"), "status": "no FAST cell",
                  "median_ms": None, "quality": {}, "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}
            rec["cells"].append(fc)
        old_ms = R.ok_ms(fc)
        old_qt = row["qa"] if row else "-"
        prev = fc.get("source") if isinstance(fc.get("source"), dict) else {}
        hist = list(prev.get("history") or [])
        if prev:
            hist.append({k: v for k, v in prev.items() if k != "history"})
        source = {"tag": src, "kind": "A/B" if "A/B" in src else "apply", "applied": a.date,
                  "previous_median_ms": old_ms, "previous_quality": fc.get("quality"),
                  "previous_hash": fc.get("hash"),
                  "baseline_ms": prev["baseline_ms"] if "baseline_ms" in prev else old_ms,
                  "baseline_quality_text": prev["baseline_quality_text"] if "baseline_quality_text" in prev
                  else old_qt, "history": hist}
        if isinstance(fc.get("source"), str):
            source["stored"] = fc["source"]
        fc.update(median_ms=new_ms, min_ms=new_ms, max_ms=new_ms, times_ms=[new_ms], rounds=1, status="ok",
                  quality={k: v for k, v in afq.parse(w["quality"]).items() if isinstance(v, float)},
                  quality_text=w["quality"] or "-", hash=None, source=source)
        st = Q_TAG_RE.sub("", fp.get("status") or "").strip("; ").strip()
        st = "%s; %s" % (st, src) if st and st != "-" else src
        fp["status"] = "%s; %s" % (st, qtag)
        if rid in (board.get("races") or {}):
            bb.add_ratios(rec["cells"])
        changed.append(rid)
        notes.append((rid, lane, ds, old_ms, row["ra"] if row else None, new_ms, src, qtag, added))

    if not changed:
        print("nothing applied: %d row(s) refused on quality" % len(refused), file=sys.stderr)
        sys.exit(2 if refused else 0)

    new_rows = {r["rid"]: r for r in R.rows_of(board)}
    parts = []
    for rid, lane, ds, old_ms, old_ra, new_ms, src, qtag, added in notes:
        nr = new_rows.get(rid) or {}
        f = nr.get("flip", "")
        parts.append("%s %s %s -> %s ms (ratio %s -> %s%s; %s; %s)%s" % (
            lane, ds, R.fmt_ms(old_ms), R.fmt_ms(new_ms), R.fmt_r(old_ra), R.fmt_r(nr.get("ra")),
            ", " + f if f.startswith("FLIP") else "", src, qtag, " [new row]" if added else ""))
    bullet = "- Board apply %s (`tools/af_board_apply.py`): %s." % (a.date, "; ".join(parts))
    fp_top["headline_history"] = [bullet] + list(fp_top.get("headline_history") or [])
    fp_top["updated"] = a.date
    h = R.headline_counts(list(new_rows.values()))
    head = "%d FAST rows / %d eligible / %d faster / gm %.3f (was %d / %d / %d / %.3f)" % (
        h["rows"], h["el"], h["fast"], h["gm"], old_head["rows"], old_head["el"], old_head["fast"], old_head["gm"])

    if a.dry_run:
        print("CHANGED ROWS (%d):" % len(changed))
        for rid in changed:
            r = new_rows.get(rid)
            if r:
                print("| %s | %s | %s -> %s ms | opp %s %s | ratio %s | %s |" % (
                    r["lane"], r["ds"], R.fmt_ms(r["before"]), R.fmt_ms(r["after"]), r["best_arm"],
                    R.fmt_ms(r["best"]), R.fmt_r(r["ra"]), r["status"]))
        print("\nHEADLINE: " + head)
        print("\nAPPLY LINE:\n" + bullet)
        if refused:
            print("\nREFUSED ON QUALITY (%d):\n  %s" % (len(refused), "\n  ".join(refused)))
            sys.exit(2)
        return
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board.setdefault("box_history", []).append({"fast_apply": os.path.basename(a.tsv), "date": a.date,
                                                "updated_cells": len(changed)})
    with open(path, "w") as fh:
        json.dump(board, fh, indent=1)
    R.write_all(a.board_dir, a.page, board)
    print("wrote %s, BOARD.md, %s: %d rows changed, headline %s" % (path, a.page, len(changed), head))
    bad = R.check(a.board_dir, a.page)
    for b in bad:
        print(b, file=sys.stderr)
    if bad:
        print("RENDER CHECK FAIL: the pages do not match board.json", file=sys.stderr)
        sys.exit(3)
    print("RENDER CHECK OK")
    if refused:
        print("REFUSED ON QUALITY (%d):\n  %s" % (len(refused), "\n  ".join(refused)))
        sys.exit(2)


if __name__ == "__main__":
    main()
