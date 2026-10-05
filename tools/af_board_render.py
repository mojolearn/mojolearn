#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Generate docs/apple-fast/BOARD_M3_FAST.md (and BOARD.md) from the one M3 board, board.json.

    python3 tools/af_board_render.py [--board-dir DIR] [--docs-dir DIR]     # write all three pages
    python3 tools/af_board_render.py --check [--board-dir DIR] [--docs-dir DIR]

Laptop, text only. bench/results/bench_board/m3ultra-0834/board.json is the single source of
truth for the M3 board. BOARD.md (bench_board.write_board), docs/apple-fast/BOARD_M3_FAST.md and
docs/apple-fast/BOARD_M3_IDENTICAL.md are generated, never hand-edited. Per mode (FAST, IDENTICAL):

- rows: every race (board["races"], plus board["extra_races"] for races measured outside this
  board: the 2026-09-29 tree board and a few AFB-only lanes) that has a mojolearn cell of that
  mode or a per-race "<mode>_page" entry;
- after = the race's cell (library mojolearn, that mode, fit phase, not our CPU);
  before = cell["source"]["baseline_ms"] (the 0.8.34 value, or the oldest page value), else
  source previous_median_ms (an ident sweep), else the cell itself when it never changed;
- best opponent = fastest ok opponent cell (tools/af_board_merge.py's rule); "(fill)" when the
  cell came from an M3 opponent fill; ratio = FAST / best opponent; flip from ratio before -> after;
- quality: FAST cell quality_text or quality, baseline quality from source, opponent quality;
- status = race["<mode>_page"]["status"] (sources and Q tags written by tools/af_board_apply.py);
- the Quality paragraph (tools/af_quality.py comparisons, as tools/af_board_quality_audit.py)
  and the headline counts are computed here; board["<mode>_page"] holds the stored prose
  (intro, headline history, notes).

--check re-renders in memory and fails (exit 1, a line per mismatch) when any of the three
pages differs from a fresh render, so a hand edit cannot survive. tools/af_board_apply.py
always ends with this check.
"""
import argparse, difflib, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import af_quality as afq

BOARD_DIR = os.path.join(REPO, "bench", "results", "bench_board", "m3ultra-0834")
DOCS = os.path.join(REPO, "docs", "apple-fast")
PAGE = os.path.join(DOCS, "BOARD_M3_FAST.md")
GENERATED = "Generated from board.json by tools/af_board_render.py; do not edit."
COLS = ["lane", "dataset", "family", "FAST before ms", "FAST after ms", "best opponent", "opp ms", "ratio before",
        "ratio after", "flip", "quality after (FAST)", "quality before (FAST)", "opponent quality", "status"]
MATERIAL = 0.01


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def fmt_ms(v):
    return "-" if v is None else ("%.0f" % v if v >= 100 else "%.1f" % v)


def fmt_r(v):
    return "-" if v is None else "%.2f" % v


def fmt_q(q):
    """Quality dict -> board text (5 significant digits stored, 4 shown, as af_board_merge)."""
    if not isinstance(q, dict) or not q:
        return "-"
    vals = {k: float("%.5g" % v) for k, v in q.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("_matches_fit")}
    if not vals:
        return "-"
    return ", ".join("%s=%.4g" % (k, v) for k, v in sorted(vals.items()))[:90]


def cell_text(s):
    return str(s).replace("|", "/").replace("\n", " ")


def is_ours(c):
    return c.get("library") == "mojolearn" or str(c.get("arm", "")).startswith("ours")


def our_cell(rec, mode="fast"):
    """The race's FAST or IDENTICAL cell (any status), fit phase, never our CPU."""
    for c in rec.get("cells") or []:
        if c.get("phase") not in (None, "fit"):
            continue
        if is_ours(c) and c.get("mode") == mode and c.get("device") != "cpu" and c.get("arm") != "ours-cpu":
            return c
    return None


def fast_cell(rec):
    return our_cell(rec, "fast")


def page_meta(rec, mode="fast"):
    return rec.get("%s_page" % mode) or {}


def opponents(rec):
    return [c for c in rec.get("cells") or [] if c.get("phase") in (None, "fit") and not is_ours(c)
            and c.get("status") == "ok" and c.get("median_ms")]


def best_opponent(rec):
    opp = opponents(rec)
    return min(opp, key=lambda c: c["median_ms"]) if opp else None


def ok_ms(c):
    return c["median_ms"] if c and c.get("status") == "ok" and c.get("median_ms") else None


def all_races(board):
    out = dict(board.get("races") or {})
    out.update(board.get("extra_races") or {})
    return out


def race_key(board, lane, ds, fam=None):
    hits = [(rid, rec) for rid, rec in all_races(board).items()
            if rec.get("lane") == lane and rec.get("dataset") == ds]
    if fam and len(hits) > 1:
        hits = [h for h in hits if h[1].get("family") == fam or
                (h[1].get("fast_page") or {}).get("family") == fam] or hits
    return hits[0] if hits else (None, None)


def eligible(row):
    st = row["status"]
    meta = page_meta(row["rec"], row["mode"])
    if row["rec"].get("opponent_note") or meta.get("excluded"):
        return False  # opponent not measured (too slow), or quality under review
    return row["ra"] is not None and row["ra"] > 0 and "HOLD" not in st and "excluded" not in st.lower()


def _src(c):
    return c.get("source") if c and isinstance(c.get("source"), dict) else {}


def has_before(c):
    s = _src(c)
    return "baseline_ms" in s or "previous_median_ms" in s


def rows_of(board, mode="fast"):
    rows = []
    for rid, rec in all_races(board).items():
        fp = page_meta(rec, mode)
        fc = our_cell(rec, mode)
        if fc is None and not fp:
            continue
        src = _src(fc)
        after = ok_ms(fc)
        if "baseline_ms" in src:
            before = _num(src["baseline_ms"])
        elif "previous_median_ms" in src:
            before = _num(src["previous_median_ms"])
        else:
            before = after
        best = best_opponent(rec)
        bms = best["median_ms"] if best else None
        rb = before / bms if before and bms else None
        ra = after / bms if after and bms else None
        flip = ""
        if rb is not None and ra is not None:
            flip = "FLIP faster" if rb > 1 > ra else ("FLIP slower" if rb < 1 < ra else "")
        flip = "; ".join(x for x in (flip, fp.get("flip_note", "")) if x)
        qa = (fc or {}).get("quality_text") or fmt_q((fc or {}).get("quality"))
        qb = src.get("baseline_quality_text") or ("-" if has_before(fc) else qa)
        qo = "-"
        if best:
            qo = best.get("quality_text") or fmt_q(best.get("quality"))
        if fp.get("status"):
            status = fp["status"]
        elif fc is None:
            status = "no %s cell" % mode.upper()
        else:
            status = "ok" if fc.get("status") == "ok" else fc.get("status", "?")[:120]
        if fp.get("excluded") and "excluded" not in status.lower():
            status += "; excluded: " + fp["excluded"]
        arm = rec.get("opponent_note") or "-"
        if best:
            arm = best["arm"] + (" (fill)" if best.get("fill") else "")
        rows.append(dict(rid=rid, lane=rec["lane"], ds=rec["dataset"], mode=mode,
                         fam=fp.get("family") or (rec.get("fast_page") or {}).get("family") or rec.get("family"),
                         before=before, after=after, best_arm=arm, best=bms, rb=rb, ra=ra, flip=flip,
                         qa=qa or "-", qb=qb or "-", qo=qo, status=status, rec=rec))
    rows.sort(key=lambda r: (r["ra"] is None, -(r["ra"] or 0.0), r["lane"], r["ds"], r["fam"] or ""))
    return rows


def quality_counts(rows):
    vs_opp = vs_before = mat_opp = mat_before = no_q = 0
    unk_rows, unk_metrics = 0, set()
    for r in rows:
        fq = r["qa"]
        if not afq.parse(fq):
            no_q += 1
            continue
        oqs = [c.get("quality_text") or c.get("quality") for c in opponents(r["rec"])]
        worst, unk = None, set()
        for oq in oqs:
            c = afq.compare(fq, oq)
            unk.update(c["unknown"])
            if c["verdict"] == afq.WORSE and (worst is None or c["worst"] < worst):
                worst = c["worst"]
        if worst is not None:
            vs_opp += 1
            mat_opp += worst < -MATERIAL
        if has_before(our_cell(r["rec"], r["mode"])) and r["qb"] not in ("", "-"):
            c = afq.compare(fq, r["qb"])
            unk.update(c["unknown"])
            if c["verdict"] == afq.WORSE:
                vs_before += 1
                mat_before += c["worst"] < -MATERIAL
        if unk:
            unk_rows += 1
            unk_metrics.update(unk)
    return dict(pairs=len(rows), vs_opp=vs_opp, mat_opp=mat_opp, vs_before=vs_before, mat_before=mat_before,
                unk_rows=unk_rows, unk_metrics=len(unk_metrics), no_q=no_q)


def headline_counts(rows):
    timed = [r for r in rows if r["after"] is not None]
    el = [r for r in rows if eligible(r)]
    fast = sum(1 for r in el if r["ra"] < 1)
    gm = statistics.geometric_mean([r["ra"] for r in el]) if el else float("nan")
    return dict(rows=len(timed), el=len(el), fast=fast, gm=gm)


CHECKS = {
    "fast": "- Quality gate: `tools/af_board_apply.py` refuses a FAST row whose quality is WORSE than FAST main or "
            "the best opponent (tools/af_quality.py, rel 1e-3, abs 1e-6) unless `--allow-quality-drop REASON` "
            "is given; every applied row carries a Q: tag in status.",
    "identical": "- IDENTICAL cells come from M3 ident sweeps (`tools/af_board_ident_update.py`, which writes "
                 "board.json and re-renders every page); same-bits checks are separate (`lq add <box> ID ...`).",
}


def render_page(board, mode="fast"):
    M = mode.upper()
    fp = board.get("%s_page" % mode) or {}
    date = (fp.get("updated") or board.get("updated") or "?")[:10]
    rows = rows_of(board, mode)
    q = quality_counts(rows)
    h = headline_counts(rows)
    out = ["# M3 %s board" % M, "", GENERATED + " Source: `bench/results/bench_board/m3ultra-0834/board.json` "
           "(the one M3 board; BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md are rendered from it).", "",
           "## Checks", "", CHECKS[mode],
           "- Render check: `python3 tools/af_board_render.py --check` fails when BOARD.md, BOARD_M3_FAST.md or "
           "BOARD_M3_IDENTICAL.md differ from a fresh render of board.json. `tools/af_board_apply.py` and "
           "`tools/af_board_ident_update.py` always run it; `tools/af_board_render_watch.sh` (called from ~/mojolearn-evidence/apple_watch.sh) runs it against "
           "origin/main and prints ALERT on a mismatch.", ""]
    if fp.get("intro"):
        out += [fp["intro"], ""]
    out += ["Quality (%s, computed): of %d lane/dataset rows, %s quality is WORSE than an opponent on %d (%d by "
            "more than 1%%) and %s after is WORSE than %s before on %d (%d by more than 1%%); %d rows carry "
            "metrics of unknown direction (%d metrics), %d have no parseable %s quality. Tolerance rel 1e-3, "
            "abs 1e-6." % (date, q["pairs"], M, q["vs_opp"], q["mat_opp"], M, M, q["vs_before"], q["mat_before"],
                           q["unk_rows"], q["unk_metrics"], q["no_q"], M), "",
            "Canonical full-board summary (%s): %d %s rows, %d eligible opponent comparisons, %d faster, "
            "geometric-mean ratio %.3f. Eligible = a ratio after and a status without HOLD or \"excluded\"; "
            "faster = ratio below 1. Ratio = %s ms / best opponent ms." % (date, h["rows"], M, h["el"], h["fast"],
                                                                            h["gm"], M), ""]
    hist = fp.get("headline_history") or []
    if hist:
        out += hist + [""]
    cols = [c.replace("FAST", M) for c in COLS]
    out += ["| " + " | ".join(cols) + " |", "|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(cell_text(x) for x in (
            r["lane"], r["ds"], r["fam"], fmt_ms(r["before"]), fmt_ms(r["after"]), r["best_arm"], fmt_ms(r["best"]),
            fmt_r(r["rb"]), fmt_r(r["ra"]), r["flip"], r["qa"], r["qb"], r["qo"], r["status"])) + " |")
    notes = fp.get("notes_md") or []
    if notes:
        out += [""] + notes
    return "\n".join(out).rstrip("\n") + "\n"


def _bb():
    import bench_board as bb
    return bb


def pages(docs_dir=DOCS):
    return {"fast": os.path.join(docs_dir, "BOARD_M3_FAST.md"), "identical": os.path.join(docs_dir, "BOARD_M3_IDENTICAL.md")}


def write_all(board_dir=BOARD_DIR, docs_dir=DOCS, board=None):
    """Write BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md from board.json (or `board`)."""
    import json
    if board is None:
        board = json.load(open(os.path.join(board_dir, "board.json")))
    _bb().write_board(board_dir, board)
    for mode, path in pages(docs_dir).items():
        with open(path, "w") as fh:
            fh.write(render_page(board, mode))
    return board


def check(board_dir=BOARD_DIR, docs_dir=DOCS):
    """List of mismatch lines (empty = all three pages match a fresh render of board.json)."""
    import json
    board = json.load(open(os.path.join(board_dir, "board.json")))
    want = [("BOARD.md", os.path.join(board_dir, "BOARD.md"), _bb().render_board(board))]
    want += [(os.path.basename(p), p, render_page(board, m)) for m, p in pages(docs_dir).items()]
    bad = []
    for name, path, fresh in want:
        have = open(path).read() if os.path.exists(path) else ""
        if have != fresh:
            diff = list(difflib.unified_diff(have.splitlines(), fresh.splitlines(), lineterm="", n=0))
            bad.append("MISMATCH %s: %d diff lines vs a fresh render of board.json; first: %s" % (
                name, len(diff), (diff[2:3] or ["?"])[0][:200]))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-dir", default=BOARD_DIR)
    ap.add_argument("--docs-dir", default=DOCS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        bad = check(a.board_dir, a.docs_dir)
        for b in bad:
            print(b)
        print("RENDER CHECK %s" % ("FAIL" if bad else "OK"))
        sys.exit(1 if bad else 0)
    board = write_all(a.board_dir, a.docs_dir)
    for mode in ("fast", "identical"):
        h = headline_counts(rows_of(board, mode))
        print("%s: %d rows, %d eligible, %d faster, gm %.3f" % (mode, h["rows"], h["el"], h["fast"], h["gm"]))


if __name__ == "__main__":
    main()
