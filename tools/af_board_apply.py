#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apply A/B winners that became FAST defaults to docs/apple-fast/BOARD_M3_FAST.md in place.

    python3 tools/af_board_apply.py WINNERS.tsv [--board PATH] [--date YYYY-MM-DD] [--dry-run]
                                    [--allow-quality-drop REASON]

Laptop, text only. WINNERS.tsv has a header row with columns
    lane  dataset  new_ms  quality  source  [family]
quality is a short "metric=value" string; source names the evidence, e.g.
"VSEARCH defaults ad265a028 M3 A/B 2026-10-04".

Per row: set "FAST after ms" and "quality after (FAST)", recompute "ratio after"
(new_ms / opp ms) and the flip ("FLIP faster" when ratio before > 1 and after < 1,
"FLIP slower" for the reverse; other flip notes are kept), append the source to
status. A (lane, dataset) missing from the table is added with family from the
input (default "algos"), opponent columns "-" and a note in status. The table is
then stable-sorted worst ratio after first (rows without a ratio last), which also
puts back any hand-edited row that sat out of order.

Quality gate (CLAUDE.md: FAST passes only if quality does not go down). Before a row
is applied, its new quality is compared (tools/af_quality.py, rel_tol 1e-3, abs_tol 1e-6)
against the row's "quality after (FAST)" (current main; "quality before (FAST)" when the
after cell is empty) and against "opponent quality". A row WORSE than either is REFUSED
and the reason printed; the other rows still apply, and the exit code is 2. Pass
--allow-quality-drop "REASON" to apply it anyway; the reason goes into its status.
Every applied row gets one status tag, replacing an older one: "Q: =main >=opp" (or
">main"), "Q: < opp", "Q: < main" (only with --allow-quality-drop) or "Q: unknown" when
either side has no comparable metric (unknown metrics are never passed silently).

Headline: the "Canonical full-board summary" paragraph counts the full page (this
table plus the original board), which this file alone cannot rebuild. So the
apply updates it by delta: rows += added rows; each changed row leaves the
eligible set with its old ratio and re-enters with its new one (eligible = numeric
ratio after and status without HOLD or "excluded"); faster = ratio < 1; the
geometric mean is rebuilt from N*ln(old gm) minus the old log ratios plus the new.
The rest of the paragraph is kept; a dated bullet listing this apply's changes is
inserted after it. --dry-run prints the changed rows and the new headline only.
"""
import argparse, math, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import af_quality as afq

BOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "apple-fast", "BOARD_M3_FAST.md")
HEAD_RE = re.compile(r"^Canonical full-board summary \((?P<date>[^)]*)\): (?P<rows>\d+) FAST rows, (?P<el>\d+) eligible "
                     r"opponent comparisons, (?P<fast>\d+) faster, geometric-mean ratio (?P<gm>[0-9.]+)\.")


def _num(x):
    try:
        v = float(str(x).replace(",", "").split()[0])
        return v if v == v else None
    except (TypeError, ValueError, IndexError):
        return None


def _fmt_ms(v):
    return "-" if v is None else ("%.0f" % v if v >= 100 else "%.1f" % v)


def _fmt_r(v):
    return "-" if v is None else "%.2f" % v


def _split(line):
    return [c.strip() for c in line.strip()[1:-1].split("|")]


def _join(cells):
    return "| " + " | ".join(cells) + " |"


class Table:
    def __init__(self, lines):
        self.hdr_i = next(i for i, l in enumerate(lines) if l.startswith("| lane |"))
        self.cols = _split(lines[self.hdr_i])
        need = ["lane", "dataset", "family", "FAST before ms", "FAST after ms", "best opponent", "opp ms",
                "ratio before", "ratio after", "flip", "quality after (FAST)", "status"]
        miss = [n for n in need if n not in self.cols]
        if miss:
            sys.exit("board table lacks columns: %s" % ", ".join(miss))
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

    def put(self, r, name, v):
        r[self.ix[name]] = v

    def ratio(self, r):
        """Precise ratio after (after ms / opp ms) when both are numbers, else the printed one."""
        a, o = _num(self.get(r, "FAST after ms")), _num(self.get(r, "opp ms"))
        if a and o:
            return a / o
        return _num(self.get(r, "ratio after"))

    def eligible(self, r):
        st = self.get(r, "status")
        ra = self.ratio(r)
        return ra is not None and ra > 0 and "HOLD" not in st and "excluded" not in st.lower()


Q_TAG_RE = re.compile(r"(;\s*)?Q: [^;]*")


def quality_gate(t, row, new_q):
    """(main compare, opp compare, tag) for new quality new_q against an existing row (or None)."""
    if row is None:
        none = afq.compare(new_q, "-")
        return none, none, "Q: unknown"
    main_q = t.get(row, "quality after (FAST)")
    if main_q in ("", "-") and "quality before (FAST)" in t.ix:
        main_q = t.get(row, "quality before (FAST)")
    opp_q = t.get(row, "opponent quality") if "opponent quality" in t.ix else "-"
    cm, co = afq.compare(new_q, main_q), afq.compare(new_q, opp_q)
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


def _sort_key(t, r):
    ra = _num(t.get(r, "ratio after"))
    return (ra is None, -(ra or 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv")
    ap.add_argument("--board", default=BOARD)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-quality-drop", metavar="REASON", default=None,
                    help="apply rows whose quality is WORSE than main or the opponent; REASON goes into status")
    a = ap.parse_args()

    lines = open(a.board).read().split("\n")
    t = Table(lines)
    tl = [l for l in open(a.tsv).read().splitlines() if l.strip() and not l.startswith("#")]
    hdr = [h.strip() for h in tl[0].split("\t")]
    for n in ("lane", "dataset", "new_ms", "quality", "source"):
        if n not in hdr:
            sys.exit("%s: header lacks column %r" % (a.tsv, n))
    wins = [dict(zip(hdr, [c.strip() for c in l.split("\t")])) for l in tl[1:]]

    head_i = next((i for i, l in enumerate(lines) if l.startswith("Canonical full-board summary")), None)
    m = HEAD_RE.match(lines[head_i]) if head_i is not None else None
    if not m:
        sys.exit("headline paragraph 'Canonical full-board summary (...): N FAST rows, ...' not found")
    n_rows, n_el, n_fast, gm = int(m["rows"]), int(m["el"]), int(m["fast"]), float(m["gm"])
    logsum = n_el * math.log(gm)

    if a.allow_quality_drop is not None and not a.allow_quality_drop.strip():
        sys.exit("--allow-quality-drop needs a reason")
    changed, notes, refused = [], [], []
    for w in wins:
        lane, ds, new_ms = w["lane"], w["dataset"], _num(w["new_ms"])
        if new_ms is None or new_ms <= 0:
            sys.exit("bad new_ms for %s %s: %r" % (lane, ds, w["new_ms"]))
        src = w["source"]
        row = next((r for r in t.rows if t.get(r, "lane") == lane and t.get(r, "dataset") == ds), None)
        cm, co, qtag = quality_gate(t, row, w["quality"])
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
        added = row is None
        if added:
            row = ["-"] * len(t.cols)
            for n, v in (("lane", lane), ("dataset", ds), ("family", w.get("family") or "algos"), ("flip", ""),
                         ("status", "ok; new row (no opponent on record)")):
                t.put(row, n, v)
            t.rows.append(row)
            n_rows += 1
            old_ms, old_ra, was_el = None, None, False
        else:
            old_ms, old_ra, was_el = _num(t.get(row, "FAST after ms")), t.ratio(row), t.eligible(row)
        if was_el:
            n_el -= 1
            n_fast -= old_ra < 1
            logsum -= math.log(old_ra)
        t.put(row, "FAST after ms", _fmt_ms(new_ms))
        t.put(row, "quality after (FAST)", w["quality"] or "-")
        opp = _num(t.get(row, "opp ms"))
        ra = new_ms / opp if opp else None
        t.put(row, "ratio after", _fmt_r(ra))
        rb = _num(t.get(row, "ratio before"))
        flip_old = t.get(row, "flip")
        if rb is not None and ra is not None:
            keep = flip_old if flip_old and not flip_old.startswith("FLIP") else ""
            flip = "FLIP faster" if rb > 1 > ra else ("FLIP slower" if rb < 1 < ra else "")
            t.put(row, "flip", "; ".join(x for x in (flip, keep) if x))
        st = Q_TAG_RE.sub("", t.get(row, "status")).strip("; ").strip()
        st = "%s; %s" % (st, src) if st and st != "-" else src
        t.put(row, "status", "%s; %s" % (st, qtag))
        if t.eligible(row):
            n_el += 1
            n_fast += t.ratio(row) < 1
            logsum += math.log(t.ratio(row))
        changed.append(row)
        f = t.get(row, "flip")
        notes.append("%s %s %s -> %s ms (ratio %s -> %s%s; %s; %s)%s" % (
            lane, ds, _fmt_ms(old_ms), _fmt_ms(new_ms), _fmt_r(old_ra), _fmt_r(ra),
            ", " + f if f.startswith("FLIP") else "", src, qtag, " [new row]" if added else ""))

    if not changed:
        print("nothing applied: %d row(s) refused on quality" % len(refused), file=sys.stderr)
        sys.exit(2 if refused else 0)

    # worst ratio after first, rows without a ratio last; stable, so ties keep their order
    t.rows.sort(key=lambda r: _sort_key(t, r))
    new_gm = math.exp(logsum / n_el) if n_el else float("nan")
    old_head = lines[head_i]
    new_head = old_head[:m.start("date")] + a.date + old_head[m.end("date"):m.start("rows")] + "%d FAST rows, %d eligible " \
        "opponent comparisons, %d faster, geometric-mean ratio %.3f." % (n_rows, n_el, n_fast, new_gm) + old_head[m.end():]
    bullet = "- Board apply %s (`tools/af_board_apply.py`): %s." % (a.date, "; ".join(notes))

    if a.dry_run:
        print("CHANGED ROWS (%d):" % len(changed))
        print(_join(t.cols))
        for r in changed:
            print(_join(r))
        print("\nHEADLINE (was %s rows / %s eligible / %s faster / gm %s):" % (m["rows"], m["el"], m["fast"], m["gm"]))
        print(new_head[:m.end() - m.start() + 40] + " ...")
        print("\nAPPLY LINE:\n" + bullet)
        if refused:
            print("\nREFUSED ON QUALITY (%d):\n  %s" % (len(refused), "\n  ".join(refused)))
            sys.exit(2)
        return
    lines[t.start:t.end] = [_join(r) for r in t.rows]
    lines[head_i] = new_head
    # the bullet list follows the headline after one blank line
    ins = head_i + 1
    if ins < len(lines) and lines[ins] == "":
        ins += 1
    lines.insert(ins, bullet)
    if ins == head_i + 1:
        lines.insert(ins, "")
    with open(a.board, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote %s: %d rows changed, headline %d rows / %d eligible / %d faster / gm %.3f" % (
        a.board, len(changed), n_rows, n_el, n_fast, new_gm))
    if refused:
        print("REFUSED ON QUALITY (%d):\n  %s" % (len(refused), "\n  ".join(refused)))
        sys.exit(2)


if __name__ == "__main__":
    main()
