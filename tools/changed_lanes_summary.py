#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One line per changed lane from a `python -m mojolearn verify --json` report.

    python3 tools/changed_lanes_summary.py <report.json> --column cuda --lanes a,b [--out summary.txt]

Prints, for every lane asked about,

    CHANGED-LANE column=<c> lane=<lane> state=VERIFIED
    CHANGED-LANE column=<c> lane=<lane> state=DIVERGENT cell=<fixture>/<part> got=<hash12> ref=<hash12> [+N more]
    CHANGED-LANE column=<c> lane=<lane> state=NO-REFERENCE (OWED: no committed record carries it)

and a last line `CHANGED-LANES column=<c> lanes=N VERIFIED=.. DIVERGENT=.. NO-REFERENCE=.. OTHER=..`.
Grep-sized on purpose (Andrew 2026-10-13: changed-lane checks fold into the
build steps; one summary line per lane). Reads the report only: it runs
nothing and computes no lane value. Exit 0 when the report was read, 2 when it
was not (the caller records that as a failed check, never as a pass).
"""
import argparse
import json
import sys

STATE_WORD = {"OWED": "NO-REFERENCE"}


def _short(value):
    return str(value)[:12] if value is not None else "-"


def summarize(report, column, lanes):
    """The lines for `lanes` (a list) from a parsed verify --json report."""
    accounting = ((report.get("lane_accounting") or {}).get("lanes")) or {}
    cells = report.get("cells") or []
    counts = {"VERIFIED": 0, "DIVERGENT": 0, "NO-REFERENCE": 0, "OTHER": 0}
    lines = []
    for lane in lanes:
        entry = accounting.get(lane)
        state = (entry or {}).get("state") or "NOT IN REPORT"
        word = STATE_WORD.get(state, state).replace(" ", "-")
        line = f"CHANGED-LANE column={column} lane={lane} state={word}"
        if state == "DIVERGENT":
            bad = [c for c in cells if c.get("lane") == lane and c.get("state") == "DIVERGENT"]
            if bad:
                c = bad[0]
                line += (f" cell={c.get('fixture')}/{c.get('part')} got={_short(c.get('value'))} "
                         f"ref={_short(c.get('reference'))}" + (f" +{len(bad) - 1} more" if len(bad) > 1 else ""))
        elif state == "OWED":
            line += " (no committed record carries a reference for a part of it)"
        elif entry and entry.get("reason") and state != "VERIFIED":
            line += " (" + str(entry["reason"])[:160] + ")"
        counts[word if word in counts else "OTHER"] += 1
        lines.append(line)
    lines.append(f"CHANGED-LANES column={column} lanes={len(lanes)} "
                 + " ".join(f"{k}={v}" for k, v in counts.items()))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report")
    ap.add_argument("--column", required=True)
    ap.add_argument("--lanes", required=True, help="comma separated")
    ap.add_argument("--out", default="", help="also append the lines here")
    a = ap.parse_args(argv)
    lanes = [x for x in a.lanes.split(",") if x]
    try:
        with open(a.report) as fh:
            report = json.load(fh)
    except (OSError, ValueError) as exc:
        lines = [f"CHANGED-LANES column={a.column} lanes={len(lanes)} REPORT-UNREADABLE ({exc})"]
        rc = 2
    else:
        lines, rc = summarize(report, a.column, lanes), 0
    print("\n".join(lines))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
