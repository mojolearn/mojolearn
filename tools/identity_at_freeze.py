#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY ONCE, ON THE FROZEN SOURCE (Andrew 2026-11-05): one identity run per release, before publish,
on the exact source that ships; the release adopts those records; nothing runs after it.

    python3 tools/identity_at_freeze.py check --root <source checkout at F> --freeze F --json report.json

`tools/release.py` runs it as the step `identity-at-freeze`, right after `freeze-commit` and before anything
is built or rented (rehearsal and cross-compile need it). Per identity column of the table committed at F
it compares F with X, the newest commit that column's admitted records were made at (the last reference
refresh that recorded it):

  * SOURCE-EQUAL: X == F, or every path that differs between X and F is a record output (the admitted
    table, the column records, the admission READMEs). Records can not change a lane's bits, so the
    column is ADOPTED by commit equality: no selector runs, nothing is owed since X.
  * DIFFERENT: the paths X..F (record outputs removed) go to F's own lane selector (tools/lane_select.py of
    the source checkout, the closure rule the recorder and the ledger use) for the column's backend; a
    harness path (tools/identity_break.py, tools/identity_lanes/) or an unattributed path owes every lane
    of the column. It prints, loudly:

        IDENTITY RECORDED AT X, FREEZE IS F: N lane(s) of <column> owed (K source path(s) differ: ...)

Exit 0 when no held column owes a lane, 4 when one does (release.py then HOLDS before any build: record once
at the freeze with tools/reference_refresh.sh F, merge ref/<F12>, and --refreeze), 2 on a usage or Git error.
The PTX column is reported but never held here: its records need the release's own sm_80 set, which exists
only after linux-builds (tools/reference_refresh.py REFRESH_PTX_SET_TGZ).

This decides nothing for the ledger: tools/release_identity_reference.py still judges every identity row at
the freeze (it uses the same source-equal rule, `source_equal`, so a record made at the frozen source is
adopted there with no selector either). A lane this report misses is still owed at the ledger; nothing here
skips, waives or narrows any row. Tests: tools/test_identity_at_freeze.py (no device, no network).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

SCHEMA = "mojolearn.identity-at-freeze.v1"
TABLE_PATH = "python/mojolearn/verify_reference/table.json"
#: Paths a reference refresh writes (tools/reference_refresh.py RECORD_OUTPUT_PATHS + TABLE): the admitted
#: table, the column records and the admission READMEs. They are what an identity run PRODUCES, never an
#: input of a lane, so a diff made only of them leaves every lane's bits where the records found them.
RECORD_OUTPUTS = (TABLE_PATH, "bench/results/identity_break/", "bench/results/reference_admissions/")
#: The harness every column runs: a change here invalidates the table rows of every lane (their
#: lane_admission harness_sha256 no longer matches), so every lane of the column is owed.
HARNESS_PATHS = ("tools/identity_break.py", "tools/identity_lanes/")
#: The backend the recorder selects each column with (tools/reference_refresh.py SELECTION).
COLUMN_BACKEND = dict(nvidia="cuda", amd="hip", cpu=None, ptx="cuda", apple="metal")
#: Columns a reference refresh records BEFORE any release build (an owed lane here holds the release at the
#: freeze). PTX needs the release's sm_80 set, built by linux-builds: reported, not held.
HELD_COLUMNS = ("nvidia", "amd", "cpu", "apple")
FIRST = 5


class GitError(RuntimeError):
    pass


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise GitError("git %s failed (%d): %s" % (" ".join(args), p.returncode, p.stderr.strip()[:200]))
    return p.stdout


def is_record_output(path):
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in RECORD_OUTPUTS)


def source_paths(paths):
    """The paths of a diff that are not record outputs, sorted."""
    return sorted(p for p in paths if p and not is_record_output(p))


def diff_paths(root, a, b):
    """Every path whose bytes differ between commits A and B (two-dot and three-dot, as
    lane_select.changed_paths takes them: a side branch's own edits count too)."""
    out = set()
    for args in (["diff", "--name-only", a, b], ["diff", "--name-only", "%s...%s" % (a, b)]):
        out |= {line.strip() for line in git(root, *args).splitlines() if line.strip()}
    return sorted(out)


def source_equal(root, a, b):
    """True when commits A and B hold the same source: equal, or differing only in record outputs."""
    if a == b:
        return True
    return not source_paths(diff_paths(root, a, b))


def is_placeholder(ent):
    return str((ent or {}).get("ref") or "").startswith("n/a:")


def column_facts(table):
    """{column: dict(lanes, newest=(commit, time))} of the table: the lanes carrying a row of the column,
    and the newest commit any of the column's referenced records was made at."""
    records = table.get("records") or []
    out = {}
    for key, cell in (table.get("cells") or {}).items():
        lane = key.partition("/")[0]
        for _part, ent in (cell or {}).items():
            if is_placeholder(ent):
                continue
            for col, idx in ((ent or {}).get("cols") or {}).items():
                idx = idx if isinstance(idx, int) else (idx[0] if isinstance(idx, list) and idx else None)
                if not isinstance(idx, int) or not 0 <= idx < len(records):
                    continue
                row = out.setdefault(col, dict(lanes=set(), newest=None))
                row["lanes"].add(lane)
                commit = records[idx].get("commit")
                t = int(records[idx].get("commit_time") or 0) if str(records[idx].get("commit_time") or "0").isdigit() else 0
                if commit and (row["newest"] is None or t > row["newest"][1]):
                    row["newest"] = (commit, t)
    return out


def load_selector(root):
    """F's own tools/lane_select.py (the closure rule of the source that ships), imported from ROOT."""
    tools = str(Path(root).resolve() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import lane_select   # noqa: E402  (the source checkout's copy)
    if Path(lane_select.__file__).resolve().parent != Path(tools):
        raise RuntimeError("lane_select imported from %s, not the source checkout %s" % (lane_select.__file__, tools))
    return lambda paths, ref, backend: lane_select.select(paths, ref=ref, backend=backend)


def judge(root, freeze, table, select=None, columns=None):
    """The per-column report (pure apart from Git and `select`). `select(paths, ref, backend)` returns
    lane_select.select's dict; it is called only for a column whose newest record is not source-equal."""
    facts = column_facts(table)
    report = dict(schema=SCHEMA, freeze=freeze, columns={}, held_columns=list(HELD_COLUMNS))
    cache = {}
    for col in sorted(columns or facts):
        f = facts.get(col)
        row = dict(column=col, held=col in HELD_COLUMNS, backend=COLUMN_BACKEND.get(col))
        report["columns"][col] = row
        if not f or not f["newest"]:
            row.update(state="no-records", owed=None, reason="the table at the freeze carries no record of this column")
            continue
        lanes = sorted(f["lanes"])
        recorded_at = f["newest"][0]
        row.update(recorded_at=recorded_at, lanes=len(lanes))
        try:
            paths = source_paths(diff_paths(root, recorded_at, freeze))
        except GitError as exc:
            row.update(state="undecided", owed=len(lanes), owed_lanes=lanes[:FIRST],
                       reason="cannot diff %s..%s: %s; every lane counts as owed" % (recorded_at[:12], freeze[:12], exc))
            continue
        row.update(source_paths=len(paths), first_paths=paths[:FIRST])
        if not paths:
            row.update(state="adopted", owed=0, reason=(
                "recorded at the freeze" if recorded_at == freeze else
                "recorded at %s; only record outputs differ from the freeze (source-equal)" % recorded_at[:12]))
            continue
        harness = [p for p in paths if any(p == h or (h.endswith("/") and p.startswith(h)) for h in HARNESS_PATHS)]
        if harness:
            row.update(state="owed", owed=len(lanes), owed_lanes=lanes[:FIRST],
                       reason="the identity harness changed since %s: %s" % (recorded_at[:12], ", ".join(harness[:FIRST])))
            continue
        if select is None:
            row.update(state="undecided", owed=len(lanes), owed_lanes=lanes[:FIRST],
                       reason="no selector; every lane counts as owed")
            continue
        key = (recorded_at, row["backend"])
        if key not in cache:
            try:
                cache[key] = select(paths, recorded_at, row["backend"])
            except (Exception, SystemExit) as exc:   # an undecided selection is never "unchanged"
                cache[key] = dict(error="%s: %s" % (type(exc).__name__, exc))
        sel = cache[key]
        if sel.get("error"):
            row.update(state="undecided", owed=len(lanes), owed_lanes=lanes[:FIRST],
                       reason="the selector failed (%s); every lane counts as owed" % sel["error"][:200])
        elif sel.get("unattributed"):
            row.update(state="owed", owed=len(lanes), owed_lanes=lanes[:FIRST],
                       reason="unattributed path(s) since %s: %s" % (recorded_at[:12], ", ".join(sel["unattributed"][:FIRST])))
        else:
            owed = sorted(set(sel.get("lanes") or ()) & set(lanes))
            row.update(state="owed" if owed else "adopted", owed=len(owed), owed_lanes=owed[:FIRST],
                       reason=("closure changed since %s" % recorded_at[:12]) if owed else
                       ("no lane's closure changed since %s (F's lane_select)" % recorded_at[:12]))
    held = [c for c, r in report["columns"].items() if r["held"] and r.get("state") in ("owed", "undecided")]
    report["held"] = held
    report["adopted"] = not held
    return report


def lines(report):
    """The loud lines release.py prints at the freeze."""
    freeze = (report.get("freeze") or "")[:12]
    out = []
    for col, r in sorted((report.get("columns") or {}).items()):
        state = r.get("state")
        if state == "adopted":
            out.append("identity %s: ADOPTED at freeze %s (%s; %s lane(s), 0 owed)" % (
                col, freeze, r.get("reason"), r.get("lanes")))
        elif state == "no-records":
            out.append("identity %s: no admitted record in the table at %s" % (col, freeze))
        else:
            out.append("IDENTITY RECORDED AT %s, FREEZE IS %s: %s lane(s) of %s owed (%s source path(s) differ: %s)%s; %s" % (
                (r.get("recorded_at") or "?")[:12], freeze, r.get("owed"), col, r.get("source_paths", "?"),
                ", ".join(r.get("first_paths") or ()) or "-", "" if r.get("held") else " [not held: recorded after linux-builds]",
                r.get("reason")))
    if report.get("held"):
        out.append("HOLD BEFORE ANY BUILD: identity is owed at the freeze on %s. Record ONCE at the frozen source "
                   "(tools/reference_refresh.sh %s), merge ref/%s into main, then rerun tools/release.py with "
                   "--refreeze. Never build first and record after." % (", ".join(report["held"]), freeze, freeze))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--root", required=True, help="the source checkout, at the frozen commit")
    c.add_argument("--freeze", required=True)
    c.add_argument("--json", required=True)
    c.add_argument("--columns", default="", help="comma list (default: every column of the table)")
    args = ap.parse_args(argv)
    try:
        freeze = git(args.root, "rev-parse", args.freeze).strip()
        head = git(args.root, "rev-parse", "HEAD").strip()
        if head != freeze:
            print("identity-at-freeze: the source checkout %s is at %s, not the freeze %s" % (args.root, head[:12], freeze[:12]),
                  file=sys.stderr)
            return 2
        table = json.loads(git(args.root, "show", "%s:%s" % (freeze, TABLE_PATH)))
    except (GitError, ValueError) as exc:
        print("identity-at-freeze: %s" % exc, file=sys.stderr)
        return 2
    try:
        select = load_selector(args.root)
    except Exception as exc:   # judge() then counts every lane of a differing column as owed
        print("identity-at-freeze: no selector (%s: %s)" % (type(exc).__name__, exc), file=sys.stderr)
        select = None
    report = judge(args.root, freeze, table, select=select,
                   columns=[c for c in args.columns.split(",") if c] or None)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for line in lines(report):
        print(line)
    return 0 if report["adopted"] else 4


if __name__ == "__main__":
    sys.exit(main())
