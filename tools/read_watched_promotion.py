#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHAT A WATCHED CPU-ONLY RUN SAID, LANE BY LANE, so a promotion rests on a
printed result rather than on an argument (lane/verifier-full-exposure,
2026-09-20).

    python3 tools/read_watched_promotion.py REPORT.json [REPORT.json ...]

`host_surface.public_reference_lanes()` states the promotion rule: a covered
lane, in a release record's scope, with a reference in the shipped table,
WATCHED TO READ CLEAN BY A CPU-ONLY RUN AT THIS COMMIT. The first three
clauses are static and other tests check them. This reads the fourth off a
`verify --all --json-out` document and says PROMOTE or HOLD for each lane,
with what the run actually read.

THE BAR IS PER PART, NOT PER LANE. A lane is clean only when every part of
every fixture read IDENTICAL or N/A and at least one read IDENTICAL. One OWED
part is enough to hold a lane: it means the shipped table has no hash for
something the run computed, which is exactly the thing a promotion claims it
does have. Printing the counts rather than a verdict alone is deliberate --
`3 OWED` and `0 OWED` are the difference between a promotion and a rewritten
reason, and a reader should not have to take this file's word for which.

IT ALSO CHECKS THE DOCUMENT IS OF THE RIGHT KIND. A run on a GPU box, in the
fast tier, or with a harness the table was not generated against would answer
a different question, so those are refused by name rather than read.
"""
import argparse
import collections
import json
import sys

CLEAN = ("IDENTICAL", "N/A")


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def problems(doc):
    """Reasons this document cannot answer the promotion question."""
    bad = []
    device = doc.get("device") or {}
    if device.get("device_class") != "cpu":
        bad.append(f"device class {device.get('device_class')!r}, not cpu: a promotion needs the "
                   "CPU-only run, because that is the install the public set exists for")
    if device.get("numeric_mode") != "identical":
        bad.append(f"numeric mode {device.get('numeric_mode')!r}, which makes no bitwise promise")
    if doc.get("repeats", 1) < 1:
        bad.append("no repeats recorded")
    return bad


def cautions(doc):
    """Things a reader should know that do not by themselves refuse the run.

    THE HARNESS DIGEST IS ONE OF THESE, and it took a moment to see why
    (2026-09-20). The shipped table keeps the harness witness of the run that
    generated it, and `merge_reference_lanes` deliberately does not update it
    on a scoped admission -- admitting a few lanes does not requalify the
    legacy cells. So `matches_table` is already false on main and refusing on
    it would refuse EVERY run, which is a check that cannot pass rather than
    one that cannot fail, and just as useless.

    It is not dropped, because the thing it warns about is real: a lane the
    harness changed since the table was generated reads OWED or DIVERGENT for
    a reason that has nothing to do with the box. That symptom is what the
    per-lane bar below already refuses, part by part, so the warning is
    printed and the arithmetic is left to the parts.
    """
    out = []
    harness = doc.get("harness") or {}
    if not harness.get("matches_table"):
        out.append("this harness is not the one the shipped table was generated with. A lane "
                   "changed since then reads OWED or DIVERGENT for that reason alone, so read "
                   "the per-lane counts below rather than the verdict")
    return out


def per_lane(doc):
    """{lane: Counter(state)} over every judged cell part, models excluded."""
    out = collections.defaultdict(collections.Counter)
    for row in doc.get("cells", []):
        if str(row.get("lane", "")).startswith("portable:"):
            continue
        out[row["lane"]][row["state"]] += 1
    return out


def verdicts(docs):
    merged = collections.defaultdict(collections.Counter)
    for doc in docs:
        for lane, counts in per_lane(doc).items():
            merged[lane].update(counts)
    rows = []
    for lane in sorted(merged):
        c = merged[lane]
        dirty = {s: n for s, n in c.items() if s not in CLEAN}
        clean = not dirty and c.get("IDENTICAL", 0) > 0
        rows.append(dict(lane=lane, clean=clean, counts=dict(c),
                         why=("" if clean else
                              "nothing was compared: every part read n/a" if not dirty else
                              ", ".join(f"{n} {s}" for s, n in sorted(dirty.items())))))
    return rows


def _doc(cells, device_class="cpu", mode="identical"):
    return dict(device=dict(device_class=device_class, numeric_mode=mode, vendor="x", commit="c" * 40),
                repeats=2, harness=dict(matches_table=True), verdict="?", exit=0, cells=cells)


def _cells(lane, **states):
    out = []
    for state, n in states.items():
        out += [dict(lane=lane, fixture="base", part="train", state=state.replace("NA", "N/A"))] * n
    return out


def self_test(verbose=True):
    """WATCH IT REFUSE BEFORE TRUSTING IT TO PROMOTE. This file's whole job is
    to say PROMOTE, so the failure that matters is it saying PROMOTE when it
    should not. Each case below is a run that must NOT promote."""
    ok = True
    cases = [
        ("a GPU document cannot settle a CPU promotion",
         _doc(_cells("a", IDENTICAL=9), device_class="apple"), None, True),
        ("the fast tier makes no bitwise promise",
         _doc(_cells("a", IDENTICAL=9), mode="fast"), None, True),
        ("one OWED part holds the lane",
         _doc(_cells("a", IDENTICAL=8) + _cells("a", OWED=1)), False, False),
        ("one DIVERGENT part holds the lane",
         _doc(_cells("a", IDENTICAL=8) + _cells("a", DIVERGENT=1)), False, False),
        ("one REFUSED part holds the lane",
         _doc(_cells("a", IDENTICAL=8) + _cells("a", REFUSED=1)), False, False),
        ("a lane whose every part is n/a compared nothing",
         _doc(_cells("a", NA=9)), False, False),
        ("a clean lane promotes", _doc(_cells("a", IDENTICAL=8) + _cells("a", NA=1)), True, False),
    ]
    for title, doc, expect_clean, expect_refuse in cases:
        refused = bool(problems(doc))
        if expect_refuse:
            if not refused:
                ok = False
                print(f"NOT REFUSED: {title}")
            elif verbose:
                print(f"refused: {title} -> {problems(doc)[0][:90]}")
            continue
        if refused:
            ok = False
            print(f"WRONGLY REFUSED: {title} -> {problems(doc)}")
            continue
        got = verdicts([doc])[0]["clean"]
        if got != expect_clean:
            ok = False
            print(f"WRONG VERDICT: {title}: clean={got}, expected {expect_clean}")
        elif verbose:
            print(f"{'promoted' if got else 'held'}: {title}")
    # AND THE ABSENT CASE, which is the one a reader is least likely to think
    # of: a lane simply not in the document at all must not read as a pass.
    rows = verdicts([_doc(_cells("a", IDENTICAL=9))])
    if any(r["lane"] == "b" for r in rows):
        ok = False
        print("NOT REFUSED: a lane the run never compared appeared in the verdicts")
    elif verbose:
        print("held: a lane the run never compared is ABSENT, not promoted")
    print(f"{'OK' if ok else 'BROKEN'}: {len(cases) + 1} cases")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reports", nargs="*")
    ap.add_argument("--self-test", action="store_true",
                    help="require the reader to refuse every run that must not promote")
    ap.add_argument("--lanes", default="", help="only these lanes, comma separated")
    a = ap.parse_args(argv)
    if a.self_test:
        return 0 if self_test() else 1
    if not a.reports:
        ap.error("give at least one verify --json-out document, or --self-test")
    docs = [load(p) for p in a.reports]
    refuse = False
    for path, doc in zip(a.reports, docs):
        bad = problems(doc)
        d = doc.get("device") or {}
        print(f"# {path}")
        print(f"#   {d.get('vendor')} (class {d.get('device_class')}), mode "
              f"{d.get('numeric_mode')}, commit {str(d.get('commit'))[:12]}, "
              f"{doc.get('repeats')} repeat(s), verdict {doc.get('verdict')} exit {doc.get('exit')}")
        for line in cautions(doc):
            print(f"#   CAUTION: {line}")
        for line in bad:
            print(f"#   REFUSED: {line}")
            refuse = True
    if refuse:
        print("\nThis document cannot settle a promotion. Nothing is promoted on it.")
        return 1
    want = {x for x in a.lanes.split(",") if x}
    rows = [r for r in verdicts(docs) if not want or r["lane"] in want]
    missing = sorted(want - {r["lane"] for r in rows})
    width = max([len(r["lane"]) for r in rows] + [10])
    print()
    for r in rows:
        counts = ", ".join(f"{s}={n}" for s, n in sorted(r["counts"].items()))
        print(f"{'PROMOTE' if r['clean'] else 'HOLD   '}  {r['lane']:<{width}}  {counts}"
              + (f"   <- {r['why']}" if r["why"] else ""))
    for lane in missing:
        print(f"ABSENT   {lane:<{width}}  the run did not compare it at all, which is not a pass")
    promoted = [r["lane"] for r in rows if r["clean"]]
    print(f"\n{len(promoted)} of {len(rows) + len(missing)} lane(s) read clean end to end")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
