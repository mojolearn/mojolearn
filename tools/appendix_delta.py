#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH HARNESS LANES THE MLSys APPENDIX DOES NOT MAP, DERIVED.

    python3 tools/appendix_delta.py            # print the delta
    python3 tools/appendix_delta.py --write    # rewrite the companion CSV
    python3 tools/appendix_delta.py --check    # fail when the CSV is stale

THE NUMBER 246 DOES NOT MOVE, AND THIS FILE EXISTS SO NOBODY MOVES IT.

`appendix-246.csv` maps the 246 algorithm/variant entries of the published
MLSys appendix onto harness lanes. It is a record of WHAT WAS PUBLISHED.
Adding a row for a lane that did not exist at publication would rewrite a
published claim, and `docs/COVERAGE_AUDIT_2026-09-18.md` already refuses the
arithmetic in as many words:

    Nor can we add 26 to 246 and claim 272 unique algorithms: the
    inventories overlap and count different things.

So the published catalog stays frozen and the DELTA lives here, derived from
`identity_break.LANES` by import on every run rather than typed once and left
to rot. That audit recorded the delta as 26 on 2026-09-18 and called catalog
maintenance "owed"; a hand-kept number is exactly what goes stale, which is
why this is a tool and not another row in a table.

WHAT A ROW HERE IS AND IS NOT. It is a harness lane with no appendix entry
mapping to it. It is NOT "an algorithm the paper missed": several are weight
formats (`*-bf16w`, `*-int8w`) and kernel variants of an algorithm the
appendix already carries, and one (`language-model-config`) is a
configuration object. Counting these as new algorithms is the same error the
audit refuses. The honest sentence is "N harness lanes are outside the
published mapping", and this prints exactly that.
"""
import argparse
import csv
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APPENDIX = ROOT / ("bench/results/cpu-verification-completion-probe/2026-09-17"
                   "/published-gap-audit/appendix-246.csv")
COMPANION = APPENDIX.with_name("appendix-delta.csv")


def mapped_lanes():
    out = set()
    for row in csv.DictReader(APPENDIX.open()):
        for lane in (row.get("lanes") or "").split(";"):
            lane = lane.strip()
            if lane:
                out.add(lane)
    return out


def delta():
    sys.path.insert(0, str(ROOT / "tools"))
    import identity_break as ib
    return sorted(set(ib.LANES) - mapped_lanes()), len(ib.LANES)


def rows_for(names):
    """A reason per lane, so the list can be argued with rather than counted."""
    out = []
    for lane in names:
        if lane.endswith(("-bf16w", "-int8w")):
            why = "weight-format variant of a lane the appendix already maps"
        elif lane.startswith(("kernel-ridge-", "nystroem-")):
            why = "kernel variant of a lane the appendix already maps"
        elif lane.startswith("gemm-"):
            why = "low-bit GEMM profile, added after publication"
        elif lane.startswith("linalg-"):
            why = "decomposition exposed under its own name, 2026-09-19"
        elif lane.startswith("par-"):
            why = "multi-device driver lane"
        elif lane.endswith("-config"):
            why = "configuration object, not an algorithm"
        else:
            why = "added after publication"
        out.append(dict(lane=lane, why_not_in_appendix=why))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    names, n_lanes = delta()
    rows = rows_for(names)
    text = "lane,why_not_in_appendix\n" + "".join(
        f"{r['lane']},{r['why_not_in_appendix']}\n" for r in rows)

    print(f"published appendix entries : 246  (FROZEN -- do not add to this)")
    print(f"harness lanes              : {n_lanes}")
    print(f"lanes outside the mapping  : {len(names)}")
    print()
    for r in rows:
        print(f"  {r['lane']:34s} {r['why_not_in_appendix']}")

    if args.write:
        COMPANION.write_text(text)
        print(f"\nwrote {COMPANION.relative_to(ROOT)}")
        return 0
    if args.check:
        have = COMPANION.read_text() if COMPANION.exists() else ""
        if have != text:
            print(f"\nSTALE: {COMPANION.relative_to(ROOT)} does not match the tree; "
                  "run --write")
            return 1
        print(f"\nOK: {COMPANION.relative_to(ROOT)} matches the tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
