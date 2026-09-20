#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOW MANY OF THE NINE PARTS DID EACH COLUMN ACTUALLY ASK?

    python3 tools/part_set_census.py            # the census
    python3 tools/part_set_census.py --json     # the same thing as data

A COLUMN CAN BE PERFECTLY CLEAN AND STILL ANSWER FOUR NINTHS OF THE QUESTION.
`_verify_reference.PARTS` is train, infer, model, batch and stepfull;
`OPTIONAL_PARTS` is batchgrad, batchscale, ragged and rlpair. Nine. But
`tools/identity_break.py` runs train/infer/model/batch/rlpair at its DEFAULTS
and needs `--step-full`, `--batch-grad`, `--batch-scale` and `--ragged` to be
ASKED for the rest. Nothing in the recording path says which of the nine a
column asked, and nothing downstream compares one column's width to another's.

WHY THAT IS NOT CAUGHT ANYWHERE ELSE, WHICH IS THE POINT OF THIS FILE:

  * `admit()` never inspects which parts a column carries. Its completeness
    check is `complete is False`, a RUN-level "did this run finish" flag.
    A five-part column is admitted exactly like a nine-part one.
  * `build_table` skips an absent part with `if value is None: continue` and
    writes no log line for it. The per-column line is "use <path>: class X,
    commit Y, N cell parts", and a smaller N is compared to nothing.
  * The absence IS caught at the far end -- `_verify_reference` line ~224
    returns OWED, "no committed record carries this cell part yet", by name,
    per part, and it is never reported IDENTICAL. But `LANE_OWED` is in
    `_verify_all.LANE_STATES_THAT_DO_NOT_GATE`, so it costs a run nothing.

So the hole is visible and free. This tool makes it countable.

MEASURED 2026-09-20, and the headline is about the records everyone cites:

    245 columns  4 parts    (batch infer model train)
    151 columns  5 parts    (+ stepfull)
     51 columns  8 parts
     31 columns  6 parts
     27 columns  5 parts    (+ rlpair, NO stepfull)
     15 columns  9 parts    <- all of them
    only 249 of 559 admissible columns carry `stepfull` AT ALL

and 2026-09-14_118-lanes, 2026-09-14_166-lanes and
2026-09-14_120-lanes-2711flip -- "the 166-lane record" and its siblings --
are THEMSELVES FOUR-PART. Nobody reading that name would guess it.

`lacks stepfull` IS ONLY A GAP WHERE THE PART APPLIES. `arima` has no decode
state and STEPFULL_DEFAULT is "n/a:no-decode-state"; counting its absent
stepfull as under-collection would be the same mistake pointing the other
way. So the per-lane sections below are restricted to the lanes the harness
DECLARES a probe for, read out of the harness rather than listed here.

THE BAR THIS SETS FOR A RECORD RUN: nine parts or it is not a record.
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IB = os.path.join(ROOT, "bench", "results", "identity_break")
TABLE = os.path.join(ROOT, "python", "mojolearn", "verify_reference", "table.json")


def _load(path, name):
    """Load a module BY FILE PATH. `mojolearn/__init__.py` runs
    `_backend.select()`, which refuses on any checkout with no built binding,
    and this is a read-only walk over JSON that needs no binding at all."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def census(root=ROOT):
    vr = _load(os.path.join(root, "python/mojolearn/_verify_reference.py"), "_psc_vr")
    harness = _load(os.path.join(root, "tools/identity_break.py"), "_psc_ib")
    all_parts = vr.PARTS + vr.OPTIONAL_PARTS

    #: which lanes DECLARE each part, read from the harness, never typed here
    declares = {"stepfull": set(getattr(harness, "STEPFULL", {}) or {})}
    for part, attr in (("batchgrad", "BATCHGRAD"), ("batchscale", "BATCHSCALE"),
                       ("ragged", "RAGGED"), ("rlpair", "RLPAIR")):
        declares[part] = set(getattr(harness, attr, {}) or {})

    def parts_of(j):
        have = set()
        for cell in j["cells"].values():
            if not isinstance(cell, dict):
                continue
            for part in all_parts:
                v = cell.get("hashes" if part == "train" else part)
                if isinstance(v, list) and v:
                    have.add(part)
        return have

    cols = []
    for path in sorted(glob.glob(os.path.join(root, "bench/results/**/*.json"), recursive=True)):
        try:
            with open(path) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(j, dict) or not isinstance(j.get("cells"), dict):
            continue
        cls = vr.device_class(j.get("vendor"), path)
        cols.append(dict(rel=os.path.relpath(path, root), j=j, cls=cls,
                         admit=vr.admit(j, path), parts=parts_of(j),
                         in_tree=os.path.abspath(path).startswith(
                             os.path.join(root, "bench", "results", "identity_break") + os.sep)))

    usable = [c for c in cols if c["admit"] is None and c["cls"] and c["in_tree"]]
    shapes = collections.Counter(tuple(sorted(c["parts"])) for c in usable)

    # per-part: (lane, class) pairs with a column but none carrying the part
    gaps = {}
    for part, lanes in declares.items():
        if not lanes:
            continue
        any_col, with_part = collections.defaultdict(list), set()
        for c in usable:
            for key, cell in c["j"]["cells"].items():
                lane = key.split("/", 1)[0]
                if lane not in lanes or not isinstance(cell, dict):
                    continue
                if cell.get("verdict") != "STABLE" or not cell.get("hashes"):
                    continue
                any_col[(lane, c["cls"])].append(c["rel"])
                v = cell.get(part)
                if isinstance(v, list) and v:
                    with_part.add((lane, c["cls"]))
        gaps[part] = dict(pairs=len(any_col),
                          missing=sorted(k for k in any_col if k not in with_part),
                          where={k: v[0] for k, v in any_col.items()})

    # what the shipped table rests on
    shipped = {}
    if os.path.exists(TABLE):
        with open(TABLE) as fh:
            t = json.load(fh)
        present = collections.defaultdict(set)
        for ck, byp in (t.get("cells") or {}).items():
            lane = ck.split("/", 1)[0]
            for part, ent in byp.items():
                for cls in (ent.get("cols") or {}):
                    present[(lane, cls)].add(part)
        for part, lanes in declares.items():
            if not lanes:
                continue
            miss = sorted((l, c) for (l, c), ps in present.items()
                          if l in lanes and part not in ps)
            shipped[part] = dict(
                missing=miss,
                also_uncommitted=sorted(set(miss) & set(gaps[part]["missing"])))
    return dict(total=len(cols), usable=len(usable),
                shapes={", ".join(k) or "(none)": v for k, v in shapes.most_common()},
                carrying_stepfull=sum(1 for c in usable if "stepfull" in c["parts"]),
                declares={k: sorted(v) for k, v in declares.items()},
                gaps=gaps, shipped=shipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = census()
    if args.json:
        json.dump(r, sys.stdout, indent=2, default=list)
        print()
        return 0
    print("columns under bench/results with an identity_break shape : %d" % r["total"])
    print("admissible, device-classed, inside bench/results/identity_break: %d" % r["usable"])
    print("of those, carrying `stepfull` at all                      : %d" % r["carrying_stepfull"])
    print("\n--- part sets, by how many columns have them ---")
    for shape, n in r["shapes"].items():
        print("  x%-5d %s" % (n, shape))
    for part in sorted(r["gaps"]):
        g = r["gaps"][part]
        print("\n--- %s: declared for %d lanes; %d (lane,class) pairs have a column, "
              "%d have NO column carrying it ---"
              % (part, len(r["declares"][part]), g["pairs"], len(g["missing"])))
        for lane, cls in g["missing"]:
            print("   %-34s %-7s e.g. %s" % (lane, cls, g["where"][(lane, cls)]))
    for part in sorted(r["shipped"]):
        s = r["shipped"][part]
        if not s["missing"]:
            continue
        print("\n--- shipped table: %d (lane,class) pairs for %s-declaring lanes "
              "with no %s entry; %d of them have no committed column either ---"
              % (len(s["missing"]), part, part, len(s["also_uncommitted"])))
        for lane, cls in s["missing"]:
            mark = "  <-- and nothing committed carries it" if (lane, cls) in s["also_uncommitted"] else ""
            print("   %-34s %-7s%s" % (lane, cls, mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
