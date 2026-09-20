#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""EVERY LANE IS EITHER REFERENCED OR DECLARED, NEVER NEITHER.

    python3 tools/lane_accounting.py             # print the accounting
    python3 tools/lane_accounting.py --check     # exit non-zero on a gap
    python3 tools/lane_accounting.py --self-test # watch it refuse, five ways
    python3 tools/lane_accounting.py --json      # the whole thing as data

WHY THIS EXISTS. There are two mechanisms that are supposed to account for a
lane, and on 2026-09-20 three lanes were sitting between them:

  * the SHIPPED REFERENCE TABLE, python/mojolearn/verify_reference/table.json,
    which is what `python -m mojolearn verify --all` on an installed wheel
    compares against. A lane with a cell there has a number to check.
  * `host_surface.PUBLIC_PENDING_LANES`, which holds a lane back from the
    public set WITH THE REASON WRITTEN DOWN, so a lane with no number says
    out loud that it has none and why.

`par-forecast-arima`, `par-forecast-holtwinters` and `par-ivf` were in
NEITHER. Neither list was wrong on its own terms, which is the whole point.
`PUBLIC_PENDING_LANES` only ever admits lanes that could become PUBLIC, and
`PUBLIC_EXCLUDED_PREFIXES` excludes every `par-` driver from the public set,
so the pending list had no business naming them; and the table simply had no
record admitted for them yet. Each mechanism refused the lane for a good
reason and handed it to the other. Nothing owned the gap, so nothing failed,
and the lanes were invisible to both sides for as long as nobody counted.

That is the shape this file refuses: not "a lane is missing evidence" -- that
is normal and is what the pending list is FOR -- but "a lane is missing from
the accounting", which reads exactly like a lane that is fine.

WHAT IS CHECKED

  1. THE READER IS NOT UNDER-READING. The lane names are read from the source
     of tools/identity_break.py, three registration forms (`@lane("x")`,
     `lane("x")(...)` and the module-level loops that spell a name with an
     f-string). A reader that silently misses a lane would make every check
     below pass vacuously -- A VERIFICATION THAT CANNOT FAIL IS WORTHLESS --
     so it is held to two registries that were written independently of it:
     every lane the shipped table carries cells for, and every lane
     PUBLIC_PENDING_LANES names, must be in the set it read. When numpy is
     importable it is additionally held to the harness's own `LANES`, which
     is authoritative; `harness_lanes()` returns None when it is not, and
     `check()` says so in its report rather than passing quietly.
  2. THE INVARIANT. Every registered lane is in the shipped table or in
     PUBLIC_PENDING_LANES. A lane in neither is named, with what it needs:
     a COVERED lane can take either road, so the message offers both; a lane
     that is not covered cannot be declared pending at all (test_host_surface
     asserts `lane in covered` of every entry), so for it the only honest
     answer is a record and an admission, and the message says only that.
  3. NO DEAD DECLARATION. A lane named in PUBLIC_PENDING_LANES that the
     harness no longer registers is a reason kept for something that is gone.
  4. NO DOUBLE STANDARD ON AN EMPTY REASON. A pending entry whose reason is
     empty or blank declares nothing; it is the gap with a name on it.

WHAT IS NOT CHECKED HERE. Whether a pending reason is TRUE -- that is
`test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`,
which checks each reason against the harness's LANE_REVISIONS, the shipped
table and the recorded run. This file checks only that a lane is accounted
for at all, which is the question that mechanism never asks, because a lane
it never sees cannot fail it.
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "tools", "identity_break.py")
SURFACE = os.path.join(ROOT, "python", "mojolearn", "host_surface.py")
TABLE = os.path.join(ROOT, "python", "mojolearn", "verify_reference", "table.json")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def surface(path=SURFACE):
    """host_surface loaded by path. It imports argparse, json and sys and
    nothing else, so this needs no binding and no numpy."""
    return _load(path, "_la_host_surface")


# ------------------------------------------------------------------- readers

def _loop_registered_lanes(text):
    """The lanes registered in a module-level loop,
    `for _name, _nu, _ls in (("matern12", ...), ...): lane(f"gp-{_name}")(...)`,
    which the `@lane("...")` pattern cannot see (the gp Matern lanes, the kde
    kernel and metric pairs, the knn and radius metrics). Read from the AST:
    each literal tuple of the loop binds the target names, and the f-string is
    spelled out from the string constants it binds. Same reader as
    test_host_surface, deliberately, so the two cannot drift."""
    out = set()
    for node in ast.parse(text).body:
        if not isinstance(node, ast.For) or not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        names = [t.id if isinstance(t, ast.Name) else None for t in targets]
        for stmt in node.body:
            call = getattr(stmt, "value", None)
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Call)
                    and isinstance(call.func.func, ast.Name) and call.func.func.id == "lane"
                    and call.func.args and isinstance(call.func.args[0], ast.JoinedStr)):
                continue
            for element in node.iter.elts:
                values = element.elts if isinstance(element, ast.Tuple) else [element]
                bound = {n: v.value for n, v in zip(names, values)
                         if n and isinstance(v, ast.Constant) and isinstance(v.value, str)}
                parts = []
                for piece in call.func.args[0].values:
                    if isinstance(piece, ast.Constant):
                        parts.append(str(piece.value))
                    elif (isinstance(piece, ast.FormattedValue)
                          and isinstance(piece.value, ast.Name) and piece.value.id in bound):
                        parts.append(bound[piece.value.id])
                    else:
                        parts = None
                        break
                if parts:
                    out.add("".join(parts))
    return out


def source_lanes(src=None, path=HARNESS):
    """Every lane name the harness SOURCE registers, no import, no numpy."""
    text = src if src is not None else open(path, "r", encoding="utf-8").read()
    found = set(re.findall(r'^@lane\("([a-z0-9-]+)"\)', text, re.M))
    found |= set(re.findall(r'^lane\("([a-z0-9-]+)"\)\(', text, re.M))
    found |= _loop_registered_lanes(text)
    return found


def harness_lanes(path=HARNESS):
    """The harness's own `LANES`, which is authoritative, or None when the
    harness cannot be imported here (it needs numpy). None is REPORTED by
    `check`, never swallowed: a cross-check that quietly stops running is the
    same failure this file exists to refuse."""
    try:
        return set(_load(path, "_la_identity_break").LANES)
    except Exception:                                    # noqa: BLE001 -- any import failure
        return None


def table_lanes(path=TABLE, table=None):
    """Every lane the SHIPPED table carries at least one cell for."""
    if table is None:
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
    return {key.partition("/")[0] for key in table.get("cells", {})}


# -------------------------------------------------------------------- check

def check(lanes=None, referenced=None, pending=None, covered=None, authoritative=None):
    """The problems, as a list of strings. Empty means every lane is
    accounted for. Every input is injectable so `self_test` can perturb one
    at a time without writing to the tree."""
    bad = []
    lanes = set(source_lanes()) if lanes is None else set(lanes)
    referenced = set(table_lanes()) if referenced is None else set(referenced)
    if pending is None or covered is None:
        mod = surface()
        pending = dict(mod.PUBLIC_PENDING_LANES) if pending is None else dict(pending)
        covered = set(mod.covered_lanes()) if covered is None else set(covered)
    else:
        pending, covered = dict(pending), set(covered)
    if authoritative is None:
        authoritative = harness_lanes()

    # 1. the reader is not under-reading
    if not lanes:
        bad.append("the lane reader found NO lanes at all in tools/identity_break.py")
    unseen = sorted(referenced - lanes)
    if unseen:
        bad.append(
            f"the lane reader did not see {len(unseen)} lane(s) that the shipped table names, so "
            f"it is under-reading tools/identity_break.py and every check below would pass "
            f"vacuously: {unseen}")
    if authoritative is not None and authoritative != lanes:
        missed, extra = sorted(authoritative - lanes), sorted(lanes - authoritative)
        bad.append(
            "the lane reader disagrees with the harness's own LANES; a new registration form "
            f"needs a reader. missed by the reader: {missed}; invented by the reader: {extra}")

    # 2. the invariant
    for lane in sorted(lanes - referenced - set(pending)):
        if lane in covered:
            bad.append(
                f"{lane}: in NEITHER the shipped table NOR PUBLIC_PENDING_LANES. An installed "
                "`verify --all` has no hash for it and nothing says why. Either admit a committed "
                "column for it (`verify --all --emit-reference --reference-table --lanes "
                f"{lane} --batch-checks`) or declare it in PUBLIC_PENDING_LANES with the reason.")
        else:
            bad.append(
                f"{lane}: in NEITHER the shipped table NOR PUBLIC_PENDING_LANES, and it is not a "
                "covered lane, so PUBLIC_PENDING_LANES cannot take it either (every entry there "
                "must be covered). What it needs is a committed admissible column and a scoped "
                f"`verify --all --emit-reference --reference-table --lanes {lane} --batch-checks`.")

    # 3. no dead declaration, 4. no empty reason
    for lane, why in sorted(pending.items()):
        if lane not in lanes:
            bad.append(f"{lane}: declared in PUBLIC_PENDING_LANES but tools/identity_break.py "
                       "registers no such lane; the reason outlived the thing it was about"
                       + ("" if authoritative is not None else
                          " (the harness could not be imported here, so this rests on the source "
                          "reader; if the lane does exist, the reader is what is wrong)"))
        if not str(why).strip():
            bad.append(f"{lane}: declared in PUBLIC_PENDING_LANES with an empty reason, which "
                       "declares nothing")
    return bad


def report(lanes=None, referenced=None, pending=None, covered=None):
    """The accounting as data."""
    lanes = set(source_lanes()) if lanes is None else set(lanes)
    referenced = set(table_lanes()) if referenced is None else set(referenced)
    mod = surface()
    pending = dict(mod.PUBLIC_PENDING_LANES) if pending is None else dict(pending)
    covered = set(mod.covered_lanes()) if covered is None else set(covered)
    authoritative = harness_lanes()
    return dict(
        registered=len(lanes),
        in_shipped_table=len(lanes & referenced),
        declared_pending=len(lanes & set(pending)),
        both=sorted(lanes & referenced & set(pending)),
        unaccounted=sorted(lanes - referenced - set(pending)),
        harness_cross_check=("ran" if authoritative is not None
                             else "SKIPPED: the harness could not be imported here (numpy)"),
        problems=check(lanes, referenced, pending, covered, authoritative),
    )


# ----------------------------------------------------------------- self test

def self_test(verbose=True):
    """WATCH IT REFUSE BEFORE TRUSTING IT TO PASS. Five perturbations, each
    of which is a real way this gap comes back, and each must be refused BY
    NAME. Nothing is written to the tree: `check` takes every input."""
    lanes = set(source_lanes())
    referenced = set(table_lanes())
    mod = surface()
    pending = dict(mod.PUBLIC_PENDING_LANES)
    covered = set(mod.covered_lanes())
    authoritative = lanes                      # pin it, so the cases below are the only variable

    def run(**kw):
        args = dict(lanes=lanes, referenced=referenced, pending=pending,
                    covered=covered, authoritative=authoritative)
        args.update(kw)
        return check(**args)

    assert run() == [], "the unperturbed tree already has problems; fix those first"
    # THE VICTIMS ARE NOT PICKED OUT OF THE TREE'S CURRENT STATE. An earlier
    # draft chose the pending victim as `pending & lanes - referenced`, and a
    # promotion that empties that set would have silently skipped two of the
    # seven cases while still printing OK: a self-test that quietly shrinks is
    # the same failure this file exists to refuse. Every case below is built
    # from a lane that certainly exists (`covered & referenced`) plus names
    # that certainly do not.
    victim = sorted(lanes & covered & referenced)[0]
    uncovered = sorted(lanes - covered)
    assert uncovered, "no uncovered lane to build the uncovered case from"
    unregistered, newborn = "zz-retired-lane", "zz-new-lane"
    assert not ({unregistered, newborn} & lanes), "the synthetic names collide with real lanes"

    # A world in which `victim` has no table cells and IS declared pending.
    # It must read CLEAN -- that is the whole claim, "declared counts too" --
    # and deleting the declaration from it must be refused.
    declared = dict(referenced=referenced - {victim},
                    pending=dict(pending, **{victim: "no reference"}))
    assert run(**declared) == [], (
        "a lane with no table cells but a pending declaration should be accounted for, and is not")
    ok = True

    cases = [
        # (a) a lane loses its table cells and nobody declares it -- the 2026-09-20
        #     shape, where a new lane simply has no record admitted yet.
        ("a referenced lane loses its table cells",
         dict(referenced=referenced - {victim}), victim),
        # (b) a pending declaration is deleted from a lane that has no cells.
        ("a pending declaration is deleted",
         dict(referenced=referenced - {victim}, pending=pending), victim),
        # (c) a brand new lane is registered and neither list is touched.
        ("a new lane is registered and nothing else",
         dict(lanes=lanes | {newborn}, authoritative=lanes | {newborn}), newborn),
        # (d) a NOT-COVERED lane falls out of the table: the message must not
        #     send the reader to a list that cannot take it.
        ("an uncovered lane loses its table cells",
         dict(referenced=referenced - {uncovered[0]}), uncovered[0]),
        # (e) the reader under-reads a lane the table names.
        ("the lane reader under-reads",
         dict(lanes=lanes - {victim}, authoritative=lanes - {victim}), victim),
        # (f) the reader disagrees with the harness's own LANES.
        ("the reader disagrees with the harness's LANES",
         dict(authoritative=lanes | {newborn}), newborn),
        # (g) a pending entry names a lane the harness no longer registers.
        ("a pending reason outlives its lane",
         dict(pending=dict(pending, **{unregistered: "no reference"})), unregistered),
        # (h) a pending entry with a blank reason.
        ("a pending reason is blank",
         dict(pending=dict(pending, **{victim: "  "})), victim),
    ]

    for title, kw, name in cases:
        bad = run(**kw)
        hit = [b for b in bad if name in b]
        if not hit:
            ok = False
            if verbose:
                print(f"NOT REFUSED: {title} ({name}) -> {bad[:2]}")
        elif verbose:
            print(f"refused: {title} -> {hit[0][:130]}")

    # The uncovered case must not advertise the pending list as a way out.
    text = " ".join(b for b in run(referenced=referenced - {uncovered[0]}) if uncovered[0] in b)
    if "declare it in PUBLIC_PENDING_LANES" in text:
        ok = False
        if verbose:
            print("NOT REFUSED: an uncovered lane was told to use PUBLIC_PENDING_LANES")
    elif verbose:
        print("refused: an uncovered lane is told to get a record, not a declaration")
    if verbose:
        print(f"{'OK' if ok else 'BROKEN'}: {len(cases)} perturbations, "
              f"{'all' if ok else 'not all'} refused by name")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit non-zero on a gap")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="perturb the inputs and require each to be refused")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    data = report()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"{data['registered']} registered lanes: {data['in_shipped_table']} carry cells in "
              f"the shipped table, {data['declared_pending']} are declared in "
              f"PUBLIC_PENDING_LANES ({len(data['both'])} in both), "
              f"{len(data['unaccounted'])} in neither")
        print(f"harness cross-check: {data['harness_cross_check']}")
        for problem in data["problems"]:
            print("  " + problem)
        print(f"{'REFUSED' if data['problems'] else 'OK'}: {len(data['problems'])} problems")
    return 1 if (args.check and data["problems"]) else 0


if __name__ == "__main__":
    sys.exit(main())
