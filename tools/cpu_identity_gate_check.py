#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The verdict steps of .github/workflows/cpu-identity-gate.yml (the CPU
training lane, 2026-09-13; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md
section 3.4), as a tool that runs on a laptop too.

  readback   import the checkout's package on this CPU-only box and read
             every host binding under mojolearn/host/ back: vendor() must be
             cpu, and each binding's <prefix>_column() must be "cpu" (the
             kernel matrix's CPU column, the comptime assert's witness).
             Exit 2 when a named binding is not built.
  column     judge a tools/identity_break.py JSON written on a CPU-only box:
             vendor starts with cpu-, commit equals --commit, host.column is
             cpu, every named host binding is in host.families with column
             cpu, at least one cell exists, every cell of a lane in --covered
             is STABLE, and every other cell is REFUSED with the by-name
             sentence "no CPU implementation of". A JSON with no cell, or a
             cell of the wrong kind, fails: a step that was supposed to
             produce a cell and produced none is a failure, never a pass.

Exit 0: every check holds. 1: a check failed. 2: the tool could not run.
"""
import argparse
import json
import os
import sys

REFUSAL = "no CPU implementation of"


def do_readback(args):
    sys.path.insert(0, os.path.abspath(args.package_root))
    try:
        import mojolearn
        from mojolearn import _backend
    except Exception as exc:
        print(f"readback: import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    vendor = mojolearn.vendor()
    print(f"readback vendor() = {vendor}")
    print(f"readback vendor_how() = {_backend.vendor_how()}")
    built = _backend.host_families_built()
    print(f"readback host_families_built = {built}")
    if vendor != "cpu":
        print("readback FAIL: this package did not take the CPU-only path", file=sys.stderr)
        return 1
    missing = [b for b in args.binding if b not in built]
    if missing:
        print(f"readback: not built: {missing}", file=sys.stderr)
        return 2
    bad = 0
    for basename in args.binding or built:
        prefix = basename[len("_mojolearn_"):]
        try:
            m = _backend.load_host_module(basename)
        except Exception as exc:
            print(f"readback FAIL {basename}: {type(exc).__name__}: {exc}", file=sys.stderr)
            bad += 1
            continue
        column = str(getattr(m, prefix + "_column")())
        detected = getattr(m, prefix + "_detected_column", None)
        detected = str(detected()) if detected is not None else "(no read-back)"
        numeric = int(getattr(m, prefix + "_numeric_mode")())
        print(f"readback {basename}: column={column} detected_column={detected} "
              f"numeric_mode={numeric} vendor={getattr(m, prefix + '_vendor')()}")
        if column != "cpu":
            print(f"readback FAIL {basename}: column {column!r} is not cpu", file=sys.stderr)
            bad += 1
    print(f"readback verdict {'OK' if not bad else 'FAIL'}")
    return 1 if bad else 0


def do_column(args):
    try:
        with open(args.json) as fh:
            j = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"column: cannot read {args.json}: {exc}", file=sys.stderr)
        return 2
    covered = set(x for x in (args.covered or "").split(",") if x)
    bindings = [x for x in (args.binding or "").split(",") if x]
    failures = []

    def need(cond, text):
        if not cond:
            failures.append(text)

    vendor = str(j.get("vendor", ""))
    need(vendor.startswith("cpu-") and len(vendor) > 4, f"vendor {vendor!r} does not start with cpu-")
    need(bool(j.get("commit")), "commit is empty")
    if args.commit:
        need(j.get("commit") == args.commit, f"commit {j.get('commit')!r} is not {args.commit!r}")
    host = j.get("host") or {}
    need(bool(host), "no host object; the run was not on a CPU-only install")
    need(host.get("column") == "cpu", f"host.column is {host.get('column')!r}, not cpu")
    need(bool(host.get("cpu_model")), "host.cpu_model is empty")
    families = host.get("families") or {}
    for b in bindings:
        need(b in families, f"host.families lacks {b}")
        need(families.get(b, {}).get("column") == "cpu", f"host.families[{b}].column is not cpu")
    cells = j.get("cells") or {}
    need(len(cells) > 0, "the JSON carries NO cell; a run that was supposed to produce cells produced none")
    need(j.get("complete", False), "the JSON is INCOMPLETE (the run was killed)")
    seen = set()
    for key, cell in cells.items():
        lane = key.split("/")[0]
        seen.add(lane)
        verdict = cell.get("verdict")
        if lane in covered:
            need(verdict == "STABLE",
                 f"{key}: covered lane reads {verdict}, not STABLE" + (f" ({cell.get('error', '')[:160]})" if verdict == "REFUSED" else ""))
        else:
            need(verdict == "REFUSED", f"{key}: uncovered lane reads {verdict}, not REFUSED; a hash from a lane with no CPU implementation is a routing bug")
            need(REFUSAL in str(cell.get("error", "")),
                 f"{key}: refused, but not by name ({str(cell.get('error', ''))[:160]!r})")
    for lane in sorted(covered - seen):
        need(False, f"covered lane {lane} has no cell in the JSON")
    print(f"column {os.path.basename(args.json)}: vendor={vendor} commit={j.get('commit')} "
          f"host.column={host.get('column')} cpu_model={host.get('cpu_model')!r} "
          f"families={sorted(families)} cells={len(cells)} lanes={sorted(seen)} covered={sorted(covered)}")
    for f in failures:
        print(f"column FAIL: {f}")
    print(f"column verdict {'OK' if not failures else 'FAIL'} ({len(failures)} failure(s))")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rb = sub.add_parser("readback", help="read every host binding back on this CPU-only box")
    rb.add_argument("--package-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    rb.add_argument("--binding", action="append", default=[], metavar="BASENAME",
                    help="a host binding that must be built and read back as cpu (repeatable)")
    col = sub.add_parser("column", help="judge an identity_break JSON written on a CPU-only box")
    col.add_argument("json")
    col.add_argument("--covered", default="", help="lanes that must be STABLE, comma separated; every other lane must be REFUSED by name")
    col.add_argument("--commit", default="", help="the commit the JSON must carry")
    col.add_argument("--binding", default="", help="host bindings that must appear in host.families with column cpu, comma separated")
    args = ap.parse_args(argv)
    if args.cmd == "readback":
        return do_readback(args)
    return do_column(args)


if __name__ == "__main__":
    sys.exit(main())
