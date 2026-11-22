#!/usr/bin/env python3
"""The record indexes: which files under the record trees (verification/records, bench/results) the readers read.

Andrew, 2026-11-25: "what is reading all of this stuff and why? maybe the
reader needs to be fixed." Three readers used to take whatever was on disk:
the verify table builder (`python/mojolearn/_verify_reference.py`
`repository_record_paths` walked every `*.json` under identity_break), and
`tools/verification_matrix.py` and `tools/part_set_census.py` (globbed
`bench/results/**/*.json` and took every JSON with a `cells` dict). They now
read only the rows of two tracked indexes:

  bench/results/identity_break/INDEX.tsv  the verify table's inputs (and the
                                          identity_break part of the matrix)
  verification/records/CELLS_INDEX.tsv    column-shaped JSONs outside
                                          identity_break (matrix, census only)

Each row is `path  sha256  class  vendor  commit`. A reader refuses a listed
file that is missing or whose bytes changed, and ignores a file no row lists.
A new record gets its row from the tool that retains it
(tools/retain_run_summary.py `retain_identity`, tools/identity_admission_guard.py
`inputs`), or by hand with `add`.

Usage:
  python3 tools/record_index.py check            # listed files intact? unlisted column JSONs?
  python3 tools/record_index.py add <file>...    # index new records (idempotent)
  python3 tools/record_index.py add --unlisted-under bench/results/identity_break   # adopt restored records
  python3 tools/record_index.py retire --reason "..." <file>...   # drop rows on purpose
  python3 tools/record_index.py generate [--force]   # ONE-TIME: from what the walk/glob took

Exit: 0 clean; 1 a listed file is missing or changed (or bad usage);
3 (check) column-shaped files no index lists (the readers ignore them).
Single process, reads JSON only; nothing is compiled or run.
"""
import argparse
import glob
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vref(root=ROOT):
    path = os.path.join(root, "python", "mojolearn", "_verify_reference.py")
    spec = importlib.util.spec_from_file_location("_record_index_vref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def walked_records(root):
    """What `repository_record_paths` walked before the index (os.walk of
    identity_break, `.admission-preview-*` pruned, every `*.json`)."""
    base = os.path.join(root, "bench", "results", "identity_break")
    out = []
    for directory, children, names in os.walk(base):
        children[:] = [n for n in children if not n.startswith(".admission-preview-")]
        out.extend(os.path.join(directory, n) for n in names if n.endswith(".json"))
    return out


def globbed_columns(root):
    """What the matrix and the census globbed before the index, over every record tree
    (bench/results and, since lane verify-home, verification/records)."""
    v = vref(root)
    return sorted(p for base in v.RECORD_ROOTS
                  for p in glob.glob(os.path.join(root, base + "**/*.json"), recursive=True))


def rows_of(v, root, paths):
    rows = []
    for path in paths:
        row = v.index_row(root, path) if os.path.isfile(path) else None
        if row is not None:
            rows.append(row)
    return rows


def generate(v, root, force):
    targets = (v.RECORD_INDEX, v.CELLS_INDEX)
    existing = [t for t in targets if os.path.exists(os.path.join(root, t))]
    if existing and not force:
        print(f"REFUSED: {existing} exist; generate is one-time (pass --force to rebuild from the tree)")
        return 1
    walked = rows_of(v, root, walked_records(root))
    ib = os.path.dirname(v.RECORD_INDEX) + "/"
    globbed = rows_of(v, root, globbed_columns(root))
    outside = [r for r in globbed if not r["path"].startswith(ib)]
    inside_glob = {r["path"] for r in globbed if r["path"].startswith(ib)}
    if inside_glob != {r["path"] for r in walked}:
        diff = sorted(inside_glob ^ {r["path"] for r in walked})
        print(f"REFUSED: the walk and the glob disagree inside identity_break ({len(diff)}): {diff[:5]}")
        return 1
    # CORRECTION (2026-11-21): a byte-identical copy, outside identity_break, of a verify input (a box's
    # remote output or an artifacts/ copy of the committed record) was counted twice by the matrix and the
    # census. It is plainly a copy: it is not indexed. Copies inside identity_break stay (table inputs).
    inputs = {r["sha256"]: r["path"] for r in walked}
    copies = [r for r in outside if r["sha256"] in inputs]
    outside = [r for r in outside if r["sha256"] not in inputs]
    for r in copies:
        print("COPY-NOT-INDEXED " + r["path"] + " = " + inputs[r["sha256"]])
    v.write_index(root, v.RECORD_INDEX, walked, v.INDEX_NOTES[v.RECORD_INDEX])
    v.write_index(root, v.CELLS_INDEX, outside, v.INDEX_NOTES[v.CELLS_INDEX])
    by_sha = {}
    for r in walked + outside:
        by_sha.setdefault(r["sha256"], []).append(r["path"])
    dups = {s: p for s, p in by_sha.items() if len(p) > 1}
    print(f"GENERATED {v.RECORD_INDEX} rows={len(walked)} {v.CELLS_INDEX} rows={len(outside)} "
          f"byte-identical groups={len(dups)} extra copies={sum(len(p) - 1 for p in dups.values())}")
    for s, p in sorted(dups.items(), key=lambda kv: kv[1]):
        print("DUPLICATE " + s[:16] + " " + " ".join(p))
    return 0


def unlisted_columns(v, root, listed, inputs, under=""):
    """Column-shaped files under the record trees (or `under`) no index lists, except a byte copy,
    outside identity_break, of an indexed identity_break record (not indexed, see generate)."""
    ib = os.path.dirname(v.RECORD_INDEX) + "/"
    out = []
    for path in globbed_columns(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if rel in listed or not rel.startswith(under):
            continue
        row = v.index_row(root, path)
        if row is not None and (rel.startswith(ib) or row["sha256"] not in inputs):
            out.append(rel)
    return out


def listed_rows(v, root, verify):
    listed, inputs, rc = set(), set(), 0
    for index in (v.RECORD_INDEX, v.CELLS_INDEX):
        try:
            rows = v.read_index(root, index, verify=verify)
        except v.RecordIndexError as exc:
            print(f"REFUSED: {exc}")
            rc = 1
            rows = v.read_index(root, index, verify=False) if os.path.isfile(os.path.join(root, index)) else []
        listed.update(rel for rel, _ in rows)
        if index == v.RECORD_INDEX:
            inputs.update(row["sha256"] for _, row in rows)
        print(f"INDEX {index} rows={len(rows)}")
    return listed, inputs, rc


def check(v, root):
    listed, inputs, rc = listed_rows(v, root, verify=True)
    unlisted = unlisted_columns(v, root, listed, inputs)
    for rel in unlisted:
        print("UNLISTED " + rel)
    print(f"CHECK listed_ok={rc == 0} unlisted={len(unlisted)}")
    return rc or (3 if unlisted else 0)


def retire(v, root, paths, reason):
    if not reason.strip():
        print("REFUSED: retire needs --reason")
        return 1
    gone = {os.path.relpath(os.path.abspath(p), root).replace(os.sep, "/") for p in paths}
    removed = 0
    for index in (v.RECORD_INDEX, v.CELLS_INDEX):
        rows = v.read_index(root, index, verify=False)
        keep = [row for rel, row in rows if rel not in gone]
        if len(keep) != len(rows):
            removed += len(rows) - len(keep)
            v.write_index(root, index, keep, v.index_header(root, index))
    print(f"RETIRED rows={removed} reason={reason!r}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    g = sub.add_parser("generate")
    g.add_argument("--force", action="store_true")
    a = sub.add_parser("add")
    a.add_argument("files", nargs="*")
    a.add_argument("--unlisted-under", metavar="DIR", default=None,
                   help="also add every column-shaped file under DIR that `check` lists as UNLISTED "
                        "(an explicit act, e.g. after records are restored from git history)")
    r = sub.add_parser("retire")
    r.add_argument("--reason", required=True)
    r.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    v = vref(root)
    try:
        if args.command == "generate":
            return generate(v, root, args.force)
        if args.command == "check":
            return check(v, root)
        if args.command == "add":
            files = list(args.files)
            if args.unlisted_under:
                under = os.path.relpath(os.path.abspath(args.unlisted_under), root).replace(os.sep, "/").rstrip("/") + "/"
                listed, inputs, _ = listed_rows(v, root, verify=False)
                files += [os.path.join(root, rel) for rel in unlisted_columns(v, root, listed, inputs, under)]
            added = v.add_to_index(root, files)
            for row in added:
                print("ADDED " + row["path"])
            print(f"ADD rows={len(added)}")
            return 0
        return retire(v, root, args.files, args.reason)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
