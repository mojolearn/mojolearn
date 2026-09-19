#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GAP REGISTRIES ARE WELL FORMED, checked rather than asserted.

Every section carries a `NOT_IMPLEMENTED.tsv` naming what of the reference
it does not implement. A reader greps one, finds a row and concludes the
thing is a gap, so a row that is stale or misshapen is worse than no row.
On 2026-09-19 an audit of the 27 files found, and this check refuses:

  * TWO INCOMPATIBLE SCHEMAS. `gbdt/` held `file, symbol, why` while every
    other file held `thing, status, why`, so column 2 was a STATUS in 26
    files and a SYMBOL in one. Seven files declared no schema at all.
  * A LOST NEWLINE. `decomposition/` line 17 carried two rows on one line
    (5 columns), so one gap had no `what` at all and could not be found.
  * A STALE ROW. `hierarchy/` said `build_mr_linkage` was NOT IMPLEMENTED
    and "the ROADMAP's Phase 1" for the 19 days after it shipped in
    `hdbscan/impl/cluster/detail/single_linkage.mojo` (a48fc8032).
  * RETIRED FRAMING. `why_not_ported`, `NOT MIRRORED`, `upstream` -- the
    words ported-is-ours retired, still in the schema headers themselves.

WHAT IS NOT CHECKED HERE. Whether a row's CLAIM is true: that needs the
section's code, and the one stale row above was found by reading it. This
check is the cheap half, run on every change; the reading is the lane's.
"""
import pathlib
import re
import sys

RETIRED = ("why_not_ported", "not mirrored", "NOT MIRRORED", "upstream\t",
           "upstream_item", "upstream_file_or_symbol")
# A STATUS IS NOT A CLOSED VOCABULARY. The registries carry statuses like
# "their dead code", "NO LONGER NEEDED, retired" and "DIVERGES, DEVIATION
# 665", and that specificity is the point -- an enum would throw it away.
# What is refused is the shape the gbdt bug had: a C++ SYMBOL sitting in
# the status column, which is what a second schema silently produces.
_SYMBOLISH = re.compile(r"^\S+$")
_CAMEL = re.compile(r"[a-z][A-Z]")


def looks_like_a_symbol(status):
    """A status names a claim; a symbol names their code. Tell them apart."""
    if not _SYMBOLISH.match(status):
        return False  # any whitespace at all and it reads as a claim
    return bool(_CAMEL.search(status)) or "::" in status or status.endswith(
        (".cu", ".cuh", ".h", ".hpp", ".pyx", ".py"))


def check(root=pathlib.Path(".")):
    bad = []
    files = sorted(root.glob("*/NOT_IMPLEMENTED.tsv"))
    if not files:
        bad.append("no NOT_IMPLEMENTED.tsv found at all")
    for p in files:
        text = p.read_text()
        for word in RETIRED:
            if word in text:
                bad.append(f"{p}: retired framing {word!r} (ported-is-ours)")
        schema = None
        for n, line in enumerate(text.split("\n"), 1):
            if line.startswith("#"):
                if schema is None and "\t" in line:
                    schema = line.lstrip("#").split("\t")
                continue
            if not line.strip():
                continue
            if schema is None:
                bad.append(f"{p}:{n}: a data row before any declared schema")
                break
            cols = line.split("\t")
            if len(cols) != len(schema):
                bad.append(f"{p}:{n}: {len(cols)} columns, schema declares "
                           f"{len(schema)} -- a lost newline or a stray tab")
                continue
            status = cols[1].strip()
            if not status:
                bad.append(f"{p}:{n}: empty status")
            elif looks_like_a_symbol(status):
                bad.append(f"{p}:{n}: status {status[:40]!r} is a SYMBOL, not a "
                           "claim -- column 2 holds a status (see gbdt, 2026-09-19)")
    return bad


def main(argv=None):
    root = pathlib.Path(argv[1]) if argv and len(argv) > 1 else pathlib.Path(".")
    bad = check(root)
    for b in bad:
        print(b)
    n = len(sorted(root.glob("*/NOT_IMPLEMENTED.tsv")))
    print(f"{'REFUSED' if bad else 'OK'}: {n} registries, {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
