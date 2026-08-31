#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Put an SPDX line and a copyright line at the top of every source file.

WHY THIS EXISTS. There are three ways someone acquires this library and
attribution travelled with only two of them.

    forks the repository   LICENSE, NOTICE and CITATION.cff come with it
    pip installs it        the wheel carries both plus the metadata
    COPIES ONE FILE        nothing travelled at all

The third is not a hypothetical and it is not malice. A developer who pastes
`gemm/original/gemm_identical.mojo` into their project receives 1,589 lines
of the hardest work in this repository with no statement of who wrote it,
what licence it carries, or that a DOI exists, and neither they nor anyone
downstream of them has any way to find out. That is how attribution is
actually lost.

IT CUTS THE OTHER WAY TOO, and that half is an obligation rather than a
preference. `NOTICE` records that everything under a `derived/` directory
derives from CatBoost at 54a8143a, or from cuML, cuVS, RAFT or FAISS at
their pinned commits. A lifted file from `derived/` carried none of that
either, so it stripped the UPSTREAM attribution that Apache 2.0 section 4
obliges this project to propagate. A `derived/` file therefore gets a third
line pointing at its lane's `DERIVATION_MAP.tsv`, where the upstream path and
the pinned commit are recorded per file.

Measured before the first run, 2026-08-31: 2 of 982 `.mojo` files and 0 of
53 `.py` files carried any copyright or SPDX line.

    python3 tools/add_source_headers.py --check    report, change nothing
    python3 tools/add_source_headers.py            write

IDEMPOTENT BY CONSTRUCTION: a file that already contains
`SPDX-License-Identifier` is left exactly as it is, so this can be re-run
after new files land and can sit in a gate.

SAFE BEFORE A MODULE DOCSTRING, which is the one thing that could have made
this dangerous. A module docstring must be the first STATEMENT, and a
comment is not a statement, so `# SPDX...` above `\"\"\"...\"\"\"` leaves the
docstring a docstring. Verified by compiling a probe with `mojo run` before
this script was pointed at anything real. A `#!` line stays first.
"""

import os
import sys

SPDX = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT = (
    "# Copyright 2026 Andrew Hendel. Part of mojolearn, "
    "https://doi.org/10.5281/zenodo.22068632"
)
DERIVED = (
    "# Derivative work: the upstream file and its pinned commit are recorded "
    "in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE."
)

SKIP_DIRS = {
    ".pixi", ".git", "upstream", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".mojo-cache",
}
# bench/results is evidence: fetched logs, cards and JSON from rented boxes.
# Editing a record would falsify it.
SKIP_PATH_PARTS = (os.sep + "bench" + os.sep + "results" + os.sep,)
EXTS = (".mojo", ".py")


def wants_header(path):
    if not path.endswith(EXTS):
        return False
    for part in SKIP_PATH_PARTS:
        if part in path:
            return False
    return True


# LANES THAT MIRROR AN UPSTREAM WITHOUT HAVING A `derived/` SUBDIRECTORY.
#
# Found 2026-08-31, and it was the reason the first proportion table was
# wrong: `gbdt/` IS the CatBoost mirror and `ensemble/` mirrors cuML, and
# neither has ever had a `derived/` directory, so a header rule keyed on the
# directory gave 191 files (89,766 lines, the most plainly derived code in
# the repository) NO provenance marker at all while 310 files under
# `derived/` had one.
#
# The line for these is deliberately about the LANE rather than the file.
# 76 of gbdt's 149 files appear in no derivation map and three say NO
# CATBOOST COUNTERPART in their own headers, so asserting file by file here
# would be asserting something unchecked. Pointing at the map and the file's
# own docstring is true of every file in the lane.
LANE_UPSTREAM = {
    "gbdt": (
        "# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file "
        "provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own "
        "docstring, and in NOTICE; files with no CatBoost counterpart say so."
    ),
    "ensemble": (
        "# This lane MIRRORS cuML's random forest. Per-file provenance is in "
        "this file's own docstring and in NOTICE."
    ),
}


def header_for(path, root):
    lines = [SPDX, COPYRIGHT]
    if (os.sep + "derived" + os.sep) in path or (os.sep + "ported" + os.sep) in path:
        lines.append(DERIVED)
    else:
        rel = os.path.relpath(path, root)
        lane = rel.split(os.sep)[0]
        # `<lane>/original/` is the lane's own work, not its mirror.
        if lane in LANE_UPSTREAM and (os.sep + "original" + os.sep) not in path:
            lines.append(LANE_UPSTREAM[lane])
    return "\n".join(lines) + "\n"


def process(path, write, root):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if "SPDX-License-Identifier" in text:
        return "skip"
    lines = text.split("\n")
    at = 0
    # A shebang stays first, and so does a coding declaration if one follows.
    if lines and lines[0].startswith("#!"):
        at = 1
        if len(lines) > 1 and "coding" in lines[1] and lines[1].lstrip().startswith("#"):
            at = 2
    new = "\n".join(lines[:at]) + ("\n" if at else "") + header_for(path, root) + "\n".join(lines[at:])
    if write:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    return "add"


def main():
    check = "--check" in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    added = skipped = 0
    added_ported = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not wants_header(p):
                continue
            what = process(p, write=not check, root=root)
            if what == "add":
                added += 1
                if (os.sep + "ported" + os.sep) in p:
                    added_ported += 1
            else:
                skipped += 1
    verb = "would add" if check else "added"
    print(f"{verb} a header to {added} files ({added_ported} of them under derived/)")
    print(f"already carried one: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
