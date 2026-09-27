#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fixture-RNG census: no NEW copy of a fixture-RNG function.

`checks/fixture_rng_gate.mojo` imports every existing copy by name and
proves each equals its canonical counterpart in `checks/fixture_rng.mojo`.
This census is its guard against falling behind: it finds every Mojo
definition whose NAME belongs to the fixture-RNG family and fails when one
is neither imported by the gate nor listed in NOT_RNG below (a name
collision that is not a fixture RNG: a digest step, a lerp, a formatter).

New check and binding code imports `checks/fixture_rng.mojo`. A new copy
fails here by file and name; the fix is to import the canonical function,
not to add the copy to the gate.

Exit 0 clean, 1 on a new copy or a stale gate entry, 2 on a usage error.
`--self-test` plants a new copy in a scratch tree and requires the census
to catch it (and to pass on the unplanted scratch tree).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = "checks/fixture_rng_gate.mojo"
CANONICAL = "checks/fixture_rng.mojo"

# The family, by name. A definition of one of these names, or of a name
# with one of the suffixes, is a fixture-RNG definition.
FAMILY_NAMES = {
    "_mix", "mix", "_mix32", "mix32", "mix64", "fmix32", "h32", "_h",
    "splitmix", "_splitmix", "splitmix64", "_splitmix64",
    "_u01", "u01", "_hashed", "hashed", "hashed_unit", "_hash01", "_hash64",
    "_gemm_mix",
}
FAMILY_SUFFIXES = ("_splitmix64", "_mix64", "_u01", "_hashed")

# Definitions with a family name that are NOT fixture RNGs (2026-09-27
# inventory). Each is (file, name). They are left alone and not gated.
NOT_RNG = {
    ("extratrees/bench/batched_ab.mojo", "mix"),        # FNV-1a digest step (nested)
    ("extratrees/bench/batchwidth_ab.mojo", "mix"),     # FNV-1a digest step (nested)
    ("tools/umap_sparse_build_bench.mojo", "mix"),      # FNV-1a digest step
    ("bench/host_rms_cpu_price_main.mojo", "_mix"),     # FNV-1a over a Float32's bytes
    ("bench/speed/classical_ladder_main.mojo", "_hashed"),  # hash + count formatter
    ("holtwinters/impl/internal/hw_eval.mojo", "_mix"),     # pinned a*x + (1-a)*y
    ("holtwinters/impl/internal/hw_estimate.mojo", "_mix"),  # pinned a*x + (1-a)*y
    ("holtwinters/host/hw_oracle.mojo", "_mix"),            # pinned a*x + (1-a)*y
}

SKIP_DIRS = {".git", ".pixi", "node_modules", "__pycache__"}
SKIP_PREFIXES = ("mamba/host/gen/",)
DEF_RE = re.compile(r"^\s*(?:fn|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\[(]")
IMPORT_RE = re.compile(r"^from\s+([A-Za-z0-9_.]+)\s+import\s+([A-Za-z0-9_]+)\s+as\s+(c_[A-Za-z0-9_]+)\s*$")


def in_family(name: str) -> bool:
    return name in FAMILY_NAMES or name.endswith(FAMILY_SUFFIXES)


def definitions(root: Path) -> set[tuple[str, str]]:
    found = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith(".mojo"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")
            if rel.startswith(SKIP_PREFIXES) or rel in (GATE, CANONICAL):
                continue
            with open(os.path.join(dirpath, f), errors="replace") as fh:
                for line in fh:
                    m = DEF_RE.match(line)
                    if m and in_family(m.group(1)):
                        found.add((rel, m.group(1)))
    return found


def gated(root: Path) -> set[tuple[str, str]]:
    out = set()
    text = (root / GATE).read_text()
    for line in text.splitlines():
        m = IMPORT_RE.match(line)
        if m:
            out.add((m.group(1).replace(".", "/") + ".mojo", m.group(2)))
    return out


def census(root: Path) -> int:
    defs = definitions(root)
    known = gated(root)
    if not known:
        print(f"fixture-RNG census: {GATE} imports no copy; refusing (the list is empty, so nothing is guarded)")
        return 1
    new = sorted(defs - known - NOT_RNG)
    stale = sorted(known - defs)
    stale_not_rng = sorted(NOT_RNG - defs)
    for f, n in new:
        print(f"NEW COPY  {f}::{n}: import it from {CANONICAL} instead of defining it")
    for f, n in stale:
        print(f"STALE     {f}::{n} is imported by the gate but no longer defined; drop it from {GATE}")
    for f, n in stale_not_rng:
        print(f"STALE     {f}::{n} is in NOT_RNG but no longer defined; drop it from this census")
    print(
        f"fixture-RNG census: {len(defs)} family definitions, {len(known)} gated, "
        f"{len(defs & NOT_RNG)} not RNG, {len(new)} new, {len(stale) + len(stale_not_rng)} stale"
    )
    return 1 if (new or stale or stale_not_rng) else 0


def self_test() -> int:
    """The census must catch a planted new copy; a census that cannot fail
    guards nothing."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "checks").mkdir()
        shutil.copy(ROOT / GATE, tmp / GATE)
        for f, n in sorted(gated(ROOT)) + sorted(NOT_RNG):
            p = tmp / f
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(f"def {n}(x: UInt64) -> UInt64:\n    return x\n")
        clean = census(tmp)
        (tmp / "checks" / "planted_new_copy.mojo").write_text(
            "def splitmix(x: UInt64) -> UInt64:\n    return x\n"
        )
        planted = census(tmp)
    if clean != 0 or planted != 1:
        print(f"SELF-TEST FAIL: clean tree exit {clean} (want 0), planted tree exit {planted} (want 1)")
        return 1
    print("SELF-TEST PASS: the census passes the clean tree and catches a planted new copy")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return census(args.root)


if __name__ == "__main__":
    sys.exit(main())
