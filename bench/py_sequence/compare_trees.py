# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BASE == NEW for every identity-lane column merged_check wrote:
    python3 compare_trees.py <base clean dir> <new clean dir>
Every string (and list of strings) inside each cell, which is where the harness
keeps its hashes, must be equal; numbers (timings, counts) are not compared.
Prints one line per lane and column and a RESULT line; exit 1 on any difference
or when nothing was compared."""
import json
import sys
from pathlib import Path


def strings(v, path=""):
    if isinstance(v, str):
        yield path, v
    elif isinstance(v, dict):
        for k in sorted(v):
            yield from strings(v[k], f"{path}/{k}")
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            yield from strings(x, f"{path}[{i}]")


def main():
    base, new = Path(sys.argv[1]), Path(sys.argv[2])
    diff = compared = 0
    for bf in sorted(base.glob("*.json")):
        nf = new / bf.name
        if not nf.is_file():
            print(f"MISSING in NEW: {bf.name}")
            diff += 1
            continue
        b = json.loads(bf.read_text()).get("cells", {})
        n = json.loads(nf.read_text()).get("cells", {})
        bad = []
        cells = 0
        for key in sorted(set(b) | set(n)):
            if key not in b or key not in n:
                bad.append(f"{key}: only in {'BASE' if key in b else 'NEW'}")
                continue
            bs, ns = dict(strings(b[key])), dict(strings(n[key]))
            if not bs:
                continue
            cells += 1
            for p in sorted(set(bs) | set(ns)):
                if bs.get(p) != ns.get(p):
                    bad.append(f"{key}{p}: {bs.get(p)!r:.40} != {ns.get(p)!r:.40}")
        compared += cells
        diff += len(bad)
        print(f"{bf.name}: {cells} cells, {'EQUAL' if not bad else 'DIFFER ' + str(len(bad))}")
        for x in bad[:6]:
            print("   ", x)
    ok = diff == 0 and compared > 0
    print(f"RESULT BASE==NEW: {'EQUAL' if ok else 'NOT EQUAL'} ({compared} cells compared, {diff} differences)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
