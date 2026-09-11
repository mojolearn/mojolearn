#!/usr/bin/env python3
"""The GBDT stage ledger of a MOJOLEARN_STAGE_TIMES=1 speed log (DEVIATION 2510
stamps): for the LAST fit in the file (the timed round; the warm-up fit comes
first), the entry table (gbdt_fit_*, train_*), the per-tree stage tables summed
over the trees, the per-tree fit wall, and the Python residual (round minus
gbdt_fit_total). A stage-timed run drains per stage; it is a split, not a
timing. Usage: summarize_stage.py <stage.log>..."""
import collections
import re
import sys

TREE = re.compile(r"\[stage-times\]\s+(\S[^\t]*?)\s*\t\s*([-\d.eE+]+) ms")
TREE_HEAD = re.compile(r"\[stage-times\] (\S+) fit: rows=(\d+) leaves=(\d+) iterations=(\d+)")
WALL = re.compile(r"accounted ([\d.]+) ms of ([\d.]+) ms fit wall")
ENTRY = re.compile(r"^\s+((?:train|gbdt_fit)_\w+)\t\s*([-\d.eE+]+) s")
ROUND = re.compile(r"^FSPEED lane=(\S+) arm=ours shape=(\S+) round=\d+ ms=([\d.]+) hash=(\S+)")


def fits_of(path):
    fits, trees, entry = [], collections.OrderedDict(), collections.OrderedDict()
    n_trees, wall, kind, round_ms, shape, lane, digest = 0, 0.0, "-", None, "-", "-", "-"
    for line in open(path, errors="replace"):
        m = ROUND.match(line)
        if m:
            lane, shape, round_ms, digest = m.group(1), m.group(2), float(m.group(3)), m.group(4)
            continue
        m = TREE_HEAD.search(line)
        if m:
            n_trees += 1
            kind = m.group(1)
            continue
        m = WALL.search(line)
        if m:
            wall += float(m.group(2))
            continue
        m = TREE.match(line)
        if m:
            trees[m.group(1)] = trees.get(m.group(1), 0.0) + float(m.group(2))
            continue
        m = ENTRY.match(line)
        if m:
            entry[m.group(1)] = float(m.group(2)) * 1e3
            if m.group(1) == "gbdt_fit_total":
                fits.append(dict(entry=entry, trees=trees, n_trees=n_trees, wall=wall, kind=kind))
                trees, entry, n_trees, wall = collections.OrderedDict(), collections.OrderedDict(), 0, 0.0
    return fits, dict(lane=lane, shape=shape, round_ms=round_ms, hash=digest)


for path in sys.argv[1:]:
    fits, info = fits_of(path)
    print("## %s" % path)
    print()
    if not fits:
        print("no gbdt_fit_total table in this log")
        print()
        continue
    f = fits[-1]
    e = f["entry"]
    print("lane %s, shape %s, round %s ms, hash %s, fits in log %d, per-tree tables %d (%s)"
          % (info["lane"], info["shape"], info["round_ms"], info["hash"], len(fits), f["n_trees"], f["kind"]))
    print()
    print("| where | stage | ms |")
    print("|---|---|---|")
    if info["round_ms"] is not None and "gbdt_fit_total" in e:
        print("| Python | round minus gbdt_fit_total | %.1f |" % (info["round_ms"] - e["gbdt_fit_total"]))
    for k, v in e.items():
        print("| entry | %s | %.1f |" % (k, v))
    if f["n_trees"]:
        print("| per-tree tables | fit wall summed over %d tables | %.1f |" % (f["n_trees"], f["wall"]))
        for k, v in sorted(f["trees"].items(), key=lambda kv: -kv[1]):
            print("| per-tree tables | %s | %.1f |" % (k, v))
        if "train_fit_with_test" in e:
            print("| loop | train_fit_with_test minus per-tree fit wall (unwrapped per-iteration work) | %.1f |"
                  % (e["train_fit_with_test"] - f["wall"]))
    print()
