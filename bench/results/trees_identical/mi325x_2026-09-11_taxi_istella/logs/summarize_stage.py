#!/usr/bin/env python3
"""The GBDT stage ledger of a MOJOLEARN_STAGE_TIMES=1 speed log (DEVIATION 2510
stamps), for the LAST fit in the file (the timed round; the warm-up fit comes
first). A fit prints, in order: one per-tree table per tree ("[stage-times]
depthwise fit: rows=..."), one whole-loop table ("[stage-times] Depthwise fit
(doc-parallel boosting)": est.* and the loop's fit wall), then the train_* and
gbdt_fit_* entry tables. Reported: the Python residual (round minus
gbdt_fit_total), the entry table, the per-tree stages summed over the trees,
the loop table, and the loop wall not covered by per-tree walls or est.*
(unwrapped per-iteration work). A stage-timed run drains per stage; it is a
split, not a timing. Usage: summarize_stage.py <stage.log>..."""
import collections
import re
import sys

ROW = re.compile(r"\[stage-times\]\s+(\S[^\t]*?)\s*\t\s*([-\d.eE+]+) ms")
TREE_HEAD = re.compile(r"\[stage-times\] (\S+) fit: rows=(\d+) leaves=(\d+) iterations=(\d+)")
LOOP_HEAD = re.compile(r"\[stage-times\] (\S+) fit \(")
WALL = re.compile(r"accounted ([\d.]+) ms of ([\d.]+) ms fit wall")
ENTRY = re.compile(r"^\s+((?:train|gbdt_fit)_\w+)\t\s*([-\d.eE+]+) s")
ROUND = re.compile(r"^FSPEED lane=(\S+) arm=ours shape=(\S+) round=\d+ ms=([\d.]+) hash=(\S+)")


def new_fit():
    return dict(entry=collections.OrderedDict(), trees=collections.OrderedDict(),
                loop=collections.OrderedDict(), n_trees=0, tree_wall=0.0,
                loop_wall=0.0, kind="-", loop_kind="-")


def fits_of(path):
    fits, cur, where = [], new_fit(), None
    info = dict(lane="-", shape="-", round_ms=None, hash="-")
    for line in open(path, errors="replace"):
        m = ROUND.match(line)
        if m:
            info.update(lane=m.group(1), shape=m.group(2), round_ms=float(m.group(3)), hash=m.group(4))
            continue
        m = TREE_HEAD.search(line)
        if m:
            cur["n_trees"] += 1
            cur["kind"] = m.group(1)
            where = "trees"
            continue
        m = LOOP_HEAD.search(line)
        if m:
            cur["loop_kind"] = m.group(1)
            where = "loop"
            continue
        m = WALL.search(line)
        if m:
            if where == "loop":
                cur["loop_wall"] += float(m.group(2))
            elif where == "trees":
                cur["tree_wall"] += float(m.group(2))
            continue
        m = ROW.match(line)
        if m and where in ("trees", "loop"):
            d = cur[where]
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
            continue
        m = ENTRY.match(line)
        if m:
            cur["entry"][m.group(1)] = float(m.group(2)) * 1e3
            if m.group(1) == "gbdt_fit_total":
                fits.append(cur)
                cur, where = new_fit(), None
    return fits, info


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
    print("lane %s, shape %s, round %s ms, hash %s, fits in log %d, per-tree tables %d (%s), loop table %s"
          % (info["lane"], info["shape"], info["round_ms"], info["hash"], len(fits), f["n_trees"],
             f["kind"], f["loop_kind"]))
    print()
    print("| where | stage | ms |")
    print("|---|---|---|")
    if info["round_ms"] is not None and "gbdt_fit_total" in e:
        print("| Python | round minus gbdt_fit_total | %.1f |" % (info["round_ms"] - e["gbdt_fit_total"]))
    for k, v in e.items():
        print("| entry | %s | %.1f |" % (k, v))
    if f["n_trees"]:
        print("| per-tree | fit wall summed over %d trees | %.1f |" % (f["n_trees"], f["tree_wall"]))
        for k, v in sorted(f["trees"].items(), key=lambda kv: -kv[1]):
            print("| per-tree | %s | %.1f |" % (k, v))
    if f["loop_wall"]:
        print("| loop | loop fit wall | %.1f |" % f["loop_wall"])
        for k, v in f["loop"].items():
            print("| loop | %s | %.1f |" % (k, v))
        print("| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | %.1f |"
              % (f["loop_wall"] - f["tree_wall"] - sum(f["loop"].values())))
    print()
