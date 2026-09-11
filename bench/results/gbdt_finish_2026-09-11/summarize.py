#!/usr/bin/env python3
"""gbdt-finish lane: before/after table and per-switch verdicts from the
speed logs of body_ab.sh (RUNS ON THE POD: python3 summarize.py /root/trees_out).

Pairs, each one switch:
  2634  baseline -> a2634
  2635  a2634    -> both
  2636  both     -> all
  all   baseline -> all   (the lane's total)
Every log is one process of 5 timed fits; a cell's time is the median over
every FSPEED line of that (set, lane, dataset) across the rounds. A shared
switch has ONE default (ENGINEERING_RULES 9): its time gate is the geometric
mean of after/before over all six (policy, dataset) cells, quality not worse
in any cell. tools/flip_verdict.py is also run per policy for the record.
"""
import collections
import glob
import math
import os
import re
import statistics
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/trees_out"
TAG = sys.argv[2] if len(sys.argv) > 2 else "ab"
LANES = ("gbdt-symmetric", "gbdt-depthwise", "gbdt-lossguide")
DATASETS = ("taxi", "istella")
PAIRS = (("2634", "baseline", "a2634"), ("2635", "a2634", "both"),
         ("2636", "both", "all"), ("total", "baseline", "all"))
REF = {("gbdt-symmetric", "taxi"): "90c3558501933f47",
       ("gbdt-depthwise", "taxi"): "40c1683b9e0eb151",
       ("gbdt-lossguide", "taxi"): "0dd8bcfc3c3a4a1d",
       ("gbdt-symmetric", "istella"): "238d3abce0cabf43",
       ("gbdt-depthwise", "istella"): "5d053cd086658072",
       ("gbdt-lossguide", "istella"): "6182fd2bee4fb941"}
LINE = re.compile(r"^FSPEED lane=(\S+) arm=ours shape=\S+ round=\d+ ms=([\d.]+) hash=(\S+)")
ACC = re.compile(r"^FSPEED-ACC lane=(\S+) arm=ours metric=(\S+) value=([-\d.eE+naif]+)")


def logs(s, lane, ds):
    return sorted(glob.glob("%s/speed/%s.%s.%s.r1000000.ours.%s.r*.log" % (OUT, s, lane, ds, TAG)))


def cell(s, lane, ds):
    ms, hashes, acc = [], collections.Counter(), collections.defaultdict(set)
    rounds = 0
    for p in logs(s, lane, ds):
        rounds += 1
        for ln in open(p, errors="replace"):
            m = LINE.match(ln)
            if m:
                ms.append(float(m.group(2)))
                hashes[m.group(3)] += 1
            a = ACC.match(ln)
            if a:
                acc[a.group(2)].add(a.group(3))
    return ms, hashes, acc, rounds


def main():
    sets = sorted({os.path.basename(p).split(".")[0] for p in glob.glob(OUT + "/speed/*.ours.%s.r*.log" % TAG)})
    print("# sets with logs: %s" % " ".join(sets))
    cells = {}
    print("| set | policy | dataset | rounds | fits | median ms | min..max | hashes (count) | ref hash held | logloss | auc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for lane in LANES:
            for s in sets:
                ms, hashes, acc, rounds = cell(s, lane, ds)
                if not ms:
                    continue
                cells[(s, lane, ds)] = (statistics.median(ms), hashes, acc)
                held = "yes" if set(hashes) == {REF[(lane, ds)]} else "NO"
                print("| %s | %s | %s | %d | %d | %.1f | %.1f..%.1f | %s | %s | %s | %s |" % (
                    s, lane.split("-")[1], ds, rounds, len(ms), statistics.median(ms), min(ms), max(ms),
                    ",".join("%s(%d)" % kv for kv in hashes.items()), held,
                    "/".join(sorted(acc.get("logloss", {"-"}))), "/".join(sorted(acc.get("auc", {"-"})))))
    print()
    for name, before, after in PAIRS:
        ratios, notes = [], []
        for ds in DATASETS:
            for lane in LANES:
                b, a = cells.get((before, lane, ds)), cells.get((after, lane, ds))
                if not b or not a:
                    notes.append("missing %s %s" % (lane, ds))
                    continue
                r = a[0] / b[0]
                ratios.append(r)
                q = "quality equal" if (b[2] == a[2] and set(b[1]) == set(a[1])) else "QUALITY OR HASH DIFFERS"
                print("switch %s %s %s: before %.1f after %.1f after/before %.4f %s" % (
                    name, lane.split("-")[1], ds, b[0], a[0], r, q))
        if ratios:
            g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            print("switch %s six-cell geomean %.4f over %d cells%s -> %s" % (
                name, g, len(ratios), (" (" + "; ".join(notes) + ")") if notes else "",
                "FLIP" if g < 1.0 and len(ratios) == 6 else "NO FLIP"))
        for lane in LANES:
            args = ["python3", "tools/flip_verdict.py", "--lane", lane]
            ok = True
            for ds in DATASETS:
                bl, al = logs(before, lane, ds), logs(after, lane, ds)
                if not bl or not al:
                    ok = False
                    break
                args += ["--%s-before" % ds] + bl + ["--%s-after" % ds] + al
            if not ok:
                continue
            res = subprocess.run(args, cwd="/root/mojolearn", capture_output=True, text=True)
            last = (res.stdout.strip().splitlines() or ["(no output) " + res.stderr.strip()[-200:]])[-1]
            print("flip_verdict %s %s: %s" % (name, lane, last))
        print()


if __name__ == "__main__":
    main()
