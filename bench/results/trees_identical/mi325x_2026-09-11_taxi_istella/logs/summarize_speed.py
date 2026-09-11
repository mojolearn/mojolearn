#!/usr/bin/env python3
"""Summarize FSPEED lines of every arm in the leg's speed logs: set, lane, dataset, rows,
mode, arm, rounds, median, min..max, hash, FSPEED-ACC. Stage logs are skipped.
Log names: <set>.<lane>.<dataset>.r<rows>.<mode>.log"""
import glob, os, re, statistics, sys
root = sys.argv[1]
print("| set | lane | dataset | rows | mode | arm | rounds | median ms | min..max | hash | FSPEED-ACC | log |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(os.path.join(root, "*.log"))):
    b = os.path.basename(p)[:-4]
    if len(b.split(".")) < 5:
        continue   # derived logs (e.g. ab_pointwise.taxi.before), not a cell
    st, lane, ds, r, mode = b.split(".")[:5]
    if mode == "stage":
        continue
    arms = {}
    for line in open(p, errors="replace"):
        m = re.match(r"FSPEED lane=\S+ arm=(\S+) shape=\S+ round=(\d+) ms=([\d.]+)(?: hash=(\w+))?", line)
        if m:
            a = arms.setdefault(m.group(1), {"ms": [], "hash": set(), "acc": ""})
            a["ms"].append(float(m.group(3)))
            if m.group(4): a["hash"].add(m.group(4))
        m = re.match(r"FSPEED-ACC lane=\S+ arm=(\S+) (.*)", line)
        if m:
            a = arms.setdefault(m.group(1), {"ms": [], "hash": set(), "acc": ""})
            a["acc"] = (a["acc"] + " " if a["acc"] else "") + m.group(2).strip()
        m = re.match(r"FSPEED-REFUSED lane=\S+ arm=(\S+) (.*)", line)
        if m:
            a = arms.setdefault(m.group(1), {"ms": [], "hash": set(), "acc": ""})
            a["acc"] = "REFUSED " + m.group(2).strip()[:80]
    if not arms:
        print(f"| {st} | {lane} | {ds} | {r[1:]} | {mode} | - | 0 | no FSPEED lines | | | | {b} |"); continue
    for arm, a in arms.items():
        ms = a["ms"]
        med = f"{statistics.median(ms):.0f}" if ms else "-"
        rng = f"{min(ms):.0f}..{max(ms):.0f}" if ms else "-"
        print(f"| {st} | {lane} | {ds} | {r[1:]} | {mode} | {arm} | {len(ms)} | {med} | {rng} | {','.join(sorted(a['hash']))} | {a['acc']} | {b} |")
