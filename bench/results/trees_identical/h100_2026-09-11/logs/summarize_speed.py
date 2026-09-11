#!/usr/bin/env python3
"""Summarize FSPEED lines of ours-only logs: set, lane, rows, rounds, median, min..max, hash, FSPEED-ACC.
Log names: <set>.<lane>.<dataset>.r<rows>.<mode>[.passN].log; stage logs are skipped."""
import glob, os, re, statistics, sys
root = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(root, "*.log"))):
    b = os.path.basename(p)[:-4]
    parts = b.split(".")
    if "stage" in parts:
        continue
    st, lane, ds, r = parts[0], parts[1], parts[2], parts[3][1:]
    tag = parts[5] if len(parts) > 5 else ""
    ms, hs, acc = [], set(), ""
    for line in open(p, errors="replace"):
        m = re.match(r"FSPEED lane=\S+ arm=ours shape=\S+ round=(\d+) ms=([\d.]+) hash=(\w+)", line)
        if m:
            ms.append(float(m.group(2))); hs.add(m.group(3))
        if line.startswith("FSPEED-ACC") and "arm=ours" in line:
            acc = (acc + " " if acc else "") + line.strip().split("arm=ours", 1)[1].strip()
    if not ms:
        rows.append((st, lane, ds, r, tag, 0, "no FSPEED ours lines", "", "", "", b)); continue
    rows.append((st, lane, ds, r, tag, len(ms), f"{statistics.median(ms):.0f}", f"{min(ms):.0f}..{max(ms):.0f}", ",".join(sorted(hs)), acc, b))
print("| set | lane | dataset | rows | pass | rounds | median ms | min..max | hash | FSPEED-ACC (ours) | log |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
