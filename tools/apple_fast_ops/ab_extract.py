#!/usr/bin/env python3
"""M3: one line per A/B tag matching argv[1] regex: tag | A ms | B ms | dB% | A quality | B quality | digest same?"""
import re, sys, glob, os, json
pat = re.compile(sys.argv[1]); out = os.path.expanduser("~/mq/out")
tags = sorted({m.group(0) for m in re.finditer(r"\b(rab[0-9a-z]+-[a-z0-9-]+)", open(os.path.expanduser("~/mq/results.txt")).read()) if pat.search(m.group(0))})
for t in tags:
    f = os.path.join(out, t + ".log")
    if not os.path.exists(f): print(t, "| no log"); continue
    arms = {}
    for ln in open(f, errors="replace"):
        m = re.match(r"AFC-AB def=([AB]) .*status=(\S+) median_ms=(\S+) digest=(\S+).*quality=(\{.*\})", ln)
        if m: arms[m.group(1)] = m.groups()[1:]
    a, b = arms.get("A"), arms.get("B")
    if not a or not b: print(t, "| missing arm", list(arms)); continue
    try: d = "%+.1f%%" % (100 * (float(b[1]) / float(a[1]) - 1))
    except Exception: d = "-"
    print("%s | A %s %s | B %s %s | %s | qA %s | qB %s | dig %s" % (t, a[0], a[1][:8], b[0], b[1][:8], d, a[3][:110], b[3][:110], "same" if a[2] == b[2] else "diff"))
