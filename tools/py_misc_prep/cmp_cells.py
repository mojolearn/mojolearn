#!/usr/bin/env python3
"""Compare two identity_break JSONs of the SAME column cell for cell (every
part a cell carries besides timings): the before route against the after
route. Exit 0 only when every cell of the named lanes met and was equal."""
import json, sys
SKIP = ("seconds", "time", "elapsed", "started", "wall", "timing", "error")
a, b = (json.load(open(p))["cells"] for p in sys.argv[1:3])
lanes = set(sys.argv[3].split(","))
keys = sorted(k for k in set(a) | set(b) if k.split("/")[0] in lanes)
bad, n = [], 0
for k in keys:
    ca, cb = a.get(k), b.get(k)
    if ca is None or cb is None or not ca.get("hashes"):
        bad.append((k, "missing or no hashes"))
        continue
    fa = {x: y for x, y in ca.items() if not any(s in x for s in SKIP)}
    fb = {x: y for x, y in cb.items() if not any(s in x for s in SKIP)}
    n += 1
    if fa != fb:
        bad.append((k, sorted(x for x in set(fa) | set(fb) if fa.get(x) != fb.get(x))))
for k, why in bad:
    print("DIFFER", k, why)
print(f"cells compared {n}, differing {len(bad)}")
print("RESULT", "SAME" if n and not bad else "DIFFER")
sys.exit(0 if n and not bad else 1)
