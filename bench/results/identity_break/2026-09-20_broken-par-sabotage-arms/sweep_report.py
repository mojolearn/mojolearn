"""Classify every par-* cell across the clean and core-only-sabotage sweeps.

Four buckets:
  STRUCTURAL   refused clean AND refused sabotaged -- no CPU host route
  CAUGHT       clean STABLE, sabotaged moved/divergent -- a working control
  INERT        clean STABLE, sabotaged STABLE at the same hash -- arm does nothing here
  BROKEN       clean STABLE, sabotaged REFUSED -- the shape this lane is about
"""
import json, sys, os
ROOT = "/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-a42cd6627cd863bdb"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import verification_matrix as matrix

clean = json.load(open(sys.argv[1]))
sab = json.load(open(sys.argv[2]))
buckets = {"STRUCTURAL": [], "CAUGHT": [], "INERT": [], "BROKEN": [], "OTHER": []}
for key in sorted(clean["cells"]):
    c, s = clean["cells"][key], sab["cells"].get(key)
    lane = key.split("/")[0]
    cv = c.get("verdict")
    sv = s.get("verdict") if s else None
    if cv == "REFUSED" and sv == "REFUSED":
        buckets["STRUCTURAL"].append((lane, (c.get("error") or "").strip().splitlines()[-1][:110]))
    elif cv == "STABLE" and sv == "REFUSED":
        buckets["BROKEN"].append((lane, (s.get("error") or "").strip().splitlines()[-1][:160]))
    elif cv == "STABLE" and sv in ("MOVED", "DIVERGENT"):
        parts = [p for p in ("train", "infer", "model", "batch")
                 if matrix.negative_control_moves(s, c, p)]
        buckets["CAUGHT"].append((lane, "%s, credited: %s" % (sv, ", ".join(parts) or "NONE")))
    elif cv == "STABLE" and sv == "STABLE":
        same = matrix.stable_digest(c, "train") == matrix.stable_digest(s, "train")
        parts = [p for p in ("train", "infer", "model", "batch")
                 if matrix.negative_control_moves(s, c, p)]
        (buckets["INERT"] if same and not parts else buckets["CAUGHT"]).append(
            (lane, "STABLE, credited: %s" % (", ".join(parts) or "NONE")))
    else:
        buckets["OTHER"].append((lane, "%s -> %s" % (cv, sv)))

for name in ("BROKEN", "CAUGHT", "INERT", "STRUCTURAL", "OTHER"):
    rows = buckets[name]
    print("\n%s  (%d)" % (name, len(rows)))
    for lane, note in rows:
        print("   %-28s %s" % (lane, note))
print("\ntotal cells: %d" % len(clean["cells"]))
