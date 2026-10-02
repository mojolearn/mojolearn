"""Summarize tree rows of a bench board: ours vs best opponent, ratio, quality.

Usage: python3 tools/aft_board_trees.py [board.json]
"""
import json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/board-0834/board/board.json")
d = json.load(open(path))
keys = ("forest", "tree", "gbdt", "/et/", "extra", "iforest", "isolation", "/rf", "shap")
for k, v in sorted(d["races"].items()):
    if k.startswith("neural/") or not any(s in k for s in keys):
        continue
    cells = v.get("cells", [])
    ours = {}
    opp = []
    for c in cells:
        arm = c.get("arm", "")
        ms = c.get("median_ms")
        q = c.get("quality") or {}
        q = {a: round(b, 5) for a, b in q.items() if isinstance(b, (int, float))}
        tag = "%s:%s:%s:%s" % (arm, c.get("status"), None if ms is None else round(ms), c.get("mode"))
        if arm.startswith("ours"):
            ours[arm] = (ms, q, tag, c.get("hash"))
        elif ms is not None and c.get("status") == "ok":
            opp.append((ms, arm, q))
    best = min(opp) if opp else None
    line = [k]
    for arm, (ms, q, tag, h) in sorted(ours.items()):
        r = (ms / best[0]) if (best and ms) else None
        line.append("%s ms=%s r=%s q=%s h=%s" % (arm, None if ms is None else round(ms), None if r is None else round(r, 2), q, h))
    if best:
        line.append("best=%s ms=%d q=%s" % (best[1], round(best[0]), best[2]))
    print(" | ".join(line))
