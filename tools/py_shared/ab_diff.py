#!/usr/bin/env python3
"""ab_diff.py BASE_DIR HEAD_DIR (every clean.*.json in HEAD_DIR must exist in BASE_DIR): every hash a lane JSON carries (clean.<lane>.<col>.json
from tools/algos_lane_check.py), base tree vs head tree, per column. Prints one line
per lane/column: SAME n (n hashes compared) or MOVED with the differing keys. Exit 1
when anything moved, a file is missing on one side, or nothing was compared."""
import json, sys
from pathlib import Path


def hashes(path):
    out = {}
    cells = json.loads(Path(path).read_text()).get("cells", {})
    for key in sorted(cells):
        c = cells[key]
        for i, parts in enumerate(c.get("parts") or []):
            for name in sorted(parts or {}):
                out[f"{key} parts[{i}].{name}"] = json.dumps(parts[name], sort_keys=True)
        for k in sorted(c):
            v = c[k]
            if k == "parts":
                continue
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                out[f"{key} {k}"] = ",".join(v)
            elif isinstance(v, str) and len(v) >= 16 and all(ch in "0123456789abcdef" for ch in v):
                out[f"{key} {k}"] = v
    return out


def main():
    base, head = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    names = sorted(p.name for p in head.glob("clean.*.json"))
    if not names:
        print("NOTHING COMPARED: no clean.*.json in the head directory")
        return 1
    for name in names:
        b, h = base / name, head / name
        if not b.is_file() or not h.is_file():
            print(f"MISSING {name}: base={b.is_file()} head={h.is_file()}")
            bad += 1
            continue
        hb, hh = hashes(b), hashes(h)
        keys = sorted(set(hb) | set(hh))
        moved = [k for k in keys if hb.get(k) != hh.get(k)]
        if not keys:
            print(f"NOTHING COMPARED {name}")
            bad += 1
        elif moved:
            print(f"MOVED {name}: {len(moved)} of {len(keys)}: " + "; ".join(moved[:8]))
            bad += 1
        else:
            print(f"SAME {name}: {len(keys)} hashes")
    print("AB RESULT: " + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
