# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane linear-apple3: tables from an ab.py output (and a referee output).

    python3 tools/linear_apple3/tab.py <ab output> [<referee output>]

Timings: one line per (rows, case), every arm's fit seconds in run order.
Quality: one line per (case, seed), every arm's metrics, the host
reference column and the scikit-learn referee side by side.
"""
import collections
import re
import sys


def main():
    f = open(sys.argv[1]).read().splitlines()
    ref = open(sys.argv[2]).read().splitlines() if len(sys.argv) > 2 else []
    t = collections.defaultdict(list)
    arms, cases, rows = [], [], []
    for l in f:
        m = re.match(r"\[(\S+) (fast|identical) (\d+)\] XLSPEED (\S+) gpu fit=(\S+) predict=(\S+) (\S+)", l)
        if m:
            a, mode, r, c, fit, pred, dig = m.groups()
            a = a if mode == "fast" else a + ":" + mode
            t[(c, r, a)].append((float(fit), float(pred), dig))
            for lst, v in ((arms, a), (cases, c), (rows, r)):
                if v not in lst:
                    lst.append(v)
    for r in rows:
        print(f"== fit s, rows {r} (arms: {', '.join(arms)})")
        for c in cases:
            cells = []
            for a in arms:
                v = t.get((c, r, a))
                if v:
                    cells.append(f"{a} " + "/".join(f"{x[0]:.3f}" for x in v) + f" [{v[-1][2][:8]}]")
            if cells:
                print(f"  {c:13s} " + " | ".join(cells))
    q = collections.defaultdict(dict)
    for l in f + ref:
        m = re.match(r"QUAL3 (\S+) (\S+) (\w+) seed=(\d+) (.*) n_iter=(\S+) fit=(\S+)", l)
        if m:
            arm, case, col, seed, ms, ni, ft = m.groups()
            q[(case, int(seed))]["host" if col == "host" else arm] = (ms, ni)
        m = re.match(r"REF3 (\S+) seed=(\d+) (.*?)( n_iter=(\S+))?$", l)
        if m:
            q[(m.group(1), int(m.group(2)))]["sklearn"] = (m.group(3), m.group(5) or "?")
    if q:
        print("== quality (per case and seed)")
    for case, seed in sorted(q):
        row = q[(case, seed)]
        print(f"  {case} seed={seed}")
        for k in row:
            print(f"      {k:10s} {row[k][0]} n_iter={row[k][1]}")


if __name__ == "__main__":
    main()
