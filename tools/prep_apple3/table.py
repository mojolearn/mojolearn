#!/usr/bin/env python3
"""lane prep-apple3: one table per numeric mode from a bench/x_prep_ab.sh job's stdout.

    python3 tools/prep_apple3/table.py <stdout> [--mode fast] [--arms a,b,...] [--md]

Seconds per arm (XPSPEED total), and whether every arm's digest equals the first arm's that has one.
"""
import argparse
import collections
import re


def parse(path):
    mode = None
    out = collections.OrderedDict()
    arms = []
    for line in open(path, errors="replace"):
        m = re.match(r"ARM (\S+) XPINFO mode=(\S+)", line)
        if m:
            mode = m.group(2)
            continue
        m = re.match(r"ARM (\S+) XPSPEED (\S+) (\S+) (\d+) (\S+) (\S+) (\d+) (\S+)", line)
        if m:
            arm, ds, case, n, tot, bind, progs, dig = m.groups()
            out.setdefault((mode, ds, case, int(n)), {})[arm] = (float(tot), float(bind), dig)
            if (mode, arm) not in arms:
                arms.append((mode, arm))
    return out, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--arms", default="")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows, arms = parse(a.path)
    names = [x for x in a.arms.split(",") if x] or [arm for mode, arm in arms if mode == a.mode]
    sep = " | " if a.md else "  "
    head = ["dataset", "case", "rows"] + names + ["digests"]
    print(("| " if a.md else "") + sep.join(head) + (" |" if a.md else ""))
    if a.md:
        print("|" + "---|" * len(head))
    for (mode, ds, case, n), v in rows.items():
        if mode != a.mode or not any(x in v for x in names):
            continue
        digs = [v[x][2] for x in names if x in v]
        cells = [ds, case, str(n)] + [f"{v[x][0]:.3f}" if x in v else "-" for x in names]
        cells.append("same" if len(set(digs)) == 1 else "DIFFER " + ",".join(d[:6] for d in digs))
        print(("| " if a.md else "") + sep.join(cells) + (" |" if a.md else ""))


if __name__ == "__main__":
    main()
