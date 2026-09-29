#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compare the DIGEST lines of the fifteen-bit gate across boxes.

Lane lane/lowbit-int15, 2026-09-29. `gemm/checks/gemm_int15_check.mojo` and
`gemm_int15_sim_check.mojo` print one line per case and plan,

    DIGEST <case> <plan> <hex>

a 64-bit FNV-1a of the output words. A case has ONE right digest: every
plan of every box, and every box's host oracle, must print it. This script
reads each box's lines and says, per case, whether they did.

    python3 tools/lowbit_int15/digests.py h100=<file> mi325x=<file> m3ultra=<file> m2pro=<file>
    python3 tools/lowbit_int15/digests.py --expect-disagree clean=<file> sabotage=<file>

A file is a gate log or a digests.tsv; lines that are not digest lines are
skipped. Exit 0 when every case agrees on every box that printed it and
every case was printed by every box; 1 otherwise. A comparison of nothing is
a failure. `--expect-disagree` inverts the verdict for a sabotage file: exit
0 only when the two files share cases and the device digests of EVERY shared
case differ.
"""
import sys


def read(path):
    out = {}
    for line in open(path, errors="replace"):
        parts = line.split()
        if len(parts) == 4 and parts[0] == "DIGEST":
            out.setdefault(parts[1], {})[parts[2]] = parts[3]
    return out


def main(argv):
    expect_disagree = "--expect-disagree" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2 or any("=" not in a for a in args):
        print(__doc__)
        return 2
    boxes = {}
    for a in args:
        name, path = a.split("=", 1)
        boxes[name] = read(path)
        if not boxes[name]:
            print(f"REFUSED: {path} ({name}) holds no DIGEST line; nothing to compare")
            return 1
    names = list(boxes)
    cases = sorted(set().union(*[set(b) for b in boxes.values()]))
    if expect_disagree:
        clean, bad = boxes[names[0]], boxes[names[1]]
        shared = sorted(set(clean) & set(bad))
        if not shared:
            print("REFUSED: the two files share no case")
            return 1
        same = []
        for c in shared:
            for plan, d in bad[c].items():
                if plan != "oracle" and plan != "simulation" and clean[c].get(plan) == d:
                    same.append((c, plan))
        print(f"{len(shared)} shared cases; device digests equal to the clean run's: {len(same)}")
        for c, plan in same[:10]:
            print(f"  NOT MOVED: {c} {plan}")
        return 1 if same else 0
    missing = []
    disagree = []
    plans = {n: set() for n in names}
    for c in cases:
        seen = {}
        for n in names:
            if c not in boxes[n]:
                missing.append((c, n))
                continue
            for plan, d in boxes[n][c].items():
                plans[n].add(plan)
                seen.setdefault(d, []).append(f"{n}:{plan}")
        if len(seen) > 1:
            disagree.append((c, seen))
    print(f"boxes: {', '.join(names)}")
    for n in names:
        print(f"  {n}: {len(boxes[n])} cases, plans {', '.join(sorted(plans[n]))}, "
              f"{sum(len(v) for v in boxes[n].values())} digests")
    print(f"cases: {len(cases)}; missing on a box: {len(missing)}; disagreeing: {len(disagree)}")
    for c, n in missing[:10]:
        print(f"  MISSING: {c} on {n}")
    for c, seen in disagree[:10]:
        print(f"  DISAGREE: {c}")
        for d, who in seen.items():
            print(f"    {d}: {', '.join(who)}")
    ok = not missing and not disagree and cases
    print("verdict=" + ("AGREE: every plan of every box and every host oracle printed the same digest for every case"
                        if ok else "DISAGREE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
