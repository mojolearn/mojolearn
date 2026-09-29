#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE identity table of the fifteen-bit GEMM, from the boxes' own records.

    python3 tools/lowbit_int15/identity_table.py <box>=<digests>[,<digests>...]:<job output> ...

Lane lane/lowbit-int15. `<digests>` are files of `DIGEST <case> <plan>
<hex>` lines (the gate's, the simulation check's); `<job output>` is the
job's printed output, from which the arms' verdicts are read (`status.tsv`
lines and `reach:` lines). A box given as `<box>=-` has no run and its
column reads "not run yet". Nothing is computed here but equality of
strings that the boxes printed.

THE REFERENCE of a case is the digest its host oracle printed (for the
simulation's cases, the digest of the exported simulation product). The
table says, per box and per plan, on how many cases the plan's digest is
that box's reference, and whether every box's reference is the same.
"""
import re
import sys

PLANS = ["oracle", "flat", "pieces", "mma", "dispatch-codes", "dispatch-planes", "device", "from-f32", "simulation"]
WHAT = {
    "oracle": "host oracle (the reference)",
    "flat": "FLAT kernel, codes, Int64 sum",
    "pieces": "PIECES kernel, planes, three Int32 sums",
    "mma": "MMA, integer matrix unit, four products",
    "dispatch-codes": "entry point for codes",
    "dispatch-planes": "entry point for planes",
    "device": "float32 in: device quantizer, split, product",
    "from-f32": "float32 in: parallel quantizer, product",
    "simulation": "the PyTorch simulation's exported product",
}


def read_digests(paths):
    out = {}
    for path in paths:
        for line in open(path, errors="replace"):
            parts = line.split()
            if len(parts) == 4 and parts[0] == "DIGEST":
                out.setdefault(parts[1], {})[parts[2]] = parts[3]
    return out


def read_arms(path):
    """phase -> (exit code, held), and the reach lines."""
    arms, reach = {}, []
    for line in open(path, errors="replace"):
        m = re.match(r"^(\S+)\t(\d+)\t\S+\texpected=(pass|fail)\theld=(yes|no)$", line.rstrip("\n"))
        if m:
            arms[m.group(1)] = (int(m.group(2)), m.group(3), m.group(4))
        elif line.startswith("reach: "):
            reach.append(line.strip()[7:])
    return arms, reach


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    boxes, order = {}, []
    for a in argv:
        name, rest = a.split("=", 1)
        order.append(name)
        if rest == "-":
            boxes[name] = None
            continue
        digs, out = rest.split(":", 1)
        boxes[name] = (read_digests(digs.split(",")), read_arms(out))
    ran = [b for b in order if boxes[b] is not None]
    cases = sorted(set().union(*[set(boxes[b][0]) for b in ran]))
    red = False

    def ref(box, case):
        row = boxes[box][0].get(case, {})
        return row.get("oracle") if "oracle" in row else None

    print(f"Cases: {len(cases)} ({sum(1 for c in cases if c.startswith('sim-'))} of them the simulation's).")
    print()
    print("| plan | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for plan in PLANS:
        cells = []
        for b in order:
            if boxes[b] is None:
                cells.append("not run yet")
                continue
            have = [c for c in cases if plan in boxes[b][0].get(c, {})]
            if not have:
                cells.append("does not run here" if plan == "mma" else "no line")
                continue
            if plan == "oracle":
                cells.append(f"{len(have)} cases")
                continue
            same = [c for c in have if boxes[b][0][c][plan] == ref(b, c)]
            cells.append(f"{len(same)} of {len(have)} equal")
            if len(same) != len(have):
                red = True
        print(f"| {WHAT[plan]} | " + " | ".join(cells) + " |")
    # the references across boxes
    disagree = []
    for c in cases:
        seen = {ref(b, c) for b in ran if ref(b, c) is not None}
        if len(seen) != 1:
            disagree.append(c)
    missing = [(c, b) for c in cases for b in ran if c not in boxes[b][0]]
    print()
    print(f"Host oracles across boxes ({', '.join(ran)}): "
          + ("THE SAME DIGEST on every case" if not disagree else f"DIFFER on {len(disagree)} cases: {disagree[:5]}")
          + (f"; {len(missing)} case lines missing on a box" if missing else "") + ".")
    if disagree or missing:
        red = True
    print()
    phases = []
    for b in ran:
        for p in boxes[b][1][0]:
            if p not in phases:
                phases.append(p)
    print("| phase (expected) | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for p in phases:
        cells, want = [], ""
        for b in order:
            if boxes[b] is None:
                cells.append("not run yet")
                continue
            got = boxes[b][1][0].get(p)
            if got is None:
                cells.append("no line")
                red = True
                continue
            want = got[1]
            word = ("PASSED" if got[0] == 0 else "FAILED") + (", as expected" if got[2] == "yes" else ", NOT AS EXPECTED")
            cells.append(word)
            if got[2] != "yes":
                red = True
        print(f"| {p} (must {want}) | " + " | ".join(cells) + " |")
    print()
    for b in ran:
        bad = [r for r in boxes[b][1][1] if "as it must" not in r]
        good = [r for r in boxes[b][1][1] if "as it must" in r]
        print(f"{b}: {len(good)} gates seen failing under the arm that must fail them"
              + (f"; NOT AS EXPECTED: {bad}" if bad else ""))
        if bad:
            red = True
    print()
    print("verdict=" + ("RED" if red else "IDENTICAL on every box that ran: every plan's digest is the host oracle's, "
                        "the host oracles agree across boxes, and every arm failed the gates it must"))
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
