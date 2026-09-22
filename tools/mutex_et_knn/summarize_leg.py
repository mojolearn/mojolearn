#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read one ET/kNN leg and print the table plus the rate it COULD have detected.

A null is reported with its bound or not at all. A zero is written up as a NULL with
what it excludes, never as a clearance.

    python3 tools/mutex_et_knn/summarize_leg.py <et_knn.json> [et_knn_mutex.log]
"""
import json, math, re, sys

FOREST_LOW, FOREST_HIGH = 0.043, 0.053


def p0(p, n):
    return math.exp(n * math.log(1.0 - p)) if n else 1.0


def bound95(n):
    """Rule of three: with 0 of n, the 95% upper bound on the rate is 3/n."""
    return 3.0 / n if n else float("inf")


def detectable(n, power=0.95):
    """The smallest rate this n would have caught with `power` probability.
    1 - (1-p)^n >= power  =>  p >= 1 - (1-power)^(1/n)."""
    return 1.0 - (1.0 - power) ** (1.0 / n) if n else float("inf")


def main():
    doc = json.load(open(sys.argv[1]))
    print("## ExtraTrees cells\n")
    print("| arm | cols | bpn | contends | fits | moved | distinct | rate | "
          "P(0) at the forest's 4.3% | 95% bound | rate detectable at 95% power |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in doc.get("cells", []):
        n, m = c["runs"], c["moved"]
        rate = ("%.2f%%" % (100.0 * m / n)) if n else "n/a"
        p = "%.2g" % p0(FOREST_LOW, n) if m == 0 else "-"
        b = "%.2f%%" % (100 * bound95(n)) if m == 0 else "-"
        d = "%.2f%%" % (100 * detectable(n)) if n else "-"
        print("| %s | %d | %d | %s | %d | %d | %d | %s | %s | %s | %s |" % (
            c["arm"], c["cols"], c["bpn_expected"],
            "YES" if c["contends"] else "no (one claimant)",
            n, m, c["distinct"], rate, p, b, d))
        if c.get("error"):
            print("|   ^ ERROR | | | | | | | %s | | | |" % c["error"][:120])
    for note in doc.get("notes", []):
        print("\nnote: %s" % note)

    if len(sys.argv) > 2:
        log = open(sys.argv[2], errors="replace").read()
        print("\n## fused kNN arms\n")
        for line in log.splitlines():
            if re.match(r"^(ARM |CONTROL |POSITIVE_CONTROL |grid_x |ETKNN-gfx942 section|"
                        r"ETKNN-gfx942 knn_|ETKNN-gfx942 et_arms_gate|ETKNN-gfx942 treesbuild)",
                        line):
                print("    " + line)
        print("\n## the gate lines, which decide whether anything above is readable\n")
        for line in log.splitlines():
            if ("EXPECT" in line or "REFUSED" in line or "VOID" in line
                    or "SAME PROGRAM" in line or "DIFFER" in line):
                print("    " + line)


if __name__ == "__main__":
    main()
