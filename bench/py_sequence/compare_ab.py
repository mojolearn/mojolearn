# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The before/after table of bench/py_sequence/ab.py: python3 compare_ab.py <OUT>
reads ab_{BASE,NEW}_{gpu,cpu}.json and prints, per case and arm, BASE and NEW
seconds and whether the digests are equal (and GPU == CPU on NEW)."""
import json
import sys
from pathlib import Path


def load(p):
    return json.loads(p.read_text()) if p.is_file() else {}


def main():
    out = Path(sys.argv[1])
    R = {(t, a): load(out / f"ab_{t}_{a}.json") for t in ("BASE", "NEW") for a in ("gpu", "cpu")}
    keys = sorted(set().union(*[set(v) for v in R.values()]))
    print("| case | arm | BASE s | NEW s | BASE digest | NEW digest | BASE == NEW |")
    print("|---|---|---|---|---|---|---|")
    bad = 0
    for k in keys:
        for a in ("gpu", "cpu"):
            b, n = R[("BASE", a)].get(k), R[("NEW", a)].get(k)
            if b is None and n is None:
                continue
            bs = f"{b['s']:.3f}" if b else "-"
            ns = f"{n['s']:.3f}" if n else "-"
            bd, nd = (b or {}).get("digest", "-"), (n or {}).get("digest", "-")
            eq = "-" if (b is None or n is None) else ("EQUAL" if bd == nd else "DIFFER")
            if eq == "DIFFER":
                bad += 1
            print(f"| {k} | {a} | {bs} | {ns} | {bd} | {nd} | {eq} |")
    xs = []
    for k in keys:
        g, c = R[("NEW", "gpu")].get(k), R[("NEW", "cpu")].get(k)
        if g and c:
            xs.append((k, g["digest"] == c["digest"]))
    print("NEW GPU == CPU: " + ", ".join(f"{k}={'EQUAL' if e else 'DIFFER'}" for k, e in xs))
    print(f"RESULT BASE==NEW digests: {bad} differ")


if __name__ == "__main__":
    main()
