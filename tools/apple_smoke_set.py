#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SMALL APPLE SET FOR ROUTINE WORK (2026-09-16).

Andrew: "for our regular test we test much much much less of mac, maybe hardly
any at all, 3-5 tests, rely on nvidia and amd and trust mojo, mac is a big
problem".

The Apple GPU is one machine, it cannot be rented, it runs one job at a time,
and a full column is hours. NVIDIA and AMD are rented, parallel and cost a few
dollars. So routine Apple work is a SMOKE SET of a few lanes, and the full
column is a release step and nothing else.

DERIVED, NOT HAND-PICKED. The set is a greedy cover over the lane-to-source map
`tools/lane_select.py` already derives by import and call-graph closure. Picking
lanes by taste produces a list that looks sensible, drifts as the tree changes,
and nobody notices; re-deriving means the set tracks the code. Run this to see
the current answer and the coverage it buys.

WHAT IS COUNTED. Device `.mojo` files only: host routes, `bindings/` glue and
`checks/` are excluded, because this set exists to exercise METAL and a host
file says nothing about it. `par-*` lanes are dropped, since the record excludes
them and a multi-device driver cannot state its claim on a one-GPU box anyway
(`tools/lane_applicability.py`).

WHAT THIS DOES NOT CLAIM, and the limit is the point rather than a footnote.
FILE COVERAGE IS NOT KERNEL COVERAGE AND NEITHER IS DEFECT COVERAGE. A lane that
reaches a file does not exercise every kernel in it, and a kernel that runs does
not mean the defect you care about could have expressed itself in the shapes
that lane uses. Today's own example: two gemm lanes reach `identical_gemm` and
fire it 63 times between them, and on the base fixture every one of those calls
has a one-float stub workspace that is never written or read, so a
workspace-read-before-write defect COULD NOT ARISE there. The smoke set is a
tripwire for gross breakage on the vendor we cannot afford to run. It is not
evidence of Apple identity, and nothing that needs such evidence may cite it.
"""
import argparse, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def derive(n, include_par=False):
    import lane_select as ls
    m = ls.lane_sources()
    srcs = m[0] if isinstance(m, tuple) else m
    lanes = {k: set(v) for k, v in srcs.items() if include_par or not k.startswith("par-")}
    if not lanes:
        raise SystemExit("apple_smoke_set: the lane map is empty; REFUSING rather than returning a set")

    def device(fs):
        return {f for f in fs if f.endswith(".mojo") and "/host/" not in f
                and not f.endswith("_host.mojo") and not f.startswith("bindings/")
                and "/checks/" not in f}

    cov = {k: device(v) for k, v in lanes.items()}
    universe = set().union(*cov.values())
    if not universe:
        raise SystemExit("apple_smoke_set: no device sources attributed; REFUSING")
    chosen, covered = [], set()
    for _ in range(n):
        best = max(cov, key=lambda k: (len(cov[k] - covered), -len(cov[k])))
        gain = len(cov[best] - covered)
        if gain == 0:
            break
        covered |= cov[best]
        chosen.append((best, gain, len(covered), len(universe)))
    return chosen, len(universe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=5, help="how many lanes (Andrew asked for 3 to 5)")
    ap.add_argument("--lanes-only", action="store_true", help="print just the comma list, for a command line")
    ap.add_argument("--include-par", action="store_true", help="do not drop par-* (they cannot state their claim on one GPU)")
    a = ap.parse_args()
    chosen, total = derive(a.n, a.include_par)
    if a.lanes_only:
        print(",".join(c[0] for c in chosen))
        return 0
    print(f"# apple smoke set, {len(chosen)} lanes, derived over {total} device .mojo files")
    for i, (lane, gain, cum, tot) in enumerate(chosen, 1):
        print(f"{i}. {lane:<30} +{gain:4d} new    {cum}/{tot}  {100*cum//tot}%")
    print("#")
    print("# A TRIPWIRE, NOT EVIDENCE. File coverage is not kernel coverage and")
    print("# neither is defect coverage. Nothing that needs Apple identity evidence")
    print("# may cite this set; that needs the full column or a targeted sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
