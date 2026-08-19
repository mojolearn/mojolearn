#!/usr/bin/env python3
"""Alternate the two benchmark processes and report medians with a verdict.

THE ALTERNATION IS THE POINT. This machine has been measured drifting two- to
threefold between thermal windows, so running all of ours and then all of
theirs compares thermal states as much as implementations. This runs
round-robin: ours, theirs, ours, theirs, and only pools samples from one
invocation of this script.

THE VERDICT MATTERS AS MUCH AS THE RATIO. If the two [min, max] ranges
overlap, the result is INDISTINGUISHABLE and the ratio is not reported as a
finding. A ratio without that test is an invitation to read noise as a
result, which this project has done before.
"""

import argparse
import collections
import pathlib
import statistics
import subprocess
import sys


def parse(text):
    out = collections.defaultdict(list)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ARM":
            out[parts[1]].append(float(parts[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mojo-bin", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sklearn-python", default=sys.executable)
    args = ap.parse_args()

    here = pathlib.Path(__file__).parent
    ours = collections.defaultdict(list)
    theirs = collections.defaultdict(list)

    for r in range(args.rounds):
        print(f"round {r + 1}/{args.rounds}: ours", flush=True)
        p = subprocess.run([args.mojo_bin], capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit("mojo arm failed")
        for k, v in parse(p.stdout).items():
            ours[k] += v

        print(f"round {r + 1}/{args.rounds}: scikit-learn", flush=True)
        p = subprocess.run(
            [args.sklearn_python, str(here / "bench_sklearn.py")],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-3000:])
            raise SystemExit("sklearn arm failed")
        for k, v in parse(p.stdout).items():
            theirs[k] += v

    print()
    print(f"{'arm':10} {'ours ms':>12} {'sklearn ms':>12} {'ratio':>8}  verdict")
    print("-" * 62)
    for arm in ["kmeans", "knn", "pca", "dbscan", "ols"]:
        a, b = ours.get(arm), theirs.get(arm)
        if not a or not b:
            print(f"{arm:10} {'--':>12} {'--':>12}")
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        overlap = not (max(a) < min(b) or max(b) < min(a))
        verdict = "INDISTINGUISHABLE" if overlap else (
            "ours faster" if ma < mb else "sklearn faster")
        print(f"{arm:10} {ma:12.2f} {mb:12.2f} {mb / ma:8.2f}x  {verdict}")
        print(f"{'':10} [{min(a):.2f}, {max(a):.2f}]  "
              f"[{min(b):.2f}, {max(b):.2f}]  n={len(a)}/{len(b)}")
    print()
    print("ratio > 1 means ours is faster. Ranges are min..max over all "
          "samples pooled from this invocation only.")


if __name__ == "__main__":
    main()
