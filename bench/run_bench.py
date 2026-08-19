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

Every `ARM <name> <ms>` line either process prints is reported, so the same
runner covers the fixed-size pair (bench_main / bench_sklearn.py) and the
scaling pair (scaling_main / scaling_sklearn.py); pick the scikit-learn side
with --sklearn-script. --markdown appends the table, with the environment
record, to a results file.
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


def arm_order(name):
    # kmeans, knn, pca, dbscan, ols first at their fixed sizes, then
    # name-grouped scaling arms in numeric size order (knn@20000 < knn@100000).
    base, _, size = name.partition("@")
    return (base, int(size) if size.isdigit() else -1)


def rows(ours, theirs):
    for arm in sorted(set(ours) | set(theirs), key=arm_order):
        a, b = ours.get(arm), theirs.get(arm)
        if not a or not b:
            yield (arm, a or b, None, None, "ONE-SIDED")
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        overlap = not (max(a) < min(b) or max(b) < min(a))
        verdict = "INDISTINGUISHABLE" if overlap else (
            "ours faster" if ma < mb else "sklearn faster")
        yield (arm, a, b, mb / ma, verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mojo-bin", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sklearn-python", default=sys.executable)
    ap.add_argument("--sklearn-script", default="bench_sklearn.py",
                    help="which scikit-learn arm to alternate with, "
                         "relative to bench/")
    ap.add_argument("--markdown", default=None,
                    help="append the table + environment record to this file")
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
            [args.sklearn_python, str(here / args.sklearn_script)],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-3000:])
            raise SystemExit("sklearn arm failed")
        for k, v in parse(p.stdout).items():
            theirs[k] += v

    print()
    print(f"{'arm':16} {'ours ms':>12} {'sklearn ms':>12} {'ratio':>8}  verdict")
    print("-" * 68)
    md = ["| arm | ours ms | sklearn ms | ratio | verdict | ours [min, max] "
          "| sklearn [min, max] | n |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, a, b, ratio, verdict in rows(ours, theirs):
        if ratio is None:
            print(f"{arm:16} {'--':>12} {'--':>12}           {verdict}")
            md.append(f"| {arm} | -- | -- | -- | {verdict} | | | |")
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"{arm:16} {ma:12.2f} {mb:12.2f} {ratio:8.2f}x  {verdict}")
        print(f"{'':16} [{min(a):.2f}, {max(a):.2f}]  "
              f"[{min(b):.2f}, {max(b):.2f}]  n={len(a)}/{len(b)}")
        md.append(
            f"| {arm} | {ma:.2f} | {mb:.2f} | {ratio:.2f}x | {verdict} "
            f"| [{min(a):.2f}, {max(a):.2f}] | [{min(b):.2f}, {max(b):.2f}] "
            f"| {len(a)}/{len(b)} |")
    print()
    print("ratio > 1 means ours is faster. Ranges are min..max over all "
          "samples pooled from this invocation only.")

    if args.markdown:
        env = subprocess.run(
            ["bash", str(here.parent / "tools" / "record_environment.sh")],
            capture_output=True, text=True).stdout
        with open(args.markdown, "a") as f:
            f.write(f"\n## {args.sklearn_script} vs {args.mojo_bin} "
                    f"({args.rounds} rounds, arms alternated per round)\n\n")
            f.write("\n".join(md))
            f.write("\n\nratio > 1 means ours is faster; INDISTINGUISHABLE "
                    "means the min..max ranges overlap and the ratio is not "
                    "a finding.\n\n```\n" + env + "```\n")


if __name__ == "__main__":
    main()
