#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fold the `timing <name> <ms> ms` lines of a timed byte LM run into a
per-shard table (lane/amd-step-time, 2026-09-24).

    python3 tools/amd_step_timing_summary.py <log> [--skip-shards N] [--tsv out.tsv]

A shard is one `byte_gradient_device`: its forward ends with ONE
`envelope.blocks_forward` line, so the number of those lines is the number of
shards timed. The first `--skip-shards` (default 1: setup and first-touch
costs) are dropped by position: a line belongs to the shard whose
`envelope.blocks_forward` precedes it... except the forward's own block lines,
which precede it; so shards are cut at `step.upload_inputs` (the first line of
every forward) instead. Per name: calls a shard, ms a shard (mean over the
kept shards), share of the summed `envelope.*`/`step.*` top level. `gemm.*`
lines (phase-timer builds) are a CROSS-CUT of the block lines, reported apart.
"""
import argparse
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^timing (\S+) ([0-9.eE+-]+) ms\s*$")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--skip-shards", type=int, default=1)
    ap.add_argument("--tsv", default="")
    args = ap.parse_args(argv)
    shards = []
    cur = None
    for raw in open(args.log, errors="replace"):
        m = LINE.match(raw.strip())
        if not m:
            continue
        name, ms = m.group(1), float(m.group(2))
        if name == "step.upload_inputs":
            cur = defaultdict(lambda: [0, 0.0])
            shards.append(cur)
        if cur is None:
            continue
        cur[name][0] += 1
        cur[name][1] += ms
    kept = shards[args.skip_shards:]
    if not kept:
        print("no shard after skipping %d of %d" % (args.skip_shards, len(shards)))
        return 1
    names = sorted({n for s in kept for n in s})
    rows = []
    for n in names:
        calls = sum(s[n][0] for s in kept) / len(kept)
        ms = sum(s[n][1] for s in kept) / len(kept)
        rows.append((n, calls, ms))
    top = sum(ms for n, _, ms in rows if n.startswith("step.") or n.startswith("envelope."))
    gemm = sum(ms for n, _, ms in rows if n.startswith("gemm."))
    out = ["name\tcalls_per_shard\tms_per_shard\tshare_of_top"]
    for n, calls, ms in sorted(rows, key=lambda r: -r[2]):
        share = ms / top if top and not n.startswith("gemm.") else float("nan")
        out.append("%s\t%.1f\t%.3f\t%.4f" % (n, calls, ms, share))
    out.append("# shards timed %d, kept %d; top level (step.* + envelope.*) %.3f ms a shard; gemm.* %.3f ms a shard"
               % (len(shards), len(kept), top, gemm))
    text = "\n".join(out)
    print(text)
    if args.tsv:
        open(args.tsv, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
