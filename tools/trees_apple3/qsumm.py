#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Pair the LGQ / MQ quality records of a trees-apple3 run by seed.

    python3 tools/trees_apple3/qsumm.py <stdout> [before-tag] [after-tag]

One line per (kind, dataset, metric): the per-seed values of both tags, the
mean of each, the mean difference after - before and how many seeds moved
each way. Lower is better for rmse and logloss, higher for acc and auc."""
import collections
import json
import re
import sys


def main(argv):
    path = argv[1]
    before = argv[2] if len(argv) > 2 else "before"
    after = argv[3] if len(argv) > 3 else "after"
    rows = collections.defaultdict(dict)
    fits = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        m = re.match(r"(?:\[\w+\] )?(LGQ|MQ) (\{.*\})", line.rstrip())
        if not m:
            continue
        j = json.loads(m[2])
        kind = j.get("est") or ("gbdt-" + j.get("policy", "Lossguide"))
        for metric in ("rmse", "logloss", "acc", "auc"):
            if metric in j:
                rows[(kind, j["ds"], metric)][(j["tag"], j["seed"])] = j[metric]
        rows[(kind, j["ds"], "digest")][(j["tag"], j["seed"])] = j.get("digest")
        fits[(kind, j["ds"], j["tag"])].append(j["fit_s"])
    for key in sorted(rows):
        kind, ds, metric = key
        vals = rows[key]
        seeds = sorted({s for (t, s) in vals if t == before} & {s for (t, s) in vals if t == after})
        if not seeds:
            continue
        b = [vals[(before, s)] for s in seeds]
        a = [vals[(after, s)] for s in seeds]
        if metric == "digest":
            same = sum(1 for x, y in zip(b, a) if x == y)
            print("%-16s %-11s digest   equal on %d of %d seeds" % (kind, ds, same, len(seeds)))
            continue
        mb, ma = sum(b) / len(b), sum(a) / len(a)
        better_low = metric in ("rmse", "logloss")
        wins = sum(1 for x, y in zip(b, a) if (y < x if better_low else y > x))
        ties = sum(1 for x, y in zip(b, a) if y == x)
        print("%-16s %-11s %-8s %s=%.6f %s=%.6f diff=%+.6f (%+.4f%%)  after better on %d, equal on %d, worse on %d of %d"
              % (kind, ds, metric, before, mb, after, ma, ma - mb, 100.0 * (ma - mb) / mb if mb else 0.0,
                 wins, ties, len(seeds) - wins - ties, len(seeds)))
    for key in sorted(fits):
        v = fits[key]
        print("fit_s %-16s %-11s %-8s mean %.2f  %s" % (key[0], key[1], key[2], sum(v) / len(v), v))


if __name__ == "__main__":
    main(sys.argv)
