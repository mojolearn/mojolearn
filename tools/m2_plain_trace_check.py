#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Recompute a traced SymmetricTree fit's first-tree per-leaf totals on the
host and print them beside the device's (lane/ordered-speed, 2026-09-29: the
M2 Pro's Plain fit differs from the CPU column from depth 1 on).

    MOJOLEARN_IDENTITY_TRACE=/tmp/t.trace MOJOLEARN_IDENTITY_TRACE_DUMP=depth0 \
      ORD_PROFILE_DUMP_MODEL=/tmp/m.txt python bench/speed/ordered_profile.py \
      --boosting Plain --trees 1 --rows 100000 --ours-ab class_weights=None
    python tools/m2_plain_trace_check.py /tmp/t.trace /tmp/m.txt 100000

Prints, for depth 0..3 of tree 0, the device `pstats` records
([leaf][stat] float32) and the host's per-leaf row counts and sums of
(y - 0.5) under the model's first splits (the first tree's gradient at the
start point 0 for Logloss; weights 1).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speed_gbdt_arm as spec  # noqa: E402


def parse_model(path):
    borders, splits = {}, []
    in_tree0 = False
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if w[0] == "feature":
            f = int(w[1])
            k = w.index("borders")
            n = int(w[k + 1])
            borders[f] = [float(x.split("/")[0]) for x in w[k + 2:k + 2 + n]]
            # the bits are authoritative
            borders[f] = [np.frombuffer(bytes.fromhex(x.split("/")[1])[::-1],
                                        dtype="<f4")[0] for x in w[k + 2:k + 2 + n]]
        elif w[0] == "tree":
            in_tree0 = int(w[1]) == 0
        elif w[0] == "split" and in_tree0:
            splits.append((int(w[3]), int(w[4])))
    return borders, splits


def main():
    trace, model, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = spec.load_dataset("taxi", "shipped", rows)
    x = np.asarray(d.X_train, dtype=np.float32)
    y = np.asarray(d.y_train, dtype=np.float64)
    borders, splits = parse_model(model)
    print("splits of tree 0:", splits)
    recs = []
    for line in open(trace):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            recs.append(p)
    seen = set()
    for depth in range(4):
        tag_end = ".depth%02d.pstats" % depth
        rec = next((r for r in recs if r[1].endswith(tag_end)
                    and r[1] not in seen), None)
        if rec is None:
            print("no record", tag_end)
            continue
        seen.add(rec[1])
        binp = "%s.%s.%s.bin" % (trace, rec[0], rec[1])
        dev = np.fromfile(binp, dtype="<f4") if os.path.exists(binp) else None
        # host: leaf id = bits of the first `depth` splits (bit k = split k)
        leaf = np.zeros(len(y), dtype=np.int64)
        for k in range(depth):
            f, b = splits[k]
            right = x[:, f] > borders[f][b]
            leaf |= right.astype(np.int64) << k
        n_leaves = 1 << depth
        cnt = np.bincount(leaf, minlength=n_leaves)
        g = np.bincount(leaf, weights=y - 0.5, minlength=n_leaves)
        print("depth", depth, "record", rec[1], "count", rec[3])
        print("  device:", None if dev is None else dev.tolist())
        print("  host counts:", cnt.tolist())
        print("  host sum(y-0.5):", [round(v, 3) for v in g.tolist()])
        hrec = next((r for r in recs if r[1].endswith(".depth%02d.hist" % depth)), None)
        hbin = None if hrec is None else "%s.%s.%s.bin" % (trace, hrec[0], hrec[1])
        if depth > 1 or hbin is None or not os.path.exists(hbin):
            continue
        hist = np.fromfile(hbin, dtype="<f4")
        feats = sorted(borders)
        nbf = sum(len(borders[f]) for f in feats)
        # host: per leaf, per binFeature (feature-major, border order), rows
        # strictly above the border (count, sum g)
        host = np.zeros((n_leaves, 2, nbf))
        col = 0
        for f in feats:
            for bb in borders[f]:
                above = x[:, f] > bb
                for lf in range(n_leaves):
                    m = above & (leaf == lf)
                    host[lf, 0, col] = m.sum()
                    host[lf, 1, col] = (y[m] - 0.5).sum()
                col += 1
        devh = hist.reshape(-1, 2, nbf)
        print("  hist device shape", devh.shape, "binFeatures", nbf)
        for lf in range(devh.shape[0]):
            for st in range(2):
                dv = devh[lf, st]
                best = None
                for hl in range(n_leaves):
                    for form in ("above", "below"):
                        ref = host[hl, st] if form == "above" else (
                            (cnt[hl] if st == 0 else g[hl]) - host[hl, st])
                        bad = int(np.sum(np.abs(dv - ref) > 1e-3 * (1 + np.abs(ref))))
                        if best is None or bad < best[0]:
                            best = (bad, hl, form)
                col = 0
                worst = []
                for f in feats:
                    for bi in range(len(borders[f])):
                        worst.append((f, bi))
                        col += 1
                bad, hl, form = best
                ref = host[hl, st] if form == "above" else ((cnt[hl] if st == 0 else g[hl]) - host[hl, st])
                idx = np.nonzero(np.abs(dv - ref) > 1e-3 * (1 + np.abs(ref)))[0]
                print("  dev leaf", lf, "stat", st, "best host leaf", hl, form,
                      "mismatched cells", bad, "first:",
                      [(worst[i], float(dv[i]), float(ref[i])) for i in idx[:6]])


if __name__ == "__main__":
    main()
