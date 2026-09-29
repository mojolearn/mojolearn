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
        if depth > 2 or hbin is None or not os.path.exists(hbin):
            continue
        feats = sorted(borders)
        cand = []   # (feature, border index, form) per host column
        for f in feats:
            for bi in range(len(borders[f])):
                cand.append((f, bi))
        hist = np.fromfile(hbin, dtype="<f4")
        nbf = len(cand)
        devh = hist.reshape(-1, 2, nbf)
        # host per leaf: rows above / at-or-below each border
        above = np.zeros((n_leaves, 2, nbf))
        for ci, (f, bi) in enumerate(cand):
            a = x[:, f] > borders[f][bi]
            for lf in range(n_leaves):
                m = a & (leaf == lf)
                above[lf, 0, ci] = m.sum()
                above[lf, 1, ci] = (y[m] - 0.5).sum()
        below = np.stack([cnt[:, None] - above[:, 0], g[:, None] - above[:, 1]], axis=1)
        if depth == 0:
            # the column map from the (trusted) root histogram
            key = {}
            for ci in range(nbf):
                for form, arr in (("above", above), ("below", below)):
                    key.setdefault((round(arr[0, 0, ci], 1), round(arr[0, 1, ci], 1)), []).append((ci, form))
            colmap = []
            unmatched = 0
            for dc in range(nbf):
                k = (round(float(devh[0, 0, dc]), 1), round(float(devh[0, 1, dc]), 1))
                c = key.get(k)
                colmap.append(c[0] if c else None)
                unmatched += c is None
            globals()["COLMAP"] = colmap
            print("  depth 0: device columns with no host match:", unmatched, "of", nbf)
            forms = {}
            for c in colmap:
                if c:
                    forms[c[1]] = forms.get(c[1], 0) + 1
            print("  forms:", forms, "first columns:", [(cand[c[0]], c[1]) if c else None for c in colmap[:12]])
            continue
        colmap = globals().get("COLMAP")
        for slot in range(devh.shape[0]):
            best = None
            for hl in range(n_leaves):
                bad_feats = {}
                bad = 0
                for dc in range(nbf):
                    c = colmap[dc]
                    if c is None:
                        continue
                    arr = above if c[1] == "above" else below
                    for st in range(2):
                        if abs(devh[slot, st, dc] - arr[hl, st, c[0]]) > 1e-3 * (1 + abs(arr[hl, st, c[0]])):
                            bad += 1
                            f = cand[c[0]][0]
                            bad_feats[f] = bad_feats.get(f, 0) + 1
                if best is None or bad < best[0]:
                    best = (bad, hl, bad_feats)
            print("  depth", depth, "device slot", slot, "best host leaf", best[1],
                  "mismatched cells", best[0], "by feature", best[2])


if __name__ == "__main__":
    main()
