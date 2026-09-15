#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN soft clustering against cuML's own, on an NVIDIA box.

    PYTHONPATH=python <python with cuml> tools/hdbscan_soft_cuml_reference.py --out ref.json

For each fixture and configuration it fits mojolearn's HDBSCAN (identical
binding, prediction_data=True) and cuML's HDBSCAN on the same float32 rows.
When the two fits agree on labels_ (so the condensed trees select the same
clusters) it asks both libraries for membership_vector on held-out rows and
for all_points_membership_vectors, and records how far mojolearn's float32
formulation (DEVIATION 1616, hdbscan/impl/detail/soft_clustering.mojo) is
from cuML's: cells bitwise equal, max absolute and relative difference over
the cells where cuML is finite, rows whose argmax differs, cuML's NaN rows
and ours. It also records sha256 of mojolearn's outputs, so the box's hashes
can be compared with other columns. A fixture whose fits disagree is recorded
as such and not compared.
"""
import argparse
import hashlib
import json
import sys

import numpy as np


def _fixtures():
    rng = np.random.default_rng(20260915)
    centers = np.asarray([[0, 0, 0, 0], [6, 6, 0, 0], [0, 6, 6, 0], [6, 0, 0, 6]], np.float32)
    a = np.concatenate([c + rng.normal(0, 0.8, (500, 4)) for c in centers]).astype(np.float32)
    a = np.concatenate([a, rng.uniform(-3, 9, (200, 4))]).astype(np.float32)
    qa = (a[rng.choice(len(a), 400, replace=False)] + rng.normal(0, 0.3, (400, 4))).astype(np.float32)
    # ties: a coarse grid (distance ties) with every row duplicated (zero distances)
    g = np.round(np.concatenate([c + rng.normal(0, 0.8, (150, 4)) for c in centers]) * 2.0) / np.float32(2.0)
    b = np.concatenate([g, g]).astype(np.float32)
    qb = (np.round((b[rng.choice(len(b), 200, replace=False)] + rng.normal(0, 0.4, (200, 4))) * 2.0) / 2.0).astype(np.float32)
    return [("blobs", np.ascontiguousarray(a), np.ascontiguousarray(qa)),
            ("dupes-grid", np.ascontiguousarray(b), np.ascontiguousarray(qb))]


CONFIGS = [
    ("eom", dict(min_cluster_size=20, min_samples=10, cluster_selection_method="eom")),
    ("leaf", dict(min_cluster_size=15, min_samples=5, cluster_selection_method="leaf")),
]


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()


def _compare(ours, theirs):
    o = np.asarray(ours, np.float32)
    t = np.asarray(theirs, np.float32)
    out = dict(shape_ours=list(o.shape), shape_cuml=list(t.shape))
    if o.shape != t.shape:
        out["verdict"] = "SHAPE_DIFFERS"
        return out
    fin = np.isfinite(t)
    out["cells"] = int(o.size)
    out["cuml_nonfinite_cells"] = int((~fin).sum())
    out["cuml_nonfinite_rows"] = int((~fin).any(axis=1).sum()) if o.ndim == 2 else 0
    out["ours_nonfinite_cells"] = int((~np.isfinite(o)).sum())
    out["bitwise_equal_cells"] = int((o.view(np.uint32) == t.view(np.uint32)).sum())
    if fin.any():
        d = np.abs(o[fin].astype(np.float64) - t[fin].astype(np.float64))
        scale = np.maximum(np.abs(t[fin].astype(np.float64)), np.finfo(np.float32).tiny)
        out["max_abs_diff"] = float(d.max())
        out["max_rel_diff"] = float((d / scale).max())
        big = np.abs(t[fin]) >= 1e-6
        out["max_rel_diff_cells_ge_1e-6"] = float((d[big] / np.abs(t[fin][big])).max()) if big.any() else 0.0
        ulps = d / np.maximum(np.spacing(np.abs(t[fin])).astype(np.float64), 1e-45)
        out["max_ulps"] = float(ulps.max())
        out["median_ulps"] = float(np.median(ulps))
    if o.ndim == 2 and o.shape[1] > 0:
        rows = fin.all(axis=1)
        out["argmax_differs_rows"] = int((np.argmax(o[rows], axis=1) != np.argmax(t[rows], axis=1)).sum())
        so = o.sum(axis=1, dtype=np.float64)
        st = t.sum(axis=1, dtype=np.float64)
        out["row_sum_max_abs_diff"] = float(np.abs(so[rows] - st[rows]).max()) if rows.any() else 0.0
        out["cuml_rows_summing_to_1"] = int((np.abs(st[rows] - 1.0) <= 1e-5).sum())
        out["ours_rows_summing_to_1_where_cuml_does"] = int(
            (np.abs(so[rows][np.abs(st[rows] - 1.0) <= 1e-5] - 1.0) <= 1e-5).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cuml", action="store_true")
    args = ap.parse_args()
    import mojolearn
    from mojolearn import hdbscan as mh
    rec = dict(tool="tools/hdbscan_soft_cuml_reference.py", runs=[])
    if not args.skip_cuml:
        import cuml
        from cuml.cluster import hdbscan as ch
        rec["cuml_version"] = cuml.__version__
    rec["mojolearn_numeric_mode"] = mojolearn.HDBSCAN().numeric_mode_used()
    for fname, x, q in _fixtures():
        for cname, cfg in CONFIGS:
            run = dict(fixture=fname, config=cname, params=cfg, n_rows=int(x.shape[0]), n_queries=int(q.shape[0]))
            m = mojolearn.HDBSCAN(prediction_data=True, **cfg).fit(x)
            ol = np.asarray(m.labels_)
            omv = np.asarray(mh.membership_vector(m, q))
            oap = np.asarray(mh.all_points_membership_vectors(m))
            run["ours"] = dict(n_clusters=int(m.n_clusters_), n_exemplars=int(m._prediction_data["n_exemplars"]),
                               labels_sha256=hashlib.sha256(ol.astype(np.int32).tobytes()).hexdigest(),
                               membership_vector_sha256=_sha(omv), all_points_sha256=_sha(oap),
                               membership_nonfinite=int((~np.isfinite(omv)).sum()),
                               all_points_nonfinite=int((~np.isfinite(oap)).sum()))
            if not args.skip_cuml:
                c = ch.HDBSCAN(prediction_data=True, **cfg).fit(x)
                cl = np.asarray(c.labels_).astype(np.int32)
                run["cuml"] = dict(n_clusters=int(c.n_clusters_))
                run["labels_equal"] = bool(np.array_equal(cl, ol))
                run["labels_disagree_rows"] = int((cl != ol).sum())
                if run["labels_equal"] and m.n_clusters_ > 0:
                    cmv = np.asarray(ch.membership_vector(c, q), dtype=np.float32)
                    cap = np.asarray(ch.all_points_membership_vectors(c), dtype=np.float32)
                    run["membership_vector"] = _compare(omv, cmv)
                    run["all_points_membership_vectors"] = _compare(oap, cap)
                else:
                    run["membership_vector"] = "not compared: the fits disagree"
            rec["runs"].append(run)
            print(json.dumps(run), flush=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
