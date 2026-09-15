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


F32 = np.float32
F32MAX = np.finfo(np.float32).max


def _walk_heights(point_parent, fallback, parents, iic, lambdas, selected):
    """merge_height_kernel (kernels/soft_clustering.cuh:14-98), in Python."""
    nr, ns = len(point_parent), len(selected)
    h = np.empty((nr, ns), F32)
    for r in range(nr):
        for c in range(ns):
            left, right = int(point_parent[r]), int(selected[c])
            tl = tr = False
            last = 0
            while left != right:
                if left > right:
                    tl, last, left = True, left, int(parents[iic[left]])
                else:
                    tr, last, right = True, right, int(parents[iic[right]])
            h[r, c] = lambdas[iic[last]] if (tl and tr) else fallback[r]
    return h


def _norm32(v):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return (v / v.sum(axis=1, dtype=F32, keepdims=True)).astype(F32)


def _dist_membership_f64_seams(x, rows, pd):
    """soft_clustering.cuh:46-149 with cuML's precision: float32 distances and
    sums, value_t(1.0 / val) through a double."""
    ns = len(pd["selected_clusters"])
    ex = x[np.asarray(pd["exemplar_idx"])[: pd["n_exemplars"]]].astype(F32)
    off = np.asarray(pd["exemplar_label_offsets"])
    rn = (rows * rows).sum(axis=1, dtype=F32)
    en = (ex * ex).sum(axis=1, dtype=F32)
    d2 = (rn[:, None] + en[None, :] - F32(2.0) * (rows @ ex.T)).astype(F32)
    d = np.sqrt(np.maximum(d2, F32(0.0))).astype(F32)
    mins = np.stack([d[:, off[c]:off[c + 1]].min(axis=1) for c in range(ns)], axis=1).astype(F32)
    with np.errstate(divide="ignore"):
        inv = (1.0 / mins.astype(np.float64)).astype(F32)
    dmv = np.where(mins > 0, inv, F32(F32MAX / F32(ns))).astype(F32)
    return _norm32(dmv)


def _softmax32(o):
    with np.errstate(invalid="ignore", over="ignore"):
        return np.exp((o - o.max(axis=1, keepdims=True)).astype(F32)).astype(F32)


def transcribe_all_points(x, pd):
    """cuML all_points_membership_vectors (soft_clustering.cuh:385-482) in
    numpy at cuML's precision, from OUR fitted prediction data."""
    par = np.asarray(pd["parents"]); lam = np.asarray(pd["lambdas"], F32)
    iic = np.asarray(pd["index_into_children"]); deaths = np.asarray(pd["deaths"], F32)
    sel = np.asarray(pd["selected_clusters"]); nl = x.shape[0]
    rows = x.astype(F32)
    dmv = _dist_membership_f64_seams(x, rows, pd)
    row_lam = lam[iic[np.arange(nl)]]
    h = _walk_heights(par[iic[np.arange(nl)]], row_lam, par, iic, lam, sel)
    vec = deaths[par[iic[np.arange(nl)]] - nl]
    with np.errstate(divide="ignore", over="ignore"):
        o = np.exp(-(vec.astype(np.float64)[:, None] + 1e-8) / h.astype(np.float64)).astype(F32)
    o = _norm32(_softmax32(o))
    am = np.argmax(h, axis=1)
    ml = np.maximum(row_lam, deaths[sel[am] - nl])
    prob = (h[np.arange(nl), am] / ml).astype(F32)
    mv = _norm32((dmv * o).astype(F32))
    return (mv * prob[:, None]).astype(F32)


def transcribe_membership_vector(x, q, pd, core, min_samples):
    """cuML membership_vector (soft_clustering.cuh:501-627) in numpy at cuML's
    precision, from OUR fitted prediction data and core distances. The
    neighborhood is a stable float64 sort, so on a distance tie it can name a
    different neighbor than the pinned (distance, index) order."""
    par = np.asarray(pd["parents"]); lam = np.asarray(pd["lambdas"], F32)
    iic = np.asarray(pd["index_into_children"]); deaths = np.asarray(pd["deaths"], F32)
    sel = np.asarray(pd["selected_clusters"]); nl = x.shape[0]
    q = q.astype(F32); core = np.asarray(core, F32)
    dmv = _dist_membership_f64_seams(x, q, pd)
    k = (min_samples - 1) * 2
    d = np.sqrt(np.maximum(((q.astype(np.float64)[:, None, :] - x.astype(np.float64)[None, :, :]) ** 2).sum(-1), 0.0))
    nb = np.argsort(d, axis=1, kind="stable")[:, :k]
    nd = np.take_along_axis(d, nb, axis=1).astype(F32)
    pcore = nd[:, min_samples - 1]
    mr = np.maximum(np.maximum(pcore[:, None], core[nb]), nd)
    ind = nb[np.arange(len(q)), np.argmin(mr, axis=1)]
    mmr = mr.min(axis=1)
    with np.errstate(divide="ignore"):
        pl = np.where(mmr > 0, (F32(1.0) / mmr).astype(F32), F32MAX).astype(F32)
    pl = np.minimum(pl, lam[iic[ind]])
    h = _walk_heights(par[iic[ind]], pl, par, iic, lam, sel)
    vec = deaths[par[iic[ind]] - nl]
    den = (vec[:, None] - h).astype(F32)
    den = np.where(den <= 0, F32(1e-8), den).astype(F32)
    with np.errstate(over="ignore", invalid="ignore"):
        o = (vec[:, None] / den).astype(F32)
    o = _norm32(_softmax32(o))
    with np.errstate(over="ignore", invalid="ignore"):
        c = (o.astype(np.float64) ** 2 * dmv.astype(np.float64) ** 0.5).astype(F32)
    c = _norm32(c)
    am = np.argmax(h, axis=1)
    ml = np.maximum(pl, deaths[sel[am] - nl]).astype(np.float64) + 1e-8
    prob = (h[np.arange(len(q)), am].astype(np.float64) / ml).astype(F32)
    return (c * prob[:, None]).astype(F32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cuml", action="store_true")
    args = ap.parse_args()
    import mojolearn
    from mojolearn import hdbscan as mh
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        return _main(args, mojolearn, mh)


def _main(args, mojolearn, mh):
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
            try:
                ms = cfg.get("min_samples") or cfg["min_cluster_size"]
                run["vs_float64_transcription"] = dict(
                    membership_vector=_compare(omv, transcribe_membership_vector(x, q, m._prediction_data, m.core_distances_, ms)),
                    all_points_membership_vectors=_compare(oap, transcribe_all_points(x, m._prediction_data)))
            except Exception as exc:  # a transcription failure is recorded, never silent
                run["vs_float64_transcription"] = f"failed: {type(exc).__name__}: {exc}"
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
