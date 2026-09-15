# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.HDBSCAN` (workstream D,
2026-09-14): a gate on the WIRING (the params list, the four integers, the
labels and core distances landing in the caller's arrays, every refusal by
name). The arithmetic is gated by `pixi run check-hdbscan`.

    cd python && python3 -m mojolearn.tests.test_hdbscan_surface

Exit 2 naming `bindings/build_hdbscan.sh` when unbuilt. Written on one
Apple M4 with no built binary in the worktree. Its first GPU run is still
owed: on the Hot Aisle MI300X at 2b2f568b0 (2026-09-14) the binding did not
build, the compiler crashed in AMDGPU instruction selection (the gfx942
banner in `hdbscan/impl/detail/stabilities.mojo`).
"""
import sys

import numpy as np

import mojolearn
from mojolearn import HDBSCAN
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((60, 2), dtype=np.float32) * 0.5
    b = rng.random((60, 2), dtype=np.float32) * 0.5 + np.float32(6.0)
    x = np.ascontiguousarray(np.concatenate([a, b]).astype(np.float32))
    planted = np.concatenate([np.zeros(60, np.int32), np.ones(60, np.int32)])
    return x, planted


def arm_fit(rep):
    x, planted = _blobs()
    m = HDBSCAN(min_cluster_size=8).fit(x)
    lab = np.asarray(m.labels_)
    rep.check("FIT", lab.shape == (120,) and lab.dtype == np.int32, "labels_ int32 (n,)", (lab.shape, lab.dtype))
    rep.check("FIT", m.n_clusters_ == 2, "two planted blobs give n_clusters_ == 2", m.n_clusters_)
    kept = lab >= 0
    agree = max(np.mean(lab[kept] == planted[kept]), np.mean(lab[kept] == 1 - planted[kept])) if kept.any() else 0.0
    rep.check("FIT", agree == 1.0, "every non-noise label agrees with its blob (up to relabeling)", agree)
    rep.check("FIT", m.n_outliers_ == int(np.sum(lab == -1)), "n_outliers_ counts the -1 labels", (m.n_outliers_, int(np.sum(lab == -1))))
    cd = np.asarray(m.core_distances_)
    rep.check("FIT", cd.shape == (120,) and np.all(cd > 0) and np.isfinite(cd).all(), "core_distances_ positive and finite")
    rep.check("FIT", m.n_boruvka_rounds_ >= 1 and m.n_condensed_clusters_ >= 1, "Boruvka rounds and condensed cluster count carried", (m.n_boruvka_rounds_, m.n_condensed_clusters_))
    m2 = HDBSCAN(min_cluster_size=8).fit(x)
    same = np.array_equal(lab, np.asarray(m2.labels_)) and np.array_equal(cd.view(np.uint32), np.asarray(m2.core_distances_).view(np.uint32))
    if mode() == "identical":
        rep.check("FIT", same, "two fits of the same input agree bit for bit on this box")
    else:
        rep.report_only("FIT", same, "two fits of the same input")
    leaf = HDBSCAN(min_cluster_size=8, cluster_selection_method="leaf").fit(x)
    rep.check("FIT", leaf.n_clusters_ >= 1, "cluster_selection_method='leaf' runs", leaf.n_clusters_)
    rep.check("FIT", np.asarray(HDBSCAN(min_cluster_size=8).fit_predict(x)).shape == (120,), "fit_predict")


def arm_refusals(rep):
    x, _ = _blobs()
    rep.raises("REFUSE", ValueError, "euclidean", "metric='manhattan' by name", HDBSCAN(metric="manhattan").fit, x)
    rep.raises("REFUSE", ValueError, "cluster_selection_method", "an unknown selection method", HDBSCAN(cluster_selection_method="x").fit, x)
    rep.raises("REFUSE", TypeError, "min_samples", "min_samples as a float", HDBSCAN(min_samples=2.5).fit, x)
    rep.raises("REFUSE", Exception, "", "cluster_selection_epsilon != 0, refused on the Mojo host", HDBSCAN(cluster_selection_epsilon=0.5).fit, x)
    rep.raises("REFUSE", Exception, "", "min_cluster_size=1, refused on the Mojo host", HDBSCAN(min_cluster_size=1).fit, x)
    rep.raises("REFUSE", Exception, "n_rows", "a single row, refused on the Mojo host", HDBSCAN().fit, x[:1])
    bad = x.copy(); bad[5, 0] = np.float32("nan")
    rep.raises("REFUSE", Exception, "", "a NaN cell, refused on the Mojo host (DEVIATION 1607)", HDBSCAN().fit, bad)
    m = HDBSCAN(min_cluster_size=8).fit(x)
    rep.raises("REFUSE", AttributeError, "1610", "probabilities_ refused by name", lambda: m.probabilities_)


def _dupes(seed=2):
    """Two planted blobs with every row DUPLICATED and one grid column, so
    k-NN distances tie many ways (the identity harness's `dupes` idea)."""
    rng = np.random.default_rng(seed)
    a = np.round(rng.random((40, 2)) * 8.0) / np.float32(16.0)
    b = np.round(rng.random((40, 2)) * 8.0) / np.float32(16.0) + np.float32(5.0)
    x = np.concatenate([a, a, b, b]).astype(np.float32)
    return np.ascontiguousarray(x)


def arm_predict(rep):
    from mojolearn import hdbscan as hd
    x, planted = _blobs()
    plain = HDBSCAN(min_cluster_size=8).fit(x)
    m = HDBSCAN(min_cluster_size=8, prediction_data=True).fit(x)
    same_fit = (np.array_equal(np.asarray(plain.labels_), np.asarray(m.labels_))
                and np.array_equal(np.asarray(plain.core_distances_).view(np.uint32), np.asarray(m.core_distances_).view(np.uint32)))
    rep.check("PREDICT", same_fit, "prediction_data=True fits the same labels_ and core_distances_ bytes")

    lab, prob = hd.approximate_predict(m, x)
    lab, prob = np.asarray(lab), np.asarray(prob)
    fitted = np.asarray(m.labels_)
    rep.check("PREDICT", lab.dtype == np.int32 and lab.shape == (120,), "labels int32 (n,), the dtype of labels_", (lab.dtype, lab.shape))
    rep.check("PREDICT", prob.dtype == np.float32 and prob.shape == (120,), "probabilities float32 (n,)", (prob.dtype, prob.shape))
    rep.check("PREDICT", bool(np.all((prob >= 0) & (prob <= 1))), "probabilities in [0, 1]")
    rep.check("PREDICT", bool(np.all(prob[lab == -1] == 0)), "a -1 label has probability 0 (kernels/predict.cuh:81-83)")
    # cuML documents the labels as those of the original clustering; on the
    # training rows of two planted blobs every row predicts its fitted label.
    agree = float(np.mean(lab == fitted))
    rep.check("PREDICT", agree == 1.0, "every training row predicts its fitted label on two planted blobs", agree)

    far = np.asarray([[1000.0, -1000.0], [-500.0, 700.0]], dtype=np.float32)
    fl, fp = hd.approximate_predict(m, far)
    rep.check("PREDICT", np.array_equal(np.asarray(fl), np.asarray([-1, -1], np.int32)) and np.all(np.asarray(fp) == 0),
              "a point far from every cluster is noise with probability 0", (list(np.asarray(fl)), list(np.asarray(fp))))

    pd = m._prediction_data
    ex = np.asarray(pd["exemplar_idx"])[: pd["n_exemplars"]]
    rep.check("PREDICT", pd["n_exemplars"] >= m.n_clusters_ and ex.size > 0, "every selected cluster has an exemplar", pd["n_exemplars"])
    xl, xp = hd.approximate_predict(m, x[ex])
    rep.check("PREDICT", np.array_equal(np.asarray(xl), fitted[ex]) and bool(np.all(np.asarray(xp) > 0)),
              "a query exactly at an exemplar predicts the exemplar's label with a positive probability")

    l64, p64 = hd.approximate_predict(m, x[:17].astype(np.float64))
    rep.check("PREDICT", np.asarray(l64).dtype == np.int32 and np.array_equal(np.asarray(l64), lab[:17])
              and np.array_equal(np.asarray(p64).view(np.uint32), prob[:17].view(np.uint32)),
              "a float64 query converts to float32 and returns the float32 query's bytes")

    l2, p2 = hd.approximate_predict(m, x)
    one = [hd.approximate_predict(m, x[i:i + 1]) for i in range(5)]
    rows_alone = all(int(np.asarray(a)[0]) == int(lab[i]) and np.asarray(b).view(np.uint32)[0] == prob.view(np.uint32)[i]
                     for i, (a, b) in enumerate(one))
    stable = np.array_equal(np.asarray(l2), lab) and np.array_equal(np.asarray(p2).view(np.uint32), prob.view(np.uint32))
    if mode() == "identical":
        rep.check("PREDICT", stable and rows_alone, "two calls agree bit for bit, and a row alone equals its row in the batch")
    else:
        rep.report_only("PREDICT", stable and rows_alone, "two calls, and rows alone")

    dx = _dupes()
    dm = HDBSCAN(min_cluster_size=6, min_samples=4, prediction_data=True).fit(dx)
    d1 = hd.approximate_predict(dm, dx)
    d2 = hd.approximate_predict(dm, dx[::-1].copy())
    rev = (np.array_equal(np.asarray(d1[0])[::-1], np.asarray(d2[0]))
           and np.array_equal(np.asarray(d1[1])[::-1].view(np.uint32), np.asarray(d2[1]).view(np.uint32)))
    if mode() == "identical":
        rep.check("PREDICT", rev, "duplicated rows and tied distances: reversing the query order reverses the answer bytes")
    else:
        rep.report_only("PREDICT", rev, "duplicated rows, reversed query order")

    rep.raises("PREDICT", ValueError, "Prediction data", "approximate_predict without prediction_data=True refused by name",
               hd.approximate_predict, plain, x)
    rep.raises("PREDICT", ValueError, "features", "a query with the wrong feature count", hd.approximate_predict, m, x[:, :1])
    bad = x[:3].copy(); bad[1, 1] = np.float32("nan")
    rep.raises("PREDICT", Exception, "1607", "a NaN query refused by name (DEVIATION 1607)", hd.approximate_predict, m, bad)
    m1 = HDBSCAN(min_cluster_size=8, min_samples=1, prediction_data=True).fit(x)
    rep.raises("PREDICT", Exception, "min_samples", "min_samples=1 has an empty prediction neighborhood, refused by name",
               hd.approximate_predict, m1, x[:4])
    rep.raises("PREDICT", TypeError, "prediction_data", "prediction_data must be a bool", HDBSCAN(prediction_data=1).fit, x)


def _bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def arm_soft(rep):
    """membership_vector and all_points_membership_vectors (cuML
    soft_clustering.cuh:385-627, DEVIATION 1616)."""
    from mojolearn import hdbscan as hd
    x, planted = _blobs()
    plain = HDBSCAN(min_cluster_size=8).fit(x)
    m = HDBSCAN(min_cluster_size=8, prediction_data=True).fit(x)
    ns = m.n_clusters_
    lab = np.asarray(m.labels_)

    mv = np.asarray(hd.membership_vector(m, x))
    rep.check("SOFT", mv.dtype == np.float32 and mv.shape == (120, ns), "membership_vector float32 (n, n_clusters_)", (mv.dtype, mv.shape))
    rep.check("SOFT", bool(np.isfinite(mv).all() and (mv >= 0).all()), "membership_vector finite and non-negative")
    s = mv.sum(axis=1, dtype=np.float64)
    rep.check("SOFT", bool(np.all(s <= 1.0 + 1e-5)), "a row sums to at most 1 (the joint distribution)", float(s.max()))
    kept = lab >= 0
    rep.check("SOFT", bool(np.all(np.argmax(mv[kept], axis=1) == lab[kept])),
              "on two planted blobs a clustered training row is most likely its own label")

    ap = np.asarray(hd.all_points_membership_vectors(m))
    rep.check("SOFT", ap.dtype == np.float32 and ap.shape == (120, ns), "all_points_membership_vectors float32 (n, n_clusters_)", (ap.dtype, ap.shape))
    rep.check("SOFT", bool(np.isfinite(ap).all() and (ap >= 0).all()), "all_points_membership_vectors finite and non-negative")
    pd = m._prediction_data
    ex = np.asarray(pd["exemplar_idx"])[: pd["n_exemplars"]]
    exs = ap[ex].sum(axis=1, dtype=np.float64)
    # An exemplar's lambda is its cluster's death, so its probability of
    # being in some cluster is exactly 1 and its row sums to 1, as cuML's does.
    rep.check("SOFT", bool(np.all(np.abs(exs - 1.0) <= 1e-5)), "every exemplar's all-points row sums to 1",
              (float(exs.min()), float(exs.max())))
    aps = ap.sum(axis=1, dtype=np.float64)
    rep.check("SOFT", bool(np.all(aps <= 1.0 + 1e-5)), "every all-points row sums to at most 1", float(aps.max()))
    rep.check("SOFT", bool(np.all(np.argmax(ap[kept], axis=1) == lab[kept])),
              "every clustered training row's largest all-points membership is its own label")

    far = np.asarray([[1000.0, -1000.0], [-500.0, 700.0]], dtype=np.float32)
    fv = np.asarray(hd.membership_vector(m, far))
    fsum = fv.sum(axis=1, dtype=np.float64)
    # A far point keeps a small floor: where the tree walk climbs both sides
    # the merge height is the split's lambda, not the query's
    # (kernels/soft_clustering.cuh:90-92), so its row is small, not zero.
    rep.check("SOFT", bool(np.isfinite(fv).all() and np.all(fsum < 0.05) and np.all(fsum < s[kept].min())),
              "a point far from every cluster has a row below 0.05 and below every clustered training row (noise)",
              (list(fsum), float(s[kept].min())))

    one = [np.asarray(hd.membership_vector(m, x[i:i + 1])) for i in range(5)]
    rows_alone = all(np.array_equal(_bits(one[i][0]), _bits(mv[i])) for i in range(5))
    again = np.array_equal(_bits(hd.membership_vector(m, x)), _bits(mv))
    small = np.array_equal(_bits(hd.membership_vector(m, x, batch_size=7)), _bits(mv))
    ap_small = np.array_equal(_bits(hd.all_points_membership_vectors(m, batch_size=13)), _bits(ap))
    if mode() == "identical":
        rep.check("SOFT", again and rows_alone, "two calls agree bit for bit, and a row alone equals its row in the batch")
        rep.check("SOFT", small and ap_small, "batch_size changes no byte of either call")
    else:
        rep.report_only("SOFT", again and rows_alone and small and ap_small, "repeat, rows alone, batch_size")

    dx = _dupes()
    dm = HDBSCAN(min_cluster_size=6, min_samples=4, prediction_data=True).fit(dx)
    d1 = np.asarray(hd.membership_vector(dm, dx))
    d2 = np.asarray(hd.membership_vector(dm, dx[::-1].copy()))
    dap = np.asarray(hd.all_points_membership_vectors(dm))
    rep.check("SOFT", bool(np.isfinite(d1).all() and np.isfinite(dap).all()),
              "duplicated rows (zero distances, FLT_MAX lambdas) give finite rows where cuML's overflow to NaN (DEVIATION 1616)")
    rep.check("SOFT", bool(np.all(d1.sum(axis=1) <= 1.0 + 1e-5) and np.all(dap.sum(axis=1) <= 1.0 + 1e-5)),
              "duplicated rows: every row sums to at most 1")
    rev = np.array_equal(_bits(d1[::-1]), _bits(d2))
    if mode() == "identical":
        rep.check("SOFT", rev, "duplicated rows and tied distances: reversing the query order reverses the membership bytes")
    else:
        rep.report_only("SOFT", rev, "duplicated rows, reversed query order")

    rep.raises("SOFT", ValueError, "Prediction data", "membership_vector without prediction_data=True refused by name",
               hd.membership_vector, plain, x)
    rep.raises("SOFT", ValueError, "Prediction data", "all_points_membership_vectors without prediction_data=True refused by name",
               hd.all_points_membership_vectors, plain)
    rep.raises("SOFT", ValueError, "batch_size", "batch_size=0 refused", hd.membership_vector, m, x, 0)
    rep.raises("SOFT", ValueError, "batch_size", "all points batch_size=-1 refused", hd.all_points_membership_vectors, m, -1)
    rep.raises("SOFT", ValueError, "features", "a query with the wrong feature count", hd.membership_vector, m, x[:, :1])
    bad = x[:3].copy(); bad[1, 1] = np.float32("nan")
    rep.raises("SOFT", Exception, "1607", "a NaN query refused by name (DEVIATION 1607)", hd.membership_vector, m, bad)
    m1 = HDBSCAN(min_cluster_size=8, min_samples=1, prediction_data=True).fit(x)
    rep.raises("SOFT", Exception, "min_samples", "min_samples=1 has an empty prediction neighborhood, refused by name",
               hd.membership_vector, m1, x[:4])

    rng = np.random.default_rng(5)
    u = rng.random((40, 2), dtype=np.float32)
    zm = HDBSCAN(min_cluster_size=25, prediction_data=True).fit(u)
    if zm.n_clusters_ == 0:
        zv = np.asarray(hd.membership_vector(zm, u[:3]))
        za = np.asarray(hd.all_points_membership_vectors(zm))
        rep.check("SOFT", zv.shape == (3, 0) and za.shape == (40, 0), "no cluster: (n, 0) arrays", (zv.shape, za.shape))
    else:
        rep.report_only("SOFT", False, f"the no-cluster fixture found {zm.n_clusters_} clusters; the (n, 0) shape is unchecked")

    # Last, because the float64 cast goes through the base binding.
    f64 = np.array_equal(_bits(hd.membership_vector(m, x[:17].astype(np.float64))), _bits(mv[:17]))
    if mode() == "identical":
        rep.check("SOFT", f64, "a float64 query converts to float32 and returns the float32 query's bytes")
    else:
        rep.report_only("SOFT", f64, "float64 query")


def arm_provenance(rep):
    rep.check("PROVENANCE", "HDBSCAN" in mojolearn.__all__ and "hdbscan" in mojolearn.__all__, "HDBSCAN and mojolearn.hdbscan exported")
    rep.check("PROVENANCE", HDBSCAN().numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_hdbscan", "build_hdbscan.sh")
    rep = Report("test_hdbscan_surface")
    # A CPU-only install fits only inside the verifier's scope
    # (_cpu_reference.py); on a GPU binding the scope changes nothing.
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        return _run(rep, out)


def _run(rep, out):
    return run("test_hdbscan_surface", [("FIT", arm_fit), ("REFUSE", arm_refusals), ("PREDICT", arm_predict),
                                        ("SOFT", arm_soft), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
