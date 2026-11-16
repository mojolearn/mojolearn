# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.HDBSCAN` (workstream D,
2026-09-14): a gate on the WIRING (the params list, the returned arrays'
dtypes and shapes, the float64 query conversion, every refusal by name).
The arithmetic is gated by `pixi run check-hdbscan`; the fit, approximate
predict and soft-membership relations of the typed entries moved to the
native check `checks/native_tests/hdbscan_surface.mojo`
(`pixi run check-native-cpu hdbscan_surface`).

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
    return np.ascontiguousarray(np.concatenate([a, b]).astype(np.float32))


def arm_fit(rep):
    x = _blobs()
    m = HDBSCAN(min_cluster_size=8).fit(x)
    lab = np.asarray(m.labels_)
    rep.check("FIT", lab.shape == (120,) and lab.dtype == np.int32, "labels_ int32 (n,)", (lab.shape, lab.dtype))
    cd = np.asarray(m.core_distances_)
    rep.check("FIT", cd.shape == (120,) and cd.dtype == np.float32, "core_distances_ float32 (n,)", (cd.shape, cd.dtype))
    rep.check("FIT", np.asarray(HDBSCAN(min_cluster_size=8).fit_predict(x)).shape == (120,), "fit_predict")


def arm_refusals(rep):
    x = _blobs()
    rep.raises("REFUSE", ValueError, "euclidean", "metric='manhattan' by name", HDBSCAN(metric="manhattan").fit, x)
    rep.raises("REFUSE", ValueError, "cluster_selection_method", "an unknown selection method", HDBSCAN(cluster_selection_method="x").fit, x)
    rep.raises("REFUSE", TypeError, "min_samples", "min_samples as a float", HDBSCAN(min_samples=2.5).fit, x)
    rep.raises("REFUSE", ValueError, "cluster_selection_epsilon", "a negative cluster_selection_epsilon by name", HDBSCAN(cluster_selection_epsilon=-0.5).fit, x)
    rep.raises("REFUSE", Exception, "", "min_cluster_size=1, refused on the Mojo host", HDBSCAN(min_cluster_size=1).fit, x)
    rep.raises("REFUSE", Exception, "n_rows", "a single row, refused on the Mojo host", HDBSCAN().fit, x[:1])
    bad = x.copy(); bad[5, 0] = np.float32("nan")
    rep.raises("REFUSE", Exception, "", "a NaN cell, refused on the Mojo host (DEVIATION 1607)", HDBSCAN().fit, bad)
    m = HDBSCAN(min_cluster_size=8).fit(x)
    pr = np.asarray(m.probabilities_)
    rep.check("FIT", pr.shape == (120,) and bool(np.all((pr >= 0) & (pr <= 1)))
              and bool(np.all(pr[np.asarray(m.labels_) == -1] == 0)), "probabilities_ in [0, 1], 0 on noise (DEVIATION 5116)")


def arm_predict(rep):
    from mojolearn import hdbscan as hd
    x = _blobs()
    plain = HDBSCAN(min_cluster_size=8).fit(x)
    m = HDBSCAN(min_cluster_size=8, prediction_data=True).fit(x)
    lab, prob = hd.approximate_predict(m, x)
    lab, prob = np.asarray(lab), np.asarray(prob)
    rep.check("PREDICT", lab.dtype == np.int32 and lab.shape == (120,), "labels int32 (n,), the dtype of labels_", (lab.dtype, lab.shape))
    rep.check("PREDICT", prob.dtype == np.float32 and prob.shape == (120,), "probabilities float32 (n,)", (prob.dtype, prob.shape))

    l64, p64 = hd.approximate_predict(m, x[:17].astype(np.float64))
    rep.check("PREDICT", np.asarray(l64).dtype == np.int32 and np.array_equal(np.asarray(l64), lab[:17])
              and np.array_equal(np.asarray(p64).view(np.uint32), prob[:17].view(np.uint32)),
              "a float64 query converts to float32 and returns the float32 query's bytes")

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
    soft_clustering.cuh:385-627, DEVIATION 1616): returned dtypes and
    shapes, refusals, the no-cluster (n, 0) shape the door returns before
    any native call, and the float64 query conversion."""
    from mojolearn import hdbscan as hd
    x = _blobs()
    plain = HDBSCAN(min_cluster_size=8).fit(x)
    m = HDBSCAN(min_cluster_size=8, prediction_data=True).fit(x)
    ns = m.n_clusters_

    mv = np.asarray(hd.membership_vector(m, x))
    rep.check("SOFT", mv.dtype == np.float32 and mv.shape == (120, ns), "membership_vector float32 (n, n_clusters_)", (mv.dtype, mv.shape))
    ap = np.asarray(hd.all_points_membership_vectors(m))
    rep.check("SOFT", ap.dtype == np.float32 and ap.shape == (120, ns), "all_points_membership_vectors float32 (n, n_clusters_)", (ap.dtype, ap.shape))

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
    # (a no-op scope since CPU training became public); on a GPU binding the scope changes nothing.
    return _run(rep, out)


def _run(rep, out):
    return run("test_hdbscan_surface", [("FIT", arm_fit), ("REFUSE", arm_refusals), ("PREDICT", arm_predict),
                                        ("SOFT", arm_soft), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
