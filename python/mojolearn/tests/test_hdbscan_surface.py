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


def arm_provenance(rep):
    rep.check("PROVENANCE", "HDBSCAN" in mojolearn.__all__ and "hdbscan" in mojolearn.__all__, "HDBSCAN and mojolearn.hdbscan exported")
    rep.check("PROVENANCE", HDBSCAN().numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_hdbscan", "build_hdbscan.sh")
    rep = Report("test_hdbscan_surface")
    return run("test_hdbscan_surface", [("FIT", arm_fit), ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
