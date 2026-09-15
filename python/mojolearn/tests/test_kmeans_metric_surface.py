# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the two KMeans arms routed through `mojolearn.cluster`
(workstream D, 2026-09-14): `metric` (cuVS's `DistanceType`) and
`oversampling_factor` (the classic sequential k-means++ arm at 0.0). The
routing must leave the default's bits alone, must reach the Mojo refusal
of the cosine metric BY NAME, and must reach the classic arm.

    cd python && python3 -m mojolearn.tests.test_kmeans_metric_surface

Exit 2 naming `bindings/build.sh` when the core binding is unbuilt.
"""
import sys

import numpy as np

from mojolearn import KMeans
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _x(seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.random((512, 4), dtype=np.float32) * 4.0).astype(np.float32))


def _wide(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((512, 4)).astype(np.float32) * np.logspace(-2, 2, 4).astype(np.float32)
    return np.ascontiguousarray(x.astype(np.float32))


def _label_not_argmin(x, centers, labels):
    """Rows whose label is not the float64 argmin of the squared distance
    to the returned centers, beyond a relative 1e-4 of float32 round-off."""
    xd = np.asarray(x, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    d2 = ((xd[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    lab = np.asarray(labels).astype(np.int64)
    chosen = d2[np.arange(len(xd)), lab]
    best = d2.min(1)
    return int(np.sum(chosen > best + 1e-4 * np.maximum(best, 1.0)))


def _bits_same(a, b):
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def arm_metric(rep):
    x = _x()
    base = KMeans(n_clusters=4, random_state=3).fit(x)
    euc = KMeans(n_clusters=4, random_state=3, metric="euclidean").fit(x)
    l2e = KMeans(n_clusters=4, random_state=3, metric="l2_expanded").fit(x)
    same = _bits_same(base.cluster_centers_, euc.cluster_centers_) and _bits_same(base.cluster_centers_, l2e.cluster_centers_) and base.inertia_ == euc.inertia_ == l2e.inertia_
    if mode() == "identical":
        rep.check("METRIC", same, "the default, 'euclidean' and 'l2_expanded' are one arm: centers and inertia bit for bit")
    else:
        rep.report_only("METRIC", same, "default vs 'euclidean' vs 'l2_expanded'")
    sq = KMeans(n_clusters=4, random_state=3, metric="l2_sqrt_expanded").fit(x)
    rep.check("METRIC", np.asarray(sq.cluster_centers_).shape == (4, 4) and np.isfinite(sq.inertia_), "'l2_sqrt_expanded' fits", sq.inertia_)
    rep.report_only("METRIC", sq.inertia_ == base.inertia_, "l2_sqrt_expanded inertia vs the squared arm (expected to MOVE: the root is taken)")
    # DEVIATION 2716: `labels_` is the argmin to the returned centers under
    # BOTH L2 metrics (the root is monotone). The final assignment used to
    # take ROOTED row norms under 'l2_sqrt_expanded', which clamped most
    # rows of a wide-magnitude fixture to 0 and gave them label 0.
    for name, xs in (("uniform [0, 4)", x), ("columns 1e-2 to 1e2", _wide())):
        for metric in ("l2_expanded", "l2_sqrt_expanded"):
            m = KMeans(n_clusters=4, random_state=3, metric=metric).fit(xs)
            bad = _label_not_argmin(xs, m.cluster_centers_, m.labels_)
            rep.check("METRIC", bad == 0, f"{metric} on {name}: every label is the argmin to the returned centers", f"{bad} of {len(xs)} are not")
    rep.raises("METRIC", Exception, "L2Expanded or L2SqrtExpanded", "metric='cosine' is routed and REFUSED BY NAME on the Mojo host (kmeans_params.mojo::validate)",
               KMeans(n_clusters=4, metric="cosine").fit, x)
    rep.raises("METRIC", Exception, "L2Expanded or L2SqrtExpanded", "metric='cosine_expanded' the same", KMeans(n_clusters=4, metric="cosine_expanded").fit, x)
    rep.raises("METRIC", ValueError, "metric must be", "an unknown metric name", KMeans(n_clusters=4, metric="manhattan").fit, x)


def arm_oversampling(rep):
    x = _x()
    base = KMeans(n_clusters=4, random_state=3).fit(x)
    two = KMeans(n_clusters=4, random_state=3, oversampling_factor=2.0).fit(x)
    if mode() == "identical":
        rep.check("OVS", _bits_same(base.cluster_centers_, two.cluster_centers_), "oversampling_factor=2.0 is the default, bit for bit")
    else:
        rep.report_only("OVS", _bits_same(base.cluster_centers_, two.cluster_centers_), "default vs 2.0")
    classic = KMeans(n_clusters=4, random_state=3, oversampling_factor=0.0).fit(x)
    rep.check("OVS", np.asarray(classic.cluster_centers_).shape == (4, 4) and np.isfinite(classic.inertia_) and classic.n_iter_ >= 1, "oversampling_factor=0.0 (classic sequential k-means++) fits", classic.inertia_)
    rep.report_only("OVS", _bits_same(base.cluster_centers_, classic.cluster_centers_), "classic vs scalable seeding centers (a different algorithm; the same answer is possible on a well separated fixture)")
    rep.raises("OVS", Exception, "oversampling_factor", "a negative oversampling_factor is refused on the Mojo host by name", KMeans(n_clusters=4, oversampling_factor=-1.0).fit, x)
    rep.raises("OVS", TypeError, "oversampling_factor", "a bool oversampling_factor", KMeans(n_clusters=4, oversampling_factor=True).fit, x)
    rnd = KMeans(n_clusters=4, random_state=3, init="random", oversampling_factor=0.0).fit(x)
    rep.check("OVS", np.isfinite(rnd.inertia_), "under init='random' the factor is unread and the fit runs")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn", "build.sh")
    rep = Report("test_kmeans_metric_surface")
    return run("test_kmeans_metric_surface", [("METRIC", arm_metric), ("OVS", arm_oversampling)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
