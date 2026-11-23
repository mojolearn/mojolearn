"""Before/after digests of the cluster-family member routes lane mf-clu6 added (2026-11-23), host column. One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested
by its words. Base: origin/lane/member-fits-6 ca3b0cfb8c (each setting below made the member admission answer False
and the class's own door ran the fit), head lane/mf-clu6 (the member runs it). Test tool, never product runtime.

Every configuration is fitted through GridSearchCV with one given split (train = test = every row, so a
precomputed matrix stays square) and a one-point grid, refit=True: the search fits the estimator as a member.
Scoring: the estimator's own score (KernelDensity, GaussianMixture, BayesianGaussianMixture) or
adjusted_rand_score on its predict. Digested: mean_test_score and the refit estimator's fitted attributes.

Rows covered (mf-cal5-work/fallback-audit.tsv): density 95, 98, 131, 136; hierarchy 76, 80, 91, 94; spectral 76,
80, 91; xcluster 120, 152, 184, 191, 241, 243, 316; kmeans 58; mixture 60, 65, 71 (71: HostGaussianMixture).

EXPECTED: every line equal, except, stated: `*_raised` lines whose words move from the door's refusal to the
member's refusal of the same input (the same words are intended); the spectral 'precomputed' COO route on a
binding without the dense entry (the member scans the float32 X, the door scanned it widened to float64: equal
for float32 input, which these configurations use); `*_scores` of a configuration with no predict (the search's
scorer raises either way). Configurations are independent: one process each across the host cores.

    python tools/mf_clu6_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(n=48, d=3, seed=3):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 3.0, 3.0]], dtype=np.float32)[:, :d]
    y = np.arange(n) % 3
    X = (centers[y] + 0.3 * rng.standard_normal((n, d))).astype(np.float32)
    return X, y


def _distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff * diff).sum(-1)).astype(np.float32)


def _config(name):
    """(estimator, X, y, scoring, attribute names) of configuration `name`."""
    import mojolearn
    import scipy.sparse as sp
    X, y = _data()
    D = _distances(X)
    ari = "adjusted_rand_score"
    if name == "kde_unknown_kernel":
        return mojolearn.KernelDensity(kernel="boxcar"), X, y, None, ["_x"]
    if name == "kde_metric_params":
        return mojolearn.KernelDensity(metric="minkowski", metric_params={"p": 2}), X, y, None, ["_x"]
    if name == "kde_l1":
        return mojolearn.KernelDensity(metric="l1", bandwidth=0.7), X, y, None, ["_x"]
    if name == "dbscan_no_prediction":
        return mojolearn.DBSCAN(eps=0.8, prediction_data=False), X, y, ari, ["labels_", "components_"]
    if name == "dbscan_precomputed":
        return (mojolearn.DBSCAN(eps=0.8, metric="precomputed", prediction_data=False), D, y, ari,
                ["labels_", "core_sample_indices_"])
    if name == "kmeans_array_init":
        return mojolearn.KMeans(n_clusters=3, init=X[:3].copy(), n_init=1), X, y, ari, ["cluster_centers_", "labels_"]
    if name == "minibatch_array_init":
        return (mojolearn.MiniBatchKMeans(n_clusters=3, init=X[:3].copy(), n_init=1, random_state=0), X, y, ari,
                ["cluster_centers_", "labels_"])
    if name == "meanshift_seeds":
        return mojolearn.MeanShift(seeds=X[::8].copy()), X, y, ari, ["cluster_centers_", "labels_"]
    if name == "affinity_precomputed":
        return (mojolearn.AffinityPropagation(affinity="precomputed", random_state=0), -(D * D), y, None,
                ["labels_", "cluster_centers_indices_"])
    if name == "affinity_preference_array":
        return (mojolearn.AffinityPropagation(preference=np.full(len(X), -5.0, dtype=np.float32), random_state=0),
                X, y, ari, ["labels_", "cluster_centers_"])
    if name == "bgmm_mean_prior":
        return (mojolearn.BayesianGaussianMixture(n_components=3, mean_prior=np.zeros(3), random_state=0), X, y,
                None, ["means_", "weights_"])
    if name == "bgmm_warm":
        return (mojolearn.BayesianGaussianMixture(n_components=3, warm_start=True, max_iter=5, random_state=0),
                X, y, None, ["means_", "weights_"])
    if name == "agglo_ward":
        return mojolearn.AgglomerativeClustering(n_clusters=3, linkage="ward"), X, y, None, ["labels_", "children_"]
    if name == "agglo_cosine_average":
        return (mojolearn.AgglomerativeClustering(n_clusters=3, linkage="average", metric="cosine"), X + 1.0, y,
                None, ["labels_", "children_"])
    if name == "agglo_threshold":
        return (mojolearn.AgglomerativeClustering(n_clusters=None, distance_threshold=2.0, linkage="single"), X,
                y, None, ["labels_", "children_", "distances_", "n_clusters_"])
    if name == "agglo_distances_predict":
        return (mojolearn.AgglomerativeClustering(n_clusters=3, linkage="single", compute_distances=True,
                                                  prediction_data=True), X, y, ari,
                ["labels_", "children_", "distances_", "_fit_X"])
    if name == "agglo_no_prediction":
        return (mojolearn.AgglomerativeClustering(n_clusters=3, linkage="single", prediction_data=False), X, y,
                None, ["labels_", "children_", "n_boruvka_rounds_"])
    if name in ("agglo_connectivity_sparse", "agglo_connectivity_dense", "agglo_connectivity_callable"):
        near = (D < 1.5).astype(np.float32)
        conn = {"agglo_connectivity_sparse": sp.csr_matrix(near), "agglo_connectivity_dense": near,
                "agglo_connectivity_callable": (lambda Z: near)}[name]
        return (mojolearn.AgglomerativeClustering(n_clusters=3, linkage="average", connectivity=conn), X, y, None,
                ["labels_", "children_", "n_connected_components_"])
    if name == "spectral_rbf":
        return (mojolearn.SpectralClustering(n_clusters=3, affinity="rbf", gamma=0.5, prediction_data=False,
                                             random_state=0), X, y, None, ["labels_", "embedding_", "affinity_matrix_"])
    if name == "spectral_precomputed":
        A = np.exp(-(D * D)).astype(np.float32)
        return (mojolearn.SpectralClustering(n_clusters=3, affinity="precomputed", random_state=0), A, y, None,
                ["labels_", "embedding_"])
    if name == "spectral_precomputed_nn":
        return (mojolearn.SpectralClustering(n_clusters=3, affinity="precomputed_nearest_neighbors", n_neighbors=6,
                                             prediction_data=False, random_state=0), D, y, None,
                ["labels_", "embedding_", "affinity_matrix_"])
    if name == "spectral_no_prediction":
        return (mojolearn.SpectralClustering(n_clusters=3, n_neighbors=6, prediction_data=False, random_state=0), X,
                y, None, ["labels_", "embedding_"])
    if name == "gmm_diag":
        return (mojolearn.GaussianMixture(n_components=3, covariance_type="diag", random_state=0), X, y, None,
                ["means_", "covariances_", "lower_bound_"])
    if name == "gmm_n_init":
        return mojolearn.GaussianMixture(n_components=3, n_init=2, random_state=0), X, y, None, ["means_", "weights_"]
    if name == "gmm_means_init":
        return (mojolearn.GaussianMixture(n_components=3, means_init=X[:3].astype(np.float64), random_state=0), X, y,
                None, ["means_", "weights_"])
    if name == "gmm_warm":
        return (mojolearn.GaussianMixture(n_components=3, warm_start=True, max_iter=5, random_state=0), X, y, None,
                ["means_", "precisions_cholesky_"])
    if name == "gmm_bad_init":
        return mojolearn.GaussianMixture(n_components=3, init_params="bogus"), X, y, None, ["means_"]
    if name == "gmm_host":
        from mojolearn._classical_host import HostGaussianMixture
        return HostGaussianMixture(n_components=3), X, y, None, ["means_"]
    raise ValueError(name)


def _unit(name):
    from mojolearn.model_selection import GridSearchCV
    out = []
    try:
        est, X, y, scoring, attrs = _config(name)
        every = np.arange(len(X))
        search = GridSearchCV(est, {"random_state": [0]} if "random_state" in est.get_params() else
                              {k: [v] for k, v in list(est.get_params().items())[:1]},
                              cv=[(every, every)], scoring=scoring, refit=True)
        search.fit(X, y)
        out.append((name + "_scores", _digest(np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64))))
        best = search.best_estimator_
        for attr in attrs:
            value = getattr(best, attr, None)
            out.append((f"{name}_{attr}", "None" if value is None else _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["kde_unknown_kernel", "kde_metric_params", "kde_l1", "dbscan_no_prediction", "dbscan_precomputed",
         "kmeans_array_init", "minibatch_array_init", "meanshift_seeds", "affinity_precomputed",
         "affinity_preference_array", "bgmm_mean_prior", "bgmm_warm", "agglo_ward", "agglo_cosine_average",
         "agglo_threshold", "agglo_distances_predict", "agglo_no_prediction", "agglo_connectivity_sparse",
         "agglo_connectivity_dense", "agglo_connectivity_callable", "spectral_rbf", "spectral_precomputed",
         "spectral_precomputed_nn", "spectral_no_prediction", "gmm_diag", "gmm_n_init", "gmm_means_init",
         "gmm_warm", "gmm_bad_init", "gmm_host"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
