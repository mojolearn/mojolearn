"""Before/after digests of the RFE cases lane mf-member8 serves (2026-11-23): RFE with no elimination round over
NearestNeighbors, RadiusNeighbors, OPTICS, MDS and SpectralEmbedding (their new member kinds), the same estimators
with a round, Cholesky, AffinityPropagation's cluster_centers_ getter and IncrementalPCA's mean_ / var_ getters, host
column (and the device image when run there). One line per configuration and output: name and sha256 of the
output bytes (dtype and shape included); a refusal is digested by its exception type and words. Base:
lane/member-fits-8 8e995c2547, head lane/mf-member8. Test tool, never product runtime.

EXPECTED (stated per unit):
  `nn_kept`, `radius_kept`, `optics_kept`, `optics_dbscan_kept`, `mds_kept`, `se_kept`, `se_rbf_kept`,
    `se_pre_kept`: base RAISED "TypeError: mojolearn: RFE requires an estimator with a native member; <cls> has
    none"; head the support (all ones), ranking (all ones) and the final estimator's fitted arrays, which equal
    the `*_own` units (the estimator's own fit on the same X: the member runs the same entries).
  `*_own`: EQUAL base and head (the estimators' own fits are untouched).
  `nn_round`, `optics_round`, `mds_round`, `se_round`: base the "has none" words, head "mojolearn: RFE needs an
    estimator with coef_ or feature_importances_" (scikit-learn fails there too).
  `cholesky_kept`: base the "has none" words, head "TypeError: Cholesky.fit() takes 2 positional arguments but 3
    were given" (scikit-learn's RFE calls fit(X, y)).
  `ap_centers`: base RAISED attrgetter's words ("'AffinityPropagation' object has no attribute
    'cluster_centers_'", the name was not in the named-block table); head the support and ranking of the
    elimination over cluster_centers_.
  `ap_pre_centers`: EQUAL words (attrgetter's; head raises them before the first round fits).
  `ipca_mean`, `ipca_var`: base attrgetter's words, head the support and ranking over mean_ / var_.
Configurations are independent: one process each across the host cores.

    python tools/mf_member8_digests.py > digests.txt
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


def _data(n=60, d=5, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 3]).astype(np.float32)
    return x, y


_ATTRS = {
    "nn": ["_index", "n_samples_fit_", "n_features_in_"],
    "radius": ["_index", "n_samples_fit_", "n_features_in_"],
    "optics": ["labels_", "reachability_", "ordering_", "core_distances_", "predecessor_", "cluster_hierarchy_",
               "n_features_in_"],
    "optics_dbscan": ["labels_", "reachability_", "ordering_", "core_distances_", "predecessor_", "n_features_in_"],
    "mds": ["embedding_", "dissimilarity_matrix_", "stress_", "n_iter_", "n_features_in_"],
    "se": ["embedding_", "n_neighbors_", "n_features_in_"],
    "se_rbf": ["embedding_", "affinity_matrix_", "gamma_", "n_features_in_"],
    "se_pre": ["embedding_", "n_neighbors_"],
}


def _make(stem):
    import mojolearn
    if stem == "nn":
        return mojolearn.NearestNeighbors(n_neighbors=3)
    if stem == "radius":
        return mojolearn.RadiusNeighbors(radius=1.5)
    if stem == "optics":
        return mojolearn.OPTICS(min_samples=4)
    if stem == "optics_dbscan":
        return mojolearn.OPTICS(min_samples=0.1, cluster_method="dbscan", eps=1.0)
    if stem == "mds":
        return mojolearn.MDS(n_components=2, random_state=0, n_init=1, max_iter=50)
    if stem == "se":
        return mojolearn.SpectralEmbedding(n_components=2, random_state=0)
    if stem == "se_rbf":
        return mojolearn.SpectralEmbedding(n_components=2, affinity="rbf", random_state=0)
    if stem == "se_pre":
        return mojolearn.SpectralEmbedding(n_components=2, affinity="precomputed", random_state=0)
    raise ValueError(stem)


def _x_for(stem):
    x, y = _data()
    if stem == "se_pre":
        a = np.exp(-np.square(x[:20, None, :] - x[None, :20, :]).sum(axis=2)).astype(np.float32)
        return a, y[:20]
    return x, y


def _fitted_arrays(est, stem):
    out = []
    for attribute in _ATTRS[stem]:
        out.append((attribute, getattr(est, attribute)))
    return out


def _run(name):
    """The configuration's outputs as (suffix, value) pairs."""
    import mojolearn
    if name.endswith("_kept") and name[:-5] in _ATTRS:
        stem = name[:-5]
        x, y = _x_for(stem)
        rfe = mojolearn.RFE(_make(stem), n_features_to_select=x.shape[1]).fit(x, y)
        return [("support", np.asarray(rfe.support_)), ("ranking", np.asarray(rfe.ranking_))] + _fitted_arrays(
            rfe.estimator_, stem)
    if name.endswith("_own") and name[:-4] in _ATTRS:
        stem = name[:-4]
        x, _ = _x_for(stem)
        return _fitted_arrays(_make(stem).fit(x), stem)
    if name.endswith("_round") and name[:-6] in _ATTRS:
        stem = name[:-6]
        x, y = _x_for(stem)
        rfe = mojolearn.RFE(_make(stem), n_features_to_select=2).fit(x, y)
        return [("support", np.asarray(rfe.support_)), ("ranking", np.asarray(rfe.ranking_))]
    if name == "cholesky_kept":
        rng = np.random.default_rng(2)
        m = rng.random((6, 6)).astype(np.float32)
        a = (m @ m.T + 6 * np.eye(6, dtype=np.float32)).astype(np.float32)
        rfe = mojolearn.RFE(mojolearn.Cholesky(jitter=0.0), n_features_to_select=6).fit(a, np.zeros(6, np.float32))
        return [("support", np.asarray(rfe.support_))]
    if name == "ap_centers":
        x, y = _data(n=40, d=6)
        rfe = mojolearn.RFE(mojolearn.AffinityPropagation(random_state=0), n_features_to_select=3,
                            importance_getter="cluster_centers_").fit(x, y)
        return [("support", np.asarray(rfe.support_)), ("ranking", np.asarray(rfe.ranking_))]
    if name == "ap_pre_centers":
        x, y = _data(n=12, d=12)
        s = -np.abs(x - x.T).astype(np.float32)
        rfe = mojolearn.RFE(mojolearn.AffinityPropagation(affinity="precomputed", random_state=0),
                            n_features_to_select=6, importance_getter="cluster_centers_").fit(s, y)
        return [("support", np.asarray(rfe.support_))]
    if name in ("ipca_mean", "ipca_var"):
        x, y = _data(n=50, d=6)
        rfe = mojolearn.RFE(mojolearn.IncrementalPCA(n_components=2), n_features_to_select=3,
                            importance_getter="mean_" if name == "ipca_mean" else "var_").fit(x, y)
        return [("support", np.asarray(rfe.support_)), ("ranking", np.asarray(rfe.ranking_))]
    raise ValueError(name)


def _unit(name):
    out = []
    try:
        for suffix, value in _run(name):
            if np.isscalar(value):
                out.append((f"{name}_{suffix}", repr(value)))
            else:
                out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ([stem + "_kept" for stem in _ATTRS] + [stem + "_own" for stem in _ATTRS]
         + ["nn_round", "optics_round", "mds_round", "se_round", "cholesky_kept", "ap_centers", "ap_pre_centers",
            "ipca_mean", "ipca_var"])


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
