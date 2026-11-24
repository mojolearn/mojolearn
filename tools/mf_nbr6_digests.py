"""Before/after digests of the member routes lane mf-nbr6 added (2026-11-23) in the XNeighbors, XDecomp and
Manifold families, host column. One line per configuration and output: name and sha256 of the output bytes
(dtype and shape included); a refusal is digested by its words. Base: origin/lane/member-fits-6 53a3adbed4 (each
setting below made the member admission answer False and the class's own door ran the fit), head lane/mf-nbr6
(the member runs it). Test tool, never product runtime.

Every configuration is fitted through GridSearchCV with one given split (train = test = every row, so a
precomputed matrix stays square), a one-point grid on the estimator's first parameter by name (never
random_state: a RandomState seed is cloned per fit), refit=True: the refit fits the estimator as a member.
Scoring: 'accuracy' on y = 1 for OneClassSVM, 'r2' for SVGP, else none (a transformer's search scores are nan
either way). Digested: mean_test_score, the refit estimator's fitted attributes and one output.

Rows covered (mf-cal5-work/fallback-audit.tsv): xneighbors 171 (no unit: the vendor 'none' image is never
importable), 248, 264, 330, 349, 382, 385, 402, 423, 426, 428, 445, 454, 474, 482, 564, 606, 634; xdecomp 247,
774, 776, 842, 1147, 1181, 1185, 1204; manifold 118, 123.

EXPECTED: every line equal, except, stated: `dl_callback_raised` (the refusal's text now names the native
alternative: 'callback is not carried (...)'); `*_raised` lines whose words move from the door's refusal to the
member's refusal of the same input; `svgp_inducing_Z_` identity aside (the bytes are equal); `lle_fast_*` where
this installation has no fast x_decomp image (then both sides raise the same words). Configurations are
independent: one process each across the host cores.

    python tools/mf_nbr6_digests.py > digests.txt
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


def _data(n=48, d=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _config(name):
    """(estimator, X, y, scoring, attribute names, output method) of configuration `name`."""
    import mojolearn
    X, y = _data()
    K = (X @ X.T).astype(np.float32)
    ones = np.ones(len(X), dtype=np.int64)
    rs = np.random.RandomState
    if name == "ocsvm_precomputed":
        return (mojolearn.OneClassSVM(kernel="precomputed", nu=0.3), K, ones, "accuracy",
                ["support_", "dual_coef_", "support_vectors_", "offset_", "n_features_in_"], "decision_function")
    if name == "ocsvm_rbf":
        return (mojolearn.OneClassSVM(nu=0.3), X, ones, "accuracy", ["support_", "dual_coef_", "offset_"],
                "decision_function")
    if name == "kpca_precomputed":
        return (mojolearn.KernelPCA(n_components=2, kernel="precomputed"), K, None, None,
                ["eigenvalues_", "eigenvectors_"], "transform")
    if name == "kpca_rbf":
        return mojolearn.KernelPCA(n_components=2, kernel="rbf"), X, None, None, ["eigenvalues_"], "transform"
    if name == "pcs_randomstate":
        return (mojolearn.PolynomialCountSketch(n_components=8, random_state=rs(0)), X, None, None,
                ["indexHash_", "bitHash_"], "transform")
    if name == "schi2_randomstate":
        return (mojolearn.SkewedChi2Sampler(n_components=8, random_state=rs(0)), np.abs(X), None, None,
                ["random_weights_", "random_offset_"], "transform")
    if name == "achi2":
        return mojolearn.AdditiveChi2Sampler(), np.abs(X), None, None, ["n_features_in_"], "transform"
    if name == "imputer_k0":
        return mojolearn.KNNImputer(n_neighbors=0), X, None, None, ["_fit_X"], "transform"
    if name == "imputer_k3":
        return mojolearn.KNNImputer(n_neighbors=3), X, None, None, ["_fit_X"], "transform"
    if name == "svgp_inducing":
        return (mojolearn.SVGP(inducing_points=X[::6].copy()), X, y, "r2", ["Z_", "_alpha", "_C", "elbo_"],
                "predict")
    if name == "svgp_rows":
        return mojolearn.SVGP(n_inducing=8), X, y, "r2", ["Z_", "_alpha", "elbo_"], "predict"
    if name == "label_spreading":
        codes = (np.arange(len(X)) % 3).astype(np.int64)
        return mojolearn.LabelSpreading(), X, codes, "accuracy", ["label_distributions_"], "predict_proba"
    if name == "nearest_centroid":
        codes = (np.arange(len(X)) % 3).astype(np.int64)
        return mojolearn.NearestCentroid(), X, codes, "accuracy", ["centroids_"], "predict"
    if name == "ica_w_init":
        return (mojolearn.FastICA(n_components=2, w_init=np.eye(2, dtype=np.float32) + 0.1), X, None, None,
                ["components_", "mixing_", "n_iter_"], "transform")
    if name == "dl_inits":
        return (mojolearn.DictionaryLearning(n_components=3, code_init=np.ones((len(X), 3), dtype=np.float32),
                                             dict_init=np.eye(3, 4, dtype=np.float32), max_iter=5, random_state=0),
                X, None, None, ["components_", "error_", "n_iter_"], "transform")
    if name == "dl_dict_init_only":
        return (mojolearn.DictionaryLearning(n_components=3, dict_init=np.eye(3, 4, dtype=np.float32), max_iter=5,
                                             random_state=0), X, None, None, ["components_"], "transform")
    if name == "dl_callback":
        return (mojolearn.DictionaryLearning(n_components=3, callback=print), X, None, None, ["components_"],
                "transform")
    if name == "mbdl_dict_init":
        return (mojolearn.MiniBatchDictionaryLearning(n_components=3, dict_init=np.eye(3, 4, dtype=np.float32),
                                                      max_iter=2, random_state=0), X, None, None,
                ["components_", "n_steps_"], "transform")
    if name == "spca_inits":
        return (mojolearn.SparsePCA(n_components=2, V_init=np.eye(2, 4, dtype=np.float32),
                                    U_init=np.ones((len(X), 2), dtype=np.float32), max_iter=5, random_state=0),
                X, None, None, ["components_", "mean_", "error_"], "transform")
    if name == "isomap_radius":
        return (mojolearn.Isomap(n_neighbors=None, radius=3.0, n_components=2), X, None, None,
                ["embedding_", "dist_matrix_", "_knn"], "transform")
    if name == "isomap_radius_far":
        return (mojolearn.Isomap(n_neighbors=None, radius=0.05, n_components=2), X, None, None,
                ["embedding_"], "transform")
    if name == "lle_fast":
        return (mojolearn.LocallyLinearEmbedding(n_neighbors=8, n_components=2, eigen_solver="dense",
                                                 numeric_mode="fast"), X, None, None, ["embedding_"], "transform")
    if name in ("umap_random", "umap_pca"):
        return (mojolearn.UMAP(n_neighbors=8, n_epochs=20, init=name[5:], random_state=3), X, None, None,
                ["embedding_"], "transform")
    if name == "umap_array":
        start = (np.arange(2 * len(X), dtype=np.float32).reshape(len(X), 2) / len(X))
        return (mojolearn.UMAP(n_neighbors=8, n_epochs=20, init=start, random_state=3), X, None, None,
                ["embedding_"], "transform")
    if name == "umap_array_bad_shape":
        return (mojolearn.UMAP(n_neighbors=8, n_epochs=20, init=np.zeros((3, 2), dtype=np.float32)), X, None,
                None, ["embedding_"], "transform")
    if name == "umap_big_seed":
        return (mojolearn.UMAP(n_neighbors=8, n_epochs=20, random_state=2 ** 60 + 1), X, None, None,
                ["embedding_", "_transform_config"], "transform")
    raise ValueError(name)


def _unit(name):
    from mojolearn.model_selection import GridSearchCV
    out = []
    try:
        est, X, y, scoring, attrs, method = _config(name)
        every = np.arange(len(X))
        first = sorted(est.get_params())[0]
        search = GridSearchCV(est, {first: [est.get_params()[first]]}, cv=[(every, every)], scoring=scoring,
                              refit=True)
        search.fit(X, y)
        out.append((name + "_scores", _digest(np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64))))
        best = search.best_estimator_
        for attr in attrs:
            value = getattr(best, attr, None)
            out.append((f"{name}_{attr}", "None" if value is None else
                        (repr(value) if isinstance(value, tuple) else _digest(value))))
        try:
            out.append((f"{name}_{method}", _digest(getattr(best, method)(X[:8]))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"{name}_{method}_raised", f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["ocsvm_precomputed", "ocsvm_rbf", "kpca_precomputed", "kpca_rbf", "pcs_randomstate", "schi2_randomstate",
         "achi2", "imputer_k0", "imputer_k3", "svgp_inducing", "svgp_rows", "label_spreading", "nearest_centroid",
         "ica_w_init", "dl_inits", "dl_dict_init_only", "dl_callback", "mbdl_dict_init", "spca_inits",
         "isomap_radius", "isomap_radius_far", "lle_fast", "umap_random", "umap_pca", "umap_array",
         "umap_array_bad_shape", "umap_big_seed"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
