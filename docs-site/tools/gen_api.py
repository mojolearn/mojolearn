"""Write the API reference pages (api/*.rst) from the installed mojolearn's __all__.

Every public name lands on exactly one family page or is listed in api_exclude.txt
(broken, unverified or unfinished: not documented); an unassigned name fails the
build, so a release cannot ship a public name the reference silently leaves out.
Usage: python tools/gen_api.py   (from docs-site/, with the documented mojolearn installed)
"""
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import pathlib
import sys

import mojolearn

# (slug, title, intro, module prefixes, extra names). First match wins, in order.
FAMILIES = [
    ("runtime", "Runtime and numeric modes",
     "Process-wide numeric mode, device and backend queries.",
     ("mojolearn._backend", "mojolearn._numeric_profile"),
     {"set_numeric_mode", "numeric_mode", "vendor", "gpu_arch", "__version__", "Array"}),
    ("boosting", "Gradient boosting", "Gradient-boosted decision trees: classification, regression and ranking losses.",
     ("mojolearn._gbdt_adapters", "mojolearn.gbdt"),
     {"GradientBoosting", "GradientBoostingClassifier", "GradientBoostingRegressor", "OrderedRMSE",
      "ExperimentalTwoLevelFeatureFreq"}),
    ("trees", "Forests and trees", "Random forests, Extra Trees, single trees, bagging, AdaBoost, isolation forests and saved-model tree inference.",
     ("mojolearn.randomforest", "mojolearn.extratrees", "mojolearn.ensemble", "mojolearn._expansion_trees",
      "mojolearn._forest_host"), {"IsolationForest", "HostGBDT"}),
    ("linear", "Linear models", "Least squares, regularized, robust, Bayesian and generalized linear models.",
     ("mojolearn.linear_model", "mojolearn._solver_impl", "mojolearn._expansion_linear"), set()),
    ("svm_kernels", "Support vector machines and kernel methods", "SVMs, kernel ridge and kernel approximations.",
     ("mojolearn.svm", "mojolearn._svm_impl", "mojolearn.kernel_methods"), set()),
    ("gaussian_process", "Gaussian processes", "Gaussian process regression and classification, with kernels.",
     ("mojolearn._gp_impl", "mojolearn._gpc_impl", "mojolearn.gaussian_process"), set()),
    ("neighbors", "Neighbors and vector search", "Exact neighbors, IVF, CAGRA and quantized indexes for vector search.",
     ("mojolearn.neighbors", "mojolearn._expansion_neighbors", "mojolearn._expansion_ann", "mojolearn.ivf"),
     {"IVFIndex", "DistributedIVFIndex"}),
    ("clustering", "Clustering and mixture models", "K-means and its variants, density and hierarchical clustering, spectral clustering and Gaussian mixtures.",
     ("mojolearn.cluster", "mojolearn.hdbscan", "mojolearn.mixture", "mojolearn._expansion_cluster",
      "mojolearn._spectral_impl", "mojolearn._hierarchy_impl", "mojolearn.kmeans", "mojolearn.dbscan"), {"hdbscan", "mixture", "DBSCAN"}),
    ("decomposition", "Decomposition and manifold learning", "PCA, SVD, matrix factorizations and embeddings for visualization.",
     ("mojolearn.decomposition", "mojolearn._expansion_decomp", "mojolearn.umap", "mojolearn._umap_impl", "mojolearn.manifold"),
     {"umap", "manifold"}),
    ("preprocessing", "Preprocessing", "Scalers, encoders, imputers and discretizers.",
     ("mojolearn.preprocessing", "mojolearn._expansion_prep"), {"preprocessing"}),
    ("density", "Density estimation", "Kernel density estimation.", ("mojolearn.density",), set()),
    ("time_series", "Time series", "ARIMA, exponential smoothing, decomposition and statistical forecasting models.",
     ("mojolearn._tsa_impl", "mojolearn._x_sequence_theta", "mojolearn._x_sequence_croston", "mojolearn._x_sequence_ets",
      "mojolearn._x_sequence_var", "mojolearn._x_sequence_stl", "mojolearn._x_sequence_prophet",
      "mojolearn._x_sequence_garch", "mojolearn._x_sequence_autoarima", "mojolearn.arima", "mojolearn._arima_impl"), set()),
    ("neural", "Neural networks", "Layers, blocks, inference sessions, trainers, optimizers and schedules.",
     ("mojolearn._x_sequence_rnn", "mojolearn._x_sequence_mlp", "mojolearn._x_sequence_norm", "mojolearn._expansion_cnn",
      "mojolearn.neural_inference", "mojolearn._mamba_impl", "mojolearn._transformer_impl", "mojolearn._samba_impl",
      "mojolearn.embedding", "mojolearn._x_sequence_moe", "mojolearn._training_impl", "mojolearn._x_sequence_optim",
      "mojolearn._x_sequence_sched", "mojolearn.training", "mojolearn.neural_network", "mojolearn.mamba",
      "mojolearn.transformer", "mojolearn._mlp_impl"),
     {"embedding", "training", "neural_network", "mamba", "transformer"}),
    ("language_models", "Language models and tokenizers", "Language-model training and inference, checkpoints, tokenizers and corpora.",
     ("mojolearn._byte_lm", "mojolearn.tokenizer", "mojolearn.language_model", "mojolearn.lm_corpus", "mojolearn.models"),
     {"tokenizer", "language_model", "lm_corpus", "models"}),
    ("multi_gpu", "Multi-GPU and cross-vendor training", "Ordered multi-GPU training, model pools, offloading and partitioned workloads.",
     ("mojolearn.parallel", "mojolearn.offload_training", "mojolearn.model_pool_training", "mojolearn.cross_vendor"),
     {"parallel_forecasting", "parallel_gaussian_process", "parallel_ivf", "parallel_model_selection"}),
    ("evaluation", "Metrics, model selection and resampling", "Scores, cross-validation, bootstrap and permutation tests.",
     ("mojolearn.metrics", "mojolearn.model_selection", "mojolearn.resample"),
     {"metrics", "model_selection", "resample"}),
    ("linalg", "Linear algebra", "GEMM, factorizations and low-bit weight storage.",
     ("mojolearn.linalg", "mojolearn.lowbit", "mojolearn.cholesky", "mojolearn._linalg"), {"linalg", "lowbit", "Cholesky", "matmul"}),
]


def family_of(name, obj):
    mod = getattr(obj, "__module__", "") or ""
    if isinstance(obj, type(sys)):
        mod = obj.__name__
    for slug, _, _, prefixes, extra in FAMILIES:
        if name in extra:
            return slug
    for slug, _, _, prefixes, extra in FAMILIES:
        if any(mod == p or mod.startswith(p + ".") or mod.startswith(p + "_") for p in prefixes):
            return slug
    return None


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    out = root / "api"
    excluded = {line.split("\t", 1)[0].strip() for line in (root / "api_exclude.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")}
    excluded = {e for e in excluded if "." not in e}  # dotted entries are members, skipped in conf.py
    stale = excluded - set(mojolearn.__all__)
    if stale:
        sys.exit(f"gen_api: api_exclude.txt names that are not public: {sorted(stale)}")
    out.mkdir(exist_ok=True)
    pages = {slug: [] for slug, *_ in FAMILIES}
    missing = []
    for name in mojolearn.__all__:
        if name == "__version__" or name in excluded:
            continue
        obj = getattr(mojolearn, name)
        slug = family_of(name, obj)
        (pages[slug] if slug else missing).append(name)
    if missing:
        sys.exit(f"gen_api: public names with no API page: {missing}")
    families = [f for f in FAMILIES if pages[f[0]]]  # a family with every name excluded gets no page
    for old in out.glob("*.rst"):
        old.unlink()
    for slug, title, intro, *_ in families:
        names = sorted(pages[slug], key=str.lower)
        body = [f".. Generated by tools/gen_api.py from mojolearn {mojolearn.__version__}. Do not edit.", "",
                title, "=" * len(title), "", intro, "", ".. currentmodule:: mojolearn", ""]
        if names:
            body += [".. autosummary::", "   :toctree: generated", "   :nosignatures:", ""]
            body += [f"   {n}" for n in names]
        (out / f"{slug}.rst").write_text("\n".join(body) + "\n")
    index = ["API reference", "=============", "",
             f"Every public name in ``mojolearn`` {mojolearn.__version__}, by family. "
             "Estimators follow the scikit-learn conventions (``fit``, ``predict``, ``transform``) "
             "unless their page says otherwise.", "", ".. toctree::", "   :maxdepth: 1", ""]
    index += [f"   {slug}" for slug, *_ in families]
    (out / "index.rst").write_text("\n".join(index) + "\n")
    print("gen_api:", {s: len(v) for s, v in pages.items()})


if __name__ == "__main__":
    main()
