# API exposure fixes, 2026-10-22 (lane/api-exposure-fixes)

Input: the API exposure audit of the 1508 native binding entries
(`~/mojolearn-evidence/audit-20261022/api_exposure.md` and `.tsv`, main 551dee1ce):
PUBLIC 1153, INTERNAL-SUPPORT 246, DEAD 109, plus one module-level finding (the
multi-GPU driver modules had no `__all__`, were not imported by `mojolearn` and were
not counted by `tools/public_runtime_inventory.py`).

## 1. Exposed names

Every module below now has a literal `__all__`, is imported by `mojolearn/__init__.py`
and listed in `mojolearn.__all__` (wired as `parallel_forecasting` /
`parallel_model_selection` are), has rows in `packaging/native_only/manifest.json`
(so `tools/public_runtime_inventory.py` counts it), carries user-voice docstrings (the
old engineering docstrings are `#` comments beside them), and has a real-path case in
`python/mojolearn/tests/test_parallel_public_drivers.py` (devices from
`MOJOLEARN_PAR_DEVICES`, default `0`; each case equals the plain one-device estimator
byte for byte).

Docs decision for all six modules: excluded in `docs-site/api_exclude.txt`
("multi-GPU: physical qualification pending (par-* lanes recorded on one device per
vendor)"), the same footing as `parallel_forecasting`, `parallel_gaussian_process`,
`parallel_ivf` and `parallel_model_selection`. Their par-* lanes have NVIDIA, AMD and
Apple reference records in `python/mojolearn/verify_reference/table.json`, but those
records are one-device columns (`tools/record_identity_column.sh`: par_devices 0);
`docs/multi_gpu/COVERAGE.md` still lists cross-vendor multi-device qualification as
remaining for every family. Assigned to the `multi_gpu` family in
`docs-site/tools/gen_api.py` for the day they qualify.

| module | public names | verifier lanes (table.json columns) |
|---|---|---|
| `parallel_classical` | `fit_logistic` | par-logistic (nv, amd, apple) |
| | `fit_exponential_smoothing` | par-holtwinters (nv, amd, apple, cpu) |
| | `fit_gram_estimator` | par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd (nv, amd, apple) |
| | `fit_arima` | par-arima (nv, amd, apple, cpu) |
| | `fit_kmeans` | par-kmeans (nv, amd, apple) |
| | `fit_coordinate_descent` | par-cd, par-cd-elasticnet (nv, amd, apple) |
| | `fit_svm`, `predict_svm` | par-svm, par-svm-svr (nv, amd, apple) |
| | `fit_gaussian_process`, `predict_gaussian_process` | par-gp (nv, amd, apple) |
| | `fit_dbscan` | par-dbscan (nv, amd, apple) |
| | `fit_gaussian_mixture`, `predict_gaussian_mixture` | par-gmm (nv, amd, apple) |
| | `bootstrap`, `permutation_test`, `monte_carlo_integrate` | par-resample (nv, amd, apple, cpu) |
| | `fit_hdbscan` | par-hdbscan (nv, amd, apple) |
| | `fit_cholesky`, `solve_cholesky` | par-cholesky (nv, amd, apple) |
| | `fit_kernel_method`, `apply_kernel_method` | par-kernel-ridge, par-nystroem (nv, amd, apple) |
| | `transform_rbf_sampler` | par-rbf-sampler (nv, amd, apple, cpu) |
| `parallel_ensemble` | `fit_boosting` | par-boosting, par-boosting-clf, par-boosting-reg, par-boosting-pointwise, par-ordered, par-border-types (nv, amd, apple) |
| | `fit_forest` | par-forest, par-forest-et, par-forest-et-clf, par-forest-reg (nv, amd, apple, cpu) |
| | `fit_isolation_forest`, `score_isolation_forest` | par-iforest (nv, amd, apple) |
| | `fit_ordered_rmse` | par-ordered-rmse (nv, amd, apple) |
| | `fit_feature_freq` | par-feature-freq (nv, amd, apple) |
| | `ParallelForestPredictor` | par-forest-pool (nv, amd, apple) |
| `parallel_graph` | `fit_graph` | par-graph-agglomerative, par-graph-spectral, par-graph-umap (nv, amd, apple) |
| | `transform_umap` | par-graph-umap (nv, amd, apple) |
| `parallel_neighbors` | `ParallelQueries` | par-queries-knn, par-queries-nn, par-queries-radius, par-queries-kde (nv, amd, apple, cpu) |
| `parallel_neighbors_reference` | `ReferenceShardedNeighbors` | par-reference-knn, par-reference-knn-reg (nv, amd, apple, cpu) |
| `parallel_preprocessing` | `fit_scaler`, `transform_scaler` | par-scaler, par-scaler-minmax (nv, amd, apple, cpu) |
| `ensemble` | `catboost_auto_learning_rate` | none directly; reached through GradientBoosting fits (gbdt lanes) |

`parallel_neighbors_reference` is product: `docs/multi_gpu/README.md` documents it,
two verifier lanes and two capacity/transport tools call it.

`ensemble.catboost_auto_learning_rate`: `ensemble.py` gets a literal `__all__` that keeps
today's public set (the boosting estimators, their option tables, scikit-learn's lazily
resolved ensemble names) and adds this function. Docs decision: excluded
(`ensemble.catboost_auto_learning_rate`, "host scalar helper behind GradientBoosting's
automatic learning rate; no verifier lane calls it directly").

Also updated: `SUPPORT_MATRIX.md` (Multi-GPU row), `tools/qualify_verifier_wheel.py`
(expanded API import guard), `tools/public_runtime_inventory.py` (family names).

Inventory (`tools/public_runtime_inventory.py --summary`, TOTAL row): before 383 names /
996 methods (120 native, 367 orchestrated, 509 python); after 418 names / 1033 methods
(120 native, 368 orchestrated, 545 python). The 36 new python-verdict methods are the
drivers' worker-pool glue; the static walk does not follow `DevicePool` requests into
the worker. Baseline refreshed in its own commit; `--check` passes.
