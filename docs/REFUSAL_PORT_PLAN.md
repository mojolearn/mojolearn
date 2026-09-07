# Historical refusal matrix: current port and evidence plan

Source-only audit. No tests, builds, measurements, models or provisioning were
run. Root qualifies completed changes serially on authorized NVIDIA/AMD hosts,
at most three CPU cores. The release snapshot and staged 0.6.0 are unchanged.

## Exact matrix and counts

The requested matrix is **round 11, commit `144aa5b`, 2026-08-23**, not a
current feature inventory. Its paper source is
[`mlsys/results/cross-vendor-identity.json`](../../mlsys/results/cross-vendor-identity.json),
consumed by [`gen_numbers.py`](../../mlsys/paper/gen_numbers.py) and
[`numbers.tex`](../../mlsys/paper/numbers.tex) (`cvTotalCells=209`).

| Family | Cases | Traced identical | Host/no-card | Refused alike |
| --- | ---: | ---: | ---: | ---: |
| Trees | 116 | 109 | 2 | 5 |
| Unsupervised/linear | 93 | 71 | 0 | 22 |
| Total | 209 | 180 | 2 | 27 |

These are per-configuration verdicts against the Apple reference on each of
NVIDIA/AMD, not 209 distinct algorithms or 27 absent features. Raw NVIDIA
sources: [tree cells](../bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/e2_cells.json),
[non-tree cells](../bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/e2u/e2u_cells.json).
Corresponding AMD sources: [tree cells](../bench/results/e1/2026-08-23_172650-mojolearn-e2-amd/e2_cells.json),
[non-tree cells](../bench/results/e1/2026-08-23_172650-mojolearn-e2-amd/e2u/e2u_cells.json).
The paper's metadata traces the original removed `E3_RESULTS.md` document to
revision `d878991361ea27f434949b17d92e773047c0b836`; restoring provenance does
not constitute a new run. Keep historical counts unchanged.

## The five tree rows are not five missing parameter ports

| Exact cell | Historical refusal | Current decision |
| --- | --- | --- |
| `et_clf_maxleaf` | `max_leaf_nodes=64` requires sklearn best-first growth, not cuML breadth-first `max_leaves` | Already implemented since Sept 1 in the ExtraTrees path; refresh installed/card evidence rather than port twice |
| `gbdt_multiclass_lossguide` | MultiClass with Lossguide lacks the corresponding CatBoost GPU trainer | Retain unsupported combination; a multiclass nonsymmetric objective/search/leaf port is substantial separate work |
| `gbdt_quantile_newton` | Newton leaf estimation is not supported for Quantile | Retain refusal; do not invent a nonzero Hessian or alias Newton to another estimator |
| `gbdt_rmse_depthwise_pointwise` | The selected pointwise searcher is oblivious; Depthwise requires the nonsymmetric searcher | Retain incompatible-option refusal; ordinary Depthwise already has its own route |
| `gbdt_rmse_nan_forbidden_refused` | NaN data with `nan_mode='Forbidden'` | Retain intentional input refusal; Min/Max are separate supported policies |

[extratrees.py](../python/mojolearn/extratrees.py) explicitly documents the new
best-first mode and its remaining sklearn differences. The
[native tree binding](../bindings/_mojolearn_trees.mojo) forwards
`max_leaf_nodes` in slot 19. A breadth-first `max_leaves` alias would change
semantics and must not replace it. No new numerical tree patch is necessary
to resolve the one genuine historical parameter gap in these five rows.

## Two host/no-card rows

`et_clf_cpu` and `et_reg_cpu` explicitly requested `device='cpu'`; their
retained JSON has `card: null` and model/prediction hashes (plus classifier
probabilities). They establish host-output agreement, not GPU stage identity.
They are not missing GPU algorithms and must not be silently relabelled.

- [ ] Add host-path trace stages only if host certification is required: RNG
  state/draws, selected samples/features, candidate thresholds/gains, frontier
  choice, partitions, leaf outputs and serialized model. Check unchanged
  outputs and an effective altered-stage control. This is instrumentation,
  not a port to GPU.
- [ ] Separately qualify the existing GPU ExtraTrees API, including best-first
  growth and heldout prediction, with actual binary/mode/vendor witnesses.
  Do not count those GPU cases as retroactive coverage of the CPU rows.

## Twenty-two non-tree rows, grouped by underlying behavior

| Exact cells | Shared feature or distinction | Plan/status from source |
| --- | --- | --- |
| `dbscan_algorithm_bad`, `knn_algorithm_kd_tree` | Explicit kd-tree/index choices | **Intentional unsupported algorithms; retain refusals per user preference** |
| `dbscan_chain_iter200`, `dbscan_maxiter1` | One convergence-limit refusal at two iteration budgets | Keep safety refusal; never return an unconverged label snapshot as IDENTICAL |
| `kmeans_init_bad` | Invalid literal `pca-ish` | Keep invalid-option refusal, not a feature request |
| `dbscan_metric_manhattan` | Non-Euclidean DBSCAN neighborhood semantics | Separate metric implementation/gates; do not alias to Euclidean |
| `dbscan_sw` | Weighted DBSCAN core-point threshold | Historical gap; current density/DBSCAN source has weighted support, requiring refreshed installed evidence |
| `kde_bw_scott_refused` | Host bandwidth rule | Explicit bandwidth-rule API work; no new density kernel needed, but specify sample/dimension rule and FP32 conversion |
| `kde_metric_cosine_refused` | KDE cosine distance | Current density source exposes expanded metric coverage; qualify normalization and kernel/metric validity, not just name acceptance |
| `knn_clf_weights_distance_refused` | Distance-weighted classifier vote | Current neighbors source exposes distance weights; qualify zero-distance/tie handling and all-zero/infinite edge rules |
| `knn_k300` | Selection capacity beyond 256 | Inspect actual selection dispatch and capacity; qualify K=257/300 with full distance/index records before changing a bound |
| `knn_metric_cosine`, `knn_metric_minkowski_p1` | Distinct cosine and L1 metric math; p=1 maps to the L1 family | Current brute-force neighbors source supports these metrics; preserve separate indexed-route refusals and refresh device evidence |
| `logreg_l1_refused` | OWL-QN rather than L-BFGS | A real nonsmooth optimizer port; changing the accepted penalty string is insufficient |
| `ols_onecol_refused`, `ols_wide_refused` | Former OLS fallback restrictions at one-column and underdetermined shapes | Current linear-model source documents scalar/minimum-norm paths; different mathematical cases, refresh both rather than duplicate a generic SVD request |
| `ols_sw_refused` | Weighted least squares | Current linear-model source validates and rescales weights; retain weighted centering/zero-mass checks |
| `pca_c8_wide`, `tsvd_c8_wide` | **Same** 200-column IDENTICAL Gram capacity restriction | One shared Gram-kernel capacity/dispatch investigation, two consumers; do not substitute vendor GEMM under IDENTICAL |
| `pca_solver_full`, `tsvd_randomized` | Distinct decomposition algorithms | Separate full-SVD/randomized implementations; neither is an alias for covariance-eigh |
| `pca_whiten` | Transform/inverse-transform rescaling | Current decomposition wrapper calls dedicated whitening exports; refresh exact installed binding and inverse behavior |

Current source entry references: [density](../python/mojolearn/density.py),
[neighbors](../python/mojolearn/neighbors.py),
[linear models](../python/mojolearn/linear_model.py),
[decomposition](../python/mojolearn/decomposition.py),
[shared Gram entry](../core/gemm.mojo). Source implementation is not current
wheel qualification. These groupings intentionally distinguish aliases,
shared kernels, invalid inputs and separate algorithms instead of equating
the refusal count with engineering work.

## Ordered implementation/evidence queue

1. Root refreshes small existing-feature cases first: ExtraTrees best-first,
   DBSCAN weights, kNN metrics/voting, OLS cases and PCA whitening. Record which
   exact installed extension supplies each export; an alpha Python overlay
   alone cannot create a missing native symbol.
2. For still-missing KDE bandwidth convenience, establish the upstream rule,
   weighting convention and conversion seams before a bounded wrapper patch.
   For Gram capacity, preserve current accumulation profile while changing
   staging/dispatch; gate both PCA and TSVD consumers.
3. Treat OWL-QN, decomposition alternatives and multiclass nonsymmetric trees
   as separate algorithm ports with independent references, not small flags.
4. Root runs meaningful correctness tests before any timing, with full raw
   model/output/stage comparisons across authorized GPUs and effective negative
   controls. Keep unsupported index algorithms and intentional invalid-input
   refusals covered. Publish refreshed counts as a new matrix, never rewrite
   the historical 209-case record.
