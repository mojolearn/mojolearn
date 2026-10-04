# Shared IDENTICAL call-path source coverage

**Source inventory and opt-in proposals.** The original inventory was not
compiled or executed; subsequent narrow integration gate results are recorded
in [README.md](README.md). They do not validate every family below. The scope is
all available algorithm families and all existing GPU vendor columns, not an
Apple-only dispatch. Applicability in this inventory does **not** mean that an
estimator has been migrated, validated, or accelerated.

This checkout starts at `c96137714`. Inventory anchors are the Mojo estimator
modules, GPU bindings, public Python modules and
[`host_surface.py`](../../python/mojolearn/host_surface.py). The latter separately
declares CPU/reference/inference families; its existing coverage records are
not evidence for these new experiments. Later Apple worktrees were not merged.

## What is implemented here

[`core/identical_callpath.mojo`](../../core/identical_callpath.mojo) provides
vendor-neutral context ownership, typed resident buffers, retained host staging,
upload/readback submission and explicit completion. It does not reinterpret
an estimator, select a new arithmetic profile, or fuse operations.

[`families.mojo`](families.mojo) adds five explicit building-block adapters:

| Adapter | Existing implementation retained | Concrete coverage and limits |
| --- | --- | --- |
| `enqueue_row_norms` | `core.row_norms.row_norm_kernel`, `NORM_TPB` | Distance/neighbor/clustering norm stage; pinned fold width and square-root flag retained |
| `enqueue_column_means` | `core.column_stats.column_mean_kernel`, `STATS_TPB` | PCA/OLS mean stage; same feature-axis block mapping |
| `enqueue_shift_columns` | `core.column_stats.shift_columns_kernel`, 256 threads | Existing unfused centering/restoration only; sign restricted to +1/-1; does not replace PCA fused covariance branch |
| `enqueue_core_gemm_nt` | `core.gemm.gemm_nt` | Existing callers of this exact function; preserves its `n == 1` GEMV route |
| `enqueue_identical_gemm` | `gemm.checks.gemm_identical.identical_gemm_into` | Existing callers of this exact dispatcher; external workspace sized with `identical_gemm_workspace_max_floats` |

These adapters require an active guarded session, check logical slot extents
and basic alias restrictions, and abort/poison the session on failure. They do
not replace estimator-level validation, traces, host decisions or model objects.
They refuse empty matrices and more than Int32 cells; an estimator retaining an
existing empty-input behavior must handle that before calling them.
Only readback the logical result extent: reserve the exact output size unless
the caller separately initializes and intentionally reads every padded element.

The generic GEMM and core GEMM adapters are deliberately separate: substituting
one for another could change the profile even in IDENTICAL mode. Preserve trial
defines/environment and all existing dispatched branches when comparing them.
`identical_gemm_into` documents plan-specific internal allocations and waits on
some NVIDIA paths; session reuse does not remove these. No adapter asserts a
zero-allocation or zero-wait complete algorithm.

## Every available family: integration inventory

`P` means a concrete shared primitive adapter above is applicable to stages;
`S` means shared typed storage/completion applies, but consumer wiring is still
needed. **Neither status means a public estimator is migrated.** Device types
below identify important payloads rather than a complete scratch-layout schema.

| Family / algorithms | Existing GPU entry or source | Buffers and session applicability | Identity obligations / required boundary |
| --- | --- | --- | --- |
| Preprocessing: MinMaxScaler, StandardScaler | `preprocessing/estimator.mojo`; `preprocessing/minmax.mojo`, `standard.mojo` | S: Float32 data/model/output, extrema/stat scratch; scaler experiments in this directory provide their own explicit adapters | Preserve finite refusal, signed extrema, near-constant handling, transform operation order, inverse/clip flags and fitted statistics |
| PCA, whitened PCA, TruncatedSVD | `decomposition/estimator.mojo`: `pca_fit_host`, `pca_fit_full_host`, `pca_transform_host`, `tsvd_fit_host`, inverse/whiten paths | P: resident data, components, means, eigensolver/GEMM scratch | Keep fused-vs-unfused centered Gram predicate, solver choice, sign/order normalization, whitening scale and input restoration; preserve eigensolver host convergence reads |
| OLS, Ridge, LogisticRegression | `glm/estimator.mojo`: OLS/ridge/QN fit and prediction entries | P: Float32 design, targets, coefficients, predictions and scratch | Preserve weighted centering, coefficient/intercept layout, Gram/GEMV dispatch and QN host line-search/termination decisions |
| Lasso, ElasticNet | `solver/estimator.mojo`: `cd_fit_host`, `cd_predict_host` | S/P: coordinate-descent buffers, prediction primitive | Preserve coordinate update order, residual initialization, convergence readbacks and penalty scaling |
| KMeans | `cluster/estimator.mojo`: `kmeans_fit`, `kmeans_predict`, `kmeans_transform` | P: data/centroid norms, distances; integer labels and sum scratch | Preserve fixed-point sum-scale certification, RNG draws, ties, empty-cluster handling and host convergence/inertia boundaries |
| DBSCAN | `dbscan/estimator.mojo`: `dbscan_fit` | S: distance data, labels, graph/queue/union scratch | Preserve traversal/label canonicalization, connectivity semantics, initialized sentinels and graph phase completion |
| HDBSCAN and inference | `hdbscan/estimator.mojo`: fit, approximate prediction and membership entries | S: floating distances/probabilities; integer graph/tree structures | Preserve MST tie ordering, hierarchy/condensation, cluster selection and prediction metadata; host model assembly remains a boundary |
| Agglomerative / single linkage | `hierarchy/estimator.mojo`: `linkage_fit_host` | S: distance/connectivity buffers and integer merges | Preserve edge ordering, tie resolution and host linkage assembly |
| Spectral clustering and prediction | `spectral/estimator.mojo`: dataset/graph fit and `spectral_predict_host` | S/P: COO graph, eigensolver/embedding Float32 storage, integer labels | Preserve graph order, eigenbasis/sign rules, clustering seed and host spectral-prediction path where selected |
| UMAP dense and sparse | `umap/estimator.mojo`, `umap/sparse_estimator.mojo`: fuzzy graph and fit-transform | S/P: distance/embedding data, sparse graph indices/weights | Preserve graph construction, RNG counters, edge order, optimizer updates and sparse offsets; no reorder of SGD |
| NearestNeighbors, KNeighborsClassifier/Regressor, RadiusNeighbors / RBC | `neighbors/estimator.mojo`: `knn_search_resident`, resident classifier/regressor entries, radius count/fill | P/S: existing resident Float32 index; UInt32 neighbor IDs; distances, counts and scratch | Preserve host index refusals, metric and query-tile planning, stable distance/index ties, voting arithmetic; radius count must finish before host-sized fill allocation |
| IVF-Flat build/search/extend | `ivf/estimator.mojo`: `ivf_flat_build_host`, `ivf_flat_search_host`, `ivf_flat_extend_host` | S/P: resident index payload, centroids, list offsets/IDs and query scratch | Preserve list ordering, quantization/search plan, probes, tie order and index lifetime; model ownership must remain explicit |
| KernelDensity | `kde/estimator.mojo`: `kde_score_samples_host`, pointer entry | S/P: resident training data, query/distance/log-score scratch | Preserve kernel/bandwidth choice, norm reduction, log-sum-exp ordering and final normalization |
| KernelRidge | `kernel_methods/estimator.mojo`: fit/predict entries | P/S: kernel matrix, dual/model data, Cholesky/GEMM scratch | Preserve kernel parameters, jitter and solve ordering; no substitution of Gram/GEMM profiles |
| Nystroem, RBFSampler | `kernel_methods/estimator.mojo`: fit, transform and transform-into entries | P/S: resident components, weights/offsets, eigenvalue and feature scratch | Preserve RNG schedule, landmark order, clipped spectrum, cosine arithmetic and exact transform layout |
| GaussianProcessRegressor/Classifier, posterior sampling | `gaussian_process/estimator.mojo`: GPR fit/LML gradient/predict/classify/sample entries | P/S: model factor, kernel matrices, means/covariance, normal draws and typed status buffers | Preserve jitter ladder, kernel profile, Cholesky/GEMM route, host optimizer and LML folds, class threshold and sample RNG |
| GaussianMixture fit and inference/sample | `mixture/estimator.mojo`: fit, score/proba/predict/sample entries | S/P: model means/covariances, responsibilities, status scratch | Preserve initialization, E/M ordering, covariance-type profile, host convergence and component-collapse checks; multi-device fold ordering remains unchanged |
| SVC and SVR | `svm/estimator.mojo`: fit, borrowed fit, predict entries | S/P: support vectors, dual coefficients, kernel tiles, working-set indices | Preserve working-set selection, convergence/KKT host decisions, support-vector ordering, class vote/threshold and kernel parameters |
| RandomForest classification/regression | `ensemble/randomforest.mojo`; `core/forest_inference.mojo`, `forest_inference_pool.mojo` | S: node/feature/leaf buffers, integer offsets and predictions | Preserve sampling seeds, split ties, quantization, tree/grove accumulation order, global-tree division and context ownership |
| ExtraTrees classification/regression | `extratrees/estimator.mojo`: device/reference/host-exact fits | S: tree data and integer node/index scratch | Preserve explicit route choice, host-exact reference behavior, RNG counters, candidate order and leaf reduction; do not turn a host reference route into a GPU route |
| Decision-tree building shared by forests | `ensemble/decisiontree/decisiontree.mojo`, `batched_levelalgo/` | S: quantile/bin storage, nodes and split scratch | Preserve node/level traversal, histogram zeroing, integer scales and split ordering |
| GradientBoosting, ordered and experimental variants | `gbdt/estimator.mojo`: `gbdt_fit`, `gbdt_predict`, multioutput/two-level entries | S: feature bins, target/permutation/CTR state, gradients and model outputs | Preserve categorical hashes, permutations, ordered folds, split scores, RNG sequence, per-round host control and serialization |
| IsolationForest | `isolation_forest/estimator.mojo`: `IsolationForestEstimator`, `iforest_run_host` | S: tree/index storage, path scores and output labels | Preserve sampling, tree path arithmetic, host percentile threshold and contamination handling |
| Cholesky factor/solve/logdet/rank-one update | `cholesky/estimator.mojo`: factor, solve, logdet, update entries | P/S: factor/model and solve scratch, status | Preserve panel/update/solve order, jitter and refusal checks; keep host factor status and logdet fold boundaries |
| GEMM, QR, symmetric eigenvalue, singular-value primitives | `bindings/_mojolearn_linalg.mojo`; `core/gemm.mojo`; `core/householder_qr.mojo` | P/S: Float32 matrices and existing workspace layouts | Same op/layout, arithmetic profile, tile/reduction and eigensolver iteration/sign order; GEMM adapter does not convert another profile |
| Low-bit GEMM and conversion | `bindings/_mojolearn_linalg.mojo`: BF16/int8 GEMM, quantize/dequantize, BF16 conversions | S: signed Int8, raw UInt16 BF16 bits, Int32 accumulators, Float32 scales/output; typed banks must preserve representations | No float reinterpretation of packed payloads; preserve rounding, saturation, scale, block layout and overflow contract |
| ARIMA fit, prediction, forecast | `arima/estimator.mojo`: pointer fit/predict/forecast entries | S: Float32 time series, fitted coefficients, state/covariance and statuses | Preserve sequential Kalman recurrence and likelihood fold, optimizer host callbacks, missing-data handling and forecast layout; an associative rewrite is outside this IDENTICAL plumbing experiment |
| KPSS and differencing selection | `tsa/estimator.mojo`: `kpss_test_host`, `select_d_host` | S: series and stationarity scratch/results | Keep trend/difference decisions and host result reads at original locations |
| Holt-Winters fit and forecast | `holtwinters/estimator.mojo`: traced/host/pointer entries | S: persistent model and seasonal state, outputs/statuses | Preserve seasonal indexing, recurrence order, smoothing parameter search, stopping decisions and fitted-state ownership |
| Metrics: classification, regression, clustering, ranking, silhouette, trustworthiness | `metrics/estimator.mojo` | S/P: Int32 labels/counts, Float32 values/distances, typed confusion matrices and scalar outputs | Preserve integer count widths, stable ranking/ties, weighted reductions, scalar host folds and refusal semantics |
| Bootstrap, permutation tests, Monte Carlo integration | `resample/estimator.mojo` | S: samples/statistics, index/permutation scratch and RNG counters | Preserve Philox counter assignment, resample order, quantiles/sorting, host confidence interval rules and owner fold order |
| MLP inference/training and optimizer/loss/clipping | `training/mlp_ops.mojo`, `training/estimator.mojo` | S/P: weights, activations, gradients, optimizer states, labels and losses | Preserve optimizer transactional validation, step count, accumulation order, clipping/loss folds and error boundaries |
| Transformer, Llama/causal LM | `transformer/impl/llama/modeling_llama.mojo`; `training/byte_lm.mojo` | S/P: resident weights/stages/KV/RoPE, activations and GEMM workspaces | Preserve attention profile, masking, position/cache transitions, RNG, normalization and output projection; reuse existing stage/model owners rather than duplicate their storage |
| Mamba/Mamba2/Mamba3 and Samba | `mamba/impl/`, `mamba/checks/*backward.mojo`; `training/samba_ops.mojo` | S/P: recurrence/convolution caches, activations, gradients and model weights | Preserve scan/chunk profile, state carry, recurrence/fold order and transactional decode/optimizer boundaries |
| Embedding forward/backward | `embedding/checks/embedding_identical.mojo`: forward/backward-into; `embedding/checks/embedding_sort.mojo` | S: token IDs, embeddings, Float32 gradients and sorting scratch | Preserve index refusals, duplicate-token stable ordering, gradient fold and padding behavior; no atomics/reordered scatter substitution |
| Multi-device classical, graph, neighbors, GP, forecast, preprocessing, training/model pools | `python/mojolearn/parallel_*.py`; algorithm-specific multi-GPU modules; `training/*pool.mojo` | S: one session per actual context/device; retained local state only | Preserve partition plan, global sample/RNG IDs, inter-device dependencies and root merge order; never share a session's buffers across contexts |
| Model selection and pipelines/orchestration | `python/mojolearn/model_selection.py`, `parallel_model_selection.py` | S only through GPU-consuming constituent estimators | Preserve folds, train/validation isolation, seeds, scoring and independent-model lifetimes; caching must not leak state between folds |

## Host-only and mixed paths

The BPE tokenizer (`python/mojolearn/tokenizer.py`, `tokenizer/host/`) is a host
integer algorithm with no GPU binding according to the host manifest. It needs
no GPU session. Vocabulary merge ordering, token IDs and byte encoding must
remain unchanged; claiming it benefits from this GPU path would be inaccurate.

CPU reference training and saved-model inference are separate existing routes.
The manifest has families `byte_lm`, `forest`, `tokenizer`, `neural`, `core`,
`linalg`, `estimators`, `metrics`, `preprocessing`, `tsa`, `solver`, `svm`,
`trees`, `rf`, `gp`, `kernel_methods`, `mixture`, `mixture_infer`, `hdbscan`,
`gp_infer`, `hdbscan_infer`, `gbdt`, `training`, `resample`, `mamba`, `arima`,
`embedding`, `embedding_infer`, `ivf`, `ivf_search`, `forecast`, `transformer`.
Those CPU bindings do not become GPU-session clients. Their arithmetic/host
ownership is unchanged. This includes host-exact tree routes, classical host
prediction, host GP optimizer work, neural host inference and CPU decoding.

Python array validation, label encoding, sparse/ragged metadata, serialization,
checkpoint I/O, corpus loading, model selection and statistical host decisions
also remain host operations. A method named `*_host` in a GPU estimator module
usually orchestrates GPU work; it must not be assumed to be CPU-only.

The original request also names LU, kernel PCA, randomized SVD, LLE,
MinCovDet, SVGP, AutoARIMA, VAR, KNN-imputer, Gaussian/sparse random projections,
AdditiveChi2Sampler and MaxAbsScaler. Distinct public implementations of all
these names are not present in this baseline inventory. TruncatedSVD,
`tsa.impl.auto_arima.select_d`, and RBFSampler are not evidence that every
similarly named requested algorithm is available or integrated. The shared
session is intended to accept their typed buffers when their actual lanes are
integrated; keep the same per-family obligations and report that work separately.

## All-vendor invariants and the remaining rollout

Apple/Metal, NVIDIA/CUDA and AMD/HIP use the same session source. The existing
numeric-mode and kernel-matrix dispatch still select the original implementation
for each target. No new warp width, tile size, reduction shape or fast arithmetic
flag is introduced. CPU-only routes remain CPU routes. New source has not been
compiled or validated on any vendor.

For a concrete family migration, first preserve its validation and complete
operation list. Reserve typed data/model/scratch slots outside the repeated
call; stage only changed inputs. Begin, upload, enqueue the exact existing
operations in the same order, queue required readbacks, finish and collect.
Keep an explicit finish at every host-dependent branch. A later iteration may
begin on the same session while retaining model and initialized device state.
Do not use batching to change numerical aggregation or remove initialization.

GEMM workspaces must use the original dispatcher sizing helper. Shapes, layouts,
dtype interpretation, aliasing, padding, signed zero, FTZ seams, special-value
refusals, scratch initialization, RNG assignment, trace points, model state and
host exception order are all observable contracts. Integer IDs cannot travel
through Float32 storage. Raw BF16 bit buffers cannot be silently replaced by
numeric Float16 conversion. Mixed-size result reads must respect logical bounds.

Future validation, only after authorization, must separately establish
same-vendor baseline-versus-session raw-bit identity and existing cross-vendor
profile identity. Include changing inputs, independent model instances, repeated
calls, tails, cold construction, host branch barriers, alias refusals and enqueue
failure cleanup. Only then can latency/allocation/launch/wait measurements justify
adopting a variant. None of those checks were run for this inventory.
