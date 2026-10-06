# A B experiment index

Repository snapshot: `252c988a5`, 2026-10-06, branch `lane/apple-fast-classical-ideas-20261006`. Paths below are repository-relative and link to the files that define, implement or record each experiment.

The new campaign contains **54 Apple FAST classical ML experiments**, all disabled by default and uncompiled, unverified and unmeasured. The shared catalog also contains **60 earlier cards**. Older Apple FAST records, IDENTICAL recipes and source toggle inventories overlap these cards; their counts must not be added as if they were distinct experiments. Existing neural records are included only as catalog references and do not expand the classical ML campaign.

Historical statuses belong to their original source, workload and evidence. A source path, registration, retained build status or historical verdict does not establish a current full-workload speed or quality result. This documentation change runs no experiment and changes no defaults.

| Collection | Entries | Definition or record |
| --- | --- | --- |
| New Apple FAST classical cards | 54 | [experiments/apple_fast_classical_20261006/ideas.json](../experiments/apple_fast_classical_20261006/ideas.json) |
| Earlier shared catalog cards | 60 | [experiments/performance_ideas/README.md](../experiments/performance_ideas/README.md) |
| IDENTICAL named candidate groups | 38 | [docs/identical/original-handoff-coverage.json](../docs/identical/original-handoff-coverage.json) |
| IDENTICAL explicit arm recipes | 75 | [tools/identical_candidate_recipes.json](../tools/identical_candidate_recipes.json) |
| IDENTICAL GEMM profiles | 5 | [experiments/identical_speed/profiles.json](../experiments/identical_speed/profiles.json) |
| IDENTICAL optimization ledger records | 9 | [docs/identical/optimization-ledger.json](../docs/identical/optimization-ledger.json) |
| Historical source toggle names including rollback controls | 266 | [docs/identical/toggle-inventory.json](../docs/identical/toggle-inventory.json) |
| Historical Apple FAST records and followups | Listed by original section below | [docs/apple-fast/EXPERIMENTS.md](../docs/apple-fast/EXPERIMENTS.md) |

## New Apple FAST classical experiments

Each card lists the **implemented mechanism**, which can be narrower than the original idea. A and B below give explicit compiler defines and runtime environment additions, not complete compiler commands. Shared prerequisites must remain equal in both arms; the lane files retain route restrictions and quality requirements. `MOJOLEARN_NUMERIC_MODE=fast` is also enforced by the shared integration. Listed workload recipes are coverage mappings with pending dataset, route and quality audits, not admitted measurements.

### Linear models and decomposition

#### AFCL-L01 Linear Gram row partition

Halve both existing Gram row partitions: 8192 to 4096 and 2048 to 1024. Existing tile-pair selection is unchanged. Workspace sizing and both Gram entry points consume the same chunk calculation.

- Implementation files: [x_linear/fast_gram.mojo](../x_linear/fast_gram.mojo).
- Caller files: [x_linear/ridge_grid.mojo](../x_linear/ridge_grid.mojo); [x_linear/device.mojo](../x_linear/device.mojo); [x_linear/ard_grid.mojo](../x_linear/ard_grid.mojo); [x_prep/device.mojo](../x_prep/device.mojo).
- Affected callers: Ridge; RidgeClassifier; RidgeCV; Lars; LassoLars; BayesianRidge; ARDRegression; LinearDiscriminantAnalysis; QuadraticDiscriminantAnalysis; other classical x_prep Gram stages using fast_sym_gram_into.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L01`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L01/manifest.json](../experiments/performance_ideas/AFCL-L01/manifest.json).
- Binding map: `x_linear`, `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/ridge`, `algos/ridge-clf`, `algos/ridge-cv`, `algos/lars`, `algos/lasso-lars`, `algos/bayesian-ridge`, `algos/ard`, `algos/lda-clf`, `algos/qda` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L02 Linear sufficient-statistic reduction

Fold chunk means, Gram entries and X-transpose-Y entries with four FP32 striped accumulators and a fixed pairwise finish instead of one dependent serial accumulator.

- Implementation files: [x_linear/fast_gram.mojo](../x_linear/fast_gram.mojo).
- Caller files: [x_linear/ridge_grid.mojo](../x_linear/ridge_grid.mojo); [x_linear/device.mojo](../x_linear/device.mojo); [x_linear/ard_grid.mojo](../x_linear/ard_grid.mojo); [x_prep/device.mojo](../x_prep/device.mojo).
- Affected callers: Ridge; RidgeClassifier; RidgeCV; Lars; LassoLars; BayesianRidge; ARDRegression; LinearDiscriminantAnalysis; QuadraticDiscriminantAnalysis; other classical x_prep Gram stages using fast_sym_gram_into.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L02`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L02/manifest.json](../experiments/performance_ideas/AFCL-L02/manifest.json).
- Binding map: `x_linear`, `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/ridge`, `algos/ridge-clf`, `algos/ridge-cv`, `algos/lars`, `algos/lasso-lars`, `algos/bayesian-ridge`, `algos/ard`, `algos/lda-clf`, `algos/qda` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L03 ElasticNetCV device work granularity

Predict two held-out rows per thread at a time, reusing every path coefficient across the two complete ascending feature chains. Preserve the existing MSE block fold and every alpha/fold.

- Implementation files: [x_linear/enetcv_fast.mojo](../x_linear/enetcv_fast.mojo).
- Caller files: [x_linear/device.mojo](../x_linear/device.mojo).
- Affected callers: LassoCV; ElasticNetCV.
- A defines: none; A environment: `MOJOLEARN_X_LINEAR_ENETCV_FAST=1`.
- B defines: `MOJOLEARN_AFCL_L03`; B environment: `MOJOLEARN_X_LINEAR_ENETCV_FAST=1`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L03/manifest.json](../experiments/performance_ideas/AFCL-L03/manifest.json).
- Binding map: `x_linear` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/lasso-cv`, `algos/enet-cv` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L04 Huber gradient loss work grouping

Use 256-row loss/gradient partials in the resident Huber solver instead of its default 512-row partials; scratch and witness launch counts follow the existing HF_FOLD-derived helpers.

- Implementation files: [x_linear/huber_fast.mojo](../x_linear/huber_fast.mojo).
- Caller files: [x_linear/device.mojo](../x_linear/device.mojo).
- Affected callers: HuberRegressor.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L04`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L04/manifest.json](../experiments/performance_ideas/AFCL-L04/manifest.json).
- Binding map: `x_linear` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/huber` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L05 Multiclass logistic gradient schedule

Use nominal 1024-row chunks and at most 512 gradient partials instead of 2048 rows and 256 partials. Workspace and launch planning share the new constants.

- Implementation files: [glm/impl/qn/fast_xtdz.mojo](../glm/impl/qn/fast_xtdz.mojo).
- Caller files: [glm/impl/qn/glm_base.mojo](../glm/impl/qn/glm_base.mojo); [glm/estimator.mojo](../glm/estimator.mojo).
- Affected callers: LogisticRegression binary and multinomial QN paths; LinearSVC QN paths; LinearSVR QN paths; QN squared/absolute regression callers using the same gradient product.
- A defines: `MOJOLEARN_QN_FAST_COALESCED_OFF`; A environment: none.
- B defines: `MOJOLEARN_QN_FAST_COALESCED_OFF`, `MOJOLEARN_AFCL_L05`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L05/manifest.json](../experiments/performance_ideas/AFCL-L05/manifest.json).
- Binding map: `estimators` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/logreg`, `classical2/linearsvc`, `classical2/linearsvr`, `algos/qn-reg` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L06 SMO gradient row reuse

Increase moved-row shared staging from 3072 to 4096 FP32 feature words. Kernel-row and alpha updates retain their complete ascending order.

- Implementation files: [svm/impl/fast_update_f.mojo](../svm/impl/fast_update_f.mojo).
- Caller files: [svm/impl/smosolver.mojo](../svm/impl/smosolver.mojo).
- Affected callers: SVC; SVR; OneClassSVM through the same SMO fused update when applicable.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L06`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L06/manifest.json](../experiments/performance_ideas/AFCL-L06/manifest.json).
- Binding map: `svm` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/svc`, `classical2/svr`, `algos/ocsvm` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L07 PCA covariance work granularity

Partition the existing compensated centered covariance into nominal 2048-row partials, then compensate the partial fold and divide once by n-1. Means and input restoration contract are retained.

- Implementation files: [decomposition/impl/linalg/detail/pca.mojo](../decomposition/impl/linalg/detail/pca.mojo).
- Caller files: [decomposition/impl/linalg/detail/pca.mojo](../decomposition/impl/linalg/detail/pca.mojo); [decomposition/estimator.mojo](../decomposition/estimator.mojo).
- Affected callers: PCA covariance-eigendecomposition route; direct compute_covariance consumers.
- A defines: `MOJOLEARN_PCA_FAST_COMPENSATED_COV`; A environment: none.
- B defines: `MOJOLEARN_PCA_FAST_COMPENSATED_COV`, `MOJOLEARN_AFCL_L07`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L07/manifest.json](../experiments/performance_ideas/AFCL-L07/manifest.json).
- Binding map: `estimators`, `kernel_methods`, `x_decomp`, `linalg`, `metrics`, `x_neighbors` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/pca` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L08 Decomposition tiled product geometry

Increase the tiled product's K slab depth from 16 to 32, retaining the 32x32 output tile, 2x2 thread microtile, transpose handling and complete partial-K coverage.

- Implementation files: [x_decomp/fast_gemm.mojo](../x_decomp/fast_gemm.mojo).
- Caller files: [x_decomp/device.mojo](../x_decomp/device.mojo); [x_decomp/kit_device.mojo](../x_decomp/kit_device.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo); [x_decomp/nmf_dev.mojo](../x_decomp/nmf_dev.mojo).
- Affected callers: NMF; FactorAnalysis; PLSRegression; PLSCanonical; CCA; ALS; randomized SVD; least squares; other classical callers of x_decomp launch_gemm / DKit.mm.
- A defines: `MOJOLEARN_DECOMP_FAST_GEMM_TILED`; A environment: none.
- B defines: `MOJOLEARN_DECOMP_FAST_GEMM_TILED`, `MOJOLEARN_AFCL_L08`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L08/manifest.json](../experiments/performance_ideas/AFCL-L08/manifest.json).
- Binding map: `x_decomp`, `linalg`, `metrics`, `x_neighbors` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/nmf`, `algos/factor-analysis`, `algos/pls`, `algos/pls-canonical`, `algos/cca`, `algos/als`, `algos/randomized-svd`, `algos/qr`, `algos/svd`, `classical/ols` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L09 TSQR panel reduction schedule

Use 128-row reflector-product partials instead of 256-row partials; preserve the scaled norm reduction, signs, complete reflector sequence and existing finish/update kernels.

- Implementation files: [x_decomp/fast_qr.mojo](../x_decomp/fast_qr.mojo).
- Caller files: [x_decomp/device.mojo](../x_decomp/device.mojo); [x_decomp/kit_device.mojo](../x_decomp/kit_device.mojo); [python/mojolearn/linalg_fast.py](../python/mojolearn/linalg_fast.py).
- Affected callers: QR through DevExec.geqrf/orgqr; randomized SVD when this QR route is selected; least squares when this QR route is selected; classical decomposition callers selecting the grid Householder route.
- A defines: `MOJOLEARN_QR_FAST_DEV`; A environment: none.
- B defines: `MOJOLEARN_QR_FAST_DEV`, `MOJOLEARN_AFCL_L09`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L09/manifest.json](../experiments/performance_ideas/AFCL-L09/manifest.json).
- Binding map: `x_decomp`, `linalg`, `metrics`, `x_neighbors` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/qr`, `algos/randomized-svd`, `classical/ols` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L10 NMF update memory traffic

Specialize OP_MUZ dispatch and process four independent coalesced output stripes per thread using the existing ew_cell(OP_MUZ) arithmetic and broadcast indices.

- Implementation files: [x_decomp/device.mojo](../x_decomp/device.mojo).
- Caller files: [x_decomp/kit_device.mojo](../x_decomp/kit_device.mojo); [x_decomp/nmf_dev.mojo](../x_decomp/nmf_dev.mojo).
- Affected callers: NMF solver=mu, beta_loss=frobenius; fit and transform.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L10`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L10/manifest.json](../experiments/performance_ideas/AFCL-L10/manifest.json).
- Binding map: `x_decomp`, `linalg`, `metrics`, `x_neighbors` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/nmf` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L11 Gaussian-process prediction batching

Use 128-query variance threadgroups in place of the default 256, preserving the complete ascending training-axis sum per query and all variance/std/clamp/probability work.

- Implementation files: [gaussian_process/afcl_prediction.mojo](../gaussian_process/afcl_prediction.mojo); [gaussian_process/checks/kernels.mojo](../gaussian_process/checks/kernels.mojo); [gaussian_process/gpc_device_var.mojo](../gaussian_process/gpc_device_var.mojo).
- Caller files: [gaussian_process/estimator.mojo](../gaussian_process/estimator.mojo); [gaussian_process/classifier.mojo](../gaussian_process/classifier.mojo); [gaussian_process/gpc_ovr.mojo](../gaussian_process/gpc_ovr.mojo).
- Affected callers: GaussianProcessRegressor variance/std prediction; GaussianProcessClassifier binary/OVR latent variance and predictive probability.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L11`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L11/manifest.json](../experiments/performance_ideas/AFCL-L11/manifest.json).
- Binding map: `gp` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/gpr`, `classical2/gpc`, `public/gpr-variance` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L12 Random-feature and kernel output work

Each thread applies the same random offset to one component in four consecutive rows; keep the same add/cos/multiply sequence, random draws and projected values.

- Implementation files: [kernel_methods/checks/random_features.mojo](../kernel_methods/checks/random_features.mojo).
- Caller files: [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo); [kernel_methods/rbf_resident.mojo](../kernel_methods/rbf_resident.mojo).
- Affected callers: RBFSampler transform with a separate projection epilogue; RBFSampler resident fit_transform including its staged pipeline.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L12`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L12/manifest.json](../experiments/performance_ideas/AFCL-L12/manifest.json).
- Binding map: `kernel_methods` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/rbf-sampler` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L13 Robust covariance C-step scheduling

Select each C-step trial's support with 128 threads rather than 256; all eight radix passes and the ascending-index tie scan remain. Initial and iterative support selection use the same geometry.

- Implementation files: [x_decomp/mcd_fast.mojo](../x_decomp/mcd_fast.mojo).
- Caller files: [x_decomp/kit_device.mojo](../x_decomp/kit_device.mojo); [x_decomp/mcd_fast.mojo](../x_decomp/mcd_fast.mojo).
- Affected callers: MinCovDet; EllipticEnvelope.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L13`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L13/manifest.json](../experiments/performance_ideas/AFCL-L13/manifest.json).
- Binding map: `x_decomp` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/min-cov-det`, `algos/elliptic-envelope` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-L14 SGD linear update work grouping

Schedule independent minibatch prediction/loss rows in 128-thread blocks instead of 256-thread blocks; witness capacity, offsets and launch count use the same new row-grid helper.

- Implementation files: [x_linear/device.mojo](../x_linear/device.mojo).
- Caller files: [python/mojolearn/_expansion_linear.py](../python/mojolearn/_expansion_linear.py); [x_linear/device.mojo](../x_linear/device.mojo); [x_linear/sgd.mojo](../x_linear/sgd.mojo).
- Affected callers: SGDClassifier; SGDRegressor; Perceptron/PassiveAggressiveClassifier/PassiveAggressiveRegressor when using the separate minibatch row kernel; SGDOneClassSVM when using the separate minibatch row kernel.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_L14`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); registration: [experiments/performance_ideas/AFCL-L14/manifest.json](../experiments/performance_ideas/AFCL-L14/manifest.json).
- Binding map: `x_linear` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/sgd-clf`, `algos/sgd-reg`, `algos/perceptron`, `algos/pa-clf`, `algos/pa-reg`, `algos/sgd-ocsvm` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

### Neighbors clustering and geometry

#### AFCL-G01 Exact kNN MMA query block geometry

MQ_SG changes 8 to 4; complete 32-lane groups halve per-block query ownership while retaining matrix tile, all index slices and top-k state. Launch and query-grid geometry share MQ_TPB/MQ_SG.

- Implementation files: [neighbors/impl/detail/fast_mma_knn.mojo](../neighbors/impl/detail/fast_mma_knn.mojo).
- Affected callers: fast_mma_knn -> fast_mma_partial_kernel / fast_mma_bigd_kernel; NearestNeighbors / KNeighborsClassifier / KNeighborsRegressor and classical exact-neighbor consumers.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G01`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_KNN_FAST_MMA_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G01/manifest.json](../experiments/performance_ideas/AFCL-G01/manifest.json).
- Binding map: `core`, `metrics`, `hdbscan` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/knn`, `classical2/knn-clf`, `classical2/knn-reg`, `classical/hdbscan`, `classical2/umap`, `classical2/spectral-embedding`, `classical2/spectral` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G02 Scalar streaming top-k candidate chunk

FKT_T changes 64 to 128 candidate rows per shared tile, amortizing staging barriers at a 16 KiB feature tile plus norms for the widest existing admitted row. Every candidate is still inserted into the per-slice top-k and all slices merge.

- Implementation files: [neighbors/impl/detail/fast_topk_knn.mojo](../neighbors/impl/detail/fast_topk_knn.mojo).
- Affected callers: fast_topk_knn -> fast_topk_partial_kernel -> fast_topk_merge_kernel; NearestNeighbors / KNeighborsClassifier / KNeighborsRegressor on the scalar fused route.
- A defines: `MOJOLEARN_KNN_FAST_MMA_OFF`; A environment: none.
- B defines: `MOJOLEARN_KNN_FAST_MMA_OFF`, `MOJOLEARN_AFCL_G02`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_KNN_FAST_TOPK_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G02/manifest.json](../experiments/performance_ideas/AFCL-G02/manifest.json).
- Binding map: `core`, `metrics`, `hdbscan` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/knn-clf`, `classical2/knn-reg` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G03 DBSCAN epsilon query block geometry

ME_SG changes 8 to 4 complete SIMD groups per threadgroup, halving query ownership and exposing more blocks. Staged index tile, candidate coverage and exact epsilon-boundary fallback remain.

- Implementation files: [dbscan/impl/neighbors/fast_mma_eps.mojo](../dbscan/impl/neighbors/fast_mma_eps.mojo).
- Affected callers: fast_mma_eps_neighborhood -> _launch_eps -> fast_mma_eps_kernel; DBSCAN.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G03`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_DBSCAN_FAST_MMA_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G03/manifest.json](../experiments/performance_ideas/AFCL-G03/manifest.json).
- Binding map: `estimators` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/dbscan` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G04 HDBSCAN core-distance staging tile

CT_TILE changes 64 to 32 reference rows per shared tile, reducing its widest threadgroup storage from 16 KiB to 8 KiB while visiting every reference and retaining the same register top-k.

- Implementation files: [hdbscan/impl/detail/fast_apple.mojo](../hdbscan/impl/detail/fast_apple.mojo); [hdbscan/impl/detail/core_tile.mojo](../hdbscan/impl/detail/core_tile.mojo).
- Affected callers: core_tile_applies -> core_tile_kernel; HDBSCAN core-distance construction with MOJOLEARN_HDB_CORE_TILE.
- A defines: `MOJOLEARN_HDB_CORE_TILE`; A environment: none.
- B defines: `MOJOLEARN_HDB_CORE_TILE`, `MOJOLEARN_AFCL_G04`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G04/manifest.json](../experiments/performance_ideas/AFCL-G04/manifest.json).
- Binding map: `hdbscan` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/hdbscan` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G05 Exhaustive centroid assignment row geometry

FUSED_NORMAL_TR changes 16 to 8 and FUSED_SKINNY_TR changes 8 to 4. Existing policy launch constants and footprint computations use the new row count; centroid column threads, feature tiles, alignment selection and exhaustive traversal stay fixed.

- Implementation files: [cluster/impl/distance/fused_distance_nn/simt_kernel.mojo](../cluster/impl/distance/fused_distance_nn/simt_kernel.mojo).
- Affected callers: cluster/impl/detail/min_cluster_distance_compute.mojo::_launch_fused and gated counterpart; KMeans; KMeans initialization/quantizer consumers including GaussianMixture, IVF and spectral clustering where that caller is used.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G05`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G05/manifest.json](../experiments/performance_ideas/AFCL-G05/manifest.json).
- Binding map: `core`, `mixture`, `metrics`, `x_cluster`, `ivf`, `x_ann` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/kmeans`, `classical2/gmm`, `classical2/ivf`, `classical2/spectral` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G06 Blocked centroid partial row granularity

BLOCK_ACC_ROWS selects 128 with the new control; A retains the existing setting (256 with the listed overrides absent). This increases independent per-feature accumulation work and partial-table size while retaining Int32 sums and scale policy.

- Implementation files: [cluster/checks/reduce_by_key.mojo](../cluster/checks/reduce_by_key.mojo).
- Affected callers: launch_accumulate_centroid_sums_blocked / launch_accumulate_weight_per_cluster_blocked and gated twins; KMeans and consumers using its blocked accumulation route.
- A defines: `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC`; A environment: none.
- B defines: `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC`, `MOJOLEARN_AFCL_G06`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_KMEANS_BLOCK_ACC_OFF`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_4096`, `MOJOLEARN_IDN_ALL_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G06/manifest.json](../experiments/performance_ideas/AFCL-G06/manifest.json).
- Binding map: `core`, `mixture`, `metrics`, `x_cluster`, `ivf`, `x_ann` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/kmeans`, `classical2/gmm`, `classical2/ivf`, `classical2/spectral` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G07 MiniBatchKMeans resident block and chunk schedule

MBF_TPB changes 256 to 128 and MBF_CH follows it. Four SIMD groups per block reduce resource footprint and increase independent groups; equality of chunk and block width preserves the existing compacted sum one-thread-per-row contract.

- Implementation files: [x_cluster/minibatch_fast.mojo](../x_cluster/minibatch_fast.mojo).
- Affected callers: MiniBatchKMeans resident mini-batch path; _mbf_assign_kernel / row-group assignment, compacted center sum, finish and reassignment kernels.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G07`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G07/manifest.json](../experiments/performance_ideas/AFCL-G07/manifest.json).
- Binding map: `x_cluster` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/minibatch-kmeans` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G08 GaussianMixture expectation row schedule

FE_TPB changes 256 to 128 on the fused path; GMM_ROW_TPB changes 128 to 64 for ordinary row kernels. Per-row component/feature traversal, stable logsumexp, refusal behavior and likelihood reduction policy remain.

- Implementation files: [mixture/checks/estep.mojo](../mixture/checks/estep.mojo).
- Affected callers: gmm_e_step / fast_estep_kernel; mixture/estimator.mojo fit, predict, predict_proba, score and score_samples callers through GMM_ROW_TPB defaults.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G08`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G08/manifest.json](../experiments/performance_ideas/AFCL-G08/manifest.json).
- Binding map: `mixture` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/gmm`, `public/gmm-outputs` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G09 IVF batched query group partition

FIVF_QPB changes 4 to 2 SIMD-owned queries per block. Narrow/wide shared query tiles shrink from 4/32 KiB to 2/16 KiB; the feature cap explicitly stays at its baseline 2048 instead of expanding with the smaller block.

- Implementation files: [ivf/impl/neighbors/ivf_flat/fast_ivf_scan.mojo](../ivf/impl/neighbors/ivf_flat/fast_ivf_scan.mojo).
- Affected callers: ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo -> fast_ivf_scan_kernel; IVFIndex and ANN callers using batched IVF-Flat scan.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G09`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_IVF_FAST_SCAN_OFF`, `MOJOLEARN_IVF_FAST_BALANCED_TASKS`, `MOJOLEARN_IVF_BALANCED_TASKS`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G09/manifest.json](../experiments/performance_ideas/AFCL-G09/manifest.json).
- Binding map: `ivf` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/ivf` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G10 UMAP CSR optimizer head block schedule

FAST_OPT_TPB changes 128 to 64 head rows per block. Each head still owns all its ordered edges, unchanged epoch snapshot and counter-derived negatives; dense and sparse FAST launchers import the same width.

- Implementation files: [umap/optimizer_fast.mojo](../umap/optimizer_fast.mojo).
- Affected callers: optimize_layout_fast; umap/sparse_optimizer.mojo FAST epoch launcher including umap_jacobi_epoch_fused_kernel; UMAP.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G10`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G10/manifest.json](../experiments/performance_ideas/AFCL-G10/manifest.json).
- Binding map: `metrics` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/umap` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G11 Spectral Lanczos sparse product row grouping

FL_SPMV_ROWS_PER_TG changes 8 to 4 complete SIMD-owned CSR rows. Existing 32-lane partition and row fold stay fixed, while shorter block lifetime can reduce scheduling effects of degree skew.

- Implementation files: [spectral/impl/sparse/solver/detail/lanczos.mojo](../spectral/impl/sparse/solver/detail/lanczos.mojo).
- Affected callers: _spmv_fast -> fl_spmv_warp_kernel; SpectralEmbedding / SpectralClustering Lanczos and restart callers.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G11`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_LANCZOS_FAST_OFF`, `MOJOLEARN_SPMV_WARP_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G11/manifest.json](../experiments/performance_ideas/AFCL-G11/manifest.json).
- Binding map: `metrics` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/spectral-embedding`, `classical2/spectral` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G12 MeanShift full-neighborhood partial chunks

MSG_T changes 256 to 512 while MSG_TPB stays 256. Two strided input-row passes amortize center staging and halve partial-table/final-fold work; the existing loop visits every row and handles ragged tails.

- Implementation files: [x_cluster/meanshift_fast.mojo](../x_cluster/meanshift_fast.mojo).
- Affected callers: _msg_part_kernel -> _msg_finish_kernel; x_cluster/device_ops.mojo::DeviceOps.meanshift grid route / MeanShift.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G12`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G12/manifest.json](../experiments/performance_ideas/AFCL-G12/manifest.json).
- Binding map: `x_cluster` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/meanshift` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G13 KDE stable score query work grouping

KDE2_QM changes 4 to 2 queries per thread. The query tile changes 64 to 32 rows; per-thread distance accumulators and shared query/reduction pages shrink. The existing query-grid-based chunk planner follows the new tile, retains every train contribution and uses the same stable max-rescaled logsumexp.

- Implementation files: [kde/impl/neighbors/kernel_density.mojo](../kde/impl/neighbors/kernel_density.mojo).
- Affected callers: kde2_dimtile_kernel -> kde2_score_samples_fast_apple / kde2_score_samples_fast_apple_to_host; KernelDensity score_samples and score, including resident scoring callers.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G13`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_KDE_DIMTILE_OFF`, `MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G13/manifest.json](../experiments/performance_ideas/AFCL-G13/manifest.json).
- Binding map: `estimators` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical/kde` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-G14 Agglomerative and shared MST reference tile grouping

MB_T changes 128 to 64 reference rows per shared/prefetched tile, retaining complete 32-row MMA admission groups. Operand/metadata shared pages and prefetch registers halve; full candidate traversal, scalar exact recomputation, lower-bound filter and value/index tie ordering remain.

- Implementation files: [hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo](../hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo).
- Affected callers: MmaBoruvka.enqueue -> fb_mma_nearest_kernel; hierarchy/impl/cluster/detail/fast_boruvka.mojo::fast_euclidean_mst; AgglomerativeClustering supported single-linkage Euclidean pairwise route; HDBSCAN mutual-reachability callers using MmaBoruvka, including optional device-round route when enabled equally in both arms.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_G14`; B environment: none.
- Defines absent from both arms: `MOJOLEARN_BORUVKA_FAST_MMA_OFF`, `MOJOLEARN_SL_FAST_BORUVKA_OFF`, `MOJOLEARN_LEGACY_NARROW_SL_BORUVKA`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); registration: [experiments/performance_ideas/AFCL-G14/manifest.json](../experiments/performance_ideas/AFCL-G14/manifest.json).
- Binding map: `solver`, `hdbscan` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/agglomerative`, `classical/hdbscan` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

### Trees and explainers

#### AFCL-T01 Quantized boosting histogram row chunks

QH_MIN_ITEMS_PER_BLOCK is 4096 with T01 instead of 8192. The same constant feeds host replica selection and per-partition active-block limits; all rows keep their existing quantization and bin destinations.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo](../gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo); [gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo](../gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo).
- Affected callers: GradientBoosting/CatBoost-style Depthwise fit; GradientBoosting/CatBoost-style Lossguide fit.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T01`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T01/manifest.json](../experiments/performance_ideas/AFCL-T01/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-depthwise`, `trees/gbdt-lossguide` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T02 Quantized boosting histogram feature groups

QH_GROUP_FEATURES becomes eight; shared allocation, whole compressed-index-word count, feature offsets, skip-bin vector width and launcher group count derive from it. This reduces the per-group histogram to 16 KiB.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo](../gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo); [gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo](../gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo).
- Affected callers: GradientBoosting/CatBoost-style Depthwise fit; GradientBoosting/CatBoost-style Lossguide fit.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T02`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T02/manifest.json](../experiments/performance_ideas/AFCL-T02/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-depthwise`, `trees/gbdt-lossguide` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T03 Per-feature binarization block width

BINARIZE_BLOCK_SIZE becomes 512 instead of 1024, keeping eight documents per thread. Imported launcher constants and in-kernel row strides use the same geometry; all 256 border slots remain loadable.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/gpu_data/kernel/binarize.mojo](../gbdt/gpu_data/kernel/binarize.mojo).
- Affected callers: Boosted-tree fit quantization; Boosted-tree resident prediction quantization.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T03`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T03/manifest.json](../experiments/performance_ideas/AFCL-T03/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-symmetric`, `trees/gbdt-symmetric-1000`, `trees/gbdt-depthwise`, `trees/gbdt-lossguide`, `trees/gbdt-ordered`, `trees/gbdt-categorical` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T04 Symmetric partition metadata geometry

SPLIT_BLOCK_SIZE becomes 128 instead of 256. Bin-update, partition-offset/size and associated fused winner/bin launches inherit the four-SIMD-group width and matching grids/shared-array sizing.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/methods/pointwise_optimization_subsets.mojo](../gbdt/methods/pointwise_optimization_subsets.mojo).
- Affected callers: Symmetric boosted-tree structure search; Ordered symmetric structure-search callers of pointwise optimization subsets.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T04`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T04/manifest.json](../experiments/performance_ideas/AFCL-T04/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-symmetric`, `trees/gbdt-symmetric-1000`, `trees/gbdt-ordered` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T05 Cooperative split-record batches

Inside existing cooperative _fold_block, a lane folds four adjacent candidate records before stepping by block_width*4, instead of one strided record. The same _record_less comparator, all candidates, shared winner reduction and descriptor writes remain.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/methods/kernel/pointwise_split_resolve.mojo](../gbdt/methods/kernel/pointwise_split_resolve.mojo).
- Affected callers: Symmetric/ordered boosted-tree fused split search.
- A defines: `MOJOLEARN_SYM_RESOLVE_BLOCK`; A environment: none.
- B defines: `MOJOLEARN_SYM_RESOLVE_BLOCK`, `MOJOLEARN_AFCL_T05`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T05/manifest.json](../experiments/performance_ideas/AFCL-T05/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-symmetric`, `trees/gbdt-symmetric-1000`, `trees/gbdt-ordered` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T06 Apple leaf-statistic partial budget

_afcl_leaf_stats_sm maps the hardware-derived statistic scheduling budget to ceil(sm/2). Both scratch sizing and all one-/two-stat gathered/non-gathered leaf-statistic launches use the same budget; remaining solver/work kernels use the original sm.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/methods/leaves_estimation/apple_fast_est.mojo](../gbdt/methods/leaves_estimation/apple_fast_est.mojo).
- Affected callers: Apple fast pointwise boosted-tree leaf estimation; Weighted/unweighted regression and binary-classification objectives admitted by apple_est_handles.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T06`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T06/manifest.json](../experiments/performance_ideas/AFCL-T06/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-symmetric`, `trees/gbdt-depthwise`, `trees/gbdt-lossguide` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T07 Resident boosted-tree prediction blocks

Resident oblivious model application uses AFCL_PREDICT_BLOCK=128 instead of 256 for both ceil-divided row grids and launches, including ordinary per-tree/four-tree and packed all-tree applications. Existing bounded grid stride covers every row.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/resident_model.mojo](../gbdt/resident_model.mojo).
- Affected callers: Resident GradientBoosting/CatBoost-style predict; Resident boosted-tree predict_proba/link callers.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T07`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T07/manifest.json](../experiments/performance_ideas/AFCL-T07/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-symmetric`, `trees/gbdt-symmetric-1000`, `trees/gbdt-depthwise`, `trees/gbdt-lossguide`, `trees/gbdt-ordered`, `trees/gbdt-categorical` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T08 RF traversal workgroup width

RF_INPUT specializations use 64 instead of 128 threads per traversal group. Ordered and row-owned groves use matching ceil-divided grids; lane-owned groves use two rather than four rows, with matching shared row staging and partial-buffer layouts. Logical 32-tree grove reductions are retained.

- Implementation files: [core/forest_inference.mojo](../core/forest_inference.mojo).
- Affected callers: RandomForestClassifier GPU inference; RandomForestRegressor GPU inference.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T08`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T08/manifest.json](../experiments/performance_ideas/AFCL-T08/manifest.json).
- Binding map: `rf`, `trees`, `x_trees` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/rf` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T09 ExtraTrees sampled-feature tiles

ET_FEATURE_TILE is eight instead of sixteen for tiled range searches and tiled regression-score kernels, including their ceil-divided feature grids. Selected feature slots and RNG/threshold assignment do not change.

- Implementation files: [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).
- Affected callers: ExtraTreesClassifier fit; ExtraTreesRegressor fit; RandomTreesEmbedding or other ExtraTrees-builder callers reaching tiled range search.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T09`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T09/manifest.json](../experiments/performance_ideas/AFCL-T09/manifest.json).
- Binding map: `trees`, `x_trees` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/et`, `algos/random-trees-embedding` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T10 IsolationForest query blocks

IF_PATH_TPB defaults to 128 instead of 256; IFLaunchKnobs.default and existing path/score launchers consume it, preserving per-row traversal and correction arithmetic.

- Implementation files: [isolation_forest/impl/isolation_tree_builder.mojo](../isolation_forest/impl/isolation_tree_builder.mojo).
- Affected callers: IsolationForest score_samples; IsolationForest decision_function/predict; IsolationForest path_lengths and fit-time contamination scoring using default launch knobs.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T10`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T10/manifest.json](../experiments/performance_ideas/AFCL-T10/manifest.json).
- Binding map: `svm` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/iforest` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T11 TreeSHAP query fold block width

QUERY_TPB=64 instead of 128 is used by query grids for direct tree units, table row units, decision-word table row units and final contribution fold. Preparation and table construction retain TPB=128.

- Implementation files: [xtrees/shap_device.mojo](../xtrees/shap_device.mojo).
- Affected callers: GPU TreeExplainer/TreeSHAP values for classical tree models.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T11`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T11/manifest.json](../experiments/performance_ideas/AFCL-T11/manifest.json).
- Binding map: `trees`, `x_trees` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/tree-shap`, `public/tree-shap-full` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-T12 CTR elementwise count preprocessing blocks

CTR_BLOCK_SIZE=128 instead of 256 changes the existing elementwise/calculation launchers and their matching per-thread row strides. CTR_DOCS_PER_THREAD stays four.

- Implementation files: [gbdt/apple_fast_classical.mojo](../gbdt/apple_fast_classical.mojo); [gbdt/ctrs/kernel/ctr_calcers.mojo](../gbdt/ctrs/kernel/ctr_calcers.mojo).
- Affected callers: Categorical boosted-tree CTR preparation; Categorical boosted-tree inference/preparation callers actually using ctr_calcers launchers.
- A defines: none; A environment: none.
- B defines: `MOJOLEARN_AFCL_T12`; B environment: none.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); registration: [experiments/performance_ideas/AFCL-T12/manifest.json](../experiments/performance_ideas/AFCL-T12/manifest.json).
- Binding map: `gbdt` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `trees/gbdt-categorical` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

### Preprocessing metrics and forecasting

#### AFCL-P01 Scaler reduction independent accumulators

Two independent per-lane FP32 sums in the two-pass col_stats path; two independent Welford chains followed by existing Chan merge in the current SI_ONEPASS/fused-transform cs_tile path.

- Implementation files: [x_prep/fastred.mojo](../x_prep/fastred.mojo); [x_prep/fastpt.mojo](../x_prep/fastpt.mojo).
- Affected callers: SimpleImputer and PowerTransformer column-statistic consumers; any OP_COL_STATS caller including scaler programs..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P01`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P01/manifest.json](../experiments/performance_ideas/AFCL-P01/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/simple-imputer`, `algos/power-transformer`, `algos/standard-scaler`, `algos/minmax-scaler`, `public/column-statistics` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P02 MaxAbs row tiling

MaxAbs direct-fit row chunks 1024/128 become 512/64; feature-width policy is untouched.

- Implementation files: [x_prep/fastmaxabs.mojo](../x_prep/fastmaxabs.mojo).
- Affected callers: MaxAbsScaler direct FAST fit..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P02`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P02/manifest.json](../experiments/performance_ideas/AFCL-P02/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/maxabs-scaler` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P03 Categorical Naive Bayes count grouping

Categorical count launch uses 64 threads instead of 128; final integer-count conversion launch is unchanged.

- Implementation files: [x_prep/fastnb.mojo](../x_prep/fastnb.mojo); [x_prep/device.mojo](../x_prep/device.mojo).
- Affected callers: CategoricalNB unweighted categorical atomic count path..
- A defines: `MOJOLEARN_NB_CAT_ATOMIC`; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_NB_CAT_ATOMIC`, `MOJOLEARN_AFCL_P03`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P03/manifest.json](../experiments/performance_ideas/AFCL-P03/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/categorical-nb` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P04 Sparse Naive Bayes row grouping

CSR histogram groups contain 32 rows instead of 64; the 256-thread block and all nonzero processing remain.

- Implementation files: [x_prep/fastnb_csr.mojo](../x_prep/fastnb_csr.mojo).
- Affected callers: MultinomialNB and ComplementNB CSR count fitting..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P04`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P04/manifest.json](../experiments/performance_ideas/AFCL-P04/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/multinomial-nb`, `algos/complement-nb` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P05 Target-encoding work grouping

TargetEncoder global/category folds use independent TE_TGR=128 rather than shared TGR=256, including scratch, strides, tree fold and launch size.

- Implementation files: [x_prep/fastprep2.mojo](../x_prep/fastprep2.mojo).
- Affected callers: TargetEncoder global and gathered-category fold stages..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P05`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P05/manifest.json](../experiments/performance_ideas/AFCL-P05/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/target-encoder` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P06 Iterative-imputer convergence fold

Existing imputer row-max convergence reduction uses 512 rather than 256 lanes, with matching shared array, stride and launch.

- Implementation files: [x_prep/fastprep2.mojo](../x_prep/fastprep2.mojo).
- Affected callers: IterativeImputer II_CONV when precomputed row sums exist..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`, `MOJOLEARN_X_PREP_FAST_II_CONV=1`.
- B defines: `MOJOLEARN_AFCL_P06`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`, `MOJOLEARN_X_PREP_FAST_II_CONV=1`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P06/manifest.json](../experiments/performance_ideas/AFCL-P06/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/iterative-imputer` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P07 Feature-selection row tile

Feature statistic row tiles use RPT=32 rather than 64, RT=256 rather than 512, retaining the 32-column by 8-row-lane block.

- Implementation files: [x_prep/select_fast.mojo](../x_prep/select_fast.mojo).
- Affected callers: f_regression, r_regression, f_classif and applicable selectors..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P07`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P07/manifest.json](../experiments/performance_ideas/AFCL-P07/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/select-f-classif`, `algos/select-f-regression`, `algos/select-r-regression`, `public/selector-scores` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P08 Regression metric independent accumulators

Two independent software-binary64 accumulation chains in uniform and weighted multioutput regression score averaging.

- Implementation files: [x_metrics/reg_epi.mojo](../x_metrics/reg_epi.mojo).
- Affected callers: Regression metric KIND_AVG epilogues for multiple outputs..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P08`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P08/manifest.json](../experiments/performance_ideas/AFCL-P08/manifest.json).
- Binding map: `x_metrics` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `public/regression-metrics` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P09 Ranking metric prefix work grouping

Weighted/unweighted curve-prefix chunk size 1024 becomes 512 in the planner and emitted stage parameters.

- Implementation files: [x_metrics/plan.mojo](../x_metrics/plan.mojo).
- Affected callers: ROC/PR/AUC curve callers and weighted-percentile consumers of CURVE_CHUNK..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P09`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P09/manifest.json](../experiments/performance_ideas/AFCL-P09/manifest.json).
- Binding map: `x_metrics` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `public/ranking-metrics` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P10 Resampling gather tile geometry

Grouped resample gather tile 32 features by 8 rows becomes 16 features by 16 rows, retaining 256 threads.

- Implementation files: [resample/gather_fast.mojo](../resample/gather_fast.mojo); [resample/estimator.mojo](../resample/estimator.mojo).
- Affected callers: Public float32 resample GPU grouped-gather path..
- A defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`, `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER`; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`, `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER`, `MOJOLEARN_AFCL_P10`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P10/manifest.json](../experiments/performance_ideas/AFCL-P10/manifest.json).
- Binding map: `resample` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/resample`, `public/resample-full` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P11 Stationarity series reduction schedule

Two independent per-thread chains for series sums/squared sums followed by the existing STATS_TPB block reduction.

- Implementation files: [tsa/impl/timeSeries/stationarity.mojo](../tsa/impl/timeSeries/stationarity.mojo).
- Affected callers: KPSS and AutoARIMA differencing selection using series_sum_kernel..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P11`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P11/manifest.json](../experiments/performance_ideas/AFCL-P11/manifest.json).
- Binding map: `tsa`, `arima` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/kpss`, `algos/select-d`, `algos/autoarima` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P12 Batched ARIMA likelihood work grouping

Kalman series block default 32 becomes 64; propagated through public likelihood/forecast helpers, packed likelihood, optimizer and retained-evaluation workspace entrances.

- Implementation files: [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo); [arima/impl/batched_arima.mojo](../arima/impl/batched_arima.mojo); [arima/impl/fast_eval_ws.mojo](../arima/impl/fast_eval_ws.mojo); [arima/estimator.mojo](../arima/estimator.mojo).
- Affected callers: ARIMA and AutoARIMA scalar-lane Kalman evaluation/forecast routes..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P12`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P12/manifest.json](../experiments/performance_ideas/AFCL-P12/manifest.json).
- Binding map: `arima` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/arima`, `algos/autoarima` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P13 Gaussian Naive Bayes class-statistic chains

Two independent per-lane chains for unweighted class-filtered sums and centered squared deviations; existing count and reduction tree remain.

- Implementation files: [x_prep/fastred.mojo](../x_prep/fastred.mojo).
- Affected callers: GaussianNB and unweighted OP_CLASS_STATS consumers..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P13`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P13/manifest.json](../experiments/performance_ideas/AFCL-P13/manifest.json).
- Binding map: `x_prep` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `algos/gaussian-nb` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

#### AFCL-P14 Holt-Winters series launch geometry

Default optimizer group width HW_OPTIM_TPB changes 128 to 64; existing callers continue using the shared default.

- Implementation files: [holtwinters/impl/internal/hw_utils.mojo](../holtwinters/impl/internal/hw_utils.mojo).
- Affected callers: HoltWinters fitting and applicable optimizer/evaluation helpers that use HW_OPTIM_TPB..
- A defines: none; A environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- B defines: `MOJOLEARN_AFCL_P14`; B environment: `MOJOLEARN_NUMERIC_MODE=fast`.
- Controls and limitations: [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); registration: [experiments/performance_ideas/AFCL-P14/manifest.json](../experiments/performance_ideas/AFCL-P14/manifest.json).
- Binding map: `tsa` in [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json). The builder can widen this map through imported modules.
- Workload recipe IDs: `classical2/ets` in [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json).

## Integration files for all new cards

| Purpose | Files |
| --- | --- |
| Original hypotheses and per-card quality questions | [experiments/apple_fast_classical_20261006/README.md](../experiments/apple_fast_classical_20261006/README.md); [experiments/apple_fast_classical_20261006/ideas.json](../experiments/apple_fast_classical_20261006/ideas.json) |
| Canonical arm controls and exact implementation notes | [experiments/apple_fast_classical_20261006/lanes/linear.json](../experiments/apple_fast_classical_20261006/lanes/linear.json); [experiments/apple_fast_classical_20261006/lanes/linear.md](../experiments/apple_fast_classical_20261006/lanes/linear.md); [experiments/apple_fast_classical_20261006/lanes/geometry.json](../experiments/apple_fast_classical_20261006/lanes/geometry.json); [experiments/apple_fast_classical_20261006/lanes/geometry.md](../experiments/apple_fast_classical_20261006/lanes/geometry.md); [experiments/apple_fast_classical_20261006/lanes/trees.json](../experiments/apple_fast_classical_20261006/lanes/trees.json); [experiments/apple_fast_classical_20261006/lanes/trees.md](../experiments/apple_fast_classical_20261006/lanes/trees.md); [experiments/apple_fast_classical_20261006/lanes/preprocessing.json](../experiments/apple_fast_classical_20261006/lanes/preprocessing.json); [experiments/apple_fast_classical_20261006/lanes/preprocessing.md](../experiments/apple_fast_classical_20261006/lanes/preprocessing.md) |
| Shared catalog and source-only arm selection | [tools/performance_ideas.py](../tools/performance_ideas.py); [experiments/apple_fast_classical_20261006/select.py](../experiments/apple_fast_classical_20261006/select.py) |
| Paired native package build source and binding coverage | [experiments/apple_fast_classical_20261006/build_pair.py](../experiments/apple_fast_classical_20261006/build_pair.py); [experiments/apple_fast_classical_20261006/build_bindings.json](../experiments/apple_fast_classical_20261006/build_bindings.json); [experiments/apple_fast_classical_20261006/BUILD_INTEGRATION.md](../experiments/apple_fast_classical_20261006/BUILD_INTEGRATION.md) |
| Full-workload mappings and paired execution source | [experiments/apple_fast_classical_20261006/run_pair.py](../experiments/apple_fast_classical_20261006/run_pair.py); [experiments/apple_fast_classical_20261006/full_workloads.json](../experiments/apple_fast_classical_20261006/full_workloads.json); [experiments/apple_fast_classical_20261006/WORKLOAD_INTEGRATION.md](../experiments/apple_fast_classical_20261006/WORKLOAD_INTEGRATION.md) |
| Production reach, prerequisites and integration contract | [experiments/apple_fast_classical_20261006/INTEGRATION_SCOPE.md](../experiments/apple_fast_classical_20261006/INTEGRATION_SCOPE.md); [experiments/apple_fast_classical_20261006/INTEGRATION.md](../experiments/apple_fast_classical_20261006/INTEGRATION.md) |
| Retained evidence export and board input definitions | [experiments/apple_fast_classical_20261006/export_measurements.py](../experiments/apple_fast_classical_20261006/export_measurements.py); [experiments/apple_fast_classical_20261006/measurement_inventory.json](../experiments/apple_fast_classical_20261006/measurement_inventory.json); [experiments/apple_fast_classical_20261006/measurement_index.json](../experiments/apple_fast_classical_20261006/measurement_index.json); [tools/performance_measurement_board.py](../tools/performance_measurement_board.py) |
| Source delivery handoff | [experiments/apple_fast_classical_20261006/DELIVERY.md](../experiments/apple_fast_classical_20261006/DELIVERY.md) |

G01 and G02 are mutually exclusive routes. Other combinations and complete proposed defaults require their own later qualification; isolated card results cannot establish combination quality. No combination is enabled by this index.

## Earlier registered A B experiments

The following 60 manifests predate AFCL. Each manifest contains its precise arm controls, entry points, quality gates and related evidence. File lists below reproduce `implementation_paths`; a listed check or adapter file is part of the experiment surface, not necessarily a production kernel. Status is the saved manifest status, not a new qualification. See [experiments/performance_ideas/IMPLEMENTATION_STATUS.md](../experiments/performance_ideas/IMPLEMENTATION_STATUS.md) for historical implementation notes and [docs/plans/PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md](../docs/plans/PERFORMANCE_EXPERIMENT_IDEAS_2026-10-05.md) for the initial ideas.

### Earlier Apple FAST cards

| ID and arm manifest | Experiment | Scope | Implementation and experiment files | Recorded status |
| --- | --- | --- | --- | --- |
| [F01](../experiments/performance_ideas/F01/manifest.json) | Actual PCA caller GEMM geometry | fast; apple | [experiments/performance_ideas/F01/caller.py](../experiments/performance_ideas/F01/caller.py); [experiments/performance_ideas/apple_fast/pair.py](../experiments/performance_ideas/apple_fast/pair.py); [experiments/apple_fast/gemm/scoped_dispatch.mojo](../experiments/apple_fast/gemm/scoped_dispatch.mojo) | `source_ready` |
| [F02](../experiments/performance_ideas/F02/manifest.json) | Share GEMM infrastructure through independent fused and unfused caller adapters | fast; apple | [experiments/apple_fast/gemm/scoped_dispatch.mojo](../experiments/apple_fast/gemm/scoped_dispatch.mojo); [x_decomp/lu_fast_mma.mojo](../x_decomp/lu_fast_mma.mojo); [experiments/performance_ideas/F02/caller.py](../experiments/performance_ideas/F02/caller.py); [cholesky/checks/fast_trsm.mojo](../cholesky/checks/fast_trsm.mojo); [bindings/_mojolearn_gp.mojo](../bindings/_mojolearn_gp.mojo); [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [core/gemm.mojo](../core/gemm.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo); [bindings/_mojolearn_scoped_gemm_probe.mojo](../bindings/_mojolearn_scoped_gemm_probe.mojo) | `source_ready` |
| [F03](../experiments/performance_ideas/F03/manifest.json) | Actual LM bounded resident session versus stateless calls | fast; apple; existing neural or training scope | [experiments/performance_ideas/F03/caller.py](../experiments/performance_ideas/F03/caller.py); [experiments/performance_ideas/apple_fast/lm_task.py](../experiments/performance_ideas/apple_fast/lm_task.py); [bindings/_mojolearn_byte_lm.mojo](../bindings/_mojolearn_byte_lm.mojo) | `source_ready` |
| [F04](../experiments/performance_ideas/F04/manifest.json) | Pair independent gather completions with owned buffers | fast; apple | [resample/estimator.mojo](../resample/estimator.mojo); [experiments/performance_ideas/F04/caller.py](../experiments/performance_ideas/F04/caller.py) | `source_ready` |
| [F05](../experiments/performance_ideas/F05/manifest.json) | Narrow softmax at full optimizer and line-search caller | fast; apple | [experiments/performance_ideas/F05/caller.py](../experiments/performance_ideas/F05/caller.py); [experiments/apple_fast/gemm/softmax_narrow.mojo](../experiments/apple_fast/gemm/softmax_narrow.mojo) | `source_ready` |
| [F06](../experiments/performance_ideas/F06/manifest.json) | Independent MCD batch bounds, active-candidate compaction and exact-support covariance reuse | fast; apple | [x_decomp/mcd_bmma.mojo](../x_decomp/mcd_bmma.mojo); [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [experiments/performance_ideas/F06/caller.py](../experiments/performance_ideas/F06/caller.py); [x_decomp/mcd_fast.mojo](../x_decomp/mcd_fast.mojo); [x_decomp/mcd_experiments.mojo](../x_decomp/mcd_experiments.mojo); [experiments/performance_ideas/F06/native_check.mojo](../experiments/performance_ideas/F06/native_check.mojo) | `source_ready` |
| [F07](../experiments/performance_ideas/F07/manifest.json) | FLASH and true GQA caller qualification with reach counters | fast; apple; existing neural or training scope | [transformer/impl/llama/afn_apple_fast.mojo](../transformer/impl/llama/afn_apple_fast.mojo); [bindings/_mojolearn_transformer.mojo](../bindings/_mojolearn_transformer.mojo); [experiments/performance_ideas/F07/caller.py](../experiments/performance_ideas/F07/caller.py) | `source_ready` |
| [F08](../experiments/performance_ideas/F08/manifest.json) | Independent LM backward fusion and view A/B training task | fast; apple; existing neural or training scope | [experiments/performance_ideas/F08/caller.py](../experiments/performance_ideas/F08/caller.py); [experiments/performance_ideas/apple_fast/lm_task.py](../experiments/performance_ideas/apple_fast/lm_task.py); [training/byte_lm_afn.mojo](../training/byte_lm_afn.mojo); [training/byte_lm_afn_grad.mojo](../training/byte_lm_afn_grad.mojo) | `source_ready` |
| [F09](../experiments/performance_ideas/F09/manifest.json) | Memory-bounded LM head in a fixed actual SGD task | fast; apple; existing neural or training scope | [experiments/performance_ideas/F09/caller.py](../experiments/performance_ideas/F09/caller.py); [training/chunked_lm_head_v2.mojo](../training/chunked_lm_head_v2.mojo); [training/afn_optim.mojo](../training/afn_optim.mojo) | `source_ready` |
| [F10](../experiments/performance_ideas/F10/manifest.json) | Independent SSD MMA and Mamba fusion caller arms | fast; apple; existing neural or training scope | [experiments/performance_ideas/F10/caller.py](../experiments/performance_ideas/F10/caller.py); [mamba/impl/modules/afn_ssd_mma.mojo](../mamba/impl/modules/afn_ssd_mma.mojo); [mamba/impl/modeling/afn_mamba1_fused.mojo](../mamba/impl/modeling/afn_mamba1_fused.mojo) | `source_ready` |
| [F11](../experiments/performance_ideas/F11/manifest.json) | Apple FAST opt-in TSQR norm and grid scheduling | fast; apple | [x_decomp/tsqr_device.mojo](../x_decomp/tsqr_device.mojo); [experiments/performance_ideas/F11/caller.py](../experiments/performance_ideas/F11/caller.py); [decomposition/impl/linalg/detail/pca.mojo](../decomposition/impl/linalg/detail/pca.mojo); [bindings/_mojolearn_estimators.mojo](../bindings/_mojolearn_estimators.mojo) | `source_ready` |
| [F12](../experiments/performance_ideas/F12/manifest.json) | Qualify resident boosting partition and leaf-input reuse | fast; apple | [experiments/performance_ideas/F12/caller.py](../experiments/performance_ideas/F12/caller.py); [gbdt/methods/sym_iter_fast.mojo](../gbdt/methods/sym_iter_fast.mojo); [gbdt/methods/leaves_estimation/apple_fast_est.mojo](../gbdt/methods/leaves_estimation/apple_fast_est.mojo); [gbdt/methods/ordered_fast_switches.mojo](../gbdt/methods/ordered_fast_switches.mojo); [gbdt/methods/pointwise_scores_calcer.mojo](../gbdt/methods/pointwise_scores_calcer.mojo); [gbdt/methods/pointwise_kernels.mojo](../gbdt/methods/pointwise_kernels.mojo); [gbdt/methods/kernel/compute_point_hist2_loop.mojo](../gbdt/methods/kernel/compute_point_hist2_loop.mojo) | `source_ready` |
| [F13](../experiments/performance_ideas/F13/manifest.json) | FAST-only shared row forest layout at divergent public callers | fast; apple | [core/forest_inference.mojo](../core/forest_inference.mojo); [experiments/performance_ideas/F13/caller.py](../experiments/performance_ideas/F13/caller.py); [xtrees/shap_device.mojo](../xtrees/shap_device.mojo); [bindings/_mojolearn_trees.mojo](../bindings/_mojolearn_trees.mojo); [bindings/_mojolearn_x_trees.mojo](../bindings/_mojolearn_x_trees.mojo) | `source_ready` |
| [F14](../experiments/performance_ideas/F14/manifest.json) | Compare exact grouped query rows and fixed-probe bounded IVF list tasks | fast; apple | [neighbors/impl/detail/fast_topk_knn.mojo](../neighbors/impl/detail/fast_topk_knn.mojo); [experiments/performance_ideas/F14/caller.py](../experiments/performance_ideas/F14/caller.py); [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo); [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo); [experiments/performance_ideas/F14/ann_caller.py](../experiments/performance_ideas/F14/ann_caller.py) | `source_ready` |
| [F15](../experiments/performance_ideas/F15/manifest.json) | Independent mini-batch labeling and bounded stopping work | fast; apple | [x_cluster/minibatch_fast.mojo](../x_cluster/minibatch_fast.mojo); [experiments/performance_ideas/F15/caller.py](../experiments/performance_ideas/F15/caller.py); [dbscan/estimator.mojo](../dbscan/estimator.mojo); [hdbscan/impl/detail/fast_apple.mojo](../hdbscan/impl/detail/fast_apple.mojo) | `source_ready` |
| [F16](../experiments/performance_ideas/F16/manifest.json) | Integrate compensated production Kalman tail with independent gradient prerequisite | fast; apple | [arima/impl/fast_eval_ws.mojo](../arima/impl/fast_eval_ws.mojo); [arima/impl/fast_eval_df.mojo](../arima/impl/fast_eval_df.mojo); [arima/checks/product_df_quality_check.mojo](../arima/checks/product_df_quality_check.mojo); [experiments/performance_ideas/F16/caller.py](../experiments/performance_ideas/F16/caller.py); [experiments/performance_ideas/apple_fast/arima_task.py](../experiments/performance_ideas/apple_fast/arima_task.py) | `source_ready` |
| [F17](../experiments/performance_ideas/F17/manifest.json) | Actual panel search A/B for stacked candidates and held evaluation state | fast; apple | [experiments/performance_ideas/F17/caller.py](../experiments/performance_ideas/F17/caller.py); [experiments/performance_ideas/apple_fast/arima_task.py](../experiments/performance_ideas/apple_fast/arima_task.py); [arima/impl/batched_arima.mojo](../arima/impl/batched_arima.mojo); [arima/impl/fast_eval_ws.mojo](../arima/impl/fast_eval_ws.mojo) | `source_ready` |
| [F18](../experiments/performance_ideas/F18/manifest.json) | Resident KDE immutable input ownership and direct preparation | fast; apple | [kde/resident_fit.mojo](../kde/resident_fit.mojo); [python/mojolearn/density.py](../python/mojolearn/density.py); [bindings/_mojolearn_estimators.mojo](../bindings/_mojolearn_estimators.mojo); [experiments/performance_ideas/F18/caller.py](../experiments/performance_ideas/F18/caller.py) | `source_ready` |
| [F19](../experiments/performance_ideas/F19/manifest.json) | Coalesced tiled gather with GPU-generated indices and direct owned outputs | fast; apple | [resample/gather_fast.mojo](../resample/gather_fast.mojo); [resample/estimator.mojo](../resample/estimator.mojo); [experiments/performance_ideas/F19/caller.py](../experiments/performance_ideas/F19/caller.py); [experiments/performance_ideas/F04/caller.py](../experiments/performance_ideas/F04/caller.py); [bindings/_mojolearn_resample.mojo](../bindings/_mojolearn_resample.mojo); [python/mojolearn/resample.py](../python/mojolearn/resample.py) | `source_ready` |
| [F20](../experiments/performance_ideas/F20/manifest.json) | Task-level optimizer fusion and FAST blocked LayerNorm qualification | fast; apple; existing neural or training scope | [sequence/layernorm.mojo](../sequence/layernorm.mojo); [training/afn_optim.mojo](../training/afn_optim.mojo); [experiments/performance_ideas/F20/caller.py](../experiments/performance_ideas/F20/caller.py) | `source_ready` |

### Shared IDENTICAL cards

| ID and arm manifest | Experiment | Scope | Implementation and experiment files | Recorded status |
| --- | --- | --- | --- | --- |
| [I01](../experiments/performance_ideas/I01/manifest.json) | Attribute existing GEMM schedules | identical; nvidia, amd, apple | [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo); [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py) | `build_passed` |
| [I02](../experiments/performance_ideas/I02/manifest.json) | Bound session GEMM scratch retention | identical; nvidia, amd, apple | [gemm/experiments/bounded_workspace.mojo](../gemm/experiments/bounded_workspace.mojo); [gemm/experiments/bounded_workspace_check.mojo](../gemm/experiments/bounded_workspace_check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [I03](../experiments/performance_ideas/I03/manifest.json) | Batch independent products on separate grid jobs | identical; nvidia, amd, apple | [gemm/experiments/grouped_jobs.mojo](../gemm/experiments/grouped_jobs.mojo); [gemm/experiments/grouped_jobs_check.mojo](../gemm/experiments/grouped_jobs_check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [I04](../experiments/performance_ideas/I04/manifest.json) | Qualify a coherently shared opt-in 64-element IDENTICAL leaf version | identical; nvidia, amd, apple | [gemm/contract.mojo](../gemm/contract.mojo); [gemm/experiments/fold_profile_probe.mojo](../gemm/experiments/fold_profile_probe.mojo); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/experiments/profile_identity_check.mojo](../gemm/experiments/profile_identity_check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [I05](../experiments/performance_ideas/I05/manifest.json) | Fuse bias after the explicit rounded product seam | identical; nvidia, amd, apple | [gemm/experiments/rounded_epilogue.mojo](../gemm/experiments/rounded_epilogue.mojo); [gemm/experiments/rounded_epilogue_check.mojo](../gemm/experiments/rounded_epilogue_check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [I06](../experiments/performance_ideas/I06/manifest.json) | qualify attention KV grid reuse across GQA tails | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I06/check.mojo](../experiments/performance_ideas/I06/check.mojo); [transformer/impl/llama/fused_attention.mojo](../transformer/impl/llama/fused_attention.mojo) | `source_ready` |
| [I07](../experiments/performance_ideas/I07/manifest.json) | exercise retained and recomputed attention backward lifetimes | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I07/check.mojo](../experiments/performance_ideas/I07/check.mojo); [experiments/performance_ideas/I07/state_cost.mojo](../experiments/performance_ideas/I07/state_cost.mojo) | `source_ready` |
| [I08](../experiments/performance_ideas/I08/manifest.json) | attribute SSD tile reuse on multiple state cases and chunk tails | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I08/check.mojo](../experiments/performance_ideas/I08/check.mojo); [mamba/impl/modules/ssd_minimal.mojo](../mamba/impl/modules/ssd_minimal.mojo); [experiments/performance_ideas/I08/backward_check.mojo](../experiments/performance_ideas/I08/backward_check.mojo); [mamba/impl/modules/mamba2_prefill_backward.mojo](../mamba/impl/modules/mamba2_prefill_backward.mojo) | `source_ready` |
| [I09](../experiments/performance_ideas/I09/manifest.json) | gate token-parallel recurrence on prefix and decode boundaries | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I09/check.mojo](../experiments/performance_ideas/I09/check.mojo) | `source_ready` |
| [I10](../experiments/performance_ideas/I10/manifest.json) | reduce ordered training status from per-tile contributions | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I10/check.mojo](../experiments/performance_ideas/I10/check.mojo); [training/checks/optimizer.mojo](../training/checks/optimizer.mojo); [training/byte_lm.mojo](../training/byte_lm.mojo) | `source_ready` |
| [I11](../experiments/performance_ideas/I11/manifest.json) | qualify canonical radix embedding updates under skew and vocabulary reuse | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/I11/check.mojo](../experiments/performance_ideas/I11/check.mojo) | `source_ready` |
| [I12](../experiments/performance_ideas/I12/manifest.json) | gate solver pass fusion on accepted iterates and refusal semantics | identical; nvidia, amd, apple | [experiments/performance_ideas/I12/check.mojo](../experiments/performance_ideas/I12/check.mojo); [glm/impl/qn/glm_base.mojo](../glm/impl/qn/glm_base.mojo); [glm/impl/qn/qn_linesearch.mojo](../glm/impl/qn/qn_linesearch.mojo); [experiments/performance_ideas/I12/trials_check.mojo](../experiments/performance_ideas/I12/trials_check.mojo); [experiments/performance_ideas/I12/sgd_check.mojo](../experiments/performance_ideas/I12/sgd_check.mojo); [x_linear/device.mojo](../x_linear/device.mojo); [x_linear/sgd.mojo](../x_linear/sgd.mojo) | `source_ready` |
| [I13](../experiments/performance_ideas/I13/manifest.json) | implement compact degree buckets for bounded RBC stable merges | identical; nvidia, amd, apple | [experiments/performance_ideas/I13/check.mojo](../experiments/performance_ideas/I13/check.mojo); [neighbors/checks/ball_cover_canonical_order.mojo](../neighbors/checks/ball_cover_canonical_order.mojo); [neighbors/checks/rbc_canonical_merge_check.mojo](../neighbors/checks/rbc_canonical_merge_check.mojo) | `source_ready` |
| [I14](../experiments/performance_ideas/I14/manifest.json) | qualify gated convergence at first fixed point on graph topologies | identical; nvidia, amd, apple | [experiments/performance_ideas/I14/check.mojo](../experiments/performance_ideas/I14/check.mojo); [experiments/performance_ideas/I14/hdbscan_stages.mojo](../experiments/performance_ideas/I14/hdbscan_stages.mojo) | `source_ready` |
| [I15](../experiments/performance_ideas/I15/manifest.json) | qualify streaming exact neighbor merge at ties and batch tails | identical; nvidia, amd, apple | [experiments/performance_ideas/I15/check.mojo](../experiments/performance_ideas/I15/check.mojo); [neighbors/impl/detail/knn_brute_force.mojo](../neighbors/impl/detail/knn_brute_force.mojo); [experiments/performance_ideas/I15/certified_caller.mojo](../experiments/performance_ideas/I15/certified_caller.mojo) | `source_ready` |
| [I16](../experiments/performance_ideas/I16/manifest.json) | attribute IVF grouped and staged scans under skewed list occupancy | identical; nvidia, amd, apple | [experiments/performance_ideas/I16/check.mojo](../experiments/performance_ideas/I16/check.mojo); [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo); [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo) | `source_ready` |
| [I17](../experiments/performance_ideas/I17/manifest.json) | qualify loss-guide frontier retention at varying live leaf capacities | identical; nvidia, amd, apple | [experiments/performance_ideas/I17/check.mojo](../experiments/performance_ideas/I17/check.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [experiments/performance_ideas/I17/caller_check.mojo](../experiments/performance_ideas/I17/caller_check.mojo) | `source_ready` |
| [I18](../experiments/performance_ideas/I18/manifest.json) | Retain bounded exact forest histograms and subtract compatible siblings | identical; nvidia, amd, apple | [experiments/performance_ideas/I18/check.mojo](../experiments/performance_ideas/I18/check.mojo); [ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo](../ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo); [ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo](../ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo); [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo); [experiments/performance_ideas/I18/forest_check.mojo](../experiments/performance_ideas/I18/forest_check.mojo) | `source_ready` |
| [I19](../experiments/performance_ideas/I19/manifest.json) | qualify reusable stable radix scratch and bounded key widths | identical; nvidia, amd, apple | [experiments/performance_ideas/I19/check.mojo](../experiments/performance_ideas/I19/check.mojo); [experiments/performance_ideas/I19/ragged_float.mojo](../experiments/performance_ideas/I19/ragged_float.mojo); [experiments/performance_ideas/I19/float_check.mojo](../experiments/performance_ideas/I19/float_check.mojo); [x_prep/ragged_quantile.mojo](../x_prep/ragged_quantile.mojo); [experiments/performance_ideas/I19/quantile_check.mojo](../experiments/performance_ideas/I19/quantile_check.mojo); [x_prep/ragged_categories.mojo](../x_prep/ragged_categories.mojo); [x_prep/ragged_select.mojo](../x_prep/ragged_select.mojo); [core/stable_radix_digits.mojo](../core/stable_radix_digits.mojo); [experiments/performance_ideas/I19/categories_check.mojo](../experiments/performance_ideas/I19/categories_check.mojo); [experiments/performance_ideas/I19/select_check.mojo](../experiments/performance_ideas/I19/select_check.mojo) | `source_ready` |
| [I20](../experiments/performance_ideas/I20/manifest.json) | Retain bounded source-owned KDE partial storage under the existing chunked fold | identical; nvidia, amd, apple | [experiments/performance_ideas/I20/check.mojo](../experiments/performance_ideas/I20/check.mojo); [kde/impl/chunk_workspace.mojo](../kde/impl/chunk_workspace.mojo); [kde/impl/neighbors/kernel_density.mojo](../kde/impl/neighbors/kernel_density.mojo); [kde/resident_fit.mojo](../kde/resident_fit.mojo); [kde/checks/reused_workspace_check.mojo](../kde/checks/reused_workspace_check.mojo) | `build_passed` |
| [I21](../experiments/performance_ideas/I21/manifest.json) | gate fused GMM components on complete EM state and likelihood | identical; nvidia, amd, apple | [experiments/performance_ideas/I21/check.mojo](../experiments/performance_ideas/I21/check.mojo); [mixture/checks/estep.mojo](../mixture/checks/estep.mojo); [experiments/performance_ideas/I21/component_check.mojo](../experiments/performance_ideas/I21/component_check.mojo); [mixture/checks/mstep.mojo](../mixture/checks/mstep.mojo) | `source_ready` |
| [I22](../experiments/performance_ideas/I22/manifest.json) | qualify blocked TSQR factors and retained reflector applies across tails | identical; nvidia, amd, apple | [experiments/performance_ideas/I22/check.mojo](../experiments/performance_ideas/I22/check.mojo); [x_decomp/tsqr_device.mojo](../x_decomp/tsqr_device.mojo); [experiments/performance_ideas/I22/reuse_check.mojo](../experiments/performance_ideas/I22/reuse_check.mojo); [experiments/performance_ideas/I22/campaign.py](../experiments/performance_ideas/I22/campaign.py) | `source_ready` |
| [I23](../experiments/performance_ideas/I23/manifest.json) | qualify independent time-series fits and batch-gradient selection state | identical; nvidia, amd, apple | [experiments/performance_ideas/I23/check.mojo](../experiments/performance_ideas/I23/check.mojo); [arima/impl/batched_arima.mojo](../arima/impl/batched_arima.mojo) | `source_ready` |
| [I24](../experiments/performance_ideas/I24/manifest.json) | fuse resident confusion and PRF input counts in one device pass | identical; nvidia, amd, apple | [experiments/performance_ideas/I24/check.mojo](../experiments/performance_ideas/I24/check.mojo); [metrics/impl/classification_joint.mojo](../metrics/impl/classification_joint.mojo) | `source_ready` |

### AMD IDENTICAL cards

| ID and arm manifest | Experiment | Scope | Implementation and experiment files | Recorded status |
| --- | --- | --- | --- | --- |
| [A01](../experiments/performance_ideas/A01/manifest.json) | Isolate AMD smaller MFMA and band routing | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `source_ready` |
| [A02](../experiments/performance_ideas/A02/manifest.json) | Compare production one/two-page staging and bounded one/two/four-plane resource controls | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo); [experiments/performance_ideas/A02/check.mojo](../experiments/performance_ideas/A02/check.mojo); [gemm/experiments/bounded_staging.mojo](../gemm/experiments/bounded_staging.mojo); [gemm/experiments/bounded_staging_check.mojo](../gemm/experiments/bounded_staging_check.mojo) | `build_passed` |
| [A03](../experiments/performance_ideas/A03/manifest.json) | Vary vector-aligned LDS operand strides | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `source_ready` |
| [A04](../experiments/performance_ideas/A04/manifest.json) | Pair independent logical groups with supported XOR membership | identical; nvidia, amd, apple | [gemm/experiments/subwave_membership.mojo](../gemm/experiments/subwave_membership.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [A05](../experiments/performance_ideas/A05/manifest.json) | Halve the packed body row accumulator live range | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `source_ready` |
| [A06](../experiments/performance_ideas/A06/manifest.json) | Qualify AMD exact fused low-dimensional neighbor selection | identical; nvidia, amd, apple | [experiments/performance_ideas/A06/check.mojo](../experiments/performance_ideas/A06/check.mojo); [neighbors/checks/fused_slot_merge_check.mojo](../neighbors/checks/fused_slot_merge_check.mojo); [neighbors/estimator.mojo](../neighbors/estimator.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [A07](../experiments/performance_ideas/A07/manifest.json) | Qualify bounded degree graph and exact integer node histogram tasks | identical; nvidia, amd, apple | [experiments/performance_ideas/A07/check.mojo](../experiments/performance_ideas/A07/check.mojo); [experiments/performance_ideas/A07/histogram_check.mojo](../experiments/performance_ideas/A07/histogram_check.mojo); [experiments/performance_ideas/A07/histogram_tasks.mojo](../experiments/performance_ideas/A07/histogram_tasks.mojo); [experiments/performance_ideas/N07/streamed_histogram.mojo](../experiments/performance_ideas/N07/streamed_histogram.mojo); [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [neighbors/checks/ball_cover_canonical_order.mojo](../neighbors/checks/ball_cover_canonical_order.mojo); [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo); [experiments/performance_ideas/A07/production_check.mojo](../experiments/performance_ideas/A07/production_check.mojo) | `source_ready` |
| [A08](../experiments/performance_ideas/A08/manifest.json) | Requalify promoted 256-row accumulation with fit interactions | identical; nvidia, amd, apple | [experiments/performance_ideas/A08/check.mojo](../experiments/performance_ideas/A08/check.mojo); [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [cluster/checks/reduce_by_key.mojo](../cluster/checks/reduce_by_key.mojo); [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) | `build_passed` |

### NVIDIA IDENTICAL cards

| ID and arm manifest | Experiment | Scope | Implementation and experiment files | Recorded status |
| --- | --- | --- | --- | --- |
| [N01](../experiments/performance_ideas/N01/manifest.json) | Isolate packed64 body tiles and their interaction | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `source_ready` |
| [N02](../experiments/performance_ideas/N02/manifest.json) | Prove and dispatch a two-level logical fold stack | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `build_passed` |
| [N03](../experiments/performance_ideas/N03/manifest.json) | Pipeline supported asynchronous operand loads with explicit completion | identical; nvidia | [gemm/experiments/async_operand_pipeline.mojo](../gemm/experiments/async_operand_pipeline.mojo); [gemm/experiments/async_operand_pipeline_check.mojo](../gemm/experiments/async_operand_pipeline_check.mojo); [gemm/experiments/async_api_probe.mojo](../gemm/experiments/async_api_probe.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [N04](../experiments/performance_ideas/N04/manifest.json) | Separate contiguous and gather staging with counted caller passes | identical; nvidia, amd, apple | [gemm/experiments/profile_campaign.py](../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | `source_ready` |
| [N05](../experiments/performance_ideas/N05/manifest.json) | Batch compatible launches with fixed-address changing inputs | identical; nvidia, amd, apple | [gemm/experiments/grouped_jobs.mojo](../gemm/experiments/grouped_jobs.mojo); [gemm/experiments/changing_batch_check.mojo](../gemm/experiments/changing_batch_check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [N06](../experiments/performance_ideas/N06/manifest.json) | Keep compact feature gradients in registers without query barriers | identical; nvidia, amd, apple; existing neural or training scope | [experiments/performance_ideas/N06/compact_grad.mojo](../experiments/performance_ideas/N06/compact_grad.mojo); [experiments/performance_ideas/N06/check.mojo](../experiments/performance_ideas/N06/check.mojo); [transformer/impl/llama/attention_v2.mojo](../transformer/impl/llama/attention_v2.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |
| [N07](../experiments/performance_ideas/N07/manifest.json) | Stream bounded feature chunks with exact private integer histograms | identical; nvidia, amd, apple | [experiments/performance_ideas/N07/streamed_histogram.mojo](../experiments/performance_ideas/N07/streamed_histogram.mojo); [experiments/performance_ideas/N07/check.mojo](../experiments/performance_ideas/N07/check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [experiments/performance_ideas/N07/production_check.mojo](../experiments/performance_ideas/N07/production_check.mojo) | `source_ready` |
| [N08](../experiments/performance_ideas/N08/manifest.json) | Fuse canonical radius distance decisions into device CSR scan/fill | identical; nvidia, amd, apple | [experiments/performance_ideas/N08/fused_threshold.mojo](../experiments/performance_ideas/N08/fused_threshold.mojo); [experiments/performance_ideas/N08/check.mojo](../experiments/performance_ideas/N08/check.mojo); [gemm/experiments/native_build.py](../gemm/experiments/native_build.py) | `build_passed` |

## Existing IDENTICAL candidate families and recipes

These explicit recipes are an older selection surface and overlap the shared catalog. The recipe file retains A/B define arrays, required builders, source freezes and route gates. The named group records retain original source references and integration/qualification limits. Runner sources are [tools/select_identical_ab.py](../tools/select_identical_ab.py); [tools/identical_speed_ab.py](../tools/identical_speed_ab.py).

### Named candidate groups

Group definitions: [docs/identical/original-handoff-coverage.json](../docs/identical/original-handoff-coverage.json). Every recipe for each group is linked through the recipe file; the arm IDs are listed in full.

| Group | Family | Recipe IDs in identical_candidate_recipes.json | Controls | Source files | Recorded disposition |
| --- | --- | --- | --- | --- | --- |
| `kmeans_convergence_chunk` | cluster | `kmeans_convergence_chunk/1`, `kmeans_convergence_chunk/2`, `kmeans_convergence_chunk/4`, `kmeans_convergence_chunk/16`, `kmeans_convergence_chunk/32` | `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_2`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_4`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_16`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_32` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `kmeans_accumulator_rows` | cluster | `kmeans_accumulator_rows/256`, `kmeans_accumulator_rows/4096` | `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_4096` | [cluster/checks/reduce_by_key.mojo](../cluster/checks/reduce_by_key.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gmm_estep_stack` | cluster | `gmm_estep_stack/1` | `MOJOLEARN_IDN_GMM_ESTEP_STACK` | [mixture/checks/estep.mojo](../mixture/checks/estep.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `dbscan_cc_chunk` | cluster | `dbscan_cc_chunk/4`, `dbscan_cc_chunk/16`, `dbscan_cc_chunk/32` | `MOJOLEARN_IDN_DBSCAN_CC_CHUNK4`, `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16`, `MOJOLEARN_IDN_DBSCAN_CC_CHUNK32` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `dbscan_cc_split` | cluster | `dbscan_cc_split/4`, `dbscan_cc_split/8`, `dbscan_cc_split/16` | `MOJOLEARN_IDN_DBSCAN_CC_SPLIT4`, `MOJOLEARN_IDN_DBSCAN_CC_SPLIT8`, `MOJOLEARN_IDN_DBSCAN_CC_SPLIT16` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `mst_scan_lanes` | cluster | `mst_scan_lanes/64`, `mst_scan_lanes/128`, `mst_scan_lanes/256` | `MOJOLEARN_IDN_MST_SCAN_LANES` | [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `dendrogram_hook_chunk` | cluster | `dendrogram_hook_chunk/2`, `dendrogram_hook_chunk/8` | `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK` | [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `hdbscan_sparse_threshold` | cluster | `hdbscan_sparse_threshold/4096`, `hdbscan_sparse_threshold/16384` | `MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` | [hdbscan/impl/cluster/detail/single_linkage.mojo](../hdbscan/impl/cluster/detail/single_linkage.mojo); [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `single_linkage_sparse_mst` | cluster | `single_linkage_sparse_mst/4096` | `MOJOLEARN_IDN_SL_SPARSE_MST`, `MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS` | [hierarchy/impl/cluster/detail/single_linkage.mojo](../hierarchy/impl/cluster/detail/single_linkage.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `affinity_convergence_chunk` | cluster | `affinity_convergence_chunk/4`, `affinity_convergence_chunk/16`, `affinity_convergence_chunk/32` | `MOJOLEARN_IDN_AP_CONV_CHUNK4`, `MOJOLEARN_IDN_AP_CONV_CHUNK16`, `MOJOLEARN_IDN_AP_CONV_CHUNK32` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `minibatch_group_size` | cluster | `minibatch_group_size/8`, `minibatch_group_size/32`, `minibatch_group_size/64` | `MOJOLEARN_IDN_MINIBATCH_GROUP8`, `MOJOLEARN_IDN_MINIBATCH_GROUP32`, `MOJOLEARN_IDN_MINIBATCH_GROUP64` | [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `eigh_block_width` | decomp | `eigh_block_width/8`, `eigh_block_width/32` | `MOJOLEARN_IDN_EIGH_BLOCK_B8`, `MOJOLEARN_IDN_EIGH_BLOCK_B32` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) | BLOCK_FAMILY_QUARANTINED_DEFAULT_DISABLED |
| `eigh_block_threshold` | decomp | `eigh_block_threshold/128` | `MOJOLEARN_IDN_EIGH_BLOCK_MIN128` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) | BLOCK_FAMILY_QUARANTINED_DEFAULT_DISABLED |
| `eigh_small_limit` | decomp | `eigh_small_limit/64`, `eigh_small_limit/128` | `MOJOLEARN_IDN_EIGH_SMALL_N64`, `MOJOLEARN_IDN_EIGH_SMALL_N128` | [x_decomp/device.mojo](../x_decomp/device.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `rf_sorted_bootstrap` | forests | `rf_sorted_bootstrap/1` | `MOJOLEARN_IDN_RF_ROWS_SORTED` | [ensemble/randomforest.mojo](../ensemble/randomforest.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `rf_device_level_loop` | forests | `rf_device_level_loop/1`, `rf_device_level_loop/2`, `rf_device_level_loop/4`, `rf_device_level_loop/8` | `MOJOLEARN_IDN_RF_DEVICE_LOOP`, `MOJOLEARN_IDN_RF_DEVICE_LOOP_K1`, `MOJOLEARN_IDN_RF_DEVICE_LOOP_K2`, `MOJOLEARN_IDN_RF_DEVICE_LOOP_K8` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/level_loop_kernels.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/level_loop_kernels.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `extra_trees_u16_bins` | forests | `extra_trees_u16_bins/1` | `MOJOLEARN_IDN_ET_BINNED_U16` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gbdt_apply_wide` | gbdt | `gbdt_apply_wide/1` | `MOJOLEARN_IDN_GBDT_APPLY_WIDE` | [gbdt/models/kernel/add_bin_values.mojo](../gbdt/models/kernel/add_bin_values.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gbdt_ctr_borders_device` | gbdt | `gbdt_ctr_borders_device/1` | `MOJOLEARN_IDN_GBDT_CTR_BORDERS_DEVICE` | [gbdt/ctrs/ctr_binarization.mojo](../gbdt/ctrs/ctr_binarization.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gp_gradient_rows4` | kernel-gp | `gp_gradient_rows4/4` | `MOJOLEARN_IDN_GP_GRAD_ROWS4` | [gaussian_process/gp_grad_items.mojo](../gaussian_process/gp_grad_items.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `nystroem_device_rr_stop` | kernel-gp | `nystroem_device_rr_stop/2`, `nystroem_device_rr_stop/4` | `MOJOLEARN_IDN_NYS_RR_DEV_STOP`, `MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `rbf_direct_cell` | kernel-gp | `rbf_direct_cell/16`, `rbf_direct_cell/64` | `MOJOLEARN_IDN_KM_RBF_CELL`, `MOJOLEARN_IDN_KM_RBF_CELL_D16` | [kernel_methods/checks/kernel_matrix.mojo](../kernel_methods/checks/kernel_matrix.mojo); [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gram_cd_column_limit` | linear | `gram_cd_column_limit/128`, `gram_cd_column_limit/256` | `MOJOLEARN_CD_IDN_GRAM_COLS_128`, `MOJOLEARN_CD_IDN_GRAM_COLS_256` | [solver/impl/cd_gram_rule.mojo](../solver/impl/cd_gram_rule.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `qn_tiled_multinomial_all` | linear | `qn_tiled_multinomial_all/1` | `MOJOLEARN_QN_TILED_MULTI_ALL` | [glm/impl/qn/qn_tiled_rule.mojo](../glm/impl/qn/qn_tiled_rule.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `qn_device_convergence` | linear | `qn_device_convergence/2`, `qn_device_convergence/4`, `qn_device_convergence/8` | `MOJOLEARN_QN_IDN_DCONV`, `MOJOLEARN_QN_IDN_DCONV_POLL_2`, `MOJOLEARN_QN_IDN_DCONV_POLL_8` | [glm/impl/qn/glm_base.mojo](../glm/impl/qn/glm_base.mojo); [glm/impl/qn/qn_dconv.mojo](../glm/impl/qn/qn_dconv.mojo); [glm/impl/qn/qn_solvers.mojo](../glm/impl/qn/qn_solvers.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `mamba3_parallel_angle` | lm | `mamba3_parallel_angle/32`, `mamba3_parallel_angle/64`, `mamba3_parallel_angle/128` | `MOJOLEARN_IDN_M3_ANGLE_PARALLEL`, `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32`, `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128` | [mamba/checks/mamba3_oracle.mojo](../mamba/checks/mamba3_oracle.mojo); [mamba/host/gen/mamba3_siso.mojo](../mamba/host/gen/mamba3_siso.mojo); [mamba/impl/ops/mamba3_siso.mojo](../mamba/impl/ops/mamba3_siso.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gemm_group_slack` | lm | `gemm_group_slack/2`, `gemm_group_slack/8` | `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2`, `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8` | [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gemm_group_s` | lm | `gemm_group_s/half`, `gemm_group_s/x2` | `MOJOLEARN_IDN_GEMM_GROUP_S_HALF`, `MOJOLEARN_IDN_GEMM_GROUP_S_X2` | [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `gemm_group_tiles_body` | lm | `gemm_group_tiles_body/1` | `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY` | [gemm/checks/gemm_identical.mojo](../gemm/checks/gemm_identical.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `ocsvm_chunk256` | neighbors | `ocsvm_chunk256/256` | `MOJOLEARN_IDN_OCSVM_CHUNK256` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `cross_entropy_fold256` | neural | `cross_entropy_fold256/256` | `MOJOLEARN_IDN_XENT_FOLD_BLOCK_256` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `lda_gram_rowtile` | prep-metrics | `lda_gram_rowtile/1` | `MOJOLEARN_IDN_GRAM_ROWTILE` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `lda_qda_gram_rows` | prep-metrics | `lda_qda_gram_rows/512`, `lda_qda_gram_rows/8192` | `MOJOLEARN_IDN_GRAM_ROWS_512`, `MOJOLEARN_IDN_GRAM_ROWS_8192` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `lda_qda_rr_min` | prep-metrics | `lda_qda_rr_min/8`, `lda_qda_rr_min/16`, `lda_qda_rr_min/48`, `lda_qda_rr_min/96` | `MOJOLEARN_IDN_RR_MIN_8`, `MOJOLEARN_IDN_RR_MIN_16`, `MOJOLEARN_IDN_RR_MIN_48`, `MOJOLEARN_IDN_RR_MIN_96` | [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `xty_tile_rows` | shared | `xty_tile_rows/64`, `xty_tile_rows/1024` | `MOJOLEARN_IDN_XTY_TILE_64`, `MOJOLEARN_IDN_XTY_TILE_1024` | [core/host_tile_fold.mojo](../core/host_tile_fold.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `device_f64_narrowing` | shared | `device_f64_narrowing/1` | `MOJOLEARN_IDN_HPDEV_CAST_F64` | [bindings/_mojolearn.mojo](../bindings/_mojolearn.mojo); [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `arima_parallel_start_ls` | timeseries | `arima_parallel_start_ls/256`, `arima_parallel_start_ls/2048` | `MOJOLEARN_IDN_X0_PAR_LS`, `MOJOLEARN_IDN_X0_PAR_LS_MIN256` | [arima/impl/idn_arma_ls.mojo](../arima/impl/idn_arma_ls.mojo); [arima/impl/idn_ls_math.mojo](../arima/impl/idn_ls_math.mojo) | NOT_QUALIFIED_LEAVE_DISABLED |
| `eigh_block_reenable` | decomp | `eigh_block_reenable/1` | `MOJOLEARN_IDN_EIGH_BLOCK_ON` | [tools/identical_wave_eigh_block_refusal.mojo](../tools/identical_wave_eigh_block_refusal.mojo); [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) | DISABLED_AFTER_OBSERVED_CONVERGENCE_FAILURE |

Full explicit A/B definitions: [tools/identical_candidate_recipes.json](../tools/identical_candidate_recipes.json). Candidate groups are not blanket claims of current route coverage; recipes may require nondefault settings or additional caller evidence.

### GEMM schedule profiles

Definitions: [experiments/identical_speed/profiles.json](../experiments/identical_speed/profiles.json); selected batch: [experiments/identical_speed/selected-batch.json](../experiments/identical_speed/selected-batch.json); runner: [tools/identical_speed_ab.py](../tools/identical_speed_ab.py).

| Profile | Control | Vendors | Hypothesis |
| --- | --- | --- | --- |
| one-page | `MOJOLEARN_GEMM_ONE_PAGE` | nvidia, amd | Lower shared-memory use can admit more resident blocks; loses double-buffer prefetch. |
| group-slack2 | `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2` | nvidia, amd | Coarser aligned leaf groups reduce workspace and fold traffic, at the cost of fewer blocks. |
| group-slack8 | `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8` | nvidia, amd | Finer aligned leaf groups expose more parallelism, at the cost of workspace and fold traffic. |
| body-tiles | `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY` | nvidia | Count actual packed-body tiles when choosing group size, reducing unnecessary split workspace. |
| packed64 | `MOJOLEARN_GEMM_KPACK_RPT4` | nvidia | A shorter packed tile reduces per-thread register demand and increases the number of independent blocks. |

### Optimization ledger entries

Record file: [docs/identical/optimization-ledger.json](../docs/identical/optimization-ledger.json). Saved status is retained without rerunning its evidence.

| Candidate | Control | Implementation recorded by ledger | Recorded status |
| --- | --- | --- | --- |
| RF fused partition / DART | `MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) | PENDING_QUALITY_AND_SPEED |
| Block Jacobi eigensolver | none | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) | DISABLED_BY_DEFAULT |
| Shared IDENTICAL callpath | none | [experiments/identical_callpath](../experiments/identical_callpath) | EXPERIMENT_NOT_FULL_ESTIMATOR_INTEGRATION |
| Combined-source NVIDIA PTX qualification | none | [tools/nvidia_baseline_gpu_batch.py](../tools/nvidia_baseline_gpu_batch.py); [tools/nvidia_baseline_qualification.py](../tools/nvidia_baseline_qualification.py); [python/mojolearn/ptx_admission.py](../python/mojolearn/ptx_admission.py) | REQUIRES_NEW_COMBINED_SOURCE_QUALIFICATION |
| CNN/SGD correctness and launch bounds | none | See the linked record | TARGETED_BITS_PASS_SPEED_PENDING |
| IDENTICAL ON (all switches) vs MOJOLEARN_IDN_ALL_OFF, medium wave, frozen af93ebe0a | none | See the linked record | MEASURED |
| KMeans accumulate rows 256 (MOJOLEARN_IDN_KMEANS_ACC_ROWS_256) | none | See the linked record | KEEP |
| GMM E-step stack (MOJOLEARN_IDN_GMM_ESTEP_STACK) | none | See the linked record | DROP |
| OCSVM chunk256 (MOJOLEARN_IDN_OCSVM_CHUNK256) | none | See the linked record | DROP |

Additional candidate disposition and pending work: [docs/identical/candidate-priority-queue.json](../docs/identical/candidate-priority-queue.json); [docs/identical/block-eigh-centering-experiment.json](../docs/identical/block-eigh-centering-experiment.json); [docs/identical/candidate-audit/candidates-2026-10-04.tsv](../docs/identical/candidate-audit/candidates-2026-10-04.tsv); [docs/identical/candidate-audit/candidate-audit-2026-10-04.md](../docs/identical/candidate-audit/candidate-audit-2026-10-04.md).

## Historical Apple FAST experiments

The retained ledger [docs/apple-fast/EXPERIMENTS.md](../docs/apple-fast/EXPERIMENTS.md) contains original branches, commits, arm tags, quality notes and KEEP/DROP/OPEN histories. Its rows include duplicate names at different checkpoints, superseded controls, removed code and work that lived only on another branch. The tables below index record names rather than reclassifying their verdicts. Repeated names within a section are consolidated; section links retain every original row. A record file is a location for the experiment history, not proof its implementation is present in this worktree. Neural and optimizer sections are historical references outside the new classical campaign.

### Untried candidates 2026-10-05

Record file and section: [Untried candidates (2026-10-05)](../docs/apple-fast/EXPERIMENTS.md#untried-candidates-2026-10-05).

- `HUBER_FAST_BLOCK512`
- `NB_CAT_ATOMIC` (FAST)
- `MI_REG_RANKMAJOR` (+ `MI_REG_SORTCOUNT`)
- `MI_WORK` (env)
- `X_PREP_FAST_II_CONV` (env)
- `X_PREP_FAST_II_GRAM_TILE` (env)
- `X_PREP_FAST_QSELECT` (env)
- `ARIMA_FAST_CONST_BOTH`
- `ARIMA_FAST_ROOT_CHECK`
- `ARIMA_FAST_KPSS_D`
- `FA_LIVEBUF` (on the FA_ITER_DEVICE default)
- `DBSCAN_FAST_CC_BATCH`
- `BPE_TRAIN_DEVICE`, `BPE_ENCODE_DEVICE`, `BPE_LIVEBUF`, `BPE_GROUP_FILTER`, `BPE_MERGE_BATCH`, `BPE_ALL` (7 rows)
- `PURITY2_2` (`_OFF`)
- `SVGP_FAST_BSPLIT`
- `AFN_OPT_FUSE_SCAN` (alone)
- `AFN_OPT_VEC4` (alone)
- `AFN_OPT_RESIDENT_STATE` (alone)
- `OPT_FAST_PIPE_CH=524288`
- `SCHED_FAST_INLINE` (env)

### TODO run later AutoARIMA vs statsforecast 2026-10-05

Record file and section: [TODO run later: AutoARIMA vs statsforecast (2026-10-05)](../docs/apple-fast/EXPERIMENTS.md#todo-run-later-autoarima-vs-statsforecast-2026-10-05).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### Trees 101

Record file and section: [Trees (101)](../docs/apple-fast/EXPERIMENTS.md#trees-101).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `TREESHAP_FAST_TABLE` | lane/apple-fast-fix-treeshap @ 8fcd690cb |
| (no switch) YetiRank 256-thread block kernel | lane/apple-fast @ 46f0cf09f |
| `DART_DEVICE` | lane/apple-fast-dart @ b192c3353 |
| `ET_PART_ROWS (_OFF)` | lane/apple-fast @ 269ffa57a |
| `FOREST_DEVICE_FINITE (_OFF)` | lane/apple-fast-rfet-scan @ 500168cfe |
| `GBDT_CTR_FAST_FREQ` | lane/apple-fast-trees-depthwise @ f743edd60 |
| `GBDT_CTR_PERM_BATCH` | lane/apple-fast-trees-depthwise @ f743edd60 |
| `GBDT_DW2_PART_VEC4` | lane/apple-fast-dwgap2 @ 23eca012b |
| `GBDT_DW2_SCAN_SMEM` | lane/apple-fast-dwgap2 @ 23eca012b |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC` | lane/apple-fast-depthwise @ 4547e0d14 |
| `GBDT_DW_MODE_SKIP` | lane/apple-fast-dwgap @ 0256457c3 |
| `GBDT_LG_EXACT_BATCH128` | lane/apple-fast-lgw128 @ ec921c123 |
| `GBDT_LG_EXACT_BATCH64` | lane/apple-fast-trees2 @ bfd1d7cc6 |
| `IF_ROWMAJOR (_OFF) + device finite scan` | lane/apple-fast-trees2 @ bfd1d7cc6 |
| `MULTICLASS_HESSIAN_BATCH` | lane/apple-fast-pairlogit @ 00bfe78f8 |
| `ORDERED_BATCH_EST` | lane/apple-fast-ordered @ 5b8722353 |
| `PAIRLOGIT_EST_REUSE + PAIRLOGIT_GROUP_FUSED` | lane/apple-fast-pairlogit @ 00bfe78f8 |
| `PAIRLOGIT_GROUP_FUSED` | lane/apple-fast-pairlogit @ 00bfe78f8 |
| `RF_HIST_COLUMNS8 (_OFF)` | lane/apple-fast @ 269ffa57a |
| `SEG_SUMS_BLOCK_SCAN (_OFF)` | lane/apple-fast-rfet-scan @ 500168cfe |
| `TE_ADA_SESSION` | lane/apple-fast-trees-ensembles @ 1669a3bcd |
| `TE_ADA_SESSION + TE_ADA_SESSION_SHARE` | lane/apple-fast-trees-ensembles @ 1669a3bcd |
| `TE_NATIVE_SPLITS` | lane/apple-fast-trees-ensembles @ 1669a3bcd |
| `YETI_EST_REUSE_SEARCH (_OFF)` | lane/apple-fast @ 098e89988 |
| `YETI_FAST_SORT` | lane/apple-fast-yetirank @ c7b35fd7c |
| `YETI_SEARCH_TASK16K` | lane/apple-fast-trees-yeti @ 65f551e39 |
| `ET_DEVICE_BATCH_65536` | lane/apple-fast @ 269ffa57a |
| `ET_TPB_256` | lane/apple-fast @ 269ffa57a |
| `GBDT_CTR_FAST_FREQ + GBDT_CTR_FAST_SCAN` | lane/apple-fast-trees-depthwise @ f743edd60 |
| `GBDT_CTR_FAST_SCAN` | lane/apple-fast-trees-depthwise @ f743edd60 |
| `GBDT_CTR_PERM_PTRS` | lane/apple-fast-trees-depthwise @ f743edd60 |
| `GBDT_DW2_COPY_ZERO` | lane/apple-fast-dwgap2 @ 23eca012b |
| `GBDT_DW_FAST_DEV_SCALE` | lane/apple-fast-gap-misc @ 5e2eec7a3 |
| `GBDT_DW_FAST_SKIP_FINAL_STATS` | lane/apple-fast-gap-misc @ 5e2eec7a3 |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC` | lane/apple-fast-depthwise @ 4547e0d14 |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC + GBDT_DW_TREE_SYNC_CHECK` | lane/apple-fast-depthwise @ 4547e0d14 |
| `GBDT_LG_EXACT_BATCH16` | lane/apple-fast-trees2 @ 50dfdcca0 |
| `GBDT_QH_FAST_FUSED_Q` | lane/apple-fast-gap-misc @ 5e2eec7a3 |
| `GBDT_SEG_SUMS_BLOCK` | lane/apple-fast-rfet-scan @ 500168cfe |
| `GBDT_SM_X4` | lane/apple-fast-trees2 @ 50dfdcca0 |
| `GBDT_SM_X8` | lane/apple-fast-trees2 @ 50dfdcca0 |
| `IF_QUERY_RAW` | lane/apple-fast-trees-io @ df6a77c21 |
| `IF_SAMPLED_UPLOAD` | lane/apple-fast-trees-io @ df6a77c21 |
| `ORDERED_FOLD_DERIVS` | lane/apple-fast-ordered @ 5b8722353 |
| `REORDER_FLAGS_SCAN_BLOCK + SEG_SCAN_BLOCK` | lane/apple-fast-trees-scan @ 43430ca0f |
| `RF_NODESPLIT_ZERO_AFTER_READ + RF_FAST_BATCH16K` | lane/apple-fast-trees2 @ bfd1d7cc6 |
| `RF_SMALL_NODE_1024` | lane/apple-fast-trees2 @ 50dfdcca0 |
| `SEG_SCAN_BLOCK` | lane/apple-fast-trees-scan @ 43430ca0f |
| `SYM_DEVICE_LEAVES` | lane/apple-fast-trees-symmetric @ ce517b4b3 |
| `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` | lane/apple-fast-trees-symmetric @ ce517b4b3 |
| `SYM_DEVICE_LEVEL` | lane/apple-fast-trees-symmetric @ ce517b4b3 |
| `SYM_DEVICE_PARTITION` | lane/apple-fast-trees-symmetric @ ce517b4b3 |
| `SYM_HIST_FAST` | lane/apple-fast-trees-yeti @ 65f551e39 |
| `SYM_HIST_FAST + YETI_SEARCH_TASK16K` | lane/apple-fast-trees-yeti @ 65f551e39 |
| `SYM_HIST_FAST + YETI_SYM_HIST_UNROLL8` | lane/apple-fast-yetirank @ c7b35fd7c |
| `SYM_NO_TAIL_DRAIN` | lane/apple-fast-trees-symmetric @ ce517b4b3 |
| `YETI_TREE_SEARCH_SCORE_GRID` | lane/apple-fast-yetirank @ c7b35fd7c |
| `CTR_INDEX_FUSED` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `CTR_ONEHOT_DEVICE` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `CTR_PREP_SHARED` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `CTR_SORT_ONCE` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `EST_ITERS_DEVICE` | lane/apple-fast-sym-est @ c8518eb52 |
| `EST_REUSE_PART` | lane/apple-fast-sym-est @ c8518eb52 |
| `EST_SHRINK_FUSED` | lane/apple-fast-sym-est @ c8518eb52 |
| `EST_STATS_FUSED` | lane/apple-fast-sym-est @ c8518eb52 |
| `GBDT_BOOT_DEVICE` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `GBDT_EVAL_FUSED` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `GBDT_EVAL_SKIP_EMPTY` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `GBDT_INDEX_PACK_DEVICE` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `GBDT_PREDICT_PACKED` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `GBDT_QUANT_DEVICE` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `MC_CLASS_BATCH_DERIV` | lane/apple-fast-sym-multi @ d2c832da0 |
| `MC_CLASS_BATCH_DERIV + MC_CLASS_BATCH_EST` | lane/apple-fast-sym-multi @ d2c832da0 |
| `MC_CLASS_BATCH_EST` | lane/apple-fast-sym-multi @ d2c832da0 |
| `ORD_ALL` (`_OFF`) | lane/apple-fast-sym-ordered @ 27b912397 |
| `ORD_FOLD_BINS_ONE` | lane/apple-fast-sym-ordered @ 27b912397 |
| `ORD_FOLD_INDEX` | lane/apple-fast-sym-ordered @ 27b912397 |
| `ORD_STD_PARALLEL` | lane/apple-fast-sym-ordered @ 27b912397 |
| `ORD_TREE_LEAN` | lane/apple-fast-sym-ordered @ 27b912397 |
| `PL_GROUP_NARROW` | lane/apple-fast-sym-multi @ d2c832da0 |
| `PL_GROUP_NARROW + PL_PAIRS_ONCE` | lane/apple-fast-sym-multi @ d2c832da0 |
| `PL_PAIRS_ONCE` | lane/apple-fast-sym-multi @ d2c832da0 |
| `SHAP_KERNEL_DEV` | lane/apple-fast-shap @ 13343dd51 |
| `SHAP_PERM_CACHE` (rollback `SHAP_PERM_CACHE_OFF`) | lane/apple-fast-shap @ 13343dd51, recovered on lane/apple-fast-rec-shap; A/B ab1 d51f4b4bf |
| `SHAP_TREE_TAB` | lane/apple-fast-shap @ 13343dd51, recovered on lane/apple-fast-rec-shap; lane/apple-fast-shap @ 13343dd51 |
| `SYM_BUF_ARENA` | lane/apple-fast-sym-iter @ 4956a2234 |
| `SYM_CTR_ALL` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `SYM_CTR_PERM_BATCH` | lane/apple-fast-sym-ctr @ 39c3c9daf |
| `SYM_DERIV_FUSED` | lane/apple-fast-sym-iter @ 4956a2234 |
| `SYM_EST_ALL` | lane/apple-fast-sym-est @ c8518eb52 |
| `SYM_FEAT_ALL` | lane/apple-fast-sym-feat @ bca0e3a48 |
| `SYM_GATHER_FUSED` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_HIST_ALL` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_HIST_MULT` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_ITER_ALL` | lane/apple-fast-sym-iter @ 4956a2234 |
| `SYM_LEAF_FROM_STATS` | lane/apple-fast-sym-iter @ 4956a2234 |
| `SYM_MULTI_ALL` | lane/apple-fast-sym-multi @ d2c832da0 |
| `SYM_PART_STATS_PAR` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_RESOLVE_BLOCK` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_REUSE_PARTITION` | lane/apple-fast-sym-iter @ 4956a2234 |
| `SYM_SCAN_SUB_FUSED` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `SYM_SORT_SWAP` | lane/apple-fast-sym-hist @ 3bb4db314 |
| `YR_TASK_FUSED` | lane/apple-fast-sym-multi @ d2c832da0 |
| `SHAP_PERM_CACHE` | lane/apple-fast-shap @ 13343dd51 |

### Linear 46

Record file and section: [Linear (46)](../docs/apple-fast/EXPERIMENTS.md#linear-46).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `SGD_FAST_PS_SIMD` | lane/apple-fast-gap-clus3 @ 2ac0505fc |
| `BAYES_GRID_GUARD` | lane/apple-fast-bayes @ 1a0bb2b2b |
| `CALIB_GNB_FOLDS` | lane/apple-fast-meta @ 17b317ae6 |
| `CD_FAST_GRID_GRAM` | lane/apple-fast-linear @ 1c7c213f8 |
| `CD_FAST_GRID_GRAM + CD_FAST_ROWMAJOR` | lane/apple-fast-linear @ 1c7c213f8 |
| `FAST_OLS_NORMAL_EQ (_OFF)` | lane/apple-fast-olsne @ 917dd5b4c |
| `LDAQDA_DEC_TILE` | lane/apple-fast-ldaqda @ 7d6a6126e |
| `LDAQDA_PAR_STAGES` | lane/apple-fast-ldaqda @ 7d6a6126e |
| `LDAQDA_RR_EIGH` | lane/apple-fast-ldaqda @ 7d6a6126e |
| `MULTIOUT_RIDGE` | lane/apple-fast-meta @ 17b317ae6 |
| `QN_FAST_BLOCKS` | lane/apple-fast-linear @ 1c7c213f8 |
| `RIDGE_NO_U` | lane/apple-fast-ridgespeed @ be56e2ead |
| `RIDGE_RESIDENT` | lane/apple-fast-ridgespeed @ be56e2ead |
| `X_LINEAR_ENETCV_FAST (grid path)` | lane/apple-fast @ 952422579 |
| `X_LINEAR_GRAM_SSE` | lane/apple-fast @ 269ffa57a |
| `X_LINEAR_LARS_FAST_GRAM` | lane/apple-fast-gram @ 3d36a676a |
| `X_LINEAR_RIDGE_FAST_GRAM` | lane/apple-fast-gram @ 3d36a676a |
| `X_PREP_CLASS_COV_GRID` | lane/apple-fast-gram @ 3d36a676a |
| `KERNEL_FAST_BAYES_JACOBI` | lane/apple-fast-kernel @ 9e851777c |
| `KERNEL_FAST_BAYES_STATS` | lane/apple-fast-kernel @ 9e851777c |
| `OLS_FAST_DEVICE_CENTER` | lane/apple-fast-core @ 9a31ebb4c |
| `QN_FAST_COALESCED_OFF` | lane/apple-fast-linear @ 1c7c213f8 |
| `QN_FAST_GRID_SUMS` | lane/apple-fast-linear @ 1c7c213f8 |
| `RIDGE_FAST_CLS1_CODES + RIDGE_FAST_CLS1_PREDICT` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `RIDGE_FAST_CLS1_PREDICT` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| (baseline, no switch) | lane/apple-fast-robust @ cfdb95e48 |
| `ARD_FAST_CLS1_BATCH` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `ARD_FAST_CLS1_BATCH + ARD_FAST_CLS1_PARTS + ARD_FAST_CLS1_STATS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `ARD_FAST_CLS1_PARTS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `ARD_FAST_CLS1_STATS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `BAYES_FAST_CLS1_BATCH` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `BAYES_FAST_CLS1_BATCH + BAYES_FAST_CLS1_PARTS + BAYES_FAST_CLS1_STATS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `BAYES_FAST_CLS1_PARTS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `BAYES_FAST_CLS1_STATS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `HUBER_DEVICE_LBFGS` | lane/apple-fast-robust @ cfdb95e48 |
| `HUBER_DEVICE_LBFGS + HUBER_FAST_BLOCK512` | lane/apple-fast-robust @ cfdb95e48 |
| `LSVR_ALL` | lane/apple-fast-linsvr @ c649076a4 (via lane/apple-fast-m2b1 65c454c87) |
| `LSVR_DEVICE_CONVERGE` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_DUAL_CD` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_EVAL_SLIM` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_FASTPATH_FIX` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_FUSED_GRAD` | lane/apple-fast-linsvr @ c649076a4 |
| `LSVR_LINESEARCH_BATCH` | lane/apple-fast-linsvr @ c649076a4 (via lane/apple-fast-m2b1) |
| `NB_CAT_ATOMIC` | lane/apple-fast-nb @ be2ea3a05 |
| `NB_TEXT_CSR` | lane/apple-fast-nb @ be2ea3a05 (via lane/apple-fast-m2b1) |
| `RIDGE_FAST_CLS1_CODES` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `ISOTONIC_FAST_NOLIST` (env `MOJOLEARN_ISOTONIC_FAST_NOLIST_OFF=1` off) | lane/apple-fast-gap-manprep @ 1db219f01 |

### Neighbors 42

Record file and section: [Neighbors (42)](../docs/apple-fast/EXPERIMENTS.md#neighbors-42).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `KDE_FAST_SLICES` | lane/apple-fast-core @ 9a31ebb4c |
| `KNN_FAST_MMA_BIGD` | lane/apple-fast @ 269ffa57a |
| `KNN_FAST_MMA_K64` | lane/apple-fast-core @ 9a31ebb4c |
| `LP_FAST_RESIDENT` | lane/apple-fast-neighbors2 @ 5fb6edd3f |
| `KNN_FAST_CLS1_PRESEED` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `KNN_FAST_CLS1_PRESEED + KNN_FAST_CLS1_SLICES2` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `KNN_FAST_CLS1_SLICES2` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `LLE_FAST_KNN` | lane/apple-fast-isotonic-knn @ 7385fcfdd |
| `NC_FAST_CLS1_LABELS + NC_FAST_CLS1_PREDICT` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `NC_FAST_CLS1_PREDICT` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `RADIUS_FAST_REUSE_COUNT` | lane/apple-fast-neighbors2 @ 5fb6edd3f |
| (baseline, no switch) | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `ANN_FAST_KNN_BIGD` | lane/apple-fast-ann @ 70833546a |
| `CAGRA_FAST_TEAM` | lane/apple-fast-ann @ 70833546a |
| `CAGRA_FAST_WIDE` | lane/apple-fast-gap-cagra @ a3ebfc4a7 |
| `CAGRA_FAST_DOT` | lane/apple-fast-gap-cagra @ a3ebfc4a7 |
| `CAGRA_FAST_IVFG` | lane/apple-fast-gap-cagra @ a3ebfc4a7 |
| `CAGRA_FAST_IVFG + CAGRA_FAST_IVFG_P32` | lane/apple-fast-gap-cagra @ a3ebfc4a7 |
| `CAGRA_FAST_SEEDS` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_SEEDS4` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_ITERS` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_SEEDS + CAGRA_FAST_ITERS` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_IVFG + CAGRA_FAST_IVFG_EXACTD` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_IVFG + IVFG_EXACTD + IVFG_P8` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_IVFG + IVFG_EXACTD + SEEDS + ITERS` | lane/apple-fast-gap-cagra @ 2b16b4322 |
| `CAGRA_FAST_IVFG_LOWD` | lane/apple-fast-w2-cagra (base b2b1c22bc) |
| `CAGRA_FAST_IVFG_LOWD_SEEDS4` | lane/apple-fast-w2-cagra (base b2b1c22bc) |
| `ISOTONIC_FAST_PAIRMERGE + ISOTONIC_FAST_PAR` | lane/apple-fast-isotonic-knn @ 7385fcfdd |
| `ISOTONIC_FAST_PAR` | lane/apple-fast-isotonic-knn @ 7385fcfdd |
| `IVFPQ_FAST_DEVICE_CODEBOOKS` | lane/apple-fast-ann @ 70833546a |
| `IVF_COARSE_RANDOM_INIT (_OFF)` | lane/apple-fast-vsearch @ 86925aef9 |
| `IVF_DEVICE_VALIDATE (_OFF)` | lane/apple-fast-vsearch @ 86925aef9 |
| `IVF_FAST_DEVICE_CSR` | lane/apple-fast-ann @ 70833546a |
| `IVF_FAST_DEVICE_TRAINSET` | lane/apple-fast-ann @ 70833546a |
| `IVF_FAST_SCAN_SELECT` | lane/apple-fast-ann @ 70833546a |
| `IVF_KMEANS_LAZY_SHIFT (_OFF)` | lane/apple-fast-vsearch @ 86925aef9 |
| `KMEANS_FAST_LAZY_SHIFT` | lane/apple-fast-vsv-promote |
| `IVF_REFINE_TEAM` | lane/apple-fast-batch @ 3150d75c1 |
| `KDE2_ALL` | lane/apple-fast-batch @ 3150d75c1 |
| `KDE2_ALL + KDE_DIMTILE` | lane/apple-fast-kde2 @ 659400b94 |
| `KDE_DIMTILE` | lane/apple-fast-batchv @ c8251211d |
| `KDE_DIMTILE + KDE_KERNEL_VARIANTS` | lane/apple-fast-kde2 @ 659400b94 |
| `KDE_DIMTILE + KDE_LSE_FUSED` | lane/apple-fast-kde2 @ 659400b94 |
| `KDE_DIMTILE + KDE_NORM_FUSED` | lane/apple-fast-kde2 @ 659400b94 |
| `KDE_DIMTILE + KDE_SAMPLE_FUSED` | lane/apple-fast-kde2 @ 659400b94 |
| `KDE_SAMPLE_FUSED` | lane/apple-fast-batch @ 3150d75c1 |
| `NC_FAST_CLS1_LABELS` | lane/apple-fast-gap-cls1 @ 4e341dc41 |
| `PQ_LUT_TILED (_OFF)` | lane/apple-fast-vsearch @ 86925aef9 |
| `PQ_SCAN_FUSED (_OFF)` | lane/apple-fast-vsearch @ 86925aef9 |
| `PQ_LUT_TILED` | lane/apple-fast-vsearch @ 86925aef9 |
| `PQ_SCAN_FUSED` | lane/apple-fast-vsearch @ 86925aef9 |
| `TSNE_FAST_SPLIT` | lane/apple-fast-ann @ 70833546a |
| `VSEARCH_ALL` | lane/apple-fast-vsv @ ad265a028 |
| `PY2MOJO_cluster_OFF` | lane/apple-fast-py2mojo-cluster @ 2b6f3bfd4 |
| `XN_FAST_IMPUTE_TILED2` | lane/apple-fast-isotonic-knn @ 7385fcfdd |
| `XN_FAST_MMA_ROUTE` | lane/apple-fast-isotonic-knn @ 7385fcfdd |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_RES | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_2L | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256 | lane/apple-fast-gap-cls2@72602a339 |
| `XN_FAST_IMPUTE_TIE_MEAN + XN_FAST_NAN_COLMISS_ONLY` | lane/apple-fast-gap-manprep @ 9130a81bc |

### Prep 42

Record file and section: [Prep (42)](../docs/apple-fast/EXPERIMENTS.md#prep-42).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `PREP_FAST_MINMAX` | lane/apple-fast-prep @ a11e43a5e |
| `SELECT_D` | lane/apple-fast-select @ 99fad7a5d |
| `SELECT_FCLS` | lane/apple-fast-select @ 99fad7a5d |
| `SELECT_FREG` | lane/apple-fast-select @ 99fad7a5d |
| `X_PREP_FAST_UNIQUE` | lane/apple-fast-prep @ a11e43a5e |
| `X_PREP_FAST_NONEG` | lane/apple-fast-prep @ a11e43a5e |
| `CV_FAST_SLICE` | lane/apple-fast-resample @ 50b96e795 |
| `CV_FAST_TRUST_FOLDS` | lane/apple-fast-resample @ 50b96e795 |
| `MI_ALL` | lane/apple-fast-miv @ 514401169 |
| `MI_CLF_RANKMAJOR` | lane/apple-fast-miv @ 5e26b1008 |
| `MI_REG_TIES` | lane/apple-fast-miv @ 514401169 |
| `MI_FAST_FOLDS` | lane/apple-fast-batch @ 3150d75c1 |
| `MI_REG_RANKMAJOR + MI_REG_SORTCOUNT` | lane/apple-fast-mi @ 6944ebb57 |
| `MI_REG_SORTCOUNT` | lane/apple-fast-mi @ 6944ebb57 |
| `MI_REG_SORTCOUNT + MI_REG_TIES` | lane/apple-fast-mi @ 6944ebb57 |
| `MI_WORK` | lane/apple-fast-mi @ 6944ebb57 |
| `PREP2_FAST_EIGH_BLOCK` | lane/apple-fast-prep2 @ 8762eb33f |
| `PREP3_LABELS` | lane/apple-fast-prep3 @ ec65873e3 |
| `PREP3_MAXABS` | lane/apple-fast-prep3 @ ec65873e3 (via lane/apple-fast-m2b1) |
| `PREP3_SPLINE` | lane/apple-fast-prep3 @ ec65873e3 |
| `PREP_FAST_CLS2_MINMAX_FUSED / _MINMAX_POOL` | lane/apple-fast-gap-cls2 @ 72602a339 |
| `PTIMPUTE_ALL` | lane/apple-fast-batchv @ 77f1f5afb |
| `PT_COLBATCH` | lane/apple-fast-batchv @ 30aa43339 |
| `PT_COLBATCH + PT_SPEC` | lane/apple-fast-batch @ 3150d75c1 |
| `PT_FOLD_NOX` | lane/apple-fast-ptimpute @ 9623cd7dc |
| `PT_FUSED_TRANSFORM` | lane/apple-fast-batchv |
| `PT_SPEC` | lane/apple-fast-batch @ 3150d75c1 |
| `RESAMPLE_FAST_GATHER` | lane/apple-fast-resample @ 50b96e795 |
| `RESAMPLE_FAST_IDX_BULK` | lane/apple-fast-resample @ 50b96e795 |
| `RESAMPLE_FAST_ONE_FOLD` | lane/apple-fast-resample @ 50b96e795; A/B ab1 d51f4b4bf |
| `RESAMPLE_FAST_PERM_SELECT` (rollback `RESAMPLE_FAST_PERM_SELECT_OFF`) | lane/apple-fast-resample @ 50b96e795; A/B ab1 d51f4b4bf |
| `RESAMPLE_FAST_RANK_SORT` | lane/apple-fast-resample @ 50b96e795; A/B ab1 d51f4b4bf |
| `SI_ONEPASS` | lane/apple-fast-batchv @ 77f1f5afb |
| `X_PREP_FAST_CLS2_PACK / _PRESENT` | lane/apple-fast-gap-cls2 @ 72602a339 |
| `X_PREP_FAST_II_CONV` | lane/apple-fast-prep2 @ 8762eb33f |
| `X_PREP_FAST_II_GRAM_TILE` | lane/apple-fast-prep2 @ 8762eb33f |
| `X_PREP_FAST_QSELECT` | lane/apple-fast-prep2 @ 8762eb33f |
| `X_PREP_FAST_TE_ENC` | lane/apple-fast-prep2 @ 8762eb33f |
| `X_PREP_FAST_TE_GLOBAL` | lane/apple-fast-prep2 @ 8762eb33f |
| MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_X_PREP_FAST_CLS2_PACK | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_X_PREP_FAST_CLS2_PRESENT | lane/apple-fast-gap-cls2@72602a339 |
| `X_PREP_FAST_STAGED_OUT` | lane/apple-fast-gap-manprep @ 1169df581 |
| `X_PREP_POOL_ARENA` | lane/apple-fast-w2-prep @ ec87a9035 |

### Decomp 34

Record file and section: [Decomp (34)](../docs/apple-fast/EXPERIMENTS.md#decomp-34).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `APPLE_FAST_GEMM_TN_V1` | lane/apple-fast-tier @ 95a09d1fd |
| `LDA_FUSED_SS` | lane/apple-fast-nb @ be2ea3a05 |
| `APPLE_FAST_GEMM_NT_TILED` | lane/apple-fast-tier @ 95a09d1fd |
| `DECOMP_FAST_SMALL_EIGH_J2` | lane/apple-fast-decomp-sparse @ 5fb1740cd |
| `LLE_SPARSE_EIG` | lane/apple-fast-lle @ f2ea1ecb5 |
| `PCA_FAST_EIG` | lane/apple-fast-pca-eig @ 9819970b1 |
| `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` | lane/apple-fast-pca-eig @ 9819970b1 |
| `PCA_FAST_NO_ALIAS` | lane/apple-fast-pca-eig @ 9819970b1 |
| `PCA_FAST_TOPK` | lane/apple-fast-pca-eig @ 9819970b1 |
| (baseline, no switch) | lane/apple-fast-pca-eig @ 9819970b1; lane/apple-fast-tier @ 95a09d1fd |
| `LU_FAST_STEP1` | lane/apple-fast-gap-linalg2 @ f3d66dd94 |
| `CHOL_FAST_DEVIO` | lane/apple-fast-gap-linalg2 @ 19fb05674 |
| `CHOL_FAST_NOSYNC` | lane/apple-fast-gap-linalg2 @ 19fb05674 |
| `CHOL_FAST_TALL` | lane/apple-fast-w3-linalg @ 97b7bcb7e |
| `DECOMP_FAST_GEMM_MMA` | lane/apple-fast-gap-linalg2-pca @ 474241154 |
| `PCA_FAST_GRAM_MMA` | lane/apple-fast-gap-linalg2-pca @ 474241154 |
| `CHOL_FAST_BLOCKED` | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `CHOL_FAST_BLOCKED + SVD_FAST_CHOLQR` | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `DECOMP_FAST_DICT_UPDATE` | lane/apple-fast-decomp-sparse @ 5fb1740cd |
| `DECOMP_FAST_GEMM_TILED` | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `DECOMP_FAST_LASSO_BLOCK` | lane/apple-fast-decomp-sparse @ 5fb1740cd |
| `DECOMP_FAST_DICT_DEV` | lane/apple-fast-gap-clus3 @ 43bc2906c |
| `DECOMP_FAST_LASSO_GRP` | lane/apple-fast-gap-clus3 @ 43bc2906c |
| `DECOMP_FAST_OMP_BLOCK` | lane/apple-fast-decomp-sparse @ 5fb1740cd; lane/apple-fast-decomp-sparse @ 5fb1740cd -> lane/apple-fast-rec-decomp |
| `FA_ALL` | lane/apple-fast-fa @ 3efbce2af |
| `FA_EIG_SMALL + FA_ITER_DEVICE` | lane/apple-fast-fa @ 3efbce2af |
| `FA_EIG_SMALL + FA_ITER_DEVICE + FA_LL_DEVICE` | lane/apple-fast-fa @ 3efbce2af |
| `FA_FAST_QRR` | lane/apple-fast-decomp-linalg @ 74d52352b; lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `FA_GRAM_ONCE` | lane/apple-fast-fa @ 3efbce2af |
| `FA_ITER_DEVICE` | lane/apple-fast-fa @ 3efbce2af |
| `FA_ITER_DEVICE + FA_LIVEBUF` | lane/apple-fast-fa @ 3efbce2af |
| `FA_TRANSFORM_FUSED` | lane/apple-fast-fa @ 3efbce2af |
| `LU_FAST_PIVOT_GRID` | lane/apple-fast-decomp-linalg @ 74d52352b |
| `DECOMP_SDK_NNNT` | lane/apple-fast-decomp-sdk-control @ 73bb9ac6e |
| `LU_FAST_MMA` | lane/apple-fast-w2-linalg @ 91a6573f5 (base 254e50a01) |
| `MCD_DEVICE_CSTEPS` | lane/apple-fast-robust @ cfdb95e48 |
| `ANN3_COARSE_SEED + IVF_FAST_SEED_DEVICE` | lane/apple-fast-fastonly2 @ eca3e33b6 (via lane/apple-fast-m2b1) |
| `QR_FAST_DEV` | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `SVD_FAST_CHOLQR` | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp |
| `XD_FAST_CLS2_GRP_DEVSCAN` | lane/apple-fast-gap-cls2 @ 72602a339 |
| `XD_FAST_CLS2_GRP_NOSCAN (+ _GRP_LAZY)` | lane/apple-fast-gap-cls2 @ 72602a339 |
| MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_XD_FAST_CLS2_GRP_LAZY | lane/apple-fast-gap-cls2@72602a339 |
| `LLE_FAST_DEV_F0` (env `MOJOLEARN_LLE_FAST_DEV_F0_OFF=1` off) | lane/apple-fast-gap-manprep @ 9130a81bc |

### Cluster 38

Record file and section: [Cluster (38)](../docs/apple-fast/EXPERIMENTS.md#cluster-38).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `GMM_FAST_BIG_CHOL` | lane/apple-fast-linear @ 1c7c213f8 |
| `MBK_ZEROCOPY` | lane/apple-fast-mbkspeed @ 44e6d93e6 |
| `X_CLUSTER_FAST_MEANSHIFT` | lane/apple-fast-cluster @ f707846c8 |
| `X_CLUSTER_FAST_MINIBATCH` | lane/apple-fast-cluster @ f707846c8 |
| `GMM_FAST_BIG_CHOL + GMM_FAST_ESTEP_STACK + GMM_FAST_GRID_COV` | lane/apple-fast-linear @ 1c7c213f8 |
| `GMM_FAST_ESTEP_STACK` | lane/apple-fast-linear @ 1c7c213f8 |
| `GMM_FAST_GRID_COV` | lane/apple-fast-linear @ 1c7c213f8 |
| `KMEANS_FAST_ROWNORM` | lane/apple-fast-core @ 9a31ebb4c |
| `KMEANS_FAST_SKIP_PREDICT` | lane/apple-fast-core @ 9a31ebb4c |
| `AFFINITY_FAST_LOOP` | lane/apple-fast-cluster2 @ ded4ea07b |
| `AP_EXACT` | lane/apple-fast-cluster2 @ ded4ea07b |
| `AP_SPLIT` | lane/apple-fast-cluster2 @ ded4ea07b |
| `BGMM_ENT` | lane/apple-fast-cluster2 @ ded4ea07b |
| `BGMM_ESTEP1` | lane/apple-fast-cluster2 @ ded4ea07b |
| `BGMM_FAST_MAHAL_GEMM` | lane/apple-fast-cluster2 @ ded4ea07b |
| `BGMM_FAST_MOMENTS_GEMM` | lane/apple-fast-cluster2 @ ded4ea07b |
| `BISECT_FAST_RESIDENT` | lane/apple-fast-cluster2 @ ded4ea07b |
| `CC_FAST` | lane/apple-fast-graph @ 1fa36a7ec; ported lane/apple-fast-rec-misc |
| `DBSCAN_FAST_CC_BATCH` | lane/apple-fast-core @ 9a31ebb4c |
| `DBSCAN_FAST_DENSEBALL` | lane/apple-fast-dbscantaxi @ 1febff7df; ported lane/apple-fast-rec-misc |
| `DBSCAN_FAST_SCAN` | lane/apple-fast-core @ 9a31ebb4c |
| `HDBSCAN2_ALL` | lane/apple-fast-hdbscan2 @ 2fdb9114f |
| `HDB_CORE_TILE` | lane/apple-fast-batchv @ c7ede6e47 |
| `HDB_DEV_BORUVKA` | lane/apple-fast-hdbscan2 @ 2fdb9114f |
| `HDB_LINKAGE_DEVICE` | lane/apple-fast-hdbscan2 @ 2fdb9114f |
| `HDB_ONE_SYNC` | lane/apple-fast-batchv @ c8251211d |
| `HDB_SELECT_DEVICE` | lane/apple-fast-hdbscan2 @ 2fdb9114f |
| `HDB_SMR_TILED` | lane/apple-fast-batchv @ c7ede6e47 |
| `OPTICS2_ALL` | lane/apple-fast-batch @ 3150d75c1 |
| `OPTICS_CORE_SQ` | lane/apple-fast-batch @ 3150d75c1 |
| `OPTICS_FAST_DEVICE_ORDER` | lane/apple-fast-cluster2 @ ded4ea07b |
| `OPTICS_FRONTIER_DEVICE` | lane/apple-fast-opv @ 9a844772e |
| `OPTICS_LIVEBUF` | lane/apple-fast-opv @ 9a844772e |
| `OPTICS_STEP_BATCH` | lane/apple-fast-batch @ 3150d75c1 |
| `X_CLUSTER_FAST_CLS2_MBK_G128 / _MBK_FIN / _MBK_POOL` | lane/apple-fast-gap-cls2 @ 72602a339 |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL | lane/apple-fast-gap-cls2@72602a339 |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128 | lane/apple-fast-gap-cls2@72602a339 (deleted before merge) |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN | lane/apple-fast-gap-cls2@72602a339 (deleted before merge) |
| `RESAMPLE_FAST_IDX_DIRECT` | lane/apple-fast-gap-manprep @ c90b63b26 |
| `BISECT_FAST_ZEROCOPY` | lane/apple-fast-gap-clus3 @ 43bc2906c |
| `X_CLUSTER_FAST_CLS3_MBK_ROWGRP` | lane/apple-fast-gap-clus3 @ 43bc2906c |

### Time series 34

Record file and section: [Time series (34)](../docs/apple-fast/EXPERIMENTS.md#time-series-34).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `ARIMA_FAST_ASYNC` | lane/apple-fast-gap-arima @ d967c0121 |
| `ARIMA_FAST_EVAL_WS` | lane/apple-fast-tsa @ cc25a4c7f |
| `ARIMA_FAST_EVAL_WS + ARIMA_FAST_LLONLY` | lane/apple-fast-tsa @ cc25a4c7f |
| `ARIMA_FAST_LLONLY` | lane/apple-fast-tsa @ cc25a4c7f |
| `ETS_TEAM` | lane/apple-fast-ets @ 03b4948d3 |
| `GARCH_COOP` | lane/apple-fast-garchspeed @ d6effc734 |
| `PROPHET_COOP` | lane/apple-fast-prophetspeed @ 1f75f9068 |
| `SELECT_D` | lane/apple-fast-gap-tsa @ e9da47064 |
| `SEQ_CROSTON_REG` | lane/apple-fast-seq @ b185f069f |
| `SEQ_FAST_FMA` | lane/apple-fast-tier @ 95a09d1fd |
| `SEQ_FAST_FMA + SEQ_THETA_REG` | lane/apple-fast-tier @ 95a09d1fd |
| `SEQ_FAST_THETA_SNAP` | lane/apple-fast-regress @ 93951fafb |
| `SEQ_FAST_THETA_SPEC` | lane/apple-fast-gap-tsa @ e9da47064 |
| `SEQ_FAST_VAR_ONECOPY` | lane/apple-fast-gap-tsa @ e9da47064 |
| `SEQ_GARCH_GRID` | lane/apple-fast-seq @ b185f069f |
| `SEQ_GARCH_GRID + SEQ_GARCH_REG` | lane/apple-fast-seq @ b185f069f |
| `SEQ_GARCH_REG` | lane/apple-fast-seq @ b185f069f |
| `SEQ_THETA_REG` | lane/apple-fast-tier @ 95a09d1fd |
| `TSA2_STL` | lane/apple-fast-tsa2 @ b27c8169b |
| `TSA2_VAR` | lane/apple-fast-tsa2 @ b27c8169b |
| `TSA_FAST_KPSS_PACK` | lane/apple-fast-gap-tsa @ e9da47064 |
| `TSA_FAST_SELD_FUSED` | lane/apple-fast-gap-tsa @ e9da47064 |
| `ARIMA_FAST_LS_NOREAD` | lane/apple-fast-gap-arima @ d967c0121 |
| `ARIMA_FAST_P_FIX` | lane/apple-fast-gap-arima @ d967c0121 |
| `ARIMA_FAST_CONST_BOTH` | lane/apple-fast-arima-quality @ 9ca86df3e |
| `ARIMA_FAST_ROOT_CHECK` | lane/apple-fast-arima-quality @ 9ca86df3e |
| `ARIMA_FAST_KPSS_D` | lane/apple-fast-arima-quality @ 9ca86df3e |
| `SEQ_FAST_VAR_ONECOPY + SEQ_FAST_VAR_SPEC` | lane/apple-fast-gap-tsa @ e9da47064 |
| `SEQ_FAST_VAR_SPEC` | lane/apple-fast-gap-tsa @ e9da47064 |
| `TSA2_KPSS` | lane/apple-fast-gap-tsa @ e9da47064; lane/apple-fast-tsa2 @ b27c8169b |
| (baseline, no switch) | lane/apple-fast-seq @ b185f069f |
| `SEQ_FAST_THETA_HOIST` | lane/apple-fast-gap-tsa @ e9da47064 |
| `SEQ_GARCH_HOST_MAX` | lane/apple-fast-seq @ b185f069f |

### Kernel GP 11

Record file and section: [Kernel / GP (11)](../docs/apple-fast/EXPERIMENTS.md#kernel--gp-11).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `APPLE_FAST_GEMM_PINNED` | lane/apple-fast-tier @ 95a09d1fd |
| `KAPPROX_DEVICE` | lane/apple-fast-kapprox @ 10d5a7970 |
| `KERNEL_FAST_GPR_RESIDENT` | lane/apple-fast-kernel @ 9e851777c |
| `SPARSE_RP_DEVICE` | lane/apple-fast-kapprox @ 10d5a7970 |
| `SVGP_FAST_GPU` | lane/apple-fast-neighbors2 @ 5fb6edd3f |
| `XN_FAST_TILED_RBF` | lane/apple-fast-neighbors2 @ 5fb6edd3f |
| `XN_PCS_SPARSE` | lane/apple-fast-neighbors2 @ 5fb6edd3f |
| (baseline, no switch) | lane/apple-fast-kapprox @ 10d5a7970 |
| `KERNEL_FAST_NYS_RR_EIGH` | lane/apple-fast-kernel @ 9e851777c |
| `KPCA_RESIDENT` | lane/apple-fast-kapprox @ 10d5a7970 |
| `XN_FAST_CLS2_OCSVM_RES / _2L / _CHUNK256` | lane/apple-fast-gap-cls2 @ 72602a339 |

### Neural 99

Record file and section: [Neural (99)](../docs/apple-fast/EXPERIMENTS.md#neural-99).

| Recorded experiment or control | Original branch and commit references |
| --- | --- |
| `MOE_DEVGROUP` | lane/apple-fast-moespeed @ b9bc99e0f |
| `MOE_REGTILE` | lane/apple-fast-moespeed @ b9bc99e0f |
| `OPT_PIPE_DOWN` | lane/apple-fast-optspeed @ 7a5b3fb2c |
| `OPT_ZERO_OPEN` | lane/apple-fast-optspeed @ 7a5b3fb2c |
| `SEQ_FAST_PIPE_DOWN` | lane/apple-fast-regress @ 93951fafb |
| `SEQ_FAST_PIPE_UP` | lane/apple-fast-regress @ 93951fafb |
| `OPT_RAW_UP` | lane/apple-fast-optspeed @ 7a5b3fb2c |
| (baseline, no switch) | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_ARENA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_FLASH` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_FUSE_MLP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_FUSE_PRE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_GQA_TILE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_NORM_SG` | lane/apple-fast-neural @ 600237d7c |
| `AFN_ATTN_ROPE_CACHE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_CNN_DIRECT` | lane/apple-fast-neural @ 600237d7c |
| `AFN_EMB_ATOMIC_BWD` | lane/apple-fast-neural @ 600237d7c |
| `AFN_EPI_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_EPI_ALL + AFN_SAMBA_FUSE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_ALL + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_BIGTILE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_DBUF + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_DIRECT_B + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_SWIZZLE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_BF16_MMA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_INT8_MMA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_SIMDGROUP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_SIMDGROUP + AFN_GEMM_SPLITK + AFN_LM_WGRAD_SPLIT` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_SIMDGROUP + AFN_LMGRAD_ALL + AFN_LM_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_SPLITK` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_TILESHAPE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LMGRAD_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_BWD_EPILOGUE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_BWD_FUSE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_BWD_NORM1_RESID` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_BWD_NOSYNC` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_HEAD_FUSE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_NOSYNC` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_PARAM_VIEWS` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LM_WGRAD_SPLIT` | lane/apple-fast-neural @ 600237d7c |
| `AFN_LOSS_FUSED` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA1_CHUNKSCAN` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA1_FUSE_IN` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA2_SSD_MMA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA3_BWD_ARENA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA3_BWD_CHUNK` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA3_SISO_FUSED` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA_ARENA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA_DEVICE_REFUSAL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA_PROJ_EPILOGUE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MAMBA_PROJ_EPILOGUE + AFN_MAMBA_PROJ_SPLITK` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MLP_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MLP_FUSED_STEP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MLP_MULTISTEP` | lane/apple-fast-neural @ 600237d7c |
| `AFN_MLP_RESIDENT` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPTIM_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPT_CLIP_FUSE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPT_FUSE_SCAN` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPT_MULTITENSOR` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPT_RESIDENT_STATE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_OPT_VEC4` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_ALL` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_ARENA` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_DEVICE_ADMIT` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_EMB_ATOMIC` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_FUSE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_SAMBA_FUSE + AFN_SAMBA_HEAD_GEMM` | lane/apple-fast-neural @ 600237d7c |
| `AF_FAST_NOFILL` | lane/apple-fast-gap-optim @ cf4513f8a |
| `AF_FAST_RESIDENT` | lane/apple-fast-gap-optim @ cf4513f8a |
| `BPE_ALL` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_ENCODE_DEVICE` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_ENCODE_DEVICE + BPE_LIVEBUF` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_GROUP_FILTER + BPE_TRAIN_DEVICE` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_LIVEBUF + BPE_TRAIN_DEVICE` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_MERGE_BATCH + BPE_TRAIN_DEVICE` | lane/apple-fast-bpe @ 3355d37b3 |
| `BPE_TRAIN_DEVICE` | lane/apple-fast-bpe @ 3355d37b3 |
| `LN_FAST_NOFILL` | lane/apple-fast-gap-optim @ cf4513f8a |
| `MOE_FAST_MMA (+ _KB32, _WIDE, _PF)` | lane/apple-fast-gap-misc @ 5e2eec7a3 |
| `OPT_FAST_MAP_DOWN` | lane/apple-fast-gap-optim @ cf4513f8a |
| `OPT_FAST_PIPE_CH` | lane/apple-fast-gap-optim @ cf4513f8a |
| `OPT_FAST_RAW_DOWN` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SCHED_FAST_INLINE` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SCHED_FAST_INLINE,SCHED_FAST_P64` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SCHED_FAST_P64` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SEQ_FAST_LSTM_SCAN` | lane/apple-fast-gap-lstm @ 0d6cbc821 |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM` | lane/apple-fast-gap-lstm @ 0d6cbc821 |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM + SEQ_FAST_LSTM_WGRAD` | lane/apple-fast-gap-lstm @ 0d6cbc821 |
| `SEQ_FAST_LSTM_WGRAD` | lane/apple-fast-gap-lstm @ 0d6cbc821 |
| `SEQ_FAST_MAP_DOWN` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SEQ_FAST_PIPE_CH` | lane/apple-fast-gap-optim @ cf4513f8a |
| `SEQ_FAST_RAW_DOWN` | lane/apple-fast-gap-optim @ cf4513f8a |
| `AFN_GEMM_EPILOGUE` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_KB` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM_CORES` | lane/apple-fast-neural @ 600237d7c |
| `AFN_GEMM2_SWZ_G` | lane/apple-fast-neural @ 600237d7c |

### Gap kapprox2 lane apple-fast-gap-kapprox2 Oct 3

Record file and section: [Gap kapprox2 (lane/apple-fast-gap-kapprox2, Oct 3)](../docs/apple-fast/EXPERIMENTS.md#gap-kapprox2-laneapple-fast-gap-kapprox2-oct-3).

- `KSHAP_FAST_BATCH` (`_OFF`)
- `KM_FAST_PTR_IN` (`_OFF`)
- `KERNEL_FAST_NYS_RR_EIGH` (`_OFF`)
- `SPLINE_FAST_FUSED` (`_OFF`)
- `SVGP_FAST_SYMTILE` (`_OFF`)
- `SVGP_FAST_COLSPLIT` (`_OFF`)
- `ACHI2_FAST_DEVCHECK`
- `XD_FAST_GRP_FUSED` (`_OFF`)
- `KSHAP_FAST_SIGNGRAM`

### GPU purity 2 lane apple-fast-purity2 Oct 3

Record file and section: [GPU purity 2 (lane/apple-fast-purity2, Oct 3)](../docs/apple-fast/EXPERIMENTS.md#gpu-purity-2-laneapple-fast-purity2-oct-3).

- `PURITY2_1` (`_OFF`)
- `PURITY2_2` (`_OFF`)

### Gap linalg2 kernel-pca incremental-pca lane apple-fast-gap-linalg2-kpca Oct 3

Record file and section: [Gap linalg2, kernel-pca + incremental-pca (lane/apple-fast-gap-linalg2-kpca, Oct 3)](../docs/apple-fast/EXPERIMENTS.md#gap-linalg2-kernel-pca--incremental-pca-laneapple-fast-gap-linalg2-kpca-oct-3).

- `KPCA_FAST_LANCZOS_DEV`
- `IPCA_FAST_DEV`

### Oct 4 manager takeover

Record file and section: [Oct 4 manager takeover](../docs/apple-fast/EXPERIMENTS.md#oct-4-manager-takeover).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### Label-direct promotion 2026-10-04

Record file and section: [Label-direct promotion (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#label-direct-promotion-2026-10-04).

- `MOJOLEARN_LABEL_DIRECT` -> `MOJOLEARN_LABEL_DIRECT_OFF`

### MCD compatibility repair review 2026-10-04

Record file and section: [MCD compatibility repair review (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#mcd-compatibility-repair-review-2026-10-04).

- `MOJOLEARN_MCD_BATCH_COMPAT`

### MCD MMA repair candidate 2026-10-04

Record file and section: [MCD MMA repair candidate (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#mcd-mma-repair-candidate-2026-10-04).

- `MOJOLEARN_MCD_BATCH_MMA`
- `MOJOLEARN_MCD_BMMA`
- `MOJOLEARN_MCD_WIDE`

### AutoARIMA order batching current-main integration 2026-10-04

Record file and section: [AutoARIMA order batching current-main integration (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#autoarima-order-batching-current-main-integration-2026-10-04).

- `ARIMA_ORDER_BATCH`, original small quality
- `ARIMA_ORDER_BATCH`, fused-tail integration

### M3 repaired checks 2026-10-04 09 00 UTC

Record file and section: [M3 repaired checks — 2026-10-04 09:00 UTC](../docs/apple-fast/EXPERIMENTS.md#m3-repaired-checks--2026-10-04-0900-utc).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### PowerTransformer compensated score failure 2026-10-04

Record file and section: [PowerTransformer compensated score failure (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#powertransformer-compensated-score-failure-2026-10-04).

- `PT_SCORE`

### AutoARIMA order-batching default promotion prepared 2026-10-04

Record file and section: [AutoARIMA order-batching default promotion prepared (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#autoarima-order-batching-default-promotion-prepared-2026-10-04).

- `ARIMA_ORDER_BATCH` → default + `ARIMA_ORDER_BATCH_OFF`

### Wave 2 kernel features lane apple-fast-w2-kfeat 2026-10-04 base 254e50a01

Record file and section: [Wave 2 kernel features (lane/apple-fast-w2-kfeat, 2026-10-04, base 254e50a01)](../docs/apple-fast/EXPERIMENTS.md#wave-2-kernel-features-laneapple-fast-w2-kfeat-2026-10-04-base-254e50a01).

- `KM_FAST_RBF_RESIDENT`
- `XN_FAST_ACHI2_DEVSCAN`
- `XN_FAST_SCHI2_MOJO_MT`
- `XD_FAST_SRP_STRAT`

### Wave 3 kernel features lane apple-fast-w3-kfeat 2026-10-04 all three DEFAULT

Record file and section: [Wave 3 kernel features (lane/apple-fast-w3-kfeat, 2026-10-04; all three DEFAULT)](../docs/apple-fast/EXPERIMENTS.md#wave-3-kernel-features-laneapple-fast-w3-kfeat-2026-10-04-all-three-default).

- `XN_FAST_ACHI2_DEVSCAN` + size gate
- `XN_FAST_SCHI2_LAZYW`
- `KM_FAST_RBF_STAGED`

### SVGP wave 2 candidates lane apple-fast-w2-svgp 2026-10-04 OPEN

Record file and section: [SVGP wave 2 candidates (lane apple-fast-w2-svgp, 2026-10-04, OPEN)](../docs/apple-fast/EXPERIMENTS.md#svgp-wave-2-candidates-lane-apple-fast-w2-svgp-2026-10-04-open).

- `SVGP_FAST_BLKCHOL`
- `SVGP_FAST_RBFTILE`
- `SVGP_FAST_BSPLIT`

### MiniBatchKMeans W2 residuals lane apple-fast-w2-clres 2026-10-04

Record file and section: [MiniBatchKMeans W2 residuals (lane/apple-fast-w2-clres, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#minibatchkmeans-w2-residuals-laneapple-fast-w2-clres-2026-10-04).

- `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP`
- `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG`

### Manager verdicts 2026-10-04 session 2 rejected or held candidates stay on their branches

Record file and section: [Manager verdicts, 2026-10-04 session 2 (rejected or held; candidates stay on their branches)](../docs/apple-fast/EXPERIMENTS.md#manager-verdicts-2026-10-04-session-2-rejected-or-held-candidates-stay-on-their-branches).

- `MOJOLEARN_EIGH_TANGENT_CACHE`
- `MOJOLEARN_CAGRA_FAST_IVFG_LOWD`
- `MOJOLEARN_SHAP_FAST_PIPE`
- `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG`
- `MOJOLEARN_ARIMA_FIT_GROUPS`

### PT centered-score WIP checkpoint 2026-10-04

Record file and section: [PT centered-score WIP checkpoint (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#pt-centered-score-wip-checkpoint-2026-10-04).

- `MOJOLEARN_GBDT_DW_FLAT_GRID`
- `MOJOLEARN_ARIMA_SLAB`
- `MOJOLEARN_X_PREP_PINNED_OUT`
- `MOJOLEARN_LU_FAST_TSLU`
- `MOJOLEARN_SEQ_FAST_VAR_FUSED`
- `MOJOLEARN_PREP3_MAXABS_POOL` / rollback `MOJOLEARN_PREP3_MAXABS_POOL_OFF`
- `MOJOLEARN_KPCA_RESIDENT` (rollback `MOJOLEARN_KPCA_RESIDENT_OFF`)
- `MOJOLEARN_CHOL_FAST_NB512`
- `MOJOLEARN_RESAMPLE_FAST_ROW_GATHER`
- `MOJOLEARN_XN_FAST_NAN_FIT_LEAN`
- `MOJOLEARN_MCD_SKIP_PINVH` (rollback `MOJOLEARN_MCD_SKIP_PINVH_OFF`)
- `MOJOLEARN_MCD_SKIP_PINVH` + `MOJOLEARN_MCD_DEFLATE`
- `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS` / rollback `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF`
- `MOJOLEARN_LU_FAST_MMA_DBUF`
- `MOJOLEARN_DECOMP_FAST_MMA_K16`

### W4 PCA pool isolated promotion 2026-10-04

Record file and section: [W4 PCA pool isolated promotion (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#w4-pca-pool-isolated-promotion-2026-10-04).

- `MOJOLEARN_ARIMA_FAST_SCALAR_LL`
- `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE` / rollback `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE_OFF`
- `MOJOLEARN_PSHAP_DELTA` / `_OFF`

### Catalog shared GEMM screen and actual caller expansion 2026-10-04

Record file and section: [Catalog shared GEMM screen and actual caller expansion (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#catalog-shared-gemm-screen-and-actual-caller-expansion-2026-10-04).

- `MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU` / `_OFF`

### Shared caller quality and process checkpoint

Record file and section: [Shared caller quality and process checkpoint](../docs/apple-fast/EXPERIMENTS.md#shared-caller-quality-and-process-checkpoint).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### 2026-10-04 scoped decomp PCA direct G1 G2 OPEN default OFF

Record file and section: [2026-10-04 scoped decomp/PCA direct G1/G2 — OPEN, default OFF](../docs/apple-fast/EXPERIMENTS.md#2026-10-04-scoped-decomppca-direct-g1g2--open-default-off).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### Recovered branch candidate 2026-10-04

Record file and section: [Recovered branch candidate, 2026-10-04](../docs/apple-fast/EXPERIMENTS.md#recovered-branch-candidate-2026-10-04).

- `MOJOLEARN_RESAMPLE_FAST_GATHER`

### MCD non-split batched G1 covariance remote source-only 2026-10-04

Record file and section: [MCD non-split batched G1 covariance (remote source-only, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#mcd-non-split-batched-g1-covariance-remote-source-only-2026-10-04).

- `MOJOLEARN_MCD_FAST_G1_GRAM`

### Main integration checkpoint remote manager 2026-10-04

Record file and section: [Main integration checkpoint, remote manager 2026-10-04](../docs/apple-fast/EXPERIMENTS.md#main-integration-checkpoint-remote-manager-2026-10-04).

The linked section records the experiment narrative, source locations, followup or disposition; it has no separately indexed control-name table.

### Completed matched timing 2026-10-04

Record file and section: [Completed matched timing, 2026-10-04](../docs/apple-fast/EXPERIMENTS.md#completed-matched-timing-2026-10-04).

- `MOJOLEARN_RSVD_FAST_DIRECT_IN` (rollback `MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF`)
- `MOJOLEARN_LLE_FAST_DEV_LU`

### Optimizer layernorm schedule recovery lane apple-fast-rec-optim 2026-10-04

Record file and section: [Optimizer / layernorm / schedule recovery (lane apple-fast-rec-optim, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#optimizer--layernorm--schedule-recovery-lane-apple-fast-rec-optim-2026-10-04).

- `TRAIN_OPT_FAST_PIPE_DOWN`
- `OPT_FAST_STREAM` + `OPT_RAW_UP` (pair)
- `OPT_FAST_STREAM` (alone, staged uploads)
- `AFN_OPT_FUSE_SCAN`
- `AFN_OPT_VEC4`
- `AFN_OPT_RESIDENT_STATE`
- `AFN_OPTIM_ALL`
- `TRAIN_OPT_FAST_PIPE_DOWN + AFN_OPTIM_ALL`
- `OPT_FAST_MAP_DOWN` / `OPT_FAST_RAW_DOWN` / `OPT_FAST_PIPE_CH=524288`
- `LN_FAST_NOFILL`
- `SEQ_FAST_MAP_DOWN`
- `SEQ_FAST_RAW_DOWN`
- `MOJOLEARN_SCHED_FAST_INLINE=1` (env)
- `SEQ_FAST_LSTM_SCAN + _SCAN_SMEM + _WGRAD` (bundle)

### Recovery lane apple-fast-rec-misc 2026-10-04

Record file and section: [Recovery lane apple-fast-rec-misc (2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#recovery-lane-apple-fast-rec-misc-2026-10-04).

- `MOJOLEARN_MCD_ORDERED_COV`
- `MOJOLEARN_EIGH_FAST_PANEL_DF`
- `MOJOLEARN_SVD_FAST_AW`
- `MOJOLEARN_LLE_FAST_DEV_LU`
- `MOJOLEARN_ARD_SIGMA_QOLD` (A = old sigma; B = default)
- `MOJOLEARN_DT_BINS_QOLD` (A = 128 bins; B = default)
- `LU_QFIX` (QOLD `MOJOLEARN_LU_QOLD`)
- `SVD_QFIX` (QOLD `MOJOLEARN_SVD_QOLD`)
- `TSVD_QFIX` (QOLD `MOJOLEARN_TSVD_QOLD`)
- `KNN_FAST_REFINE` (old: `MOJOLEARN_KNN_REFINE_QOLD`)
- `IVF_COARSE_FAISS_INIT` (old: `MOJOLEARN_IVF_COARSE_INIT_QOLD`)

### Quality fixes classifiers lane apple-fast-q-clf 2026-10-04 reconciled 2026-10-05 from Verdicts batch 4

Record file and section: [Quality fixes, classifiers (lane/apple-fast-q-clf, 2026-10-04; reconciled 2026-10-05 from Verdicts batch 4)](../docs/apple-fast/EXPERIMENTS.md#quality-fixes-classifiers-laneapple-fast-q-clf-2026-10-04-reconciled-2026-10-05-from-verdicts-batch-4).

- `MOJOLEARN_SGD_PERC_QOLD` (A) / none (B)
- `MOJOLEARN_PROBA64_QOLD` (A) / none (B)

### Verdicts batch 3 M3 afc_ab_def lane apple-fast-rec-ab3 0ca521cc5 2026-10-04

Record file and section: [Verdicts batch 3 (M3 afc_ab_def, lane/apple-fast-rec-ab3 @ 0ca521cc5, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#verdicts-batch-3-m3-afc_ab_def-laneapple-fast-rec-ab3--0ca521cc5-2026-10-04).

- `X_PREP_FAST_TE_GLOBAL`
- `X_PREP_FAST_TE_ENC`
- `PREP2_FAST_EIGH_BLOCK`
- `HUBER_DEVICE_LBFGS`
- `FA_FAST_QRR`
- `CHOL_FAST_BLOCKED`
- `FA_GRAM_ONCE`
- `FA_ITER_DEVICE`
- `FA_ALL`
- `MOJOLEARN_FA_GRAM_QOLD` (A: `-D MOJOLEARN_FA_ALL -D MOJOLEARN_FA_GRAM_QOLD`) / none (B: `-D MOJOLEARN_FA_ALL`), route token `MOJOLEARN_FA_GRAM_DF`

### Verdicts batch 4 M3 afc_ab_def full board size 1 run per arm 2026-10-04 tags rab lane apple-fast-verdicts-4

Record file and section: [Verdicts batch 4 (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tags rab*; lane/apple-fast-verdicts-4)](../docs/apple-fast/EXPERIMENTS.md#verdicts-batch-4-m3-afc_ab_def-full-board-size-1-run-per-arm-2026-10-04-tags-rab-laneapple-fast-verdicts-4).

- `CV_FAST_SLICE` (rollback `MOJOLEARN_CV_FAST_SLICE_OFF`)
- `CV_FAST_TRUST_FOLDS` (rollback `MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF`)
- `SHAP_TREE_TAB` (rollback `MOJOLEARN_SHAP_TREE_TAB_OFF`)
- `CC_FAST` (rollback `MOJOLEARN_CC_FAST_OFF`)
- `MCD_DEFLATE` (rollback `MOJOLEARN_MCD_DEFLATE_OFF`)
- `SYM_CTR_ALL` (rollback `MOJOLEARN_SYM_CTR_ALL_OFF`)
- `SYM_EST_ALL` (rollback `MOJOLEARN_SYM_EST_ALL_OFF`)
- `PL_GROUP_NARROW` (rollback `MOJOLEARN_PL_GROUP_NARROW_OFF`)
- `CTR_INDEX_FUSED`
- `CTR_SORT_ONCE`
- `EST_STATS_FUSED`
- `FA_ITER_DEVICE` + `FA_GRAM_DF` (rollback `MOJOLEARN_FA_ITER_DEVICE_OFF`)
- `FA_ALL` (with FA_GRAM_DF)
- `FA_TRANSFORM_FUSED`
- `LU_QFIX` (old: `MOJOLEARN_LU_QOLD`)
- `PROBA64` (old: `MOJOLEARN_PROBA64_QOLD`)
- `RF_DT_DEFAULT_BINS` 256 (old: `MOJOLEARN_DT_BINS_QOLD`)
- `ARD_FAST_EQ` (old: `MOJOLEARN_ARD_SIGMA_QOLD`)
- `TSVD_QFIX` (old: `MOJOLEARN_TSVD_QOLD`)
- `SVD_QFIX` (now opt-in `MOJOLEARN_SVD_QFIX`)
- `SGD_PERC_AVG` (now opt-in `MOJOLEARN_SGD_PERC_AVG`)
- `IVF_COARSE_FAISS_INIT` (now opt-in `MOJOLEARN_IVF_COARSE_FAISS_INIT`)
- `KNN_FAST_REFINE` (now opt-in `MOJOLEARN_KNN_FAST_REFINE`)
- `EIGH_FAST_PANEL_DF`
- `MCD_ORDERED_COV`
- `MCD_FAST_G1_GRAM`
- `DBSCAN_FAST_DENSEBALL`
- `OPT_FAST_MAP_DOWN`
- `OPT_FAST_RAW_DOWN`
- `SEQ_FAST_RAW_DOWN`
- `SYM_HIST_ALL`
- `SYM_FEAT_ALL`
- `SYM_ITER_ALL`
- `SYM_MULTI_ALL`

### lane apple-fast-s-seq 2026-10-05 LSTM scan fix width sequence targets

Record file and section: [lane/apple-fast-s-seq (2026-10-05): LSTM scan fix + width, sequence targets](../docs/apple-fast/EXPERIMENTS.md#laneapple-fast-s-seq-2026-10-05-lstm-scan-fix--width-sequence-targets).

- `SEQ_FAST_LSTM_SCAN` (fixed)
- `SEQ_FAST_LSTM_SCAN + _SCAN_SMEM`
- `SEQ_FAST_LSTM_SCAN + _SCAN_SMEM + _WGRAD` (bundle)
- `SEQ_FAST_LSTM_SCAN + _SCAN_WIDE (+ _WGRAD)`

### Benchmark-shaped limits removed round 2 lane apple-fast-no-narrow-2 2026-10-04

Record file and section: [Benchmark-shaped limits removed, round 2 (lane/apple-fast-no-narrow-2, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#benchmark-shaped-limits-removed-round-2-laneapple-fast-no-narrow-2-2026-10-04).

- `MOJOLEARN_LEGACY_NARROW_CD_GRAM`
- `MOJOLEARN_LEGACY_NARROW_FIVF_DIM`
- `MOJOLEARN_LEGACY_NARROW_ACHI2_DEVSCAN`
- `MOJOLEARN_LEGACY_NARROW_SL_BORUVKA`
- `MOJOLEARN_LEGACY_NARROW_SPECTRAL_NCV`
- `MOJOLEARN_LEGACY_NARROW_MC_HESS_BATCH`
- `MOJOLEARN_LEGACY_NARROW_ET_RM`
- `MOJOLEARN_LEGACY_NARROW_SYM_RIDX`
- `MOJOLEARN_LEGACY_NARROW_GEMV_PINNED`
- `MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS`
- `MOJOLEARN_LEGACY_NARROW_EIGH_TD`
- `MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N`
- `MOJOLEARN_LEGACY_NARROW_FX_D`

### lane apple-fast-general-speed verdicts 2026-10-05 lane apple-fast-verdicts-5

Record file and section: [lane/apple-fast-general-speed verdicts (2026-10-05, lane/apple-fast-verdicts-5)](../docs/apple-fast/EXPERIMENTS.md#laneapple-fast-general-speed-verdicts-2026-10-05-laneapple-fast-verdicts-5).

- `ARD_EQ_ONEPASS` (rollback `ARD_EQ_ONEPASS_OFF`)

### Small launch-bound inputs lane apple-fast-s-small 2026-10-05 READY-AB

Record file and section: [Small launch-bound inputs (lane/apple-fast-s-small, 2026-10-05, READY-AB)](../docs/apple-fast/EXPERIMENTS.md#small-launch-bound-inputs-laneapple-fast-s-small-2026-10-05-ready-ab).

- `MOJOLEARN_KM_FAST_RBF_PIPE`

### Lane apple-fast-s-ts round 2 speed 2026-10-04 READY-AB candidates

Record file and section: [Lane apple-fast-s-ts (round 2 speed, 2026-10-04): READY-AB candidates](../docs/apple-fast/EXPERIMENTS.md#lane-apple-fast-s-ts-round-2-speed-2026-10-04-ready-ab-candidates).

- `MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT`
- `MOJOLEARN_ARIMA_FAST_D_CONCURRENT` (with GROUPS_CONCURRENT)
- `MOJOLEARN_ARIMA_FAST_SEARCH_REUSE`
- `MOJOLEARN_ARIMA_FAST_CSS_SEARCH`
- `MOJOLEARN_ARIMA_FAST_STEPWISE`
- `MOJOLEARN_ARIMA_FAST_CSS_SEARCH` + `MOJOLEARN_ARIMA_FAST_STEPWISE`
- `MOJOLEARN_SEQ_FAST_VAR_COOP`
- `MOJOLEARN_SEQ_FAST_VAR_NODRAIN`
- `MOJOLEARN_SCHED_FAST_TABLE`

### Verdicts applied on lane apple-fast-verdicts-5 2026-10-05

Record file and section: [Verdicts applied on lane/apple-fast-verdicts-5 (2026-10-05)](../docs/apple-fast/EXPERIMENTS.md#verdicts-applied-on-laneapple-fast-verdicts-5-2026-10-05).

- `AF_FAST_RESIDENT` (rollback `AF_FAST_RESIDENT_OFF`)
- `AF_FAST_NOFILL` (rollback `AF_FAST_NOFILL_OFF`)
- `KM_FAST_RBF_PIPE` (rollback `KM_FAST_RBF_PIPE_OFF`)
- `MOE_FAST_MMA_KB32 + _WIDE + _PF` (bundle, on top of the default `MOE_FAST_MMA`)

### Speed round 2 linear algebra lane apple-fast-s-linalg 2026-10-04

Record file and section: [Speed round 2: linear algebra (lane/apple-fast-s-linalg, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#speed-round-2-linear-algebra-laneapple-fast-s-linalg-2026-10-04).

- `MOJOLEARN_TSVD_FAST_CHOLQR3`
- `MOJOLEARN_RSVD_FAST_DEVSCAN`
- `MOJOLEARN_DECOMP_FAST_ORTH_WS`
- `MOJOLEARN_MBK_FAST_DEVSCAN`
- `MOJOLEARN_PCA_FAST_COLMEAN`
- `MOJOLEARN_TSVD_FAST_POOL`
- `MOJOLEARN_TSVD_FAST_COLVAR`
- `MOJOLEARN_LU_FAST_RESIDENT`
- `MOJOLEARN_CHOL_FAST_POOLIO`

### Speed round 2 SHAP manifold resample lane apple-fast-s-shap 2026-10-04

Record file and section: [Speed round 2: SHAP / manifold / resample (lane/apple-fast-s-shap, 2026-10-04)](../docs/apple-fast/EXPERIMENTS.md#speed-round-2-shap--manifold--resample-laneapple-fast-s-shap-2026-10-04).

- `LLE_FAST_NULL_CANON` (with `LLE_FAST_DEV_LU`)
- `KSHAP_FAST_OVERLAP`
- `PSHAP_FAST_OVERLAP`
- `RESAMPLE_FAST_GATHER_NARROW`
- `RESAMPLE_FAST_TAKE`

### Verdicts applied on lane apple-fast-verdicts-6 2026-10-05

Record file and section: [Verdicts applied on lane/apple-fast-verdicts-6 (2026-10-05)](../docs/apple-fast/EXPERIMENTS.md#verdicts-applied-on-laneapple-fast-verdicts-6-2026-10-05).

- `LU_FAST_RESIDENT` (rollback `LU_FAST_RESIDENT_OFF`)
- `CHOL_FAST_POOLIO` (rollback `CHOL_FAST_POOLIO_OFF`)
- `MBK_FAST_DEVSCAN` (rollback `MBK_FAST_DEVSCAN_OFF`)
- `RSVD_FAST_DEVSCAN` + `DECOMP_FAST_ORTH_WS` (rollbacks `RSVD_FAST_DEVSCAN_OFF`, `DECOMP_FAST_ORTH_WS_OFF`)
- `TSVD_FAST_POOL` (rollback `TSVD_FAST_POOL_OFF`)
- `TSVD_FAST_COLVAR` (rollback `TSVD_FAST_COLVAR_OFF`)
- `PCA_FAST_COLMEAN` (rollback `PCA_FAST_COLMEAN_OFF`)
- `ARIMA_FAST_SEARCH_REUSE` (rollback `ARIMA_FAST_SEARCH_REUSE_OFF`)
- `SCHED_FAST_TABLE` (rollback `SCHED_FAST_TABLE_OFF`)
- `SEQ_FAST_VAR_COOP` (rollback `SEQ_FAST_VAR_COOP_OFF`)
- `HUBER_FAST_BLOCK512` (rollback `HUBER_FAST_BLOCK512_OFF`)
- `SVGP_FAST_BSPLIT` (rollback `SVGP_FAST_BSPLIT_OFF`)
- `ARIMA_FAST_GROUPS_CONCURRENT`
- `ARIMA_FAST_D_CONCURRENT`
- `TSVD_FAST_CHOLQR3`
- `SEQ_FAST_VAR_NODRAIN`
- `KSHAP_FAST_OVERLAP`
- `PSHAP_FAST_OVERLAP`
- `LLE_FAST_NULL_CANON`
- `AFN_OPT_FUSE_SCAN`, `AFN_OPT_VEC4`, `AFN_OPT_RESIDENT_STATE`
- `ARIMA_FAST_CSS_SEARCH`
- `ARIMA_FAST_STEPWISE`
- `ARIMA_FAST_CSS_SEARCH` + `ARIMA_FAST_STEPWISE`

### IDENTICAL shared GEMM screening 2026-10-05

Record file and section: [IDENTICAL shared GEMM screening, 2026-10-05](../docs/apple-fast/EXPERIMENTS.md#identical-shared-gemm-screening-2026-10-05).

- `MOJOLEARN_GEMM_ONE_PAGE`
- `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2`
- `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8`
- `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY`
- `MOJOLEARN_GEMM_KPACK_RPT4`

## Historical IDENTICAL source toggle inventory

Inventory file: [docs/identical/toggle-inventory.json](../docs/identical/toggle-inventory.json). This is a retained source-location index for 266 flag names, including rollback switches, shared primitives and pre-existing neural paths. A flag is not necessarily an independent A/B experiment. Saved line numbers can drift, so links point to the recorded files and the exact flag names provide the lookup key. Current defaults and current qualification must be read from the source and relevant experiment record.

| Control name | Recorded source files |
| --- | --- |
| `MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_ADAPT_M_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_ADA_SESSION_OFF` | [xtrees/api.mojo](../xtrees/api.mojo) |
| `MOJOLEARN_IDN_AF_RESIDENT_OFF` | [sequence/opt_resident.mojo](../sequence/opt_resident.mojo) |
| `MOJOLEARN_IDN_ALL_OFF` | [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo); [arima/impl/fast_lbfgs_async.mojo](../arima/impl/fast_lbfgs_async.mojo); [arima/impl/fast_order_search.mojo](../arima/impl/fast_order_search.mojo); [arima/impl/fast_order_state.mojo](../arima/impl/fast_order_state.mojo); [arima/impl/idn_ls_math.mojo](../arima/impl/idn_ls_math.mojo); [bindings/_mojolearn.mojo](../bindings/_mojolearn.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo); [bindings/_mojolearn_embedding.mojo](../bindings/_mojolearn_embedding.mojo); [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo); [cholesky/estimator.mojo](../cholesky/estimator.mojo); [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo); [core/ctx_key.mojo](../core/ctx_key.mojo); [core/host_tile_fold.mojo](../core/host_tile_fold.mojo); [core/xtdz_coalesced.mojo](../core/xtdz_coalesced.mojo); [dbscan/estimator.mojo](../dbscan/estimator.mojo); [dbscan/impl/dbscan.mojo](../dbscan/impl/dbscan.mojo); [dbscan/impl/runner.mojo](../dbscan/impl/runner.mojo); [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo); [decomposition/estimator.mojo](../decomposition/estimator.mojo); [decomposition/mean_switch.mojo](../decomposition/mean_switch.mojo); [decomposition/pca_rr_switch.mojo](../decomposition/pca_rr_switch.mojo); [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [ensemble/decisiontree/batched_levelalgo/quantiles.mojo](../ensemble/decisiontree/batched_levelalgo/quantiles.mojo); [ensemble/host/rf_oracle.mojo](../ensemble/host/rf_oracle.mojo); [ensemble/oob_device.mojo](../ensemble/oob_device.mojo); [ensemble/randomforest.mojo](../ensemble/randomforest.mojo); [ensemble/weighted_bootstrap_device.mojo](../ensemble/weighted_bootstrap_device.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [gaussian_process/estimator.mojo](../gaussian_process/estimator.mojo); [gaussian_process/gp_grad_items.mojo](../gaussian_process/gp_grad_items.mojo); [gaussian_process/gpc_ovr.mojo](../gaussian_process/gpc_ovr.mojo); [gaussian_process/gpr_resident.mojo](../gaussian_process/gpr_resident.mojo); [gbdt/ctrs/ctr_binarization.mojo](../gbdt/ctrs/ctr_binarization.mojo); [gbdt/data/pairs.mojo](../gbdt/data/pairs.mojo); [gbdt/gpu_util/kernel/bootstrap.mojo](../gbdt/gpu_util/kernel/bootstrap.mojo); [gbdt/grid_creator/binarization.mojo](../gbdt/grid_creator/binarization.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../gbdt/methods/doc_parallel_boosting.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/leaves_estimation/leaves_estimation.mojo](../gbdt/methods/leaves_estimation/leaves_estimation.mojo); [gbdt/methods/leaves_estimation/pointwise_oracle.mojo](../gbdt/methods/leaves_estimation/pointwise_oracle.mojo); [gbdt/models/add_non_symmetric_tree_doc_parallel.mojo](../gbdt/models/add_non_symmetric_tree_doc_parallel.mojo); [gbdt/models/kernel/add_bin_values.mojo](../gbdt/models/kernel/add_bin_values.mojo); [gbdt/targets/kernel/yeti_rank.mojo](../gbdt/targets/kernel/yeti_rank.mojo); [gbdt/train.mojo](../gbdt/train.mojo); [glm/estimator.mojo](../glm/estimator.mojo); [glm/impl/qn/glm_base.mojo](../glm/impl/qn/glm_base.mojo); [glm/impl/qn/qn_tiled_rule.mojo](../glm/impl/qn/qn_tiled_rule.mojo); [hdbscan/impl/cluster/detail/sparse_mr_mst.mojo](../hdbscan/impl/cluster/detail/sparse_mr_mst.mojo); [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo); [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo); [hierarchy/impl/cluster/detail/single_linkage.mojo](../hierarchy/impl/cluster/detail/single_linkage.mojo); [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo); [holtwinters/estimator.mojo](../holtwinters/estimator.mojo); [isolation_forest/impl/isolation_forest.mojo](../isolation_forest/impl/isolation_forest.mojo); [ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo); [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo); [kde/impl/neighbors/kernel_density.mojo](../kde/impl/neighbors/kernel_density.mojo); [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo); [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo); [mamba/host/gen/mamba3_siso.mojo](../mamba/host/gen/mamba3_siso.mojo); [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/idn_gemm_ws.mojo](../mamba/impl/modules/idn_gemm_ws.mojo); [mamba/impl/ops/mamba3_siso.mojo](../mamba/impl/ops/mamba3_siso.mojo); [metrics/impl/stats/detail/sf_epilogue_core.mojo](../metrics/impl/stats/detail/sf_epilogue_core.mojo); [metrics/impl/stats/detail/trustworthiness_score.mojo](../metrics/impl/stats/detail/trustworthiness_score.mojo); [mixture/chol_order.mojo](../mixture/chol_order.mojo); [mixture/estimator.mojo](../mixture/estimator.mojo); [mixture/nk_order.mojo](../mixture/nk_order.mojo); [neighbors/impl/knn/knn.mojo](../neighbors/impl/knn/knn.mojo); [neighbors/impl/selection/knn.mojo](../neighbors/impl/selection/knn.mojo); [sequence/exec_device.mojo](../sequence/exec_device.mojo); [sequence/layernorm.mojo](../sequence/layernorm.mojo); [sequence/moe_reg.mojo](../sequence/moe_reg.mojo); [sequence/opt_resident.mojo](../sequence/opt_resident.mojo); [sequence/prophet.mojo](../sequence/prophet.mojo); [solver/impl/cd_gram_rule.mojo](../solver/impl/cd_gram_rule.mojo); [spectral/impl/labels_device.mojo](../spectral/impl/labels_device.mojo); [spectral/impl/preprocessing/detail/fast_graph.mojo](../spectral/impl/preprocessing/detail/fast_graph.mojo); [spectral/impl/sparse/linalg/detail/laplacian.mojo](../spectral/impl/sparse/linalg/detail/laplacian.mojo); [spectral/impl/sparse/linalg/detail/symmetrize.mojo](../spectral/impl/sparse/linalg/detail/symmetrize.mojo); [spectral/impl/sparse/solver/detail/lanczos.mojo](../spectral/impl/sparse/solver/detail/lanczos.mojo); [spectral/spmv_order.mojo](../spectral/spmv_order.mojo); [svm/impl/distance/kernel_matrices.mojo](../svm/impl/distance/kernel_matrices.mojo); [svm/impl/smosolver.mojo](../svm/impl/smosolver.mojo); [training/dev_tensors.mojo](../training/dev_tensors.mojo); [training/estimator.mojo](../training/estimator.mojo); [training/samba_ops.mojo](../training/samba_ops.mojo); [transformer/impl/llama/fused_attention.mojo](../transformer/impl/llama/fused_attention.mojo); [transformer/impl/llama/modeling_llama.mojo](../transformer/impl/llama/modeling_llama.mojo); [tsa/impl/select_d_fast.mojo](../tsa/impl/select_d_fast.mojo); [tsa/impl/timeSeries/stationarity.mojo](../tsa/impl/timeSeries/stationarity.mojo); [umap/optimizer_identical_device.mojo](../umap/optimizer_identical_device.mojo); [x_ann/tsne_core.mojo](../x_ann/tsne_core.mojo); [x_cluster/bisect.mojo](../x_cluster/bisect.mojo); [x_cluster/bodies.mojo](../x_cluster/bodies.mojo); [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo); [x_cluster/device_post.mojo](../x_cluster/device_post.mojo); [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo); [x_cnn/device.mojo](../x_cnn/device.mojo); [x_cnn/ops.mojo](../x_cnn/ops.mojo); [x_decomp/api.mojo](../x_decomp/api.mojo); [x_decomp/device.mojo](../x_decomp/device.mojo); [x_decomp/lu_fast.mojo](../x_decomp/lu_fast.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo); [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo); [x_decomp/tsqr_device.mojo](../x_decomp/tsqr_device.mojo); [x_linear/device.mojo](../x_linear/device.mojo); [x_linear/finite_device.mojo](../x_linear/finite_device.mojo); [x_linear/glm_ydom.mojo](../x_linear/glm_ydom.mojo); [x_metrics/cls_epi.mojo](../x_metrics/cls_epi.mojo); [x_neighbors/items.mojo](../x_neighbors/items.mojo); [x_neighbors/iter_device.mojo](../x_neighbors/iter_device.mojo); [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo); [x_neighbors/ocsvm_init.mojo](../x_neighbors/ocsvm_init.mojo); [x_neighbors/svgp_ff.mojo](../x_neighbors/svgp_ff.mojo); [x_prep/blocked.mojo](../x_prep/blocked.mojo); [x_prep/fam2.mojo](../x_prep/fam2.mojo); [x_prep/fastnb.mojo](../x_prep/fastnb.mojo); [x_prep/fastnb_csr.mojo](../x_prep/fastnb_csr.mojo); [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo); [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo); [x_prep/label_fast.mojo](../x_prep/label_fast.mojo); [x_prep/prep3.mojo](../x_prep/prep3.mojo); [x_prep/pt_blocked.mojo](../x_prep/pt_blocked.mojo); [x_prep/select_blocked.mojo](../x_prep/select_blocked.mojo); [xtrees/agnostic_device.mojo](../xtrees/agnostic_device.mojo); [xtrees/api.mojo](../xtrees/api.mojo); [xtrees/dart_device.mojo](../xtrees/dart_device.mojo); [xtrees/dart_units.mojo](../xtrees/dart_units.mojo); [xtrees/shap_device.mojo](../xtrees/shap_device.mojo) |
| `MOJOLEARN_IDN_AP_CONV_CHUNK16` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_AP_CONV_CHUNK32` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_AP_CONV_CHUNK4` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_AP_DEVICE_CONV_OFF` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_AP_R_TOP2_OFF` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_ARIMA_ASYNC_OFF` | [arima/impl/fast_lbfgs_async.mojo](../arima/impl/fast_lbfgs_async.mojo) |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF` | [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo) |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF` | [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo) |
| `MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF` | [arima/impl/fast_eval_ws.mojo](../arima/impl/fast_eval_ws.mojo) |
| `MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF` | [arima/impl/fast_order_search.mojo](../arima/impl/fast_order_search.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo) |
| `MOJOLEARN_IDN_ARIMA_LLONLY_OFF` | [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo) |
| `MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF` | [arima/impl/fast_order_state.mojo](../arima/impl/fast_order_state.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo) |
| `MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF` | [arima/impl/fast_order_search.mojo](../arima/impl/fast_order_search.mojo) |
| `MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF` | [arima/impl/fast_order_search.mojo](../arima/impl/fast_order_search.mojo); [bindings/_mojolearn_arima_host.mojo](../bindings/_mojolearn_arima_host.mojo) |
| `MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF` | [arima/impl/fast_order_search.mojo](../arima/impl/fast_order_search.mojo) |
| `MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF` | [transformer/impl/llama/fused_attention.mojo](../transformer/impl/llama/fused_attention.mojo) |
| `MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF` | [transformer/impl/llama/fused_attention.mojo](../transformer/impl/llama/fused_attention.mojo) |
| `MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF` | [x_cluster/device_ops.mojo](../x_cluster/device_ops.mojo) |
| `MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF` | [x_cluster/bodies.mojo](../x_cluster/bodies.mojo) |
| `MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF` | [x_cluster/bisect.mojo](../x_cluster/bisect.mojo) |
| `MOJOLEARN_IDN_CACHE_CTX_KEY_OFF` | [core/ctx_key.mojo](../core/ctx_key.mojo) |
| `MOJOLEARN_IDN_CD_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_CE_PIPE_UP_OFF` | [training/estimator.mojo](../training/estimator.mojo) |
| `MOJOLEARN_IDN_CLASS_ONEPASS_OFF` | [x_prep/blocked.mojo](../x_prep/blocked.mojo) |
| `MOJOLEARN_IDN_CLS_EPI_OFF` | [x_metrics/cls_epi.mojo](../x_metrics/cls_epi.mojo); [x_metrics/units.mojo](../x_metrics/units.mojo) |
| `MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_CODE_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_DART_DEVICE_OFF` | [xtrees/dart_device.mojo](../xtrees/dart_device.mojo); [xtrees/dart_host.mojo](../xtrees/dart_host.mojo); [xtrees/dart_units.mojo](../xtrees/dart_units.mojo) |
| `MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF` | [dbscan/impl/runner.mojo](../dbscan/impl/runner.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK32` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK4` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT16` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT4` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT8` | [dbscan/impl/sparse/detail/csr.mojo](../dbscan/impl/sparse/detail/csr.mojo) |
| `MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF` | [dbscan/estimator.mojo](../dbscan/estimator.mojo) |
| `MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF` | [dbscan/estimator.mojo](../dbscan/estimator.mojo); [dbscan/impl/dbscan.mojo](../dbscan/impl/dbscan.mojo) |
| `MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF` | [dbscan/impl/runner.mojo](../dbscan/impl/runner.mojo) |
| `MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF` | [dbscan/impl/runner.mojo](../dbscan/impl/runner.mojo) |
| `MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF` | [dbscan/impl/dbscan.mojo](../dbscan/impl/dbscan.mojo) |
| `MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF` | [decomposition/estimator.mojo](../decomposition/estimator.mojo); [decomposition/mean_switch.mojo](../decomposition/mean_switch.mojo) |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK` | [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo) |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF` | [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo) |
| `MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF` | [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo) |
| `MOJOLEARN_IDN_DENDRO_UNION_OFF` | [hierarchy/impl/cluster/detail/dendrogram_device.mojo](../hierarchy/impl/cluster/detail/dendrogram_device.mojo) |
| `MOJOLEARN_IDN_EIGH_BLOCK_B32` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) |
| `MOJOLEARN_IDN_EIGH_BLOCK_B8` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) |
| `MOJOLEARN_IDN_EIGH_BLOCK_MIN128` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) |
| `MOJOLEARN_IDN_EIGH_BLOCK_OFF` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) |
| `MOJOLEARN_IDN_EIGH_BLOCK_ON` | [x_decomp/rr_block.mojo](../x_decomp/rr_block.mojo) |
| `MOJOLEARN_IDN_EIGH_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_EIGH_SMALL_N128` | [x_decomp/device.mojo](../x_decomp/device.mojo) |
| `MOJOLEARN_IDN_EIGH_SMALL_N64` | [x_decomp/device.mojo](../x_decomp/device.mojo) |
| `MOJOLEARN_IDN_EIGH_SMALL_OFF` | [x_decomp/device.mojo](../x_decomp/device.mojo) |
| `MOJOLEARN_IDN_ET_BINNED_U16` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_ET_PART_ROWS_OFF` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_ET_RESCUE_DEVICE_OFF` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_ET_STAGE_LIVE_OFF` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_ET_TILED_SEARCH_OFF` | [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_GATES_OFF` | [cholesky/estimator.mojo](../cholesky/estimator.mojo); [sequence/moe_reg.mojo](../sequence/moe_reg.mojo); [x_decomp/lu_fast.mojo](../x_decomp/lu_fast.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo); [x_neighbors/iter_device.mojo](../x_neighbors/iter_device.mojo); [x_prep/prep3.mojo](../x_prep/prep3.mojo); [xtrees/shap_device.mojo](../xtrees/shap_device.mojo) |
| `MOJOLEARN_IDN_GBDT_APPLY_WIDE` | [gbdt/models/kernel/add_bin_values.mojo](../gbdt/models/kernel/add_bin_values.mojo) |
| `MOJOLEARN_IDN_GBDT_BOOT_SEEDS_DEVICE_OFF` | [gbdt/gpu_util/kernel/bootstrap.mojo](../gbdt/gpu_util/kernel/bootstrap.mojo) |
| `MOJOLEARN_IDN_GBDT_CTR_BORDERS_DEVICE` | [gbdt/ctrs/ctr_binarization.mojo](../gbdt/ctrs/ctr_binarization.mojo) |
| `MOJOLEARN_IDN_GBDT_CTR_FREQ_DEVICE_OFF` | [gbdt/train.mojo](../gbdt/train.mojo) |
| `MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF` | [gbdt/methods/doc_parallel_boosting.mojo](../gbdt/methods/doc_parallel_boosting.mojo) |
| `MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF` | [gbdt/methods/leaves_estimation/leaves_estimation.mojo](../gbdt/methods/leaves_estimation/leaves_estimation.mojo) |
| `MOJOLEARN_IDN_GBDT_MULTI_HESS_ONE_WAIT_OFF` | [gbdt/methods/leaves_estimation/pointwise_oracle.mojo](../gbdt/methods/leaves_estimation/pointwise_oracle.mojo) |
| `MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF` | [gbdt/models/add_non_symmetric_tree_doc_parallel.mojo](../gbdt/models/add_non_symmetric_tree_doc_parallel.mojo) |
| `MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF` | [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo) |
| `MOJOLEARN_IDN_GBDT_ORDERED_RMSE_DEVICE_GRID_OFF` | [gbdt/grid_creator/binarization.mojo](../gbdt/grid_creator/binarization.mojo) |
| `MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF` | [gbdt/data/pairs.mojo](../gbdt/data/pairs.mojo) |
| `MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` | [gbdt/models/kernel/add_bin_values.mojo](../gbdt/models/kernel/add_bin_values.mojo) |
| `MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF` | [gbdt/targets/kernel/yeti_rank.mojo](../gbdt/targets/kernel/yeti_rank.mojo) |
| `MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF` | [mixture/chol_order.mojo](../mixture/chol_order.mojo) |
| `MOJOLEARN_IDN_GMM_INIT_DEVICE_OFF` | [mixture/estimator.mojo](../mixture/estimator.mojo) |
| `MOJOLEARN_IDN_GMM_NK_LEVELS_OFF` | [mixture/nk_order.mojo](../mixture/nk_order.mojo) |
| `MOJOLEARN_IDN_GMM_ONE_DRAIN_OFF` | [mixture/estimator.mojo](../mixture/estimator.mojo) |
| `MOJOLEARN_IDN_GMM_PROBA_DEVICE_OFF` | [mixture/estimator.mojo](../mixture/estimator.mojo) |
| `MOJOLEARN_IDN_GMM_SCORE_DEVICE_OFF` | [mixture/estimator.mojo](../mixture/estimator.mojo) |
| `MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF` | [gaussian_process/gpc_ovr.mojo](../gaussian_process/gpc_ovr.mojo) |
| `MOJOLEARN_IDN_GPC_OVR_OFF` | [bindings/_mojolearn_gp.mojo](../bindings/_mojolearn_gp.mojo); [gaussian_process/gpc_ovr.mojo](../gaussian_process/gpc_ovr.mojo) |
| `MOJOLEARN_IDN_GPR_PTR_OFF` | [gaussian_process/gpr_resident.mojo](../gaussian_process/gpr_resident.mojo) |
| `MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF` | [gaussian_process/estimator.mojo](../gaussian_process/estimator.mojo) |
| `MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF` | [gaussian_process/gp_grad_items.mojo](../gaussian_process/gp_grad_items.mojo) |
| `MOJOLEARN_IDN_GP_GRAD_ROWS4` | [gaussian_process/gp_grad_items.mojo](../gaussian_process/gp_grad_items.mojo) |
| `MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF` | [gaussian_process/estimator.mojo](../gaussian_process/estimator.mojo) |
| `MOJOLEARN_IDN_GRAM_BLOCKED_OFF` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) |
| `MOJOLEARN_IDN_GRAM_ROWS_512` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) |
| `MOJOLEARN_IDN_GRAM_ROWS_8192` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) |
| `MOJOLEARN_IDN_GRAM_ROWTILE` | [x_prep/gram_blocked.mojo](../x_prep/gram_blocked.mojo) |
| `MOJOLEARN_IDN_GRAPH_M_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_GROUP_M_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_ONE_SYNC_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF` | [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` | [hdbscan/impl/cluster/detail/single_linkage.mojo](../hdbscan/impl/cluster/detail/single_linkage.mojo); [hdbscan/impl/detail/idn_switches.mojo](../hdbscan/impl/detail/idn_switches.mojo) |
| `MOJOLEARN_IDN_HPDEV_` | [bindings/_mojolearn.mojo](../bindings/_mojolearn.mojo) |
| `MOJOLEARN_IDN_HPDEV_CAST_F64` | [bindings/_mojolearn.mojo](../bindings/_mojolearn.mojo); [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HPDEV_ELEM_OFF` | [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HPDEV_FOLDS_OFF` | [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HPDEV_INIT_OFF` | [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HPDEV_LABELS_OFF` | [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HPDEV_REDUCE_OFF` | [bindings/hotpath_device.mojo](../bindings/hotpath_device.mojo) |
| `MOJOLEARN_IDN_HW_FIT_DIRECT_OFF` | [holtwinters/estimator.mojo](../holtwinters/estimator.mojo) |
| `MOJOLEARN_IDN_HW_FORECAST_DIRECT_OFF` | [holtwinters/estimator.mojo](../holtwinters/estimator.mojo) |
| `MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF` | [isolation_forest/impl/isolation_forest.mojo](../isolation_forest/impl/isolation_forest.mojo) |
| `MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF` | [isolation_forest/impl/isolation_forest.mojo](../isolation_forest/impl/isolation_forest.mojo) |
| `MOJOLEARN_IDN_IF_RESIDENT_OFF` | [isolation_forest/impl/isolation_forest.mojo](../isolation_forest/impl/isolation_forest.mojo) |
| `MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF` | [ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo) |
| `MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF` | [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo) |
| `MOJOLEARN_IDN_KALMAN_RD_OFF` | [arima/impl/batched_kalman.mojo](../arima/impl/batched_kalman.mojo) |
| `MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF` | [kde/impl/neighbors/kernel_density.mojo](../kde/impl/neighbors/kernel_density.mojo) |
| `MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_16` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_2` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_32` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_4` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_DEVICE_CONV_OFF` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_INCR_INIT_OFF` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KMEANS_INIT_PSI_DEVICE_OFF` | [cluster/impl/detail/kmeans.mojo](../cluster/impl/detail/kmeans.mojo) |
| `MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_KM_PTR_IN_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_KM_RBF_CELL` | [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo) |
| `MOJOLEARN_IDN_KM_RBF_CELL_D16` | [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo) |
| `MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF` | [neighbors/impl/selection/knn.mojo](../neighbors/impl/selection/knn.mojo) |
| `MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF` | [neighbors/impl/selection/knn.mojo](../neighbors/impl/selection/knn.mojo) |
| `MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF` | [neighbors/impl/knn/knn.mojo](../neighbors/impl/knn/knn.mojo) |
| `MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF` | [tsa/estimator.mojo](../tsa/estimator.mojo); [tsa/impl/select_d_fast.mojo](../tsa/impl/select_d_fast.mojo) |
| `MOJOLEARN_IDN_KPSS_SCAN_OFF` | [tsa/impl/timeSeries/stationarity.mojo](../tsa/impl/timeSeries/stationarity.mojo) |
| `MOJOLEARN_IDN_KRR_DEV_SCALE_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_KRR_PTR_IN_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_LABEL_INV_OFF` | [x_prep/fam2.mojo](../x_prep/fam2.mojo) |
| `MOJOLEARN_IDN_LABEL_OFF` | [x_prep/label_fast.mojo](../x_prep/label_fast.mojo) |
| `MOJOLEARN_IDN_LANCZOS_POOL_OFF` | [spectral/impl/sparse/solver/detail/lanczos.mojo](../spectral/impl/sparse/solver/detail/lanczos.mojo) |
| `MOJOLEARN_IDN_LAYER_DEV_IO_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF` | [transformer/impl/llama/modeling_llama.mojo](../transformer/impl/llama/modeling_llama.mojo) |
| `MOJOLEARN_IDN_LLE_DEV_F0_OFF` | [x_decomp/api.mojo](../x_decomp/api.mojo) |
| `MOJOLEARN_IDN_LP_RESIDENT_OFF` | [x_neighbors/iter_device.mojo](../x_neighbors/iter_device.mojo) |
| `MOJOLEARN_IDN_LU_GESV_OFF` | [x_decomp/api.mojo](../x_decomp/api.mojo) |
| `MOJOLEARN_IDN_LU_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128` | [mamba/host/gen/mamba3_siso.mojo](../mamba/host/gen/mamba3_siso.mojo); [mamba/impl/ops/mamba3_siso.mojo](../mamba/impl/ops/mamba3_siso.mojo) |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32` | [mamba/host/gen/mamba3_siso.mojo](../mamba/host/gen/mamba3_siso.mojo); [mamba/impl/ops/mamba3_siso.mojo](../mamba/impl/ops/mamba3_siso.mojo) |
| `MOJOLEARN_IDN_M3_ANGLE_PARALLEL` | [mamba/host/gen/mamba3_siso.mojo](../mamba/host/gen/mamba3_siso.mojo); [mamba/impl/ops/mamba3_siso.mojo](../mamba/impl/ops/mamba3_siso.mojo) |
| `MOJOLEARN_IDN_M3_SESSION_STAGE_REUSE_OFF` | [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo) |
| `MOJOLEARN_IDN_MAMBA3_REPORTS_ON_REQUEST_OFF` | [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo) |
| `MOJOLEARN_IDN_MAMBA_ALLOC_NOWAIT_OFF` | [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo) |
| `MOJOLEARN_IDN_MAMBA_ARENA_OFF` | [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo) |
| `MOJOLEARN_IDN_MAMBA_DEVICE_REFUSAL_OFF` | [mamba/impl/modules/afn_defines.mojo](../mamba/impl/modules/afn_defines.mojo) |
| `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF` | [mamba/impl/modules/idn_gemm_ws.mojo](../mamba/impl/modules/idn_gemm_ws.mojo) |
| `MOJOLEARN_IDN_MAXIMIZE_DEV_OFF` | [training/estimator.mojo](../training/estimator.mojo) |
| `MOJOLEARN_IDN_METRIC_EPI_OFF` | [metrics/impl/stats/detail/sf_epilogue_core.mojo](../metrics/impl/stats/detail/sf_epilogue_core.mojo) |
| `MOJOLEARN_IDN_MINIBATCH_GROUP32` | [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo) |
| `MOJOLEARN_IDN_MINIBATCH_GROUP64` | [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo) |
| `MOJOLEARN_IDN_MINIBATCH_GROUP8` | [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo) |
| `MOJOLEARN_IDN_MINIBATCH_GROUP_OFF` | [x_cluster/minibatch.mojo](../x_cluster/minibatch.mojo) |
| `MOJOLEARN_IDN_MST_LABEL_JUMP_OFF` | [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo) |
| `MOJOLEARN_IDN_MST_PAR_COMPACT_OFF` | [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo) |
| `MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF` | [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo) |
| `MOJOLEARN_IDN_MST_SCAN_LANES` | [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo) |
| `MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF` | [x_prep/fastnb.mojo](../x_prep/fastnb.mojo) |
| `MOJOLEARN_IDN_NB_CSR_DENSE_OFF` | [x_prep/blocked.mojo](../x_prep/blocked.mojo) |
| `MOJOLEARN_IDN_NB_CSR_OFF` | [bindings/_mojolearn_x_prep.mojo](../bindings/_mojolearn_x_prep.mojo); [x_prep/fastnb_csr.mojo](../x_prep/fastnb_csr.mojo) |
| `MOJOLEARN_IDN_NB_ONEPASS_OFF` | [x_prep/blocked.mojo](../x_prep/blocked.mojo) |
| `MOJOLEARN_IDN_NC_CHUNKED_OFF` | [x_neighbors/items.mojo](../x_neighbors/items.mojo) |
| `MOJOLEARN_IDN_NYS_DEV_BASIS_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_NYS_DEV_ORDER_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo) |
| `MOJOLEARN_IDN_NYS_RR_EIGH_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo); [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo) |
| `MOJOLEARN_IDN_OCSVM_2L` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) |
| `MOJOLEARN_IDN_OCSVM_2L_OFF` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) |
| `MOJOLEARN_IDN_OCSVM_CHUNK256` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) |
| `MOJOLEARN_IDN_OCSVM_DEV_INIT_OFF` | [x_neighbors/ocsvm_init.mojo](../x_neighbors/ocsvm_init.mojo) |
| `MOJOLEARN_IDN_OCSVM_RES` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) |
| `MOJOLEARN_IDN_OCSVM_RES_OFF` | [x_neighbors/ocsvm_dev.mojo](../x_neighbors/ocsvm_dev.mojo) |
| `MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF` | [x_decomp/api.mojo](../x_decomp/api.mojo) |
| `MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF` | [x_cluster/device_post.mojo](../x_cluster/device_post.mojo) |
| `MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF` | [training/estimator.mojo](../training/estimator.mojo) |
| `MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF` | [sequence/opt_resident.mojo](../sequence/opt_resident.mojo) |
| `MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF` | [training/estimator.mojo](../training/estimator.mojo) |
| `MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF` | [sequence/opt_resident.mojo](../sequence/opt_resident.mojo) |
| `MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_PAD_M_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_PARTIAL_CODES_OFF` | [x_prep/fam2.mojo](../x_prep/fam2.mojo) |
| `MOJOLEARN_IDN_PERM_DRAW_OFF` | [x_prep/fam2.mojo](../x_prep/fam2.mojo) |
| `MOJOLEARN_IDN_PIN_WEIGHTS_OFF` | [x_cnn/device.mojo](../x_cnn/device.mojo) |
| `MOJOLEARN_IDN_PROPHET_COOP_OFF` | [sequence/prophet.mojo](../sequence/prophet.mojo); [sequence/prophet_coop.mojo](../sequence/prophet_coop.mojo) |
| `MOJOLEARN_IDN_PT_BLOCKED_OFF` | [x_prep/pt_blocked.mojo](../x_prep/pt_blocked.mojo) |
| `MOJOLEARN_IDN_QR_R_DIRECT_OFF` | [x_decomp/device.mojo](../x_decomp/device.mojo) |
| `MOJOLEARN_IDN_QR_R_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_RAND_RESIDENT_OFF` | [x_decomp/api.mojo](../x_decomp/api.mojo) |
| `MOJOLEARN_IDN_RBF_FUSED_OFF` | [kernel_methods/estimator.mojo](../kernel_methods/estimator.mojo); [kernel_methods/host/km_host_oracle.mojo](../kernel_methods/host/km_host_oracle.mojo) |
| `MOJOLEARN_IDN_RES_MOVES_OFF` | [x_decomp/api.mojo](../x_decomp/api.mojo) |
| `MOJOLEARN_IDN_RF_COLS40_OFF` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/level_loop_kernels.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/level_loop_kernels.mojo) |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP_K1` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP_K2` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP_K8` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF` | [ensemble/decisiontree/batched_levelalgo/builder.mojo](../ensemble/decisiontree/batched_levelalgo/builder.mojo) |
| `MOJOLEARN_IDN_RF_HIST_COLUMNS4_AMD_OFF` | [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo) |
| `MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF` | [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo) |
| `MOJOLEARN_IDN_RF_HIST_ZERO_OFF` | [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo) |
| `MOJOLEARN_IDN_RF_OOB_DEVICE_OFF` | [ensemble/oob_device.mojo](../ensemble/oob_device.mojo) |
| `MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF` | [ensemble/decisiontree/batched_levelalgo/quantiles.mojo](../ensemble/decisiontree/batched_levelalgo/quantiles.mojo); [ensemble/host/rf_oracle.mojo](../ensemble/host/rf_oracle.mojo) |
| `MOJOLEARN_IDN_RF_ROWS_SORTED` | [ensemble/randomforest.mojo](../ensemble/randomforest.mojo) |
| `MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF` | [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo) |
| `MOJOLEARN_IDN_RF_WEIGHTED_BOOTSTRAP_DEVICE_OFF` | [ensemble/host/rf_oracle.mojo](../ensemble/host/rf_oracle.mojo); [ensemble/weighted_bootstrap_device.mojo](../ensemble/weighted_bootstrap_device.mojo) |
| `MOJOLEARN_IDN_RF_WEIGHT_ROWS_DEVICE_OFF` | [ensemble/randomforest.mojo](../ensemble/randomforest.mojo) |
| `MOJOLEARN_IDN_RR_EIGH_OFF` | [x_prep/device.mojo](../x_prep/device.mojo); [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) |
| `MOJOLEARN_IDN_RR_MIN_16` | [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) |
| `MOJOLEARN_IDN_RR_MIN_48` | [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) |
| `MOJOLEARN_IDN_RR_MIN_8` | [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) |
| `MOJOLEARN_IDN_RR_MIN_96` | [x_prep/host/rr_eigh_host.mojo](../x_prep/host/rr_eigh_host.mojo) |
| `MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF` | [training/samba_ops.mojo](../training/samba_ops.mojo) |
| `MOJOLEARN_IDN_SELECT_BLOCKED_OFF` | [x_prep/select_blocked.mojo](../x_prep/select_blocked.mojo) |
| `MOJOLEARN_IDN_SELECT_D_OFF` | [tsa/impl/auto_arima.mojo](../tsa/impl/auto_arima.mojo); [tsa/impl/select_d_fast.mojo](../tsa/impl/select_d_fast.mojo) |
| `MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF` | [sequence/exec_device.mojo](../sequence/exec_device.mojo) |
| `MOJOLEARN_IDN_SEQ_PIPE_UP_OFF` | [sequence/exec_device.mojo](../sequence/exec_device.mojo) |
| `MOJOLEARN_IDN_SHAP_DEVICE_MODEL` | [xtrees/agnostic.mojo](../xtrees/agnostic.mojo); [xtrees/agnostic_device.mojo](../xtrees/agnostic_device.mojo); [xtrees/api.mojo](../xtrees/api.mojo) |
| `MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF` | [xtrees/agnostic_device.mojo](../xtrees/agnostic_device.mojo); [xtrees/api.mojo](../xtrees/api.mojo) |
| `MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS` | [hierarchy/impl/cluster/detail/single_linkage.mojo](../hierarchy/impl/cluster/detail/single_linkage.mojo) |
| `MOJOLEARN_IDN_SL_SPARSE_MST` | [hierarchy/impl/cluster/detail/single_linkage.mojo](../hierarchy/impl/cluster/detail/single_linkage.mojo) |
| `MOJOLEARN_IDN_SMR_RADIX_RANK_OFF` | [hdbscan/impl/cluster/detail/sparse_mr_mst.mojo](../hdbscan/impl/cluster/detail/sparse_mr_mst.mojo) |
| `MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF` | [spectral/impl/preprocessing/detail/fast_graph.mojo](../spectral/impl/preprocessing/detail/fast_graph.mojo) |
| `MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF` | [spectral/impl/labels_device.mojo](../spectral/impl/labels_device.mojo) |
| `MOJOLEARN_IDN_SPECTRAL_LAP_DEVICE_OFF` | [spectral/impl/sparse/linalg/detail/laplacian.mojo](../spectral/impl/sparse/linalg/detail/laplacian.mojo) |
| `MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF` | [spectral/impl/sparse/solver/detail/lanczos.mojo](../spectral/impl/sparse/solver/detail/lanczos.mojo) |
| `MOJOLEARN_IDN_SPMV_LANES_OFF` | [spectral/spmv_order.mojo](../spectral/spmv_order.mojo) |
| `MOJOLEARN_IDN_STATS_BLOCKED_OFF` | [x_prep/blocked.mojo](../x_prep/blocked.mojo) |
| `MOJOLEARN_IDN_SVD_RESIDENT_OFF` | [bindings/_mojolearn_x_decomp.mojo](../bindings/_mojolearn_x_decomp.mojo); [x_decomp/resident.mojo](../x_decomp/resident.mojo) |
| `MOJOLEARN_IDN_SVGP_DOT2_OFF` | [x_neighbors/svgp_ff.mojo](../x_neighbors/svgp_ff.mojo) |
| `MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF` | [spectral/impl/sparse/linalg/detail/symmetrize.mojo](../spectral/impl/sparse/linalg/detail/symmetrize.mojo) |
| `MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF` | [training/dev_tensors.mojo](../training/dev_tensors.mojo) |
| `MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF` | [xtrees/api.mojo](../xtrees/api.mojo) |
| `MOJOLEARN_IDN_TRUST_DEV_OFF` | [metrics/impl/stats/detail/trustworthiness_score.mojo](../metrics/impl/stats/detail/trustworthiness_score.mojo) |
| `MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF` | [x_ann/tsne_core.mojo](../x_ann/tsne_core.mojo) |
| `MOJOLEARN_IDN_TSQR_GRID_OFF` | [x_decomp/tsqr_device.mojo](../x_decomp/tsqr_device.mojo) |
| `MOJOLEARN_IDN_TSQR_NORM_OFF` | [x_decomp/tsqr_device.mojo](../x_decomp/tsqr_device.mojo) |
| `MOJOLEARN_IDN_UMAP_DEVICE_CSR_OFF` | [umap/optimizer_identical_device.mojo](../umap/optimizer_identical_device.mojo) |
| `MOJOLEARN_IDN_WDRAW_OFF` | [x_prep/fam2.mojo](../x_prep/fam2.mojo) |
| `MOJOLEARN_IDN_WPICK_OFF` | [x_prep/fam2.mojo](../x_prep/fam2.mojo) |
| `MOJOLEARN_IDN_X0_PAR_LS` | [arima/impl/idn_arma_ls.mojo](../arima/impl/idn_arma_ls.mojo); [arima/impl/idn_ls_math.mojo](../arima/impl/idn_ls_math.mojo) |
| `MOJOLEARN_IDN_X0_PAR_LS_MIN256` | [arima/impl/idn_ls_math.mojo](../arima/impl/idn_ls_math.mojo) |
| `MOJOLEARN_IDN_XD_SWEEP_OFF` | [x_decomp/device.mojo](../x_decomp/device.mojo) |
| `MOJOLEARN_IDN_XENT_DEV_FOLD_OFF` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_XENT_FOLD_BLOCK_256` | [x_cnn/ops.mojo](../x_cnn/ops.mojo) |
| `MOJOLEARN_IDN_XN_KNN_LEAN_OFF` | [x_neighbors/iter_device.mojo](../x_neighbors/iter_device.mojo) |
| `MOJOLEARN_IDN_XN_KNN_PRUNE_OFF` | [x_neighbors/iter_device.mojo](../x_neighbors/iter_device.mojo) |
| `MOJOLEARN_IDN_XN_UNIT_DEV_OFF` | [x_neighbors/ocsvm_init.mojo](../x_neighbors/ocsvm_init.mojo) |
| `MOJOLEARN_IDN_XTY_TILED_OFF` | [core/host_tile_fold.mojo](../core/host_tile_fold.mojo); [core/xtdz_coalesced.mojo](../core/xtdz_coalesced.mojo) |
| `MOJOLEARN_IDN_XTY_TILE_1024` | [core/host_tile_fold.mojo](../core/host_tile_fold.mojo) |
| `MOJOLEARN_IDN_XTY_TILE_64` | [core/host_tile_fold.mojo](../core/host_tile_fold.mojo) |

## Other existing A B drivers and records

These files expose older comparisons, controls or orchestration. They may reuse the experiments above. Some are quality-only, reduced fixtures, negative controls or source-migration tools; their presence does not establish a full-dataset performance experiment.

| Experiment family or purpose | Files | Scope |
| --- | --- | --- |
| Classical pass comparisons for LU SGD LARS and IVF | [tools/classical_pass_ab.py](../tools/classical_pass_ab.py) | Several old arms were removed; its full-size mode is not a complete old/new A/B for every case. |
| Classical GEMM split callers | [tools/gemm_ksplit_classical_ab.py](../tools/gemm_ksplit_classical_ab.py) | Caller comparisons for classical estimators. |
| Forest training and work counts | [tools/forest_train_ab.py](../tools/forest_train_ab.py); [tools/forest_experiment_board.py](../tools/forest_experiment_board.py) | Forest arm timing and experiment record tooling. |
| Boosting partition cache | [tools/gbdt_partition_cache_ab.py](../tools/gbdt_partition_cache_ab.py) | Full-model output and work-count comparison. |
| Boosting fused paths accuracy and grow policy | [tools/gbdt_fused_ab.sh](../tools/gbdt_fused_ab.sh); [tools/gbdt_accuracy_ab.sh](../tools/gbdt_accuracy_ab.sh); [tools/grow_policy_ab.sh](../tools/grow_policy_ab.sh) | Existing shell A/B launchers. |
| Apple tree and IDENTICAL tree lanes | [tools/trees_apple_ab.sh](../tools/trees_apple_ab.sh); [tools/trees_identical_ab.sh](../tools/trees_identical_ab.sh); [tools/trees_apple3/laptop_ab.sh](../tools/trees_apple3/laptop_ab.sh); [tools/aft_ab.sh](../tools/aft_ab.sh); [tools/criteo_ours_cat_ab.py](../tools/criteo_ours_cat_ab.py) | Tree campaign and dataset-specific orchestration; historical recipes require current policy. |
| Forest repeatability and mutex comparisons | [tools/rf_nondeterminism/make_rf_ab_body.sh](../tools/rf_nondeterminism/make_rf_ab_body.sh); [tools/rf_nondeterminism/rf_ab_body.template.sh](../tools/rf_nondeterminism/rf_ab_body.template.sh); [tools/mutex_et_knn/et_ab.py](../tools/mutex_et_knn/et_ab.py); [tools/device_mutex_ab_leg.sh](../tools/device_mutex_ab_leg.sh) | Repeatability or coordination comparisons, not new AFCL candidates. |
| Apple ANN and geometry campaign | [tools/ann_apple2_ab.sh](../tools/ann_apple2_ab.sh); [tools/ann_apple3_tab.py](../tools/ann_apple3_tab.py) | ANN A/B orchestration and retained tables. |
| Apple linear model campaign | [tools/linear_apple3/ab.py](../tools/linear_apple3/ab.py); [tools/linear_apple3/tab.py](../tools/linear_apple3/tab.py) | Job-driven linear comparisons and tables. |
| Kernel PCA quality | [tools/kpca_quality_ab.py](../tools/kpca_quality_ab.py); [tools/kpca_quality_ab.sh](../tools/kpca_quality_ab.sh) | Quality-only A/B surface. |
| MCD compatibility | [tools/mcd_compat_ab.sh](../tools/mcd_compat_ab.sh) | Compatibility and repair comparison; related history in the Apple ledger. |
| ARIMA quality and order batching | [tools/arima_fast_quality_ab.sh](../tools/arima_fast_quality_ab.sh); [docs/apple-fast/ab/arima-orders-evidence/raw-ab.txt](../docs/apple-fast/ab/arima-orders-evidence/raw-ab.txt); [docs/apple-fast/ab/gap26/ab-notes.txt](../docs/apple-fast/ab/gap26/ab-notes.txt); [docs/apple-fast/ab/gap26/ab-tags.txt](../docs/apple-fast/ab/gap26/ab-tags.txt) | Retained forecasting comparison definitions and evidence references. |
| Label direct route | [docs/apple-fast/ab/label-direct.md](../docs/apple-fast/ab/label-direct.md); [tools/label_fast_quality.py](../tools/label_fast_quality.py) | Existing route and quality experiment record. |
| Python to native preprocessing migration | [tools/py_misc_prep/ab.py](../tools/py_misc_prep/ab.py); [tools/py_consolidated/ab_arms.py](../tools/py_consolidated/ab_arms.py); [tools/py_shared/ab_job.sh](../tools/py_shared/ab_job.sh); [tools/py_shared/ab_diff.py](../tools/py_shared/ab_diff.py) | Historical route comparisons; does not authorize Python numerical runtime. |
| General Apple FAST arm tools | [tools/afc_ab.sh](../tools/afc_ab.sh); [tools/afc_ab_def.sh](../tools/afc_ab_def.sh); [tools/apple_fast_ops/m2_build_ab.py](../tools/apple_fast_ops/m2_build_ab.py); [tools/apple_fast_ops/ab_extract.py](../tools/apple_fast_ops/ab_extract.py) | Shared arm build or extraction helpers, not separate optimization ideas. |
| FAST replication comparisons | [tools/fast_replication_ab.sh](../tools/fast_replication_ab.sh); [tools/fast_replication_ab_all.sh](../tools/fast_replication_ab_all.sh) | Existing replication campaign drivers. |
| Shared full-workload queue and measurement records | [tools/performance_full_ab_queue.py](../tools/performance_full_ab_queue.py); [tools/full_ab_watch.py](../tools/full_ab_watch.py); [tools/cell_ab_job.sh](../tools/cell_ab_job.sh); [experiments/performance_ideas/measurements/full_ab_20261006/inventory.json](../experiments/performance_ideas/measurements/full_ab_20261006/inventory.json); [experiments/performance_ideas/measurements/20261006/inventory.json](../experiments/performance_ideas/measurements/20261006/inventory.json) | Orchestration and historical inventories; no jobs scheduled by this document. |
| IDENTICAL callpath experiment source | [experiments/identical_callpath/README.md](../experiments/identical_callpath/README.md); [experiments/identical_callpath/COVERAGE.md](../experiments/identical_callpath/COVERAGE.md) | Callpath and identity experiment context. |
| CPU worker resource comparisons | [tools/host_threads_ab_check.py](../tools/host_threads_ab_check.py); [docs/identical/cpu-race-resource-audit-20261005.json](../docs/identical/cpu-race-resource-audit-20261005.json) | Historical resource-policy evidence; not a new kernel speed card. |
| Existing neural runtime environment experiments | [tools/neural_experiments.py](../tools/neural_experiments.py) | Outside new classical scope; named runtime arms remain in this source file. |
| Existing neural Apple and sequence comparisons | [tools/afn_ab.py](../tools/afn_ab.py); [tools/afn_ab.sh](../tools/afn_ab.sh); [tools/sequence_apple_ab.sh](../tools/sequence_apple_ab.sh); [tools/apple_speed_neural/ab.sh](../tools/apple_speed_neural/ab.sh); [tools/apple_speed_cnn/ab.sh](../tools/apple_speed_cnn/ab.sh) | Historical neural scope only. |
| Existing language model pool and shard arms | [tools/byte_lm_pool_ab_matrix.sh](../tools/byte_lm_pool_ab_matrix.sh); [tools/byte_lm_pool_ab_compare.py](../tools/byte_lm_pool_ab_compare.py); [tools/lm_shards_ab_matrix.sh](../tools/lm_shards_ab_matrix.sh); [tools/lm_shards_ab_compare.py](../tools/lm_shards_ab_compare.py); [tools/transformer_ab_input_sha.py](../tools/transformer_ab_input_sha.py) | Historical neural scope only; matrix, comparison and input evidence helpers. |
| Multi GPU negative controls | [docs/multi_gpu/PAR_SABOTAGE_ARMS.md](../docs/multi_gpu/PAR_SABOTAGE_ARMS.md); [tools/embedding_sabotage_arm.sh](../tools/embedding_sabotage_arm.sh); [tools/par_harness/sabotage_par_read_shift.patch](../tools/par_harness/sabotage_par_read_shift.patch) | Fault-injection control records, not performance candidates. |

## Keeping this index current

Update the relevant canonical card or lane file first, then update its index entry. Keep original experiment IDs and historical branches so rejected or superseded ideas remain findable. New full-workload results belong in retained evidence and board-tool inputs; documentation edits must not manufacture results, carry acceptance across source versions or silently enable a switch.
