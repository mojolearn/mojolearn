# Apple FAST classical ML A/B ideas — 2026-10-06

See the [consolidated A/B experiment index](../AB_EXPERIMENT_INDEX.md) for every
new card's implementation files, controls, callers and registration, together
with the existing shared catalog and historical experiment records.

This is a source-only experiment branch from `fd6cf8045`. The owner explicitly
requested a new worktree, a thorough idea inventory before delegation, implementation,
commit and push, and **no compilation, verification or measurement**. Those commands
must not be run in this task, including through Git hooks or remote jobs. No speed or
quality result is claimed. New candidates default OFF and are Apple FAST only.

Different floating-point bits from the previous version are allowed. Quality is not
negotiable: preserve estimator semantics, input/refusal rules, trained-state lifetime,
randomness contracts, convergence settings and task quality. No reduced datasets,
fewer trees, fewer starts, looser tolerances, fewer neighbors/probes/epochs, lowered
precision, host numerical detours or Python runtime work are proposed. Any shape
rule must follow a documented memory/occupancy/cost argument covering neighboring
shapes. Existing production defaults and IDENTICAL behavior remain the A arm.

## Coverage and prioritization

The 54 independently selectable cards cover linear and generalized linear models,
SVM, decomposition, NMF/ALS/PLS, Gaussian processes and kernel features; exact and
approximate neighbors, clustering, mixture models and manifold learning; boosted,
randomized and isolation trees plus explainers; preprocessing, naive Bayes, feature
selection, metrics, resampling and classical statistical forecasting. Neural models,
MLP, CNN, attention, transformers, Mamba, neural optimizers and tokenizers are excluded.
Shared kernels are included only through classical caller-scoped controls.

Start future qualification with low numerical-risk launch/layout changes (P02–P07,
P10, T03/T04/T07–T12), then reduction/product schedules (L01–L09, G01–G09), then
iterative/ill-conditioned callers (L10–L12, G10–G12, P11/P12). This is an engineering
priority, not a measured ranking. The implementation may use the closest supported
schedule in the named call path; each lane records the exact mechanism, actual reach,
prerequisite defines and limitations. A manifest alone is not an implementation.

Existing F01–F20 work under `experiments/performance_ideas/` is prior art. These cards
extend specific schedules or arithmetic implementations; they must not relabel an
unchanged earlier opt-in as a new optimization. A/B comparisons of existing unqualified
paths need their prerequisite enabled in BOTH arms so only the new mechanism differs.

## A/B and future quality contract (not executed now)

For each card, A is the frozen existing implementation and B enables only the new
card. The lane manifest records exact defines, reachable callers and dependencies.
Existing benchmark-fitted rule removals, if encountered, require old-rule B and
new-rule A per AGENTS.md, recorded explicitly. Do not silently change such a rule.

Before any later measurement, map all affected callers to the saved full workloads
in `tools/classical_two_datasets.py`, `tools/bench_board_more.py`,
`tools/bench_board_algos.py`, `tools/knn_datasets.py`, and the tree recipes in
`tools/bench_board.py` / `bench/speed/forest_speed_arm.py`. Record the dataset version,
hash, split, actual dimensions, full workload settings, mode, source/binary/compiler,
hardware and exact toggles. Existing intrinsic row caps remain unresolved until the
full intended workload is explicit; a `--rows full` flag alone proves nothing.

Future timing includes preparation, fitting, synchronization and consumed outputs;
report inference, cold use and repeated use separately. Use one excluded warmup and
one scored sample per arm under the repository policy, retain failed attempts and
report uncertainty. For every affected estimator use its full data and existing task
quality policy, including the admitted opponent quality floor where applicable.
Compare quality to A with independent metrics, plus neighboring shapes and a non-board
workload; output bit equality is not the Apple FAST quality gate. Keep nonfinite,
empty/constant, ragged-tail, highly skewed, ill-conditioned and concurrent-live-model
cases in future correctness coverage where relevant. No claims can be admitted now.

Interactions need their own future A/B: Gram+linear folds; product+QR/PCA/NMF;
neighbor distances+top-k; assignment+center reduction; histogram+partition+leaf;
binarization+CTR; scaler/selector+downstream fit; metric reductions+score consumption;
ARIMA differencing+likelihood. Measure the complete proposed configuration, not only
isolated winners. Keep neutral, rejected, failed and unsupported arms visible and OFF.
Use board tools for future board updates; this source-only branch changes no boards.

## Experiment cards

### AFCL-L01 — Linear Gram row partition

- Callers: Ridge, RidgeClassifier, RidgeCV, LDA/QDA. Source area: `x_linear/fast_gram.mojo`.
- A: Current row chunks.
- B: Smaller row chunks in the existing partial-Gram grid.
- Why it might help: More independent threadgroups for narrow Grams, at the cost of scratch and final-fold work.
- Quality and edge cases: Coefficients, heldout R2/logloss, rank, class covariance and ill-conditioned inputs.
- Intended control: `MOJOLEARN_AFCL_L01`; OFF until later qualification.

### AFCL-L02 — Linear sufficient-statistic reduction

- Callers: Ridge and multi-output linear models. Source area: `x_linear/fast_gram.mojo`.
- A: Serial chunk-partial fold.
- B: Striped or pairwise chunk-partial reduction.
- Why it might help: Reduce dependent accumulation latency while retaining centered statistics.
- Quality and edge cases: Large offsets, nearly constant columns, sample weights and multiple targets.
- Intended control: `MOJOLEARN_AFCL_L02`; OFF until later qualification.

### AFCL-L03 — ElasticNetCV device work granularity

- Callers: LassoCV and ElasticNetCV. Source area: `x_linear/enetcv_fast.mojo`.
- A: Current residual/prediction schedule.
- B: Alternative row work grouping for the same alpha/fold path.
- Why it might help: Improve occupancy without changing regularization search or stopping tolerance.
- Quality and edge cases: Selected alpha, dual gap, coefficients, heldout RMSE and fold isolation.
- Intended control: `MOJOLEARN_AFCL_L03`; OFF until later qualification.

### AFCL-L04 — Huber gradient/loss work grouping

- Callers: HuberRegressor. Source area: `x_linear/huber_fast.mojo`.
- A: Current gradient and loss row chunks.
- B: Smaller bounded partial chunks or fused existing output work.
- Why it might help: Shorten long serial chains and reduce repeated row traffic.
- Quality and edge cases: Robust objective, outlier behavior, convergence and heldout RMSE.
- Intended control: `MOJOLEARN_AFCL_L04`; OFF until later qualification.

### AFCL-L05 — Multiclass logistic gradient schedule

- Callers: LogisticRegression and linear classification. Source area: `glm/impl/qn/fast_xtdz.mojo`.
- A: Current X-transpose times residual geometry.
- B: Alternative row/chunk geometry for the same FP32 product.
- Why it might help: Balance output parallelism against repeated residual reads.
- Quality and edge cases: Heldout logloss and accuracy, optimizer status, extreme logits and class imbalance.
- Intended control: `MOJOLEARN_AFCL_L05`; OFF until later qualification.

### AFCL-L06 — SMO gradient row reuse

- Callers: SVC, SVR and OneClassSVM where caller applies. Source area: `svm/impl/fast_update_f.mojo`.
- A: Current fused row update.
- B: Reuse feature loads or independent row work within the fused update.
- Why it might help: Reduce repeated kernel-vector memory work without reducing working set or iteration budget.
- Quality and edge cases: Margin, support-vector state, regression residual, anomaly ranking and refusal semantics.
- Intended control: `MOJOLEARN_AFCL_L06`; OFF until later qualification.

### AFCL-L07 — PCA covariance work granularity

- Callers: PCA and covariance consumers. Source area: `decomposition/`.
- A: Current covariance partial schedule.
- B: Alternative bounded centered-covariance chunks.
- Why it might help: Increase parallelism while retaining the numerically safer centered formulation.
- Quality and edge cases: Explained variance, reconstruction, orthogonality and small singular directions.
- Intended control: `MOJOLEARN_AFCL_L07`; OFF until later qualification.

### AFCL-L08 — Decomposition tiled product geometry

- Callers: NMF, FactorAnalysis, PLS, ALS, SVD and least squares callers. Source area: `x_decomp/fast_gemm.mojo`.
- A: Existing tiled product geometry.
- B: Alternative slab depth or microtile on the same product route.
- Why it might help: Trade shared-memory reuse against occupancy; preserve transpose and tail handling.
- Quality and edge cases: Factor/solve residuals plus every affected estimator task metric; no standalone GEMM admission.
- Intended control: `MOJOLEARN_AFCL_L08`; OFF until later qualification.

### AFCL-L09 — TSQR panel reduction schedule

- Callers: QR, randomized SVD, least squares. Source area: `x_decomp/fast_qr.mojo`.
- A: Current norm/update launch geometry.
- B: Alternative panel row grouping using existing supported kernels.
- Why it might help: Reduce launch or reduction latency with unchanged rank and iteration policy.
- Quality and edge cases: Orthogonality, reconstruction, residual, rank-deficient and tall/skinny cases.
- Intended control: `MOJOLEARN_AFCL_L09`; OFF until later qualification.

### AFCL-L10 — NMF update memory traffic

- Callers: NMF. Source area: `x_decomp/`.
- A: Separate existing update passes.
- B: Fuse elementwise multiply/divide and nonnegative write where outputs permit.
- Why it might help: Remove a materialized intermediate without changing solver objective.
- Quality and edge cases: Reconstruction objective, nonnegative factors, zero denominators and heldout transform.
- Intended control: `MOJOLEARN_AFCL_L10`; OFF until later qualification.

### AFCL-L11 — Gaussian-process prediction batching

- Callers: GaussianProcessRegressor and GaussianProcessClassifier. Source area: `gaussian_process/`.
- A: Current prediction/variance tile schedule.
- B: Alternative bounded query batch or row tile.
- Why it might help: Amortize launch and kernel reads while bounding workspace.
- Quality and edge cases: Predictive mean, nonnegative variance, uncertainty calibration, logloss and ill-conditioning.
- Intended control: `MOJOLEARN_AFCL_L11`; OFF until later qualification.

### AFCL-L12 — Random-feature and kernel output work

- Callers: RBFSampler, Nystroem and KernelRidge. Source area: `kernel_methods/`.
- A: Current kernel/projection epilogue.
- B: Coalesced or fused projection epilogue with unchanged random draws.
- Why it might help: Save row traffic in reusable classical feature maps.
- Quality and edge cases: Feature-map quality, downstream task quality, seed behavior and kernel/solve residual.
- Intended control: `MOJOLEARN_AFCL_L12`; OFF until later qualification.

### AFCL-G01 — Exact nearest-neighbor distance geometry

- Callers: NearestNeighbors, KNeighborsClassifier and KNeighborsRegressor. Source area: `neighbors/impl/detail/fast_mma_knn.mojo`.
- A: Current MMA query/index tile.
- B: Alternative hardware-sized query/index tile.
- Why it might help: Change operand reuse and partial-distance pressure without reducing the index.
- Quality and edge cases: Tie-aware exact recall, no duplicate IDs, vote/regression quality and distance error.
- Intended control: `MOJOLEARN_AFCL_G01`; OFF until later qualification.

### AFCL-G02 — Streaming top-k merge granularity

- Callers: Exact nearest-neighbor callers. Source area: `neighbors/impl/detail/fast_topk_knn.mojo`.
- A: Current candidate chunk/merge schedule.
- B: Bounded alternative candidate merge grouping.
- Why it might help: Trade scratch writes for merge work without dropping candidates.
- Quality and edge cases: Exhaustive candidate coverage, ties, k tails, k greater than tile and duplicate points.
- Intended control: `MOJOLEARN_AFCL_G02`; OFF until later qualification.

### AFCL-G03 — DBSCAN epsilon-neighborhood schedule

- Callers: DBSCAN. Source area: `dbscan/impl/neighbors/fast_mma_eps.mojo`.
- A: Current pairwise tile schedule.
- B: Alternative row tile for complete epsilon neighborhoods.
- Why it might help: Reuse distance operands while retaining every edge that passes epsilon.
- Quality and edge cases: Cluster-pair agreement, noise/core membership, epsilon boundary and dense components.
- Intended control: `MOJOLEARN_AFCL_G03`; OFF until later qualification.

### AFCL-G04 — HDBSCAN core-distance schedule

- Callers: HDBSCAN. Source area: `hdbscan/impl/detail/fast_apple.mojo`.
- A: Current core-distance/query grouping.
- B: Alternative bounded core-distance work grouping.
- Why it might help: Increase occupancy without pruning neighbors or altering min_samples.
- Quality and edge cases: Cluster quality, noise, membership probabilities, MST/linkage validity and ties.
- Intended control: `MOJOLEARN_AFCL_G04`; OFF until later qualification.

### AFCL-G05 — KMeans assignment work grouping

- Callers: KMeans and KMeans-based consumers. Source area: `cluster/`.
- A: Current assignment tile.
- B: Alternative query/centroid tiling of exhaustive assignment.
- Why it might help: Improve centroid reuse without changing starts or distance semantics.
- Quality and edge cases: Full-data inertia, heldout assignment, empty clusters and unchanged training budget.
- Intended control: `MOJOLEARN_AFCL_G05`; OFF until later qualification.

### AFCL-G06 — KMeans center partial chunks

- Callers: KMeans. Source area: `cluster/`.
- A: Current center partial granularity.
- B: Alternative row chunks for center accumulation.
- Why it might help: Balance occupancy, contention and scratch fold work.
- Quality and edge cases: Inertia, finite centers, counts, weighted rows and high cluster imbalance.
- Intended control: `MOJOLEARN_AFCL_G06`; OFF until later qualification.

### AFCL-G07 — MiniBatchKMeans resident schedule

- Callers: MiniBatchKMeans. Source area: `x_cluster/minibatch_fast.mojo`.
- A: Current resident mini-batch launch geometry.
- B: Alternative block/chunk grouping preserving every batch and stopping observation.
- Why it might help: Reduce idle lanes and launch work without reducing iterations or samples.
- Quality and edge cases: Inertia, labels, counts, reassignment, stopping point semantics and fixed random draws.
- Intended control: `MOJOLEARN_AFCL_G07`; OFF until later qualification.

### AFCL-G08 — GMM expectation row scheduling

- Callers: GaussianMixture. Source area: `mixture/`.
- A: Current E-step launch/chunk schedule.
- B: Alternative bounded row/component tile.
- Why it might help: Reuse component parameters while retaining stable logsumexp.
- Quality and edge cases: Heldout likelihood, responsibilities, covariance positivity and converged status.
- Intended control: `MOJOLEARN_AFCL_G08`; OFF until later qualification.

### AFCL-G09 — IVF query work partition

- Callers: IVFIndex and applicable ANN callers. Source area: `ivf/impl/neighbors/ivf_flat/fast_ivf_scan.mojo`.
- A: Current scan task size.
- B: Alternative bounded list/query task chunk.
- Why it might help: Balance skewed inverted lists without changing nprobe or candidate set.
- Quality and edge cases: Recall at fixed probes, filters, no duplicate IDs, exhaustive-probe control and index identity.
- Intended control: `MOJOLEARN_AFCL_G09`; OFF until later qualification.

### AFCL-G10 — UMAP optimizer edge grouping

- Callers: UMAP. Source area: `umap/optimizer_fast.mojo`.
- A: Current edge launch work.
- B: Alternative edge block size or reuse within existing seeded schedule.
- Why it might help: Change occupancy without skipping edges or negative samples.
- Quality and edge cases: Trustworthiness, downstream task quality, fixed epochs/seed/negative-sample count.
- Intended control: `MOJOLEARN_AFCL_G10`; OFF until later qualification.

### AFCL-G11 — Spectral and LLE local product schedule

- Callers: SpectralEmbedding, SpectralClustering, LLE and related manifold callers. Source area: `x_neighbors/`.
- A: Current local solve/product schedule.
- B: Alternative bounded row or eigensolver work grouping.
- Why it might help: Reduce serial work while preserving graph and solver convergence settings.
- Quality and edge cases: Embedding neighborhood quality, eigensystem residual, clustering quality and disconnected graphs.
- Intended control: `MOJOLEARN_AFCL_G11`; OFF until later qualification.

### AFCL-G12 — MeanShift neighborhood row grouping

- Callers: MeanShift. Source area: `x_cluster/meanshift_fast.mojo`.
- A: Current neighborhood tile.
- B: Alternative complete pairwise neighborhood tile.
- Why it might help: Improve load reuse without bandwidth approximation or seed subsampling.
- Quality and edge cases: Cluster centers, cluster-pair agreement, bandwidth boundaries and convergence status.
- Intended control: `MOJOLEARN_AFCL_G12`; OFF until later qualification.

### AFCL-T01 — Boosting histogram row chunks

- Callers: GradientBoosting and CatBoost-style trees. Source area: `gbdt/`.
- A: Current histogram row chunk.
- B: Alternative bounded row chunk.
- Why it might help: Balance histogram scratch traffic and available independent groups.
- Quality and edge cases: Heldout RMSE/logloss/ranking quality, identical hyperparameters and all rows consumed.
- Intended control: `MOJOLEARN_AFCL_T01`; OFF until later qualification.

### AFCL-T02 — Boosting histogram feature grouping

- Callers: Symmetric and applicable depthwise boosting. Source area: `gbdt/`.
- A: Current features per task.
- B: Alternative contiguous feature group.
- Why it might help: Reuse row/bin loads without changing bins or selected features.
- Quality and edge cases: Quality across numerical/categorical columns, missing values and multiclass objectives.
- Intended control: `MOJOLEARN_AFCL_T02`; OFF until later qualification.

### AFCL-T03 — Binarization launch geometry

- Callers: Boosted-tree fit and prediction. Source area: `gbdt/gpu_data/`.
- A: Current binarization work grouping.
- B: Alternative coalesced row/feature launch.
- Why it might help: Reduce launch overhead and improve sequential input reads.
- Quality and edge cases: Bin boundary semantics, NaNs, categorical codes and final fitted task quality.
- Intended control: `MOJOLEARN_AFCL_T03`; OFF until later qualification.

### AFCL-T04 — Tree row-partition schedule

- Callers: Depthwise/lossguide/symmetric boosting where applicable. Source area: `gbdt/`.
- A: Current partition launch geometry.
- B: Alternative bounded partition block.
- Why it might help: Reduce scattered read/write overhead while preserving membership.
- Quality and edge cases: Every row exactly once per partition, missing routing, child counts and task quality.
- Intended control: `MOJOLEARN_AFCL_T04`; OFF until later qualification.

### AFCL-T05 — Split-score reduction schedule

- Callers: Boosted-tree split selection. Source area: `gbdt/`.
- A: Current split-candidate grouping.
- B: Alternative cooperative or grouped score reduction.
- Why it might help: Expose more independent score work without reducing candidates.
- Quality and edge cases: Best-score/tie semantics, constrained splits, monotonic settings and final quality.
- Intended control: `MOJOLEARN_AFCL_T05`; OFF until later qualification.

### AFCL-T06 — Leaf estimation partial chunks

- Callers: Boosted-tree regression/classification/ranking. Source area: `gbdt/methods/leaves_estimation/`.
- A: Current leaf-statistic chunk.
- B: Alternative bounded row/leaf grouping.
- Why it might help: Reduce contention and long partial sums at fixed leaf solver settings.
- Quality and edge cases: Leaf objectives, sample weights, small Hessians, ranking groups and heldout quality.
- Intended control: `MOJOLEARN_AFCL_T06`; OFF until later qualification.

### AFCL-T07 — Boosted-tree prediction grouping

- Callers: Boosted-tree predict and predict_proba. Source area: `gbdt/models/`.
- A: Current row/tree launch layout.
- B: Alternative rows or trees per block with same tree order.
- Why it might help: Improve model reuse and output traffic.
- Quality and edge cases: Prediction error, accuracy/logloss, categorical CTR routing and repeated-call lifetime.
- Intended control: `MOJOLEARN_AFCL_T07`; OFF until later qualification.

### AFCL-T08 — RandomForest traversal geometry

- Callers: RandomForestClassifier and RandomForestRegressor. Source area: `ensemble/`.
- A: Current traversal launch.
- B: Alternative contiguous query grouping.
- Why it might help: Reuse node cache lines without pruning trees.
- Quality and edge cases: Heldout quality, probabilities, uneven depth, multioutput and OOB behavior.
- Intended control: `MOJOLEARN_AFCL_T08`; OFF until later qualification.

### AFCL-T09 — ExtraTrees training work grouping

- Callers: ExtraTreesClassifier and ExtraTreesRegressor. Source area: `extratrees/`.
- A: Current training histogram/work chunks.
- B: Alternative bounded row or feature grouping.
- Why it might help: Improve parallelism without changing sampled thresholds or feature sets.
- Quality and edge cases: Fixed RNG mapping, impurity behavior, task quality and completed tree count.
- Intended control: `MOJOLEARN_AFCL_T09`; OFF until later qualification.

### AFCL-T10 — IsolationForest score grouping

- Callers: IsolationForest. Source area: `isolation_forest/`.
- A: Current score traversal launch.
- B: Alternative bounded query grouping.
- Why it might help: Amortize node traffic with unchanged tree ensemble and correction.
- Quality and edge cases: Anomaly ranking, path lengths, contamination decisions and degenerate leaves.
- Intended control: `MOJOLEARN_AFCL_T10`; OFF until later qualification.

### AFCL-T11 — TreeSHAP query scheduling

- Callers: TreeExplainer and classical tree explainers. Source area: `xtrees/shap_device.mojo`.
- A: Current query/feature work layout.
- B: Alternative adjacent-row or block grouping.
- Why it might help: Reuse model data while computing all contributions.
- Quality and edge cases: Additivity, reference contributions, interactions when supported and memory lifetime.
- Intended control: `MOJOLEARN_AFCL_T11`; OFF until later qualification.

### AFCL-T12 — Categorical CTR preprocessing schedule

- Callers: Categorical boosted-tree fit and inference. Source area: `gbdt/ctrs/`.
- A: Current CTR/count task grouping.
- B: Alternative bounded row/category chunks.
- Why it might help: Improve occupancy without changing category ordering or smoothing.
- Quality and edge cases: Leakage prevention, unseen categories, ordered statistics, ranking groups and final quality.
- Intended control: `MOJOLEARN_AFCL_T12`; OFF until later qualification.

### AFCL-P01 — Scaler reduction independent accumulators

- Callers: StandardScaler, PowerTransformer and class-statistic consumers. Source area: `x_prep/fastred.mojo`.
- A: One running accumulation chain per lane.
- B: Two independent accumulators over the same observations.
- Why it might help: Hide load/add dependency latency without approximating statistics.
- Quality and edge cases: Large offsets, NaNs, zero variance, transformed outputs and downstream model quality.
- Intended control: `MOJOLEARN_AFCL_P01`; OFF until later qualification.

### AFCL-P02 — MaxAbs row tiling

- Callers: MaxAbsScaler. Source area: `x_prep/fastmaxabs.mojo`.
- A: Current row tile.
- B: Half-sized row tiles with unchanged feature grouping.
- Why it might help: Expose more work on narrow inputs at the cost of partial-fold traffic.
- Quality and edge cases: Exact finite maxima, NaN/zero handling and transformed output.
- Intended control: `MOJOLEARN_AFCL_P02`; OFF until later qualification.

### AFCL-P03 — Categorical Naive Bayes count grouping

- Callers: CategoricalNB. Source area: `x_prep/fastnb.mojo`.
- A: Current categorical atomic launch.
- B: Alternative threads per block.
- Why it might help: Change occupancy and atomic contention without changing integer counts.
- Quality and edge cases: Class/feature counts, smoothing, unseen categories and heldout logloss.
- Intended control: `MOJOLEARN_AFCL_P03`; OFF until later qualification.

### AFCL-P04 — Sparse Naive Bayes row grouping

- Callers: MultinomialNB, BernoulliNB and ComplementNB where admitted. Source area: `x_prep/fastnb_csr.mojo`.
- A: Current CSR rows per block.
- B: Half-sized CSR row group.
- Why it might help: Bound row-search and shared-metadata pressure on skewed sparse rows.
- Quality and edge cases: Counts, class priors, negative-input refusal, posterior quality and empty rows.
- Intended control: `MOJOLEARN_AFCL_P04`; OFF until later qualification.

### AFCL-P05 — Target-encoding work grouping

- Callers: TargetEncoder. Source area: `x_prep/fastprep2.mojo`.
- A: Current category/fold row tile.
- B: Alternative cooperative group size for existing sums.
- Why it might help: Reduce long row reductions while preserving cross-fitting and leakage rules.
- Quality and edge cases: Fold isolation, category smoothing, unseen-category behavior and downstream task quality.
- Intended control: `MOJOLEARN_AFCL_P05`; OFF until later qualification.

### AFCL-P06 — Iterative-imputer convergence fold

- Callers: IterativeImputer. Source area: `x_prep/fastprep2.mojo`.
- A: Current row-sum/max work grouping.
- B: Alternative convergence reduction work grouping.
- Why it might help: Reduce reduction latency without changing tolerance or observation frequency.
- Quality and edge cases: Imputed values, observed entries unchanged, stopping semantics and downstream quality.
- Intended control: `MOJOLEARN_AFCL_P06`; OFF until later qualification.

### AFCL-P07 — Feature-selection row tile

- Callers: f_regression, r_regression and f_classif selectors. Source area: `x_prep/select_fast.mojo`.
- A: Current observations per lane.
- B: Half as many observations per lane.
- Why it might help: Trade increased grid parallelism against partial scratch/fold cost.
- Quality and edge cases: Scores, p-values, selected features, class imbalance and force_finite behavior.
- Intended control: `MOJOLEARN_AFCL_P07`; OFF until later qualification.

### AFCL-P08 — Regression metric independent accumulators

- Callers: Regression metrics and scoring callers. Source area: `x_metrics/reg_epi.mojo`.
- A: Current serial output/statistic fold.
- B: Independent accumulators or grouped partial work over same rows.
- Why it might help: Shorten dependency chains in metric reduction.
- Quality and edge cases: Independent FP64 metric agreement, sample weights, multioutput and finite policy.
- Intended control: `MOJOLEARN_AFCL_P08`; OFF until later qualification.

### AFCL-P09 — Ranking metric prefix work grouping

- Callers: ROC/PR/AUC and ranking metric callers. Source area: `x_metrics/ranking.mojo`.
- A: Current ranking chunk/work schedule.
- B: Alternative bounded work grouping with stable equal-score groups.
- Why it might help: Improve occupancy without dropping thresholds or changing ties.
- Quality and edge cases: Exact threshold membership, tie groups, weighted areas and degenerate labels.
- Intended control: `MOJOLEARN_AFCL_P09`; OFF until later qualification.

### AFCL-P10 — Resampling gather tile geometry

- Callers: bootstrap, permutation_test and resample. Source area: `resample/gather_fast.mojo`.
- A: Current gathered row/column tile.
- B: Alternative fixed hardware-sized tile.
- Why it might help: Coalesce wide rows and improve narrow-row occupancy; same drawn indices.
- Quality and edge cases: Exact paired-array row draws, old-output lifetime and statistical interval/p-value quality.
- Intended control: `MOJOLEARN_AFCL_P10`; OFF until later qualification.

### AFCL-P11 — Stationarity series reduction schedule

- Callers: AutoARIMA differencing selection and KPSS. Source area: `tsa/impl/select_d_fast.mojo`.
- A: Current series reduction group.
- B: Alternative supported per-series group width.
- Why it might help: Reduce serial per-series work without changing tested orders or significance.
- Quality and edge cases: KPSS statistics, differencing decisions, finite-input refusals and forecast quality.
- Intended control: `MOJOLEARN_AFCL_P11`; OFF until later qualification.

### AFCL-P12 — Batched ARIMA likelihood work grouping

- Callers: ARIMA and AutoARIMA. Source area: `arima/impl/batched_kalman.mojo`.
- A: Current batch launch geometry.
- B: Alternative series-per-block or block width using existing recurrence.
- Why it might help: Increase independent-series occupancy while preserving full recurrence and optimizer settings.
- Quality and edge cases: Likelihood/gradient quality, selected orders, optimizer status and heldout forecast error.
- Intended control: `MOJOLEARN_AFCL_P12`; OFF until later qualification.

### AFCL-G13 — KDE stable score work grouping

- Callers: KernelDensity. Source area: `kde/`.
- A: Current query/train tile.
- B: Alternative bounded query grouping preserving stable logsumexp.
- Why it might help: Reuse training data while keeping every density contribution.
- Quality and edge cases: Independent density/log-likelihood error, extreme bandwidths, far queries and heldout likelihood.
- Intended control: `MOJOLEARN_AFCL_G13`; OFF until later qualification.

### AFCL-G14 — Agglomerative graph/MST work grouping

- Callers: AgglomerativeClustering. Source area: `hierarchy/`.
- A: Current edge or row scheduling.
- B: Alternative bounded complete edge work grouping.
- Why it might help: Balance independent distance or Boruvka work without dropping edges.
- Quality and edge cases: Linkage validity, cluster-pair quality, ties, disconnected graphs and supported linkage semantics.
- Intended control: `MOJOLEARN_AFCL_G14`; OFF until later qualification.

### AFCL-L13 — Robust covariance C-step scheduling

- Callers: MinCovDet and EllipticEnvelope. Source area: `x_decomp/mcd_fast.mojo`.
- A: Current per-trial work grouping.
- B: Alternative row or independent-trial schedule.
- Why it might help: Reuse covariance factors and balance active trials without reducing starts or C steps.
- Quality and edge cases: Robust covariance/support/location, contamination behavior, anomaly ranking and singular cases.
- Intended control: `MOJOLEARN_AFCL_L13`; OFF until later qualification.

### AFCL-L14 — SGD linear update work grouping

- Callers: SGDClassifier, SGDRegressor and applicable linear SGD callers. Source area: `glm/`.
- A: Current gradient/prediction launch geometry.
- B: Alternative feature or row grouping preserving exact training examples and epoch order.
- Why it might help: Improve occupancy or load reuse without reducing epochs or changing learning-rate policy.
- Quality and edge cases: Heldout loss/accuracy/RMSE, seeded shuffle semantics, weights, sparse inputs and convergence.
- Intended control: `MOJOLEARN_AFCL_L14`; OFF until later qualification.

### AFCL-P13 — Gaussian Naive Bayes class-statistic chains

- Callers: GaussianNB and applicable class-statistic callers. Source area: `x_prep/fastred.mojo`.
- A: Single per-lane statistic chain.
- B: Independent class-filtered accumulation chains.
- Why it might help: Shorten dependent sums while retaining every class observation and centered variance.
- Quality and edge cases: Class counts/means/variances, variance smoothing, rare classes and heldout logloss.
- Intended control: `MOJOLEARN_AFCL_P13`; OFF until later qualification.

### AFCL-P14 — Holt-Winters series launch geometry

- Callers: HoltWinters and applicable ETS forecasting callers. Source area: `holtwinters/`.
- A: Current series block width.
- B: Alternative independent-series block width.
- Why it might help: Amortize scheduling without modifying serial recurrence, season length or optimizer work.
- Quality and edge cases: Heldout forecast error, smoothing parameters, initial states, seasonal tails and refusal handling.
- Intended control: `MOJOLEARN_AFCL_P14`; OFF until later qualification.

## Broader mechanisms to carry into these cards

Retain input/model/workspace residency only where existing ownership and invalidation
can express it safely. Fuse adjacent elementwise passes only when public intermediate
outputs are not required. Bound scratch from actual allocation cost; preserve recovery
on allocation/refusal paths. Balance irregular tasks by work, not dataset names.
Stable logsumexp, centered moments and compensated reductions can trade additional
arithmetic for safe faster matrix products, but must be evaluated by final model
quality. Do not revive previously rejected candidates unchanged or import neural
GEMM changes globally. Unsupported stream, graph, subgroup or compiler capabilities
must be recorded as requests for Modular; do not patch toolchain internals.

These are optimization axes within the cards, not claims of implemented memory
pools, graph capture, mixed precision or algorithm substitutions.

## Lane handoff rule

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

Propagate that reminder to any further delegated agent. Do not compile, verify,
measure, launch a GPU job, run a manifest checker or run Git hooks in this task.
Lane manifests and implementation notes will distinguish source written from
uncompiled, unverified and unmeasured behavior. All quality and speed remain pending.

## Source delivery and selecting arms

The [shared experiment integration](INTEGRATION.md) now registers every card in
`tools/performance_ideas.py`, with paired build packages, full-workload adapters
and source-only board inputs. The standalone selector below remains available.

The implementation records are the authority for actual source scope; the cards
above preserve the hypotheses written before delegation. Some broad ideas narrowed
to a supported caller schedule or extended a fusion that already existed. These
records name those differences explicitly:

- [Linear models, numerical methods and kernels](lanes/linear.md), with [A/B controls](lanes/linear.json).
- [Neighbors, clustering, manifold learning and density](lanes/geometry.md), with [A/B controls](lanes/geometry.json).
- [Trees and tree explainers](lanes/trees.md), with [A/B controls](lanes/trees.json).
- [Preprocessing, metrics, resampling and forecasting](lanes/preprocessing.md), with [A/B controls](lanes/preprocessing.json).

[select.py](select.py) is an offline configuration emitter. It supports `--list`,
one or more card IDs, and `--factorial` for up to six interacting cards. It emits
exact bare compiler defines and environment requirements as JSON, and has no
build, verification, measurement or command-execution facility. It was written
but **not run**, including its list mode. Example future configuration-only use:

```sh
python3 experiments/apple_fast_classical_20261006/select.py AFCL-L01
python3 experiments/apple_fast_classical_20261006/select.py AFCL-P11 AFCL-P12 --factorial
```

Each define is opt-in by **presence**. To disable a candidate, omit its define;
`-D MOJOLEARN_AFCL_P01=0` still defines it and does not disable it. Start each arm
from a clean configuration using the established FAST Apple binding recipe,
without IDENTICAL/DETERMINISTIC or unrelated experiment/rollback defines. Existing
route prerequisites are intentionally equal in A/B. The G01 MMA route and G02's
MMA-disabled route are mutually exclusive and the selector refuses that pairing.
Other path reach and combination admission still require future qualification.
No combined all-on configuration is an accepted default or a performance result.

Implementation evidence here consists only of source, the idea inventory and
lane handoffs. There are no new build, test, correctness, quality or timing logs.
All those stages were deliberately skipped by the owner's instruction. The branch
is not a qualified release; all new candidate switches remain OFF. Git commit/push
uses hooks disabled and a CI-skip commit marker to avoid initiating checks.
