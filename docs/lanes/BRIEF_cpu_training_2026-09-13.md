# BRIEF, CPU training bitwise identical to the GPU vendors (2026-09-13)

Branch `lane/cpu-training-plan`, off `main` at `0dcc1204e`. Read-only
investigation, nothing built, nothing run. Andrew's goal, Sep 13. Make
TRAINING bitwise identical between the three GPU vendors and a plain CPU for
every lane of `tools/identity_break.py` (28 lanes, 252 train cells plus the
infer and model columns), and eventually every public estimator.

What exists today. The byte LM ships a CPU training step
(`bindings/build_byte_lm_host.sh`, `training/byte_lm_host_backward.mojo`,
`python/mojolearn/_byte_lm_host.py::LanguageModelHostTrainer`) gated on seven
free runners by `.github/workflows/byte-lm-cpu-gate.yml`, 640 of 640 arrays
equal on all 128 recorded steps (`docs/BYTE_LM_CPU_TRAINING.md`). Forest and
GBDT CPU INFERENCE exists on `lane/forest-host-inference`
(`docs/lanes/BRIEF_forest_host_inference_2026-09-13.md` on that branch, seven
CPUs reproducing the Metal predictions, run 34780078300). Everything else is
GPU-only by policy, `SUPPORT_MATRIX.md:54` ("No CPU implementation of any
estimator, block or trainer other than the byte LM, and no CPU fallback for
them").

The thesis holds, with a smaller reach than "most fits". Every IDENTICAL
kernel is gated against a host oracle, but the oracle covers the WHOLE fit for
ten of the 28 lanes, a stage for eleven, and nothing for seven (the two random
forests, the four GBDT lanes and DBSCAN). The per-lane census is section 1.

Reading conventions. "Whole fit" means every arithmetic step from the lane's
inputs to the arrays the train column hashes, at the shipped default, in a
routine that takes no `DeviceContext`. "Stage" means one kernel's fold has a
host twin and the rest does not. Hours are for a host fit entry that
reproduces the GPU bits, including the binding, the Python route, and the
gate row, and they assume the pattern of section 3 exists once (its own hours
are in phase 0 of section 4). Every "none" below is an absence claim backed by
a grep whose output is printed, not counted.

## 1. Per lane

### 1.0 Lane to estimator to binding to package

`grep -n '^@lane' tools/identity_break.py` gives 28 lanes at lines 242 to
447. The binding each estimator calls resolves through
`python/mojolearn/_backend.py:155-231` (`_MODULES`) and `:1214-1237`
(`_build_script`); the script pins the CPU target and, on Linux, one GPU
architecture (`bindings/build_rf.sh:53-74`, `bindings/build.sh:199-253`).

| lane | estimator (Python file:line) | binding module, build script | Mojo package |
|---|---|---|---|
| rf-clf, rf-reg | `RandomForestClassifier` `randomforest.py:645`, `RandomForestRegressor` `:759` | `_mojolearn_rf`, `build_rf.sh` | `ensemble/` |
| et-clf, et-reg | `ExtraTreesClassifier` `extratrees.py:444`, `ExtraTreesRegressor` `:542` | `_mojolearn_trees`, `build_trees.sh` | `extratrees/` |
| gbdt-symmetric, gbdt-depthwise, gbdt-lossguide, gbdt-rmse | `GradientBoosting` `ensemble.py:401`, fit `:1110` | `_mojolearn_gbdt`, `build_gbdt.sh` | `gbdt/` |
| kmeans | `KMeans` `cluster.py:24`, fit `:106` | `_mojolearn`, `build.sh` | `cluster/` |
| knn, knn-clf, knn-reg | `NearestNeighbors` `neighbors.py:296`, `KNeighborsClassifier` `:563`, `KNeighborsRegressor` `:787` | `_mojolearn`, `build.sh` | `neighbors/` |
| dbscan | `DBSCAN` `density.py:45`, fit `:185` | `_mojolearn_estimators`, `build_estimators.sh` | `dbscan/` |
| pca, tsvd | `PCA` `decomposition.py:33`, `TruncatedSVD` `:374` | `_mojolearn_estimators` | `decomposition/` |
| ols, ridge, logistic | `LinearRegression` `linear_model.py:400`, `Ridge` `:579`, `LogisticRegression` `:732` | `_mojolearn_estimators` (not the solver binding) | `glm/` |
| lasso, elasticnet | `ElasticNet` `_solver_impl.py:52`, `Lasso` `:384` | `_mojolearn_solver`, `build_solver.sh` | `solver/` |
| svc | `SVC` `_svm_impl.py:186`, fit `:478` | `_mojolearn_svm`, `build_svm.sh` | `svm/` |
| kde | `KernelDensity` `density.py:286` | `_mojolearn_estimators` | `kde/` |
| agglomerative | `AgglomerativeClustering` `_hierarchy_impl.py:49`, fit `:256` | `_mojolearn_solver` | `hierarchy/` |
| spectral | `SpectralClustering` `_spectral_impl.py:113`, fit `:426` | `_mojolearn_metrics`, `build_metrics.sh` | `spectral/` |
| holtwinters | `ExponentialSmoothing` `_tsa_impl.py:280`, fit `:457` | `_mojolearn_tsa`, `build_tsa.sh` | `holtwinters/` |
| gemm-pinned | `linalg.matmul` `_linalg_impl.py:316` | `_mojolearn_linalg`, `build_linalg.sh` | `gemm/` |
| metrics | `_metrics_impl.py:261, 339, 488, 529, 764` | `_mojolearn_metrics` | `metrics/` |
| iforest | `IsolationForest` `_iforest_impl.py:368` | `_mojolearn_svm` (shared with SVC) | `isolation_forest/` |

### 1.1 The census

Columns. (a) the GPU fit entry, binding function then the Mojo routine it
calls; (b) the host implementation of the fit, with its GPU-import status;
(c) coverage, whole fit or stage; (d) fitted state and serialization; (e)
hours and the main risk.

**rf-clf, rf-reg.**
(a) `rf_classifier_fit_binding` `bindings/_mojolearn_rf.mojo:480` and
`rf_regressor_fit_binding` `:506`, both through `fit_forest`
`ensemble/randomforest.mojo:2299`, which takes `ctx: DeviceContext` and
`DeviceBuffer` inputs; the binding creates the context at `:388`. Kernels
from quantiles through leaves are listed in
`ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
(`bin_dataset_kernel :2971`, `build_histograms_kernel :2149`,
`find_best_splits_kernel :2744`, `leaf_kernel :1837`).
(b) NONE. `DecisionTree.fit` raises "not implemented: fit a one-tree forest
through ensemble.randomforest.fit_forest instead"
(`ensemble/decisiontree/decisiontree.mojo:504`). The host pieces are stages
only, `host_lower_bound` `ensemble/checks/builder_kernels_check.mojo:215`,
`_host_reduce` `ensemble/checks/split_check.mojo:227`, `host_sort`
`ensemble/checks/quantiles_check.mojo:477`, and a C++ arithmetic oracle
`ensemble/tools/cuml_oracle/oracle.cpp` whose consumer says it "DOES NOT
SETTLE ... an end-to-end tree" (`ensemble/checks/cuml_oracle_check.mojo:24-31`).
The RNG is host-callable end to end, `uniform_int_host` `core/philox.mojo:235`
(bit-equal to the device draw, `ensemble/checks/philox_check.mojo:351, 436,
617`), `uniform_double_host` `:357`, `shuffled_feature`
`core/shuffle_iterator.mojo:299`, `fnv1a32_hash_seed_tree`
`ensemble/decisiontree/batched_levelalgo/random_utils.mojo:136`.
(c) nothing of the fit. The production fit reads one kernel-matrix row,
`column_shared_limit` at `ensemble/decisiontree/batched_levelalgo/builder.mojo:2480`.
(d) five flat arrays `_offsets, _colid, _quesval, _left_child, _leaves`
(`_forest_protocol.py:18`, dtypes `:344`), `save` `randomforest.py:510`
through `_serialize.write_npz` `:547`, `load` `:550`. The saved bytes are a
pure function of the arrays, so a host fit that produces the same arrays
passes the model column with no format change.
(e) 80 hours for both lanes together (one host restatement of the cuML
batched-level builder, quantiles, binning, histogram, gain and tie rule,
split reduce, partition, leaf). Main risk, the histogram and split-reduce
fold order and the quantile sort (a segmented radix sort on the device; the
host must produce the same bins on ties, `ensemble/checks/quantiles_check.mojo`
is the reference for that stage).

**et-clf, et-reg.**
(a) `et_classifier_fit_binding` `bindings/_mojolearn_trees.mojo:295` and
`et_regressor_fit_binding` `:374`, refusing a non-GPU device slot at `:330`
and `:407` ("Extra Trees training is GPU-only; device (slot 20) must be 1"),
then `fit_extra_trees_classifier_device` `extratrees/estimator.mojo:597` and
`fit_extra_trees_regressor_device` `:715`.
(b) WHOLE FIT, and it is the arm the device fit is gated against.
`fit_extra_trees_classifier_reference` `extratrees/estimator.mojo:571` and
`fit_extra_trees_regressor_reference` `:691` take no context and run
`fit_classification` `extratrees/impl/randomforest/randomforest.mojo:205` and
`train_classification` / `train_regression`
`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:1078, 1241`.
The classifier claim at `extratrees/estimator.mojo:617-621` is "The forest it
returns is the SAME forest the host arm returns, tree for tree and node for
node". The regressor claim at `:742-748` is that the tree STRUCTURE is
bit-identical but the LEAF VALUES "differ by at most one quantization step",
because the device consumes labels quantized by `quantize_labels` `:669` and
the host arm takes Float64 means. The whole ET RNG is host code with no GPU
import, `extratrees/checks/pcg_rng.mojo` (row seed `:97`, `PCGenerator :124`,
`uniform_threshold :407`). The two fit files DO import the GPU host module,
printed below, so a host build compiles them with kernel bodies present.
(c) whole fit for the classifier; whole structure plus a restated leaf mean
for the regressor. No kernel-matrix row is read in `extratrees/`.
(d) the same five arrays as RF plus `depth_cap_bound_, max_depth_resolved_,
max_features_`; `save` `extratrees.py:303`, `write_npz` at `:343`, `load` `:346`.
(e) 10 hours et-clf, 12 hours et-reg. Main risk, whether
`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo` (kernel bodies,
`std.gpu` imports at `:95`) compiles with no accelerator target. The byte LM
probe measured that `max.gpu.host` IMPORTS compile host-only on seven runners
(`docs/BYTE_LM_CPU_TRAINING.md:346-348`), but no file with a kernel body has
been compiled that way. If it does not, the host trainer is moved into an
import-only module the way `core/forest_host_predict.mojo` was, which is
mechanical and adds about 8 hours. Second risk, the regressor's leaf mean
must be restated over the quantized labels to match the device bits, and the
check at `:749-753` REQUIRES the host arm to differ, so the new arm cannot be
the existing reference unchanged.

**gbdt-symmetric, gbdt-depthwise, gbdt-lossguide, gbdt-rmse.**
(a) `gbdt_fit_binding` `bindings/_mojolearn_gbdt.mojo:169`, context created
unconditionally at `:387-389`, then `gbdt_fit` `gbdt/estimator.mojo:406`,
`train` `gbdt/train.mojo:667`, `fit_with_test`
`gbdt/methods/doc_parallel_boosting.mojo:853`. Symmetric and RMSE run
`run_tree_layout_traced`
`gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:4628`;
Depthwise and Lossguide run `fit_non_symmetric_tree`
`greedy_search_helper_depthwise.mojo:725`.
(b) NONE. The word "greedy" in `gbdt/methods/greedy_subsets_searcher/` names
CatBoost's GPU searcher and "oracle" names either the device leaf estimator
(`gbdt/methods/leaves_estimation/pointwise_oracle.mojo`, context at `:194`)
or the committed CatBoost fixtures (`bench/oracle*.txt`, differ at
`checks/oracle_check.mojo:543`). Every check named in the lane charter opens
a `DeviceContext` (printed below); the two that do not,
`checks/oracle_sweep_main.mojo` and `checks/gbdt_sub_byte_identity_check.py`,
are fixture differs and a subprocess driver. The one genuine host reference
in the set is `host_weighted_quantile` `checks/exact_estimation_check.mojo:102`,
a Quantile and MAE leaf stage that is off the four lanes' path. What IS
already host code on the fit path, with no GPU import, is the border search
(`gbdt/grid_creator/binarization.mojo:203, 366, 470, 787`), the grid policy
and compressed-index layout, the options, the model text
(`gbdt/models/model_text.mojo`), the Lossguide leaf selection
(`greedy_search_helper_lossguide.mojo`, whole file), the RNG
(`gbdt/data/permutation.mojo`, `gbdt/train.mojo:595, 605`) and the
overfitting detector. At the lanes' defaults the permutation is the identity
(`gbdt/methods/oblivious_tree_fold_tasks.mojo:273`), no bootstrap, no feature
fraction, `random_strength` 0.
(c) nothing of the fit. Twenty-three kernel-matrix rows are read (the GBDT
census table in this brief's source sweep, `checks/kernel_matrix.mojo:377-606`
and `:1037`); under IDENTICAL every numeric one resolves to the pinned
`COLUMN_BIT_IDENTICAL` reading, and the fold order a host trainer must
reproduce is pinned at `write_reduces_histograms_kernel`
`gbdt/methods/greedy_subsets_searcher/kernel/histogram_utils.mojo:649`,
`write_reduces_from_fixed_kernel :712`, `scan_histograms_kernel :257`,
`deterministic_sum_lanes_kernel` `gbdt/targets/kernel/pointwise_targets.mojo:808`,
and `PINNED_PARTITION_CHUNKS_SM = 32` `checks/kernel_matrix.mojo:596`.
(d) the model is a text (`model_` `ensemble.py:1274`) plus `bias_`,
`best_iteration_`, the loss curves; `save` `ensemble.py:1515` writes the text
as `<u1` bytes through `write_npz` at `:1578`, `load` `:1581`. A host trainer
emitting the same text round-trips through the existing format.
(e) 120 hours for symmetric plus RMSE (binarize to cindex, target planes,
histograms in the pinned fold order, scan and subtract, split scoring with
the CatBoost tie rule, partition move, leaf estimation, model text), 40 more
for Depthwise and Lossguide. Main risk, the histogram fold order and the
sub-byte peel bounds (DEVIATION 2600, `hist_binary.mojo:236-240`,
`hist_half_byte.mojo:246-250`) which a serial host loop reproduces only if
it folds block partials in the same order the device does, not row order.

**kmeans.**
(a) `kmeans_fit_binding` `bindings/_mojolearn.mojo:363`, context at `:424`,
`kmeans_fit` `cluster/estimator.mojo:209`, `kmeans_fit_main`
`cluster/impl/detail/kmeans.mojo:921`. Init is k-means++ on the device
(`:246`, kernels in `cluster/checks/plus_plus.mojo`), each Lloyd iteration is
`min_cluster_and_distance_compute`
`cluster/impl/detail/min_cluster_distance_compute.mojo:167` (fused arm
`cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:246`) and the
center fold is a fixed-point Int32 accumulation (`sums_i32`, `weight_i32`,
scale from `plan_sum_scale` `cluster/estimator.mojo:143`).
(b) NONE. `cluster/` has no oracle file; `cluster/tools/sklearn_reference.py`
is a scikit-learn comparison, and `cluster/checks/kmeans_check.mojo` holds
Float64 stage truths inside a context (`:552-609`, `:721-827`). The RNG is
host code, `HostRng` `cluster/impl/detail/kmeans.mojo:113` (splitmix64), and
the one device draw `scalable_uniform` `cluster/checks/scalable_init.mojo:62`
is a pure hash of `(seed, i)`.
(c) nothing of the fit. Rows read, `K_LIB_FUSED_DISTANCE_NN`,
`K_LIB_REDUCE_BY_KEY`, `K_LIB_PLUS_PLUS`, `K_LIB_ROW_NORM`
(`cluster/impl/distance/unfused_distance_nn.mojo:59-63`,
`cluster/checks/reduce_by_key.mojo:54-58`, `cluster/checks/plus_plus.mojo:25-28`);
three of them are classified float folds
(`cluster/checks/kmeans_identity_check.mojo:736-741`) whose strided partials
then halving tree a host replay must reproduce.
(d) `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `sum_scale_`,
`weight_scale_` (`cluster.py:182-194`). No save or load.
(e) 24 hours. Main risk, the row-norm and reduce-by-key fold shapes (the
same `_host_row_norm_halving` shape `kde/checks/kde_oracle.mojo:94` already
restates for KDE) and the fixed-point center fold's scale choice
(`choose_scale_kernel` runs on the device for GBDT; for k-means the scale is
planned on the host at `cluster/estimator.mojo:143`, which helps).

**knn, knn-clf, knn-reg.**
(a) fit stores the array (`neighbors.py:460`); the arithmetic is
`knn_search_binding` `bindings/_mojolearn.mojo:183`, `knn_classify_binding`
`:238`, `knn_regress_binding` `:312`, then `knn_search`
`neighbors/estimator.mojo:316`, `knn_classifier_predict` `:743`,
`knn_regressor_predict` `:971`. Distances are the register tile
`neighbors/impl/detail/knn_brute_force.mojo:843-946`, selection the composite
key small-k selector or the 64-bit radix (`:1100`), votes
`neighbors/impl/selection/knn.mojo:383, 455, 510`.
(b) STAGE. `neighbors/checks/metric_oracle.mojo` (no GPU import, printed
below) covers the distance row, `oracle_row_norm :53`,
`oracle_metric_distance :89`, `oracle_distance_weights :217`. The selection
is order-independent by construction (the k smallest `(distance, index)`
keys under a total order, `checks/kernel_matrix.mojo:1054`), so a serial host
selection is the same bits by definition; the vote and average folds
(`class_probs_kernel`, `regress_avg_kernel`) have no host twin. No RNG.
Known defect `neighbors/NOT_IMPLEMENTED.tsv:31`, `core/row_norms.mojo:102`
uses the stdlib `sqrt`, not `identical_sqrt`; the host twin must call the
same function.
(c) distance stage only. Rows read, `K_LIB_SELECT_WARPSORT`, `K_LIB_ROW_NORM`,
`knn_query_tile_for`, `knn_distance_*_for`
(`neighbors/impl/detail/knn_brute_force.mojo:78-109`,
`neighbors/estimator.mojo:232`); the Apple-only identical rows are the
concern of section 2.
(d) `_index`, `_y_cols`, `_classes_list` (`neighbors.py`). No save or load.
(e) 12 hours for the three lanes. Main risk, the Apple zero-FMA repair and
preflight rows (`checks/kernel_matrix.mojo:1210-1235`) which a host build
compiles as the Apple column today (section 2), and the vote's float fold
order in `class_probs_kernel`.

**dbscan.**
(a) `dbscan_fit_binding` `bindings/_mojolearn_estimators.mojo:58`, context
at `:110`, `dbscan_fit` `dbscan/estimator.mojo:118`, `dbscan_fit_impl`
`dbscan/impl/dbscan.mojo:147`. Per batch, `eps_neighborhood_kernel`
`dbscan/impl/vertexdeg/algo.mojo:394`, `core_points_kernel`
`dbscan/impl/corepoints/compute.mojo:49`, the CSR scans
`dbscan/impl/adjgraph/algo.mojo:107-274`, and `propagate_label_kernel`
`dbscan/impl/label/merge_labels.mojo:66` iterated to a fixed point.
(b) NONE. `dbscan/checks/` has no oracle file; `dbscan_check.mojo` compares
against host scans and a Float64 adjacency inside a context (`:263-291`,
`:541-586`). No RNG, no transcendental (squared distances against
`dbscan_metric_threshold` `dbscan/impl/neighbors/epsilon_neighborhood.mojo:210`).
(c) nothing of the fit. Rows read, `K_LIB_EPS_NEIGHBORHOOD`,
`K_LIB_WEIGHTED_VERTEX_DEG` (`dbscan/impl/vertexdeg/algo.mojo:154-158`).
(d) `labels_`, `n_iter_` (`density.py:279-281`). No save or load.
(e) 16 hours. Main risk, the label propagation's fixed point is
order-independent (a minimum over a connected component) but `n_iter_` is
the device's pass count, which a serial union-find will not reproduce; the
train column hashes `labels_` only, so the host may take a different
algorithm and must still report `n_iter_` by replaying the device's
synchronous passes.

**pca, tsvd.**
(a) `pca_fit_binding` `bindings/_mojolearn_estimators.mojo:118`,
`tsvd_fit_binding` `:304`, `pca_fit_host` `decomposition/estimator.mojo:53`,
`tsvd_fit_host` `:219`. The covariance or Gram is `gemm_tn` `core/gemm.mojo:205`
(split-K under 128 features, `core/gram_splitk.mojo:942`, else
`identical_gemm_into`), the eigensolve is one-block cyclic Jacobi
`decomposition/checks/jacobi_eigh_device.mojo:243` at 15 sweeps and 1e-7,
then `sign_flip_kernel` `decomposition/impl/linalg/detail/pca.mojo:143`, and
the truncation is already host Float64 (`order_truncate_spectrum` `:203`).
(b) STAGE, and the stage is explicitly disclaimed. `jacobi_eigh`
`decomposition/checks/jacobi_eigh.mojo:44` is Float64 at 1e-12 and 60 sweeps,
and its header at `:5-10` reads "THIS IS THE ORACLE, NOT A PATH ANY FIT
TAKES ... the two are not expected to agree bit for bit". The Gram has a
host answer, `gemm_oracle` `gemm/checks/gemm_oracle.mojo:514` at `OP_TN`, but
the split-K Gram is gated per cell against a Float64 tolerance, not bits
(`checks/gram_splitk_check.mojo:3, 47`), so the host must restate the
split-K fold, not call `gemm_oracle`. The centering kernels
(`core/column_stats.mojo:104, 153, 324`) have no host twin.
(c) stage. Rows read, `lib_block_size_for[K_LIB_JACOBI_EIGH]`
(`jacobi_eigh_device.mojo:27`), `lib_block_size_for[K_LIB_GRAM_SPLITK]`
(`core/gram_splitk.mojo:174`).
(d) `components_`, `mean_`, `explained_variance_`, `singular_values_`
(`decomposition.py:295-336`, `:449-461`). No save or load.
(e) 10 hours pca, 6 hours tsvd. Main risk, a Float32 host Jacobi at the
device's settings must reproduce the device rotation order and the
cross-lane folds `_folded_and_broadcast` `jacobi_eigh_device.mojo:77` exactly,
and `sqrt` in `order_truncate_spectrum` is the stdlib Float64 `sqrt`
(`pca.mojo:6, 241`), fine on a CPU since Float64 `sqrt` is correctly rounded
everywhere, but it is not an `identical_*` seam and must stay that way.

**ols, ridge, logistic.**
(a) `ols_fit_binding` `bindings/_mojolearn_estimators.mojo:372`,
`ridge_fit_binding` `:412`, `qn_fit_binding` `:434`; `ols_fit_host`
`glm/estimator.mojo:77` then `lstsq_eig_traced`
`glm/impl/linalg/detail/lstsq.mojo:269` (Gram by `gemm_tn`, Jacobi
eigendecomposition, pseudo-inverse with cutoff `:173`, no Cholesky);
`ridge_fit_host` `:242` then `ridge_eig_traced` `glm/impl/ridge.mojo:189`
(`svd_eig_traced` `glm/impl/linalg/detail/svd.mojo:89` plus six elementwise
kernels `ridge.mojo:119-157`); `qn_fit_host` `:284` then the L-BFGS and
OWL-QN solvers `glm/impl/qn/qn_solvers.mojo`, line search
`qn_linesearch.mojo`, loss kernel `logistic_loss_dz_kernel`
`glm/impl/qn/glm_logistic.mojo:72`.
(b) NONE for all three. `glm/checks/` has no oracle file; every check is
device-against-host-transfer (`check_ridge_device_equals_host`
`glm/checks/ridge_check.mojo:389`, `check_logistic_device_equals_host`
`glm/checks/logistic_check.mojo:532`) and `_host_fit_coefs`
`glm/checks/ols_check.mojo:1082` calls `ols_fit_host`, the device path. The
transcendental seam is already portable, `identical_exp` and
`identical_log` at `glm/impl/qn/glm_logistic.mojo:43, 57-58, 66`,
`identical_exp64` at `glm/estimator.mojo:50, 384`.
(c) nothing. Rows read transitively through the Jacobi and Gram kernels.
(d) `coef_`, `intercept_`, `_x_mean`, `_y_mean` (`linear_model.py:543-557`,
`:655-691`); logistic adds `_w`, `n_iter_`, `objective_` (`:922-932`). No
save or load.
(e) 12 hours ols, 8 hours ridge (both reuse the phase-2 host Jacobi and Gram
from pca), 24 hours logistic. Main risk for logistic, a whole quasi-Newton
loop with line search whose every GEMV is a `gemm.fp32.v1` dot, so the host
must fold each dot at the contract leaf size (`contract_leaf_size`
`gemm/checks/gemm_oracle.mojo:117`), and any Float64 accumulator in the
solver's host glue must stay where the device has it.

**lasso, elasticnet.**
(a) `cd_fit_binding` `bindings/_mojolearn_solver.mojo:56`, `cd_fit_host`
`solver/estimator.mojo:80`, `cd_fit_traced` `solver/impl/cd.mojo:297`;
coordinate descent whose per-coordinate dots are `gemv_n` at the contract
leaf size (`cd.mojo:88-101`), three small kernels `:172, 182, 249`.
(b) WHOLE FIT. `cd_oracle_fit` `solver/checks/cd_oracle.mojo:288`, header
`:1-31`, "cdFit on the host, stage for stage ... every row-length reduction
is gemm_oracle_cell at the contract's leaf size, IMPORTED from the gemm lane,
so the fold tree is the device's by construction". No RNG (the default is
cyclic).
(c) whole fit at the shipped default. No kernel-matrix row is read in
`solver/`.
(d) `coef_`, `intercept_`, `n_iter_` (`_solver_impl.py:341-357`). No save or
load.
(e) 8 hours for both. Main risk, `cd_oracle.mojo:85` imports
`solver/checks/profile_dot.mojo`, whose `:43` imports `max.gpu.host`; the
byte LM probe says such an import compiles host-only, and if it does not the
two pure host functions `profile_dot_host :95` and `serial_dot_host :101`
move to an import-only module.

**svc.**
(a) `svc_fit_binding` `bindings/_mojolearn_svm.mojo:96`,
`svc_fit_host_borrowed` `svm/estimator.mojo:215`, `_svc_fit_staged`
`svm/impl/svc_impl.mojo:228`; kernel matrix through `identical_gemm_into`
and `rbf_fused_tile_kernel` `svm/impl/distance/kernel_matrices.mojo:183`, SMO
in `smo_block_solve_kernel` `svm/impl/smoblocksolve.mojo:127`.
(b) WHOLE FIT. `smo_oracle_fit` `svm/checks/smo_oracle.mojo:535`, header
`:3-11`, "the SAME SMO, serial ... one thread, one loop, ascending, through
ftz, identical_mul_add, identical_exp and gemm_oracle_cell"; the decision
function is `smo_oracle_decision :814`, so the infer column has its host
twin too. No GPU import (printed below). The RBF exponential is
`identical_exp` on both sides (`kernel_matrices.mojo:20-31`).
(c) whole fit. Rows read, `svm_block_solve_schedule_for` and
`svm_block_solve_tree_arity_for` (`svm/impl/smoblocksolve.mojo:157, 165`),
scheduling of a fold whose order the oracle spells serially.
(d) `support_`, `support_vectors_`, `dual_coef_`, `intercept_`, `n_iter_`
(`_svm_impl.py:529-543`). No save or load.
(e) 8 hours. Main risk, the working-set selection order across SMO
iterations (the oracle claims it, the gate is
`svm/checks/svc_check` style arms, and the lane's `max_iter=200` fixture
must converge the same way on both sides).

**kde.**
(a) fit stores `_x` (`density.py:433`); `kde_score_samples_binding`
`bindings/_mojolearn_estimators.mojo:392`, `kde_score_samples_host`
`kde/estimator.mojo:70`, context at `:110`; fused
`kde_fused_logsumexp_kernel` `kde/impl/neighbors/kernel_density.mojo:1112`.
(b) WHOLE. `oracle_score_samples` `kde/checks/kde_oracle.mojo:256`, serial
Float32 through `identical_mul_add`, `ftz`, `identical_exp`, `identical_log`,
`identical_sqrt`, with the row-norm halving fold `_host_row_norm_halving :94`
and the score fold `oracle_logsumexp_row :224`. No GPU import (printed below).
(c) whole score path. No kernel-matrix row is read in `kde/`.
(d) `_x`, `_w`, `n_samples_fit_`. No save or load.
(e) 4 hours. Main risk, none beyond the normalization constant, whose host
twin already exists (`log_kernel_norm_identical`
`kde/impl/neighbors/kernel_density.mojo:697`).

**agglomerative.**
(a) `linkage_fit_binding` `bindings/_mojolearn_solver.mojo:152`,
`linkage_fit_host` `hierarchy/estimator.mojo:97`, `single_linkage`
`hierarchy/impl/cluster/detail/single_linkage.mojo:106`; pairwise distances
through `pinned_distance_tile_kernel`, Boruvka MST
`hierarchy/impl/sparse/solver/mst_solver.mojo:457`, dendrogram on the host
`build_dendrogram_host` `hierarchy/impl/cluster/detail/agglomerative.mojo:119`.
(b) WHOLE FIT by a different MST algorithm. `oracle_linkage`
`hierarchy/checks/linkage_oracle.mojo:593` runs `host_pinned_distance_matrix
:260`, `host_kruskal :335` (Kruskal under the same total order, deliberately
not Boruvka), `host_dendrogram :467`, `host_extract_flattened_clusters :482`.
The file imports `max.gpu.host` at `:53` for `build_fixture :184` only
(printed below).
(c) whole fit. No kernel-matrix row is read in `hierarchy/`.
(d) `labels_`, `children_`, `n_boruvka_rounds_` (`_hierarchy_impl.py:299-306`).
No save or load.
(e) 8 hours. Main risk, `n_boruvka_rounds_` is a device pass count the
Kruskal host cannot produce (the train column hashes `labels_` only, so the
lane passes; the attribute would have to be reported from a Boruvka replay
or documented as device-only), and equal edge weights, where Kruskal and
Boruvka pick the same edge only because the total order carries the index
(`linkage_oracle.mojo:11-20`).

**spectral.**
(a) `spectral_fit_predict_dataset_binding` `bindings/_mojolearn_metrics.mojo:605`,
`spectral_fit_predict_dataset_host` `spectral/estimator.mojo:130`; the kNN
affinity graph `spectral/impl/preprocessing/detail/spectral_embedding.mojo:134`,
the normalized Laplacian `spectral/impl/sparse/linalg/detail/laplacian.mojo:224`,
Lanczos `spectral/impl/sparse/solver/detail/lanczos.mojo:471, 662` with the
projected solve already on the host (`symmetric_eig_host`
`spectral/checks/symmetric_eig_host.mojo:197`, "part of the numerical plan"
`:12-15`), then k-means on the embedding
`spectral/impl/cluster/detail/spectral.mojo:121-136` with `params.seed`.
(b) STAGE, a large one. `oracle_embedding` `spectral/checks/spectral_oracle.mojo:518`
covers Laplacian, Lanczos and the embedding (no GPU import, printed below);
the Lanczos start vector `lanczos_v0` `lanczos.mojo:920` is a host `def`.
Missing on the host, the affinity graph (the kNN of section knn) and the
final k-means (the kmeans lane).
(c) stage. No kernel-matrix row is read in `spectral/`.
(d) `labels_`, `embedding_` (`_spectral_impl.py:512-525`). No save or load.
(e) 12 hours after kmeans and knn land. Main risk, inherited from kmeans.

**holtwinters.**
(a) `holtwinters_fit_binding` `bindings/_mojolearn_tsa.mojo:65`,
`holtwinters_fit_ptr` `holtwinters/estimator.mojo:279`; STL decompose
`holtwinters/impl/internal/hw_decompose.mojo:81-286`, BFGS
`hw_optim.mojo:582`, eval `hw_eval.mojo:223`, forecast `hw_forecast.mojo:27`.
(b) WHOLE FIT. `oracle_fit[dt]` `holtwinters/checks/hw_oracle.mojo:307`,
"HoltWintersFitHelper on the host: transpose, decompose, BFGS, final eval"
(`:320-321`), plus `oracle_forecast :566`; the Float32 arm is the device's
bits under IDENTICAL (`:6-14`). No GPU import (printed below). The only
transcendental is `identical_sqrt` (`hw_optim.mojo:24, 238`).
(c) whole fit. No kernel-matrix row is read.
(d) `level_`, `trend_`, `season_`, `sse_`, `alpha_`, `beta_`, `gamma_`
(`_tsa_impl.py:520-540`). No save or load.
(e) 6 hours. Main risk, none identified beyond the binding.

**gemm-pinned.**
(a) `gemm_binding` `bindings/_mojolearn_linalg.mojo:115`, `identical_gemm_host`
`gemm/host_entry.mojo:55` ("no arithmetic in this file", `:20-26`),
`identical_gemm_into` `gemm/checks/gemm_identical.mojo:2641`.
(b) WHOLE, and it is the definition. `gemm_oracle` `gemm/checks/gemm_oracle.mojo:514`
is "The NORMATIVE answer of profile mojolearn.identical.gemm.fp32.v1" (`:34-37`),
every shape and all three ops, the partition count a parameter so the
split-K path is the same oracle (`:42-46`), "NO DEVICE KERNEL LIVES HERE,
DELIBERATELY ... A host-only oracle also runs with no GPU present" (`:66-71`).
The group fold of the shipped ksplit dispatch is proven the contract tree on
the host by `check_stack_fold_is_the_contract_tree`
`gemm/checks/gemm_device_check.mojo:290` (every P in 1 to 2049) and
`check_group_fold_is_the_contract_tree` `gemm/checks/gemm_step_arms_check.mojo:396`.
Caveat at `:52-62`, under FAST the helpers compile away and the oracle is
not the contract; a host build is IDENTICAL only, which the pattern already
enforces (`build_byte_lm_host.sh:10`).
(c) whole op. Rows read on the device side only (`gemm_identical.mojo:148-163`).
(d) stateless.
(e) 4 hours. Main risk, none; `gemm_oracle` already compiles host-only inside
the byte LM binding (`training/byte_lm_host_backward.mojo:55`).

**metrics.**
(a) `accuracy_score_binding` `bindings/_mojolearn_metrics.mojo:114`,
`adjusted_rand_score_binding :158`, `v_measure_score_binding :284`,
`r2_score_binding :317`, `silhouette_binding :491`; hosts in
`metrics/estimator.mojo:106, 147, 263, 291, 337`, each opening a context.
(b) silhouette WHOLE, `_oracle_silhouette` `metrics/checks/silhouette_check.mojo:129`
over `host_l2sqrt_unexpanded` `metrics/checks/pinned_distance.mojo:61` (no
GPU import) and `host_tree_sum` `metrics/checks/pinned_sum.mojo:138` (that
file imports `std.gpu` at `:57-61` for its kernel half, printed below). The
other four have NONE; `metrics/checks/{classification,label,regression}_metrics_check.mojo`
hold inline host expressions inside a context. Accuracy is an integer count,
ARI and v-measure are integer contingency plus `identical_log`, r2 is
`sse_ssto_chunks_kernel` `metrics/impl/stats/detail/scores.mojo:203` whose
fold `host_tree_sum` already has the shape of.
(c) one of five metrics. No kernel-matrix row is read.
(d) functions, no state.
(e) 10 hours. Main risk, the lane also fits a `KMeans` for its labels
(`identity_break.py:423`), so the metrics cell cannot pass before kmeans does.

**iforest.**
(a) `iforest_run_binding` `bindings/_mojolearn_svm.mojo:423`, `iforest_run_host`
`isolation_forest/estimator.mojo:336` (despite its name it creates the
context at `:397`), `IsolationForest.fit` `isolation_forest/impl/isolation_forest.mojo:551`,
one fit kernel `build_isolation_trees_global_kernel`
`isolation_forest/impl/isolation_tree_builder.mojo:599`, and the scoring
kernels re-run the fit on every call (DEVIATION 874, `_iforest_impl.py:339-340,
369-372`).
(b) WHOLE FIT AND SCORE. `oracle_fit` `isolation_forest/checks/if_oracle.mojo:239`,
`oracle_path_lengths :337`, `oracle_scores :356`, "a SECOND, independent
transcription of cuML's isolation tree builder and scorer, serial Float32
through the same identical_* helpers and the same XORWOW implementation"
(`:3-30`), consumption order at `:12-17`. No GPU import (printed below). The
RNG `isolation_forest/impl/rng/xorwow.mojo` has no GPU import and the fit
already runs 16 host draws as its trace probe (`isolation_forest.mojo:648-660`).
(c) whole. No kernel-matrix row is read.
(d) `_x` (the training matrix, kept), `offset_`, `max_samples_`
(`_iforest_impl.py:363-381`); the forest never reaches Python and there is no
save or load, so the model column stays `n/a:no-save` on the CPU column too.
(e) 12 hours. Main risk, the oracle's in-memory shape `OracleForest`
`if_oracle.mojo:87` differs from `IsolationForestModel`
`isolation_forest.mojo:145`, so the binding needs an adapter producing the
`IFRunOutputs` the Python side reads, and the fit-on-every-score design must
be kept or the identity surface changes.

### 1.2 The printed greps behind the absence claims

The 17 lanes with a "none" or "stage" verdict rest on the sweeps' greps over
their package directories (`ensemble extratrees isolation_forest` 432 lines,
`gbdt` 455, `cluster neighbors dbscan kde hierarchy spectral metrics` 736,
`decomposition glm solver svm holtwinters tsa gemm cholesky` 679, exit 0 on
each), and on `find` returning no `mojo_only`, `host` or `reference`
subdirectory in any package. The whole-fit files' status, run on `0dcc1204e`
and printed in full:

```
$ git grep -n -E "from gpu|import gpu|max\.gpu|DeviceContext|has_accelerator|std\.gpu" -- \
    isolation_forest/checks/if_oracle.mojo solver/checks/cd_oracle.mojo \
    holtwinters/checks/hw_oracle.mojo svm/checks/smo_oracle.mojo kde/checks/kde_oracle.mojo \
    hierarchy/checks/linkage_oracle.mojo spectral/checks/spectral_oracle.mojo \
    spectral/checks/symmetric_eig_host.mojo metrics/checks/pinned_distance.mojo \
    metrics/checks/pinned_sum.mojo neighbors/checks/metric_oracle.mojo \
    decomposition/checks/jacobi_eigh.mojo extratrees/checks/pcg_rng.mojo \
    isolation_forest/impl/rng/xorwow.mojo core/philox.mojo core/shuffle_iterator.mojo \
    gbdt/data/permutation.mojo solver/checks/profile_dot.mojo \
    extratrees/impl/randomforest/randomforest.mojo \
    extratrees/impl/decisiontree/batched_levelalgo/builder.mojo
core/philox.mojo:6:from std.gpu import block_dim, block_idx, thread_idx
core/philox.mojo:8:from max.gpu.host import DeviceBuffer, DeviceContext
core/philox.mojo:267:    ctx: DeviceContext,
core/philox.mojo:318:    ctx: DeviceContext,
extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:94:from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:95:from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, thread_idx
(... 23 more `ctx: DeviceContext` signatures in that file, device-arm entries ...)
extratrees/impl/randomforest/randomforest.mojo:100:from max.gpu.host import DeviceContext
extratrees/impl/randomforest/randomforest.mojo:328:    ctx: DeviceContext,
extratrees/impl/randomforest/randomforest.mojo:466:    ctx: DeviceContext,
hierarchy/checks/linkage_oracle.mojo:53:from max.gpu.host import DeviceContext, HostBuffer
hierarchy/checks/linkage_oracle.mojo:184:def build_fixture(ctx: DeviceContext, fix: Int) raises -> HostBuffer[DType.float32]:
metrics/checks/pinned_sum.mojo:57:from std.gpu import block_dim, block_idx, grid_dim, thread_idx
metrics/checks/pinned_sum.mojo:59:from max.gpu.memory import AddressSpace
metrics/checks/pinned_sum.mojo:60:from max.gpu.primitives.block import sum as block_sum
metrics/checks/pinned_sum.mojo:61:from max.gpu.sync import barrier
solver/checks/profile_dot.mojo:43:from max.gpu.host import DeviceBuffer, DeviceContext
solver/checks/profile_dot.mojo:72:    ctx: DeviceContext,
```

Every other file in that list is absent from the output, which is the
printed form of "no GPU import". The same grep over the byte LM host files
returns only prose (`bindings/_mojolearn_byte_lm_host.mojo:4`,
`training/byte_lm_host.mojo:27`), and over `checks/numerics.mojo` and
`checks/kernel_matrix.mojo` nothing at all; both are already inside the
certified CPU binding.

### 1.3 Tally

| verdict | lanes | count |
|---|---|---|
| whole-fit host oracle exists | et-clf, et-reg (leaf mean restated), lasso, elasticnet, svc, kde, agglomerative, holtwinters, gemm-pinned, iforest | 10 |
| stage only | knn, knn-clf, knn-reg, pca, tsvd, ols, ridge, logistic, spectral, metrics, kmeans (planned scale and RNG only) | 11 |
| nothing | rf-clf, rf-reg, gbdt x4, dbscan | 7 |

Save and load exist on eight lanes only (RF, ET, the four GBDT), all through
`_serialize.write_npz` `python/mojolearn/_serialize.py:267`; every other lane
records `n/a:no-save` on every column and will on the CPU column too.

## 2. The kernel matrix has no CPU column

`checks/kernel_matrix.mojo:20-31` defines eight columns, `COLUMN_BIT_IDENTICAL
= 0`, `COLUMN_APPLE`, `COLUMN_NVIDIA`, `COLUMN_AMD`, `COLUMN_AMD_RDNA`,
`COLUMN_QUALCOMM`, `COLUMN_INTEL`, `COLUMN_SPEC_BASELINE`, `COLUMN_COUNT = 8`.
`grep -n COLUMN_ checks/kernel_matrix.mojo` prints no `COLUMN_CPU` or
`COLUMN_HOST`. `TARGET_COLUMN` at `:348-360` takes the `-D MOJOLEARN_COLUMN_*`
define first, then the accelerator probes, and its final fallthrough is
`COLUMN_APPLE`. **A host build with no accelerator target therefore compiles
as the Apple column today.** The byte LM CPU binding has run that way on
seven runners and matched the GPU bits, which shows the fallthrough is
bit-inert for the rows the byte LM reaches, not that it is harmless in
general.

Which rows still branch on the column under IDENTICAL. Most numeric rows
resolve to the pinned reading regardless of column (`block_size_for :399`
takes `IDENTITY_FLOOR_BLOCK`, `lane_width_for :418` returns
`PINNED_REPLICATION_LANES`, `reduce_width_for :434`, `deterministic_flush_for
:466` is true, `hist_smem_mode_for :578`, `partition_chunks_sm_for :598`,
`lib_block_size_for :770` resolves float folds to `COLUMN_BIT_IDENTICAL`). The
ones that do not, and what a CPU column should answer:

| row | file:line | answer per column today | COLUMN_CPU should answer |
|---|---|---|---|
| `lib_hardware_ftz_fma_for` | `:814-824` | NVIDIA true, else false (the `mul.rn.ftz` fold flush, DEVIATION 2706) | false, the software spelling |
| `lib_gemm_kernel_body_for` | `:846` | NVIDIA 1, else 0 | 0 |
| `lib_gemm_block_parallelism_for` | `:834` | NVIDIA 132, AMD 110, else 0 | 0, the host oracle folds the contract tree without groups |
| `knn_distance_zero_fma_repair_for` | `:1210-1220` | identical and Apple | false, a CPU FMA rounds once and does not pre-round underflow |
| `knn_distance_preflight_for`, `_metadata_for` | `:1222-1245` | identical and Apple | false |
| `knn_distance_hardware_flush_for` | `:1248` | identical and NVIDIA | false |
| `knn_distance_rows_for` | `:1256-1264` | 8 on NVIDIA identical, else 4 | 4, scheduling |
| `_knn_identical_round_column` | `:1044-1052` | the four buildable GPU columns | true, so the 2026-09-09 IDENTICAL defaults apply on the host too |
| `quantize_search_for` | `:1037` | Apple two-level, else linear | linear, scheduling |
| `column_compares_flush_subnormals` | `:206` | Apple | false, `ftz` is explicit under IDENTICAL |
| `column_lane_width`, `_is_fixed` | `:257, 273` | 32 or 64 | 1 and true, which makes `sub_byte_lane_sync_for :457` answer `SYNC_BLOCK`, irrelevant on a host with no barrier |
| `column_shared_limit` | `:173` | 32 KB to 64 KB | unlimited, or the AMD reading; it sizes device pages only |
| `column_is_buildable` | `:62` | the five compute columns | true, the host compiles today |

What a `COLUMN_CPU` needs. `comptime COLUMN_CPU = 8`, `COLUMN_COUNT = 9`, a
`column_name` entry "cpu" (`:42`), the rows above, `TARGET_COLUMN` reading
`-D MOJOLEARN_COLUMN_CPU` before the accelerator probes, and `DETECTED_COLUMN`
`:364-368` answering `COLUMN_CPU` when `has_accelerator()` is false instead
of `COLUMN_APPLE`, so `column_is_simulated :372` stops calling a host build a
simulated Apple build. Every `build_*_host.sh` passes `-D MOJOLEARN_COLUMN_CPU`
(`build.sh:302-303` already turns `MOJOLEARN_TARGET_COLUMN` into that define
for the GPU scripts), and every host binding carries `comptime assert
TARGET_COLUMN == COLUMN_CPU` so an accidental Apple fallthrough is a build
error rather than a matched hash by luck. The byte LM and forest host
bindings would take the same assert once the column exists. 111 Mojo files
read `TARGET_COLUMN` (`git grep -ln TARGET_COLUMN -- '*.mojo' | wc -l`), so
the change is one file plus the asserts, and the risk is a row that lists
the buildable columns by name and would exclude the new one (the two
`column == COLUMN_APPLE or ... or COLUMN_AMD_RDNA` lists at `:520-531` and
`:566-577` are FAST-only and never reached under IDENTICAL).

## 3. The fourth column

### 3.1 How identity_break behaves on a GPU-less box today

`import mojolearn` on a box with no device node and no driver library raises
at `_backend.py:701-712` ("NO SUPPORTED GPU FOUND ON THIS BOX, and there is
no CPU path in this package"), unless `host_binding_built()` `:802` is true,
in which case `_select_cpu_only` `:826` installs a `_NoGpuBinding` stub for
every name in `_MODULES`, sets `_SELECTED = mode`, and `vendor()` `:1319`
answers `'cpu'`. On that install `tools/identity_break.py::run` passes its
mode check (`numeric_mode()` `:1240-1298` reads `default_mode()`, which is
`_SELECTED`, and the `hasattr` probe on the stub raises `ImportError`, which
`:1289-1291` catches), and then every lane's first binding call raises by
name from `:813-823`, so all 252 cells read REFUSED with the stub's message.
Nothing is hashed, which is the right baseline. The forest lane widened
`host_binding_built()` to the forest host `.so`
(`lane/forest-host-inference` `_backend.py:802-812`) and added a separate
class `HostForest`; identity_break calls `ml.RandomForestClassifier`, so a
separate class does not reach the lanes.

### 3.2 The host binding set

One `.so` per family under `python/mojolearn/host/`, named
`_mojolearn_<family>_host.so`, built by `bindings/build_<family>_host.sh`
flag for flag with `build_byte_lm_host.sh` (IDENTICAL only, `--target-cpu
apple-m1` or `x86-64-v3`, no `MOJOLEARN_GPU_ARCHS`, `-D MOJOLEARN_COLUMN_CPU`,
`MOJOLEARN_BUILD_EXTRA_DEFINES` for the sabotage arm). Each exports the same
function names the GPU binding exports for the fits it covers, plus
`<prefix>_vendor()` answering `cpu`, `<prefix>_numeric_mode()` answering 1,
and `<prefix>_sabotage()`. The host helpers `bindings/host_helpers.mojo` from
the forest lane (`all_finite_f32`, `argmax_rows_f32`, `gather_i64`, ...) are
shared.

`_backend.py` grows `_HOST_MODULES`, a map from `_MODULES` name to host `.so`
basename, and `binding(name, mode)` on a CPU-only install returns the host
module when it is built and the `_NoGpuBinding` stub otherwise. Two rules
keep this honest. The host set loads only when `_CPU_ONLY is not None`, so a
box with a GPU never serves host arithmetic under a GPU label (the vendor
read-back cross-check at `_check_vendor` applies to the host set with `cpu`
as its expected string). And a family whose host binding lacks a function
raises by name at that function, so a lane with no host fit still reads
REFUSED, never a hash of something else. The Python estimator classes need
no change where the host binding exports the same names with the same
address contract; the exceptions are ET (the binding refuses `device != 1`
at `bindings/_mojolearn_trees.mojo:330`, so the host binding drops that
check) and iforest (the adapter in section 1).

### 3.3 What the CPU column records

`identity_break.py` records `vendor` from `--vendor`, whose default is
`platform.machine()` (`:798`), and `commit` from `MOJOLEARN_COMMIT` (`:540`).
The three GPU columns committed on 2026-09-13 show the weakness of a typed
value; `bench/results/identity_break/2026-09-13_three-columns/nvidia-h100-sm_90a.json`
carries `"vendor": "box-arch"`, a literal the leg script failed to expand,
and all three carry `"commit": ""`. The CPU column must derive its label.

`--vendor cpu-<cpu>` where `<cpu>` is the CPU model string from
`/proc/cpuinfo` `model name`, `lscpu` on ARM, or `sysctl machdep.cpu.brand_string`
on macOS, lowercased and hyphenated (the byte LM workflow already extracts
it, `.github/workflows/byte-lm-cpu-gate.yml:139-160`), and the JSON gains a
`host` object beside `vendor`, written by `run` when `ml.vendor() == 'cpu'`:

| key | source |
|---|---|
| `cpu_model` | the string above |
| `arch` | `platform.machine()` |
| `target_cpu` | the `--target-cpu` the build used (`x86-64-v3`, `apple-m1`, or `none` on aarch64), read back from the binding as `<prefix>_target_cpu()` |
| `column` | `<prefix>_column()` answering "cpu", the comptime assert's witness |
| `mojo_version` | `pixi run mojo --version`, recorded by the workflow into the JSON via `MOJOLEARN_MOJO_VERSION` |
| `commit` | `MOJOLEARN_COMMIT`, required non-empty when `vendor` starts with `cpu-`; `run` refuses otherwise |
| `families` | the host bindings found built, so a REFUSED cell is attributable to an unbuilt family rather than a bug |

The three columns are diffed with `--diff apple-m4.json nvidia-h100-sm_90a.json
amd-mi325x-gfx942.json cpu-<model>.json`. `_diff` already skips a REFUSED
column (`:729`), so an uncovered lane reads `IDENTICAL x3` and a covered
lane `IDENTICAL x4`; the gate must require `x4` on the covered list by name,
because `x3` on a lane that should be covered is the CPU binding refusing,
not a pass. A `--require-columns 4 --lanes <list>` flag on `--diff`, exiting
non-zero when any named lane has fewer than four real hashes, is the whole
addition. The infer and model columns diff the same way; the model column
is meaningful on the CPU for ET only in phase 1 (RF and GBDT are phase 2b),
and its RELOAD check runs through the forest host predict from
`lane/forest-host-inference`.

### 3.4 The workflow on the seven free runners

`.github/workflows/identity-cpu-column.yml`, the byte LM gate's matrix
(`byte-lm-cpu-gate.yml:89-110`, ARM64 Cobalt 100, macOS M1, five x86-64
draws on two images, `fail-fast: false`), sparse checkout of
`bench/results/identity_break/2026-09-13_three-columns/` only. Per runner,
in order. Runner facts to `runner.txt`. Build the host set for the families
of the current phase. `MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT=$GITHUB_SHA
nice -n 19 python3 tools/identity_break.py --lanes <covered> --json
cpu-<slot>.json --vendor cpu-<model>` under `timeout` (the tool writes the
JSON after every lane, `:613`). `--diff` against the three committed GPU
columns with `--require-columns 4`. Build the sabotage set with
`MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1"` (one define,
honored in `gemm_oracle` by reversing the leaf fold, which reaches every
phase-1 lane except ET and iforest, and in `pcg_rng.mojo` and `xorwow.mojo`
by advancing one extra draw, which reaches those two), run the same lanes,
and require `--diff` to exit non-zero with DIVERGENT on every covered lane.
Upload `cpu-gate-out/` as the artifact. A green job enters the brief's
certified table only after its uploaded JSON is read and its `host.cpu_model`
names the CPU, the byte LM gate's rule.

Cost. The fixtures are 20,000 by 16 float32 (`identity_break.py:104`), nine
of them, two fits per cell. The serial host oracles were written for
correctness, not speed; `smo_oracle_fit` on 2,000 rows and `oracle_fit` for
Holt-Winters on 512 points are seconds, the ET reference at 16 trees depth 8
on 20,000 rows and `oracle_fit` for iforest at 16 trees are the ones to
time first. UNMEASURED. The runner limit is 60 minutes; if a lane does not
fit, `--fixtures` narrows the column and the README says so.

## 4. The plan

Ordered by evidence value over hours. Evidence value is cells closed on the
fourth column times how much of the public claim they carry; the forests and
GBDT carry the most (they are the lanes with a model column and the product's
lead) and cost the most, so they sit after the cheap whole-fit lanes and
before the estimators outside identity_break.

### Phase 0, the pattern, 16 hours

`COLUMN_CPU` in the kernel matrix with the asserts (section 2), 4 hours.
`_HOST_MODULES` and the CPU-only routing in `_backend.py`, the `host` object
and `--require-columns` in `identity_break.py`, 6 hours. The workflow, the
sabotage define, the README under `bench/results/identity_break/`, 6 hours.
The forest host inference lane is a prerequisite for the ET model column and
should be merged first.

### Phase 1, lanes whose host oracle covers the whole fit, 72 hours

| lane | host routine | hours |
|---|---|---|
| gemm-pinned | `gemm_oracle` `gemm/checks/gemm_oracle.mojo:514` | 4 |
| kde | `oracle_score_samples` `kde/checks/kde_oracle.mojo:256` | 4 |
| holtwinters | `oracle_fit`, `oracle_forecast` `holtwinters/checks/hw_oracle.mojo:307, 566` | 6 |
| lasso, elasticnet | `cd_oracle_fit` `solver/checks/cd_oracle.mojo:288` | 8 |
| svc | `smo_oracle_fit`, `smo_oracle_decision` `svm/checks/smo_oracle.mojo:535, 814` | 8 |
| agglomerative | `oracle_linkage` `hierarchy/checks/linkage_oracle.mojo:593` | 8 |
| et-clf | `fit_extra_trees_classifier_reference` `extratrees/estimator.mojo:571` | 10 |
| et-reg | `fit_extra_trees_regressor_reference` `:691` plus quantized leaf means | 12 |
| iforest | `oracle_fit`, `oracle_scores` `isolation_forest/checks/if_oracle.mojo:239, 356` | 12 |

Ten lanes, 90 of 252 train cells, 72 infer cells, 18 model cells (ET). The
order inside the phase is by hours; gemm-pinned first because it is also the
sabotage carrier and proves the pattern end to end in an afternoon.

### Phase 2a, lanes needing a restated fold, 134 hours

| lane | what is restated | hours |
|---|---|---|
| pca, tsvd | Float32 host Jacobi at the device settings, column stats, split-K Gram fold | 16 |
| ols, ridge | the same Jacobi and Gram, `xty`, the pseudo-inverse cutoff, ridge's six elementwise kernels | 20 |
| knn, knn-clf, knn-reg | selection and the vote and average folds over `metric_oracle` | 12 |
| kmeans | k-means++ over `HostRng`, the Lloyd loop, the fixed-point center fold, the tie rule | 24 |
| metrics | accuracy, ARI, v-measure, r2 host folds beside `_oracle_silhouette`; needs kmeans | 10 |
| spectral | the affinity kNN and the final k-means around `oracle_embedding`; needs kmeans and knn | 12 |
| dbscan | eps neighborhood, core points, propagation with the device pass count | 16 |
| logistic | a host L-BFGS and OWL-QN with line search over contract-leaf dots | 24 |

Eleven lanes, 99 train cells.

### Phase 2b, new host trainers, 240 hours

| lane | what is new | hours |
|---|---|---|
| rf-clf, rf-reg | the cuML batched-level builder on the host, quantiles through leaves, over the existing host RNG | 80 |
| gbdt-symmetric, gbdt-rmse | binarize to cindex, target planes, histograms in the pinned fold order, scan and subtract, split scoring, partition, leaf estimation, model text | 120 |
| gbdt-depthwise, gbdt-lossguide | the non-symmetric driver over the same stages, Lossguide selection is already host code | 40 |

Seven lanes, 63 train cells, the 54 remaining model cells. Phase 2b closes
the column; it is last because its hours dwarf the rest and its evidence
already has three GPU witnesses, so the CPU witness adds one vendor to a
claim that is not in doubt, while phase 1 adds a CPU witness to ten lanes for
a tenth of the cost.

Totals. Phase 0 16, phase 1 72, phase 2a 134, phase 2b 240, in all 462 hours
for the 28 lanes, before the risks in section 5 land.

### Phase 3, estimators not in identity_break

One line each, GPU identity card and host oracle status, from the sweep over
`bench/results/`, `IDENTITY_PATHS.md` and `docs/CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md`.
"no card found" means the grep for the name under `bench/results/**/*.card`
printed nothing. Each of these also needs a lane in `identity_break.py`
before a CPU column can say anything about it.

| estimator | GPU identity card | host oracle | hours to a CPU fit |
|---|---|---|---|
| SVR (`_svm_impl.py:643`) | no card found, only the SVC card `bench/results/e1/2026-08-24_184912-*/lanes/svm.identical.card`; audit `docs/CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md:90` | yes, `smo_oracle_fit` solves EPSILON_SVR through the same loop (`svm/checks/smo_oracle.mojo:40-49`) | 4 after svc |
| ARIMA (`_arima_impl.py:174`) | yes, Apple and AMD cards `bench/results/e1/2026-08-28_*/lanes/arima.identical.card`, three-vendor claim at `IDENTITY_PATHS.md:361` (its cited `CERT_2026-08-31.md` is not in this worktree) | yes, `arima/checks/kalman_oracle.mojo`, `fit_oracle.mojo`, no GPU import | 12 |
| GaussianProcessRegressor (`_gp_impl.py:213`) | Apple and AMD cards `gp.identical.card`, no NVIDIA (audit `:95`) | yes, `gaussian_process/checks/gp_oracle.mojo:69-91` over the Cholesky and GEMM oracles | 8 |
| UMAP (`_umap_impl.py:42`) | no card found; Apple verdict only `bench/results/identity_continuation_2026-09-10/umap-apple/verdict.txt` | yes, the portable host math and serial optimizer (`umap/PORTABLE_HOST_MATH.md`), but the IDENTICAL contract is the DEVICE optimizer (`checks/kernel_matrix.mojo:1177`, "The two produce DIFFERENT bits") | 24, the device epoch fold must be restated, the host loop is not it |
| RadiusNeighbors (`neighbors.py:909`) | no card found; two-vendor text leg `bench/results/identity/RBC_551_APPLE_M4.txt`, eps 8.0 diverged on AMD (`IDENTITY_PATHS.md:364`) | inside a GPU check, `_host_dist_sq` `neighbors/checks/radius_check.mojo:72` | 6 after knn |
| StandardScaler, MinMaxScaler (`preprocessing.py:183, 69`) | no card found; Apple smokes only, "no throughput or cross-vendor qualification" (`bench/results/standard_scaler_2026-09-10/RESULTS.md`) | none found; `preprocessing/estimator.mojo:4` imports `max.gpu.host` and every `*_host` entry opens a context, `preprocessing/checks/` is empty | 6 for both, two column folds |
| SambaStack (`_samba_impl.py:181`) | no card found | none found; `training/samba_ops.mojo:15-17` imports the GPU, only `tools/samba_torch_reference.py` | 40 |
| Mamba1Block, Mamba2Block, Mamba3Block (`_mamba_impl.py:387, 619, 940`) | yes, `mamba*.identical.card` on three vendors, historical (audit `:109-115`) | yes, `mamba/checks/mamba{,2,3,_backward}_oracle.mojo`, no GPU import | 16 for the three forwards, 12 more for backward |
| TransformerBlock (`_transformer_impl.py:351`) | yes, `transformer.identical.card` (audit `:116-117`) | yes, `transformer/checks/transformer_oracle.mojo`, `transformer_backward_oracle.mojo`, already compiled host-only inside the byte LM binding | 8 |
| SmallMLPTrainer (`_mlp_impl.py:255`) | no card found; NVIDIA and AMD raw trajectory `bench/results/resume/2026-09-07-root-mlp-amd/`, Metal OPEN (audit `:36, 120-121`) | none found; `training/mlp_ops.mojo:11-13` imports the GPU | 24 |
| LanguageModelTrainer (`language_model.py:29`) | no card, three-vendor raw capture `bench/results/resume/2026-09-07-root-byte-lm-three-vendor/` | yes, and shipped as `LanguageModelHostTrainer` | 0, it is the pattern; the identity_break lane is the missing piece |
| LanguageModelInference, LanguageModelHostTrainer (`_byte_lm_host.py:144, 251`) | n/a, they are the CPU surfaces; `docs/BYTE_LM_CPU_INFERENCE.md`, `docs/BYTE_LM_CPU_TRAINING.md`; the trainer docstring says "NOT CERTIFIED YET" (`:258`) | they are it | 0 |

## 5. The risks, in order

1. **Kernel bodies in a host-only compile.** The measured fact is that
   `max.gpu.host` IMPORTS compile with no accelerator target
   (`docs/BYTE_LM_CPU_TRAINING.md:346-348`, seven runners). Phase 1's ET
   reference and phase 2a's solvers live in files that also DEFINE kernels
   (`std.gpu` at `extratrees/.../builder.mojo:95`, `gemm_identical.mojo:118`).
   Whether a kernel body compiles host-only is unmeasured. The forest host
   lane avoided the question by restating three statements in an
   import-only module; ET's trainer is thousands of lines and cannot be
   restated cheaply. First action of phase 1, one probe compile of a module
   that imports `fit_extra_trees_classifier_reference` with no accelerator
   target, on a free runner, the byte LM probe's shape. Phase 1's hours
   assume it passes; add about 8 hours per family if it does not.

2. **The Apple fallthrough of `TARGET_COLUMN`.** Every host build today is
   compiled as `COLUMN_APPLE` (`checks/kernel_matrix.mojo:360`). The byte LM
   reaches no Apple-only row, so it has been harmless there. The kNN lanes
   reach three (`:1210-1245`), which repair an Apple FMA underflow a CPU
   does not have, and would move kNN bits on the host if left in. Section 2's
   `COLUMN_CPU` with a comptime assert in every host binding retires this
   class of error before the first kNN host fit.

3. **Fold order restated by hand.** Every phase-2 and phase-2b lane rests on
   a serial loop reproducing a device fold whose order is a property of the
   launch (block partials then a halving tree for the `K_LIB_*` float folds,
   `write_reduces_histograms_kernel` for GBDT). The stage checks exist for
   each but none is a whole-fit gate, so the first CPU column of those lanes
   will find every fold the author got wrong at once, with the diff naming
   only the lane and the parts key. The three-vendor GPU columns are the
   reference and the sabotage arm is the control; the debugging cost is in
   the hours, the risk is that it is under.

4. **The fit-on-every-score and pass-count attributes.** iforest re-fits on
   every score (DEVIATION 874), DBSCAN reports `n_iter_`, agglomerative
   reports `n_boruvka_rounds_`. The train column hashes none of the counts,
   so the lanes can pass while the attributes are device-only; the brief
   for each lane must say which attributes the CPU column does not certify.

5. **ftz on x86 without the flag.** `checks/numerics.mojo:73` flushes by
   bits, not by hardware mode, so the CPU column does not depend on MXCSR;
   the seven-runner byte LM result confirms it for the seams the byte LM
   reaches. The `denormal` fixture of identity_break is the first fixture to
   read on every new lane for that reason.

6. **A runner's 60 minutes.** Serial oracles on 20,000-row fixtures, two fits
   per cell, nine fixtures, ten lanes. Unmeasured. If it does not fit, the
   column is split across jobs by `--lanes`, which the JSON's per-lane dump
   already supports.

## 6. What this brief does not say

Nothing here was compiled or run. The whole-fit verdicts are the oracle
files' own claims plus their gate files' existence, read at `0dcc1204e`; the
GPU-import census is a printed grep; the hours are estimates against the
byte LM and forest host lanes' actual cost. The certified table for the CPU
column is empty until a workflow report is read.

## Phase 1 results

Branch `lane/cpu-training-phase1`, off `main` at `e32ddfaf3`, 2026-09-13.
Every lane below was run on this Mac (Apple M4, host build `--target-cpu
apple-m1`, Mojo 1.0.0) through a CPU-only package view (`python/mojolearn`
without `identical/`, `fast/`, `deterministic/` and the tests, `host/`
holding the phase 1 bindings, so `mojolearn.vendor()` is `cpu` and
`--vendor cpu-apple-m4-host`), then diffed with `tools/identity_break.py
--diff` against the three committed GPU columns
`bench/results/identity_break/2026-09-13_46-lanes/{apple-m4,nvidia-h100-sm_90a,amd-mi325x-gfx942}.json`
with `--require-columns 4 --lanes <lane>`. The whole-column summary line
carries the 46-lane column's own `DIVERGENT=8` (gbdt-feature-freq) and
`ONE-COLUMN=18` (the two byte LM CPU lanes the GPU legs did not build); the
lane's verdict is its nine rows and the `require-columns` line. Each lane's
sabotage arm is the same binding built with `-D MOJOLEARN_HOST_SABOTAGE=1`,
loaded through `MOJOLEARN_HOST_DIR` with `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`.
The seven-runner gate (`.github/workflows/cpu-identity-gate.yml`,
`COVERED_LANES`) has not run on this branch; the certified table is still
empty until a workflow report is read.

### gemm-pinned, IDENTICAL x4

Binding `bindings/_mojolearn_linalg_host.mojo` (`bindings/build_linalg_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_linalg"]`, exporting `gemm`,
`linalg_numeric_mode`, `linalg_vendor` (answering `cpu`),
`linalg_profile_version` under the GPU binding's address contract, over
`gemm/checks/gemm_oracle.mojo::gemm_oracle`. `python/mojolearn/_linalg_impl.py`
is unchanged.

| fixture | hash (all four columns) |
|---|---|
| base | 931036cbc84c1ce0 |
| ties | 8ddd60b67526c1b1 |
| hashed | 06057c0eeb212963 |
| wide | 01701582fa724ce4 |
| denormal | 026720f450eda8a3 |
| denormal_ftz | 026720f450eda8a3 |
| dupes | 4d5d6e4a621ef9af |
| odd | d5a7a31e551215d9 |
| negative | 1ef67a1833b19819 |

`require-columns 4 over ['gemm-pinned']: OK`; the nine rows read
`IDENTICAL x4`. Infer and model are `n/a:function` and `n/a:no-save`, as on
the GPU columns. Sabotage (`GEMM_ORACLE_HOST_SABOTAGE`, every leaf walked
descending): 8 of 9 cells `DIVERGENT` with `parts differ: small,wide`
(base 260c0a74b97fa901 against 931036cbc84c1ce0); `ties` stays
`IDENTICAL x4` because an integer grid sums exactly in any order, which is
why the gate requires `DIVERGENT` in the summary and not on every cell.

### kde, IDENTICAL x4

Binding `bindings/_mojolearn_estimators_host.mojo` (`bindings/build_estimators_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_estimators"]`, exporting
`kde_score_samples`, `estimators_numeric_mode`, `estimators_vendor` under
the GPU binding's address contract, over
`kde/checks/kde_oracle.mojo::oracle_score_samples`, with the GPU entry's
validation in the GPU entry's order (`kernel_from_name`, `metric_from_name`,
`kde_fit_validate`, `kde_validate_data_ptr`, all host code in
`kde/impl/neighbors/kernel_density.mojo`). `python/mojolearn/density.py` is
unchanged. Every other `_mojolearn_estimators` function (dbscan, pca, tsvd,
ols, ridge, logistic) is absent from the host binding and refuses by name.
Risk 1 of section 5, answered for this family: `kde_oracle.mojo` imports
`kde/impl/neighbors/kernel_density.mojo`, `kde/impl/distance/distance_ops.mojo`
and `core/row_norms.mojo`, each of which defines kernels and imports
`std.gpu`, and the host-only build (`--target-cpu apple-m1`, no accelerator
target) compiled them in about a minute.

| fixture | train (all four columns) | infer (all four columns) |
|---|---|---|
| base | e0d6e3d0623d6112 | 1a2c3054b661b72a |
| ties | 0a800c7b3cca66e4 | 63b4da91ce458ab7 |
| hashed | dc279c30b7c5d070 | f0d1cae79c88ca8a |
| wide | 36c3cbb2ef71f768 | 39622885a3c508a9 |
| denormal | 2128392fe1eb228a | 21caf753bc5b6352 |
| denormal_ftz | 2128392fe1eb228a | 21caf753bc5b6352 |
| dupes | e0d6e3d0623d6112 | 1a2c3054b661b72a |
| odd | 8a2b6beee09a98f3 | e49f797e443af916 |
| negative | 84ca35b6a157c265 | 30a64b99c13a89c4 |

`require-columns 4 over ['kde']: OK`; the nine train rows and the nine
infer rows read `IDENTICAL x4`; model is `n/a:no-save`. Sabotage
(`KDE_ORACLE_HOST_SABOTAGE`, every logsumexp row summed descending): 9 of 9
train cells and 9 of 9 infer cells `DIVERGENT`, `parts differ: scores`
(base a58ce84d395f4d1e against e0d6e3d0623d6112).

### holtwinters, IDENTICAL x4

Binding `bindings/_mojolearn_tsa_host.mojo` (`bindings/build_tsa_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_tsa"]`, exporting
`holtwinters_fit`, `holtwinters_forecast`, `tsa_vendor` under the GPU
binding's address contract and packed layouts, over
`holtwinters/checks/hw_oracle.mojo::oracle_fit[DType.float32]` and
`oracle_forecast`, with the GPU entry's validation in its order
(`holtwinters_fit_ptr`'s extent guards, `seasonal_from_name`,
`holtwinters_validate_params`, `holtwinters_validate_data`).
`python/mojolearn/_tsa_impl.py` is unchanged. `kpss_test` and `select_d`
are absent and refuse by name. `hw_oracle.mojo` imports `runner.mojo`,
`hw_decompose.mojo` and `hw_optim.mojo`, each with kernels and `std.gpu`;
the host-only build compiled them (risk 1, answered for this family too).
The attributes the CPU column certifies are the ones the train column
hashes, the forecast; `n_iter_` and `criterion_` are written from the
oracle's `niter` and `criterion`, which `hw_check::_compare_fit` holds
bitwise to the device's, but no identity_break cell hashes them.

| fixture | forecast hash (all four columns) |
|---|---|
| base | 9781c2061a287937 |
| ties | b2a5df45db1395b0 |
| hashed | 66f688c307cabc3b |
| wide | 560cdc45a69cc987 |
| denormal | db35c88da059e9e1 |
| denormal_ftz | db35c88da059e9e1 |
| dupes | 9781c2061a287937 |
| odd | 8fb4dd76d8080a46 |
| negative | 8fa4116875bdf59c |

`require-columns 4 over ['holtwinters']: OK`; the nine rows read
`IDENTICAL x4`; infer is `n/a:forecast`, model `n/a:no-save`. Sabotage
(`HW_ORACLE_HOST_SABOTAGE`, the SSE fused multiply-add split into two
roundings): 6 of 9 cells `DIVERGENT`, `parts differ: forecast` (base
3bf92be9d53726e1 against 9781c2061a287937); `denormal`, `denormal_ftz` and
`wide` stay `IDENTICAL x4`, because the lane's series (a cumulative sum of
column 0 plus 50) is nearly constant on those fixtures and BFGS ends at the
same parameters under both spellings of the loss. The gate requires
`DIVERGENT` in the summary, which holds; a fixture the arm cannot move is
recorded, not hidden.

### lasso and elasticnet, IDENTICAL x4

Binding `bindings/_mojolearn_solver_host.mojo` (`bindings/build_solver_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_solver"]`, exporting `cd_fit`,
`cd_predict`, `solver_vendor` under the GPU binding's address contract
(column-major design, the nine-value and three-value params lists). The
fit is `solver/checks/cd_oracle.mojo::cd_oracle_fit` at `profile=True`
(every reduction a `gemm_oracle_cell`); the predict restates
`linearRegH`'s IDENTICAL arm, `gemm_oracle(x, coef, OP_TN, n_rows, 1,
n_cols)` then `ftz(v + intercept)`, because `solver/impl/functions/linear_reg.mojo`
defines the kernels beside it. The guards are `cd_fit_host`'s then
`cd_fit_traced`'s, in their order and words (`sample_weight` and `shuffle`
refused by name as on the device). `python/mojolearn/_solver_impl.py` is
unchanged. `linkage_fit` is absent until the agglomerative lane.
`cd_oracle.mojo` imports `solver/checks/profile_dot.mojo`, which imports
`max.gpu.host` and `gemm/checks/gemm_identical.mojo` (the kernels); the
host-only build compiled it (the risk named in section 1.1 for this lane,
answered).

A FINDING ON THE WAY. The first run REFUSED every cell at
`_mojolearn.transpose_f32`: `python/mojolearn/_buffer.py::_native` resolves
the input converters (`transpose_f32`, `cast_colmajor_f64_to_f32`,
`cast_f64_to_f32`, `all_finite_*`) from the BASE binding, and `cdFit`'s
Fortran-order design goes through `transpose_f32`, which no host binding
carried (`_host_native` covers only the byte LM and forest host sets, none
of which has the transpose). Section 3.2's "the Python estimator classes
need no change" holds, but the base binding's host helpers are a
dependency of every lane whose Python layer converts an array, not of
kmeans and knn alone. `bindings/_mojolearn_core_host.mojo`
(`bindings/build_core_host.sh`, routed by `_HOST_MODULES["_mojolearn"]`)
carries the eleven helpers under the base binding's names (the transpose
pair MIRRORS `_tiled_transpose_to_f32` element for element; the rest are
`bindings/host_helpers.mojo`), and nothing else, so kmeans and knn keep
refusing by name. It moves bytes and folds nothing, so it has no sabotage
arm; `core_host_sabotage()` reports the define so a sabotage set loads as
one set.

| fixture | lasso train | lasso infer | elasticnet train | elasticnet infer |
|---|---|---|---|---|
| base | 2fa3301e45a46091 | 7334b38e7b651e27 | 2cf742083831ee9d | 483bce4b8cc1a137 |
| ties | 1e207f82f555270b | ff61f45db38db46d | 118eac19578010ab | a767a79709358382 |
| hashed | 0446105edc307f9c | 482da47ba9b49877 | 00b6923d62a41c0a | db0926772dbaaff4 |
| wide | d0df889b0ede05a7 | 1f85238c927b088e | ef8800eaf0ca4f87 | e37614bc5b2ef0af |
| denormal | 5374b434a96382d6 | b92e9bfda9b725b4 | 3add60ff69fd788c | 0047832ca8c561a3 |
| denormal_ftz | 5374b434a96382d6 | b92e9bfda9b725b4 | 3add60ff69fd788c | 0047832ca8c561a3 |
| dupes | beafb1d480c34cf4 | 727d68f61ef51bfb | 97d4c8c235d38bfe | fc18f039274052f1 |
| odd | 5593579edd6113d6 | 1c17e2eb1ef6a2e2 | 1a518e703b7dd1d9 | 75a88217f1114867 |
| negative | c332c713694ff28d | 0167126d36a4243d | 7624315e7deffe10 | a21e833e3e925b89 |

`require-columns 4 over ['lasso']: OK` and `require-columns 4 over
['elasticnet']: OK`; all 36 rows (nine train and nine infer per lane) read
`IDENTICAL x4`; model is `n/a:no-save`. Sabotage (the GEMM oracle's
descending leaf, reached through every `gemm_oracle_cell` reduction of the
CD oracle): 9 of 9 train and 9 of 9 infer cells `DIVERGENT` on each lane,
`parts differ: coef,predict` (lasso base 61a6ae20cfafca49 against
2fa3301e45a46091; elasticnet base 3bde50d4452e548a against
2cf742083831ee9d).

### svc, IDENTICAL x4

Binding `bindings/_mojolearn_svm_host.mojo` (`bindings/build_svm_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_svm"]`, exporting `svc_fit`,
`svc_predict`, `svm_vendor`, `svm_numeric_mode` under the GPU binding's
address contract (worst-case sized outputs, the five float64 info slots),
over `svm/checks/smo_oracle.mojo::smo_oracle_fit[DType.float32]` and
`smo_oracle_decision`, with the GPU entry's guards in its order
(`svc_fit_host_borrowed`, `svc_fit_borrowed`, `_svc_label_model`), the
one-vs-rest targets by `ovr_labels_kernel`'s rule, the support matrix by
`CollectSupportVectorMatrix`'s gather, and `applyPrediction`'s epilogue
`label0 if val < 0 else label1`. `python/mojolearn/_svm_impl.py` is
unchanged. `svr_fit`, `svr_predict` and `iforest_run` are absent and refuse
by name. `smo_oracle.mojo` imports `svm/impl/smosolver.mojo` (kernels,
`std.gpu`) for `fold_order_for` and `hash_f32_list`; the host-only build
compiled it. The lane's `max_iter=200` fixture converges the same way on
all four columns (the risk named in section 1.1). `n_iter_` is the oracle's
inner iteration count, held to the device per outer iteration by
`svc_check` but hashed by no cell.

| fixture | train (decision, predict) | infer |
|---|---|---|
| base | dec306e940b8a444 | 4aee3fcc51c4761e |
| ties | 2093351e33d072ed | e400ffe98db4b786 |
| hashed | 274bf9087fb0750e | 94cb46a9aa6a811c |
| wide | 550d1845e417ca18 | ee2e31ef0e3bbfcb |
| denormal | 549c0b5c319d84cf | ebc593040592ca1e |
| denormal_ftz | 549c0b5c319d84cf | ebc593040592ca1e |
| dupes | b2f0d9a49c9f241c | 6e5c253fa1d7ea5b |
| odd | 61b1676a7e40c914 | acd970db5a3ae605 |
| negative | 667155f7eda847aa | 685bebcb29a0cf8c |

`require-columns 4 over ['svc']: OK`; the nine train rows and the nine
infer rows read `IDENTICAL x4`; model is `n/a:no-save`. Sabotage
(`SMO_ORACLE_HOST_SABOTAGE`, the SMO oracle's GEMM leaf walked descending):
8 of 9 train cells `DIVERGENT` (`parts differ: decision`, and on `wide`
the predicted labels too) and 8 of 9 infer cells; `ties` stays
`IDENTICAL x4`, the integer grid summing exactly in any order.

## Phase 1b results (2026-09-14): agglomerative, et-clf, et-reg, iforest

Branch `lane/cpu-training-phase1b`, off `main` at `6796ceff9`. The four
lanes of the phase 1 table above that the 2026-09-13 merge left open, run
on this Mac exactly as the phase 1 lanes were (Apple M4, host build
`--target-cpu apple-m1`, Mojo 1.0.0, the CPU-only package path with the
host bindings under `python/mojolearn/host/`, `--vendor cpu-apple-m4`),
diffed against the three 2026-09-14 GPU columns
`bench/results/identity_break/2026-09-14_46-lanes/{apple-m4,nvidia-h100-sm_90a,amd-mi300x-gfx942}.json`
with `--require-columns 4 --lanes <lane>`. The JSONs, the diffs and the
sabotage runs are under `bench/results/identity_break/2026-09-14_cpu-phase1b/`
(its README is the index). The whole-column summary line reads
`IDENTICAL=414` for every lane below; the lane's verdict is its rows and
the `require-columns` line. The seven-runner gate has not run on this
branch yet; the certified table waits for its reports.

Two pattern changes landed with these lanes. The six per-family build
scripts were folded into `bindings/build_host_family.sh` (one script, the
family as its argument; each `bindings/build_<family>_host.sh` is a
two-line wrapper naming its family, so the workflow, the GPU legs and a
developer call what they always called; the estimators wrapper was left as
the copy it was because another lane is editing that family), and the
sabotage define now reaches every phase 1 lane: `hierarchy/checks/linkage_oracle.mojo`
(`host_kruskal` walks the sorted keys descending, the maximum spanning
tree), `extratrees/checks/pcg_rng.mojo` (`SplitKey.generator` burns one
draw on every keyed stream) and `isolation_forest/impl/rng/xorwow.mojo`
(`curand_uniform` advances one extra step). Section 3.4 above described
the last two as already honored; on 2026-09-13 neither file carried the
define (a `git grep MOJOLEARN_HOST_SABOTAGE` found it in `gemm_oracle`,
`kde_oracle`, `hw_oracle` and `smo_oracle` only), so that sentence was a
plan, not a fact, until this lane.

### agglomerative, IDENTICAL x4

`linkage_fit` in `bindings/_mojolearn_solver_host.mojo` over
`hierarchy/checks/linkage_oracle.mojo`: `host_pinned_distance_matrix`
(the IDENTICAL tile's arithmetic), `host_kruskal` (Kruskal under the
device's total order, so the MST is the device's Boruvka MST), `host_dendrogram`,
`host_extract_flattened_clusters`, with the device path's guards in its
order and words. `python/mojolearn/_hierarchy_impl.py` is unchanged
except a comment: `n_boruvka_rounds_` reads -1 on a CPU-only install,
because the host runs Kruskal and there is no Boruvka pass to count (the
risk named in section 1.1 (e), resolved by reporting it as device-only;
no cell hashes it). `n_connected_components_` is 1, the pairwise arm's
literal.

| fixture | labels hash (all four columns) |
|---|---|
| base | ad11b976bfbfc20b |
| ties | 7dea8a094150b8a9 |
| hashed | 196c79a60c16166c |
| wide | 24b2baa4533fb568 |
| denormal | b05982504c7e8646 |
| denormal_ftz | b05982504c7e8646 |
| dupes | ad11b976bfbfc20b |
| odd | c747a33eff11bbe1 |
| negative | 795f9975eb5ed985 |

`require-columns 4 over ['agglomerative']: OK`; infer is
`n/a:transductive`, model `n/a:no-save`, as on the GPU columns. Sabotage
(`LINKAGE_ORACLE_HOST_SABOTAGE`): 9 of 9 `DIVERGENT`, `parts differ:
labels` (base 4302f3be04ed038b against ad11b976bfbfc20b). The arm is on
the ORDER, not on a fold: a reversed distance fold moves distances by an
ulp and leaves a 4-cluster cut alone on most fixtures, which is not a
control that can fail. 24 s for the lane on this Mac.

### et-clf and et-reg, IDENTICAL x4, model column included

`bindings/_mojolearn_trees_host.mojo` (`bindings/build_trees_host.sh`),
routed by `_backend._HOST_MODULES["_mojolearn_trees"]`, exporting the
eight `et_*_fit` entries (plain, `_export`, `_rowmajor`,
`_rowmajor_export`; the 22-slot params list of the GPU binding, slot 20
accepted at 1, the one refusal the host binding drops), `forest_export`,
`forest_export_legacy`, `forest_export_release`, `et_predict` (over
`core/forest_host_predict.mojo`, the forest host binding's walk, so the
model column's RELOAD check on a CPU-only install predicts through the
CPU inference path), `trees_vendor`, `trees_numeric_mode`.
`python/mojolearn/extratrees.py` and `_forest_protocol.py` are unchanged.

THE FIT IS NOT THE EXISTING REFERENCE. Section 1.1 named
`fit_extra_trees_classifier_reference` and `fit_extra_trees_regressor_reference`
as the host routines, and the regressor's leaf restatement as the risk.
Reading the two searches side by side showed a second gap: the host
regressor `node_split_random_mse` orders candidates by sklearn's Float64
proxy (DEVIATION 153) where the device orders by cuML's exact `Int64` MSE
key over the QUANTIZED labels (DEVIATION 189, `regression_key`, with a
node-uniform right shift of 14 bits at 20,000 rows). The two orderings
agree in exact arithmetic and can disagree bit for bit on a near-tie the
shift turns into an exact tie (then resolved by DEVIATION 463's keyed
rank), so structure identity between that reference and the device is a
measured fact on the check's fixtures, not a property of the code. The
CPU column therefore runs a HOST RESTATEMENT OF THE DEVICE SEARCH,
`train_tree_exact` (`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`,
with `fit_forest_exact` in `randomforest.mojo` and
`fit_extra_trees_classifier_host_exact` / `fit_extra_trees_regressor_host_exact`
in `estimator.mojo`): per (node, feature) `node_feature_score_host`, the
score kernel's own sequential oracle; the candidate as
`score_to_candidate_kernel` forms it; `SplitExact.update` in slot order;
the readback's `MIN_FINITE` fix; `split_not_valid`, `partition_samples`,
`NodeQueue.push`; DEVIATION 205's rescue keyed as the device keys it; and
the leaf pass as `leaf_kernel` computes it over the device's label plane
(class ids, or `quantize_labels`'s fixed point with `Float32(1 / scale)`).
Both objectives go through it. Best-first growth (`max_leaf_nodes`) is
refused by name on that path. The docstring of
`fit_extra_trees_regressor_device` at `extratrees/estimator.mojo`, which
said the split decision "is made on integer sums on both sides", now says
what each side orders by; `extratrees/checks/device_regression_check.mojo:11-15`
says the same thing and is a check file, reported here rather than
edited.

A FINDING ON THE WAY. The first et-clf run REFUSED every cell at
`_mojolearn.encode_labels_i32`: `_labels.encode_labels` resolves its
native encoder from the base binding, and `bindings/_mojolearn_core_host.mojo`
carries the converters, the finiteness scans, the gathers and the
argmaxes but not `encode_labels_*`. That file is under another lane's
edit, so `_labels._encode_labels_native` now takes the Python routine on
a CPU-only install when the native encoder is missing (the routine is the
encoder's definition and `tests/test_labels_native.py` holds the native
copy equal to it; a GPU install still fails loudly). Adding the encoders
to the core host binding is the cleaner close and is owed to that file's
owner.

| fixture | et-clf train | et-clf model | et-reg train | et-reg model |
|---|---|---|---|---|
| base | c586b27a3b049614 | 58a53490a62fba59 | 754d8c127ecfc04d | 1c0c6b20cc9be5bd |
| ties | 604f59c7a4203eeb | e63de8292d8384e4 | e7072e1c8fe0083c | 0ae34e3bcc2b7391 |
| hashed | eb7aaadcd0843a6d | 94580dd625aaa11a | 4e0a0d9a4670ed5f | 029c99465d40dd66 |
| wide | ee8b318d6bf698b2 | f792edfcd53b6aff | b745e53515f59cac | 6da81ae280958f67 |
| denormal | b40377fb35e52909 | 5d348c7133844b3e | 19a23f6fea44befd | 8704471f45cc32d6 |
| denormal_ftz | b40377fb35e52909 | 5d348c7133844b3e | 19a23f6fea44befd | 8704471f45cc32d6 |
| dupes | a24dcc8a93f699ff | ff82d3ce7bd5651f | 9a63fea590f79a67 | d99a54bf0717266d |
| odd | 1da918353bb9f10e | ed0c7e9a34735dbd | c9fe7c90673c2661 | d631aaa061d3f2e4 |
| negative | e1409786c462d97b | 974e58a34d03e07b | 3fd3dff94cc79fc2 | 34d9143f7fc1fec7 |

`require-columns 4 over ['et-clf', 'et-reg']: OK`; all 54 rows (nine
train, nine infer and nine model per lane; the infer hashes are in
`diff.et.txt`, base et-clf b728e73e5f84514c, et-reg f67822ca39ef408b)
read `IDENTICAL x4`; the model column is the saved bytes, so the five
model arrays a CPU fits are byte for byte the three GPUs'. Sabotage
(`PCG_HOST_SABOTAGE`): 18 of 18 train, infer and model cells `DIVERGENT`
(et-clf base d630c867d817d8fd, et-reg base ea112e2954936a35). The first
placement of that hook, inside `uniform_threshold`, was measured INERT on
this path (the restated search draws through `draw_threshold_device`,
which never calls it; `cpu-apple-m4.sabotage.agglomerative-et.json` is
that run, kept), and the hook moved to `SplitKey.generator`, which every
keyed draw goes through. The ET lanes take 15 s together on this Mac
(the "time first" item of section 3.4: the restated search at 16 trees,
depth 8, 20,000 by 16 is well under a second per fit).

### iforest, IDENTICAL x4

`iforest_run` in `bindings/_mojolearn_svm_host.mojo` over
`isolation_forest/checks/if_oracle.mojo::oracle_fit`, `oracle_path_lengths`
and `oracle_scores` (the XORWOW tables rebuilt on the host per call,
DEVIATION 683), the parameter resolution of `IsolationForestEstimator.fit`
and the estimator's epilogues (`-paper`, `- Float32(offset_)`,
`-(paper > Float32(-offset_) ? 1 : -1)`, the contamination quantile
through `percentile_linear`), under the GPU binding's 16-slot contract
with DEVIATION 874's fit-on-every-call kept, so the identity surface is
the GPU's and `python/mojolearn/_iforest_impl.py` is unchanged. The
contamination-quantile arm is restated but no identity_break lane runs it
(the lane fits at `contamination='auto'`); it is a non-default path in
rule 8's sense until a lane covers it.

A FINDING ON THE WAY. The first run diverged on `denormal` alone (train
`scores` 045af056aca03f84 against 54ea9b5c8c3fc38b, infer likewise), with
`denormal_ftz` identical: the device stages every input cell through
`ftz` at upload (`_upload_f32`, `_upload_rowmajor_as_colmajor`) for the
training matrix and for every query, and the oracle reads its lists raw.
The binding now flushes both matrices at read, once, at the same
boundary; nothing else moved.

| fixture | train (scores, predict) | infer (all four columns) |
|---|---|---|
| base | 8703d4a008ffd13d | c4247bb3a6c4675e |
| ties | 5043855ee14799da | see `diff.iforest.txt` |
| hashed | 72260557474879ae | see `diff.iforest.txt` |
| wide | 8703d4a008ffd13d | see `diff.iforest.txt` |
| denormal | 54ea9b5c8c3fc38b | f3dcde6748050803 |
| denormal_ftz | 54ea9b5c8c3fc38b | f3dcde6748050803 |
| dupes | b879bcf3dcf99f38 | see `diff.iforest.txt` |
| odd | 9b849a13c8d35adb | see `diff.iforest.txt` |
| negative | a5b7676e39778af5 | see `diff.iforest.txt` |

`require-columns 4 over ['iforest']: OK`; the nine train rows and the nine
infer rows read `IDENTICAL x4`; model is `n/a:no-save` on all four.
Sabotage (`XORWOW_HOST_SABOTAGE`): 9 of 9 train and infer cells
`DIVERGENT`, `parts differ: scores,predict` (base 9d499389c350b685). 8 s
for the lane on this Mac.

STILL OWED FOR IFOREST: the GPU-side model export. The forest host
inference brief (`docs/lanes/BRIEF_forest_host_inference_2026-09-13.md`,
its iforest entry and rank 9) says no fitted model exists to save and a
GPU-side export comes first; that is still true. A `save` on the Python
class would turn the model cell from `n/a:no-save` into a hash on the CPU
column while the three GPU columns still read `n/a`, so `--require-columns 4`
would fail on the model cell until the GPU columns are rerun with the same
export, and the GPU bindings are not rebuilt on this Mac. The export
(the four node arrays, the tree offsets, `c_normalization`, `offset_`,
`max_samples_`, materialized once after the device fit) and a host
`score_samples` / `predict` over the saved arrays are one lane with a GPU
leg; the host scorer for it is `oracle_path_lengths` / `oracle_scores`
over an `OracleForest` rebuilt from the arrays, which this binding
already runs.

## Workstream E (2026-09-14): knn, knn-clf, knn-reg, pca, pca-whiten, tsvd, ols, ridge, dbscan, WRITTEN AND COMPILE-CHECKED, NOT MEASURED

Branch `lane/cpu-training-e`, off `origin/main` at 59ee1b98b. Nothing in this
section is a bit result; the four-column diff has not run. What exists follows.

- knn, knn-clf, knn-reg. The fit stores the index (`neighbors.py:578`) and
  the train cell is `kneighbors`, `predict` and `predict_proba` on training
  rows, the same host search, vote and mean the knn host inference lane
  serves through `_mojolearn_core_host` (`core/knn_host_predict.mojo`). No
  code was added; the three lanes are declared as the core family's
  training lanes in `python/mojolearn/host_surface.py`, so the CPU identity
  gate runs them and demands IDENTICAL x4. The `knn-clf-distance` and
  `knn-reg-distance` twins wait for a GPU record that carries them (the
  2026-09-14 47-lane record does not).
- pca, pca-whiten, tsvd. `decomposition/host/pca_oracle.mojo`, the fit
  restated from its kernels, namely `column_mean_kernel` (STATS_TPB lane partials,
  the halving tree), the split-K Gram (128 pinned chunks, the fused centered
  read, the serial chunk fold), `scale_in_place_kernel`, a Float32 Jacobi at
  the device's 15 sweeps and 1e-7 with the JACOBI_TPB lane folds and the
  DEVIATION 2671 merged phase, `sign_flip_kernel`, the Float64 tail. Past
  128 columns the Gram is `gemm_oracle` at OP_TN, unmeasured on any
  fixture. Exported as `pca_fit` and `tsvd_fit` from
  `bindings/_mojolearn_estimators_host.mojo`; `pca_fit_full` stays absent.
- ols, ridge. `glm/host/glm_oracle.mojo` carries `lstsq_eig` (the Gram and Jacobi
  above, `xty_kernel`, the DEVIATION 2620 equilibration and 2621 cutoff,
  `divide_columns_by_nonzero_kernel`, the pinned `gemm_nt` and `gemv_n`
  cells of `core/classical_host_predict.mojo`) and `svd_eig` plus
  `ridge_solve` with every elementwise kernel of `glm/impl/matrix/math.mojo`
  restated. `n_cols > n_rows` (`lstsq_min_norm`) is refused by name.
  Exported as `ols_fit` and `ridge_fit`.
- dbscan. `dbscan/host/dbscan_oracle.mojo`, the oracle the census said did
  not exist, written as a second spelling of the device path. The ball cover
  index (`rbc_n_landmarks`, `_floyd_sample` at seed 12345, the strict-`<`
  nearest landmark rooted by `identical_sqrt`, the member lists ranked by
  distance then index, the last member's distance as the radius) and the
  eps query (`block_rbc_kernel_eps_csr_pass`, the landmark test against
  `(eps + radius)^2`, the member scan in RBC_LANES chunks from the ragged
  tail backward with its `cur_r_dist - min_warp_dist > eps` exit) are
  replayed with `eps_dist_sq`'s fold; the brute arm keeps
  `eps_unexp_neigh_kernel`'s fold (the query value unflushed). Then
  `weak_cc` as the one-thread schedule in row order (the same fixed point;
  the pass count `n_iter_` is that schedule's and no column hashes it) and
  `make_monotonic` plus the scikit-learn relabel. One batch of every row.
  `sample_weight` and an explicit `max_mbytes_per_batch` are refused by
  name. Exported as `dbscan_fit`. The `dbscan-brute-l1` and
  `dbscan-weighted` twins wait for a GPU record that carries them.
- The sabotage arm is the estimators family's `-D MOJOLEARN_HOST_SABOTAGE=1`,
  under which the DBSCAN core test asks one neighbor more and the Gram reduce walks its chunks descending. On the Apple M4, production
  and sabotage builds of the binding compiled and, on a 3000 x 6 draw, the
  sabotage build moved every PCA, tSVD, OLS and ridge output (the mean is
  inert by construction, it has no Gram); production runs twice gave the
  same bytes and agreed with numpy at 1e-6 relative. That is a plumbing
  check, not identity.
- The seven-runner gate (run 34869406147) refused ols and ridge on every
  runner at `_mojolearn.column_mean_f64`, the centering helper
  `linear_model.py` reaches through `_buffer._native` before the estimators
  host binding is called; on a CPU-only install that resolves to the core
  host binding, which did not carry it. `column_mean_f64`,
  `center_columns_f32` and `scale_rows_f32` now live in
  `bindings/host_helpers.mojo` (the base binding's definitions on the
  calling thread, the same chains in the same order) and are exported by
  `_mojolearn_core_host`. The Mac reproduced the refusal first (the same
  ImportError, by name, with the CPU-only path forced through
  `MOJOLEARN_HOST_DIR` on a worktree with no GPU set) and, with the fix,
  all nine lane bodies on the `base` fixture read IDENTICAL against the
  three 47-lane GPU columns (train parts and infer hash, one fixture, one
  repeat; a witness, not the gate).
- The test module is `cd python && python3 -m mojolearn.tests.test_cpu_training_e`.

To measure, on each GPU box and on a CPU-only box (the CPU identity gate
runs the same steps on seven runners), the commands are these.

    python3 tools/identity_break.py --lanes knn,knn-clf,knn-reg,pca,pca-whiten,tsvd,ols,ridge,dbscan --json <box>.json
    python3 tools/identity_break.py --diff <apple.json> <nvidia.json> <amd.json> <cpu.json> \
        --require-columns 4 --lanes knn,knn-clf,knn-reg,pca,pca-whiten,tsvd,ols,ridge,dbscan
    MOJOLEARN_HOST_OUTDIR=<sab> MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" sh bindings/build_estimators_host.sh
    MOJOLEARN_HOST_DIR=<sab> MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/identity_break.py --lanes pca,pca-whiten,tsvd,ols,ridge,dbscan --json <cpu-sab>.json
    (the diff of <cpu-sab>.json against the three GPU columns must exit non-zero with DIVERGENT)

The three 2026-09-14 47-lane GPU columns already carry every one of these
lanes, so no new GPU record is needed for the first diff.

## Workstream E batch 2 (2026-09-14): kmeans, SIMULATED IDENTICAL x3 ON base, GATE OWED

Branch `lane/cpu-training-e2`, off `origin/main` at 2b7f991b6. The gate has
not run; what is measured is one fixture on one box, stated as such.

- kmeans. `cluster/host/kmeans_oracle.mojo`, the fit restated from its
  kernels in the order `kmeans_fit` then `fit_predict` reach them: the
  splitmix64 `HostRng`, `plan_sum_scale` and `choose_scale` (the latter
  imported, it is host code already), `row_norm_kernel` at NORM_TPB with
  the halving tree, the fused assignment (one ascending fma chain per cell,
  the epilog, the self-neighbor guard, `argmin_op`'s total order),
  `_sum_device`'s two-stage fold at REDUCE_BY_KEY_TPB with the map inside,
  the three-stage device scan with the library block scan replayed at a
  32-wide warp (Hillis-Steele inside the warp, the warp totals scanned by
  warp 0, max/v26.5.0's `block.mojo:672` and `warp.mojo:1084`), the binary
  search, the classic k-means++ over the candidates (the pinned `gemm_nt`
  cell, `candidate_cost_kernel`'s chains, the Float64 argmin), k-means||
  (the counter-hash uniforms, `scalable_keep`, the flag scan and stable
  scatter, the float count histogram, the recluster under fresh defaults
  with the inner scales), the quantized Int32 scatter-add,
  `finalize_centroids_kernel`, the shift test, the post-loop assignment and
  the weighted inertia, then `fit_predict`'s fresh assignment. Exported as
  `kmeans_fit` from `bindings/_mojolearn_core_host.mojo` with the GPU
  binding's ten-value params list; `KMeans.fit` is unchanged.
- THE ONE DEVICE FACT THE SOURCE DID NOT SAY. The first host build read
  DIVERGENT on every column. The Apple M4 GPU (the Sep 13 base binding,
  which reproduces the record's `kmeans/base` hashes) was traced with
  `MOJOLEARN_IDENTITY_TRACE` and its candidate norms dumped: 132 candidates
  against the host's 134, round 0 of k-means|| identical (19 of 19 rows),
  round 1 disjoint. The round seed crosses to `sample_flags_kernel` as two
  Int32 halves and the kernel's `lo.cast[uint32]().cast[uint64]()`
  SIGN-EXTENDS on the device, so the seed it hashes is `(hi << 32) |
  sext64(lo)`; round 0's low half was positive and round 1's negative.
  `host_round_seed_as_the_device_reassembles_it` spells that value. With
  it the host card and the Apple card agree on all 1190 stages
  (`tools/identity_trace_diff.py`, `fit.x_norm` through `fit.labels`, the
  recluster included) and the lane reads IDENTICAL against the Apple,
  NVIDIA and AMD 47-lane columns on `base` (one fixture, one repeat, the
  CPU-only path forced through MOJOLEARN_HOST_DIR). The three GPU columns
  agree on every k-means cell, so the three vendors share the reassembly;
  a host with the whole 64-bit draw is the side that is wrong.
- The block scan's warp width. The library scan folds at the hardware
  width (64 on the MI300X), so the AMD `csum` may differ from the 32-wide
  replay in bits that reach `binary_search_kernel` only when a target lands
  inside that gap; the three GPU columns' agreement on every k-means cell
  is the evidence that it did not on the seven fixtures. Stated in the
  oracle's header, not claimed away.
- The sabotage arm is the core family's `-D MOJOLEARN_HOST_SABOTAGE=1`,
  under which every quantized centroid-sum cell carries one extra unit
  (an order walked differently would not reliably move an argmin or an
  Int32 sum). On the M4 the sabotage build read DIVERGENT on `centers` and
  `labels` against all three columns on `base`.
- kmeans-random, kmeans-array and kmeans-weighted share the entry (`init`
  random and array, supplied weights are restated) but the 47-lane record
  carries no cell for them, so they are not declared covered; they wait
  for a GPU record that carries them.
- The test module is `cd python && python3 -m mojolearn.tests.test_cpu_training_e2`.

The measurement owed is the seven-runner CPU identity gate on the lane
(`--require-columns 4 --lanes ... kmeans`), every fixture, two repeats.

## Workstream E batch 2 (2026-09-14): metrics, SIMULATED IDENTICAL x3 ON base, GATE OWED

- metrics. `metrics/host/metrics_oracle.mojo`, the five metrics of the lane
  and the four label metrics that share their integer kernels, restated
  from `metrics/impl/stats/detail/`: the integer count, histogram and
  contingency matrix as serial loops (their device forms are integer
  atomics whose sums no order moves), DEVIATION 653's slab tree
  (`PINNED_SUM_W` 256, the halving fold, the chunk totals ascending) for
  every float sum, `r2_epilogue` with DEVIATION 657, the IDENTICAL Float32
  arms of entropy and mutual information (host code on the GPU path too),
  homogeneity, completeness and the v-measure in Float64, and the batched
  silhouette row by row (DEVIATION 654's per-cluster tree, the positional
  min, `sil_op` with DEVIATION 656, the tree over the scores). A new host
  family, `metrics`: `bindings/_mojolearn_metrics_host.mojo` exports the
  nine entries under the GPU binding's names with its `params` lists,
  `bindings/build_metrics_host.sh` is the shim, and the manifest routes
  `_mojolearn_metrics` to it, so `mojolearn.metrics` runs unchanged. The
  spectral, UMAP, ranking, classification, regression-error, KL and
  trustworthiness entries stay absent and refuse by name.
- On the M4's CPU-only path the lane (its KMeans through the core host
  binding, then the five metrics) reads IDENTICAL against the Apple, NVIDIA
  and AMD 47-lane columns on `base`, one fixture, one repeat. The gate has
  not run.
- The sabotage arm is the family's `-D MOJOLEARN_HOST_SABOTAGE=1`, under
  which every slab tree's chunk boundaries shift by one value
  (`pinned_sum.mojo::sabotage_shifted_host_tree_sum`'s partition, the one
  its check measured to move a sum where a rotation inside a chunk cannot);
  it reaches r2 and the silhouette, and the integer metrics do not move. On
  the M4 the sabotage set read DIVERGENT on `silhouette` against all three
  columns on `base` (r2 did not move on that fixture's target; the summary
  is what the gate requires).
- The test module is `cd python && python3 -m mojolearn.tests.test_cpu_training_e2`
  (the kmeans checks and these in one file).

## Workstream E batch 2 (2026-09-14): spectral, SIMULATED IDENTICAL x3 ON base, GATE OWED

- spectral. The host oracle `spectral/checks/spectral_oracle.mojo`, the
  bit-for-bit reference the device Lanczos is gated against, MOVED to
  `spectral/host/spectral_oracle.mojo` (the checks file re-exports it and
  keeps the dense Float64 cross-check, whose Jacobi import the host binding
  does not carry); its imports are host modules only, with the three Lanczos
  clamps and `lanczos_v0` spelled in the host file rather than imported
  from the device Lanczos module, and `contract_leaf_size` from
  `gemm/host/gemm_oracle.mojo`. Around it the fit is restated from
  `spectral/impl/cluster/detail/spectral.mojo` and `spectral/impl/
  preprocessing/detail/spectral_embedding.mojo`: the k-NN self-join through
  `core/knn_host_predict.mojo::host_knn_search` at L2SqrtExpanded, the
  `(i, neighbor, 1.0)` COO, `coo_symmetrize_kernel` row by row over the
  zero-filled `2 nnz` output, `coo_sort` and `coo_remove_scalar(0)` (host
  code already), `oracle_embedding` (`norm_laplacian` true, `drop_first`
  false), then k-means on the row-major embedding exactly as
  `fit_predict_graph` sets it up (`plan_sum_scale` over the embedding,
  `choose_scale(n, n)`, unit weights, cuVS defaults with the seed, `n_init`
  and `oversampling_factor = 0.0`, the classic k-means++) through
  `cluster/host/kmeans_oracle.mojo::host_fit_main` and the fresh
  assignment. Exported as `spectral_fit_predict_dataset` from the metrics
  host binding under the GPU binding's name and params list;
  `spectral_fit_predict_graph` (the `spectral-precomputed` lane, absent
  from the 47-lane record) stays absent.
- On the M4's CPU-only path the lane reads IDENTICAL against the Apple,
  NVIDIA and AMD 47-lane columns on `base`, one fixture, one repeat, at the
  first build. The gate has not run.
- The sabotage arm is the family's `-D MOJOLEARN_HOST_SABOTAGE=1`, under
  which the recluster is seeded one draw off (on top of the core family's
  extra unit per quantized cell, which `host_fit_main` carries into this
  binding); on the M4 the sabotage set read DIVERGENT on `labels` against
  all three columns on `base`.
- The test module is `cd python && python3 -m mojolearn.tests.test_cpu_training_e2`.
