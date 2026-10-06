# AFCL production source integration scope

This is a source-reading and programming handoff for the 54 Apple FAST classical
cards. Status remains **source written, uncompiled, unverified and unmeasured**.
No builder, compiler, binary import, manifest/checker, test, GPU job or benchmark
was executed for this handoff. The source locations below identify intended
production dispatch; they are not observations that a compiled workload took it.

The existing production call paths consume the new constants or candidate branches.
Source tracing did not identify a missing kernel launch, module import or Python
ABI export that required widening an estimator's algorithm or adding a new public
entry point. Consequently this followup does not change numerical kernels or
defaults. It records the dependency and coverage boundaries needed by the paired
builder and full-workload runner integration. The original lane records retain
the exact mechanism, controls, caller scope and future quality obligations.

## Compilation units and package wiring

These controls are compile-time `is_defined` switches. Merely exporting a runtime
environment variable named `MOJOLEARN_AFCL_*` does not select a kernel. Supplying a
define with value `0` still selects an `is_defined` control; omission selects A.
FAST uses the existing build recipe's `MOJOLEARN_NUMERIC_MODE=fast`, with neither
`MOJOLEARN_NUMERIC_IDENTICAL` nor `MOJOLEARN_NUMERIC_DETERMINISTIC` defined. Every
new candidate additionally requires Apple in its source guard.

Existing GPU builder commands pass `MOJOLEARN_MOJO_BUILD_FLAGS` into `mojo build`
with repository and bindings include paths. Older builders also accept separate
channels: for example GP/kernel-methods/core use `MOJOLEARN_BUILD_EXTRA_DEFINES`,
and GBDT uses `MOJOLEARN_EXTRA_DEFINES`. A paired build must control those channels
together; an inherited define can preempt a route even when the emitted AFCL
configuration is correct. The shell builders contain import/smoke and artifact
checks after compilation. Reading them here does not authorize executing any stage.

The source modules are already imported through native binding entry points:

| Source area | Existing native entry points relevant to staging |
| --- | --- |
| Linear Gram, CV, Huber and SGD | `bindings/_mojolearn_x_linear.mojo`; L01/L02 also enter `bindings/_mojolearn_x_prep.mojo` through class-covariance/Gram programs |
| QN gradient and PCA covariance | `bindings/_mojolearn_estimators.mojo` imports `glm.estimator` and `decomposition.estimator` |
| Decomposition products, QR, NMF and robust covariance | `bindings/_mojolearn_x_decomp.mojo`; existing `x_decomp_fast_defines` export exposes the older QR/product route flags where the Python shell needs them |
| SMO | `bindings/_mojolearn_svm.mojo` and its imported solver |
| GP and random features | `bindings/_mojolearn_gp.mojo` and `bindings/_mojolearn_kernel_methods.mojo`; `gaussian_process/afcl_prediction.mojo` is imported by the existing GPR/GPC production modules |
| Core neighbors/KMeans and their consumers | `bindings/_mojolearn.mojo` plus extensions containing their transitive classical consumers; staging only one consumer is incomplete |
| DBSCAN/KDE and mixture | `bindings/_mojolearn_estimators.mojo` and `bindings/_mojolearn_mixture.mojo` |
| HDBSCAN, IVF and expanded clustering | `bindings/_mojolearn_hdbscan.mojo`, `bindings/_mojolearn_ivf.mojo`, `bindings/_mojolearn_x_cluster.mojo`, plus transitive hierarchy/manifold consumers |
| Boosting, RF, ExtraTrees/IsolationForest and SHAP | Existing GBDT/RF/trees/x_trees bindings; shared RF inference specialization stays guarded by `RF_INPUT` |
| Preprocessing and metrics | `bindings/_mojolearn_x_prep.mojo` and `bindings/_mojolearn_x_metrics.mojo`; existing planned-program/native exports remain the entrance |
| Resampling and statistical forecasting | `bindings/_mojolearn_resample.mojo`, `bindings/_mojolearn_arima.mojo`, `bindings/_mojolearn_tsa.mojo`; TSA imports both stationarity and Holt-Winters estimators |

This table describes source entrances, not a minimal complete extension dependency
set for every public estimator. A/B packages must stage a coherent affected native
extension set. For example, rebuilding only x_linear cannot establish L01/L02
coverage for LDA/QDA in x_prep. Likewise a downstream estimator can carry its own
compiled copy of an imported neighbor, hierarchy or product kernel. Existing
Python API dispatch and native exports suffice; no Python numerical runtime was
added. New module files are source imports, not dynamically registered plugins.

## Per-card production dispatch and exclusions

The listed path is the source implementation or its immediate production launcher.
Further caller and quality details are in the corresponding lane JSON/Markdown.

| Card | Production connection | Boundary that must remain explicit |
| --- | --- | --- |
| L01 | `x_linear/fast_gram.mojo`: chunk helpers feed `fast_gram_into` and `fast_sym_gram_into`; callers include ridge/device/ARD and x_prep | Existing unweighted/grid-Gram entrances only; weighted bypasses and rollback routes do not exercise it |
| L02 | The same module's `_fg_fold` is consumed by means, Gram/X'Y and symmetric final folds | Same broad transitive callers as L01; not limited to Ridge |
| L03 | `x_linear/enetcv_fast.mojo::ef_mse_kernel` is enqueued by the existing fast CV driver | `MOJOLEARN_X_LINEAR_ENETCV_FAST=1` in both arms; noneligible fold layouts retain fallback |
| L04 | `x_linear/huber_fast.mojo::HF_FOLD` drives partial count, scratch and resident solver launches | Resident Huber route and its baseline block512 policy stay enabled |
| L05 | `glm/impl/qn/fast_xtdz.mojo` workspace and `fast_xtdz_into` share the chunk/block constants; `glm_base` calls them | Disable the competing coalesced route in both arms; distributed and capacity fallbacks remain |
| L06 | `svm/impl/fast_update_f.mojo` shared-page constant sizes moved-row staging; `smosolver` calls `fast_update_f` | Only inherited fused RBF/linear capacity; other kernels and wider rows bypass it |
| L07 | `decomposition/impl/linalg/detail/pca.mojo::compute_covariance` allocates and launches compensated partial/finish kernels | Existing compensated covariance prerequisite in both arms; not default MMA covariance or every PCA solver |
| L08 | `x_decomp/device.mojo::launch_gemm` consumes `fast_gemm` tiled constants; `DKit.mm` and DevExec call it | Existing tiled-product prerequisite in both arms; ordered products bypass it |
| L09 | `fast_qr` reflector partials are called by DevExec geqrf/orgqr; the existing binding reports QR_FAST_DEV | Opt-in grid Householder route, not a new blocked-TSQR implementation |
| L10 | `x_decomp/device.mojo::launch_ew` dispatches OP_MUZ to the striped kernel; NMF's `DKit.ew3` reaches it | Frobenius multiplicative updates; CD and KL/IS formulas are outside this schedule |
| L11 | GPR `gp_predictive_variance` and GPC `gpc_latent_var_launch` import the same prediction-width helper | Variance/std/probability work only; fit, solves and mean-only prediction do not change |
| L12 | `kernel_methods/checks/random_features.mojo::km_feature_map_epilogue` launches the rows4 kernel | Separate RBFSampler epilogue only; already-fused projection, Nystroem and KernelRidge are excluded |
| L13 | Both initial and iterative `mf_select_kernel` launches in `x_decomp/mcd_fast.mojo` use MF_SELECT_TPB | Existing accepted batched MCD route; no revival of rejected covariance/eigensolve arms |
| L14 | `x_linear/device.mojo` row launch/witness helpers and `_sgd_mb_rows_kernel_body` share the row width | Separate minibatch row kernel only; per-sample, chunked minibatch and sparse routes are excluded |
| G01 | `neighbors/impl/detail/fast_mma_knn.mojo` MQ_SG/MQ_TPB feed narrow/big-feature launches and query ownership | Existing MMA eligibility and k/register limits remain; all affected downstream callers need coverage |
| G02 | `neighbors/impl/detail/fast_topk_knn.mojo` FKT_T feeds staged candidate traversal and existing merge pipeline | MMA must be off in both arms; slice count and final merge are unchanged |
| G03 | `dbscan/impl/neighbors/fast_mma_eps.mojo` ME_SG/ME_TPB feed `_launch_eps` and query groups | Existing epsilon MMA eligibility and scalar boundary fallback remain |
| G04 | `hdbscan/impl/detail/core_tile.mojo` imports AFCL_G04; CT_TILE controls storage and reference traversal | HDB_CORE_TILE enabled in both arms; normal kNN core-distance route does not exercise it |
| G05 | `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo` geometry is consumed by existing fused assignment launches | Exhaustive centroid assignment only; existing alignment/cost routing remains |
| G06 | `cluster/checks/reduce_by_key.mojo` BLOCK_ACC_ROWS feeds scratch count and row partitions; KMeans selects blocked accumulation | Existing experimental blocked route in both arms; legacy row overrides must be absent for the stated A |
| G07 | `x_cluster/minibatch_fast.mojo` MBF_TPB/MBF_CH and derived row groups feed resident launches | Resident route only; no new batch draws or training budget |
| G08 | `mixture/checks/estep.mojo` FE_TPB and GMM_ROW_TPB feed fused/default E-step launches | Explicit row_tpb overrides can bypass the ordinary default width |
| G09 | `ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo` imports FIVF_QPB for query grid and block width | Existing batched scan; fixed admission capacity and complete probes/lists remain |
| G10 | `umap/sparse_optimizer.mojo` imports FAST_OPT_TPB from `optimizer_fast` for ordinary and fused launches | FAST CSR/dense-graph optimizer routes only; graph and RNG contracts unchanged |
| G11 | `spectral/impl/sparse/solver/detail/lanczos.mojo::_spmv_fast` shares FL_SPMV_ROWS_PER_TG with its warp kernel | Existing long-row CSR SpMV route; LLE's separate dense solver is excluded |
| G12 | `x_cluster/meanshift_fast.mojo` MSG_T controls seed-row chunks and scratch/final folds | Existing grid MeanShift; every bandwidth-tested input remains covered |
| G13 | `kde/impl/neighbors/kernel_density.mojo` KDE2_QM controls dimension-tiled query groups and accumulator banks | Existing stable tiled FAST score path only; full train contributions retained |
| G14 | `hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo` MB_T is consumed by MmaBoruvka search | Euclidean MMA MST route and HDBSCAN consumers; unsupported metrics/linkages stay refused |
| T01 | Quantized histogram kernel and launcher import QH_MIN_ITEMS_PER_BLOCK | One-byte/two-stat Depthwise/Lossguide family; symmetric histogram is excluded |
| T02 | Quantized histogram kernel and launcher share QH_GROUP_FEATURES and derived words/shared allocation | Same nonsymmetric family; sub-byte/multistat fallbacks are excluded |
| T03 | GBDT train/resident modules import BINARIZE_BLOCK_SIZE for per-feature binarization | Packed index/predict paths bypass it; keep those opt-in/umbrella flags absent when using this arm |
| T04 | `gbdt/methods/pointwise_optimization_subsets.mojo` SPLIT_BLOCK_SIZE feeds bin/partition metadata launches | Symmetric metadata/bin scheduling; physical radix partition and nonsymmetric partitioners unchanged |
| T05 | `gbdt/methods/kernel/pointwise_split_resolve.mojo::_fold_block` consumes AFCL_RESOLVE_RECORDS | SYM_RESOLVE_BLOCK enabled in both arms; fused level/pointwise entrances must remain enabled |
| T06 | `gbdt/methods/leaves_estimation/apple_fast_est.mojo` uses `_afcl_leaf_stats_sm` for scratch and statistic launches | Existing admitted Apple leaf objectives only; multiclass/multi-RMSE/group/ranking exclusions remain |
| T07 | `gbdt/resident_model.mojo` consumes AFCL_PREDICT_BLOCK for oblivious model row grids/launches | Resident oblivious application only; nonresident/nonsymmetric routes are excluded |
| T08 | `core/forest_inference.mojo::_afcl_rf_block[RF_INPUT]` feeds traversal and grove geometry | RF_INPUT=True only; ExtraTrees, legacy host and OOB training paths retain their schedules |
| T09 | ExtraTrees builder uses ET_FEATURE_TILE in tiled range/regression score kernels and feature grids | Existing tiled row-major eligibility only; classification score family is not replaced |
| T10 | Isolation tree builder's IF_PATH_TPB feeds default IFLaunchKnobs and path/score launches | Explicit path_tpb overrides can bypass it; tree construction is unchanged |
| T11 | `xtrees/shap_device.mojo` query helper and all row/fold launches use QUERY_TPB | Preparation/table construction and unsupported interaction APIs remain unchanged |
| T12 | `gbdt/ctrs/kernel/ctr_calcers.mojo` CTR_BLOCK_SIZE feeds elementwise/count strides and callers | Only callers using these CTR launchers; independent CTR frequency/sort routes retain their schedules |
| P01 | x_prep imports fastred/fastpt; both two-pass sums and current cs_tile Welford paths consume AFCL_P01 | OP_COL_STATS family, not separate core scaler kernels or every statistic kernel |
| P02 | Existing x_prep MaxAbs direct ABI calls `fastmaxabs` row-chunk helpers | PREP3_MAXABS route must remain enabled; pools/lifetime unchanged |
| P03 | x_prep device dispatch imports NB_CAT_ATOMIC and the candidate count width | NB_CAT_ATOMIC enabled in both arms; weighted fallback excluded |
| P04 | x_prep binding exports existing CSR fit functions from `fastnb_csr` | Multinomial/Complement CSR fit; dense, Bernoulli and prediction traversals excluded |
| P05 | `prep2_fast_stage` uses TE_TGR for global/gathered-category kernels and their local storage | TE_GLOBAL/TE_ENC enabled in both; nongathered fallback remains |
| P06 | `prep2_fast_stage` calls `ii_conv_fast_kernel` with II_CONV_TGR when hq[7] identifies row sums | Runtime II_CONV enabled in both; no row-sum availability claim for bypassing plans |
| P07 | `select_fast` shared row-tile constants feed feature-statistic scratch and launches | Existing f/r regression and class-statistic selector routes; large-class fallback remains |
| P08 | `x_metrics/reg_epi.mojo` KIND_AVG consumes the guarded independent binary64 chains | Multioutput regression averaging only; scalar-output metric folds do not change |
| P09 | `x_metrics/plan.mojo` emits matching curve-prefix chunk parameters consumed by the planned runtime | Prefix staging only; sort/tie grouping and bounded-plan fallback remain |
| P10 | `resample/estimator.mojo` imports GATHER_COLS/GATHER_ROWS for the tiled float32 gather launch | Existing gather/wait/tiled prerequisites in both; nonreplacement needs device permutation too |
| P11 | `tsa/impl/timeSeries/stationarity.mojo::series_sum_kernel` feeds KPSS/differencing callers | Series sums/squares only; other stationarity algorithms and scan policies unchanged |
| P12 | KALMAN_TPB propagates through batched ARIMA, estimator and fast retained-evaluation workspace | Scalar-lane Kalman default only; explicit widths/time-scan routes remain separate |
| P13 | x_prep fast-fold dispatch reaches `fastred` unweighted class statistics | GaussianNB/OP_CLASS_STATS route only; native naive_bayes and weighted alternatives excluded |
| P14 | Holt-Winters estimator imports HW_OPTIM_TPB for default optimization/evaluation helpers | Explicit optimizer widths and HW_PREDICT_TPB retain their behavior |

## A/B prerequisite and interaction rules

The lane manifests already encode these matched prerequisites. They must reach
every applicable compilation unit in both packages:

| Card | Matched prerequisite |
| --- | --- |
| L03 | Runtime `MOJOLEARN_X_LINEAR_ENETCV_FAST=1` |
| L05 | Define `MOJOLEARN_QN_FAST_COALESCED_OFF`; keep QN_FAST_XTDZ enabled |
| L07 | Define `MOJOLEARN_PCA_FAST_COMPENSATED_COV` |
| L08 | Define `MOJOLEARN_DECOMP_FAST_GEMM_TILED` |
| L09 | Define `MOJOLEARN_QR_FAST_DEV` |
| G02 | Define `MOJOLEARN_KNN_FAST_MMA_OFF`; keep scalar top-k enabled |
| G04 | Define `MOJOLEARN_HDB_CORE_TILE` |
| G06 | Define `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC`; keep unrelated accumulator overrides absent |
| T05 | Define `MOJOLEARN_SYM_RESOLVE_BLOCK`; retain fused level/pointwise routes |
| P03 | Define `MOJOLEARN_NB_CAT_ATOMIC` |
| P06 | Runtime `MOJOLEARN_X_PREP_FAST_II_CONV=1` and an existing row-sum plan |
| P10 | Defines `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`, `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER`; nonreplacement also requires `MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE` |

G01 and G02 are competing routes: G02's required MMA-off define prevents G01's
MMA geometry from running. A combined configuration must not claim both. Existing
unqualified prerequisites are not promoted by using them in an A/B. L07/L08/L09,
G04/G06, T05 and P03/P10 therefore compare schedules inside their stated route,
not automatically against the current production default.

Keep legacy rollback/umbrella controls and explicit launch widths consistent.
In particular, packed GBDT index/predict flags or `MOJOLEARN_SYM_FEAT_ALL` can
bypass T03; `MOJOLEARN_RBF_FUSED` can bypass L12's nonresident epilogue; and the
`*_OFF` flags named in lane records can disable an inherited route entirely.
The configuration selector's source presently records the numeric-mode exclusions
and G01/G02 conflict; geometry lane manifests also carry
`defines_absent_in_both_arms`. Build integration must consume or enforce those
requirements rather than accepting contradictory inherited compiler flags.

## Scratch, launch and ABI lifetime boundaries

The source schedules use coupled planning/launch state rather than independent
new buffer sizes: Gram chunks and `fg_part_words`; QN workspace and launch partial
count; PCA's explicitly allocated partial table and compensated finish; QR's
`fq_dot_blocks`; MCD selection's shared page and launch width; SGD's row-grid
helper and witness offsets; kNN/IVF/UMAP query ownership and imported widths;
KMeans/MeanShift row partial helpers; histogram feature groups and compressed
words; RF grove row staging; TargetEncoder local storage; resample gather tile;
and Kalman defaults in retained-evaluation entrances.

This coupling explains the intended source integration. It does not prove scratch
bounds, supported threadgroup sizes, code generation, kernel execution or numerical
behavior. Tail shapes, live model/output lifetimes, weighted/empty/nonfinite inputs,
quality thresholds and full-workload boundaries remain future qualification work.
No existing ABI signature or ownership protocol was changed to introduce these
controls; no new Python data loops or numerical fallback was added.

## Remaining work and evidence status

There is no source-discovered missing production connection requiring a kernel or
ABI edit in this handoff. Narrower implemented scope is deliberate and visible;
it must not be silently counted as every estimator named in an original broad
idea. The paired builder and full-workload runner integration are authored by
their respective lanes, and all of those tools also remain unexecuted.

Runtime reach is unproven for all 54 cards until separately authorized execution.
Before any future timing, map actual full dataset/version/hash/dimensions/settings,
resolve intrinsic caps, use a coherent frozen source/binary package, and retain
per-estimator task quality with the complete operation boundary. Do not infer
default eligibility from source plumbing or reuse unrelated component results.

Evidence retained here consists of this source-integration scope record and the
four lane manifests/notes. There are no build, validation, timing or GPU logs
because those actions were not run. The parent owns commit and push.
