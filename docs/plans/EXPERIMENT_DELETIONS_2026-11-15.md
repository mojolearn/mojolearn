# Experiment defines: inventory and owed deletions (2026-11-15)

Read-only inventory of every compile-time `MOJOLEARN_*` define read in Mojo code (`is_defined[...]` and `get_defined_*[...]`) on `origin/preinteg/wave2-20261113` @ fcc934a37c, joined with `docs/apple-fast/EXPERIMENTS.md`, `experiments/six_lane_integration/grid_controls/*.json`, `docs/TOMBSTONES.md`, `experiments/removed/*.patch`, the refusal guards in `core/experiment_guards_*.mojo` / `core/six_lane_experiment_guards.mojo`, and the `#` comments within 8 lines of each read site. Rules applied: AGENTS.md "Experiments", "Deleting an experiment" (2026-11-14: a decided A/B whose losing arm or superseded old path is still in code is an owed deletion), "A measured winner is never deleted on reasoning". Protected and never pruned: `experiments/removed/`, `docs/TOMBSTONES.md`, `docs/apple-fast/EXPERIMENTS.md`.

Scripts and raw joins: `~/mojolearn-evidence/briefs-20261114/experiment-decisions-inventory/` (`scripts/scan.py`, `scripts/decide.py`, `scripts/report.py`; `scan.json` holds every read site with file:line, `decide.json` every define with its full site list). The scan ran on 10 of 10 laptop cores (5,290 `.mojo` files split across workers).

## Totals

| | count |
|---|---|
| defines read in Mojo | 2345 |
| live defines (read outside the refusal guards) | 1948 |
| read sites: live / refusal-guard | 2732 / 561 (total 3293) |
| uses of a define through a `comptime` alias | 10436 |
| DELETE-LOSER | 27 defines, 28 live sites, 295 branch lines |
| DELETE-OLD-PATH | 81 defines, 81 live sites, 439 branch lines |
| KEEP-OPEN | 1300 defines, 1932 live sites, 18290 branch lines |
| KEEP | 540 defines, 691 live sites, 7079 branch lines |
| REMOVED | 397 defines, 0 live sites, 0 branch lines |

"Branch lines" is a lower bound: the lines of the `if` arm taken when the define is set (or the `else` of an `if not`), plus the condition and `else` lines; a compound condition counts 1. Helper functions reachable only from the deleted arm are not counted; the deletion lane removes them too and reports the real count.

Decisions:

- **DELETE-LOSER**: default-off arm whose own record (EXPERIMENTS row naming it first, or its grid control) is a loser (DROPPED / NEUTRAL / SUPERSEDED / removed). Default unchanged: no bits change, no board row needs re-timing.
- **DELETE-OLD-PATH**: `_OFF` / legacy switch whose new default won its record (KEPT / PROMOTED / DEFAULT); for `identical` the record names both NVIDIA and AMD, `fast` (Apple) records are M3 A/Bs. The default path stays; the old path goes. No bits change, no board row needs re-timing.
- **KEEP-OPEN**: no decisive record for this define (owed A/B named in the reason), a record that names one vendor only, a record that is indirect, or a record that contradicts the code (promoted but still opt-in).
- **KEEP**: not an experiment (sabotage arm, check, build or mode control), read only by checks/tests/benches, or Apple `fast` opt-in code with no verdict (Apple fast is frozen, Andrew 2026-10-22: it may stay opt-in, no A/B will run).
- **REMOVED**: only its refusal guard (or a compatibility assert) remains: already deleted.

Re-timing: every DELETE in this plan removes a non-default arm, so no row of `bench/results` needs re-timing; no board recipe sets a define (checked: `bench/` sets none outside A/B scripts). KEEP-OPEN rows owe an A/B on NVIDIA L40S (AMD at the release campaign's next AMD leg) on the rows of the estimators their family covers.

## Biggest owed deletions

| define | decision | branch lines | live sites | evidence |
|---|---|---|---|---|
| `MOJOLEARN_GBDT_EVAL_FUSED` | DELETE-LOSER | 122 | 1 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE |
| `MOJOLEARN_GBDT_PREDICT_PACKED` | KEEP-OPEN | 50 | 1 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:171) but the prerequisite of the unmeasured AFT_P05 card (core/apple_fast_tree_experiments.mojo, P.json: PREDICT_PACKED in both arms; AFT_P05 code sits in the packed `_apply_all` path); orchestrator decides (delete P05 with it, or keep) |
| `MOJOLEARN_SVGP_FAST_BSPLIT_OFF` | DELETE-OLD-PATH | 26 | 1 | EXP:1424 KEEP (FAST+Apple default, `_OFF`) |
| `MOJOLEARN_GBDT_QUANT_DEVICE` | KEEP-OPEN | 18 | 1 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:172) but implied by GBDT_INDEX_PACK_DEVICE (G06 prerequisite) and held equal in G05; goes with INDEX_PACK_DEVICE |
| `MOJOLEARN_IDN_SEQ_REG_OFF` | DELETE-OLD-PATH | 16 | 1 | grid seq-reg.json promoted:lane/i17-decide-linalg-seq |
| `MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF` | DELETE-OLD-PATH | 15 | 1 | grid samba-resident.json promoted:ge123e6f9 |
| `MOJOLEARN_TSVD_FAST_POOL_OFF` | DELETE-OLD-PATH | 14 | 1 | EXP:1417 KEEP (FAST+Apple default, `_OFF`) |
| `MOJOLEARN_AF_FAST_NOFILL_OFF` | DELETE-OLD-PATH | 13 | 1 | EXP:1377 KEEP (FAST+Apple default, `_OFF`) |
| `MOJOLEARN_AF_FAST_RESIDENT_OFF` | DELETE-OLD-PATH | 13 | 1 | EXP:1376 KEEP (FAST+Apple default, `_OFF`) |
| `MOJOLEARN_AP_EXACT` | DELETE-LOSER | 13 | 2 | EXP:495 SUPERSEDED |

## Deletion lanes

Each lane owns every code file and grid json its defines touch; no code or grid file is in two lanes (defines that share a file are in the same lane). The shared append-only records (`docs/TOMBSTONES.md`, `docs/apple-fast/EXPERIMENTS.md`, the family guard module `core/experiment_guards_<family>.mojo`) are appended by every lane in its own dated section at the end of the file (guard module: its own block at the end of `_check_configuration`); the integrator keeps both sides of an append conflict. Every deletion carries the four tombstone parts: inline `# TOMBSTONE:` + `# Restore:` at each site, the TOMBSTONES.md index row and section, `experiments/removed/<DEFINE>.patch` (reverse diff, code files only), and the `removed` refusal, grid entry moved to `removed`, EXPERIMENTS.md row. Before deleting, the lane reads the evidence row named here; a row that does not say what this plan says moves the define to KEEP-OPEN in the lane report, it is not deleted. A lane deletes the losing branch and every helper reachable only from it; it never touches the default arm, a hand-tuned kernel or verify code.

### expdel-01-cluster (20 defines, 134 branch lines, 17 files)

Defines: `MOJOLEARN_ANN_FAST_KNN_BIGD_OFF` (old-path), `MOJOLEARN_AP_EXACT` (loser), `MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF` (old-path), `MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF` (old-path), `MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF` (old-path), `MOJOLEARN_CAGRA_FAST_IVFG_OFF` (old-path), `MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF` (old-path), `MOJOLEARN_MBK_FAST_DEVSCAN_OFF` (old-path), `MOJOLEARN_SVGP_FAST_BLKCHOL_OFF` (old-path), `MOJOLEARN_SVGP_FAST_BSPLIT_OFF` (old-path), `MOJOLEARN_SVGP_FAST_COLSPLIT_OFF` (old-path), `MOJOLEARN_SVGP_FAST_RBFTILE_OFF` (old-path), `MOJOLEARN_SVGP_FAST_SYMTILE_OFF` (old-path), `MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF` (old-path), `MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF` (old-path), `MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF` (old-path), `MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF` (old-path), `MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF` (old-path), `MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF` (old-path), `MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF` (old-path)

Files: `bindings/_mojolearn_x_neighbors.mojo`, `x_ann/cagra_device.mojo`, `x_ann/fast_env.mojo`, `x_ann/ivf_pq_device.mojo`, `x_ann/knn_device.mojo`, `x_cluster/affinity.mojo`, `x_cluster/bisect_fast.mojo`, `x_cluster/device_execution.mojo`, `x_cluster/device_ops.mojo`, `x_cluster/meanshift_fast.mojo`, `x_cluster/minibatch.mojo`, `x_cluster/minibatch_fast.mojo`, `x_cluster/minibatch_ptr.mojo`, `x_neighbors/iter_device.mojo`, `x_neighbors/kfeat_dev.mojo`, `x_neighbors/ocsvm_dev.mojo`, `x_neighbors/ocsvm_resident_native_device.mojo`

### expdel-02-decomposition-series (18 defines, 86 branch lines, 19 files)

Lane result (2026-11-15, lane/expdel-02-decomposition-series): 17 deleted with the four tombstone parts; `MOJOLEARN_CHOL_FAST_POOLIO_OFF` moved to KEEP-OPEN (its row below: the FAST record does not cover the IDENTICAL default it also controls).

Defines: `MOJOLEARN_ARIMA_FAST_EVAL_WS_OFF` (old-path), `MOJOLEARN_ARIMA_FAST_LLONLY_OFF` (old-path), `MOJOLEARN_CHOL_FAST_POOLIO_OFF` (old-path), `MOJOLEARN_CV_FAST_SLICE_OFF` (old-path), `MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF` (old-path), `MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF` (old-path), `MOJOLEARN_DECOMP_FAST_LASSO_GRP_OFF` (old-path), `MOJOLEARN_DECOMP_FAST_ORTH_WS_OFF` (old-path), `MOJOLEARN_FA_FAST_QRR_OFF` (old-path), `MOJOLEARN_FA_TRANSFORM_FUSED` (loser), `MOJOLEARN_IDN_LU_RESIDENT_PAIR_OFF` (old-path), `MOJOLEARN_IPCA_FAST_DEV_OFF` (old-path), `MOJOLEARN_PCA_FAST_GRAM_MMA_OFF` (old-path), `MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF` (old-path), `MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_OFF` (old-path), `MOJOLEARN_TSA_FAST_SELD_FUSED_OFF` (old-path), `MOJOLEARN_TSVD_FAST_POOL_OFF` (old-path), `MOJOLEARN_XD_FAST_GRP_FUSED_OFF` (old-path)

Files: `arima/impl/batched_fit.mojo`, `arima/impl/batched_kalman.mojo`, `arima/impl/fast_order_state.mojo`, `bindings/_mojolearn_x_decomp.mojo`, `cholesky/estimator.mojo`, `decomposition/estimator.mojo`, `decomposition/impl/linalg/detail/pca.mojo`, `experiments/six_lane_integration/grid_controls/lu-resident-pair.json`, `resample/estimator.mojo`, `tsa/impl/timeSeries/kpss_fused.mojo`, `x_decomp/device.mojo`, `x_decomp/dict_fast.mojo`, `x_decomp/fa_fast.mojo`, `x_decomp/kit_device.mojo`, `x_decomp/lanczos_dev.mojo`, `x_decomp/lasso_grp.mojo`, `x_decomp/pca_randomized_dev.mojo`, `x_decomp/resident.mojo`, `x_decomp/s_linalg_fast.mojo`

### expdel-03-linear (20 defines, 79 branch lines, 14 files)

Outcome (lane/expdel-03-linear, 2026-11-15): 19 deleted; `MOJOLEARN_SGD_FAST_PS_SIMD_OFF` moved to KEEP-OPEN (reason in its row below).

Defines: `MOJOLEARN_ARD_FAST_CLS1_BATCH_OFF` (old-path), `MOJOLEARN_ARD_FAST_CLS1_PARTS_OFF` (old-path), `MOJOLEARN_BAYES_FAST_CLS1_BATCH_OFF` (old-path), `MOJOLEARN_BAYES_FAST_CLS1_PARTS_OFF` (old-path), `MOJOLEARN_CD_FAST_GRID_GRAM_OFF` (old-path), `MOJOLEARN_CD_FAST_ROWMAJOR_OFF` (old-path), `MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF` (old-path), `MOJOLEARN_HUBER_FAST_BLOCK512_OFF` (old-path), `MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF` (old-path), `MOJOLEARN_PREP_FAST_MINMAX_OFF` (old-path), `MOJOLEARN_QN_FAST_BLOCKS_OFF` (old-path), `MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF` (old-path), `MOJOLEARN_SGDOC_FAST_TAIL_OFF` (old-path), `MOJOLEARN_SGD_FAST_PS_SIMD_OFF` (old-path), `MOJOLEARN_SGD_PERC_AVG` (loser), `MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF` (old-path), `MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF` (old-path), `MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF` (old-path), `MOJOLEARN_X_PREP_FAST_TE_ENC_OFF` (old-path), `MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF` (old-path)

Files: `bindings/_mojolearn_estimators.mojo`, `glm/impl/qn/glm_base.mojo`, `glm/linear_fit_device.mojo`, `glm/linear_route_controls.mojo`, `preprocessing/minmax.mojo`, `solver/estimator.mojo`, `solver/impl/cd.mojo`, `x_linear/cls1_fast.mojo`, `x_linear/device.mojo`, `x_linear/fast_gram.mojo`, `x_linear/huber_fast.mojo`, `x_linear/sgd.mojo`, `x_prep/device.mojo`, `x_prep/fastprep2.mojo`

### expdel-04-neural (14 defines, 105 branch lines, 18 files)

Result (lane expdel-04-neural, 2026-11-15): 10 deleted (AFN_OPTIM_ALL, AF_FAST_NOFILL_OFF, LN_FAST_NOFILL_OFF, SEQ_FAST_PIPE_UP_OFF, SEQ_FAST_PIPE_DOWN_OFF, SEQ_FAST_THETA_SNAP_OFF, SEQ_FAST_THETA_SPEC_OFF, IDN_SEQ_REG_OFF, SEQ_FAST_VAR_COOP_OFF, IDN_SAMBA_RESIDENT_STEP_OFF); 4 moved to KEEP-OPEN after reading their evidence (ATTN_DQ_MFMA, ATTN_DKDV_MFMA, IDN_ATTN_TILE_ORDER, NI12_IMPLICIT_CONV_OFF: see their rows). Most old paths stay because other builds (IDENTICAL, FAST off Apple, IDN_ALL_OFF, CHUNKED_LM_HEAD_V2) still compile them; only the switch term goes.

Defines: `MOJOLEARN_AFN_OPTIM_ALL` (loser), `MOJOLEARN_AF_FAST_NOFILL_OFF` (old-path), `MOJOLEARN_ATTN_DKDV_MFMA` (loser), `MOJOLEARN_ATTN_DQ_MFMA` (loser), `MOJOLEARN_IDN_ATTN_TILE_ORDER` (loser), `MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF` (old-path), `MOJOLEARN_IDN_SEQ_REG_OFF` (old-path), `MOJOLEARN_LN_FAST_NOFILL_OFF` (old-path), `MOJOLEARN_NI12_IMPLICIT_CONV_OFF` (old-path), `MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF` (old-path), `MOJOLEARN_SEQ_FAST_PIPE_UP_OFF` (old-path), `MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF` (old-path), `MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF` (old-path), `MOJOLEARN_SEQ_FAST_VAR_COOP_OFF` (old-path)

Files: `autograd/program_output_copy.mojo`, `bindings/_mojolearn_training.mojo`, `experiments/six_lane_integration/grid_controls/neural-seq-train-dedupe.json`, `experiments/six_lane_integration/grid_controls/neural-small.json`, `experiments/six_lane_integration/grid_controls/samba-resident.json`, `experiments/six_lane_integration/grid_controls/seq-reg.json`, `sequence/exec_device.mojo`, `sequence/ops.mojo`, `sequence/ops_impl.mojo`, `sequence/theta.mojo`, `sequence/theta_spec.mojo`, `training/afn_optim.mojo`, `training/native_primitive_dispatch.mojo`, `training/native_resident_dispatch.mojo`, `training/neural_identical_experiments.mojo`, `training/samba_resident.mojo`, `transformer/impl/llama/fused_attention.mojo`, `x_cnn/device.mojo`

### expdel-05-trees (18 defines, 243 branch lines, 13 files)

Defines: `MOJOLEARN_CTR_ONEHOT_DEVICE` (loser), `MOJOLEARN_CTR_PREP_SHARED` (loser), `MOJOLEARN_EST_ITERS_DEVICE` (loser), `MOJOLEARN_GBDT_BOOT_DEVICE` (loser), `MOJOLEARN_GBDT_EVAL_FUSED` (loser), `MOJOLEARN_GBDT_EVAL_SKIP_EMPTY` (loser), `MOJOLEARN_GBDT_INDEX_PACK_DEVICE` (loser), `MOJOLEARN_GBDT_PREDICT_PACKED` (loser), `MOJOLEARN_GBDT_QUANT_DEVICE` (loser), `MOJOLEARN_KSHAP_FAST_BATCH_OFF` (old-path), `MOJOLEARN_SYM_BUF_ARENA` (loser), `MOJOLEARN_SYM_FEAT_ALL` (loser), `MOJOLEARN_SYM_HIST_ALL` (loser), `MOJOLEARN_SYM_ITER_ALL` (loser), `MOJOLEARN_SYM_MULTI_ALL` (loser), `MOJOLEARN_SYM_RESOLVE_BLOCK` (loser), `MOJOLEARN_SYM_SORT_SWAP` (loser), `MOJOLEARN_YR_TASK_FUSED` (loser)

Files: `bindings/_mojolearn_gbdt.mojo`, `gbdt/ctrs/fast_prep.mojo`, `gbdt/estimator.mojo`, `gbdt/gpu_data/sym_feat_switches.mojo`, `gbdt/grid_creator/gls_borders_device.mojo`, `gbdt/methods/doc_parallel_boosting.mojo`, `gbdt/methods/kernel/pointwise_split_resolve.mojo`, `gbdt/methods/kernel/sym_fast.mojo`, `gbdt/methods/leaves_estimation/apple_fast_est.mojo`, `gbdt/methods/sym_iter_fast.mojo`, `gbdt/resident_model.mojo`, `gbdt/targets/kernel/yeti_rank.mojo`, `xtrees/agnostic_device.mojo`

Lane result (expdel-05-trees, 2026-11-15): 11 deleted (`SYM_FEAT_ALL`, `GBDT_BOOT_DEVICE`, `GBDT_EVAL_SKIP_EMPTY`,
`GBDT_EVAL_FUSED`, `SYM_HIST_ALL`, `SYM_SORT_SWAP`, `SYM_ITER_ALL`, `SYM_BUF_ARENA`, `SYM_MULTI_ALL`, `YR_TASK_FUSED`,
`KSHAP_FAST_BATCH_OFF`). 7 moved to KEEP-OPEN after reading the evidence rows: `CTR_ONEHOT_DEVICE`, `CTR_PREP_SHARED`,
`EST_ITERS_DEVICE` are default ON through their umbrellas `SYM_CTR_ALL` / `SYM_EST_ALL` (EXP:160/161/163 "KEPT/DEFAULT via",
EXP:1303 "ride their umbrellas, now on"), so they are winners, not losers; `GBDT_INDEX_PACK_DEVICE` + `GBDT_QUANT_DEVICE`,
`GBDT_PREDICT_PACKED` and `SYM_RESOLVE_BLOCK` lost (NEUTRAL) but are the prerequisites of the unmeasured Apple FAST cards
AFT_G06, AFT_P05 and AFCL_T05 (both arms); deleting them removes those cards' arms, which is the orchestrator's call.

### expdel-06-base-cluster-kernel (18 defines, 87 branch lines, 19 files)

Defines: `MOJOLEARN_AF_FAST_RESIDENT_OFF` (old-path), `MOJOLEARN_APPLE_FAST_GEMM_TN_V1_OFF` (old-path), `MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` (loser), `MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF` (old-path), `MOJOLEARN_KDE_DIMTILE` (loser), `MOJOLEARN_KDE_FAST_SLICES_OFF` (old-path), `MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF` (old-path), `MOJOLEARN_KMEANS_LAZY_SHIFT_IDN_OFF` (old-path), `MOJOLEARN_KMEANS_LIVE_CORES_OFF` (old-path), `MOJOLEARN_KM_FAST_PTR_IN_OFF` (old-path), `MOJOLEARN_KM_FAST_RBF_PIPE_OFF` (old-path), `MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF` (old-path), `MOJOLEARN_KM_FAST_RBF_STAGED_OFF` (old-path), `MOJOLEARN_KNN_FAST_MMA_K64_OFF` (old-path), `MOJOLEARN_KNN_FAST_REFINE` (loser), `MOJOLEARN_SPLINE_FAST_FUSED_OFF` (old-path), `MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF` (old-path), `MOJOLEARN_X_PREP_FAST_UNIQUE_OFF` (old-path)

Files: `bindings/_mojolearn_kernel_methods.mojo`, `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_x_sequence.mojo`, `cluster/host/kmeans_oracle.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `cluster/impl/kmeans_params.mojo`, `core/gemm.mojo`, `experiments/six_lane_integration/grid_controls/classical-fixes.json`, `experiments/six_lane_integration/grid_controls/fg-ivf.json`, `experiments/six_lane_integration/grid_controls/gap-ivf.json`, `hdbscan/impl/detail/idn_switches.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo`, `kde/impl/neighbors/kernel_density.mojo`, `kernel_methods/estimator.mojo`, `kernel_methods/rbf_resident.mojo`, `neighbors/estimator.mojo`, `neighbors/impl/detail/fast_mma_knn.mojo`, `sequence/opt_resident.mojo`, `x_neighbors/nan_cells_device.mojo`

Lane result (expdel-06-base-cluster-kernel, 2026-11-15): 14 deleted (`AF_FAST_RESIDENT_OFF`, `APPLE_FAST_GEMM_TN_V1_OFF`,
`IVF_FAST_SEED_DEVICE_OFF`, `KDE_FAST_SLICES_OFF`, `KERNEL_FAST_NYS_RR_EIGH_OFF`, `KM_FAST_PTR_IN_OFF`, `KM_FAST_RBF_PIPE_OFF`,
`KM_FAST_RBF_STAGED_OFF`, `KM_FAST_RBF_RESIDENT_OFF`, `KNN_FAST_MMA_K64_OFF`, `KNN_FAST_REFINE`, `X_PREP_FAST_UNIQUE_OFF`,
`SPLINE_FAST_FUSED_OFF`, `XN_FAST_NAN_COLMISS_ONLY_OFF`). 4 moved to KEEP-OPEN after reading the evidence rows:
`IDN_HDB_SPARSE_MIN_ROWS` (unmeasured threshold candidate, not a loser), `KDE_DIMTILE` (its own row EXP:339 is a KEEP; the DROP
rows are combos), `KMEANS_LAZY_SHIFT_IDN_OFF` and `KMEANS_LIVE_CORES_OFF` (IDENTICAL `new default` by plan, no NVIDIA + AMD A/B;
EXP:1671 PENDING). No grid_controls file was touched (only the KEEP-OPEN defines have entries).

## What stays OPEN and why

| reason | defines |
|---|---|
| no verdict recorded: owed A/B default vs the opt-in arm | 668 |
| no verdict recorded: owed A/B default vs the old path | 391 |
| record is indirect: confirm the verdict, then delete or A/B | 160 |
| promoted on one vendor in the record: owed the other vendor before the old path goes | 60 |
| record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) | 12 |
| old-path switch whose record reads as dropped: reconcile which arm lost | 9 |

KEEP-OPEN by family: neural 394, trees 272, cluster 187, base 183, linear 114, decomposition 72, series 44, kernel 34. Of these, 50 are the neural MERGED-UNMEASURED opt-ins (Andrew, Oct 3: merged default-off without an A/B, "don't lose the work"); they stay until their A/B runs.

## Every define

Columns: verdict = evidence class (WIN / LOSE / OPEN / none) and the newest evidence row; sites = live read sites (+ alias uses) / guard sites; lines = branch lines deletable; tomb = tombstone parts already present (i inline, c central TOMBSTONES.md mention, p patch, r refusal); first site = the first live read site (the full list per define is in `decide.json`).

### base (395)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_AF_FAST_RESIDENT_OFF` | DELETE-OLD-PATH | WIN EXP:1376 KEEP (FAST+Apple default, `_OFF`) | 1+9/0 | 13 | - | sequence/opt_resident.mojo:110 | expdel-06 |
| `MOJOLEARN_APPLE_FAST_GEMM_TN_V1_OFF` | DELETE-OLD-PATH | WIN EXP:415 KEPT 7484503b6 | 1+2/0 | 3 | - | core/gemm.mojo:130 | expdel-06 |
| `MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF` | DELETE-OLD-PATH | WIN EXP:858 DEFAULT (FAST+Apple), rollback `MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF`: M3 one run  | 1+4/0 | 5 | - | kernel_methods/rbf_resident.mojo:71 | expdel-06 |
| `MOJOLEARN_SPLINE_FAST_FUSED_OFF` | DELETE-OLD-PATH | WIN EXP:695 KEPT | 1+3/0 | 4 | - | bindings/_mojolearn_x_prep.mojo:213 | expdel-06 |
| `MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF` | DELETE-OLD-PATH | WIN EXP:870 DEFAULT (FAST+Apple), rollback MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF (needs SCHI2_MO | 1+3/0 | 4 | - | x_neighbors/kfeat_dev.mojo:74 | expdel-01 |
| `MOJOLEARN_X_PREP_FAST_UNIQUE_OFF` | DELETE-OLD-PATH | WIN EXP:369 KEPT f419ea9f1 | 1+3/0 | 4 | - | bindings/_mojolearn_x_prep.mojo:198 | expdel-06 |
| `MOJOLEARN_134_CONTROL` | KEEP-OPEN | none  | 2+3/0 | 56 | - | checks/determinism_soak.mojo:97 | owed A/B: the default vs MOJOLEARN_134_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_2030_FUSED_EST_MOVE` | KEEP-OPEN | LOSE EXP:1611 DROP_REDUNDANT, alias deleted | 1+6/0 | 45 | c | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:207 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_2031_SYM_RIDX_SPLITS` | KEEP-OPEN | none  | 1+8/0 | 22 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:572 | owed A/B: the default vs MOJOLEARN_2031_SYM_RIDX_SPLITS on NV+AMD, rows of its family |
| `MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF` | KEEP-OPEN | none  | 1+8/0 | 28 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:575 | owed A/B: default vs the old path MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_2560_MEMSET_FILL` | KEEP-OPEN | none  | 2+2/0 | 8 | - | core/device_zero.mojo:36 | owed A/B: the default vs MOJOLEARN_2560_MEMSET_FILL on NV+AMD, rows of its family |
| `MOJOLEARN_2711_GRAM_STRIDED_DEAD` | KEEP-OPEN | none  | 1+3/0 | 4 | - | core/gram_splitk.mojo:225 | owed A/B: the default vs MOJOLEARN_2711_GRAM_STRIDED_DEAD on NV+AMD, rows of its family |
| `MOJOLEARN_537_GEMM_IDENT_SWAP` | KEEP-OPEN | none  | 1+2/0 | 6 | - | core/gemm.mojo:397 | owed A/B: the default vs MOJOLEARN_537_GEMM_IDENT_SWAP on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T08` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/forest_inference.mojo:49 | owed A/B: the default vs MOJOLEARN_AFCL_T08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_FUSE` | KEEP-OPEN | LOSE EXP:2363 DELETED (superseded) | 1+4/0 | 102 | c | training/samba_afn.mojo:135 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_AFT_P01` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:21 | owed A/B: the default vs MOJOLEARN_AFT_P01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P03` | KEEP-OPEN | none  | 1+1/0 | 2 | - | core/apple_fast_tree_experiments.mojo:35 | owed A/B: the default vs MOJOLEARN_AFT_P03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P04` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:29 | owed A/B: the default vs MOJOLEARN_AFT_P04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P06` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:44 | owed A/B: the default vs MOJOLEARN_AFT_P06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P08` | KEEP-OPEN | none  | 1+1/0 | 2 | - | core/apple_fast_tree_experiments.mojo:52 | owed A/B: the default vs MOJOLEARN_AFT_P08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P09` | KEEP-OPEN | none  | 1+1/0 | 2 | - | core/apple_fast_tree_experiments.mojo:53 | owed A/B: the default vs MOJOLEARN_AFT_P09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:57 | owed A/B: the default vs MOJOLEARN_AFT_P10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P11` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:62 | owed A/B: the default vs MOJOLEARN_AFT_P11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P12` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/apple_fast_tree_experiments.mojo:63 | owed A/B: the default vs MOJOLEARN_AFT_P12 on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_DIRECT_OUT` | KEEP-OPEN | none  | 1+2/0 | 5 | - | x_ann/switches.mojo:63 | owed A/B: the default vs MOJOLEARN_ANN3_DIRECT_OUT on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL` | KEEP-OPEN | OPEN EXP:2415 PENDING (A = unfilled store is the default) | 1+3/0 | 16 | - | bindings/byte_lm_trainer_native_door.mojo:1459 | owed A/B: the default vs MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL on NV+AMD, rows of its family |
| `MOJOLEARN_C31_DEVICE_BUCKETS` | KEEP-OPEN | none  | 1+3/0 | 8 | - | experiments/classical_identical_ideas/graph_controls.mojo:72 | owed A/B: the default vs MOJOLEARN_C31_DEVICE_BUCKETS on NV+AMD, rows of its family |
| `MOJOLEARN_C32_COUNT_FUSION` | KEEP-OPEN | none  | 1+2/0 | 7 | - | experiments/classical_identical_ideas/graph_controls.mojo:73 | owed A/B: the default vs MOJOLEARN_C32_COUNT_FUSION on NV+AMD, rows of its family |
| `MOJOLEARN_C32_EMIT_FUSION` | KEEP-OPEN | none  | 1+2/0 | 7 | - | experiments/classical_identical_ideas/graph_controls.mojo:74 | owed A/B: the default vs MOJOLEARN_C32_EMIT_FUSION on NV+AMD, rows of its family |
| `MOJOLEARN_C35_ROWS_128` | KEEP-OPEN | none  | 1+2/0 | 3 | - | experiments/classical_identical_ideas/graph_controls.mojo:79 | owed A/B: the default vs MOJOLEARN_C35_ROWS_128 on NV+AMD, rows of its family |
| `MOJOLEARN_C39_RETAIN_STATE` | KEEP-OPEN | none  | 1+10/0 | 12 | c | experiments/classical_identical_ideas/graph_controls.mojo:101 | owed A/B: the default vs MOJOLEARN_C39_RETAIN_STATE on NV+AMD, rows of its family |
| `MOJOLEARN_C40_ACTIVE_SEEDS` | KEEP-OPEN | none  | 1+5/0 | 13 | - | experiments/classical_identical_ideas/graph_controls.mojo:103 | owed A/B: the default vs MOJOLEARN_C40_ACTIVE_SEEDS on NV+AMD, rows of its family |
| `MOJOLEARN_C40_SEED_TILES` | KEEP-OPEN | none  | 1+2/0 | 3 | - | experiments/classical_identical_ideas/graph_controls.mojo:102 | owed A/B: the default vs MOJOLEARN_C40_SEED_TILES on NV+AMD, rows of its family |
| `MOJOLEARN_C41_FUSED_MINIMA` | KEEP-OPEN | none  | 1+3/0 | 23 | - | experiments/classical_identical_ideas/graph_controls.mojo:104 | owed A/B: the default vs MOJOLEARN_C41_FUSED_MINIMA on NV+AMD, rows of its family |
| `MOJOLEARN_C42_ACTIVE_TRIANGLE` | KEEP-OPEN | none  | 1+10/0 | 24 | - | experiments/classical_identical_ideas/graph_controls.mojo:105 | owed A/B: the default vs MOJOLEARN_C42_ACTIVE_TRIANGLE on NV+AMD, rows of its family |
| `MOJOLEARN_C44_SAMPLING_DESCRIPTORS` | KEEP-OPEN | none  | 1+3/0 | 8 | - | experiments/classical_identical_ideas/graph_controls.mojo:107 | owed A/B: the default vs MOJOLEARN_C44_SAMPLING_DESCRIPTORS on NV+AMD, rows of its family |
| `MOJOLEARN_C53_CENTER4` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/stats_controls.mojo:19 | owed A/B: the default vs MOJOLEARN_C53_CENTER4 on NV+AMD, rows of its family |
| `MOJOLEARN_C54_PREDICT_TILES` | KEEP-OPEN | none  | 1+8/0 | 45 | - | experiments/classical_identical_ideas/stats_controls.mojo:22 | owed A/B: the default vs MOJOLEARN_C54_PREDICT_TILES on NV+AMD, rows of its family |
| `MOJOLEARN_C58_TEAM_MIB` | KEEP-OPEN | none  | 1+2/2 | 3 | c | experiments/classical_identical_ideas/stats_controls.mojo:43 | owed A/B: the default vs MOJOLEARN_C58_TEAM_MIB on NV+AMD, rows of its family |
| `MOJOLEARN_C62_SAME_COMPONENT_SKIP` | KEEP-OPEN | OPEN EXP:1560 unmeasured | 1+3/0 | 31 | - | experiments/classical_identical_ideas/graph_controls.mojo:114 | owed A/B: the default vs MOJOLEARN_C62_SAME_COMPONENT_SKIP on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C01_LEAF` | KEEP-OPEN | LOSE EXP:1530 DROPPED-slower | 1+3/0 | 4 | c | experiments/classical_identical_ideas/shared_controls.mojo:15 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_CLASSICAL_C01_MEAN` | KEEP-OPEN | LOSE EXP:1530 DROPPED-slower | 1+4/0 | 14 | c | experiments/classical_identical_ideas/shared_controls.mojo:23 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR` | KEEP-OPEN | none  | 1+6/0 | 7 | - | experiments/classical_identical_ideas/shared_controls.mojo:97 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA` | KEEP-OPEN | none  | 1+2/0 | 3 | - | experiments/classical_identical_ideas/shared_controls.mojo:25 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C04_LDA` | KEEP-OPEN | LOSE EXP:1532 DROPPED-serial | 1+6/0 | 7 | c | experiments/classical_identical_ideas/shared_controls.mojo:35 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_CLASSICAL_C05_OLS_PHASE` | KEEP-OPEN | none  | 1+4/0 | 11 | - | experiments/classical_identical_ideas/shared_controls.mojo:102 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C05_OLS_PHASE on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH` | KEEP-OPEN | none  | 1+3/0 | 7 | - | experiments/classical_identical_ideas/shared_controls.mojo:36 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C06_NORM_ROWS` | KEEP-OPEN | OPEN EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit | 1+3/1 | 4 | c | experiments/classical_identical_ideas/shared_controls.mojo:40 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C06_NORM_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD` | KEEP-OPEN | OPEN EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit | 1+2/1 | 10 | - | experiments/classical_identical_ideas/shared_controls.mojo:47 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C07_RADIX_BITS` | KEEP-OPEN | none  | 1+2/0 | 3 | c | experiments/classical_identical_ideas/shared_controls.mojo:56 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C07_RADIX_BITS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C07_RADIX_ROWS` | KEEP-OPEN | none  | 1+2/0 | 3 | c | experiments/classical_identical_ideas/shared_controls.mojo:57 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C07_RADIX_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C08_ONEHOT_FT` | KEEP-OPEN | none  | 1+4/0 | 5 | - | experiments/classical_identical_ideas/shared_controls.mojo:67 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C08_ONEHOT_FT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C08_ORDINAL_FT` | KEEP-OPEN | none  | 1+4/0 | 5 | - | experiments/classical_identical_ideas/shared_controls.mojo:68 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C08_ORDINAL_FT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF` | KEEP-OPEN | WIN EXP:1740 PROMOTED (IDENTICAL default) | 1+4/0 | 5 | - | experiments/classical_identical_ideas/shared_controls.mojo:84 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_C09_REG_BUNDLE` | KEEP-OPEN | none  | 1+2/0 | 15 | - | experiments/classical_identical_ideas/shared_controls.mojo:85 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C09_REG_BUNDLE on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C10_RANK_REUSE` | KEEP-OPEN | none  | 1+2/0 | 17 | - | experiments/classical_identical_ideas/shared_controls.mojo:86 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C10_RANK_REUSE on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C11_DRAW_GATHER` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/shared_controls.mojo:87 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C11_DRAW_GATHER on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS` | KEEP-OPEN | none  | 1+2/0 | 13 | - | experiments/classical_identical_ideas/shared_controls.mojo:88 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/linear_controls.mojo:131 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C17_OVR` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/linear_controls.mojo:158 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C17_OVR on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT` | KEEP-OPEN | none  | 1+5/0 | 12 | - | experiments/classical_identical_ideas/linear_controls.mojo:170 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C19_SGD_CHUNK` | KEEP-OPEN | none  | 1+2/3 | 3 | c | experiments/classical_identical_ideas/linear_controls.mojo:176 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C19_SGD_CHUNK on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C20_PAIR_LOAD` | KEEP-OPEN | none  | 1+3/0 | 16 | - | experiments/classical_identical_ideas/linear_controls.mojo:180 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C20_PAIR_LOAD on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C21_EXTREMA` | KEEP-OPEN | none  | 1+2/0 | 23 | - | experiments/classical_identical_ideas/linear_controls.mojo:181 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C21_EXTREMA on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C22_TRIANGLE` | KEEP-OPEN | none  | 1+3/0 | 13 | - | experiments/classical_identical_ideas/linear_controls.mojo:182 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C22_TRIANGLE on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C24_PANEL8` | KEEP-OPEN | WIN grid classical-decomp.json promoted:ge123e6f9 | 1+2/0 | 3 | - | experiments/classical_identical_ideas/linear_controls.mojo:206 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF` | KEEP-OPEN | WIN EXP:1638 PROMOTED | 1+2/0 | 3 | - | experiments/classical_identical_ideas/linear_controls.mojo:223 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF` | KEEP-OPEN | WIN EXP:1984 PROMOTED (restored; IDENTICAL default) | 1+6/0 | 7 | c | experiments/classical_identical_ideas/linear_controls.mojo:236 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_C56_LDA_INPUT` | KEEP-OPEN | none  | 1+5/0 | 6 | - | experiments/classical_identical_ideas/shared_controls.mojo:98 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C56_LDA_INPUT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C56_QDA_PROJECT` | KEEP-OPEN | none  | 1+2/0 | 3 | c | experiments/classical_identical_ideas/stats_controls.mojo:30 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C56_QDA_PROJECT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS` | KEEP-OPEN | WIN EXP:1992 PROMOTED (IDENTICAL default, arm 64) | 1+3/4 | 4 | c | experiments/classical_identical_ideas/linear_controls.mojo:72 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF` | KEEP-OPEN | WIN EXP:1992 PROMOTED (IDENTICAL default, arm 64) | 1+1/1 | 2 | - | experiments/classical_identical_ideas/linear_controls.mojo:70 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS` | KEEP-OPEN | WIN EXP:2068 KEEP (promoted IDENTICAL default) | 1+3/3 | 4 | - | experiments/classical_identical_ideas/linear_controls.mojo:128 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF` | KEEP-OPEN | WIN EXP:2068 KEEP (promoted IDENTICAL default) | 1+1/1 | 2 | - | experiments/classical_identical_ideas/linear_controls.mojo:127 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` | KEEP-OPEN | LOSE EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off | 1+11/0 | 18 | - | experiments/classical_identical_ideas/linear_controls.mojo:156 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_COLUMN_CPU` | KEEP-OPEN | none  | 28+729/1 | 1036 | - | autograd/kernels/mamba3.mojo:1498 | owed A/B: the default vs MOJOLEARN_COLUMN_CPU on NV+AMD, rows of its family |
| `MOJOLEARN_DART_ROUND_REF_OFF` | KEEP-OPEN | OPEN EXP:2393 PENDING | 1+1/0 | 4 | - | bindings/trees_dart_rounds_run.mojo:52 | owed A/B: default vs the old path MOJOLEARN_DART_ROUND_REF_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_DT_BINS_QOLD` | KEEP-OPEN | WIN EXP:1285 QUALITY-FIX KEPT: FAST default | 1+4/0 | 5 | - | bindings/_mojolearn_rf.mojo:814 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT` | KEEP-OPEN | none  | 4+0/0 | 41 | - | core/byte_lm_entry_table_sessions.mojo:226 | owed A/B: the default vs MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SCALAR_STAGE_COMPARE` | KEEP-OPEN | none  | 3+0/0 | 12 | - | bindings/_mojolearn_trees.mojo:273 | owed A/B: the default vs MOJOLEARN_ET_SCALAR_STAGE_COMPARE on NV+AMD, rows of its family |
| `MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH` | KEEP-OPEN | none  | 2+0/0 | 2 | - | core/identical_callpath.mojo:127 | owed A/B: the default vs MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_DEVICE_FINITE_OFF` | KEEP-OPEN | WIN EXP:106 KEPT 69f7a41fd | 1+12/0 | 23 | - | ensemble/device_finite.mojo:26 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_FOREST_DEVICE_FINITE_OFF |
| `MOJOLEARN_FOREST_ORDERED_RESIDENT` | KEEP-OPEN | none  | 1+0/0 | 1 | - | core/forest_inference_model.mojo:80 | owed A/B: the default vs MOJOLEARN_FOREST_ORDERED_RESIDENT on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_PER_MODEL_IO` | KEEP-OPEN | none  | 1+4/0 | 6 | - | core/forest_inference_model.mojo:51 | owed A/B: the default vs MOJOLEARN_FOREST_PER_MODEL_IO on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_PINNED_STAGE` | KEEP-OPEN | none  | 1+4/0 | 15 | - | core/forest_inference_model.mojo:37 | owed A/B: the default vs MOJOLEARN_FOREST_PINNED_STAGE on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_SCALAR_GROVES` | KEEP-OPEN | none  | 1+0/0 | 1 | - | core/forest_inference.mojo:363 | owed A/B: the default vs MOJOLEARN_FOREST_SCALAR_GROVES on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_SEPARATE_NODES` | KEEP-OPEN | LOSE EXP:1504 DROP, defines deleted | 1+3/0 | 4 | c | core/forest_inference.mojo:124 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_FOREST_SHARED_ROWS` | KEEP-OPEN | none  | 1+4/0 | 52 | - | core/forest_inference.mojo:85 | owed A/B: the default vs MOJOLEARN_FOREST_SHARED_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AF_RESIDENT_OFF` | KEEP-OPEN | none  | 1+9/0 | 13 | - | sequence/opt_resident.mojo:111 | owed A/B: default vs the old path MOJOLEARN_IDN_AF_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CACHE_CTX_KEY_DEVICE_ID` | KEEP-OPEN | none  | 1+0/0 | 3 | - | core/ctx_key.mojo:35 | owed A/B: the default vs MOJOLEARN_IDN_CACHE_CTX_KEY_DEVICE_ID on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CACHE_CTX_KEY_OFF` | KEEP-OPEN | none  | 1+2/0 | 9 | - | core/ctx_key.mojo:21 | owed A/B: default vs the old path MOJOLEARN_IDN_CACHE_CTX_KEY_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CLASS_ONEPASS_OFF` | KEEP-OPEN | none  | 1+8/0 | 9 | - | x_prep/blocked.mojo:75 | owed A/B: default vs the old path MOJOLEARN_IDN_CLASS_ONEPASS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CODE_RESIDENT_OFF` | KEEP-OPEN | none  | 1+2/0 | 5 | - | x_decomp/resident.mojo:724 | owed A/B: default vs the old path MOJOLEARN_IDN_CODE_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | gbdt/models/kernel/add_bin_values.mojo:50 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_STEP_OLD` | KEEP-OPEN | OPEN EXP:2423 PENDING | 2+2/0 | 6 | - | checks/gbdt_step_cost_check.mojo:57 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_STEP_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GFF_GRAM_TILED_OFF` | KEEP-OPEN | WIN EXP:2096 KEEP (promoted) | 1+1/0 | 2 | - | experiments/classical_identical_ideas/fg_linear_controls.mojo:121 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_GPC_OVR_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | gaussian_process/gpc_ovr.mojo:89 | owed A/B: default vs the old path MOJOLEARN_IDN_GPC_OVR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GPR_PTR_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | gaussian_process/gpr_resident.mojo:101 | owed A/B: default vs the old path MOJOLEARN_IDN_GPR_PTR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GRAM_BLOCKED_OFF` | KEEP-OPEN | none  | 1+9/0 | 10 | - | x_prep/gram_blocked.mojo:49 | owed A/B: default vs the old path MOJOLEARN_IDN_GRAM_BLOCKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF` | KEEP-OPEN | WIN EXP:1832 PROMOTED | 1+3/0 | 4 | - | experiments/classical_identical_ideas/fg_linear_controls.mojo:98 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_GRAM_ROWS_512` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_prep/gram_blocked.mojo:53 | owed A/B: the default vs MOJOLEARN_IDN_GRAM_ROWS_512 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GRAM_ROWS_8192` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_prep/gram_blocked.mojo:53 | owed A/B: the default vs MOJOLEARN_IDN_GRAM_ROWS_8192 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GRAM_ROWTILE` | KEEP-OPEN | none  | 1+5/0 | 6 | - | x_prep/gram_blocked.mojo:51 | owed A/B: the default vs MOJOLEARN_IDN_GRAM_ROWTILE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_CAST_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/label_device_steps.mojo:30 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_CAST_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_ELEM_OFF` | KEEP-OPEN | none  | 2+6/0 | 8 | - | core/label_device_steps.mojo:27 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_ELEM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_FOLDS_OFF` | KEEP-OPEN | none  | 2+13/0 | 15 | - | core/base_helpers_device.mojo:44 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_FOLDS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_INIT_OFF` | KEEP-OPEN | none  | 2+5/0 | 7 | - | bindings/hotpath_device_door.mojo:124 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_INIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_ISUM_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/label_device_steps.mojo:29 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_ISUM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_LABELS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/hotpath_device_run.mojo:21 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_LABELS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_NORMAL_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | bindings/hotpath_device_door.mojo:128 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_NORMAL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HPDEV_REDUCE_OFF` | KEEP-OPEN | none  | 2+4/0 | 6 | - | core/label_device_steps.mojo:28 | owed A/B: default vs the old path MOJOLEARN_IDN_HPDEV_REDUCE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IF_FIT_EXPORT` | KEEP-OPEN | none  | 1+2/0 | 4 | - | bindings/trees_public_iforest_door.mojo:78 | owed A/B: the default vs MOJOLEARN_IDN_IF_FIT_EXPORT on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IF_FIT_OWN_X` | KEEP-OPEN | OPEN EXP:2396 PENDING | 1+1/0 | 2 | - | bindings/trees_public_iforest_door.mojo:63 | owed A/B: the default vs MOJOLEARN_IDN_IF_FIT_OWN_X on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF` | KEEP-OPEN | none  | 1+4/0 | 7 | - | isolation_forest/impl/isolation_forest.mojo:584 | owed A/B: default vs the old path MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IF_RESIDENT_OFF` | KEEP-OPEN | none  | 2+9/0 | 11 | - | bindings/_mojolearn_svm.mojo:104 | owed A/B: default vs the old path MOJOLEARN_IDN_IF_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LABEL_OFF` | KEEP-OPEN | none  | 1+13/0 | 15 | - | x_prep/label_fast.mojo:31 | owed A/B: default vs the old path MOJOLEARN_IDN_LABEL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LM_RESIDENT_GENERATE` | KEEP-OPEN | OPEN EXP:2262 PENDING | 2+3/0 | 6 | - | bindings/causal_lm_public_door.mojo:127 | owed A/B: the default vs MOJOLEARN_IDN_LM_RESIDENT_GENERATE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD` | KEEP-OPEN | OPEN EXP:2421 PENDING (A = 0.8.55's `_mamba1_run_arena` / `_mamba2_run_arena` form is the defa | 1+4/0 | 5 | - | autograd/block_step_ops.mojo:216 | owed A/B: default vs the old path MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD` | KEEP-OPEN | OPEN EXP:2422 PENDING (A = caps cached on the session, one held call per forward) | 1+1/0 | 2 | - | core/neural_entry_table.mojo:198 | owed A/B: default vs the old path MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NB_CSR_DENSE_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_prep/blocked.mojo:685 | owed A/B: default vs the old path MOJOLEARN_IDN_NB_CSR_DENSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NB_CSR_OFF` | KEEP-OPEN | none  | 1+5/0 | 9 | - | x_prep/fastnb_csr.mojo:72 | owed A/B: default vs the old path MOJOLEARN_IDN_NB_CSR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NB_ONEPASS_OFF` | KEEP-OPEN | none  | 1+8/0 | 9 | - | x_prep/blocked.mojo:43 | owed A/B: default vs the old path MOJOLEARN_IDN_NB_ONEPASS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OLS_PREDICT_TILED` | KEEP-OPEN | OPEN EXP:293 RUN OWED | 1+1/0 | 2 | - | core/gemm.mojo:744 | owed A/B: the default vs MOJOLEARN_IDN_OLS_PREDICT_TILED on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PT_BLOCKED_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | x_prep/pt_blocked.mojo:50 | owed A/B: default vs the old path MOJOLEARN_IDN_PT_BLOCKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_QUERY_X_HELD_OFF` | KEEP-OPEN | OPEN EXP:2419 RUN OWED (nv+amd, owed row fix-svc-query-resident, B arm query-x-held-off) | 1+7/0 | 17 | - | core/query_x_held.mojo:53 | owed A/B: default vs the old path MOJOLEARN_IDN_QUERY_X_HELD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RADIX_TILE128` | KEEP-OPEN | none  | 3+0/0 | 9 | - | experiments/performance_ideas/I19/check.mojo:36 | owed A/B: the default vs MOJOLEARN_IDN_RADIX_TILE128 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RAGGED_FLOAT_RADIX` | KEEP-OPEN | none  | 2+0/0 | 14 | - | experiments/performance_ideas/I19/ragged_float.mojo:84 | owed A/B: the default vs MOJOLEARN_IDN_RAGGED_FLOAT_RADIX on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE` | KEEP-OPEN | none  | 3+0/0 | 27 | - | experiments/performance_ideas/I19/check.mojo:33 | owed A/B: the default vs MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RIDGE_RESIDENT_OFF` | KEEP-OPEN | LOSE EXP:1833 _OFF arm ratios: ridge istella NV 2.22x / AMD 1.28x slower; taxi NV 0.87x / AMD  | 1+1/0 | 2 | - | experiments/classical_identical_ideas/fg_linear_controls.mojo:36 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_SCRATCH_POOL` | KEEP-OPEN | none  | 1+9/0 | 62 | - | core/scratch_pool.mojo:41 | owed A/B: the default vs MOJOLEARN_IDN_SCRATCH_POOL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SCRATCH_POOL_OFF` | KEEP-OPEN | none  | 1+9/0 | 62 | - | core/scratch_pool.mojo:42 | owed A/B: default vs the old path MOJOLEARN_IDN_SCRATCH_POOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEG_RADIX8_OFF` | KEEP-OPEN | none  | 1+2/0 | 8 | - | core/segmented_sort.mojo:136 | owed A/B: default vs the old path MOJOLEARN_IDN_SEG_RADIX8_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SELECT_BLOCKED_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | x_prep/select_blocked.mojo:33 | owed A/B: default vs the old path MOJOLEARN_IDN_SELECT_BLOCKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SVD_RESIDENT_OFF` | KEEP-OPEN | none  | 1+2/0 | 5 | - | x_decomp/resident.mojo:565 | owed A/B: default vs the old path MOJOLEARN_IDN_SVD_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | providers/python/xtrees_api_pyapi_door.mojo:955 | owed A/B: default vs the old path MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_REUSE` | KEEP-OPEN | none  | 4+0/0 | 7 | - | experiments/performance_ideas/I22/reuse_check.mojo:23 | owed A/B: the default vs MOJOLEARN_IDN_TSQR_REUSE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XTY_TILE_1024` | KEEP-OPEN | none  | 1+11/0 | 12 | - | core/host_tile_fold.mojo:50 | owed A/B: the default vs MOJOLEARN_IDN_XTY_TILE_1024 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XTY_TILE_64` | KEEP-OPEN | none  | 1+11/0 | 12 | - | core/host_tile_fold.mojo:49 | owed A/B: the default vs MOJOLEARN_IDN_XTY_TILE_64 on NV+AMD, rows of its family |
| `MOJOLEARN_IF_ROWMAJOR_OFF` | KEEP-OPEN | WIN EXP:115 KEPT 269ffa57a | 1+6/0 | 12 | - | isolation_forest/impl/isolation_tree_builder.mojo:169 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_IF_ROWMAJOR_OFF |
| `MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF` | KEEP-OPEN | WIN grid fg-ivf.json new default (read_ivf.md A3) | 1+3/0 | 4 | - | x_ann/switches.mojo:172 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF |
| `MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF` | KEEP-OPEN | WIN grid fg-ivf.json new default (read_ivf.md A7) | 1+4/0 | 5 | - | x_ann/switches.mojo:186 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF |
| `MOJOLEARN_KNN_CERTIFIED_MMA_OFF` | KEEP-OPEN | none  | 1+5/0 | 9 | - | neighbors/impl/detail/knn_brute_force.mojo:256 | owed A/B: default vs the old path MOJOLEARN_KNN_CERTIFIED_MMA_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KPCA_RESIDENT_OFF` | KEEP-OPEN | WIN EXP:923 DEFAULT; M2 default + OFF both rc=0, merged bc61f3560; Python import PASS | 1+3/0 | 4 | - | x_neighbors/kapprox_dev.mojo:30 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL` | KEEP-OPEN | OPEN EXP:2409 PENDING (A = 0.8.55's unfilled outputs is the default) | 1+2/0 | 5 | - | bindings/neural_prefill_native_door.mojo:109 | owed A/B: the default vs MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL` | KEEP-OPEN | none  | 2+0/0 | 2 | - | bindings/_mojolearn_mamba.mojo:1319 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL on NV+AMD, rows of its family |
| `MOJOLEARN_MEASURE_I06_GQA` | KEEP-OPEN | none  | 1+0/0 | 2 | - | experiments/performance_ideas/attention_time.mojo:27 | owed A/B: the default vs MOJOLEARN_MEASURE_I06_GQA on NV+AMD, rows of its family |
| `MOJOLEARN_MEASURE_I07_LIFETIME` | KEEP-OPEN | none  | 2+0/0 | 7 | - | experiments/performance_ideas/attention_time.mojo:32 | owed A/B: the default vs MOJOLEARN_MEASURE_I07_LIFETIME on NV+AMD, rows of its family |
| `MOJOLEARN_MLP_SESSIONS_PACKED` | KEEP-OPEN | OPEN EXP:2381 RUN OWED | 2+2/0 | 50 | i | bindings/small_mlp_native_door.mojo:72 | owed A/B: the default vs MOJOLEARN_MLP_SESSIONS_PACKED on NV+AMD, rows of its family |
| `MOJOLEARN_MSEL_UNITS_GPU_PARALLEL` | KEEP-OPEN | WIN EXP:2371 DEFAULT OFF (crashes on Metal today: member fits share core/neural_context.mojo' | 1+1/0 | 3 | ic | bindings/native_unit_threads_unpack.mojo:52 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_MULTIOUT_RIDGE_OFF` | KEEP-OPEN | WIN EXP:244 KEPT 24ed76679 | 1+7/0 | 11 | - | glm/impl/ridge_multi.mojo:44 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MULTIOUT_RIDGE_OFF |
| `MOJOLEARN_MUTEX_SPIN_RELAXED` | KEEP-OPEN | none  | 1+1/0 | 5 | - | core/device_mutex.mojo:94 | owed A/B: the default vs MOJOLEARN_MUTEX_SPIN_RELAXED on NV+AMD, rows of its family |
| `MOJOLEARN_NATIVE_CONV_BAD_GRADIENT` | KEEP-OPEN | none  | 1+0/0 | 2 | - | packaging/native_only/cases/autograd_conv_core.mojo:141 | owed A/B: the default vs MOJOLEARN_NATIVE_CONV_BAD_GRADIENT on NV+AMD, rows of its family |
| `MOJOLEARN_NATIVE_OPTIMIZER_BAD_REFERENCE` | KEEP-OPEN | none  | 1+0/0 | 3 | - | packaging/native_only/cases/sequence_optimizer_core.mojo:110 | owed A/B: the default vs MOJOLEARN_NATIVE_OPTIMIZER_BAD_REFERENCE on NV+AMD, rows of its family |
| `MOJOLEARN_NATIVE_SVM_BAD_PREDICTION` | KEEP-OPEN | none  | 2+0/0 | 4 | - | packaging/native_only/cases/svm_core.mojo:97 | owed A/B: the default vs MOJOLEARN_NATIVE_SVM_BAD_PREDICTION on NV+AMD, rows of its family |
| `MOJOLEARN_NB_TEXT_CSR_OFF` | KEEP-OPEN | WIN EXP:290 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | 1+2/0 | 3 | - | x_prep/fastnb_csr.mojo:50 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_NB_TEXT_CSR_OFF |
| `MOJOLEARN_NEURAL_TABLE_ROUTE_OFF` | KEEP-OPEN | OPEN EXP:2391 PENDING | 1+9/0 | 34 | - | bindings/neural_block_public_door.mojo:191 | owed A/B: default vs the old path MOJOLEARN_NEURAL_TABLE_ROUTE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_NV_AMD_STEPS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/strided_walk.mojo:76 | owed A/B: default vs the old path MOJOLEARN_NV_AMD_STEPS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_PROGRAM_RUN_ENTRY` | KEEP-OPEN | WIN EXP:2394 RESTORED (A = run_program, the 0.8.55 route, default) | 1+3/0 | 4 | - | providers/python/autograd_program_pyapi_door.mojo:338 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_RF_MUTEX_CLAIM_STOCK` | KEEP-OPEN | none  | 1+1/0 | 2 | - | core/device_mutex.mojo:77 | owed A/B: the default vs MOJOLEARN_RF_MUTEX_CLAIM_STOCK on NV+AMD, rows of its family |
| `MOJOLEARN_SAMBA_PROGRAM_FORWARD` | KEEP-OPEN | OPEN EXP:2399 PENDING | 1+1/0 | 2 | - | training/neural_identical_experiments.mojo:118 | owed A/B: the default vs MOJOLEARN_SAMBA_PROGRAM_FORWARD on NV+AMD, rows of its family |
| `MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL` | KEEP-OPEN | OPEN EXP:2416 PENDING (A = per-registry key is the default) | 1+4/0 | 13 | - | core/weight_epoch.mojo:116 | owed A/B: the default vs MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL on NV+AMD, rows of its family |
| `MOJOLEARN_SCRATCH` | KEEP-OPEN | none  | 1+4/0 | 13 | - | tools/lowbit_amd_tuned/scratch_first.mojo:22 | owed A/B: the default vs MOJOLEARN_SCRATCH on NV+AMD, rows of its family |
| `MOJOLEARN_SEG_SUMS_SERIAL` | KEEP-OPEN | none  | 1+4/0 | 13 | - | core/segmented_sort.mojo:114 | owed A/B: the default vs MOJOLEARN_SEG_SUMS_SERIAL on NV+AMD, rows of its family |
| `MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF` | KEEP-OPEN | OPEN EXP:1789 owed (nv + amd RACE moe, OFF arm vs default) | 1+1/0 | 2 | - | bindings/_mojolearn_x_sequence.mojo:119 | owed A/B: default vs the old path MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SESSION_CALL_DESCRIPTOR` | KEEP-OPEN | OPEN EXP:2395 RESTORED (A = direct typed call when the calling image owns the registry, defaul | 2+2/0 | 14 | - | providers/python/autograd_ops_frontend_pyapi_door.mojo:133 | owed A/B: the default vs MOJOLEARN_SESSION_CALL_DESCRIPTOR on NV+AMD, rows of its family |
| `MOJOLEARN_STEP_GLUE_TRIAL` | KEEP-OPEN | none  | 1+7/0 | 9 | - | core/step_glue.mojo:63 | owed A/B: the default vs MOJOLEARN_STEP_GLUE_TRIAL on NV+AMD, rows of its family |
| `MOJOLEARN_TE_ADA_SESSION_SHARE_OFF` | KEEP-OPEN | WIN EXP:123 KEPT ed49f2b11 | 1+2/0 | 3 | - | providers/python/xtrees_api_pyapi_door.mojo:1012 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_TE_ADA_SESSION_SHARE_OFF |
| `MOJOLEARN_TE_NATIVE_SPLITS_OFF` | KEEP-OPEN | WIN EXP:124 KEPT ed49f2b11 | 1+2/0 | 3 | - | providers/python/xtrees_api_pyapi_door.mojo:1009 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_TE_NATIVE_SPLITS_OFF |
| `MOJOLEARN_TRAIN_LIVE_STATUS` | KEEP-OPEN | none  | 2+0/0 | 4 | - | experiments/performance_ideas/I10/check.mojo:79 | owed A/B: the default vs MOJOLEARN_TRAIN_LIVE_STATUS on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION` | KEEP-OPEN | OPEN EXP:2406 PENDING (A = 0.8.55's per-token shape: admit x, one y, cached_tokens read/write, | 1+1/0 | 2 | - | providers/python/transformer_decode_pyapi_door.mojo:95 | owed A/B: the default vs MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR` | KEEP-OPEN | OPEN EXP:2408 PENDING (A = the step input admitted as is) | 2+5/0 | 20 | - | providers/python/transformer_decode_pyapi_door.mojo:73 | owed A/B: the default vs MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_FRESH_PREFILL` | KEEP-OPEN | OPEN EXP:2376 PENDING (A = 0.8.66's session fresh prefill is the default) | 1+0/0 | 1 | icp | bindings/_mojolearn_transformer.mojo:1230 | owed A/B: the default vs MOJOLEARN_TRANSFORMER_FRESH_PREFILL on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL` | KEEP-OPEN | LOSE EXP:2355 DELETED (superseded) | 1+0/0 | 1 | ic | bindings/_mojolearn_transformer.mojo:1230 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_TREES_C50_GB_PACKED` | KEEP-OPEN | none  | 1+12/1 | 105 | - | core/forest_experiments.mojo:65 | owed A/B: the default vs MOJOLEARN_TREES_C50_GB_PACKED on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_FOREST_ROUTE` | KEEP-OPEN | none  | 1+3/0 | 4 | c | core/forest_experiments.mojo:32 | owed A/B: the default vs MOJOLEARN_TREES_FOREST_ROUTE on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS` | KEEP-OPEN | none  | 1+2/0 | 11 | - | core/forest_inference.mojo:218 | owed A/B: the default vs MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T32_SHARED_ROWS` | KEEP-OPEN | none  | 1+2/0 | 3 | - | core/forest_experiments.mojo:15 | owed A/B: the default vs MOJOLEARN_TREES_T32_SHARED_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T33_COST_SCHEDULE` | KEEP-OPEN | none  | 1+3/0 | 17 | - | core/forest_experiments.mojo:16 | owed A/B: the default vs MOJOLEARN_TREES_T33_COST_SCHEDULE on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T34_CHUNK_FOLD` | KEEP-OPEN | none  | 1+10/0 | 80 | - | core/forest_experiments.mojo:17 | owed A/B: the default vs MOJOLEARN_TREES_T34_CHUNK_FOLD on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T37_WORKSPACE` | KEEP-OPEN | none  | 1+6/0 | 14 | - | core/forest_experiments.mojo:35 | owed A/B: the default vs MOJOLEARN_TREES_T37_WORKSPACE on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T38_FUSED_LABELS` | KEEP-OPEN | none  | 1+6/0 | 30 | - | core/forest_experiments.mojo:36 | owed A/B: the default vs MOJOLEARN_TREES_T38_FUSED_LABELS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T39_AUXILIARY_WALK` | KEEP-OPEN | none  | 1+2/0 | 5 | - | core/forest_experiments.mojo:63 | owed A/B: the default vs MOJOLEARN_TREES_T39_AUXILIARY_WALK on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T44_METADATA_CACHE` | KEEP-OPEN | none  | 2+4/0 | 24 | - | core/forest_experiments.mojo:62 | owed A/B: the default vs MOJOLEARN_TREES_T44_METADATA_CACHE on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256` | KEEP-OPEN | none  | 1+8/0 | 9 | - | umap/optimizer_identical_device.mojo:72 | owed A/B: the default vs MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256 on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64` | KEEP-OPEN | none  | 1+8/0 | 9 | - | umap/optimizer_identical_device.mojo:71 | owed A/B: the default vs MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64 on NV+AMD, rows of its family |
| `MOJOLEARN_XCLUSTER_ROW_ASSIGN` | KEEP-OPEN | OPEN EXP:1518 4`, and per-family direct-distance defines KNN / KDE / DBSCAN / GRAPH / IVF with | 1+3/0 | 4 | c | experiments/classical_identical_ideas/graph_controls.mojo:90 | owed A/B: the default vs MOJOLEARN_XCLUSTER_ROW_ASSIGN on NV+AMD, rows of its family |
| `MOJOLEARN_XPREP_ARENA_POOL_OFF` | KEEP-OPEN | LOSE EXP:1787 CONFIRMED (kept default) | 2+5/0 | 9 | - | core/device_store.mojo:71 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` | KEEP-OPEN | WIN EXP:888 DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` | 1+4/0 | 10 | - | x_cluster/minibatch_fast.mojo:140 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_X_PREP_FAST_CLS2_PRESENT_OFF` | KEEP-OPEN | WIN EXP:407 KEEP, FAST+Apple default (`_OFF` off) | 1+2/0 | 3 | - | x_prep/cat_cls2.mojo:45 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_2002_SABOTAGE` | KEEP | none  | 1+2/0 | 3 | - | core/device_liveness.mojo:94 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_2042_FAST_NO_LOOKBACK` | KEEP | none  | 2+3/0 | 6 | - | checks/gbdt_partition_route_check.mojo:17 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_2043_FAST_FUSED_ONE_BYTE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:872 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_2044_FAST_NO_RIDX_ONLY` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:952 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_2045_FAST_NO_QUANT_HIST` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:886 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_2046_FAST_SHARED_I32` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:968 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:863 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF` | KEEP | none  | 1+2/0 | 3 | - | core/strided_walk.mojo:62 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_APPLE_GEMM_NT_MMA_OFF` | KEEP | none  | 1+2/0 | 3 | - | core/gemm.mojo:203 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_APPLE_GEMM_NT_MMA_SABOTAGE` | KEEP | none  | 1+0/0 | 3 | - | core/gemm.mojo:308 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_APPLE_GEMM_NT_TILED_OFF` | KEEP | none  | 1+3/0 | 4 | - | core/gemm.mojo:105 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_APPLE_STEP_UNROLL_OFF` | KEEP | none  | 1+2/0 | 3 | - | core/strided_walk.mojo:51 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ATTN_APPLE_R3_ONLY` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_attn.mojo:241 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN` | KEEP | none  | 1+0/3 | 2 | r | checks/kernel_matrix_attn.mojo:158 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN` | KEEP | none  | 1+0/3 | 2 | r | checks/kernel_matrix_attn.mojo:160 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN` | KEEP | none  | 1+0/3 | 2 | r | checks/kernel_matrix_attn.mojo:162 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN` | KEEP | none  | 1+0/3 | 2 | r | checks/kernel_matrix_attn.mojo:164 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_NO_LAUNCH_BOUND` | KEEP | none  | 2+0/0 | 4 | - | checks/kernel_matrix_attn.mojo:111 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M` | KEEP | none  | 2+5/0 | 19 | - | checks/batch_invariance_check.mojo:1188 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BIAS_SILU_PRICE` | KEEP | none  | 1+2/0 | 15 | - | bench/bias_gelu_price_main.mojo:21 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_C53_BGMM_STATS` | KEEP | none  | 1+2/0 | 3 | - | experiments/classical_identical_ideas/stats_controls.mojo:20 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CLASSICAL_C02_STATS_PAIR` | KEEP | none  | 1+2/0 | 3 | - | experiments/classical_identical_ideas/shared_controls.mojo:24 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CLASSICAL_C08_TARGET_CODES` | KEEP | none  | 1+4/0 | 5 | - | experiments/classical_identical_ideas/shared_controls.mojo:66 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_COLUMN_AMD` | KEEP | none  | 1+415/0 | 416 | - | checks/kernel_matrix.mojo:706 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_AMD_RDNA` | KEEP | none  | 1+415/0 | 416 | - | checks/kernel_matrix.mojo:707 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_APPLE` | KEEP | none  | 1+415/0 | 416 | - | checks/kernel_matrix.mojo:704 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_INTEL` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:710 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_NVIDIA` | KEEP | none  | 1+415/0 | 416 | - | checks/kernel_matrix.mojo:705 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_QUALCOMM` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:709 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COLUMN_SPEC_BASELINE` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:711 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_COVTYPE_4K` | KEEP | none  | 1+2/0 | 3 | - | checks/searcher_parity_covtype_check.mojo:80 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EM_LOOP_CHECK_EVERY` | KEEP | OPEN EXP:2320 PENDING | 1+3/0 | 4 | - | core/em_loop_chunk.mojo:34 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1735 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK_BOUNDED` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1804 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1578 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1647 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_PREFLIGHT_METADATA` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1533 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_RESIDENT_CACHE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1854 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_SELECTOR_BOUND` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1770 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE` | KEEP | none  | 2+0/0 | 4 | - | checks/kernel_matrix.mojo:1686 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_SMEM_WIDE_ONLY` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1722 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_TOPK_KEY32` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1834 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1383 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_KNN_WARPBOUND_GUARD` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1476 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1372 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FAST_BLOCK_SCAN_SHUFFLE` | KEEP | none  | 1+2/0 | 3 | - | core/block_scan.mojo:139 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_FAST_LCG_MODULO` | KEEP | none  | 1+4/0 | 7 | - | core/shuffle_iterator.mojo:107 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_FMAX_SABOTAGE_HARDWARE_MAX` | KEEP | none  | 1+4/0 | 68 | - | checks/portable_fmax_check.mojo:117 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FMAX_SABOTAGE_NO_NAN_CANON` | KEEP | none  | 1+4/0 | 21 | - | checks/portable_fmax_check.mojo:119 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FMAX_SABOTAGE_SIGN_UNFLIPPED` | KEEP | none  | 1+4/0 | 24 | - | checks/portable_fmax_check.mojo:118 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_FAST_SHARED_ROWS` | KEEP | none  | 1+4/0 | 52 | - | core/forest_inference.mojo:85 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_FOREST_GROVES_SABOTAGE` | KEEP | none  | 1+3/0 | 9 | - | core/forest_host_groves.mojo:80 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_HOST_SABOTAGE` | KEEP | none  | 4+5/0 | 14 | - | bindings/_mojolearn_gbdt.mojo:151 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_POOL_FAULT` | KEEP | none  | 2+0/0 | 25 | - | core/forest_inference_pool.mojo:531 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_POOL_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 8 | - | core/forest_inference_pool.mojo:150 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_PROFILE` | KEEP | none  | 1+7/0 | 19 | - | core/forest_inference_model.mojo:38 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FOREST_ROW_THREADS` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1656 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FOREST_ROW_THREADS_OFF` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1654 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FOREST_ROW_THREADS_SABOTAGE` | KEEP | none  | 1+2/0 | 10 | - | core/forest_inference.mojo:112 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_FTZ_FCMP_OFF` | KEEP | none  | 1+0/0 | 1 | - | checks/numerics.mojo:218 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FTZ_HW_OFF` | KEEP | none  | 1+0/0 | 1 | - | checks/numerics.mojo:253 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FTZ_NO_CLASS` | KEEP | none  | 1+0/0 | 1 | - | checks/numerics.mojo:206 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_FTZ_TWO_TEST` | KEEP | none  | 1+0/0 | 1 | - | checks/numerics.mojo:218 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_CTR_HOST_SABOTAGE` | KEEP | none  | 1+3/0 | 10 | - | core/gbdt_host_ctr.mojo:38 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_FULL_PARTITION_STATS` | KEEP | none  | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:210 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_IDENTICAL_RIDX_OFF` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:951 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GBDT_ID_RIDX` | KEEP | none  | 1+0/0 | 1 | - | checks/kernel_matrix.mojo:943 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GELU_NO_VENDOR_WITNESS` | KEEP | none  | 1+3/0 | 10 | - | checks/portable_gelu_check.mojo:142 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GELU_SABOTAGE_CEPHES_ZERO` | KEEP | none  | 1+4/0 | 22 | - | checks/portable_gelu_check.mojo:139 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GELU_SABOTAGE_DOUBLE_CONST` | KEEP | none  | 1+4/0 | 22 | - | checks/portable_gelu_check.mojo:136 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GELU_SABOTAGE_FUSED_INNER` | KEEP | none  | 1+4/0 | 19 | - | checks/portable_gelu_check.mojo:131 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GELU_SABOTAGE_ONE_GELU` | KEEP | none  | 1+5/0 | 34 | - | checks/portable_gelu_check.mojo:125 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_DETECT_SEAM` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_gemm.mojo:72 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_GATHER_FULL_TILE` | KEEP | none  | 2+0/0 | 16 | - | bench/gemm_step_price_main.mojo:423 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_KPACK_RPT4` | KEEP | OPEN EXP:1458 unmeasured | 2+28/0 | 30 | - | bench/gemm_schedule_ab.mojo:47 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LEGACY_REUSE_GROUP_WS` | KEEP | none  | 1+5/0 | 6 | - | gemm/checks/gemm_identical.mojo:7211 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_KPACK_NARROW` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_gemm.mojo:168 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_LEAF_SPLIT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_gemm.mojo:93 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_MFMA` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_gemm.mojo:189 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_WINDOW_ADMIT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_gemm.mojo:148 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_ONE_PAGE` | KEEP | OPEN EXP:1454 unmeasured | 5+88/0 | 93 | - | bench/gemm_schedule_ab.mojo:39 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_PRICE_SABOTAGE` | KEEP | none  | 1+0/0 | 1 | - | bench/gemm_step_price_main.mojo:158 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMV_FAST_VENDOR` | KEEP | none  | 1+0/0 | 1 | - | core/gemm.mojo:812 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_GRAM_PARALLEL_SABOTAGE` | KEEP | none  | 3+0/0 | 15 | - | core/blocked_moments_multi_gpu.mojo:137 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HOST_GEMM_SIMD_SABOTAGE` | KEEP | none  | 1+1/0 | 5 | - | core/host_gemm_simd.mojo:51 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HOST_SABOTAGE` | KEEP | none  | 63+314/0 | 923 | i | arima/host/arima_oracle.mojo:133 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HOST_SIMD_SABOTAGE` | KEEP | none  | 1+1/0 | 3 | - | core/host_simd_identical.mojo:40 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HOTPATH_SABOTAGE` | KEEP | none  | 1+23/0 | 55 | - | core/hotpath_impl.mojo:34 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_PREDICT_SABOTAGE` | KEEP | none  | 1+3/0 | 9 | - | core/holtwinters_host_predict_impl.mojo:9 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION` | KEEP | none  | 2+2/0 | 4 | - | checks/gbdt_partition_route_check.mojo:18 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2` | KEEP | WIN grid neural-gemm-attn-dedupe.json promoted:ge123e6f9 | 2+13/1 | 15 | - | bench/gemm_schedule_ab.mojo:41 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_GROUP_SLACK_OFF` | KEEP | WIN EXP:1724 PROMOTED | 2+10/1 | 12 | - | bench/gemm_schedule_ab.mojo:44 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GMM_COMPONENT_BATCH` | KEEP | none  | 1+3/0 | 30 | - | mixture/checks/estep.mojo:1108 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_NV_FMA_ASM_OLD` | KEEP | OPEN EXP:2413 PENDING (A = `llvm.nvvm.fma.rn.f`, 0.8.55's form) | 1+1/0 | 3 | - | checks/nv_fma_rn.mojo:32 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD` | KEEP | OPEN EXP:2414 PENDING (A = `llvm.nvvm.fma.rn.f/.d(a, b, -0.0)`, `llvm.nvvm.fma.rn.ftz.f(x, 1,  | 1+3/0 | 7 | - | checks/numerics.mojo:157 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_STATS_BLOCKED_OFF` | KEEP | none  | 1+7/0 | 8 | - | x_prep/blocked.mojo:62 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT15_EPILOGUE_SABOTAGE` | KEEP | none  | 1+4/0 | 8 | - | gemm/checks/gemm_int15_epilogue.mojo:42 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KDE_FAST_IMMUTABLE_FIT` | KEEP | none  | 1+2/0 | 3 | - | kde/resident_fit.mojo:88 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_BLOCK_TOPK_ALL_K` | KEEP | none  | 1+3/0 | 4 | c | checks/kernel_matrix.mojo:1760 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_BOUNDED_FIRST_TILE_1024` | KEEP | none  | 1+7/0 | 8 | - | checks/kernel_matrix.mojo:1820 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_BOUNDED_FIRST_TILE_16384` | KEEP | none  | 1+7/0 | 8 | - | checks/kernel_matrix.mojo:1822 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_BOUNDED_FIRST_TILE_8192` | KEEP | none  | 1+7/0 | 8 | - | checks/kernel_matrix.mojo:1821 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_FLUSHED_CHAIN` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1576 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1632 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_GENERIC_K` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1443 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_INSERT_CHAIN` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1471 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1381 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1370 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_LIST_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1768 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1733 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1498 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_NO_METADATA` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1541 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_NO_PREFLIGHT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1523 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_NO_RESIDENT_CACHE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1852 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_NO_ZERO_FMA_REPAIR` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1513 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_REGISTER_TILE_ONLY` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1684 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_ROWS4` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1563 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1392 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_SOFTWARE_FLUSH` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1549 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1445 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_TOPK_KEY64` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1832 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_TREE_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1460 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_UNBOUNDED_TOPK` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1802 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_IDENTICAL_UNFUSED_SELECT` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1645 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_QUERY_TILE_ARM_1024` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1613 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_QUERY_TILE_ARM_2048` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1615 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_QUERY_TILE_ARM_4096` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1617 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_QUERY_TILE_ARM_512` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1611 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K` | KEEP | none  | 1+4/0 | 5 | - | checks/kernel_matrix.mojo:1794 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_VECTOR_REQUEST_CHECK` | KEEP | none  | 2+0/0 | 6 | - | bench/knn_vector_request_main.mojo:64 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KPSS_DECISION_SABOTAGE` | KEEP | none  | 1+3/0 | 5 | - | core/kpss_host_test_impl.mojo:12 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_LOWBIT_PRICE_SABOTAGE` | KEEP | none  | 1+3/0 | 12 | - | bench/gemm_lowbit_price_main.mojo:377 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MUTEX_SECTION_SABOTAGE` | KEEP | none  | 1+1/0 | 2 | - | core/device_mutex.mojo:85 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_NO_ZERO_FMA_REPAIR` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1246 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_NUMERIC_FAST` | KEEP | OPEN EXP:1590 All OPEN: written, not compiled (lanes never compile), not measured; queue lines | 5+7/0 | 14 | - | core/package_profile.mojo:15 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_NUMERIC_IDENTICAL` | KEEP | none  | 4+4/7 | 8 | - | autograd/session_lifetime.mojo:15 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_QR_PARALLEL_SABOTAGE` | KEEP | none  | 2+0/0 | 12 | - | core/qr_multi_gpu.mojo:75 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_RESNORM_BATCH16` | KEEP | none  | 1+0/0 | 1 | - | bench/residual_norm_price_main.mojo:16 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RESNORM_BATCH2` | KEEP | none  | 1+0/0 | 1 | - | bench/residual_norm_price_main.mojo:18 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RESNORM_BATCH4` | KEEP | none  | 1+0/0 | 1 | - | bench/residual_norm_price_main.mojo:17 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RF_FAST_BATCH32K` | KEEP | none  | 1+4/0 | 5 | - | core/mojolearn_rf_run.mojo:233 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RUNTIME_MODEL_SELECTION_SABOTAGE` | KEEP | none  | 1+1/0 | 12 | - | bindings/runtime_model_selection_entry_pack.mojo:11 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_STEP_GLUE_DEFAULT_EVERY_COLUMN` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1273 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_ARITY_16` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:37 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_ARITY_64` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:39 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_SCHED_FUSED_TREE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:52 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_SCHED_RARY_TREE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:54 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_SCHED_TREE` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:46 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_SCHED_WARP` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:48 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_SCHED_WARP_LANE0` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix_svm.mojo:50 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SVM_TREE_FOLDS` | KEEP | none  | 2+0/0 | 3 | - | checks/kernel_matrix_svm.mojo:21 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_TARGET_SCRATCH_OFF` | KEEP | WIN EXP:747 KEEP for promotion: recorded maintenance windows clear; one pair per arm, varian | 1+0/0 | 1 | - | bindings/_mojolearn_x_prep.mojo:387 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE` | KEEP | none  | 1+2/0 | 5 | - | core/labeled_reference_host_predict.mojo:53 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_UMAP_IDENTICAL_LIVE_ROW` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1878 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD` | KEEP | none  | 1+0/0 | 2 | - | checks/kernel_matrix.mojo:1876 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_X_PREP_FAST_FIT_TRANSFORM_FUSED` | KEEP | OPEN EXP:1597 OPEN | 1+2/0 | 4 | - | preprocessing/estimator.mojo:346 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_AUTOGRAD_MAMBA3_OFF` | REMOVED | LOSE EXP:2191 **removed** | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_C37_FUSED_ACCUMULATE` | REMOVED | LOSE EXP:1725 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_C37_FUSED_ROWS` | REMOVED | LOSE EXP:1725 DROP (slower), code deleted | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_C52_PAIR_128` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_C52_PAIR_512` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_C58_TEAM64` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C19_ORDERED_128` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C19_ORDERED_32` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS` | REMOVED | LOSE EXP:1742 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS` | REMOVED | LOSE EXP:1743 DROP (slower), code deleted | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_GEMM_DIAG` | REMOVED | none  | 0+0/2 | 0 | - | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF` | REMOVED | LOSE EXP:2202 DELETED (serial shape) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_DRADIX_RANGE_PASSES` | REMOVED | WIN EXP:2060 KEEP (promoted 2026-10-11, w2-decide-algos; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GBDT_MULTI_HESS_ONE_WAIT_OFF` | REMOVED | LOSE EXP:2137 REMOVED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE` | REMOVED | LOSE EXP:1607 DROPPED-slower, define deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64` | REMOVED | LOSE EXP:1482 DROP | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_FS2` | REMOVED | LOSE EXP:1606 DROPPED-noise, define deleted | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8` | REMOVED | WIN EXP:1724 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY` | REMOVED | LOSE EXP:1608 DROPPED-noise, define deleted (EXPERIMENTS row "GROUP_TILES_BODY OPEN" above clo | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_NV_MICROTILE` | REMOVED | LOSE EXP:2037 LOSER, DELETED 2026-10-11 (c38d7bc89) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_TILE_SHORT_K` | REMOVED | LOSE EXP:1605 DROPPED, define deleted (arms equal measured losers / A/A) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1` | REMOVED | LOSE EXP:1571 RETIRED (merged, guard refuses) | 0+0/1 | 0 | c | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_16` | REMOVED | none  | 0+0/1 | 0 | c | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_2` | REMOVED | none  | 0+0/1 | 0 | c | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_32` | REMOVED | none  | 0+0/1 | 0 | c | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_4` | REMOVED | none  | 0+0/1 | 0 | c | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_IDN_LAB_CODES_DIRECT` | REMOVED | LOSE EXP:1653 LOSER, DELETED 2026-10-11 (de9ce7b39) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LAB_ONE_ENTRY` | REMOVED | LOSE EXP:1654 LOSER, DELETED 2026-10-11 (de9ce7b39) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LAB_UNIQUE_HIST` | REMOVED | LOSE EXP:1652 LOSER, DELETED 2026-10-11 (de9ce7b39) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MAMBA1_CHUNKED_SCAN` | REMOVED | LOSE EXP:2270 REJECTED (deleted 2026-10-28) | 0+0/2 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_XTY_TILED_OFF` | REMOVED | LOSE EXP:2202 DELETED (serial shape) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KMEANS_FAST_LAZY_SHIFT` | REMOVED | LOSE EXP:1926 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LOSS_SABOTAGE_REDUCE_SERIAL` | REMOVED | LOSE EXP:2204 DELETED (serial shape) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MI_ALL` | REMOVED | LOSE EXP:1879 DELETED (quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MI_FAST_FOLDS` | REMOVED | LOSE EXP:1880 DELETED (speed + quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_NI14_BOUNDED_COL2IM` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NI14_TILED_COL2IM` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_PCA_FAST_COLMEAN_OFF` | REMOVED | LOSE EXP:2202 DELETED (serial shape) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_PTIMPUTE_ALL` | REMOVED | LOSE EXP:1886 DELETED (quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PT_COLBATCH` | REMOVED | LOSE EXP:2280 DELETED (DROP quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PT_FOLD_NOX` | REMOVED | LOSE EXP:1889 DELETED (no gain) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PT_FUSED_TRANSFORM` | REMOVED | LOSE EXP:1888 DELETED (quality, with COLBATCH) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PT_SPEC` | REMOVED | LOSE EXP:1887 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_QN_TILED_MULTI_ALL` | REMOVED | LOSE EXP:2203 DELETED (serial shape) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_QN_TILED_MULTI_OFF` | REMOVED | LOSE EXP:2203 DELETED (serial shape) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_TSVD_FAST_COLVAR_OFF` | REMOVED | LOSE EXP:2202 DELETED (serial shape) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_XPREP_DEVICE_CODES` | REMOVED | LOSE EXP:1785 DELETED (loser) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_XPREP_NO_SLOT_HOP` | REMOVED | LOSE EXP:1786 DELETED (loser on the average) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_XPREP_PINNED_UPLOAD` | REMOVED | none  | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_XPREP_PINNED_UPLOAD_OFF` | REMOVED | LOSE EXP:1776 REVERTED + DELETED (loser; the old pageable copy is faster on both vendors) | 0+0/1 | 0 | icr | - | only the refusal guard remains |

### cluster (351)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_AP_EXACT` | DELETE-LOSER | LOSE EXP:495 SUPERSEDED | 2+4/0 | 13 | - | x_cluster/affinity.mojo:36 | expdel-01 |
| `MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` | KEEP-OPEN | LOSE grid classical-fixes.json removed | 1+1/0 | 2 | c | hdbscan/impl/detail/idn_switches.mojo:87 | expdel-06: KEEP-OPEN (2026-11-15): not a loser: an unmeasured IDENTICAL threshold candidate (default 46340 = PAIRWISE_MAX_ROWS, no change); classical-fixes.json `removed` says only "not registered: no board effect" (the board's 100,000 rows already take the sparse arm). Owed: A/B of =4096 / =16384 vs default on NVIDIA + AMD on HDBSCAN designs below 46,341 rows (non-board) |
| `MOJOLEARN_KNN_FAST_REFINE` | DELETE-LOSER | LOSE EXP:1291 REVERTED: no gain | 1+3/0 | 12 | c | neighbors/estimator.mojo:136 | expdel-06 |
| `MOJOLEARN_ANN_FAST_KNN_BIGD_OFF` | DELETE-OLD-PATH | WIN EXP:311 KEPT 8d43ec357 | 1+2/0 | 3 | - | x_ann/fast_env.mojo:34 | expdel-01 |
| `MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF` | DELETE-OLD-PATH | WIN EXP:524 KEPT, FAST+Apple default (`_OFF` off) | 1+4/0 | 5 | - | x_cluster/bisect_fast.mojo:48 | expdel-01 |
| `MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF` | DELETE-OLD-PATH | WIN EXP:321 KEPT 3852df59b (in the bundle) | 1+2/0 | 3 | - | x_ann/fast_env.mojo:71 | expdel-01 |
| `MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF` | DELETE-OLD-PATH | WIN EXP:325 DEFAULT (FAST+Apple), rollback `MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF` | 1+3/0 | 4 | c | x_ann/fast_env.mojo:93 | expdel-01 |
| `MOJOLEARN_CAGRA_FAST_IVFG_OFF` | DELETE-OLD-PATH | WIN EXP:323 KEPT 3852df59b | 1+4/0 | 5 | - | x_ann/fast_env.mojo:69 | expdel-01 |
| `MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF` | DELETE-OLD-PATH | WIN EXP:328 KEPT 8d43ec357 | 1+2/0 | 3 | - | x_ann/fast_env.mojo:44 | expdel-01 |
| `MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF` | DELETE-OLD-PATH | WIN EXP:462 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | 1+2/0 | 12 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:157 | expdel-06 |
| `MOJOLEARN_KMEANS_LAZY_SHIFT_IDN_OFF` | KEEP-OPEN | WIN grid gap-ivf.json new default (plan gaps-2026-10-08 section 6 item 3) | 1+3/0 | 4 | - | cluster/impl/kmeans_params.mojo:46 | expdel-06: KEEP-OPEN (2026-11-15): no measurement: gap-ivf.json `new default (plan gaps-2026-10-08 section 6 item 3)` is a plan, not an A/B; IDENTICAL bits move for early-converging lazy_shift fits. Owed: NVIDIA + AMD A/B (ivf, ivf-pq istella/taxi) default vs _OFF |
| `MOJOLEARN_KMEANS_LIVE_CORES_OFF` | KEEP-OPEN | WIN grid fg-ivf.json new default (read_ivf.md B5) | 1+2/0 | 3 | - | cluster/impl/detail/min_cluster_distance_compute.mojo:102 | expdel-06: KEEP-OPEN (2026-11-15): no measurement: EXP:1671 `PENDING A/B` (owed); fg-ivf.json `new default (read_ivf.md B5)` is a plan. Owed: NVIDIA + AMD A/B (ivf, ivf-pq coarse, k-means taxi/istella + one non-board k-means dataset) default vs _OFF |
| `MOJOLEARN_KNN_FAST_MMA_K64_OFF` | DELETE-OLD-PATH | WIN EXP:301 KEPT 857fd5804 | 1+0/0 | 1 | - | neighbors/impl/detail/fast_mma_knn.mojo:134 | expdel-06 |
| `MOJOLEARN_MBK_FAST_DEVSCAN_OFF` | DELETE-OLD-PATH | WIN EXP:1415 KEEP (FAST+Apple default, `_OFF`) | 1+3/0 | 4 | - | x_cluster/minibatch_ptr.mojo:64 | expdel-01 |
| `MOJOLEARN_SVGP_FAST_BLKCHOL_OFF` | DELETE-OLD-PATH | WIN EXP:879 DEFAULT (FAST+Apple), rollback `MOJOLEARN_SVGP_FAST_BLKCHOL_OFF`: M3 one run per | 1+2/0 | 3 | - | x_neighbors/iter_device.mojo:2318 | expdel-01 |
| `MOJOLEARN_SVGP_FAST_BSPLIT_OFF` | DELETE-OLD-PATH | WIN EXP:1424 KEEP (FAST+Apple default, `_OFF`) | 1+9/0 | 26 | - | x_neighbors/iter_device.mojo:2476 | expdel-01 |
| `MOJOLEARN_SVGP_FAST_COLSPLIT_OFF` | DELETE-OLD-PATH | WIN EXP:697 KEPT | 1+3/0 | 9 | - | x_neighbors/iter_device.mojo:2182 | expdel-01 |
| `MOJOLEARN_SVGP_FAST_RBFTILE_OFF` | DELETE-OLD-PATH | WIN EXP:880 DEFAULT (FAST+Apple), rollback `MOJOLEARN_SVGP_FAST_RBFTILE_OFF`: M3 one run per | 1+2/0 | 3 | - | x_neighbors/iter_device.mojo:1996 | expdel-01 |
| `MOJOLEARN_SVGP_FAST_SYMTILE_OFF` | DELETE-OLD-PATH | WIN EXP:696 KEPT | 1+3/0 | 9 | - | x_neighbors/iter_device.mojo:2176 | expdel-01 |
| `MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF` | DELETE-OLD-PATH | WIN EXP:869 DEFAULT (FAST+Apple), rollback MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF: M3 one run p | 1+3/0 | 4 | - | x_neighbors/kfeat_dev.mojo:69 | expdel-01 |
| `MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF` | DELETE-OLD-PATH | WIN EXP:580 KEPT 417ba1ded | 1+6/0 | 8 | - | x_neighbors/ocsvm_dev.mojo:79 | expdel-01 |
| `MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF` | DELETE-OLD-PATH | WIN EXP:359 KEPT (FAST+Apple default, `_OFF`) | 1+0/0 | 1 | - | x_neighbors/nan_cells_device.mojo:217 | expdel-06 |
| `MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF` | DELETE-OLD-PATH | WIN EXP:860 DEFAULT (FAST+Apple), rollback `MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF`: M3 one run | 1+4/0 | 5 | - | x_neighbors/kfeat_dev.mojo:70 | expdel-01 |
| `MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF` | DELETE-OLD-PATH | WIN EXP:525 KEPT, FAST+Apple default (`_OFF` off) | 1+3/0 | 4 | - | x_cluster/minibatch_fast.mojo:120 | expdel-01 |
| `MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF` | DELETE-OLD-PATH | WIN EXP:487 KEPT 4b1311c12 | 1+5/0 | 6 | - | x_cluster/meanshift_fast.mojo:48 | expdel-01 |
| `MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF` | DELETE-OLD-PATH | WIN EXP:488 KEPT 4b1311c12 | 1+12/0 | 13 | - | x_cluster/minibatch_fast.mojo:58 | expdel-01 |
| `MOJOLEARN_AFCL_G01` | KEEP-OPEN | none  | 1+2/1 | 3 | - | neighbors/impl/detail/fast_mma_knn.mojo:68 | owed A/B: the default vs MOJOLEARN_AFCL_G01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G02` | KEEP-OPEN | none  | 1+2/0 | 3 | - | neighbors/impl/detail/fast_topk_knn.mojo:48 | owed A/B: the default vs MOJOLEARN_AFCL_G02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G03` | KEEP-OPEN | none  | 1+2/0 | 3 | - | dbscan/impl/neighbors/fast_mma_eps.mojo:69 | owed A/B: the default vs MOJOLEARN_AFCL_G03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G05` | KEEP-OPEN | none  | 1+3/0 | 4 | - | cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:198 | owed A/B: the default vs MOJOLEARN_AFCL_G05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G07` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/minibatch_fast.mojo:66 | owed A/B: the default vs MOJOLEARN_AFCL_G07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G09` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ivf/impl/neighbors/ivf_flat/fast_ivf_scan.mojo:35 | owed A/B: the default vs MOJOLEARN_AFCL_G09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | umap/optimizer_fast.mojo:21 | owed A/B: the default vs MOJOLEARN_AFCL_G10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G11` | KEEP-OPEN | none  | 1+2/0 | 3 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:866 | owed A/B: the default vs MOJOLEARN_AFCL_G11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G12` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/meanshift_fast.mojo:56 | owed A/B: the default vs MOJOLEARN_AFCL_G12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_G14` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo:68 | owed A/B: the default vs MOJOLEARN_AFCL_G14 on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_CAGRA_TEAM` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_ann/switches.mojo:97 | owed A/B: the default vs MOJOLEARN_ANN3_CAGRA_TEAM on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_COARSE_SEED_OFF` | KEEP-OPEN | OPEN EXP:2135 OPEN | 1+1/0 | 2 | - | x_ann/switches.mojo:51 | owed A/B: default vs the old path MOJOLEARN_ANN3_COARSE_SEED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_PQ_SEED` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_ann/switches.mojo:43 | owed A/B: the default vs MOJOLEARN_ANN3_PQ_SEED on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_PREPARE_OFF` | KEEP-OPEN | none  | 1+4/0 | 22 | - | x_ann/switches.mojo:37 | owed A/B: default vs the old path MOJOLEARN_ANN3_PREPARE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_ROW_THREADS` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_ann/switches.mojo:74 | owed A/B: the default vs MOJOLEARN_ANN3_ROW_THREADS on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_SCAN_SELECT` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_ann/switches.mojo:82 | owed A/B: the default vs MOJOLEARN_ANN3_SCAN_SELECT on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_TRAINSET_COPY` | KEEP-OPEN | none  | 1+1/0 | 7 | - | x_ann/switches.mojo:68 | owed A/B: the default vs MOJOLEARN_ANN3_TRAINSET_COPY on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_TSNE_RB32` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_ann/switches.mojo:57 | owed A/B: the default vs MOJOLEARN_ANN3_TSNE_RB32 on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_TSNE_RB64` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_ann/switches.mojo:58 | owed A/B: the default vs MOJOLEARN_ANN3_TSNE_RB64 on NV+AMD, rows of its family |
| `MOJOLEARN_ANN3_TSNE_STEP_ROWS` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_ann/switches.mojo:90 | owed A/B: the default vs MOJOLEARN_ANN3_TSNE_STEP_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_ANN_COARSE_TILED_OFF` | KEEP-OPEN | OPEN EXP:1651 OPEN (A/B owed) | 1+2/0 | 3 | - | x_ann/ivf_scan_device.mojo:97 | owed A/B: default vs the old path MOJOLEARN_ANN_COARSE_TILED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ANN_SERIAL_SCAN` | KEEP-OPEN | none  | 2+3/0 | 16 | - | x_ann/ivf_scan_device.mojo:1102 | owed A/B: the default vs MOJOLEARN_ANN_SERIAL_SCAN on NV+AMD, rows of its family |
| `MOJOLEARN_AP_SPLIT_OFF` | KEEP-OPEN | WIN EXP:496 KEPT deee07721 | 1+2/0 | 3 | - | x_cluster/affinity.mojo:41 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_AP_SPLIT_OFF |
| `MOJOLEARN_C33_CHUNK_4` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/graph_controls.mojo:76 | owed A/B: the default vs MOJOLEARN_C33_CHUNK_4 on NV+AMD, rows of its family |
| `MOJOLEARN_C33_FROZEN_CHUNKS` | KEEP-OPEN | LOSE EXP:2323 DELETED (superseded) | 1+4/0 | 5 | ic | experiments/classical_identical_ideas/graph_controls.mojo:75 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_C34_PARALLEL_EDGES` | KEEP-OPEN | none  | 1+3/0 | 6 | - | experiments/classical_identical_ideas/graph_controls.mojo:77 | owed A/B: the default vs MOJOLEARN_C34_PARALLEL_EDGES on NV+AMD, rows of its family |
| `MOJOLEARN_C35_PACKED_LISTS` | KEEP-OPEN | none  | 1+8/0 | 30 | c | experiments/classical_identical_ideas/graph_controls.mojo:78 | owed A/B: the default vs MOJOLEARN_C35_PACKED_LISTS on NV+AMD, rows of its family |
| `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | KEEP-OPEN | none  | 1+11/0 | 53 | - | experiments/classical_identical_ideas/graph_controls.mojo:106 | owed A/B: the default vs MOJOLEARN_C43_RESIDENT_NORMALIZATION on NV+AMD, rows of its family |
| `MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | dbscan/impl/runner.mojo:242 | owed A/B: default vs the old path MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC` | KEEP-OPEN | none  | 1+5/0 | 26 | - | cluster/impl/detail/kmeans.mojo:171 | owed A/B: the default vs MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC on NV+AMD, rows of its family |
| `MOJOLEARN_FUSED_STAGE_FTZ_OFF` | KEEP-OPEN | none  | 1+4/0 | 6 | - | cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:218 | owed A/B: default vs the old path MOJOLEARN_FUSED_STAGE_FTZ_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GMM_INIT_FILL` | KEEP-OPEN | none  | 1+1/0 | 10 | - | mixture/estimator.mojo:648 | owed A/B: the default vs MOJOLEARN_GMM_INIT_FILL on NV+AMD, rows of its family |
| `MOJOLEARN_HDB_LINKAGE_DEVICE_OFF` | KEEP-OPEN | LOSE EXP:509 DROPPED-noise | 1+2/0 | 3 | - | hdbscan/impl/detail/fast_apple.mojo:72 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_HDB_ONE_SYNC` | KEEP-OPEN | WIN EXP:1914 KEPT (owed row withdrawn) | 1+3/0 | 4 | c | hdbscan/impl/detail/fast_apple.mojo:58 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_HDB_SELECT_DEVICE` | KEEP-OPEN | WIN EXP:1915 KEPT (owed row withdrawn) | 1+2/0 | 3 | c | hdbscan/impl/detail/fast_apple.mojo:82 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_HDB_SMR_TILED` | KEEP-OPEN | LOSE EXP:1911 DELETED | 1+4/0 | 11 | c | hdbscan/impl/detail/fast_apple.mojo:30 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_HDB_SMR_TILED_OFF` | KEEP-OPEN | LOSE EXP:1911 DELETED | 1+4/0 | 11 | - | hdbscan/impl/detail/fast_apple.mojo:31 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_ann/knn_device.mojo:48 | owed A/B: default vs the old path MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AP_CONV_CHUNK16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/device_ops.mojo:690 | owed A/B: the default vs MOJOLEARN_IDN_AP_CONV_CHUNK16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AP_CONV_CHUNK32` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/device_ops.mojo:691 | owed A/B: the default vs MOJOLEARN_IDN_AP_CONV_CHUNK32 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AP_CONV_CHUNK4` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/device_ops.mojo:688 | owed A/B: the default vs MOJOLEARN_IDN_AP_CONV_CHUNK4 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AP_DEVICE_CONV_OFF` | KEEP-OPEN | none  | 1+2/0 | 33 | - | x_cluster/device_ops.mojo:683 | owed A/B: default vs the old path MOJOLEARN_IDN_AP_DEVICE_CONV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AP_R_TOP2_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cluster/device_ops.mojo:611 | owed A/B: default vs the old path MOJOLEARN_IDN_AP_R_TOP2_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_cluster/device_ops.mojo:1666 | owed A/B: default vs the old path MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF` | KEEP-OPEN | none  | 1+2/0 | 12 | - | x_cluster/bodies.mojo:557 | owed A/B: default vs the old path MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF` | KEEP-OPEN | none  | 1+3/0 | 24 | - | x_cluster/bisect.mojo:69 | owed A/B: default vs the old path MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF` | KEEP-OPEN | none  | 1+3/0 | 42 | - | x_cluster/bisect.mojo:47 | owed A/B: default vs the old path MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF` | KEEP-OPEN | WIN EXP:1971 PROMOTED (IDENTICAL default, together) | 1+1/0 | 2 | - | neighbors/impl/ball_cover/registers.mojo:982 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | dbscan/impl/runner.mojo:271 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16` | KEEP-OPEN | none  | 1+4/0 | 5 | - | dbscan/impl/sparse/detail/csr.mojo:129 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_CHUNK16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK32` | KEEP-OPEN | none  | 1+0/0 | 1 | - | dbscan/impl/sparse/detail/csr.mojo:131 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_CHUNK32 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK4` | KEEP-OPEN | none  | 1+4/0 | 5 | - | dbscan/impl/sparse/detail/csr.mojo:126 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_CHUNK4 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF` | KEEP-OPEN | none  | 1+6/0 | 59 | - | dbscan/impl/sparse/detail/csr.mojo:108 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF` | KEEP-OPEN | none  | 1+3/0 | 13 | - | dbscan/impl/sparse/detail/csr.mojo:177 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT16` | KEEP-OPEN | none  | 1+0/0 | 1 | - | dbscan/impl/sparse/detail/csr.mojo:151 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_SPLIT16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT4` | KEEP-OPEN | none  | 1+5/0 | 6 | - | dbscan/impl/sparse/detail/csr.mojo:146 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_SPLIT4 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT8` | KEEP-OPEN | none  | 1+5/0 | 6 | - | dbscan/impl/sparse/detail/csr.mojo:149 | owed A/B: the default vs MOJOLEARN_IDN_DBSCAN_CC_SPLIT8 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | dbscan/estimator.mojo:148 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF` | KEEP-OPEN | none  | 2+4/0 | 10 | - | dbscan/estimator.mojo:161 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF` | KEEP-OPEN | WIN EXP:2010 KEEP | 1+1/0 | 2 | - | dbscan/impl/edge_free.mojo:116 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF` | KEEP-OPEN | WIN EXP:1971 PROMOTED (IDENTICAL default, together) | 1+4/0 | 5 | - | neighbors/impl/ball_cover/registers.mojo:980 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | dbscan/impl/runner.mojo:230 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | dbscan/impl/runner.mojo:255 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF` | KEEP-OPEN | none  | 1+2/0 | 30 | - | dbscan/impl/dbscan.mojo:85 | owed A/B: default vs the old path MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK` | KEEP-OPEN | none  | 1+4/0 | 5 | - | hierarchy/impl/cluster/detail/dendrogram_device.mojo:102 | owed A/B: the default vs MOJOLEARN_IDN_DENDRO_HOOK_CHUNK on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | hierarchy/impl/cluster/detail/dendrogram_device.mojo:106 | owed A/B: default vs the old path MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF` | KEEP-OPEN | none  | 1+2/0 | 29 | - | hierarchy/impl/cluster/detail/dendrogram_device.mojo:69 | owed A/B: default vs the old path MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_DENDRO_UNION_OFF` | KEEP-OPEN | OPEN EXP:1715 owed | 1+1/0 | 2 | - | hierarchy/impl/cluster/detail/dendrogram_device.mojo:86 | owed A/B: default vs the old path MOJOLEARN_IDN_DENDRO_UNION_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF` | KEEP-OPEN | none  | 1+4/0 | 6 | - | mixture/chol_order.mojo:85 | owed A/B: default vs the old path MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GMM_NK_LEVELS_OFF` | KEEP-OPEN | none  | 1+3/0 | 16 | - | mixture/nk_order.mojo:32 | owed A/B: default vs the old path MOJOLEARN_IDN_GMM_NK_LEVELS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hdbscan/impl/detail/idn_switches.mojo:35 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | hdbscan/impl/detail/idn_switches.mojo:51 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_MST_SEED_KNN_OFF` | KEEP-OPEN | LOSE EXP:2064 LOSER-noise (deleted 826f3d57e; docs/TOMBSTONES.md) | 1+5/0 | 6 | - | hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:615 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_HDB_ONE_SYNC_OFF` | KEEP-OPEN | WIN EXP:1914 KEPT (owed row withdrawn) | 1+2/0 | 3 | - | hdbscan/impl/detail/idn_switches.mojo:27 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF` | KEEP-OPEN | OPEN EXP:1715 owed | 1+3/0 | 19 | - | hdbscan/impl/hdbscan_door_impl.mojo:25 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hdbscan/impl/detail/idn_switches.mojo:70 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | hdbscan/impl/detail/idn_switches.mojo:77 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC_OFF` | KEEP-OPEN | OPEN EXP:1714 owed | 1+4/0 | 22 | - | hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:710 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF` | KEEP-OPEN | WIN EXP:1915 KEPT (owed row withdrawn) | 1+1/0 | 2 | - | hdbscan/impl/detail/idn_switches.mojo:43 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hdbscan/impl/detail/idn_switches.mojo:61 | owed A/B: default vs the old path MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ivf/impl/neighbors/ivf_flat/ivf_finite_device.mojo:45 | owed A/B: default vs the old path MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF` | KEEP-OPEN | none  | 1+6/0 | 12 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:123 | owed A/B: default vs the old path MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF` | KEEP-OPEN | none  | 1+3/0 | 10 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo:234 | owed A/B: default vs the old path MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | cluster/impl/detail/kmeans.mojo:164 | owed A/B: default vs the old path MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK` | KEEP-OPEN | LOSE EXP:1571 RETIRED (merged, guard refuses) | 1+2/1 | 3 | c | cluster/impl/detail/kmeans.mojo:233 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_KMEANS_DEVICE_CONV_OFF` | KEEP-OPEN | LOSE EXP:2274 REFACTORED (2026-11-08) | 1+4/0 | 34 | c | cluster/impl/detail/kmeans.mojo:210 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | cluster/impl/detail/kmeans.mojo:185 | owed A/B: default vs the old path MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KMEANS_INCR_INIT_OFF` | KEEP-OPEN | LOSE EXP:1517 DROPPED (deleted) | 1+2/0 | 3 | - | cluster/impl/detail/kmeans.mojo:894 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_KMEANS_INIT_PSI_DEVICE_OFF` | KEEP-OPEN | LOSE EXP:1517 DROPPED (deleted) | 1+3/0 | 17 | - | cluster/impl/detail/kmeans.mojo:924 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_KNN_CERTIFIED_REACH` | KEEP-OPEN | none  | 3+0/0 | 9 | - | neighbors/impl/detail/knn_brute_force.mojo:274 | owed A/B: the default vs MOJOLEARN_IDN_KNN_CERTIFIED_REACH on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | neighbors/impl/selection/knn.mojo:115 | owed A/B: default vs the old path MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF` | KEEP-OPEN | none  | 1+2/0 | 20 | - | neighbors/impl/selection/knn.mojo:100 | owed A/B: default vs the old path MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER` | KEEP-OPEN | OPEN EXP:2420 PENDING (A = 0.8.55's form: only the multi-output label transpose resolves the b | 1+1/0 | 2 | - | neighbors/neighbors_knn_run.mojo:433 | owed A/B: the default vs MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD` | KEEP-OPEN | OPEN EXP:2412 PENDING (A = int32 labels copied straight into the label columns) | 1+1/0 | 3 | - | neighbors/neighbors_knn_run.mojo:342 | owed A/B: default vs the old path MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF` | KEEP-OPEN | none  | 1+2/0 | 5 | - | neighbors/impl/knn/knn.mojo:50 | owed A/B: default vs the old path MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LANCZOS_POOL_OFF` | KEEP-OPEN | none  | 1+4/0 | 23 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:2012 | owed A/B: default vs the old path MOJOLEARN_IDN_LANCZOS_POOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF` | KEEP-OPEN | none  | 1+4/0 | 32 | - | spectral/spmv_order.mojo:51 | owed A/B: default vs the old path MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MBK_RESIDENT_OFF` | KEEP-OPEN | WIN EXP:2050 KEEP (promoted 2026-10-11; _OFF restores) | 1+3/0 | 4 | - | x_cluster/minibatch_ptr.mojo:88 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_MEANSHIFT_GRID_OFF` | KEEP-OPEN | none  | 1+9/0 | 18 | - | x_cluster/bodies.mojo:127 | owed A/B: default vs the old path MOJOLEARN_IDN_MEANSHIFT_GRID_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MINIBATCH_GROUP32` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_cluster/minibatch.mojo:57 | owed A/B: the default vs MOJOLEARN_IDN_MINIBATCH_GROUP32 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MINIBATCH_GROUP64` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_cluster/minibatch.mojo:58 | owed A/B: the default vs MOJOLEARN_IDN_MINIBATCH_GROUP64 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MINIBATCH_GROUP8` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_cluster/minibatch.mojo:55 | owed A/B: the default vs MOJOLEARN_IDN_MINIBATCH_GROUP8 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MST_LABEL_JUMP_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hierarchy/impl/sparse/solver/detail/mst_kernels.mojo:81 | owed A/B: default vs the old path MOJOLEARN_IDN_MST_LABEL_JUMP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MST_PAR_COMPACT_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | hierarchy/impl/sparse/solver/detail/mst_kernels.mojo:104 | owed A/B: default vs the old path MOJOLEARN_IDN_MST_PAR_COMPACT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | hierarchy/impl/sparse/solver/detail/mst_kernels.mojo:96 | owed A/B: default vs the old path MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MST_SCAN_LANES` | KEEP-OPEN | none  | 1+1/0 | 2 | - | hierarchy/impl/sparse/solver/detail/mst_kernels.mojo:112 | owed A/B: the default vs MOJOLEARN_IDN_MST_SCAN_LANES on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NC_CHUNKED_OFF` | KEEP-OPEN | none  | 1+9/0 | 38 | - | x_neighbors/items.mojo:58 | owed A/B: default vs the old path MOJOLEARN_IDN_NC_CHUNKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NC_FIT_LEAN_OFF` | KEEP-OPEN | WIN EXP:2049 KEEP (promoted 2026-10-11; _OFF restores) | 1+0/0 | 1 | - | x_neighbors/nan_cells_device.mojo:240 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_OCSVM_2L_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | x_neighbors/ocsvm_dev.mojo:96 | owed A/B: default vs the old path MOJOLEARN_IDN_OCSVM_2L_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OCSVM_CHUNK256` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_neighbors/ocsvm_dev.mojo:99 | owed A/B: the default vs MOJOLEARN_IDN_OCSVM_CHUNK256 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OCSVM_RES_OFF` | KEEP-OPEN | none  | 1+8/0 | 11 | - | x_neighbors/ocsvm_dev.mojo:93 | owed A/B: default vs the old path MOJOLEARN_IDN_OCSVM_RES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cluster/device_post.mojo:373 | owed A/B: default vs the old path MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF` | KEEP-OPEN | WIN EXP:1775 PROMOTED (IDENTICAL default) | 1+3/0 | 4 | - | x_ann/switches.mojo:130 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_PQ_LUT_TILED_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_ann/vsearch_fast.mojo:79 | owed A/B: default vs the old path MOJOLEARN_IDN_PQ_LUT_TILED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF` | KEEP-OPEN | WIN EXP:2065 KEEP (promoted 2026-10-11, quality up; _OFF restores) | 1+2/0 | 3 | - | x_ann/switches.mojo:154 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_SL_SPARSE_MST` | KEEP-OPEN | none  | 1+2/0 | 27 | - | hierarchy/impl/cluster/detail/single_linkage.mojo:81 | owed A/B: the default vs MOJOLEARN_IDN_SL_SPARSE_MST on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SMR_RADIX_RANK_OFF` | KEEP-OPEN | none  | 1+2/0 | 47 | - | hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:1104 | owed A/B: default vs the old path MOJOLEARN_IDN_SMR_RADIX_RANK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | spectral/impl/preprocessing/detail/fast_graph.mojo:67 | owed A/B: default vs the old path MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF` | KEEP-OPEN | none  | 1+4/0 | 9 | - | spectral/impl/labels_device.mojo:25 | owed A/B: default vs the old path MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:2317 | owed A/B: default vs the old path MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SPMV_LANES_OFF` | KEEP-OPEN | none  | 1+7/0 | 35 | - | spectral/spmv_order.mojo:31 | owed A/B: default vs the old path MOJOLEARN_IDN_SPMV_LANES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SVGP_BLKCHOL_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_neighbors/iter_device.mojo:2327 | owed A/B: default vs the old path MOJOLEARN_IDN_SVGP_BLKCHOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF` | KEEP-OPEN | WIN EXP:2036 KEEP (default) | 1+1/0 | 2 | - | x_neighbors/svgp_b32_gate.mojo:52 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_SVGP_DOT2_OFF` | KEEP-OPEN | none  | 1+3/0 | 48 | - | x_neighbors/svgp_ff.mojo:64 | owed A/B: default vs the old path MOJOLEARN_IDN_SVGP_DOT2_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SVGP_RBFTILE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_neighbors/iter_device.mojo:2004 | owed A/B: default vs the old path MOJOLEARN_IDN_SVGP_RBFTILE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | spectral/impl/sparse/linalg/detail/symmetrize.mojo:65 | owed A/B: default vs the old path MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF` | KEEP-OPEN | WIN EXP:1822 PROMOTED (IDENTICAL default, with REP_TILE) | 1+3/0 | 9 | - | x_ann/knn_device.mojo:358 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF` | KEEP-OPEN | none  | 1+4/0 | 8 | - | x_ann/tsne_device.mojo:209 | owed A/B: default vs the old path MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF` | KEEP-OPEN | none  | 1+11/0 | 72 | - | x_ann/tsne_core.mojo:62 | owed A/B: default vs the old path MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSNE_REP_TILE_OFF` | KEEP-OPEN | WIN EXP:1823 PROMOTED (IDENTICAL default, with KNN_TILE_SELECT) | 1+5/0 | 11 | - | x_ann/tsne_device.mojo:488 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_XN_KNN_LEAN_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_neighbors/iter_device.mojo:884 | owed A/B: default vs the old path MOJOLEARN_IDN_XN_KNN_LEAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XN_KNN_PRUNE_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_neighbors/iter_device.mojo:885 | owed A/B: default vs the old path MOJOLEARN_IDN_XN_KNN_PRUNE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XN_SKETCH_CTR_OFF` | KEEP-OPEN | none  | 1+8/0 | 13 | - | x_neighbors/kfeat_rng.mojo:55 | owed A/B: default vs the old path MOJOLEARN_IDN_XN_SKETCH_CTR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XN_UNIT_DEV_OFF` | KEEP-OPEN | none  | 1+7/0 | 10 | - | x_neighbors/ocsvm_init.mojo:121 | owed A/B: default vs the old path MOJOLEARN_IDN_XN_UNIT_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF` | KEEP-OPEN | OPEN EXP:2135 OPEN | 1+4/0 | 5 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:174 | owed A/B: default vs the old path MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IVF_DEVICE_VALIDATE_OFF` | KEEP-OPEN | WIN EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote | 1+5/0 | 13 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:198 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IVF_DIRECT_DISTANCE_OFF` | KEEP-OPEN | WIN EXP:1642 PROMOTED | 1+17/0 | 38 | - | experiments/classical_identical_ideas/graph_controls.mojo:51 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo:198 | owed A/B: default vs the old path MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IVF_IDENTICAL_SCAN_OFF` | KEEP-OPEN | none  | 1+7/0 | 10 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo:194 | owed A/B: default vs the old path MOJOLEARN_IVF_IDENTICAL_SCAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF` | KEEP-OPEN | LOSE EXP:1803 CONFIRMED (kept default) | 1+1/0 | 2 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo:123 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF` | KEEP-OPEN | LOSE EXP:1803 CONFIRMED (kept default) | 1+2/0 | 3 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo:145 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IVF_KMEANS_LAZY_SHIFT_OFF` | KEEP-OPEN | WIN EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote | 2+3/0 | 5 | - | cluster/impl/detail/kmeans.mojo:1511 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IVF_TRAINSET_STRIDE_OFF` | KEEP-OPEN | WIN grid gap-ivf.json new default (plan gaps-2026-10-08 section 6 item 2) | 1+2/0 | 3 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo:107 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_IVF_TRAINSET_STRIDE_OFF |
| `MOJOLEARN_KMEANS_BLOCK_ACC_OFF` | KEEP-OPEN | none  | 1+5/0 | 27 | - | cluster/impl/detail/kmeans.mojo:169 | owed A/B: default vs the old path MOJOLEARN_KMEANS_BLOCK_ACC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KMEANS_FAST_PP_NOSYNC_OFF` | KEEP-OPEN | LOSE EXP:1517 DROPPED (deleted) | 1+0/0 | 1 | - | cluster/impl/detail/kmeans.mojo:421 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_KMEANS_ID_INCR_INIT_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | cluster/impl/detail/kmeans.mojo:907 | owed A/B: default vs the old path MOJOLEARN_KMEANS_ID_INCR_INIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KMEANS_ROW_ASSIGN` | KEEP-OPEN | WIN EXP:1927 KEPT (owed row withdrawn) | 1+3/0 | 4 | c | experiments/classical_identical_ideas/graph_controls.mojo:63 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512` | KEEP-OPEN | none  | 1+7/0 | 18 | - | neighbors/estimator.mojo:288 | owed A/B: the default vs MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512 on NV+AMD, rows of its family |
| `MOJOLEARN_KNN_LEGACY_QUERY_TILE` | KEEP-OPEN | none  | 1+7/0 | 8 | - | neighbors/estimator.mojo:287 | owed A/B: default vs the old path MOJOLEARN_KNN_LEGACY_QUERY_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_KNN_SCRATCH_POOL_OFF` | KEEP-OPEN | OPEN EXP:1658 owed | 1+0/0 | 1 | - | neighbors/impl/detail/knn_brute_force.mojo:486 | owed A/B: default vs the old path MOJOLEARN_KNN_SCRATCH_POOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KNN_SELECT_TRIAL` | KEEP-OPEN | none  | 4+12/0 | 184 | - | neighbors/checks/select_smallk_identical_candidate.mojo:433 | owed A/B: the default vs MOJOLEARN_KNN_SELECT_TRIAL on NV+AMD, rows of its family |
| `MOJOLEARN_KNN_WIDE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_ann/knn_device.mojo:210 | owed A/B: default vs the old path MOJOLEARN_KNN_WIDE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF` | KEEP-OPEN | none  | 1+4/0 | 53 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:1268 | owed A/B: default vs the old path MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LANCZOS_ID_DEV_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:1262 | owed A/B: default vs the old path MOJOLEARN_LANCZOS_ID_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N` | KEEP-OPEN | OPEN EXP:1335 OWED | 1+0/0 | 4 | - | x_ann/cagra_fast_knn.mojo:431 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_SHAPE_KNN_BOARD` | KEEP-OPEN | none  | 1+0/0 | 12 | - | neighbors/impl/detail/knn_brute_force.mojo:200 | owed A/B: default vs the old path MOJOLEARN_LEGACY_SHAPE_KNN_BOARD on NV+AMD, rows of its family |
| `MOJOLEARN_MBK_ZEROCOPY_OFF` | KEEP-OPEN | WIN EXP:486 KEPT 51493c7a8 | 1+4/0 | 5 | - | x_cluster/minibatch_ptr.mojo:46 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MBK_ZEROCOPY_OFF |
| `MOJOLEARN_MEANSHIFT_BLOCK` | KEEP-OPEN | none  | 1+2/0 | 11 | - | x_cluster/device_ops.mojo:453 | owed A/B: the default vs MOJOLEARN_MEANSHIFT_BLOCK on NV+AMD, rows of its family |
| `MOJOLEARN_MINIBATCH_ONE_PASS` | KEEP-OPEN | none  | 1+2/0 | 7 | - | x_cluster/minibatch.mojo:41 | owed A/B: the default vs MOJOLEARN_MINIBATCH_ONE_PASS on NV+AMD, rows of its family |
| `MOJOLEARN_MOMENTS_ROWS` | KEEP-OPEN | none  | 1+2/0 | 22 | - | x_cluster/device_ops.mojo:1035 | owed A/B: the default vs MOJOLEARN_MOMENTS_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF` | KEEP-OPEN | LOSE EXP:516 DROP | 2+4/0 | 8 | - | x_cluster/device_ops.mojo:1248 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_OPTICS_SIMD` | KEEP-OPEN | none  | 1+2/0 | 17 | - | x_cluster/optics.mojo:39 | owed A/B: the default vs MOJOLEARN_OPTICS_SIMD on NV+AMD, rows of its family |
| `MOJOLEARN_PQ_ASSIGN_UNSTAGED` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_ann/ivf_pq_device.mojo:156 | owed A/B: the default vs MOJOLEARN_PQ_ASSIGN_UNSTAGED on NV+AMD, rows of its family |
| `MOJOLEARN_PQ_LUT_TILED_OFF` | KEEP-OPEN | WIN EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote | 1+5/0 | 6 | - | x_ann/vsearch_fast.mojo:49 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PQ_SCAN_FUSED_OFF` | KEEP-OPEN | LOSE EXP:1801 DELETED (loser) | 1+5/0 | 6 | - | x_ann/vsearch_fast.mojo:59 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_PROBA64_QOLD` | KEEP-OPEN | WIN EXP:1284 QUALITY-FIX KEPT: FAST default | 2+18/0 | 24 | - | x_neighbors/proba64_nc.mojo:22 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PURITY_4_OFF` | KEEP-OPEN | none  | 1+3/0 | 13 | - | x_neighbors/items.mojo:643 | owed A/B: default vs the old path MOJOLEARN_PURITY_4_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_PURITY_5_OFF` | KEEP-OPEN | none  | 1+1/0 | 19 | - | x_neighbors/svgp_ff.mojo:491 | owed A/B: default vs the old path MOJOLEARN_PURITY_5_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SMR_TILED_OFF` | KEEP-OPEN | none  | 1+5/0 | 25 | - | hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:185 | owed A/B: default vs the old path MOJOLEARN_SMR_TILED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SPMV_WARP_OFF` | KEEP-OPEN | none  | 1+3/0 | 6 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:857 | owed A/B: default vs the old path MOJOLEARN_SPMV_WARP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_TSNE_SPLIT_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | x_ann/tsne_device.mojo:303 | owed A/B: default vs the old path MOJOLEARN_TSNE_SPLIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_FUSED_EPOCH_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | umap/optimizer_fast.mojo:215 | owed A/B: default vs the old path MOJOLEARN_UMAP_FUSED_EPOCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_INIT_TOL` | KEEP-OPEN | none  | 1+3/0 | 8 | - | umap/spectral_init.mojo:27 | owed A/B: the default vs MOJOLEARN_UMAP_INIT_TOL on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_INIT_TOL_OFF` | KEEP-OPEN | none  | 1+3/0 | 8 | - | umap/spectral_init.mojo:28 | owed A/B: default vs the old path MOJOLEARN_UMAP_INIT_TOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_UMAP_LIVE_BOTH_ARM` | KEEP-OPEN | none  | 2+10/0 | 25 | - | umap/host/umap_oracle.mojo:142 | owed A/B: the default vs MOJOLEARN_UMAP_LIVE_BOTH_ARM on NV+AMD, rows of its family |
| `MOJOLEARN_VSEARCH_ALL` | KEEP-OPEN | WIN EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote | 1+0/0 | 1 | c | x_ann/vsearch_fast.mojo:29 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_WARD_ROUNDS` | KEEP-OPEN | none  | 1+2/0 | 7 | - | x_cluster/agglo.mojo:156 | owed A/B: the default vs MOJOLEARN_WARD_ROUNDS on NV+AMD, rows of its family |
| `MOJOLEARN_XCLUSTER_KPP_DISTINCT` | KEEP-OPEN | LOSE EXP:1517 DROPPED (deleted) | 1+3/0 | 8 | c | experiments/classical_identical_ideas/graph_controls.mojo:100 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_XC_ALLOC` | KEEP-OPEN | none  | 1+2/0 | 5 | - | x_cluster/optics.mojo:40 | owed A/B: the default vs MOJOLEARN_XC_ALLOC on NV+AMD, rows of its family |
| `MOJOLEARN_XN_FAST_CLS2_OCSVM_2L_OFF` | KEEP-OPEN | WIN EXP:357 KEEP, FAST+Apple default (`_OFF` off) | 1+5/0 | 6 | - | x_neighbors/ocsvm_dev.mojo:95 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256_OFF` | KEEP-OPEN | WIN EXP:358 KEEP, FAST+Apple default (`_OFF` off) | 1+2/0 | 3 | - | x_neighbors/ocsvm_dev.mojo:98 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF` | KEEP-OPEN | WIN EXP:1049 DEFAULT: M2 default/rollback PASS, final M3 knn-lean-gpu-default-quality-2026100 | 1+2/0 | 3 | - | x_neighbors/nan_cells_device.mojo:226 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_XN_IMPUTE_NO_SPLIT` | KEEP-OPEN | none  | 1+0/0 | 2 | - | x_neighbors/iter_device.mojo:1248 | owed A/B: the default vs MOJOLEARN_XN_IMPUTE_NO_SPLIT on NV+AMD, rows of its family |
| `MOJOLEARN_XN_IMPUTE_ROWDIST_OFF` | KEEP-OPEN | OPEN EXP:1650 OPEN (A/B owed) | 1+2/0 | 3 | - | x_neighbors/iter_device.mojo:1527 | owed A/B: default vs the old path MOJOLEARN_XN_IMPUTE_ROWDIST_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_XN_KNN_ROWWISE` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_neighbors/iter_device.mojo:1094 | owed A/B: the default vs MOJOLEARN_XN_KNN_ROWWISE on NV+AMD, rows of its family |
| `MOJOLEARN_XN_PCS_CELL` | KEEP-OPEN | none  | 1+0/0 | 2 | - | x_neighbors/iter_device.mojo:826 | owed A/B: the default vs MOJOLEARN_XN_PCS_CELL on NV+AMD, rows of its family |
| `MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF` | KEEP-OPEN | WIN EXP:520 KEEP, FAST+Apple default (`_OFF` off) | 1+5/0 | 8 | - | x_cluster/minibatch_fast.mojo:104 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_AFCL_G06` | KEEP | none  | 1+2/0 | 3 | - | cluster/checks/reduce_by_key.mojo:789 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFCL_G08` | KEEP | none  | 1+3/0 | 4 | - | mixture/checks/estep.mojo:144 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_BGMM_FAST_MOMENTS_GEMM_OFF` | KEEP | OPEN EXP:500 HOLD-quality | 1+2/0 | 3 | - | x_cluster/device_ops.mojo:1247 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_BORUVKA_FAST_MMA_OFF` | KEEP | none  | 1+4/0 | 6 | - | hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo:55 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CAGRA_FAST_ITERS_OFF` | KEEP | OPEN EXP:2126 owed (boxes down) | 1+3/0 | 4 | - | x_ann/fast_env.mojo:126 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CAGRA_FAST_SEEDS_OFF` | KEEP | OPEN EXP:2126 owed (boxes down) | 1+4/0 | 5 | - | x_ann/fast_env.mojo:124 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CLUSTER_FAST_STOP_GROUP8` | KEEP | none  | 1+5/0 | 6 | - | x_cluster/minibatch_fast.mojo:89 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DBSCAN_FAST_CC_BATCH` | KEEP | OPEN EXP:503 OPEN | 1+0/0 | 1 | - | dbscan/impl/sparse/detail/csr.mojo:85 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DBSCAN_FAST_DENSE_ADJ` | KEEP | none  | 1+0/0 | 1 | - | dbscan/impl/dbscan.mojo:346 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DBSCAN_FAST_DIRECT_OUT` | KEEP | none  | 1+0/0 | 1 | - | dbscan/estimator.mojo:171 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DBSCAN_FAST_MMA_OFF` | KEEP | none  | 1+3/0 | 5 | - | dbscan/impl/neighbors/fast_mma_eps.mojo:60 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DBSCAN_PARALLEL_SABOTAGE` | KEEP | none  | 2+0/0 | 8 | - | dbscan/impl/multi_gpu.mojo:148 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GMM_ESTEP_STACK_OFF` | KEEP | none  | 1+2/0 | 3 | - | mixture/checks/estep.mojo:1032 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_BIG_CHOL_OFF` | KEEP | LOSE EXP:489 DROPPED-slower | 1+3/0 | 4 | - | mixture/checks/mstep.mojo:1424 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_FF_COV_OFF` | KEEP | none  | 1+4/0 | 5 | - | mixture/checks/mstep.mojo:1093 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_FUSED_CHOL_OFF` | KEEP | none  | 1+6/0 | 7 | - | mixture/checks/mstep.mojo:963 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_FUSED_COV_OFF` | KEEP | none  | 1+5/0 | 6 | - | mixture/checks/mstep.mojo:791 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_FUSED_ESTEP_OFF` | KEEP | none  | 1+3/0 | 4 | - | mixture/checks/estep.mojo:519 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_PINNED_COV` | KEEP | none  | 1+6/0 | 7 | - | mixture/checks/mstep.mojo:693 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_FAST_SERIAL_MEANLL` | KEEP | none  | 1+4/0 | 9 | - | mixture/checks/estep.mojo:637 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_MEANLL_AHEAD_OFF` | KEEP | none  | 1+3/0 | 4 | - | mixture/checks/estep.mojo:735 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_MEANLL_PIPE_OFF` | KEEP | none  | 1+0/0 | 7 | - | mixture/checks/estep.mojo:780 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_NK_AHEAD_OFF` | KEEP | none  | 1+3/0 | 4 | - | mixture/checks/mstep.mojo:271 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_NK_T_OFF` | KEEP | none  | 1+2/0 | 13 | - | mixture/checks/mstep.mojo:280 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GMM_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 5 | - | mixture/multi_gpu.mojo:247 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HDBSCAN_PREDICT_SABOTAGE` | KEEP | none  | 1+3/0 | 12 | - | hdbscan/host/hdbscan_host_oracle.mojo:173 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 4 | - | hierarchy/impl/cluster/detail/multi_gpu.mojo:57 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IDN_GMM_CENTER_PAIR` | KEEP | none  | 1+2/0 | 3 | - | mixture/checks/mstep.mojo:624 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GMM_ESTEP_STACK` | KEEP | none  | 1+2/0 | 3 | - | mixture/checks/estep.mojo:1026 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF` | KEEP | none  | 1+0/0 | 1 | - | cluster/checks/reduce_by_key.mojo:802 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_KMEANS_ACC_ROWS_4096` | KEEP | none  | 1+2/0 | 3 | - | cluster/checks/reduce_by_key.mojo:795 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_KMEANS_CENTROID_FOLD_OFF` | KEEP | WIN EXP:1639 PROMOTED | 1+7/0 | 20 | - | cluster/checks/reduce_by_key.mojo:861 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IVF_FAST_SCAN_OFF` | KEEP | none  | 1+3/0 | 4 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo:254 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_IVF_FAST_TRAINSET_OFF` | KEEP | none  | 1+3/0 | 4 | - | ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:137 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KMEANS_FAST_INCR_INIT_OFF` | KEEP | none  | 1+6/0 | 17 | - | cluster/impl/detail/kmeans.mojo:902 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KMEANS_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 10 | - | cluster/multi_gpu.mojo:93 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KMEANS_PP_NOSYNC_APPLE_ONLY` | KEEP | none  | 1+0/0 | 1 | - | cluster/impl/detail/kmeans.mojo:418 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_APPLE_MMA_OFF` | KEEP | none  | 1+2/0 | 3 | - | neighbors/impl/detail/knn_brute_force.mojo:235 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_APPLE_MMA_SABOTAGE` | KEEP | none  | 1+0/0 | 3 | - | neighbors/checks/apple_mma_distance.mojo:159 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_BOUNDED_DIAG_NOBOUND` | KEEP | none  | 1+3/0 | 4 | - | neighbors/checks/smem_distance_tile.mojo:129 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_BOUNDED_DIAG_NOBREAK` | KEEP | none  | 1+2/0 | 3 | - | neighbors/checks/smem_distance_tile.mojo:128 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_BOUNDED_TOPK_SABOTAGE` | KEEP | none  | 1+1/0 | 3 | - | neighbors/checks/smem_distance_tile.mojo:125 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE` | KEEP | none  | 1+0/0 | 2 | - | neighbors/checks/pinned_distance_tile.mojo:231 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_FAST_FORCE_FUSED` | KEEP | none  | 1+0/0 | 2 | - | neighbors/impl/detail/knn_brute_force.mojo:2233 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_FAST_FORCE_TILED` | KEEP | none  | 1+0/0 | 2 | - | neighbors/impl/detail/knn_brute_force.mojo:2235 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_FAST_MMA_OFF` | KEEP | none  | 1+2/1 | 4 | - | neighbors/impl/detail/fast_mma_knn.mojo:94 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_FAST_QUERY_GROUP4` | KEEP | none  | 1+5/0 | 6 | - | neighbors/impl/detail/fast_topk_knn.mojo:92 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_FAST_TOPK_OFF` | KEEP | none  | 1+3/0 | 5 | - | neighbors/impl/detail/fast_topk_knn.mojo:57 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_KNN_FUSED_PLAIN_CHAIN` | KEEP | none  | 1+2/0 | 3 | - | neighbors/checks/fused_distance_select_identical.mojo:98 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_FUSED_SELECT_SABOTAGE` | KEEP | none  | 1+1/0 | 2 | - | neighbors/checks/fused_distance_select_identical.mojo:99 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_FUSED_UNROLL_16` | KEEP | none  | 1+7/0 | 8 | - | neighbors/checks/fused_distance_select_identical.mojo:89 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_FUSED_UNROLL_4` | KEEP | none  | 1+7/0 | 8 | - | neighbors/checks/fused_distance_select_identical.mojo:90 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_PHASE_TIMERS` | KEEP | OPEN EXP:2420 PENDING (A = 0.8.55's form: only the multi-output label transpose resolves the b | 5+29/0 | 137 | - | bindings/host_archive_lifetime_door.mojo:21 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_RESIDENT_CACHE_SABOTAGE` | KEEP | none  | 1+1/0 | 6 | - | neighbors/impl/detail/knn_brute_force.mojo:412 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_SELECTOR_BOUND_C2` | KEEP | none  | 1+4/0 | 5 | - | neighbors/checks/knn_selector_bound_compact.mojo:115 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_SELECTOR_BOUND_C4` | KEEP | none  | 1+4/0 | 5 | - | neighbors/checks/knn_selector_bound_compact.mojo:114 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KNN_SELECTOR_BOUND_FALLBACK_SABOTAGE` | KEEP | none  | 1+1/0 | 2 | - | neighbors/checks/knn_selector_bound_compact.mojo:107 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE` | KEEP | none  | 1+2/0 | 9 | - | neighbors/checks/knn_selector_bound_compact.mojo:106 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_SELECTOR_BOUND_TIMING_NOFLAG` | KEEP | none  | 1+2/0 | 3 | - | neighbors/checks/knn_selector_bound_compact.mojo:120 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_SMEM_TILE_SABOTAGE` | KEEP | none  | 1+1/0 | 3 | - | neighbors/checks/smem_distance_tile.mojo:121 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KNN_STATS_DUMP_ASM` | KEEP | none  | 1+0/0 | 1 | - | neighbors/checks/knn_selector_kernel_stats_main.mojo:29 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_LANCZOS_FAST_OFF` | KEEP | none  | 1+5/0 | 6 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:680 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_LANCZOS_RESTART_FAST_OFF` | KEEP | none  | 1+2/0 | 3 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:1051 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_LEGACY_SHAPE_KNN_K10_15` | KEEP | none  | 2+0/0 | 22 | - | neighbors/checks/fused_distance_select_identical.mojo:346 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_NEIGHBORS_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 4 | - | neighbors/impl/multi_gpu.mojo:118 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_PQ_FAST_TRAINSET_OFF` | KEEP | none  | 1+3/0 | 4 | - | x_ann/ivf_pq_device.mojo:45 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_PURITY2_1_OFF` | KEEP | WIN EXP:720 KEPT 196b61747 | 1+0/0 | 2 | - | mixture/checks/estep.mojo:721 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RBC_CANON_DEGREE_BUCKETS` | KEEP | none  | 1+0/0 | 1 | - | neighbors/checks/ball_cover_canonical_order.mojo:337 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RBC_CANON_MERGE` | KEEP | none  | 1+0/0 | 1 | - | neighbors/checks/ball_cover_canonical_order.mojo:342 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_RBC_FAST_EPS_OFF` | KEEP | none  | 1+4/0 | 5 | - | neighbors/impl/ball_cover/fast_rbc_eps.mojo:39 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_REQUIRE_CDNA_TARGET` | KEEP | none  | 1+0/0 | 2 | - | neighbors/checks/fused_logical32_check.mojo:77 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_REQUIRE_RDNA_TARGET` | KEEP | none  | 1+0/0 | 2 | - | neighbors/checks/lane_minimum_check.mojo:52 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_SL_FAST_BORUVKA_OFF` | KEEP | none  | 1+2/0 | 3 | - | hierarchy/impl/cluster/detail/single_linkage.mojo:56 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF` | KEEP | none  | 1+0/0 | 1 | - | spectral/impl/preprocessing/detail/fast_graph.mojo:79 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SPECTRAL_NCV_FAST_OFF` | KEEP | none  | 1+2/0 | 3 | - | spectral/impl/preprocessing/detail/spectral_embedding.mojo:97 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE` | KEEP | none  | 1+2/0 | 9 | - | spectral/host/spectral_predict_host.mojo:93 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_MAXITER` | KEEP | none  | 1+1/0 | 3 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:238 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_NCV` | KEEP | none  | 1+1/0 | 3 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:230 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_SIGN_FLIP` | KEEP | none  | 1+3/0 | 7 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:214 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_SPMV_ROTATE` | KEEP | none  | 1+5/0 | 31 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:210 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_STD_SQRT` | KEEP | none  | 1+2/0 | 8 | - | spectral/impl/sparse/solver/detail/lanczos.mojo:218 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SPECTRAL_SABOTAGE_SWEEP_CAP` | KEEP | none  | 1+1/0 | 3 | - | spectral/checks/symmetric_eig_host.mojo:91 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SYMMETRIZE_FAST_OFF` | KEEP | none  | 1+0/0 | 1 | - | spectral/impl/sparse/linalg/detail/symmetrize.mojo:77 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF` | KEEP | OPEN EXP:2125 owed (boxes down) | 1+4/0 | 5 | - | x_neighbors/items.mojo:1319 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_XN_FUSED_SABOTAGE` | KEEP | none  | 1+0/0 | 12 | - | x_neighbors/iter_device.mojo:1806 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF` | KEEP | none  | 2+0/0 | 2 | - | x_cluster/minibatch.mojo:346 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_AFCL_G04` | REMOVED | LOSE EXP:1912 DELETED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFCL_L08` | REMOVED | LOSE EXP:1907 DELETED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_BGMM_ESTEP1` | REMOVED | LOSE EXP:1903 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_C29_STREAM_TOPK` | REMOVED | LOSE EXP:1569 DROPPED (deleted, guard refuses) | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_C29_TILE` | REMOVED | LOSE EXP:1570 DROPPED (deleted, guard refuses) | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_C29_TILE_128` | REMOVED | LOSE grid classical-misc.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_CAGRA_FAST_IVFG_LOWD` | REMOVED | LOSE EXP:1904 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CAGRA_FAST_SEEDS4` | REMOVED | LOSE EXP:1905 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CC_FAST` | REMOVED | WIN EXP:1656 PROMOTED to every vendor and tier | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_CC_FAST_OFF` | REMOVED | WIN EXP:1656 PROMOTED to every vendor and tier | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_DBSCAN_DIRECT_DISTANCE` | REMOVED | LOSE EXP:2021 DROPPED-noise | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_GRAPH_DIRECT_DISTANCE` | REMOVED | LOSE EXP:1727 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_HDBSCAN2_ALL` | REMOVED | LOSE EXP:1911 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_HDB_CORE_TILE` | REMOVED | LOSE EXP:1912 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_HDB_DEV_BORUVKA` | REMOVED | LOSE EXP:1913 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP` | REMOVED | WIN EXP:1971 PROMOTED (IDENTICAL default, together) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_DBSCAN_EPS_TILE` | REMOVED | WIN EXP:1971 PROMOTED (IDENTICAL default, together) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GMM_COV_SYM` | REMOVED | LOSE EXP:1626 DROP (slower AND quality), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GMM_ONE_DRAIN_OFF` | REMOVED | LOSE EXP:2319 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_HDB_MST_SEED_KNN` | REMOVED | LOSE EXP:2064 LOSER-noise (deleted 826f3d57e; docs/TOMBSTONES.md) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_HDB_ROUND_LEAN` | REMOVED | LOSE EXP:2064 LOSER-noise (deleted 826f3d57e; docs/TOMBSTONES.md) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_KMEANS_CENTROID_FOLD` | REMOVED | WIN EXP:1639 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LP_RESIDENT_OFF` | REMOVED | LOSE EXP:2325 DELETED (rollback of a default) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MBK_RESIDENT` | REMOVED | WIN EXP:2050 KEEP (promoted 2026-10-11; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MINIBATCH_GROUP_OFF` | REMOVED | LOSE EXP:2322 DELETED (superseded) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NC_FIT_LEAN` | REMOVED | WIN EXP:2049 KEEP (promoted 2026-10-11; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS` | REMOVED | WIN EXP:1775 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PQ_SCAN_FUSED` | REMOVED | LOSE EXP:1801 DELETED (loser) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PQ_TRAINSET_STRIDED` | REMOVED | WIN EXP:2065 KEEP (promoted 2026-10-11, quality up; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TSNE_REP_RB4` | REMOVED | LOSE EXP:2061 LOSER (deleted 826f3d57e; docs/TOMBSTONES.md) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_BALANCED_TASKS` | REMOVED | LOSE EXP:2361 DELETED (loser) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_COARSE_FAISS_INIT` | REMOVED | LOSE EXP:1917 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_COARSE_INIT_QOLD` | REMOVED | LOSE EXP:1917 DELETED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_DIRECT_DISTANCE` | REMOVED | WIN EXP:1642 PROMOTED | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_FAST_BALANCED_AUDIT` | REMOVED | LOSE EXP:2297 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_FAST_BALANCED_TASKS` | REMOVED | LOSE EXP:2297 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_FAST_STRIDED_COARSE` | REMOVED | LOSE EXP:2134 DROPPED (deleted 1a76955b5) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_IDN_STRIDED_INIT` | REMOVED | WIN EXP:1800 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_LAYOUT_SCATTER` | REMOVED | LOSE EXP:1802 DELETED (loser) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IVF_REFINE_TEAM` | REMOVED | LOSE EXP:1918 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KAPPROX_DEVICE` | REMOVED | LOSE EXP:1919 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KMEANS_DIRECT_DISTANCE` | REMOVED | WIN EXP:1927 KEPT (owed row withdrawn) | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_KNN_AMD_BLOCK_TOPK_ALL_K` | REMOVED | LOSE EXP:1952 DROPPED-slower, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KNN_AMD_EXACT_CHAIN` | REMOVED | LOSE EXP:1954 DROPPED-noise, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KNN_AMD_LEGACY_SCHEDULE` | REMOVED | LOSE EXP:1951 DROPPED-slower, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KNN_DIRECT_DISTANCE` | REMOVED | LOSE EXP:2020 DROPPED-slower | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_KNN_IDENTICAL_WIDEK_RADIX` | REMOVED | LOSE EXP:1953 DROPPED-slower, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LP_FAST_RESIDENT_OFF` | REMOVED | LOSE EXP:2325 DELETED (rollback of a default) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_PURITY_3_OFF` | REMOVED | LOSE EXP:2228 DELETED (dead code) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_PURITY_6_OFF` | REMOVED | LOSE EXP:2228 DELETED (dead code) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_SPARSE_RP_DEVICE` | REMOVED | LOSE EXP:1920 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG` | REMOVED | LOSE EXP:1934 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |

### decomposition (175)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_FA_TRANSFORM_FUSED` | DELETE-LOSER | LOSE EXP:458 NEUTRAL | 1+1/0 | 2 | - | x_decomp/fa_fast.mojo:156 | expdel-02 |
| `MOJOLEARN_CHOL_FAST_POOLIO_OFF` | KEEP-OPEN | expdel-02: KEEP-OPEN (2026-11-15): the record (EXP:1414, rab15-cholpoolio) is M3 FAST only, but `CHOL_FAST_POOLIO = CHOL_FAST_DEVIO and not _OFF` and CHOL_FAST_DEVIO is also the IDENTICAL default on every vendor (lane/idn-gates), so POOLIO is the unmeasured IDENTICAL default on NVIDIA and AMD and `_OFF` is their only way back to the DEVIO staging body of `cholesky_factor_devio`; owed: NV + AMD A/B of cholesky synthetic (and the GP rows through `cholesky_factor_binding`) POOLIO vs `_OFF`, then delete the staging body or gate POOLIO to FAST + Apple. The source comment "FAST + Apple only" is wrong for IDENTICAL | 1+3/0 | 4 | - | cholesky/estimator.mojo:280 | expdel-02 |
| `MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF` | DELETE-OLD-PATH | WIN EXP:437 KEPT, FAST+Apple default (`_OFF` off) | 1+4/0 | 7 | - | x_decomp/dict_fast.mojo:44 | expdel-02 |
| `MOJOLEARN_DECOMP_FAST_LASSO_GRP_OFF` | DELETE-OLD-PATH | WIN EXP:438 KEPT, FAST+Apple default (`_OFF` off) | 1+3/0 | 4 | - | x_decomp/lasso_grp.mojo:35 | expdel-02 |
| `MOJOLEARN_DECOMP_FAST_ORTH_WS_OFF` | DELETE-OLD-PATH | WIN EXP:1416 KEEP (FAST+Apple default, `_OFF`) | 1+7/0 | 10 | - | x_decomp/s_linalg_fast.mojo:78 | expdel-02 |
| `MOJOLEARN_FA_FAST_QRR_OFF` | DELETE-OLD-PATH | WIN EXP:1257 ACCEPT/DEFAULT: FAST+Apple default | 1+1/0 | 2 | - | x_decomp/device.mojo:1124 | expdel-02 |
| `MOJOLEARN_IDN_LU_RESIDENT_PAIR_OFF` | DELETE-OLD-PATH | WIN grid lu-resident-pair.json promoted:lane/i17-decide-linalg-seq | 1+6/0 | 7 | - | x_decomp/s_linalg_fast.mojo:126 | expdel-02 |
| `MOJOLEARN_IPCA_FAST_DEV_OFF` | DELETE-OLD-PATH | WIN EXP:735 KEPT | 1+1/0 | 2 | - | x_decomp/lanczos_dev.mojo:130 | expdel-02 |
| `MOJOLEARN_PCA_FAST_GRAM_MMA_OFF` | DELETE-OLD-PATH | WIN EXP:431 KEPT | 1+2/0 | 3 | - | decomposition/impl/linalg/detail/pca.mojo:52 | expdel-02 |
| `MOJOLEARN_TSVD_FAST_POOL_OFF` | DELETE-OLD-PATH | WIN EXP:1417 KEEP (FAST+Apple default, `_OFF`) | 1+5/0 | 14 | - | decomposition/estimator.mojo:103 | expdel-02 |
| `MOJOLEARN_XD_FAST_GRP_FUSED_OFF` | DELETE-OLD-PATH | WIN EXP:699 KEPT | 1+3/0 | 4 | - | x_decomp/resident.mojo:964 | expdel-02 |
| `MOJOLEARN_AFCL_L10` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_decomp/device.mojo:521 | owed A/B: the default vs MOJOLEARN_AFCL_L10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L13` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_decomp/mcd_fast.mojo:223 | owed A/B: the default vs MOJOLEARN_AFCL_L13 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_KB` | KEEP-OPEN | OPEN EXP:683 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+13/0 | 14 | - | gemm/afn_apple_fast.mojo:169 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_KB on NV+AMD, rows of its family |
| `MOJOLEARN_C57_CANDIDATE_STATE` | KEEP-OPEN | none  | 1+3/0 | 8 | - | experiments/classical_identical_ideas/stats_controls.mojo:32 | owed A/B: the default vs MOJOLEARN_C57_CANDIDATE_STATE on NV+AMD, rows of its family |
| `MOJOLEARN_CHOL_HOST_SCALAR` | KEEP-OPEN | none  | 1+3/0 | 15 | - | cholesky/host/chol_oracle.mojo:271 | owed A/B: the default vs MOJOLEARN_CHOL_HOST_SCALAR on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C14_GROUP_RHS` | KEEP-OPEN | none  | 1+5/0 | 35 | - | experiments/classical_identical_ideas/linear_controls.mojo:75 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C14_GROUP_RHS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C23_MCD` | KEEP-OPEN | OPEN EXP:1533 not measured alone | 1+3/0 | 6 | c | experiments/classical_identical_ideas/linear_controls.mojo:192 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C23_MCD on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C24_TREE4` | KEEP-OPEN | WIN grid classical-decomp.json promoted:ge123e6f9 | 1+6/0 | 7 | - | experiments/classical_identical_ideas/linear_controls.mojo:224 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_CLASSICAL_C26_PRODUCTS` | KEEP-OPEN | none  | 1+9/0 | 30 | - | experiments/classical_identical_ideas/linear_controls.mojo:237 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C26_PRODUCTS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED` | KEEP-OPEN | none  | 1+5/0 | 19 | - | experiments/classical_identical_ideas/linear_controls.mojo:238 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C27_COMPONENTS` | KEEP-OPEN | none  | 1+3/0 | 12 | - | experiments/classical_identical_ideas/linear_controls.mojo:239 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C27_COMPONENTS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS` | KEEP-OPEN | none  | 1+3/0 | 8 | - | experiments/classical_identical_ideas/linear_controls.mojo:247 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C27_NORM_VECTOR` | KEEP-OPEN | none  | 1+5/0 | 14 | - | experiments/classical_identical_ideas/linear_controls.mojo:248 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C27_NORM_VECTOR on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES` | KEEP-OPEN | none  | 1+8/0 | 31 | - | experiments/classical_identical_ideas/linear_controls.mojo:240 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF` | KEEP-OPEN | WIN EXP:1993 PROMOTED (IDENTICAL default) | 1+7/0 | 8 | - | x_decomp/tsqr_core.mojo:173 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_DECOMP_FAST_GEMM_MMA_OFF` | KEEP-OPEN | LOSE EXP:435 DROPPED-slower | 1+3/0 | 4 | - | x_decomp/device.mojo:1862 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_FA_GRAM_ONCE` | KEEP-OPEN | OPEN EXP:1260 HOLD-quality | 1+1/0 | 2 | - | x_decomp/fa_fast.mojo:110 | owed A/B: the default vs MOJOLEARN_FA_GRAM_ONCE on NV+AMD, rows of its family |
| `MOJOLEARN_FA_GRAM_QOLD` | KEEP-OPEN | WIN EXP:1263 KEPT (QUALITY-FIX) | 1+1/0 | 2 | - | x_decomp/fa_fast.mojo:178 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_FA_ITER_DEVICE_OFF` | KEEP-OPEN | WIN EXP:1280 ACCEPT/DEFAULT: FAST+Apple default | 1+2/0 | 3 | - | x_decomp/fa_fast.mojo:132 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_FA_ITER_DEVICE_OFF |
| `MOJOLEARN_FA_LIVEBUF` | KEEP-OPEN | OPEN EXP:457 OPEN | 2+6/0 | 22 | - | x_decomp/fa_fast.mojo:133 | owed A/B: the default vs MOJOLEARN_FA_LIVEBUF on NV+AMD, rows of its family |
| `MOJOLEARN_FA_SVD_GROUP_OFF` | KEEP-OPEN | WIN EXP:2145 DEFAULT (bit-neutral) | 1+8/0 | 9 | - | x_decomp/fa_fast.mojo:192 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_CD_RESIDENT_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_decomp/resident.mojo:546 | owed A/B: default vs the old path MOJOLEARN_IDN_CD_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CHOL_OPS_FIT` | KEEP-OPEN | OPEN EXP:581 RUN OWED | 1+3/0 | 4 | - | cholesky/chol_ops.mojo:112 | owed A/B: the default vs MOJOLEARN_IDN_CHOL_OPS_FIT on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_COLVAR_FUSED_OFF` | KEEP-OPEN | WIN EXP:1946 PROMOTED (IDENTICAL default) | 1+2/0 | 3 | - | decomposition/estimator.mojo:544 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_EIGH_BLOCK_B32` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_decomp/rr_block.mojo:73 | owed A/B: the default vs MOJOLEARN_IDN_EIGH_BLOCK_B32 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_BLOCK_B8` | KEEP-OPEN | none  | 1+13/0 | 14 | - | x_decomp/rr_block.mojo:75 | owed A/B: the default vs MOJOLEARN_IDN_EIGH_BLOCK_B8 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_BLOCK_MIN128` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_decomp/rr_block.mojo:79 | owed A/B: the default vs MOJOLEARN_IDN_EIGH_BLOCK_MIN128 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_BLOCK_OFF` | KEEP-OPEN | none  | 3+16/0 | 32 | - | tools/identical_wave_eigh_block_refusal.mojo:18 | owed A/B: default vs the old path MOJOLEARN_IDN_EIGH_BLOCK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_BLOCK_ON` | KEEP-OPEN | none  | 3+14/0 | 30 | ic | tools/identical_wave_eigh_block_refusal.mojo:17 | owed A/B: the default vs MOJOLEARN_IDN_EIGH_BLOCK_ON on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_RESIDENT_OFF` | KEEP-OPEN | none  | 1+6/0 | 11 | - | x_decomp/resident.mojo:794 | owed A/B: default vs the old path MOJOLEARN_IDN_EIGH_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EIGH_TRIDIAG_OFF` | KEEP-OPEN | WIN EXP:476 KEEP (default) | 1+2/0 | 3 | - | x_decomp/eigh_tridiag_cells.mojo:29 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_GATES_OFF` | KEEP-OPEN | none  | 7+25/0 | 77 | - | cholesky/estimator.mojo:237 | owed A/B: default vs the old path MOJOLEARN_IDN_GATES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GEMM_TN_CENTERED` | KEEP-OPEN | OPEN grid fg-pca.json hold:postmerge-20261010 (lane/postmerge-act-7; fg2 board-bri | 1+2/0 | 7 | - | decomposition/impl/linalg/detail/pca.mojo:262 | owed A/B: the default vs MOJOLEARN_IDN_GEMM_TN_CENTERED on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ICA_LIM_DEV_OFF` | KEEP-OPEN | none  | 1+4/0 | 6 | - | x_decomp/api.mojo:302 | owed A/B: default vs the old path MOJOLEARN_IDN_ICA_LIM_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LLE_DEV_F0_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_decomp/api.mojo:274 | owed A/B: default vs the old path MOJOLEARN_IDN_LLE_DEV_F0_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LLE_PAD_DEV_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_decomp/api.mojo:312 | owed A/B: default vs the old path MOJOLEARN_IDN_LLE_PAD_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LU_GESV_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | x_decomp/api.mojo:267 | owed A/B: default vs the old path MOJOLEARN_IDN_LU_GESV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LU_RESIDENT_OFF` | KEEP-OPEN | none  | 1+4/0 | 8 | - | x_decomp/resident.mojo:604 | owed A/B: default vs the old path MOJOLEARN_IDN_LU_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MCD_CHOL` | KEEP-OPEN | OPEN EXP:581 RUN OWED | 1+7/0 | 10 | - | x_decomp/mcd.mojo:76 | owed A/B: the default vs MOJOLEARN_IDN_MCD_CHOL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NMF_L2_CELL_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_decomp/api.mojo:297 | owed A/B: default vs the old path MOJOLEARN_IDN_NMF_L2_CELL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PCA_LEAN_SCRATCH_OFF` | KEEP-OPEN | OPEN EXP:1689 owed | 1+1/0 | 2 | - | decomposition/impl/linalg/detail/pca.mojo:284 | owed A/B: default vs the old path MOJOLEARN_IDN_PCA_LEAN_SCRATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_POLAR_MAX_DEV_OFF` | KEEP-OPEN | none  | 1+4/0 | 6 | - | x_decomp/api.mojo:307 | owed A/B: default vs the old path MOJOLEARN_IDN_POLAR_MAX_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_QR_R_DIRECT_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_decomp/device.mojo:272 | owed A/B: default vs the old path MOJOLEARN_IDN_QR_R_DIRECT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_QR_R_INPLACE_OFF` | KEEP-OPEN | none  | 1+3/0 | 11 | - | x_decomp/resident.mojo:676 | owed A/B: default vs the old path MOJOLEARN_IDN_QR_R_INPLACE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_QR_R_RESIDENT_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_decomp/resident.mojo:667 | owed A/B: default vs the old path MOJOLEARN_IDN_QR_R_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RAND_RESIDENT_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_decomp/api.mojo:281 | owed A/B: default vs the old path MOJOLEARN_IDN_RAND_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RES_MOVES_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_decomp/api.mojo:289 | owed A/B: default vs the old path MOJOLEARN_IDN_RES_MOVES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_GRID_OFF` | KEEP-OPEN | none  | 1+5/0 | 23 | - | x_decomp/tsqr_device.mojo:131 | owed A/B: default vs the old path MOJOLEARN_IDN_TSQR_GRID_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_NORM_OFF` | KEEP-OPEN | none  | 1+3/0 | 18 | - | x_decomp/tsqr_device.mojo:143 | owed A/B: default vs the old path MOJOLEARN_IDN_TSQR_NORM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_STRIP_UPDATE` | KEEP-OPEN | none  | 1+2/0 | 9 | - | x_decomp/tsqr_device.mojo:151 | owed A/B: the default vs MOJOLEARN_IDN_TSQR_STRIP_UPDATE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_TREE_PAR_OFF` | KEEP-OPEN | OPEN EXP:479 RUN OWED | 1+1/0 | 2 | - | x_decomp/tsqr_core.mojo:111 | owed A/B: default vs the old path MOJOLEARN_IDN_TSQR_TREE_PAR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSQR_WY_PAIR_OFF` | KEEP-OPEN | OPEN EXP:479 RUN OWED | 1+2/0 | 8 | - | x_decomp/tsqr_core.mojo:108 | owed A/B: default vs the old path MOJOLEARN_IDN_TSQR_WY_PAIR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TSVD_ONE_UPLOAD_OFF` | KEEP-OPEN | LOSE EXP:1998 _OFF arm 1.610x combined (slower) | 1+3/0 | 12 | - | decomposition/estimator.mojo:759 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_XD_SVD_GROUP_OFF` | KEEP-OPEN | WIN EXP:2145 DEFAULT (bit-neutral) | 1+2/0 | 3 | - | x_decomp/device.mojo:228 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_XD_SWEEP_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_decomp/device.mojo:212 | owed A/B: default vs the old path MOJOLEARN_IDN_XD_SWEEP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LDA_FUSED_SS_OFF` | KEEP-OPEN | WIN EXP:416 KEPT a7b8b9513 | 1+2/0 | 5 | - | x_decomp/lda_fast.mojo:50 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LDA_FUSED_SS_OFF |
| `MOJOLEARN_LDA_V_CAP_HALF_PAGE` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_decomp/lda_fast.mojo:114 | owed A/B: the default vs MOJOLEARN_LDA_V_CAP_HALF_PAGE on NV+AMD, rows of its family |
| `MOJOLEARN_LDA_V_CAP_RULE_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_decomp/lda_fast.mojo:112 | owed A/B: default vs the old path MOJOLEARN_LDA_V_CAP_RULE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_LDA_FUSED_V` | KEEP-OPEN | WIN EXP:1343 ACCEPT (merged via lane/apple-fast-verdicts-5) | 2+6/0 | 10 | - | x_decomp/lda_fast.mojo:112 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LEGACY_NARROW_LDA_FUSED_V |
| `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE_OFF` | KEEP-OPEN | WIN EXP:1007 DEFAULT merged; promotion0769e1d92 M2default/OFF rc0. Existing parallel LUstep g | 1+4/0 | 25 | - | x_decomp/lu_fast.mojo:71 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_LU_FAST_STEP1_OFF` | KEEP-OPEN | WIN EXP:460 DEFAULT (FAST+Apple), rollback `MOJOLEARN_LU_FAST_MMA_OFF`; w2-lumma-quality PAS | 1+8/0 | 9 | - | x_decomp/lu_fast.mojo:57 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_LU_QOLD` | KEEP-OPEN | WIN EXP:1283 QUALITY-FIX KEPT: FAST default | 1+8/0 | 9 | - | x_decomp/qfix.mojo:67 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_MCD_BATCH_MMA_OFF` | KEEP-OPEN | LOSE EXP:2281 DELETED (HOLD-quality) | 1+20/0 | 57 | i | x_decomp/mcd_fast.mojo:111 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_MCD_BMMA_OFF` | KEEP-OPEN | WIN EXP:804 DEFAULT (FAST+Apple, needs MCD_BMMA), rollback `MOJOLEARN_MCD_WIDE_OFF` | 1+7/0 | 22 | - | x_decomp/mcd_fast.mojo:144 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_MCD_DEFLATE_OFF` | KEEP-OPEN | WIN EXP:1273 ACCEPT/DEFAULT: FAST+Apple default | 1+2/0 | 3 | - | x_decomp/mcd_fast.mojo:194 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MCD_DEFLATE_OFF |
| `MOJOLEARN_MCD_SKIP_PINVH_OFF` | KEEP-OPEN | OPEN EXP:929 HOLD (opt-in) | 1+2/0 | 3 | - | x_decomp/mcd_fast.mojo:181 | owed A/B: default vs the old path MOJOLEARN_MCD_SKIP_PINVH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_MCD_WIDE_OFF` | KEEP-OPEN | WIN EXP:804 DEFAULT (FAST+Apple, needs MCD_BMMA), rollback `MOJOLEARN_MCD_WIDE_OFF` | 1+9/0 | 10 | - | x_decomp/mcd_fast.mojo:159 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PCA_RR_EIGH_OFF` | KEEP-OPEN | none  | 1+5/0 | 36 | - | decomposition/pca_rr_switch.mojo:21 | owed A/B: default vs the old path MOJOLEARN_PCA_RR_EIGH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RSVD_FAST_DEVSCAN_OFF` | KEEP-OPEN | WIN EXP:2377 KEPT default again (restored; Apple FAST paused, owed row r1-rsvd-devscan) | 1+4/0 | 5 | c | x_decomp/s_linalg_fast.mojo:58 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TSVD_QOLD` | KEEP-OPEN | WIN EXP:1287 QUALITY-FIX KEPT: FAST default | 1+6/0 | 7 | - | x_decomp/qfix.mojo:66 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_XDECOMP_HOST_SERIAL` | KEEP-OPEN | none  | 1+0/0 | 4 | - | x_decomp/host.mojo:188 | owed A/B: the default vs MOJOLEARN_XDECOMP_HOST_SERIAL on NV+AMD, rows of its family |
| `MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN_OFF` | KEEP-OPEN | WIN EXP:467 KEEP, FAST+Apple default (`_OFF` off) | 1+3/0 | 4 | - | x_decomp/resident.mojo:946 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` | KEEP | none  | 2+0/0 | 7 | - | cholesky/checks/trsm.mojo:1089 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHOL_APPLE_LEFT_OFF` | KEEP | none  | 1+9/0 | 11 | - | cholesky/checks/potrf.mojo:1092 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_APPLE_LEFT_SABOTAGE` | KEEP | none  | 1+0/0 | 2 | - | cholesky/checks/potrf.mojo:1383 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHOL_APPLE_MMA_SYRK` | KEEP | none  | 1+2/0 | 12 | - | cholesky/checks/potrf.mojo:965 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_APPLE_MMA_SYRK_SABOTAGE` | KEEP | none  | 1+0/0 | 2 | - | cholesky/checks/potrf.mojo:1084 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHOL_BACK_FTZ_ON` | KEEP | none  | 1+3/0 | 8 | - | cholesky/checks/trsm.mojo:231 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_BACK_RING_OFF` | KEEP | none  | 1+2/0 | 3 | - | cholesky/checks/trsm.mojo:569 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_BACK_U16` | KEEP | none  | 1+4/0 | 5 | - | cholesky/checks/trsm.mojo:234 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_BACK_U64` | KEEP | none  | 1+4/0 | 5 | - | cholesky/checks/trsm.mojo:234 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_BACK_UNROLL_OFF` | KEEP | none  | 1+4/0 | 5 | - | cholesky/checks/trsm.mojo:233 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_DEFER_INFO_OFF` | KEEP | none  | 1+6/0 | 7 | - | cholesky/checks/potrf.mojo:1121 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_DEVIO_OFF` | KEEP | OPEN EXP:1395 READY-AB | 1+2/0 | 3 | - | cholesky/estimator.mojo:239 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CHOL_FAST_NB128` | KEEP | none  | 1+3/0 | 4 | - | cholesky/checks/potrf.mojo:460 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_NB64` | KEEP | none  | 1+3/0 | 4 | - | cholesky/checks/potrf.mojo:459 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_NOSYNC_OFF` | KEEP | LOSE EXP:432 DROPPED-slower+quality 2026-10-04 (see verdicts batch 3) | 1+2/0 | 3 | - | cholesky/checks/potrf.mojo:431 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_PINNED` | KEEP | none  | 1+16/0 | 19 | - | cholesky/checks/potrf.mojo:412 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_SERIAL_SOLVE` | KEEP | none  | 1+3/0 | 4 | - | cholesky/checks/trsm.mojo:96 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_SHARED_SUB` | KEEP | OPEN EXP:2378 PENDING (restored; Apple FAST paused, owed row r1-chol-shared-sub) | 1+2/0 | 12 | ic | cholesky/checks/fast_trsm.mojo:47 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_FAST_TALL_OFF` | KEEP | WIN EXP:429 DEFAULT (FAST+Apple), rollback MOJOLEARN_CHOL_FAST_TALL_OFF | 1+9/0 | 10 | - | cholesky/checks/chol_fast_tall.mojo:57 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_LEFT_BLOCK_ADMIT` | KEEP | none  | 1+1/0 | 34 | - | cholesky/checks/potrf.mojo:1141 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_LEFT_KB32` | KEEP | none  | 1+2/0 | 3 | - | cholesky/checks/potrf.mojo:1137 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_LEFT_LOOKAHEAD_OFF` | KEEP | none  | 1+5/0 | 10 | - | cholesky/checks/potrf.mojo:1127 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_MR4` | KEEP | none  | 1+1/0 | 8 | - | cholesky/checks/trsm.mojo:480 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_MULTI_RHS_OFF` | KEEP | none  | 1+1/0 | 2 | - | cholesky/checks/trsm.mojo:346 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_STRIP_NO_ADMIT` | KEEP | none  | 1+4/0 | 5 | - | cholesky/checks/potrf_strip.mojo:109 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_STRIP_OFF` | KEEP | none  | 1+1/0 | 2 | - | cholesky/checks/potrf_strip.mojo:101 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_STRIP_SABOTAGE` | KEEP | none  | 1+0/0 | 2 | - | cholesky/checks/potrf_strip.mojo:347 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHOL_STRIP_W` | KEEP | none  | 1+2/0 | 3 | - | cholesky/checks/potrf_strip.mojo:76 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_CHOL_SWEEP_SOLVES_OFF` | KEEP | none  | 1+4/0 | 5 | - | cholesky/checks/trsm.mojo:215 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_DECOMP_FAST_OMP_BLOCK` | KEEP | OPEN EXP:449 OPEN (measured, verdict owed) | 1+4/0 | 11 | - | x_decomp/omp_block.mojo:50 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DECOMP_FAST_TSQR_GRID` | KEEP | none  | 1+5/0 | 23 | - | x_decomp/tsqr_device.mojo:131 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_DECOMP_FAST_TSQR_NORM` | KEEP | none  | 1+3/0 | 18 | - | x_decomp/tsqr_device.mojo:143 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF` | KEEP | OPEN EXP:931 M3 43765.6 -> 700.3 ms; original no-regression criterion PASS all 8 fixtures, ze | 1+2/0 | 3 | - | x_decomp/eigh_tridiag.mojo:112 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_IDN_CHOL_ONE_WAIT_OFF` | KEEP | none  | 1+4/0 | 14 | - | cholesky/checks/potrf.mojo:1459 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_JACOBI_FAST_NARROW` | KEEP | none  | 1+4/0 | 5 | - | decomposition/checks/jacobi_eigh_device.mojo:91 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_LU_FAST_MMA_NB` | KEEP | none  | 1+4/0 | 5 | - | x_decomp/lu_fast_mma.mojo:74 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_LU_FAST_MMA_OFF` | KEEP | OPEN EXP:1401 READY-AB | 1+4/0 | 5 | - | x_decomp/lu_fast_mma.mojo:72 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_LU_FAST_RESIDENT_OFF` | KEEP | OPEN EXP:2031 OPEN (A/B owed) | 1+5/0 | 6 | - | x_decomp/s_linalg_fast.mojo:101 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_MCD_PROFILE` | KEEP | none  | 1+11/0 | 27 | - | x_decomp/mcd_fast.mojo:199 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_PCA_FAST_POOL_OFF` | KEEP | OPEN EXP:1392 READY-AB | 1+2/0 | 3 | - | decomposition/estimator.mojo:88 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_PCA_RR_FLAG_TEST_OFF` | KEEP | none  | 1+1/0 | 2 | - | decomposition/pca_rr_switch.mojo:45 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_XD_FAST_CLS2_GRP_LAZY` | KEEP | none  | 1+2/0 | 3 | - | x_decomp/resident.mojo:949 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN` | KEEP | OPEN EXP:466 PARKED | 1+3/0 | 4 | - | x_decomp/resident.mojo:938 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_XD_FAST_SRP_STRAT_OFF` | KEEP | OPEN EXP:861 sparse-rp / istella (quality HOLD), taxi | 1+5/0 | 6 | - | x_decomp/resident.mojo:988 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_XD_FAST_SVD_ONE_BLOCK` | KEEP | OPEN EXP:2366 PENDING | 1+1/0 | 4 | - | x_decomp/device.mojo:246 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_AFCL_L07` | REMOVED | LOSE EXP:2299 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFCL_L09` | REMOVED | LOSE EXP:1868 DELETED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_PCA_COV` | REMOVED | LOSE EXP:1995 DROPPED (noise/slower), code deleted | 0+0/2 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO` | REMOVED | WIN EXP:1993 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_DECOMP_FAST_GEMM_TILED` | REMOVED | LOSE EXP:1907 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_EIGH_FAST_PANEL_DF` | REMOVED | LOSE EXP:2286 DELETED (DROPPED-slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_EIGH_FAST_PANEL_DF_OFF` | REMOVED | LOSE EXP:2286 DELETED (DROPPED-slower) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_FA_ALL` | REMOVED | LOSE EXP:1910 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_FA_EIG_SMALL` | REMOVED | LOSE EXP:1909 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_FA_FAST_GRAM_SPREAD` | REMOVED | LOSE EXP:2295 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_FA_LL_DEVICE` | REMOVED | LOSE EXP:1910 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_EIGH_SMALL_N128` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_EIGH_SMALL_N64` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_EIGH_SMALL_OFF` | REMOVED | LOSE EXP:2316 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_JACOBI_ROUND_ROBIN` | REMOVED | WIN EXP:1831 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MCD_CHOL_OFF` | REMOVED | OPEN EXP:581 RUN OWED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_OLS_R_TRISOLVE` | REMOVED | LOSE grid gap-linalg.json removed | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PCA_DEVICE_TRUNCATE` | REMOVED | LOSE EXP:1949 DROPPED-noise, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK` | REMOVED | LOSE EXP:1834 DROPPED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS` | REMOVED | LOSE EXP:1692 DROPPED 2026-10-09 (broken on NVIDIA;see lane/postmerge-act-5 section) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_RR_ONE_BLOCK_D` | REMOVED | LOSE EXP:2098 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_RR_ONE_BLOCK_D_STEPS` | REMOVED | OPEN EXP:2047 OPEN | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_RSVD_DIRECT_IN_OFF` | REMOVED | WIN EXP:2377 KEPT default again (restored; Apple FAST paused, owed row r1-rsvd-devscan) | 0+0/1 | 0 | ir | - | only the refusal guard remains |
| `MOJOLEARN_IDN_RSVD_ORTH_LEAN` | REMOVED | LOSE EXP:2102 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SVD_TALL_LEAN` | REMOVED | LOSE EXP:2099 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT` | REMOVED | WIN EXP:1822 PROMOTED (IDENTICAL default, with REP_TILE) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TSNE_REP_TILE` | REMOVED | WIN EXP:1823 PROMOTED (IDENTICAL default, with KNN_TILE_SELECT) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_XD_KIT_GEMM` | REMOVED | LOSE EXP:2101 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LEGACY_NARROW_MCD_G1_GRAM` | REMOVED | LOSE EXP:2284 DELETED (NEUTRAL) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_LLE_FAST_DEV_LU` | REMOVED | LOSE EXP:2283 DELETED (HOLD-quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LLE_FAST_NULL_CANON` | REMOVED | LOSE EXP:1928 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LU_FAST_SHARED_SUB` | REMOVED | LOSE EXP:2314 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_BATCH_COMPAT` | REMOVED | LOSE EXP:2281 DELETED (HOLD-quality) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_DEVICE_CSTEPS` | REMOVED | LOSE EXP:1929 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_FAST_ACTIVE_COMPACT` | REMOVED | LOSE EXP:2302 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_FAST_BOUND_BATCH` | REMOVED | LOSE EXP:2302 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_FAST_COV_REUSE` | REMOVED | LOSE EXP:2301 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_FAST_G1_GRAM` | REMOVED | LOSE EXP:2284 DELETED (NEUTRAL) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_FAST_G1_GRAM_AUDIT` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_MCD_ORDERED_COV` | REMOVED | LOSE EXP:2285 DELETED (NEUTRAL) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PCA_FAST_COMPENSATED_COV` | REMOVED | LOSE EXP:2299 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_QR_FAST_DEV` | REMOVED | LOSE EXP:1868 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF` | REMOVED | WIN EXP:2377 KEPT default again (restored; Apple FAST paused, owed row r1-rsvd-devscan) | 0+0/1 | 0 | ir | - | only the refusal guard remains |
| `MOJOLEARN_SVD_FAST_CHOLQR` | REMOVED | LOSE EXP:1863 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SVD_QFIX` | REMOVED | LOSE EXP:1931 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SVD_QOLD` | REMOVED | LOSE EXP:1932 DELETED | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_TSVD_FAST_CHOLQR3` | REMOVED | LOSE EXP:1933 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |

### kernel (52)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_KDE_DIMTILE` | KEEP-OPEN | LOSE EXP:343 DROP | 1+3/0 | 4 | c | kde/impl/neighbors/kernel_density.mojo:2748 | expdel-06: KEEP-OPEN (2026-11-15): the define's own row is a WIN (EXP:339 `KDE_DIMTILE` KEEP, istella 137.2 -> 69.2 ms; the tile pass is now the FAST+Apple default at every d); EXP:343 and its siblings are the DROPPED combos (already deleted 2026-10-09). The define now only overrides MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE: orchestrator decides whether to delete it as redundant with LEGACY_NARROW |
| `MOJOLEARN_KDE_FAST_SLICES_OFF` | DELETE-OLD-PATH | WIN EXP:299 KEPT 857fd5804 | 1+0/0 | 1 | - | kde/impl/neighbors/kernel_density.mojo:4429 | expdel-06 |
| `MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF` | DELETE-OLD-PATH | WIN EXP:694 KEPT | 1+0/0 | 1 | - | kernel_methods/estimator.mojo:1330 | expdel-06 |
| `MOJOLEARN_KM_FAST_PTR_IN_OFF` | DELETE-OLD-PATH | WIN EXP:693 KEPT | 1+4/0 | 5 | - | kernel_methods/estimator.mojo:288 | expdel-06 |
| `MOJOLEARN_KM_FAST_RBF_PIPE_OFF` | DELETE-OLD-PATH | WIN EXP:1378 KEEP (FAST+Apple default, `_OFF`) | 1+2/0 | 3 | - | kernel_methods/rbf_resident.mojo:118 | expdel-06 |
| `MOJOLEARN_KM_FAST_RBF_STAGED_OFF` | DELETE-OLD-PATH | WIN EXP:871 DEFAULT (FAST+Apple), rollback MOJOLEARN_KM_FAST_RBF_STAGED_OFF: M3 one run per  | 1+5/0 | 9 | - | kernel_methods/rbf_resident.mojo:94 | expdel-06 |
| `MOJOLEARN_AFCL_G13` | KEEP-OPEN | none  | 1+2/0 | 3 | - | kde/impl/neighbors/kernel_density.mojo:2778 | owed A/B: the default vs MOJOLEARN_AFCL_G13 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L11` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gaussian_process/afcl_prediction.mojo:14 | owed A/B: the default vs MOJOLEARN_AFCL_L11 on NV+AMD, rows of its family |
| `MOJOLEARN_C52_PAIR_ROWS` | KEEP-OPEN | none  | 2+15/3 | 34 | c | experiments/classical_identical_ideas/stats_controls.mojo:16 | owed A/B: the default vs MOJOLEARN_C52_PAIR_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_FAMILY_CTX_PER_CALL` | KEEP-OPEN | none  | 5+4/0 | 13 | - | gaussian_process/classifier.mojo:54 | owed A/B: the default vs MOJOLEARN_FAMILY_CTX_PER_CALL on NV+AMD, rows of its family |
| `MOJOLEARN_GP_BULK_DOWNLOAD` | KEEP-OPEN | none  | 1+0/0 | 11 | - | gaussian_process/estimator.mojo:575 | owed A/B: the default vs MOJOLEARN_GP_BULK_DOWNLOAD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF` | KEEP-OPEN | none  | 1+2/0 | 19 | - | gaussian_process/estimator.mojo:539 | owed A/B: default vs the old path MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gaussian_process/gp_grad_items.mojo:33 | owed A/B: default vs the old path MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GP_GRAD_ROWS4` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gaussian_process/gp_grad_items.mojo:35 | owed A/B: the default vs MOJOLEARN_IDN_GP_GRAD_ROWS4 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | gaussian_process/estimator.mojo:548 | owed A/B: default vs the old path MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GP_VAR_SEG_OFF` | KEEP-OPEN | none  | 1+4/0 | 12 | - | gaussian_process/gp_var_seg.mojo:40 | owed A/B: default vs the old path MOJOLEARN_IDN_GP_VAR_SEG_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | kde/impl/neighbors/kernel_density.mojo:3466 | owed A/B: default vs the old path MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KDE_PARTIAL_POOL` | KEEP-OPEN | none  | 1+8/0 | 44 | - | kde/impl/chunk_workspace.mojo:28 | owed A/B: the default vs MOJOLEARN_IDN_KDE_PARTIAL_POOL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KDE_QTILE_FORCE` | KEEP-OPEN | OPEN EXP:2398 PENDING | 1+2/0 | 4 | - | kde/impl/neighbors/kernel_density.mojo:3668 | owed A/B: the default vs MOJOLEARN_IDN_KDE_QTILE_FORCE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KDE_QTILE_OFF` | KEEP-OPEN | WIN EXP:2059 KEEP by average (promoted 2026-10-11, istella regression flagged; _OFF restores) | 1+4/0 | 5 | - | kde/impl/neighbors/kernel_density.mojo:3639 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF` | KEEP-OPEN | none  | 1+2/0 | 16 | - | kernel_methods/estimator.mojo:296 | owed A/B: default vs the old path MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KM_DOT_CELL_D16` | KEEP-OPEN | none  | 2+4/0 | 6 | - | kernel_methods/checks/kernel_matrix.mojo:533 | owed A/B: the default vs MOJOLEARN_IDN_KM_DOT_CELL_D16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KM_DOT_CELL_OFF` | KEEP-OPEN | none  | 2+6/0 | 63 | - | kernel_methods/checks/kernel_matrix.mojo:531 | owed A/B: default vs the old path MOJOLEARN_IDN_KM_DOT_CELL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KM_PTR_IN_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | kernel_methods/estimator.mojo:283 | owed A/B: default vs the old path MOJOLEARN_IDN_KM_PTR_IN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KM_RBF_CELL` | KEEP-OPEN | none  | 2+6/0 | 48 | - | kernel_methods/checks/kernel_matrix.mojo:475 | owed A/B: the default vs MOJOLEARN_IDN_KM_RBF_CELL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KM_RBF_CELL_D16` | KEEP-OPEN | none  | 2+4/0 | 6 | - | kernel_methods/checks/kernel_matrix.mojo:478 | owed A/B: the default vs MOJOLEARN_IDN_KM_RBF_CELL_D16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KRR_DEV_SCALE_OFF` | KEEP-OPEN | none  | 1+5/0 | 13 | - | kernel_methods/estimator.mojo:460 | owed A/B: default vs the old path MOJOLEARN_IDN_KRR_DEV_SCALE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KRR_PTR_IN_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | kernel_methods/estimator.mojo:480 | owed A/B: default vs the old path MOJOLEARN_IDN_KRR_PTR_IN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_DEV_BASIS_OFF` | KEEP-OPEN | none  | 1+3/0 | 6 | - | kernel_methods/estimator.mojo:2080 | owed A/B: default vs the old path MOJOLEARN_IDN_NYS_DEV_BASIS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_DEV_ORDER_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | kernel_methods/estimator.mojo:1920 | owed A/B: default vs the old path MOJOLEARN_IDN_NYS_DEV_ORDER_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | kernel_methods/estimator.mojo:2068 | owed A/B: default vs the old path MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP` | KEEP-OPEN | none  | 1+3/0 | 7 | - | kernel_methods/estimator.mojo:1612 | owed A/B: the default vs MOJOLEARN_IDN_NYS_RR_DEV_STOP on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4` | KEEP-OPEN | none  | 1+2/0 | 3 | - | kernel_methods/estimator.mojo:1614 | owed A/B: the default vs MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NYS_RR_EIGH_OFF` | KEEP-OPEN | none  | 2+6/0 | 24 | - | kernel_methods/estimator.mojo:1457 | owed A/B: default vs the old path MOJOLEARN_IDN_NYS_RR_EIGH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RBF_FUSED_OFF` | KEEP-OPEN | none  | 2+6/0 | 9 | - | kernel_methods/estimator.mojo:195 | owed A/B: default vs the old path MOJOLEARN_IDN_RBF_FUSED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KDE_BLOCK_SUM_OFF` | KEEP-OPEN | none  | 1+1/0 | 22 | - | kde/impl/neighbors/kernel_density.mojo:2002 | owed A/B: default vs the old path MOJOLEARN_KDE_BLOCK_SUM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_KDE_DIMTILE_OFF` | KEEP-OPEN | LOSE EXP:343 DROP | 1+3/0 | 27 | - | kde/impl/neighbors/kernel_density.mojo:2757 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_KDE_DIRECT_DISTANCE` | KEEP-OPEN | none  | 1+9/0 | 27 | - | experiments/classical_identical_ideas/graph_controls.mojo:25 | owed A/B: the default vs MOJOLEARN_KDE_DIRECT_DISTANCE on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE` | KEEP-OPEN | none  | 1+1/0 | 2 | - | kde/impl/neighbors/kernel_density.mojo:2763 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE on NV+AMD, rows of its family |
| `MOJOLEARN_RBF_FUSED` | KEEP-OPEN | none  | 1+2/0 | 11 | - | kernel_methods/estimator.mojo:176 | owed A/B: the default vs MOJOLEARN_RBF_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L12` | KEEP | none  | 1+0/0 | 1 | - | kernel_methods/checks/random_features.mojo:736 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GP_GRAD_SABOTAGE` | KEEP | none  | 1+1/0 | 3 | - | gaussian_process/host/gp_theta.mojo:62 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GP_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 7 | - | gaussian_process/checks/kernels.mojo:1654 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IDN_GP_GRAD_PRESCALE_OFF` | KEEP | none  | 1+5/0 | 25 | - | gaussian_process/checks/kernel_gradient.mojo:86 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GP_PRESCALE_OFF` | KEEP | none  | 1+2/0 | 4 | - | gaussian_process/checks/kernels.mojo:761 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_KM_LAP_CELL_OFF` | KEEP | none  | 1+4/0 | 14 | - | kernel_methods/checks/kernel_matrix.mojo:597 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF` | KEEP | none  | 1+3/0 | 7 | - | kde/resident_fit.mojo:89 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CLASSICAL_C20_ROW_CACHE` | REMOVED | LOSE EXP:1568 DROPPED (deleted, guard refuses) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE` | REMOVED | WIN EXP:1984 PROMOTED (restored; IDENTICAL default) | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CHOL_NB128` | REMOVED | LOSE EXP:471 LOSER, DELETED 2026-10-11 (dd9a3d31b) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF` | REMOVED | LOSE EXP:2317 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SVC_ONE_BLOCK` | REMOVED | WIN EXP:2062 KEEP (promoted 2026-10-11; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |

### linear (207)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_SGD_PERC_AVG` | DELETE-LOSER | LOSE EXP:1289 REVERTED: last iterate is the default | 1+4/0 | 6 | c | x_linear/sgd.mojo:443 | expdel-03 |
| `MOJOLEARN_ARD_FAST_CLS1_BATCH_OFF` | DELETE-OLD-PATH | WIN EXP:272 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:53 | expdel-03 |
| `MOJOLEARN_ARD_FAST_CLS1_PARTS_OFF` | DELETE-OLD-PATH | WIN EXP:273 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:52 | expdel-03 |
| `MOJOLEARN_BAYES_FAST_CLS1_BATCH_OFF` | DELETE-OLD-PATH | WIN EXP:276 KEPT 9decae29f | 1+2/0 | 3 | - | x_linear/cls1_fast.mojo:50 | expdel-03 |
| `MOJOLEARN_BAYES_FAST_CLS1_PARTS_OFF` | DELETE-OLD-PATH | WIN EXP:277 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:49 | expdel-03 |
| `MOJOLEARN_CD_FAST_GRID_GRAM_OFF` | DELETE-OLD-PATH | WIN EXP:239 KEPT 0ae9c28cd | 1+3/0 | 4 | - | solver/impl/cd.mojo:337 | expdel-03 |
| `MOJOLEARN_CD_FAST_ROWMAJOR_OFF` | DELETE-OLD-PATH | WIN EXP:239 KEPT 0ae9c28cd | 1+4/0 | 5 | - | solver/impl/cd.mojo:338 | expdel-03 |
| `MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF` | DELETE-OLD-PATH | WIN EXP:240 KEPT 94cb5ba59 | 1+8/0 | 12 | - | glm/linear_route_controls.mojo:19 | expdel-03 |
| `MOJOLEARN_HUBER_FAST_BLOCK512_OFF` | DELETE-OLD-PATH | WIN EXP:1423 KEEP (FAST+Apple default, `_OFF`) | 1+3/0 | 4 | - | x_linear/huber_fast.mojo:84 | expdel-03 |
| `MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF` | DELETE-OLD-PATH | WIN EXP:1255 ACCEPT/DEFAULT: FAST+Apple default | 1+5/0 | 6 | - | x_prep/fastprep2.mojo:82 | expdel-03 |
| `MOJOLEARN_PREP_FAST_MINMAX_OFF` | DELETE-OLD-PATH | WIN EXP:365 KEPT f419ea9f1 | 1+6/0 | 9 | - | preprocessing/minmax.mojo:34 | expdel-03 |
| `MOJOLEARN_QN_FAST_BLOCKS_OFF` | DELETE-OLD-PATH | WIN EXP:245 KEPT 0ae9c28cd | 1+4/0 | 5 | - | glm/impl/qn/glm_base.mojo:318 | expdel-03 |
| `MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF` | DELETE-OLD-PATH | WIN EXP:291 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:54 | expdel-03 |
| `MOJOLEARN_SGDOC_FAST_TAIL_OFF` | DELETE-OLD-PATH | WIN EXP:745 DEFAULT (FAST+Apple), rollback MOJOLEARN_SGDOC_FAST_TAIL_OFF; parallel algorithm | 1+2/0 | 3 | - | x_linear/device.mojo:2733 | expdel-03 |
| `MOJOLEARN_SGD_FAST_PS_SIMD_OFF` | KEEP-OPEN | WIN EXP:235 KEPT (small d only), FAST+Apple default for d <= 32 (`_OFF` off) | 1+2/0 | 3 | - | x_linear/device.mojo:2752 | expdel-03: KEEP-OPEN (2026-11-15): EXP:235 KEPT SIMD for small d only (sgd-ocsvm taxi d 11: 58,334 -> 42,252 ms) and records the block form WINNING at istella d 220 (75,506 -> 106,443 ms, +41% with SIMD); the current gate (d <= SPS_MAX_D = 256, x_linear/device.mojo, replacement of the benchmark-tuned d <= 32 cap) is UNMEASURED and covers istella, so `_OFF` is the only route to the measured winner there; owed A/B of the register-bound gate on sgd-ocsvm istella (with SGDOC_FAST_TAIL on) before the old path goes |
| `MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF` | DELETE-OLD-PATH | WIN EXP:250 KEPT c5e1bbeb6 | 1+0/0 | 1 | - | x_linear/device.mojo:707 | expdel-03 |
| `MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF` | DELETE-OLD-PATH | WIN EXP:251 KEPT c5e1bbeb6 | 1+0/0 | 1 | - | x_linear/fast_gram.mojo:49 | expdel-03 |
| `MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF` | DELETE-OLD-PATH | WIN EXP:408 KEPT (FAST+Apple default, `_OFF`) | 1+0/0 | 1 | - | x_prep/device.mojo:61 | expdel-03 |
| `MOJOLEARN_X_PREP_FAST_TE_ENC_OFF` | DELETE-OLD-PATH | WIN EXP:1254 ACCEPT/DEFAULT: FAST+Apple default | 1+3/0 | 4 | - | x_prep/fastprep2.mojo:115 | expdel-03 |
| `MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF` | DELETE-OLD-PATH | WIN EXP:1253 ACCEPT/DEFAULT: FAST+Apple default | 1+3/0 | 4 | - | x_prep/fastprep2.mojo:100 | expdel-03 |
| `MOJOLEARN_AFCL_L01` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/fast_gram.mojo:56 | owed A/B: the default vs MOJOLEARN_AFCL_L01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L02` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/fast_gram.mojo:58 | owed A/B: the default vs MOJOLEARN_AFCL_L02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L03` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/enetcv_fast.mojo:53 | owed A/B: the default vs MOJOLEARN_AFCL_L03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L04` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/huber_fast.mojo:92 | owed A/B: the default vs MOJOLEARN_AFCL_L04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L05` | KEEP-OPEN | none  | 1+0/0 | 1 | - | glm/impl/qn/fast_xtdz.mojo:35 | owed A/B: the default vs MOJOLEARN_AFCL_L05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L06` | KEEP-OPEN | none  | 1+0/0 | 1 | - | svm/impl/fast_update_f.mojo:36 | owed A/B: the default vs MOJOLEARN_AFCL_L06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_L14` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/device.mojo:751 | owed A/B: the default vs MOJOLEARN_AFCL_L14 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P01` | KEEP-OPEN | none  | 2+8/0 | 34 | - | x_prep/fastpt.mojo:54 | owed A/B: the default vs MOJOLEARN_AFCL_P01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P02` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_prep/fastmaxabs.mojo:37 | owed A/B: the default vs MOJOLEARN_AFCL_P02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P03` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/fastnb.mojo:32 | owed A/B: the default vs MOJOLEARN_AFCL_P03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P04` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/fastnb_csr.mojo:81 | owed A/B: the default vs MOJOLEARN_AFCL_P04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P05` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/fastprep2.mojo:122 | owed A/B: the default vs MOJOLEARN_AFCL_P05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P06` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/fastprep2.mojo:123 | owed A/B: the default vs MOJOLEARN_AFCL_P06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P07` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/select_fast.mojo:84 | owed A/B: the default vs MOJOLEARN_AFCL_P07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P13` | KEEP-OPEN | none  | 1+5/0 | 18 | - | x_prep/fastred.mojo:35 | owed A/B: the default vs MOJOLEARN_AFCL_P13 on NV+AMD, rows of its family |
| `MOJOLEARN_ARD_SIGMA_QOLD` | KEEP-OPEN | OPEN EXP:2124 owed (boxes down) | 1+6/0 | 7 | - | x_linear/bayes.mojo:519 | owed A/B: the default vs MOJOLEARN_ARD_SIGMA_QOLD on NV+AMD, rows of its family |
| `MOJOLEARN_BAYES_GRID_GUARD_OFF` | KEEP-OPEN | WIN EXP:249 KEPT 269ffa57a | 1+3/0 | 4 | - | x_linear/device.mojo:142 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CALIB_GNB_FOLDS_OFF` | KEEP-OPEN | WIN EXP:237 KEPT 24ed76679 | 1+2/0 | 3 | - | x_prep/calib.mojo:57 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_CALIB_GNB_FOLDS_OFF |
| `MOJOLEARN_CD_FUSED_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | solver/impl/cd.mojo:752 | owed A/B: default vs the old path MOJOLEARN_CD_FUSED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CD_GRAM_BLOCKS` | KEEP-OPEN | none  | 1+2/0 | 29 | - | solver/impl/cd.mojo:253 | owed A/B: the default vs MOJOLEARN_CD_GRAM_BLOCKS on NV+AMD, rows of its family |
| `MOJOLEARN_CD_IDN_GRAM_COLS_128` | KEEP-OPEN | none  | 2+4/0 | 7 | - | solver/impl/cd_gram_rule.mojo:31 | owed A/B: the default vs MOJOLEARN_CD_IDN_GRAM_COLS_128 on NV+AMD, rows of its family |
| `MOJOLEARN_CD_IDN_GRAM_COLS_256` | KEEP-OPEN | none  | 2+4/0 | 7 | - | solver/impl/cd_gram_rule.mojo:30 | owed A/B: the default vs MOJOLEARN_CD_IDN_GRAM_COLS_256 on NV+AMD, rows of its family |
| `MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | solver/impl/cd_gram_rule.mojo:55 | owed A/B: default vs the old path MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CD_IDN_GRAM_OFF` | KEEP-OPEN | none  | 1+5/0 | 9 | - | solver/impl/cd_gram_rule.mojo:21 | owed A/B: default vs the old path MOJOLEARN_CD_IDN_GRAM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CD_IDN_GRAM_WIDE` | KEEP-OPEN | none  | 2+4/0 | 7 | - | solver/impl/cd_gram_rule.mojo:30 | owed A/B: the default vs MOJOLEARN_CD_IDN_GRAM_WIDE on NV+AMD, rows of its family |
| `MOJOLEARN_CD_THREE_LAUNCH_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | solver/impl/cd.mojo:560 | owed A/B: default vs the old path MOJOLEARN_CD_THREE_LAUNCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CD_TWO_STEP_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | solver/impl/cd.mojo:594 | owed A/B: default vs the old path MOJOLEARN_CD_TWO_STEP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT` | KEEP-OPEN | none  | 1+4/0 | 16 | - | experiments/classical_identical_ideas/shared_controls.mojo:93 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS` | KEEP-OPEN | none  | 1+18/0 | 46 | - | experiments/classical_identical_ideas/linear_controls.mojo:249 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C16_GLM_FUSED` | KEEP-OPEN | none  | 1+7/0 | 28 | - | experiments/classical_identical_ideas/linear_controls.mojo:157 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C16_GLM_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_C17_LS_TRIALS` | KEEP-OPEN | none  | 2+3/1 | 5 | - | experiments/classical_identical_ideas/linear_controls.mojo:159 | owed A/B: the default vs MOJOLEARN_CLASSICAL_C17_LS_TRIALS on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED` | KEEP-OPEN | none  | 1+14/0 | 97 | - | experiments/classical_identical_ideas/linear_controls.mojo:89 | owed A/B: the default vs MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED on NV+AMD, rows of its family |
| `MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD` | KEEP-OPEN | none  | 1+9/0 | 53 | - | x_prep/idn_fold.mojo:82 | owed A/B: the default vs MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD on NV+AMD, rows of its family |
| `MOJOLEARN_GLM_CELLS_MIXED` | KEEP-OPEN | none  | 2+0/0 | 3 | - | x_linear/glm_host.mojo:175 | owed A/B: the default vs MOJOLEARN_GLM_CELLS_MIXED on NV+AMD, rows of its family |
| `MOJOLEARN_HUBER_DEVICE_LBFGS_OFF` | KEEP-OPEN | WIN EXP:1256 ACCEPT/DEFAULT: FAST+Apple default | 1+3/0 | 4 | - | x_linear/huber_fast.mojo:71 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_HUBER_DEVICE_LBFGS_OFF |
| `MOJOLEARN_HUBER_FAST_BLOCK512` | KEEP-OPEN | WIN EXP:1423 KEEP (FAST+Apple default, `_OFF`) | 1+3/0 | 7 | - | x_linear/huber_fast.mojo:71 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF` | KEEP-OPEN | WIN EXP:2060 KEEP (promoted 2026-10-11, w2-decide-algos; _OFF restores) | 1+4/0 | 7 | - | x_prep/dradix.mojo:80 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_II_CONV_TREE_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_prep/idn_fold.mojo:88 | owed A/B: default vs the old path MOJOLEARN_IDN_II_CONV_TREE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_II_GRAM_TILE_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_prep/idn_fold.mojo:85 | owed A/B: default vs the old path MOJOLEARN_IDN_II_GRAM_TILE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF` | KEEP-OPEN | WIN EXP:1831 PROMOTED | 1+5/0 | 41 | - | experiments/classical_identical_ideas/fg_linear_controls.mojo:72 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_LARS_GRID_GRAM_OFF` | KEEP-OPEN | WIN EXP:2032 KEEP (default) | 1+3/0 | 19 | - | x_linear/lars.mojo:62 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_prep/fastnb.mojo:40 | owed A/B: default vs the old path MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_QN_EXACT_TRIALS` | KEEP-OPEN | none  | 3+0/1 | 4 | - | experiments/performance_ideas/I12/trials_check.mojo:105 | owed A/B: the default vs MOJOLEARN_IDN_QN_EXACT_TRIALS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RAGGED_RADIX_SELECT` | KEEP-OPEN | none  | 1+3/0 | 35 | - | x_prep/ragged_select.mojo:19 | owed A/B: the default vs MOJOLEARN_IDN_RAGGED_RADIX_SELECT on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RR_EIGH_OFF` | KEEP-OPEN | none  | 1+14/0 | 19 | - | x_prep/host/rr_eigh_host.mojo:50 | owed A/B: default vs the old path MOJOLEARN_IDN_RR_EIGH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RR_MIN_16` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_prep/host/rr_eigh_host.mojo:62 | owed A/B: the default vs MOJOLEARN_IDN_RR_MIN_16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RR_MIN_48` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_prep/host/rr_eigh_host.mojo:63 | owed A/B: the default vs MOJOLEARN_IDN_RR_MIN_48 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RR_MIN_8` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_prep/host/rr_eigh_host.mojo:61 | owed A/B: the default vs MOJOLEARN_IDN_RR_MIN_8 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RR_MIN_96` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_prep/host/rr_eigh_host.mojo:63 | owed A/B: the default vs MOJOLEARN_IDN_RR_MIN_96 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF` | KEEP-OPEN | WIN EXP:2062 KEEP (promoted 2026-10-11; _OFF restores) | 1+2/0 | 3 | - | svm/impl/smosolver.mojo:332 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_TE_ENC_TREE_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_prep/idn_fold.mojo:84 | owed A/B: default vs the old path MOJOLEARN_IDN_TE_ENC_TREE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_prep/idn_fold.mojo:83 | owed A/B: default vs the old path MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XPREP_RADIX_OFF` | KEEP-OPEN | none  | 1+3/0 | 6 | - | x_prep/dradix.mojo:42 | owed A/B: default vs the old path MOJOLEARN_IDN_XPREP_RADIX_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_LABEL_DIRECT_OFF` | KEEP-OPEN | WIN EXP:762 KEEP, default merged on main1975c2dc5 | 1+3/0 | 4 | - | x_prep/label_fast.mojo:25 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_LDAQDA_DEC_TILE_OFF` | KEEP-OPEN | WIN EXP:241 KEPT b2dd5dfe5 | 1+0/0 | 1 | - | x_prep/device.mojo:177 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LDAQDA_DEC_TILE_OFF |
| `MOJOLEARN_LDAQDA_PAR_STAGES_OFF` | KEEP-OPEN | WIN EXP:242 KEPT b2dd5dfe5 | 1+0/0 | 1 | - | x_prep/device.mojo:175 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LDAQDA_PAR_STAGES_OFF |
| `MOJOLEARN_LDAQDA_RR_EIGH_OFF` | KEEP-OPEN | WIN EXP:243 KEPT b2dd5dfe5 | 1+0/0 | 1 | - | x_prep/device.mojo:179 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LDAQDA_RR_EIGH_OFF |
| `MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS` | KEEP-OPEN | OPEN EXP:1333 OWED | 1+1/0 | 3 | - | x_prep/fastpt.mojo:125 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_FX_D` | KEEP-OPEN | OPEN EXP:1336 OWED | 1+1/0 | 3 | - | glm/impl/qn/fast_xtdz.mojo:48 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_FX_D on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_QN_ALL` | KEEP-OPEN | none  | 1+2/0 | 4 | - | glm/impl/qn/glm_base.mojo:544 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_QN_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD` | KEEP-OPEN | none  | 1+1/0 | 3 | - | x_linear/device.mojo:2762 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_SVM_WS` | KEEP-OPEN | LOSE EXP:1344 NEUTRAL, not merged | 1+3/0 | 4 | - | svm/impl/workingset.mojo:109 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_LSVR_ALL_OFF` | KEEP-OPEN | LOSE EXP:283 DROPPED-noise | 1+3/0 | 4 | - | glm/impl/qn/glm_base.mojo:534 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_LSVR_EVAL_SLIM_OFF` | KEEP-OPEN | WIN EXP:285 KEPT lane/apple-fast-m2b1-main | 1+10/0 | 13 | - | glm/impl/qn/glm_base.mojo:531 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LSVR_EVAL_SLIM_OFF |
| `MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF` | KEEP-OPEN | WIN EXP:288 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | 1+15/0 | 20 | - | glm/impl/qn/glm_base.mojo:532 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF |
| `MOJOLEARN_MI_CLF_RANKMAJOR` | KEEP-OPEN | WIN EXP:374 KEEP (FAST + Apple default, _OFF) | 1+2/0 | 8 | - | x_prep/device.mojo:215 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_MI_CLF_RANKMAJOR_OFF` | KEEP-OPEN | WIN EXP:374 KEEP (FAST + Apple default, _OFF) | 1+0/0 | 1 | - | x_prep/device.mojo:216 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MI_CLF_RANKMAJOR_OFF |
| `MOJOLEARN_MI_REG_RANKMAJOR` | KEEP-OPEN | OPEN EXP:377 OPEN | 1+3/0 | 4 | - | x_prep/device.mojo:207 | owed A/B: the default vs MOJOLEARN_MI_REG_RANKMAJOR on NV+AMD, rows of its family |
| `MOJOLEARN_MI_REG_SORTCOUNT` | KEEP-OPEN | WIN EXP:379 KEEP (see MI_REG_TIES row) | 1+2/0 | 10 | - | x_prep/device.mojo:208 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_MI_REG_TIES` | KEEP-OPEN | WIN EXP:379 KEEP (see MI_REG_TIES row) | 1+3/0 | 4 | - | x_prep/device.mojo:205 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_MI_REG_TIES_OFF` | KEEP-OPEN | WIN EXP:379 KEEP (see MI_REG_TIES row) | 1+0/0 | 1 | - | x_prep/device.mojo:206 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MI_REG_TIES_OFF |
| `MOJOLEARN_NB_CAT_ATOMIC` | KEEP-OPEN | OPEN EXP:289 OPEN | 1+2/0 | 25 | - | x_prep/fastnb.mojo:44 | owed A/B: the default vs MOJOLEARN_NB_CAT_ATOMIC on NV+AMD, rows of its family |
| `MOJOLEARN_PREP3_MAXABS_OFF` | KEEP-OPEN | WIN EXP:383 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | 1+9/0 | 11 | - | x_prep/prep3.mojo:33 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_PREP3_MAXABS_OFF |
| `MOJOLEARN_PREP3_MAXABS_POOL_OFF` | KEEP-OPEN | WIN EXP:920 DEFAULT; M2 default/_OFF both rc=0, merged. Only MaxAbs paths extracted; resampl | 1+3/0 | 4 | - | x_prep/prep3.mojo:42 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED_OFF` | KEEP-OPEN | WIN EXP:405 KEEP, FAST+Apple default (`_OFF` off) | 1+8/0 | 27 | - | preprocessing/minmax.mojo:55 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL_OFF` | KEEP-OPEN | WIN EXP:404 KEEP, FAST+Apple default (`_OFF` off) | 1+8/0 | 19 | - | preprocessing/minmax.mojo:56 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PT_SCORE` | KEEP-OPEN | LOSE EXP:2280 DELETED (DROP quality) | 1+5/0 | 11 | c | x_prep/fastpt.mojo:66 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_PT_SCORE_STABLE_OFF` | KEEP-OPEN | WIN EXP:1890 KEPT (default path) | 1+0/0 | 1 | c | x_prep/pt_center.mojo:22 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_PURITY2_2_OFF` | KEEP-OPEN | OPEN EXP:721 OPEN | 1+0/0 | 1 | - | glm/impl/qn/glm_base.mojo:1514 | owed A/B: default vs the old path MOJOLEARN_PURITY2_2_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_PURITY_1_OFF` | KEEP-OPEN | none  | 2+0/0 | 30 | - | glm/estimator.mojo:212 | owed A/B: default vs the old path MOJOLEARN_PURITY_1_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_QN_DEV_ARGMAX_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | glm/estimator.mojo:840 | owed A/B: default vs the old path MOJOLEARN_QN_DEV_ARGMAX_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_QN_FAST_XTDZ_OFF` | KEEP-OPEN | LOSE EXP:1930 DELETED | 1+6/0 | 7 | c | glm/impl/qn/glm_base.mojo:108 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_QN_IDN_FUSED_OFF` | KEEP-OPEN | none  | 1+5/0 | 39 | - | glm/impl/qn/glm_base.mojo:1222 | owed A/B: default vs the old path MOJOLEARN_QN_IDN_FUSED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_QN_IDN_SLIM_OFF` | KEEP-OPEN | none  | 1+4/0 | 13 | - | glm/impl/qn/glm_base.mojo:1040 | owed A/B: default vs the old path MOJOLEARN_QN_IDN_SLIM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF` | KEEP-OPEN | none  | 1+2/0 | 18 | - | glm/estimator.mojo:602 | owed A/B: default vs the old path MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_QN_TILED_OFF` | KEEP-OPEN | LOSE EXP:2203 DELETED (serial shape) | 2+8/0 | 10 | - | glm/host/qn_oracle.mojo:156 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_RIDGE_FF_ALWAYS` | KEEP-OPEN | none  | 1+1/0 | 3 | - | x_linear/ridge.mojo:78 | owed A/B: the default vs MOJOLEARN_RIDGE_FF_ALWAYS on NV+AMD, rows of its family |
| `MOJOLEARN_RIDGE_NO_U_OFF` | KEEP-OPEN | WIN EXP:246 KEPT 2caff7b0c | 1+4/0 | 19 | - | glm/impl/ridge.mojo:115 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_RIDGE_NO_U_OFF |
| `MOJOLEARN_RIDGE_RESIDENT_OFF` | KEEP-OPEN | WIN EXP:247 KEPT 2caff7b0c | 2+10/0 | 29 | - | glm/impl/ridge.mojo:116 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_RIDGE_RESIDENT_OFF |
| `MOJOLEARN_SELECT_FCLS_OFF` | KEEP-OPEN | WIN EXP:367 KEPT e2bfb8422 | 1+3/0 | 4 | - | x_prep/select_fast.mojo:65 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SELECT_FCLS_OFF |
| `MOJOLEARN_SELECT_FREG_OFF` | KEEP-OPEN | WIN EXP:368 KEPT 4198d5a9c | 1+3/0 | 4 | - | x_prep/select_fast.mojo:64 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SELECT_FREG_OFF |
| `MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF` | KEEP-OPEN | OPEN EXP:16 / `MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF` (both modes, default on; PENDING) / sk-kern | 1+2/0 | 3 | - | x_linear/sgd.mojo:1043 | owed A/B: default vs the old path MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_AUTO_STEPS_256_OFF` | KEEP-OPEN | OPEN EXP:17 SGDClassifier/SGDRegressor/Perceptron/PassiveAggressive pair 'auto': batch = n_s | 1+3/0 | 4 | - | x_linear/sgd.mojo:1066 | owed A/B: default vs the old path MOJOLEARN_SGD_AUTO_STEPS_256_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_linear/device.mojo:947 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF` | KEEP-OPEN | none  | 1+4/0 | 8 | - | x_linear/device.mojo:1158 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_DEV_FINITE_OFF` | KEEP-OPEN | none  | 1+4/0 | 8 | - | x_linear/device.mojo:1157 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_DEV_FINITE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_MB_FUSE_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | x_linear/device.mojo:962 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_MB_FUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_OVR_PAR_OFF` | KEEP-OPEN | none  | 1+4/0 | 9 | - | x_linear/device.mojo:1380 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_OVR_PAR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_IDN_PS_WARP_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | x_linear/device.mojo:2974 | owed A/B: default vs the old path MOJOLEARN_SGD_IDN_PS_WARP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF` | KEEP-OPEN | none  | 1+3/0 | 5 | - | x_linear/sgd.mojo:1018 | owed A/B: default vs the old path MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | x_linear/sgd.mojo:1005 | owed A/B: default vs the old path MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SI_ONEPASS` | KEEP-OPEN | LOSE EXP:1888 DELETED (quality, with COLBATCH) | 1+4/0 | 7 | c | x_prep/fastpt.mojo:91 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_SI_ONEPASS_OFF` | KEEP-OPEN | LOSE EXP:1888 DELETED (quality, with COLBATCH) | 1+0/0 | 1 | - | x_prep/fastpt.mojo:92 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_SVM_IDN_FUSED_TILE_OFF` | KEEP-OPEN | none  | 1+2/0 | 6 | - | svm/impl/distance/kernel_matrices.mojo:393 | owed A/B: default vs the old path MOJOLEARN_SVM_IDN_FUSED_TILE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF` | KEEP-OPEN | none  | 1+2/0 | 15 | - | svm/impl/smosolver.mojo:164 | owed A/B: default vs the old path MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF` | KEEP-OPEN | none  | 1+3/0 | 15 | - | svm/impl/smosolver.mojo:180 | owed A/B: default vs the old path MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SVM_RBF_EPILOGUE_1D` | KEEP-OPEN | none  | 1+2/0 | 9 | - | svm/impl/distance/kernel_matrices.mojo:113 | owed A/B: the default vs MOJOLEARN_SVM_RBF_EPILOGUE_1D on NV+AMD, rows of its family |
| `MOJOLEARN_SVM_WS1024` | KEEP-OPEN | none  | 1+5/0 | 6 | - | svm/impl/workingset.mojo:99 | owed A/B: the default vs MOJOLEARN_SVM_WS1024 on NV+AMD, rows of its family |
| `MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF` | KEEP-OPEN | none  | 1+15/0 | 29 | - | x_linear/finite_device.mojo:31 | owed A/B: default vs the old path MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_X_LINEAR_ARD_LEAD` | KEEP-OPEN | none  | 1+0/0 | 6 | - | x_linear/bayes.mojo:647 | owed A/B: the default vs MOJOLEARN_X_LINEAR_ARD_LEAD on NV+AMD, rows of its family |
| `MOJOLEARN_X_LINEAR_NO_STAGED_FOLD` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_linear/tops.mojo:353 | owed A/B: the default vs MOJOLEARN_X_LINEAR_NO_STAGED_FOLD on NV+AMD, rows of its family |
| `MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF` | KEEP-OPEN | WIN EXP:252 KEPT c5e1bbeb6 | 1+0/0 | 1 | - | x_prep/device.mojo:473 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF |
| `MOJOLEARN_X_PREP_FAST_CLS2_PACK_OFF` | KEEP-OPEN | WIN EXP:406 KEEP, FAST+Apple default (`_OFF` off) | 1+7/0 | 8 | - | x_prep/cat_cls2.mojo:43 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_X_PREP_POOL_ARENA_OFF` | KEEP-OPEN | WIN EXP:409 DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_PREP_POOL_ARENA_OFF` | 1+0/0 | 1 | - | x_prep/device.mojo:103 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_X_PREP_POOL_ARENA_OFF |
| `MOJOLEARN_ARD_FAST_CLS1_STATS_OFF` | KEEP | WIN EXP:274 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:51 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BAYES_FAST_CLS1_STATS_OFF` | KEEP | WIN EXP:278 KEPT 9decae29f | 1+1/0 | 2 | - | x_linear/cls1_fast.mojo:48 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CD_FAST_GRAM_OFF` | KEEP | none  | 1+8/0 | 13 | - | solver/impl/cd.mojo:193 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CD_SABOTAGE_NO_FTZ_RESID` | KEEP | none  | 1+2/0 | 5 | - | solver/host/cd_oracle.mojo:98 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON` | KEEP | none  | 1+3/0 | 6 | - | solver/checks/record_canon.mojo:67 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CD_SABOTAGE_SOFT_SWAP` | KEEP | none  | 1+4/0 | 12 | - | solver/impl/cd.mojo:160 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX` | KEEP | none  | 1+5/0 | 8 | - | solver/impl/cd.mojo:174 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GLM_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 4 | - | glm/impl/qn/multi_gpu.mojo:99 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GLM_TRACE` | KEEP | none  | 3+0/0 | 3 | - | x_linear/glm.mojo:387 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OLS_FAST_SMALL_NT` | KEEP | none  | 1+2/0 | 11 | - | glm/impl/linalg/detail/lstsq.mojo:169 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SGDOC_FAST_TAIL_LONG` | KEEP | none  | 1+2/0 | 3 | - | x_linear/device.mojo:2735 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SGD_FAST_EPOCH_RESIDENT` | KEEP | OPEN EXP:1596 OPEN | 1+4/0 | 5 | - | x_linear/device.mojo:1068 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SGD_LOOP_CHECK_EVERY` | KEEP | OPEN EXP:2360 owed (nv+amd, arm B = `-D MOJOLEARN_SGD_LOOP_CHECK_EVERY=1`, + one non-board sha | 1+3/0 | 4 | c | x_linear/device.mojo:1803 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SOFTMAX_SABOTAGE` | KEEP | none  | 1+1/0 | 9 | - | glm/impl/qn/glm_softmax.mojo:112 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SOLVER_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 5 | - | solver/multi_gpu.mojo:76 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_FAST_SORT_OFF` | KEEP | none  | 1+5/0 | 12 | - | svm/impl/workingset.mojo:91 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SVM_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 4 | - | svm/multi_gpu.mojo:130 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_ARG_TIE_HIGH` | KEEP | none  | 1+3/0 | 10 | - | svm/checks/pinned_argreduce.mojo:66 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX` | KEEP | none  | 1+1/0 | 3 | - | svm/impl/smoblocksolve.mojo:118 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX_SWAP` | KEEP | none  | 1+1/0 | 4 | - | svm/impl/smoblocksolve.mojo:119 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_FMAX_NOKEY` | KEEP | none  | 1+1/0 | 4 | - | svm/impl/smoblocksolve.mojo:117 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE` | KEEP | none  | 1+2/0 | 4 | - | svm/impl/smosolver.mojo:124 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_NO_FTZ` | KEEP | none  | 2+5/0 | 11 | - | svm/impl/distance/kernel_matrices.mojo:79 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_STD_EXP` | KEEP | none  | 2+5/0 | 11 | - | svm/impl/distance/kernel_matrices.mojo:75 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SVM_SABOTAGE_WS_TIE` | KEEP | none  | 1+2/0 | 9 | - | svm/impl/workingset.mojo:78 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_WITNESS_SABOTAGE` | KEEP | none  | 1+0/0 | 4 | - | x_linear/witness.mojo:45 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_X_LINEAR_FAST_FMA` | KEEP | none  | 1+0/0 | 1 | - | x_linear/ops.mojo:86 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_CHOL_FAST_BLOCKED` | REMOVED | LOSE EXP:1906 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CHOL_FAST_NB512` | REMOVED | LOSE EXP:2282 DELETED (HOLD-speed) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN` | REMOVED | WIN EXP:1740 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF` | REMOVED | LOSE EXP:1744 DROP (off arm slower), switch deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_C24_ROWS2048` | REMOVED | WIN EXP:1638 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE` | REMOVED | LOSE EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_SGD_EPOCH_CONTIGUOUS` | REMOVED | LOSE EXP:2321 DELETED (removed, never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS` | REMOVED | LOSE EXP:1994 DROPPED (slower), code deleted | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_DBSCAN_FAST_DENSEBALL` | REMOVED | LOSE EXP:1935 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_BM_GRAM_RB4` | REMOVED | LOSE EXP:2038 LOSER, DELETED 2026-10-11 (bde9c2243) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CD_GRAM_EPOCHS_64` | REMOVED | LOSE EXP:1950 DROPPED-slower, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_COLVAR_FUSED` | REMOVED | WIN EXP:1946 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE` | REMOVED | LOSE EXP:1973 DROPPED (noise; deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_DBSCAN_CC_HOOK` | REMOVED | LOSE EXP:1974 DROPPED (noise; deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_OZAKI_LINALG` | REMOVED | LOSE EXP:1728 S=4: NV 69.6 -> 276.6 (3.972), AMD 28.5 -> 58.8 (2.061), 2.86x combined; S=5: NV | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GFF_GRAM_TILED` | REMOVED | WIN EXP:2096 KEEP (promoted) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GRAM_FF_FALLBACK` | REMOVED | WIN EXP:1832 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GS_SOLVE_ONE_LAUNCH` | REMOVED | LOSE EXP:2097 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_KDE_QTILE` | REMOVED | WIN EXP:2059 KEEP by average (promoted 2026-10-11, istella regression flagged; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LINEAR_PINNED_UPLOAD_OFF` | REMOVED | LOSE grid fg-linear.json removed | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LU_DELAYED_UPDATE` | REMOVED | LOSE EXP:477 LOSER, DELETED 2026-10-11 (aaf94d8a9) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LU_NB128` | REMOVED | LOSE EXP:2016 DROPPED-slower | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF` | REMOVED | LOSE EXP:1732 DROP (the off arm is slower), switch deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SGD_EPOCH_KERNEL` | REMOVED | LOSE EXP:1784 DELETED (noise) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TSQR_RSTACK_TREE` | REMOVED | LOSE EXP:2100 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KDE2_ALL` | REMOVED | LOSE EXP:1921 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KDE_KERNEL_VARIANTS` | REMOVED | LOSE EXP:1922 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KDE_LSE_FUSED` | REMOVED | LOSE EXP:1923 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KDE_NORM_FUSED` | REMOVED | LOSE EXP:1924 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KDE_SAMPLE_FUSED` | REMOVED | LOSE EXP:1925 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_LEGACY_NARROW_SCOPED_GEMM` | REMOVED | LOSE EXP:2310 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_LEGACY_NARROW_SOFTMAX_G2` | REMOVED | LOSE EXP:2313 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_QN_FAST_COALESCED_OFF` | REMOVED | LOSE EXP:1930 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_QN_IDN_DCONV` | REMOVED | LOSE EXP:1625 DROP (neutral / slower), code deleted | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_QN_IDN_DCONV_POLL_2` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_QN_IDN_DCONV_POLL_8` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_AUDIT` | REMOVED | LOSE EXP:2315 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_G1_DENSE` | REMOVED | LOSE EXP:2305 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_G1_GRAM` | REMOVED | LOSE EXP:2306 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_G1_TALL` | REMOVED | LOSE EXP:2304 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_G2_NARROW` | REMOVED | LOSE EXP:2307 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_PCA` | REMOVED | LOSE EXP:2309 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCOPED_GEMM_SPLIT` | REMOVED | LOSE EXP:2308 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SOFTMAX_FAST_G2_NARROW` | REMOVED | LOSE EXP:2311 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SOFTMAX_G2_AUDIT` | REMOVED | LOSE EXP:2312 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_X_LINEAR_JACOBI_LEAD` | REMOVED | LOSE EXP:2357 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |

### neural (718)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_AFN_OPTIM_ALL` | DELETE-LOSER | LOSE EXP:1209 NEUTRAL | 1+3/0 | 4 | c | training/afn_optim.mojo:112 | expdel-04 |
| `MOJOLEARN_ATTN_DKDV_MFMA` | KEEP-OPEN | LOSE grid neural-small.json removed | 1+2/0 | 3 | c | transformer/impl/llama/fused_attention.mojo:6753 | KEEP-OPEN (expdel-04 2026-11-15: the grid `removed` entry says "Not gridded (AMD)": the arm engages only at BJ == 32 and no-ops at the AMD tile rule; never measured, no losing A/B; owed: a BJ 16 body or pairing, then an AMD A/B) |
| `MOJOLEARN_ATTN_DQ_MFMA` | KEEP-OPEN | LOSE grid neural-small.json removed | 1+2/0 | 3 | c | transformer/impl/llama/fused_attention.mojo:5865 | KEEP-OPEN (expdel-04 2026-11-15: the grid `removed` entry says "Not gridded (AMD)": the dq MFMA kernel aborts rc 134, "repair before gridding"; never measured, no losing A/B; owed: repair, then an AMD A/B) |
| `MOJOLEARN_IDN_ATTN_TILE_ORDER` | KEEP-OPEN | LOSE grid neural-small.json removed | 1+5/0 | 6 | c | transformer/impl/llama/fused_attention.mojo:1509 | KEEP-OPEN (expdel-04 2026-11-15: the grid `removed` entry covers only the never-built `desc` / `paired` values; the live arms `asc` and `zdot_asc` are control status "new", unmeasured, and the H100 leg measured `_bswz` costing zdot 14.79 -> 15.61 ms, so `zdot_asc` is not a known loser; owed: NV + AMD A/B of asc and zdot_asc) |
| `MOJOLEARN_AF_FAST_NOFILL_OFF` | DELETE-OLD-PATH | WIN EXP:1377 KEEP (FAST+Apple default, `_OFF`) | 1+2/0 | 13 | - | sequence/ops_impl.mojo:51 | expdel-04 |
| `MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF` | DELETE-OLD-PATH | WIN grid samba-resident.json promoted:ge123e6f9 | 1+9/0 | 15 | - | training/neural_identical_experiments.mojo:101 | expdel-04 |
| `MOJOLEARN_IDN_SEQ_REG_OFF` | DELETE-OLD-PATH | WIN grid seq-reg.json promoted:lane/i17-decide-linalg-seq | 1+5/0 | 16 | - | sequence/theta.mojo:63 | expdel-04 |
| `MOJOLEARN_LN_FAST_NOFILL_OFF` | DELETE-OLD-PATH | WIN EXP:1211 KEEP: FAST+Apple default | 1+3/0 | 10 | - | sequence/ops_impl.mojo:56 | expdel-04 |
| `MOJOLEARN_NI12_IMPLICIT_CONV_OFF` | KEEP-OPEN | WIN grid neural-seq-train-dedupe.json promoted 2026-10-10 (lane/postmerge-act-6; post-merge neural | 1+6/0 | 7 | - | x_cnn/device.mojo:101 | KEEP-OPEN (expdel-04 2026-11-15: x_cnn/device.mojo MEASURE OWED note: grid g50ebe26a5 has the _OFF arm FASTER on cnn-clf synthetic, NV 193.6 -> 173.9 ms v1446, AMD 303.5 -> 274.0 ms b0080, avg 0.90x, while the promotion A/B won conv2d / resnet-block 0.69x-0.74x; the old path is a measured winner on cnn-clf; owed: per-algorithm confirm on both vendors) |
| `MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF` | DELETE-OLD-PATH | WIN EXP:591 KEPT f6a9e7c04 | 1+6/0 | 7 | - | sequence/exec_device.mojo:151 | expdel-04 |
| `MOJOLEARN_SEQ_FAST_PIPE_UP_OFF` | DELETE-OLD-PATH | WIN EXP:592 KEPT f6a9e7c04 | 1+2/0 | 3 | - | sequence/exec_device.mojo:148 | expdel-04 |
| `MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF` | DELETE-OLD-PATH | WIN EXP:542 KEPT f6a9e7c04 | 1+2/0 | 5 | - | sequence/theta.mojo:88 | expdel-04 |
| `MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF` | DELETE-OLD-PATH | WIN EXP:543 KEPT 6c0ba4379 | 1+2/0 | 3 | - | sequence/theta_spec.mojo:44 | expdel-04 |
| `MOJOLEARN_SEQ_FAST_VAR_COOP_OFF` | DELETE-OLD-PATH | WIN EXP:1422 KEEP (FAST+Apple default, `_OFF`) | 1+6/0 | 10 | - | sequence/ops.mojo:119 | expdel-04 |
| `MOJOLEARN_AFN26_ATTN_FLASH_BK16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:486 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_FLASH_BK16 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_ATTN_FLASH_TQ16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:487 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_FLASH_TQ16 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_ATTN_NORM_TPB128` | KEEP-OPEN | none  | 1+2/1 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:128 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_NORM_TPB128 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_ATTN_NORM_TPB512` | KEEP-OPEN | none  | 1+2/1 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:129 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_NORM_TPB512 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_ATTN_PROJ_BM32` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:918 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_PROJ_BM32 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_ATTN_PROJ_KB16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:917 | owed A/B: the default vs MOJOLEARN_AFN26_ATTN_PROJ_KB16 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_CNN_CHANNELS16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/afn_direct.mojo:55 | owed A/B: the default vs MOJOLEARN_AFN26_CNN_CHANNELS16 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_CNN_COLS_ONCE` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/afn_direct.mojo:59 | owed A/B: the default vs MOJOLEARN_AFN26_CNN_COLS_ONCE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_CNN_K16` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/afn_direct.mojo:56 | owed A/B: the default vs MOJOLEARN_AFN26_CNN_K16 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_CNN_ROWS32` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/afn_direct.mojo:54 | owed A/B: the default vs MOJOLEARN_AFN26_CNN_ROWS32 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_BWD_EPILOGUE` | KEEP-OPEN | LOSE EXP:2201 DELETED (removed) | 1+2/1 | 15 | cr | autograd/kernels/transformer_afn.mojo:80 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_AFN26_LM_BWD_FUSE` | KEEP-OPEN | none  | 1+4/0 | 5 | - | training/byte_lm_afn.mojo:68 | owed A/B: the default vs MOJOLEARN_AFN26_LM_BWD_FUSE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_BWD_NORM1_RESID` | KEEP-OPEN | none  | 1+1/0 | 2 | - | autograd/kernels/transformer_afn.mojo:85 | owed A/B: the default vs MOJOLEARN_AFN26_LM_BWD_NORM1_RESID on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_CE_BLOCK128` | KEEP-OPEN | none  | 1+8/0 | 9 | - | training/byte_lm_afn.mojo:107 | owed A/B: the default vs MOJOLEARN_AFN26_LM_CE_BLOCK128 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_CE_BLOCK512` | KEEP-OPEN | none  | 1+8/0 | 9 | - | training/byte_lm_afn.mojo:108 | owed A/B: the default vs MOJOLEARN_AFN26_LM_CE_BLOCK512 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_HEAD_FUSE` | KEEP-OPEN | none  | 1+1/1 | 2 | i | training/byte_lm_afn.mojo:90 | owed A/B: the default vs MOJOLEARN_AFN26_LM_HEAD_FUSE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_NOSYNC` | KEEP-OPEN | none  | 1+1/1 | 2 | i | training/byte_lm_afn.mojo:41 | owed A/B: the default vs MOJOLEARN_AFN26_LM_NOSYNC on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_PARAM_VIEWS` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/byte_lm_afn.mojo:81 | owed A/B: the default vs MOJOLEARN_AFN26_LM_PARAM_VIEWS on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_WGRAD_CAP8` | KEEP-OPEN | none  | 1+2/0 | 3 | - | autograd/kernels/transformer_afn.mojo:105 | owed A/B: the default vs MOJOLEARN_AFN26_LM_WGRAD_CAP8 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_WGRAD_MIN128` | KEEP-OPEN | none  | 1+3/0 | 4 | - | autograd/kernels/transformer_afn.mojo:98 | owed A/B: the default vs MOJOLEARN_AFN26_LM_WGRAD_MIN128 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` | KEEP-OPEN | none  | 1+2/0 | 20 | - | autograd/kernels/transformer_afn.mojo:75 | owed A/B: the default vs MOJOLEARN_AFN26_LM_WGRAD_SPLIT on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LOSS_BLOCK128` | KEEP-OPEN | none  | 1+2/0 | 3 | - | training/afn_optim.mojo:163 | owed A/B: the default vs MOJOLEARN_AFN26_LOSS_BLOCK128 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LOSS_BLOCK512` | KEEP-OPEN | none  | 1+2/0 | 3 | - | training/afn_optim.mojo:164 | owed A/B: the default vs MOJOLEARN_AFN26_LOSS_BLOCK512 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_LOSS_FUSED` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/afn_optim.mojo:139 | owed A/B: the default vs MOJOLEARN_AFN26_LOSS_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_OPT_BLOCK128` | KEEP-OPEN | none  | 1+22/0 | 23 | - | training/afn_optim.mojo:154 | owed A/B: the default vs MOJOLEARN_AFN26_OPT_BLOCK128 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_OPT_BLOCK512` | KEEP-OPEN | none  | 1+22/0 | 23 | - | training/afn_optim.mojo:155 | owed A/B: the default vs MOJOLEARN_AFN26_OPT_BLOCK512 on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_OPT_CLIP_FUSE` | KEEP-OPEN | none  | 1+4/0 | 5 | - | training/afn_optim.mojo:119 | owed A/B: the default vs MOJOLEARN_AFN26_OPT_CLIP_FUSE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN26_OPT_MULTITENSOR` | KEEP-OPEN | none  | 1+4/0 | 16 | - | training/afn_optim.mojo:128 | owed A/B: the default vs MOJOLEARN_AFN26_OPT_MULTITENSOR on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_ALL` | KEEP-OPEN | OPEN EXP:595 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+7/0 | 8 | - | transformer/impl/llama/afn_apple_fast.mojo:74 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_AUDIT` | KEEP-OPEN | none  | 1+2/0 | 6 | - | transformer/impl/llama/afn_apple_fast.mojo:856 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_AUDIT on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_FLASH` | KEEP-OPEN | OPEN EXP:597 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | - | transformer/impl/llama/afn_apple_fast.mojo:106 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_FLASH on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_FUSE_MLP` | KEEP-OPEN | OPEN EXP:598 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:114 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_FUSE_MLP on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_FUSE_PRE` | KEEP-OPEN | OPEN EXP:599 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:110 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_FUSE_PRE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_GQA_TILE` | KEEP-OPEN | OPEN EXP:600 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | - | transformer/impl/llama/afn_apple_fast.mojo:93 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_GQA_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_NORM_SG` | KEEP-OPEN | OPEN EXP:601 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:77 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_NORM_SG on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_ATTN_ROPE_CACHE` | KEEP-OPEN | OPEN EXP:602 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 3 | - | transformer/impl/llama/afn_apple_fast.mojo:81 | owed A/B: the default vs MOJOLEARN_AFN_ATTN_ROPE_CACHE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_CNN_DIRECT` | KEEP-OPEN | OPEN EXP:603 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+8/0 | 45 | - | x_cnn/afn_direct.mojo:45 | owed A/B: the default vs MOJOLEARN_AFN_CNN_DIRECT on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_EPI_ALL` | KEEP-OPEN | OPEN EXP:606 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 2+7/0 | 54 | - | mamba/impl/modules/afn_proj_gemm.mojo:83 | owed A/B: the default vs MOJOLEARN_AFN_EPI_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF` | KEEP-OPEN | none  | 1+2/0 | 5 | - | transformer/impl/llama/afn_apple_fast.mojo:480 | owed A/B: default vs the old path MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_ALL` | KEEP-OPEN | OPEN EXP:607 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+5/0 | 6 | - | gemm/afn_apple_fast2.mojo:77 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_BIGTILE` | KEEP-OPEN | OPEN EXP:608 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 10 | - | gemm/afn_apple_fast2.mojo:79 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_BIGTILE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_DBUF` | KEEP-OPEN | OPEN EXP:609 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 4 | - | gemm/afn_apple_fast2.mojo:82 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_DBUF on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_DIRECT_B` | KEEP-OPEN | OPEN EXP:610 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 5 | - | gemm/afn_apple_fast2.mojo:85 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_DIRECT_B on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_SWIZZLE` | KEEP-OPEN | OPEN EXP:685 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 4 | - | gemm/afn_apple_fast2.mojo:88 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_SWIZZLE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM2_SWZ_G` | KEEP-OPEN | OPEN EXP:685 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 4 | - | gemm/afn_apple_fast2.mojo:99 | owed A/B: the default vs MOJOLEARN_AFN_GEMM2_SWZ_G on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_ALL` | KEEP-OPEN | OPEN EXP:612 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+6/0 | 7 | - | gemm/afn_apple_fast.mojo:105 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_BF16_MMA` | KEEP-OPEN | OPEN EXP:613 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+8/0 | 22 | - | gemm/afn_apple_fast.mojo:132 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_BF16_MMA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_CORES` | KEEP-OPEN | OPEN EXP:684 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 2+7/0 | 9 | - | gemm/afn_apple_fast.mojo:171 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_CORES on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_EPILOGUE` | KEEP-OPEN | OPEN EXP:682 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+7/0 | 69 | - | gemm/afn_apple_fast.mojo:138 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_EPILOGUE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_INT8_MMA` | KEEP-OPEN | OPEN EXP:614 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 3 | - | gemm/afn_apple_fast.mojo:135 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_INT8_MMA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_SIMDGROUP` | KEEP-OPEN | OPEN EXP:617 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 4 | - | gemm/afn_apple_fast.mojo:107 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_SIMDGROUP on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_SPLITK` | KEEP-OPEN | OPEN EXP:618 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+7/0 | 25 | - | gemm/afn_apple_fast.mojo:110 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_SPLITK on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_GEMM_TILESHAPE` | KEEP-OPEN | OPEN EXP:619 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+6/0 | 16 | - | gemm/afn_apple_fast.mojo:113 | owed A/B: the default vs MOJOLEARN_AFN_GEMM_TILESHAPE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LMGRAD_ALL` | KEEP-OPEN | OPEN EXP:620 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 4 | c | autograd/kernels/transformer_afn.mojo:71 | owed A/B: the default vs MOJOLEARN_AFN_LMGRAD_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_ALL` | KEEP-OPEN | OPEN EXP:621 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 5 | - | training/byte_lm_afn.mojo:30 | owed A/B: the default vs MOJOLEARN_AFN_LM_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_BWD_EPILOGUE` | KEEP-OPEN | LOSE EXP:2201 DELETED (removed) | 1+2/1 | 15 | icr | autograd/kernels/transformer_afn.mojo:79 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_AFN_LM_BWD_FUSE` | KEEP-OPEN | OPEN EXP:623 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 5 | - | training/byte_lm_afn.mojo:67 | owed A/B: the default vs MOJOLEARN_AFN_LM_BWD_FUSE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_BWD_NORM1_RESID` | KEEP-OPEN | OPEN EXP:624 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+1/0 | 2 | - | autograd/kernels/transformer_afn.mojo:84 | owed A/B: the default vs MOJOLEARN_AFN_LM_BWD_NORM1_RESID on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | KEEP-OPEN | OPEN EXP:625 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 5 | - | training/byte_lm_afn.mojo:56 | owed A/B: default vs the old path MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_HEAD_FUSE` | KEEP-OPEN | LOSE EXP:2340 DELETED (their step deleted; default off) | 1+1/1 | 2 | ic | training/byte_lm_afn.mojo:89 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_AFN_LM_NOSYNC` | KEEP-OPEN | LOSE EXP:2340 DELETED (their step deleted; default off) | 1+1/1 | 2 | ic | training/byte_lm_afn.mojo:40 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_AFN_LM_PARAM_VIEWS` | KEEP-OPEN | OPEN EXP:628 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+1/0 | 2 | - | training/byte_lm_afn.mojo:80 | owed A/B: the default vs MOJOLEARN_AFN_LM_PARAM_VIEWS on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LM_WGRAD_SPLIT` | KEEP-OPEN | OPEN EXP:629 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 20 | - | autograd/kernels/transformer_afn.mojo:74 | owed A/B: the default vs MOJOLEARN_AFN_LM_WGRAD_SPLIT on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_LOSS_FUSED` | KEEP-OPEN | OPEN EXP:630 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+1/0 | 2 | - | training/afn_optim.mojo:138 | owed A/B: the default vs MOJOLEARN_AFN_LOSS_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` | KEEP-OPEN | OPEN EXP:631 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | - | mamba/impl/modules/afn_defines.mojo:162 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA1_CHUNKSCAN on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA1_FUSE_IN` | KEEP-OPEN | OPEN EXP:632 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | c | mamba/impl/modules/afn_defines.mojo:174 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA1_FUSE_IN on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA2_SSD_MMA` | KEEP-OPEN | OPEN EXP:633 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | - | mamba/impl/modules/afn_defines.mojo:186 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA2_SSD_MMA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA3_BWD_ARENA` | KEEP-OPEN | OPEN EXP:634 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 2+10/0 | 50 | - | autograd/kernels/mamba3_prefill.mojo:42 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA3_BWD_ARENA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA3_BWD_CHUNK` | KEEP-OPEN | OPEN EXP:635 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 2+2/0 | 4 | - | autograd/kernels/mamba3.mojo:1501 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA3_BWD_CHUNK on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` | KEEP-OPEN | OPEN EXP:636 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | - | mamba/impl/modules/afn_defines.mojo:198 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA3_SISO_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA_ALL` | KEEP-OPEN | OPEN EXP:637 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+7/0 | 8 | - | mamba/impl/modules/afn_defines.mojo:53 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA_ARENA` | KEEP-OPEN | OPEN EXP:638 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+0/0 | 1 | c | mamba/impl/modules/afn_defines.mojo:210 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA_ARENA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL` | KEEP-OPEN | OPEN EXP:639 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+2/0 | 4 | - | mamba/impl/modules/afn_defines.mojo:223 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE` | KEEP-OPEN | OPEN EXP:641 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+3/0 | 5 | - | mamba/impl/modules/afn_proj_gemm.mojo:85 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_MAMBA_PROJ_SPLITK` | KEEP-OPEN | OPEN EXP:641 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 20 | - | mamba/impl/modules/afn_proj_gemm.mojo:88 | owed A/B: the default vs MOJOLEARN_AFN_MAMBA_PROJ_SPLITK on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_OPT_CLIP_FUSE` | KEEP-OPEN | OPEN EXP:647 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 5 | - | training/afn_optim.mojo:118 | owed A/B: the default vs MOJOLEARN_AFN_OPT_CLIP_FUSE on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_OPT_MULTITENSOR` | KEEP-OPEN | OPEN EXP:649 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 16 | - | training/afn_optim.mojo:127 | owed A/B: the default vs MOJOLEARN_AFN_OPT_MULTITENSOR on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_ALL` | KEEP-OPEN | OPEN EXP:652 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 5+17/0 | 60 | - | autograd/kernels/mamba3.mojo:1502 | owed A/B: the default vs MOJOLEARN_AFN_SAMBA_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_ARENA` | KEEP-OPEN | OPEN EXP:653 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+6/0 | 21 | - | training/samba_afn.mojo:138 | owed A/B: the default vs MOJOLEARN_AFN_SAMBA_ARENA on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT` | KEEP-OPEN | OPEN EXP:654 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+12/0 | 23 | - | training/samba_afn.mojo:141 | owed A/B: the default vs MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_EMB_ATOMIC` | KEEP-OPEN | OPEN EXP:655 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+5/0 | 25 | - | training/samba_afn.mojo:144 | owed A/B: the default vs MOJOLEARN_AFN_SAMBA_EMB_ATOMIC on NV+AMD, rows of its family |
| `MOJOLEARN_AFN_SAMBA_HEAD_GEMM` | KEEP-OPEN | OPEN EXP:657 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+4/0 | 50 | - | training/samba_afn.mojo:152 | owed A/B: the default vs MOJOLEARN_AFN_SAMBA_HEAD_GEMM on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_AMMA_ADMIT` | KEEP-OPEN | none  | 1+10/0 | 41 | - | transformer/impl/llama/fused_attention.mojo:8893 | owed A/B: the default vs MOJOLEARN_ATTN_AMMA_ADMIT on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_ARM_TRIAL` | KEEP-OPEN | none  | 1+27/0 | 511 | - | transformer/impl/llama/fused_attention.mojo:388 | owed A/B: the default vs MOJOLEARN_ATTN_ARM_TRIAL on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_CACHE_WAIT` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/modeling_llama.mojo:4092 | owed A/B: the default vs MOJOLEARN_ATTN_CACHE_WAIT on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DKDV_AMMA` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:6154 | owed A/B: the default vs MOJOLEARN_ATTN_DKDV_AMMA on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DKDV_BJ16` | KEEP-OPEN | none  | 2+4/0 | 7 | - | transformer/impl/llama/fused_attention.mojo:4730 | owed A/B: the default vs MOJOLEARN_ATTN_DKDV_BJ16 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DKDV_BJ32` | KEEP-OPEN | none  | 2+4/0 | 7 | - | transformer/impl/llama/fused_attention.mojo:4731 | owed A/B: the default vs MOJOLEARN_ATTN_DKDV_BJ32 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DKDV_NO_T4` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/impl/llama/fused_attention.mojo:6541 | owed A/B: the default vs MOJOLEARN_ATTN_DKDV_NO_T4 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DKDV_T4` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/impl/llama/fused_attention.mojo:6540 | owed A/B: the default vs MOJOLEARN_ATTN_DKDV_T4 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DQ_TQ16` | KEEP-OPEN | none  | 2+7/0 | 10 | - | transformer/impl/llama/fused_attention.mojo:4749 | owed A/B: the default vs MOJOLEARN_ATTN_DQ_TQ16 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DQ_TQ32` | KEEP-OPEN | none  | 2+7/0 | 10 | - | transformer/impl/llama/fused_attention.mojo:4750 | owed A/B: the default vs MOJOLEARN_ATTN_DQ_TQ32 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_DQ_TQ64` | KEEP-OPEN | none  | 2+7/0 | 10 | - | transformer/impl/llama/fused_attention.mojo:4751 | owed A/B: the default vs MOJOLEARN_ATTN_DQ_TQ64 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_ES_TQ16` | KEEP-OPEN | none  | 2+3/0 | 8 | - | transformer/checks/attention_zdot_routing_check.mojo:16 | owed A/B: the default vs MOJOLEARN_ATTN_ES_TQ16 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_ES_TQ8` | KEEP-OPEN | none  | 2+3/0 | 8 | - | transformer/checks/attention_zdot_routing_check.mojo:19 | owed A/B: the default vs MOJOLEARN_ATTN_ES_TQ8 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_EXACT_TAIL_GUARD` | KEEP-OPEN | none  | 1+5/0 | 14 | - | transformer/impl/llama/fused_attention.mojo:193 | owed A/B: the default vs MOJOLEARN_ATTN_EXACT_TAIL_GUARD on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_FWD_AMMA_P1_BARRIER` | KEEP-OPEN | none  | 2+0/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:9067 | owed A/B: the default vs MOJOLEARN_ATTN_FWD_AMMA_P1_BARRIER on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_FWD_AMMA_P2_BARRIER` | KEEP-OPEN | none  | 1+3/0 | 5 | - | transformer/impl/llama/fused_attention.mojo:9094 | owed A/B: the default vs MOJOLEARN_ATTN_FWD_AMMA_P2_BARRIER on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_FWD_TQ16` | KEEP-OPEN | none  | 2+21/0 | 23 | - | transformer/impl/llama/fused_attention.mojo:642 | owed A/B: the default vs MOJOLEARN_ATTN_FWD_TQ16 on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_FWD_TQ_RULE` | KEEP-OPEN | none  | 1+4/0 | 11 | - | transformer/impl/llama/fused_attention.mojo:651 | owed A/B: the default vs MOJOLEARN_ATTN_FWD_TQ_RULE on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_KV_CORNER_GUARD` | KEEP-OPEN | none  | 1+5/0 | 14 | - | transformer/impl/llama/fused_attention.mojo:193 | owed A/B: the default vs MOJOLEARN_ATTN_KV_CORNER_GUARD on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_LEGACY_CORNER` | KEEP-OPEN | none  | 1+10/0 | 21 | - | transformer/impl/llama/fused_attention.mojo:190 | owed A/B: default vs the old path MOJOLEARN_ATTN_LEGACY_CORNER on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_NO_BWD_CORNER` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:225 | owed A/B: the default vs MOJOLEARN_ATTN_NO_BWD_CORNER on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_NO_MODE_FLUSH` | KEEP-OPEN | none  | 1+5/0 | 6 | - | transformer/impl/llama/fused_attention.mojo:5893 | owed A/B: the default vs MOJOLEARN_ATTN_NO_MODE_FLUSH on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_NO_SCRATCH_CACHE` | KEEP-OPEN | none  | 1+6/0 | 8 | - | transformer/impl/llama/fused_attention.mojo:9625 | owed A/B: the default vs MOJOLEARN_ATTN_NO_SCRATCH_CACHE on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_REPAIR_MASKED_TAIL` | KEEP-OPEN | none  | 1+10/0 | 63 | - | transformer/impl/llama/fused_attention.mojo:189 | owed A/B: the default vs MOJOLEARN_ATTN_REPAIR_MASKED_TAIL on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_STASH_ZDOT_AMMA` | KEEP-OPEN | none  | 1+3/0 | 4 | - | transformer/impl/llama/fused_attention.mojo:9450 | owed A/B: the default vs MOJOLEARN_ATTN_STASH_ZDOT_AMMA on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_STICKY` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/impl/llama/fused_attention.mojo:218 | owed A/B: the default vs MOJOLEARN_ATTN_STICKY on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_TILE_RULE_OFF` | KEEP-OPEN | none  | 2+8/0 | 10 | - | transformer/impl/llama/fused_attention.mojo:653 | owed A/B: default vs the old path MOJOLEARN_ATTN_TILE_RULE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_ZDOT_AMMA_FOLD_BARRIER` | KEEP-OPEN | none  | 1+0/0 | 2 | - | transformer/impl/llama/fused_attention.mojo:9412 | owed A/B: the default vs MOJOLEARN_ATTN_ZDOT_AMMA_FOLD_BARRIER on NV+AMD, rows of its family |
| `MOJOLEARN_ATTN_ZDOT_STASH_FOLD_BARRIER` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/impl/llama/fused_attention.mojo:5634 | owed A/B: the default vs MOJOLEARN_ATTN_ZDOT_STASH_FOLD_BARRIER on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF` | KEEP-OPEN | OPEN EXP:2178 PENDING | 1+3/0 | 4 | - | autograd/ops.mojo:157 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_EMBEDDING_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | embedding/embedding_words_run.mojo:552 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_EMBEDDING_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_FOLD_V2_OFF` | KEEP-OPEN | none  | 1+13/0 | 29 | - | autograd/ops.mojo:271 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_FOLD_V2_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | autograd/data.mojo:36 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1` | KEEP-OPEN | OPEN EXP:2160 PENDING | 1+2/0 | 12 | - | sequence/gemm_tiled.mojo:379 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1 on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | autograd/tape.mojo:75 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_HAS_INFERENCE` | KEEP-OPEN | none  | 1+2/0 | 5 | - | autograd/model.mojo:39 | owed A/B: the default vs MOJOLEARN_AUTOGRAD_HAS_INFERENCE on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF` | KEEP-OPEN | OPEN EXP:2174 PENDING | 1+3/0 | 6 | - | autograd/ops.mojo:275 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF` | KEEP-OPEN | OPEN EXP:2172 PENDING | 1+11/0 | 71 | - | autograd/rules_mamba.mojo:282 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_MLP_UNFUSED` | KEEP-OPEN | none  | 1+3/0 | 4 | c | sequence/mlp_fit.mojo:115 | owed A/B: the default vs MOJOLEARN_AUTOGRAD_MLP_UNFUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_MODEL_UNFUSED` | KEEP-OPEN | none  | 1+4/0 | 15 | - | autograd/model.mojo:59 | owed A/B: the default vs MOJOLEARN_AUTOGRAD_MODEL_UNFUSED on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_OPTIM` | KEEP-OPEN | OPEN EXP:2380 lane/r2-restore-unmeasured-defaults @ e048e42e81 (hand OP_OPT update restored as | 2+12/0 | 30 | icp | sequence/mlp_fit.mojo:150 | owed A/B: the default vs MOJOLEARN_AUTOGRAD_OPTIM on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF` | KEEP-OPEN | OPEN EXP:2169 PENDING | 1+1/0 | 2 | - | autograd/optim.mojo:94 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | autograd/optim.mojo:72 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_PAIR_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | autograd/tape.mojo:94 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_PAIR_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | autograd/ops.mojo:119 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_PASS3_DB_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | autograd/ops.mojo:135 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_PASS3_DB_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF` | KEEP-OPEN | OPEN EXP:2171 PENDING | 2+3/0 | 10 | - | autograd/rules_lm.mojo:151 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF` | KEEP-OPEN | OPEN EXP:2177 PENDING | 1+0/0 | 1 | - | autograd/ops.mojo:156 | owed A/B: default vs the old path MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_BN_FOLD_BLOCK_OFF` | KEEP-OPEN | none  | 2+4/0 | 9 | - | x_cnn/checks/oracle.mojo:111 | owed A/B: default vs the old path MOJOLEARN_BN_FOLD_BLOCK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_ALL` | KEEP-OPEN | OPEN EXP:660 OPEN | 1+5/0 | 6 | - | tokenizer/fast/bpe_device.mojo:52 | owed A/B: the default vs MOJOLEARN_BPE_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_ENCODE_DEVICE` | KEEP-OPEN | OPEN EXP:662 OPEN | 1+1/0 | 2 | - | tokenizer/fast/bpe_device.mojo:60 | owed A/B: the default vs MOJOLEARN_BPE_ENCODE_DEVICE on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_GROUP_FILTER` | KEEP-OPEN | OPEN EXP:663 OPEN | 1+3/0 | 10 | - | tokenizer/fast/bpe_device.mojo:55 | owed A/B: the default vs MOJOLEARN_BPE_GROUP_FILTER on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_LIVEBUF` | KEEP-OPEN | OPEN EXP:664 OPEN | 1+6/0 | 34 | - | tokenizer/fast/bpe_device.mojo:53 | owed A/B: the default vs MOJOLEARN_BPE_LIVEBUF on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_MERGE_BATCH` | KEEP-OPEN | OPEN EXP:665 OPEN | 1+4/0 | 30 | - | tokenizer/fast/bpe_device.mojo:54 | owed A/B: the default vs MOJOLEARN_BPE_MERGE_BATCH on NV+AMD, rows of its family |
| `MOJOLEARN_BPE_TRAIN_DEVICE` | KEEP-OPEN | OPEN EXP:666 OPEN | 1+1/0 | 2 | - | tokenizer/fast/bpe_device.mojo:57 | owed A/B: the default vs MOJOLEARN_BPE_TRAIN_DEVICE on NV+AMD, rows of its family |
| `MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE` | KEEP-OPEN | none  | 1+2/0 | 8 | - | autograd/kernels/transformer.mojo:433 | owed A/B: default vs the old path MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_CE_UNALIASED` | KEEP-OPEN | none  | 1+3/0 | 10 | - | training/byte_lm.mojo:219 | owed A/B: the default vs MOJOLEARN_BYTE_LM_CE_UNALIASED on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_COPY_EMB_HEAD` | KEEP-OPEN | none  | 1+4/0 | 12 | - | training/byte_lm.mojo:894 | owed A/B: the default vs MOJOLEARN_BYTE_LM_COPY_EMB_HEAD on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_GENERIC_OPS` | KEEP-OPEN | OPEN EXP:2375 PENDING (A = restored ops are the default) | 2+5/0 | 7 | c | autograd/block_step_ops.mojo:1854 | owed A/B: the default vs MOJOLEARN_BYTE_LM_GENERIC_OPS on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM` | KEEP-OPEN | none  | 1+7/0 | 8 | - | training/byte_lm_host_kernels.mojo:151 | owed A/B: default vs the old path MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_PROGRAM_STEP` | KEEP-OPEN | OPEN EXP:2342 RESTORED hand step as default; A/B owed | 1+12/0 | 49 | c | training/byte_lm_config.mojo:16 | owed A/B: the default vs MOJOLEARN_BYTE_LM_PROGRAM_STEP on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_RELEASE_EAGER` | KEEP-OPEN | none  | 1+4/0 | 7 | - | training/byte_lm.mojo:216 | owed A/B: the default vs MOJOLEARN_BYTE_LM_RELEASE_EAGER on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_RETAIN_EAGER` | KEEP-OPEN | none  | 1+4/0 | 5 | - | training/byte_lm.mojo:217 | owed A/B: the default vs MOJOLEARN_BYTE_LM_RETAIN_EAGER on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_STICKY_EAGER` | KEEP-OPEN | none  | 1+2/0 | 5 | - | training/byte_lm.mojo:213 | owed A/B: the default vs MOJOLEARN_BYTE_LM_STICKY_EAGER on NV+AMD, rows of its family |
| `MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | training/byte_lm_host_kernels.mojo:128 | owed A/B: default vs the old path MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CNN_GENERIC_OPS` | KEEP-OPEN | OPEN EXP:2374 PENDING (A = restored fused block is the default) | 1+5/0 | 6 | c | autograd/cnn_block_op.mojo:71 | owed A/B: the default vs MOJOLEARN_CNN_GENERIC_OPS on NV+AMD, rows of its family |
| `MOJOLEARN_CNN_PREDICT_PROGRAM_COPY` | KEEP-OPEN | OPEN EXP:2411 PENDING (A = unfilled X/output buffers, cached replay, one-chunk output in place | 1+6/0 | 12 | - | sequence/cnn_classifier_program.mojo:124 | owed A/B: the default vs MOJOLEARN_CNN_PREDICT_PROGRAM_COPY on NV+AMD, rows of its family |
| `MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/impl/llama/modeling_llama.mojo:2281 | owed A/B: the default vs MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM on NV+AMD, rows of its family |
| `MOJOLEARN_DROPOUT2D_CH_MASK_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/ops.mojo:1283 | owed A/B: default vs the old path MOJOLEARN_DROPOUT2D_CH_MASK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS` | KEEP-OPEN | none  | 2+9/0 | 11 | - | autograd/embedding_ops.mojo:72 | owed A/B: the default vs MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS on NV+AMD, rows of its family |
| `MOJOLEARN_EMB_DEVICE_SCRATCH_OFF` | KEEP-OPEN | none  | 1+3/0 | 60 | - | embedding/embedding_words_run.mojo:123 | owed A/B: default vs the old path MOJOLEARN_EMB_DEVICE_SCRATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_EMB_RESIDENT_OFF` | KEEP-OPEN | none  | 1+4/0 | 55 | - | embedding/embedding_words_run.mojo:122 | owed A/B: default vs the old path MOJOLEARN_EMB_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_EMB_SERIAL_RUN_BEGIN` | KEEP-OPEN | none  | 1+0/0 | 10 | - | autograd/kernels/embedding.mojo:810 | owed A/B: the default vs MOJOLEARN_EMB_SERIAL_RUN_BEGIN on NV+AMD, rows of its family |
| `MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL` | KEEP-OPEN | none  | 2+5/0 | 10 | - | autograd/kernels/embedding_sort.mojo:19 | owed A/B: the default vs MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_ETS_TEAM_OFF` | KEEP-OPEN | WIN EXP:535 KEPT aec9f8c33 | 1+4/0 | 5 | - | sequence/ets_team.mojo:55 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_ETS_TEAM_OFF |
| `MOJOLEARN_FUSE_RESIDUAL1_NORM2` | KEEP-OPEN | none  | 2+2/0 | 4 | - | transformer/impl/llama/modeling_llama.mojo:6405 | owed A/B: the default vs MOJOLEARN_FUSE_RESIDUAL1_NORM2 on NV+AMD, rows of its family |
| `MOJOLEARN_GAPLM_ZDOT_KREG` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:8121 | owed A/B: the default vs MOJOLEARN_GAPLM_ZDOT_KREG on NV+AMD, rows of its family |
| `MOJOLEARN_GARCH_COOP_OFF` | KEEP-OPEN | LOSE EXP:564 SUPERSEDED | 1+9/0 | 10 | - | sequence/fit_team.mojo:141 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_GEMM_FAST_RAGGED_K_OFF` | KEEP-OPEN | WIN EXP:2123 KEPT, FAST default (`_OFF`) | 1+3/0 | 6 | - | gemm/afn_apple_fast.mojo:165 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_HOST_RMS_FORCE_SERIAL` | KEEP-OPEN | none  | 1+1/0 | 3 | - | training/host/samba_ops_oracle.mojo:107 | owed A/B: the default vs MOJOLEARN_HOST_RMS_FORCE_SERIAL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ACT_RETAIN` | KEEP-OPEN | LOSE EXP:2365 DELETED (superseded) | 2+4/4 | 6 | c | mamba/impl/mamba_shell_impl.mojo:1087 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_ACT_RETAIN_OFF` | KEEP-OPEN | LOSE EXP:2365 DELETED (superseded) | 2+4/1 | 6 | - | mamba/impl/mamba_shell_impl.mojo:1086 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF` | KEEP-OPEN | none  | 1+3/0 | 6 | - | x_cnn/ops.mojo:87 | owed A/B: default vs the old path MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ADAPT_M_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2160 | owed A/B: default vs the old path MOJOLEARN_IDN_ADAPT_M_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AF_COL_CHUNK_FOLD` | KEEP-OPEN | none  | 1+3/0 | 6 | - | sequence/adafactor.mojo:29 | owed A/B: the default vs MOJOLEARN_IDN_AF_COL_CHUNK_FOLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ALL_OFF` | KEEP-OPEN | LOSE EXP:1732 DROP (the off arm is slower), switch deleted | 403+1335/0 | 3911 | c | arima/impl/batched_kalman.mojo:186 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF` | KEEP-OPEN | none  | 1+6/0 | 25 | - | transformer/impl/llama/fused_attention.mojo:2852 | owed A/B: default vs the old path MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/modeling_llama.mojo:4091 | owed A/B: default vs the old path MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_FOLD_V2` | KEEP-OPEN | none  | 1+1/0 | 2 | - | transformer/impl/llama/attn_fold_v2.mojo:67 | owed A/B: the default vs MOJOLEARN_IDN_ATTN_FOLD_V2 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_HEAD_SHARE` | KEEP-OPEN | LOSE EXP:1483 DROP | 2+2/1 | 5 | c | experiments/performance_ideas/attention_time.mojo:23 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF` | KEEP-OPEN | none  | 1+9/0 | 10 | - | transformer/impl/llama/fused_attention.mojo:9641 | owed A/B: default vs the old path MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF` | KEEP-OPEN | none  | 1+4/0 | 31 | - | transformer/impl/llama/fused_attention.mojo:2345 | owed A/B: default vs the old path MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:9612 | owed A/B: default vs the old path MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ATTN_STASH` | KEEP-OPEN | WIN EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf | 5+4/1 | 9 | - | experiments/performance_ideas/attention_time.mojo:44 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF` | KEEP-OPEN | OPEN EXP:2168 PENDING | 1+2/0 | 19 | - | autograd/row_ops.mojo:58 | owed A/B: default vs the old path MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF` | KEEP-OPEN | none  | 1+2/0 | 15 | - | x_cnn/ops.mojo:81 | owed A/B: default vs the old path MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | autograd/kernels/transformer.mojo:252 | owed A/B: default vs the old path MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CE_DENOM_ROWFOLD` | KEEP-OPEN | none  | 1+5/1 | 14 | - | training/neural_identical_experiments.mojo:52 | owed A/B: the default vs MOJOLEARN_IDN_CE_DENOM_ROWFOLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CE_GRAD_FUSED` | KEEP-OPEN | none  | 1+2/0 | 3 | c | training/neural_identical_experiments.mojo:44 | owed A/B: the default vs MOJOLEARN_IDN_CE_GRAD_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CE_PIPE_UP_OFF` | KEEP-OPEN | none  | 1+2/0 | 4 | - | training/estimator.mojo:1355 | owed A/B: default vs the old path MOJOLEARN_IDN_CE_PIPE_UP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CE_TOKEN_FOLD` | KEEP-OPEN | LOSE EXP:2009 DROPPED (noise), code deleted | 1+2/3 | 3 | icr | training/neural_ab_profile_contract.mojo:13 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2` | KEEP-OPEN | WIN src# # descriptor op 16), the measured winner: IDN_SAMBA_RESIDENT_STEP PROMOTED | 2+17/1 | 26 | - | training/neural_identical_experiments.mojo:66 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_CNN_CONV_RELU` | KEEP-OPEN | none  | 1+3/3 | 4 | c | x_cnn/ops.mojo:45 | owed A/B: the default vs MOJOLEARN_IDN_CNN_CONV_RELU on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF` | KEEP-OPEN | WIN EXP:2084 KEEP (IDENTICAL default; `_OFF` restores) | 1+2/0 | 3 | - | x_cnn/device.mojo:2192 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:277 | owed A/B: default vs the old path MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ELEM_VEC4_OFF` | KEEP-OPEN | OPEN EXP:2175 PENDING | 1+2/0 | 3 | - | autograd/fold_kernels.mojo:71 | owed A/B: default vs the old path MOJOLEARN_IDN_ELEM_VEC4_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | autograd/kernels/embedding_sort.mojo:53 | owed A/B: default vs the old path MOJOLEARN_IDN_EMB_RADIX_SORT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ETS_TEAM_OFF` | KEEP-OPEN | WIN EXP:2087 KEEP (IDENTICAL default; `_OFF` restores) | 1+11/0 | 17 | - | sequence/ets_team.mojo:83 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_FOLD_STAGED_OFF` | KEEP-OPEN | OPEN EXP:2173 PENDING | 1+2/0 | 3 | - | autograd/fold_kernels.mojo:67 | owed A/B: default vs the old path MOJOLEARN_IDN_FOLD_STAGED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF` | KEEP-OPEN | none  | 1+5/0 | 42 | - | x_cnn/ops.mojo:1587 | owed A/B: default vs the old path MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GARCH_COOP_OFF` | KEEP-OPEN | none  | 1+6/0 | 30 | - | sequence/garch.mojo:92 | owed A/B: default vs the old path MOJOLEARN_IDN_GARCH_COOP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GARCH_FOLD32_OFF` | KEEP-OPEN | none  | 1+6/0 | 39 | - | sequence/garch.mojo:76 | owed A/B: default vs the old path MOJOLEARN_IDN_GARCH_FOLD32_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GARCH_LEAD_REG_OFF` | KEEP-OPEN | none  | 1+2/0 | 8 | - | sequence/fit_team.mojo:177 | owed A/B: default vs the old path MOJOLEARN_IDN_GARCH_LEAD_REG_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_cnn/ops.mojo:1833 | owed A/B: default vs the old path MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GEMM_INT_SLICES` | KEEP-OPEN | LOSE EXP:2263 pending | 2+1/0 | 6 | - | gemm/host/linalg_native.mojo:66 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF` | KEEP-OPEN | OPEN EXP:2181 PENDING | 1+3/0 | 14 | - | sequence/gemm_tiled.mojo:486 | owed A/B: default vs the old path MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GEMM_SPLITK_OFF` | KEEP-OPEN | OPEN EXP:2180 PENDING (BIT VERSION) | 1+5/0 | 6 | - | sequence/gemm_split.mojo:46 | owed A/B: default vs the old path MOJOLEARN_IDN_GEMM_SPLITK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GEMM_STAGE64` | KEEP-OPEN | LOSE src# #: recurrent trainers ran 2.7x slower on the AMD MI325X after GEMM_STAGE_V2 | 1+3/0 | 4 | - | sequence/ops.mojo:409 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_GEMM_STAGE_OFF` | KEEP-OPEN | OPEN EXP:2164 PENDING | 1+3/0 | 4 | - | sequence/ops.mojo:410 | owed A/B: default vs the old path MOJOLEARN_IDN_GEMM_STAGE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GRAPH_M_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2162 | owed A/B: default vs the old path MOJOLEARN_IDN_GRAPH_M_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GROUP_M_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2166 | owed A/B: default vs the old path MOJOLEARN_IDN_GROUP_M_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LAYER_DEV_IO_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2169 | owed A/B: default vs the old path MOJOLEARN_IDN_LAYER_DEV_IO_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF` | KEEP-OPEN | none  | 1+2/0 | 26 | - | transformer/impl/llama/modeling_llama.mojo:4077 | owed A/B: default vs the old path MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LM_VIEWS` | KEEP-OPEN | none  | 1+2/4 | 3 | c | training/neural_identical_experiments.mojo:37 | owed A/B: the default vs MOJOLEARN_IDN_LM_VIEWS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LN_ONE_PASS_OFF` | KEEP-OPEN | OPEN EXP:2170 PENDING | 1+3/0 | 4 | - | sequence/layernorm.mojo:299 | owed A/B: default vs the old path MOJOLEARN_IDN_LN_ONE_PASS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_LN_ROW_WARP_OFF` | KEEP-OPEN | WIN EXP:1945 PROMOTED (IDENTICAL default) | 2+9/0 | 30 | - | sequence/checks/oracle.mojo:702 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH` | KEEP-OPEN | none  | 1+5/1 | 6 | - | mamba/impl/ops/m1_persistent_scan.mojo:55 | owed A/B: the default vs MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS` | KEEP-OPEN | none  | 1+3/1 | 4 | - | mamba/impl/ops/m1_persistent_scan.mojo:56 | owed A/B: the default vs MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M1_SCAN` | KEEP-OPEN | LOSE EXP:2071 LOSER (deleted) | 2+5/1 | 7 | ic | mamba/host/gen/selective_scan_interface.mojo:263 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_M2_BWD_CELL_OFF` | KEEP-OPEN | none  | 2+14/0 | 82 | - | autograd/kernels/mamba2_ssd.mojo:69 | owed A/B: default vs the old path MOJOLEARN_IDN_M2_BWD_CELL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M2_CB_LOWER_OFF` | KEEP-OPEN | none  | 3+12/0 | 15 | - | mamba/checks/mamba2_oracle.mojo:103 | owed A/B: default vs the old path MOJOLEARN_IDN_M2_CB_LOWER_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M2_CS_PT` | KEEP-OPEN | none  | 2+4/2 | 6 | c | mamba/host/gen/ssd_minimal.mojo:762 | owed A/B: the default vs MOJOLEARN_IDN_M2_CS_PT on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M2_GRAD_FOLD` | KEEP-OPEN | none  | 3+4/3 | 7 | c | mamba/impl/ops/neural_gradient_profile.mojo:21 | owed A/B: the default vs MOJOLEARN_IDN_M2_GRAD_FOLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M2_RETAIN_GL_OFF` | KEEP-OPEN | WIN EXP:2013 PROMOTED | 2+4/0 | 6 | - | mamba/host/gen/ssd_minimal.mojo:869 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_M2_SSD_TILES_OFF` | KEEP-OPEN | none  | 1+6/0 | 42 | - | mamba/impl/modules/afn_defines.mojo:148 | owed A/B: default vs the old path MOJOLEARN_IDN_M2_SSD_TILES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M2_YD_ROWS` | KEEP-OPEN | WIN EXP:2012 PROMOTED | 2+6/2 | 8 | c | mamba/host/gen/ssd_minimal.mojo:761 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_M2_YOFF_EXP` | KEEP-OPEN | none  | 2+7/3 | 9 | c | mamba/host/gen/ssd_minimal.mojo:1051 | owed A/B: the default vs MOJOLEARN_IDN_M2_YOFF_EXP on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128` | KEEP-OPEN | none  | 2+24/0 | 26 | - | mamba/host/gen/mamba3_siso.mojo:538 | owed A/B: the default vs MOJOLEARN_IDN_M3_ANGLE_BLOCK_128 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32` | KEEP-OPEN | none  | 2+24/0 | 26 | - | mamba/host/gen/mamba3_siso.mojo:537 | owed A/B: the default vs MOJOLEARN_IDN_M3_ANGLE_BLOCK_32 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF` | KEEP-OPEN | WIN EXP:1755 PROMOTED (IDENTICAL default) | 2+6/0 | 18 | - | autograd/kernels/mamba3.mojo:1335 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF` | KEEP-OPEN | none  | 2+8/0 | 10 | - | autograd/kernels/mamba3.mojo:1315 | owed A/B: default vs the old path MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_ANGLE_PARALLEL` | KEEP-OPEN | none  | 3+6/0 | 126 | - | mamba/checks/mamba3_oracle.mojo:185 | owed A/B: the default vs MOJOLEARN_IDN_M3_ANGLE_PARALLEL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF` | KEEP-OPEN | none  | 2+2/0 | 4 | - | autograd/kernels/mamba3_prefill.mojo:36 | owed A/B: default vs the old path MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE` | KEEP-OPEN | none  | 1+4/0 | 16 | c | autograd/block_step_ops.mojo:135 | owed A/B: the default vs MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF` | KEEP-OPEN | WIN EXP:2014 PROMOTED | 2+0/0 | 2 | - | autograd/kernels/mamba3.mojo:2474 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2133 | owed A/B: the default vs MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD` | KEEP-OPEN | OPEN EXP:2418 PENDING (A = no per-call stage allocation, no h_last copy, no host state round t | 1+4/0 | 10 | - | autograd/block_step_ops.mojo:188 | owed A/B: default vs the old path MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA_ARENA_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | mamba/impl/modules/afn_defines.mojo:111 | owed A/B: default vs the old path MOJOLEARN_IDN_MAMBA_ARENA_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC` | KEEP-OPEN | OPEN EXP:2410 PENDING (A = the restored hand chain is the default) | 1+6/0 | 11 | - | autograd/rules_mamba.mojo:229 | owed A/B: the default vs MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA_BWD_FOLDS` | KEEP-OPEN | OPEN EXP:2271 PENDING | 2+4/0 | 14 | - | autograd/kernels/mamba_bwd_folds.mojo:54 | owed A/B: the default vs MOJOLEARN_IDN_MAMBA_BWD_FOLDS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF` | KEEP-OPEN | none  | 1+5/0 | 33 | - | mamba/impl/modules/idn_gemm_ws.mojo:44 | owed A/B: default vs the old path MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MAXIMIZE_DEV_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/estimator.mojo:693 | owed A/B: default vs the old path MOJOLEARN_IDN_MAXIMIZE_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | sequence/mlp.mojo:48 | owed A/B: default vs the old path MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | sequence/mlp.mojo:49 | owed A/B: default vs the old path MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF` | KEEP-OPEN | WIN EXP:2373 KEPT default again (restored from removed) | 1+5/1 | 6 | ic | autograd/small_mlp_op.mojo:91 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF` | KEEP-OPEN | WIN EXP:1944 PROMOTED (IDENTICAL default) | 1+7/0 | 21 | - | sequence/moe_grouped_fold.mojo:40 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_MOE_STABLE_PACK` | KEEP-OPEN | none  | 1+1/0 | 2 | c | sequence/moe_group.mojo:45 | owed A/B: the default vs MOJOLEARN_IDN_MOE_STABLE_PACK on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | KEEP-OPEN | none  | 2+25/3 | 27 | - | gemm/experiments/neural_profile.mojo:32 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_CHAINS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_FILL_BLOCKS` | KEEP-OPEN | none  | 1+4/2 | 5 | - | gemm/neural_dispatch.mojo:55 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_FILL_BLOCKS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_FOLD_SHARED` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gemm/neural_dispatch.mojo:190 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_FOLD_SHARED on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_GEMM_ARM` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gemm/neural_dispatch.mojo:54 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_GEMM_ARM on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL` | KEEP-OPEN | none  | 2+12/0 | 15 | - | gemm/neural_dispatch.mojo:59 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_GEMM_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE` | KEEP-OPEN | LOSE EXP:1767 DELETED (loser; AMD flat, NV 1.9x) | 1+4/2 | 9 | icr | gemm/experiments/neural_tiled_v2.mojo:32 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE` | KEEP-OPEN | LOSE EXP:1746 DROP (slower), arm deleted | 1+2/1 | 3 | ic | gemm/experiments/neural_switches.mojo:58 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES` | KEEP-OPEN | none  | 1+1/1 | 2 | - | gemm/experiments/neural_switches.mojo:65 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_NN07` | KEEP-OPEN | none  | 1+7/0 | 16 | - | gemm/experiments/neural_grouped.mojo:28 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_NN07 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_NN12` | KEEP-OPEN | none  | 1+10/1 | 11 | - | gemm/experiments/neural_plans.mojo:34 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_NN12 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_NN16` | KEEP-OPEN | none  | 1+7/0 | 8 | - | gemm/experiments/neural_streaming.mojo:33 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_NN16 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_PAIR_CONTROL` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gemm/neural_dispatch.mojo:60 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_PAIR_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS` | KEEP-OPEN | none  | 1+3/1 | 4 | - | gemm/neural_dispatch.mojo:56 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_STAGE_DEPTH` | KEEP-OPEN | none  | 1+2/2 | 3 | - | gemm/neural_dispatch.mojo:51 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_STAGE_DEPTH on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_STAGE_PAD` | KEEP-OPEN | none  | 1+2/2 | 3 | - | gemm/neural_dispatch.mojo:52 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_STAGE_PAD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gemm/neural_dispatch.mojo:53 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NEURAL_STREAM_GROUP` | KEEP-OPEN | none  | 1+3/1 | 4 | - | gemm/neural_dispatch.mojo:50 | owed A/B: the default vs MOJOLEARN_IDN_NEURAL_STREAM_GROUP on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NM_SPEC_OFF` | KEEP-OPEN | WIN EXP:2086 KEEP (IDENTICAL default; `_OFF` restores) | 2+6/0 | 8 | - | sequence/ets_spec.mojo:42 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_NORM` | KEEP-OPEN | none  | 10+4/1 | 14 | c | training/byte_lm_host.mojo:836 | owed A/B: the default vs MOJOLEARN_IDN_NORM on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_NORM_FIXED_LANES` | KEEP-OPEN | OPEN EXP:2246 PENDING | 1+16/0 | 49 | - | transformer/experiments/norm_fixed_lanes.mojo:48 | owed A/B: the default vs MOJOLEARN_IDN_NORM_FIXED_LANES on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/opt_gate.mojo:86 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_GATE_SCAN_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | training/opt_gate.mojo:76 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_GATE_SCAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/estimator.mojo:719 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF` | KEEP-OPEN | none  | 1+2/0 | 5 | - | sequence/opt_resident.mojo:75 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF` | KEEP-OPEN | LOSE EXP:1878 DELETED (noise) | 1+4/0 | 7 | - | training/estimator.mojo:419 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | training/opt_gate.mojo:82 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF` | KEEP-OPEN | none  | 1+3/0 | 7 | - | sequence/opt_resident.mojo:78 | owed A/B: default vs the old path MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF` | KEEP-OPEN | none  | 1+2/0 | 23 | - | x_cnn/ops.mojo:84 | owed A/B: default vs the old path MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PAD_M_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2164 | owed A/B: default vs the old path MOJOLEARN_IDN_PAD_M_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PIN_WEIGHTS_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:2173 | owed A/B: default vs the old path MOJOLEARN_IDN_PIN_WEIGHTS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PROPHET_COOP_OFF` | KEEP-OPEN | none  | 1+4/0 | 10 | - | sequence/prophet.mojo:46 | owed A/B: default vs the old path MOJOLEARN_IDN_PROPHET_COOP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ROPE` | KEEP-OPEN | none  | 2+2/1 | 4 | c | training/neural_identical_experiments.mojo:21 | owed A/B: the default vs MOJOLEARN_IDN_ROPE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ROW_COOP_OFF` | KEEP-OPEN | OPEN EXP:2167 PENDING | 1+13/0 | 20 | - | autograd/row_ops.mojo:55 | owed A/B: default vs the old path MOJOLEARN_IDN_ROW_COOP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SAMBA_ATTN_RECOMPUTE_OFF` | KEEP-OPEN | WIN EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf | 1+2/0 | 3 | - | training/neural_identical_experiments.mojo:134 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF` | KEEP-OPEN | none  | 1+5/0 | 12 | - | training/samba_ops.mojo:101 | owed A/B: default vs the old path MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_AF_COOP_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | sequence/exec_device.mojo:111 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_AF_COOP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | sequence/exec_device.mojo:99 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_GEMM_MICRO` | KEEP-OPEN | none  | 1+3/0 | 4 | - | sequence/gemm_tiled.mojo:445 | owed A/B: the default vs MOJOLEARN_IDN_SEQ_GEMM_MICRO on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF` | KEEP-OPEN | OPEN EXP:2179 PENDING | 1+3/0 | 4 | - | sequence/gemm_tiled.mojo:446 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF` | KEEP-OPEN | none  | 1+6/0 | 28 | - | sequence/gemm_tiled.mojo:165 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF` | KEEP-OPEN | none  | 1+10/0 | 11 | - | sequence/gemm_tiled.mojo:146 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF` | KEEP-OPEN | OPEN EXP:2163 PENDING | 1+5/0 | 73 | - | sequence/gemm_tiled.mojo:396 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_LN_COOP_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | sequence/exec_device.mojo:115 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_LN_COOP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_LN_LEAF` | KEEP-OPEN | none  | 1+0/2 | 1 | c | sequence/layernorm.mojo:81 | owed A/B: the default vs MOJOLEARN_IDN_SEQ_LN_LEAF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF` | KEEP-OPEN | none  | 1+6/0 | 25 | - | sequence/exec_device.mojo:152 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_PIPE_UP_OFF` | KEEP-OPEN | none  | 1+2/0 | 20 | - | sequence/exec_device.mojo:149 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_PIPE_UP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF` | KEEP-OPEN | none  | 1+3/0 | 24 | - | sequence/ops.mojo:79 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_TILED_PREDICT` | KEEP-OPEN | none  | 1+3/0 | 4 | - | autograd/model.mojo:82 | owed A/B: the default vs MOJOLEARN_IDN_SEQ_TILED_PREDICT on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_WGRAD` | KEEP-OPEN | none  | 5+3/3 | 8 | c | sequence/checks/oracle.mojo:196 | owed A/B: the default vs MOJOLEARN_IDN_SEQ_WGRAD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF` | KEEP-OPEN | none  | 4+6/0 | 28 | - | sequence/checks/oracle.mojo:192 | owed A/B: default vs the old path MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SSD_OCCUPANCY` | KEEP-OPEN | OPEN EXP:2239 PENDING | 1+4/0 | 18 | - | mamba/impl/modules/ssd_minimal.mojo:809 | owed A/B: the default vs MOJOLEARN_IDN_SSD_OCCUPANCY on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | training/dev_tensors.mojo:49 | owed A/B: default vs the old path MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE` | KEEP-OPEN | none  | 1+2/0 | 3 | c | training/neural_identical_experiments.mojo:25 | owed A/B: the default vs MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRAIN_SCRATCH` | KEEP-OPEN | none  | 2+6/4 | 22 | c | training/neural_ab_lifetime.mojo:16 | owed A/B: the default vs MOJOLEARN_IDN_TRAIN_SCRATCH on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRAIN_SWIGLU` | KEEP-OPEN | none  | 1+1/0 | 2 | c | training/neural_identical_experiments.mojo:31 | owed A/B: the default vs MOJOLEARN_IDN_TRAIN_SWIGLU on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRANSPOSE_TILE_OFF` | KEEP-OPEN | OPEN EXP:2176 PENDING | 1+2/0 | 3 | - | autograd/fold_kernels.mojo:75 | owed A/B: default vs the old path MOJOLEARN_IDN_TRANSPOSE_TILE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/device.mojo:805 | owed A/B: default vs the old path MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF` | KEEP-OPEN | none  | 1+2/0 | 22 | - | x_cnn/ops.mojo:91 | owed A/B: default vs the old path MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XENT_DEV_FOLD_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | x_cnn/ops.mojo:1822 | owed A/B: default vs the old path MOJOLEARN_IDN_XENT_DEV_FOLD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_XENT_FOLD_BLOCK_256` | KEEP-OPEN | none  | 1+4/0 | 6 | - | x_cnn/ops.mojo:1841 | owed A/B: the default vs MOJOLEARN_IDN_XENT_FOLD_BLOCK_256 on NV+AMD, rows of its family |
| `MOJOLEARN_INT_SLICES_FORCE_FLAT` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gemm/int_slices.mojo:96 | owed A/B: the default vs MOJOLEARN_INT_SLICES_FORCE_FLAT on NV+AMD, rows of its family |
| `MOJOLEARN_LN_FOLD_BLOCK_OFF` | KEEP-OPEN | none  | 1+5/0 | 8 | - | sequence/layernorm.mojo:68 | owed A/B: default vs the old path MOJOLEARN_LN_FOLD_BLOCK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE` | KEEP-OPEN | none  | 1+1/0 | 2 | - | autograd/kernels/mamba3.mojo:1215 | owed A/B: the default vs MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2032 | owed A/B: the default vs MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT` | KEEP-OPEN | none  | 3+8/0 | 25 | - | mamba/checks/mamba3_oracle.mojo:186 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/impl/mamba_shell_impl.mojo:145 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER` | KEEP-OPEN | none  | 1+3/0 | 20 | - | mamba/impl/mamba_shell_impl.mojo:119 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2025 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2058 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_QKS` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2097 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_QKS on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_REFUSAL` | KEEP-OPEN | none  | 1+4/0 | 11 | - | autograd/kernels/mamba_refusal.mojo:21 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_REFUSAL on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_STATEPASS` | KEEP-OPEN | none  | 2+2/0 | 37 | - | mamba/host/gen/mamba3_siso.mojo:1125 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_STATEPASS on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_YINTRA` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2139 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_YINTRA on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_LEGACY_YSTATE` | KEEP-OPEN | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3_siso.mojo:2170 | owed A/B: default vs the old path MOJOLEARN_MAMBA3_LEGACY_YSTATE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_S16_QK_NAIVE` | KEEP-OPEN | none  | 1+2/0 | 5 | - | autograd/kernels/mamba3.mojo:291 | owed A/B: the default vs MOJOLEARN_MAMBA3_S16_QK_NAIVE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_S16_V_NAIVE` | KEEP-OPEN | none  | 1+2/0 | 5 | - | autograd/kernels/mamba3.mojo:30 | owed A/B: the default vs MOJOLEARN_MAMBA3_S16_V_NAIVE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE` | KEEP-OPEN | none  | 3+7/0 | 37 | - | autograd/kernels/mamba3.mojo:2207 | owed A/B: the default vs MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_S17_TAIL_NAIVE` | KEEP-OPEN | none  | 1+6/0 | 7 | - | autograd/kernels/mamba3.mojo:39 | owed A/B: the default vs MOJOLEARN_MAMBA3_S17_TAIL_NAIVE on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_TILED_INCREMENT` | KEEP-OPEN | none  | 2+0/0 | 4 | - | mamba/host/gen/mamba3_siso.mojo:2023 | owed A/B: the default vs MOJOLEARN_MAMBA3_TILED_INCREMENT on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA3_TILED_YINTRA` | KEEP-OPEN | none  | 2+0/0 | 4 | - | mamba/host/gen/mamba3_siso.mojo:2137 | owed A/B: the default vs MOJOLEARN_MAMBA3_TILED_YINTRA on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA_2712_UNBOUNDED` | KEEP-OPEN | none  | 2+5/0 | 7 | - | mamba/host/gen/ssd_minimal.mojo:104 | owed A/B: the default vs MOJOLEARN_MAMBA_2712_UNBOUNDED on NV+AMD, rows of its family |
| `MOJOLEARN_MAMBA_CALL_PROGRAM` | KEEP-OPEN | OPEN EXP:2404 PENDING (A = 0.8.55's per-call shape is the default) | 1+3/0 | 4 | - | autograd/block_step_ops.mojo:160 | owed A/B: the default vs MOJOLEARN_MAMBA_CALL_PROGRAM on NV+AMD, rows of its family |
| `MOJOLEARN_MLP_LOSSBLOCK_OFF` | KEEP-OPEN | OPEN EXP:2159 PENDING | 1+6/0 | 7 | - | sequence/mlp.mojo:57 | owed A/B: default vs the old path MOJOLEARN_MLP_LOSSBLOCK_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | sequence/mlp.mojo:66 | owed A/B: default vs the old path MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_MOE_DEVGROUP_OFF` | KEEP-OPEN | WIN EXP:587 KEPT 499e1b72c | 1+3/0 | 4 | - | sequence/moe_reg.mojo:64 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MOE_DEVGROUP_OFF |
| `MOJOLEARN_MOE_FAST_MMA_OFF` | KEEP-OPEN | LOSE EXP:1956 DELETED (slower, bundle) | 1+3/0 | 4 | c | sequence/moe_mma.mojo:53 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_MOE_REGTILE_OFF` | KEEP-OPEN | WIN EXP:588 KEPT 499e1b72c | 1+2/0 | 3 | - | sequence/moe_reg.mojo:62 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MOE_REGTILE_OFF |
| `MOJOLEARN_NI01_TRAINING_WORKSPACE` | KEEP-OPEN | none  | 1+1/1 | 2 | - | training/neural_gemm_workspace.mojo:17 | owed A/B: the default vs MOJOLEARN_NI01_TRAINING_WORKSPACE on NV+AMD, rows of its family |
| `MOJOLEARN_NI07_GROUPED_PROJECTIONS` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gemm/experiments/neural_tiled_v2.mojo:23 | owed A/B: the default vs MOJOLEARN_NI07_GROUPED_PROJECTIONS on NV+AMD, rows of its family |
| `MOJOLEARN_NI10_BUDGETED_CNN_TAPE` | KEEP-OPEN | none  | 1+0/1 | 2 | - | x_cnn/ops.mojo:65 | owed A/B: the default vs MOJOLEARN_NI10_BUDGETED_CNN_TAPE on NV+AMD, rows of its family |
| `MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE` | KEEP-OPEN | none  | 1+0/1 | 2 | - | x_cnn/ops.mojo:67 | owed A/B: the default vs MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE on NV+AMD, rows of its family |
| `MOJOLEARN_NI11_DIRECT_CONV_TAPS64` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:88 | owed A/B: the default vs MOJOLEARN_NI11_DIRECT_CONV_TAPS64 on NV+AMD, rows of its family |
| `MOJOLEARN_NI14_COL2IM` | KEEP-OPEN | OPEN EXP:1615 n/a | 1+3/2 | 4 | c | x_cnn/neural_col2im.mojo:27 | owed A/B: the default vs MOJOLEARN_NI14_COL2IM on NV+AMD, rows of its family |
| `MOJOLEARN_NI15_BIAS_NO_ONES` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/device.mojo:103 | owed A/B: the default vs MOJOLEARN_NI15_BIAS_NO_ONES on NV+AMD, rows of its family |
| `MOJOLEARN_NI17_BALANCED_NORM` | KEEP-OPEN | none  | 1+4/0 | 30 | - | x_cnn/ops.mojo:934 | owed A/B: the default vs MOJOLEARN_NI17_BALANCED_NORM on NV+AMD, rows of its family |
| `MOJOLEARN_NI55_GRAPH_FEATURE4` | KEEP-OPEN | none  | 1+1/1 | 2 | - | x_cnn/neural_aux_contract.mojo:17 | owed A/B: the default vs MOJOLEARN_NI55_GRAPH_FEATURE4 on NV+AMD, rows of its family |
| `MOJOLEARN_NI57_GRAPH_TREE64` | KEEP-OPEN | none  | 1+3/0 | 8 | - | x_cnn/neural_aux_contract.mojo:18 | owed A/B: the default vs MOJOLEARN_NI57_GRAPH_TREE64 on NV+AMD, rows of its family |
| `MOJOLEARN_NI58_SAGE_FEATURE4` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/neural_aux_contract.mojo:19 | owed A/B: the default vs MOJOLEARN_NI58_SAGE_FEATURE4 on NV+AMD, rows of its family |
| `MOJOLEARN_NI59_DROPOUT_CHANNEL` | KEEP-OPEN | none  | 1+1/1 | 2 | - | x_cnn/neural_aux_contract.mojo:20 | owed A/B: the default vs MOJOLEARN_NI59_DROPOUT_CHANNEL on NV+AMD, rows of its family |
| `MOJOLEARN_NI60_DROPOUT_APPLY4` | KEEP-OPEN | none  | 1+1/1 | 2 | - | x_cnn/neural_aux_contract.mojo:21 | owed A/B: the default vs MOJOLEARN_NI60_DROPOUT_APPLY4 on NV+AMD, rows of its family |
| `MOJOLEARN_NN14_BOUNDED_IM2COL` | KEEP-OPEN | WIN EXP:1943 PROMOTED (IDENTICAL default) | 1+19/1 | 77 | - | x_cnn/ops.mojo:1961 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_NN18_DENSE_TILE_CONTROL` | KEEP-OPEN | none  | 1+2/0 | 4 | - | transformer/impl/llama/fused_attention.mojo:2198 | owed A/B: the default vs MOJOLEARN_NN18_DENSE_TILE_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_NN21_SCALE_MASK` | KEEP-OPEN | none  | 1+1/0 | 2 | - | transformer/experiments/attention_schedules.mojo:195 | owed A/B: the default vs MOJOLEARN_NN21_SCALE_MASK on NV+AMD, rows of its family |
| `MOJOLEARN_NN29_RING_PAIR` | KEEP-OPEN | none  | 1+1/0 | 2 | - | transformer/experiments/attention_schedules.mojo:29 | owed A/B: the default vs MOJOLEARN_NN29_RING_PAIR on NV+AMD, rows of its family |
| `MOJOLEARN_NN35_CONV_PAIR` | KEEP-OPEN | none  | 4+8/0 | 66 | - | mamba/host/gen/mamba2.mojo:637 | owed A/B: the default vs MOJOLEARN_NN35_CONV_PAIR on NV+AMD, rows of its family |
| `MOJOLEARN_NN37_TRIANGLE_TASKS` | KEEP-OPEN | none  | 2+4/0 | 31 | - | mamba/host/gen/ssd_minimal.mojo:379 | owed A/B: the default vs MOJOLEARN_NN37_TRIANGLE_TASKS on NV+AMD, rows of its family |
| `MOJOLEARN_NN40_OWNED_WEIGHTS` | KEEP-OPEN | none  | 2+5/0 | 12 | - | mamba/host/mamba_host_shell_impl.mojo:312 | owed A/B: the default vs MOJOLEARN_NN40_OWNED_WEIGHTS on NV+AMD, rows of its family |
| `MOJOLEARN_NN40_WS_HEADROOM` | KEEP-OPEN | none  | 1+2/0 | 4 | - | mamba/impl/modules/idn_gemm_ws.mojo:54 | owed A/B: the default vs MOJOLEARN_NN40_WS_HEADROOM on NV+AMD, rows of its family |
| `MOJOLEARN_NN42_INPUT_BIAS_FUSED` | KEEP-OPEN | none  | 1+2/0 | 5 | - | sequence/ops.mojo:45 | owed A/B: the default vs MOJOLEARN_NN42_INPUT_BIAS_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_NN44_EXPERT_BISECT` | KEEP-OPEN | none  | 1+2/0 | 12 | - | sequence/moe_tiled.mojo:36 | owed A/B: the default vs MOJOLEARN_NN44_EXPERT_BISECT on NV+AMD, rows of its family |
| `MOJOLEARN_NN46_GATHER_BOUNDS` | KEEP-OPEN | none  | 1+4/0 | 24 | - | x_cnn/ops.mojo:47 | owed A/B: the default vs MOJOLEARN_NN46_GATHER_BOUNDS on NV+AMD, rows of its family |
| `MOJOLEARN_NN47_APPLY_RUNNING` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_cnn/ops.mojo:48 | owed A/B: the default vs MOJOLEARN_NN47_APPLY_RUNNING on NV+AMD, rows of its family |
| `MOJOLEARN_NN48_CSR_TILES` | KEEP-OPEN | none  | 1+2/1 | 6 | - | x_cnn/device.mojo:1906 | owed A/B: the default vs MOJOLEARN_NN48_CSR_TILES on NV+AMD, rows of its family |
| `MOJOLEARN_NN50_OWNED_RUNS` | KEEP-OPEN | none  | 1+3/0 | 14 | - | autograd/kernels/embedding_runs.mojo:23 | owed A/B: the default vs MOJOLEARN_NN50_OWNED_RUNS on NV+AMD, rows of its family |
| `MOJOLEARN_NN53_HEAD_CHUNK2048` | KEEP-OPEN | none  | 1+2/1 | 3 | - | training/chunked_lm_head_profile.mojo:17 | owed A/B: the default vs MOJOLEARN_NN53_HEAD_CHUNK2048 on NV+AMD, rows of its family |
| `MOJOLEARN_NN53_HEAD_CHUNK512` | KEEP-OPEN | none  | 1+2/2 | 3 | - | training/chunked_lm_head_profile.mojo:12 | owed A/B: the default vs MOJOLEARN_NN53_HEAD_CHUNK512 on NV+AMD, rows of its family |
| `MOJOLEARN_NN58_ACCUMULATE_STATUS` | KEEP-OPEN | none  | 2+9/0 | 42 | - | training/host/samba_ops_oracle.mojo:440 | owed A/B: the default vs MOJOLEARN_NN58_ACCUMULATE_STATUS on NV+AMD, rows of its family |
| `MOJOLEARN_NN59_CONTROL` | KEEP-OPEN | none  | 1+2/0 | 4 | - | training/residual_dropout_contract.mojo:17 | owed A/B: the default vs MOJOLEARN_NN59_CONTROL on NV+AMD, rows of its family |
| `MOJOLEARN_NN59_DROPOUT_RESIDUAL` | KEEP-OPEN | none  | 1+2/0 | 3 | - | training/residual_dropout_contract.mojo:15 | owed A/B: the default vs MOJOLEARN_NN59_DROPOUT_RESIDUAL on NV+AMD, rows of its family |
| `MOJOLEARN_NN63_CANONICAL_SHARD_MERGE` | KEEP-OPEN | none  | 1+5/0 | 45 | - | training/neural_ab_shards.mojo:18 | owed A/B: the default vs MOJOLEARN_NN63_CANONICAL_SHARD_MERGE on NV+AMD, rows of its family |
| `MOJOLEARN_OPT_FAST_STREAM_OFF` | KEEP-OPEN | LOSE EXP:1204 DROPPED-slower (alone) | 1+4/0 | 12 | - | sequence/opt_resident.mojo:141 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_OPT_PIPE_DOWN_OFF` | KEEP-OPEN | WIN EXP:589 KEPT 5990c5946 | 1+2/0 | 3 | - | sequence/opt_resident.mojo:74 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_OPT_PIPE_DOWN_OFF |
| `MOJOLEARN_OPT_RAW_UP_OFF` | KEEP-OPEN | LOSE EXP:1204 DROPPED-slower (alone) | 1+3/0 | 11 | - | sequence/opt_resident.mojo:61 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_OPT_TRUST_INPUTS` | KEEP-OPEN | none  | 3+0/0 | 5 | - | training/checks/optimizer.mojo:1705 | owed A/B: the default vs MOJOLEARN_OPT_TRUST_INPUTS on NV+AMD, rows of its family |
| `MOJOLEARN_OPT_ZERO_OPEN_OFF` | KEEP-OPEN | WIN EXP:590 KEPT 5990c5946 | 1+3/0 | 4 | - | sequence/opt_resident.mojo:77 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_OPT_ZERO_OPEN_OFF |
| `MOJOLEARN_PROPHET_COOP_OFF` | KEEP-OPEN | WIN EXP:537 KEPT 8c2b7de51 | 1+4/0 | 7 | - | sequence/prophet_coop.mojo:58 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_PROPHET_COOP_OFF |
| `MOJOLEARN_SEQ_CROSTON_REG_OFF` | KEEP-OPEN | WIN EXP:539 KEPT 2dcdd949f | 1+2/0 | 3 | - | sequence/croston.mojo:38 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SEQ_CROSTON_REG_OFF |
| `MOJOLEARN_SEQ_FAST_FMA_OFF` | KEEP-OPEN | WIN EXP:542 KEPT f6a9e7c04 | 1+2/0 | 3 | - | sequence/ops.mojo:67 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_SEQ_FAST_VAR_ONECOPY_OFF` | KEEP-OPEN | LOSE EXP:558 DROPPED-noise | 1+2/0 | 3 | - | sequence/ops.mojo:105 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_SEQ_GARCH_GRID_OFF` | KEEP-OPEN | WIN EXP:546 KEPT 2dcdd949f | 1+4/0 | 5 | - | sequence/garch.mojo:61 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SEQ_GARCH_GRID_OFF |
| `MOJOLEARN_SEQ_GARCH_REG_OFF` | KEEP-OPEN | WIN EXP:547 KEPT 2dcdd949f | 1+3/0 | 4 | - | sequence/garch.mojo:42 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SEQ_GARCH_REG_OFF |
| `MOJOLEARN_SEQ_REC_STAGE64` | KEEP-OPEN | none  | 1+4/0 | 5 | - | sequence/ops.mojo:427 | owed A/B: the default vs MOJOLEARN_SEQ_REC_STAGE64 on NV+AMD, rows of its family |
| `MOJOLEARN_SEQ_RNN_SCAN` | KEEP-OPEN | LOSE EXP:2166 TIE (chain stays the default for G >= 2) | 1+2/0 | 4 | - | sequence/recurrent_scan.mojo:125 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_SEQ_RNN_SCAN_APPLE` | KEEP-OPEN | WIN EXP:2165 KEPT (scan default for G = 1) | 1+4/0 | 5 | - | sequence/recurrent_scan.mojo:103 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_SEQ_RNN_SCAN_OFF` | KEEP-OPEN | LOSE EXP:2166 TIE (chain stays the default for G >= 2) | 1+3/0 | 4 | - | sequence/recurrent_scan.mojo:130 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_SEQ_THETA_REG_OFF` | KEEP-OPEN | WIN EXP:548 KEPT d1fa9223b | 1+2/0 | 3 | - | sequence/theta.mojo:59 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SEQ_THETA_REG_OFF |
| `MOJOLEARN_SMALL_MLP_GENERIC_OPS` | KEEP-OPEN | OPEN EXP:2372 PENDING (A = restored tuned op is the default) | 1+2/0 | 13 | ic | autograd/small_mlp_program.mojo:108 | owed A/B: the default vs MOJOLEARN_SMALL_MLP_GENERIC_OPS on NV+AMD, rows of its family |
| `MOJOLEARN_TRAIN_OPT_FAST_PIPE_DOWN_OFF` | KEEP-OPEN | LOSE EXP:1209 NEUTRAL | 1+4/0 | 8 | - | training/estimator.mojo:745 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_TRANSFORMER_GENERIC_PREFILL` | KEEP-OPEN | OPEN EXP:2376 PENDING (A = 0.8.66's session fresh prefill is the default) | 1+0/0 | 1 | c | autograd/block_step_ops.mojo:1743 | owed A/B: the default vs MOJOLEARN_TRANSFORMER_GENERIC_PREFILL on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER` | KEEP-OPEN | none  | 2+0/0 | 2 | - | transformer/transformer_binding_impl.mojo:161 | owed A/B: default vs the old path MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER on NV+AMD, rows of its family |
| `MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY` | KEEP-OPEN | none  | 1+0/0 | 1 | - | transformer/transformer_binding_impl.mojo:260 | owed A/B: default vs the old path MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY on NV+AMD, rows of its family |
| `MOJOLEARN_TSA2_STL_OFF` | KEEP-OPEN | WIN EXP:549 KEPT 550806bc0 | 1+2/0 | 3 | - | sequence/ops.mojo:95 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_TSA2_STL_OFF |
| `MOJOLEARN_TSA2_VAR_OFF` | KEEP-OPEN | WIN EXP:550 KEPT 550806bc0 | 1+7/0 | 8 | - | sequence/ops.mojo:94 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_TSA2_VAR_OFF |
| `MOJOLEARN_XCNN_NO_BN_BLOCK` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:1506 | owed A/B: the default vs MOJOLEARN_XCNN_NO_BN_BLOCK on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_DIRECT_CONV` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:282 | owed A/B: the default vs MOJOLEARN_XCNN_NO_DIRECT_CONV on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_IM2COL_TAPS` | KEEP-OPEN | none  | 1+1/0 | 4 | - | x_cnn/device.mojo:244 | owed A/B: the default vs MOJOLEARN_XCNN_NO_IM2COL_TAPS on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_MMA_SPLIT` | KEEP-OPEN | none  | 2+0/0 | 2 | - | x_cnn/device.mojo:533 | owed A/B: the default vs MOJOLEARN_XCNN_NO_MMA_SPLIT on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_NT_TUNE` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:119 | owed A/B: the default vs MOJOLEARN_XCNN_NO_NT_TUNE on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_POOL_BWD_TILE` | KEEP-OPEN | none  | 1+0/0 | 1 | - | x_cnn/ops.mojo:418 | owed A/B: the default vs MOJOLEARN_XCNN_NO_POOL_BWD_TILE on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_RES_POOL` | KEEP-OPEN | none  | 1+3/0 | 4 | - | x_cnn/device.mojo:871 | owed A/B: the default vs MOJOLEARN_XCNN_NO_RES_POOL on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_NO_TILED_LAYOUT` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_cnn/device.mojo:208 | owed A/B: the default vs MOJOLEARN_XCNN_NO_TILED_LAYOUT on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_ONES_CACHE_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/device.mojo:787 | owed A/B: default vs the old path MOJOLEARN_XCNN_ONES_CACHE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF` | KEEP-OPEN | OPEN EXP:1792 owed (nv + amd RACE) | 1+3/0 | 5 | - | x_cnn/device.mojo:895 | owed A/B: default vs the old path MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF` | KEEP-OPEN | OPEN EXP:1791 owed (nv + amd RACE batchnorm2d,dropout2d) | 1+2/0 | 3 | - | x_cnn/device.mojo:886 | owed A/B: default vs the old path MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_XCNN_TILED_ROWS` | KEEP-OPEN | none  | 1+1/0 | 2 | - | x_cnn/device.mojo:213 | owed A/B: the default vs MOJOLEARN_XCNN_TILED_ROWS on NV+AMD, rows of its family |
| `MOJOLEARN_ACCUMULATE_POOL_FAULT` | KEEP | none  | 1+2/0 | 5 | - | training/accumulate_multi_gpu.mojo:20 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_AFN26_EMB_GATHER8` | KEEP | none  | 1+2/0 | 11 | - | embedding/checks/embedding_fast_apple.mojo:43 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFN26_EMB_RESIDENT` | KEEP | none  | 1+2/0 | 3 | - | embedding/checks/embedding_fast_apple.mojo:52 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFN26_EMB_SCRATCH` | KEEP | none  | 1+1/0 | 2 | - | embedding/checks/embedding_fast_apple.mojo:53 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFN26_EMB_THREADS128` | KEEP | none  | 1+2/1 | 3 | - | embedding/checks/embedding_fast_apple.mojo:47 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFN26_EMB_THREADS64` | KEEP | none  | 1+2/1 | 3 | - | embedding/checks/embedding_fast_apple.mojo:46 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_AFN_EMB_ATOMIC_BWD` | KEEP | OPEN EXP:604 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+6/0 | 55 | - | embedding/checks/embedding_fast_apple.mojo:40 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_GEMM_NARROW_MMA_OFF` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:4531 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_GEMM_SKINNY_OFF` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:4540 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_LEAF_PREFETCH_OFF` | KEEP | none  | 1+3/0 | 11 | - | gemm/checks/gemm_identical.mojo:542 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_ADMIT_174` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:1449 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_DB` | KEEP | none  | 1+4/0 | 28 | - | gemm/checks/gemm_identical.mojo:5063 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_FM` | KEEP | none  | 1+5/0 | 6 | - | gemm/checks/gemm_identical.mojo:5057 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_FN` | KEEP | none  | 1+5/0 | 6 | - | gemm/checks/gemm_identical.mojo:5058 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_GROUP_M` | KEEP | none  | 1+4/0 | 5 | - | gemm/checks/gemm_identical.mojo:5060 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_KB` | KEEP | none  | 1+9/0 | 10 | - | gemm/checks/gemm_identical.mojo:5059 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_KB_WIDE` | KEEP | none  | 1+5/0 | 6 | - | gemm/checks/gemm_identical.mojo:5066 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SGM` | KEEP | none  | 1+13/0 | 14 | - | gemm/checks/gemm_identical.mojo:5055 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SGN` | KEEP | none  | 1+13/0 | 14 | - | gemm/checks/gemm_identical.mojo:5056 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SMALL_BLOCKS` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:5093 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SMALL_FM` | KEEP | none  | 1+6/0 | 7 | - | gemm/checks/gemm_identical.mojo:5091 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SMALL_FN` | KEEP | none  | 1+6/0 | 7 | - | gemm/checks/gemm_identical.mojo:5092 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SMALL_KB` | KEEP | none  | 1+4/0 | 5 | - | gemm/checks/gemm_identical.mojo:5105 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SMALL_OFF` | KEEP | none  | 1+1/0 | 2 | - | gemm/checks/gemm_identical.mojo:5090 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SPLIT_BLOCKS` | KEEP | none  | 2+10/0 | 12 | - | gemm/checks/gemm_identical.mojo:5073 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_APPLE_MMA_SPLIT_MIN_LEAVES` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:5074 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ATTN_AMMA_PROBE_P1` | KEEP | none  | 1+0/0 | 4 | - | transformer/impl/llama/fused_attention.mojo:9081 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_AMMA_PROBE_P12` | KEEP | none  | 1+0/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:9127 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_APPLE_ERECOMP` | KEEP | none  | 1+6/0 | 51 | - | transformer/impl/llama/fused_attention.mojo:1611 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ATTN_APPLE_ESTASH_FORCE_DENY` | KEEP | none  | 1+1/0 | 2 | - | transformer/impl/llama/fused_attention.mojo:1634 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ATTN_NO_APPLE_MMA` | KEEP | none  | 1+4/0 | 5 | - | transformer/impl/llama/fused_attention.mojo:8880 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ATTN_OPERAND_DUMP` | KEEP | none  | 1+3/0 | 12 | - | transformer/impl/llama/fused_attention.mojo:2013 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_PHASE_TIMERS` | KEEP | none  | 1+1/0 | 3 | - | transformer/impl/llama/fused_attention.mojo:389 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_REPAIR_SAB_DQ` | KEEP | none  | 1+3/0 | 4 | - | transformer/impl/llama/fused_attention.mojo:192 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_REPAIR_SAB_Z` | KEEP | none  | 1+4/0 | 5 | - | transformer/impl/llama/fused_attention.mojo:191 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_SABOTAGE_AMMA` | KEEP | none  | 4+0/0 | 11 | - | transformer/impl/llama/fused_attention.mojo:6273 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ATTN_TAIL_GUARD_SABOTAGE` | KEEP | none  | 1+3/0 | 7 | - | transformer/impl/llama/fused_attention.mojo:194 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_AUTOGRAD_PROBE_CPU` | KEEP | none  | 1+0/0 | 4 | - | autograd/checks/gpu_arguments.mojo:45 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BPE_TRAINER_SABOTAGE` | KEEP | none  | 1+1/0 | 3 | - | tokenizer/train/bpe_train.mojo:69 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BYTE_LM_FAULT_INJECT` | KEEP | none  | 1+4/0 | 8 | - | training/byte_lm.mojo:192 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BYTE_LM_HOST_SABOTAGE` | KEEP | none  | 3+11/0 | 16 | c | training/byte_lm_host.mojo:109 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE` | KEEP | none  | 2+0/0 | 7 | - | training/byte_lm_logits.mojo:303 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BYTE_LM_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 16 | - | training/byte_lm_parallel.mojo:424 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BYTE_POOL_FAULT_INJECT` | KEEP | none  | 1+2/0 | 6 | - | training/byte_lm_optimizer_pool.mojo:18 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHECK_CORRUPT_DQ_PRESERVE` | KEEP | none  | 1+0/0 | 2 | - | transformer/checks/attention_masked_tail_check.mojo:139 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CHECK_CORRUPT_Z_PRESERVE` | KEEP | none  | 1+0/0 | 2 | - | transformer/checks/attention_masked_tail_check.mojo:81 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CLASS_PROBE_SABOTAGE` | KEEP | none  | 1+0/0 | 3 | - | gemm/checks/gemm_seam_probe.mojo:126 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_CLIP_POOL_FAULT` | KEEP | none  | 1+1/0 | 4 | - | training/clip_multi_gpu.mojo:30 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_EMB_AUTO_SORT_OFF` | KEEP | none  | 1+2/0 | 5 | - | embedding/checks/embedding_identical.mojo:187 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_EMB_SABOTAGE_SORT_KEY_ID_ONLY_UNSTABLE` | KEEP | none  | 1+2/0 | 5 | - | autograd/kernels/embedding_sort.mojo:27 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_EXECUTOR_TIMERS` | KEEP | LOSE EXP:2274 REFACTORED (2026-11-08) | 1+4/0 | 24 | c | autograd/program_loop.mojo:73 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_ADMIT_NEVER` | KEEP | none  | 1+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:2588 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_AMD_K768_PLAN64` | KEEP | none  | 2+0/0 | 5 | - | gemm/checks/gemm_identical.mojo:7835 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_AMD_NO_K768_TUNED` | KEEP | none  | 2+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:7833 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_ARM_TRIAL` | KEEP | none  | 1+13/1 | 195 | - | gemm/checks/gemm_identical.mojo:6162 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_DETECT_ALWAYS_EXACT` | KEEP | none  | 1+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:1531 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_FOLD_SPECIALIZE_TRIAL` | KEEP | none  | 1+0/0 | 1 | - | gemm/checks/gemm_identical.mojo:8982 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_HOST_ROWS_SABOTAGE` | KEEP | none  | 2+1/0 | 5 | - | gemm/checks/gemm_host_rows_check.mojo:200 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_INLINE_ZERO_FMA_REPAIR` | KEEP | none  | 1+7/0 | 8 | - | gemm/checks/gemm_identical.mojo:1398 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_KPACK_CPT4` | KEEP | LOSE grid neural-small.json removed | 1+29/0 | 30 | c | gemm/checks/gemm_identical.mojo:8574 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LB512` | KEEP | none  | 1+11/0 | 12 | - | gemm/checks/gemm_identical.mojo:1705 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LB768` | KEEP | none  | 1+11/0 | 12 | - | gemm/checks/gemm_identical.mojo:1706 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LEAF_SPLIT_KPACK_BODY` | KEEP | none  | 1+0/0 | 7 | - | gemm/checks/gemm_identical.mojo:7819 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LEGACY_CLASS_FLUSH` | KEEP | none  | 1+3/0 | 4 | - | gemm/checks/gemm_identical.mojo:1339 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_LEGACY_STAGE_FTZ` | KEEP | none  | 1+11/0 | 22 | - | gemm/checks/gemm_identical.mojo:1324 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_MFMA_GMIN1` | KEEP | none  | 1+7/0 | 8 | - | gemm/checks/gemm_identical.mojo:7192 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_MFMA_GMIN16` | KEEP | none  | 1+7/0 | 8 | - | gemm/checks/gemm_identical.mojo:7190 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_MFMA_GMIN2` | KEEP | none  | 1+7/0 | 8 | - | gemm/checks/gemm_identical.mojo:7191 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_MFMA_NO_ADMIT` | KEEP | none  | 1+7/0 | 8 | - | gemm/checks/gemm_identical.mojo:2881 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_MFMA_NO_GROUPS` | KEEP | none  | 2+0/0 | 4 | - | gemm/checks/gemm_identical.mojo:3294 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_APPLE_MMA` | KEEP | none  | 2+13/0 | 15 | - | gemm/checks/gemm_identical.mojo:5052 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_APPLE_MMA_FAST` | KEEP | none  | 1+5/0 | 6 | - | gemm/checks/gemm_identical.mojo:5121 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NO_LAUNCH_BOUND` | KEEP | none  | 1+11/0 | 12 | - | gemm/checks/gemm_identical.mojo:1704 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_NV_FS4_OFF` | KEEP | none  | 1+0/0 | 1 | - | gemm/checks/gemm_identical.mojo:7876 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_PREFETCH_EARLY` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:1711 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS` | KEEP | none  | 2+5/0 | 7 | - | gemm/checks/gemm_identical.mojo:1437 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_APPLE_MMA` | KEEP | none  | 1+0/0 | 4 | - | gemm/checks/gemm_identical.mojo:5455 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_DETECT_NEVER` | KEEP | none  | 1+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:1533 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL` | KEEP | none  | 1+13/0 | 47 | - | gemm/checks/gemm_identical.mojo:238 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_FOLD_STRIDE` | KEEP | none  | 2+9/0 | 45 | - | gemm/checks/gemm_identical.mojo:229 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_GATHER_FTZ` | KEEP | none  | 1+0/0 | 1 | - | gemm/checks/gemm_identical.mojo:8793 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_GATHER_FULL` | KEEP | none  | 1+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:8781 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE` | KEEP | none  | 2+8/0 | 15 | - | gemm/checks/gemm_identical.mojo:255 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_NODE_ORDER` | KEEP | none  | 1+5/0 | 14 | - | gemm/checks/gemm_identical.mojo:244 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_TUNED_STAGE_FTZ` | KEEP | none  | 2+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:2044 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_WINDOW_ADMIT_STEP` | KEEP | none  | 1+0/0 | 2 | - | gemm/checks/gemm_identical.mojo:1480 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_SABOTAGE_WS_FOLD` | KEEP | none  | 1+0/0 | 1 | - | gemm/checks/gemm_identical.mojo:9505 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GEMM_STAGE_FTZ` | KEEP | none  | 1+11/0 | 21 | - | gemm/checks/gemm_identical.mojo:1325 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_STEP_T3` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_step_arms.mojo:33 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_GEMM_UNPINNED_ARM` | KEEP | none  | 2+3/0 | 5 | - | gemm/checks/gemm_unpinned.mojo:229 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_BN_GRID_STATS_OFF` | KEEP | WIN EXP:2085 KEEP (IDENTICAL default; `_OFF` restores) | 1+1/0 | 2 | - | x_cnn/ops.mojo:1113 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IDN_EMB_TOUCHED_ROWS_OFF` | KEEP | none  | 1+2/0 | 3 | - | embedding/checks/embedding_identical.mojo:277 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_EMB_UNIQUE_ROWS_OFF` | KEEP | OPEN EXP:1655 PENDING A/B | 1+1/0 | 2 | - | embedding/checks/embedding_identical.mojo:315 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_AMD_BAND_MFMA_OFF` | KEEP | none  | 1+3/0 | 4 | - | gemm/checks/gemm_identical.mojo:3272 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_GROUP_S_HALF` | KEEP | none  | 1+12/0 | 13 | - | gemm/checks/gemm_identical.mojo:7169 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_GROUP_S_X2` | KEEP | none  | 1+12/0 | 13 | - | gemm/checks/gemm_identical.mojo:7171 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_LDS_PAD_WORDS` | KEEP | none  | 1+21/0 | 22 | - | gemm/checks/gemm_identical.mojo:8599 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_MFMA16_OFF` | KEEP | none  | 1+3/0 | 4 | - | gemm/checks/gemm_identical.mojo:3382 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_MFMA_NO_LONE_GROUP_OFF` | KEEP | none  | 1+3/0 | 10 | - | gemm/checks/gemm_identical.mojo:3253 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_MFMA_REUSE_WS_OFF` | KEEP | none  | 1+4/0 | 10 | - | gemm/checks/gemm_identical.mojo:3237 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_NV_STEP_KPACK_OFF` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:4200 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_SPLIT_MIN_LEAVES` | KEEP | none  | 1+3/0 | 4 | - | gemm/checks/gemm_identical.mojo:4626 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_GEMM_TILE_MIN_BLOCKS` | KEEP | none  | 1+5/1 | 6 | - | gemm/checks/gemm_identical.mojo:4415 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_IDN_MAMBA_RMS_TREE_OFF` | KEEP | none  | 1+5/0 | 18 | - | mamba/checks/mamba_rms_fold.mojo:34 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT15_APPLE_CHUNK_SABOTAGE` | KEEP | none  | 1+3/0 | 8 | - | gemm/checks/gemm_int15_apple.mojo:130 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE` | KEEP | none  | 1+3/0 | 7 | - | gemm/checks/gemm_int15_apple_tuned.mojo:183 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT15_EXPONENT_SABOTAGE` | KEEP | none  | 1+8/0 | 20 | - | gemm/checks/gemm_int15_epilogue.mojo:45 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT15_FORCE_FLAT` | KEEP | none  | 1+3/0 | 5 | - | gemm/checks/gemm_int15.mojo:168 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT15_NO_APPLE_UNIT` | KEEP | none  | 1+5/0 | 12 | - | gemm/checks/gemm_int15.mojo:190 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT15_ONE_LAUNCH` | KEEP | none  | 1+2/0 | 5 | - | gemm/checks/gemm_int15.mojo:179 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT15_PIECE_SABOTAGE` | KEEP | none  | 1+5/0 | 16 | - | gemm/checks/gemm_int15.mojo:156 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT15_QUANT_SABOTAGE` | KEEP | none  | 1+3/0 | 11 | - | gemm/checks/gemm_int15.mojo:159 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_AMD_SABOTAGE` | KEEP | none  | 1+2/0 | 7 | - | gemm/checks/gemm_int8_mma_amd.mojo:136 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_AMD_TRACE` | KEEP | none  | 1+1/0 | 3 | - | gemm/checks/gemm_int8_mma_amd_check.mojo:144 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_APPLE_CHUNK_SABOTAGE` | KEEP | none  | 1+2/0 | 4 | - | gemm/checks/gemm_int8_apple_chunk.mojo:98 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_FORCE_FLAT` | KEEP | none  | 1+2/0 | 4 | - | gemm/checks/gemm_lowbit.mojo:112 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT8_MMA_REFERENCE` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_int8_mma.mojo:145 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT8_MMA_UNSTATED_LOADS` | KEEP | none  | 1+2/0 | 5 | - | gemm/checks/gemm_int8_mma.mojo:108 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT8_PIECES_SABOTAGE` | KEEP | none  | 1+4/0 | 21 | - | gemm/checks/gemm_int8_mma_tuned.mojo:153 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_TUNED_SABOTAGE` | KEEP | none  | 1+6/0 | 31 | - | gemm/checks/gemm_int8_mma_tuned.mojo:142 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_INT8_TUNED_UNSTATED` | KEEP | none  | 1+1/0 | 4 | - | gemm/checks/gemm_int8_mma_tuned.mojo:150 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_INT_SLICES_SABOTAGE` | KEEP | none  | 1+3/0 | 6 | - | gemm/int_slices.mojo:92 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_LEGACY_SHAPE_AMD_K768` | KEEP | none  | 1+0/0 | 4 | - | gemm/checks/gemm_identical.mojo:8715 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_LEGACY_SHAPE_APPLE_PLAN64` | KEEP | none  | 2+0/0 | 18 | - | gemm/checks/gemm_identical.mojo:4676 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_LEGACY_SHAPE_NV_FS4` | KEEP | none  | 1+0/0 | 7 | - | gemm/checks/gemm_identical.mojo:7868 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_LN_FAST_BLOCK_FOLD` | KEEP | none  | 1+5/0 | 8 | - | sequence/layernorm.mojo:68 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_LOWBIT_CONVERT_SABOTAGE` | KEEP | none  | 2+6/0 | 14 | - | gemm/host/gemm_int15_oracle.mojo:85 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_LOWBIT_SABOTAGE` | KEEP | none  | 9+20/0 | 57 | - | gemm/checks/gemm_int15.mojo:153 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA2_SABOTAGE_PAIR_DT_B` | KEEP | none  | 2+12/0 | 60 | - | mamba/host/gen/ssd_minimal.mojo:117 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA3_LEGACY_TRACE_SLICES` | KEEP | none  | 2+0/0 | 2 | - | mamba/host/gen/mamba3.mojo:1377 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA3_PHASE_TIMERS` | KEEP | none  | 7+0/0 | 20 | - | mamba/host/gen/mamba3.mojo:1442 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_AT_END` | KEEP | none  | 1+2/0 | 3 | - | mamba/checks/mamba3_oracle.mojo:188 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_PER_CHUNK` | KEEP | none  | 1+2/0 | 3 | - | mamba/checks/mamba3_oracle.mojo:187 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_POISON` | KEEP | none  | 3+9/0 | 27 | - | mamba/host/gen/modeling_mamba.mojo:561 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_POISON_NOBAND` | KEEP | none  | 2+22/0 | 24 | - | mamba/host/gen/modeling_mamba.mojo:565 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_POISON_OVERREAD` | KEEP | none  | 2+0/0 | 2 | - | mamba/host/gen/modeling_mamba.mojo:562 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_SABOTAGE_S1_FOLD_DESCENDING` | KEEP | none  | 1+5/0 | 18 | - | mamba/checks/mamba_rms_fold.mojo:36 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_SABOTAGE_S5_EXP2` | KEEP | none  | 2+8/0 | 18 | - | mamba/host/gen/selective_scan_interface.mojo:218 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED` | KEEP | none  | 2+6/0 | 16 | - | mamba/host/gen/selective_scan_interface.mojo:199 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_NI01_GEMM_GEOMETRIC_WORKSPACE` | KEEP | none  | 1+3/0 | 16 | - | gemm/checks/gemm_identical.mojo:302 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_NI06_GEMM_ONE_PAGE` | KEEP | none  | 1+2/0 | 3 | - | gemm/checks/gemm_identical.mojo:306 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_NN49_EMB_VECTOR4` | KEEP | none  | 1+1/0 | 2 | - | embedding/checks/embedding_identical.mojo:286 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_NN57_NORM_PROFILE` | KEEP | none  | 1+12/0 | 27 | - | training/neural_ab_profile_contract.mojo:21 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_RECORD` | KEEP | none  | 1+9/0 | 20 | - | training/checks/optimizer.mojo:430 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_OPT_SABOTAGE_FTZ_LATE` | KEEP | none  | 1+4/0 | 20 | - | training/checks/optimizer.mojo:277 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_MHAT_FORM` | KEEP | none  | 1+5/0 | 21 | - | training/checks/optimizer.mojo:258 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_MOMENT_LERP` | KEEP | none  | 1+3/0 | 9 | - | training/checks/optimizer.mojo:248 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_POW_EXPLOG` | KEEP | none  | 1+3/0 | 5 | - | training/checks/optimizer.mojo:327 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_POW_RUNNING` | KEEP | none  | 1+4/0 | 13 | - | training/checks/optimizer.mojo:322 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_RECIP_MUL` | KEEP | none  | 1+3/0 | 10 | - | training/checks/optimizer.mojo:229 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_RSQRT` | KEEP | none  | 1+3/0 | 15 | - | training/checks/optimizer.mojo:225 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_OPT_SABOTAGE_SQ_ASSOC` | KEEP | none  | 1+3/0 | 9 | - | training/checks/optimizer.mojo:251 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_QUANT_PAR_SABOTAGE` | KEEP | none  | 2+3/0 | 9 | - | gemm/checks/gemm_int8_mma_tuned.mojo:157 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_SEQ_FAST_PIPE_CH` | KEEP | OPEN EXP:680 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | 1+10/0 | 11 | - | sequence/exec_device.mojo:156 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_STEP_PHASE_TIMERS` | KEEP | none  | 1+23/10 | 81 | c | core/step_phase.mojo:44 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_TOKENIZER_BATCH_SABOTAGE` | KEEP | none  | 1+1/0 | 13 | - | tokenizer/impl/tokenizer_host_shell_impl.mojo:40 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_TOKENIZER_HOST_SABOTAGE` | KEEP | none  | 1+3/0 | 13 | - | tokenizer/impl/tokenizer_host_shell_impl.mojo:39 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_TRANSFORMER_HOST_STAGED_ATTENTION` | KEEP | none  | 1+2/0 | 3 | - | transformer/checks/transformer_oracle.mojo:1690 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_TUNED_NO_DEV` | KEEP | none  | 1+3/0 | 6 | - | gemm/checks/gemm_int15_apple_tuned_dev.mojo:84 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_XCNN_NO_FAST_TUNE` | KEEP | none  | 1+3/0 | 4 | - | x_cnn/device.mojo:124 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_AFN26_MAMBA1_CHUNKS16` | REMOVED | none  | 0+0/1 | 0 | - | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_AFN26_MAMBA1_CHUNKS64` | REMOVED | none  | 0+0/1 | 0 | - | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_AFN26_MAMBA3_THREADS256` | REMOVED | none  | 0+0/1 | 0 | - | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_AFN26_MAMBA3_THREADS64` | REMOVED | none  | 0+0/1 | 0 | - | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_AFN26_OPT_FUSE_SCAN` | REMOVED | LOSE EXP:1876 DELETED (noise) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFN26_OPT_RESIDENT_STATE` | REMOVED | LOSE EXP:1878 DELETED (noise) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFN26_OPT_VEC4` | REMOVED | LOSE EXP:1877 DELETED (noise) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_ATTN_ARENA` | REMOVED | LOSE EXP:2368 DELETED (dead) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_MLP_ALL` | REMOVED | LOSE EXP:2337 DELETED (superseded, unmeasured) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_MLP_FUSED_STEP` | REMOVED | LOSE EXP:2334 DELETED (superseded, unmeasured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_MLP_MULTISTEP` | REMOVED | LOSE EXP:2336 DELETED (superseded, unmeasured) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_MLP_RESIDENT` | REMOVED | LOSE EXP:2335 DELETED (superseded, unmeasured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_OPT_FUSE_SCAN` | REMOVED | LOSE EXP:1876 DELETED (noise) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_OPT_RESIDENT_STATE` | REMOVED | LOSE EXP:1878 DELETED (noise) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AFN_OPT_VEC4` | REMOVED | LOSE EXP:1877 DELETED (noise) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_ATTN_V1_PACKED_ESTASH` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_CNN` | REMOVED | LOSE EXP:2198 DELETED (removed) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_CNN_OFF` | REMOVED | LOSE EXP:2198 DELETED (removed) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_DATA` | REMOVED | WIN EXP:2208 PROMOTED (same OP_MLP_PERM / OP_SEQ_IOTA launches and key; inline launches delet | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_LM` | REMOVED | LOSE EXP:2192 REMOVED | 0+0/2 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_LM_OFF` | REMOVED | LOSE EXP:2192 REMOVED | 0+0/2 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_LOSS` | REMOVED | WIN EXP:2206 rnn_fit: PROMOTED (blocked fold replaces the serial one-thread OP_SUM; new bits  | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_MAMBA` | REMOVED | LOSE EXP:2190 **removed** | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_MAMBA_OFF` | REMOVED | OPEN EXP:2410 PENDING (A = the restored hand chain is the default) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_MLP_OFF` | REMOVED | LOSE EXP:2187 DELETED-superseded | 0+0/2 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE` | REMOVED | LOSE EXP:2155 DROPPED-noise | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_MOE_OFF` | REMOVED | LOSE EXP:2327 DELETED (removed) | 0+0/2 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_PREDICT` | REMOVED | LOSE EXP:2205 DELETED-folded-in (tape is the one predict path) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_RNN` | REMOVED | LOSE EXP:2188 removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_RNN_OFF` | REMOVED | LOSE EXP:2188 removed | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_SAMBA` | REMOVED | LOSE EXP:2195 **DELETED** | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_SAMBA_OFF` | REMOVED | LOSE EXP:2195 **DELETED** | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_SMALL_MLP` | REMOVED | LOSE EXP:2199 removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF` | REMOVED | LOSE EXP:2199 removed | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_TRANSFORMER` | REMOVED | LOSE EXP:2200 DELETED (removed) | 0+0/2 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_AUTOGRAD_TRANSFORMER_OFF` | REMOVED | LOSE EXP:2200 DELETED (removed) | 0+0/2 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_C58_FORECAST4` | REMOVED | LOSE EXP:1741 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_AF_VEC_FUSED` | REMOVED | LOSE EXP:1835 DROPPED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ATTENTION_V2` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ATTN_DEFER_REGIME` | REMOVED | LOSE EXP:2116 LOSER (LOSER, deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE` | REMOVED | LOSE EXP:1483 DROP | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ATTN_SOFTMAX` | REMOVED | LOSE EXP:2008 DROPPED (slower), code deleted | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ATTN_V2_TQ` | REMOVED | LOSE EXP:2008 DROPPED (slower), code deleted | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_BN_GRID_STATS` | REMOVED | WIN EXP:2085 KEEP (IDENTICAL default; `_OFF` restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CNN_BN_FUSE` | REMOVED | LOSE EXP:1947 DROPPED-noise, DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CNN_DX_IMPLICIT` | REMOVED | LOSE EXP:2088 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF` | REMOVED | LOSE EXP:2347 DELETED (route removed) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CNN_RESIDENT_GRADS` | REMOVED | WIN EXP:2084 KEEP (IDENTICAL default; `_OFF` restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CNN_WGRAD_LEAF1024` | REMOVED | LOSE EXP:2089 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_EMB_SCAN_DEFERRED` | REMOVED | LOSE EXP:1948 DROPPED-slower, DELETED (the _OFF arm won) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_EMB_SCAN_DEFERRED_OFF` | REMOVED | LOSE EXP:1948 DROPPED-slower, DELETED (the _OFF arm won) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GEMM_FILL_SPLITK` | REMOVED | LOSE EXP:2113 LOSER (NOISE/LOSER, deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_GPOOL_WARP` | REMOVED | LOSE EXP:2090 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_GROUPED_ADAM` | REMOVED | LOSE EXP:2069 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_LEAN_STEP` | REMOVED | LOSE grid lm-lean-step.json removed | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_LOGITS_LEAN` | REMOVED | LOSE EXP:2115 LOSER (NOISE, deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_OWNED_TOKENS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_PARAM_VIEWS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LM_RESIDENT_TOKENS` | REMOVED | LOSE EXP:2344 DELETED (no reader since lane x7; never promoted) | 0+0/4 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LN_ROW_WARP` | REMOVED | WIN EXP:1945 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN` | REMOVED | LOSE grid neural-fusions.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M1_STATE_WINDOW` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M2_GRAD_LEAF128` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M2_YOFF_EXP_CACHE` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE` | REMOVED | WIN EXP:1755 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_M3_S17_TAIL_DBUF` | REMOVED | LOSE EXP:2070 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MAMBA1_ELEMENTWISE_FUSED` | REMOVED | LOSE EXP:2017 DROPPED-slower | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MOE_GROUPED_GEMM` | REMOVED | WIN EXP:1944 PROMOTED (IDENTICAL default) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_MOE_RESIDENT` | REMOVED | LOSE EXP:2091 LOSER (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES` | REMOVED | LOSE EXP:2019 DROPPED-slower | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_LEAF` | REMOVED | LOSE EXP:1482 DROP | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN01` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN02` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN03` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN04` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN05` | REMOVED | LOSE EXP:1758 DELETED (loser) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN06` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN08` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN09` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN10` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN11` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NEURAL_NN15` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NM_SPEC` | REMOVED | WIN EXP:2086 KEEP (IDENTICAL default; `_OFF` restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NN20_SPLIT_KV` | REMOVED | LOSE EXP:1745 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES` | REMOVED | LOSE grid neural-fusions.json removed | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PARAM_GRAD_VIEWS` | REMOVED | LOSE EXP:2343 DELETED (its step deleted; default off, never promoted) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_RMS_ROW_BLOCK` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ROPE_CACHE` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SAMBA_FORWARD_TAPE` | REMOVED | LOSE EXP:2365 DELETED (superseded) | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SAMBA_RESIDENT_STEP` | REMOVED | OPEN EXP:2399 PENDING | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SEQ_FMA` | REMOVED | LOSE grid seq-reg.json removed | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SEQ_LN_LEAF32` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN` | REMOVED | LOSE EXP:1491 DROPPED-rule | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_SEQ_WGRAD_LEAF256` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TF_SESSION_LEAN` | REMOVED | LOSE EXP:2117 LOSER (LOSER, deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_TRAIN_ONE_WAIT` | REMOVED | LOSE EXP:2339 DELETED (its step deleted; default off, never promoted) | 0+0/1 | 0 | ic | - | only the refusal guard remains (guard is a compatibility assert) |
| `MOJOLEARN_MOE_FAST_MMA_KB32` | REMOVED | LOSE EXP:1882 DELETED (slower, bundle) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MOE_FAST_MMA_PF` | REMOVED | LOSE EXP:1956 DELETED (slower, bundle) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MOE_FAST_MMA_WIDE` | REMOVED | LOSE EXP:1955 DELETED (slower, bundle) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_NI02_GEMM_STREAM_PARTIALS` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NI08_GEMM_LEAF_256` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NI09_TILED_BIAS` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NI12_IMPLICIT_CONV` | REMOVED | WIN EXP:1943 PROMOTED (IDENTICAL default) | 0+0/2 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_NI16_CONV_RELU_FUSED` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NI18_FUSED_EPOCH_GATHER` | REMOVED | LOSE EXP:2346 DELETED (route removed) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_NN17_GQA_FOUR_HEADS` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN20_BALANCED_SUMMARY_TREE` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN22_EAGER_DKDV_PAIR` | REMOVED | LOSE EXP:1484 DROP (unmeasured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_NN23_ROWDOT_DS` | REMOVED | LOSE EXP:1485 DROP (unmeasured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_NN24_NORM_LANES8` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN25_RMS_SPLIT_SCALE` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN26_TRAIN_SWIGLU` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN27_QK_ROPE_PAIR` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN28_DEAD_TRAINING_CACHE` | REMOVED | LOSE grid neural-gemm-attn-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN31_BOUNDED_CHECKPOINTS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN32_RETAIN_FORWARD` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN33_CSTATE_P16` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN33_YDIAG_ROWS8` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN34_AFFINE_PREFIX` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN36_SHARED_DECAY` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN39_M2_GRAD_TREE` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN43_WGRAD_FIXED128` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN44_STABLE_GROUP` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN45_CONV_RELU` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN52_CE_WEIGHT_GRAD` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN54_LOSS_PROFILE` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN55_BLOCK_STATUS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN56_GROUPED_ADAM` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN60_BLOCK_VIEWS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN60_EMB_HEAD_VIEWS` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN62_LIFETIME_ARENA` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_NN64_SESSION_PACK` | REMOVED | LOSE grid neural-seq-train-dedupe.json removed | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_OPT_FAST_MAP_DOWN` | REMOVED | LOSE EXP:1883 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_OPT_FAST_PIPE_CH` | REMOVED | LOSE EXP:1885 DELETED (batch DROPPED-slower; arm untried) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_OPT_FAST_RAW_DOWN` | REMOVED | LOSE EXP:1884 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SCHED_FAST_TABLE` | REMOVED | LOSE EXP:2227 DELETED (dead code) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SCHED_FAST_TABLE_OFF` | REMOVED | LOSE EXP:2227 DELETED (dead code) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_LSTM_SCAN` | REMOVED | LOSE EXP:1891 DELETED (quality, broken) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM` | REMOVED | LOSE EXP:1892 DELETED (quality, broken bundle) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE` | REMOVED | none  | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_LSTM_WGRAD` | REMOVED | LOSE EXP:1893 DELETED (quality, broken bundle) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_MAP_DOWN` | REMOVED | LOSE EXP:1894 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_RAW_DOWN` | REMOVED | LOSE EXP:1895 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SEQ_FAST_VAR_NODRAIN` | REMOVED | LOSE EXP:1896 DELETED (slower) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |

### series (93)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_ARIMA_FAST_EVAL_WS_OFF` | DELETE-OLD-PATH | WIN EXP:533 KEPT 6c54b7e87 | 1+4/0 | 6 | - | arima/impl/batched_kalman.mojo:290 | expdel-02 |
| `MOJOLEARN_ARIMA_FAST_LLONLY_OFF` | DELETE-OLD-PATH | WIN EXP:534 KEPT 6c54b7e87 | 1+6/0 | 8 | - | arima/impl/batched_kalman.mojo:268 | expdel-02 |
| `MOJOLEARN_CV_FAST_SLICE_OFF` | DELETE-OLD-PATH | WIN EXP:1269 ACCEPT/DEFAULT: FAST+Apple default | 1+2/0 | 3 | - | resample/estimator.mojo:242 | expdel-02 |
| `MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF` | DELETE-OLD-PATH | WIN EXP:1270 ACCEPT/DEFAULT: FAST+Apple default | 1+2/0 | 3 | - | resample/estimator.mojo:258 | expdel-02 |
| `MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF` | DELETE-OLD-PATH | WIN EXP:523 KEPT (FAST+Apple default, `_OFF`) | 1+0/0 | 1 | - | resample/estimator.mojo:2202 | expdel-02 |
| `MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_OFF` | DELETE-OLD-PATH | WIN EXP:395 KEEP (FAST+Apple default, `_OFF`) on lane/apple-fast-recovery | 1+3/0 | 4 | - | resample/estimator.mojo:220 | expdel-02 |
| `MOJOLEARN_TSA_FAST_SELD_FUSED_OFF` | DELETE-OLD-PATH | WIN EXP:552 KEPT 6c0ba4379 | 1+1/0 | 2 | - | tsa/impl/timeSeries/kpss_fused.mojo:71 | expdel-02 |
| `MOJOLEARN_AFCL_P08` | KEEP-OPEN | none  | 1+5/0 | 21 | - | x_metrics/reg_epi.mojo:54 | owed A/B: the default vs MOJOLEARN_AFCL_P08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P09` | KEEP-OPEN | none  | 1+2/0 | 3 | - | x_metrics/plan.mojo:35 | owed A/B: the default vs MOJOLEARN_AFCL_P09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | resample/gather_fast.mojo:31 | owed A/B: the default vs MOJOLEARN_AFCL_P10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P11` | KEEP-OPEN | none  | 1+3/0 | 16 | - | tsa/impl/timeSeries/stationarity.mojo:110 | owed A/B: the default vs MOJOLEARN_AFCL_P11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P12` | KEEP-OPEN | none  | 1+2/0 | 3 | - | arima/impl/batched_kalman.mojo:165 | owed A/B: the default vs MOJOLEARN_AFCL_P12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_P14` | KEEP-OPEN | none  | 1+2/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:159 | owed A/B: the default vs MOJOLEARN_AFCL_P14 on NV+AMD, rows of its family |
| `MOJOLEARN_ARIMA_ASYNC_ALL` | KEEP-OPEN | OPEN EXP:2379 lane/r2-restore-unmeasured-defaults @ 1155f6ab56 (arima/impl/batched_fit.mojo `_ | 1+2/0 | 3 | c | arima/impl/fast_lbfgs_async.mojo:92 | owed A/B: the default vs MOJOLEARN_ARIMA_ASYNC_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_ARIMA_FAST_SEARCH_REUSE` | KEEP-OPEN | WIN EXP:1420 KEEP (FAST+Apple default, `_OFF`) | 1+4/0 | 16 | - | arima/impl/fast_order_search.mojo:111 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF` | KEEP-OPEN | WIN EXP:746 KEEP: default/OFF promotion 311d5233e; taxi RMSE 74.6591 remains worse than oppo | 1+3/0 | 5 | - | arima/impl/fast_eval_ws.mojo:75 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF |
| `MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | arima/impl/batched_arima.mojo:737 | owed A/B: default vs the old path MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ARIMA_ORDER_BATCH_OFF` | KEEP-OPEN | WIN EXP:850 KEEP, default merged on main dc2285bc0; manager default/OFF builds both rc0. Sou | 2+10/0 | 15 | - | arima/impl/fast_order_state.mojo:46 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_ARIMA_ORDER_BATCH_OFF |
| `MOJOLEARN_C58_SERIES4` | KEEP-OPEN | none  | 1+3/0 | 4 | - | experiments/classical_identical_ideas/stats_controls.mojo:36 | owed A/B: the default vs MOJOLEARN_C58_SERIES4 on NV+AMD, rows of its family |
| `MOJOLEARN_C59_TRIAL_STATE` | KEEP-OPEN | none  | 1+4/0 | 7 | - | experiments/classical_identical_ideas/stats_controls.mojo:44 | owed A/B: the default vs MOJOLEARN_C59_TRIAL_STATE on NV+AMD, rows of its family |
| `MOJOLEARN_C60_LAG4` | KEEP-OPEN | none  | 1+5/0 | 35 | - | experiments/classical_identical_ideas/stats_controls.mojo:46 | owed A/B: the default vs MOJOLEARN_C60_LAG4 on NV+AMD, rows of its family |
| `MOJOLEARN_HW_EST_BLOCK64` | KEEP-OPEN | none  | 1+4/0 | 5 | - | holtwinters/impl/internal/hw_utils.mojo:253 | owed A/B: the default vs MOJOLEARN_HW_EST_BLOCK64 on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | arima/impl/batched_kalman.mojo:208 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF` | KEEP-OPEN | none  | 2+5/0 | 8 | - | arima/impl/batched_kalman.mojo:196 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF` | KEEP-OPEN | none  | 1+3/0 | 12 | - | arima/impl/fast_eval_ws.mojo:78 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF` | KEEP-OPEN | none  | 2+5/0 | 7 | - | arima/impl/fast_order_search.mojo:82 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_LLONLY_OFF` | KEEP-OPEN | none  | 2+4/0 | 7 | - | arima/impl/batched_kalman.mojo:190 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_LLONLY_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF` | KEEP-OPEN | none  | 2+10/0 | 27 | - | arima/impl/fast_order_state.mojo:49 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF` | KEEP-OPEN | none  | 1+6/0 | 7 | - | arima/impl/fast_order_search.mojo:76 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF` | KEEP-OPEN | none  | 2+8/0 | 12 | - | arima/impl/fast_order_search.mojo:77 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | arima/impl/fast_order_search.mojo:79 | owed A/B: default vs the old path MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_BOOT_SELECT_OFF` | KEEP-OPEN | none  | 1+2/0 | 17 | - | resample/order_select.mojo:45 | owed A/B: default vs the old path MOJOLEARN_IDN_BOOT_SELECT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GEMM_LEAF` | KEEP-OPEN | WIN EXP:1766 PROMOTED for the MLP step only (IDENTICAL default on mlp-train-step); every othe | 1+23/3 | 24 | c | gemm/contract.mojo:132 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_KALMAN_RD_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | arima/impl/batched_kalman.mojo:186 | owed A/B: default vs the old path MOJOLEARN_IDN_KALMAN_RD_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF` | KEEP-OPEN | none  | 1+3/0 | 13 | - | tsa/impl/select_d_fast.mojo:66 | owed A/B: default vs the old path MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_KPSS_SCAN_OFF` | KEEP-OPEN | none  | 1+2/0 | 10 | - | tsa/impl/timeSeries/stationarity.mojo:309 | owed A/B: default vs the old path MOJOLEARN_IDN_KPSS_SCAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_METRIC_EPI_OFF` | KEEP-OPEN | none  | 1+7/0 | 34 | - | metrics/impl/stats/detail/sf_epilogue_core.mojo:52 | owed A/B: default vs the old path MOJOLEARN_IDN_METRIC_EPI_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SELECT_D_OFF` | KEEP-OPEN | none  | 1+2/0 | 18 | - | tsa/impl/select_d_fast.mojo:56 | owed A/B: default vs the old path MOJOLEARN_IDN_SELECT_D_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SIL_NO_WAIT_OFF` | KEEP-OPEN | none  | 1+3/0 | 18 | - | metrics/impl/stats/detail/batched/silhouette_score.mojo:103 | owed A/B: default vs the old path MOJOLEARN_IDN_SIL_NO_WAIT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRUST_DEV_OFF` | KEEP-OPEN | none  | 1+4/0 | 25 | - | metrics/impl/stats/detail/trustworthiness_score.mojo:97 | owed A/B: default vs the old path MOJOLEARN_IDN_TRUST_DEV_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_TRUST_NO_STAGE_OFF` | KEEP-OPEN | none  | 1+3/0 | 10 | - | metrics/impl/stats/detail/trustworthiness_score.mojo:111 | owed A/B: default vs the old path MOJOLEARN_IDN_TRUST_NO_STAGE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_X0_PAR_LS` | KEEP-OPEN | none  | 1+2/0 | 29 | - | arima/impl/idn_ls_math.mojo:38 | owed A/B: the default vs MOJOLEARN_IDN_X0_PAR_LS on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_X0_PAR_LS_MIN256` | KEEP-OPEN | none  | 1+3/0 | 4 | - | arima/impl/idn_ls_math.mojo:41 | owed A/B: the default vs MOJOLEARN_IDN_X0_PAR_LS_MIN256 on NV+AMD, rows of its family |
| `MOJOLEARN_KALMAN_TIME_SCAN` | KEEP-OPEN | none  | 1+5/0 | 38 | - | arima/impl/batched_kalman.mojo:227 | owed A/B: the default vs MOJOLEARN_KALMAN_TIME_SCAN on NV+AMD, rows of its family |
| `MOJOLEARN_KALMAN_TIME_SCAN_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | arima/impl/batched_kalman.mojo:228 | owed A/B: default vs the old path MOJOLEARN_KALMAN_TIME_SCAN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_METRICS_JOINT_COUNTS` | KEEP-OPEN | none  | 1+0/0 | 1 | - | metrics/impl/classification_joint.mojo:50 | owed A/B: the default vs MOJOLEARN_METRICS_JOINT_COUNTS on NV+AMD, rows of its family |
| `MOJOLEARN_PERM_COUNT_RANK` | KEEP-OPEN | none  | 1+1/0 | 2 | - | resample/estimator.mojo:624 | owed A/B: the default vs MOJOLEARN_PERM_COUNT_RANK on NV+AMD, rows of its family |
| `MOJOLEARN_RESAMPLE_FAST_GATHER_OFF` | KEEP-OPEN | LOSE EXP:1131 DROPPED-slower (reconciled 2026-10-05: matched timing r2-20261004: taxi 68.49 -> | 1+0/0 | 1 | - | resample/estimator.mojo:2337 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES` | KEEP-OPEN | OPEN EXP:1404 READY-AB | 1+3/0 | 4 | - | resample/estimator.mojo:2546 | owed A/B: the default vs MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES on NV+AMD, rows of its family |
| `MOJOLEARN_SELECT_D_OFF` | KEEP-OPEN | WIN EXP:552 KEPT 6c0ba4379 | 1+3/0 | 4 | - | tsa/impl/select_d_fast.mojo:43 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TSA_FAST_KPSS_PACK_OFF` | KEEP-OPEN | LOSE EXP:1848 DELETED | 1+1/0 | 2 | - | tsa/impl/timeSeries/kpss_fused.mojo:61 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_ARIMA_EXOG_SABOTAGE` | KEEP | none  | 1+4/0 | 7 | - | arima/host/arima_oracle.mojo:147 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF` | KEEP | none  | 1+3/0 | 4 | - | arima/impl/batched_arima.mojo:732 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ARIMA_FAST_CONST_BOTH` | KEEP | OPEN EXP:555 READY-AB, merged unmeasured | 1+3/0 | 5 | - | arima/impl/fast_arima_quality.mojo:71 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ARIMA_FAST_KPSS_D` | KEEP | OPEN EXP:557 READY-AB, merged unmeasured | 1+6/0 | 26 | - | arima/impl/fast_arima_quality.mojo:88 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ARIMA_FAST_ROOT_CHECK` | KEEP | OPEN EXP:556 READY-AB, merged unmeasured | 1+6/0 | 11 | - | arima/impl/fast_arima_quality.mojo:80 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ARIMA_PREDICT_SABOTAGE` | KEEP | none  | 1+5/0 | 9 | - | arima/host/arima_oracle.mojo:142 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_CLAMP_GE` | KEEP | none  | 1+2/0 | 10 | - | holtwinters/impl/internal/hw_utils.mojo:91 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_CRIT_ORDER` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:109 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_HW_MAX_CLAMP` | KEEP | none  | 1+2/0 | 8 | - | holtwinters/impl/internal/hw_utils.mojo:85 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_LS_LAST` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:116 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_LS_TIE` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:102 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_NO_FTZ` | KEEP | none  | 1+2/0 | 5 | - | holtwinters/impl/internal/hw_utils.mojo:72 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_NO_ZERO_DIR_GUARD` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:75 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_PACK_FORECAST_SWAP` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/checks/hw_check.mojo:1910 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_PACK_SERIES_MAJOR` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/checks/hw_check.mojo:1903 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_PACK_STRIDE_BATCH` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/checks/hw_check.mojo:1904 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_PACK_SWAP_FLAGS` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/checks/hw_check.mojo:1909 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_PACK_SWAP_SSE_ALPHA` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/checks/hw_check.mojo:1905 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_ROTATE_CONV` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:61 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_STD_SQRT` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:70 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_HW_SABOTAGE_SWAP_FMA` | KEEP | none  | 1+1/0 | 3 | - | holtwinters/impl/internal/hw_utils.mojo:95 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_KALMAN_FAST_RD_OFF` | KEEP | none  | 1+0/0 | 1 | - | arima/impl/batched_kalman.mojo:222 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE` | KEEP | none  | 1+2/0 | 3 | - | resample/estimator.mojo:2407 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW` | KEEP | OPEN EXP:1404 READY-AB | 1+0/0 | 1 | - | resample/estimator.mojo:2545 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER_OFF` | KEEP | none  | 1+0/0 | 1 | - | resample/estimator.mojo:2374 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR_OFF` | KEEP | none  | 1+0/0 | 1 | - | resample/estimator.mojo:2358 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 3 | - | resample/multi_gpu.mojo:59 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_X0_FAST_LS_OFF` | KEEP | none  | 1+2/0 | 3 | - | arima/impl/estimate_x0.mojo:104 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_ARIMA_FAST_ASYNC_OFF` | REMOVED | LOSE EXP:2326 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_CSS_SEARCH` | REMOVED | LOSE EXP:1844 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_D_CONCURRENT` | REMOVED | LOSE EXP:1847 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT` | REMOVED | LOSE EXP:1846 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_K3_DF_PROBE` | REMOVED | LOSE EXP:2292 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL` | REMOVED | LOSE EXP:2292 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_ARIMA_FAST_STEPWISE` | REMOVED | LOSE EXP:1845 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_C58_SHARED_PREP` | REMOVED | LOSE EXP:1624 DROP (noise), code deleted | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_C60_DIFF_REUSE` | REMOVED | LOSE EXP:1555 DROPPED, code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ARIMA_ASYNC_OFF` | REMOVED | LOSE EXP:2326 DELETED (superseded) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ETS_TEAM` | REMOVED | WIN EXP:2087 KEEP (IDENTICAL default; `_OFF` restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_RESAMPLE_FAST_ONE_FOLD` | REMOVED | LOSE EXP:1867 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_RESAMPLE_FAST_RANK_SORT` | REMOVED | LOSE EXP:1866 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_TSA2_KPSS` | REMOVED | LOSE EXP:1848 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |

### trees (354)

| define | decision | verdict + evidence | sites | lines | tomb | first site | reason / lane |
|---|---|---|---|---|---|---|---|
| `MOJOLEARN_CTR_ONEHOT_DEVICE` | KEEP-OPEN | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+4/0 | 5 | - | gbdt/ctrs/fast_prep.mojo:161 | expdel-05: KEEP-OPEN (2026-11-15): default ON through SYM_CTR_ALL (EXP:160 KEPT/DEFAULT via SYM_CTR_ALL, EXP:1274 ACCEPT -8.0%); not a loser: its old path is the SYM_CTR_ALL_OFF rollback (owed: A/B of SYM_CTR_ALL on current main, fast_prep.mojo note) |
| `MOJOLEARN_CTR_PREP_SHARED` | KEEP-OPEN | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+3/0 | 4 | - | gbdt/ctrs/fast_prep.mojo:136 | expdel-05: KEEP-OPEN (2026-11-15): default ON through SYM_CTR_ALL (EXP:161 KEPT/DEFAULT via SYM_CTR_ALL, EXP:1274); not a loser: its old path is the SYM_CTR_ALL_OFF rollback |
| `MOJOLEARN_EST_ITERS_DEVICE` | KEEP-OPEN | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+4/0 | 5 | - | gbdt/methods/leaves_estimation/apple_fast_est.mojo:129 | expdel-05: KEEP-OPEN (2026-11-15): default ON through SYM_EST_ALL (EXP:163 KEPT/DEFAULT via SYM_EST_ALL, EXP:1275 ACCEPT -2.9%); not a loser: its old path is the SYM_EST_ALL_OFF rollback |
| `MOJOLEARN_GBDT_BOOT_DEVICE` | DELETE-LOSER | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+1/0 | 2 | - | gbdt/gpu_data/sym_feat_switches.mojo:93 | expdel-05 |
| `MOJOLEARN_GBDT_EVAL_FUSED` | DELETE-LOSER | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+10/0 | 122 | - | gbdt/gpu_data/sym_feat_switches.mojo:73 | expdel-05 |
| `MOJOLEARN_GBDT_EVAL_SKIP_EMPTY` | DELETE-LOSER | LOSE EXP:169 NEUTRAL | 1+1/0 | 2 | - | gbdt/gpu_data/sym_feat_switches.mojo:62 | expdel-05 |
| `MOJOLEARN_GBDT_INDEX_PACK_DEVICE` | KEEP-OPEN | LOSE EXP:170 NEUTRAL | 1+2/0 | 3 | - | gbdt/gpu_data/sym_feat_switches.mojo:35 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:170) but the prerequisite of the unmeasured AFT_G06 card (experiments/apple_fast_trees/G.json: INDEX_PACK_DEVICE in both arms); deleting it removes G06's arm: orchestrator decides (delete G06 with it, or keep) |
| `MOJOLEARN_GBDT_PREDICT_PACKED` | KEEP-OPEN | LOSE EXP:171 NEUTRAL | 1+5/0 | 50 | - | gbdt/gpu_data/sym_feat_switches.mojo:83 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:171) but the prerequisite of the unmeasured AFT_P05 card (core/apple_fast_tree_experiments.mojo, P.json: PREDICT_PACKED in both arms; AFT_P05 code sits in the packed `_apply_all` path); orchestrator decides (delete P05 with it, or keep) |
| `MOJOLEARN_GBDT_QUANT_DEVICE` | KEEP-OPEN | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+8/0 | 18 | - | gbdt/gpu_data/sym_feat_switches.mojo:52 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:172) but implied by GBDT_INDEX_PACK_DEVICE (G06 prerequisite) and held equal in G05; goes with INDEX_PACK_DEVICE |
| `MOJOLEARN_SYM_BUF_ARENA` | DELETE-LOSER | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+0/0 | 1 | - | gbdt/methods/sym_iter_fast.mojo:112 | expdel-05 |
| `MOJOLEARN_SYM_FEAT_ALL` | DELETE-LOSER | LOSE EXP:1300 RECORD NEUTRAL: stays off | 1+6/0 | 7 | - | gbdt/gpu_data/sym_feat_switches.mojo:21 | expdel-05 |
| `MOJOLEARN_SYM_HIST_ALL` | DELETE-LOSER | LOSE EXP:1299 RECORD NEUTRAL: stays off | 1+3/0 | 4 | - | gbdt/methods/kernel/sym_fast.mojo:32 | expdel-05 |
| `MOJOLEARN_SYM_ITER_ALL` | DELETE-LOSER | LOSE EXP:1301 RECORD NEUTRAL: stays off | 1+2/0 | 3 | - | gbdt/methods/sym_iter_fast.mojo:97 | expdel-05 |
| `MOJOLEARN_SYM_MULTI_ALL` | DELETE-LOSER | LOSE EXP:1302 RECORD NEUTRAL | 1+0/0 | 1 | - | gbdt/targets/kernel/yeti_rank.mojo:250 | expdel-05 |
| `MOJOLEARN_SYM_RESOLVE_BLOCK` | KEEP-OPEN | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+2/0 | 10 | - | gbdt/methods/kernel/sym_fast.mojo:53 | expdel-05: KEEP-OPEN (2026-11-15): NEUTRAL (EXP:200/224) but both arms of the unmeasured AFCL_T05 card (gbdt/apple_fast_classical.mojo) require it; orchestrator decides (delete T05 with it, or keep) |
| `MOJOLEARN_SYM_SORT_SWAP` | DELETE-LOSER | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+1/0 | 2 | - | gbdt/methods/kernel/sym_fast.mojo:42 | expdel-05 |
| `MOJOLEARN_YR_TASK_FUSED` | DELETE-LOSER | LOSE EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE | 1+0/0 | 1 | - | gbdt/targets/kernel/yeti_rank.mojo:249 | expdel-05 |
| `MOJOLEARN_KSHAP_FAST_BATCH_OFF` | DELETE-OLD-PATH | WIN EXP:692 KEPT | 1+2/0 | 3 | - | xtrees/agnostic_device.mojo:42 | expdel-05 |
| `MOJOLEARN_2010_ROWS_SORTED` | KEEP-OPEN | none  | 2+5/0 | 36 | - | ensemble/randomforest.mojo:161 | owed A/B: the default vs MOJOLEARN_2010_ROWS_SORTED on NV+AMD, rows of its family |
| `MOJOLEARN_2011_HIST_ITEMS1` | KEEP-OPEN | none  | 1+7/0 | 8 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:242 | owed A/B: the default vs MOJOLEARN_2011_HIST_ITEMS1 on NV+AMD, rows of its family |
| `MOJOLEARN_2012_SMEM_COPIES4` | KEEP-OPEN | none  | 1+3/0 | 4 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:642 | owed A/B: the default vs MOJOLEARN_2012_SMEM_COPIES4 on NV+AMD, rows of its family |
| `MOJOLEARN_2030_NO_FUSED_EST_MOVE` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:216 | owed A/B: the default vs MOJOLEARN_2030_NO_FUSED_EST_MOVE on NV+AMD, rows of its family |
| `MOJOLEARN_2502_RETRY_PURE` | KEEP-OPEN | none  | 1+5/0 | 7 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:612 | owed A/B: the default vs MOJOLEARN_2502_RETRY_PURE on NV+AMD, rows of its family |
| `MOJOLEARN_2551_DEVICE_PARTITION` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo:63 | owed A/B: the default vs MOJOLEARN_2551_DEVICE_PARTITION on NV+AMD, rows of its family |
| `MOJOLEARN_2551_DEVICE_PARTITION_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo:65 | owed A/B: default vs the old path MOJOLEARN_2551_DEVICE_PARTITION_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_2580_LEVEL_QUANT` | KEEP-OPEN | none  | 1+10/0 | 19 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:608 | owed A/B: the default vs MOJOLEARN_2580_LEVEL_QUANT on NV+AMD, rows of its family |
| `MOJOLEARN_2581_GROUP_WIDTH` | KEEP-OPEN | none  | 1+9/0 | 12 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:653 | owed A/B: the default vs MOJOLEARN_2581_GROUP_WIDTH on NV+AMD, rows of its family |
| `MOJOLEARN_2581_GROUP_WIDTH_OFF` | KEEP-OPEN | none  | 1+9/0 | 10 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:655 | owed A/B: default vs the old path MOJOLEARN_2581_GROUP_WIDTH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_2635_LINEAR_BOUNDS` | KEEP-OPEN | none  | 1+1/0 | 7 | - | gbdt/grid_creator/binarization.mojo:79 | owed A/B: the default vs MOJOLEARN_2635_LINEAR_BOUNDS on NV+AMD, rows of its family |
| `MOJOLEARN_2710_TENSOR_ACC_DEAD` | KEEP-OPEN | none  | 1+1/0 | 4 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:3866 | owed A/B: the default vs MOJOLEARN_2710_TENSOR_ACC_DEAD on NV+AMD, rows of its family |
| `MOJOLEARN_3040_YETI_BLOCK` | KEEP-OPEN | none  | 1+0/0 | 2 | - | gbdt/targets/kernel/yeti_rank.mojo:162 | owed A/B: the default vs MOJOLEARN_3040_YETI_BLOCK on NV+AMD, rows of its family |
| `MOJOLEARN_3040_YETI_SEQUENTIAL` | KEEP-OPEN | none  | 1+0/0 | 2 | - | gbdt/targets/kernel/yeti_rank.mojo:160 | owed A/B: the default vs MOJOLEARN_3040_YETI_SEQUENTIAL on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T01` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:21 | owed A/B: the default vs MOJOLEARN_AFCL_T01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T02` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:26 | owed A/B: the default vs MOJOLEARN_AFCL_T02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T03` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:31 | owed A/B: the default vs MOJOLEARN_AFCL_T03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T04` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:36 | owed A/B: the default vs MOJOLEARN_AFCL_T04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T05` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:42 | owed A/B: the default vs MOJOLEARN_AFCL_T05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T06` | KEEP-OPEN | none  | 1+2/0 | 4 | - | gbdt/apple_fast_classical.mojo:48 | owed A/B: the default vs MOJOLEARN_AFCL_T06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T07` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:53 | owed A/B: the default vs MOJOLEARN_AFCL_T07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T09` | KEEP-OPEN | none  | 1+2/0 | 3 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4008 | owed A/B: the default vs MOJOLEARN_AFCL_T09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:238 | owed A/B: the default vs MOJOLEARN_AFCL_T10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T11` | KEEP-OPEN | none  | 1+2/0 | 3 | - | xtrees/shap_device.mojo:42 | owed A/B: the default vs MOJOLEARN_AFCL_T11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFCL_T12` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_classical.mojo:59 | owed A/B: the default vs MOJOLEARN_AFCL_T12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F01` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:166 | owed A/B: the default vs MOJOLEARN_AFT_F01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F02` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:240 | owed A/B: the default vs MOJOLEARN_AFT_F02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F03` | KEEP-OPEN | none  | 1+8/0 | 9 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3634 | owed A/B: the default vs MOJOLEARN_AFT_F03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F04` | KEEP-OPEN | none  | 1+4/0 | 5 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:1564 | owed A/B: the default vs MOJOLEARN_AFT_F04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F05` | KEEP-OPEN | none  | 1+4/0 | 5 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:2021 | owed A/B: the default vs MOJOLEARN_AFT_F05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F06` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3854 | owed A/B: the default vs MOJOLEARN_AFT_F06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F07` | KEEP-OPEN | none  | 1+2/0 | 3 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4101 | owed A/B: the default vs MOJOLEARN_AFT_F07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F08` | KEEP-OPEN | none  | 1+4/0 | 5 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4325 | owed A/B: the default vs MOJOLEARN_AFT_F08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F09` | KEEP-OPEN | none  | 1+5/0 | 6 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:730 | owed A/B: the default vs MOJOLEARN_AFT_F09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F10` | KEEP-OPEN | none  | 1+1/0 | 2 | - | isolation_forest/impl/isolation_tree_builder.mojo:204 | owed A/B: the default vs MOJOLEARN_AFT_F10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F11` | KEEP-OPEN | none  | 1+10/0 | 11 | - | isolation_forest/impl/isolation_tree_builder.mojo:1281 | owed A/B: the default vs MOJOLEARN_AFT_F11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_F12` | KEEP-OPEN | none  | 1+2/0 | 24 | - | isolation_forest/impl/isolation_tree_builder.mojo:1004 | owed A/B: the default vs MOJOLEARN_AFT_F12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G01` | KEEP-OPEN | none  | 1+2/0 | 11 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:18 | owed A/B: the default vs MOJOLEARN_AFT_G01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G02` | KEEP-OPEN | none  | 1+2/0 | 19 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:21 | owed A/B: the default vs MOJOLEARN_AFT_G02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G03` | KEEP-OPEN | none  | 1+2/0 | 26 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:24 | owed A/B: the default vs MOJOLEARN_AFT_G03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G04` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:27 | owed A/B: the default vs MOJOLEARN_AFT_G04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G05` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:30 | owed A/B: the default vs MOJOLEARN_AFT_G05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G06` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:33 | owed A/B: the default vs MOJOLEARN_AFT_G06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G07` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:36 | owed A/B: the default vs MOJOLEARN_AFT_G07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G08` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:39 | owed A/B: the default vs MOJOLEARN_AFT_G08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G09` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:42 | owed A/B: the default vs MOJOLEARN_AFT_G09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:45 | owed A/B: the default vs MOJOLEARN_AFT_G10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G11` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:48 | owed A/B: the default vs MOJOLEARN_AFT_G11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_G12` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/gpu_data/apple_fast_trees_experiments.mojo:51 | owed A/B: the default vs MOJOLEARN_AFT_G12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N01` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:14 | owed A/B: the default vs MOJOLEARN_AFT_N01 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N02` | KEEP-OPEN | none  | 1+7/0 | 11 | - | gbdt/apple_fast_tree_experiments.mojo:15 | owed A/B: the default vs MOJOLEARN_AFT_N02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N03` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:16 | owed A/B: the default vs MOJOLEARN_AFT_N03 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N04` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:17 | owed A/B: the default vs MOJOLEARN_AFT_N04 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N05` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:18 | owed A/B: the default vs MOJOLEARN_AFT_N05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N06` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:19 | owed A/B: the default vs MOJOLEARN_AFT_N06 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N07` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/apple_fast_tree_experiments.mojo:20 | owed A/B: the default vs MOJOLEARN_AFT_N07 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N08` | KEEP-OPEN | none  | 1+4/0 | 5 | - | gbdt/apple_fast_tree_experiments.mojo:21 | owed A/B: the default vs MOJOLEARN_AFT_N08 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N09` | KEEP-OPEN | none  | 1+6/0 | 13 | - | gbdt/apple_fast_tree_experiments.mojo:22 | owed A/B: the default vs MOJOLEARN_AFT_N09 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N10` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:23 | owed A/B: the default vs MOJOLEARN_AFT_N10 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N11` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/apple_fast_tree_experiments.mojo:24 | owed A/B: the default vs MOJOLEARN_AFT_N11 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_N12` | KEEP-OPEN | none  | 1+2/0 | 12 | - | gbdt/apple_fast_tree_experiments.mojo:25 | owed A/B: the default vs MOJOLEARN_AFT_N12 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P02` | KEEP-OPEN | none  | 1+2/1 | 3 | - | core/apple_fast_tree_experiments.mojo:27 | owed A/B: the default vs MOJOLEARN_AFT_P02 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P05` | KEEP-OPEN | none  | 1+3/0 | 45 | - | core/apple_fast_tree_experiments.mojo:40 | owed A/B: the default vs MOJOLEARN_AFT_P05 on NV+AMD, rows of its family |
| `MOJOLEARN_AFT_P07` | KEEP-OPEN | none  | 1+1/1 | 2 | - | core/apple_fast_tree_experiments.mojo:48 | owed A/B: the default vs MOJOLEARN_AFT_P07 on NV+AMD, rows of its family |
| `MOJOLEARN_AGN_IDN_SYN_POOL_OFF` | KEEP-OPEN | none  | 1+3/0 | 7 | - | xtrees/agnostic_device.mojo:53 | owed A/B: default vs the old path MOJOLEARN_AGN_IDN_SYN_POOL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_CTR_INDEX_FUSED` | KEEP-OPEN | WIN EXP:1274 ACCEPT/DEFAULT: FAST+Apple default | 1+3/0 | 4 | - | gbdt/ctrs/fast_prep.mojo:154 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_CTR_SORT_ONCE` | KEEP-OPEN | WIN EXP:1274 ACCEPT/DEFAULT: FAST+Apple default | 1+5/0 | 6 | - | gbdt/ctrs/fast_prep.mojo:143 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_DART_DEVICE_OFF` | KEEP-OPEN | WIN EXP:104 KEPT 741a5495d | 1+10/0 | 19 | - | xtrees/dart_device.mojo:86 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_DART_DEVICE_OFF |
| `MOJOLEARN_ET_BINNED_OFF` | KEEP-OPEN | none  | 1+5/0 | 7 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:2602 | owed A/B: default vs the old path MOJOLEARN_ET_BINNED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_BATCH_16384` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/estimator.mojo:500 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_BATCH_16384 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_BATCH_32768` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/estimator.mojo:496 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_BATCH_32768 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_BATCH_4096` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/estimator.mojo:498 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_BATCH_4096 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_LOOP_K1` | KEEP-OPEN | none  | 1+7/0 | 8 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:5512 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_LOOP_K1 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_LOOP_K2` | KEEP-OPEN | none  | 1+7/0 | 8 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:5513 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_LOOP_K2 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_DEVICE_LOOP_K8` | KEEP-OPEN | none  | 1+7/0 | 8 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:5514 | owed A/B: the default vs MOJOLEARN_ET_DEVICE_LOOP_K8 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:294 | owed A/B: the default vs MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER on NV+AMD, rows of its family |
| `MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH` | KEEP-OPEN | none  | 1+0/0 | 11 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:618 | owed A/B: the default vs MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH on NV+AMD, rows of its family |
| `MOJOLEARN_ET_MAX_ACC_16` | KEEP-OPEN | none  | 2+2/0 | 10 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3925 | owed A/B: the default vs MOJOLEARN_ET_MAX_ACC_16 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_MAX_ACC_32` | KEEP-OPEN | none  | 1+2/0 | 8 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:5183 | owed A/B: the default vs MOJOLEARN_ET_MAX_ACC_32 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_MAX_ACC_4` | KEEP-OPEN | none  | 2+2/0 | 10 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3921 | owed A/B: the default vs MOJOLEARN_ET_MAX_ACC_4 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_MAX_ACC_8` | KEEP-OPEN | none  | 2+2/0 | 10 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3923 | owed A/B: the default vs MOJOLEARN_ET_MAX_ACC_8 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:1732 | owed A/B: the default vs MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS on NV+AMD, rows of its family |
| `MOJOLEARN_ET_PART_FLAGS_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4149 | owed A/B: default vs the old path MOJOLEARN_ET_PART_FLAGS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_PART_ROWS` | KEEP-OPEN | WIN EXP:105 KEPT 269ffa57a | 1+11/0 | 12 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4108 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_ET_PART_ROWS_OFF` | KEEP-OPEN | WIN EXP:105 KEPT 269ffa57a | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4111 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_ET_PART_ROWS_OFF |
| `MOJOLEARN_ET_RANGE_TILED` | KEEP-OPEN | none  | 1+5/0 | 8 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3993 | owed A/B: the default vs MOJOLEARN_ET_RANGE_TILED on NV+AMD, rows of its family |
| `MOJOLEARN_ET_RELAXED_PUBLISH` | KEEP-OPEN | none  | 1+3/0 | 16 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:323 | owed A/B: the default vs MOJOLEARN_ET_RELAXED_PUBLISH on NV+AMD, rows of its family |
| `MOJOLEARN_ET_RM_LINE_HW_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4060 | owed A/B: default vs the old path MOJOLEARN_ET_RM_LINE_HW_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_RM_NARROW` | KEEP-OPEN | none  | 1+3/0 | 19 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4044 | owed A/B: the default vs MOJOLEARN_ET_RM_NARROW on NV+AMD, rows of its family |
| `MOJOLEARN_ET_RM_NARROW_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4046 | owed A/B: default vs the old path MOJOLEARN_ET_RM_NARROW_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_ROW_MAJOR` | KEEP-OPEN | none  | 1+10/0 | 13 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3941 | owed A/B: the default vs MOJOLEARN_ET_ROW_MAJOR on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SCORE_TILED` | KEEP-OPEN | none  | 1+4/0 | 6 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4031 | owed A/B: the default vs MOJOLEARN_ET_SCORE_TILED on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_1` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:80 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_1 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_128` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:66 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_128 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_16` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:72 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_16 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_2` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:78 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_2 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_256` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:64 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_256 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_32` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:70 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_32 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_4` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:76 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_4 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_64` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:68 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_64 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEARCH_RPT_8` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:74 | owed A/B: the default vs MOJOLEARN_ET_SEARCH_RPT_8 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SEQCST_PUBLISH` | KEEP-OPEN | none  | 1+3/0 | 9 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:324 | owed A/B: the default vs MOJOLEARN_ET_SEQCST_PUBLISH on NV+AMD, rows of its family |
| `MOJOLEARN_ET_SHARED_CLASS_COUNTS` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:1734 | owed A/B: the default vs MOJOLEARN_ET_SHARED_CLASS_COUNTS on NV+AMD, rows of its family |
| `MOJOLEARN_ET_STAGE_FULL_CAPACITY` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4173 | owed A/B: the default vs MOJOLEARN_ET_STAGE_FULL_CAPACITY on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TIE_MAX_COLID` | KEEP-OPEN | none  | 1+1/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/split.mojo:78 | owed A/B: the default vs MOJOLEARN_ET_TIE_MAX_COLID on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3959 | owed A/B: default vs the old path MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TILED_SEARCH_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3946 | owed A/B: default vs the old path MOJOLEARN_ET_TILED_SEARCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TPB_1024` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3857 | owed A/B: the default vs MOJOLEARN_ET_TPB_1024 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TPB_128` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3861 | owed A/B: the default vs MOJOLEARN_ET_TPB_128 on NV+AMD, rows of its family |
| `MOJOLEARN_ET_TPB_512` | KEEP-OPEN | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3859 | owed A/B: the default vs MOJOLEARN_ET_TPB_512 on NV+AMD, rows of its family |
| `MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF` | KEEP-OPEN | none  | 1+0/1 | 1 | - | core/forest_inference_model.mojo:81 | owed A/B: default vs the old path MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF` | KEEP-OPEN | OPEN EXP:2273 CANDIDATE (perf fix, 2026-11-05) | 1+1/0 | 2 | - | gbdt/grid_creator/gls_borders_device.mojo:44 | owed A/B: default vs the old path MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_BT_HW_F64_OFF` | KEEP-OPEN | OPEN EXP:2272 CANDIDATE (perf fix, 2026-11-05) | 1+2/0 | 3 | - | gbdt/grid_creator/border_types_cells.mojo:48 | owed A/B: default vs the old path MOJOLEARN_GBDT_BT_HW_F64_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF` | KEEP-OPEN | WIN EXP:108 KEPT 57ddea946 | 1+2/0 | 3 | - | gbdt/methods/doc_parallel_boosting.mojo:293 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF |
| `MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF` | KEEP-OPEN | LOSE EXP:45 pointwise-searcher SymmetricTree GBDT rows (catboost-style regressor and classif | 1+3/0 | 5 | - | gbdt/methods/doc_parallel_boosting.mojo:2906 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF` | KEEP-OPEN | WIN EXP:110 KEPT c55b8c377 | 1+2/0 | 19 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:353 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF |
| `MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF` | KEEP-OPEN | LOSE EXP:137 DROPPED-noise | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:287 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_GBDT_DW_MODE_SKIP_OFF` | KEEP-OPEN | WIN EXP:112 KEPT 7736fa5b8 | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo:76 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_GBDT_DW_MODE_SKIP_OFF |
| `MOJOLEARN_GBDT_FUSED_LEVEL_OFF` | KEEP-OPEN | none  | 2+8/0 | 13 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:287 | owed A/B: default vs the old path MOJOLEARN_GBDT_FUSED_LEVEL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:182 | owed A/B: the default vs MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_ID_DEFER_COPY` | KEEP-OPEN | WIN EXP:1503 KEPT as switch | 1+6/0 | 69 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:178 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_GBDT_LG_BATCH` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1193 | owed A/B: the default vs MOJOLEARN_GBDT_LG_BATCH on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_BATCH2` | KEEP-OPEN | none  | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1200 | owed A/B: the default vs MOJOLEARN_GBDT_LG_BATCH2 on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_BATCH32` | KEEP-OPEN | none  | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1197 | owed A/B: the default vs MOJOLEARN_GBDT_LG_BATCH32 on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_BATCH4` | KEEP-OPEN | none  | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1199 | owed A/B: the default vs MOJOLEARN_GBDT_LG_BATCH4 on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_BATCH8` | KEEP-OPEN | none  | 1+5/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1198 | owed A/B: the default vs MOJOLEARN_GBDT_LG_BATCH8 on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_BATCH_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1194 | owed A/B: default vs the old path MOJOLEARN_GBDT_LG_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH128` | KEEP-OPEN | WIN EXP:113 KEPT 9722e5a2b | 1+2/0 | 3 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1317 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF` | KEEP-OPEN | WIN EXP:113 KEPT 9722e5a2b | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1322 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH32` | KEEP-OPEN | none  | 2+4/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1320 | owed A/B: the default vs MOJOLEARN_GBDT_LG_EXACT_BATCH32 on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH64` | KEEP-OPEN | WIN EXP:114 KEPT 269ffa57a | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1335 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1306 | owed A/B: default vs the old path MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_LG_EXACT_ID_OFF` | KEEP-OPEN | none  | 1+7/0 | 8 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1283 | owed A/B: default vs the old path MOJOLEARN_GBDT_LG_EXACT_ID_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_NS_INHERIT_ID_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1393 | owed A/B: default vs the old path MOJOLEARN_GBDT_NS_INHERIT_ID_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF` | KEEP-OPEN | none  | 1+4/0 | 5 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1399 | owed A/B: default vs the old path MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_NS_PREDICT_BITVEC` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gbdt/models/ns_bitvector_predict.mojo:52 | owed A/B: the default vs MOJOLEARN_GBDT_NS_PREDICT_BITVEC on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF` | KEEP-OPEN | WIN EXP:2148 DEFAULT (bit-neutral) | 1+1/0 | 2 | - | gbdt/methods/ordered_boosting.mojo:277 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_GBDT_RIDX_GATHER_COST` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:533 | owed A/B: the default vs MOJOLEARN_GBDT_RIDX_GATHER_COST on NV+AMD, rows of its family |
| `MOJOLEARN_GBDT_WALK_DEV_STEP_OFF` | KEEP-OPEN | WIN EXP:2147 DEFAULT (bit-neutral) | 1+5/0 | 6 | - | gbdt/methods/leaves_estimation/device_walker.mojo:112 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_DART_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | xtrees/dart_units.mojo:43 | owed A/B: default vs the old path MOJOLEARN_IDN_DART_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ET_DEVICE_LOOP` | KEEP-OPEN | OPEN EXP:2401 RUN OWED | 1+1/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:6961 | owed A/B: the default vs MOJOLEARN_IDN_ET_DEVICE_LOOP on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ET_PART_ROWS_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4130 | owed A/B: default vs the old path MOJOLEARN_IDN_ET_PART_ROWS_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ET_STAGE_LIVE_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4168 | owed A/B: default vs the old path MOJOLEARN_IDN_ET_STAGE_LIVE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ET_TILED_SEARCH_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3977 | owed A/B: default vs the old path MOJOLEARN_IDN_ET_TILED_SEARCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_APPLY_WIDE` | KEEP-OPEN | none  | 1+3/1 | 4 | - | gbdt/models/kernel/add_bin_values.mojo:65 | owed A/B: the default vs MOJOLEARN_IDN_GBDT_APPLY_WIDE on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/methods/doc_parallel_boosting.mojo:286 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/methods/leaves_estimation/leaves_estimation.mojo:107 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/methods/leaves_estimation/leaves_estimation.mojo:140 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:274 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:345 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF` | KEEP-OPEN | none  | 1+4/0 | 6 | - | gbdt/data/pairs.mojo:46 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD` | KEEP-OPEN | OPEN EXP:2417 RUN OWED | 1+0/0 | 1 | - | gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:573 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | gbdt/methods/leaves_estimation/leaves_estimation.mojo:162 | owed A/B: default vs the old path MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF` | KEEP-OPEN | LOSE EXP:1584 DROP, define deleted (no-op) | 1+0/0 | 1 | c | gbdt/targets/kernel/yeti_rank.mojo:186 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | isolation_forest/impl/isolation_forest.mojo:616 | owed A/B: default vs the old path MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_IF_SCORES_F64_OFF` | KEEP-OPEN | OPEN EXP:2397 PENDING | 1+2/0 | 3 | - | isolation_forest/impl/isolation_forest.mojo:874 | owed A/B: default vs the old path MOJOLEARN_IDN_IF_SCORES_F64_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ORD_CAT_PLANES_OFF` | KEEP-OPEN | OPEN grid fg-gbdt-ordered.json pending | 1+1/0 | 2 | - | gbdt/trees_identical_switches.mojo:186 | owed A/B: default vs the old path MOJOLEARN_IDN_ORD_CAT_PLANES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ORD_INDEX_SHARED` | KEEP-OPEN | OPEN grid fg-gbdt-ordered.json pending | 1+1/0 | 2 | - | gbdt/trees_identical_switches.mojo:232 | owed A/B: the default vs MOJOLEARN_IDN_ORD_INDEX_SHARED on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/methods/ordered_boosting.mojo:296 | owed A/B: default vs the old path MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF` | KEEP-OPEN | none  | 1+7/0 | 93 | - | gbdt/methods/ordered_boosting.mojo:319 | owed A/B: default vs the old path MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_PSHAP_RESIDENT_OFF` | KEEP-OPEN | WIN EXP:2051 KEEP (promoted 2026-10-11; _OFF restores) | 1+2/0 | 3 | - | xtrees/agnostic_device.mojo:102 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_IDN_RF_COLS40_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:155 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_COLS40_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP` | KEEP-OPEN | OPEN EXP:2402 RUN OWED | 1+1/0 | 2 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:1017 | owed A/B: the default vs MOJOLEARN_IDN_RF_DEVICE_LOOP on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF` | KEEP-OPEN | none  | 1+5/0 | 6 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:980 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:2719 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_HIST_ZERO_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:541 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_HIST_ZERO_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF` | KEEP-OPEN | none  | 2+2/0 | 4 | - | ensemble/decisiontree/batched_levelalgo/quantiles.mojo:543 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:465 | owed A/B: default vs the old path MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF` | KEEP-OPEN | none  | 1+1/0 | 2 | - | xtrees/agnostic_device.mojo:74 | owed A/B: default vs the old path MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_IF_DIAG_ENTRY_RETURN` | KEEP-OPEN | none  | 1+1/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:302 | owed A/B: the default vs MOJOLEARN_IF_DIAG_ENTRY_RETURN on NV+AMD, rows of its family |
| `MOJOLEARN_IF_DIAG_GATHER_ONLY` | KEEP-OPEN | none  | 1+1/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:303 | owed A/B: the default vs MOJOLEARN_IF_DIAG_GATHER_ONLY on NV+AMD, rows of its family |
| `MOJOLEARN_IF_DIAG_NO_RECORD` | KEEP-OPEN | none  | 1+1/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:305 | owed A/B: the default vs MOJOLEARN_IF_DIAG_NO_RECORD on NV+AMD, rows of its family |
| `MOJOLEARN_IF_DIAG_NO_REJECT` | KEEP-OPEN | none  | 1+1/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:304 | owed A/B: the default vs MOJOLEARN_IF_DIAG_NO_REJECT on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_ET_BINNED` | KEEP-OPEN | none  | 1+1/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:2610 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_ET_BINNED on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_ET_RM` | KEEP-OPEN | OPEN EXP:1330 OWED | 1+2/0 | 10 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4077 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_ET_RM on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_ORD_ALL` | KEEP-OPEN | none  | 1+1/0 | 4 | - | gbdt/methods/ordered_fast_switches.mojo:68 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_ORD_ALL on NV+AMD, rows of its family |
| `MOJOLEARN_LEGACY_NARROW_SYM_RIDX` | KEEP-OPEN | OPEN EXP:1331 OWED | 1+2/0 | 3 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:641 | owed A/B: default vs the old path MOJOLEARN_LEGACY_NARROW_SYM_RIDX on NV+AMD, rows of its family |
| `MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF` | KEEP-OPEN | WIN EXP:116 KEPT 28adffe85 | 1+0/0 | 1 | - | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1609 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF |
| `MOJOLEARN_ORDERED_BATCH_EST_OFF` | KEEP-OPEN | WIN EXP:117 KEPT cf45a2732 | 1+8/0 | 9 | - | gbdt/methods/ordered_boosting.mojo:1918 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_ORDERED_BATCH_EST_OFF |
| `MOJOLEARN_ORD_ALL_OFF` | KEEP-OPEN | LOSE EXP:180 DROPPED-noise (standalone) | 1+11/0 | 43 | - | gbdt/methods/ordered_fast_switches.mojo:61 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_ORD_DOC_ID_STORAGE` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/methods/ordered_fast_switches.mojo:95 | owed A/B: the default vs MOJOLEARN_ORD_DOC_ID_STORAGE on NV+AMD, rows of its family |
| `MOJOLEARN_ORD_HIST_FOLD_SKIP` | KEEP-OPEN | OPEN grid fg-gbdt-ordered.json pending | 1+2/0 | 5 | - | gbdt/trees_identical_switches.mojo:209 | owed A/B: the default vs MOJOLEARN_ORD_HIST_FOLD_SKIP on NV+AMD, rows of its family |
| `MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS` | KEEP-OPEN | OPEN grid fg-gbdt-ordered.json pending | 1+1/0 | 2 | - | gbdt/trees_identical_switches.mojo:213 | owed A/B: the default vs MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS on NV+AMD, rows of its family |
| `MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF` | KEEP-OPEN | OPEN grid fg-gbdt-ordered.json pending | 1+1/0 | 2 | - | gbdt/trees_identical_switches.mojo:168 | owed A/B: default vs the old path MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF` | KEEP-OPEN | WIN EXP:118 KEPT 28adffe85 | 1+0/0 | 1 | - | gbdt/targets/kernel/pair_logit_group.mojo:120 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF |
| `MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF` | KEEP-OPEN | WIN EXP:119 KEPT 28adffe85 | 1+0/0 | 1 | - | gbdt/targets/kernel/pair_logit_group.mojo:96 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF |
| `MOJOLEARN_PL_GROUP_NARROW_OFF` | KEEP-OPEN | LOSE EXP:1302 RECORD NEUTRAL | 1+0/0 | 1 | - | gbdt/targets/kernel/pair_logit_group.mojo:146 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_PSHAP_DELTA_OFF` | KEEP-OPEN | OPEN EXP:1595 OPEN | 1+2/0 | 3 | - | xtrees/agnostic_device.mojo:111 | owed A/B: default vs the old path MOJOLEARN_PSHAP_DELTA_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_BINS_COLUMN_MAJOR` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/dataset.mojo:88 | owed A/B: the default vs MOJOLEARN_RF_BINS_COLUMN_MAJOR on NV+AMD, rows of its family |
| `MOJOLEARN_RF_COLS10` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:175 | owed A/B: the default vs MOJOLEARN_RF_COLS10 on NV+AMD, rows of its family |
| `MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX` | KEEP-OPEN | none  | 2+0/0 | 2 | - | ensemble/decisiontree/batched_levelalgo/split.mojo:762 | owed A/B: the default vs MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX on NV+AMD, rows of its family |
| `MOJOLEARN_RF_FBS_WARP_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3615 | owed A/B: default vs the old path MOJOLEARN_RF_FBS_WARP_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER` | KEEP-OPEN | none  | 1+6/0 | 89 | - | ensemble/randomforest.mojo:205 | owed A/B: the default vs MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER on NV+AMD, rows of its family |
| `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF` | KEEP-OPEN | none  | 1+6/0 | 15 | - | ensemble/randomforest.mojo:202 | owed A/B: default vs the old path MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_HIST_COLUMNS2` | KEEP-OPEN | none  | 3+2/0 | 5 | - | ensemble/bench/rf_bench.mojo:588 | owed A/B: the default vs MOJOLEARN_RF_HIST_COLUMNS2 on NV+AMD, rows of its family |
| `MOJOLEARN_RF_HIST_COLUMNS4` | KEEP-OPEN | none  | 3+3/0 | 6 | - | ensemble/bench/rf_bench.mojo:588 | owed A/B: the default vs MOJOLEARN_RF_HIST_COLUMNS4 on NV+AMD, rows of its family |
| `MOJOLEARN_RF_HIST_COLUMNS4_OFF` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3000 | owed A/B: default vs the old path MOJOLEARN_RF_HIST_COLUMNS4_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_HIST_COLUMNS8` | KEEP-OPEN | WIN EXP:120 KEPT 269ffa57a | 1+13/0 | 14 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3029 | record says promoted but the code is still opt-in: reconcile (flip to default with _OFF, or owed A/B) |
| `MOJOLEARN_RF_HIST_COLUMNS8_OFF` | KEEP-OPEN | WIN EXP:120 KEPT 269ffa57a | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3035 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_RF_HIST_COLUMNS8_OFF |
| `MOJOLEARN_RF_HIST_SIMD_AGG_OFF` | KEEP-OPEN | none  | 1+3/0 | 4 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:2733 | owed A/B: default vs the old path MOJOLEARN_RF_HIST_SIMD_AGG_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_ROWS_SORTED_OFF` | KEEP-OPEN | none  | 1+5/0 | 10 | - | ensemble/randomforest.mojo:164 | owed A/B: default vs the old path MOJOLEARN_RF_ROWS_SORTED_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_SMALL_NODE_FUSED` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:524 | owed A/B: the default vs MOJOLEARN_RF_SMALL_NODE_FUSED on NV+AMD, rows of its family |
| `MOJOLEARN_RF_SPLIT_CANDIDATES` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:487 | owed A/B: the default vs MOJOLEARN_RF_SPLIT_CANDIDATES on NV+AMD, rows of its family |
| `MOJOLEARN_RF_SPLIT_CANDIDATES_OFF` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:490 | owed A/B: default vs the old path MOJOLEARN_RF_SPLIT_CANDIDATES_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_RF_TRIAL_COLS20` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:180 | owed A/B: the default vs MOJOLEARN_RF_TRIAL_COLS20 on NV+AMD, rows of its family |
| `MOJOLEARN_RF_TRIAL_COLS40` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:181 | owed A/B: the default vs MOJOLEARN_RF_TRIAL_COLS40 on NV+AMD, rows of its family |
| `MOJOLEARN_SHAP_PERM_CACHE_OFF` | KEEP-OPEN | OPEN EXP:1403 READY-AB | 1+5/0 | 10 | - | xtrees/agnostic_device.mojo:133 | owed A/B: default vs the old path MOJOLEARN_SHAP_PERM_CACHE_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_SHAP_TREE_TAB_OFF` | KEEP-OPEN | WIN EXP:1271 ACCEPT/DEFAULT: FAST+Apple default | 1+5/0 | 6 | - | xtrees/shap_device.mojo:90 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SHAP_TREE_TAB_OFF |
| `MOJOLEARN_SYM_CTR_ALL_OFF` | KEEP-OPEN | WIN EXP:1274 ACCEPT/DEFAULT: FAST+Apple default | 1+5/0 | 6 | - | gbdt/ctrs/fast_prep.mojo:126 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SYM_CTR_ALL_OFF |
| `MOJOLEARN_SYM_EST_ALL_OFF` | KEEP-OPEN | WIN EXP:1275 ACCEPT/DEFAULT: FAST+Apple default | 1+3/0 | 4 | - | gbdt/methods/leaves_estimation/apple_fast_est.mojo:118 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_SYM_EST_ALL_OFF |
| `MOJOLEARN_TREESHAP_ACC_SHARED` | KEEP-OPEN | none  | 1+2/0 | 3 | - | xtrees/shap_device.mojo:120 | owed A/B: the default vs MOJOLEARN_TREESHAP_ACC_SHARED on NV+AMD, rows of its family |
| `MOJOLEARN_TREESHAP_FAST_TABLE_OFF` | KEEP-OPEN | WIN EXP:186 KEPT/DEFAULT | 1+6/0 | 7 | - | xtrees/shap_device.mojo:57 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_C45_GBDT` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/trees_identical_switches.mojo:112 | owed A/B: the default vs MOJOLEARN_TREES_C45_GBDT on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_C47_IF_LIVE_TASKS` | KEEP-OPEN | none  | 1+2/0 | 4 | - | isolation_forest/impl/isolation_tree_builder.mojo:163 | owed A/B: the default vs MOJOLEARN_TREES_C47_IF_LIVE_TASKS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_C48` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:63 | owed A/B: the default vs MOJOLEARN_TREES_C48 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_C48_IF_PARTITION` | KEEP-OPEN | none  | 1+3/0 | 38 | - | isolation_forest/impl/isolation_tree_builder.mojo:164 | owed A/B: the default vs MOJOLEARN_TREES_C48_IF_PARTITION on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_C50_IF_PACKED` | KEEP-OPEN | none  | 1+3/0 | 13 | - | isolation_forest/impl/isolation_forest.mojo:115 | owed A/B: the default vs MOJOLEARN_TREES_C50_IF_PACKED on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_CTR_SORTFREE_SUMS` | KEEP-OPEN | none  | 1+7/0 | 55 | - | gbdt/trees_identical_switches.mojo:141 | owed A/B: the default vs MOJOLEARN_TREES_CTR_SORTFREE_SUMS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_HIST_PACKED_GH` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/trees_hist_switches.mojo:30 | owed A/B: the default vs MOJOLEARN_TREES_HIST_PACKED_GH on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_HIST_REP_SM` | KEEP-OPEN | WIN EXP:1640 PROMOTED | 1+7/0 | 8 | c | gbdt/trees_small_switches.mojo:73 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_HIST_REP_SM_OFF` | KEEP-OPEN | WIN EXP:1640 PROMOTED | 1+3/0 | 4 | - | gbdt/trees_small_switches.mojo:68 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_HIST_WARP_AGG` | KEEP-OPEN | none  | 1+2/0 | 3 | - | gbdt/trees_hist_switches.mojo:35 | owed A/B: the default vs MOJOLEARN_TREES_HIST_WARP_AGG on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_IF_BUILD_TPB` | KEEP-OPEN | none  | 2+4/0 | 6 | - | isolation_forest/impl/isolation_tree_builder.mojo:196 | owed A/B: the default vs MOJOLEARN_TREES_IF_BUILD_TPB on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_LG_LEVEL_ROUNDS` | KEEP-OPEN | none  | 1+3/0 | 8 | - | gbdt/trees_identical_switches.mojo:136 | owed A/B: the default vs MOJOLEARN_TREES_LG_LEVEL_ROUNDS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF` | KEEP-OPEN | WIN EXP:1637 PROMOTED | 1+3/0 | 8 | - | gbdt/trees_identical_switches.mojo:130 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_PLATT_ONE_READ_OFF` | KEEP-OPEN | OPEN EXP:229 PENDING | 1+2/0 | 4 | - | xtrees/ops_device_boost.mojo:787 | owed A/B: default vs the old path MOJOLEARN_TREES_PLATT_ONE_READ_OFF on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_RF_SAMPLE` | KEEP-OPEN | LOSE EXP:1610 ALWAYS_OFF_DELETE, define deleted | 2+8/0 | 10 | c | ensemble/tree_identical_ideas.mojo:45 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_TREES_T02` | KEEP-OPEN | LOSE EXP:1581 DROP, define deleted (neutral + duplicate) | 1+0/0 | 1 | c | ensemble/tree_identical_ideas.mojo:21 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_TREES_T03` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:22 | owed A/B: the default vs MOJOLEARN_TREES_T03 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T04` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:23 | owed A/B: the default vs MOJOLEARN_TREES_T04 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T05` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:24 | owed A/B: the default vs MOJOLEARN_TREES_T05 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T05_ROW_VISITS` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/tree_identical_ideas.mojo:26 | owed A/B: the default vs MOJOLEARN_TREES_T05_ROW_VISITS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T06` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:27 | owed A/B: the default vs MOJOLEARN_TREES_T06 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T07` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:28 | owed A/B: the default vs MOJOLEARN_TREES_T07 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T08` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:29 | owed A/B: the default vs MOJOLEARN_TREES_T08 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T08_BITS` | KEEP-OPEN | none  | 1+5/0 | 6 | - | ensemble/tree_identical_ideas.mojo:31 | owed A/B: the default vs MOJOLEARN_TREES_T08_BITS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T08_LAYOUT` | KEEP-OPEN | none  | 1+7/0 | 8 | - | ensemble/tree_identical_ideas.mojo:30 | owed A/B: the default vs MOJOLEARN_TREES_T08_LAYOUT on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T11_LEVELS` | KEEP-OPEN | LOSE EXP:1582 DROP, defines deleted (no gain + duplicate) | 2+3/0 | 5 | c | ensemble/tree_identical_ideas.mojo:54 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_TREES_T12` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:56 | owed A/B: the default vs MOJOLEARN_TREES_T12 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T12_BYTES` | KEEP-OPEN | none  | 1+4/0 | 5 | - | ensemble/tree_identical_ideas.mojo:57 | owed A/B: the default vs MOJOLEARN_TREES_T12_BYTES on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T13` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:58 | owed A/B: the default vs MOJOLEARN_TREES_T13 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T13_BYTES` | KEEP-OPEN | none  | 1+3/0 | 4 | - | ensemble/tree_identical_ideas.mojo:59 | owed A/B: the default vs MOJOLEARN_TREES_T13_BYTES on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T14` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:60 | owed A/B: the default vs MOJOLEARN_TREES_T14 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T14_EXACT` | KEEP-OPEN | none  | 1+16/0 | 19 | - | ensemble/tree_identical_ideas.mojo:61 | owed A/B: the default vs MOJOLEARN_TREES_T14_EXACT on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T15` | KEEP-OPEN | none  | 1+0/0 | 1 | - | ensemble/tree_identical_ideas.mojo:62 | owed A/B: the default vs MOJOLEARN_TREES_T15 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T17_BATCH` | KEEP-OPEN | WIN EXP:1641 PROMOTED | 1+2/0 | 3 | c | gbdt/trees_identical_switches.mojo:29 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_T18` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:34 | owed A/B: the default vs MOJOLEARN_TREES_T18 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T19_DEFER` | KEEP-OPEN | LOSE EXP:1627 DROP (quality loss), code deleted | 1+2/0 | 3 | c | gbdt/trees_identical_switches.mojo:85 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_TREES_T21` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:42 | owed A/B: the default vs MOJOLEARN_TREES_T21 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T21_STREAMS` | KEEP-OPEN | none  | 1+7/0 | 8 | c | gbdt/trees_identical_switches.mojo:48 | owed A/B: the default vs MOJOLEARN_TREES_T21_STREAMS on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T22_OFF` | KEEP-OPEN | WIN EXP:1636 PROMOTED | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:63 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_T23` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:93 | owed A/B: the default vs MOJOLEARN_TREES_T23 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T24` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:96 | owed A/B: the default vs MOJOLEARN_TREES_T24 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T25` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:66 | owed A/B: the default vs MOJOLEARN_TREES_T25 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T26` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:99 | owed A/B: the default vs MOJOLEARN_TREES_T26 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T28` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:71 | owed A/B: the default vs MOJOLEARN_TREES_T28 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T28_PREFIX` | KEEP-OPEN | none  | 1+5/0 | 8 | - | gbdt/trees_identical_switches.mojo:89 | owed A/B: the default vs MOJOLEARN_TREES_T28_PREFIX on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T28_SORT` | KEEP-OPEN | none  | 1+4/0 | 5 | - | gbdt/trees_identical_switches.mojo:87 | owed A/B: the default vs MOJOLEARN_TREES_T28_SORT on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T29` | KEEP-OPEN | WIN EXP:1869 KEPT (code stays) | 1+0/0 | 1 | c | gbdt/trees_identical_switches.mojo:74 | record is indirect (WIN in a row or comment that does not name this define first): confirm the verdict, then d |
| `MOJOLEARN_TREES_T30` | KEEP-OPEN | none  | 1+0/0 | 1 | - | gbdt/trees_identical_switches.mojo:81 | owed A/B: the default vs MOJOLEARN_TREES_T30 on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T30_ADABOOST` | KEEP-OPEN | none  | 1+3/0 | 4 | - | gbdt/trees_identical_switches.mojo:109 | owed A/B: the default vs MOJOLEARN_TREES_T30_ADABOOST on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T40_ROWMAJOR` | KEEP-OPEN | none  | 1+2/0 | 3 | - | isolation_forest/impl/isolation_tree_builder.mojo:160 | owed A/B: the default vs MOJOLEARN_TREES_T40_ROWMAJOR on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T41_TREE_BATCH` | KEEP-OPEN | none  | 1+1/0 | 2 | - | isolation_forest/impl/isolation_tree_builder.mojo:161 | owed A/B: the default vs MOJOLEARN_TREES_T41_TREE_BATCH on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T42_CORRECTION_CACHE` | KEEP-OPEN | none  | 1+2/0 | 4 | - | isolation_forest/impl/isolation_tree_builder.mojo:162 | owed A/B: the default vs MOJOLEARN_TREES_T42_CORRECTION_CACHE on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T43_RADIX_SELECT` | KEEP-OPEN | none  | 1+2/0 | 19 | - | isolation_forest/impl/isolation_forest.mojo:1599 | owed A/B: the default vs MOJOLEARN_TREES_T43_RADIX_SELECT on NV+AMD, rows of its family |
| `MOJOLEARN_TREES_T45_LINEAR_UNWIND` | KEEP-OPEN | none  | 1+2/0 | 21 | - | xtrees/shap.mojo:85 | owed A/B: the default vs MOJOLEARN_TREES_T45_LINEAR_UNWIND on NV+AMD, rows of its family |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST` | KEEP-OPEN | none  | 2+5/1 | 12 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:105 | owed A/B: the default vs MOJOLEARN_TREE_EXACT_SIBLING_HIST on NV+AMD, rows of its family |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT` | KEEP-OPEN | none  | 4+0/0 | 17 | - | ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo:33 | owed A/B: the default vs MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT on NV+AMD, rows of its family |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES` | KEEP-OPEN | none  | 1+2/0 | 3 | - | ensemble/decisiontree/batched_levelalgo/builder.mojo:106 | owed A/B: the default vs MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES on NV+AMD, rows of its family |
| `MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF` | KEEP-OPEN | WIN EXP:125 KEPT 269ffa57a | 1+0/0 | 2 | - | gbdt/targets/kernel/yeti_rank.mojo:219 | promoted, but the record names one vendor: owed NV+AMD A/B default vs MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF |
| `MOJOLEARN_YETI_FAST_SORT_OFF` | KEEP-OPEN | LOSE EXP:228 NEUTRAL | 1+0/0 | 2 | - | gbdt/targets/kernel/yeti_rank.mojo:284 | record is indirect (LOSE in a row or comment that does not name this define first): confirm the verdict, then  |
| `MOJOLEARN_YETI_SEARCH_TASK16K_OFF` | KEEP-OPEN | LOSE EXP:155 DROPPED-noise | 1+0/0 | 2 | - | gbdt/targets/kernel/yeti_rank.mojo:535 | record says the old-path switch or its base was dropped: reconcile which arm lost (owed A/B default vs MOJOLEA |
| `MOJOLEARN_2041_FAST_CHUNKS_PIN` | KEEP | none  | 1+1/0 | 2 | - | gbdt/gpu_util/partitions_reduce.mojo:299 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_3041_ORACLE_ALLOC_PER_TREE` | KEEP | none  | 1+0/0 | 2 | - | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1582 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_3041_ORACLE_POOL` | KEEP | none  | 1+0/0 | 2 | - | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1584 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_BORDER_TYPES_SABOTAGE` | KEEP | none  | 1+1/0 | 4 | - | gbdt/grid_creator/binarization.mojo:927 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_EST_STATS_FUSED` | KEEP | WIN EXP:1275 ACCEPT/DEFAULT: FAST+Apple default | 1+4/0 | 5 | - | gbdt/methods/leaves_estimation/apple_fast_est.mojo:123 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_CYCLE_STATS` | KEEP | none  | 2+0/0 | 32 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:7701 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_KEY_UNSALTED` | KEEP | none  | 1+1/0 | 2 | - | extratrees/checks/pcg_rng.mojo:377 | read only by checks/tests/benches (not a product path) |
| `MOJOLEARN_ET_SAB_LEAF_ZERO_FILL` | KEEP | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:3054 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_SAB_PHASE_SETUP` | KEEP | none  | 1+0/0 | 2 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:1705 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_SAB_PLAIN_PUBLISH` | KEEP | none  | 1+1/0 | 4 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:318 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_SAB_RELAXED_PUBLISH` | KEEP | none  | 1+1/0 | 6 | - | extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:369 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS` | KEEP | none  | 1+0/0 | 3 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:4428 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ET_TIE_STATS` | KEEP | none  | 2+0/0 | 78 | - | extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:5304 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_BINARY_SABOTAGE` | KEEP | none  | 1+1/0 | 2 | - | gbdt/host/gbdt_oracle.mojo:229 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_EVAL_SABOTAGE` | KEEP | none  | 1+2/0 | 4 | - | gbdt/host/gbdt_oracle_eval.mojo:85 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_ORACLE_POOL_SABOTAGE` | KEEP | none  | 1+1/0 | 10 | - | gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1702 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_PARALLEL_SABOTAGE` | KEEP | none  | 2+0/0 | 9 | - | gbdt/methods/pointwise_multi_gpu.mojo:147 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_PART_STATS_WORK` | KEEP | none  | 1+2/0 | 7 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:212 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_RESIDENT_SABOTAGE` | KEEP | none  | 1+1/0 | 2 | - | gbdt/resident_model.mojo:182 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_SAB_SKIP_RIGHT_PART_STATS` | KEEP | none  | 1+0/0 | 1 | - | gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:3740 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_GBDT_YETI_SABOTAGE` | KEEP | none  | 1+2/0 | 3 | - | gbdt/targets/kernel/yeti_rank.mojo:196 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IF_DIAG_TRACE` | KEEP | none  | 1+6/0 | 13 | - | isolation_forest/impl/isolation_forest.mojo:98 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_IF_SABOTAGE_U64_SWAP` | KEEP | none  | 1+1/0 | 4 | - | isolation_forest/impl/isolation_tree_builder.mojo:288 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ISOLATION_FOREST_PARALLEL_SABOTAGE` | KEEP | none  | 1+0/0 | 11 | - | isolation_forest/multi_gpu.mojo:152 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_ORDERED_SABOTAGE` | KEEP | none  | 2+6/0 | 14 | - | gbdt/host/gbdt_oracle_ordered.mojo:1514 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_PROBE_ACQUIRE_CAS` | KEEP | none  | 1+3/0 | 12 | - | ensemble/checks/acquire_rmw_probe.mojo:69 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_RF_BENCH_BINARY` | KEEP | none  | 1+10/0 | 11 | - | ensemble/bench/rf_bench.mojo:66 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_RF_FAST_HIST_ZERO_OFF` | KEEP | none  | 1+0/0 | 1 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:557 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_RF_FAST_SAMPLE_PER_COLUMN` | KEEP | none  | 1+3/0 | 4 | - | ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:477 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SHAP_FAST_ROW_PAIR` | KEEP | none  | 1+3/1 | 5 | - | xtrees/shap_device.mojo:98 | Apple fast is frozen (Andrew 2026-10-22): opt-in fast code may stay; no A/B will run |
| `MOJOLEARN_SNAP_SABOTAGE` | KEEP | none  | 3+0/0 | 6 | - | gbdt/methods/greedy_subsets_searcher/kernel/histogram_utils.mojo:192 | not an experiment: sabotage arm, check, build or mode control |
| `MOJOLEARN_EST_REUSE_PART` | REMOVED | LOSE EXP:1849 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_EST_SHRINK_FUSED` | REMOVED | LOSE EXP:1850 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_ADA_SESSION_OFF` | REMOVED | LOSE EXP:2362 DELETED (escape of a kept winner) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_CTR_PERM_GRID` | REMOVED | LOSE EXP:2104 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_LG_FUSED_CHAIN` | REMOVED | LOSE EXP:2103 NOISE (deleted) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_IDN_PSHAP_RESIDENT` | REMOVED | WIN EXP:2051 KEEP (promoted 2026-10-11; _OFF restores) | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_KSHAP_FAST_DEVICE_MODEL` | REMOVED | LOSE EXP:2294 DELETED (never measured) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_KSHAP_FAST_OVERLAP` | REMOVED | LOSE EXP:1864 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MC_CLASS_BATCH_DERIV` | REMOVED | LOSE EXP:1851 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MC_CLASS_BATCH_EST` | REMOVED | LOSE EXP:1852 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_MSEL_UNITS_SERIAL` | REMOVED | LOSE EXP:2370 REMOVED (redundant) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_PL_PAIRS_ONCE` | REMOVED | LOSE EXP:1853 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_PSHAP_FAST_DEVICE_MODEL` | REMOVED | LOSE EXP:2294 DELETED (never measured) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_PSHAP_FAST_OVERLAP` | REMOVED | LOSE EXP:1865 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_RF_FAST_QSAMPLE` | REMOVED | LOSE EXP:2136 DROPPED (deleted 1a76955b5) | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_DERIV_FUSED` | REMOVED | LOSE EXP:1854 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_GATHER_FUSED` | REMOVED | LOSE EXP:1857 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_HIST_MULT` | REMOVED | LOSE EXP:1859 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_LEAF_FROM_STATS` | REMOVED | LOSE EXP:1855 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_PART_STATS_PAR` | REMOVED | LOSE EXP:1858 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_REUSE_PARTITION` | REMOVED | LOSE EXP:1856 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_SYM_SCAN_SUB_FUSED` | REMOVED | LOSE EXP:1860 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_TE_ADA_SESSION_OFF` | REMOVED | LOSE EXP:2362 DELETED (escape of a kept winner) | 0+0/1 | 0 | icr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_C47_GBDT` | REMOVED | LOSE EXP:1612 DROP_REDUNDANT, define deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_HIST_MULTISTAT` | REMOVED | LOSE EXP:1729 DROP (slower), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_HIST_REP_BPSM` | REMOVED | LOSE EXP:1613 MERGED into `MOJOLEARN_TREES_HIST_REP_SM` (arms 64, 0 = device x2, 1 = device x4 | 0+0/1 | 0 | cr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_ORD_STD_GRIDFOLD` | REMOVED | WIN EXP:1637 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_TREES_T09` | REMOVED | LOSE EXP:1610 ALWAYS_OFF_DELETE, define deleted | 0+0/1 | 0 | icpr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_T19` | REMOVED | LOSE EXP:1627 DROP (quality loss), code deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_T22` | REMOVED | WIN EXP:1636 PROMOTED | 0+0/1 | 0 | r | - | only the refusal guard remains |
| `MOJOLEARN_TREES_T27` | REMOVED | LOSE EXP:1611 DROP_REDUNDANT, alias deleted | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_TREES_T29_VERSIONED` | REMOVED | WIN EXP:1869 KEPT (code stays) | 0+0/1 | 0 | cpr | - | only the refusal guard remains |
| `MOJOLEARN_YETI_TREE_SEARCH_SCORE_GRID` | REMOVED | LOSE EXP:1861 DELETED | 0+0/1 | 0 | icpr | - | only the refusal guard remains |

