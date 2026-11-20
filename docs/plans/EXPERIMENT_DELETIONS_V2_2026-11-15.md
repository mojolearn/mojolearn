# Experiment deletions V2: every undecided define goes (2026-11-15)

Rule: AGENTS.md "Experiments", "An unmeasured arm is deleted" (Andrew, 2026-11-15, origin/main ce56ae89bd). Source: the inventory of `docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md` regenerated on `origin/preinteg/wave2-20261113` @ 76810c7c02 (scripts and joins: `~/mojolearn-evidence/briefs-20261114/experiment-decisions-inventory/v2/`, `scripts/v2.py`).

Excluded: defines that expdel-01, -04 and -05 already deleted (now refusal-only, docs/TOMBSTONES.md); the 56 defines of the in-flight lanes expdel-02, -03 and -06 (they stay with those lanes); the KEEP set (sabotage arms, check/build/mode switches, defines read only by checks/tests/benches) and the refusal guards.

## Decisions

| decision | rule | defines | live sites | branch lines (lower bound) |
|---|---|---|---|---|
| DELETE-ARM | a | 565 | 712 | 7338 |
| DELETE-ARM | e | 59 | 71 | 657 |
| DELETE-ARM | f | 50 | 58 | 664 |
| DELETE-EXPERIMENT | g | 102 | 103 | 598 |
| DELETE-OLD-PATH | b | 430 | 473 | 4305 |
| DELETE-OLD-PATH | d | 61 | 64 | 369 |
| DELETE-OLD-PATH | e | 122 | 541 | 4981 |
| PROMOTE | c | 1 | 1 | 16 |
| **total** | | **1390** | 2023 | 18928 |

Rules (the coordinator's letters): a = no-verdict opt-in arm; b = no-verdict old path beside a default; c = recorded promoted but still opt-in -> PROMOTE; d = one-vendor promotion -> old path goes, the owed vendor is written in the tombstone; e = indirect or reconcile record, read on 2026-11-15; f = neural MERGED-UNMEASURED; g = never-run Apple FAST programme card (AFT_*, AFCL_*) and the prerequisites held only for it; v1 = carried from the first plan (decided record).

**The mechanical edit (makes "the default stays byte-identical" checkable without a build):** DELETE-ARM replaces every `is_defined["D"]()` with `False` (a `get_defined_int["D", v]()` with its default `v`), then folds the constant: the branch taken only when D is set and every helper reachable only from it go. DELETE-OLD-PATH does the same for the `_OFF` / legacy / QOLD define. Any other define in the same expression keeps its read (an umbrella such as `SYM_CTR_ALL` or `IDN_ALL_OFF` stays as it is). DELETE-EXPERIMENT does the same for every card of a programme and then deletes the programme file when nothing reads it any more. A lane never edits a default arm, a hand-tuned kernel, a sabotage arm or verify code. Each deletion carries the four tombstone parts (inline `# TOMBSTONE:` + `# Restore:`, the TOMBSTONES.md row and section, `experiments/removed/<DEFINE>.patch`, the `removed` refusal + grid entry moved to `removed` + EXPERIMENTS.md row); rule d writes "promoted on <vendor> only; <other vendor> A/B owed" in the tombstone.

Bits: no DELETE changes bits or needs re-timing (the default arm is untouched). PROMOTE changes bits: see its lane.

## PROMOTE

- `MOJOLEARN_ARIMA_FAST_SEARCH_REUSE`: EXP:1420 KEEP (M3 rab16: synthetic 18229.7 -> 13688.1, taxi-hourly 16386.5 -> 9872.2) but code is opt-in (arima/impl/fast_order_search.mojo:111): becomes the FAST Apple default, switch goes. Bits: Apple fast AutoARIMA output only; no identical column changes.

Read and not promoted (the other 11 "promoted but opt-in" rows): `CLASSICAL_C24_PANEL8`, `CLASSICAL_C24_TREE4` (unpromoted grid arms); `ET_PART_ROWS`, `GBDT_LG_EXACT_BATCH128`, `HUBER_FAST_BLOCK512`, `MI_CLF_RANKMAJOR`, `MI_REG_SORTCOUNT`, `MI_REG_TIES`, `RF_HIST_COLUMNS8`, `VSEARCH_ALL` (force-on knobs for an arm that is already the default); `GBDT_LG_EXACT_BATCH64` (the KEPT row is the FAST width; its IDENTICAL arm never ran). The 167 indirect / reconcile records were read: none is an opt-in measured winner that is not already the default (the promoted ones are int knobs whose default is already the winner, e.g. `TREES_T17_BATCH`, `TREES_HIST_REP_SM`, `IDN_M2_YD_ROWS`, `IDN_GEMM_LEAF`, or umbrellas split into new defines).

## Lanes

A lane owns a set of code files (no code file is in two lanes) and makes the mechanical edit for every listed define inside them. An alias that other files import keeps its name, defined as the folded constant, so importers in other lanes compile before and after; the integrator removes aliases left as an unused constant after the wave. The owner lane of a define (its first live read site) writes `experiments/removed/<DEFINE>.patch`, the TOMBSTONES.md section, the refusal and the records; any other lane touching it adds the inline tombstones and `experiments/removed/<DEFINE>.<lane>.patch`, named in the owner's section. Shared append-only files (`docs/TOMBSTONES.md`, `docs/apple-fast/EXPERIMENTS.md`, `core/experiment_guards_*.mojo`, `core/six_lane_experiment_guards.mojo`, the grid json under `experiments/six_lane_integration/grid_controls/`) get one dated block per lane at the end; the integrator keeps both sides. Lanes start after expdel-02/-03/-06 merge (some files overlap theirs). Order: the DELETE lanes in any order, in parallel; the PROMOTE lane last (it changes bits).

| lane | defines touched (owned) | decisions | code files | est. branch lines in its files |
|---|---|---|---|---|
| expdel2-01-base | 26 (6 owned) | ARM 11, OLD-PATH 15 | 52 | 157 |
| expdel2-02-base | 44 (14 owned) | ARM 20, OLD-PATH 24 | 22 | 364 |
| expdel2-03-base | 43 (36 owned) | ARM 22, EXPERIMENT 13, OLD-PATH 8 | 198 | 845 |
| expdel2-04-base | 45 (30 owned) | ARM 24, OLD-PATH 21 | 25 | 484 |
| expdel2-05-base | 35 (30 owned) | ARM 25, OLD-PATH 10 | 9 | 110 |
| expdel2-06-base | 28 (28 owned) | ARM 23, OLD-PATH 5 | 1 | 58 |
| expdel2-07-base | 45 (40 owned) | ARM 39, EXPERIMENT 1, OLD-PATH 5 | 50 | 278 |
| expdel2-08-base | 43 (16 owned) | ARM 23, OLD-PATH 20 | 49 | 297 |
| expdel2-09-base | 30 (14 owned) | ARM 15, EXPERIMENT 2, OLD-PATH 13 | 19 | 161 |
| expdel2-10-cluster | 44 (37 owned) | ARM 9, EXPERIMENT 2, OLD-PATH 33 | 21 | 552 |
| expdel2-11-cluster | 45 (38 owned) | ARM 18, EXPERIMENT 1, OLD-PATH 26 | 15 | 587 |
| expdel2-12-cluster | 45 (37 owned) | ARM 12, EXPERIMENT 3, OLD-PATH 30 | 26 | 566 |
| expdel2-13-cluster | 43 (31 owned) | ARM 8, EXPERIMENT 2, OLD-PATH 33 | 23 | 452 |
| expdel2-14-cluster | 45 (39 owned) | ARM 21, OLD-PATH 24 | 11 | 559 |
| expdel2-15-cluster | 42 (29 owned) | ARM 17, EXPERIMENT 2, OLD-PATH 23 | 18 | 356 |
| expdel2-16-decomposition | 45 (25 owned) | ARM 11, EXPERIMENT 1, OLD-PATH 33 | 22 | 429 |
| expdel2-17-decomposition | 42 (24 owned) | ARM 19, EXPERIMENT 1, OLD-PATH 22 | 22 | 369 |
| expdel2-18-decomposition | 35 (24 owned) | ARM 13, OLD-PATH 22 | 7 | 292 |
| expdel2-19-kernel | 39 (34 owned) | ARM 15, EXPERIMENT 2, OLD-PATH 22 | 22 | 683 |
| expdel2-20-linear | 45 (37 owned) | ARM 14, EXPERIMENT 2, OLD-PATH 29 | 22 | 579 |
| expdel2-21-linear | 44 (35 owned) | ARM 17, EXPERIMENT 5, OLD-PATH 22 | 26 | 446 |
| expdel2-22-linear | 45 (37 owned) | ARM 15, EXPERIMENT 6, OLD-PATH 24 | 14 | 319 |
| expdel2-23-neural | 41 (32 owned) | ARM 24, OLD-PATH 17 | 20 | 367 |
| expdel2-24-neural | 44 (31 owned) | ARM 22, OLD-PATH 22 | 20 | 451 |
| expdel2-25-neural | 36 (29 owned) | ARM 31, OLD-PATH 5 | 26 | 662 |
| expdel2-26-neural | 45 (27 owned) | ARM 29, OLD-PATH 16 | 13 | 575 |
| expdel2-27-neural | 32 (28 owned) | ARM 23, OLD-PATH 9 | 10 | 441 |
| expdel2-28-neural | 42 (16 owned) | ARM 16, OLD-PATH 26 | 15 | 422 |
| expdel2-29-neural | 45 (39 owned) | ARM 11, OLD-PATH 34 | 17 | 673 |
| expdel2-30-neural | 37 (29 owned) | ARM 21, OLD-PATH 16 | 10 | 369 |
| expdel2-31-neural | 45 (30 owned) | ARM 35, OLD-PATH 10 | 19 | 326 |
| expdel2-32-neural | 45 (31 owned) | ARM 36, OLD-PATH 9 | 27 | 652 |
| expdel2-33-neural | 33 (29 owned) | ARM 26, OLD-PATH 7 | 6 | 208 |
| expdel2-34-neural | 43 (38 owned) | ARM 36, OLD-PATH 7 | 1 | 1059 |
| expdel2-35-neural | 41 (31 owned) | ARM 25, OLD-PATH 16 | 5 | 363 |
| expdel2-36-neural-series-linear | 38 (31 owned) | ARM 14, EXPERIMENT 3, OLD-PATH 21 | 13 | 496 |
| expdel2-37-series | 44 (38 owned) | ARM 16, EXPERIMENT 3, OLD-PATH 25 | 25 | 472 |
| expdel2-38-trees | 45 (34 owned) | ARM 25, EXPERIMENT 6, OLD-PATH 14 | 9 | 194 |
| expdel2-39-trees | 32 (28 owned) | ARM 28, OLD-PATH 4 | 6 | 218 |
| expdel2-40-trees | 40 (33 owned) | ARM 22, EXPERIMENT 4, OLD-PATH 14 | 1 | 226 |
| expdel2-41-trees | 45 (42 owned) | ARM 21, EXPERIMENT 23, OLD-PATH 1 | 5 | 143 |
| expdel2-42-trees | 44 (32 owned) | ARM 13, EXPERIMENT 18, OLD-PATH 13 | 13 | 246 |
| expdel2-43-trees | 43 (21 owned) | ARM 18, EXPERIMENT 12, OLD-PATH 13 | 11 | 286 |
| expdel2-44-trees | 44 (26 owned) | ARM 9, EXPERIMENT 13, OLD-PATH 22 | 17 | 596 |
| expdel2-45-trees | 45 (40 owned) | ARM 28, EXPERIMENT 3, OLD-PATH 14 | 5 | 136 |
| expdel2-46-trees | 33 (29 owned) | ARM 15, EXPERIMENT 5, OLD-PATH 13 | 8 | 332 |
| expdel2-47-promote | 8 (5 owned) | OLD-PATH 7, PROMOTE 1 | 1 | 68 |

### expdel2-01-base (26 defines, 52 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_537_GEMM_IDENT_SWAP` | DELETE-ARM | a | 1+2 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_SAMBA_FUSE` | DELETE-ARM | e | 1+4 | 102 | EXP:2363 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_ANN3_DIRECT_OUT` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` | DELETE-OLD-PATH | e | 1+11 | 18 | EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DT_BINS_QOLD` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:1285 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_ET_SCALAR_STAGE_COMPARE` | DELETE-ARM | a | 3+0 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_DEVICE_FINITE_OFF` | DELETE-OLD-PATH | d | 1+12 | 23 | EXP:106 KEPT 69f7a41fd -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2` | DELETE-ARM | e | 2+17 | 26 | src# # descriptor op 16), the measured winner: IDN_SAMBA_RESIDENT_STEP PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_GPC_OVR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GPR_PTR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_RESIDENT_OFF` | DELETE-OLD-PATH | b | 2+9 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IF_ROWMAJOR_OFF` | DELETE-OLD-PATH | d | 1+6 | 12 | EXP:115 KEPT 269ffa57a -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | grid fg-ivf.json new default (read_ivf.md A3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | grid fg-ivf.json new default (read_ivf.md A7) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RIDGE_RESIDENT_OFF` | DELETE-OLD-PATH | d | 2+10 | 29 | EXP:247 KEPT 2caff7b0c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TRANSFORMER_FRESH_PREFILL` | DELETE-ARM | a | 1+0 | 1 | EXP:2376 PENDING (A = 0.8.66's session fresh prefill is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:2355 DELETED (superseded) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF` | DELETE-OLD-PATH | e | 1+5 | 8 | EXP:520 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` | DELETE-OLD-PATH | e | 1+4 | 10 | EXP:888 DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_PREP_FAST_FIT_TRANSFORM_FUSED` | DELETE-ARM | a | 1+2 | 4 | EXP:1597 OPEN -- no A/B recorded (Apple fast opt-in) |

Files: `autograd/program_save.mojo`, `bench/attention_step_price_main.mojo`, `bench/gemm_card_main.mojo`, `bench/gemm_excp_ab_main.mojo`, `bench/gemm_int15_apple_tuned_price_main.mojo`, `bench/gemm_ladder_main.mojo`, `bench/gemm_lowbit_amd_price_main.mojo`, `bench/gemm_schedule_ab.mojo`, `bench/gemm_step_price_main.mojo`, `bench/knn_index_layout_main.mojo`, `bench/knn_reference_price_main.mojo`, `bench/knn_vector_request_main.mojo`, `bench/linalg_trace_main.mojo`, `bench/speed/gemm_speed_main.mojo`, `bench/umap_phase_price_main.mojo`, `bench/unsupervised_trace_main.mojo`, `bindings/_mojolearn_embedding_host.mojo`, `bindings/_mojolearn_estimators.mojo`, `bindings/_mojolearn_estimators_host.mojo`, `bindings/_mojolearn_gp.mojo`, `bindings/_mojolearn_ivf_host.mojo`, `bindings/_mojolearn_mamba.mojo`, `bindings/_mojolearn_metrics_host.mojo`, `bindings/_mojolearn_preprocessing.mojo`, `bindings/_mojolearn_preprocessing_host.mojo`, `bindings/_mojolearn_resample_host.mojo`, `bindings/_mojolearn_rf.mojo`, `bindings/_mojolearn_runtime_byte_trainer_host.mojo`, `bindings/_mojolearn_runtime_catalog_host.mojo`, `bindings/_mojolearn_runtime_classical_host.mojo`, `bindings/_mojolearn_runtime_decomp_host.mojo`, `bindings/_mojolearn_runtime_family_host.mojo`, `bindings/_mojolearn_runtime_linear_host.mojo`, `bindings/_mojolearn_runtime_manifold_host.mojo`, `bindings/_mojolearn_runtime_model_selection_host.mojo`, `bindings/_mojolearn_runtime_neural_host.mojo`, `bindings/_mojolearn_runtime_parallel_host.mojo`, `bindings/_mojolearn_runtime_prep_host.mojo`, `bindings/_mojolearn_runtime_sequential_host.mojo`, `bindings/_mojolearn_runtime_trees_host.mojo`, `bindings/_mojolearn_runtime_xseq_host.mojo`, `bindings/_mojolearn_svm.mojo`, `bindings/_mojolearn_training.mojo`, `bindings/_mojolearn_transformer.mojo`, `bindings/_mojolearn_trees.mojo`, `bindings/_mojolearn_x_ann.mojo`, `bindings/_mojolearn_x_ann_host.mojo`, `bindings/_mojolearn_x_cluster.mojo`, `bindings/_mojolearn_x_cluster_host.mojo`, `bindings/_mojolearn_x_decomp_host.mojo`, `bindings/_mojolearn_x_linear_host.mojo`, `bindings/_mojolearn_x_metrics_host.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `DT_BINS_QOLD`, `ET_SCALAR_STAGE_COMPARE`, `IDN_IF_RESIDENT_OFF`, `MAMBA3_LEGACY_FRESH_PREFILL`, `TRANSFORMER_FRESH_PREFILL`, `TRANSFORMER_LEGACY_FRESH_PREFILL`

### expdel2-02-base (44 defines, 22 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_134_CONTROL` | DELETE-ARM | a | 2+3 | 56 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2030_FUSED_EST_MOVE` | DELETE-ARM | e | 1+6 | 45 | EXP:1611 DROP_REDUNDANT, alias deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL` | DELETE-ARM | a | 1+3 | 16 | EXP:2415 PENDING (A = unfilled store is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C04_LDA` | DELETE-ARM | e | 1+6 | 7 | EXP:1532 DROPPED-serial -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C08_ONEHOT_FT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_ORDINAL_FT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C56_LDA_INPUT` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DART_ROUND_REF_OFF` | DELETE-OLD-PATH | b | 1+1 | 4 | EXP:2393 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_FOREST_SEPARATE_NODES` | DELETE-ARM | e | 1+3 | 4 | EXP:1504 DROP, defines deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_AF_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+9 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CLASS_ONEPASS_OFF` | DELETE-OLD-PATH | b | 1+8 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_STEP_OLD` | DELETE-OLD-PATH | b | 2+2 | 6 | EXP:2423 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GRAM_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+9 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GRAM_ROWS_512` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GRAM_ROWS_8192` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GRAM_ROWTILE` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_HPDEV_INIT_OFF` | DELETE-OLD-PATH | b | 2+5 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_NORMAL_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_FIT_EXPORT` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_IF_FIT_OWN_X` | DELETE-ARM | a | 1+1 | 2 | EXP:2396 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_LABEL_OFF` | DELETE-OLD-PATH | b | 1+13 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LM_RESIDENT_GENERATE` | DELETE-ARM | a | 2+3 | 6 | EXP:2262 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD` | DELETE-OLD-PATH | b | 1+4 | 10 | EXP:2418 PENDING (A = no per-call stage allocation, no h_last copy, no host state round t -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:2421 PENDING (A = 0.8.55's `_mamba1_run_arena` / `_mamba2_run_arena` form is the defa -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_CSR_DENSE_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_CSR_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_ONEPASS_OFF` | DELETE-OLD-PATH | b | 1+8 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PT_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RR_EIGH_OFF` | DELETE-OLD-PATH | b | 1+14 | 19 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SELECT_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL` | DELETE-ARM | a | 1+2 | 5 | EXP:2409 PENDING (A = 0.8.55's unfilled outputs is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MLP_SESSIONS_PACKED` | DELETE-ARM | a | 2+2 | 50 | EXP:2381 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MSEL_UNITS_GPU_PARALLEL` | DELETE-ARM | e | 1+1 | 3 | EXP:2371 DEFAULT OFF (crashes on Metal today: member fits share core/neural_context.mojo' -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_NB_TEXT_CSR_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:290 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_NEURAL_TABLE_ROUTE_OFF` | DELETE-OLD-PATH | b | 1+9 | 34 | EXP:2391 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_PREP3_MAXABS_OFF` | DELETE-OLD-PATH | d | 1+9 | 11 | EXP:383 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PROBA64_QOLD` | DELETE-OLD-PATH | e | 2+18 | 24 | EXP:1284 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SAMBA_PROGRAM_FORWARD` | DELETE-ARM | a | 1+1 | 2 | EXP:2399 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:1789 owed (nv + amd RACE moe, OFF arm vs default) -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_T38_FUSED_LABELS` | DELETE-ARM | a | 1+6 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_PREP_FAST_CLS2_PACK_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:406 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_PREP_FAST_CLS2_PRESENT_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:407 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |

Files: `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_x_prep_host.mojo`, `bindings/_mojolearn_x_sequence.mojo`, `bindings/_mojolearn_x_sequence_host.mojo`, `bindings/_mojolearn_x_trees_host.mojo`, `bindings/byte_lm_trainer_native_door.mojo`, `bindings/causal_lm_public_door.mojo`, `bindings/forest_inference_binding_door.mojo`, `bindings/hotpath_device_door.mojo`, `bindings/native_unit_threads_unpack.mojo`, `bindings/neural_block_public_door.mojo`, `bindings/neural_prefill_native_door.mojo`, `bindings/samba_inference_native_door.mojo`, `bindings/small_mlp_native_door.mojo`, `bindings/training_extra_native_door.mojo`, `bindings/trees_dart_rounds_run.mojo`, `bindings/trees_public_iforest_door.mojo`, `checks/determinism_soak.mojo`, `checks/forest_inference_model.mojo`, `checks/gbdt_fused_fit_check.mojo`, `checks/gbdt_large_partition_check.mojo`, `checks/gbdt_step_cost_check.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `134_CONTROL`, `BYTE_LM_LOGITS_ZERO_FILL`, `DART_ROUND_REF_OFF`, `IDN_GBDT_STEP_OLD`, `IDN_HPDEV_INIT_OFF`, `IDN_HPDEV_NORMAL_OFF`, `IDN_IF_FIT_EXPORT`, `IDN_IF_FIT_OWN_X`, `IDN_LM_RESIDENT_GENERATE`, `MAMBA3_FORWARD_ZERO_FILL`, `MLP_SESSIONS_PACKED`, `MSEL_UNITS_GPU_PARALLEL`, `NEURAL_TABLE_ROUTE_OFF`, `SEQ_MOE_DEVICE_IO_OFF`

Lane result (expdel2-02-base, 2026-11-15): KEPT, not deleted: `COLUMN_CPU` (the CPU-column build switch, coordinator correction) and `134_CONTROL` (the DEVIATION 134f positive control, a sabotage arm of checks/determinism_soak.mojo and greedy_search_helper.mojo, in the KEEP set). The 13 other owned defines are deleted with their records; 10 non-owner defines are folded in this lane's files; the remaining 19 have only mode-dependent alias reads here, which stay as they are.

### expdel2-03-base (43 defines, 198 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_2031_SYM_RIDX_SPLITS` | DELETE-ARM | a | 1+8 | 22 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF` | DELETE-OLD-PATH | b | 1+8 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_2560_MEMSET_FILL` | DELETE-ARM | a | 2+2 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFCL_T08` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P02` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P03` | DELETE-EXPERIMENT | g | 1+1 | 2 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P05` | DELETE-EXPERIMENT | g | 1+3 | 45 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P06` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P07` | DELETE-EXPERIMENT | g | 1+1 | 2 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P08` | DELETE-EXPERIMENT | g | 1+1 | 2 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P09` | DELETE-EXPERIMENT | g | 1+1 | 2 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P12` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT` | DELETE-ARM | a | 4+0 | 41 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FAST_BLOCK_SCAN_SHUFFLE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_FOREST_FAST_SHARED_ROWS` | DELETE-ARM | a | 1+4 | 52 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_FOREST_SCALAR_GROVES` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_SEPARATE_NODES` | DELETE-ARM | e | 1+3 | 4 | EXP:1504 DROP, defines deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_FOREST_SHARED_ROWS` | DELETE-ARM | a | 1+4 | 52 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CACHE_CTX_KEY_DEVICE_ID` | DELETE-ARM | a | 1+0 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CACHE_CTX_KEY_OFF` | DELETE-OLD-PATH | b | 1+2 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_FOLDS_OFF` | DELETE-OLD-PATH | b | 2+13 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+4 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QUERY_X_HELD_OFF` | DELETE-OLD-PATH | b | 1+7 | 17 | EXP:2419 RUN OWED (nv+amd, owed row fix-svc-query-resident, B arm query-x-held-off) -- no A/B recorded for the old path |
| `MOJOLEARN_MULTIOUT_RIDGE_OFF` | DELETE-OLD-PATH | d | 1+7 | 11 | EXP:244 KEPT 24ed76679 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MUTEX_SPIN_RELAXED` | DELETE-ARM | a | 1+1 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_MUTEX_CLAIM_STOCK` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_C50_GB_PACKED` | DELETE-ARM | a | 1+12 | 105 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_FOREST_ROUTE` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS` | DELETE-ARM | a | 1+2 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T32_SHARED_ROWS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T33_COST_SCHEDULE` | DELETE-ARM | a | 1+3 | 17 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T34_CHUNK_FOLD` | DELETE-ARM | a | 1+10 | 80 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T37_WORKSPACE` | DELETE-ARM | a | 1+6 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T38_FUSED_LABELS` | DELETE-ARM | a | 1+6 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T39_AUXILIARY_WALK` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T44_METADATA_CACHE` | DELETE-ARM | a | 2+4 | 24 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XPREP_ARENA_POOL_OFF` | DELETE-OLD-PATH | e | 2+5 | 9 | EXP:1787 CONFIRMED (kept default) -- record read: the default is the recorded winner or the kept path |

Files: `checks/gbdt_sym_ridx_fit_check.mojo`, `checks/native_tests/adafactor.mojo`, `checks/native_tests/agglo_native.mojo`, `checks/native_tests/ann_cagra_filtered.mojo`, `checks/native_tests/ann_ivf_surface.mojo`, `checks/native_tests/ann_ivfpq_host_filter.mojo`, `checks/native_tests/ann_ops.mojo`, `checks/native_tests/arima_state.mojo`, `checks/native_tests/bf16_conversion.mojo`, `checks/native_tests/bf16_linalg.mojo`, `checks/native_tests/byte_lm_host_column.mojo`, `checks/native_tests/byte_lm_models_loader_f16.mojo`, `checks/native_tests/byte_lm_parallel_byte_native.mojo`, `checks/native_tests/byte_lm_trainer_session.mojo`, `checks/native_tests/causal_lm_entry_suite.mojo`, `checks/native_tests/causal_lm_session.mojo`, `checks/native_tests/centroid_fit_native.mojo`, `checks/native_tests/cluster_kmeans_query.mojo`, `checks/native_tests/cnn_batchnorm.mojo`, `checks/native_tests/cnn_blocks.mojo`, `checks/native_tests/cnn_conv.mojo`, `checks/native_tests/cnn_pool.mojo`, `checks/native_tests/cnn_trainer.mojo`, `checks/native_tests/core_chunked_lm_head_denormals.mojo`, `checks/native_tests/core_convert.mojo`, `checks/native_tests/core_helpers.mojo`, `checks/native_tests/core_import_lower.mojo`, `checks/native_tests/core_ne2_rounds.mojo`, `checks/native_tests/core_ops_import_kinds.mojo`, `checks/native_tests/core_ops_layernorm_view.mojo`, `checks/native_tests/core_ops_program.mojo`, `checks/native_tests/core_ops_session.mojo`, `checks/native_tests/core_partitioned_program.mojo`, `checks/native_tests/core_program_fit_stop.mojo`, `checks/native_tests/core_program_loop.mojo`, `checks/native_tests/core_ragged_lengths.mojo`, `checks/native_tests/core_safetensors_f16.mojo`, `checks/native_tests/core_stratified_group_fp64.mojo`, `checks/native_tests/core_verifier_gap_methods.mojo`, `checks/native_tests/core_vjp_rank.mojo`, `checks/native_tests/dbscan_native.mojo`, `checks/native_tests/dbscan_ownership.mojo`, `checks/native_tests/decomp_classical_infer_one_call.mojo`, `checks/native_tests/decomp_entry_table_suite.mojo`, `checks/native_tests/decomp_holtwinters_inference.mojo`, `checks/native_tests/decomp_ne2_g_decomp_entries.mojo`, `checks/native_tests/eigh_route.mojo`, `checks/native_tests/embedding_ne3_entries.mojo`, `checks/native_tests/embedding_surface.mojo`, `checks/native_tests/exec_owned_reuse_suite.mojo`, `checks/native_tests/gbdt_ova_link.mojo`, `checks/native_tests/gbdt_public_fits.mojo`, `checks/native_tests/gbdt_resident_ns_predict.mojo`, `checks/native_tests/gp_entry_table.mojo`, `checks/native_tests/gp_native.mojo`, `checks/native_tests/gp_normalization.mojo`, `checks/native_tests/gp_theta.mojo`, `checks/native_tests/gpc_score.mojo`, `checks/native_tests/gpr_score.mojo`, `checks/native_tests/hdbscan_native.mojo`, `checks/native_tests/hdbscan_surface.mojo`, `checks/native_tests/holtwinters_state.mojo`, `checks/native_tests/iforest_forest_export.mojo`, `checks/native_tests/int8_linalg.mojo`, `checks/native_tests/ipca_native.mojo`, `checks/native_tests/kde_native.mojo`, `checks/native_tests/kernel_methods_cells.mojo`, `checks/native_tests/kernel_methods_krr_options.mojo`, `checks/native_tests/kmeans_native.mojo`, `checks/native_tests/kmeans_variants_native.mojo`, `checks/native_tests/kpca_native.mojo`, `checks/native_tests/labels_threshold.mojo`, `checks/native_tests/linalg_decompositions.mojo`, `checks/native_tests/linalg_numerical.mojo`, `checks/native_tests/linear_owner.mojo`, `checks/native_tests/linear_svc_native.mojo`, `checks/native_tests/linear_svr_native.mojo`, `checks/native_tests/lof_native.mojo`, `checks/native_tests/logistic_label_weights.mojo`, `checks/native_tests/logistic_numeric.mojo`, `checks/native_tests/lowbit_weights.mojo`, `checks/native_tests/mamba12_forward_exec.mojo`, `checks/native_tests/mamba3_output_coverage.mojo`, `checks/native_tests/mamba3_prefill_session.mojo`, `checks/native_tests/mamba3_prefix_step.mojo`, `checks/native_tests/mamba_decode_sessions.mojo`, `checks/native_tests/mani_decomp_repeat.mojo`, `checks/native_tests/metrics_selection_host_p2.mojo`, `checks/native_tests/metrics_smallfam_public.mojo`, `checks/native_tests/metrics_weighted_score.mojo`, `checks/native_tests/mixture_gmm_sample.mojo`, `checks/native_tests/mixture_native.mojo`, `checks/native_tests/native_device_concurrency_suite.mojo`, `checks/native_tests/nearest_centroid_native.mojo`, `checks/native_tests/nearest_centroid_numeric.mojo`, `checks/native_tests/neighbors_knn_metrics.mojo`, `checks/native_tests/neighbors_nearest_centroid_native_capability.mojo`, `checks/native_tests/neighbors_ops.mojo`, `checks/native_tests/neighbors_umap_surface.mojo`, `checks/native_tests/neighbors_weighted_largek.mojo`, `checks/native_tests/neural_entry_suite.mojo`, `checks/native_tests/nmf_native.mojo`, `checks/native_tests/ocsvm_native.mojo`, `checks/native_tests/oob_exact_sum.mojo`, `checks/native_tests/ops_higher.mojo`, `checks/native_tests/ops_optimizer.mojo`, `checks/native_tests/ops_optimizer_requests.mojo`, `checks/native_tests/ops_session.mojo`, `checks/native_tests/optics_native.mojo`, `checks/native_tests/optimizer_descriptor.mojo`, `checks/native_tests/optimizer_device.mojo`, `checks/native_tests/optimizer_maximize.mojo`, `checks/native_tests/optimizer_owner.mojo`, `checks/native_tests/pca_native.mojo`, `checks/native_tests/pca_owned_ops.mojo`, `checks/native_tests/pca_whitening.mojo`, `checks/native_tests/pcs_native.mojo`, `checks/native_tests/prep_entry_table_suite.mojo`, `checks/native_tests/prep_label_binarizer_inverse_window.mojo`, `checks/native_tests/prep_robust_scaler_native_lifecycle.mojo`, `checks/native_tests/prophet_state.mojo`, `checks/native_tests/query_x_held.mojo`, `checks/native_tests/rankn_conv.mojo`, `checks/native_tests/rankn_graph.mojo`, `checks/native_tests/rankn_math.mojo`, `checks/native_tests/rankn_views.mojo`, `checks/native_tests/resample_cross_vendor_fold.mojo`, `checks/native_tests/resample_cv_row_order.mojo`, `checks/native_tests/ridge_multi_entry_suite.mojo`, `checks/native_tests/run_cpu.mojo`, `checks/native_tests/run_gpu.mojo`, `checks/native_tests/runner.mojo`, `checks/native_tests/sequence_croston.mojo`, `checks/native_tests/sequence_garch.mojo`, `checks/native_tests/sequence_moe.mojo`, `checks/native_tests/sequence_norm.mojo`, `checks/native_tests/sequence_optimizer.mojo`, `checks/native_tests/sequence_rnn.mojo`, `checks/native_tests/sequence_samba_surface.mojo`, `checks/native_tests/sequence_stl.mojo`, `checks/native_tests/sequence_theta.mojo`, `checks/native_tests/sequence_var.mojo`, `checks/native_tests/solver_estimators_ops.mojo`, `checks/native_tests/solver_qn_objectives.mojo`, `checks/native_tests/spectral_assignment.mojo`, `checks/native_tests/spectral_dense.mojo`, `checks/native_tests/spectral_native.mojo`, `checks/native_tests/spectral_prediction.mojo`, `checks/native_tests/spectral_rbf.mojo`, `checks/native_tests/spectral_resident.mojo`, `checks/native_tests/svc_accuracy_native.mojo`, `checks/native_tests/svc_probability.mojo`, `checks/native_tests/svr_native.mojo`, `checks/native_tests/theta_state.mojo`, `checks/native_tests/training_autograd_model.mojo`, `checks/native_tests/training_autograd_save_resume.mojo`, `checks/native_tests/training_autograd_usability.mojo`, `checks/native_tests/training_chunked_lm_head_door.mojo`, `checks/native_tests/training_distributed_program.mojo`, `checks/native_tests/training_moe_backward_ownership.mojo`, `checks/native_tests/training_neural_inference.mojo`, `checks/native_tests/training_neural_rng_slices.mojo`, `checks/native_tests/training_primitives.mojo`, `checks/native_tests/training_small_mlp.mojo`, `checks/native_tests/transformer_block_session.mojo`, `checks/native_tests/transformer_decode_door.mojo`, `checks/native_tests/transformer_host_block.mojo`, `checks/native_tests/transformer_ops_program.mojo`, `checks/native_tests/trees_entry_glue_suite.mojo`, `checks/native_tests/tsa_holtwinters_initialization.mojo`, `checks/native_tests/tsa_sequence_ets.mojo`, `checks/native_tests/tsa_training_arima.mojo`, `checks/native_tests/tsvd_fit.mojo`, `checks/native_tests/tsvd_native.mojo`, `checks/native_tests/umap_ops.mojo`, `checks/native_tests/x_linear_ops.mojo`, `checks/native_tests/xcnn_entry_table_suite.mojo`, `checks/rtf_seam.mojo`, `checks/vendor_correctness_check.mojo`, `cholesky/checks/fast_trsm.mojo`, `core/apple_fast_tree_experiments.mojo`, `core/base_helpers_device.mojo`, `core/block_reduce.mojo`, `core/block_scan.mojo`, `core/byte_lm_entry_table_sessions.mojo`, `core/ctx_key.mojo`, `core/device_arena.mojo`, `core/device_mutex.mojo`, `core/device_query_x.mojo`, `core/device_store.mojo`, `core/device_zero.mojo`, `core/device_zero_check.mojo`, `core/entry_table.mojo`, `core/forest_auxiliary_units.mojo`, `core/forest_experiments.mojo`, `core/forest_host_groves.mojo`, `core/forest_host_predict.mojo`, `core/forest_inference.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2560_MEMSET_FILL`, `AFCL_T08`, `AFT_P01`, `AFT_P02`, `AFT_P03`, `AFT_P04`, `AFT_P05`, `AFT_P06`, `AFT_P07`, `AFT_P08`, `AFT_P09`, `AFT_P10`, `AFT_P11`, `AFT_P12`, `ENTRY_TABLE_HEAP_OUTPUT`, `FAST_BLOCK_SCAN_SHUFFLE`, `FOREST_FAST_SHARED_ROWS`, `FOREST_SCALAR_GROVES`, `FOREST_SEPARATE_NODES`, `FOREST_SHARED_ROWS`, `IDN_CACHE_CTX_KEY_DEVICE_ID`, `IDN_CACHE_CTX_KEY_OFF`, `IDN_HPDEV_FOLDS_OFF`, `MUTEX_SPIN_RELAXED`, `RF_MUTEX_CLAIM_STOCK`, `TREES_C50_GB_PACKED`, `TREES_FOREST_ROUTE`, `TREES_PREDICT_ROW_OUTPUTS`, `TREES_T32_SHARED_ROWS`, `TREES_T33_COST_SCHEDULE`, `TREES_T34_CHUNK_FOLD`, `TREES_T37_WORKSPACE`, `TREES_T38_FUSED_LABELS`, `TREES_T39_AUXILIARY_WALK`, `TREES_T44_METADATA_CACHE`, `XPREP_ARENA_POOL_OFF`

### expdel2-04-base (45 defines, 25 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_2711_GRAM_STRIDED_DEAD` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_537_GEMM_IDENT_SWAP` | DELETE-ARM | a | 1+2 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_APPLE_GEMM_NT_MMA_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_APPLE_GEMM_NT_TILED_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_APPLE_STEP_UNROLL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_CLASSICAL_C06_NORM_ROWS` | DELETE-ARM | a | 1+3 | 4 | EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD` | DELETE-ARM | a | 1+2 | 10 | EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT` | DELETE-ARM | a | 4+0 | 41 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FAST_LCG_MODULO` | DELETE-ARM | a | 1+4 | 7 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_FOREST_DEVICE_FINITE_OFF` | DELETE-OLD-PATH | d | 1+12 | 23 | EXP:106 KEPT 69f7a41fd -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_FOREST_ORDERED_RESIDENT` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_FOREST_PER_MODEL_IO` | DELETE-ARM | a | 1+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_PINNED_STAGE` | DELETE-ARM | a | 1+4 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GEMV_FAST_VENDOR` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GPC_OVR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GPR_PTR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_CAST_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_ELEM_OFF` | DELETE-OLD-PATH | b | 2+6 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_FOLDS_OFF` | DELETE-OLD-PATH | b | 2+13 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_ISUM_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_LABELS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_REDUCE_OFF` | DELETE-OLD-PATH | b | 2+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:2422 PENDING (A = caps cached on the session, one held call per forward) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OLS_PREDICT_TILED` | DELETE-ARM | a | 1+1 | 2 | EXP:293 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_QUERY_X_HELD_OFF` | DELETE-OLD-PATH | b | 1+7 | 17 | EXP:2419 RUN OWED (nv+amd, owed row fix-svc-query-resident, B arm query-x-held-off) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SCRATCH_POOL` | DELETE-ARM | a | 1+9 | 62 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SCRATCH_POOL_OFF` | DELETE-OLD-PATH | b | 1+9 | 62 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEG_RADIX8_OFF` | DELETE-OLD-PATH | b | 1+2 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XTY_TILE_1024` | DELETE-ARM | a | 1+11 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_XTY_TILE_64` | DELETE-ARM | a | 1+11 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KDE_FAST_IMMUTABLE_FIT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_MULTIOUT_RIDGE_OFF` | DELETE-OLD-PATH | d | 1+7 | 11 | EXP:244 KEPT 24ed76679 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_NV_AMD_STEPS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_FAST_BATCH32K` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SEG_SUMS_SERIAL` | DELETE-ARM | a | 1+4 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_STEP_GLUE_TRIAL` | DELETE-ARM | a | 1+7 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_C50_GB_PACKED` | DELETE-ARM | a | 1+12 | 105 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T34_CHUNK_FOLD` | DELETE-ARM | a | 1+10 | 80 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T37_WORKSPACE` | DELETE-ARM | a | 1+6 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T38_FUSED_LABELS` | DELETE-ARM | a | 1+6 | 30 | no record -- no A/B recorded for the opt-in arm |

Files: `core/forest_inference_model.mojo`, `core/forest_inference_run.mojo`, `core/gbdt_host_predict.mojo`, `core/gemm.mojo`, `core/gp_native_device.mojo`, `core/gram_splitk.mojo`, `core/host_tile_fold.mojo`, `core/hotpath_device_run.mojo`, `core/identical_callpath.mojo`, `core/kde_native_device.mojo`, `core/label_device_steps.mojo`, `core/linear_multi_device.mojo`, `core/mojolearn_rf_run.mojo`, `core/msel_helpers_device.mojo`, `core/native_descriptor_call.mojo`, `core/native_descriptor_fixed_call.mojo`, `core/neural_entry_table.mojo`, `core/query_x_held.mojo`, `core/row_norms.mojo`, `core/scan_by_key.mojo`, `core/scratch_pool.mojo`, `core/segmented_sort.mojo`, `core/shuffle_iterator.mojo`, `core/step_glue.mojo`, `core/strided_walk.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2711_GRAM_STRIDED_DEAD`, `537_GEMM_IDENT_SWAP`, `APPLE_FAST_STEP_UNROLL_OFF`, `APPLE_GEMM_NT_MMA_OFF`, `APPLE_GEMM_NT_TILED_OFF`, `APPLE_STEP_UNROLL_OFF`, `EXPERIMENT_IDENTICAL_CALLPATH`, `FAST_LCG_MODULO`, `FOREST_ORDERED_RESIDENT`, `FOREST_ORDERED_RESIDENT_OFF`, `FOREST_PER_MODEL_IO`, `FOREST_PINNED_STAGE`, `GEMV_FAST_VENDOR`, `IDN_HPDEV_CAST_OFF`, `IDN_HPDEV_ELEM_OFF`, `IDN_HPDEV_ISUM_OFF`, `IDN_HPDEV_LABELS_OFF`, `IDN_HPDEV_REDUCE_OFF`, `IDN_MAMBA3_FORWARD_DOOR_OLD`, `IDN_OLS_PREDICT_TILED`, `IDN_QUERY_X_HELD_OFF`, `IDN_SCRATCH_POOL`, `IDN_SCRATCH_POOL_OFF`, `IDN_SEG_RADIX8_OFF`, `IDN_XTY_TILE_1024`, `IDN_XTY_TILE_64`, `NV_AMD_STEPS_OFF`, `RF_FAST_BATCH32K`, `SEG_SUMS_SERIAL`, `STEP_GLUE_TRIAL`

### expdel2-05-base (35 defines, 9 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_C31_DEVICE_BUCKETS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C32_COUNT_FUSION` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C32_EMIT_FUSION` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C33_CHUNK_4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C33_FROZEN_CHUNKS` | DELETE-ARM | e | 1+4 | 5 | EXP:2323 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_C34_PARALLEL_EDGES` | DELETE-ARM | a | 1+3 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C35_PACKED_LISTS` | DELETE-ARM | a | 1+8 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C35_ROWS_128` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C39_RETAIN_STATE` | DELETE-ARM | a | 1+10 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C40_ACTIVE_SEEDS` | DELETE-ARM | a | 1+5 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C40_SEED_TILES` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C41_FUSED_MINIMA` | DELETE-ARM | a | 1+3 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C42_ACTIVE_TRIANGLE` | DELETE-ARM | a | 1+10 | 24 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | DELETE-ARM | a | 1+12 | 54 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C44_SAMPLING_DESCRIPTORS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C62_SAME_COMPONENT_SKIP` | DELETE-ARM | a | 1+3 | 31 | EXP:1560 unmeasured -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C01_MEAN` | DELETE-ARM | e | 1+4 | 14 | EXP:1530 DROPPED-slower -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` | DELETE-OLD-PATH | e | 1+11 | 18 | EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_DEVICE_FINITE_OFF` | DELETE-OLD-PATH | d | 1+12 | 23 | EXP:106 KEPT 69f7a41fd -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_COLVAR_FUSED_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1946 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GFF_GRAM_TILED_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:2096 KEEP (promoted) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1832 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF` | DELETE-OLD-PATH | e | 1+5 | 41 | EXP:1831 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_RIDGE_RESIDENT_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1833 _OFF arm ratios: ridge istella NV 2.22x / AMD 1.28x slower; taxi NV 0.87x / AMD  -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_TSVD_ONE_UPLOAD_OFF` | DELETE-OLD-PATH | e | 1+3 | 12 | EXP:1998 _OFF arm 1.610x combined (slower) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_XTY_TILE_1024` | DELETE-ARM | a | 1+11 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_XTY_TILE_64` | DELETE-ARM | a | 1+11 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IVF_DIRECT_DISTANCE_OFF` | DELETE-OLD-PATH | e | 1+17 | 38 | EXP:1642 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_KDE_DIRECT_DISTANCE` | DELETE-ARM | a | 1+9 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KMEANS_ROW_ASSIGN` | DELETE-ARM | e | 1+3 | 4 | EXP:1927 KEPT (owed row withdrawn) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_PCA_FAST_POOL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1392 READY-AB -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL` | DELETE-ARM | a | 1+4 | 13 | EXP:2416 PENDING (A = per-registry key is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCLUSTER_KPP_DISTINCT` | DELETE-ARM | e | 1+3 | 8 | EXP:1517 DROPPED (deleted) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_XCLUSTER_ROW_ASSIGN` | DELETE-ARM | a | 1+3 | 4 | EXP:1518 4`, and per-family direct-distance defines KNN / KDE / DBSCAN / GRAPH / IVF with -- no A/B recorded for the opt-in arm |

Files: `core/weight_epoch.mojo`, `core/xtdz_coalesced.mojo`, `decomposition/checks/jacobi_sm120a_probe.mojo`, `decomposition/estimator.mojo`, `embedding/checks/embedding_fast_apple.mojo`, `ensemble/checks/acquire_rmw_probe.mojo`, `ensemble/device_finite.mojo`, `experiments/classical_identical_ideas/fg_linear_controls.mojo`, `experiments/classical_identical_ideas/graph_controls.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `C31_DEVICE_BUCKETS`, `C32_COUNT_FUSION`, `C32_EMIT_FUSION`, `C33_CHUNK_4`, `C33_FROZEN_CHUNKS`, `C34_PARALLEL_EDGES`, `C35_PACKED_LISTS`, `C35_ROWS_128`, `C39_RETAIN_STATE`, `C40_ACTIVE_SEEDS`, `C40_SEED_TILES`, `C41_FUSED_MINIMA`, `C42_ACTIVE_TRIANGLE`, `C43_RESIDENT_NORMALIZATION`, `C44_SAMPLING_DESCRIPTORS`, `C62_SAME_COMPONENT_SKIP`, `FOREST_DEVICE_FINITE_OFF`, `IDN_COLVAR_FUSED_OFF`, `IDN_GFF_GRAM_TILED_OFF`, `IDN_GRAM_FF_FALLBACK_OFF`, `IDN_JACOBI_ROUND_ROBIN_OFF`, `IDN_RIDGE_RESIDENT_OFF`, `IDN_TSVD_ONE_UPLOAD_OFF`, `IVF_DIRECT_DISTANCE_OFF`, `KDE_DIRECT_DISTANCE`, `KMEANS_ROW_ASSIGN`, `PCA_FAST_POOL_OFF`, `SAMBA_WEIGHT_EPOCH_GLOBAL`, `XCLUSTER_KPP_DISTINCT`, `XCLUSTER_ROW_ASSIGN`

### expdel2-06-base (28 defines, 1 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS` | DELETE-ARM | a | 1+18 | 46 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C14_GROUP_RHS` | DELETE-ARM | a | 1+5 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C16_GLM_FUSED` | DELETE-ARM | a | 1+7 | 28 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C17_LS_TRIALS` | DELETE-ARM | a | 2+3 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C17_OVR` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT` | DELETE-ARM | a | 1+5 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C19_SGD_CHUNK` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C20_PAIR_LOAD` | DELETE-ARM | a | 1+3 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C21_EXTREMA` | DELETE-ARM | a | 1+2 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C22_TRIANGLE` | DELETE-ARM | a | 1+3 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C23_MCD` | DELETE-ARM | a | 1+3 | 6 | EXP:1533 not measured alone -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C24_PANEL8` | DELETE-ARM | e | 1+2 | 3 | grid classical-decomp.json promoted:ge123e6f9 -- grid C24_TSQR promoted the rows2048 arm; panel8 was a measured arm that was not promoted |
| `MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1638 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_C24_TREE4` | DELETE-ARM | e | 1+6 | 7 | grid classical-decomp.json promoted:ge123e6f9 -- grid C24_TSQR promoted the rows2048 arm; tree4 was a measured arm that was not promoted |
| `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1984 PROMOTED (restored; IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_C26_PRODUCTS` | DELETE-ARM | a | 1+9 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED` | DELETE-ARM | a | 1+5 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_COMPONENTS` | DELETE-ARM | a | 1+3 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_NORM_VECTOR` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES` | DELETE-ARM | a | 1+8 | 31 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS` | DELETE-ARM | e | 1+3 | 4 | EXP:1992 PROMOTED (IDENTICAL default, arm 64) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1992 PROMOTED (IDENTICAL default, arm 64) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS` | DELETE-ARM | e | 1+3 | 4 | EXP:2068 KEEP (promoted IDENTICAL default) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:2068 KEEP (promoted IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` | DELETE-OLD-PATH | e | 1+11 | 18 | EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED` | DELETE-ARM | a | 1+14 | 97 | no record -- no A/B recorded for the opt-in arm |

Files: `experiments/classical_identical_ideas/linear_controls.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `CLASSICAL_C13_LOGCV_WEIGHTS`, `CLASSICAL_C14_GROUP_RHS`, `CLASSICAL_C15_FACTOR_SOLVE`, `CLASSICAL_C16_GLM_FUSED`, `CLASSICAL_C17_LS_TRIALS`, `CLASSICAL_C17_OVR`, `CLASSICAL_C18_RESIDUAL_NEXT`, `CLASSICAL_C19_SGD_CHUNK`, `CLASSICAL_C20_PAIR_LOAD`, `CLASSICAL_C21_EXTREMA`, `CLASSICAL_C22_TRIANGLE`, `CLASSICAL_C23_MCD`, `CLASSICAL_C24_PANEL8`, `CLASSICAL_C24_ROWS2048_OFF`, `CLASSICAL_C24_TREE4`, `CLASSICAL_C25_PROJECTION_REUSE_OFF`, `CLASSICAL_C26_PRODUCTS`, `CLASSICAL_C26_UPDATE_FUSED`, `CLASSICAL_C27_COMPONENTS`, `CLASSICAL_C27_FA_COMPONENTS`, `CLASSICAL_C27_NORM_VECTOR`, `CLASSICAL_C28_BUCKET_SOLVES`, `CLASSICAL_ENETCV_FOLD_BLOCKS`, `CLASSICAL_ENETCV_FOLD_BLOCKS_OFF`, `CLASSICAL_ENETCV_SCORE_BLOCKS`, `CLASSICAL_ENETCV_SCORE_BLOCKS_OFF`, `CLASSICAL_LINEAR_GRAM_SOLVE_OFF`, `CLASSICAL_RIDGECV_FF_BLOCKED`

### expdel2-07-base (45 defines, 50 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFT_P06` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_C52_PAIR_ROWS` | DELETE-ARM | a | 2+15 | 34 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C53_CENTER4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C54_PREDICT_TILES` | DELETE-ARM | a | 1+8 | 45 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C57_CANDIDATE_STATE` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C58_SERIES4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C58_TEAM_MIB` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C59_TRIAL_STATE` | DELETE-ARM | a | 1+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C60_LAG4` | DELETE-ARM | a | 1+5 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C01_LEAF` | DELETE-ARM | e | 1+3 | 4 | EXP:1530 DROPPED-slower -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C01_MEAN` | DELETE-ARM | e | 1+4 | 14 | EXP:1530 DROPPED-slower -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C04_LDA` | DELETE-ARM | e | 1+6 | 7 | EXP:1532 DROPPED-serial -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C05_OLS_PHASE` | DELETE-ARM | a | 1+4 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH` | DELETE-ARM | a | 1+3 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C06_NORM_ROWS` | DELETE-ARM | a | 1+3 | 4 | EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD` | DELETE-ARM | a | 1+2 | 10 | EXP:1519 4`; new `CLASSICAL_C06_SMALL_D_THREAD` (thread per row at d <= 32, incumbent bit -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C07_RADIX_BITS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C07_RADIX_ROWS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT` | DELETE-ARM | a | 1+4 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_ONEHOT_FT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_ORDINAL_FT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:1740 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_C09_REG_BUNDLE` | DELETE-ARM | a | 1+2 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C10_RANK_REUSE` | DELETE-ARM | a | 1+2 | 17 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C11_DRAW_GATHER` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS` | DELETE-ARM | a | 1+2 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C56_LDA_INPUT` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C56_QDA_PROJECT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ATTN_HEAD_SHARE` | DELETE-ARM | e | 2+2 | 5 | EXP:1483 DROP -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_ATTN_STASH` | DELETE-ARM | e | 5+4 | 9 | EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_GPC_OVR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GPR_PTR_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RADIX_TILE128` | DELETE-ARM | a | 3+0 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RAGGED_FLOAT_RADIX` | DELETE-ARM | a | 2+0 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE` | DELETE-ARM | a | 3+0 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TSQR_REUSE` | DELETE-ARM | a | 4+0 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KNN_CERTIFIED_MMA_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MEASURE_I06_GQA` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MEASURE_I07_LIFETIME` | DELETE-ARM | a | 2+0 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRAIN_LIVE_STATUS` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |

Files: `experiments/classical_identical_ideas/shared_controls.mojo`, `experiments/classical_identical_ideas/stats_controls.mojo`, `experiments/identical_callpath/storage.mojo`, `experiments/performance_ideas/I10/check.mojo`, `experiments/performance_ideas/I15/certified_caller.mojo`, `experiments/performance_ideas/I19/check.mojo`, `experiments/performance_ideas/I19/ragged_float.mojo`, `experiments/performance_ideas/I22/reuse_check.mojo`, `experiments/performance_ideas/I22/time.mojo`, `experiments/performance_ideas/attention_time.mojo`, `gaussian_process/checks/kernels.mojo`, `gaussian_process/gpc_ovr.mojo`, `gaussian_process/gpr_resident.mojo`, `gbdt/gpu_util/kernel/reorder_single_pass.mojo`, `gbdt/gpu_util/kernel/segmented_scan.mojo`, `gbdt/models/kernel/resident_link.mojo`, `gemm/checks/amd_excp_probe.mojo`, `gemm/checks/amd_mfma_probe.mojo`, `gemm/checks/amd_mfma_probe2.mojo`, `gemm/checks/amd_mfma_probe3.mojo`, `gemm/checks/gemm_apple_devload_probe1.mojo`, `gemm/checks/gemm_apple_devload_probe2.mojo`, `gemm/checks/gemm_int15_apple.mojo`, `gemm/checks/gemm_int15_apple_tuned.mojo`, `gemm/checks/gemm_int15_apple_tuned_check.mojo`, `gemm/checks/gemm_int15_apple_tuned_dev.mojo`, `gemm/checks/gemm_int15_apple_tuned_threads_probe.mojo`, `gemm/checks/gemm_int15_check.mojo`, `gemm/checks/gemm_int15_sim_check.mojo`, `gemm/checks/gemm_int15_tuned.mojo`, `gemm/checks/gemm_int15_tuned_check.mojo`, `gemm/checks/gemm_int8_apple_chunk.mojo`, `gemm/checks/gemm_int8_apple_chunk_check.mojo`, `gemm/checks/gemm_int8_mma_amd.mojo`, `gemm/checks/gemm_int8_mma_amd_check.mojo`, `gemm/checks/gemm_int8_pieces_tuned_check.mojo`, `gemm/checks/gemm_int_slices_check.mojo`, `gemm/checks/gemm_lowbit_check.mojo`, `gemm/checks/gemm_production_dispatch_check.mojo`, `gemm/checks/gemm_rtf_boundary_check.mojo`, `gemm/checks/gemm_seam_probe.mojo`, `gemm/checks/quantize_int8_par_check.mojo`, `gemm/experiments/async_operand_pipeline_check.mojo`, `gemm/experiments/bounded_staging_check.mojo`, `gemm/experiments/changing_batch_check.mojo`, `gemm/experiments/grouped_jobs_check.mojo`, `gemm/experiments/profile_check.mojo`, `gemm/experiments/rounded_epilogue_check.mojo`, `gemm/host_transport.mojo`, `gemm/tn_centered.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `C52_PAIR_ROWS`, `C53_CENTER4`, `C54_PREDICT_TILES`, `C57_CANDIDATE_STATE`, `C58_SERIES4`, `C58_TEAM_MIB`, `C59_TRIAL_STATE`, `C60_LAG4`, `CLASSICAL_C01_LEAF`, `CLASSICAL_C01_MEAN`, `CLASSICAL_C02_LINEAR_PAIR`, `CLASSICAL_C03_FINITE_EXTREMA`, `CLASSICAL_C04_LDA`, `CLASSICAL_C05_OLS_PHASE`, `CLASSICAL_C05_PHASE_SCRATCH`, `CLASSICAL_C06_NORM_ROWS`, `CLASSICAL_C06_SMALL_D_THREAD`, `CLASSICAL_C07_RADIX_BITS`, `CLASSICAL_C07_RADIX_ROWS`, `CLASSICAL_C08_GROUPED_OUTPUT`, `CLASSICAL_C08_ONEHOT_FT`, `CLASSICAL_C08_ORDINAL_FT`, `CLASSICAL_C08_UNIQUE_SCAN_OFF`, `CLASSICAL_C09_REG_BUNDLE`, `CLASSICAL_C10_RANK_REUSE`, `CLASSICAL_C11_DRAW_GATHER`, `CLASSICAL_C12_SPARSE_COUNTS`, `CLASSICAL_C56_LDA_INPUT`, `CLASSICAL_C56_QDA_PROJECT`, `IDN_ATTN_HEAD_SHARE`, `IDN_ATTN_STASH`, `IDN_GPC_OVR_OFF`, `IDN_GPR_PTR_OFF`, `IDN_RADIX_TILE128`, `IDN_RAGGED_FLOAT_RADIX`, `IDN_RAGGED_RADIX_NIBBLE`, `IDN_TSQR_REUSE`, `MEASURE_I06_GQA`, `MEASURE_I07_LIFETIME`, `TRAIN_LIVE_STATUS`

### expdel2-08-base (43 defines, 49 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_APPLE_STEP_UNROLL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_C31_DEVICE_BUCKETS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C32_COUNT_FUSION` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C32_EMIT_FUSION` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C44_SAMPLING_DESCRIPTORS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C53_CENTER4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C58_TEAM_MIB` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C20_PAIR_LOAD` | DELETE-ARM | a | 1+3 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C21_EXTREMA` | DELETE-ARM | a | 1+2 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C56_QDA_PROJECT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` | DELETE-OLD-PATH | e | 1+11 | 18 | EXP:2015 OLS taxi avg 0.762, istella 0.665 (Gram off faster); ridge taxi 5.76x slower off -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1971 PROMOTED (IDENTICAL default, together) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:1971 PROMOTED (IDENTICAL default, together) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1832 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_HPDEV_ELEM_OFF` | DELETE-OLD-PATH | b | 2+6 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_INIT_OFF` | DELETE-OLD-PATH | b | 2+5 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HPDEV_REDUCE_OFF` | DELETE-OLD-PATH | b | 2+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QUERY_X_HELD_OFF` | DELETE-OLD-PATH | b | 1+7 | 17 | EXP:2419 RUN OWED (nv+amd, owed row fix-svc-query-resident, B arm query-x-held-off) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MULTIOUT_RIDGE_OFF` | DELETE-OLD-PATH | d | 1+7 | 11 | EXP:244 KEPT 24ed76679 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_NATIVE_CONV_BAD_GRADIENT` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NATIVE_OPTIMIZER_BAD_REFERENCE` | DELETE-ARM | a | 1+0 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NATIVE_SVM_BAD_PREDICTION` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PROGRAM_RUN_ENTRY` | DELETE-ARM | e | 1+3 | 4 | EXP:2394 RESTORED (A = run_program, the 0.8.55 route, default) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_PROPHET_COOP_OFF` | DELETE-OLD-PATH | d | 1+4 | 7 | EXP:537 KEPT 8c2b7de51 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_RBC_FAST_EPS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_RIDGE_RESIDENT_OFF` | DELETE-OLD-PATH | d | 2+10 | 29 | EXP:247 KEPT 2caff7b0c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SCRATCH` | DELETE-ARM | a | 1+4 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SESSION_CALL_DESCRIPTOR` | DELETE-ARM | a | 2+2 | 14 | EXP:2395 RESTORED (A = direct typed call when the calling image owns the registry, defaul -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TE_ADA_SESSION_SHARE_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:123 KEPT ed49f2b11 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TE_NATIVE_SPLITS_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:124 KEPT ed49f2b11 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION` | DELETE-ARM | a | 1+1 | 2 | EXP:2406 PENDING (A = 0.8.55's per-token shape: admit x, one y, cached_tokens read/write, -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR` | DELETE-ARM | a | 2+5 | 20 | EXP:2408 PENDING (A = the step input admitted as is) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T44_METADATA_CACHE` | DELETE-ARM | a | 2+4 | 24 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_UMAP_LIVE_BOTH_ARM` | DELETE-ARM | a | 2+10 | 25 | no record -- no A/B recorded for the opt-in arm |

Files: `glm/host/gram_solve_host.mojo`, `glm/host/linear_fit_native.mojo`, `glm/impl/center_device.mojo`, `glm/impl/qn/glm_softmax.mojo`, `glm/impl/ridge_multi.mojo`, `glm/linear_fit_device.mojo`, `kernel_methods/krr_resident.mojo`, `mamba/native_block_dispatch.mojo`, `mixture/checks/mstep.mojo`, `naive_bayes/da.mojo`, `neighbors/checks/ball_cover_canonical_order.mojo`, `neighbors/checks/fused_distance_select_identical.mojo`, `neighbors/checks/fused_logical32_check.mojo`, `neighbors/checks/knn_check.mojo`, `neighbors/checks/knn_distance_fma_boundary_check.mojo`, `neighbors/checks/knn_identity_check.mojo`, `neighbors/checks/lane_minimum_check.mojo`, `neighbors/checks/pinned_distance_tile.mojo`, `neighbors/checks/smem_distance_tile.mojo`, `neighbors/impl/ball_cover/registers.mojo`, `neighbors/impl/detail/fused_l2_knn.mojo`, `neighbors/impl/distance/detail/pairwise_distance_base.mojo`, `neighbors/impl/topk/logical_warp32.mojo`, `neighbors/knn_main.mojo`, `packaging/native_only/cases/autograd_conv_core.mojo`, `packaging/native_only/cases/sequence_optimizer_core.mojo`, `packaging/native_only/cases/svm_core.mojo`, `providers/python/autograd_ops_frontend_pyapi_door.mojo`, `providers/python/autograd_program_public_request_pyapi_pack.mojo`, `providers/python/autograd_program_pyapi_door.mojo`, `providers/python/autograd_session_public_request_pyapi_pack.mojo`, `providers/python/transformer_decode_pyapi_door.mojo`, `providers/python/xtrees_api_pyapi_door.mojo`, `sequence/fit_team_impl.mojo`, `sequence/model_outputs_device.mojo`, `svm/impl/kernelcache.mojo`, `svm/impl/smoblocksolve.mojo`, `tools/amd_codegen/probe_gemm.mojo`, `tools/amd_codegen/probe_mfma_gemm.mojo`, `tools/amd_codegen/probe_step_gemm.mojo`, `tools/identical_wave_gbdt_depth_gate.mojo`, `tools/lowbit_amd_tuned/scratch_first.mojo`, `tools/nvidia_step_time/probe_gemm_ptx.mojo`, `training/byte_lm_binding_impl.mojo`, `training/byte_lm_native_dispatch.mojo`, `training/byte_lm_stateless_run.mojo`, `transformer/attn_step_session.mojo`, `transformer/checks/transformer_fused_check.mojo`, `umap/optimizer_identical_device.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `IDN_DBSCAN_ADJ_BITMAP_OFF`, `IDN_DBSCAN_EPS_TILE_OFF`, `IDN_TREE_SHAPE_NATIVE_OFF`, `MULTIOUT_RIDGE_OFF`, `NATIVE_CONV_BAD_GRADIENT`, `NATIVE_OPTIMIZER_BAD_REFERENCE`, `NATIVE_SVM_BAD_PREDICTION`, `PROGRAM_RUN_ENTRY`, `SCRATCH`, `SESSION_CALL_DESCRIPTOR`, `TE_ADA_SESSION_SHARE_OFF`, `TE_NATIVE_SPLITS_OFF`, `TRANSFORMER_DECODE_PER_TOKEN_ADMISSION`, `TRANSFORMER_DECODE_PER_TOKEN_DOOR`, `UMAP_IDENTICAL_OPT_TPB_256`, `UMAP_IDENTICAL_OPT_TPB_64`

Lane result (expdel2-08-base, 2026-11-15): 10 owned defines deleted with their records (both DBSCAN `_OFF`, `IDN_TREE_SHAPE_NATIVE_OFF`, `MULTIOUT_RIDGE_OFF`, `PROGRAM_RUN_ENTRY`, `SESSION_CALL_DESCRIPTOR`, `TE_ADA_SESSION_SHARE_OFF`, `TE_NATIVE_SPLITS_OFF`, both `TRANSFORMER_DECODE_PER_TOKEN_*`); 16 non-owner defines folded in this lane's files. KEPT, not deleted: `NATIVE_CONV_BAD_GRADIENT`, `NATIVE_OPTIMIZER_BAD_REFERENCE`, `NATIVE_SVM_BAD_PREDICTION` (fixture-only negative controls of packaging/native_only: sabotage arms, in the KEEP set), `SCRATCH` (the A/B toggle of the standalone probe tools/lowbit_amd_tuned/scratch_first.mojo, which scratch_first.sh builds both ways), `UMAP_IDENTICAL_OPT_TPB_64` / `_256` (the launch-width builds of the UMAP fingerprint gate, tools/umap_lane_pod_run.sh), `COLUMN_CPU` (correction) and the `IDN_ALL_OFF` umbrella reads (Method). The other defines are read here only through mode-dependent aliases owned by other lanes' files, which stay; three refusal texts / docstrings no longer name deleted switches.

### expdel2-09-base (30 defines, 19 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_P04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P09` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_C40_ACTIVE_SEEDS` | DELETE-ARM | a | 1+5 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C40_SEED_TILES` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C04_LDA` | DELETE-ARM | e | 1+6 | 7 | EXP:1532 DROPPED-serial -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH` | DELETE-ARM | a | 1+3 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT` | DELETE-ARM | a | 1+4 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C09_REG_BUNDLE` | DELETE-ARM | a | 1+2 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C10_RANK_REUSE` | DELETE-ARM | a | 1+2 | 17 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS` | DELETE-ARM | a | 1+2 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT` | DELETE-ARM | a | 1+5 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS` | DELETE-ARM | e | 1+3 | 4 | EXP:1992 PROMOTED (IDENTICAL default, arm 64) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CLASS_ONEPASS_OFF` | DELETE-OLD-PATH | b | 1+8 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_LEAF` | DELETE-ARM | e | 1+23 | 24 | EXP:1766 PROMOTED for the MLP step only (IDENTICAL default on mlp-train-step); every othe -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_GRAM_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+9 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GRAM_ROWS_512` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GRAM_ROWS_8192` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GRAM_ROWTILE` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_LABEL_OFF` | DELETE-OLD-PATH | b | 1+13 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MEANSHIFT_GRID_OFF` | DELETE-OLD-PATH | b | 1+9 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_CSR_DENSE_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_CSR_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_ONEPASS_OFF` | DELETE-OLD-PATH | b | 1+8 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PT_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SELECT_BLOCKED_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KPCA_RESIDENT_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:923 DEFAULT; M2 default + OFF both rc=0, merged bc61f3560; Python import PASS -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_NB_TEXT_CSR_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:290 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_X_PREP_FAST_CLS2_PACK_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:406 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |

Files: `x_cluster/meanshift_idn.mojo`, `x_decomp/descriptor_mani_device.mojo`, `x_decomp/lle_iter_dev.mojo`, `x_decomp/qr_sliced_device.mojo`, `x_decomp/svd_ops.mojo`, `x_linear/cd.mojo`, `x_linear/enetcv_blocks.mojo`, `x_linear/moments_grid.mojo`, `x_linear/spearman_device.mojo`, `x_metrics/plan.mojo`, `x_neighbors/kapprox_dev.mojo`, `x_prep/blocked.mojo`, `x_prep/fastnb_csr.mojo`, `x_prep/gram_blocked.mojo`, `x_prep/pt_blocked.mojo`, `x_prep/select_blocked.mojo`, `x_prep/units.mojo`, `xtrees/folds_device.mojo`, `xtrees/label_weights_native.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_P04`, `AFCL_P09`, `IDN_CLASS_ONEPASS_OFF`, `IDN_GRAM_BLOCKED_OFF`, `IDN_GRAM_ROWS_512`, `IDN_GRAM_ROWS_8192`, `IDN_GRAM_ROWTILE`, `IDN_NB_CSR_DENSE_OFF`, `IDN_NB_CSR_OFF`, `IDN_NB_ONEPASS_OFF`, `IDN_PT_BLOCKED_OFF`, `IDN_SELECT_BLOCKED_OFF`, `KPCA_RESIDENT_OFF`, `NB_TEXT_CSR_OFF`

### expdel2-10-cluster (44 defines, 21 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_G05` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ANN3_PREPARE_OFF` | DELETE-OLD-PATH | b | 1+4 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | DELETE-ARM | a | 1+12 | 54 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DBSCAN_FAST_DENSE_ADJ` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_DBSCAN_FAST_DIRECT_OUT` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_DBSCAN_FAST_MMA_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC` | DELETE-ARM | a | 1+5 | 26 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FUSED_STAGE_FTZ_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF` | DELETE-OLD-PATH | b | 2+4 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:2010 KEEP -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF` | DELETE-OLD-PATH | b | 1+2 | 30 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK` | DELETE-ARM | e | 1+2 | 3 | EXP:1571 RETIRED (merged, guard refuses) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_KMEANS_DEVICE_CONV_OFF` | DELETE-OLD-PATH | e | 1+4 | 34 | EXP:2274 REFACTORED (2026-11-08) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KMEANS_INCR_INIT_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1517 DROPPED (deleted) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_KMEANS_INIT_PSI_DEVICE_OFF` | DELETE-OLD-PATH | e | 1+3 | 17 | EXP:1517 DROPPED (deleted) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_OCSVM_2L_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OCSVM_CHUNK256` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_OCSVM_RES_OFF` | DELETE-OLD-PATH | b | 1+8 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:2036 KEEP (default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_SVGP_DOT2_OFF` | DELETE-OLD-PATH | b | 1+3 | 48 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_SKETCH_CTR_OFF` | DELETE-OLD-PATH | b | 1+8 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_UNIT_DEV_OFF` | DELETE-OLD-PATH | b | 1+7 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IVF_KMEANS_LAZY_SHIFT_OFF` | DELETE-OLD-PATH | e | 2+3 | 5 | EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_KMEANS_BLOCK_ACC_OFF` | DELETE-OLD-PATH | b | 1+5 | 27 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KMEANS_FAST_INCR_INIT_OFF` | DELETE-OLD-PATH | b | 1+6 | 17 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KMEANS_FAST_PP_NOSYNC_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1517 DROPPED (deleted) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_KMEANS_ID_INCR_INIT_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KMEANS_PP_NOSYNC_APPLE_ONLY` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_KMEANS_ROW_ASSIGN` | DELETE-ARM | e | 1+3 | 4 | EXP:1927 KEPT (owed row withdrawn) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_KPCA_RESIDENT_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:923 DEFAULT; M2 default + OFF both rc=0, merged bc61f3560; Python import PASS -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PROBA64_QOLD` | DELETE-OLD-PATH | e | 2+18 | 24 | EXP:1284 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PURITY_5_OFF` | DELETE-OLD-PATH | b | 1+1 | 19 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_XN_FAST_CLS2_OCSVM_2L_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:357 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:358 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |

Files: `bindings/_mojolearn_x_neighbors.mojo`, `bindings/_mojolearn_x_neighbors_host.mojo`, `bindings/ivf_device_run.mojo`, `checks/native_tests/core_ne3_fused_entries.mojo`, `cluster/host/kmeans_oracle.mojo`, `cluster/impl/detail/classical_assignment.mojo`, `cluster/impl/detail/kmeans.mojo`, `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo`, `cluster/impl/kmeans_params.mojo`, `dbscan/estimator.mojo`, `dbscan/impl/dbscan.mojo`, `dbscan/impl/edge_free.mojo`, `dbscan/impl/neighbors/fast_mma_eps.mojo`, `dbscan/impl/runner.mojo`, `x_neighbors/ocsvm_dev.mojo`, `x_neighbors/ocsvm_init.mojo`, `x_neighbors/ocsvm_resident_native_device.mojo`, `x_neighbors/proba64_nc.mojo`, `x_neighbors/svgp_b32_gate.mojo`, `x_neighbors/svgp_ff.mojo`, `x_prep/proba64.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G03`, `AFCL_G05`, `DBSCAN_FAST_DENSE_ADJ`, `DBSCAN_FAST_DIRECT_OUT`, `DBSCAN_FAST_MMA_OFF`, `DBSCAN_RBC_KEEP_COUNTS_OFF`, `EXPERIMENTAL_KMEANS_BLOCK_ACC`, `FUSED_STAGE_FTZ_OFF`, `IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF`, `IDN_DBSCAN_COSINE_DEVICE_OFF`, `IDN_DBSCAN_DIRECT_OUT_OFF`, `IDN_DBSCAN_EDGE_FREE_OFF`, `IDN_DBSCAN_KEEP_COUNTS_OFF`, `IDN_DBSCAN_RBC_DEAD_READS_OFF`, `IDN_DBSCAN_RBC_ONE_BATCH_OFF`, `IDN_KMEANS_BLOCK_ACC_AMD_OFF`, `IDN_KMEANS_CONV_CHUNK`, `IDN_KMEANS_DEVICE_CONV_OFF`, `IDN_KMEANS_FOLD_STORE_OFF`, `IDN_KMEANS_INCR_INIT_OFF`, `IDN_KMEANS_INIT_PSI_DEVICE_OFF`, `IDN_OCSVM_2L_OFF`, `IDN_OCSVM_CHUNK256`, `IDN_OCSVM_RES_OFF`, `IDN_SVGP_B_FP32_GATED_OFF`, `IDN_SVGP_DOT2_OFF`, `IDN_XN_UNIT_DEV_OFF`, `IVF_KMEANS_LAZY_SHIFT_OFF`, `KMEANS_BLOCK_ACC_OFF`, `KMEANS_FAST_INCR_INIT_OFF`, `KMEANS_FAST_PP_NOSYNC_OFF`, `KMEANS_ID_INCR_INIT_OFF`, `KMEANS_PP_NOSYNC_APPLE_ONLY`, `PROBA64_QOLD`, `PURITY_5_OFF`, `XN_FAST_CLS2_OCSVM_2L_OFF`, `XN_FAST_CLS2_OCSVM_CHUNK256_OFF`

### expdel2-11-cluster (45 defines, 15 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G14` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_BORUVKA_FAST_MMA_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_C33_CHUNK_4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C33_FROZEN_CHUNKS` | DELETE-ARM | e | 1+4 | 5 | EXP:2323 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_C34_PARALLEL_EDGES` | DELETE-ARM | a | 1+3 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C62_SAME_COMPONENT_SKIP` | DELETE-ARM | a | 1+3 | 31 | EXP:1560 unmeasured -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DBSCAN_FAST_CC_BATCH` | DELETE-ARM | a | 1+0 | 1 | EXP:503 OPEN -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_HDB_LINKAGE_DEVICE_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:509 DROPPED-noise -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_HDB_ONE_SYNC` | DELETE-ARM | e | 1+3 | 4 | EXP:1914 KEPT (owed row withdrawn) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_HDB_SELECT_DEVICE` | DELETE-ARM | e | 1+2 | 3 | EXP:1915 KEPT (owed row withdrawn) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_HDB_SMR_TILED` | DELETE-ARM | e | 1+4 | 11 | EXP:1911 DELETED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_HDB_SMR_TILED_OFF` | DELETE-OLD-PATH | e | 1+4 | 11 | EXP:1911 DELETED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK32` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DBSCAN_CC_CHUNK4` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF` | DELETE-OLD-PATH | b | 1+6 | 59 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF` | DELETE-OLD-PATH | b | 1+3 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT16` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT4` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DBSCAN_CC_SPLIT8` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF` | DELETE-OLD-PATH | b | 1+2 | 29 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_DENDRO_UNION_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:1715 owed -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_MST_SEED_KNN_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:2064 LOSER-noise (deleted 826f3d57e; docs/TOMBSTONES.md) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_HDB_ONE_SYNC_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1914 KEPT (owed row withdrawn) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF` | DELETE-OLD-PATH | b | 1+3 | 19 | EXP:1715 owed -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC_OFF` | DELETE-OLD-PATH | b | 1+4 | 22 | EXP:1714 owed -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1915 KEPT (owed row withdrawn) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MST_LABEL_JUMP_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MST_PAR_COMPACT_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MST_SCAN_LANES` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_OCSVM_RES_OFF` | DELETE-OLD-PATH | b | 1+8 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SL_SPARSE_MST` | DELETE-ARM | a | 1+2 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SMR_RADIX_RANK_OFF` | DELETE-OLD-PATH | b | 1+2 | 47 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SL_FAST_BORUVKA_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_SMR_TILED_OFF` | DELETE-OLD-PATH | b | 1+5 | 25 | no record -- no A/B recorded for the old path |

Files: `dbscan/impl/sparse/detail/csr.mojo`, `docs/handoffs/20261111-weekly-limit/resume-evidence/native-upgrade-120/combined-native/combined_gpu.mojo`, `hdbscan/impl/cluster/detail/single_linkage.mojo`, `hdbscan/impl/cluster/detail/sparse_mr_mst.mojo`, `hdbscan/impl/detail/extract.mojo`, `hdbscan/impl/detail/fast_apple.mojo`, `hdbscan/impl/detail/idn_switches.mojo`, `hdbscan/impl/detail/soft_clustering.mojo`, `hdbscan/impl/detail/tree_device.mojo`, `hdbscan/impl/hdbscan_door_impl.mojo`, `hdbscan/impl/runner.mojo`, `hierarchy/impl/cluster/detail/dendrogram_device.mojo`, `hierarchy/impl/cluster/detail/fast_mma_boruvka.mojo`, `hierarchy/impl/cluster/detail/single_linkage.mojo`, `hierarchy/impl/sparse/solver/detail/mst_kernels.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G14`, `BORUVKA_FAST_MMA_OFF`, `DBSCAN_FAST_CC_BATCH`, `HDB_LINKAGE_DEVICE_OFF`, `HDB_ONE_SYNC`, `HDB_SELECT_DEVICE`, `HDB_SMR_TILED`, `HDB_SMR_TILED_OFF`, `IDN_DBSCAN_CC_CHUNK16`, `IDN_DBSCAN_CC_CHUNK32`, `IDN_DBSCAN_CC_CHUNK4`, `IDN_DBSCAN_CC_GATED_OFF`, `IDN_DBSCAN_CC_SHORTCUT_OFF`, `IDN_DBSCAN_CC_SPLIT16`, `IDN_DBSCAN_CC_SPLIT4`, `IDN_DBSCAN_CC_SPLIT8`, `IDN_DENDRO_HOOK_CHUNK`, `IDN_DENDRO_HOOK_CHUNK_OFF`, `IDN_DENDRO_RADIX_SORT_OFF`, `IDN_DENDRO_UNION_OFF`, `IDN_HDB_CONDENSE_TWO_READS_OFF`, `IDN_HDB_MR_FUSED_GUARD_OFF`, `IDN_HDB_MST_SEED_KNN_OFF`, `IDN_HDB_ONE_SYNC_OFF`, `IDN_HDB_OUT_MEMCPY_OFF`, `IDN_HDB_PREDICT_DEVICE_CAST_OFF`, `IDN_HDB_PREDICT_LEAN_OFF`, `IDN_HDB_ROUND_ONE_SYNC_OFF`, `IDN_HDB_SELECT_ONE_READ_OFF`, `IDN_HDB_SOFT_LEAN_OFF`, `IDN_MST_LABEL_JUMP_OFF`, `IDN_MST_PAR_COMPACT_OFF`, `IDN_MST_ROUNDS_DEVICE_OFF`, `IDN_MST_SCAN_LANES`, `IDN_SL_SPARSE_MST`, `IDN_SMR_RADIX_RANK_OFF`, `SL_FAST_BORUVKA_OFF`, `SMR_TILED_OFF`

### expdel2-12-cluster (45 defines, 26 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_G02` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_G09` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ANN3_COARSE_SEED_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:2135 OPEN -- no A/B recorded for the old path |
| `MOJOLEARN_ANN3_PREPARE_OFF` | DELETE-OLD-PATH | b | 1+4 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ANN3_TRAINSET_COPY` | DELETE-ARM | a | 1+1 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C35_PACKED_LISTS` | DELETE-ARM | a | 1+8 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C35_ROWS_128` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GMM_INIT_FILL` | DELETE-ARM | a | 1+1 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GMM_NK_LEVELS_OFF` | DELETE-OLD-PATH | b | 1+3 | 16 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF` | DELETE-OLD-PATH | b | 1+6 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF` | DELETE-OLD-PATH | b | 1+3 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KNN_CERTIFIED_REACH` | DELETE-ARM | a | 3+0 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF` | DELETE-OLD-PATH | b | 1+2 | 20 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER` | DELETE-ARM | a | 1+1 | 2 | EXP:2420 PENDING (A = 0.8.55's form: only the multi-output label transpose resolves the b -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD` | DELETE-OLD-PATH | b | 1+1 | 3 | EXP:2412 PENDING (A = int32 labels copied straight into the label columns) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:2135 OPEN -- no A/B recorded for the old path |
| `MOJOLEARN_IVF_DEVICE_VALIDATE_OFF` | DELETE-OLD-PATH | e | 1+5 | 13 | EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IVF_DIRECT_DISTANCE_OFF` | DELETE-OLD-PATH | e | 1+17 | 38 | EXP:1642 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IVF_FAST_SCAN_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_IVF_FAST_TRAINSET_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IVF_IDENTICAL_SCAN_OFF` | DELETE-OLD-PATH | b | 1+7 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1803 CONFIRMED (kept default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1803 CONFIRMED (kept default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IVF_TRAINSET_STRIDE_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | grid gap-ivf.json new default (plan gaps-2026-10-08 section 6 item 2) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_KNN_APPLE_MMA_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KNN_CERTIFIED_MMA_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KNN_FAST_FORCE_FUSED` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_KNN_FAST_FORCE_TILED` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_KNN_FAST_MMA_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KNN_FAST_QUERY_GROUP4` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_KNN_FAST_TOPK_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512` | DELETE-ARM | a | 1+7 | 18 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KNN_LEGACY_QUERY_TILE` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KNN_SCRATCH_POOL_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | EXP:1658 owed -- no A/B recorded for the old path |
| `MOJOLEARN_KNN_SELECT_TRIAL` | DELETE-ARM | a | 4+12 | 184 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_LEGACY_SHAPE_KNN_BOARD` | DELETE-OLD-PATH | b | 1+0 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RBC_FAST_EPS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded; old path beside the Apple fast default |

Files: `ivf/checks/ivf_check.mojo`, `ivf/checks/ivf_oracle.mojo`, `ivf/host/ivf_host.mojo`, `ivf/impl/neighbors/ivf_flat/fast_ivf_scan.mojo`, `ivf/impl/neighbors/ivf_flat/identical_ivf_scan.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_finite_device.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_query_device.mojo`, `mixture/checks/gmm_oracle.mojo`, `mixture/chol_order.mojo`, `mixture/estimator.mojo`, `mixture/host/gmm_host_oracle.mojo`, `mixture/nk_order.mojo`, `neighbors/checks/query_batch_check.mojo`, `neighbors/checks/select_smallk_identical_candidate.mojo`, `neighbors/estimator.mojo`, `neighbors/impl/ball_cover/fast_rbc_eps.mojo`, `neighbors/impl/detail/fast_mma_knn.mojo`, `neighbors/impl/detail/fast_topk_knn.mojo`, `neighbors/impl/detail/knn_brute_force.mojo`, `neighbors/impl/knn/knn.mojo`, `neighbors/impl/selection/knn.mojo`, `neighbors/neighbors_knn_run.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G01`, `AFCL_G02`, `AFCL_G09`, `GMM_INIT_FILL`, `IDN_GMM_FUSED_CHOL_OFF`, `IDN_GMM_NK_LEVELS_OFF`, `IDN_IVF_DEVICE_FINITE_OFF`, `IDN_IVF_DEVICE_SCALE_OFF`, `IDN_IVF_DEVICE_SQRT_OFF`, `IDN_KNN_CERTIFIED_REACH`, `IDN_KNN_DEVICE_WEIGHTS_OFF`, `IDN_KNN_DIRECT_RELABEL_OFF`, `IDN_KNN_DOOR_FIT_XN_EAGER`, `IDN_KNN_I32_LABEL_RANGE_OLD`, `IDN_KNN_VOTE_CACHE_OFF`, `IVF_COARSE_RANDOM_INIT_OFF`, `IVF_DEVICE_VALIDATE_OFF`, `IVF_FAST_SCAN_OFF`, `IVF_FAST_TRAINSET_OFF`, `IVF_IDENTICAL_ANY_DIM_OFF`, `IVF_IDENTICAL_SCAN_OFF`, `IVF_IDN_RECLUSTER_CAP_OFF`, `IVF_IDN_STRIDED_INIT_OFF`, `IVF_TRAINSET_STRIDE_OFF`, `KNN_APPLE_MMA_OFF`, `KNN_CERTIFIED_MMA_OFF`, `KNN_FAST_FORCE_FUSED`, `KNN_FAST_FORCE_TILED`, `KNN_FAST_MMA_OFF`, `KNN_FAST_QUERY_GROUP4`, `KNN_FAST_TOPK_OFF`, `KNN_IDENTICAL_QUERY_TILE_512`, `KNN_LEGACY_QUERY_TILE`, `KNN_SCRATCH_POOL_OFF`, `KNN_SELECT_TRIAL`, `LEGACY_SHAPE_KNN_BOARD`, `RBC_FAST_EPS_OFF`

### expdel2-13-cluster (43 defines, 23 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_G11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ANN3_CAGRA_TEAM` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_PREPARE_OFF` | DELETE-OLD-PATH | b | 1+4 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ANN3_SCAN_SELECT` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN_COARSE_TILED_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1651 OPEN (A/B owed) -- no A/B recorded for the old path |
| `MOJOLEARN_ANN_SERIAL_SCAN` | DELETE-ARM | a | 2+3 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | DELETE-ARM | a | 1+12 | 54 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CAGRA_FAST_ITERS_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:2126 owed (boxes down) -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_CAGRA_FAST_SEEDS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:2126 owed (boxes down) -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LANCZOS_POOL_OFF` | DELETE-OLD-PATH | b | 1+4 | 23 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF` | DELETE-OLD-PATH | b | 1+4 | 32 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1775 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_PQ_LUT_TILED_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2065 KEEP (promoted 2026-10-11, quality up; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+4 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SPMV_LANES_OFF` | DELETE-OLD-PATH | b | 1+7 | 35 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF` | DELETE-OLD-PATH | e | 1+3 | 9 | EXP:1822 PROMOTED (IDENTICAL default, with REP_TILE) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF` | DELETE-OLD-PATH | b | 1+11 | 72 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KNN_WIDE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LANCZOS_FAST_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF` | DELETE-OLD-PATH | b | 1+4 | 53 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LANCZOS_ID_DEV_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LANCZOS_RESTART_FAST_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N` | DELETE-OLD-PATH | b | 1+0 | 4 | EXP:1335 OWED -- no A/B recorded for the old path |
| `MOJOLEARN_PQ_ASSIGN_UNSTAGED` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PQ_FAST_TRAINSET_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_PQ_LUT_TILED_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PQ_SCAN_FUSED_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:1801 DELETED (loser) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_SPECTRAL_NCV_FAST_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_SPMV_WARP_OFF` | DELETE-OLD-PATH | b | 1+3 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SYMMETRIZE_FAST_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_UMAP_FUSED_EPOCH_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_UMAP_INIT_TOL` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_UMAP_INIT_TOL_OFF` | DELETE-OLD-PATH | b | 1+3 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_UMAP_LIVE_BOTH_ARM` | DELETE-ARM | a | 2+10 | 25 | no record -- no A/B recorded for the opt-in arm |

Files: `spectral/host/spectral_oracle.mojo`, `spectral/impl/cluster/detail/spectral.mojo`, `spectral/impl/labels_device.mojo`, `spectral/impl/preprocessing/detail/fast_graph.mojo`, `spectral/impl/preprocessing/detail/spectral_embedding.mojo`, `spectral/impl/sparse/linalg/detail/laplacian.mojo`, `spectral/impl/sparse/linalg/detail/symmetrize.mojo`, `spectral/impl/sparse/solver/detail/lanczos.mojo`, `spectral/impl/spectral_predict.mojo`, `spectral/spmv_order.mojo`, `umap/host/umap_oracle.mojo`, `umap/optimizer_fast.mojo`, `umap/spectral_init.mojo`, `x_ann/cagra_core.mojo`, `x_ann/cagra_device.mojo`, `x_ann/cagra_fast_knn.mojo`, `x_ann/fast_env.mojo`, `x_ann/host/ivf_pq_host.mojo`, `x_ann/host/tsne_host.mojo`, `x_ann/ivf_pq_device.mojo`, `x_ann/ivf_scan_device.mojo`, `x_ann/knn_device.mojo`, `x_ann/resident.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G10`, `AFCL_G11`, `ANN_COARSE_TILED_OFF`, `ANN_SERIAL_SCAN`, `CAGRA_FAST_ITERS_OFF`, `CAGRA_FAST_SEEDS_OFF`, `IDN_ANN_KNN_PRUNE_OFF`, `IDN_LANCZOS_POOL_OFF`, `IDN_LAP_DEGREE_LANES_OFF`, `IDN_SPECTRAL_GRAPH_DEVICE_OFF`, `IDN_SPECTRAL_LABELS_DEVICE_OFF`, `IDN_SPECTRAL_VECS_DEVICE_OFF`, `IDN_SPMV_LANES_OFF`, `IDN_SYMMETRIZE_SORTED_OFF`, `IDN_TSNE_KNN_TILE_SELECT_OFF`, `KNN_WIDE_OFF`, `LANCZOS_FAST_OFF`, `LANCZOS_ID_DEV_FUSE_OFF`, `LANCZOS_ID_DEV_OFF`, `LANCZOS_RESTART_FAST_OFF`, `LEGACY_NARROW_IVFG_MIN_N`, `PQ_ASSIGN_UNSTAGED`, `PQ_FAST_TRAINSET_OFF`, `SPECTRAL_GRAPH_FAST_OFF`, `SPECTRAL_NCV_FAST_OFF`, `SPMV_WARP_OFF`, `SYMMETRIZE_FAST_OFF`, `UMAP_FUSED_EPOCH_OFF`, `UMAP_INIT_TOL`, `UMAP_INIT_TOL_OFF`, `UMAP_LIVE_BOTH_ARM`

### expdel2-14-cluster (45 defines, 11 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_ANN3_CAGRA_TEAM` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_COARSE_SEED_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:2135 OPEN -- no A/B recorded for the old path |
| `MOJOLEARN_ANN3_DIRECT_OUT` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_PQ_SEED` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_PREPARE_OFF` | DELETE-OLD-PATH | b | 1+4 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ANN3_ROW_THREADS` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_SCAN_SELECT` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_TRAINSET_COPY` | DELETE-ARM | a | 1+1 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_TSNE_RB32` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_TSNE_RB64` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ANN3_TSNE_STEP_ROWS` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AP_SPLIT_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:496 KEPT deee07721 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_BGMM_FAST_MOMENTS_GEMM_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:500 HOLD-quality -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_C39_RETAIN_STATE` | DELETE-ARM | a | 1+10 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C42_ACTIVE_TRIANGLE` | DELETE-ARM | a | 1+10 | 24 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_AP_CONV_CHUNK16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_AP_CONV_CHUNK32` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_AP_CONV_CHUNK4` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_AP_DEVICE_CONV_OFF` | DELETE-OLD-PATH | b | 1+2 | 33 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_AP_R_TOP2_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF` | DELETE-OLD-PATH | b | 1+2 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF` | DELETE-OLD-PATH | b | 1+3 | 24 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF` | DELETE-OLD-PATH | b | 1+3 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MEANSHIFT_GRID_OFF` | DELETE-OLD-PATH | b | 1+9 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1775 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_PQ_LUT_TILED_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2065 KEEP (promoted 2026-10-11, quality up; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF` | DELETE-OLD-PATH | b | 1+4 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF` | DELETE-OLD-PATH | b | 1+11 | 72 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSNE_REP_TILE_OFF` | DELETE-OLD-PATH | e | 1+5 | 11 | EXP:1823 PROMOTED (IDENTICAL default, with KNN_TILE_SELECT) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | grid fg-ivf.json new default (read_ivf.md A3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | grid fg-ivf.json new default (read_ivf.md A7) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MEANSHIFT_BLOCK` | DELETE-ARM | a | 1+2 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MOMENTS_ROWS` | DELETE-ARM | a | 1+2 | 22 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF` | DELETE-OLD-PATH | e | 2+4 | 8 | EXP:516 DROP -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PQ_LUT_TILED_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PQ_SCAN_FUSED_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:1801 DELETED (loser) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TSNE_SPLIT_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_VSEARCH_ALL` | DELETE-ARM | e | 1+0 | 1 | EXP:352 KEPT (FAST+Apple default, `_OFF`) lane/apple-fast-vsv-promote -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the de |
| `MOJOLEARN_WARD_ROUNDS` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCLUSTER_KPP_DISTINCT` | DELETE-ARM | e | 1+3 | 8 | EXP:1517 DROPPED (deleted) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_XCLUSTER_ROW_ASSIGN` | DELETE-ARM | a | 1+3 | 4 | EXP:1518 4`, and per-family direct-distance defines KNN / KDE / DBSCAN / GRAPH / IVF with -- no A/B recorded for the opt-in arm |

Files: `x_ann/switches.mojo`, `x_ann/tsne_core.mojo`, `x_ann/tsne_device.mojo`, `x_ann/vsearch_fast.mojo`, `x_cluster/affinity.mojo`, `x_cluster/agglo.mojo`, `x_cluster/bisect.mojo`, `x_cluster/bodies.mojo`, `x_cluster/checks/oracles.mojo`, `x_cluster/common.mojo`, `x_cluster/device_ops.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `ANN3_CAGRA_TEAM`, `ANN3_COARSE_SEED_OFF`, `ANN3_DIRECT_OUT`, `ANN3_PQ_SEED`, `ANN3_PREPARE_OFF`, `ANN3_ROW_THREADS`, `ANN3_SCAN_SELECT`, `ANN3_TRAINSET_COPY`, `ANN3_TSNE_RB32`, `ANN3_TSNE_RB64`, `ANN3_TSNE_STEP_ROWS`, `AP_SPLIT_OFF`, `BGMM_FAST_MOMENTS_GEMM_OFF`, `IDN_AP_CONV_CHUNK16`, `IDN_AP_CONV_CHUNK32`, `IDN_AP_CONV_CHUNK4`, `IDN_AP_DEVICE_CONV_OFF`, `IDN_AP_R_TOP2_OFF`, `IDN_BGMM_MOMENTS_POOL_OFF`, `IDN_BGMM_NK_LEVELS_OFF`, `IDN_BISECT_DEVICE_CENTER_OFF`, `IDN_BISECT_DEVICE_SCORES_OFF`, `IDN_MEANSHIFT_GRID_OFF`, `IDN_PQ_DEVICE_CODEBOOKS_OFF`, `IDN_PQ_LUT_TILED_OFF`, `IDN_PQ_TRAINSET_STRIDED_OFF`, `IDN_TSNE_LANES_SPLIT_OFF`, `IDN_TSNE_LANE_FOLD_OFF`, `IDN_TSNE_REP_TILE_OFF`, `IVF_PQ_ONE_UPLOAD_OFF`, `IVF_PQ_RESIDENT_FIT_OFF`, `MEANSHIFT_BLOCK`, `MOMENTS_ROWS`, `OPTICS_FAST_DEVICE_ORDER_OFF`, `PQ_LUT_TILED_OFF`, `PQ_SCAN_FUSED_OFF`, `TSNE_SPLIT_OFF`, `VSEARCH_ALL`, `WARD_ROUNDS`

### expdel2-15-cluster (42 defines, 18 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_G12` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_C33_CHUNK_4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C33_FROZEN_CHUNKS` | DELETE-ARM | e | 1+4 | 5 | EXP:2323 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_C41_FUSED_MINIMA` | DELETE-ARM | a | 1+3 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | DELETE-ARM | a | 1+12 | 54 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C54_PREDICT_TILES` | DELETE-ARM | a | 1+8 | 45 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLUSTER_FAST_STOP_GROUP8` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MBK_RESIDENT_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:2050 KEEP (promoted 2026-10-11; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MINIBATCH_GROUP32` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MINIBATCH_GROUP64` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MINIBATCH_GROUP8` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NC_CHUNKED_OFF` | DELETE-OLD-PATH | b | 1+9 | 38 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NC_FIT_LEAN_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:2049 KEEP (promoted 2026-10-11; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_OCSVM_RES_OFF` | DELETE-OLD-PATH | b | 1+8 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SVGP_BLKCHOL_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SVGP_RBFTILE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_KNN_LEAN_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_KNN_PRUNE_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_SKETCH_CTR_OFF` | DELETE-OLD-PATH | b | 1+8 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XN_UNIT_DEV_OFF` | DELETE-OLD-PATH | b | 1+7 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MBK_ZEROCOPY_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | EXP:486 KEPT 51493c7a8 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MINIBATCH_ONE_PASS` | DELETE-ARM | a | 1+2 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF` | DELETE-OLD-PATH | e | 2+4 | 8 | EXP:516 DROP -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_OPTICS_SIMD` | DELETE-ARM | a | 1+2 | 17 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PROBA64_QOLD` | DELETE-OLD-PATH | e | 2+18 | 24 | EXP:1284 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PURITY_4_OFF` | DELETE-OLD-PATH | b | 1+3 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_XCLUSTER_KPP_DISTINCT` | DELETE-ARM | e | 1+3 | 8 | EXP:1517 DROPPED (deleted) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_XC_ALLOC` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:2125 owed (boxes down) -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1049 DEFAULT: M2 default/rollback PASS, final M3 knn-lean-gpu-default-quality-2026100 -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XN_IMPUTE_NO_SPLIT` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XN_IMPUTE_ROWDIST_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1650 OPEN (A/B owed) -- no A/B recorded for the old path |
| `MOJOLEARN_XN_KNN_ROWWISE` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XN_PCS_CELL` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF` | DELETE-OLD-PATH | e | 1+5 | 8 | EXP:520 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` | DELETE-OLD-PATH | e | 1+4 | 10 | EXP:888 DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` -- record read: the default is the recorded winner or the kept path |

Files: `x_cluster/device_post.mojo`, `x_cluster/meanshift_fast.mojo`, `x_cluster/minibatch.mojo`, `x_cluster/minibatch_fast.mojo`, `x_cluster/minibatch_ptr.mojo`, `x_cluster/optics.mojo`, `x_neighbors/checks/oracles.mojo`, `x_neighbors/graph_par.mojo`, `x_neighbors/items.mojo`, `x_neighbors/iter_device.mojo`, `x_neighbors/iter_host.mojo`, `x_neighbors/kfeat_rng.mojo`, `x_neighbors/lp_knn.mojo`, `x_neighbors/nan_cells_device.mojo`, `x_neighbors/native_descriptor_aux_device.mojo`, `x_neighbors/native_descriptor_aux_host.mojo`, `x_neighbors/native_descriptor_device.mojo`, `x_neighbors/native_descriptor_host.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G07`, `AFCL_G12`, `CLUSTER_FAST_STOP_GROUP8`, `IDN_MBK_RESIDENT_OFF`, `IDN_MINIBATCH_GROUP32`, `IDN_MINIBATCH_GROUP64`, `IDN_MINIBATCH_GROUP8`, `IDN_NC_CHUNKED_OFF`, `IDN_NC_FIT_LEAN_OFF`, `IDN_OPTICS_FUSED_STEP_OFF`, `IDN_SVGP_BLKCHOL_OFF`, `IDN_SVGP_RBFTILE_OFF`, `IDN_XN_KNN_LEAN_OFF`, `IDN_XN_KNN_PRUNE_OFF`, `IDN_XN_SKETCH_CTR_OFF`, `MBK_ZEROCOPY_OFF`, `MINIBATCH_ONE_PASS`, `OPTICS_SIMD`, `PURITY_4_OFF`, `XC_ALLOC`, `XN_FAST_IMPUTE_TIE_MEAN_OFF`, `XN_FAST_NAN_FIT_LEAN_GPU_OFF`, `XN_IMPUTE_NO_SPLIT`, `XN_IMPUTE_ROWDIST_OFF`, `XN_KNN_ROWWISE`, `XN_PCS_CELL`, `X_CLUSTER_FAST_CLS2_MBK_POOL_OFF`, `X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF`, `X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF`

### expdel2-16-decomposition (45 defines, 22 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_L10` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFN_GEMM_KB` | DELETE-ARM | f | 1+13 | 14 | EXP:683 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_CD_IDN_GRAM_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CHOL_FAST_DEVIO_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1395 READY-AB -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_CHOL_HOST_SCALAR` | DELETE-ARM | a | 1+3 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C01_MEAN` | DELETE-ARM | e | 1+4 | 14 | EXP:1530 DROPPED-slower -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CLASSICAL_C14_GROUP_RHS` | DELETE-ARM | a | 1+5 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES` | DELETE-ARM | a | 1+8 | 31 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:1993 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DECOMP_FAST_GEMM_MMA_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:435 DROPPED-slower -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_DECOMP_FAST_OMP_BLOCK` | DELETE-ARM | a | 1+4 | 11 | EXP:449 OPEN (measured, verdict owed) -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:931 M3 43765.6 -> 700.3 ms; original no-regression criterion PASS all 8 fixtures, ze -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CHOL_OPS_FIT` | DELETE-ARM | a | 1+3 | 4 | EXP:581 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CODE_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_BLOCK_OFF` | DELETE-OLD-PATH | b | 3+16 | 32 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_BLOCK_ON` | DELETE-ARM | a | 3+14 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+6 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_TRIDIAG_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:476 KEEP (default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_TN_CENTERED` | DELETE-ARM | a | 1+2 | 7 | grid fg-pca.json hold:postmerge-20261010 (lane/postmerge-act-7; fg2 board-bri -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ICA_LIM_DEV_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LLE_DEV_F0_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LLE_PAD_DEV_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LU_GESV_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LU_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+4 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NMF_L2_CELL_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PCA_LEAN_SCRATCH_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:1689 owed -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_POLAR_MAX_DEV_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QR_R_DIRECT_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QR_R_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RAND_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RES_MOVES_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SVD_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XD_SVD_GROUP_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2145 DEFAULT (bit-neutral) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_XD_SWEEP_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LU_FAST_MMA_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:1401 READY-AB -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_LU_FAST_STEP1_OFF` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:460 DEFAULT (FAST+Apple), rollback `MOJOLEARN_LU_FAST_MMA_OFF`; w2-lumma-quality PAS -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LU_QOLD` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:1283 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PCA_RR_EIGH_OFF` | DELETE-OLD-PATH | b | 1+5 | 36 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_TSVD_QOLD` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1287 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XD_FAST_SVD_ONE_BLOCK` | DELETE-ARM | a | 1+1 | 4 | EXP:2366 PENDING -- no A/B recorded (Apple fast opt-in) |

Files: `bindings/_mojolearn_x_decomp.mojo`, `cholesky/checks/chol_fast_tall.mojo`, `cholesky/chol_ops.mojo`, `cholesky/estimator.mojo`, `cholesky/host/chol_oracle.mojo`, `decomposition/host/pca_oracle.mojo`, `decomposition/impl/linalg/detail/pca.mojo`, `decomposition/pca_rr_switch.mojo`, `kernel_methods/impl/kernel_ridge/kernel_ridge.mojo`, `mixture/checks/gmm_check.mojo`, `providers/python/x_decomp_qfix_pyapi_pack.mojo`, `tools/identical_wave_eigh_block_refusal.mojo`, `tools/identical_wave_eigh_centered_pivot.mojo`, `tools/identical_wave_route_gate.mojo`, `x_decomp/api.mojo`, `x_decomp/cells.mojo`, `x_decomp/decomp_owner_dev.mojo`, `x_decomp/descriptor_device.mojo`, `x_decomp/device.mojo`, `x_decomp/eigh_ops_host.mojo`, `x_decomp/eigh_tridiag.mojo`, `x_decomp/eigh_tridiag_cells.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_L10`, `CHOL_FAST_DEVIO_OFF`, `CHOL_HOST_SCALAR`, `DECOMP_FAST_GEMM_MMA_OFF`, `EIGH_FAST_TRIDIAG_PANELS_OFF`, `IDN_CHOL_OPS_FIT`, `IDN_EIGH_BLOCK_OFF`, `IDN_EIGH_BLOCK_ON`, `IDN_EIGH_TRIDIAG_OFF`, `IDN_GATES_OFF`, `IDN_GEMM_TN_CENTERED`, `IDN_ICA_LIM_DEV_OFF`, `IDN_LLE_DEV_F0_OFF`, `IDN_LLE_PAD_DEV_OFF`, `IDN_LU_GESV_OFF`, `IDN_NMF_L2_CELL_OFF`, `IDN_PCA_LEAN_SCRATCH_OFF`, `IDN_POLAR_MAX_DEV_OFF`, `IDN_QR_R_DIRECT_OFF`, `IDN_RAND_RESIDENT_OFF`, `IDN_RES_MOVES_OFF`, `IDN_XD_SVD_GROUP_OFF`, `IDN_XD_SWEEP_OFF`, `PCA_RR_EIGH_OFF`, `XD_FAST_SVD_ONE_BLOCK`

Lane result (expdel2-16-decomposition, 2026-11-15): all 25 owned defines deleted with their records; 5 non-owner defines folded in this lane's files (`CLASSICAL_C14_GROUP_RHS`, `CLASSICAL_C28_BUCKET_SOLVES`, `DECOMP_FAST_OMP_BLOCK`, `CLASSICAL_C01_MEAN`, `IDN_ALL_OFF`). KEPT: `COLUMN_CPU` (correction). The other non-owner defines are read here only through other lanes' aliases whose value stays the default (int knob or mode term); comments naming them were updated. Left for other lanes: `XD_FAST_SVD_ONE_BLOCK` and `CHOL_OPS_FIT` stay `False` and `launch_classical_code_rows` + its two kernels and `kernel_ridge_ops_route` stay only for imports in other lanes' files (x_decomp/kit_device.mojo, gaussian_process/*, mixture/checks/mstep.mojo, kernel_methods/estimator.mojo); gemm/tn_centered.mojo (expdel2-07) and x_decomp/omp_block.mojo have no reader left in this lane's files.

### expdel2-17-decomposition (42 defines, 22 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_L13` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFN_GEMM_KB` | DELETE-ARM | f | 1+13 | 14 | EXP:683 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_C57_CANDIDATE_STATE` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C23_MCD` | DELETE-ARM | a | 1+3 | 6 | EXP:1533 not measured alone -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C24_TREE4` | DELETE-ARM | e | 1+6 | 7 | grid classical-decomp.json promoted:ge123e6f9 -- grid C24_TSQR promoted the rows2048 arm; tree4 was a measured arm that was not promoted |
| `MOJOLEARN_CLASSICAL_C26_PRODUCTS` | DELETE-ARM | a | 1+9 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED` | DELETE-ARM | a | 1+5 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_COMPONENTS` | DELETE-ARM | a | 1+3 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C27_NORM_VECTOR` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES` | DELETE-ARM | a | 1+8 | 31 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:1993 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DECOMP_FAST_GEMM_MMA_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:435 DROPPED-slower -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_DECOMP_FAST_OMP_BLOCK` | DELETE-ARM | a | 1+4 | 11 | EXP:449 OPEN (measured, verdict owed) -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_FA_GRAM_ONCE` | DELETE-ARM | a | 1+1 | 2 | EXP:1260 HOLD-quality -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FA_GRAM_QOLD` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1263 KEPT (QUALITY-FIX) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_FA_ITER_DEVICE_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:1280 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_FA_LIVEBUF` | DELETE-ARM | a | 2+6 | 22 | EXP:457 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FA_SVD_GROUP_OFF` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:2145 DEFAULT (bit-neutral) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_EIGH_BLOCK_OFF` | DELETE-OLD-PATH | b | 3+16 | 32 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_BLOCK_ON` | DELETE-ARM | a | 3+14 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+6 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MCD_CHOL` | DELETE-ARM | a | 1+7 | 10 | EXP:581 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_LDA_FUSED_SS_OFF` | DELETE-OLD-PATH | d | 1+2 | 5 | EXP:416 KEPT a7b8b9513 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LDA_V_CAP_HALF_PAGE` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_LDA_V_CAP_RULE_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_LDA_FUSED_V` | DELETE-OLD-PATH | d | 2+6 | 10 | EXP:1343 ACCEPT (merged via lane/apple-fast-verdicts-5) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LU_FAST_MMA_NB` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_LU_FAST_MMA_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:1401 READY-AB -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE_OFF` | DELETE-OLD-PATH | e | 1+4 | 25 | EXP:1007 DEFAULT merged; promotion0769e1d92 M2default/OFF rc0. Existing parallel LUstep g -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LU_FAST_STEP1_OFF` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:460 DEFAULT (FAST+Apple), rollback `MOJOLEARN_LU_FAST_MMA_OFF`; w2-lumma-quality PAS -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LU_QOLD` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:1283 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_MCD_BATCH_MMA_OFF` | DELETE-OLD-PATH | e | 1+20 | 57 | EXP:2281 DELETED (HOLD-quality) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_MCD_BMMA_OFF` | DELETE-OLD-PATH | e | 1+7 | 22 | EXP:804 DEFAULT (FAST+Apple, needs MCD_BMMA), rollback `MOJOLEARN_MCD_WIDE_OFF` -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_MCD_DEFLATE_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:1273 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MCD_SKIP_PINVH_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:929 HOLD (opt-in) -- no A/B recorded for the old path |
| `MOJOLEARN_MCD_WIDE_OFF` | DELETE-OLD-PATH | e | 1+9 | 10 | EXP:804 DEFAULT (FAST+Apple, needs MCD_BMMA), rollback `MOJOLEARN_MCD_WIDE_OFF` -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TSVD_QOLD` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1287 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XDECOMP_HOST_SERIAL` | DELETE-ARM | a | 1+0 | 4 | no record -- no A/B recorded for the opt-in arm |

Files: `x_decomp/fa_em.mojo`, `x_decomp/fa_em_dev.mojo`, `x_decomp/fa_fast.mojo`, `x_decomp/host.mojo`, `x_decomp/ica.mojo`, `x_decomp/ica_dev.mojo`, `x_decomp/ipca_owner_dev.mojo`, `x_decomp/kit_device.mojo`, `x_decomp/lda_fast.mojo`, `x_decomp/lda_fit_dev.mojo`, `x_decomp/lu_fast.mojo`, `x_decomp/lu_fast_mma.mojo`, `x_decomp/mcd.mojo`, `x_decomp/mcd_bmma.mojo`, `x_decomp/mcd_fast.mojo`, `x_decomp/nmf.mojo`, `x_decomp/nmf_dev.mojo`, `x_decomp/omp_block.mojo`, `x_decomp/pls.mojo`, `x_decomp/pls_dev.mojo`, `x_decomp/qfix.mojo`, `x_decomp/qr_sliced.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_L13`, `DECOMP_FAST_OMP_BLOCK`, `FA_GRAM_ONCE`, `FA_GRAM_QOLD`, `FA_ITER_DEVICE_OFF`, `FA_LIVEBUF`, `FA_SVD_GROUP_OFF`, `IDN_MCD_CHOL`, `LDA_FUSED_SS_OFF`, `LDA_V_CAP_HALF_PAGE`, `LDA_V_CAP_RULE_OFF`, `LEGACY_NARROW_LDA_FUSED_V`, `LU_FAST_MMA_NB`, `LU_FAST_MMA_OFF`, `LU_FAST_PIVOT_SHUFFLE_OFF`, `LU_FAST_STEP1_OFF`, `LU_QOLD`, `MCD_BATCH_MMA_OFF`, `MCD_BMMA_OFF`, `MCD_DEFLATE_OFF`, `MCD_SKIP_PINVH_OFF`, `MCD_WIDE_OFF`, `TSVD_QOLD`, `XDECOMP_HOST_SERIAL`

### expdel2-18-decomposition (35 defines, 7 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_CLASSICAL_C24_PANEL8` | DELETE-ARM | e | 1+2 | 3 | grid classical-decomp.json promoted:ge123e6f9 -- grid C24_TSQR promoted the rows2048 arm; panel8 was a measured arm that was not promoted |
| `MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1638 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_C24_TREE4` | DELETE-ARM | e | 1+6 | 7 | grid classical-decomp.json promoted:ge123e6f9 -- grid C24_TSQR promoted the rows2048 arm; tree4 was a measured arm that was not promoted |
| `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:1993 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DECOMP_FAST_TSQR_GRID` | DELETE-ARM | a | 1+5 | 23 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_DECOMP_FAST_TSQR_NORM` | DELETE-ARM | a | 1+3 | 18 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CD_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_CODE_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_BLOCK_B32` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_BLOCK_B8` | DELETE-ARM | a | 1+13 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_BLOCK_MIN128` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_BLOCK_OFF` | DELETE-OLD-PATH | b | 3+16 | 32 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EIGH_BLOCK_ON` | DELETE-ARM | a | 3+14 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_EIGH_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+6 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_LU_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+4 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QR_R_INPLACE_OFF` | DELETE-OLD-PATH | b | 1+3 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_QR_R_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SVD_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSQR_GRID_OFF` | DELETE-OLD-PATH | b | 1+5 | 23 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSQR_NORM_OFF` | DELETE-OLD-PATH | b | 1+3 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSQR_REUSE` | DELETE-ARM | a | 4+0 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TSQR_STRIP_UPDATE` | DELETE-ARM | a | 1+2 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TSQR_TREE_PAR_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:479 RUN OWED -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TSQR_WY_PAIR_OFF` | DELETE-OLD-PATH | b | 1+2 | 8 | EXP:479 RUN OWED -- no A/B recorded for the old path |
| `MOJOLEARN_LU_FAST_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | EXP:2031 OPEN (A/B owed) -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_LU_QOLD` | DELETE-OLD-PATH | e | 1+8 | 9 | EXP:1283 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_RSVD_FAST_DEVSCAN_OFF` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:2377 KEPT default again (restored; Apple FAST paused, owed row r1-rsvd-devscan) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TSVD_QOLD` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1287 QUALITY-FIX KEPT: FAST default -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:467 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XD_FAST_CLS2_GRP_LAZY` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN` | DELETE-ARM | a | 1+3 | 4 | EXP:466 PARKED -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_XD_FAST_SRP_STRAT_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | EXP:861 sparse-rp / istella (quality HOLD), taxi -- no A/B recorded; old path beside the Apple fast default |

Files: `x_decomp/resident.mojo`, `x_decomp/rr_block.mojo`, `x_decomp/s_linalg_fast.mojo`, `x_decomp/tsqr_core.mojo`, `x_decomp/tsqr_device.mojo`, `x_decomp/tsqr_multi_gpu.mojo`, `x_decomp/tsvd_descriptor_boundary_run.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `CLASSICAL_RSVD_TSQR_ORTHO_OFF`, `DECOMP_FAST_TSQR_GRID`, `DECOMP_FAST_TSQR_NORM`, `IDN_CD_RESIDENT_OFF`, `IDN_CODE_RESIDENT_OFF`, `IDN_EIGH_BLOCK_B32`, `IDN_EIGH_BLOCK_B8`, `IDN_EIGH_BLOCK_MIN128`, `IDN_EIGH_RESIDENT_OFF`, `IDN_LU_RESIDENT_OFF`, `IDN_QR_R_INPLACE_OFF`, `IDN_QR_R_RESIDENT_OFF`, `IDN_SVD_RESIDENT_OFF`, `IDN_TSQR_GRID_OFF`, `IDN_TSQR_NORM_OFF`, `IDN_TSQR_STRIP_UPDATE`, `IDN_TSQR_TREE_PAR_OFF`, `IDN_TSQR_WY_PAIR_OFF`, `LU_FAST_RESIDENT_OFF`, `RSVD_FAST_DEVSCAN_OFF`, `XD_FAST_CLS2_GRP_DEVSCAN_OFF`, `XD_FAST_CLS2_GRP_LAZY`, `XD_FAST_CLS2_GRP_NOSCAN`, `XD_FAST_SRP_STRAT_OFF`

### expdel2-19-kernel (39 defines, 22 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_G13` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L11` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_C52_PAIR_ROWS` | DELETE-ARM | a | 2+15 | 34 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1984 PROMOTED (restored; IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FAMILY_CTX_PER_CALL` | DELETE-ARM | a | 5+4 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GP_BULK_DOWNLOAD` | DELETE-ARM | a | 1+0 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF` | DELETE-OLD-PATH | b | 1+2 | 19 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GP_GRAD_ROWS4` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GP_VAR_SEG_OFF` | DELETE-OLD-PATH | b | 1+4 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KDE_PARTIAL_POOL` | DELETE-ARM | a | 1+8 | 44 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KDE_QTILE_FORCE` | DELETE-ARM | a | 1+2 | 4 | EXP:2398 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KDE_QTILE_OFF` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:2059 KEEP by average (promoted 2026-10-11, istella regression flagged; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF` | DELETE-OLD-PATH | b | 1+2 | 16 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KM_DOT_CELL_D16` | DELETE-ARM | a | 2+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KM_DOT_CELL_OFF` | DELETE-OLD-PATH | b | 2+6 | 63 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KM_PTR_IN_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KM_RBF_CELL` | DELETE-ARM | a | 2+6 | 48 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KM_RBF_CELL_D16` | DELETE-ARM | a | 2+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_KRR_DEV_SCALE_OFF` | DELETE-OLD-PATH | b | 1+5 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KRR_PTR_IN_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NYS_DEV_BASIS_OFF` | DELETE-OLD-PATH | b | 1+3 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NYS_DEV_ORDER_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP` | DELETE-ARM | a | 1+3 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NYS_RR_EIGH_OFF` | DELETE-OLD-PATH | b | 2+6 | 24 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RBF_FUSED_OFF` | DELETE-OLD-PATH | b | 2+6 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KDE_BLOCK_SUM_OFF` | DELETE-OLD-PATH | b | 1+1 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_KDE_DIMTILE_OFF` | DELETE-OLD-PATH | e | 1+3 | 27 | EXP:343 DROP -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_KDE_DIRECT_DISTANCE` | DELETE-ARM | a | 1+9 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF` | DELETE-OLD-PATH | b | 1+3 | 7 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KDE_FAST_IMMUTABLE_FIT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RBF_FUSED` | DELETE-ARM | a | 1+2 | 11 | no record -- no A/B recorded for the opt-in arm |

Files: `bindings/_mojolearn_kernel_methods.mojo`, `checks/native_tests/kde_ownership.mojo`, `gaussian_process/afcl_prediction.mojo`, `gaussian_process/checks/gp_oracle.mojo`, `gaussian_process/classifier.mojo`, `gaussian_process/estimator.mojo`, `gaussian_process/gp_grad_items.mojo`, `gaussian_process/gp_var_seg.mojo`, `gaussian_process/host/gpc_steps.mojo`, `gaussian_process/host/gpr_oracle.mojo`, `kde/checks/reused_workspace_check.mojo`, `kde/host/kde_oracle.mojo`, `kde/impl/chunk_workspace.mojo`, `kde/impl/distance/distance.mojo`, `kde/impl/neighbors/kernel_density.mojo`, `kde/pair_lse.mojo`, `kde/resident_fit.mojo`, `kernel_methods/checks/kernel_matrix.mojo`, `kernel_methods/estimator.mojo`, `kernel_methods/host/km_host_oracle.mojo`, `kernel_methods/rbf_resident.mojo`, `svm/estimator.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_G13`, `AFCL_L11`, `FAMILY_CTX_PER_CALL`, `GP_BULK_DOWNLOAD`, `IDN_GP_BULK_DOWNLOAD_OFF`, `IDN_GP_GRAD_ROWFOLD_OFF`, `IDN_GP_GRAD_ROWS4`, `IDN_GP_PREDICT_LAZY_L_OFF`, `IDN_GP_VAR_SEG_OFF`, `IDN_KDE_CHUNK_LSE_OFF`, `IDN_KDE_PARTIAL_POOL`, `IDN_KDE_QTILE_FORCE`, `IDN_KDE_QTILE_OFF`, `IDN_KM_BULK_DOWNLOAD_OFF`, `IDN_KM_DOT_CELL_D16`, `IDN_KM_DOT_CELL_OFF`, `IDN_KM_PTR_IN_OFF`, `IDN_KM_RBF_CELL`, `IDN_KM_RBF_CELL_D16`, `IDN_KRR_DEV_SCALE_OFF`, `IDN_KRR_PTR_IN_OFF`, `IDN_NYS_DEV_BASIS_OFF`, `IDN_NYS_DEV_ORDER_OFF`, `IDN_NYS_FIT_PTR_IN_OFF`, `IDN_NYS_RR_DEV_STOP`, `IDN_NYS_RR_DEV_STOP_B4`, `IDN_NYS_RR_EIGH_OFF`, `IDN_RBF_FUSED_OFF`, `KDE_BLOCK_SUM_OFF`, `KDE_DIMTILE_OFF`, `KDE_FAST_DIRECT_PREP_OFF`, `KDE_FAST_IMMUTABLE_FIT`, `LEGACY_NARROW_KDE_DIMTILE`, `RBF_FUSED`

### expdel2-20-linear (45 defines, 22 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_L05` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L06` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_CD_FAST_GRAM_OFF` | DELETE-OLD-PATH | b | 1+8 | 13 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_CD_FUSED_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CD_GRAM_BLOCKS` | DELETE-ARM | a | 1+2 | 29 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CD_IDN_GRAM_COLS_128` | DELETE-ARM | a | 2+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CD_IDN_GRAM_COLS_256` | DELETE-ARM | a | 2+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CD_IDN_GRAM_OFF` | DELETE-OLD-PATH | b | 1+5 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CD_IDN_GRAM_WIDE` | DELETE-ARM | a | 2+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CD_THREE_LAUNCH_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CD_TWO_STEP_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C05_OLS_PHASE` | DELETE-ARM | a | 1+4 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C17_LS_TRIALS` | DELETE-ARM | a | 2+3 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C22_TRIANGLE` | DELETE-ARM | a | 1+3 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF` | DELETE-OLD-PATH | e | 1+5 | 41 | EXP:1831 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_QN_EXACT_TRIALS` | DELETE-ARM | a | 3+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2062 KEEP (promoted 2026-10-11; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LEGACY_NARROW_FX_D` | DELETE-OLD-PATH | b | 1+1 | 3 | EXP:1336 OWED -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_QN_ALL` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LSVR_ALL_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:283 DROPPED-noise -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LSVR_EVAL_SLIM_OFF` | DELETE-OLD-PATH | d | 1+10 | 13 | EXP:285 KEPT lane/apple-fast-m2b1-main -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF` | DELETE-OLD-PATH | d | 1+15 | 20 | EXP:288 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_OLS_FAST_SMALL_NT` | DELETE-ARM | a | 1+2 | 11 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED_OFF` | DELETE-OLD-PATH | e | 1+8 | 27 | EXP:405 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL_OFF` | DELETE-OLD-PATH | e | 1+8 | 19 | EXP:404 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PURITY2_2_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | EXP:721 OPEN -- no A/B recorded for the old path |
| `MOJOLEARN_PURITY_1_OFF` | DELETE-OLD-PATH | b | 2+0 | 30 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_QN_DEV_ARGMAX_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_QN_FAST_XTDZ_OFF` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:1930 DELETED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_QN_IDN_FUSED_OFF` | DELETE-OLD-PATH | b | 1+5 | 39 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_QN_IDN_SLIM_OFF` | DELETE-OLD-PATH | b | 1+4 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF` | DELETE-OLD-PATH | b | 1+2 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_QN_TILED_OFF` | DELETE-OLD-PATH | e | 2+8 | 10 | EXP:2203 DELETED (serial shape) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_RIDGE_NO_U_OFF` | DELETE-OLD-PATH | d | 1+4 | 19 | EXP:246 KEPT 2caff7b0c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_RIDGE_RESIDENT_OFF` | DELETE-OLD-PATH | d | 2+10 | 29 | EXP:247 KEPT 2caff7b0c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SVM_IDN_FUSED_TILE_OFF` | DELETE-OLD-PATH | b | 1+2 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF` | DELETE-OLD-PATH | b | 1+2 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF` | DELETE-OLD-PATH | b | 1+3 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SVM_RBF_EPILOGUE_1D` | DELETE-ARM | a | 1+2 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SVM_WS1024` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_PREP_FAST_FIT_TRANSFORM_FUSED` | DELETE-ARM | a | 1+2 | 4 | EXP:1597 OPEN -- no A/B recorded (Apple fast opt-in) |

Files: `decomposition/host/jacobi_select_host.mojo`, `experiments/performance_ideas/I12/trials_check.mojo`, `glm/checks/logistic_check.mojo`, `glm/checks/qn_losses_check.mojo`, `glm/estimator.mojo`, `glm/host/qn_oracle.mojo`, `glm/impl/linalg/detail/lstsq.mojo`, `glm/impl/linalg/detail/lstsq_min_norm.mojo`, `glm/impl/linalg/detail/svd.mojo`, `glm/impl/qn/fast_xtdz.mojo`, `glm/impl/qn/glm_base.mojo`, `glm/impl/qn/qn_linesearch.mojo`, `glm/impl/ridge.mojo`, `glm/linear_route_controls.mojo`, `glm/linear_svc_native_device.mojo`, `preprocessing/estimator.mojo`, `preprocessing/minmax.mojo`, `solver/impl/cd.mojo`, `solver/impl/cd_gram_rule.mojo`, `svm/impl/distance/kernel_matrices.mojo`, `svm/impl/fast_update_f.mojo`, `svm/impl/smosolver.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_L05`, `AFCL_L06`, `CD_FAST_GRAM_OFF`, `CD_FUSED_OFF`, `CD_GRAM_BLOCKS`, `CD_IDN_GRAM_COLS_128`, `CD_IDN_GRAM_COLS_256`, `CD_IDN_GRAM_COST_RULE_OFF`, `CD_IDN_GRAM_OFF`, `CD_IDN_GRAM_WIDE`, `CD_THREE_LAUNCH_OFF`, `CD_TWO_STEP_OFF`, `IDN_QN_EXACT_TRIALS`, `IDN_SVC_ONE_BLOCK_OFF`, `LEGACY_NARROW_FX_D`, `LEGACY_NARROW_QN_ALL`, `LSVR_ALL_OFF`, `LSVR_EVAL_SLIM_OFF`, `LSVR_LINESEARCH_BATCH_OFF`, `OLS_FAST_SMALL_NT`, `PREP_FAST_CLS2_MINMAX_FUSED_OFF`, `PREP_FAST_CLS2_MINMAX_POOL_OFF`, `PURITY2_2_OFF`, `PURITY_1_OFF`, `QN_DEV_ARGMAX_OFF`, `QN_FAST_XTDZ_OFF`, `QN_IDN_FUSED_OFF`, `QN_IDN_SLIM_OFF`, `QN_OVR_ONE_UPLOAD_OFF`, `QN_TILED_OFF`, `RIDGE_NO_U_OFF`, `RIDGE_RESIDENT_OFF`, `SVM_IDN_FUSED_TILE_OFF`, `SVM_IDN_FUSED_UPDATE_F_OFF`, `SVM_IDN_NAN_SCAN_ONE_OFF`, `SVM_RBF_EPILOGUE_1D`, `X_PREP_FAST_FIT_TRANSFORM_FUSED`

### expdel2-21-linear (44 defines, 26 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_L01` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L02` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L03` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L04` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_L14` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ARD_SIGMA_QOLD` | DELETE-OLD-PATH | b | 1+6 | 7 | EXP:2124 owed (boxes down) -- no A/B recorded for the old path |
| `MOJOLEARN_BAYES_GRID_GUARD_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:249 KEPT 269ffa57a -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CALIB_GNB_FOLDS_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:237 KEPT 24ed76679 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS` | DELETE-ARM | a | 1+18 | 46 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C14_GROUP_RHS` | DELETE-ARM | a | 1+5 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C16_GLM_FUSED` | DELETE-ARM | a | 1+7 | 28 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C17_OVR` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C19_SGD_CHUNK` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED` | DELETE-ARM | a | 1+14 | 97 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GLM_CELLS_MIXED` | DELETE-ARM | a | 2+0 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_HUBER_DEVICE_LBFGS_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:1256 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_HUBER_FAST_BLOCK512` | DELETE-ARM | e | 1+3 | 7 | EXP:1423 KEEP (FAST+Apple default, `_OFF`) -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_LARS_GRID_GRAM_OFF` | DELETE-OLD-PATH | e | 1+3 | 19 | EXP:2032 KEEP (default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD` | DELETE-OLD-PATH | b | 1+1 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_SVM_WS` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1344 NEUTRAL, not merged -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_RIDGE_FF_ALWAYS` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SGDOC_FAST_TAIL_LONG` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:16 / `MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF` (both modes, default on; PENDING) / sk-kern -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_AUTO_STEPS_256_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:17 SGDClassifier/SGDRegressor/Perceptron/PassiveAggressive pair 'auto': batch = n_s -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_FAST_EPOCH_RESIDENT` | DELETE-ARM | a | 1+4 | 5 | EXP:1596 OPEN -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF` | DELETE-OLD-PATH | b | 1+4 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_IDN_DEV_FINITE_OFF` | DELETE-OLD-PATH | b | 1+4 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_IDN_MB_FUSE_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_IDN_OVR_PAR_OFF` | DELETE-OLD-PATH | b | 1+4 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_IDN_PS_WARP_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_SVM_FAST_SORT_OFF` | DELETE-OLD-PATH | b | 1+5 | 12 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_SVM_WS1024` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF` | DELETE-OLD-PATH | b | 1+15 | 29 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_X_LINEAR_ARD_LEAD` | DELETE-ARM | a | 1+0 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_LINEAR_FAST_FMA` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_X_LINEAR_NO_STAGED_FOLD` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X_PREP_FAST_CLS2_PACK_OFF` | DELETE-OLD-PATH | e | 1+7 | 8 | EXP:406 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_PREP_FAST_CLS2_PRESENT_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:407 KEEP, FAST+Apple default (`_OFF` off) -- record read: the default is the recorded winner or the kept path |

Files: `svm/impl/workingset.mojo`, `x_linear/ard_grid.mojo`, `x_linear/bayes.mojo`, `x_linear/cd_grid.mojo`, `x_linear/device.mojo`, `x_linear/enetcv_fast.mojo`, `x_linear/fast_gram.mojo`, `x_linear/finite_device.mojo`, `x_linear/glm_host.mojo`, `x_linear/huber_fast.mojo`, `x_linear/huber_grid.mojo`, `x_linear/lars.mojo`, `x_linear/logcv.mojo`, `x_linear/logcv_grid.mojo`, `x_linear/logcv_host.mojo`, `x_linear/ops.mojo`, `x_linear/ops_entry_device.mojo`, `x_linear/quantile_grid.mojo`, `x_linear/ridge.mojo`, `x_linear/ridge_descriptor_device.mojo`, `x_linear/ridgecv.mojo`, `x_linear/sgd.mojo`, `x_linear/tops.mojo`, `x_linear/x_linear_host_impl.mojo`, `x_prep/calib.mojo`, `x_prep/cat_cls2.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_L01`, `AFCL_L02`, `AFCL_L03`, `AFCL_L04`, `AFCL_L14`, `ARD_SIGMA_QOLD`, `BAYES_GRID_GUARD_OFF`, `CALIB_GNB_FOLDS_OFF`, `GLM_CELLS_MIXED`, `HUBER_DEVICE_LBFGS_OFF`, `HUBER_FAST_BLOCK512`, `IDN_LARS_GRID_GRAM_OFF`, `LEGACY_NARROW_SGD_PS_SIMD`, `LEGACY_NARROW_SVM_WS`, `RIDGE_FF_ALWAYS`, `SGDOC_FAST_TAIL_LONG`, `SGD_ADAPTIVE_BATCH_OFF`, `SGD_AUTO_STEPS_256_OFF`, `SGD_FAST_EPOCH_RESIDENT`, `SGD_IDN_CHUNK_WIDE_OFF`, `SGD_IDN_DEV_FINITE_APPLE_OFF`, `SGD_IDN_DEV_FINITE_OFF`, `SGD_IDN_MB_FUSE_OFF`, `SGD_IDN_OVR_PAR_OFF`, `SGD_IDN_PS_WARP_OFF`, `SGD_OC_IDN_MB_DEFAULT_OFF`, `SGD_OC_MB_IMPLICIT_OFF`, `SVM_FAST_SORT_OFF`, `SVM_WS1024`, `XLIN_IDN_DEV_FINITE_OFF`, `X_LINEAR_ARD_LEAD`, `X_LINEAR_FAST_FMA`, `X_LINEAR_NO_STAGED_FOLD`, `X_PREP_FAST_CLS2_PACK_OFF`, `X_PREP_FAST_CLS2_PRESENT_OFF`

Lane result (expdel2-21-linear, 2026-11-15): 34 of the 35 owned defines deleted with their records; 8 non-owner defines folded in this lane's files (`IDN_ALL_OFF`, `CLASSICAL_C13_LOGCV_WEIGHTS`, `CLASSICAL_C14_GROUP_RHS`, `CLASSICAL_C15_FACTOR_SOLVE`, `CLASSICAL_C16_GLM_FUSED`, `CLASSICAL_C17_OVR`, `CLASSICAL_C19_SGD_CHUNK`, `CLASSICAL_RIDGECV_FF_BLOCKED`). KEPT: `COLUMN_CPU` (correction); `IDN_LARS_GRID_GRAM_OFF` (KEEP-OPEN: its old path, main's moments grid, is the measured winner on lars istella, NV 165.6 vs 340.7 ms, AMD 390.2 vs 1155.2 ms, x_linear/lars.mojo OPEN FOLLOW-UP; "A measured winner is never deleted on reasoning"). Left for other lanes: x_linear/ridge_ff_blocks.mojo (the rff_* kernels), x_linear/glm.mojo `_unit_all` have no reader left in this lane's files.

### expdel2-22-linear (45 defines, 14 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_P01` | DELETE-EXPERIMENT | g | 2+8 | 34 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P02` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P05` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P06` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P13` | DELETE-EXPERIMENT | g | 1+5 | 18 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_CLASSICAL_C07_RADIX_BITS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C07_RADIX_ROWS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT` | DELETE-ARM | a | 1+4 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF` | DELETE-OLD-PATH | e | 1+4 | 5 | EXP:1740 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD` | DELETE-ARM | a | 1+9 | 53 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF` | DELETE-OLD-PATH | e | 1+4 | 7 | EXP:2060 KEEP (promoted 2026-10-11, w2-decide-algos; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_II_CONV_TREE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_II_GRAM_TILE_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LABEL_OFF` | DELETE-OLD-PATH | b | 1+13 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RR_EIGH_OFF` | DELETE-OLD-PATH | b | 1+14 | 19 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RR_MIN_16` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RR_MIN_48` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RR_MIN_8` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RR_MIN_96` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TE_ENC_TREE_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XPREP_RADIX_OFF` | DELETE-OLD-PATH | b | 1+3 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LABEL_DIRECT_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:762 KEEP, default merged on main1975c2dc5 -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_LDAQDA_DEC_TILE_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:241 KEPT b2dd5dfe5 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LDAQDA_PAR_STAGES_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:242 KEPT b2dd5dfe5 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LDAQDA_RR_EIGH_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:243 KEPT b2dd5dfe5 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS` | DELETE-OLD-PATH | b | 1+1 | 3 | EXP:1333 OWED -- no A/B recorded for the old path |
| `MOJOLEARN_MI_CLF_RANKMAJOR` | DELETE-ARM | e | 1+2 | 8 | EXP:374 KEEP (FAST + Apple default, _OFF) -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_MI_CLF_RANKMAJOR_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:374 KEEP (FAST + Apple default, _OFF) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MI_REG_RANKMAJOR` | DELETE-ARM | a | 1+3 | 4 | EXP:377 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MI_REG_SORTCOUNT` | DELETE-ARM | e | 1+2 | 10 | EXP:379 KEEP (see MI_REG_TIES row) -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_MI_REG_TIES` | DELETE-ARM | e | 1+3 | 4 | EXP:379 KEEP (see MI_REG_TIES row) -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_MI_REG_TIES_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:379 KEEP (see MI_REG_TIES row) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_NB_CAT_ATOMIC` | DELETE-ARM | a | 1+2 | 25 | EXP:289 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PREP3_MAXABS_OFF` | DELETE-OLD-PATH | d | 1+9 | 11 | EXP:383 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PREP3_MAXABS_POOL_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:920 DEFAULT; M2 default/_OFF both rc=0, merged. Only MaxAbs paths extracted; resampl -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PT_SCORE` | DELETE-ARM | e | 1+5 | 11 | EXP:2280 DELETED (DROP quality) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_SI_ONEPASS` | DELETE-ARM | e | 1+4 | 7 | EXP:1888 DELETED (quality, with COLBATCH) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_SI_ONEPASS_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1888 DELETED (quality, with COLBATCH) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XPREP_ARENA_POOL_OFF` | DELETE-OLD-PATH | e | 2+5 | 9 | EXP:1787 CONFIRMED (kept default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:252 KEPT c5e1bbeb6 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_X_PREP_POOL_ARENA_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:409 DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_PREP_POOL_ARENA_OFF` -- promoted on one vendor in the record; tombstone notes the owed other vendor |

Files: `x_prep/device.mojo`, `x_prep/dradix.mojo`, `x_prep/fastmaxabs.mojo`, `x_prep/fastnb.mojo`, `x_prep/fastprep2.mojo`, `x_prep/fastpt.mojo`, `x_prep/fastred.mojo`, `x_prep/host/program.mojo`, `x_prep/host/rr_eigh_host.mojo`, `x_prep/host/target.mojo`, `x_prep/idn_blocked.mojo`, `x_prep/idn_fold.mojo`, `x_prep/iterative.mojo`, `x_prep/label_fast.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_P01`, `AFCL_P02`, `AFCL_P03`, `AFCL_P05`, `AFCL_P06`, `AFCL_P13`, `CLASSICAL_TE_BLOCKED_FOLD`, `IDN_DRADIX_RANGE_PASSES_OFF`, `IDN_II_CONV_TREE_OFF`, `IDN_II_GRAM_TILE_OFF`, `IDN_LABEL_OFF`, `IDN_NB_CAT_ATOMIC_OFF`, `IDN_RR_EIGH_OFF`, `IDN_RR_MIN_16`, `IDN_RR_MIN_48`, `IDN_RR_MIN_8`, `IDN_RR_MIN_96`, `IDN_TE_ENC_TREE_OFF`, `IDN_TE_GLOBAL_TREE_OFF`, `IDN_XPREP_RADIX_OFF`, `LABEL_DIRECT_OFF`, `LDAQDA_DEC_TILE_OFF`, `LDAQDA_PAR_STAGES_OFF`, `LDAQDA_RR_EIGH_OFF`, `LEGACY_NARROW_FASTPT_ROWS`, `MI_CLF_RANKMAJOR`, `MI_CLF_RANKMAJOR_OFF`, `MI_REG_RANKMAJOR`, `MI_REG_SORTCOUNT`, `MI_REG_TIES`, `MI_REG_TIES_OFF`, `NB_CAT_ATOMIC`, `PT_SCORE`, `SI_ONEPASS`, `SI_ONEPASS_OFF`, `X_PREP_CLASS_COV_GRID_OFF`, `X_PREP_POOL_ARENA_OFF`

### expdel2-23-neural (41 defines, 20 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_MAMBA3_BWD_ARENA` | DELETE-ARM | f | 2+10 | 50 | EXP:634 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA3_BWD_CHUNK` | DELETE-ARM | f | 2+2 | 4 | EXP:635 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_ALL` | DELETE-ARM | f | 5+17 | 60 | EXP:652 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AUTOGRAD_FOLD_V2_OFF` | DELETE-OLD-PATH | b | 1+13 | 29 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_MODEL_UNFUSED` | DELETE-ARM | a | 1+4 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_GENERIC_OPS` | DELETE-ARM | a | 2+5 | 7 | EXP:2375 PENDING (A = restored ops are the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CNN_GENERIC_OPS` | DELETE-ARM | a | 1+5 | 6 | EXP:2374 PENDING (A = restored fused block is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS` | DELETE-ARM | a | 2+9 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_SERIAL_RUN_BEGIN` | DELETE-ARM | a | 1+0 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL` | DELETE-ARM | a | 2+5 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CE_GRAD_FUSED` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ELEM_VEC4_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:2175 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_FOLD_STAGED_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:2173 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_BWD_CELL_OFF` | DELETE-OLD-PATH | b | 2+14 | 82 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_GRAD_FOLD` | DELETE-ARM | a | 3+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M2_SSD_TILES_OFF` | DELETE-OLD-PATH | b | 1+6 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF` | DELETE-OLD-PATH | e | 2+6 | 18 | EXP:1755 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF` | DELETE-OLD-PATH | b | 2+8 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF` | DELETE-OLD-PATH | b | 2+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE` | DELETE-ARM | a | 1+4 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF` | DELETE-OLD-PATH | e | 2+0 | 2 | EXP:2014 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD` | DELETE-OLD-PATH | b | 1+4 | 10 | EXP:2418 PENDING (A = no per-call stage allocation, no h_last copy, no host state round t -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD` | DELETE-OLD-PATH | b | 1+4 | 5 | EXP:2421 PENDING (A = 0.8.55's `_mamba1_run_arena` / `_mamba2_run_arena` form is the defa -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA_BWD_FOLDS` | DELETE-ARM | a | 2+4 | 14 | EXP:2271 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF` | DELETE-OLD-PATH | e | 1+7 | 21 | EXP:1944 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TRANSPOSE_TILE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:2176 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_LEGACY_REFUSAL` | DELETE-OLD-PATH | b | 1+4 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_S16_QK_NAIVE` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_S16_V_NAIVE` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE` | DELETE-ARM | a | 3+7 | 37 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_S17_TAIL_NAIVE` | DELETE-ARM | a | 1+6 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA_CALL_PROGRAM` | DELETE-ARM | a | 1+3 | 4 | EXP:2404 PENDING (A = 0.8.55's per-call shape is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN14_BOUNDED_IM2COL` | DELETE-ARM | e | 1+19 | 77 | EXP:1943 PROMOTED (IDENTICAL default) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_NN50_OWNED_RUNS` | DELETE-ARM | a | 1+3 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRANSFORMER_GENERIC_PREFILL` | DELETE-ARM | a | 1+0 | 1 | EXP:2376 PENDING (A = 0.8.66's session fresh prefill is the default) -- no A/B recorded for the opt-in arm |

Files: `autograd/block_step_ops.mojo`, `autograd/checks/fold_cases.mojo`, `autograd/checks/pass1_cases.mojo`, `autograd/cnn_block_op.mojo`, `autograd/data.mojo`, `autograd/embedding_ops.mojo`, `autograd/fold.mojo`, `autograd/fold_kernels.mojo`, `autograd/kernels/cnn.mojo`, `autograd/kernels/embedding.mojo`, `autograd/kernels/embedding_runs.mojo`, `autograd/kernels/embedding_sort.mojo`, `autograd/kernels/lm.mojo`, `autograd/kernels/mamba1_scan.mojo`, `autograd/kernels/mamba2_ssd.mojo`, `autograd/kernels/mamba3.mojo`, `autograd/kernels/mamba3_prefill.mojo`, `autograd/kernels/mamba_bwd_folds.mojo`, `autograd/kernels/mamba_refusal.mojo`, `autograd/kernels/moe.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN_MAMBA3_BWD_ARENA`, `AFN_MAMBA3_BWD_CHUNK`, `AFN_SAMBA_ALL`, `AUTOGRAD_GATHER_EPOCH_OFF`, `BYTE_LM_GENERIC_OPS`, `CNN_GENERIC_OPS`, `COLUMN_CPU`, `EMB_AUTO_SORT_MIN_CELLS`, `EMB_SERIAL_RUN_BEGIN`, `EMB_SORT_NEGATIVE_CONTROL`, `IDN_ELEM_VEC4_OFF`, `IDN_EMB_RADIX_SORT_OFF`, `IDN_FOLD_STAGED_OFF`, `IDN_M2_BWD_CELL_OFF`, `IDN_M3_ANGLE_CARRY_CACHE_OFF`, `IDN_M3_ANGLE_DT_SUFFIX_OFF`, `IDN_M3_BWD_SKIP_DEAD_OFF`, `IDN_M3_DECODE_WINDOW_REUSE`, `IDN_M3_S17_TAIL_PIPE_OFF`, `IDN_MAMBA12_FORWARD_COPIES_OLD`, `IDN_MAMBA12_FORWARD_DOOR_OLD`, `IDN_MAMBA_BWD_FOLDS`, `IDN_TRANSPOSE_TILE_OFF`, `MAMBA3_ANGLE_DT_NAIVE`, `MAMBA3_LEGACY_REFUSAL`, `MAMBA3_S16_QK_NAIVE`, `MAMBA3_S16_V_NAIVE`, `MAMBA3_S17_OPERANDS_NAIVE`, `MAMBA3_S17_TAIL_NAIVE`, `MAMBA_CALL_PROGRAM`, `NN50_OWNED_RUNS`, `TRANSFORMER_GENERIC_PREFILL`

### expdel2-24-neural (44 defines, 20 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN26_LM_BWD_EPILOGUE` | DELETE-ARM | e | 1+2 | 15 | EXP:2201 DELETED (removed) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_AFN26_LM_BWD_FUSE` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_BWD_NORM1_RESID` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_WGRAD_CAP8` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_WGRAD_MIN128` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_WGRAD_SPLIT` | DELETE-ARM | a | 1+2 | 20 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_LMGRAD_ALL` | DELETE-ARM | f | 1+3 | 4 | EXP:620 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_BWD_EPILOGUE` | DELETE-ARM | e | 1+2 | 15 | EXP:2201 DELETED (removed) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_AFN_LM_BWD_FUSE` | DELETE-ARM | f | 1+4 | 5 | EXP:623 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_BWD_NORM1_RESID` | DELETE-ARM | f | 1+1 | 2 | EXP:624 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | DELETE-ARM | f | 1+4 | 5 | EXP:625 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_WGRAD_SPLIT` | DELETE-ARM | f | 1+2 | 20 | EXP:629 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:2178 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_FOLD_V2_OFF` | DELETE-OLD-PATH | b | 1+13 | 29 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_HAS_INFERENCE` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF` | DELETE-OLD-PATH | b | 1+3 | 6 | EXP:2174 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF` | DELETE-OLD-PATH | b | 1+11 | 71 | EXP:2172 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_MODEL_UNFUSED` | DELETE-ARM | a | 1+4 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:2169 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_PAIR_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_PASS3_DB_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF` | DELETE-OLD-PATH | b | 2+3 | 10 | EXP:2171 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | EXP:2177 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE` | DELETE-OLD-PATH | b | 1+2 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS` | DELETE-ARM | a | 2+9 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF` | DELETE-OLD-PATH | b | 1+2 | 19 | EXP:2168 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC` | DELETE-ARM | a | 1+6 | 11 | EXP:2410 PENDING (A = the restored hand chain is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:2373 KEPT default again (restored from removed) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ROW_COOP_OFF` | DELETE-OLD-PATH | b | 1+13 | 20 | EXP:2167 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF` | DELETE-OLD-PATH | b | 1+6 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_TILED_PREDICT` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF` | DELETE-OLD-PATH | b | 4+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LN_FAST_BLOCK_FOLD` | DELETE-ARM | a | 1+5 | 8 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_LN_FOLD_BLOCK_OFF` | DELETE-OLD-PATH | b | 1+5 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_NN50_OWNED_RUNS` | DELETE-ARM | a | 1+3 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SEQ_FAST_PIPE_CH` | DELETE-ARM | a | 1+10 | 11 | EXP:680 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SMALL_MLP_GENERIC_OPS` | DELETE-ARM | a | 1+2 | 13 | EXP:2372 PENDING (A = restored tuned op is the default) -- no A/B recorded for the opt-in arm |

Files: `autograd/kernels/transformer.mojo`, `autograd/kernels/transformer_afn.mojo`, `autograd/layers.mojo`, `autograd/lm_embedding_op.mojo`, `autograd/loss_ops.mojo`, `autograd/losses.mojo`, `autograd/metrics.mojo`, `autograd/model.mojo`, `autograd/ops.mojo`, `autograd/optim.mojo`, `autograd/pass1_ops.mojo`, `autograd/program_output_copy.mojo`, `autograd/row_ops.mojo`, `autograd/rules_lm.mojo`, `autograd/rules_mamba.mojo`, `autograd/rules_recurrent.mojo`, `autograd/rules_transformer.mojo`, `autograd/small_mlp_op.mojo`, `autograd/small_mlp_program.mojo`, `autograd/tape.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN26_LM_BWD_EPILOGUE`, `AFN26_LM_BWD_NORM1_RESID`, `AFN26_LM_WGRAD_CAP8`, `AFN26_LM_WGRAD_MIN128`, `AFN26_LM_WGRAD_SPLIT`, `AFN_LMGRAD_ALL`, `AFN_LM_BWD_EPILOGUE`, `AFN_LM_BWD_NORM1_RESID`, `AFN_LM_WGRAD_SPLIT`, `AUTOGRAD_CURVE_WORD_OFF`, `AUTOGRAD_FOLD_V2_OFF`, `AUTOGRAD_GRAD_ALIAS_OFF`, `AUTOGRAD_HAS_INFERENCE`, `AUTOGRAD_LOSS_REDUCED_OFF`, `AUTOGRAD_MAMBA_EXPORT_OFF`, `AUTOGRAD_MODEL_UNFUSED`, `AUTOGRAD_OPT_CLIP_FUSE_OFF`, `AUTOGRAD_OPT_FUSE_OFF`, `AUTOGRAD_PAIR_OFF`, `AUTOGRAD_PASS2_FOLD_OFF`, `AUTOGRAD_PASS3_DB_OFF`, `AUTOGRAD_REFUSE_DEFER_OFF`, `AUTOGRAD_TRANSPOSE_VIEW_OFF`, `BWD_NORM_LEGACY_SHAPE_RULE`, `IDN_AUC_BLOCK_SORT_OFF`, `IDN_BWD_ENTRY_ALIAS_OFF`, `IDN_MAMBA_BACKWARD_GENERIC`, `IDN_MLP_GEMM_LEAF128_OFF`, `IDN_ROW_COOP_OFF`, `IDN_SEQ_TILED_PREDICT`, `SMALL_MLP_GENERIC_OPS`

### expdel2-25-neural (36 defines, 26 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_GEMM2_ALL` | DELETE-ARM | f | 1+5 | 6 | EXP:607 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM2_BIGTILE` | DELETE-ARM | f | 1+3 | 10 | EXP:608 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM2_DBUF` | DELETE-ARM | f | 1+3 | 4 | EXP:609 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM2_DIRECT_B` | DELETE-ARM | f | 1+4 | 5 | EXP:610 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM2_SWIZZLE` | DELETE-ARM | f | 1+3 | 4 | EXP:685 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM2_SWZ_G` | DELETE-ARM | f | 1+3 | 4 | EXP:685 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_ALL` | DELETE-ARM | f | 1+6 | 7 | EXP:612 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_BF16_MMA` | DELETE-ARM | f | 1+8 | 22 | EXP:613 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_CORES` | DELETE-ARM | f | 2+7 | 9 | EXP:684 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_EPILOGUE` | DELETE-ARM | f | 1+7 | 69 | EXP:682 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_INT8_MMA` | DELETE-ARM | f | 1+2 | 3 | EXP:614 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_KB` | DELETE-ARM | f | 1+13 | 14 | EXP:683 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_SIMDGROUP` | DELETE-ARM | f | 1+3 | 4 | EXP:617 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_SPLITK` | DELETE-ARM | f | 1+7 | 25 | EXP:618 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_GEMM_TILESHAPE` | DELETE-ARM | f | 1+6 | 16 | EXP:619 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AUTOGRAD_EMBEDDING_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS` | DELETE-ARM | a | 2+9 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EMB_DEVICE_SCRATCH_OFF` | DELETE-OLD-PATH | b | 1+3 | 60 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_EMB_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+4 | 55 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL` | DELETE-ARM | a | 2+5 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GEMM_FAST_RAGGED_K_OFF` | DELETE-OLD-PATH | e | 1+3 | 6 | EXP:2123 KEPT, FAST default (`_OFF`) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2` | DELETE-ARM | e | 2+17 | 26 | src# # descriptor op 16), the measured winner: IDN_SAMBA_RESIDENT_STEP PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_GEMM_INT_SLICES` | DELETE-ARM | e | 2+1 | 6 | EXP:2263 pending -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE` | DELETE-ARM | e | 1+4 | 9 | EXP:1767 DELETED (loser; AMD flat, NV 1.9x) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE` | DELETE-ARM | e | 1+2 | 3 | EXP:1746 DROP (slower), arm deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN07` | DELETE-ARM | a | 1+7 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN12` | DELETE-ARM | a | 1+10 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN16` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NORM_FIXED_LANES` | DELETE-ARM | a | 1+16 | 49 | EXP:2246 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_INT_SLICES_FORCE_FLAT` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI07_GROUPED_PROJECTIONS` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN14_BOUNDED_IM2COL` | DELETE-ARM | e | 1+19 | 77 | EXP:1943 PROMOTED (IDENTICAL default) -- record read: loser, superseded, default-equal force knob, or never measured |

Files: `autograd/transformer_primitives_ops.mojo`, `bindings/_mojolearn_training_host.mojo`, `bindings/_mojolearn_x_cnn.mojo`, `bindings/_mojolearn_x_cnn_host.mojo`, `checks/kernel_matrix.mojo`, `cholesky/checks/potrf.mojo`, `cluster/checks/reduce_by_key.mojo`, `embedding/checks/embedding_identical.mojo`, `embedding/embedding_words_run.mojo`, `experiments/performance_ideas/I07/check.mojo`, `experiments/performance_ideas/I21/component_check.mojo`, `gaussian_process/checks/kernel_gradient.mojo`, `gemm/afn_apple_fast.mojo`, `gemm/afn_apple_fast2.mojo`, `gemm/apple_fast_bf16_public_run.mojo`, `gemm/checks/gemm_identical.mojo`, `gemm/checks/gemm_lowbit.mojo`, `gemm/experiments/neural_grouped.mojo`, `gemm/experiments/neural_plans.mojo`, `gemm/experiments/neural_profile.mojo`, `gemm/experiments/neural_streaming.mojo`, `gemm/experiments/neural_switches.mojo`, `gemm/experiments/neural_tiled_v2.mojo`, `gemm/host/linalg_native.mojo`, `gemm/host_entry.mojo`, `gemm/int_slices.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN_GEMM2_ALL`, `AFN_GEMM2_BIGTILE`, `AFN_GEMM2_DBUF`, `AFN_GEMM2_DIRECT_B`, `AFN_GEMM2_SWIZZLE`, `AFN_GEMM2_SWZ_G`, `AFN_GEMM_ALL`, `AFN_GEMM_BF16_MMA`, `AFN_GEMM_CORES`, `AFN_GEMM_EPILOGUE`, `AFN_GEMM_INT8_MMA`, `AFN_GEMM_KB`, `AFN_GEMM_SIMDGROUP`, `AFN_GEMM_SPLITK`, `AFN_GEMM_TILESHAPE`, `AUTOGRAD_EMBEDDING_OFF`, `EMB_DEVICE_SCRATCH_OFF`, `EMB_RESIDENT_OFF`, `GEMM_FAST_RAGGED_K_OFF`, `IDN_GEMM_INT_SLICES`, `IDN_NEURAL_CHAINS`, `IDN_NEURAL_GEMM_EPILOGUE`, `IDN_NEURAL_GEMM_SCHEDULE`, `IDN_NEURAL_GEMM_SCHEDULE_ROLES`, `IDN_NEURAL_NN07`, `IDN_NEURAL_NN12`, `IDN_NEURAL_NN16`, `INT_SLICES_FORCE_FLAT`, `NI07_GROUPED_PROJECTIONS`

### expdel2-26-neural (45 defines, 13 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_MAMBA3_BWD_ARENA` | DELETE-ARM | f | 2+10 | 50 | EXP:634 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA3_BWD_CHUNK` | DELETE-ARM | f | 2+2 | 4 | EXP:635 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_ALL` | DELETE-ARM | f | 5+17 | 60 | EXP:652 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M1_SCAN` | DELETE-ARM | e | 2+5 | 7 | EXP:2071 LOSER (deleted) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_M2_BWD_CELL_OFF` | DELETE-OLD-PATH | b | 2+14 | 82 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_CB_LOWER_OFF` | DELETE-OLD-PATH | b | 3+12 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_GRAD_FOLD` | DELETE-ARM | a | 3+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M2_SSD_TILES_OFF` | DELETE-OLD-PATH | b | 1+6 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128` | DELETE-ARM | a | 2+24 | 26 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32` | DELETE-ARM | a | 2+24 | 26 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF` | DELETE-OLD-PATH | e | 2+6 | 18 | EXP:1755 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF` | DELETE-OLD-PATH | b | 2+8 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_ANGLE_PARALLEL` | DELETE-ARM | a | 3+6 | 126 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF` | DELETE-OLD-PATH | b | 2+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF` | DELETE-OLD-PATH | e | 2+0 | 2 | EXP:2014 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MAMBA_BWD_FOLDS` | DELETE-ARM | a | 2+4 | 14 | EXP:2271 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF` | DELETE-OLD-PATH | b | 1+5 | 33 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_FILL_BLOCKS` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_FOLD_SHARED` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_GEMM_ARM` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL` | DELETE-ARM | a | 2+12 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN07` | DELETE-ARM | a | 1+7 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN12` | DELETE-ARM | a | 1+10 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_PAIR_CONTROL` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_STAGE_DEPTH` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_STAGE_PAD` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_STREAM_GROUP` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT` | DELETE-OLD-PATH | b | 3+8 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_QKS` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_STATEPASS` | DELETE-OLD-PATH | b | 2+2 | 37 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_YINTRA` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_YSTATE` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE` | DELETE-ARM | a | 3+7 | 37 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_TILED_INCREMENT` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_TILED_YINTRA` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN35_CONV_PAIR` | DELETE-ARM | a | 4+8 | 66 | no record -- no A/B recorded for the opt-in arm |

Files: `gemm/neural_dispatch.mojo`, `mamba/checks/mamba2_oracle.mojo`, `mamba/checks/mamba3_oracle.mojo`, `mamba/checks/mamba_rms_fold.mojo`, `mamba/host/gen/mamba2.mojo`, `mamba/host/gen/mamba2_ssd_backward.mojo`, `mamba/host/gen/mamba3_backward.mojo`, `mamba/host/gen/mamba3_prefill_backward.mojo`, `mamba/host/gen/mamba3_siso.mojo`, `mamba/host/gen/mamba_bwd_folds.mojo`, `mamba/host/gen/modeling_mamba.mojo`, `mamba/host/gen/selective_scan_backward.mojo`, `mamba/host/gen/selective_scan_interface.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `IDN_M1_SCAN`, `IDN_M2_CB_LOWER_OFF`, `IDN_M3_ANGLE_BLOCK_128`, `IDN_M3_ANGLE_BLOCK_32`, `IDN_M3_ANGLE_PARALLEL`, `IDN_M3_YINTRA_RESOURCE_TILE`, `IDN_NEURAL_FILL_BLOCKS`, `IDN_NEURAL_FOLD_SHARED`, `IDN_NEURAL_GEMM_ARM`, `IDN_NEURAL_GEMM_CONTROL`, `IDN_NEURAL_PAIR_CONTROL`, `IDN_NEURAL_RETAINED_FLOATS`, `IDN_NEURAL_STAGE_DEPTH`, `IDN_NEURAL_STAGE_PAD`, `IDN_NEURAL_STAGE_SWIZZLE`, `IDN_NEURAL_STREAM_GROUP`, `MAMBA3_BALANCED_INCREMENT_TILE`, `MAMBA3_LEGACY_ANGLE_INCREMENT`, `MAMBA3_LEGACY_INCREMENT_TILE`, `MAMBA3_LEGACY_INCREMENT_V`, `MAMBA3_LEGACY_QKS`, `MAMBA3_LEGACY_STATEPASS`, `MAMBA3_LEGACY_YINTRA`, `MAMBA3_LEGACY_YSTATE`, `MAMBA3_TILED_INCREMENT`, `MAMBA3_TILED_YINTRA`, `NN35_CONV_PAIR`

### expdel2-27-neural (32 defines, 10 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_EPI_ALL` | DELETE-ARM | f | 2+7 | 54 | EXP:606 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA1_CHUNKSCAN` | DELETE-ARM | f | 1+0 | 1 | EXP:631 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA1_FUSE_IN` | DELETE-ARM | f | 1+0 | 1 | EXP:632 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA2_SSD_MMA` | DELETE-ARM | f | 1+0 | 1 | EXP:633 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA3_SISO_FUSED` | DELETE-ARM | f | 1+0 | 1 | EXP:636 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA_ALL` | DELETE-ARM | f | 1+7 | 8 | EXP:637 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA_ARENA` | DELETE-ARM | f | 1+0 | 1 | EXP:638 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL` | DELETE-ARM | f | 1+2 | 4 | EXP:639 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE` | DELETE-ARM | f | 1+3 | 5 | EXP:641 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_MAMBA_PROJ_SPLITK` | DELETE-ARM | f | 1+4 | 20 | EXP:641 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ACT_RETAIN` | DELETE-ARM | e | 2+4 | 6 | EXP:2365 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_ACT_RETAIN_OFF` | DELETE-OLD-PATH | e | 2+4 | 6 | EXP:2365 DELETED (superseded) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M2_CB_LOWER_OFF` | DELETE-OLD-PATH | b | 3+12 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_CS_PT` | DELETE-ARM | a | 2+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M2_RETAIN_GL_OFF` | DELETE-OLD-PATH | e | 2+4 | 6 | EXP:2013 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M2_SSD_TILES_OFF` | DELETE-OLD-PATH | b | 1+6 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_M2_YD_ROWS` | DELETE-ARM | e | 2+6 | 8 | EXP:2012 PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_M2_YOFF_EXP` | DELETE-ARM | a | 2+7 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MAMBA_ARENA_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF` | DELETE-OLD-PATH | b | 1+5 | 33 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SSD_OCCUPANCY` | DELETE-ARM | a | 1+4 | 18 | EXP:2239 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER` | DELETE-OLD-PATH | b | 1+3 | 20 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA_2712_UNBOUNDED` | DELETE-ARM | a | 2+5 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN35_CONV_PAIR` | DELETE-ARM | a | 4+8 | 66 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN37_TRIANGLE_TASKS` | DELETE-ARM | a | 2+4 | 31 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN40_OWNED_WEIGHTS` | DELETE-ARM | a | 2+5 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN40_WS_HEADROOM` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |

Files: `mamba/host/gen/ssd_minimal.mojo`, `mamba/host/mamba_host_shell_impl.mojo`, `mamba/impl/mamba_shell_impl.mojo`, `mamba/impl/modeling/modeling_mamba.mojo`, `mamba/impl/modules/afn_defines.mojo`, `mamba/impl/modules/afn_proj_gemm.mojo`, `mamba/impl/modules/idn_gemm_ws.mojo`, `mamba/impl/modules/mamba2.mojo`, `mamba/impl/modules/ssd_minimal.mojo`, `mamba/impl/ops/m1_persistent_scan.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN_EPI_ALL`, `AFN_MAMBA1_CHUNKSCAN`, `AFN_MAMBA1_FUSE_IN`, `AFN_MAMBA2_SSD_MMA`, `AFN_MAMBA3_SISO_FUSED`, `AFN_MAMBA_ALL`, `AFN_MAMBA_ARENA`, `AFN_MAMBA_DEVICE_REFUSAL`, `AFN_MAMBA_PROJ_EPILOGUE`, `AFN_MAMBA_PROJ_SPLITK`, `IDN_ACT_RETAIN`, `IDN_ACT_RETAIN_OFF`, `IDN_M1_PERSISTENT_SCAN_CH`, `IDN_M1_PERSISTENT_SCAN_TOKENS`, `IDN_M2_CS_PT`, `IDN_M2_RETAIN_GL_OFF`, `IDN_M2_SSD_TILES_OFF`, `IDN_M2_YD_ROWS`, `IDN_M2_YOFF_EXP`, `IDN_MAMBA_ARENA_OFF`, `IDN_MAMBA_GEMM_WS_OFF`, `IDN_SSD_OCCUPANCY`, `MAMBA3_LEGACY_CALLER_TRANSFER`, `MAMBA3_LEGACY_DIRECT_TRANSFER`, `MAMBA_2712_UNBOUNDED`, `NN37_TRIANGLE_TASKS`, `NN40_OWNED_WEIGHTS`, `NN40_WS_HEADROOM`

### expdel2-28-neural (42 defines, 15 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_CNN_GENERIC_OPS` | DELETE-ARM | a | 1+5 | 6 | EXP:2374 PENDING (A = restored fused block is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CNN_PREDICT_PROGRAM_COPY` | DELETE-ARM | a | 1+6 | 12 | EXP:2411 PENDING (A = unfilled X/output buffers, cached replay, one-chunk output in place -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ETS_TEAM_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | EXP:535 KEPT aec9f8c33 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_AF_COL_CHUNK_FOLD` | DELETE-ARM | a | 1+3 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ETS_TEAM_OFF` | DELETE-OLD-PATH | e | 1+11 | 17 | EXP:2087 KEEP (IDENTICAL default; `_OFF` restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GEMM_SPLITK_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | EXP:2180 PENDING (BIT VERSION) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LN_ROW_WARP_OFF` | DELETE-OLD-PATH | e | 2+9 | 30 | EXP:1945 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_M1_SCAN` | DELETE-ARM | e | 2+5 | 7 | EXP:2071 LOSER (deleted) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_M2_GRAD_FOLD` | DELETE-ARM | a | 3+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128` | DELETE-ARM | a | 2+24 | 26 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32` | DELETE-ARM | a | 2+24 | 26 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_ANGLE_PARALLEL` | DELETE-ARM | a | 3+6 | 126 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF` | DELETE-OLD-PATH | e | 1+7 | 21 | EXP:1944 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_NM_SPEC_OFF` | DELETE-OLD-PATH | e | 2+6 | 8 | EXP:2086 KEEP (IDENTICAL default; `_OFF` restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ROW_COOP_OFF` | DELETE-OLD-PATH | b | 1+13 | 20 | EXP:2167 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_AF_COOP_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF` | DELETE-OLD-PATH | b | 1+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_LN_COOP_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF` | DELETE-OLD-PATH | b | 1+6 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_PIPE_UP_OFF` | DELETE-OLD-PATH | b | 1+2 | 20 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_WGRAD` | DELETE-ARM | a | 5+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF` | DELETE-OLD-PATH | b | 4+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT` | DELETE-OLD-PATH | b | 3+8 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_QKS` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_STATEPASS` | DELETE-OLD-PATH | b | 2+2 | 37 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_YINTRA` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_LEGACY_YSTATE` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MAMBA3_TILED_INCREMENT` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MAMBA3_TILED_YINTRA` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_MOE_FAST_MMA_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1956 DELETED (slower, bundle) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PROPHET_COOP_OFF` | DELETE-OLD-PATH | d | 1+4 | 7 | EXP:537 KEPT 8c2b7de51 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SEQ_CROSTON_REG_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:539 KEPT 2dcdd949f -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SEQ_FAST_PIPE_CH` | DELETE-ARM | a | 1+10 | 11 | EXP:680 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SEQ_RNN_SCAN_APPLE` | DELETE-ARM | e | 1+4 | 5 | EXP:2165 KEPT (scan default for G = 1) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TSA2_VAR_OFF` | DELETE-OLD-PATH | d | 1+7 | 8 | EXP:550 KEPT 550806bc0 -- promoted on one vendor in the record; tombstone notes the owed other vendor |

Files: `mamba/impl/ops/mamba3_siso.mojo`, `mamba/impl/ops/neural_gradient_profile.mojo`, `mamba/impl/ops/selective_scan_interface.mojo`, `mixture/checks/estep.mojo`, `sequence/adafactor.mojo`, `sequence/checks/oracle.mojo`, `sequence/checks/rnn_scan_check.mojo`, `sequence/cnn_classifier_program.mojo`, `sequence/coop.mojo`, `sequence/croston.mojo`, `sequence/dispatch.mojo`, `sequence/ets_spec.mojo`, `sequence/ets_team.mojo`, `sequence/exec.mojo`, `sequence/exec_device.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `CNN_PREDICT_PROGRAM_COPY`, `ETS_TEAM_OFF`, `IDN_AF_COL_CHUNK_FOLD`, `IDN_ETS_TEAM_OFF`, `IDN_LN_ROW_WARP_OFF`, `IDN_M2_GRAD_FOLD`, `IDN_NM_SPEC_OFF`, `IDN_SEQ_AF_COOP_OFF`, `IDN_SEQ_COOP_NVAMD_OFF`, `IDN_SEQ_LN_COOP_OFF`, `IDN_SEQ_PIPE_DOWN_OFF`, `IDN_SEQ_PIPE_UP_OFF`, `IDN_SEQ_WGRAD`, `IDN_SEQ_WGRAD_BLOCKED_OFF`, `SEQ_CROSTON_REG_OFF`, `SEQ_FAST_PIPE_CH`

Lane result (expdel2-28-neural, 2026-11-15): all 16 owned defines deleted with their records (refusals in core/experiment_guards_neural.mojo and, for the ETS / Croston series switches, core/experiment_guards_series.mojo; grid controls ni54_af_col_fold, m2_grad_fold, seq_wgrad, idn_ets_team, idn_nm_spec, ln_row_warp moved to removed). 20 non-owner defines folded in this lane's files (`IDN_ALL_OFF`, the Mamba-3 angle / tile / legacy switches, `IDN_M1_SCAN`, `CNN_GENERIC_OPS`, comment-only `PROPHET_COOP_OFF`, `IDN_GEMM_SPLITK_OFF`, `SEQ_RNN_SCAN_APPLE`). KEPT: `COLUMN_CPU` (correction). No read in this lane's files: `IDN_MOE_GROUPED_GEMM_OFF`, `IDN_ROW_COOP_OFF`, `IDN_SEQ_GEMM_TILED_OFF`, `MOE_FAST_MMA_OFF`, `TSA2_VAR_OFF`. Left for other lanes: sequence/recurrent.mojo and sequence/layernorm.mojo still read `IDN_SEQ_WGRAD`, `IDN_SEQ_WGRAD_BLOCKED_OFF`, `IDN_LN_ROW_WARP_OFF` (now refused); sequence/adafactor_candidates.mojo af_col_chunk_kernel and mamba/impl/ops/identical_scan_window.mojo have no reader left in this lane's files; the generated mamba/host/gen twins still carry the deleted Mamba arms.

### expdel2-29-neural (45 defines, 17 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1` | DELETE-OLD-PATH | b | 1+2 | 12 | EXP:2160 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_AUTOGRAD_MLP_UNFUSED` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AUTOGRAD_OPTIM` | DELETE-ARM | a | 2+12 | 30 | EXP:2380 lane/r2-restore-unmeasured-defaults @ e048e42e81 (hand OP_OPT update restored as -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_CNN_GENERIC_OPS` | DELETE-ARM | a | 1+5 | 6 | EXP:2374 PENDING (A = restored fused block is the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GARCH_COOP_OFF` | DELETE-OLD-PATH | e | 1+9 | 10 | EXP:564 SUPERSEDED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GARCH_COOP_OFF` | DELETE-OLD-PATH | b | 1+6 | 30 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GARCH_FOLD32_OFF` | DELETE-OLD-PATH | b | 1+6 | 39 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GARCH_LEAD_REG_OFF` | DELETE-OLD-PATH | b | 1+2 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF` | DELETE-OLD-PATH | b | 1+3 | 14 | EXP:2181 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_SPLITK_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | EXP:2180 PENDING (BIT VERSION) -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_STAGE64` | DELETE-ARM | e | 1+3 | 4 | src# #: recurrent trainers ran 2.7x slower on the AMD MI325X after GEMM_STAGE_V2 -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_GEMM_STAGE_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:2164 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LN_ONE_PASS_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:2170 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LN_ROW_WARP_OFF` | DELETE-OLD-PATH | e | 2+9 | 30 | EXP:1945 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF` | DELETE-OLD-PATH | e | 1+7 | 21 | EXP:1944 PROMOTED (IDENTICAL default) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_MOE_STABLE_PACK` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_GEMM_MICRO` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | EXP:2179 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF` | DELETE-OLD-PATH | b | 1+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF` | DELETE-OLD-PATH | b | 1+10 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF` | DELETE-OLD-PATH | b | 1+5 | 73 | EXP:2163 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_LN_LEAF` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF` | DELETE-OLD-PATH | b | 1+3 | 24 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LN_FAST_BLOCK_FOLD` | DELETE-ARM | a | 1+5 | 8 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_LN_FOLD_BLOCK_OFF` | DELETE-OLD-PATH | b | 1+5 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MLP_LOSSBLOCK_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | EXP:2159 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MOE_DEVGROUP_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:587 KEPT 499e1b72c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_MOE_FAST_MMA_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1956 DELETED (slower, bundle) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_MOE_REGTILE_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:588 KEPT 499e1b72c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_NN42_INPUT_BIAS_FUSED` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN44_EXPERT_BISECT` | DELETE-ARM | a | 1+2 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SEQ_FAST_FMA_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:542 KEPT f6a9e7c04 -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SEQ_FAST_VAR_ONECOPY_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:558 DROPPED-noise -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SEQ_GARCH_GRID_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | EXP:546 KEPT 2dcdd949f -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SEQ_GARCH_REG_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:547 KEPT 2dcdd949f -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SEQ_REC_STAGE64` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TSA2_STL_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:549 KEPT 550806bc0 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TSA2_VAR_OFF` | DELETE-OLD-PATH | d | 1+7 | 8 | EXP:550 KEPT 550806bc0 -- promoted on one vendor in the record; tombstone notes the owed other vendor |

Files: `sequence/fit_team.mojo`, `sequence/garch.mojo`, `sequence/gemm_split.mojo`, `sequence/gemm_tiled.mojo`, `sequence/layernorm.mojo`, `sequence/layernorm_core.mojo`, `sequence/mlp.mojo`, `sequence/mlp_fit.mojo`, `sequence/model_device_dispatch.mojo`, `sequence/moe.mojo`, `sequence/moe_group.mojo`, `sequence/moe_grouped_fold.mojo`, `sequence/moe_mma.mojo`, `sequence/moe_reg.mojo`, `sequence/moe_tiled.mojo`, `sequence/ops.mojo`, `sequence/ops_impl.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AUTOGRAD_GEMM_DISPATCH_V1`, `AUTOGRAD_MLP_UNFUSED`, `AUTOGRAD_OPTIM`, `GARCH_COOP_OFF`, `IDN_GARCH_COOP_OFF`, `IDN_GARCH_FOLD32_OFF`, `IDN_GARCH_LEAD_REG_OFF`, `IDN_GEMM_ROUTE_P6_OFF`, `IDN_GEMM_SPLITK_OFF`, `IDN_GEMM_STAGE64`, `IDN_GEMM_STAGE_OFF`, `IDN_LN_ONE_PASS_OFF`, `IDN_MLP_BLOCKED_FOLDS_OFF`, `IDN_MLP_EPOCH_DEV_OFF`, `IDN_MOE_GROUPED_GEMM_OFF`, `IDN_MOE_STABLE_PACK`, `IDN_SEQ_GEMM_MICRO`, `IDN_SEQ_GEMM_MICRO_OFF`, `IDN_SEQ_GEMM_TILED_OFF`, `IDN_SEQ_GEMM_TILED_V2_OFF`, `IDN_SEQ_GEMM_TILED_V3_OFF`, `IDN_SEQ_LN_LEAF`, `IDN_SEQ_SOFTMAX_ONE_EXP_OFF`, `LN_FAST_BLOCK_FOLD`, `LN_FOLD_BLOCK_OFF`, `MLP_LOSSBLOCK_OFF`, `MLP_LOSSBLOCK_ROWS_OFF`, `MOE_DEVGROUP_OFF`, `MOE_FAST_MMA_OFF`, `MOE_REGTILE_OFF`, `NN42_INPUT_BIAS_FUSED`, `NN44_EXPERT_BISECT`, `SEQ_FAST_FMA_OFF`, `SEQ_FAST_VAR_ONECOPY_OFF`, `SEQ_GARCH_GRID_OFF`, `SEQ_GARCH_REG_OFF`, `SEQ_REC_STAGE64`, `TSA2_STL_OFF`, `TSA2_VAR_OFF`

### expdel2-30-neural (37 defines, 10 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN26_LOSS_BLOCK128` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LOSS_BLOCK512` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LOSS_FUSED` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_OPT_BLOCK128` | DELETE-ARM | a | 1+22 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_OPT_BLOCK512` | DELETE-ARM | a | 1+22 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_OPT_CLIP_FUSE` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_OPT_MULTITENSOR` | DELETE-ARM | a | 1+4 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_LOSS_FUSED` | DELETE-ARM | f | 1+1 | 2 | EXP:630 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_OPT_CLIP_FUSE` | DELETE-ARM | f | 1+4 | 5 | EXP:647 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_OPT_MULTITENSOR` | DELETE-ARM | f | 1+4 | 16 | EXP:649 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AUTOGRAD_OPTIM` | DELETE-ARM | a | 2+12 | 30 | EXP:2380 lane/r2-restore-unmeasured-defaults @ e048e42e81 (hand OP_OPT update restored as -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_ALL` | DELETE-ARM | a | 1+5 | 6 | EXP:660 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_ENCODE_DEVICE` | DELETE-ARM | a | 1+1 | 2 | EXP:662 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_GROUP_FILTER` | DELETE-ARM | a | 1+3 | 10 | EXP:663 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_LIVEBUF` | DELETE-ARM | a | 1+6 | 34 | EXP:664 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_MERGE_BATCH` | DELETE-ARM | a | 1+4 | 30 | EXP:665 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BPE_TRAIN_DEVICE` | DELETE-ARM | a | 1+1 | 2 | EXP:666 OPEN -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ETS_TEAM_OFF` | DELETE-OLD-PATH | d | 1+4 | 5 | EXP:535 KEPT aec9f8c33 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_AF_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+9 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ETS_TEAM_OFF` | DELETE-OLD-PATH | e | 1+11 | 17 | EXP:2087 KEEP (IDENTICAL default; `_OFF` restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_NM_SPEC_OFF` | DELETE-OLD-PATH | e | 2+6 | 8 | EXP:2086 KEEP (IDENTICAL default; `_OFF` restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF` | DELETE-OLD-PATH | b | 1+3 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PROPHET_COOP_OFF` | DELETE-OLD-PATH | b | 1+4 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_WGRAD` | DELETE-ARM | a | 5+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF` | DELETE-OLD-PATH | b | 4+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_OPT_FAST_STREAM_OFF` | DELETE-OLD-PATH | e | 1+4 | 12 | EXP:1204 DROPPED-slower (alone) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_OPT_PIPE_DOWN_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:589 KEPT 5990c5946 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_OPT_RAW_UP_OFF` | DELETE-OLD-PATH | e | 1+3 | 11 | EXP:1204 DROPPED-slower (alone) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_OPT_ZERO_OPEN_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:590 KEPT 5990c5946 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PROPHET_COOP_OFF` | DELETE-OLD-PATH | d | 1+4 | 7 | EXP:537 KEPT 8c2b7de51 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SEQ_RNN_SCAN` | DELETE-ARM | e | 1+2 | 4 | EXP:2166 TIE (chain stays the default for G >= 2) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_SEQ_RNN_SCAN_APPLE` | DELETE-ARM | e | 1+4 | 5 | EXP:2165 KEPT (scan default for G = 1) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_SEQ_RNN_SCAN_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:2166 TIE (chain stays the default for G >= 2) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SEQ_THETA_REG_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:548 KEPT d1fa9223b -- promoted on one vendor in the record; tombstone notes the owed other vendor |

Files: `sequence/opt_resident.mojo`, `sequence/prophet.mojo`, `sequence/prophet_coop.mojo`, `sequence/recurrent.mojo`, `sequence/recurrent_scan.mojo`, `sequence/theta.mojo`, `sequence/theta_spec.mojo`, `sequence/xseq_glue_device.mojo`, `tokenizer/fast/bpe_device.mojo`, `training/afn_optim.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN26_LOSS_BLOCK128`, `AFN26_LOSS_BLOCK512`, `AFN26_LOSS_FUSED`, `AFN26_OPT_BLOCK128`, `AFN26_OPT_BLOCK512`, `AFN26_OPT_CLIP_FUSE`, `AFN26_OPT_MULTITENSOR`, `AFN_LOSS_FUSED`, `AFN_OPT_CLIP_FUSE`, `AFN_OPT_MULTITENSOR`, `BPE_ALL`, `BPE_ENCODE_DEVICE`, `BPE_GROUP_FILTER`, `BPE_LIVEBUF`, `BPE_MERGE_BATCH`, `BPE_TRAIN_DEVICE`, `IDN_AF_RESIDENT_OFF`, `IDN_OPT_PIPE_DOWN_OFF`, `IDN_OPT_ZERO_OPEN_OFF`, `IDN_PROPHET_COOP_OFF`, `OPT_FAST_STREAM_OFF`, `OPT_PIPE_DOWN_OFF`, `OPT_RAW_UP_OFF`, `OPT_ZERO_OPEN_OFF`, `PROPHET_COOP_OFF`, `SEQ_RNN_SCAN`, `SEQ_RNN_SCAN_APPLE`, `SEQ_RNN_SCAN_OFF`, `SEQ_THETA_REG_OFF`

### expdel2-31-neural (45 defines, 19 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN26_LM_BWD_FUSE` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_CE_BLOCK128` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_CE_BLOCK512` | DELETE-ARM | a | 1+8 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_HEAD_FUSE` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_NOSYNC` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_LM_PARAM_VIEWS` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_LM_ALL` | DELETE-ARM | f | 1+4 | 5 | EXP:621 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_BWD_FUSE` | DELETE-ARM | f | 1+4 | 5 | EXP:623 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF` | DELETE-ARM | f | 1+4 | 5 | EXP:625 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_LM_HEAD_FUSE` | DELETE-ARM | e | 1+1 | 2 | EXP:2340 DELETED (their step deleted; default off) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_AFN_LM_NOSYNC` | DELETE-ARM | e | 1+1 | 2 | EXP:2340 DELETED (their step deleted; default off) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_AFN_LM_PARAM_VIEWS` | DELETE-ARM | f | 1+1 | 2 | EXP:628 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_ATTN_EXACT_TAIL_GUARD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_KV_CORNER_GUARD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_LEGACY_CORNER` | DELETE-OLD-PATH | b | 1+10 | 21 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ATTN_REPAIR_MASKED_TAIL` | DELETE-ARM | a | 1+10 | 63 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_CE_UNALIASED` | DELETE-ARM | a | 1+3 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_COPY_EMB_HEAD` | DELETE-ARM | a | 1+4 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_GENERIC_OPS` | DELETE-ARM | a | 2+5 | 7 | EXP:2375 PENDING (A = restored ops are the default) -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_BYTE_LM_PROGRAM_STEP` | DELETE-ARM | a | 1+12 | 49 | EXP:2342 RESTORED hand step as default; A/B owed -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_RELEASE_EAGER` | DELETE-ARM | a | 1+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_RETAIN_EAGER` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_STICKY_EAGER` | DELETE-ARM | a | 1+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CE_DENOM_ROWFOLD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CE_PIPE_UP_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2` | DELETE-ARM | e | 2+17 | 26 | src# # descriptor op 16), the measured winner: IDN_SAMBA_RESIDENT_STEP PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_MAXIMIZE_DEV_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL` | DELETE-ARM | a | 2+12 | 15 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_NN16` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NORM` | DELETE-ARM | a | 10+4 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NORM_FIXED_LANES` | DELETE-ARM | a | 1+16 | 49 | EXP:2246 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF` | DELETE-OLD-PATH | e | 1+4 | 7 | EXP:1878 DELETED (noise) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TRAIN_SCRATCH` | DELETE-ARM | a | 2+6 | 22 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN53_HEAD_CHUNK2048` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN53_HEAD_CHUNK512` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_OPT_TRUST_INPUTS` | DELETE-ARM | a | 3+0 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRAIN_LIVE_STATUS` | DELETE-ARM | a | 2+0 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRAIN_OPT_FAST_PIPE_DOWN_OFF` | DELETE-OLD-PATH | e | 1+4 | 8 | EXP:2429 DELETED 2026-11-15 (NEUTRAL) -- record read: the default is the recorded winner or the kept path |

Files: `training/byte_lm.mojo`, `training/byte_lm_afn.mojo`, `training/byte_lm_config.mojo`, `training/byte_lm_host.mojo`, `training/byte_lm_host_execution.mojo`, `training/byte_lm_host_kernels.mojo`, `training/byte_lm_host_session.mojo`, `training/byte_lm_layer_pool.mojo`, `training/byte_lm_logits.mojo`, `training/byte_lm_program.mojo`, `training/checks/loss.mojo`, `training/checks/loss_oracle.mojo`, `training/checks/optimizer.mojo`, `training/checks/optimizer_oracle.mojo`, `training/chunked_lm_head_profile.mojo`, `training/clip_multi_gpu.mojo`, `training/dev_tensors.mojo`, `training/estimator.mojo`, `training/host/native_primitive_dispatch.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN26_LM_BWD_FUSE`, `AFN26_LM_CE_BLOCK128`, `AFN26_LM_CE_BLOCK512`, `AFN26_LM_HEAD_FUSE`, `AFN26_LM_NOSYNC`, `AFN26_LM_PARAM_VIEWS`, `AFN_LM_ALL`, `AFN_LM_BWD_FUSE`, `AFN_LM_BWD_NOSYNC_OFF`, `AFN_LM_HEAD_FUSE`, `AFN_LM_NOSYNC`, `AFN_LM_PARAM_VIEWS`, `BYTE_LM_CE_UNALIASED`, `BYTE_LM_COPY_EMB_HEAD`, `BYTE_LM_HOST_LEGACY_GEMM`, `BYTE_LM_PROGRAM_STEP`, `BYTE_LM_RELEASE_EAGER`, `BYTE_LM_RETAIN_EAGER`, `BYTE_LM_STICKY_EAGER`, `BYTE_LM_VSUM_HD_GENERIC_OFF`, `IDN_CE_PIPE_UP_OFF`, `IDN_MAXIMIZE_DEV_OFF`, `IDN_NORM`, `IDN_OPT_PARAMS_RESIDENT_OFF`, `IDN_OPT_SCRATCH_POOL_OFF`, `IDN_TRAIN_DEV_TENSORS_OFF`, `NN53_HEAD_CHUNK2048`, `NN53_HEAD_CHUNK512`, `OPT_TRUST_INPUTS`, `TRAIN_OPT_FAST_PIPE_DOWN_OFF`

### expdel2-32-neural (45 defines, 27 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_EPI_ALL` | DELETE-ARM | f | 2+7 | 54 | EXP:606 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_ALL` | DELETE-ARM | f | 5+17 | 60 | EXP:652 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_ARENA` | DELETE-ARM | f | 1+6 | 21 | EXP:653 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT` | DELETE-ARM | f | 1+12 | 23 | EXP:654 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_EMB_ATOMIC` | DELETE-ARM | f | 1+5 | 25 | EXP:655 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_SAMBA_FUSE` | DELETE-ARM | e | 1+4 | 102 | EXP:2363 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_AFN_SAMBA_HEAD_GEMM` | DELETE-ARM | f | 1+4 | 50 | EXP:657 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_ATTN_ES_TQ16` | DELETE-ARM | a | 2+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_ES_TQ8` | DELETE-ARM | a | 2+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_HOST_RMS_FORCE_SERIAL` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ACT_RETAIN` | DELETE-ARM | e | 2+4 | 6 | EXP:2365 DELETED (superseded) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_ACT_RETAIN_OFF` | DELETE-OLD-PATH | e | 2+4 | 6 | EXP:2365 DELETED (superseded) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ATTN_STASH` | DELETE-ARM | e | 5+4 | 9 | EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_CE_DENOM_ROWFOLD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CE_GRAD_FUSED` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CE_TOKEN_FOLD` | DELETE-ARM | e | 1+2 | 3 | EXP:2009 DROPPED (noise), code deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2` | DELETE-ARM | e | 2+17 | 26 | src# # descriptor op 16), the measured winner: IDN_SAMBA_RESIDENT_STEP PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_LM_VIEWS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_M2_GRAD_FOLD` | DELETE-ARM | a | 3+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NORM` | DELETE-ARM | a | 10+4 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NORM_FIXED_LANES` | DELETE-ARM | a | 1+16 | 49 | EXP:2246 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OPT_GATE_SCAN_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ROPE` | DELETE-ARM | a | 2+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SAMBA_ATTN_RECOMPUTE_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF` | DELETE-OLD-PATH | b | 1+5 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SEQ_WGRAD` | DELETE-ARM | a | 5+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF` | DELETE-OLD-PATH | b | 4+6 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TRAIN_SCRATCH` | DELETE-ARM | a | 2+6 | 22 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_TRAIN_SWIGLU` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI01_TRAINING_WORKSPACE` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN21_SCALE_MASK` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN29_RING_PAIR` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN58_ACCUMULATE_STATUS` | DELETE-ARM | a | 2+9 | 42 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN59_CONTROL` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN59_DROPOUT_RESIDUAL` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN63_CANONICAL_SHARD_MERGE` | DELETE-ARM | a | 1+5 | 45 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_OPT_TRUST_INPUTS` | DELETE-ARM | a | 3+0 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SAMBA_PROGRAM_FORWARD` | DELETE-ARM | a | 1+1 | 2 | EXP:2399 PENDING -- no A/B recorded for the opt-in arm |

Files: `training/host/samba_ops_oracle.mojo`, `training/native_primitive_dispatch.mojo`, `training/native_resident_dispatch.mojo`, `training/neural_ab_lifetime.mojo`, `training/neural_ab_pointwise.mojo`, `training/neural_ab_profile_contract.mojo`, `training/neural_ab_profiles.mojo`, `training/neural_ab_shards.mojo`, `training/neural_arithmetic_profile.mojo`, `training/neural_gemm_workspace.mojo`, `training/neural_identical_experiments.mojo`, `training/opt_gate.mojo`, `training/optimizer_kernels_gpu.mojo`, `training/optimizer_public.mojo`, `training/residual_dropout_contract.mojo`, `training/samba_afn.mojo`, `training/samba_ops.mojo`, `training/samba_resident.mojo`, `transformer/checks/attention_v1_memory_profile_check.mojo`, `transformer/checks/attention_zdot_routing_check.mojo`, `transformer/checks/transformer_backward_oracle.mojo`, `transformer/checks/transformer_oracle.mojo`, `transformer/experiments/attention_schedules.mojo`, `transformer/experiments/checkpoint_contract.mojo`, `transformer/experiments/norm_fixed_lanes.mojo`, `transformer/experiments/norm_profile_contract.mojo`, `transformer/experiments/profile.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN_SAMBA_ARENA`, `AFN_SAMBA_DEVICE_ADMIT`, `AFN_SAMBA_EMB_ATOMIC`, `AFN_SAMBA_FUSE`, `AFN_SAMBA_HEAD_GEMM`, `ATTN_ES_TQ16`, `ATTN_ES_TQ8`, `HOST_RMS_FORCE_SERIAL`, `IDN_CE_DENOM_ROWFOLD`, `IDN_CE_GRAD_FUSED`, `IDN_CE_TOKEN_FOLD`, `IDN_CHUNKED_LM_HEAD_V2`, `IDN_LM_VIEWS`, `IDN_NORM_FIXED_LANES`, `IDN_OPT_CLIP_BATCHED_OFF`, `IDN_OPT_GATE_SCAN_OFF`, `IDN_OPT_SGD_ONE_LAUNCH_OFF`, `IDN_ROPE`, `IDN_SAMBA_ATTN_RECOMPUTE_OFF`, `IDN_SAMBA_DEV_REFUSE_OFF`, `IDN_TRAIN_NO_DECODE_CACHE`, `IDN_TRAIN_SCRATCH`, `IDN_TRAIN_SWIGLU`, `NI01_TRAINING_WORKSPACE`, `NN21_SCALE_MASK`, `NN29_RING_PAIR`, `NN58_ACCUMULATE_STATUS`, `NN59_CONTROL`, `NN59_DROPOUT_RESIDUAL`, `NN63_CANONICAL_SHARD_MERGE`, `SAMBA_PROGRAM_FORWARD`

### expdel2-33-neural (33 defines, 6 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN26_ATTN_FLASH_BK16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_ATTN_FLASH_TQ16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_ATTN_NORM_TPB128` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_ATTN_NORM_TPB512` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_ATTN_PROJ_BM32` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_ATTN_PROJ_KB16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_CNN_CHANNELS16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_CNN_COLS_ONCE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_CNN_K16` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN26_CNN_ROWS32` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_ATTN_ALL` | DELETE-ARM | f | 1+7 | 8 | EXP:595 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_AUDIT` | DELETE-ARM | a | 1+2 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFN_ATTN_FLASH` | DELETE-ARM | f | 1+0 | 1 | EXP:597 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_FUSE_MLP` | DELETE-ARM | f | 1+2 | 3 | EXP:598 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_FUSE_PRE` | DELETE-ARM | f | 1+2 | 3 | EXP:599 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_GQA_TILE` | DELETE-ARM | f | 1+0 | 1 | EXP:600 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_NORM_SG` | DELETE-ARM | f | 1+2 | 3 | EXP:601 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_ATTN_ROPE_CACHE` | DELETE-ARM | f | 1+2 | 3 | EXP:602 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_CNN_DIRECT` | DELETE-ARM | f | 1+8 | 45 | EXP:603 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF` | DELETE-OLD-PATH | b | 1+2 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ATTN_CACHE_WAIT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_BN_FOLD_BLOCK_OFF` | DELETE-OLD-PATH | b | 2+4 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FUSE_RESIDUAL1_NORM2` | DELETE-ARM | a | 2+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ATTN_FOLD_V2` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF` | DELETE-OLD-PATH | b | 1+2 | 26 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LM_RESIDENT_GENERATE` | DELETE-ARM | a | 2+3 | 6 | EXP:2262 PENDING -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI07_GROUPED_PROJECTIONS` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER` | DELETE-OLD-PATH | b | 2+0 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |

Files: `transformer/impl/llama/afn_apple_fast.mojo`, `transformer/impl/llama/attn_fold_v2.mojo`, `transformer/impl/llama/modeling_llama.mojo`, `transformer/transformer_binding_impl.mojo`, `x_cnn/afn_direct.mojo`, `x_cnn/checks/oracle.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFN26_ATTN_FLASH_BK16`, `AFN26_ATTN_FLASH_TQ16`, `AFN26_ATTN_NORM_TPB128`, `AFN26_ATTN_NORM_TPB512`, `AFN26_ATTN_PROJ_BM32`, `AFN26_ATTN_PROJ_KB16`, `AFN26_CNN_CHANNELS16`, `AFN26_CNN_COLS_ONCE`, `AFN26_CNN_K16`, `AFN26_CNN_ROWS32`, `AFN_ATTN_ALL`, `AFN_ATTN_AUDIT`, `AFN_ATTN_FLASH`, `AFN_ATTN_FUSE_MLP`, `AFN_ATTN_FUSE_PRE`, `AFN_ATTN_GQA_TILE`, `AFN_ATTN_NORM_SG`, `AFN_ATTN_ROPE_CACHE`, `AFN_CNN_DIRECT`, `AFN_FLASH_HD_GENERIC_OFF`, `ATTN_CACHE_WAIT`, `BN_FOLD_BLOCK_OFF`, `DISABLE_RESIDUAL2_NEXT_NORM`, `FUSE_RESIDUAL1_NORM2`, `IDN_ATTN_CACHE_NOWAIT_OFF`, `IDN_ATTN_FOLD_V2`, `IDN_LLAMA_REFUSE_BATCH_OFF`, `TRANSFORMER_LEGACY_CALLER_TRANSFER`, `TRANSFORMER_LEGACY_WEIGHT_COPY`

### expdel2-34-neural (43 defines, 1 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_ATTN_AMMA_ADMIT` | DELETE-ARM | a | 1+10 | 41 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_APPLE_ERECOMP` | DELETE-ARM | a | 1+6 | 51 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_ATTN_APPLE_ESTASH_FORCE_DENY` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_ATTN_ARM_TRIAL` | DELETE-ARM | a | 1+27 | 511 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DKDV_AMMA` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DKDV_BJ16` | DELETE-ARM | a | 2+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DKDV_BJ32` | DELETE-ARM | a | 2+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DKDV_MFMA` | DELETE-ARM | a | 1+2 | 3 | grid neural-small.json removed -- expdel-04 note: not gridded on AMD (kernel aborts rc 134); never measured |
| `MOJOLEARN_ATTN_DKDV_NO_T4` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DKDV_T4` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DQ_MFMA` | DELETE-ARM | a | 1+2 | 3 | grid neural-small.json removed -- expdel-04 note: not gridded on AMD (kernel aborts rc 134); never measured |
| `MOJOLEARN_ATTN_DQ_TQ16` | DELETE-ARM | a | 2+7 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DQ_TQ32` | DELETE-ARM | a | 2+7 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_DQ_TQ64` | DELETE-ARM | a | 2+7 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_ES_TQ16` | DELETE-ARM | a | 2+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_ES_TQ8` | DELETE-ARM | a | 2+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_EXACT_TAIL_GUARD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_FWD_AMMA_P1_BARRIER` | DELETE-ARM | a | 2+0 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_FWD_AMMA_P2_BARRIER` | DELETE-ARM | a | 1+3 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_FWD_TQ16` | DELETE-ARM | a | 2+21 | 23 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_FWD_TQ_RULE` | DELETE-ARM | a | 1+4 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_KV_CORNER_GUARD` | DELETE-ARM | a | 1+5 | 14 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_LEGACY_CORNER` | DELETE-OLD-PATH | b | 1+10 | 21 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ATTN_NO_APPLE_MMA` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_ATTN_NO_BWD_CORNER` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_NO_MODE_FLUSH` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_NO_SCRATCH_CACHE` | DELETE-ARM | a | 1+6 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_REPAIR_MASKED_TAIL` | DELETE-ARM | a | 1+10 | 63 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_STASH_ZDOT_AMMA` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_STICKY` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_TILE_RULE_OFF` | DELETE-OLD-PATH | b | 2+8 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ATTN_ZDOT_AMMA_FOLD_BARRIER` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ATTN_ZDOT_STASH_FOLD_BARRIER` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GAPLM_ZDOT_KREG` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF` | DELETE-OLD-PATH | b | 1+6 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ATTN_HEAD_SHARE` | DELETE-ARM | e | 2+2 | 5 | EXP:1483 DROP -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF` | DELETE-OLD-PATH | b | 1+9 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF` | DELETE-OLD-PATH | b | 1+4 | 31 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ATTN_STASH` | DELETE-ARM | e | 5+4 | 9 | EXP:1757 PROMOTED for Samba only (IDENTICAL default on the Samba train step); LM / transf -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_ATTN_TILE_ORDER` | DELETE-ARM | a | 1+5 | 6 | grid neural-small.json removed -- expdel-04 note: grid entry not measured |
| `MOJOLEARN_NN18_DENSE_TILE_CONTROL` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |

Files: `transformer/impl/llama/fused_attention.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `ATTN_AMMA_ADMIT`, `ATTN_APPLE_ERECOMP`, `ATTN_APPLE_ESTASH_FORCE_DENY`, `ATTN_ARM_TRIAL`, `ATTN_DKDV_AMMA`, `ATTN_DKDV_BJ16`, `ATTN_DKDV_BJ32`, `ATTN_DKDV_MFMA`, `ATTN_DKDV_NO_T4`, `ATTN_DKDV_T4`, `ATTN_DQ_MFMA`, `ATTN_DQ_TQ16`, `ATTN_DQ_TQ32`, `ATTN_DQ_TQ64`, `ATTN_EXACT_TAIL_GUARD`, `ATTN_FWD_AMMA_P1_BARRIER`, `ATTN_FWD_AMMA_P2_BARRIER`, `ATTN_FWD_TQ16`, `ATTN_FWD_TQ_RULE`, `ATTN_KV_CORNER_GUARD`, `ATTN_LEGACY_CORNER`, `ATTN_NO_APPLE_MMA`, `ATTN_NO_BWD_CORNER`, `ATTN_NO_MODE_FLUSH`, `ATTN_NO_SCRATCH_CACHE`, `ATTN_REPAIR_MASKED_TAIL`, `ATTN_STASH_ZDOT_AMMA`, `ATTN_STICKY`, `ATTN_TILE_RULE_OFF`, `ATTN_ZDOT_AMMA_FOLD_BARRIER`, `ATTN_ZDOT_STASH_FOLD_BARRIER`, `GAPLM_ZDOT_KREG`, `IDN_ATTN_BWD_SCAN_REUSE_OFF`, `IDN_ATTN_ONE_FLAG_WAIT_OFF`, `IDN_ATTN_SCAN_CACHE_OFF`, `IDN_ATTN_SCRATCH_CACHE_OFF`, `IDN_ATTN_TILE_ORDER`, `NN18_DENSE_TILE_CONTROL`

Lane result (expdel2-34-neural, 2026-11-15): 37 of the 38 owned defines deleted with their records (refusals in core/experiment_guards_neural.mojo; grid controls attn_tile_order, attn_fwd_tq_rule, attn_tile_rule_off moved to removed; the DQ / DKDV MFMA removed entries annotated). 5 non-owner defines folded in this file (`ATTN_ES_TQ16`, `ATTN_ES_TQ8`, `IDN_ALL_OFF`, `IDN_ATTN_HEAD_SHARE`, `IDN_ATTN_STASH`). KEEP: `ATTN_ARM_TRIAL`, the only switch that compiles the attention sabotage instantiations the arms check and the step-price bench exercise. Kernels kept although no launcher runs them: `fused_bwd_dq_mfma_kernel`, `fused_bwd_dkdv_mfma_kernel` (imported by tools/amd_codegen/probe_attn_dq.mojo). No bit change.

### expdel2-35-neural (41 defines, 5 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFN_CNN_DIRECT` | DELETE-ARM | f | 1+8 | 45 | EXP:603 MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) -- neural MERGED-UNMEASURED opt-in (Oct 3), never measured |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ADAPT_M_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_CNN_CONV_RELU` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2084 KEEP (IDENTICAL default; `_OFF` restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF` | DELETE-OLD-PATH | b | 1+5 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GRAPH_M_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GROUP_M_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_LAYER_DEV_IO_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE` | DELETE-ARM | e | 1+4 | 9 | EXP:1767 DELETED (loser; AMD flat, NV 1.9x) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_PAD_M_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_PIN_WEIGHTS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XENT_DEV_FOLD_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_NI11_DIRECT_CONV_TAPS64` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI12_IMPLICIT_CONV_OFF` | DELETE-OLD-PATH | d | 1+6 | 7 | grid neural-seq-train-dedupe.json promoted 2026-10-10 (lane/postmerge-act-6; post-merge neural -- expdel-04 note: MEASURE OWED after g50ebe26a5; default stays, owed vendor recorded in the tombstone |
| `MOJOLEARN_NI14_COL2IM` | DELETE-ARM | a | 1+3 | 4 | EXP:1615 n/a -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI15_BIAS_NO_ONES` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI55_GRAPH_FEATURE4` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI57_GRAPH_TREE64` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI58_SAGE_FEATURE4` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI59_DROPOUT_CHANNEL` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI60_DROPOUT_APPLY4` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN14_BOUNDED_IM2COL` | DELETE-ARM | e | 1+19 | 77 | EXP:1943 PROMOTED (IDENTICAL default) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_NN47_APPLY_RUNNING` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN48_CSR_TILES` | DELETE-ARM | a | 1+2 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_BN_BLOCK` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_DIRECT_CONV` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_FAST_TUNE` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_XCNN_NO_IM2COL_TAPS` | DELETE-ARM | a | 1+1 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_MMA_SPLIT` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_NT_TUNE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_RES_POOL` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_NO_TILED_LAYOUT` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_XCNN_ONES_CACHE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF` | DELETE-OLD-PATH | b | 1+3 | 5 | EXP:1792 owed (nv + amd RACE) -- no A/B recorded for the old path |
| `MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1791 owed (nv + amd RACE batchnorm2d,dropout2d) -- no A/B recorded for the old path |
| `MOJOLEARN_XCNN_TILED_ROWS` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |

Files: `x_cnn/device.mojo`, `x_cnn/host/gemm_host.mojo`, `x_cnn/host/ops_host.mojo`, `x_cnn/neural_aux_contract.mojo`, `x_cnn/neural_col2im.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `IDN_ADAPT_M_OFF`, `IDN_CNN_RESIDENT_GRADS_OFF`, `IDN_DIRECT_CONV_NVAMD_OFF`, `IDN_GRAPH_M_OFF`, `IDN_GROUP_M_OFF`, `IDN_LAYER_DEV_IO_OFF`, `IDN_PAD_M_OFF`, `IDN_PIN_WEIGHTS_OFF`, `IDN_XCNN_BIAS_FOLD_OFF`, `NI11_DIRECT_CONV_TAPS64`, `NI12_IMPLICIT_CONV_OFF`, `NI14_COL2IM`, `NI15_BIAS_NO_ONES`, `NI55_GRAPH_FEATURE4`, `NI57_GRAPH_TREE64`, `NI58_SAGE_FEATURE4`, `NI59_DROPOUT_CHANNEL`, `NI60_DROPOUT_APPLY4`, `NN48_CSR_TILES`, `XCNN_NO_BN_BLOCK`, `XCNN_NO_DIRECT_CONV`, `XCNN_NO_FAST_TUNE`, `XCNN_NO_IM2COL_TAPS`, `XCNN_NO_MMA_SPLIT`, `XCNN_NO_NT_TUNE`, `XCNN_NO_RES_POOL`, `XCNN_NO_TILED_LAYOUT`, `XCNN_ONES_CACHE_OFF`, `XCNN_RES_FREE_ASYNC_OFF`, `XCNN_RES_NEW_NOZERO_OFF`, `XCNN_TILED_ROWS`

Lane result (expdel2-35-neural, 2026-11-15): 30 owned deleted with records; 5 non-owner folds (IDN_ALL_OFF, AFN_CNN_DIRECT, NN14_BOUNDED_IM2COL, IDN_CNN_CONV_RELU, NN47_APPLY_RUNNING); no edit needed for IDN_GAP_BLOCK_FOLD_OFF, IDN_XENT_DEV_FOLD_OFF, IDN_NEURAL_CHAINS, IDN_NEURAL_GEMM_EPILOGUE. KEPT: `NI12_IMPLICIT_CONV_OFF` (its _OFF arm is a measured cnn-clf winner, grid g50ebe26a5 avg 0.90x) and `COLUMN_CPU`. The tiled col2im kernel (x_cnn/neural_col2im.mojo) stays while autograd/kernels/cnn.mojo imports it.

### expdel2-36-neural-series-linear (38 defines, 13 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_P07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P08` | DELETE-EXPERIMENT | g | 1+5 | 21 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P11` | DELETE-EXPERIMENT | g | 1+3 | 16 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_BN_FOLD_BLOCK_OFF` | DELETE-OLD-PATH | b | 2+4 | 9 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_C60_LAG4` | DELETE-ARM | a | 1+5 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DROPOUT2D_CH_MASK_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF` | DELETE-OLD-PATH | b | 1+3 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF` | DELETE-OLD-PATH | b | 1+2 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_CNN_CONV_RELU` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF` | DELETE-OLD-PATH | b | 1+5 | 42 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_LEAF` | DELETE-ARM | e | 1+23 | 24 | EXP:1766 PROMOTED for the MLP step only (IDENTICAL default on mlp-train-step); every othe -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF` | DELETE-OLD-PATH | b | 1+3 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KPSS_SCAN_OFF` | DELETE-OLD-PATH | b | 1+2 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_NEURAL_CHAINS` | DELETE-ARM | a | 2+25 | 27 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF` | DELETE-OLD-PATH | b | 1+2 | 23 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RAGGED_RADIX_SELECT` | DELETE-ARM | a | 1+3 | 35 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_SELECT_D_OFF` | DELETE-OLD-PATH | b | 1+2 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF` | DELETE-OLD-PATH | b | 1+2 | 22 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XENT_DEV_FOLD_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_XENT_FOLD_BLOCK_256` | DELETE-ARM | a | 1+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI10_BUDGETED_CNN_TAPE` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI17_BALANCED_NORM` | DELETE-ARM | a | 1+4 | 30 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NI57_GRAPH_TREE64` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN14_BOUNDED_IM2COL` | DELETE-ARM | e | 1+19 | 77 | EXP:1943 PROMOTED (IDENTICAL default) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_NN46_GATHER_BOUNDS` | DELETE-ARM | a | 1+4 | 24 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_NN47_APPLY_RUNNING` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PREP3_MAXABS_OFF` | DELETE-OLD-PATH | d | 1+9 | 11 | EXP:383 KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PREP3_MAXABS_POOL_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:920 DEFAULT; M2 default/_OFF both rc=0, merged. Only MaxAbs paths extracted; resampl -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_PT_SCORE_STABLE_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1890 KEPT (default path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SELECT_D_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:552 KEPT 6c0ba4379 -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SELECT_FCLS_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:367 KEPT e2bfb8422 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_SELECT_FREG_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:368 KEPT 4198d5a9c -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TSA_FAST_KPSS_PACK_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:1848 DELETED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_XCNN_NO_POOL_BWD_TILE` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |

Files: `tsa/impl/auto_arima.mojo`, `tsa/impl/select_d_fast.mojo`, `tsa/impl/timeSeries/kpss_fused.mojo`, `tsa/impl/timeSeries/stationarity.mojo`, `x_cnn/ops.mojo`, `x_metrics/host/program.mojo`, `x_metrics/par.mojo`, `x_metrics/reg_epi.mojo`, `x_metrics/seams/x_metrics_check.mojo`, `x_prep/prep3.mojo`, `x_prep/pt_center.mojo`, `x_prep/ragged_select.mojo`, `x_prep/select_fast.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_P07`, `AFCL_P08`, `AFCL_P11`, `DROPOUT2D_CH_MASK_OFF`, `IDN_ADAPT_BWD_BOUNDED_OFF`, `IDN_AVGPOOL_BWD_TILE_OFF`, `IDN_CNN_CONV_RELU`, `IDN_GAP_BLOCK_FOLD_OFF`, `IDN_GCN_LOOPS_DEV_OFF`, `IDN_KPSS_ONE_WAIT_OFF`, `IDN_KPSS_SCAN_OFF`, `IDN_PAD_BWD_BOUNDED_OFF`, `IDN_RAGGED_RADIX_SELECT`, `IDN_SELECT_D_OFF`, `IDN_XCNN_SOFTMAX_ONE_EXP_OFF`, `IDN_XENT_DEV_FOLD_OFF`, `IDN_XENT_FOLD_BLOCK_256`, `NI10_BUDGETED_CNN_TAPE`, `NI10_RECOMPUTE_CNN_TAPE`, `NI17_BALANCED_NORM`, `NN14_BOUNDED_IM2COL`, `NN46_GATHER_BOUNDS`, `NN47_APPLY_RUNNING`, `PREP3_MAXABS_OFF`, `PREP3_MAXABS_POOL_OFF`, `PT_SCORE_STABLE_OFF`, `SELECT_D_OFF`, `SELECT_FCLS_OFF`, `SELECT_FREG_OFF`, `TSA_FAST_KPSS_PACK_OFF`, `XCNN_NO_POOL_BWD_TILE`

### expdel2-37-series (44 defines, 25 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_P10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P12` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_P14` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ARIMA_ASYNC_ALL` | DELETE-ARM | a | 1+2 | 3 | EXP:2379 lane/r2-restore-unmeasured-defaults @ 1155f6ab56 (arima/impl/batched_fit.mojo `_ -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_ARIMA_FAST_CONST_BOTH` | DELETE-ARM | a | 1+3 | 5 | EXP:555 READY-AB, merged unmeasured -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_ARIMA_FAST_KPSS_D` | DELETE-ARM | a | 1+6 | 26 | EXP:557 READY-AB, merged unmeasured -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF` | DELETE-OLD-PATH | d | 1+3 | 5 | EXP:746 KEEP: default/OFF promotion 311d5233e; taxi RMSE 74.6591 remains worse than oppo -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ARIMA_ORDER_BATCH_OFF` | DELETE-OLD-PATH | d | 2+10 | 15 | EXP:850 KEEP, default merged on main dc2285bc0; manager default/OFF builds both rc0. Sou -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_C58_SERIES4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_C59_TRIAL_STATE` | DELETE-ARM | a | 1+4 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_CLASSICAL_C11_DRAW_GATHER` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_HW_EST_BLOCK64` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF` | DELETE-OLD-PATH | b | 2+5 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF` | DELETE-OLD-PATH | b | 1+3 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+5 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_LLONLY_OFF` | DELETE-OLD-PATH | b | 2+4 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF` | DELETE-OLD-PATH | b | 2+10 | 27 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+8 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_BOOT_SELECT_OFF` | DELETE-OLD-PATH | b | 1+2 | 17 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GEMM_LEAF` | DELETE-ARM | e | 1+23 | 24 | EXP:1766 PROMOTED for the MLP step only (IDENTICAL default on mlp-train-step); every othe -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_IDN_KALMAN_RD_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF` | DELETE-OLD-PATH | b | 1+3 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_METRIC_EPI_OFF` | DELETE-OLD-PATH | b | 1+7 | 34 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_SIL_NO_WAIT_OFF` | DELETE-OLD-PATH | b | 1+3 | 18 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TRUST_DEV_OFF` | DELETE-OLD-PATH | b | 1+4 | 25 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_TRUST_NO_STAGE_OFF` | DELETE-OLD-PATH | b | 1+3 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_X0_PAR_LS` | DELETE-ARM | a | 1+2 | 29 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_X0_PAR_LS_MIN256` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KALMAN_FAST_RD_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_KALMAN_TIME_SCAN` | DELETE-ARM | a | 1+5 | 38 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_KALMAN_TIME_SCAN_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_METRICS_JOINT_COUNTS` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PERM_COUNT_RANK` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW` | DELETE-ARM | a | 1+0 | 1 | EXP:1404 READY-AB -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_RESAMPLE_FAST_GATHER_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1131 DROPPED-slower (reconciled 2026-10-05: matched timing r2-20261004: taxi 68.49 -> -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES` | DELETE-ARM | a | 1+3 | 4 | EXP:1404 READY-AB -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_X0_FAST_LS_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded; old path beside the Apple fast default |

Files: `arima/impl/batched_arima.mojo`, `arima/impl/batched_fit.mojo`, `arima/impl/batched_kalman.mojo`, `arima/impl/estimate_x0.mojo`, `arima/impl/fast_arima_quality.mojo`, `arima/impl/fast_eval_ws.mojo`, `arima/impl/fast_lbfgs_async.mojo`, `arima/impl/fast_order_state.mojo`, `arima/impl/idn_ls_math.mojo`, `bindings/_mojolearn_arima.mojo`, `bindings/_mojolearn_arima_host.mojo`, `gemm/contract.mojo`, `holtwinters/impl/internal/hw_estimate.mojo`, `holtwinters/impl/internal/hw_estimate_launch.mojo`, `holtwinters/impl/internal/hw_utils.mojo`, `metrics/host/metrics_oracle.mojo`, `metrics/impl/classification_joint.mojo`, `metrics/impl/stats/detail/adjusted_rand_index.mojo`, `metrics/impl/stats/detail/batched/silhouette_score.mojo`, `metrics/impl/stats/detail/sf_epilogue_core.mojo`, `metrics/impl/stats/detail/trustworthiness_score.mojo`, `resample/estimator.mojo`, `resample/gather_fast.mojo`, `resample/order_select.mojo`, `tsa/estimator.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_P10`, `AFCL_P12`, `AFCL_P14`, `ARIMA_ASYNC_ALL`, `ARIMA_FAST_BATCH_GRAD_OFF`, `ARIMA_FAST_CONST_BOTH`, `ARIMA_FAST_KPSS_D`, `ARIMA_FUSED_EVAL_TAIL_OFF`, `ARIMA_ID_BATCH_GRAD_OFF`, `ARIMA_ORDER_BATCH_OFF`, `HW_EST_BLOCK64`, `IDN_ALL_OFF`, `IDN_ARIMA_EVAL_WS_EXOG_OFF`, `IDN_ARIMA_EVAL_WS_OFF`, `IDN_ARIMA_FUSED_TAIL_OFF`, `IDN_ARIMA_LLONLY_OFF`, `IDN_ARIMA_ORDER_BATCH_OFF`, `IDN_BOOT_SELECT_OFF`, `IDN_GEMM_LEAF`, `IDN_KALMAN_RD_OFF`, `IDN_METRIC_EPI_OFF`, `IDN_SIL_NO_WAIT_OFF`, `IDN_TRUST_DEV_OFF`, `IDN_TRUST_NO_STAGE_OFF`, `IDN_X0_PAR_LS`, `IDN_X0_PAR_LS_MIN256`, `KALMAN_FAST_RD_OFF`, `KALMAN_TIME_SCAN`, `KALMAN_TIME_SCAN_OFF`, `METRICS_JOINT_COUNTS`, `PERM_COUNT_RANK`, `RESAMPLE_FAST_DEVICE_PERMUTE`, `RESAMPLE_FAST_GATHER_NARROW`, `RESAMPLE_FAST_GATHER_OFF`, `RESAMPLE_FAST_TILED_GATHER_OFF`, `RESAMPLE_FAST_WAIT_PAIR_OFF`, `RESAMPLE_GATHER_NARROW_BYTES`, `X0_FAST_LS_OFF`

### expdel2-38-trees (45 defines, 9 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_2010_ROWS_SORTED` | DELETE-ARM | a | 2+5 | 36 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2011_HIST_ITEMS1` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2012_SMEM_COPIES4` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2502_RETRY_PURE` | DELETE-ARM | a | 1+5 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFT_F01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F02` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F03` | DELETE-EXPERIMENT | g | 1+8 | 9 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F04` | DELETE-EXPERIMENT | g | 1+4 | 5 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F05` | DELETE-EXPERIMENT | g | 1+4 | 5 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_QUANT_DEVICE` | DELETE-EXPERIMENT | g | 1+8 | 18 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_RF_COLS40_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RF_DEVICE_LOOP` | DELETE-ARM | a | 1+1 | 2 | EXP:2402 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RF_HIST_ZERO_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_BINS_COLUMN_MAJOR` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_COLS10` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_FAST_HIST_ZERO_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded; old path beside the Apple fast default |
| `MOJOLEARN_RF_FAST_SAMPLE_PER_COLUMN` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_RF_FBS_WARP_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_HIST_COLUMNS2` | DELETE-ARM | a | 3+2 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_HIST_COLUMNS4` | DELETE-ARM | a | 3+3 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_HIST_COLUMNS4_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_HIST_COLUMNS8` | DELETE-ARM | e | 1+13 | 14 | EXP:120 KEPT 269ffa57a -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_RF_HIST_COLUMNS8_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:120 KEPT 269ffa57a -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_RF_HIST_SIMD_AGG_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_ROWS_SORTED_OFF` | DELETE-OLD-PATH | b | 1+5 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_SMALL_NODE_FUSED` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_SPLIT_CANDIDATES` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_SPLIT_CANDIDATES_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_TRIAL_COLS20` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_TRIAL_COLS40` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T05_ROW_VISITS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T08_BITS` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T08_LAYOUT` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T11_LEVELS` | DELETE-ARM | e | 2+3 | 5 | EXP:1582 DROP, defines deleted (no gain + duplicate) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T13_BYTES` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T14_EXACT` | DELETE-ARM | a | 1+16 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST` | DELETE-ARM | a | 2+5 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT` | DELETE-ARM | a | 4+0 | 17 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |

Files: `bindings/_mojolearn_gbdt.mojo`, `ensemble/bench/rf_bench.mojo`, `ensemble/checks/rf_column_tiles_fingerprint.mojo`, `ensemble/decisiontree/batched_levelalgo/builder.mojo`, `ensemble/decisiontree/batched_levelalgo/dataset.mojo`, `ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`, `ensemble/decisiontree/batched_levelalgo/objectives.mojo`, `ensemble/decisiontree/batched_levelalgo/quantiles.mojo`, `ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2011_HIST_ITEMS1`, `2012_SMEM_COPIES4`, `2502_RETRY_PURE`, `AFT_F01`, `AFT_F02`, `AFT_F03`, `AFT_F04`, `AFT_F05`, `IDN_RF_COLS40_OFF`, `IDN_RF_DEVICE_LOOP`, `IDN_RF_FUSED_PARTITION_OFF`, `IDN_RF_HIST_SIMD_AGG_OFF`, `IDN_RF_HIST_ZERO_OFF`, `IDN_RF_QBIN_DEVICE_OFF`, `IDN_RF_SAMPLE_PER_NODE_OFF`, `RF_BINS_COLUMN_MAJOR`, `RF_COLS10`, `RF_FAST_HIST_ZERO_OFF`, `RF_FAST_SAMPLE_PER_COLUMN`, `RF_FBS_WARP_OFF`, `RF_HIST_COLUMNS2`, `RF_HIST_COLUMNS4`, `RF_HIST_COLUMNS4_OFF`, `RF_HIST_COLUMNS8`, `RF_HIST_COLUMNS8_OFF`, `RF_HIST_SIMD_AGG_OFF`, `RF_SMALL_NODE_FUSED`, `RF_SPLIT_CANDIDATES`, `RF_SPLIT_CANDIDATES_OFF`, `RF_TRIAL_COLS20`, `RF_TRIAL_COLS40`, `TREE_EXACT_SIBLING_HIST`, `TREE_EXACT_SIBLING_HIST_AUDIT`, `TREE_EXACT_SIBLING_HIST_BYTES`

### expdel2-39-trees (32 defines, 6 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_2010_ROWS_SORTED` | DELETE-ARM | a | 2+5 | 36 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_DEVICE_BATCH_16384` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_DEVICE_BATCH_32768` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_DEVICE_BATCH_4096` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+2 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX` | DELETE-ARM | a | 2+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER` | DELETE-ARM | a | 1+6 | 89 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF` | DELETE-OLD-PATH | b | 1+6 | 15 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_RF_ROWS_SORTED_OFF` | DELETE-OLD-PATH | b | 1+5 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_C48` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_RF_SAMPLE` | DELETE-ARM | e | 2+8 | 10 | EXP:1610 ALWAYS_OFF_DELETE, define deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T02` | DELETE-ARM | e | 1+0 | 1 | EXP:1581 DROP, define deleted (neutral + duplicate) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T03` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T04` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T05` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T05_ROW_VISITS` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T06` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T07` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T08` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T08_BITS` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T08_LAYOUT` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T11_LEVELS` | DELETE-ARM | e | 2+3 | 5 | EXP:1582 DROP, defines deleted (no gain + duplicate) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T12` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T12_BYTES` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T13` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T13_BYTES` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T14` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T14_EXACT` | DELETE-ARM | a | 1+16 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T15` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST` | DELETE-ARM | a | 2+5 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT` | DELETE-ARM | a | 4+0 | 17 | no record -- no A/B recorded for the opt-in arm |

Files: `ensemble/decisiontree/batched_levelalgo/split.mojo`, `ensemble/host/rf_oracle.mojo`, `ensemble/randomforest.mojo`, `ensemble/tree_identical_ideas.mojo`, `experiments/performance_ideas/I18/forest_check.mojo`, `extratrees/estimator.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2010_ROWS_SORTED`, `ET_DEVICE_BATCH_16384`, `ET_DEVICE_BATCH_32768`, `ET_DEVICE_BATCH_4096`, `RF_EXP_NO_SPLIT_MUTEX`, `RF_FUSED_BOOTSTRAP_GATHER`, `RF_FUSED_BOOTSTRAP_GATHER_OFF`, `RF_ROWS_SORTED_OFF`, `TREES_C48`, `TREES_RF_SAMPLE`, `TREES_T02`, `TREES_T03`, `TREES_T04`, `TREES_T05`, `TREES_T05_ROW_VISITS`, `TREES_T06`, `TREES_T07`, `TREES_T08`, `TREES_T08_BITS`, `TREES_T08_LAYOUT`, `TREES_T11_LEVELS`, `TREES_T12`, `TREES_T12_BYTES`, `TREES_T13`, `TREES_T13_BYTES`, `TREES_T14`, `TREES_T14_EXACT`, `TREES_T15`

### expdel2-40-trees (40 defines, 1 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_T09` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F06` | DELETE-EXPERIMENT | g | 1+0 | 1 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F08` | DELETE-EXPERIMENT | g | 1+4 | 5 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_ET_BINNED_OFF` | DELETE-OLD-PATH | b | 1+5 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_DEVICE_LOOP_K1` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_DEVICE_LOOP_K2` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_DEVICE_LOOP_K8` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_MAX_ACC_16` | DELETE-ARM | a | 2+2 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_MAX_ACC_32` | DELETE-ARM | a | 1+2 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_MAX_ACC_4` | DELETE-ARM | a | 2+2 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_MAX_ACC_8` | DELETE-ARM | a | 2+2 | 10 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_PART_FLAGS_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_PART_ROWS` | DELETE-ARM | e | 1+11 | 12 | EXP:105 KEPT 269ffa57a -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_ET_PART_ROWS_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:105 KEPT 269ffa57a -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_ET_RANGE_TILED` | DELETE-ARM | a | 1+5 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_RM_LINE_HW_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_RM_NARROW` | DELETE-ARM | a | 1+3 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_RM_NARROW_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_ROW_MAJOR` | DELETE-ARM | a | 1+10 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SCALAR_STAGE_COMPARE` | DELETE-ARM | a | 3+0 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SCORE_TILED` | DELETE-ARM | a | 1+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_STAGE_FULL_CAPACITY` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_TILED_SEARCH_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_ET_TPB_1024` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_TPB_128` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_TPB_512` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_FOREST_DEVICE_FINITE_OFF` | DELETE-OLD-PATH | d | 1+12 | 23 | EXP:106 KEPT 69f7a41fd -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ET_DEVICE_LOOP` | DELETE-ARM | a | 1+1 | 2 | EXP:2401 RUN OWED -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ET_PART_ROWS_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ET_STAGE_LIVE_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ET_TILED_SEARCH_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_ET_BINNED` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_ET_RM` | DELETE-OLD-PATH | b | 1+2 | 10 | EXP:1330 OWED -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_T11_LEVELS` | DELETE-ARM | e | 2+3 | 5 | EXP:1582 DROP, defines deleted (no gain + duplicate) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T12_BYTES` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T13_BYTES` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T14_EXACT` | DELETE-ARM | a | 1+16 | 19 | no record -- no A/B recorded for the opt-in arm |

Files: `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_T09`, `AFT_F06`, `AFT_F07`, `AFT_F08`, `ET_BINNED_OFF`, `ET_DEVICE_LOOP_K1`, `ET_DEVICE_LOOP_K2`, `ET_DEVICE_LOOP_K8`, `ET_MAX_ACC_16`, `ET_MAX_ACC_32`, `ET_MAX_ACC_4`, `ET_MAX_ACC_8`, `ET_PART_FLAGS_OFF`, `ET_PART_ROWS`, `ET_PART_ROWS_OFF`, `ET_RANGE_TILED`, `ET_RM_LINE_HW_OFF`, `ET_RM_NARROW`, `ET_RM_NARROW_OFF`, `ET_ROW_MAJOR`, `ET_SCORE_TILED`, `ET_STAGE_FULL_CAPACITY`, `ET_TILED_SEARCH_IDENTICAL_OFF`, `ET_TILED_SEARCH_OFF`, `ET_TPB_1024`, `ET_TPB_128`, `ET_TPB_512`, `IDN_ET_DEVICE_LOOP`, `IDN_ET_PART_ROWS_OFF`, `IDN_ET_STAGE_LIVE_OFF`, `IDN_ET_TILED_SEARCH_OFF`, `LEGACY_NARROW_ET_BINNED`, `LEGACY_NARROW_ET_RM`

Lane result (expdel2-40-trees, 2026-11-15): 31 owned defines deleted; 6 non-owner folds (ET_SCALAR_STAGE_COMPARE, IDN_ALL_OFF, TREES_T11_LEVELS/T12_BYTES/T13_BYTES/T14_EXACT); KEEP `IDN_ET_DEVICE_LOOP` (B arm of the owed et-device-loop-early-exit A/B, used by checks/native_tests/et_device_loop_early_exit_gpu.sh) and `ET_MAX_ACC_32` (reference arm of extratrees/tools/check_accumulator_dispatch.sh); `FOREST_DEVICE_FINITE_OFF` already gone (expdel2-05-base).

### expdel2-41-trees (45 defines, 5 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_T01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T02` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T05` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T06` | DELETE-EXPERIMENT | g | 1+2 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T12` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F09` | DELETE-EXPERIMENT | g | 1+5 | 6 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N02` | DELETE-EXPERIMENT | g | 1+7 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N05` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N06` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N07` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N08` | DELETE-EXPERIMENT | g | 1+4 | 5 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N09` | DELETE-EXPERIMENT | g | 1+6 | 13 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N12` | DELETE-EXPERIMENT | g | 1+2 | 12 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_CTR_INDEX_FUSED` | DELETE-ARM | e | 1+3 | 4 | EXP:1274 ACCEPT/DEFAULT: FAST+Apple default -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_CTR_ONEHOT_DEVICE` | DELETE-EXPERIMENT | g | 1+4 | 5 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_CTR_PREP_SHARED` | DELETE-EXPERIMENT | g | 1+3 | 4 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_CTR_SORT_ONCE` | DELETE-ARM | e | 1+5 | 6 | EXP:1274 ACCEPT/DEFAULT: FAST+Apple default -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH` | DELETE-ARM | a | 1+0 | 11 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_RELAXED_PUBLISH` | DELETE-ARM | a | 1+3 | 16 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_1` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_128` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_16` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_2` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_256` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_32` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_4` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_64` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEARCH_RPT_8` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SEQCST_PUBLISH` | DELETE-ARM | a | 1+3 | 9 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_SHARED_CLASS_COUNTS` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ET_TIE_MAX_COLID` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SYM_CTR_ALL_OFF` | DELETE-OLD-PATH | d | 1+5 | 6 | EXP:1274 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TREES_T14_EXACT` | DELETE-ARM | a | 1+16 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T28_PREFIX` | DELETE-ARM | a | 1+5 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T28_SORT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |

Files: `extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`, `extratrees/impl/decisiontree/batched_levelalgo/split.mojo`, `gbdt/apple_fast_classical.mojo`, `gbdt/apple_fast_tree_experiments.mojo`, `gbdt/ctrs/fast_prep.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_T01`, `AFCL_T02`, `AFCL_T03`, `AFCL_T04`, `AFCL_T05`, `AFCL_T06`, `AFCL_T07`, `AFCL_T12`, `AFT_F09`, `AFT_N01`, `AFT_N02`, `AFT_N03`, `AFT_N04`, `AFT_N05`, `AFT_N06`, `AFT_N07`, `AFT_N08`, `AFT_N09`, `AFT_N10`, `AFT_N11`, `AFT_N12`, `CTR_INDEX_FUSED`, `CTR_ONEHOT_DEVICE`, `CTR_PREP_SHARED`, `CTR_SORT_ONCE`, `ET_EXP_NO_SEARCH_BARRIER`, `ET_EXP_RACY_RANGE_PUBLISH`, `ET_NO_SHARED_CLASS_COUNTS`, `ET_RELAXED_PUBLISH`, `ET_SEARCH_RPT_1`, `ET_SEARCH_RPT_128`, `ET_SEARCH_RPT_16`, `ET_SEARCH_RPT_2`, `ET_SEARCH_RPT_256`, `ET_SEARCH_RPT_32`, `ET_SEARCH_RPT_4`, `ET_SEARCH_RPT_64`, `ET_SEARCH_RPT_8`, `ET_SEQCST_PUBLISH`, `ET_SHARED_CLASS_COUNTS`, `ET_TIE_MAX_COLID`, `SYM_CTR_ALL_OFF`

Lane result (expdel2-41-trees, 2026-11-15): 40 of the 42 owned deleted with records; 3 non-owner folds (TREES_T14_EXACT, TREES_T28_SORT, TREES_T28_PREFIX). KEPT: `ET_SHARED_CLASS_COUNTS` and `ET_NO_SHARED_CLASS_COUNTS` (the arms of extratrees/tools/check_shared_score.sh and the trees_shared_counts_mask witness) and `COLUMN_CPU`.

### expdel2-42-trees (44 defines, 13 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_134_CONTROL` | DELETE-ARM | a | 2+3 | 56 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2031_SYM_RIDX_SPLITS` | DELETE-ARM | a | 1+8 | 22 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF` | DELETE-OLD-PATH | b | 1+8 | 28 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_2041_FAST_CHUNKS_PIN` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_2580_LEVEL_QUANT` | DELETE-ARM | a | 1+10 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2581_GROUP_WIDTH` | DELETE-ARM | a | 1+9 | 12 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2581_GROUP_WIDTH_OFF` | DELETE-OLD-PATH | b | 1+9 | 10 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_2635_LINEAR_BOUNDS` | DELETE-ARM | a | 1+1 | 7 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2710_TENSOR_ACC_DEAD` | DELETE-ARM | a | 1+1 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFCL_T03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T12` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G01` | DELETE-EXPERIMENT | g | 1+2 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G02` | DELETE-EXPERIMENT | g | 1+2 | 19 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G03` | DELETE-EXPERIMENT | g | 1+2 | 26 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G05` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G06` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G08` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G09` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G12` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N02` | DELETE-EXPERIMENT | g | 1+7 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | EXP:2273 CANDIDATE (perf fix, 2026-11-05) -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_BT_HW_F64_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:2272 CANDIDATE (perf fix, 2026-11-05) -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF` | DELETE-OLD-PATH | d | 1+2 | 3 | EXP:108 KEPT 57ddea946 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF` | DELETE-OLD-PATH | e | 1+3 | 5 | EXP:45 pointwise-searcher SymmetricTree GBDT rows (catboost-style regressor and classif -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_GBDT_FUSED_LEVEL_OFF` | DELETE-OLD-PATH | b | 2+8 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_INDEX_PACK_DEVICE` | DELETE-EXPERIMENT | g | 1+2 | 3 | EXP:170 NEUTRAL -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the card |
| `MOJOLEARN_GBDT_NS_PREDICT_BITVEC` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_PREDICT_PACKED` | DELETE-EXPERIMENT | g | 1+5 | 50 | EXP:171 NEUTRAL -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the card |
| `MOJOLEARN_GBDT_QUANT_DEVICE` | DELETE-EXPERIMENT | g | 1+8 | 18 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_GBDT_RIDX_GATHER_COST` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_SYM_RIDX` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1331 OWED -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_C45_GBDT` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_CTR_SORTFREE_SUMS` | DELETE-ARM | a | 1+7 | 55 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_HIST_REP_SM` | DELETE-ARM | e | 1+7 | 8 | EXP:1640 PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_HIST_REP_SM_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1640 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF` | DELETE-OLD-PATH | e | 1+3 | 8 | EXP:1637 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_T21_STREAMS` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |

Files: `gbdt/ctrs/kernel/ctr_calcers.mojo`, `gbdt/data/pairs.mojo`, `gbdt/estimator.mojo`, `gbdt/gpu_data/apple_fast_trees_experiments.mojo`, `gbdt/gpu_data/kernel/binarize.mojo`, `gbdt/gpu_data/sym_feat_switches.mojo`, `gbdt/gpu_util/partitions_reduce.mojo`, `gbdt/grid_creator/binarization.mojo`, `gbdt/grid_creator/border_types_cells.mojo`, `gbdt/grid_creator/gls_borders_device.mojo`, `gbdt/host/gbdt_oracle_ordered.mojo`, `gbdt/methods/doc_parallel_boosting.mojo`, `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2031_SYM_RIDX_SPLITS`, `2031_SYM_RIDX_SPLITS_OFF`, `2041_FAST_CHUNKS_PIN`, `2580_LEVEL_QUANT`, `2581_GROUP_WIDTH`, `2581_GROUP_WIDTH_OFF`, `2635_LINEAR_BOUNDS`, `2710_TENSOR_ACC_DEAD`, `AFT_G01`, `AFT_G02`, `AFT_G03`, `AFT_G04`, `AFT_G05`, `AFT_G06`, `AFT_G07`, `AFT_G08`, `AFT_G09`, `AFT_G10`, `AFT_G11`, `AFT_G12`, `GBDT_BT_COLUMN_BLOCKS_OFF`, `GBDT_BT_HW_F64_OFF`, `GBDT_CTR_PERM_BATCH_OFF`, `GBDT_DEVICE_ROUNDS_OFF`, `GBDT_FUSED_LEVEL_OFF`, `GBDT_INDEX_PACK_DEVICE`, `GBDT_PREDICT_PACKED`, `GBDT_QUANT_DEVICE`, `GBDT_RIDX_GATHER_COST`, `IDN_GBDT_CTR_PERM_BATCH_OFF`, `IDN_GBDT_PAIRLOGIT_GROUP_OFF`, `LEGACY_NARROW_SYM_RIDX`

Lane result (expdel2-42-trees, 2026-11-15): 30 of the 32 owned deleted with records; 7 non-owner folds (IDN_ALL_OFF, AFCL_T03, AFCL_T12, AFT_N02, TREES_C45_GBDT, TREES_CTR_SORTFREE_SUMS, TREES_T21_STREAMS). KEPT: `134_CONTROL` (positive control), `2710_TENSOR_ACC_DEAD` (negative control of tools/feature_freq_identity_leg.sh), `GBDT_NS_PREDICT_BITVEC` (arms of tools/gbdt_ns_bitvec_check.sh), `GBDT_DEVICE_ROUNDS_OFF` (B arm owed on NVIDIA + AMD). No edit needed here: TREES_HIST_REP_SM, TREES_HIST_REP_SM_OFF, TREES_ORD_STD_GRIDFOLD_OFF.

### expdel2-43-trees (43 defines, 11 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_T01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T02` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G03` | DELETE-EXPERIMENT | g | 1+2 | 26 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G08` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N01` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N02` | DELETE-EXPERIMENT | g | 1+7 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N03` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N05` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N06` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF` | DELETE-OLD-PATH | d | 1+2 | 19 | EXP:110 KEPT c55b8c377 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:137 DROPPED-noise -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_GBDT_DW_MODE_SKIP_OFF` | DELETE-OLD-PATH | d | 1+5 | 6 | EXP:112 KEPT 7736fa5b8 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_GBDT_FUSED_LEVEL_OFF` | DELETE-OLD-PATH | b | 2+8 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_ID_DEFER_COPY` | DELETE-ARM | e | 1+6 | 69 | EXP:1503 KEPT as switch -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_GBDT_LG_BATCH` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_BATCH2` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_BATCH32` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_BATCH4` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_BATCH8` | DELETE-ARM | a | 1+5 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_BATCH_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH128` | DELETE-ARM | e | 1+2 | 3 | EXP:113 KEPT 9722e5a2b -- force-on knob: the arm is already the FAST Apple default through its _OFF switch (or umbrella); deleting the read leaves the default byte-identical |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:113 KEPT 9722e5a2b -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH32` | DELETE-ARM | a | 2+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH64` | DELETE-ARM | e | 1+0 | 1 | EXP:114 KEPT 269ffa57a -- EXP:114 KEPT is the FAST width (superseded by 128); the IDENTICAL arm 64 (greedy_search_helper_depthwise.mojo:1335) was never measured |
| `MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF` | DELETE-OLD-PATH | b | 1+0 | 1 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_LG_EXACT_ID_OFF` | DELETE-OLD-PATH | b | 1+7 | 8 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_NS_INHERIT_ID_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF` | DELETE-OLD-PATH | b | 1+4 | 5 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_STEP_OLD` | DELETE-OLD-PATH | b | 2+2 | 6 | EXP:2423 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_ORD_HIST_FOLD_SKIP` | DELETE-ARM | a | 1+2 | 5 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_SYM_RESOLVE_BLOCK` | DELETE-EXPERIMENT | g | 1+2 | 10 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_TREES_C45_GBDT` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_HIST_PACKED_GH` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_HIST_WARP_AGG` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_LG_LEVEL_ROUNDS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T17_BATCH` | DELETE-ARM | e | 1+2 | 3 | EXP:1641 PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T19_DEFER` | DELETE-ARM | e | 1+2 | 3 | EXP:1627 DROP (quality loss), code deleted -- record read: loser, superseded, default-equal force knob, or never measured |

Files: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/dw2_level.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/split_chain_fused.mojo`, `gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo`, `gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo`, `gbdt/methods/kernel/pointwise_scores.mojo`, `gbdt/methods/kernel/pointwise_split_resolve.mojo`, `gbdt/methods/kernel/sym_fast.mojo`, `gbdt/methods/kernel_add_model_value.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `GBDT_DW2_SCAN_SMEM_OFF`, `GBDT_DW_FUSED_CHAIN_OFF`, `GBDT_DW_MODE_SKIP_OFF`, `GBDT_IDENTICAL_SPLIT_COPY`, `GBDT_ID_DEFER_COPY`, `GBDT_LG_BATCH`, `GBDT_LG_BATCH2`, `GBDT_LG_BATCH32`, `GBDT_LG_BATCH4`, `GBDT_LG_BATCH8`, `GBDT_LG_BATCH_OFF`, `GBDT_LG_EXACT_BATCH128`, `GBDT_LG_EXACT_BATCH128_OFF`, `GBDT_LG_EXACT_BATCH32`, `GBDT_LG_EXACT_BATCH64`, `GBDT_LG_EXACT_BATCH_OFF`, `GBDT_LG_EXACT_ID_OFF`, `GBDT_NS_INHERIT_ID_OFF`, `GBDT_NS_INHERIT_PARTITION_OFF`, `IDN_GBDT_NS_SCALE_DEVICE_OFF`, `SYM_RESOLVE_BLOCK`

Lane result (expdel2-43-trees, 2026-11-15): 41 defines folded in the 11 files (21 owned with records, 20 non-owner with markers and patches); KEPT `IDN_GBDT_STEP_OLD` (restore switch, A/B owed this wave) and `COLUMN_CPU`. `ORD_HIST_FOLD_SKIP`: the ord_small_fold_* kernels stay in pointwise_hist2_one_byte_templ.mojo while pointwise_kernels.mojo imports them; `TREES_C45_GBDT`: fused_root_scan_score_kernel in compute_scores.mojo is left unreferenced.

### expdel2-44-trees (44 defines, 17 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_2030_FUSED_EST_MOVE` | DELETE-ARM | e | 1+6 | 45 | EXP:1611 DROP_REDUNDANT, alias deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_2030_NO_FUSED_EST_MOVE` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2551_DEVICE_PARTITION` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_2551_DEVICE_PARTITION_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_AFCL_T04` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T06` | DELETE-EXPERIMENT | g | 1+2 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G01` | DELETE-EXPERIMENT | g | 1+2 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G02` | DELETE-EXPERIMENT | g | 1+2 | 19 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G07` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_G12` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N07` | DELETE-EXPERIMENT | g | 1+3 | 4 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N08` | DELETE-EXPERIMENT | g | 1+4 | 5 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N12` | DELETE-EXPERIMENT | g | 1+2 | 12 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_P05` | DELETE-EXPERIMENT | g | 1+3 | 45 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_EST_ITERS_DEVICE` | DELETE-EXPERIMENT | g | 1+4 | 5 | EXP:1303 RECORD NEUTRAL: stay off alone (EST_ITERS_DEVICE, CTR_PREP_SHARED, CTR_ONEHOT_DE -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the |
| `MOJOLEARN_GBDT_FUSED_LEVEL_OFF` | DELETE-OLD-PATH | b | 2+8 | 13 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_GBDT_NS_PREDICT_BITVEC` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF` | DELETE-OLD-PATH | e | 1+1 | 2 | EXP:2148 DEFAULT (bit-neutral) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_GBDT_PREDICT_PACKED` | DELETE-EXPERIMENT | g | 1+5 | 50 | EXP:171 NEUTRAL -- expdel-05 note: NEUTRAL; held only as both-arm prerequisite of an unmeasured Apple FAST card: goes with the card |
| `MOJOLEARN_GBDT_WALK_DEV_STEP_OFF` | DELETE-OLD-PATH | e | 1+5 | 6 | EXP:2147 DEFAULT (bit-neutral) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_APPLY_WIDE` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF` | DELETE-OLD-PATH | b | 1+3 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF` | DELETE-OLD-PATH | b | 1+4 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD` | DELETE-OLD-PATH | b | 1+0 | 1 | EXP:2417 RUN OWED -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+7 | 93 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_LEGACY_NARROW_ORD_ALL` | DELETE-OLD-PATH | b | 1+1 | 4 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:116 KEPT 28adffe85 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_ORDERED_BATCH_EST_OFF` | DELETE-OLD-PATH | d | 1+8 | 9 | EXP:117 KEPT cf45a2732 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_ORD_ALL_OFF` | DELETE-OLD-PATH | e | 1+11 | 43 | EXP:180 DROPPED-noise (standalone) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_ORD_DOC_ID_STORAGE` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:118 KEPT 28adffe85 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF` | DELETE-OLD-PATH | d | 1+0 | 1 | EXP:119 KEPT 28adffe85 -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PL_GROUP_NARROW_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:2454 DELETED (NEUTRAL) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_SYM_EST_ALL_OFF` | DELETE-OLD-PATH | d | 1+3 | 4 | EXP:1275 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TREES_C50_GB_PACKED` | DELETE-ARM | a | 1+12 | 105 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_CTR_SORTFREE_SUMS` | DELETE-ARM | a | 1+7 | 55 | no record -- no A/B recorded for the opt-in arm |

Files: `gbdt/methods/leaves_estimation/apple_fast_est.mojo`, `gbdt/methods/leaves_estimation/device_walker.mojo`, `gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo`, `gbdt/methods/leaves_estimation/leaves_estimation.mojo`, `gbdt/methods/leaves_estimation/pointwise_oracle.mojo`, `gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo`, `gbdt/methods/ordered_boosting.mojo`, `gbdt/methods/ordered_fast_switches.mojo`, `gbdt/methods/pointwise_kernels.mojo`, `gbdt/methods/pointwise_optimization_subsets.mojo`, `gbdt/models/add_non_symmetric_tree_doc_parallel.mojo`, `gbdt/models/kernel/add_bin_values.mojo`, `gbdt/models/ns_bitvector_predict.mojo`, `gbdt/resident_model.mojo`, `gbdt/targets/kernel/pair_logit.mojo`, `gbdt/targets/kernel/pair_logit_group.mojo`, `gbdt/targets/kernel/query_rmse.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `2030_FUSED_EST_MOVE`, `2030_NO_FUSED_EST_MOVE`, `2551_DEVICE_PARTITION`, `2551_DEVICE_PARTITION_OFF`, `EST_ITERS_DEVICE`, `GBDT_NS_PREDICT_BITVEC`, `GBDT_ORD_TREE_BUFS_OFF`, `GBDT_WALK_DEV_STEP_OFF`, `IDN_GBDT_APPLY_WIDE`, `IDN_GBDT_EST_ONE_STEP_DEVICE_OFF`, `IDN_GBDT_MC_ONE_STEP_DEVICE_OFF`, `IDN_GBDT_NS_PREDICT_PACKED_OFF`, `IDN_GBDT_PREBINNED_INFER_OLD`, `IDN_GBDT_PREDICT_FOUR_OFF`, `IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF`, `IDN_ORD_ONE_STEP_DEVICE_OFF`, `IDN_ORD_STD_SCALE_DEVICE_OFF`, `LEGACY_NARROW_ORD_ALL`, `MULTICLASS_HESSIAN_BATCH_OFF`, `ORDERED_BATCH_EST_OFF`, `ORD_ALL_OFF`, `ORD_DOC_ID_STORAGE`, `PAIRLOGIT_EST_REUSE_OFF`, `PAIRLOGIT_GROUP_FUSED_OFF`, `PL_GROUP_NARROW_OFF`, `SYM_EST_ALL_OFF`

Lane result (expdel2-44-trees, 2026-11-15): 25 of the 26 owned deleted with records; 17 non-owner folds in this lane's files (IDN_ALL_OFF, AFCL_T04/T06, AFT_G01/G02/G07/G11/G12/N07/N08/N12/P05, GBDT_FUSED_LEVEL_OFF, GBDT_PREDICT_PACKED, IDN_GBDT_PAIRLOGIT_GROUP_OFF doc only, TREES_C50_GB_PACKED, TREES_CTR_SORTFREE_SUMS). KEPT: `IDN_GBDT_PREBINNED_INFER_OLD` (EXP:2417 RUN OWED, its B arm is owed this wave) and `COLUMN_CPU`. No edit needed: `ORD_ALL` reads in oblivious_tree_doc_parallel_structure_searcher.mojo (the alias stays the FAST+Apple expression). Left for the integrator: `IDN_APPLY_WIDE` (False), `launch_oblivious_apply_wide`, `uniform_positive_depth` and `compute_bins_and_add_uniform_kernel` in gbdt/models/kernel/add_bin_values.mojo stay while gbdt/methods/doc_parallel_boosting.mojo imports them; `GBDT_NS_BITVEC` (False) and `GBDT_WALK_DEV_STEP` (True) likewise. Tooling that still passes a deleted define: pixi task check-gbdt-per-round-2551-host-partition (-D MOJOLEARN_2551_DEVICE_PARTITION_OFF), tools/gbdt_arms_leg.sh, tools/gbdt_fused_ab.sh, tools/gbdt_ns_bitvec_check.sh, tools/apple_slow_paths_check.py, tools/identical_candidate_recipes.json, checks/gbdt_fused_fit_check.mojo.

### expdel2-45-trees (45 defines, 5 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_3040_YETI_BLOCK` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_3040_YETI_SEQUENTIAL` | DELETE-ARM | a | 1+0 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_AFT_N09` | DELETE-EXPERIMENT | g | 1+6 | 13 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_N11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1584 DROP, define deleted (no-op) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+4 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_RESIDENT_OFF` | DELETE-OLD-PATH | b | 2+9 | 11 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_IF_SCORES_F64_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:2397 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ORD_CAT_PLANES_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ORD_INDEX_SHARED` | DELETE-ARM | a | 1+1 | 2 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ORD_HIST_FOLD_SKIP` | DELETE-ARM | a | 1+2 | 5 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS` | DELETE-ARM | a | 1+1 | 2 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | grid fg-gbdt-ordered.json pending -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_C45_GBDT` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_C50_IF_PACKED` | DELETE-ARM | a | 1+3 | 13 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_CTR_SORTFREE_SUMS` | DELETE-ARM | a | 1+7 | 55 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_HIST_PACKED_GH` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_HIST_REP_SM` | DELETE-ARM | e | 1+7 | 8 | EXP:1640 PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_HIST_REP_SM_OFF` | DELETE-OLD-PATH | e | 1+3 | 4 | EXP:1640 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_HIST_WARP_AGG` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_LG_LEVEL_ROUNDS` | DELETE-ARM | a | 1+3 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF` | DELETE-OLD-PATH | e | 1+3 | 8 | EXP:1637 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_T17_BATCH` | DELETE-ARM | e | 1+2 | 3 | EXP:1641 PROMOTED -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T18` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T19_DEFER` | DELETE-ARM | e | 1+2 | 3 | EXP:1627 DROP (quality loss), code deleted -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T21` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T21_STREAMS` | DELETE-ARM | a | 1+7 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T22_OFF` | DELETE-OLD-PATH | e | 1+0 | 1 | EXP:1636 PROMOTED -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_T23` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T24` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T25` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T26` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T28` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T28_PREFIX` | DELETE-ARM | a | 1+5 | 8 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T28_SORT` | DELETE-ARM | a | 1+4 | 5 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T29` | DELETE-ARM | e | 1+0 | 1 | EXP:1869 KEPT (code stays) -- record read: loser, superseded, default-equal force knob, or never measured |
| `MOJOLEARN_TREES_T30` | DELETE-ARM | a | 1+0 | 1 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T30_ADABOOST` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T43_RADIX_SELECT` | DELETE-ARM | a | 1+2 | 19 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF` | DELETE-OLD-PATH | d | 1+0 | 2 | EXP:125 KEPT 269ffa57a -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_YETI_FAST_SORT_OFF` | DELETE-OLD-PATH | e | 1+0 | 2 | EXP:228 NEUTRAL -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_YETI_SEARCH_TASK16K_OFF` | DELETE-OLD-PATH | e | 1+0 | 2 | EXP:155 DROPPED-noise -- record read: the default is the recorded winner or the kept path |

Files: `gbdt/targets/kernel/yeti_rank.mojo`, `gbdt/trees_hist_switches.mojo`, `gbdt/trees_identical_switches.mojo`, `gbdt/trees_small_switches.mojo`, `isolation_forest/impl/isolation_forest.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `3040_YETI_BLOCK`, `3040_YETI_SEQUENTIAL`, `IDN_GBDT_YETI_BLOCK_AMD_OFF`, `IDN_IF_EPILOGUE_DEVICE_OFF`, `IDN_IF_QUERY_DEVICE_OFF`, `IDN_IF_SCORES_F64_OFF`, `IDN_ORD_CAT_PLANES_OFF`, `IDN_ORD_INDEX_SHARED`, `ORD_HIST_FOLD_SKIP`, `ORD_HIST_FOLD_SKIP_DOCS`, `ORD_SKIP_UNSEARCHED_PERM_OFF`, `TREES_C45_GBDT`, `TREES_C50_IF_PACKED`, `TREES_CTR_SORTFREE_SUMS`, `TREES_HIST_PACKED_GH`, `TREES_HIST_REP_SM`, `TREES_HIST_REP_SM_OFF`, `TREES_HIST_WARP_AGG`, `TREES_LG_LEVEL_ROUNDS`, `TREES_ORD_STD_GRIDFOLD_OFF`, `TREES_T17_BATCH`, `TREES_T18`, `TREES_T19_DEFER`, `TREES_T21`, `TREES_T21_STREAMS`, `TREES_T22_OFF`, `TREES_T23`, `TREES_T24`, `TREES_T25`, `TREES_T26`, `TREES_T28`, `TREES_T28_PREFIX`, `TREES_T28_SORT`, `TREES_T29`, `TREES_T30`, `TREES_T30_ADABOOST`, `TREES_T43_RADIX_SELECT`, `YETI_EST_REUSE_SEARCH_OFF`, `YETI_FAST_SORT_OFF`, `YETI_SEARCH_TASK16K_OFF`

### expdel2-46-trees (33 defines, 8 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_AFCL_T10` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFCL_T11` | DELETE-EXPERIMENT | g | 1+2 | 3 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F10` | DELETE-EXPERIMENT | g | 1+1 | 2 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F11` | DELETE-EXPERIMENT | g | 1+10 | 11 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AFT_F12` | DELETE-EXPERIMENT | g | 1+2 | 24 | no record -- never-run Apple FAST programme card |
| `MOJOLEARN_AGN_IDN_SYN_POOL_OFF` | DELETE-OLD-PATH | b | 1+3 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_COLUMN_CPU` | KEEP (build switch: the CPU column; corrected 2026-11-15 by the orchestrator, not an experiment) | a | 28+743 | 1074 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_DART_DEVICE_OFF` | DELETE-OLD-PATH | d | 1+10 | 19 | EXP:104 KEPT 741a5495d -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_DART_DEVICE_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_GATES_OFF` | DELETE-OLD-PATH | e | 7+25 | 77 | EXP:2479 DELETED (old path) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_PSHAP_RESIDENT_OFF` | DELETE-OLD-PATH | e | 1+2 | 3 | EXP:2051 KEEP (promoted 2026-10-11; _OFF restores) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF` | DELETE-OLD-PATH | b | 1+1 | 2 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IF_DIAG_ENTRY_RETURN` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IF_DIAG_GATHER_ONLY` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IF_DIAG_NO_RECORD` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IF_DIAG_NO_REJECT` | DELETE-ARM | a | 1+1 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_IF_ROWMAJOR_OFF` | DELETE-OLD-PATH | d | 1+6 | 12 | EXP:115 KEPT 269ffa57a -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_PSHAP_DELTA_OFF` | DELETE-OLD-PATH | b | 1+2 | 3 | EXP:1595 OPEN -- no A/B recorded for the old path |
| `MOJOLEARN_SHAP_FAST_ROW_PAIR` | DELETE-ARM | a | 1+3 | 5 | no record -- no A/B recorded (Apple fast opt-in) |
| `MOJOLEARN_SHAP_PERM_CACHE_OFF` | DELETE-OLD-PATH | b | 1+5 | 10 | EXP:1403 READY-AB -- no A/B recorded for the old path |
| `MOJOLEARN_SHAP_TREE_TAB_OFF` | DELETE-OLD-PATH | d | 1+5 | 6 | EXP:1271 ACCEPT/DEFAULT: FAST+Apple default -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_TREESHAP_ACC_SHARED` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREESHAP_FAST_TABLE_OFF` | DELETE-OLD-PATH | e | 1+6 | 7 | EXP:186 KEPT/DEFAULT -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_TREES_C47_IF_LIVE_TASKS` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_C48_IF_PARTITION` | DELETE-ARM | a | 1+3 | 38 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_IF_BUILD_TPB` | DELETE-ARM | a | 2+4 | 6 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_PLATT_ONE_READ_OFF` | DELETE-OLD-PATH | b | 1+2 | 4 | EXP:229 PENDING -- no A/B recorded for the old path |
| `MOJOLEARN_TREES_T30_ADABOOST` | DELETE-ARM | a | 1+3 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T40_ROWMAJOR` | DELETE-ARM | a | 1+2 | 3 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T41_TREE_BATCH` | DELETE-ARM | a | 1+1 | 2 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T42_CORRECTION_CACHE` | DELETE-ARM | a | 1+2 | 4 | no record -- no A/B recorded for the opt-in arm |
| `MOJOLEARN_TREES_T45_LINEAR_UNWIND` | DELETE-ARM | a | 1+2 | 21 | no record -- no A/B recorded for the opt-in arm |

Files: `isolation_forest/impl/isolation_tree_builder.mojo`, `xtrees/agnostic_device.mojo`, `xtrees/dart_device.mojo`, `xtrees/dart_units.mojo`, `xtrees/glue_runs.mojo`, `xtrees/ops_device_boost.mojo`, `xtrees/shap.mojo`, `xtrees/shap_device.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `AFCL_T10`, `AFCL_T11`, `AFT_F10`, `AFT_F11`, `AFT_F12`, `AGN_IDN_SYN_POOL_OFF`, `DART_DEVICE_OFF`, `IDN_DART_DEVICE_OFF`, `IDN_PSHAP_RESIDENT_OFF`, `IDN_SHAP_DEVICE_MODEL_OFF`, `IF_DIAG_ENTRY_RETURN`, `IF_DIAG_GATHER_ONLY`, `IF_DIAG_NO_RECORD`, `IF_DIAG_NO_REJECT`, `IF_ROWMAJOR_OFF`, `PSHAP_DELTA_OFF`, `SHAP_FAST_ROW_PAIR`, `SHAP_PERM_CACHE_OFF`, `SHAP_TREE_TAB_OFF`, `TREESHAP_ACC_SHARED`, `TREESHAP_FAST_TABLE_OFF`, `TREES_C47_IF_LIVE_TASKS`, `TREES_C48_IF_PARTITION`, `TREES_IF_BUILD_TPB`, `TREES_PLATT_ONE_READ_OFF`, `TREES_T40_ROWMAJOR`, `TREES_T41_TREE_BATCH`, `TREES_T42_CORRECTION_CACHE`, `TREES_T45_LINEAR_UNWIND`

Lane result (lane/expdel2-46-trees, 2026-11-15): 32 defines folded (29 owned with records, 3 non-owner: IDN_ALL_OFF, IDN_GATES_OFF, TREES_T30_ADABOOST); COLUMN_CPU kept (correction). No KEEPs beyond it: no define here is a check arm, a sabotage control or a measured-winning off arm, and no default-launched kernel was removed.

### expdel2-47-promote (8 defines, 1 code files)

| define | decision | rule | sites | lines | evidence / why |
|---|---|---|---|---|---|
| `MOJOLEARN_ARIMA_FAST_SEARCH_REUSE` | PROMOTE | c | 1+4 | 16 | EXP:1420 KEEP (FAST+Apple default, `_OFF`) -- EXP:1420 KEEP (M3 rab16: synthetic 18229.7 -> 13688.1, taxi-hourly 16386.5 -> 9872.2) but code is opt-in (arima/impl/fast_order_search.mojo:111): becomes  |
| `MOJOLEARN_ARIMA_ORDER_BATCH_OFF` | DELETE-OLD-PATH | d | 2+10 | 15 | EXP:850 KEEP, default merged on main dc2285bc0; manager default/OFF builds both rc0. Sou -- promoted on one vendor in the record; tombstone notes the owed other vendor |
| `MOJOLEARN_IDN_ALL_OFF` | DELETE-OLD-PATH | e | 403+1335 | 3911 | EXP:2438 DELETED 2026-11-15 (old path switch; resident default kept) -- record read: the default is the recorded winner or the kept path |
| `MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+5 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF` | DELETE-OLD-PATH | b | 2+10 | 27 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF` | DELETE-OLD-PATH | b | 1+6 | 7 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF` | DELETE-OLD-PATH | b | 2+8 | 12 | no record -- no A/B recorded for the old path |
| `MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF` | DELETE-OLD-PATH | b | 1+5 | 6 | no record -- no A/B recorded for the old path |

Files: `arima/impl/fast_order_search.mojo`

Owned here (patch, TOMBSTONES.md section, refusal, grid and EXPERIMENTS.md records): `ARIMA_FAST_SEARCH_REUSE`, `IDN_ARIMA_IC_DEVICE_OFF`, `IDN_ARIMA_ORDER_CAP_OFF`, `IDN_ARIMA_ORDER_DEVICE_OFF`, `IDN_ARIMA_ORDER_SEASONAL_OFF`

## Tooling that names deleted defines (one tooling lane)

Live tooling that still names a define this plan deletes: one tooling lane moves grid control entries to `removed`, regenerates the generated grid plans, and drops the arms from A/B scripts, build lists and idea programmes. 1033 further files are dated records (dated experiment directories, grid results, docs) that keep the names as history and are not edited; the protected records (`experiments/removed/`, `docs/TOMBSTONES.md`, `docs/apple-fast/EXPERIMENTS.md`, `bench/results/`) are not counted. A file under `python/` or `checks/` that names a define is checked by that lane against verify (AGENTS.md "The verify command is protected") before any edit.

| file | deleted defines named |
|---|---|
| `.github/workflows/native-checks.yml` | 1 |
| `autograd/checks/mlp_cases_digests.tsv` | 1 |
| `bench/neighbors_apple3_jobs/job3_cmd.txt` | 2 |
| `bench/neighbors_apple3_jobs/job4c_cmd.txt` | 2 |
| `bench/speed/rf_higgs_columns_ab.py` | 2 |
| `bench/apple_fast_round2_ab.py` | 1 |
| `bench/neighbors_apple3_jobs/job4b_cmd.txt` | 1 |
| `bench/speed/gbdt_ns_bitvec_check.py` | 1 |
| `bindings/build.sh` | 1 |
| `bindings/build_byte_lm.sh` | 1 |
| `bindings/build_host_family.sh` | 1 |
| `bindings/build_tokenizer_fast.sh` | 1 |
| `bindings/build_training.sh` | 1 |
| `bindings/build_transformer.sh` | 1 |
| `bindings/merged_build.sh` | 1 |
| `checks/gbdt_sym_ridx_ab.py` | 1 |
| `checks/native_tests/et_device_loop_early_exit_gpu.sh` | 1 |
| `ensemble/bench/rf_candidate_sweep.py` | 2 |
| `ensemble/checks/rf_column_tiles_public.sh` | 1 |
| `experiments/six_lane_integration/catalog.json` | 339 |
| `experiments/six_lane_integration/build_coverage.json` | 296 |
| `experiments/six_lane_integration/build_plan.json` | 291 |
| `experiments/six_lane_integration/grid/grid-plan.json` | 241 |
| `experiments/six_lane_integration/grid/grid-build-plan.json` | 142 |
| `experiments/six_lane_integration/grid-fast/grid-plan.json` | 115 |
| `experiments/six_lane_integration/modular_blockers.json` | 111 |
| `experiments/neural_identical_ab/experiment_inventory.json` | 108 |
| `experiments/six_lane_integration/grid-fast/grid-build-plan.json` | 106 |
| `experiments/six_lane_integration/grid-fast/grid-fast-queue.txt` | 106 |
| `experiments/classical_identical_ideas/AB_EXPERIMENT_INVENTORY.json` | 73 |
| `experiments/six_lane_integration/grid_controls/trees-cleanup.json` | 56 |
| `experiments/classical_identical_ideas/C01/manifest.json` | 52 |
| `experiments/six_lane_integration/grid_controls/neural-seq-train-dedupe.json` | 49 |
| `experiments/neural_identical_ab/arms.json` | 43 |
| `experiments/classical_identical_ideas/interactions.json` | 39 |
| `experiments/neural_identical_ab/lanes/attention_integration_inventory.json` | 38 |
| `experiments/neural_identical_ab/implementation.json` | 37 |
| `experiments/apple_fast_trees/N.json` | 24 |
| `experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json` | 22 |
| `experiments/six_lane_integration/grid_controls/classical-misc.json` | 21 |
| `experiments/apple_fast_trees/G.json` | 20 |
| `experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json` | 20 |
| `experiments/classical_identical_ideas/linear_implementation.json` | 19 |
| `experiments/classical_identical_ideas/graph_implementation.json` | 16 |
| `experiments/neural_identical_ab/lanes/gemm_integration_inventory.json` | 16 |
| `experiments/apple_fast_trees/P.json` | 15 |
| `experiments/apple_fast_trees/F.json` | 14 |
| `experiments/performance_ideas/I14/compile_matrix.json` | 14 |
| `experiments/performance_ideas/I14/native_arms.json` | 14 |
| `experiments/neural_identical_ab/lanes/gemm.json` | 13 |
| `experiments/six_lane_integration/grid-fast/controls/untried-20261005.json` | 12 |
| `experiments/neural_identical_ab/lanes/state_cnn.json` | 11 |
| `experiments/six_lane_integration/grid_controls/trees-small.json` | 11 |
| `experiments/classical_identical_ideas/shared_implementation.json` | 10 |
| `experiments/neural_identical_ab/lanes/training_integration_inventory.json` | 9 |
| `experiments/performance_ideas/I14/manifest.json` | 9 |
| `experiments/six_lane_integration/grid_controls/neural-small.json` | 9 |
| `experiments/classical_identical_ideas/C29/manifest.json` | 8 |
| `experiments/classical_identical_ideas/stats_implementation.json` | 8 |
| `experiments/neural_identical_ab/lanes/training.json` | 8 |
| `experiments/neural_identical_ab/lanes/training_source.py` | 8 |
| `experiments/six_lane_integration/grid_controls/classical-decomp.json` | 8 |
| `experiments/six_lane_integration/grid_controls/classical-kmeans.json` | 8 |
| `experiments/performance_ideas/F15/manifest.json` | 7 |
| `experiments/six_lane_integration/grid_controls/classical-cv.json` | 7 |
| `experiments/six_lane_integration/grid_controls/fg-tsne-dbscan.json` | 7 |
| `experiments/six_lane_integration/grid_controls/trees-structural.json` | 7 |
| `experiments/classical_identical_ideas/C02/manifest.json` | 6 |
| `experiments/classical_identical_ideas/C13/manifest.json` | 6 |
| `experiments/performance_ideas/F15/compile_audit.json` | 6 |
| `experiments/six_lane_integration/grid_controls/classical-encoders.json` | 6 |
| `experiments/six_lane_integration/grid_controls/classical-nbda.json` | 6 |
| `experiments/six_lane_integration/grid_controls/fg-gbdt-ordered.json` | 6 |
| `experiments/six_lane_integration/grid_controls/fg-ivf.json` | 6 |
| `experiments/neural_identical_ab/lanes/attention.json` | 5 |
| `experiments/performance_ideas/F10/compile_audit.json` | 5 |
| `experiments/performance_ideas/F10/manifest.json` | 5 |
| `experiments/classical_identical_ideas/C03/manifest.json` | 4 |
| `experiments/classical_identical_ideas/C07/manifest.json` | 4 |
| `experiments/classical_identical_ideas/C16/manifest.json` | 4 |
| `experiments/classical_identical_ideas/C58/manifest.json` | 4 |
| `experiments/neural_identical_ab/lanes/training_integration_source.py` | 4 |
| `experiments/performance_ideas/F12/compile_audit.json` | 4 |
| `experiments/performance_ideas/F12/manifest.json` | 4 |
| `experiments/performance_ideas/I12/compile_matrix.json` | 4 |
| `experiments/performance_ideas/I12/native_arms.json` | 4 |
| `experiments/performance_ideas/I19/compile_matrix.json` | 4 |
| `experiments/performance_ideas/I19/native_arms.json` | 4 |
| `experiments/performance_ideas/I22/campaign.py` | 4 |
| `experiments/performance_ideas/I22/compile_matrix.json` | 4 |
| `experiments/performance_ideas/I22/native_arms.json` | 4 |
| `experiments/six_lane_integration/grid_controls/fg-linear.json` | 4 |
| `experiments/six_lane_integration/grid_controls/fg-pca.json` | 4 |
| `experiments/six_lane_integration/grid_controls/neural-fusions.json` | 4 |
| `experiments/classical_identical_ideas/C20/manifest.json` | 3 |
| `experiments/classical_identical_ideas/C27/manifest.json` | 3 |
| `experiments/performance_ideas/F07/compile_audit.json` | 3 |
| `experiments/performance_ideas/F07/manifest.json` | 3 |
| `experiments/performance_ideas/F08/manifest.json` | 3 |
| `experiments/performance_ideas/F13/compile_audit.json` | 3 |
| `experiments/performance_ideas/F13/manifest.json` | 3 |
| `experiments/performance_ideas/I15/compile_matrix.json` | 3 |
| `experiments/performance_ideas/I15/manifest.json` | 3 |
| `experiments/performance_ideas/I15/native_arms.json` | 3 |
| `experiments/six_lane_integration/grid_controls/apple-slow-paths.json` | 3 |
| `experiments/six_lane_integration/grid_controls/classical-cv-folds.json` | 3 |
| `experiments/six_lane_integration/grid_controls/classical-fixes.json` | 3 |
| `experiments/six_lane_integration/grid_controls/gap-tsqr.json` | 3 |
| `experiments/six_lane_integration/grid_controls/neighbors-small-ports.json` | 3 |
| `experiments/six_lane_integration/grid_controls/neural-ce-denom.json` | 3 |
| `experiments/six_lane_integration/grid_controls/neural-io-2.json` | 3 |
| `experiments/classical_identical_ideas/C05/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C06/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C17/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C24/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C26/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C32/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C33/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C35/manifest.json` | 2 |
| `experiments/classical_identical_ideas/C40/manifest.json` | 2 |
| `experiments/neural_identical_ab/lanes/lm_head_integration_inventory.json` | 2 |
| `experiments/neural_identical_ab/lanes/residual.json` | 2 |
| `experiments/performance_ideas/F04/manifest.json` | 2 |
| `experiments/performance_ideas/F08/compile_audit.json` | 2 |
| `experiments/performance_ideas/F11/compile_audit.json` | 2 |
| `experiments/performance_ideas/F11/manifest.json` | 2 |
| `experiments/performance_ideas/F18/manifest.json` | 2 |
| `experiments/performance_ideas/F19/manifest.json` | 2 |
| `experiments/performance_ideas/F20/compile_audit.json` | 2 |
| `experiments/performance_ideas/F20/manifest.json` | 2 |
| `experiments/performance_ideas/I08/compile_matrix.json` | 2 |
| `experiments/performance_ideas/I08/native_arms.json` | 2 |
| `experiments/performance_ideas/I10/compile_matrix.json` | 2 |
| `experiments/performance_ideas/I10/manifest.json` | 2 |
| `experiments/performance_ideas/I10/native_arms.json` | 2 |
| `experiments/performance_ideas/I17/compile_matrix.json` | 2 |
| `experiments/performance_ideas/I17/native_arms.json` | 2 |
| `experiments/performance_ideas/I18/compile_matrix.json` | 2 |
| `experiments/performance_ideas/I18/manifest.json` | 2 |
| `experiments/performance_ideas/I22/manifest.json` | 2 |
| `experiments/seq-predict-tiled/compile.json` | 2 |
| `experiments/six_lane_integration/grid-fast/controls/apple-fast-round2.json` | 2 |
| `experiments/six_lane_integration/grid_controls/apple-fast-round2.json` | 2 |
| `experiments/six_lane_integration/grid_controls/classical-structural.json` | 2 |
| `experiments/six_lane_integration/grid_controls/classical-te-gmm.json` | 2 |
| `experiments/six_lane_integration/grid_controls/fg-knn-nb.json` | 2 |
| `experiments/six_lane_integration/grid_controls/gap-gemm-layers.json` | 2 |
| `experiments/six_lane_integration/grid_controls/gap-ivf.json` | 2 |
| `experiments/six_lane_integration/grid_controls/gap-small-members.json` | 2 |
| `experiments/six_lane_integration/grid_controls/layernorm-idn.json` | 2 |
| `experiments/six_lane_integration/grid_controls/samba-resident.json` | 2 |
| `experiments/six_lane_integration/grid_controls/trees-hist-ideas.json` | 2 |
| `experiments/six_lane_integration/grid_controls/trees-predict-ideas.json` | 2 |
| `experiments/classical_identical_ideas/C08/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C09/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C10/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C11/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C12/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C14/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C15/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C18/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C19/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C21/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C22/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C28/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C31/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C34/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C36/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C39/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C41/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C42/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C43/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C44/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C52/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C53/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C54/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C56/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C57/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C59/manifest.json` | 1 |
| `experiments/classical_identical_ideas/C60/manifest.json` | 1 |
| `experiments/overhead-mamba-bwd/compile.json` | 1 |
| `experiments/overhead-mamba-bwd/pod_smoke_5b.sh` | 1 |
| `experiments/overhead-module-lookup/compile.json` | 1 |
| `experiments/performance_ideas/F14/compile_audit.json` | 1 |
| `experiments/performance_ideas/F14/manifest.json` | 1 |
| `experiments/performance_ideas/F17/compile_audit.json` | 1 |
| `experiments/performance_ideas/F17/manifest.json` | 1 |
| `experiments/performance_ideas/F18/compile_audit.json` | 1 |
| `experiments/performance_ideas/F19/compile_audit.json` | 1 |
| `experiments/performance_ideas/I06/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I06/manifest.json` | 1 |
| `experiments/performance_ideas/I06/native_arms.json` | 1 |
| `experiments/performance_ideas/I07/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I07/manifest.json` | 1 |
| `experiments/performance_ideas/I07/native_arms.json` | 1 |
| `experiments/performance_ideas/I11/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I11/manifest.json` | 1 |
| `experiments/performance_ideas/I11/native_arms.json` | 1 |
| `experiments/performance_ideas/I12/manifest.json` | 1 |
| `experiments/performance_ideas/I19/manifest.json` | 1 |
| `experiments/performance_ideas/I20/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I20/manifest.json` | 1 |
| `experiments/performance_ideas/I21/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I21/native_arms.json` | 1 |
| `experiments/performance_ideas/I23/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I23/manifest.json` | 1 |
| `experiments/performance_ideas/I23/native_arms.json` | 1 |
| `experiments/performance_ideas/I24/compile_matrix.json` | 1 |
| `experiments/performance_ideas/I24/manifest.json` | 1 |
| `experiments/six_lane_integration/grid_controls/apple-holes.json` | 1 |
| `experiments/six_lane_integration/grid_controls/cnn-resident-glue.json` | 1 |
| `experiments/six_lane_integration/grid_controls/ets-team-idn.json` | 1 |
| `experiments/six_lane_integration/grid_controls/gap-linalg.json` | 1 |
| `experiments/six_lane_integration/grid_controls/gap-shap-nb.json` | 1 |
| `experiments/six_lane_integration/grid_controls/gap-svc-tsne.json` | 1 |
| `experiments/six_lane_integration/grid_controls/hdb-ivf.json` | 1 |
| `experiments/six_lane_integration/grid_controls/kde-qtile.json` | 1 |
| `experiments/six_lane_integration/grid_controls/lars-grid-gram.json` | 1 |
| `experiments/six_lane_integration/grid_controls/mamba-folds.json` | 1 |
| `experiments/six_lane_integration/grid_controls/seq-nm-spec.json` | 1 |
| `experiments/six_lane_integration/grid_controls/svgp-b-gated.json` | 1 |
| `experiments/six_lane_integration/grid_controls/xprep-one-copy.json` | 1 |
| `extratrees/tools/check_shared_score.sh` | 2 |
| `extratrees/NOT_IMPLEMENTED.tsv` | 1 |
| `extratrees/tools/check_accumulator_dispatch.sh` | 1 |
| `pixi.toml` | 3 |
| `python/mojolearn/_expansion_prep.py` | 9 |
| `python/mojolearn/_expansion_trees.py` | 4 |
| `python/mojolearn/_expansion_cnn.py` | 3 |
| `python/mojolearn/linalg_fast.py` | 2 |
| `python/mojolearn/_backend.py` | 1 |
| `python/mojolearn/_expansion_decomp.py` | 1 |
| `python/mojolearn/_expansion_linear.py` | 1 |
| `python/mojolearn/_native_neural_bridge.py` | 1 |
| `python/mojolearn/_x_sequence_device.py` | 1 |
| `python/mojolearn/_x_sequence_moe.py` | 1 |
| `python/mojolearn/_x_sequence_norm.py` | 1 |
| `python/mojolearn/preprocessing.py` | 1 |
| `python/mojolearn/tests/_legacy_neighbors_graph.py` | 1 |
| `python/mojolearn/tests/_legacy_neighbors_loc.py` | 1 |
| `python/mojolearn/tests/_legacy_neighbors_semi.py` | 1 |
| `python/mojolearn/tests/test_byte_lm_step_routes.py` | 1 |
| `python/mojolearn/tests/test_cpu_training_mamba.py` | 1 |
| `python/mojolearn/tests/test_iforest_save_forest.py` | 1 |
| `python/mojolearn/tests/test_mamba_native_public.py` | 1 |
| `python/mojolearn/tests/test_samba_inference_native.py` | 1 |
| `tools/identical_candidate_recipes.json` | 45 |
| `tools/six_lane_ab.py` | 20 |
| `tools/identity_lanes/coverage_declared.tsv` | 13 |
| `tools/mamba_host_gen.py` | 9 |
| `tools/mamba3_statepass_leg.sh` | 6 |
| `tools/neural_identical_ideas.py` | 5 |
| `tools/gbdt_arms_leg.sh` | 4 |
| `tools/lm_attention_policy_body.sh` | 4 |
| `tools/mcd_compat_ab.sh` | 4 |
| `tools/six_lane_catalog.py` | 4 |
| `tools/test_xseq_native_provider.py` | 4 |
| `tools/forest_groves_body.sh` | 3 |
| `tools/linear_apple3/job2.json` | 3 |
| `tools/lm_attention_guard_body.sh` | 3 |
| `tools/lm_attention_norefuse_body.sh` | 3 |
| `tools/lm_attention_repair_body.sh` | 3 |
| `tools/lm_attention_tail_body.sh` | 3 |
| `tools/lm_attention_vendor_train_body.sh` | 3 |
| `tools/test_apple_bf16_default.py` | 3 |
| `tools/test_six_lane_grid_lq.py` | 3 |
| `tools/alias_y_estash_l40s_body.sh` | 2 |
| `tools/amd_step_time2_leg1.sh` | 2 |
| `tools/apple_fast_tree_ideas.py` | 2 |
| `tools/apple_slow_paths_check.py` | 2 |
| `tools/attention_alias_y_amd_body.sh` | 2 |
| `tools/attention_tq16_amd_leg.sh` | 2 |
| `tools/bench_board_algos.py` | 2 |
| `tools/box_merged_verify.sh` | 2 |
| `tools/gbdt_fused_ab.sh` | 2 |
| `tools/identical_wave_plan.json` | 2 |
| `tools/knn_query_batch_probe.sh` | 2 |
| `tools/linear_apple3/local_compile_core.json` | 2 |
| `tools/linear_apple3/m2.json` | 2 |
| `tools/lm_attention_default_body.sh` | 2 |
| `tools/lm_attention_fallback_body.sh` | 2 |
| `tools/lm_attention_trigger_body.sh` | 2 |
| `tools/lm_attention_vendor_extra_body.sh` | 2 |
| `tools/mamba3_increment_tile_leg.sh` | 2 |
| `tools/mamba3_yintra_tile_leg.sh` | 2 |
| `tools/miv_quality.sh` | 2 |
| `tools/packed_estash_l40s_body.sh` | 2 |
| `tools/six_lane_grid_lq.py` | 2 |
| `tools/sym_arms_box.sh` | 2 |
| `tools/test_six_lane_grid.py` | 2 |
| `tools/transformer_fresh_prefill_leg.sh` | 2 |
| `tools/umap_lane_pod_run.sh` | 2 |
| `tools/verify_coverage_audit.py` | 2 |
| `tools/afn_ab.sh` | 1 |
| `tools/apple_speed_cnn/ab.sh` | 1 |
| `tools/apple_speed_neural/attn_trial.sh` | 1 |
| `tools/attention_regs_leg.sh` | 1 |
| `tools/attention_step_leg.sh` | 1 |
| `tools/autograd_smoke.py` | 1 |
| `tools/batchv_quality.sh` | 1 |
| `tools/check_embedding_plan_sort.sh` | 1 |
| `tools/check_forest_resident_layouts.sh` | 1 |
| `tools/classical_pass_ab.py` | 1 |
| `tools/classical_pass_run.py` | 1 |
| `tools/device_mutex_ab_leg.sh` | 1 |
| `tools/e2_remote_leg.sh` | 1 |
| `tools/fast_replication_ab_all.sh` | 1 |
| `tools/feature_freq_identity_leg.sh` | 1 |
| `tools/gbdt_ns_bitvec_check.sh` | 1 |
| `tools/identical_forest_state_fixture.py` | 1 |
| `tools/identical_wave_native_build.py` | 1 |
| `tools/identical_wave_reuse.py` | 1 |
| `tools/identical_wave_revision.py` | 1 |
| `tools/identical_wave_runner.py` | 1 |
| `tools/identical_wave_supplement.py` | 1 |
| `tools/identity_lanes/coverage_sabotage_na.tsv` | 1 |
| `tools/idn_all_checks.py` | 1 |
| `tools/jacobi_sm120a_probe_leg.sh` | 1 |
| `tools/kfeat_pair.py` | 1 |
| `tools/kfeat_quality.py` | 1 |
| `tools/kmeans_linear_resume_chain.sh` | 1 |
| `tools/knn_lean_gpu_pair.py` | 1 |
| `tools/knn_residual_phase_probe.sh` | 1 |
| `tools/knn_selection_gate.py` | 1 |
| `tools/knn_selection_gate.sh` | 1 |
| `tools/knn_selector_kernel_stats.sh` | 1 |
| `tools/knn_selector_probe.py` | 1 |
| `tools/linear_apple3/local2.json` | 1 |
| `tools/linear_apple3/local_compile.json` | 1 |
| `tools/lm_attention_amd_identity_body.sh` | 1 |
| `tools/lm_attention_batch_body.sh` | 1 |
| `tools/lm_ce_alias_body.sh` | 1 |
| `tools/lm_ce_alias_probe.py` | 1 |
| `tools/logical_shard_identity.py` | 1 |
| `tools/long_context_identity_leg.sh` | 1 |
| `tools/lowbit_amd_tuned/cross_build.sh` | 1 |
| `tools/lowbit_amd_tuned/scratch_first.sh` | 1 |
| `tools/mamba_poison_gate.sh` | 1 |
| `tools/mutex_et_knn/et_knn_mutex_leg.sh` | 1 |
| `tools/mutex_et_knn/make_et_knn_body.sh` | 1 |
| `tools/neural_identical_ab.py` | 1 |
| `tools/neural_identical_integration.py` | 1 |
| `tools/nvidia_step_time/attn_arms.sh` | 1 |
| `tools/nvidia_step_time/leg4.sh` | 1 |
| `tools/ops/native_checks_report.py` | 1 |
| `tools/pshap_delta_pair.py` | 1 |
| `tools/pt_score_stable_verified.py` | 1 |
| `tools/s16_job.sh` | 1 |
| `tools/sgd_batch_ab.py` | 1 |
| `tools/sgdoc_tail_pair.py` | 1 |
| `tools/step_glue_leg.sh` | 1 |
| `tools/test_bincache.py` | 1 |
| `tools/test_bincache_defines.py` | 1 |
| `tools/test_byte_lm_session_steps_source.py` | 1 |
| `tools/test_experiment_guards.py` | 1 |
| `tools/test_generation_native_abi.py` | 1 |
| `tools/test_identical_candidate_build.py` | 1 |
| `tools/test_identical_wave_reuse.py` | 1 |
| `tools/test_native_checks_ci.py` | 1 |
| `tools/test_neural_provider_reuse.py` | 1 |
| `tools/test_release_candidate_controller.py` | 1 |
| `tools/test_release_measurement_smoke.py` | 1 |
| `tools/trees_apple_ab.sh` | 1 |
| `tools/w2_clres_quality.py` | 1 |
| `tools/xprep_pool_pair.py` | 1 |
| `x_neighbors/gen.py` | 1 |

## Corrections

- `MOJOLEARN_COLUMN_CPU` was listed DELETE-ARM in every lane that reads it (743 sites): it is the CPU-column build switch
  (`checks/kernel_matrix.mojo` TARGET_COLUMN, `checks/native_tests/runner.mojo`, every host binding, verify's CPU column). It is
  KEEP in every lane; no lane folds it. Corrected before any lane merged.

## Doubts

- Rule a/b/e also delete arms that are real work still owed a measurement, among them `MSEL_UNITS_GPU_PARALLEL` (CV units in parallel; crashes on Metal), `IDN_GEMM_INT_SLICES` (research profile, pending), `IDN_GEMM_STAGE64` (NVIDIA A/B never ran), `TREES_T29` (PairLogit generated objective), `HDB_ONE_SYNC` / `HDB_SELECT_DEVICE` (F15 B/A 0.87-0.95 but the full-data result was inconclusive). Under the 2026-11-15 rule they go and stay recoverable at their sha.
- `python/mojolearn/*` files name some of these defines (env-form switches or route tokens). The tooling lane checks each against verify before editing; a runtime env read of a deleted define is removed with it.
- Branch lines are a lower bound (helpers reachable only from the deleted arm are not counted); classification of rule b (no record) rests on the absence of an EXPERIMENTS.md row, a grid control or a source comment naming the define.

