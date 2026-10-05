# Apple FAST lost candidates (2026-10-04)

This audit only read code and docs. Nothing was compiled, tested, timed or queued.

## Scope and method

- origin/main `ac8a753af`. Branches scanned after `git fetch origin`: 294 origin branches matching `lane/apple-fast*`, `agent/*apple*` and `lane/*fast*`. Of these, 123 have commits outside main and 104 of those change `.mojo` or `bindings/` files.
- For every branch, the scan recorded its head, the number of commits outside main, and the `MOJOLEARN_*` defines it adds (`git diff origin/main...BR`). Build and harness noise was dropped: `BENCH_INSTALLED`, `NUMERIC_MODE`, `VENDOR`, `COMPILE_JOBS`, `MOJO_BUILD_FLAGS`, `BUILD_LOCK_HELD`, `COLUMN_*` and prefixes.
- A define counts as on main when the define, or its `_OFF` form, appears anywhere in the origin/main tree.
- Outcomes come from three places:
  - `docs/apple-fast/EXPERIMENTS.md`, its verdict column;
  - `CONSOLIDATED_CANDIDATE_QUEUE_2026-10-04.md`, `CANDIDATE_RECOVERY_2026-10-04.md`, `MANAGER_REVIEW_2026-10-04.md` and `docs/apple-fast/ab/*`;
  - the M3 queue and results snapshot `~/mojolearn-evidence/apple-fast/branch-audit-20261004/queue-snapshot.json`. Its `results.txt` carries `defines=` and `median_ms` for each tag.
- Most snapshot results hold the **B arm only** (`AFC-DEF-SUMMARY arm=B`). Where the doc gives no baseline, "B vs board" compares that B against the FAST-after value in `BOARD_M3_FAST.md`. The two numbers come from different heads, so treat this as a **lead, not a verdict**.
- Raw tables are in `~/mojolearn-evidence/apple-fast-lost-candidates/`:
  - `branches.tsv`: every branch;
  - `unmerged.tsv`: branches with new defines;
  - `alldefs.tsv` and `master.tsv`: one row per define with its docs, queue and results;
  - `open_rows.tsv`: every OPEN or HELD EXPERIMENTS row cross-checked against the M3 results;
  - the scripts that produced them.

Status key:
- `merged-with-outcome`: on main, and a verdict is recorded.
- `merged-no-outcome`: on main, but EXPERIMENTS still says OPEN or gives no verdict.
- `unmerged-unmeasured`: never timed. This includes runs that failed to compile on M3.
- `unmerged-measured-loser`: a run shows no gain or a quality HOLD.
- `unmerged-measured-unrecorded`: a B-arm timing or quality receipt exists, but no doc records it.
- `superseded`: main already beats the branch's B arm, or a later define replaced it.

## Lanes where FAST is slower than the best opponent (BOARD_M3_FAST.md, ratio after > 1)

sgd 18.7, layernorm 18.6, adagrad 13.8, adam 11.7, adamw 10.9, rmsprop 7.7, adamax 7.4, nadam 6.2, autoarima taxi-hourly 4.83 and synthetic 3.22, lr-exponential 2.93, lstm-reg/clf 2.6-2.7, cholesky 2.37, label-binarizer taxi 2.29, **lle taxi 2.06**, **kernel-shap istella 1.99**, **min-cov-det and elliptic-envelope istella 1.98**, var 1.8-1.9, moe 1.93, sparse-rp taxi 1.90, additive-chi2 taxi 1.69 and istella 1.16, **lu-factor 1.53**, **lu-solve 1.45**, **randomized-svd istella 1.45 and taxi 1.22**, target-encoder taxi 1.38, gaussian-rp taxi 1.31, multilabel-binarizer taxi 1.30, **resample taxi 1.21 and istella 1.16**, rbf-sampler istella 1.20, **permutation-shap istella 1.20**, minibatch-kmeans istella 1.17, adafactor 1.15, pca istella 1.06, gbdt-depthwise taxi 1.04.

## Ranked: unmerged candidates on slow lanes with no recorded outcome

Ranked by likely gain on a lane with ratio > 1.

| # | define | branch @ head | lane (ratio) | status | evidence | what it needs next |
|---|---|---|---|---|---|---|
| 1 | `LLE_FAST_DEV_LU` | lane/apple-fast-w4-decomp-harness-r1 @ e9d72edb5 (code from w4-decomp @ 34b4f6c72) | lle taxi (2.06) | unmerged-measured-unrecorded | results `w2-w4d-lle-q` W4-PAIR **PASS**; `w2-w4d-lle-taxi-r1` B = 1039.0 ms vs board FAST 2570 ms (lead -60%; would give ratio about 0.83). Main's EXPERIMENTS has no row; the OPEN row exists only on the branch (fca401be0). | M3 A/B timing with an A arm from the same build (one run per arm); then merge. The arm also carries `RSVD_FAST_DIRECT_IN`, so split the defines or keep them together. |
| 2 | `RSVD_FAST_DIRECT_IN` | same branch @ e9d72edb5 | randomized-svd istella (1.45), taxi (1.22) | unmerged-measured-unrecorded | `w2-w4d-rsvd-q` **PASS**; B istella 501.3 ms vs board 533 (lead -6%); B taxi 198.5 vs 199 (flat) | M3 A/B with its own arm; small gain, so it moves only if the A/B confirms. |
| 3 | `SHAP_KERNEL_DEV` | lane/apple-fast-shap @ 13343dd51 | kernel-shap istella (1.99) | unmerged-unmeasured | EXPERIMENTS L121 OPEN; M3 `shap-kernel-dev` failed to parse (rc=1); never timed | Rebase onto current main (Oct 2 base), then M2 build, M3 quality, M3 timing. `KSHAP_FAST_BATCH` is already on main, so check overlap first. |
| 4 | `SHAP_PERM_CACHE` | lane/apple-fast-shap @ 13343dd51 | permutation-shap istella (1.20) | unmerged-unmeasured | L122 OPEN; `shap-perm-cache` failed to parse; never timed | Rebase, M2 build, M3 quality and timing. `PSHAP_DELTA` and `SHAP_FAST_PIPE` landed since: check overlap. |
| 5 | `RESAMPLE_FAST_IDX_BULK` | lane/apple-fast-resample @ 50b96e795 | resample taxi (1.21) | unmerged-unmeasured | L289 OPEN; `resample-rs-idxbulk-taxi` failed to parse | Rebase (Oct 2 base), M2 build, M3 quality and timing. Do this together with the device `RESAMPLE_FAST_GATHER` recovery that CANDIDATE_RECOVERY already lists. |
| 6 | `MCD_ORDERED_COV` | lane/apple-fast-mcd-ordered-quality-r3 @ f35a57bd4 | min-cov-det and elliptic-envelope istella (1.98) | unmerged-measured-loser (quality), unrecorded | results `mcd-ordered-direct-q-r3` rc=1 "quality command failed or HOLD" (compare.log on M3); no doc row | Record the HOLD in EXPERIMENTS. Retry only with a fix for the compare failure. The bigger MCD lever is `MCD_SKIP_PINVH` (below). |
| 7 | `EIGH_FAST_PANEL_DF` | lane/apple-fast-eigh-panel-df @ 21eeacf90 | eigh synthetic (HOLD-quality vs opponent) | unmerged-measured-loser (quality), unrecorded | `eigh-panel-df-q-v1` rc=1 "HOLD: strict no-regression failed"; no doc row | Record the HOLD in EXPERIMENTS. Eigh accuracy against numpy is the open board issue, so this is still the right target with a different compensation. |
| 8 | `CV_FAST_SLICE`, `CV_FAST_TRUST_FOLDS`, `RESAMPLE_FAST_ONE_FOLD`, `RESAMPLE_FAST_RANK_SORT`, `RESAMPLE_FAST_PERM_SELECT` | lane/apple-fast-resample @ 50b96e795 | resample family (cross-val-score, bootstrap and permutation-test already beat the opponent, ratio < 0.5) | unmerged-unmeasured | L267/268/290-292 OPEN; all failed to parse on M3 | Low priority: these lanes already win. Rebase together with #5. |
| 9 | `DECOMP_SDK_NNNT` | lane/apple-fast-decomp-sdk-control @ 73bb9ac6e | pca istella (1.06) and decomp GEMM | unmerged-unmeasured, WIP | WIP checkpoint "SDK decomp buffer adapter for handoff"; no doc or queue entry | Finish it or record it as abandoned. The scoped G1/G5 GEMM timings (CANDIDATE_RECOVERY) cover the same pca lane. |
| 10 | `SVD_FAST_AW` | lane/apple-fast-gap-linalg2-svd @ 35f7b95c4 | svd (istella 0.72 and taxi 0.98: already winning) | unmerged-unmeasured | commit message says "PARKED, uncompiled; target dropped"; no doc row | Record it as parked in EXPERIMENTS. Nothing more to run. |

### Unmerged, B-arm measured on M3, but the EXPERIMENTS row still says OPEN (lanes already win)

These were run as `*-x` re-tags on `lane/apple-fast-batch@3150d75c1` after the prebuilt-arm batch, but no verdict was written back. The lanes already beat their opponent, so they rank below the table above. Before any merge, each one needs an A arm from the same build plus a quality check.

| define(s) | branch @ head | lane | B on M3 (ms) | board FAST (ms) | lead |
|---|---|---|---|---|---|
| `FA_EIG_SMALL + FA_ITER_DEVICE (+ FA_LL_DEVICE)` | lane/apple-fast-fa @ 3efbce2af | factor-analysis taxi / istella | taxi 39.5 / 30.2; istella 5884 / 5879 | 308 / 10351 | large on taxi; about -43% on istella |
| `FA_ITER_DEVICE`, `FA_LIVEBUF` | lane/apple-fast-fa @ 3efbce2af | factor-analysis istella / taxi | istella 1253 / 1258; taxi 168.7 | 10351 / 308 | about -88% on istella: largest classical lead found |
| `FA_GRAM_ONCE`, `FA_TRANSFORM_FUSED` | lane/apple-fast-fa @ 3efbce2af | factor-analysis | gram: taxi 277, istella 1262; transform: istella 6201, taxi 243 | 308 / 10351 | gram -88% istella |
| `IVF_COARSE_RANDOM_INIT`, `IVF_DEVICE_VALIDATE`, `IVF_KMEANS_LAZY_SHIFT`, `IVF_REFINE_TEAM`, `PQ_LUT_TILED`, `PQ_SCAN_FUSED` | lane/apple-fast-vsearch @ 86925aef9 (lane/apple-fast-vsv @ 5640672ca) | ivf, ivf-pq, ivf-refine | ivf-pq istella about 5170-6080 | ivf-pq istella 1676 | **superseded**: main is about 3x faster |
| `DECOMP_FAST_DICT_UPDATE`, `DECOMP_FAST_LASSO_BLOCK`, `DECOMP_FAST_OMP_BLOCK` | lane/apple-fast-decomp-sparse @ 5fb1740cd | dict-learning, mb-dict-learning, sparse-pca, sparse-coder | mbdl-upd istella 5902; lasso-block mbdl 3243; spca-upd taxi 1526 | mb-dict-learning istella 3185 | upd: superseded; lasso about flat; needs an A arm |
| `DECOMP_FAST_GEMM_TILED`, `QR_FAST_DEV`, `SVD_FAST_CHOLQR`, `CHOL_FAST_BLOCKED`, `FA_FAST_QRR`, `LU_FAST_PIVOT_GRID` | lane/apple-fast-decomp-linalg @ 74d52352b | rsvd, qr, svd, cholesky, fa, lu | rsvd istella 683; svd istella 15620; chol 476; lu 1233 | 533; 1781; 261; 654 | **superseded**: main is faster on every one |
| `SGDOC_FAST_PAR` | lane/apple-fast-sgdoc-parallel @ 61710a5c8 | sgd-ocsvm taxi / istella | 16.5 / 181 | 154 / 407 | EXPERIMENTS: **HOLD** (anomaly behavior, flagged 0). merged-with-outcome as HOLD; the flagged-rate question decides it |
| `HUBER_DEVICE_LBFGS (+ HUBER_FAST_BLOCK512)` | lane/apple-fast-robust @ cfdb95e48 | huber taxi | 225 / 196 | 215 | about -9%; EXPERIMENTS L181/182 OPEN "not merged yet" |
| `SHAP_TREE_TAB` | lane/apple-fast-shap @ 13343dd51 | tree-shap istella | 21.4 | 25.1 | about -15%; tree-shap already wins (ratio 0.17) |
| `ISOTONIC_FAST_PAIRMERGE + _PAR` | lane/apple-fast-isotonic-knn @ 7385fcfdd | isotonic istella | 91.4 (M3) | 29.4 | **superseded** |
| `PREP3_SPLINE` | lane/apple-fast-prep3 @ ec65873e3 | spline istella | 11.8 | 11.6 | noise; record DROPPED-noise |
| `LSVR_DEVICE_CONVERGE`, `LSVR_DUAL_CD` | lane/apple-fast-linsvr @ c649076a4 | linearsvr | taxi 26.2 / 57.1; istella 218 | 27 / 217 | noise or slower; record DROPPED |
| `BPE_*` (train-device, merge-batch, livebuf, group-filter) | lane/apple-fast-bpe @ 3355d37b3 | bpe-train enwik8 | 224-253 | not on board | needs an A arm; neural |
| `PREP2_FAST_EIGH_BLOCK` | lane/apple-fast-prep2 @ 8762eb33f | iterative-imputer taxi | 185.1 | 218 | about -15%; lane already wins (0.21) |

### Unmerged and never timed: symmetric GBDT pack (prebuilt arms never ran)

The rows "A/B queued (lane/apple-fast-batch prebuilt arms)" never reached the M3 queue snapshot. None of these tags appears in `queue.txt` or `results.txt`. Their lanes already win (gbdt-symmetric 0.27-0.40, multiclass 0.24-0.33, pairlogit 0.37, yetirank 0.47, categorical 0.46), so they rank below the slow lanes.

| branch @ head | defines | EXPERIMENTS rows |
|---|---|---|
| lane/apple-fast-sym-hist @ 3bb4db314 | `SYM_GATHER_FUSED`, `SYM_HIST_MULT`, `SYM_PART_STATS_PAR`, `SYM_RESOLVE_BLOCK`, `SYM_SCAN_SUB_FUSED`, `SYM_SORT_SWAP`, `SYM_HIST_ALL` | L130-140 OPEN |
| lane/apple-fast-sym-feat @ bca0e3a48 | `GBDT_BOOT_DEVICE`, `GBDT_EVAL_FUSED`, `GBDT_EVAL_SKIP_EMPTY`, `GBDT_INDEX_PACK_DEVICE`, `GBDT_PREDICT_PACKED`, `GBDT_QUANT_DEVICE`, `SYM_FEAT_ALL` | L104-109, 129 OPEN |
| lane/apple-fast-sym-iter @ 4956a2234 | `SYM_BUF_ARENA`, `SYM_DERIV_FUSED`, `SYM_LEAF_FROM_STATS`, `SYM_REUSE_PARTITION`, `SYM_ITER_ALL` | L124, 127, 133-134, 138 OPEN |
| lane/apple-fast-sym-est @ c8518eb52 | `EST_ITERS_DEVICE` (no doc row), `EST_REUSE_PART`, `EST_SHRINK_FUSED`, `EST_STATS_FUSED`, `SYM_EST_ALL` | L101-103, 128 OPEN |
| lane/apple-fast-sym-multi @ d2c832da0 | `MC_CLASS_BATCH_DERIV`, `MC_CLASS_BATCH_EST`, `PL_GROUP_NARROW`, `PL_PAIRS_ONCE`, `YR_TASK_FUSED`, `SYM_MULTI_ALL` | L110-120, 135, 141 OPEN |
| lane/apple-fast-sym-ctr @ 39c3c9daf | `CTR_INDEX_FUSED`, `CTR_ONEHOT_DEVICE`, `CTR_PREP_SHARED`, `CTR_SORT_ONCE`, `SYM_CTR_PERM_BATCH`, `SYM_CTR_ALL` | L97-100, 125-126 OPEN |

Next for each branch: rebase onto main (Oct 3 base), M2 build, M3 quality (AUC and NDCG equal), then M3 timing. Run the `*_ALL` umbrella first, one run per arm.

### Other unmerged, never timed

| define | branch @ head | lane | status / evidence | next |
|---|---|---|---|---|
| `DBSCAN_FAST_DENSEBALL` | lane/apple-fast-dbscantaxi @ 1febff7df | dbscan taxi / istella | L380 OPEN; `dbscantaxi-ab-x` and `-ist-x` failed | Fix the M3 failure, then M3 quality and timing. dbscan taxi has no board FAST row because it times out. |
| `CC_FAST` | lane/apple-fast-graph @ 1fa36a7ec | connected-components taxi (0.68) | L378 OPEN; failed to parse | Low: the lane already wins. Rebase or drop. |
| `CHOL_FAST_DEVIO`, `CHOL_FAST_NOSYNC`, `LU_FAST_STEP1` | lane/apple-fast-gap-linalg2 @ 799a99f1f | cholesky, lu | EXPERIMENTS L323 KEPT (`CHOL_FAST_DEVIO`); LU_STEP1 DEFAULT PASS | The branch's last 4 commits sit outside main, but the defines are on main through `_OFF`: superseded. |
| `MCD_DEVICE_CSTEPS_OFF`, `ANN3_COARSE_SEED`, `PREP3_MAXABS` | lane/apple-fast-m2b1 @ 83d456722 | various | KEEP rows recorded | superseded |
| `SCHI2_FAST_DEVRNG` | lane/apple-fast-gap-kapprox2 @ ff139585f | skewed-chi2 | LEDGER "WIN istella -85%"; `XN_FAST_SCHI2_MOJO_MT` became the default instead | superseded |
| `PCA_FAST_EIG`, `PCA_FAST_TOPK`, `PCA_FAST_NO_ALIAS` | lane/apple-fast-pca-eig @ 9819970b1 | pca | DROP recorded | unmerged-measured-loser |
| `OPTICS_*` | lane/apple-fast-optics2 / opv | optics | DROP recorded | unmerged-measured-loser |
| `IF_QUERY_RAW`, `IF_SAMPLED_UPLOAD`; `SEG_SCAN_BLOCK`, `REORDER_FLAGS_SCAN_BLOCK`; `SYM_DEVICE_*` | lane/apple-fast-trees-io / -scan / -symmetric | trees | DROP recorded | unmerged-measured-loser |
| `LLE_FAST_KNN`, `LLE_SPARSE_EIG`, `XN_FAST_IMPUTE_TILED2` | lane/apple-fast-lle / -isotonic-knn | lle, knn-imputer | DROP or slower recorded | unmerged-measured-loser |

## Merged or recorded candidates on slow lanes (not lost, listed for completeness)

| define | branch @ head | lane (ratio) | status | evidence | next |
|---|---|---|---|---|---|
| `MCD_SKIP_PINVH` (+ `MCD_DEFLATE`) | lane/apple-fast-w4-mcd @ d0b30bfe9 | min-cov-det / EE istella (1.98) | merged-with-outcome: **HOLD** | EXPERIMENTS: MCD 86854 -> 31891 ms, EE 86637 -> 31781; masks unchanged, final covariance differs; DEFLATE flags Jaccard 0.9989 | **Biggest single lever on a slow lane** (would flip 1.98 to about 0.73). Needs a quality decision: is a covariance difference with an unchanged mask acceptable? No new code needed. |
| `LU_FAST_TSLU` | lane/apple-fast-w4-tslu @ 5867b9fbe | lu-factor (1.53) / lu-solve (1.45) | merged-with-outcome: DROP-quality | factor 849 -> 589, solve 842 -> 585; hard-matrix residuals worse | A compensated panel is needed before a retry. |
| `CHOL_FAST_NB512`, `CHOL_FAST_TRI_SYRK` | lane/apple-fast-w4-linalg @ 5867b9fbe; lane/apple-fast-chol-20261004 @ 76edfb514 | cholesky (2.37) | HOLD-speed / DROPPED-slower | 260 -> 266; 269 -> 288 | none |
| `LU_FAST_MMA_DBUF`, `LU_FAST_PIVOT_SHUFFLE`, `DECOMP_FAST_MMA_K16` | lane/apple-fast-lu-mma-dbuf, -lu-pivot-shuffle, -decomp-mma-k16 | lu, decomp GEMM | recorded in EXPERIMENTS and the consolidated queue | see rows | none |
| `ARIMA_SLAB`, `ARIMA_FIT_GROUPS`, `ARIMA_FAST_SCALAR_LL` (K3) | lane/apple-fast-w3-arima @ 40fedfcea; -w2-ts @ 596d0abbb; -arima-scalar-k3-r2 @ 6f09497ab (+ k3-df-reference @ 9723a7383, k3-tail-probe @ 102e0d70a) | autoarima (4.83 / 3.22) | HOLD (noise, -2%) / recorded / HOLD (kernel quality) | EXPERIMENTS rows | The K3 df-reference and tail-probe branches are later quality-harness commits for the HOLD; the parallel-time scan design is on the arima-scan-* branches (docs only). |
| `X_PREP_PINNED_OUT` | lane/apple-fast-w3-prep @ 92e4661e6 | label-binarizer (2.29), multilabel (1.30), target-encoder (1.38) | DROP (cost moves to the caller's first read) | EXPERIMENTS | none |
| `TARGET_SCRATCH` | lane/apple-fast-target-current @ 4d1ea20b2 | target-encoder | promoted (main 12fdd6697) | EXPERIMENTS | none |
| `GBDT_DW_FLAT_GRID`, `GBDT_DW_BRIDGE_SCAN`, `GBDT_QH_FAST_FUSED_Q`, `GBDT_DW_FAST_DEV_SCALE` | lane/apple-fast-w3-dw, -dwcurrent, -gap-misc | gbdt-depthwise taxi (1.04) | DROP-speed / DROPPED-noise | EXPERIMENTS | none |
| `GBDT_FAST_DEPTHWISE_HOST_PARTITION` | lane/gbdt-depthwise-fast @ cc652d70b (Sep 20) | gbdt-depthwise taxi (1.04) | superseded: the define is on main; 1 old commit outside main | none | none |
| `SEQ_FAST_VAR_FUSED` | lane/apple-fast-w4-small @ 74d233862 | var (1.93 / 1.82) | PARKED (compile error, one-block launch) | EXPERIMENTS | Needs a parallel rewrite; that is the only VAR lead. |
| `RESAMPLE_FAST_ROW_GATHER` | lane/apple-fast-w4-small @ 74d233862 | resample (1.21 / 1.16) | held, recorded (CANDIDATE_RECOVERY) | B taxi 53.3 vs board 62.8 | see CANDIDATE_RECOVERY |
| `KSHAP_FAST_BATCH`, `PSHAP_DELTA`, `SHAP_FAST_PIPE` | gap-kapprox2-kshap, -w4-shap @ 20bbdc372, -w2-shap @ abd933572 | kernel-shap (1.99), permutation-shap (1.20) | KEPT / on board / recorded | board kernel-shap 15325 = KSHAP_BATCH; pshap 14789 = PSHAP_DELTA B | none |
| `APPLE_FAST_SHARED_GEMM_G1 / _G5` (+ `_AUDIT`, `_COUNTERS`) | lane/apple-fast-shared-scoped-timing @ 0849df9ac, -g5 @ 768632c27 (+ downstream and gemm-g1g5 series) | pca istella (1.06) | timing submitted (CANDIDATE_RECOVERY, queue position after 1571) | no results in the snapshot yet | Wait for the M3 results. |
| `MOE_FAST_MMA`, `_KB32`, `_PF`, `_WIDE` | lane/apple-fast-gap-misc @ 5e2eec7a3 | moe (1.93) | merged-no-outcome (EXPERIMENTS L540 "MERGED-UNMEASURED opt-in (neural; Andrew Oct 3)") | no A/B run | M3 A/B timing on moe synthetic, one run per arm. |
| 84 `AFN_*` neural opt-ins (`AFN_OPT_VEC4`, `AFN_OPT_MULTITENSOR`, `AFN_OPT_RESIDENT_STATE`, `AFN_OPT_FUSE_SCAN`, `AFN_OPT_CLIP_FUSE`, `AF_FAST_NOFILL`, `AF_FAST_RESIDENT`, ...) | lane/apple-fast-neural @ 600237d7c, lane/apple-fast-gap-optim @ cf4513f8a | optimizers (sgd 18.7 ... adafactor 1.15), layernorm, lstm | merged-no-outcome (MERGED-UNMEASURED, Andrew Oct 3) | EXPERIMENTS L467-531 | Check whether the `AFN_OPT_*` switches reach the board optimizer lanes (adam / sgd synthetic); those are the worst board ratios. |

## Merged without an outcome (stale OPEN rows; the board already shows the gain)

The branch is fully on main (0 commits ahead) and the board FAST value equals the recorded B arm, but EXPERIMENTS still says OPEN. These rows need a verdict written, not a run.

| define | branch | evidence |
|---|---|---|
| `ARD_FAST_CLS1_*`, `BAYES_FAST_CLS1_*`, `RIDGE_FAST_CLS1_CODES`, `NC_FAST_CLS1_LABELS` | lane/apple-fast-gap-cls1 | ard taxi B 8.3 = board 8.3; bayesian-ridge 15.3 = 15.3; ridge-clf 19.0 = 19; nearest-centroid 20.9 = 20.9 (L173-180, 193, 245) |
| `CAGRA_FAST_IVFG + EXACTD + SEEDS + ITERS` (and the other CAGRA rows) | lane/apple-fast-gap-cagra | cagra istella B 1254.3 = board 1254 (L214-224) |
| `PURITY2_1` | lane/apple-fast-purity2 | gmm istella B 5859 = board 5858 (L592) |
| `KPCA_RESIDENT` | lane/apple-fast-w4-decomp / -kapprox | kernel-pca istella 201 and taxi 127.5 = board (L452) |
| `HDB_DEV_BORUVKA`, `HDB_LINKAGE_DEVICE`, `HDB_SELECT_DEVICE` | lane/apple-fast-hdbscan2 (via lane/apple-fast-batch) | hdbscan taxi B 432.6 vs board 434 (L384-387) |
| `XD_FAST_CLS2_GRP_*`, `X_CLUSTER_FAST_CLS2_MBK_*`, `PREP_FAST_CLS2_MINMAX_*`, `X_PREP_FAST_CLS2_*`, `XN_FAST_CLS2_OCSVM_*` | lane/apple-fast-gap-cls2 (0 ahead) | rows L281, 294, 350-351, 395, 453 say "gapcls2-* never queued", and no tag appears in queue or results. **merged-no-outcome and unmeasured, on slow lanes gaussian-rp taxi (1.31) and minibatch-kmeans istella (1.17)**: M3 A/B owed. |
| `ANN_FAST_KNN_BIGD`, `CAGRA_FAST_TEAM`, `IVFPQ_FAST_DEVICE_CODEBOOKS`, `IVF_FAST_DEVICE_CSR`, `IVF_FAST_DEVICE_TRAINSET`, `IVF_FAST_SCAN_SELECT`, `TSNE_FAST_SPLIT` | lane/apple-fast-ann (0 ahead) | all M3 runs rc=1 (ivf_pq_device.mojo build error at that head); never timed; lanes already win |
| cluster2 (`AFFINITY_FAST_LOOP`, `AP_*`, `BGMM_*`, `BISECT_FAST_RESIDENT`, `OPTICS_FAST_DEVICE_ORDER`), `KERNEL_FAST_NYS_RR_EIGH`, `NB_CAT_ATOMIC`, `MI_*`, `X_PREP_FAST_II_*` / `QSELECT` / `TE_*`, `DBSCAN_FAST_SCAN` / `_CC_BATCH`, `SEQ_GARCH_HOST_MAX` | lane/apple-fast-cluster2, -kernel, -nb, -mi, -prep2, -core, -seq | failed to parse, ENSURE-SO failure, or timeout on M3; never timed; every lane already wins. `X_PREP_FAST_TE_*` touches target-encoder taxi (1.38): worth one M3 A/B. |

## Top 10 actions (by expected gain on slow lanes)

1. `MCD_SKIP_PINVH` (on main, HOLD): make the quality decision. 86.3 s -> about 31.9 s would flip MCD and EE istella from 1.98 to about 0.73.
2. `LLE_FAST_DEV_LU` (lane/apple-fast-w4-decomp-harness-r1 @ e9d72edb5): quality PASS, B 1039 vs board 2570 on lle taxi (2.06). Needs an M3 A arm, a merge, and an EXPERIMENTS row.
3. `RSVD_FAST_DIRECT_IN` (same branch): quality PASS; lead -6% on randomized-svd istella (1.45). Needs its own M3 A/B.
4. gap-cls2 `XD_FAST_CLS2_GRP_*` and `X_CLUSTER_FAST_CLS2_MBK_*` (on main, never measured): gaussian-rp taxi (1.31), minibatch-kmeans istella (1.17). Needs M3 A/B.
5. `SHAP_KERNEL_DEV` (lane/apple-fast-shap @ 13343dd51): kernel-shap (1.99); never compiled on M3. Rebase, M2 build, M3 quality and timing.
6. `SHAP_PERM_CACHE` (same branch): permutation-shap (1.20). Same steps.
7. `RESAMPLE_FAST_IDX_BULK` (lane/apple-fast-resample @ 50b96e795): resample taxi (1.21); never compiled. Same steps, together with the device gather.
8. `MOE_FAST_MMA*` (on main, unmeasured): moe (1.93). Needs M3 A/B.
9. `AFN_OPT_*` and `AF_FAST_*` (on main, unmeasured): check that they reach the board optimizer lanes (ratios 6-19), then run M3 A/B.
10. `X_PREP_FAST_TE_ENC` and `TE_GLOBAL` (lane/apple-fast-prep2, both arms failed with ENSURE-SO): target-encoder taxi (1.38). Needs an M2 build fix, then M3 A/B.

Record-only follow-ups (no runs):
- write HOLD rows for `MCD_ORDERED_COV` and `EIGH_FAST_PANEL_DF`;
- write PARKED rows for `SVD_FAST_AW` and `DECOMP_SDK_NNNT`;
- write verdicts for the stale OPEN rows (gap-cls1, gap-cagra, purity2, KPCA_RESIDENT, hdbscan2);
- mark the decomp-linalg, vsearch and isotonic B arms superseded.
