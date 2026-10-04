# Candidate audit 2026-10-04 (lane/candidate-audit, read-only)

Base: `origin/integration/identical-all-20261004` = ecd80e346, `origin/main` = 6f0da4d55. Main is not an ancestor of
integration. Raw data: `~/mojolearn-evidence/candidate-audit/` (`branches.jsonl`, `branch_categories.json`,
`worktrees.jsonl`, `candidates.tsv` 561 rows with file:line, scan scripts). "Unique" = no-merge commits that are in
neither integration nor main after patch-id (`--cherry-pick`) filtering.

## 1. Unmerged branches and worktrees

There are 1546 heads on origin. 378 are ancestors of neither integration nor main. 262 have patch-unique commits,
and 5 of those are content-absorbed (every changed file equals integration or main).

| Class | Count | Notes |
|---|---|---|
| Apple FAST / Apple (out of scope, listed only) | 141 | `lane/apple-fast-*` (Oct 2-4, many `-r1..-r5` review chains), `board/m3-*`, lowbit-apple |
| Measurement / diagnostic / tooling jobs | 41 | `*-measure`, `stage-diag`, `grid/`, `snapshot/`, `trial/`, `sabotage-no-host-routes` (do not merge) |
| Release / docs / fix tooling (old releases) | 19 | release/0.8.x, alpha-api, fix/release-* (Sep): abandoned |
| Docs/tooling without Mojo | 20 | incl. `lane/lm-attention-fallback` (the dirty default checkout's branch) |
| CPU/host-route work | 13 | neural-pass13/19/77/78/83, host-token-split-2, neighbors-rest, gbdt-rest: CPU routes (undo list) or superseded |
| Speed/correctness code, reviewed below | 23 | see the next table |
| Content-absorbed | 5 | ann-apple-base, apple-identical-steps, gemm-p1-fold-bypass, host-token-split-measure, ordered-speed-prespeed |

IDENTICAL/speed and correctness branches with unique commits. "Patch" means the unique Mojo commits checked against
the integration tree with `git apply --check`.

| Branch (tip) | Uniq | Class | State / evidence | Action |
|---|---|---|---|---|
| **LOCAL ONLY** `/private/tmp/mojolearn-family-assessment` `diagnose/amd-oob` (13db66e48) | 8 unpushed, 5 unique | correctness fix | `xtrees/oob.mojo` exact OOB rounding apart from the mean division (0d3f1af1a, 6187c9a30), AMD OOB capture tool, PTX admission commits. Lives in /private/tmp (wiped on reboot). Bundled to `candidate-audit/diagnose-amd-oob-13db66e48.bundle` | push the branch now; lane R1 |
| `lane/hr2-kpca-seq` (a67c406fd) | 3 | IDENTICAL speed + CPU-out | GARCH and Prophet fits on one warp per series, fold32 order on every column (`MOJOLEARN_SEQ_WARP_FIT=0` A/B). Not in integration (define absent). 1 patch applies, 2 conflict. This is the owed GARCH item; integration reverted its own GARCH_REG/GRID (2504caa9b) | lane T1 |
| `lane/neural-pass57` | 1 | speed (AMD/NV) | fused attention dkdv BJ16 for MI325X/L40S (`MOJOLEARN_ATTN_DKDV_BJ16`), never measured, absent | lane R1 |
| `lane/neural-pass67` | 1 | speed | llama forward: per-layer hard wait after the KV update (`ATTN_CACHE_WAIT`), patch applies, absent | lane R1 |
| `lane/neural-pass48` | 1 | speed | `core/scratch_pool.mojo` process pool for host-read scratch (`SCRATCH_POOL`), absent, conflicts | lane R1 |
| `lane/neural-pass59` | 1 | speed | fused attention forward r2/MFMA shared page; Oct 1 log: TQ16 slower on Apple; NV/AMD not measured | lane R1 (low) |
| `lane/neural-pass5` | 2 | speed | mamba3 backward S17 tail chain (`MAMBA3_S17_TAIL_ARM`), absent, conflicts | lane R1 (low) |
| `lane/linfit-speed` (Sep 29) | 45 | speed | `SGD_IN_FIT_KERNEL`, `X_LINEAR_GLM_TEAM`: absent; 19 of 20 patches conflict | abandoned unless re-derived |
| `lane/algos-ann-speed` (Sep 27, WIP) | 6 | speed | CAGRA chunked queries, t-SNE repulsion tiling: absent, conflicts | abandoned / reference |
| `lane/forest-train-speed` (Sep 17) | 1 | speed | ExtraTrees regression score width 4 on NVIDIA, "NOT YET BUILT" | abandoned |
| `lane/rf-score-weighted-nondeterminism` (Sep 16) | 17 | correctness diag | RF weighted-score atomic probe; superseded by the RF lifetime fix 366a56a0a? unverified | review once |
| `codex/forest-grove-pool` (Sep 14) | 6 | speed | pooled forest prediction driver (H100 qualified Sep): absent | review once |
| `lane/neural-pass36`, `-pass41` | 4, 6 | speed | DROPPED: #40 LU fused 46% slower; #46 QR tiles lost on L40S | none |
| `lane/neural-pass144`, local `lane/merged` (3) | 1, 3 | CPU-out | superseded: cgr-decomp round-robin eigh on every column; SGD host arm already removed | none |
| `lane/neural-apple4`, `claude/tender-franklin-*` | 1 | speed | regs2h AMD S16: merged as PR #50; these are duplicates | none |
| `lane/no-shape-tuning` (Oct 4) | 3 (6 mojo files) | no-bench-tuning | gemm/neighbors checks range rules; all 3 patches apply | hand to running lane no-bench-tuning |
| `lane/kmeans-mojo-1-2`, `trial/mojo-1-2-at-0819` | 7, 4 | toolchain | Mojo 1.2 migration (426 files) | parked |
| `byte-lm-resume-20260906`, `codex/certification-source`, `alpha-api-20260906` | - | old snapshots | Sep 6 | abandoned |

Local worktrees: 359 git checkouts and 20 non-git dirs. 14 hold commits on no origin branch; 10 have tracked or `.mojo`
edits (`.pixi`, leases and logs ignored).

| Worktree | Branch | Issue |
|---|---|---|
| /private/tmp/mojolearn-family-assessment | diagnose/amd-oob | 8 unpushed commits (above) |
| ~/mojolearn-wt/neural-pass139 | lane/neural-pass139 | 2 local-only: SGDOneClassSVM minibatch 4096 mean step, exact k* tie flags (`x_linear/sgd.mojo`, `device.mojo`). Not in integration |
| ~/mojolearn-wt/neural-pass94 | lane/neural-pass94 | 2 local-only: LogisticRegressionCV one block per fold (`x_linear/logcv.mojo`). Integration has `logcv_grid.mojo`; likely superseded |
| ~/mojolearn-wt/neural-pass33 | lane/neural-pass33 | uncommitted `x_decomp/device.mojo` |
| ~/mojolearn-wt/merged-nbmetrics-fix | lane/merged-nbmetrics-fix | uncommitted `core/knn_host_predict.mojo` (host) |
| ~/mojolearn-wt/apple-fast-arima-k3-df-gpu | same | uncommitted `bindings/_mojolearn_arima.mojo` (Apple FAST) |
| ~/mojolearn-wt/idle60-20260930 | lane/merge-tmp | 17,583 dirty paths: an abandoned merge, 2445 commits behind main. Reclaim disk |
| ~/mojolearn-wt/identical-all | integration | 1 unpushed commit f57406e6b (wave-ops, live) |
| ~/mojolearn-wt/no-bench-tuning | lane/no-bench-tuning | 3 unpushed and 2 dirty: running lane |
| others (apple-board-ident, scoped-pca-caller, algos-decomp, lowbit-evidence, metal-embedded-proof x4, nonapple3-integration, review-ann, neural-pass84 host route, mojolearn-apple-experiments) | - | Apple, evidence or old: listed only |

Default checkout `/Users/andrewhendel/CascadeProjects/mojolearn`: `git status` only, 10 dirty entries (untouched).

## 2. Candidate states (561 rows in `candidate-audit/candidates.tsv`; lost and unplanned first)

| State | Rows | Meaning |
|---|---|---|
| LOST | 20 | a report names it, no plan, recipe or inventory has it |
| Unplanned | 34 | define in code, absent from toggle-inventory, recipes and ledger (default ON, reverted by IDN_ALL_OFF) |
| Never written | 150 | 15 CPU items owed from handoff section 4, 131 fam2 "not implemented", 4 pass-1 items |
| Inventory only, no wave builder | 12 | not compiled by any of the 52 builders |
| In recipes only | 66 | opt-in arms; 34 of them were never compiled under any build |
| In wave plan | 194 | default-ON switches covered only by the IDN_ALL_OFF arm (not measured one by one) |
| DROPPED | 6 | 5 block-eigh arms (quarantined), `2661_NONSYM_GROUP_WIDTH` (H100, Sep 21) |
| KEPT | 0 | the ledger has no promotion yet |

LOST (file:line):
- `solver/impl/cd_gram_rule.mojo:28` `CD_IDN_GRAM_WIDE`, a new opt-in arm absent from inventory and recipes.
- `python/mojolearn/_expansion_cnn.py:213` `XCNN_DEVICE_IO_OFF`: an env switch not tied to `IDN_ALL_OFF`, so the
  wave's OFF arm never reverts it. This is an OFF-arm correctness defect.
- `gemm/checks/gemm_identical.mojo`: `GEMM_KPACK_RPT4/CPT4` :7089/:7091, `MFMA_GMIN1/2/16` :5815-5817,
  `MFMA_NO_GROUPS` :3201, `AMD_NO_K768_TUNED` :6454, `AMD_K768_PLAN64` :6456, `LEAF_SPLIT_KPACK_BODY` :6440,
  `SPLIT_CELLS` :3344, `TILE_MIN_K` :3413, `TILE_MIN_BLOCKS` :3472, `TILE_SHORT_K` :3454. These overlap no-bench-tuning.
- `ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo` `IVF_SCAN_GROUPED` :262, `IVF_SCAN_STAGED` :273.
- `python/mojolearn/_expansion_prep.py:233` `XPREP_BLOCKED`; `checks/kernel_matrix.mojo:945` `GBDT_ID_RIDX`;
  `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:162` `GBDT_ID_DEFER_COPY`.

Unplanned (34, default ON). The inventory only lists `MOJOLEARN_IDN_*`, so other prefixes are missing:
- x_linear/device.mojo: `SGD_IDN_CHUNK_WIDE` :742, `MB_FUSE` :756, `DEV_FINITE(_APPLE)` :830-831, `OVR_PAR` :1026,
  `PS_WARP` :2152. Also `x_linear/finite_device.mojo:31`, `x_linear/glm_ydom.mojo:32`.
- glm: `QN_OVR_ONE_UPLOAD` estimator.mojo:702, `QN_DEV_ARGMAX` :907, `QN_IDN_SLIM` glm_base.mojo:1014,
  `QN_IDN_FUSED` :1196, `QN_TILED_MULTI` qn_tiled_rule.mojo:22.
- svm: `SVM_IDN_FUSED_UPDATE_F` smosolver.mojo:163, `NAN_SCAN_ONE` :179, `SVM_IDN_FUSED_TILE` kernel_matrices.mojo:394.
  Also `CD_IDN_GRAM` solver/impl/cd_gram_rule.mojo:19.
- neural: x_cnn/ops.mojo `BN_FOLD_BLOCK` :874, `DROPOUT2D_CH_MASK` :1034; x_cnn/device.mojo:583 `XCNN_ONES_CACHE`;
  sequence/layernorm.mojo:22 `LN_FOLD_BLOCK`; embedding `EMB_RESIDENT`/`EMB_DEVICE_SCRATCH`
  (bindings/_mojolearn_embedding.mojo:147-148), `EMB_AUTO_SORT`/`AUTO_SORT_MIN_CELLS` (checks :189/:191).
- other: decomposition/pca_rr_switch.mojo:21/:45 `PCA_RR_EIGH`/`PCA_RR_FLAG_TEST`; xtrees/agnostic_device.mojo:50
  `AGN_IDN_SYN_POOL`; x_prep/prep3.mojo:33 `PREP3_MAXABS`; arima/impl/fast_order_state.mojo:46 `ARIMA_ORDER_BATCH`.
  `IDN_CHOL_ONE_WAIT` potrf.mojo:1468, `IDN_GP_PRESCALE` kernels.mojo:747, `IDN_GP_GRAD_PRESCALE` kernel_gradient.mojo:86,
  `IDN_GEMM_MFMA_REUSE_WS` gemm_identical.mojo:3181.

In the inventory, but no wave builder compiles them:
- hdbscan/impl/detail/idn_switches.mojo: 6 `IDN_HDB_*` at :26-64.
- `IDN_IVF_DEVICE_SCALE` ivf_flat_build.mojo:104 and `IDN_IVF_DEVICE_SQRT` ivf_flat_search.mojo:204.
- `IDN_TSNE_LANE_FOLD` x_ann/tsne_core.mojo:62 and `IDN_CLS_EPI` x_metrics/cls_epi.mojo:35.

Unmerged-branch candidates not in any plan (section 1): `SEQ_WARP_FIT` (GARCH/Prophet), `ATTN_DKDV_BJ16`,
`ATTN_CACHE_WAIT`, `SCRATCH_POOL`, the OOB exact rounding fix, and SGDOneClass minibatch (neural-pass139).

Never written, by family: cluster 35, kernel-gp 18, forests 11, neural 9, prep-metrics 9, gbdt 9, linear 8,
neighbors 10, decomp 7, timeseries 7, lm 6, shared 6, plus the 15 owed CPU items. Rows are in the TSV.

## 3. Fixes owed

| # | Defect | Evidence | Owner now |
|---|---|---|---|
| F1 | v2 identity attempt 1 rc=1: `ModuleNotFoundError identical_sgd_dependency_proof` (harness staging) | `lq/medium-wave-a4d-qualified-v2/identity-controller.log` (both boxes) | wave-ops (live, attempt2 running on NV, STOPPED on AMD) |
| F2 | `resident_qr_memory` supplement FAIL on both arms: `No module named 'bench_board_probe'` (harness path). After attempt2 the gates stand at ON 23 PASS/1 FAIL/1 PENDING and OFF 20 PASS/1 FAIL/4 N/A on both vendors | `supplements/resident_qr_memory--{on,off}/receipt.json`, `quality-reconciliation-2.json` | wave-ops |
| F3 | DART ON quality pending adjudication | `dart-adjudication/` on AMD | dart-stats |
| F4 | `XCNN_DEVICE_IO_OFF` not under `IDN_ALL_OFF`: the OFF arm stays half ON | `_expansion_cnn.py:213` | unowned, lane N1 |
| F5 | Six report-flagged files are in no wave builder, so they have no compile evidence: hdbscan `predict.mojo`, `soft_clustering.mojo`; ivf_flat `build`, `search`; x_ann `tsne_core`, `tsne_device` | candidates.tsv | plan: wave-ops; code: lanes C1/K1 |
| F6 | 34 opt-in arms never compiled (kmeans/dbscan/affinity chunk, `MINIBATCH_GROUP*`, `NYS_RR_DEV_STOP*`, `QN_IDN_DCONV_POLL_*`, `QN_TILED_MULTI_ALL`, `XENT_FOLD_BLOCK_256`, `HPDEV_CAST_F64`, `X0_PAR_LS*`, `GBDT_APPLY_WIDE`, `EIGH_SMALL_N*`, `MST_SCAN_LANES`) | candidates.tsv | wave-ops step 5 (recipes) |
| F7 | 15 CPU-in-GPU groups NOT_CLOSED: tokenizer encode/gather, GARCH team, ARIMA aic/bic, OCSVM weighted gather_i32, IVF finiteness, GCN self-loops, Bisecting centering, HDBSCAN approximate_predict, silhouette countLabels, trustworthiness staging, shared helpers (cast_elements, reduce_stat, normal_init_f32, CV gathers), decomp Python leftovers (NMF/FastICA/varimax/LLE), neighbor MT19937 sketch, GBDT multiclass leaf solve + symmetric drain merge, host-route baseline false positives | `original-handoff-coverage.json` `original_remaining_cpu_items` | unowned, lanes below |
| F8 | Old handoff section 7 step 2: resident QR holds X twice (`x_decomp/resident.mojo:683-712`, open); small-eigh one-block vs host-route guard (59d161581 partial, verify); "GBDT predict fix" (no source found, owner must name it). Done: IF teardown 6c56a8f0b, DBSCAN context 713a7c09e, KDE check e4ac33dea, `SVD_FULL_MEAN_LAUNCH` gone | git log | lane D1 (QR); GBDT predict: ask |
| F9 | Merge hotspots without a semantic hunk audit: `x_prep/prep3.mojo` MaxAbs FAST+Apple guard, `x_decomp/resident.mojo` GRP_FAST_FUSED guard (:880), `_expansion_prep.py` PowerTransformer anchor, two dual-registration bindings, svd_full mean call, `bindings/_mojolearn.mojo` imports | handoff section 9 | lane D1 (read-only review part) |
| F10 | GARCH REG/GRID reverted (2504caa9b); the device route is `fit_team`, still lead-thread only | fam2-timeseries.md:40-43 | lane T1 |
| F11 | Work at risk: /private/tmp checkout with an unpushed OOB fix; idle60 worktree (17.6k dirty paths) | section 1 | orchestrator: push / reclaim |
| F12 | 4 Metal OFF compiles CANCELLED_USER; Apple verification deferred | `codex-compile-live-summary.json` | deferred (no Apple now) |

## 3b. Skipped to protect old bits (owner rule: only same-version cross-vendor identity matters)

Sources searched: `fam/fam*.md`, `idn-all/`, `no-bench-tuning/inv*.md`, `docs/identical/*`, `docs/apple-fast/EXPERIMENTS.md`.
Excluded: block eigh (fam-decomp:39, later dropped for convergence) and atomic or nondeterminism items. Already
written later, so not lost: QN_TILED C>1 (now `QN_TILED_MULTI`), CNN xent device fold (`XENT_FOLD_BLOCK_256`),
Mamba-3 parallel angle (`IDN_M3_ANGLE_PARALLEL`, default-OFF candidate in recipes), and the XTY tile 64/1024 arms.

| # | File | Idea | Why skipped | Retry now? | Lane |
|---|---|---|---|---|---|
| B1 | `gemm/contract.mojo` + every host oracle (fam2-lm.md:35) | GEMM fold-order arm: wider leaf or a different partition | moves bits of every GEMM caller; "not attempted blind" | YES, the largest NV/AMD lever; after no-bench-tuning lands | new G2 |
| B2 | gemm plan rules (no-bench-tuning SUBAGENT_PROMPT:49) | plan rewrites limited to "bits-safe" / documented bit-equal plans | prompt told the lane to keep bits | YES: fold into B1 | G2 |
| B3 | `solver/impl/cd_gram_rule.mojo:26` `CD_IDN_GRAM_MAX_COLS=64` (inv-linear:12) | widen the Gram CD column cap | Gram vs row sweeps fold differently | YES; `CD_IDN_GRAM_WIDE` (LOST) is this arm; validate n_cols 48/64/65/128 | L1 |
| B4 | `glm/impl/qn/qn_linesearch.mojo` (fam2-linear.md:37, fam2-linear "not impl" 2) | batched QN line search in IDENTICAL (K steps priced in one pass) | z = x.xp + step*x.drt is not a full evaluation's word, so forward arithmetic on 4 columns would change | MAYBE: the gain is 1-2 evals per fit; worth it for small-iteration fits | L1 |
| B5 | `python/mojolearn/_arima_impl.py:397` (fam-timeseries.md:40) | ARIMA aic/bic on the device in float32 | float64 to float32 changes bits | YES: it is also an owed CPU item (F7) | T1 |
| B6 | `xtrees/oob.mojo`, forests epilogue (fam2-forests.md:90) | RF OOB accuracy/r2 epilogue on the device | changes `oob_score_` bits | YES: CPU in the GPU path; pair with the diagnose/amd-oob exact-sum fix | R1 |
| B7 | `_expansion_decomp.py` NMF `_cd_side` (fam2-decomp.md:38) | l2 ridge via an `axpy` with `diag_mask` in float32 | float32 rounding moves bits on all four columns | YES (decomp leftover, F7) | D1 |
| B8 | `x_neighbors/kfeat_dev.mojo` and others, MT19937 tables (fam2-neighbors.md:38) | counter-based device RNG instead of a numpy-compatible stream | keeps sklearn random_state bits | YES (handoff: "old bits do not matter") | K1 |
| B9 | `x_cnn` global average pool forward (fam-neural.md:30, fam2-neural.md:43) | blocked fold over H*W | bits change for a small window | MAYBE: cheap, small gain | N1 |
| B10 | `spectral/impl/sparse/linalg/detail/laplacian.mojo` `degree_kernel` (fam2-cluster.md:274) | lane+tree degree fold (serial chain per row) | bits change, `host_laplacian` must follow; once per fit | MAYBE: a serial chain on a GPU path | C1 |
| B11 | `kernel_methods` `KM_RBF_CELL` for polynomial/sigmoid (fam2-kernel-gp.md:92) | fused cell kernel | changes the dot's fold order on both columns | YES for small d | new KG1 |
| B12 | Nystroem transform fused cross-kernel cell (fam-kernel-gp.md:37) | distance + epilogue in one cell | `svm/` kernel_op form restated, so bits move | YES for small d | KG1 |
| B13 | `gp_variance_kernel` / `gpc_latent_var_kernel` (fam-kernel-gp.md:36, fam2-kernel-gp.md:53) | blocked fold per test point | bits plus host column; helps only tiny n_star | LOW | KG1 |
| B14 | `DevExec.cholesky` in x_decomp (fam-decomp.md:41, fam2-decomp.md:49) | blocked right-looking Cholesky | bits change; no Python caller found | LOW, only if a caller appears | - |

## 4. Proposed code-only fix lanes (disjoint files; none overlaps the running lanes)

Running lanes and the files they own: wave-ops (`tools/identical_wave_*`, `tools/idn_all_checks.py`, `docs/identical/*`,
recipes and plan, harness F1/F2); dart-stats (`tools/identical_quality_noise*`, dart); no-bench-tuning (gemm dispatch,
`gemm/checks/*`, shape rules in any file it touches, now `gbdt/methods/ordered_fast_switches.mojo`, prep glue);
amd-portable (`_backend.py`, `ptx_admission.py`, `gpu_plugins.py`, build flags). The GEMM LOST arms wait for
no-bench-tuning.

| Lane | Files owned | Work |
|---|---|---|
| T1 ts-seq | `sequence/{fit_team,garch,prophet,fold32,exec_device,pyapi}.mojo`, `bindings/_mojolearn_x_sequence.mojo`, `python/mojolearn/_arima_impl.py` | Port hr2-kpca-seq `SEQ_WARP_FIT` (GARCH/Prophet warp per series, fold32 on every column) under IDENTICAL with `_OFF` + IDN_ALL_OFF; ARIMA aic/bic on all-column arithmetic (F7, F10) |
| R1 rescue | `xtrees/oob.mojo` + its test, `transformer/impl/llama/{fused_attention,modeling_llama}.mojo`, `core/scratch_pool.mojo`, `core/device_scan.mojo`, `mamba/impl/modules/mamba3_backward.mojo` + host gen | Land the diagnose/amd-oob OOB fix; re-apply `ATTN_DKDV_BJ16`, `ATTN_CACHE_WAIT`, `SCRATCH_POOL`, S17 tail as default-OFF candidates with recipes |
| C1 cluster-metrics | `x_cluster/*` (bisecting), `hdbscan/impl/detail/{predict,soft_clustering}.mojo`, `metrics/impl/stats/detail/{batched/silhouette_score,trustworthiness_score}.mojo`, `cluster/impl/detail/kmeans.mojo` | Bisecting centering, HDBSCAN approximate_predict, silhouette counts, trustworthiness staging on the device; compile-check the 6 `IDN_HDB_*` files (F5, F7) |
| K1 neighbors-ann | `ivf/impl/neighbors/ivf_flat/*`, `x_neighbors/kfeat_dev.mojo`, `x_ann/tsne_*` | IVF finiteness on the device, MT19937 sketch on the device, recipes for `IVF_SCAN_GROUPED/STAGED`, tsne/ivf builder coverage (F5, F7) |
| S1 shared-helpers | `bindings/hotpath_helpers.mojo`, `bindings/hotpath_device.mojo`, `core/hotpath_device.mojo`, `python/mojolearn/_buffer.py` | `cast_elements`, `reduce_stat` sums, `normal_init_f32`, resident CV fold gathers, OCSVM weighted `gather_i32` kept on the device (F7) |
| N1 lm-neural | `bindings/_mojolearn_tokenizer*.mojo`, `python/mojolearn/{tokenizer,lm_corpus}.py`, `python/mojolearn/_expansion_cnn.py` | Tokenizer encode + `tokens_gather_rows` on the device; GCN self-loops on the device; tie `XCNN_DEVICE_IO_OFF` to IDN_ALL_OFF (F4, F7) |
| G1 gbdt | `gbdt/methods/leaves_estimation/*`, `gbdt/train.mojo`, symmetric drain, `greedy_search_helper_depthwise.mojo`, `checks/kernel_matrix.mojo` | Multiclass one-step leaf solve on the device, symmetric-fit drain merge; recipes for `GBDT_ID_RIDX`/`DEFER_COPY` (F7) |
| D1 decomp | `python/mojolearn/_expansion_decomp.py`, `x_decomp/resident.mojo`, `x_prep/prep3.mojo` (review only), `python/mojolearn/_expansion_prep.py` | 4 Python numeric leftovers on the device; resident QR without a second X; semantic audit of the merge hotspots (F8, F9); recipe for `XPREP_BLOCKED` |
| L1 linear-svm | `x_linear/{sgd,device,logcv}.mojo`, `glm/*`, `svm/impl/*`, `solver/impl/cd_gram_rule.mojo` | Port neural-pass139 SGDOneClass minibatch/k* ties; decide neural-pass94; list the 34 unplanned defines (most are here) for wave-ops to add to inventory/recipes; `CD_IDN_GRAM_WIDE` recipe |

| KG1 kernel-gp | `kernel_methods/*` (KM_RBF_CELL), Nystroem transform, `gaussian_process/*` variance kernels | B11-B13: new fold orders on device and host column together, behind `_OFF` and IDN_ALL_OFF |
| G2 gemm-fold (after no-bench-tuning merges) | `gemm/contract.mojo`, `gemm/*` IDENTICAL plans, `gemm/host`, the contract docs | B1/B2: wider-leaf or partition fold arm with every host oracle following it; then time the 13 LOST GEMM arms |

The bit-skip items from 3b are added to the lanes above: B3/B4 to L1, B5 to T1, B6 to R1, B7 to D1, B8 to K1,
B9 to N1 and B10 to C1. Each changes device and host column together in the new version.

The inventory, recipe and plan edits for the unplanned/LOST rows belong to wave-ops files. Each lane writes its rows
to `~/mojolearn-evidence/candidate-audit/additions-<lane>.tsv`, and wave-ops merges them, so no two lanes edit those
JSON files.
