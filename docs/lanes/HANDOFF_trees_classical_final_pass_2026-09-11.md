# Handoff: trees and classical ML, 2026-09-11 night (0.8.3 shipped, final speed pass wound down)

Written by the orchestrator session bd222cd8 on Andrew's order ("let them
wind down when done... and then create handoff file with next steps"). The
neural session (mojolearn-d1, continuing mojolearn-df) owns transformer/,
gemm/, core/step_phase.mojo and the byte LM, and DEVIATIONS 2640 to 2659.
This file covers trees and classical only. Read ENGINEERING_RULES.md
sections 9 and 10 and bench/OPPONENT_REFERENCE.md before touching a lane.

## 1. State at wind-down

- Nothing of this session is renting. RunPod lists only the
  `samba-train-*` pod, which belongs to another program (never touch it).
  No Hot Aisle VM and no DigitalOcean droplet of this session exists; every
  box below was deleted and verified gone (HTTP 404).
- Six speed lanes ran on RunPod NVIDIA H100 (one pod each) for about 30
  minutes before the wind-down. None rented anything else.
- Every measurement in this file is our IDENTICAL arm against the
  opponent's FAST arm on the same pod, unless a row says otherwise. Ratios
  are ours divided by the opponent's time (below 1 means ours took less).

## 2. What landed on main

| item | main commit | result |
|---|---|---|
| OLS conditioning, DEVIATIONS 2620, 2621 (tall) and 2622 (wide) | df77d6c1, 26683dba | power-of-two Gram equilibration and a relative eigenvalue cutoff; Istella-S R2 -115.6 to 0.332; `glm/ols_main.mojo` 17/17 on M4, H100 and MI300X |
| NVIDIA SVC above 512 rows, DEVIATION 2623 | f7a10cd9 | kernel-matrix row gives NVIDIA above width 512 the halving-tree schedule; `svm/svc_main.mojo` 44/44 on H100, M4, MI300X; fits at n=400/600/2000 hash 457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0 on all three |
| pointwise searcher identity, DEVIATION 2624 | 36ca51fd | document-block multiplier 1 in the ordered tiers; new check passes on H100, MI300X, M4; pointwise and greedy hashes on five 1M synthetic fixtures equal H100 vs MI300X; the opt-in pointwise arm is 1.6x to 2.2x slower |
| 0.8.3 release merged | a824d9da | see section 3 |
| SVM block-solve schedule row, DEVIATIONS 2627, 2628 | 3671a1e5 | opt-in schedules, default unchanged; see section 5 |

## 3. Release 0.8.3 (published 2026-09-11)

- Contents: 2620 to 2623 only, on top of v0.8.2. Branch release-0.8.3,
  wheel commit f8b65ee2, published-docs commit 0768c753, tags
  alpha-api-0.8.3-20260911 and v0.8.3.
- PyPI: Linux x86-64 (sm_89, sm_90a, gfx942) 20:17Z, run 34643281339,
  sha256 c8c2975fdb70fd51...; macOS arm64 20:35Z, run 34643372856, sha256
  60573c840b4b7b01....
- Finish line: `pip install mojolearn==0.8.3` from PyPI on the Apple M4
  imports 0.8.3 and gives the three SVC hashes above.
- Installed Linux qualification was partial. On gfx942 (DigitalOcean) and
  sm_90a (RunPod H100) every IDENTICAL job passed and all 29 smoke lane
  hashes were equal across the two. Every FAST and DETERMINISTIC job failed
  in `run_installed.py` asking `_backend.binding('_mojolearn')` in a tier
  that DEVIATION 2490 removed. The fix to the job loop (cc117fdf) is on main
  but was not on the release branch, because `tools/linux_surface_qualification.sh`
  is in the native inventory the build proofs bind. sm_89 was never
  qualified installed (no L40S, RTX 4090 or L4 stock on RunPod). An
  installed-wheel SVC check on a RunPod H100 gave the three hashes above.
- Evidence (copied out of the scratchpad):
  `~/mojolearn-evidence/release-0.8.3-2026-09-11/` and
  `~/mojolearn-evidence/release-0.8.2-2026-09-11/`.

## 4. Same-pod H100 baselines from the final pass (source 36ca51fd)

### Gradient boosting (1M rows, 100 trees, depth 6, 5 interleaved rounds)

| policy | dataset | opponent | opponent ms | ours ms | ratio | ours logloss / AUC | opponent logloss / AUC |
|---|---|---|---|---|---|---|---|
| symmetric | taxi | CatBoost GPU | 690.9 | 349.5 | 0.51x | 0.525735 / 0.619460 | 0.525904 / 0.618511 |
| depthwise | taxi | CatBoost GPU | 852.6 | 473.1 | 0.55x | 0.525086 / 0.621421 | 0.525047 / 0.621092 |
| depthwise | taxi | XGBoost GPU | 366.7 | 473.1 | 1.29x | same | 0.525227 / 0.620101 |
| lossguide | taxi | CatBoost GPU | 1151.7 | 1007.6 | 0.87x | 0.525504 / 0.619386 | 0.525110 / 0.620541 |
| lossguide | taxi | XGBoost GPU | 497.2 | 1007.6 | 2.03x | same | 0.525227 / 0.620101 |
| symmetric | Istella-S | CatBoost GPU | 1526.9 | 1373.6 | 0.90x | 0.138653 / 0.966990 | 0.138982 / 0.966853 |
| depthwise | Istella-S | CatBoost GPU | 1676.0 | 1926.9 | 1.15x | 0.126517 / 0.971896 | 0.125819 / 0.972797 |
| depthwise | Istella-S | XGBoost GPU | 1693.8 | 1926.9 | 1.14x | same | 0.125110 / 0.973774 |
| lossguide | Istella-S | CatBoost GPU | 2424.8 | 2428.0 | 1.00x | 0.122045 / 0.975037 | 0.121363 / 0.975309 |
| lossguide | Istella-S | XGBoost GPU | 1883.4 | 2428.0 | 1.29x | same | 0.125110 / 0.973774 |

Model hashes held one value per cell in 5 of 5 rounds (taxi symmetric
90c3558501933f47, depthwise 40c1683b9e0eb151, lossguide 0dd8bcfc3c3a4a1d;
Istella-S 238d3abce0cabf43, 5d053cd086658072, 6182fd2bee4fb941). LightGBM's
pip wheel has no CUDA learner, so it has no row.

### Forests (1M rows, 3 interleaved rounds)

| family | dataset | opponent and device | opponent ms | ours ms | ratio | logloss ours / opponent |
|---|---|---|---|---|---|---|
| RandomForest | taxi | cuML 26.08, GPU | 1980 | 877 | 0.44x | 0.525910 / 0.525800 |
| RandomForest | Istella-S | cuML 26.08, GPU | 3668 | 2113 | 0.58x | 0.145560 / 0.145504 |
| ExtraTrees | taxi | scikit-learn 1.9.1, CPU, 24-core pod quota | 4075 | 2058 | 0.51x | 0.527541 / 0.527011 |
| ExtraTrees | Istella-S | scikit-learn 1.9.1, CPU, 24-core pod quota | 16259 | 6073 | 0.37x | 0.188191 / 0.187901 |
| IsolationForest | taxi | cuML 26.08, GPU | 63 | not measured | owed | - |
| IsolationForest | Istella-S | cuML 26.08, GPU | 1526 | not measured | owed | - |

cuML 26.08 does ship IsolationForest (the Aug 28 85 ms row was it, against
our FAST arm). Our iforest arm was refused by a harness bug, fixed on
lane/forest-speed 759e1aac but not re-run. RF and ET hashes equal the AMD
rows (RF taxi d8f64dae01de00bd, Istella-S 574b24d0d7af51d0; ET taxi
e683f121d11f59dd, Istella-S 40b1c5b03ba40420); identity_break 45/45.

### k-NN and UMAP (cuML 26.8.0)

| workload | cuML ms | ours ms | ratio | quality |
|---|---|---|---|---|
| kNN taxi, 400k index x 4k queries, d11, k10 | 8.19 | 24.79 | 3.03x | recall@10 ours 0.99915, cuML 0.99925 |
| kNN Istella-S, same shape, d220, k10 | 51.11 | 124.71 | 2.44x | recall@10 ours 0.923025, cuML 0.92205 |
| kNN synthetic 400k x 4k x d32, k10 | 10.05 | 23.66 | 2.36x | same neighbors in the same order as cuML |
| kNN synthetic, k15 | 10.22 | 26.21 | 2.57x | same |
| UMAP taxi, 100k rows | 5297 | 3564 | NOT QUOTED | trustworthiness ours 0.906, cuML 0.966; our rounds bimodal |

### Linear models and clustering (taxi 4,000,000 x 11, 5 interleaved rounds)

| family | cuML ms | ours ms (main) | ratio | quality |
|---|---|---|---|---|
| LinearRegression | 23.12 | 263.5 | 11.40x | R2 0.908836 cuML, 0.908837 ours |
| PCA | 20.99 | 28.69 | 1.37x | EVR sum 0.99786 / 0.997861 |
| KMeans | 128.2 | 304.0 | 2.37x | inertia 1.20192e8 cuML, 1.20628e8 ours |

cuML KMeans gave a different centroid digest every round; ours held one.
Istella-S was not measured (download 980 s).

### SVC and KDE (earlier same-pod races, still current on main)

| family | dataset | cuML ms | ours ms | ratio |
|---|---|---|---|---|
| SVC, 10k training rows | taxi | 415 | 857 | 2.06x |
| SVC, 10k training rows | Istella-S | 20.64 | 70.7 | 3.43x |
| KDE, 100k x 2k queries | taxi (d11) | 2.37 | 39.2 | 16.55x |
| KDE, 100k x 2k queries | Istella-S (d220) | 6.79 | 219 | 32.25x |

## 5. Lane branches (pushed, NOT merged unless stated)

| lane | branch tip | what is on it | why not merged, and what merging needs |
|---|---|---|---|
| svm-speed | 48f92b19 | 61feb052 MERGED (3671a1e5); 48f92b19 R-ary 32-way thread-carrying tree, UNBUILT | build and time it; see 6.5 |
| kde-speed | 21bb9a91 | DEVIATION 2625 tiled IDENTICAL KDE (euclidean, l1, chebyshev), same bits (gate `check_kde_tiled_equals_staged` 33,300 scores, 0 differ; `kde_check` PASS on H100), ON BY DEFAULT in the branch | the race against cuML never ran on it; device-only synthetic taxi shape 32.3 to 18.9 ms, Istella-S shape 124 to 118 ms; needs 6.2 |
| linear-cluster-speed | 904733ff | DEVIATIONS 2632 (OLS host means, centering, uninitialized centered buffer) and 2633 (k-means scale pass), same bits (OLS fb86358654367fa0, k-means 89520efe99a08d5f); harness fixes (cuML k-means tol 0 refused by cuVS, new `ours-base` arm) | taxi only: OLS 264.7 to 177.8 ms (0.67), k-means 304.6 to 216.2 ms (0.71); Istella-S race and M4/AMD gates owed; needs 6.1 |
| gbdt-speed | 55967003 | DEVIATION 2634 (skip CTR target prep without categorical columns, built, never timed) and 2635 (binary-search border candidates, NEVER COMPILED), both default on | needs 6.3 |
| forest-speed | e3bce76a | DEVIATIONS 2637 (RF/ET row-major staging straight into pinned memory) and 2638 (iforest lends X by address), UNBUILT; 759e1aac harness fix; results | needs 6.4 |
| knn-speed | 63740f29 | DEVIATION 2629 exact distance chain without per-step flush, opt-in, same bits, flat (synthetic k10 23.65 to 23.85 ms, Istella-S 124.71 to 119.49 ms inside 6 to 10% noise); `tools/umap_two_datasets.py`; the committed default-off path was never compiled | little to merge; take the harness and rows if wanted, see 6.6 |

The lanes' own OPPONENT_REFERENCE sections live on their branches; merge
them with the code, or copy the section 4 tables here into
bench/OPPONENT_REFERENCE.md if the code is dropped.

## 6. Next steps, in the order I would take them

Rules for all of them: runs on RunPod NVIDIA H100 through `tools/trees_leg.sh`
(one pod per lane, watchdog baked in, reap and see 404); subagents never
run anything on the Mac; the flip rule of section 9 (geomean of after/before
over taxi and Istella-S below 1, quality not worse); same bits proven on the
H100, then the orchestrator gates the Apple M4 and an AMD box (Hot Aisle
first) before a merge.

1. **Finish linear-cluster-speed (smallest step to a merged win).** On one
   H100, race `ours,ours-base,cuml-gpu` on Istella-S for LinearRegression,
   PCA and KMeans (commands in bench/results/linear_cluster_speed_2026-09-11/README
   on the branch), compute the geomean by hand (`tools/flip_verdict.py` reads
   another log format). On the M4: `pixi run check-kmeans`,
   `tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo`,
   `test_native_helpers.py`, `helpers_ident.py` against 36ca51fd. Then
   merge. On Istella-S the 220-column device Jacobi is expected to dominate
   OLS and PCA; fewer synchronization points per rotation keep bits.
2. **KDE (largest ratio).** Port the SIMD accumulator from
   `kde/checks/kde_dist_attribution.mojo` into `kde_tiled_logk_kernel`
   (it cut the tiled log-kernel stage 108 to 30.6 ms with the same hash),
   stage the caller's memory without List copies (about 95 ms of host time
   on Istella-S), run `kde_check` on H100, M4 and AMD, then the race
   `MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_LANES=kde MOJOLEARN_CTD_DATASETS=taxi,istella MOJOLEARN_CTD_ROUNDS=5 MOJOLEARN_CTD_EXTRA_ARMS=ours-before`
   (the `ours-before` arm is new and untested).
3. **GBDT host setup and lossguide.** Build 2634 and 2635, then
   `sh tools/trees_identical_ab.sh build a2634 gbdt`, `ib baseline`,
   `ib a2634`, `diff baseline a2634`, `pixi run check-greedylogsum`,
   `pixi run check-binarization`, `python3 checks/gbdt_sub_byte_identity_check.py`,
   and `bench/results/gbdt_speed_2026-09-11/body_ab2634.sh` into
   `tools/flip_verdict.py`. Then batch the Istella-S compressed index uploads
   (about 100 ms per fit) and remove one of lossguide's two host drains per
   grow iteration (`greedy_search_helper_depthwise.mojo:2305-2308`), which
   is the 2.03x against XGBoost on taxi.
4. **Forests.** Build c4056486, `sh tools/trees_identical_ab.sh ib rowmajor rf-clf,rf-reg,et-clf,et-reg,iforest`
   and `diff baseline rowmajor`, `pixi run check-if`, then speed cells for
   RF, ET and our first iforest baseline on both datasets. ExtraTrees on
   Istella-S (about 6 s) holds most of the remaining forest time.
5. **SVC.** Build and time the R-ary tree (`bench/results/svm_speed_2026-09-11/variants.sh`
   with rary32, rary16, rary32nt, fused, tree). Any flip goes through the
   NVIDIA row only (Metal refuses FUSED_TREE at width 1024, threadgroup
   memory 36872 > 32768). Then the roughly 36 ms of Istella-S work outside
   the solver (41 MB kernel tile allocation and poison fill, X list copy,
   host finite check). cuML runs as many inner iterations as we do, so the
   remaining gap is per-iteration cost.
6. **k-NN.** Widen the NVIDIA query tile from 512 to 2048 or 4096 (256 to
   512 saved 5.8%, tiling cannot move bits) and shrink unused radix scratch;
   then mirror cuML's fused distance plus top-k (cuVS
   `knn_brute_force.cuh:447-451`), which never writes a distance matrix.
   Our time today is distance 15.3 ms and selection 7 to 10 ms across about
   160 launches.
7. **UMAP.** Find why our trustworthiness trails cuML's (0.906 against
   0.966 on taxi 100k) before quoting any UMAP time; the Istella-S race is
   owed (`python3 tools/umap_two_datasets.py race --dataset istella --rows 100000 ...`
   on lane/knn-speed).

Also owed, not speed:

- Pointwise speed win-back after 2624: per-block scratch slots summed in a
  fixed order with a vendor-independent multiplier, keeping 2624's hashes;
  Apple M4 pointwise model hashes were never compared.
- Before the next Linux release: confirm the installed FAST and
  DETERMINISTIC smoke passes on main (`run_installed.py` asks
  `_backend.binding('_mojolearn')` in every tier), and qualify sm_89
  installed.
- DBSCAN and HDBSCAN at 1M rows against cuML (never measured); scikit-learn
  KDE Istella-S row on AMD.
- DEVIATION 2624 collision: `origin/lane/cpu-speed` (byte LM CPU kernels,
  another session) also uses 2624 and must renumber before it merges; the
  neural session was told.

## 7. DEVIATION numbers

Trees and classical used 2620 to 2629 and 2631 to 2638 (2626, 2631, 2636
unused so far; 2639 held for the orchestrator). 2630 and 2640 to 2659 are
neural's. Take the next trees or classical number from 2660 up after
checking `git grep -n "DEVIATION 26[6-9]"` on main and the open branches.

## 8. Evidence and worktrees

- Lanes: `~/mojolearn-evidence/{kde,svm,knn,linear-cluster,gbdt,forest}-speed-2026-09-11/`;
  small copies under `bench/results/*_2026-09-11/` on each branch.
- Earlier tonight: `~/mojolearn-evidence/svc-cuda-1024-2026-09-11`,
  `pointwise-hash-drift-2026-09-11`, `ptw2624-amd-mi300x-hotaisle`,
  `verify083-amd-mi300x-hotaisle`, `trees-h100-pointwise-ab-2026-09-11`,
  `classical-h100-kde-svc-istella-2026-09-11`.
- The lane and release worktrees live in the orchestrator's scratchpad
  under /private/tmp and will not survive a reboot; every branch is pushed
  and the release evidence is copied, so they can be removed.
