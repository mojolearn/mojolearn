# neighbors-apple: progress

Apple (Metal) speed for the neighbors family: k-NN, radius neighbors, KDE,
SVC / SVR, KernelRidge, GP regressor / classifier, Nystroem, RBFSampler.
Brief: ~/mojolearn-evidence/apple_speed_brief.md. Branch lane/neighbors-apple,
merged later by the gate runners. Home Mac for speed jobs: m4pro-b.

Board: `bench/x_neighbors_speed.py` (taxi and HIGGS, first eight columns
standardized; each case one load run, then the minimum of REPS timed runs of
fit and of predict; a digest of the outputs; a quality number for FAST).
Arms: `bench/x_neighbors_ab.sh` (build-define arms, forward then reverse
order, default rebuilt at the end).

## Changes on the branch

| commit | change | modes | bits |
|---|---|---|---|
| 9765af975 | SMO outer loop: device fold order, rescan-free scatter, lagged NaN read (FAST_SMO_SYNCS) now also IDENTICAL on Apple; fused update_f stays FAST | IDENTICAL | same by construction |
| b2ab8d372 | svm, kernel_methods, gaussian_process estimators: one process-lifetime DeviceContext per binding and tier | both | same by construction |
| 8f71669af | kNN A/B arm define `MOJOLEARN_EXPERIMENTAL_KNN_WARPBOUND_GUARD` | arm only | - |
| 30fbae151 | merge origin/lane/merged (board renamed `bench/x_neighbors_apple_speed.py`; the GPU-speed lane owns `bench/x_neighbors_speed.py`) | - | - |
| c81d6a3b2 | SMO `fold_order_rank_kernel` ranks by (index, position) like the host `fold_order_for` (SVR's projected working-set indices repeat; FAST SVR had the same defect) | both | IDENTICAL: restores the host order; FAST: fixes a slot collision |
| 5a56ce5c3 | Apple IDENTICAL k-NN: simdgroup-matrix candidates + pinned-chain rescoring + per-query certificate, tiled arm for uncertified queries (`neighbors/impl/detail/certified_mma_knn.mojo`); `-D MOJOLEARN_KNN_CERTIFIED_MMA_OFF`; per-call context arm `-D MOJOLEARN_FAMILY_CTX_PER_CALL` | IDENTICAL | same keys by construction; digests equal |
| d022f6d6a | kNN IDENTICAL on Apple: compile-time k=10/15 selector + warp-bound guard | IDENTICAL | digests equal |
| 0979a063c | Cholesky, Apple left-looking IDENTICAL: `info` read once after the loop, guarded panel kernels (`-D MOJOLEARN_CHOL_DEFER_INFO_OFF`) | IDENTICAL | same words by construction |
| 59e15eef3 | `trsm_lower` on Apple: sweep with 8 right-hand sides per block (`-D MOJOLEARN_CHOL_MULTI_RHS_OFF`) | both | same chains |
| 38c9834dd | `build_gp.sh` smoke: return_cov is honored now (the stale refusal assert failed every FAST GP build on Apple after the merge) | FAST build | - |

## Speed requests

- m4pro-b 1790580105714 (ca87ddca4, base) and 1790581015947 (fc211b48e:
  SMO syncs + context; kNN define arms). Raw: ~/mojolearn-evidence/neighbors-apple/{baseline_ca87,job2_fc21}.txt
- m3ultra 1790584752251 (0979a063c, merged tree; A/B arms in one job).
  Raw: ~/mojolearn-evidence/neighbors-apple/job3_m3ultra.txt

m4pro-b was 20 deep at 08:40Z, so the A/B arms (both arms in ONE job,
forward then reverse order) ran on m3ultra; every arm pair below is one Mac,
one job.

## Identity requests (withdrawn: Andrew 2026-09-28, no verification in the lane)

- 1790581787983-neighbors-fc211b48ea: 29 lanes (svc*, svr*, x-neighbors svm
  lanes, kernel-ridge*, nystroem*, rbf-sampler, gp*, gpc*), sabotage
  `~/mojolearn-evidence/neighbors-apple/device_outputs_sabotage.patch`
  (device entry points only: SVM b + 1e-3, kernel_methods / GP first
  uploaded and first downloaded float moved).

## Before -> after

IDENTICAL, fit / predict seconds, digests equal between the arms of each row.

| algorithm | Mac | shape | before | after | digest |
|---|---|---|---|---|---|
| NearestNeighbors.kneighbors | M4 Pro | taxi 200k x 10k, k 10 | 0.493 (ca87) | 0.366 (k=10 selector arm) | 9ff7546966de542d |
| NearestNeighbors.kneighbors | M3 Ultra | taxi 200k x 10k, k 10 | 0.185 (old tiled) / 0.162 (tiled + selector) | 0.037 (certified) | 9ff7546966de542d |
| NearestNeighbors.kneighbors | M3 Ultra | HIGGS 200k x 10k, k 10 | 0.179 / 0.161 | 0.028 | a29e4118b681bb53 |
| KNeighborsClassifier.predict | M3 Ultra | taxi / HIGGS | 0.194 / 0.187 | 0.046 / 0.040 | 4c13da7c.. / 3c0e1faa.. |
| KNeighborsRegressor.predict | M3 Ultra | taxi / HIGGS | 0.186 / 0.180 | 0.037 / 0.029 | dc704b68.. / 9f94067d.. |
| kneighbors k=20 | M3 Ultra | taxi / HIGGS | 0.285 / 0.284 | 0.054 / 0.044 | 34a8923f.. / f8814dc5.. |
| kneighbors, tied grid data | M3 Ultra | taxi / HIGGS 100k x 5k | 0.053 / 0.053 | 0.052 / 0.037 (88% / fewer queries fall back) | 52412ee0.. / 575ce1bc.. |
| SVC.fit | M4 Pro | taxi 10k rbf | 1.022 (ca87) | 0.906 (SMO syncs) | 788a3d6c392d6e37 |
| SVC.fit | M4 Pro | HIGGS 10k rbf | 0.159 | 0.136 | 24d6f4ea808d3a21 |
| SVR.fit | M4 Pro | taxi 10k rbf | 0.138 | 0.122 | 0ce89d3cc2d003ea |
| SVC.fit (arm: all Apple SMO sync removal + EPT off) | M3 Ultra | taxi / HIGGS | 2.98 / 0.289 | 0.745 / 0.099 | equal |
| SVR.fit (same arm) | M3 Ultra | taxi / HIGGS | 0.318 / 0.338 | 0.095 / 0.110 | equal (HIGGS SVR 879a30f0.. = ca87 base after c81d6a3b2) |
| process-lifetime context (KRR, Nystroem, RBFSampler, GPR, GPC) | M3 Ultra | | | no measurable change | equal |

Certified k-NN fallback counts (MOJOLEARN_STAGE_TIMES=1): taxi 185 of 10,000
queries (taxi repeats rows), HIGGS 0, k=20 taxi 26, tied grid data 4,389 of
5,000 (answered by the tiled arm, as designed).
