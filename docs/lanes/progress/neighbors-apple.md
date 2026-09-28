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
| daa913db1 / 3aa68f5a2 | back-substitution ring kernel (x in threadgroup memory) -- MEASURED 3.6x SLOWER, reverted | IDENTICAL | digests equal |
| fd048ed6a | Jacobi eigh: FAST on Apple launches 256 wide like IDENTICAL (scheduling) | FAST | FAST words unchanged |
| 593ce0826 | back substitution, second form: x in threadgroup memory (masked ring of the newest values + staged older ones), 32-step register prefetch (`-D MOJOLEARN_CHOL_BACK_RING_OFF`) | IDENTICAL (and FAST where the pinned solve runs) | same chain; digests equal |
| 312b5da1a | Nystroem / RBFSampler transform into caller memory through a pinned 64 MB staging buffer (`-D MOJOLEARN_KM_DIRECT_OUT`) | both | same words |
| efd02268a, 9ec71f7a4 | MOJOLEARN_STAGE_TIMES=1 walls for GPC fit, KRR solve, RBFSampler transform (timing only) | - | - |
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

Cholesky / GP / KRR (M3 Ultra, arms in one job, digests equal in every pair):

| algorithm | mode | shape | before | after | change |
|---|---|---|---|---|---|
| KernelRidge.fit | IDENTICAL | taxi / HIGGS 10k rbf | 1.979 / 1.528 | 1.913 / 1.466 | deferred Cholesky info (0979a063c) |
| GaussianProcessRegressor.fit | IDENTICAL | taxi / HIGGS 3k | 0.375 / 0.313 | 0.351 / 0.292 | deferred info |
| GaussianProcessClassifier.fit | IDENTICAL | taxi / HIGGS 3k | 2.118 / 1.477 | 1.981 / 1.357 | deferred info |
| GaussianProcessClassifier.predict_proba | IDENTICAL | taxi / HIGGS 3k x 3k | 0.357 / 0.362 | 0.218 / 0.216 | 8-RHS sweep (59e15eef3) |
| GaussianProcessClassifier.predict_proba | FAST | taxi / HIGGS 3k x 3k | 1.509 / 1.514 | 0.182 / 0.184 | 8-RHS sweep (FAST's serial column solve before) |
| KernelRidge.fit | IDENTICAL | taxi / HIGGS 10k rbf | 1.913 / 1.466 | 1.045 / 0.601 | back-substitution ring v2 (593ce0826); cho_solve 1,257 -> 390 ms |
| GaussianProcessRegressor.fit | IDENTICAL | taxi / HIGGS 3k | 0.348 / 0.290 | 0.269 / 0.206 | ring v2 |
| GaussianProcessClassifier.fit | IDENTICAL | taxi / HIGGS 3k | 1.985 / 1.382 | 1.523 / 0.981 | ring v2 (solve 732 -> 269 ms over 6 Newton steps) |
| Nystroem.fit | FAST | taxi / HIGGS 4k, 300 comps | 3.063 / 2.301 | 0.834 / 0.630 | Jacobi 256-wide (fd048ed6a); not an in-job A/B: two jobs on the M3 Ultra |

Profiles (MOJOLEARN_STAGE_TIMES=1, M3 Ultra): KernelRidge 10k potrf 640 ms
(taxi) / 191 ms (HIGGS; taxi's tiny kernel values fail the matrix unit's
exponent admission and are recomputed on the rounded chain), cho_solve 390 ms
after ring v2. GPC fit taxi, 6 Newton steps: factor 990 ms, solve 270 ms,
B matrix (host) 123 ms, matvec 67 ms. RBFSampler 1M x 500 transform: gemm
70 ms, epilogue 39 ms, copy into the caller's array 905 ms (fixed by
312b5da1a, measurement pending).

HOME MAC (m4pro-b, M4 Pro), request 1790584533292 at d022f6d6a, arms in one
job (forward and reverse), digests equal in every pair. Raw:
~/mojolearn-evidence/neighbors-apple/job3_m4prob.txt

| algorithm | shape | old tiled (before) | tiled + k=10 selector + warp guard | certified (after) |
|---|---|---|---|---|
| NearestNeighbors.kneighbors | taxi 200k x 10k, k 10 | 0.497 | 0.432 | 0.100 |
| NearestNeighbors.kneighbors | HIGGS 200k x 10k, k 10 | 0.491 | 0.435 | 0.088 |
| KNeighborsClassifier.predict | taxi / HIGGS | 0.504 / 0.497 | 0.440 / 0.440 | 0.112 / 0.097 |
| KNeighborsRegressor.predict | taxi / HIGGS | 0.497 / 0.491 | 0.435 / 0.436 | 0.106 / 0.092 |
| kneighbors k=20 | taxi / HIGGS | 1.085 / 1.087 | 1.121 / 1.084 | 0.234 / 0.188 |
| kneighbors, tied grid data | taxi / HIGGS 100k x 5k | 0.129 / 0.129 | 0.115 / 0.115 | 0.141 / 0.100 (most queries fall back: +25 ms on taxi) |
| SVC.fit (SMO sync + EPT arm) | taxi / HIGGS 10k | 2.831 / 0.287 | | 0.888 / 0.131 |
| SVR.fit (same arm) | taxi / HIGGS 10k | 0.313 / 0.344 | | 0.121 / 0.153 |
