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
| 0e481b572 | Cholesky panel factor + panel solve on Apple: diagonal block in threadgroup memory, rows in registers (`-D MOJOLEARN_CHOL_PANEL_STAGE_OFF`, `-D MOJOLEARN_CHOL_PANEL_SOLVE_STAGE_OFF`) | both on Apple | same chains |
| 0dcb77d80 | transform output copy split over the host cores | both | same words |
| efd02268a, 9ec71f7a4 | MOJOLEARN_STAGE_TIMES=1 walls for GPC fit, KRR solve, RBFSampler transform (timing only) | - | - |
| 0e481b572 -> be86ece38 | Cholesky panel staging: measured neutral, REVERTED (no code left on the branch) | - | - |
| 395bc882a | GPC fit: Laplace Newton loop keeps K, B and the factor on the device, B by `gpc_b_matrix_kernel` (`-D MOJOLEARN_GPC_HOST_NEWTON` keeps the host round trips) | IDENTICAL and FAST GPC fit | same arithmetic by construction; digests equal |
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

RBFSampler.transform 1M x 500 (M3 Ultra, arms in one job): copy into the
caller's array 907 -> 745 ms with pinned staging (312b5da1a), transform
1.130 -> 0.964 s; the remaining copy is the first touch of the caller's
fresh 2 GB. With the copy split over the host cores (0dcb77d80, arms in one
job, M3 Ultra, digests equal): RBFSampler.transform 1M x 500 1.132 -> 0.380 s
(taxi) / 1.135 -> 0.379 s (HIGGS); Nystroem.transform 100k x 300 0.077 ->
0.033 s / 0.076 -> 0.032 s. Panel factor/solve staging (0e481b572): neutral
(KRR potrf 648 -> 640 ms, GPC factor 1,000 -> 995 ms over 6 steps).

## FAST quality (paired, bench/x_neighbors_fast_quality.py)

M3 Ultra, request 1790588783350 at eb2a28caf (includes every FAST change of
the branch: the SVR fold-order fix, the 8-RHS sweep, Jacobi 256 wide). Five
seeded row samples x two datasets, both tiers fitted on the same rows and
scored on the same held-out rows; FAST minus IDENTICAL (accuracy, R^2,
recall@10, mean log density). Raw: ~/mojolearn-evidence/neighbors-apple/quality_m3ultra.txt

| case | taxi mean / worst | HIGGS mean / worst |
|---|---|---|
| SVC acc | +0.000000 / +0.000000 | +0.000000 / +0.000000 |
| SVR R^2 | +0.000002 / -0.000017 | -0.000014 / -0.000048 |
| KernelRidge R^2 | 0 / 0 | 0 / -0.000001 |
| GPR R^2 | 0 / 0 | -0.000001 / -0.000002 |
| GPC acc | 0 / 0 | -0.000200 / -0.001000 (1 of 1,000 held-out rows, one seed) |
| KNeighborsClassifier acc | +0.000100 / 0 | 0 / 0 |
| KNeighborsRegressor R^2 | -0.000001 / -0.000003 | 0 / 0 |
| NearestNeighbors recall@10 | 0 / 0 | 0 / 0 |
| KernelDensity mean log density | 0 / 0 | 0 / 0 |

FAST quality holds against the reference arithmetic on every case.

## FINAL (wind-down, 2026-09-28)

The lane is closed. Branch tip holds no half-done work: the working tree was
clean, no `refs/wip/*neighbors-apple*` snapshot exists on origin, and the last
code commit (395bc882a) was already built and A/B-measured by the previous
agent before the usage limit (results recorded below). Nothing new was
submitted in the wind-down.

### Results read in the wind-down (m4pro-b, arms in one job, forward and reverse)

Raw: `~/mojolearn-evidence/neighbors-apple/<request>.txt`.

| request | commit | arm | case | default | arm (old path) | digests |
|---|---|---|---|---|---|---|
| 1790594682975 | 395bc882a | `MOJOLEARN_GPC_HOST_NEWTON` | IDENTICAL GPC.fit taxi / HIGGS 3k | 0.825 / 0.456 s | 1.352 / 0.889 s | equal (fc039349.. / 0de2f247..) |
| 1790594566171 | be86ece38 | `MOJOLEARN_KM_DIRECT_OUT` | IDENTICAL RBFSampler.transform 1M x 500 taxi / HIGGS | 0.545 / 0.537 s | 1.061 / 1.057 s | equal |
| 1790594566171 | be86ece38 | `MOJOLEARN_KM_DIRECT_OUT` | IDENTICAL Nystroem.transform 100k x 300 taxi / HIGGS | 0.060 / 0.051 s | 0.090 / 0.073 s | equal |
| 1790588737156 | e1e90ed59 | all old arms (MULTI_RHS_OFF, KM_DIRECT_OUT, JACOBI_FAST_NARROW, FAMILY_CTX_PER_CALL) | FAST Nystroem.fit taxi / HIGGS 4k | 0.711 / 0.537 s | 2.532 / 1.901 s | equal |
| 1790588737156 | e1e90ed59 | same | FAST GPC.predict_proba taxi / HIGGS 3k | 0.239 / 0.234 s | 1.286 / 1.285 s | equal |

Host-Newton GPC profile (taxi, 6 steps): B matrix 105 ms, factor 883 ms,
matvec 59 ms, solve 230 ms; the device loop removes the B build, the L
round trips and the K re-uploads.

### Default vs opt-in

Default ON (the speed paths): certified simdgroup-matrix k-NN (5a56ce5c3),
k=10/15 selector + warp guard (d022f6d6a), SMO sync removal in IDENTICAL
(9765af975) with the fold-order fix (c81d6a3b2), process-lifetime family
context (b2ab8d372), deferred Cholesky info (0979a063c), 8-RHS trsm sweep
(59e15eef3), back-substitution ring v2 (593ce0826), pinned staging for kernel
transforms (312b5da1a), multi-core output copy (0dcb77d80), Jacobi 256 wide in
FAST (fd048ed6a), GPC device Newton loop (395bc882a).

Opt-out defines (A/B arms, each restores the old path):
`MOJOLEARN_KNN_CERTIFIED_MMA_OFF`, `MOJOLEARN_FAMILY_CTX_PER_CALL`,
`MOJOLEARN_SVM_IDENTICAL_SYNCS_OFF`, `MOJOLEARN_CHOL_DEFER_INFO_OFF`,
`MOJOLEARN_CHOL_MULTI_RHS_OFF`, `MOJOLEARN_CHOL_BACK_RING_OFF`,
`MOJOLEARN_KM_DIRECT_OUT`, `MOJOLEARN_JACOBI_FAST_NARROW`,
`MOJOLEARN_GPC_HOST_NEWTON`. Opt-in arm only:
`MOJOLEARN_EXPERIMENTAL_KNN_WARPBOUND_GUARD`.

### Unproven: needs the integration identity check

Every change above has speed evidence with digests equal between arms on one
Mac, but NO identity-lane check passed on this branch (request
1790581787983 was withdrawn by Andrew on 2026-09-28 and is still listed
PENDING in the steward; it only covered fc211b48e). The integration lane
must run the identity lanes (svc*, svr*, x-neighbors*, kernel-ridge*,
nystroem*, rbf-sampler, gp*, gpc*, knn / nearest-neighbors / radius / KDE)
against these default-changing commits:

- 9765af975 SMO IDENTICAL syncs on Apple
- b2ab8d372 process-lifetime DeviceContext
- c81d6a3b2 SMO fold_order_rank_kernel (index, position) ranking
- d022f6d6a k-NN k=10/15 selector + warp guard
- 5a56ce5c3 certified simdgroup-matrix k-NN
- 0979a063c deferred Cholesky info
- 59e15eef3 trsm 8-RHS sweep
- fd048ed6a Jacobi FAST 256 wide
- 593ce0826 back-substitution ring v2
- 312b5da1a pinned staging transform output
- 0dcb77d80 multi-core transform copy
- 395bc882a GPC device Newton loop (newest; built and timed only on m4pro-b)
- 38c9834dd build_gp.sh smoke fix (build only)

### Known issues

- SpectralEmbedding with a knn affinity fails at 20k rows in neighbors
  `select_radix` (k > 1024); found by the decomp-apple lane. NOT fixed here;
  the integration lane should reproduce it on the merged tree and fix or
  refuse it by name.
- Certified k-NN on heavily tied data (the tied grid case) sends most
  queries to the tiled fallback: +25 ms on taxi on the M4 Pro versus the old
  tiled path (0.141 vs 0.115 s); correct, slower on that shape only.
- Older steward FAILs for `neighbors-b09e62fd87` / `neighbors-8cecb1f247`
  (x-neighbors-svc-sigmoid, svm-weights, gamma-scale, svc-multiclass gpu
  arm on m3ultra-b / m2pro) predate this lane's final state and were not
  re-examined; the integration check supersedes them.
