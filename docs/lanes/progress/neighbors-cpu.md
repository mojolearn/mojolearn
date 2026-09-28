# neighbors-cpu: progress

Lane `neighbors-cpu` (speed fan-out, CPU-speed lane of the neighbors family:
kNN / radius / ball cover, KDE, SVC / SVR, KernelRidge / Nystroem /
RBFSampler, GP regression / classification, and the x_neighbors
expansion ops). Branch `lane/neighbors-cpu`, worktree
`~/mojolearn-wt/neighbors-cpu`, pod `neighbors-cpu` (RunPod H100, Xeon 8470,
cgroup quota 22 CPUs of 208 visible).

## Rules this lane holds

- IDENTICAL CPU bits unchanged before/after, at every thread count, and
  still equal to the GPU (the lane check).
- The CPU tier is IDENTICAL only (`bindings/build_host_family.sh` refuses
  any other mode), so there is no FAST CPU arm to speed up; FAST is a GPU
  tier for this family.
- Speed comes from SIMD across CELLS (one lane = one output cell, the
  scalar statement sequence unchanged), removing per-cell allocations, and
  (once `core/host_parallel.mojo` is on main) threads over independent
  rows. Never a reordered fold.

## New shared host pieces (lane neighbors-cpu)

| file | what | proof |
|---|---|---|
| `core/host_simd_identical.mojo` | `ftz_v`, `twiddle_v`, `untwiddle`, `isnan_v`, `expf_v`, `logf_v`: the scalar seams lane-wise | `pixi run check-host-simd-identical`: ALL 2^32 float32 words through expf_v / logf_v / ftz_v vs portable_expf / portable_logf / ftz: 0 mismatches. Sabotage (`-D MOJOLEARN_HOST_SIMD_SABOTAGE`, the r^2 coefficient of expf_v one unit): FAIL. MEASURED: `SIMD.ne(x, x)` reads False on a NaN lane (ordered compare), so the NaN test is on the bits. |
| `core/host_gemm_simd.mojo` | `host_gemm_identical`: `gemm_oracle`'s bits, lanes over output columns (narrow products transposed so lanes run over rows) | `pixi run check-host-gemm-simd`: NN/NT/TN, k in {0,1,7,11,128,129,220,300,1000,2049}, m in {1,3,4,9}, n in {1,5,8,17}, operands with subnormals, signed zeros, 1e19: 15,810 cells, 0 mismatches; separation guard (serial vs tree fold differ). Sabotage (`-D MOJOLEARN_HOST_GEMM_SIMD_SABOTAGE`, stride pairing in the fold): FAIL (3,142 cells). |

## What changed, per module (all bit-for-bit; digests before == after)

- `core/knn_host_predict.mojo`: brute k-NN (every metric) and the ball cover
  scans (radius count / fill, rbc k-NN) through a block engine: the index
  packed once per call (flushed, feature-major, 8 columns per block), 4
  query rows x 8 index columns per register tile, selection by the same
  composite keys with a vector pre-test that only skips columns the carry
  insertion rejects.
- `kde/host/kde_oracle.mojo`: score_samples through the same kind of block
  engine; log kernels lane-wise (logf_v), the log-sum-exp keeps its serial
  max scan and serial sum, exponentials W at a time (expf_v).
- `svm/host/smo_oracle.mojo`: the contract dot cell allocation-free (stack
  leaves, in-place balanced fold); the fit's square tile and UpdateF through
  a SIMD kernel-cell block (8 cells, lane-wise leaves, fold, epilogue); the
  block solve's three scans and f update W lanes at a time (the scans keep
  the unique best of a strict total order); the decision path
  allocation-free.
- `cholesky/host/chol_oracle.mojo`: trailing update 8 cells per register;
  both triangular solves 8 right-hand sides per register
  (`-D MOJOLEARN_CHOL_HOST_SCALAR` builds the old loops).
- `kernel_methods/host/km_host_oracle.mojo`, `gaussian_process/host/*`:
  `gemm_oracle` -> `host_gemm_identical`.

## Before -> after (CPU IDENTICAL, pod neighbors-cpu, default thread count)

Data from R2 (`tools/classical_two_datasets.py prep`, the bench board's
blocks). Harness `~/mojolearn-evidence/neighbors-cpu/cpu_time.py` (one fit,
one predict, sha256 of every output). Seconds; digests equal in every row.

Full bench shapes, default thread count (the pod's 104 "physical cores"
on a 22-CPU quota), before = main f237f1996, after = branch tip at the
first full run (knn index packed per task then; see the second table):

| dataset | algorithm | shape (fit / predict rows x d) | fit s before -> after | predict s before -> after |
|---|---|---|---|---|
| taxi | NearestNeighbors euclidean k=10 | 400,000 / 4,000 x 11 | - | 3.38 -> 2.94 |
| taxi | manhattan | same | - | 3.43 -> 1.47 |
| taxi | cosine | same | - | 3.74 -> 1.64 |
| taxi | KNeighborsClassifier | same | - | 3.36 -> 2.79 |
| taxi | KNeighborsRegressor distance | same | - | 3.56 -> 2.72 |
| istella | NearestNeighbors euclidean | 400,000 / 4,000 x 220 | - | 47.7 -> 32.9 |
| istella | manhattan | same | - | 70.3 -> 31.9 |
| istella | cosine | same | - | 76.7 -> 32.7 |
| taxi | KernelDensity gaussian | 100,000 / 2,000 x 11 | - | 0.63 -> 0.31 |
| istella | KernelDensity gaussian | 100,000 / 2,000 x 220 | - | 8.06 -> 3.62 |
| istella | KernelDensity epanechnikov | same | - | 7.50 -> 3.71 |
| taxi | SVC rbf | 10,000 / 10,000 x 11 | 31.9 -> 5.62 | 0.88 -> 0.14 |
| taxi | SVR rbf | same | 15.8 -> 2.46 | 0.73 -> 0.12 |
| istella | SVC rbf | 10,000 / 10,000 x 220 | 14.6 -> 1.69 | 1.05 -> 0.46 |
| istella | SVR rbf | same | 50.8 -> 5.96 | 2.49 -> 1.08 |
| taxi | KernelRidge rbf | 4,000 / 10,000 x 10 | 29.2 -> 4.08 | 1.95 -> 1.57 |
| istella | KernelRidge rbf | 4,000 / 10,000 x 219 | 39.7 -> 4.38 | 29.7 -> 3.02 |
| taxi | Nystroem rbf, 1,000 components | 4,000 / 10,000 x 11 | 68.3 -> 63.4 (Jacobi eigh, decomp's) | 28.1 -> 1.65 |
| istella | Nystroem | 4,000 / 10,000 x 220 | 53.2 -> 48.4 | 35.2 -> 1.96 |
| istella | RBFSampler 1,000 components | same | - | 7.94 -> 0.58 |
| istella | GaussianProcessRegressor, return_std | 4,000 / 10,000 x 219 | 54.9 -> 11.8 | 590 -> 50.6 |
| taxi | GaussianProcessClassifier | 4,000 / 10,000 x 11 | 227 -> 30.0 | 199 -> 34.7 |
| istella | GaussianProcessClassifier | 4,000 / 10,000 x 220 | 216 -> 43.3 | 603 -> 49.0 |

After the index is packed once per call and the tile kernels are
specialized per step kind at compile time (same pod, default threads,
1,000 queries): istella euclidean 12.7 -> 2.01 s, manhattan 17.7 -> 1.30,
cosine 18.4 -> 1.89, ball cover k-NN 16.2 -> 1.26, radius manhattan
39.8 -> 4.04; taxi manhattan 0.88 -> 0.08. Single thread, 100,000 x 220
index, 128 queries: 1.5 -> 9.8 G cell-steps/s. KDE istella 200 queries:
0.86 -> 0.30 s. GP kernel matrix from pre-scaled operands: GPR predict
(800 / 400 rows) 0.75 -> 0.09 s, GPC fit 1.85 -> 0.47 s. Every digest
equal to the base in every row above (NearestNeighbors euclidean,
sqeuclidean, manhattan, chebyshev, cosine, minkowski p=3, ball cover x3
metrics, RadiusNeighbors x4 metrics, KNN classifier/regressor, KDE x7
kernel/metric/weight arms, SVC rbf/poly/linear/sigmoid, SVR, KernelRidge
rbf/poly/laplacian, Nystroem, RBFSampler, GPR rbf / Matern 1/2, 3/2, 5/2
ARD / optimizer, GPC, on taxi and istella).

## Gate (NVIDIA H100 pod, CPU Xeon 8470; branch at 87e54aa7f + origin/main da6c16849)

`tools/algos_lane_check.sh` on the 52 family lanes lane_select picks
(cholesky, gp x9, gpc x2, kde x7, kernel-ridge x4, knn x11, nystroem x4,
radius x4, rbf-sampler, svc x3, svr x2, x-neighbors svm lanes x4):
- clean: every lane AGREE (cuda column vs CPU column, batch/infer/model/
  train 9 fixtures each).
- sabotage, two host-only source patches (they touch only this lane's new
  vector code, so a DISAGREE proves the lanes run it):
  - `sab_gp_sqdist.patch` (GP `_gpr_sqdist_v` output x 1.0000001) on the 11
    gp/gpc lanes: RESULT PASS (AGREE, DISAGREE on all 11, AGREE after
    reversal).
  - `sab_ftz_v_ulp.patch` (`ftz_v` one unit up on every lane) on the other
    41: see below. (On gp it made the CPU arm's own batch check fail,
    BATCH_MOVED on predict(return_std): a row alone takes the scalar trsm
    tail, 64 rows the vector solve, and the patch moved only the vector
    one; that is the patch, not the clean build, whose batch column AGREEs.)
- Tooling gap found: `needed_bindings` drops the ubiquitous
  `_mojolearn_preprocessing` that gp-normalize-y's fit imports
  (`_gp_impl.py:720`, StandardScaler for normalize_y); the GPU arm refused
  until it was built by hand. Reported.

## OWED / NEXT

- Thread the serial host loops (SVM tile, Cholesky, gemm cells, packing,
  x_neighbors generated ops) with `core/host_parallel.mojo` once lane cpu
  merges it; not before (DEVIATION 5900).
- `host_predict_task_count` sizes tasks by `num_physical_cores()` (104 on
  this pod) while the container's cgroup quota is 22 CPUs: 104 tasks
  time-slice on 22 CPUs. Reported to the orchestrator (shared core file).
- Nystroem fit at n_components=1000 is the Jacobi eigh of
  `decomposition/host/pca_oracle.mojo` (decomp's file): 55-68 s.
