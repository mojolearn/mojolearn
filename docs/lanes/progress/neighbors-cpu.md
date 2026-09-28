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

(table filled below)

## OWED / NEXT

- Thread the serial host loops (SVM tile, Cholesky, gemm cells, packing,
  x_neighbors generated ops) with `core/host_parallel.mojo` once lane cpu
  merges it; not before (DEVIATION 5900).
- `host_predict_task_count` sizes tasks by `num_physical_cores()` (104 on
  this pod) while the container's cgroup quota is 22 CPUs: 104 tasks
  time-slice on 22 CPUs. Reported to the orchestrator (shared core file).
- Nystroem fit at n_components=1000 is the Jacobi eigh of
  `decomposition/host/pca_oracle.mojo` (decomp's file): 55-68 s.
