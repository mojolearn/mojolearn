# neighbors-apple3: progress

Apple FAST speed round 3 for the neighbors family (k-NN, radius, KDE, SVM /
SVR, GP / GPC, kernel approximation, KernelRidge, and the x_neighbors
expansion). Brief: ~/mojolearn-evidence/apple3_speed_brief.md. Branch
lane/neighbors-apple3, forked from lane/apple3-merged 6856b5f8f. Rounds one
and two: docs/lanes/progress/neighbors-apple.md, neighbors-apple2.md.

Boards: `bench/x_neighbors_apple_speed.py` (round one),
`bench/x_neighbors_apple2_speed.py` (round two). Job script:
`bench/x_neighbors_apple2_job.sh`. Paired FAST quality:
`bench/x_neighbors_fast_quality.py`. Commands and raw output:
~/mojolearn-evidence/neighbors-apple3/.

## Targets (from the round-one and round-two measurements, FAST, Apple)

Ranked by FAST seconds at the recorded shapes (M4 Pro 1790604321269 and
1790588737156, M3 Ultra 1790614983391):

| algorithm | FAST s (taxi / HIGGS) | Mac | where the time is |
|---|---|---|---|
| Nystroem.fit 4k, 300 components | 0.74 / 0.64 | M4 Pro | the device Jacobi eigh of the 300 x 300 basis kernel (one block, one rotation at a time) |
| KernelPCA.fit 500 | 0.60 / 0.40 | M3 Ultra | the host Jacobi (scalar, strided) |
| RBFSampler.transform 1M x 500 | 0.56 / 0.59 | M4 Pro | copy out 250 ms, gemm 96 ms, epilogue 97 ms |
| SVC.fit 10k rbf | 0.44 / 0.09 | M4 Pro | block solve 0.37 s of 0.44 (99,358 inner iterations in 34 outer) |
| GaussianProcessClassifier.fit 3k | 0.42 / 0.37 | M4 Pro | phases pending |
| SkewedChi2Sampler.transform 1M x 500 | 0.37 / 0.38 | M3 Ultra | phases pending |
| LabelPropagation.fit 5k | 0.32 / 0.11 | M3 Ultra | phases pending |
| KernelRidge.fit 10k | 0.30 / 0.30 | M4 Pro | potrf 247 ms, cho_solve 23 ms |
| GaussianProcessClassifier.predict_proba 3k x 3k | 0.16 to 0.22 | both | the multi-RHS sweep |
| GaussianProcessRegressor.fit 3k | 0.19 / 0.18 | M4 Pro | phases pending |

Listed by round two as opt-in, unproven, left or not flipped:
`-D MOJOLEARN_CHOL_MR4` (opt-in, slower than the merged default on both
Macs: stays opt-in), KernelPCA.fit (host Jacobi, untouched), the identical
radix select's device-memory hazard (the consolidation follow-up rewrote it
with no validation: checked by this lane's first job on the base).

## Speed requests

Commands: bench/neighbors_apple3_jobs/job<N>_cmd.txt. Raw output:
bench/neighbors_apple3_jobs/<request>.txt (copies in
~/mojolearn-evidence/neighbors-apple3/).

| request | Mac | commit | what | result |
|---|---|---|---|---|
| 1790626615651 (job 1) | m4pro-b | 6856b5f8f (the base) | both boards, FAST and IDENTICAL, MOJOLEARN_STAGE_TIMES=1: does the family build and run on the base, and the phases | PASS: every binding builds in both modes, every case runs (SVGP on taxi refuses as in round two: not positive definite at these hyperparameters) |
| 1790627549034 (job 2) | m4pro-b | b32f7312d | A/B of the first three opt-in arms, quality, IDENTICAL digests | PASS; results below |
| 1790628892129 (job 3) | m3ultra-b | b48433cfd | every opt-in arm, both boards, quality, stages | LOST: queued behind five jobs when m3ultra-b was terminated at 21:20Z; it never started. Nothing of it was measured |

## Changes on the branch

Every change is OPT-IN (a build define) until its A/B shows a gain and its
quality check passes. Nothing is default yet.

| commit | change | define | modes | bits |
|---|---|---|---|---|
| 3b273d56d | SVC / SVR: the SMO block solve on the host cores over a copy of the square kernel tile (`svm/impl/host_block_solve.mojo`): the device kernel's selections and updates as 16 lane vector passes, no barriers | `-D MOJOLEARN_SVM_HOST_BLOCK_SOLVE` | FAST, Apple | FAST may move by a rounding (the compiler's products and sums); quality: svc, svr |
| 3b273d56d | Nystroem.fit: the basis kernel's eigendecomposition on the host (`x_neighbors/fast_eigh.mojo`), not the one-block device Jacobi | `-D MOJOLEARN_NYS_HOST_EIGH` | FAST, Apple | FAST moves (another rotation formula and stopping rule); quality: nystroem |
| 3b273d56d | KernelPCA.fit: `symmetric_eig_host`'s Jacobi with its rotations as vectors over contiguous rows (full symmetric matrix, transposed basis) | `-D MOJOLEARN_XN_EIGH_ROWS` | any | the same statements per cell: the IDENTICAL arm shows whether the words are the scalar routine's |
| 8fca1b2be | large outputs: the device buffer mapped into the host and copied once over the host cores, not staged (RBFSampler, Nystroem, SkewedChi2Sampler, PolynomialCountSketch, AdditiveChi2Sampler and every x_neighbors output of 16 MB or more) | `-D MOJOLEARN_KM_MAPPED_OUT`, `-D MOJOLEARN_XN_MAPPED_DOWN` | any | a copy |
| e7a3001d7 | GaussianProcessClassifier.predict_proba: the latent variance on the device (`gpc_scale_rows` and `gpc_latent_var` as kernels, one thread per column, the fold over i ascending) | `-D MOJOLEARN_GPC_DEVICE_VAR` | FAST, Apple | the host's statements; quality: gpc |
| e7a3001d7 | GaussianProcessClassifier.fit: K stays on the device for the Newton loop | `-D MOJOLEARN_GPC_RESIDENT_K` | FAST, Apple | a copy less |
| e7a3001d7 | GP downloads: one vector copy into a list of the final length, not n appends | `-D MOJOLEARN_GP_BULK_DOWNLOAD` | any | a copy |
| 427f09b31 | LabelPropagation / LabelSpreading fit: the iteration in batches of 16 steps behind one drain, the stopping and finiteness decisions made on the host for every state in order | `-D MOJOLEARN_XN_LP_BATCH` | any | the same kernels, order and decisions |
| 1325de25f | RBFSampler.transform: projection, offset, cosine and scale in one kernel per cell when X has at most 64 features | `-D MOJOLEARN_RBF_FUSED` | FAST, Apple | FAST may move (the projection's fold order); quality: the kernel approximation error |

Timing only: `MOJOLEARN_STAGE_TIMES=1` now prints GPC_FIT_DEVICE_STAGES and
GPC_PREDICT_STAGES (e7a3001d7).

IDENTICAL code touched without a define (no arithmetic): Nystroem's device
eigendecomposition moved into `_nystroem_device_eigh` (the same launches in
the same order), and the SMO solver allocates seven one-word host buffers
it does not use. Job 3 prints the IDENTICAL digests at the head against the
base's (job 1).

## Shared code touched

None so far. `x_neighbors/fast_eigh.mojo` and `svm/impl/host_block_solve.mojo`
are new files of this family; kernel_methods imports the first.

## Results

### Job 1, request 1790626615651 (m4pro-b, M4 Pro, the base 6856b5f8f)

Every IDENTICAL digest equals round two's record (nn 9ff75469.., nn-k20
34a8923f.., nn-ties 52412ee0.., svc 788a3d6c.., gpr 07ca0943.., gpc
fc039349.. / 0de2f247.., lof 428f08c2.. / 8dbff18c.., ocsvm f0d8cd1a.. /
f89bc8b4.., labelprop 0cafc4a9.. / b02924bf.., spectral-knn a171b2a3.. /
a07a07bb.., nn-k2000 1ce567d0.. / 3d34b599..), so the consolidation's radix
follow-up (the rescanning select) builds on Metal and keeps the large-k
digests on the M4 Pro.

FAST seconds on the base (fit / predict or transform), the before numbers
on the M4 Pro:

| algorithm | taxi | HIGGS | phases (MOJOLEARN_STAGE_TIMES) |
|---|---|---|---|
| Nystroem 4k, 300 components, transform 100k | 0.741 / 0.120 | 0.560 / 0.119 | |
| KernelPCA 500, transform 10k | 0.546 / 0.014 | 0.371 / 0.014 | |
| RBFSampler.transform 1M x 500 | 0.537 | 0.537 | gemm 93 ms, epilogue 92 ms, copy out 236 ms |
| SkewedChi2Sampler.transform 1M x 500 | 0.531 | 0.530 | |
| SVC 10k | 0.398 / 0.003 | 0.082 / 0.004 | taxi block solve 0.37 s of the fit |
| GaussianProcessClassifier 3k, predict_proba 3k | 0.407 / 0.192 | 0.365 / 0.186 | |
| KernelRidge 10k | 0.283 / 0.033 | 0.283 / 0.032 | potrf 230 ms, cho_solve 22 ms |
| PolynomialCountSketch.transform 200k x 500 | 0.270 | 0.272 | |
| SpectralEmbedding(knn) 20k | 0.241 | 0.249 | |
| LabelPropagation 5k | 0.239 / 0.021 | 0.109 / 0.022 | |
| GaussianProcessRegressor 3k | 0.184 / 0.017 | 0.185 / 0.018 | |
| SVR 10k | 0.079 / 0.003 | 0.090 / 0.003 | |
| LabelSpreading 5k | 0.088 / 0.031 | 0.095 / 0.034 | |
| NearestNeighbors(k=2000) 20k x 2k | 0.087 | 0.088 | |
| LocalOutlierFactor 20k, score 5k | 0.078 / 0.045 | 0.072 / 0.033 | |
| kneighbors k=20, 200k x 10k | 0.073 | 0.058 | |
| radius 100k x 5k | 0.059 | 0.066 | |
| KNNImputer.transform 5k x 50k | 0.063 | 0.055 | |
| kneighbors / kNN classifier / regressor k=10, 200k x 10k | 0.040 to 0.048 | 0.032 to 0.039 | |
| everything else on the two boards | under 0.06 | under 0.06 | |

### Job 2, request 1790627549034 (m4pro-b, M4 Pro, b32f7312d), arms in one job, forward and reverse

Every arm builds. IDENTICAL digests at the head equal the base's (svc
788a3d6c.. / 24d6f4ea.., svr 0ce89d3c.. / 879a30f0.., krr 5654a1e2.. /
775b2534.., nystroem 7cb0754a.. / 8deb9714.., rbf 5b7fc0c7.. / 1020aa75..,
kpca cb9a04d3.. / 7a04ae1e..).

| algorithm | mode | arm | taxi before | taxi after | HIGGS before | HIGGS after | digests |
|---|---|---|---|---|---|---|---|
| Nystroem.fit 4k, 300 components | FAST | `-D MOJOLEARN_NYS_HOST_EIGH` | 0.743 / 0.735 | 0.110 / 0.109 | 0.561 / 0.555 | 0.082 / 0.079 | FAST moves (33c168f7.. -> a37f4287.., 712ceeaa.. -> 72e7ecae..) |
| KernelPCA.fit 500 | FAST | `-D MOJOLEARN_XN_EIGH_ROWS` | 0.561 / 0.544 | 0.425 / 0.427 | 0.379 / 0.369 | 0.290 / 0.290 | EQUAL (8a8e5785.., effcc1c8..) |
| KernelPCA.fit 500 | IDENTICAL | `-D MOJOLEARN_XN_EIGH_ROWS` | 1.857 / 1.860 | 1.194 / 1.195 | 1.262 / 1.265 | 0.816 / 0.815 | EQUAL (cb9a04d3.., 7a04ae1e..) |
| SVC.fit 10k | FAST | `-D MOJOLEARN_SVM_HOST_BLOCK_SOLVE` | 0.368 / 0.368 | 0.979 / 0.981 | 0.067 / 0.067 | 0.153 / 0.155 | EQUAL (788a3d6c.., e7da63c8..) |
| SVR.fit 10k | FAST | same | 0.066 / 0.066 | 0.160 / 0.159 | 0.072 / 0.074 | 0.180 / 0.179 | EQUAL (ff9a2f1b.., 1f64b96f..) |

- The host block solve is SLOWER (about 9 us an inner iteration on the
  host against 3.7 us on the device, from the three fits' iteration
  counts) although its words are the device solve's. It stays opt-in; job
  3 times its stages and two narrower lane widths.
- The row-vector Jacobi keeps the scalar routine's words in BOTH modes
  (the IDENTICAL digests are equal), 1.3x in FAST and 1.55x in IDENTICAL.
- Paired quality (FAST minus IDENTICAL, 5 seeds, the score is minus the
  relative kernel error for Nystroem): svc, svr and kpca are the same
  numbers in both arms (their FAST digests did not move). Nystroem with the
  host eigendecomposition is LOWER than the before arm on every seed, by
  0.000004 to 0.000031 on taxi (kernel error about 0.010 to 0.016) and by
  0.000005 on HIGGS (kernel error about 0.028): mean +0.000675 against
  +0.000697 (taxi), -0.000005 against 0.000000 (HIGGS). That is not "matches
  or beats", so `-D MOJOLEARN_NYS_HOST_EIGH` as measured stays OPT-IN. Job 3
  carries two more arms of it: the device solver's own statements on the
  host (`-D MOJOLEARN_NYS_HOST_EIGH_TWIN`, which can keep FAST's words) and
  the binary64 tridiagonal QL solve (`-D MOJOLEARN_NYS_HOST_EIGH_QL`).

### State at 21:35Z Sep 28

m3ultra-b and the three M4 Macs are gone. Measured so far: jobs 1 and 2 (M4
Pro). NOT BUILT and NOT MEASURED, all opt-in: the mapped download and
upload, the fused RBFSampler kernel, the GPC device variance / resident K /
bulk download, the batched label propagation loop, the sparse
PolynomialCountSketch convolution, the binary64 QL eigensolver, the Nystroem
device-twin host solve, the lazily zeroed outputs, and the host block
solve's 4 and 8 lane arms. Each unbuilt arm lives in its own module,
imported only by the build that selects it, so none can break a default
build.

Default on so far: the row-vector Jacobi on Apple (job 2, digests equal in
both modes).

Base digests against round two (job 1 against requests 1790614983391,
1790618924175, 1790613586332): 62 of 62 board-two rows equal in both modes,
which includes lane py-dn-kern's fused KernelPCA.transform, OneClassSVM
score and SVGP (unproven in their own file). The FAST k = 2,000 k-NN and
FAST spectral rows differ, as they do run to run inside one arm (FAST's
selector does not pin ties).

Host-side checks run on the laptop, each a few seconds on one core, no
build and no Metal: a Python transcription of the QL solver against NumPy
(n up to 65, residual under 4e-15), the batched label propagation control
flow against the reference loop (400 random cases, 0 differ), and the lazy
output allocation.
