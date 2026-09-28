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

| request | Mac | commit | what |
|---|---|---|---|
| 1790626615651 | m4pro-b | 6856b5f8f (the base) | both boards, FAST and IDENTICAL, MOJOLEARN_STAGE_TIMES=1: does the family build and run on the base, and the phases |

## Changes on the branch

(none yet)

## Results

(pending)
