# Mac M4, 2026-08-27: the six sklearn-opponent classical lanes, first Apple run

MacBook Pro M4, FAST, commit 9a4e759, `nice 19`, one lane per process,
dump-then-race per bench/speed/CLASSICAL_SPEED.md, 3 timed rounds after one
warm-up. These are the ONLY classical lanes whose harness has a legal Apple
opponent today (sklearn/scipy CPU); the cuml-opponent lanes refuse here by
name and their Mac opponents are a harness extension still owed.

THE READING: five of six are FIXED-COST rows at tiny shipped fixtures
(24x2 ... 48x4), where sklearn's launch-free CPU path beats our
GPU-context-paying arm 14-35x. This is the same mechanism as our 2-16x
wins over cuML-GPU on small fixtures on the H100, with the roles reversed:
whoever pays the per-call fixed cost loses the small-fixture lane. It is a
statement about call overhead at these shapes, not about kernels. At real
sizes on this box (the 2026-08-20 algorithm-matched board, 4M x 32) we are
2.7-4x FASTER than sklearn on ols/pca/kmeans. What these six lanes owe is
load-ladder fixtures; their shipped sizes were set for orbit parity, not
for measurement.

# The FAST path against the vendor, one rented NVIDIA box

Parsed from `/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a109690e-5801-4e96-ba60-c28d87725bdc/scratchpad/mac_classical_run`, 18 arm logs.

Device(s) reported by the arms themselves: Apple_M4, arm64-10cores

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| gmm | ours | SEPARATED.24x2K3 | 6 | 35.120 | 31.354 | 44.962 | 57.396 | b934e26ee2bc5277 |
| gmm | sklearn-cpu | SEPARATED.24x2K3 | 3 | 0.989 | 0.984 | 1.277 | 33.926 | 6536aef537e0af7e |
| gp | ours | ard.12x3s6 | 6 | 6.842 | 5.650 | 7.912 | 11.651 | 13c0ba0b989d05ff |
| gp | sklearn-cpu | ard.12x3s6 | 3 | 0.276 | 0.256 | 0.307 | 2.358 | 27fa3ce9b8f7824f |
| nystroem | ours | RBF.16x5q8 | 6 | 6.278 | 5.498 | 7.855 | 12.102 | 41209b855b2d0248 |
| nystroem | sklearn-cpu | RBF.16x5q8 | 3 | 0.453 | 0.440 | 0.519 | 4.142 | - |
| rbfsampler | ours | 3x5q8 | 6 | 3.208 | 2.112 | 3.726 | 8.959 | dabf82a3ba80c8ab |
| rbfsampler | sklearn-cpu | 3x5q8 | 3 | 0.169 | 0.168 | 0.188 | 2.707 | - |
| resample | ours | 200x2r10000 | 6 | 8.646 | 8.290 | 9.383 | 203.155 | 38bb5e267f2ef221 |
| resample | scipy-cpu | 200x2r10000 | 3 | 34.663 | 34.306 | 35.227 | 37.644 | f8eacc6a5e8d8b8b |
| spectral | ours | 48x4c3 | 6 | 84.643 | 77.193 | 85.517 | 96.565 | 46aa77c9e4ba34c0 |
| spectral | sklearn-cpu | 48x4c3 | 3 | 3.180 | 3.156 | 4.542 | 25.428 | - |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| gmm | SEPARATED.24x2K3 | sklearn-cpu | 35.120 | 0.989 | 35.52 | we are 35.52x SLOWER |
| gp | ard.12x3s6 | sklearn-cpu | 6.842 | 0.276 | 24.78 | we are 24.78x SLOWER |
| nystroem | RBF.16x5q8 | sklearn-cpu | 6.278 | 0.453 | 13.87 | we are 13.87x SLOWER |
| rbfsampler | 3x5q8 | sklearn-cpu | 3.208 | 0.169 | 19.04 | we are 19.04x SLOWER |
| resample | 200x2r10000 | scipy-cpu | 8.646 | 34.663 | 0.25 | we are 4.01x FASTER |
| spectral | 48x4c3 | sklearn-cpu | 84.643 | 3.180 | 26.62 | we are 26.62x SLOWER |

## FIXED-COST rows -- NOT a speed claim

Every row here is a lane whose fixture is small enough that both
arms are dominated by launch and dispatch latency, plus the Python
call overhead the vendor arm pays inside the clock and ours does
not. Read each as WHAT ONE FIT COSTS END TO END on this box.

A ratio here is not wrong, it is UNCLAIMABLE: it does not tell you
whose kernel is faster. Some of these lanes cannot be made bigger
for stated reasons -- `hdbscan`'s dense mutual-reachability arm
materializes an m x m matrix, and inventing a larger fixture to
make the number look like throughput would be inventing a dataset.
Others simply have no size knob yet, and that is owed work.

They are ranked and kept rather than deleted because the fixed
cost is a real thing a user pays on a small problem.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | gmm | SEPARATED.24x2K3 | 35.120 | sklearn-cpu | 0.989 | **35.52x SLOWER** |
| 2 | spectral | 48x4c3 | 84.643 | sklearn-cpu | 3.180 | **26.62x SLOWER** |
| 3 | gp | ard.12x3s6 | 6.842 | sklearn-cpu | 0.276 | **24.78x SLOWER** |
| 4 | rbfsampler | 3x5q8 | 3.208 | sklearn-cpu | 0.169 | **19.04x SLOWER** |
| 5 | nystroem | RBF.16x5q8 | 6.278 | sklearn-cpu | 0.453 | **13.87x SLOWER** |
| 6 | resample | 200x2r10000 | 8.646 | scipy-cpu | 34.663 | **4.01x FASTER** |

6 rows in total have an opponent: 0 throughput, 6 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| gmm | cuml-gpu | RAPIDS ships no GaussianMixture; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| gp | cuml-gpu | RAPIDS ships no Gaussian process regressor; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| nystroem | cuml-gpu | RAPIDS ships no Nystroem; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| rbfsampler | cuml-gpu | RAPIDS ships no RBFSampler / random Fourier features; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| resample | cuml-gpu | RAPIDS ships no bootstrap; the arm below is SciPy on the CPU and is labeled scipy-cpu |
| spectral | cuml-gpu | RAPIDS ships no SpectralClustering estimator; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

## Notes the arms printed

- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu optimizer=None and normalize_y=False on both sides: our gpr_fit_host implements no hyperparameter optimizer (DEVIATION 1761), so an arm that optimized would be doing different work
- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu scikit-learn works in float64 and we work in float32; that is a real difference in the amount of arithmetic and it cannot be turned off on their side
- `classical.nystroem.vendor.log`: lane=nystroem arm=sklearn-cpu hash=- : the BASIS SAMPLE differs. Ours permutes with a pinned Philox stream and theirs with numpy RandomState, so the two fit different rows. The WORK is the same (sample q rows, form q x q, eigendecompose, scale, cross kernel, matmul) and that is what is being timed
- `classical.rbfsampler.vendor.log`: lane=rbfsampler arm=sklearn-cpu hash=- : the random draws differ (a pinned Philox stream against numpy RandomState.normal). Both arms draw a d x q weight matrix and a q offset, then one matmul and one cosine over 3 rows
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu statistic=mean of column 0, method=percentile, n_resamples=10000, confidence_level passed explicitly. Our bootstrap_host resamples ROWS of the two-column sample (SciPy's paired=True shape) and computes the mean of column 0, which is what this arm does
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu with_bca_diagnostics is False on our side, so neither arm computes a jackknife
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu affinity=nearest_neighbors with the same n_neighbors, same n_clusters, same n_components, same n_init: both arms build a kNN affinity, take a Lanczos eigendecomposition of the normalized Laplacian and run k-means on the embedding. The eigensolvers are two Lanczos implementations (ours restarts, theirs is ARPACK)
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu hash=- : cluster label NUMBERING is arbitrary in both and the two are at best a permutation of each other

