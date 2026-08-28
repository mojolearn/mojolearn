# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/mac-2026-08-28_122551-classical/logs`, 45 arm logs.

Device(s) reported by the arms themselves: Apple_M4, arm64-10cores

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| cd | ours | 2048x16 | 3 | 7.630 | 7.088 | 7.719 | 12.265 | 3bc909a47ebc9bda |
| cholesky | ours | RBF.64x64r4 | 3 | 4.737 | 4.724 | 5.665 | 9.485 | 48c6594203c35bb4 |
| dbscan | ours | 4000x16 | 3 | 22.431 | 21.806 | 24.249 | 27.730 | 4869c285f078c0a5 |
| gmm | ours | SEPARATED.24x2K3 | 3 | 32.283 | 31.892 | 32.818 | 40.541 | b934e26ee2bc5277 |
| gmm | sklearn-cpu | SEPARATED.24x2K3 | 3 | 1.115 | 1.003 | 1.346 | 33.755 | 6536aef537e0af7e |
| gp | ours | ard.12x3s6 | 3 | 6.125 | 6.055 | 6.834 | 11.156 | 13c0ba0b989d05ff |
| gp | sklearn-cpu | ard.12x3s6 | 3 | 0.290 | 0.262 | 0.354 | 3.581 | 27fa3ce9b8f7824f |
| hdbscan | ours | blobs96.96x4 | 3 | 21.551 | 21.361 | 21.610 | 30.620 | 8e1998063cb24ce6 |
| holtwinters | ours | 7x72f12 | 3 | 17.816 | 17.679 | 17.855 | 295.645 | c6f5e5b9d9daf52c |
| ivf | ours | 512x8q64L8p3k8 | 3 | 86.763 | 80.209 | 98.823 | 139.128 | 0908d00a25ccbed4 |
| kde | ours | 1024x256x8 | 3 | 2.691 | 2.474 | 2.760 | 6.430 | 110eb573543dcc0f |
| kmeans | ours | 4000000x32k64i20 | 3 | 748.774 | 744.228 | 767.884 | 796.930 | c5f012ba7bff508e |
| knn | ours | 400000x32q4000k10 | 3 | 692.947 | 688.458 | 696.668 | 715.372 | c07ab69619951ff1 |
| knn | ours-fused | 400000x32q4000k10 | 3 | 699.921 | 688.035 | 703.564 | 705.101 | c07ab69619951ff1 |
| knn | ours-tiled | 400000x32q4000k10 | 3 | 616.570 | 615.361 | 635.150 | 612.737 | MOVED(3) |
| kpss | ours | 8x520 | 3 | 0.681 | 0.680 | 1.104 | 69.855 | c5237dfab4ee6cac |
| krr | ours | RBF.16x5 | 3 | 4.962 | 4.915 | 6.907 | 9.475 | b464758a7777e093 |
| linkage | ours | blobs_dups.102x5 | 3 | 14.035 | 9.714 | 14.751 | 17.408 | 5d3155fc940e61d2 |
| metrics | ours | lab2053.flt2053.sil521x4.tru301x6 | 3 | 9.491 | 9.079 | 10.600 | 15.581 | 9581966d0ec159e6 |
| nystroem | ours | RBF.16x5q8 | 3 | 5.816 | 5.291 | 7.278 | 10.840 | 41209b855b2d0248 |
| nystroem | sklearn-cpu | RBF.16x5q8 | 3 | 0.483 | 0.472 | 0.573 | 5.169 | - |
| ols | ours | 4000000x32 | 3 | 33.500 | 33.362 | 33.677 | 144.922 | c16cc39e6e050b8c |
| pca | ours | 4000000x32c8 | 3 | 36.315 | 36.200 | 36.845 | 282.365 | 38e551d5c5b3f0dc |
| rbfsampler | ours | 3x5q8 | 3 | 2.580 | 2.558 | 3.300 | 5.706 | dabf82a3ba80c8ab |
| rbfsampler | sklearn-cpu | 3x5q8 | 3 | 0.167 | 0.151 | 0.194 | 1.941 | - |
| resample | ours | 200x2r10000 | 3 | 7.673 | 6.900 | 7.768 | 12.940 | 38bb5e267f2ef221 |
| resample | scipy-cpu | 200x2r10000 | 3 | 35.149 | 34.057 | 35.181 | 37.910 | f8eacc6a5e8d8b8b |
| spectral | ours | 48x4c3 | 3 | 75.866 | 75.210 | 77.981 | 98.785 | 46aa77c9e4ba34c0 |
| spectral | sklearn-cpu | 48x4c3 | 3 | 3.648 | 3.291 | 4.407 | 27.077 | - |
| svm | ours | F2.xor.240x2 | 3 | 12.044 | 11.535 | 18.319 | 22.769 | 8621097bdb3daded |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| cd | 2048x16 | (none ran) | 7.630 | - | - | NO OPPONENT ON THIS BOX |
| cholesky | RBF.64x64r4 | (none ran) | 4.737 | - | - | NO OPPONENT ON THIS BOX |
| dbscan | 4000x16 | (none ran) | 22.431 | - | - | NO OPPONENT ON THIS BOX |
| gmm | SEPARATED.24x2K3 | sklearn-cpu | 32.283 | 1.115 | 28.95 | we are 28.95x SLOWER |
| gp | ard.12x3s6 | sklearn-cpu | 6.125 | 0.290 | 21.14 | we are 21.14x SLOWER |
| hdbscan | blobs96.96x4 | (none ran) | 21.551 | - | - | NO OPPONENT ON THIS BOX |
| holtwinters | 7x72f12 | (none ran) | 17.816 | - | - | NO OPPONENT ON THIS BOX |
| ivf | 512x8q64L8p3k8 | (none ran) | 86.763 | - | - | NO OPPONENT ON THIS BOX |
| kde | 1024x256x8 | (none ran) | 2.691 | - | - | NO OPPONENT ON THIS BOX |
| kmeans | 4000000x32k64i20 | (none ran) | 748.774 | - | - | NO OPPONENT ON THIS BOX |
| knn | 400000x32q4000k10 | ours-fused | 692.947 | 699.921 | 0.99 | we are 1.01x FASTER |
| knn | 400000x32q4000k10 | ours-tiled | 692.947 | 616.570 | 1.12 | we are 1.12x SLOWER |
| kpss | 8x520 | (none ran) | 0.681 | - | - | NO OPPONENT ON THIS BOX |
| krr | RBF.16x5 | (none ran) | 4.962 | - | - | NO OPPONENT ON THIS BOX |
| linkage | blobs_dups.102x5 | (none ran) | 14.035 | - | - | NO OPPONENT ON THIS BOX |
| metrics | lab2053.flt2053.sil521x4.tru301x6 | (none ran) | 9.491 | - | - | NO OPPONENT ON THIS BOX |
| nystroem | RBF.16x5q8 | sklearn-cpu | 5.816 | 0.483 | 12.03 | we are 12.03x SLOWER |
| ols | 4000000x32 | (none ran) | 33.500 | - | - | NO OPPONENT ON THIS BOX |
| pca | 4000000x32c8 | (none ran) | 36.315 | - | - | NO OPPONENT ON THIS BOX |
| rbfsampler | 3x5q8 | sklearn-cpu | 2.580 | 0.167 | 15.41 | we are 15.41x SLOWER |
| resample | 200x2r10000 | scipy-cpu | 7.673 | 35.149 | 0.22 | we are 4.58x FASTER |
| spectral | 48x4c3 | sklearn-cpu | 75.866 | 3.648 | 20.80 | we are 20.80x SLOWER |
| svm | F2.xor.240x2 | (none ran) | 12.044 | - | - | NO OPPONENT ON THIS BOX |

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
| 1 | gmm | SEPARATED.24x2K3 | 32.283 | sklearn-cpu | 1.115 | **28.95x SLOWER** |
| 2 | gp | ard.12x3s6 | 6.125 | sklearn-cpu | 0.290 | **21.14x SLOWER** |
| 3 | spectral | 48x4c3 | 75.866 | sklearn-cpu | 3.648 | **20.80x SLOWER** |
| 4 | rbfsampler | 3x5q8 | 2.580 | sklearn-cpu | 0.167 | **15.41x SLOWER** |
| 5 | nystroem | RBF.16x5q8 | 5.816 | sklearn-cpu | 0.483 | **12.03x SLOWER** |
| 6 | resample | 200x2r10000 | 7.673 | scipy-cpu | 35.149 | **4.58x FASTER** |

6 rows in total have an opponent: 0 throughput, 6 fixed-cost.

## Our own arms, A/B

Two of OUR kernels on the same row. This is not a speed claim
against anyone; it asks whether our own dispatch is choosing the
right arm on THIS vendor. An arm chosen by a measurement taken on
a different vendor is the failure this table exists to catch.

| lane | shape | arm | median ms | vs shipped `ours` |
|---|---|---|---|---|
| knn | 400000x32q4000k10 | ours | 692.947 | (the shipped choice) |
| knn | 400000x32q4000k10 | ours-fused | 699.921 | 1.01x slower |
| knn | 400000x32q4000k10 | ours-tiled | 616.570 | **1.12x FASTER** |

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| cd | cuml-gpu | ModuleNotFoundError("No module named 'cuml'") |
| cholesky | torch-gpu | ModuleNotFoundError("No module named 'torch'") |
| dbscan | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| gmm | cuml-gpu | RAPIDS ships no GaussianMixture; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| gp | cuml-gpu | RAPIDS ships no Gaussian process regressor; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| hdbscan | cuml-gpu | ModuleNotFoundError("No module named 'cuml'") |
| holtwinters | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| ivf | cuvs-gpu | import failed: ModuleNotFoundError("No module named 'cupy'") |
| kde | cuml-gpu | ModuleNotFoundError("No module named 'cuml'") |
| kmeans | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| knn | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| kpss | cuml-gpu | this RAPIDS build exposes no cuml.tsa.stationarity.kpss_test; falling back to statsmodels on the CPU, labeled statsmodels-cpu |
| kpss | statsmodels-cpu | import failed: ModuleNotFoundError("No module named 'statsmodels'") |
| krr | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| linkage | cuml-gpu | ModuleNotFoundError("No module named 'cuml'") |
| metrics | cuml-gpu | import failed (one of the eleven cuml metrics is missing on this build): ModuleNotFoundError("No module named 'cuml'") |
| nystroem | cuml-gpu | RAPIDS ships no Nystroem; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| ols | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| pca | cuml-gpu | import failed: ModuleNotFoundError("No module named 'cuml'") |
| rbfsampler | cuml-gpu | RAPIDS ships no RBFSampler / random Fourier features; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| resample | cuml-gpu | RAPIDS ships no bootstrap; the arm below is SciPy on the CPU and is labeled scipy-cpu |
| spectral | cuml-gpu | RAPIDS ships no SpectralClustering estimator; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| svm | cuml-gpu | ModuleNotFoundError("No module named 'cuml'") |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| knn | ours-tiled | 400000x32q4000k10 | 3 |

## Notes the arms printed

- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu optimizer=None and normalize_y=False on both sides: our gpr_fit_host implements no hyperparameter optimizer (DEVIATION 1761), so an arm that optimized would be doing different work
- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu scikit-learn works in float64 and we work in float32; that is a real difference in the amount of arithmetic and it cannot be turned off on their side
- `classical.knn.ours.log`: lane=knn arm=ours-tiled hash moved across rounds: cd95ccfa10c243d9 df3b2b6a8cffd1a5
- `classical.nystroem.vendor.log`: lane=nystroem arm=sklearn-cpu hash=- : the BASIS SAMPLE differs. Ours permutes with a pinned Philox stream and theirs with numpy RandomState, so the two fit different rows. The WORK is the same (sample q rows, form q x q, eigendecompose, scale, cross kernel, matmul) and that is what is being timed
- `classical.rbfsampler.vendor.log`: lane=rbfsampler arm=sklearn-cpu hash=- : the random draws differ (a pinned Philox stream against numpy RandomState.normal). Both arms draw a d x q weight matrix and a q offset, then one matmul and one cosine over 3 rows
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu statistic=mean of column 0, method=percentile, n_resamples=10000, confidence_level passed explicitly. Our bootstrap_host resamples ROWS of the two-column sample (SciPy's paired=True shape) and computes the mean of column 0, which is what this arm does
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu with_bca_diagnostics is False on our side, so neither arm computes a jackknife
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu affinity=nearest_neighbors with the same n_neighbors, same n_clusters, same n_components, same n_init: both arms build a kNN affinity, take a Lanczos eigendecomposition of the normalized Laplacian and run k-means on the embedding. The eigensolvers are two Lanczos implementations (ours restarts, theirs is ARPACK)
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu hash=- : cluster label NUMBERING is arbitrary in both and the two are at best a permutation of each other

