# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-28_040832-nvidia-speed-classical/remote/logs`, 45 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| cd | cuml-gpu | 2048x16 | 5 | 1.283 | 1.206 | 1.645 | 297.456 | 1708eb34307fb7a0 |
| cd | ours | 2048x16 | 5 | 1.137 | 1.132 | 1.153 | 440.208 | 3bc909a47ebc9bda |
| cholesky | ours | RBF.64x64r4 | 5 | 0.341 | 0.335 | 0.401 | 517.663 | 3cb8ff7130e4285b |
| cholesky | torch-gpu | RBF.64x64r4 | 5 | 0.162 | 0.155 | 0.317 | 75.848 | 8ec71c2a1e3fbcca |
| dbscan | cuml-gpu | 4000x16 | 5 | 1.093 | 1.057 | 1.404 | 213.667 | 4869c285f078c0a5 |
| dbscan | ours | 4000x16 | 5 | 0.682 | 0.678 | 0.690 | 1.332 | 4869c285f078c0a5 |
| gmm | ours | SEPARATED.24x2K3 | 5 | 2.456 | 2.431 | 2.631 | 528.590 | b934e26ee2bc5277 |
| gp | ours | ard.12x3s6 | 5 | 0.464 | 0.459 | 0.544 | 416.013 | fb8d7954ab8a7807 |
| hdbscan | cuml-gpu | blobs96.96x4 | 5 | 4.417 | 4.358 | 4.963 | 407.974 | 5199770714da8525 |
| hdbscan | ours | blobs96.96x4 | 5 | 1.162 | 1.147 | 1.230 | 519.873 | 8e1998063cb24ce6 |
| holtwinters | cuml-gpu | 7x72f12 | 5 | 8.156 | 8.118 | 8.560 | 333.927 | 97c0d626da007a91 |
| holtwinters | ours | 7x72f12 | 5 | 10.399 | 10.397 | 10.406 | 417.169 | 4fb56da659c4ec4b |
| ivf | cuvs-gpu | 512x8q64L8p3k8 | 5 | 14.584 | 10.174 | 54.461 | 383.699 | cb8f1cef2816d70f |
| ivf | ours | 512x8q64L8p3k8 | 5 | 7.143 | 7.104 | 7.279 | 535.649 | 0908d00a25ccbed4 |
| kde | cuml-gpu | 1024x256x8 | 5 | 0.385 | 0.354 | 0.628 | 28.468 | 1a75f5b84752fcd8 |
| kde | ours | 1024x256x8 | 5 | 0.203 | 0.202 | 0.254 | 0.460 | 4ec9d98c61d3ae83 |
| kmeans | cuml-gpu | 4000000x32k64i20 | 5 | 120.217 | 118.707 | 124.247 | 2808.991 | - |
| kmeans | ours | 4000000x32k64i20 | 5 | 155.353 | 155.314 | 155.528 | 156.089 | c5f012ba7bff508e |
| knn | cuml-gpu | 400000x32q4000k10 | 5 | 9.555 | 9.536 | 9.719 | 44.406 | - |
| knn | ours | 400000x32q4000k10 | 5 | 26.404 | 26.319 | 26.415 | 27.034 | 507b4848cbdb4ef4 |
| knn | ours-fused | 400000x32q4000k10 | 5 | 26.409 | 26.346 | 26.434 | 26.438 | 507b4848cbdb4ef4 |
| knn | ours-tiled | 400000x32q4000k10 | 5 | 53.926 | 53.909 | 53.996 | 161.880 | deed74e7eb93792a |
| kpss | cuml-gpu | 8x520 | 5 | 0.436 | 0.399 | 0.607 | 170.632 | 7270ce3a3ef261c5 |
| kpss | ours | 8x520 | 5 | 0.130 | 0.130 | 0.135 | 459.265 | 91a858d57a2f7156 |
| krr | cuml-gpu | RBF.16x5 | 5 | 1.825 | 1.784 | 2.483 | 1107.911 | bf101d3b60c2f648 |
| krr | ours | RBF.16x5 | 5 | 0.300 | 0.275 | 0.451 | 523.864 | 984fc59aeb95ca83 |
| linkage | cuml-gpu | blobs_dups.102x5 | 5 | 2.318 | 2.259 | 2.897 | 2610.790 | 1ff4250dd4aab576 |
| linkage | ours | blobs_dups.102x5 | 5 | 0.963 | 0.957 | 1.020 | 112.522 | d486f76f518b97e2 |
| metrics | cuml-gpu | lab2053.flt2053.sil521x4.tru301x6 | 5 | 12.626 | 11.172 | 19.354 | 7185.641 | - |
| metrics | ours | lab2053.flt2053.sil521x4.tru301x6 | 5 | 0.687 | 0.681 | 0.709 | 416.840 | 975cb8a31a11a5bf |
| nystroem | ours | RBF.16x5q8 | 5 | 0.418 | 0.391 | 0.574 | 612.693 | 63bde03a6cd53d4c |
| ols | cuml-gpu | 4000000x32 | 5 | 118.837 | 118.779 | 119.174 | 285.665 | 6a660f0646505394 |
| ols | ours | 4000000x32 | 5 | 8.555 | 8.541 | 8.570 | 105.683 | 1bd9a4419821cd59 |
| pca | cuml-gpu | 4000000x32c8 | 5 | 119.669 | 119.487 | 119.889 | 199.060 | 3105bf5678f6f63f |
| pca | ours | 4000000x32c8 | 5 | 9.278 | 9.277 | 9.500 | 114.389 | 5ff46ab4c9ad7cac |
| rbfsampler | ours | 3x5q8 | 5 | 0.193 | 0.137 | 0.304 | 521.663 | ee5d3f738c3c8d8a |
| resample | ours | 200x2r10000 | 5 | 0.529 | 0.524 | 2.294 | 417.728 | 38bb5e267f2ef221 |
| spectral | ours | 48x4c3 | 5 | 6.428 | 6.360 | 6.518 | 526.992 | 553b52e4f09df7cb |
| svm | cuml-gpu | F2.xor.240x2 | 5 | 4.864 | 4.806 | 5.194 | 3449.468 | 28f0580866fb9354 |
| svm | ours | F2.xor.240x2 | 5 | 1.333 | 1.305 | 1.340 | 540.687 | 89a8b6ec1d86118a |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| cd | 2048x16 | cuml-gpu | 1.137 | 1.283 | 0.89 | we are 1.13x FASTER |
| cholesky | RBF.64x64r4 | torch-gpu | 0.341 | 0.162 | 2.11 | we are 2.11x SLOWER |
| dbscan | 4000x16 | cuml-gpu | 0.682 | 1.093 | 0.62 | we are 1.60x FASTER |
| gmm | SEPARATED.24x2K3 | (none ran) | 2.456 | - | - | NO OPPONENT ON THIS BOX |
| gp | ard.12x3s6 | (none ran) | 0.464 | - | - | NO OPPONENT ON THIS BOX |
| hdbscan | blobs96.96x4 | cuml-gpu | 1.162 | 4.417 | 0.26 | we are 3.80x FASTER |
| holtwinters | 7x72f12 | cuml-gpu | 10.399 | 8.156 | 1.27 | we are 1.27x SLOWER |
| ivf | 512x8q64L8p3k8 | cuvs-gpu | 7.143 | 14.584 | 0.49 | we are 2.04x FASTER |
| kde | 1024x256x8 | cuml-gpu | 0.203 | 0.385 | 0.53 | we are 1.89x FASTER |
| kmeans | 4000000x32k64i20 | cuml-gpu | 155.353 | 120.217 | 1.29 | we are 1.29x SLOWER |
| knn | 400000x32q4000k10 | cuml-gpu | 26.404 | 9.555 | 2.76 | we are 2.76x SLOWER |
| knn | 400000x32q4000k10 | ours-fused | 26.404 | 26.409 | 1.00 | we are 1.00x FASTER |
| knn | 400000x32q4000k10 | ours-tiled | 26.404 | 53.926 | 0.49 | we are 2.04x FASTER |
| kpss | 8x520 | cuml-gpu | 0.130 | 0.436 | 0.30 | we are 3.35x FASTER |
| krr | RBF.16x5 | cuml-gpu | 0.300 | 1.825 | 0.16 | we are 6.08x FASTER |
| linkage | blobs_dups.102x5 | cuml-gpu | 0.963 | 2.318 | 0.42 | we are 2.41x FASTER |
| metrics | lab2053.flt2053.sil521x4.tru301x6 | cuml-gpu | 0.687 | 12.626 | 0.05 | we are 18.37x FASTER |
| nystroem | RBF.16x5q8 | (none ran) | 0.418 | - | - | NO OPPONENT ON THIS BOX |
| ols | 4000000x32 | cuml-gpu | 8.555 | 118.837 | 0.07 | we are 13.89x FASTER |
| pca | 4000000x32c8 | cuml-gpu | 9.278 | 119.669 | 0.08 | we are 12.90x FASTER |
| rbfsampler | 3x5q8 | (none ran) | 0.193 | - | - | NO OPPONENT ON THIS BOX |
| resample | 200x2r10000 | (none ran) | 0.529 | - | - | NO OPPONENT ON THIS BOX |
| spectral | 48x4c3 | (none ran) | 6.428 | - | - | NO OPPONENT ON THIS BOX |
| svm | F2.xor.240x2 | cuml-gpu | 1.333 | 4.864 | 0.27 | we are 3.65x FASTER |

## Where we lose most, ranked -- THROUGHPUT rows

These are the rows big enough for the ratio to be about
arithmetic. THIS IS THE ONLY TABLE ANY SPEED CLAIM MAY BE DRAWN
FROM. Against the fastest opponent that actually ran on each row,
because losing 70x to a TF32 arm and 12x to an FP32 arm is one
fact and not two.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | knn | 400000x32q4000k10 | 26.404 | cuml-gpu | 9.555 | **2.76x SLOWER** |
| 2 | kmeans | 4000000x32k64i20 | 155.353 | cuml-gpu | 120.217 | **1.29x SLOWER** |
| 3 | pca | 4000000x32c8 | 9.278 | cuml-gpu | 119.669 | **12.90x FASTER** |
| 4 | ols | 4000000x32 | 8.555 | cuml-gpu | 118.837 | **13.89x FASTER** |

**2 of 4 THROUGHPUT rows with an opponent are wins for us.**

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
| 1 | cholesky | RBF.64x64r4 | 0.341 | torch-gpu | 0.162 | **2.11x SLOWER** |
| 2 | holtwinters | 7x72f12 | 10.399 | cuml-gpu | 8.156 | **1.27x SLOWER** |
| 3 | cd | 2048x16 | 1.137 | cuml-gpu | 1.283 | **1.13x FASTER** |
| 4 | dbscan | 4000x16 | 0.682 | cuml-gpu | 1.093 | **1.60x FASTER** |
| 5 | kde | 1024x256x8 | 0.203 | cuml-gpu | 0.385 | **1.89x FASTER** |
| 6 | ivf | 512x8q64L8p3k8 | 7.143 | cuvs-gpu | 14.584 | **2.04x FASTER** |
| 7 | linkage | blobs_dups.102x5 | 0.963 | cuml-gpu | 2.318 | **2.41x FASTER** |
| 8 | kpss | 8x520 | 0.130 | cuml-gpu | 0.436 | **3.35x FASTER** |
| 9 | svm | F2.xor.240x2 | 1.333 | cuml-gpu | 4.864 | **3.65x FASTER** |
| 10 | hdbscan | blobs96.96x4 | 1.162 | cuml-gpu | 4.417 | **3.80x FASTER** |
| 11 | krr | RBF.16x5 | 0.300 | cuml-gpu | 1.825 | **6.08x FASTER** |
| 12 | metrics | lab2053.flt2053.sil521x4.tru301x6 | 0.687 | cuml-gpu | 12.626 | **18.37x FASTER** |

16 rows in total have an opponent: 4 throughput, 12 fixed-cost.

## Our own arms, A/B

Two of OUR kernels on the same row. This is not a speed claim
against anyone; it asks whether our own dispatch is choosing the
right arm on THIS vendor. An arm chosen by a measurement taken on
a different vendor is the failure this table exists to catch.

| lane | shape | arm | median ms | vs shipped `ours` |
|---|---|---|---|---|
| knn | 400000x32q4000k10 | ours | 26.404 | (the shipped choice) |
| knn | 400000x32q4000k10 | ours-fused | 26.409 | 1.00x slower |
| knn | 400000x32q4000k10 | ours-tiled | 53.926 | 2.04x slower |

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| gmm | cuml-gpu | RAPIDS ships no GaussianMixture; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| gmm | sklearn-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and sklearn-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |
| gp | cuml-gpu | RAPIDS ships no Gaussian process regressor; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| gp | sklearn-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and sklearn-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |
| nystroem | cuml-gpu | RAPIDS ships no Nystroem; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| nystroem | sklearn-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and sklearn-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |
| rbfsampler | cuml-gpu | RAPIDS ships no RBFSampler / random Fourier features; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| rbfsampler | sklearn-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and sklearn-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |
| resample | cuml-gpu | RAPIDS ships no bootstrap; the arm below is SciPy on the CPU and is labeled scipy-cpu |
| resample | scipy-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and scipy-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |
| spectral | cuml-gpu | RAPIDS ships no SpectralClustering estimator; the arm below is scikit-learn on the CPU and is labeled sklearn-cpu |
| spectral | sklearn-cpu | GPU-PATH-ONLY: this box is a GPU vendor's box and sklearn-cpu is their CPU path. On NVIDIA and AMD we compare against the vendor's GPU arm only; the CPU arm is the MacBook's. This lane therefore has no legal opponent here, which is a fact about the vendor's GPU coverage and not a failure of this run. |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

## Notes the arms printed

- `classical.cd.vendor.log`: lane=cd arm=cuml-gpu alpha, max_iter, tol, fit_intercept and selection are all passed explicitly; cuML's tol default is 1e-3 and ours is the fixture's, and they are made equal here
- `classical.cholesky.vendor.log`: lane=cholesky arm=torch-gpu torch.linalg.cholesky is cuSOLVER potrf and torch.cholesky_solve is potrs; the ridge, the logdet and the solve are all inside the clock on both sides
- `classical.dbscan.vendor.log`: lane=dbscan arm=cuml-gpu calc_core_sample_indices=False: our dbscan_fit_impl returns labels only, and computing their core-sample index array would be work our arm does not do
- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu optimizer=None and normalize_y=False on both sides: our gpr_fit_host implements no hyperparameter optimizer (DEVIATION 1761), so an arm that optimized would be doing different work
- `classical.gp.vendor.log`: lane=gp arm=sklearn-cpu scikit-learn works in float64 and we work in float32; that is a real difference in the amount of arithmetic and it cannot be turned off on their side
- `classical.hdbscan.vendor.log`: lane=hdbscan arm=cuml-gpu min_samples, min_cluster_size, metric and cluster_selection_method are passed explicitly. At 96 rows both arms are dominated by launch latency; this is the fixture the lane ships and it has no size knob
- `classical.holtwinters.vendor.log`: lane=holtwinters arm=cuml-gpu seasonal=additive, seasonal_periods, start_periods, ts_num and eps all passed explicitly. The dump is series-major (batch_size x n), which is cuML's own (ts_num, n) layout, so no transpose is needed
- `classical.ivf.vendor.log`: lane=ivf arm=cuvs-gpu ONE build plus ONE search inside the clock, matching ivf_flat_build_and_search_host. metric=sqeuclidean, kmeans_trainset_fraction=1.0, n_probes and n_lists passed explicitly (cuVS defaults n_probes to 20)
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kmeans.vendor.log`: lane=kmeans arm=cuml-gpu output is 16008192 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu cuML returns EUCLIDEAN distances and our arm returns SQUARED ones (is_sqrt=False); sqrt is monotone so the neighbor sets and their order agree, and the extra root is one elementwise pass on their side
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu output is 480000 bytes, above MOJOLEARN_SPEED_HASH_MAX (262144): hash omitted, not computed and discarded
- `classical.kpss.vendor.log`: lane=kpss arm=cuml-gpu cuml.tsa.stationarity.kpss_test at d=1, D=0, s=0, pval_threshold=0.05, the same batch of 8 series
- `classical.krr.vendor.log`: lane=krr arm=cuml-gpu kernel=rbf, gamma and alpha passed explicitly; fit plus predict inside the clock, matching our lane
- `classical.linkage.vendor.log`: lane=linkage arm=cuml-gpu linkage=single connectivity=pairwise metric=euclidean; cuML implements single linkage only, which is the arm our single_linkage was ported from
- `classical.linkage.vendor.log`: lane=linkage arm=cuml-gpu distance keyword on this build is metric=euclidean
- `classical.metrics.vendor.log`: lane=metrics arm=cuml-gpu ELEVEN metrics, the same eleven the Mojo lane times. rand_index is in neither pass: cuML ships no plain Rand index and an arm computing one more metric than the other is not a comparison
- `classical.metrics.vendor.log`: lane=metrics arm=cuml-gpu hash=- for this lane: the eleven return values are Python floats of two different widths on the two sides and folding them would compare formatting, not answers
- `classical.nystroem.vendor.log`: lane=nystroem arm=sklearn-cpu hash=- : the BASIS SAMPLE differs. Ours permutes with a pinned Philox stream and theirs with numpy RandomState, so the two fit different rows. The WORK is the same (sample q rows, form q x q, eigendecompose, scale, cross kernel, matmul) and that is what is being timed
- `classical.ols.vendor.log`: lane=ols arm=cuml-gpu algorithm=eig fit_intercept=False, algorithm-matched with lstsq_eig (normal equations, eigen route)
- `classical.pca.vendor.log`: lane=pca arm=cuml-gpu svd_solver=jacobi, algorithm-matched: their iterative eigendecomposition against our pca_fit, which forms the covariance and runs jacobi_eigh_device on it
- `classical.rbfsampler.vendor.log`: lane=rbfsampler arm=sklearn-cpu hash=- : the random draws differ (a pinned Philox stream against numpy RandomState.normal). Both arms draw a d x q weight matrix and a q offset, then one matmul and one cosine over 3 rows
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu statistic=mean of column 0, method=percentile, n_resamples=10000, confidence_level passed explicitly. Our bootstrap_host resamples ROWS of the two-column sample (SciPy's paired=True shape) and computes the mean of column 0, which is what this arm does
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu with_bca_diagnostics is False on our side, so neither arm computes a jackknife
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu affinity=nearest_neighbors with the same n_neighbors, same n_clusters, same n_components, same n_init: both arms build a kNN affinity, take a Lanczos eigendecomposition of the normalized Laplacian and run k-means on the embedding. The eigensolvers are two Lanczos implementations (ours restarts, theirs is ARPACK)
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu hash=- : cluster label NUMBERING is arbitrary in both and the two are at best a permutation of each other
- `classical.svm.vendor.log`: lane=svm arm=cuml-gpu FIT ONLY. C, gamma, tol and nochange_steps are passed explicitly; cache_size is 0 on our side and left at cuML's default here because a cache size is a memory policy, not a parameter of the answer

