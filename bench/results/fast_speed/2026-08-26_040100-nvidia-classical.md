# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs`, 45 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| cd | cuml-gpu | 2048x16 | 3 | 1.463 | 1.413 | 1.884 | 333.341 | 1708eb34307fb7a0 |
| cd | ours | 2048x16 | 3 | 1.176 | 1.174 | 1.189 | 496.657 | 3bc909a47ebc9bda |
| cholesky | ours | RBF.64x64r4 | 3 | 0.316 | 0.314 | 0.332 | 518.614 | d43ded17a30cf369 |
| cholesky | torch-gpu | RBF.64x64r4 | 3 | 0.215 | 0.195 | 0.409 | 93.809 | 8ec71c2a1e3fbcca |
| dbscan | cuml-gpu | 4000x16 | 3 | 1.304 | 1.252 | 1.643 | 233.321 | 4869c285f078c0a5 |
| dbscan | ours | 4000x16 | 3 | 0.711 | 0.711 | 0.716 | 1.497 | 4869c285f078c0a5 |
| gmm | ours | SEPARATED.24x2K3 | 3 | 2.390 | 2.382 | 2.667 | 665.663 | b934e26ee2bc5277 |
| gp | ours | ard.12x3s6 | 3 | 0.543 | 0.511 | 0.597 | 507.215 | fb8d7954ab8a7807 |
| hdbscan | cuml-gpu | blobs96.96x4 | 3 | 5.503 | 5.021 | 5.743 | 493.392 | 5199770714da8525 |
| hdbscan | ours | blobs96.96x4 | 3 | 1.311 | 1.287 | 1.381 | 681.648 | 8e1998063cb24ce6 |
| holtwinters | cuml-gpu | 7x72f12 | 3 | 8.996 | 8.861 | 9.423 | 381.802 | 97c0d626da007a91 |
| holtwinters | ours | 7x72f12 | 3 | 10.496 | 10.494 | 10.503 | 509.641 | 4fb56da659c4ec4b |
| ivf | cuvs-gpu | 512x8q64L8p3k8 | 3 | 21.531 | 20.756 | 23.495 | 479.339 | cb8f1cef2816d70f |
| ivf | ours | 512x8q64L8p3k8 | 3 | 8.755 | 8.605 | 9.049 | 716.734 | 0908d00a25ccbed4 |
| kde | cuml-gpu | 1024x256x8 | 3 | 0.462 | 0.440 | 0.659 | 29.744 | 1a75f5b84752fcd8 |
| kde | ours | 1024x256x8 | 3 | 0.207 | 0.205 | 0.244 | 0.531 | 4ec9d98c61d3ae83 |
| kmeans | cuml-gpu | 4000000x32k64i20 | 3 | 236.789 | 236.782 | 238.013 | 4029.594 | - |
| kmeans | ours | 4000000x32k64i20 | 3 | 155.067 | 154.767 | 155.119 | 155.559 | c5f012ba7bff508e |
| knn | cuml-gpu | 400000x32q4000k10 | 3 | 10.822 | 10.260 | 10.892 | 52.314 | - |
| knn | ours | 400000x32q4000k10 | 3 | 233.827 | 232.877 | 233.974 | 359.588 | MOVED(3) |
| kpss | cuml-gpu | 8x520 | 3 | 0.572 | 0.554 | 0.724 | 183.178 | 7270ce3a3ef261c5 |
| kpss | ours | 8x520 | 3 | 0.128 | 0.127 | 0.132 | 521.438 | 91a858d57a2f7156 |
| krr | cuml-gpu | RBF.16x5 | 3 | 2.256 | 2.212 | 2.634 | 1348.565 | bf101d3b60c2f648 |
| krr | ours | RBF.16x5 | 3 | 0.336 | 0.307 | 0.523 | 644.150 | 7a4f6340f5352471 |
| linkage | cuml-gpu | blobs_dups.102x5 | 3 | 2.557 | 2.521 | 2.800 | 3341.870 | 1ff4250dd4aab576 |
| linkage | ours | blobs_dups.102x5 | 3 | 0.986 | 0.972 | 1.047 | 133.320 | d486f76f518b97e2 |
| metrics | cuml-gpu | lab2053.flt2053.sil521x4.tru301x6 | 3 | 13.437 | 13.384 | 15.920 | 9028.270 | - |
| metrics | ours | lab2053.flt2053.sil521x4.tru301x6 | 3 | 0.832 | 0.826 | 0.874 | 646.842 | 53c1c906884da70e |
| nystroem | ours | RBF.16x5q8 | 3 | 0.424 | 0.404 | 0.591 | 620.817 | 6fc983bed8f114ea |
| ols | cuml-gpu | 4000000x32 | 3 | 176.938 | 176.621 | 177.806 | 825.213 | 6a660f0646505394 |
| ols | ours | 4000000x32 | 3 | 8.607 | 8.603 | 8.626 | 129.013 | 1bd9a4419821cd59 |
| pca | ours | 4000000x32c8 | 3 | 9.327 | 9.326 | 9.369 | 131.907 | 5ff46ab4c9ad7cac |
| rbfsampler | ours | 3x5q8 | 3 | 0.295 | 0.154 | 0.486 | 467.434 | b20f76c0e12fe7fd |
| resample | ours | 200x2r10000 | 3 | 0.595 | 0.579 | 0.627 | 507.870 | 38bb5e267f2ef221 |
| spectral | ours | 48x4c3 | 3 | 6.803 | 6.785 | 6.818 | 649.911 | 553b52e4f09df7cb |
| svm | cuml-gpu | F2.xor.240x2 | 3 | 5.243 | 5.138 | 6.175 | 4371.879 | 28f0580866fb9354 |
| svm | ours | F2.xor.240x2 | 3 | 1.447 | 1.438 | 1.473 | 658.623 | 89a8b6ec1d86118a |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| cd | 2048x16 | cuml-gpu | 1.176 | 1.463 | 0.80 | we are 1.24x FASTER |
| cholesky | RBF.64x64r4 | torch-gpu | 0.316 | 0.215 | 1.47 | we are 1.47x SLOWER |
| dbscan | 4000x16 | cuml-gpu | 0.711 | 1.304 | 0.54 | we are 1.83x FASTER |
| gmm | SEPARATED.24x2K3 | (none ran) | 2.390 | - | - | NO OPPONENT ON THIS BOX |
| gp | ard.12x3s6 | (none ran) | 0.543 | - | - | NO OPPONENT ON THIS BOX |
| hdbscan | blobs96.96x4 | cuml-gpu | 1.311 | 5.503 | 0.24 | we are 4.20x FASTER |
| holtwinters | 7x72f12 | cuml-gpu | 10.496 | 8.996 | 1.17 | we are 1.17x SLOWER |
| ivf | 512x8q64L8p3k8 | cuvs-gpu | 8.755 | 21.531 | 0.41 | we are 2.46x FASTER |
| kde | 1024x256x8 | cuml-gpu | 0.207 | 0.462 | 0.45 | we are 2.24x FASTER |
| kmeans | 4000000x32k64i20 | cuml-gpu | 155.067 | 236.789 | 0.65 | we are 1.53x FASTER |
| knn | 400000x32q4000k10 | cuml-gpu | 233.827 | 10.822 | 21.61 | we are 21.61x SLOWER |
| kpss | 8x520 | cuml-gpu | 0.128 | 0.572 | 0.22 | we are 4.47x FASTER |
| krr | RBF.16x5 | cuml-gpu | 0.336 | 2.256 | 0.15 | we are 6.71x FASTER |
| linkage | blobs_dups.102x5 | cuml-gpu | 0.986 | 2.557 | 0.39 | we are 2.59x FASTER |
| metrics | lab2053.flt2053.sil521x4.tru301x6 | cuml-gpu | 0.832 | 13.437 | 0.06 | we are 16.14x FASTER |
| nystroem | RBF.16x5q8 | (none ran) | 0.424 | - | - | NO OPPONENT ON THIS BOX |
| ols | 4000000x32 | cuml-gpu | 8.607 | 176.938 | 0.05 | we are 20.56x FASTER |
| pca | 4000000x32c8 | (none ran) | 9.327 | - | - | NO OPPONENT ON THIS BOX |
| rbfsampler | 3x5q8 | (none ran) | 0.295 | - | - | NO OPPONENT ON THIS BOX |
| resample | 200x2r10000 | (none ran) | 0.595 | - | - | NO OPPONENT ON THIS BOX |
| spectral | 48x4c3 | (none ran) | 6.803 | - | - | NO OPPONENT ON THIS BOX |
| svm | F2.xor.240x2 | cuml-gpu | 1.447 | 5.243 | 0.28 | we are 3.62x FASTER |

## Scale UNKNOWN for every row in this run

No arm in these logs emitted `scale=` in its FSPEED header, so
this run is OLDER than the throughput/fixed-cost declaration and
the split below could not be made. The rows are listed under
FIXED-COST because that is the conservative bucket, but the label
is NOT a measurement here -- some of these lanes are genuinely
large (kmeans ships 4,000,000 x 32) and some are genuinely tiny
(krr ships 16 rows). Check each shape tag by hand before quoting
anything from this file, or re-run so the arms declare it.

## Rows, ranked -- scale UNDECLARED, check each shape by hand

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
| 1 | knn | 400000x32q4000k10 | 233.827 | cuml-gpu | 10.822 | **21.61x SLOWER** |
| 2 | cholesky | RBF.64x64r4 | 0.316 | torch-gpu | 0.215 | **1.47x SLOWER** |
| 3 | holtwinters | 7x72f12 | 10.496 | cuml-gpu | 8.996 | **1.17x SLOWER** |
| 4 | cd | 2048x16 | 1.176 | cuml-gpu | 1.463 | **1.24x FASTER** |
| 5 | kmeans | 4000000x32k64i20 | 155.067 | cuml-gpu | 236.789 | **1.53x FASTER** |
| 6 | dbscan | 4000x16 | 0.711 | cuml-gpu | 1.304 | **1.83x FASTER** |
| 7 | kde | 1024x256x8 | 0.207 | cuml-gpu | 0.462 | **2.24x FASTER** |
| 8 | ivf | 512x8q64L8p3k8 | 8.755 | cuvs-gpu | 21.531 | **2.46x FASTER** |
| 9 | linkage | blobs_dups.102x5 | 0.986 | cuml-gpu | 2.557 | **2.59x FASTER** |
| 10 | svm | F2.xor.240x2 | 1.447 | cuml-gpu | 5.243 | **3.62x FASTER** |
| 11 | hdbscan | blobs96.96x4 | 1.311 | cuml-gpu | 5.503 | **4.20x FASTER** |
| 12 | kpss | 8x520 | 0.128 | cuml-gpu | 0.572 | **4.47x FASTER** |
| 13 | krr | RBF.16x5 | 0.336 | cuml-gpu | 2.256 | **6.71x FASTER** |
| 14 | metrics | lab2053.flt2053.sil521x4.tru301x6 | 0.832 | cuml-gpu | 13.437 | **16.14x FASTER** |
| 15 | ols | 4000000x32 | 8.607 | cuml-gpu | 176.938 | **20.56x FASTER** |

15 rows in total have an opponent: 0 throughput, 15 fixed-cost.

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
| pca | unknown | raised at run time: ValueError("Expected `svd_solver` to be one of ['auto', 'full', 'jacobi'], got 'covariance_eigh'") |
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

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| knn | ours | 400000x32q4000k10 | 3 |

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
- `classical.knn.ours.log`: lane=knn arm=ours hash moved across rounds: d8462c5bf79c3796 21eba8645a77dd02
- `classical.knn.vendor.log`: lane=knn arm=cuml-gpu cuML returns EUCLIDEAN distances and our arm returns SQUARED ones (is_sqrt=False); sqrt is monotone so the neighbor sets and their order agree, and the extra root is one elementwise pass on their side
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
- `classical.pca.vendor.log`: lane=pca arm=cuml-gpu svd_solver=covariance_eigh, algorithm-matched
- `classical.rbfsampler.vendor.log`: lane=rbfsampler arm=sklearn-cpu hash=- : the random draws differ (a pinned Philox stream against numpy RandomState.normal). Both arms draw a d x q weight matrix and a q offset, then one matmul and one cosine over 3 rows
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu statistic=mean of column 0, method=percentile, n_resamples=10000, confidence_level passed explicitly. Our bootstrap_host resamples ROWS of the two-column sample (SciPy's paired=True shape) and computes the mean of column 0, which is what this arm does
- `classical.resample.vendor.log`: lane=resample arm=scipy-cpu with_bca_diagnostics is False on our side, so neither arm computes a jackknife
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu affinity=nearest_neighbors with the same n_neighbors, same n_clusters, same n_components, same n_init: both arms build a kNN affinity, take a Lanczos eigendecomposition of the normalized Laplacian and run k-means on the embedding. The eigensolvers are two Lanczos implementations (ours restarts, theirs is ARPACK)
- `classical.spectral.vendor.log`: lane=spectral arm=sklearn-cpu hash=- : cluster label NUMBERING is arbitrary in both and the two are at best a permutation of each other
- `classical.svm.vendor.log`: lane=svm arm=cuml-gpu FIT ONLY. C, gamma, tol and nochange_steps are passed explicitly; cache_size is 0 on our side and left at cuML's default here because a cache size is a memory policy, not a parameter of the answer

