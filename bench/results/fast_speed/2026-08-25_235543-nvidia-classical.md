# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-25_235543-nvidia-speed-classical/remote/logs`, 45 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| cd | cuml-gpu | 2048x16 | 3 | 1.438 | 1.302 | 1.629 | 317.842 | 1708eb34307fb7a0 |
| cd | ours | 2048x16 | 3 | 1.139 | 1.139 | 1.162 | 425.331 | 3bc909a47ebc9bda |
| cholesky | ours | RBF.64x64r4 | 3 | 0.311 | 0.310 | 0.316 | 408.921 | d43ded17a30cf369 |
| cholesky | torch-gpu | RBF.64x64r4 | 3 | 0.179 | 0.168 | 0.304 | 82.174 | 8ec71c2a1e3fbcca |
| dbscan | cuml-gpu | 4000x16 | 3 | 2.412 | 2.279 | 2.465 | 520.174 | 4869c285f078c0a5 |
| dbscan | ours | 4000x16 | 3 | 0.693 | 0.692 | 0.703 | 1.364 | 4869c285f078c0a5 |
| gmm | ours | SEPARATED.24x2K3 | 3 | 2.437 | 2.410 | 2.441 | 524.895 | b934e26ee2bc5277 |
| gp | ours | ard.12x3s6 | 3 | 0.499 | 0.482 | 0.561 | 408.017 | fb8d7954ab8a7807 |
| hdbscan | cuml-gpu | blobs96.96x4 | 3 | 4.481 | 4.395 | 4.782 | 432.490 | 5199770714da8525 |
| hdbscan | ours | blobs96.96x4 | 3 | 1.225 | 1.202 | 1.285 | 504.607 | 8e1998063cb24ce6 |
| holtwinters | cuml-gpu | 7x72f12 | 3 | 8.288 | 8.223 | 8.758 | 373.209 | 97c0d626da007a91 |
| holtwinters | ours | 7x72f12 | 3 | 10.422 | 10.417 | 10.426 | 425.105 | 4fb56da659c4ec4b |
| ivf | cuvs-gpu | 512x8q64L8p3k8 | 3 | 10.412 | 10.146 | 10.636 | 363.678 | cb8f1cef2816d70f |
| ivf | ours | 512x8q64L8p3k8 | 3 | 7.223 | 7.206 | 7.533 | 535.900 | 0908d00a25ccbed4 |
| kde | cuml-gpu | 1024x256x8 | 3 | 0.408 | 0.384 | 0.629 | 45.243 | 1a75f5b84752fcd8 |
| kde | ours | 1024x256x8 | 3 | 0.202 | 0.199 | 0.226 | 0.465 | 4ec9d98c61d3ae83 |
| kmeans | cuml-gpu | 4000000x32k64i20 | 3 | 233.434 | 231.537 | 234.459 | 3056.490 | - |
| kmeans | ours | 4000000x32k64i20 | 3 | 167.145 | 167.080 | 167.628 | 168.609 | c5f012ba7bff508e |
| knn | cuml-gpu | 400000x32q4000k10 | 3 | 10.224 | 9.994 | 10.406 | 48.325 | - |
| knn | ours | 400000x32q4000k10 | 3 | 243.862 | 243.711 | 244.249 | 409.580 | MOVED(3) |
| kpss | cuml-gpu | 8x520 | 3 | 0.759 | 0.716 | 1.220 | 292.670 | 7270ce3a3ef261c5 |
| kpss | ours | 8x520 | 3 | 0.126 | 0.126 | 0.128 | 410.350 | 91a858d57a2f7156 |
| krr | cuml-gpu | RBF.16x5 | 3 | 1.918 | 1.874 | 2.349 | 1165.801 | bf101d3b60c2f648 |
| krr | ours | RBF.16x5 | 3 | 0.315 | 0.290 | 0.480 | 536.693 | 7a4f6340f5352471 |
| linkage | cuml-gpu | blobs_dups.102x5 | 3 | 2.308 | 2.290 | 2.554 | 2578.711 | 1ff4250dd4aab576 |
| linkage | ours | blobs_dups.102x5 | 3 | 0.973 | 0.967 | 1.048 | 112.313 | d486f76f518b97e2 |
| metrics | cuml-gpu | lab2053.flt2053.sil521x4.tru301x6 | 3 | 11.344 | 11.229 | 14.112 | 6984.392 | - |
| metrics | ours | lab2053.flt2053.sil521x4.tru301x6 | 3 | 0.839 | 0.838 | 0.869 | 528.960 | 53c1c906884da70e |
| nystroem | ours | RBF.16x5q8 | 3 | 0.395 | 0.378 | 0.562 | 516.150 | 6fc983bed8f114ea |
| ols | cuml-gpu | 4000000x32 | 3 | 173.848 | 173.675 | 175.366 | 720.321 | 6a660f0646505394 |
| rbfsampler | ours | 3x5q8 | 3 | 0.143 | 0.131 | 0.280 | 404.047 | b20f76c0e12fe7fd |
| resample | ours | 200x2r10000 | 3 | 0.551 | 0.517 | 0.746 | 405.009 | 38bb5e267f2ef221 |
| spectral | ours | 48x4c3 | 3 | 6.772 | 6.760 | 6.797 | 514.726 | 553b52e4f09df7cb |
| svm | cuml-gpu | F2.xor.240x2 | 3 | 4.868 | 4.831 | 4.890 | 3372.206 | 28f0580866fb9354 |
| svm | ours | F2.xor.240x2 | 3 | 1.368 | 1.355 | 1.384 | 524.688 | 89a8b6ec1d86118a |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| cd | 2048x16 | cuml-gpu | 1.139 | 1.438 | 0.79 | we are 1.26x FASTER |
| cholesky | RBF.64x64r4 | torch-gpu | 0.311 | 0.179 | 1.74 | we are 1.74x SLOWER |
| dbscan | 4000x16 | cuml-gpu | 0.693 | 2.412 | 0.29 | we are 3.48x FASTER |
| gmm | SEPARATED.24x2K3 | (none ran) | 2.437 | - | - | NO OPPONENT ON THIS BOX |
| gp | ard.12x3s6 | (none ran) | 0.499 | - | - | NO OPPONENT ON THIS BOX |
| hdbscan | blobs96.96x4 | cuml-gpu | 1.225 | 4.481 | 0.27 | we are 3.66x FASTER |
| holtwinters | 7x72f12 | cuml-gpu | 10.422 | 8.288 | 1.26 | we are 1.26x SLOWER |
| ivf | 512x8q64L8p3k8 | cuvs-gpu | 7.223 | 10.412 | 0.69 | we are 1.44x FASTER |
| kde | 1024x256x8 | cuml-gpu | 0.202 | 0.408 | 0.50 | we are 2.02x FASTER |
| kmeans | 4000000x32k64i20 | cuml-gpu | 167.145 | 233.434 | 0.72 | we are 1.40x FASTER |
| knn | 400000x32q4000k10 | cuml-gpu | 243.862 | 10.224 | 23.85 | we are 23.85x SLOWER |
| kpss | 8x520 | cuml-gpu | 0.126 | 0.759 | 0.17 | we are 6.01x FASTER |
| krr | RBF.16x5 | cuml-gpu | 0.315 | 1.918 | 0.16 | we are 6.09x FASTER |
| linkage | blobs_dups.102x5 | cuml-gpu | 0.973 | 2.308 | 0.42 | we are 2.37x FASTER |
| metrics | lab2053.flt2053.sil521x4.tru301x6 | cuml-gpu | 0.839 | 11.344 | 0.07 | we are 13.52x FASTER |
| nystroem | RBF.16x5q8 | (none ran) | 0.395 | - | - | NO OPPONENT ON THIS BOX |
| rbfsampler | 3x5q8 | (none ran) | 0.143 | - | - | NO OPPONENT ON THIS BOX |
| resample | 200x2r10000 | (none ran) | 0.551 | - | - | NO OPPONENT ON THIS BOX |
| spectral | 48x4c3 | (none ran) | 6.772 | - | - | NO OPPONENT ON THIS BOX |
| svm | F2.xor.240x2 | cuml-gpu | 1.368 | 4.868 | 0.28 | we are 3.56x FASTER |

## Where we lose most, ranked

Against the FASTEST opponent that actually ran on each row, because
losing 70x to a TF32 arm and 12x to an FP32 arm is one fact and not
two. This is the optimization queue: it is a measurement, not an
intuition about which kernel feels slow.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | knn | 400000x32q4000k10 | 243.862 | cuml-gpu | 10.224 | **23.85x SLOWER** |
| 2 | cholesky | RBF.64x64r4 | 0.311 | torch-gpu | 0.179 | **1.74x SLOWER** |
| 3 | holtwinters | 7x72f12 | 10.422 | cuml-gpu | 8.288 | **1.26x SLOWER** |
| 4 | cd | 2048x16 | 1.139 | cuml-gpu | 1.438 | **1.26x FASTER** |
| 5 | kmeans | 4000000x32k64i20 | 167.145 | cuml-gpu | 233.434 | **1.40x FASTER** |
| 6 | ivf | 512x8q64L8p3k8 | 7.223 | cuvs-gpu | 10.412 | **1.44x FASTER** |
| 7 | kde | 1024x256x8 | 0.202 | cuml-gpu | 0.408 | **2.02x FASTER** |
| 8 | linkage | blobs_dups.102x5 | 0.973 | cuml-gpu | 2.308 | **2.37x FASTER** |
| 9 | dbscan | 4000x16 | 0.693 | cuml-gpu | 2.412 | **3.48x FASTER** |
| 10 | svm | F2.xor.240x2 | 1.368 | cuml-gpu | 4.868 | **3.56x FASTER** |
| 11 | hdbscan | blobs96.96x4 | 1.225 | cuml-gpu | 4.481 | **3.66x FASTER** |
| 12 | kpss | 8x520 | 0.126 | cuml-gpu | 0.759 | **6.01x FASTER** |
| 13 | krr | RBF.16x5 | 0.315 | cuml-gpu | 1.918 | **6.09x FASTER** |
| 14 | metrics | lab2053.flt2053.sil521x4.tru301x6 | 0.839 | cuml-gpu | 11.344 | **13.52x FASTER** |

**11 of 14 rows with an opponent are wins for us.**

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
- `classical.knn.ours.log`: lane=knn arm=ours hash moved across rounds: 60696ef51c0f93d6 a8b524266ef7e58e
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

