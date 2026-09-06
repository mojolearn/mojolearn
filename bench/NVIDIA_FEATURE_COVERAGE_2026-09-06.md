# NVIDIA feature coverage — September 6, 2026

**The requested current FAST / IDENTICAL / one-external matrix is incomplete.** Historical NVIDIA measurements exist for many families, but most lack an IDENTICAL timing arm, current source, or admission checks. This inventory records retained evidence; its author ran no builds, tests, timings or provisioning. Root alone may execute the queued work.

Machine-readable inventory: [NVIDIA_FEATURE_COVERAGE_2026-09-06.json](NVIDIA_FEATURE_COVERAGE_2026-09-06.json). Each row selects exactly one comparator. Historical actual comparators remain recorded without relabeling; replacing a historical baseline requires a fresh three-arm session. Public/native family coverage does not imply every parameter is implemented or certified.

## Meaning of the cells

Every current-source FAST and IDENTICAL measurement cell is missing in this audit. “Stale” preserves a real older measurement; “invalid” preserves failed execution/admission; “inapplicable” marks absent equivalent GPU coverage or I/O. A CPU fallback is explicitly labeled and supplies no GPU comparison. Selected APIs must be checked against the installed external package before use.

External correctness uses a stated tolerance or quality gate. Mojo IDENTICAL means named cross-vendor raw-byte fixtures with matching provenance, not CatBoost/cuML bitwise equality. Historical identity cards are recorded separately in the JSON; no row claims all configurations are certified. Identity sources: [support matrix](../SUPPORT_MATRIX.md), [path ledger](../IDENTITY_PATHS.md).

## Feature inventory

| Priority | Feature | One selected comparator | Retained timing status / hole | Identity scope |
|---|---|---|---|---|
| P0 | Symmetric GBDT plain fit | CatBoost GPU (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.gbdt-symmetric.r1000000.log) | historical_fixture_only |
| P2 | RandomForestClassifier fit | cuML RandomForestClassifier (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.rf.r1000000.log) | historical_fixture_only |
| P2 | GBDT depthwise fit | XGBoost GPU (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.gbdt-depthwise.r1000000.log) | historical_fixture_only |
| P2 | GBDT lossguide fit | XGBoost GPU (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.gbdt-lossguide.r1000000.log) | historical_fixture_only |
| P0 | Symmetric GBDT prediction | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | OrderedRMSE single-permutation fit/predict | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | Categorical/CTR FeatureFreq and weighted partitions | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | Symmetric-tree ranking objectives and derivatives | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | Other supported symmetric-tree losses, gradients and Hessians | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | Symmetric-tree bootstrap, weights, leaf estimation and score configuration | CatBoost GPU (CUDA) | missing | scoped_partial |
| P0 | GBDT/OrderedRMSE model save/load and restored predictions | CatBoost GPU (CUDA) | missing | scoped_partial |
| P2 | RandomForestRegressor fit/predict | cuML RandomForestRegressor (CUDA) | missing | historical_fixture_only |
| P2 | RandomForestClassifier prediction | cuML RandomForestClassifier (CUDA) | missing | historical_fixture_only |
| P2 | ExtraTreesClassifier fit/predict | LightGBM CUDA RF extra_trees=true (CUDA) | FAST stale, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.et.r1000000.log) | historical_fixture_only |
| P2 | ExtraTreesRegressor fit/predict | LightGBM CUDA RF extra_trees=true (CUDA) | FAST stale, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-08-28_030908-nvidia-speed-forest/remote/logs/forest.et.r1000000.log) | historical_fixture_only |
| P2 | IsolationForest fit/score | scikit-learn IsolationForest (CPU) | missing; GPU external inapplicable | historical_fixture_only |
| P1 | NearestNeighbors kNN search | cuML NearestNeighbors (CUDA) | FAST stale, IDENTICAL stale, external stale — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/final-supplement/torch-knn-32/results.json) | historical_fixture_only |
| P2 | KNeighborsClassifier fit/query | cuML KNeighborsClassifier (CUDA) | missing | historical_fixture_only |
| P2 | KNeighborsRegressor fit/query | cuML KNeighborsRegressor (CUDA) | missing | historical_fixture_only |
| P2 | RadiusNeighbors fit/query | cuML NearestNeighbors radius_neighbors (CUDA) | missing | historical_fixture_only |
| P2 | KMeans | cuML KMeans (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.kmeans.ours.log) | historical_fixture_only |
| P2 | DBSCAN | cuML DBSCAN (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.dbscan.ours.log) | historical_fixture_only |
| P2 | SVC | cuML SVC (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.svm.ours.log) | historical_fixture_only |
| P2 | LinearRegression | cuML LinearRegression (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.ols.ours.log) | historical_fixture_only |
| P2 | ElasticNet | cuML ElasticNet (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.cd.ours.log) | historical_fixture_only |
| P2 | KernelDensity | cuML KernelDensity (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.kde.ours.log) | historical_fixture_only |
| P2 | AgglomerativeClustering | cuML AgglomerativeClustering (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.linkage.ours.log) | historical_fixture_only |
| P2 | ExponentialSmoothing | cuML ExponentialSmoothing (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.holtwinters.ours.log) | nvidia_pending |
| P2 | KPSS and select_d | cuML stationarity tests (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.kpss.ours.log) | nvidia_pending |
| P2 | HDBSCAN | cuML HDBSCAN (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.hdbscan.ours.log) | historical_fixture_only |
| P2 | IVF approximate neighbors | cuVS IVF (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.ivf.ours.log) | historical_fixture_only |
| P2 | Kernel ridge | cuML KernelRidge (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.krr.ours.log) | historical_fixture_only |
| P2 | Cholesky solve | PyTorch CUDA linalg (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.cholesky.ours.log) | historical_fixture_only |
| P2 | PCA fit/transform | cuML PCA (CUDA) | FAST stale, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.pca.vendor.log) | historical_fixture_only |
| P2 | Lasso | cuML Lasso (CUDA) | missing | historical_fixture_only |
| P2 | Ridge | cuML Ridge (CUDA) | missing | historical_fixture_only |
| P2 | LogisticRegression | cuML LogisticRegression (CUDA) | missing | historical_fixture_only |
| P2 | SVR | cuML SVR (CUDA) | missing | historical_fixture_only |
| P2 | TruncatedSVD | cuML TruncatedSVD (CUDA) | missing | historical_fixture_only |
| P2 | ARIMA fit/filter/forecast | cuML ARIMA (CUDA) | missing | historical_fixture_only |
| P2 | GaussianProcessRegressor | scikit-learn GaussianProcessRegressor (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.gp.vendor.log); GPU external inapplicable | nvidia_pending |
| P2 | SpectralClustering | scikit-learn SpectralClustering (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.spectral.vendor.log); GPU external inapplicable | nvidia_pending |
| P2 | Gaussian mixture | scikit-learn GaussianMixture (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.gmm.vendor.log); GPU external inapplicable | unverified_in_this_audit |
| P2 | Nystroem | scikit-learn Nystroem (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.nystroem.vendor.log); GPU external inapplicable | unverified_in_this_audit |
| P2 | Random Fourier features | scikit-learn RBFSampler (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.rbfsampler.vendor.log); GPU external inapplicable | unverified_in_this_audit |
| P2 | Bootstrap resampling | SciPy bootstrap (CPU) | FAST stale, IDENTICAL missing, external inapplicable — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.resample.vendor.log); GPU external inapplicable | unverified_in_this_audit |
| P2 | Label metrics | cuML metrics (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.metrics.ours.log) | historical_fixture_only |
| P2 | Regression/KL metrics | cuML metrics (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.metrics.ours.log) | historical_fixture_only |
| P2 | Silhouette | cuML metrics (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.metrics.ours.log) | historical_fixture_only |
| P2 | Trustworthiness | cuML metrics (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-26_040100-nvidia-speed-classical/remote/logs/classical.metrics.ours.log) | historical_fixture_only |
| P1 | GEMV | PyTorch CUDA strict FP32 (CUDA) | FAST stale, IDENTICAL stale, external stale — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/public-gemv/results.json) | historical_fixture_only |
| P1 | NT | PyTorch CUDA strict FP32 (CUDA) | FAST stale, IDENTICAL stale, external stale — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/public-nt/results.json) | historical_fixture_only |
| P1 | GRAM | PyTorch CUDA strict FP32 (CUDA) | FAST stale, IDENTICAL stale, external stale — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/public-gram/results.json) | historical_fixture_only |
| P2 | General GEMM orientations/shapes | PyTorch CUDA matmul (CUDA) | FAST stale, IDENTICAL missing, external stale — [evidence](../bench/results/e1g/2026-08-28_040316-nvidia-speed-gemmseq/remote/logs/gemm.gemm.ours.log) | historical_fixture_only |
| P2 | Other exposed linalg reductions/solves | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Embedding forward | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Embedding backward | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Training losses forward | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Training loss derivatives | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Optimizer update/state | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Composed training forward/backward/update | PyTorch CUDA (CUDA) | missing | local_or_scoped_only |
| P2 | Training checkpoints save/load | PyTorch serialization (CPU/storage) | missing; GPU external inapplicable | local_or_scoped_only |
| P1 | MAMBA1 forward | mamba-ssm CUDA (CUDA) | FAST stale, IDENTICAL stale, external stale — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/seq-mamba-fast-1.log) | scoped_partial |
| P1 | MAMBA1 backward | mamba-ssm CUDA (CUDA) | FAST missing, IDENTICAL missing, external missing — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/last-checks/mamba1/result.json) | scoped_partial |
| P1 | MAMBA1 state_decode | mamba-ssm CUDA (CUDA) | missing | scoped_partial |
| P1 | MAMBA2 forward | mamba-ssm CUDA (CUDA) | FAST missing, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/last-checks/mamba2/result.json) | scoped_partial |
| P1 | MAMBA2 backward | mamba-ssm CUDA (CUDA) | FAST missing, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/last-checks/mamba2/result.json) | scoped_partial |
| P1 | MAMBA2 state_decode | mamba-ssm CUDA (CUDA) | missing | scoped_partial |
| P1 | MAMBA3 forward | upstream Mamba3 CUDA/Triton (CUDA) | FAST missing, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/last-checks/results.tsv) | scoped_partial |
| P1 | MAMBA3 backward | upstream Mamba3 CUDA/Triton (CUDA) | FAST missing, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/last-checks/results.tsv) | scoped_partial |
| P1 | MAMBA3 state_decode | upstream Mamba3 CUDA/Triton (CUDA) | missing | scoped_partial |
| P1 | Transformer forward | PyTorch CUDA strict FP32 (CUDA) | FAST invalid, IDENTICAL invalid, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/results.tsv) | scoped_partial |
| P1 | Transformer backward | PyTorch CUDA strict FP32 (CUDA) | missing | scoped_partial |
| P1 | Transformer state_decode | PyTorch CUDA strict FP32 (CUDA) | missing | scoped_partial |
| P1 | Transformer attention | PyTorch CUDA strict FP32 (CUDA) | missing | scoped_partial |
| P1 | Transformer mlp | PyTorch CUDA strict FP32 (CUDA) | missing | scoped_partial |
| P1 | Transformer rmsnorm | PyTorch CUDA strict FP32 (CUDA) | missing | scoped_partial |
| P2 | Mamba1 selective scan primitive | mamba-ssm CUDA (CUDA) | FAST stale, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-08-28_040316-nvidia-speed-gemmseq/remote/logs/seq.selective_scan.torch.log) | historical_fixture_only |
| P1 | UMAP fit/fit_transform | cuML UMAP (CUDA) | FAST missing, IDENTICAL missing, external invalid — [evidence](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/results.tsv) | scoped_partial |
| P1 | UMAP held-out transform | cuML UMAP (CUDA) | missing | scoped_partial |

## Evidence that can be reused without exaggeration

- September 5 RTX4090 campaign, base `1d1f8fdd` plus retained supplemental files: seven-round public GEMV, NT, Gram and kNN comparisons passed admission. kNN actual comparator was PyTorch CUDA because cuML installation timed out. These include host transfer/output costs. [Campaign record](results/e1g/2026-09-05_074536-nvidia-mamba/README.md).
- August 28 H100 forest run at `3d65d706`: real FAST/CatBoost GPU symmetric-tree and FAST/cuML RF measurements at HIGGS 1m/2m/5m with five rounds and AUC/logloss; no IDENTICAL timing arm. ExtraTrees LightGBM CUDA failed because its build lacked CUDA. These are historical partial comparisons, not current certification.
- August 26 H100 classical run: actual FAST/GPU samples for KMeans, DBSCAN, SVC, OLS, coordinate descent, KDE, linkage, Holt-Winters, KPSS, metrics, HDBSCAN, IVF, kernel ridge and Cholesky. Only three rounds, separate arm runs, no IDENTICAL or accuracy admission. PCA external refused its unsupported solver.
- September 5 Mamba1 forward timing passed on `mamba130m.prefill.t8`; small whole-block forward/backward upstream agreement is correctness evidence. Mamba2 upstream attempt failed stride constraints; Mamba3 timed out. Transformer timing failed numerical admission. No accepted backward throughput matrix follows from native gradient certificates.
- Current OrderedRMSE and UMAP AMD/NVIDIA installed identity comparison is [retained separately](results/resume/2026-09-06-installed-gap-closure/installed-lane-comparison.json). It does not approve a failed whole-wheel NVIDIA candidate or establish CatBoost/cuML parity.

| RTX4090 public fixture | FAST ms | IDENTICAL ms | Actual PyTorch CUDA ms |
|---|---:|---:|---:|
| kNN 100k index, q32,d32,k10 | 3.981918 | 8.165275 | 1.107609 |
| kNN q128 | 4.516717 | 23.475313 | 1.386469 |
| kNN q1000 | 6.850315 | 215.468543 | 5.638816 |
| GEMV 2048 squared | 1.680231 | 1.718041 | 1.067451 |
| NT 16384x64x64 | 2.040811 | 1.523541 | 0.884171 |
| Gram 65536x32 | 3.785673 | 3.776442 | 1.096630 |

NT has IDENTICAL faster than FAST; investigate under the performance policy rather than weakening IDENTICAL. Do not merge these public-API timings with kernel-only identity-cost runs from a different session.

## Root-only queue and existing commands

Commands below are source-inspected recipes, **not executed results**. First freeze source and qualify the exact installed NVIDIA artifact. Use one job at a time, no subagent measurement, and this CPU cap in the measurement shell:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MAX_JOBS=2 MOJOLEARN_CPU_THREADS=2
```

1. P0: close symmetric-tree support/correctness and certify forward derivatives/leaf calculations, ordered/ranking/categorical slices against named fixtures. Repair the forest runner before timings: its current FAST-only selection does not satisfy the target.
2. P1: reuse the ready public three-arm runners for bounded kNN, UMAP and linear-algebra gaps; repair sequence admission before quoting timings.
3. P2: extend classical and forest adapters with IDENTICAL and one selected external; add missing estimator-specific fixtures. A primitive timing does not qualify every estimator using it.

### public_knn — existing_three_arm_runner

```sh
python tools/nvidia_public_compare.py --lane knn --knn-external cuml --index 100000 --queries 32 --k 10 --rounds 7 --out /tmp/mojolearn-current-knn-q32
```

Repeat sequentially with queries128/1000 and distinct output directories; same installed source.

### public_gemv — existing_three_arm_runner

```sh
python tools/nvidia_public_compare.py --lane gemv --dim 2048 --rounds 7 --out /tmp/mojolearn-current-gemv
```

### public_nt — existing_three_arm_runner

```sh
python tools/nvidia_public_compare.py --lane nt --rows 16384 --rounds 7 --out /tmp/mojolearn-current-nt
```

### public_gram — existing_three_arm_runner

```sh
python tools/nvidia_public_compare.py --lane gram --gram-rows 65536 --rounds 7 --out /tmp/mojolearn-current-gram
```

### public_umap — existing_three_arm_runner

```sh
python tools/nvidia_public_compare.py --lane umap --umap-rows 256 --umap-epochs 50 --rounds 7 --out /tmp/mojolearn-current-umap256
```

Require installed cuML and quality admission; start256, then1024 only if admitted.

### forest_adapter — implementation_required_before_measurement

```sh
MOJOLEARN_SPEED_SIZE=smoke MOJOLEARN_SPEED_ROUNDS=7 python bench/speed/forest_speed_arm.py --lane gbdt-symmetric --dataset synthclf --devices gpu
```

This existing command is FAST+opponents only, NOT three-arm admission. Add explicit fast/identical workers, exactly one selected external arm, same fixtures and output checks before use for target measurements. Existing --ours-only permits arm isolation. Do not run nvidia_forest_bench.sh: six-arm historical default violates requested one-comparator policy.

### classical_adapter — implementation_required_before_measurement

```sh
MOJOLEARN_SPEED_LANE=kmeans MOJOLEARN_SPEED_ROUNDS=7 python tools/speed_cuml_arm.py
```

Existing external leg only; combine prepared FAST/IDENTICAL classical binaries in seven rotating rounds. Use MOJOLEARN_SPEED_DUMP where supported to share exact input bytes. Fix refused PCA solver; no all-lane uncontrolled launch.

### mamba_external — correctness_or_external_only_runner

```sh
python tools/mamba_external_compare.py --family mamba1 --native /tmp/current-mamba1/native --out /tmp/current-mamba1-external --samples 0
```

Fresh native manifest required. samples7 adds external-only timing, NOT a Mojo ratio; backward includes forward. Fix Mamba2 stride and Mamba3 availability, then add paired native forward/backward timed regions with matching objective.

### sequence_adapter — external_admission_leg_only

```sh
MOJOLEARN_CPU_THREADS=2 python tools/speed_torch_seq.py --lane transformer --row 2 --rounds 1 --warmups 1 --no-bf16 --dump-dir /tmp/current-transformer-identical --mojo-log /tmp/current-transformer-fast.log --mojo-log /tmp/current-transformer-identical.log
```

Prepared current FAST/IDENTICAL dumps required; run against each mode dump to gate both outputs; explicitly select one strict-FP32 arm in orchestrator and rotate seven rounds. Mamba uses lane mamba row8 and --require-fused. Other emitted diagnostic precision arms are not additional selected comparisons.

### gemm_adapter — implementation_required_before_measurement

Pair bench/speed/gemm_speed_main.mojo with tools/speed_gemm_arm.py only after exact common input/precision/output admission and one-comparator selection are enforced.

### new_adapter — missing

No admitted operation-specific three-arm runner identified; implement bounded same-fixture adapter before measurement.

The historical `tools/nvidia_forest_bench.sh` and general all-family remote speed launches must not be used unchanged: they can launch multiple comparator arms or excessive work. No rental command is proposed by this inventory. The root agent owns scheduling, exact input/precision checks, source/artifact qualification and all execution.
