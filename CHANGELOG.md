# Changelog

## 0.2.0 (2026-08-28)

Fourteen new public names and two new submodules, over lanes that were
finished and gated at the kernel level before 0.1.0 shipped and were simply
unreachable from Python. **Their cross-vendor standing is NOT uniform and
each class states its own.** Read the class, not this list.

### Estimators

- `SVC` (cuML's `SVC`; binary C-SVC only. There is no `SVR`: `svmType !=
  C_SVC` raises by name in `svm/ported/svm/svm_parameter.mojo`, and
  epsilon-SVR is rung 2 in `svm/UNPORTED.tsv`)
- `Lasso`, `ElasticNet` (cuML's `solver='cd'` arm, `cd.cuh::cdFit`;
  DEVIATIONS 610-613 and 880)
- `AgglomerativeClustering` (cuML's `hierarchy/linkage.cu` down through
  cuVS's `cluster/detail/` to RAFT's Boruvka MST; single linkage;
  DEVIATIONS 620-624 and 881)
- `Ridge` (cuML `ridgeFit`, the `eig` arm: `svdEig` + `ridgeSolve`;
  DEVIATION 545) and `LogisticRegression` (cuML `qnFit`, the binary L-BFGS
  sigmoid arm with the Armijo line search; DEVIATIONS 546-549. OWL-QN / l1,
  softmax / multiclass, `sample_weight` and `class_weight` are refused by
  name). Under IDENTICAL every reduction in the objective, the gradient and
  the solver is a pinned fold where cuML's are float atomics, so the
  iteration count is part of the certificate (`qn.n_iter`). Gates:
  `glm/mojo_only/ridge_check.mojo`, `glm/mojo_only/logistic_check.mojo`,
  both modes, in `pixi run check-linalg-identity`; E2U cells `ridge_*`,
  `logreg_*`.
- `KNeighborsClassifier`, `KNeighborsRegressor` (the cuVS/RAFT/FAISS k-NN
  path 0.1.0 already shipped as `NearestNeighbors`, with the vote and the
  mean on top)
- `KernelDensity` (cuML's `kde/`; DEVIATIONS 600-604. `bandwidth='scott'`
  and `'silverman'` are refused by name: compute the number and pass it, so
  the number that ran is the number you passed)
- `IsolationForest` (cuML's `IsolationForest`; see Known issues)
- `SpectralClustering` (cuML's `fit_predict` through cuVS's kNN
  connectivity graph, the normalized graph Laplacian and RAFT's
  thick-restart Lanczos, with every closed vendor library on that path
  replaced by a named, numbered stand-in; DEVIATIONS 770-781)
- `ExponentialSmoothing` (`cuml.tsa.ExponentialSmoothing`, backed by
  `holtwinters/`; DEVIATIONS 660-665 and 697-699)

### Functions and submodules

- `mojolearn.metrics`, fourteen scoring functions: `accuracy_score`,
  `r2_score`, `rand_score`, `adjusted_rand_score`, `homogeneity_score`,
  `completeness_score`, `v_measure_score`,
  `homogeneity_completeness_v_measure`, `entropy`, `mutual_info_score`,
  `kl_divergence`, `silhouette_score`, `silhouette_samples`,
  `trustworthiness`. The names and argument names are scikit-learn's,
  because that is what a caller types; **the defaults and the semantics are
  cuML's**, and the three differences that matter are written on the
  function: `entropy` and `mutual_info_score` are in NATS as RAFT computes
  them, `r2_score` bakes in `force_finite=True` and refuses
  `force_finite=False` by name (DEVIATION 657), and `kl_divergence` does
  not normalize `P` and `Q`.
- `mojolearn.linalg`: `matmul`, plus `profile()`, `numeric_mode()` and
  `require_identical()`. A numerical primitive rather than an estimator,
  for callers who need a matrix product that returns the same bits on three
  vendors and will never fit a model here. Profile
  `mojolearn.identical.gemm.fp32.v1`; contract
  `gemm/IDENTICAL_FP32_CONTRACT.md`.
- `kpss_test` and `select_d` (`cuml.tsa.stationarity.kpss_test` and
  auto_arima's "choose the hyper-parameter d" block, both backed by `tsa/`;
  DEVIATIONS 671-672)

`ARIMA`, `SVR` and `RadiusNeighbors` are still absent and are named in
`mojolearn._NOT_YET`: importing one raises with the line where the thing
that exists stops, rather than an `AttributeError`. `ARIMA` in particular
has no `fit` -- `arima/` ports the batched Kalman filter likelihood, its
gradient and predict, but `estimate_x0` and the batched L-BFGS driver are
NOT PORTED, and those are exactly what produces the coefficients every
existing entry point requires as input.

### Cross-vendor standing

Every lane in this release runs the same pinned path under
`MOJOLEARN_NUMERIC_MODE=identical`. What differs between them is not whether
they are pinned but **where the pinned card has been diffed**: bit-identity
is a claim about two machines agreeing, and checking it requires the lane to
have run on a second and third machine.

At E3 round 11 (commit `144aa5b`, 2026-08-23) the six classical lanes had
their cards emitted on an NVIDIA H100 (CUDA) and an AMD MI325X (HIP) and
diffed against the Apple M4's (Metal), stage for stage, bit-identical on all
three: gemm 60 card stages, cd 20, kde 7, linkage 8, svm 32, metrics 34. So
`mojolearn.linalg`, `Lasso`, `ElasticNet`, `KernelDensity`,
`AgglomerativeClustering`, `SVC` and `mojolearn.metrics` stand on measured
three-vendor cards.

`IsolationForest`, `SpectralClustering`, `ExponentialSmoothing`,
`kpss_test`, `select_d`, `Ridge` and `LogisticRegression` have run on one
Apple M4 and nowhere else. Their gates are green there and their pins come
from the same source as the six above, so they are expected to match, but
the leg is OWED and until it runs that is an expectation and not a
measurement. Each class says so rather than inheriting a neighbour's
certificate. Round 11 is why the distinction is kept: running the second and
third box is what surfaced the OLS launch-invariance failure and the
subnormal KL operand in Known issues below, neither of which is visible on
one machine.

`metrics/`'s card has since grown from 34 stages to 61, and the
three-vendor leg on the grown card is OWED.

**The FAST arm, which is the default, makes no cross-vendor claim at all.**
Unchanged from 0.1.0, and the FAST cards do differ between vendors for
every lane but gemm; that is recorded, not a defect.

### Tooling

- `python -m mojolearn verify` checks this build against a reference card,
  `--json` for machine-readable output. Exit codes, because people put this
  in continuous integration: 0 VERIFIED, 1 MISMATCH (a diverging stage is
  named), 2 USAGE, 3 REFUSED (the process loaded the FAST binaries, which
  make no identity claim, so nothing was judged), 4 CANNOT RUN.
- `python -m mojolearn env` reports what the process loaded without
  touching the GPU; `python -m mojolearn check-fixture` rebuilds and hashes
  the pinned fixture without a GPU and without an extension call.
- The `mojolearn` console script continues to point at
  `mojolearn_diagnostics`, which lives outside the package on purpose so it
  still runs when importing the extensions is the thing being diagnosed.

### Performance (FAST arm)

The FAST path changed substantially this cycle; IDENTICAL is untouched by
it and every A/B fit across the round was byte-identical.

Measured on one MacBook Pro M4 on 2026-08-27 at `20729ea`, FAST both
sides, three timed rounds after a warm-up, against the only arms the
opponents ship on Apple silicon
(`bench/results/fast_speed/2026-08-27-APPLE-trees-evening.md`):

| lane | dataset | opponent | verdict |
|---|---|---|---|
| gbdt-symmetric | higgs 1M x 28 | catboost-cpu | 1.31x FASTER (was parity) |
| gbdt-symmetric | year 464k x 90 | catboost-cpu | 1.20x FASTER (was 1.10x) |
| rf | higgs 1M x 28 | sklearn-rf-cpu | 3.07x FASTER |
| et | higgs 1M x 28 | sklearn-et-cpu | 1.25x FASTER, and ahead on accuracy |
| et | covtype 523k x 54 | sklearn-et-cpu | 1.13x SLOWER |
| rf | covtype 523k x 54 | sklearn-rf-cpu | 1.45x SLOWER |
| gbdt-depthwise | year 464k x 90 | catboost-cpu | 1.28x SLOWER |
| gbdt-lossguide | year 464k x 90 | catboost-cpu | 1.73x SLOWER |

covtype's deep-narrow, launch-bound shape remains the CPU arms' best
ground. depthwise and lossguide get their first Mac rows here and both are
behind.

Landed after that board and NOT yet measured on Apple: DEV 1902 (ridx-only
splits), 1911-1914 (quantized shared-memory histograms), 1916-1919 (random
forest launch batching) and 1921-1923 (k-NN warp-select and dispatch),
merged and gated at `b90ab1c`. Several arms in that round are routed to the
NVIDIA and AMD FAST columns by the kernel matrix and do not touch this
wheel at all.

### Known issues

- `IsolationForest` carries OPEN DEVIATION 750. cuML's `curand_u64` builds
  a 64-bit draw out of two unsequenced `curand()` calls and C++ does not
  say which becomes the high word. Both readings conform and they give
  DIFFERENT forests from the same seed. This port takes the first draw as
  the high word, by name, and that choice has never been checked against a
  cuML binary. Until it is, agreement with cuML there is a belief and not a
  measurement.
- `check_ols_is_launch_invariant` fails on both the H100 and the MI325X in
  both numeric modes: two identical OLS fits in one process disagree at
  coefficient 0. Open since E3 leg 10 and NOT seen on Apple, which is the
  only platform this wheel targets. Localization is still owed
  (`E3_RESULTS.md`, round 11 verdict).

### Packaging

- The wheel now carries TEN extensions in two numeric modes rather than
  five: the 0.1.0 set plus `_mojolearn_svm.so`, `_mojolearn_solver.so`,
  `_mojolearn_metrics.so`, `_mojolearn_tsa.so` and `_mojolearn_linalg.so`,
  each built FAST and IDENTICAL by
  `packaging/macos/build_release_wheel.sh`, which refuses a stale file by
  mtime.
- Requirements are unchanged: Apple silicon from the M1 up, macOS 11 or
  later, Python 3.10 through 3.14, numpy 1.24 or later, and a GPU. There is
  still no CPU path and no Linux wheel.

## 0.1.0 (2026-08-23)

First public release. One wheel, `pip install mojolearn`, macOS arm64,
Python 3.10 through 3.14, GPU only.

### Estimators

- `GradientBoosting` (CatBoost GPU oblivious trees; 12 losses plus
  MultiClass; depthwise and lossguide growth; CTRs; eval sets; overfitting
  detector; save and load)
- `RandomForestClassifier`, `RandomForestRegressor` (cuML design)
- `ExtraTreesClassifier`, `ExtraTreesRegressor` (cuML design)
- `KMeans`, `NearestNeighbors` (cuVS, RAFT, FAISS designs)
- `DBSCAN`, `PCA`, `TruncatedSVD`, `LinearRegression` (cuML and RAFT designs)

### Numeric modes

- `FAST` (default) and `IDENTICAL` ship in the same wheel as two binary sets;
  `MOJOLEARN_NUMERIC_MODE=identical` at import selects the identical set and
  `mojolearn.numeric_mode()` reads the loaded mode back from the binary.
- Cross-vendor certificates at this release's source: E2 sub-feature matrix
  99 cells, 0 divergent, Apple M4 (Metal) against NVIDIA H100 (CUDA) and
  against AMD MI325X (HIP); see `E2_RESULTS.md`, `E1_RESULTS.md` and the
  ledger `IDENTITY_PATHS.md`.

### Packaging

- The release build compiles all five extensions in both numeric modes from
  a clean checkout, stages the MAX runtime dylibs once for all ten, and gates
  the finished wheel by installing it into a clean venv under every claimed
  interpreter and fitting every estimator family in both modes.
- Built extensions are no longer tracked in git.
- One release workflow, `release-provenance.yml`, on an ephemeral self-hosted
  runner started by `tools/release_runner.sh`; the runbook is
  `docs/PYPI_RELEASE.md`.
- `.zenodo.json` and `CITATION.cff` carry the author ORCID; every GitHub
  release is archived on Zenodo with a DOI.

### Known limits

- No CPU path. No Linux or NVIDIA wheel yet (source builds only).
- Performance measured on one M4; see the README benchmark section and its
  stated accuracy gap on higgs GBDT.
- `MultiClassOneVsAll` is not on the Python surface.
