# Changelog

## Unreleased

Ships after 0.2.0, which is macOS only. Nothing below has run yet; every
script named is committed unrun under the no-run order of 2026-08-29 and
`docs/LINUX_WHEEL.md` section 9 lists what a run must confirm.

### The Linux wheel, designed and tooled

- ONE PyPI name, `mojolearn`, two wheels per release, the macOS arm64 wheel
  and a Linux x86_64 wheel carrying BOTH a CUDA and a HIP binary set in all
  three numeric tiers (six sets, sixty extensions), the vendor picked at
  import. Layout `mojolearn/{cuda,hip}/{,deterministic,identical}/*.so`,
  MAX runtime under `.libs/`. `docs/LINUX_WHEEL.md` is the design note.
- `mojolearn.vendor()` and `estimator.vendor_used()` report the accelerator
  API the loaded binary was COMPILED for, read back out of the binary through a new
  `<binding>_vendor()` export on every binding (`mojo_only/vendor.mojo`, a
  compile-time constant from `std.sys.info.has_*_gpu_accelerator()`). The
  selector refuses at import when a binary's answer disagrees with the
  directory it was loaded from, the same refusal as the tier read-back.
  `python -m mojolearn verify` and `mojolearn doctor` print it.
- `MOJOLEARN_VENDOR=cuda|hip` picks the directory on Linux; otherwise a box
  probe (device nodes and driver libraries) picks it, and a box with neither
  refuses at import with a message naming every path and library it looked
  for. There is no CPU path.
- `packaging/linux/`: `build_sets.sh` (on a box), `stage_libs.py` (ELF
  closure), `pack_wheel.py` (pure Python, on the Mac), `audit.sh`
  (auditwheel and twine in docker), `smoke.py`, `sabotage.py`, `nogpu.py`,
  and `leg.sh` (one command per vendor over the existing RunPod and
  DigitalOcean legs). `tools/e1_bootstrap.sh` phase 9 gained
  `MOJOLEARN_P9_ONLY_DIAG` and both legs pass the diag knobs through.
- `packaging/macos/smoke.py` asserts the vendor read-back (`metal`).

## 0.2.0 (2026-08-29)

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

On 2026-08-28 all three boxes ran at ONE commit, `a0a0eee` (Apple M4 leg
`bench/results/e1/2026-08-28_130918-MacBook-Air-1-terrabyte`, NVIDIA H100
leg `2026-08-28_131651-runpod-nvidia`, AMD MI325X leg
`2026-08-28_173933-mojolearn-e2-amd`). Every IDENTICAL card is byte-identical
Apple-to-NVIDIA and Apple-to-AMD: cd 23 stages, gemm 61, iforest 124, kde 9,
linkage 10, metrics 64, svm 35, and the `ridge_*` and `logreg_*` E2U cells.
So `mojolearn.linalg`, `Lasso`, `ElasticNet`, `KernelDensity`,
`AgglomerativeClustering`, `SVC`, `mojolearn.metrics`, `IsolationForest`,
`Ridge` and `LogisticRegression` all stand on measured three-vendor cards.
This also closes two items that were open at E3 round 11: `metrics`'s grown
card (34 stages then, 64 now) has had its three-vendor leg, and
`isolation_forest` joined phase 8 and ran on both boxes.

`SpectralClustering`, `ExponentialSmoothing`, `kpss_test` and `select_d` do
not stand on a three-vendor card, and the reason changed on 2026-08-28. All
three lanes now emit identity cards and are listed in the round: `spectral`
and `holtwinters` joined phase 8 in `241aed6`, and `tsa` in `f081b7f`, whose
message records that `tsa/tsa_main.mojo` had built a complete eleven-stage
card since it was written and that the comment claiming otherwise was the
only thing keeping it out. `spectral` and `holtwinters` have an Apple card
and an AMD card; `tsa` has an Apple card.

What is missing is a SHARED COMMIT. The Apple cards for `spectral` and
`holtwinters` were taken at `5fd95b3` and `241aed6`, the AMD cards at
`26eb8ba`, and `tsa`'s Apple card at `869d416`, so no two of them are
comparable and no NVIDIA leg has run any of the three. Their pins come from
the same source as the lanes above, so they are expected to match, but the
comparison has not been performed and each class says so rather than
inheriting a neighbor's certificate.

**The FAST arm, which is the default, makes no cross-vendor claim at all.**
Unchanged from 0.1.0, and the FAST cards do differ between vendors for
every lane but gemm; that is recorded, not a defect.

### A third numeric tier, `deterministic`, and it ships

`MOJOLEARN_NUMERIC_MODE` became a three-rung ladder on 2026-08-29
(DEVIATIONS 1940 and 1941). `fast` promises nothing but speed, `deterministic`
promises the same bits run to run on one box and says nothing about a second,
and `identical` promises both plus the same bits on Metal, CUDA and HIP.
Determinism is a strict subset of identity, never a sibling, so
`PIN_DETERMINISM` is true under both upper tiers.

The tier is worth having because the reproducibility most callers actually
want was only purchasable by taking `identical` whole, and identity is not
free. Three consecutive FAST runs of one binary on one fixture on one M4
returned three different sorted k-NN index sets, and across every FAST board
taken on 2026-08-28 our own arm's per-round output hash moved between rounds
on 4 of 179 rows.

**0.2.0 carries all three tiers.** This entry said "`fast` and `identical`
only ... once the lane closes", and the lane closed the same day; the old
text is replaced rather than appended to, because a changelog that describes
a wheel it does not describe is worse than one that says nothing.

What closed it is a MEASUREMENT of the promise, not a pin count.
`tools/repeat_run_stability.py` refits one lane repeatedly in a single
process and compares raw output bytes with no tolerance. At one commit on
three rented columns on 2026-08-29:

| column | `fast` | `deterministic` | `identical` |
|---|---|---|---|
| Apple M4, Metal | MOVED in 8 of 10 attempts | STABLE 10/10 | STABLE 10/10 |
| NVIDIA RTX 4090, CUDA | MOVED, 24 calls 24 answers | STABLE | STABLE |
| AMD MI325X, HIP | MOVED, 6 answers in 24 | STABLE | STABLE |

The pin side is 15 files keyed to `PIN_DETERMINISM`. The class is small
because this tree uses no float `atomicAdd` anywhere -- three of the four
classification scopes came back empty, which is the right answer for a tree
built that way rather than a gap. And the tier is not the top one in
disguise: under `deterministic` the output hashes DIFFER between vendors on
10 of 12 comparable lanes, `gemm-vendor` among them, so it keeps MAX
`matmul`, cuBLAS and rocBLAS at full speed. All three of those were measured
run-to-run stable at 256x4096 @ 4096x128, a wide k chosen to provoke a
split-K epilogue, which is what earns them that exemption.

`ce2e843` is still the reason to be careful with this tier: for the length of
one commit it was defined, empty, and would have taken the float atomic flush
while calling itself deterministic, which is the worst failure a tier named
that way can have. That is what `verify_wheel.sh` now guards -- it installs
the finished wheel into a clean venv under every claimed interpreter and fits
every estimator family in EVERY shipped mode, and a tier that did not build
raises from a missing-binary stub BY NAME rather than serving fast arithmetic
under another label. Which tiers a wheel carries is one variable,
`MOJOLEARN_RELEASE_MODES`, read by both scripts.

### The mode is a PARAMETER now, and there is still ONE install

Until `035922c` the numeric mode was reachable only as
`MOJOLEARN_NUMERIC_MODE`, an environment variable read ONCE by
`_backend.select()` before the first estimator was imported. That was always
one install -- the wheel has carried every tier's binaries since 0.1.0 -- but
it was a choice you had to make from the shell, could not change after import,
and could not make differently for two estimators in one script.

    mojolearn.set_numeric_mode("deterministic")            # process default, in code
    rf = mojolearn.RandomForestClassifier(numeric_mode="identical")   # one estimator
    est.numeric_mode = "fast"                              # after construction
    est.numeric_mode_used()                                # what THIS instance will call
    mojolearn.numeric_mode()                               # the process default

`MOJOLEARN_NUMERIC_MODE` still works and still sets the STARTING default, so
nothing written against the old spelling breaks. There is no extra to install,
nothing to rebuild and nothing to reinstall: `pip install mojolearn` is the
whole surface and it always was.

WHY IT IS A BINDING LOOKUP AND NOT A FLAG. `PIN_DETERMINISM` and
`PIN_CROSS_VENDOR` are comptime, which is what lets the fast build carry none
of the pinning code rather than branch past it. So a parameter cannot flip
something inside one binary; it selects WHICH BINARY answers, which is why
every call site resolves through `self._bind()` at call time instead of
binding a module-level name at import. The keyword is injected by
`__init_subclass__` rather than written into eleven constructor signatures,
because eleven copies is eleven chances for one to drift silently -- an
estimator that ignored the keyword would run the process default while
reporting the tier it was asked for.

THAT THE TIERS COEXIST IS MEASURED, not assumed. Each `.so` carries its own
Mojo runtime and opens its own device context, so "two of them in one process
will fight" was the real risk. All three were loaded together and called
INTERLEAVED -- fast, deterministic, identical, fast -- twice, on one
256x4096 @ 4096x128 product on an M4, and each returned its own arithmetic
every time; a call made after the identical set did not inherit its answer.
17 estimators x 3 tiers pass.

TWO BUGS THIS FOUND, both in the reporting rather than the arithmetic, and
both shipped in 0.1.0. `mojolearn doctor` reported "import probe: failed" on a
healthy install, because it `json.dumps`'d the `numeric_mode` FUNCTION object.
And `numeric_mode()` reported the IMPORT-TIME tier, so after
`set_numeric_mode("deterministic")` it still answered `fast` -- the one
function whose entire job is that a run cannot be mislabeled was mislabeling
it. Both are fixed in `42abc57`. Neither was caught by a green test, and what
caught them was running what the documentation claimed.

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

Landed after that board were DEV 1902 (ridx-only splits), 1911-1914 (quantized
shared-memory histograms), 1916-1919 (random forest launch batching) and
1921-1923 (k-NN warp-select and dispatch), merged and gated at `b90ab1c`.
Measured on Apple at `a8d838e` on 2026-08-28
(`bench/results/fast_speed/2026-08-28-APPLE-forest.md`). There, rf on covtype is
16.89x faster than lightgbm-cpu, gbdt-symmetric on year is 1.14x faster than
catboost-cpu and et on covtype is 1.12x faster than sklearn; depthwise is
1.08x, rf 1.23x and lossguide 1.47x behind their CPU arms. Several arms in
that round are routed to the NVIDIA and AMD FAST columns by the kernel
matrix and do not touch this wheel at all.

### Known issues

- `ftz`, the denormal-policy helper that IDENTITY_PATHS row 10 is built on,
  DID NOTHING ON THE GPU until 2026-08-28 (DEVIATION 1938). Its guard was two
  float comparisons, and a flush-to-zero backend evaluates those with the
  operand already flushed, so `x != 0.0` was false for every subnormal, the
  branch never fired, and the helper returned subnormals untouched. Measured
  on the M4 under IDENTICAL against a host twin, three of four patterns
  disagreed. It hid because everywhere a denormal also passes through
  arithmetic the hardware flushes it anyway; it is visible only where a value
  reaches a seam BY COPY. **This was a cross-vendor hole, not an Apple one**,
  since on a denormal-honoring backend the guard fired and on an FTZ backend
  it did not. The guard now tests the exponent and mantissa fields, which no
  backend's FTZ can defeat. **Every binary published before this release,
  including the 0.1.0 wheel on PyPI, carries the inert helper.**

- `IsolationForest` carries OPEN DEVIATION 750. cuML's `curand_u64` builds
  a 64-bit draw out of two unsequenced `curand()` calls and C++ does not
  say which becomes the high word. Both readings conform and they give
  DIFFERENT forests from the same seed. This port takes the first draw as
  the high word, by name, and that choice has never been checked against a
  cuML binary. Until it is, agreement with cuML there is a belief and not a
  measurement.
- `check_ols_is_launch_invariant` still fails on both the H100 and the
  MI325X in both numeric modes, most recently in the 2026-08-28 legs at
  `a0a0eee`: two identical OLS fits in one process disagree at coefficient 0
  (`0xbbb60202` vs `0xbbb87825` on the H100 under FAST). The gate's own text
  is the right reading of it -- nothing on that path uses a float atomic, so
  this is not an ordering hazard but an uninitialized read or a race, and it
  is a defect in BOTH modes. It has NOT been reproduced on Apple, which is
  the only platform this wheel targets, and the card fixture's two fits agree
  so there is no traced repro yet. Open since E3 leg 10; localization owed.

### Packaging

- The wheel now carries TEN extensions in THREE numeric tiers rather than
  five: the 0.1.0 set plus `_mojolearn_svm.so`, `_mojolearn_solver.so`,
  `_mojolearn_metrics.so`, `_mojolearn_tsa.so` and `_mojolearn_linalg.so`,
  each built fast, deterministic and identical by
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
