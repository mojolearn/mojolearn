# Changelog

This file records release-level changes, not the development diary. Git history and archived evidence
contain the detailed investigation record.

## 0.8.3 (unreleased 2026-09-11)

A patch on the 0.8.2 line. Branch release-0.8.3 starts at tag v0.8.2 (438a6e66) and carries
only the fixes below, their checks and the version bump; main's later speed defaults are not
in it.

- Fixed SVC on NVIDIA GPUs. Every `SVC` or `SVR` fit with more than 512 training rows failed
  on CUDA with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES since 0.8.0: the block solve's working set
  is `min(1024, n_train)`, and CUDA refuses the width 1024 kernel's reduction schedule. A
  kernel-matrix row now gives NVIDIA above width 512 the earlier halving-tree schedule, which
  selects the same elements, so the model bits are unchanged on every vendor (DEVIATION 2623).
  Verified on an H100 (fits at 400, 600 and 2,000 rows equal to the pre-0.8.0 schedule's bits,
  `svm/svc_main.mojo` 44/44 IDENTICAL, which failed seven gates before) and on the Apple M4.
- Fixed IDENTICAL `LinearRegression` on badly scaled designs. The float32 eigensolver's
  squared Frobenius norm overflowed on Gram matrices with eigenvalues near 1e19 and stopped
  before any rotation, and an absolute 1e-10 eigenvalue cutoff made the model's rank depend on
  the data's units. Istella-S (2,043,304 x 220) returned R^2 -115.6 where scikit-learn gets
  0.164. The Gram matrix is now equilibrated by exact power-of-two scales and the cutoff is
  relative, `n * eps32 * max|eigenvalue|` (DEVIATIONS 2620, 2621 for more rows than
  features; 2622 for more features than rows). Istella-S now gives R^2 0.332; taxi keeps R^2
  0.908837 with different coefficient bits. Coefficient hashes are identical across vendors.

## 0.8.2 (published 2026-09-11)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) and macOS arm64 wheel, both from
commit 438a6e66 (tags alpha-api-0.8.2-20260911 and v0.8.2), on PyPI 2026-09-11 18:20Z and
18:35Z (release runs 34632585818 and 34632738047).

A patch on the 0.8.1 line. Branch release-0.8.2 starts at tag v0.8.1 (343ffa35) and carries
only the fix below, its check and the version bump; main's later GBDT speed defaults
(DEVIATIONS 2550, 2551, 2581) are not in it. No change on Apple or NVIDIA: the fixed and
unfixed builds give identical bits there.

- Fixed IDENTICAL GBDT on AMD GPUs. The binary, half-byte and 5-/6-bit histogram kernels
  skipped a block-wide sync on some threads of AMD's 64-lane layout, so a fit on data with
  tied values (few distinct values per column) could differ between runs in one process
  and from NVIDIA and Apple. 0.8.1 moved on the `ties` fixture on an MI300X. Every thread
  now makes the same trips (DEVIATION 2600). Verified: 36/36 identity cells equal to the
  H100 on an MI300X and on the Apple M4, taxi 1M symmetric one hash in 10/10 rounds.
- Added `checks/gbdt_sub_byte_identity_check.py`, which fits the binary, half-byte, 5-bit
  and 6-bit arms twice and compares each to an H100 reference.

## 0.8.1 (published 2026-09-11)

A patch on the 0.8.0 line. Branch release-0.8.1 starts at tag v0.8.0
(4a3c22c3, which is the 0.8.0 Linux build commit 9392320e plus two
qualification-tool commits) and carries only this fix, its regression check
and the version bump, because main has changed Random Forest outputs and the
attention default since 0.8.0. The 0.8.0 extension sets are not reused (their
build proofs bind `python/mojolearn/_buffer.py` and `_version.py`), so all
three Linux sets are rebuilt from this branch. No numerics change.

- Fixed buffer conversions in a process that also imports cuML. They were
  all refused with "argument 2: expected LP__PyBuffer instance instead of
  pointer to _PyBuffer", because `treelite.model` retypes the
  `ctypes.pythonapi.PyObject_GetBuffer` function pointer that ctypes caches
  process-wide and `_buffer.py` shared it. mojolearn now takes private
  function pointers for `PyObject_GetBuffer`, `PyBuffer_Release` and
  `PyMemoryView_FromMemory`.
- Added `tools/check_buffer_foreign_argtypes.py`, which reproduces that
  failure without cuML (`--real-cuml` imports cuML instead). The Linux and
  macOS release smokes arm the same foreign argtypes before their fits.

## 0.8.0 (published 2026-09-10)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) from commit 9392320e,
tag alpha-api-0.8.0-20260910; macOS arm64 wheel from tag v0.8.0 at 4a3c22c3.
Both on PyPI 2026-09-11 02:46Z and 02:52Z. Installed-wheel qualification on
the Linux architectures was not run (release policy of 2026-09-09: build,
retag, publish; test what changed). DEVIATION 2500 (labels in the base
binding) landed on main during the build and is not in these wheels.

- Only the three tree families (GBDT, Random Forest, Extra Trees) ship the
  `fast` and `deterministic` tiers. Every other binding, including SVC, SVR,
  isolation forest, k-means, k-NN, PCA, the linear models, UMAP, GP, ARIMA,
  preprocessing and the neural surface, builds and ships `identical` only.
  Asking one of them for a lower tier raises a named error, from
  `numeric_mode=` and from `MOJOLEARN_NUMERIC_MODE`, instead of an
  ImportError about a missing extension. Cross-vendor bitwise identity is
  the product; a fast tier ships only where it has a measured win over the
  opponent's own CPU. This removes 26 extension files from every three-tier
  wheel (13 bindings x 2 retired tiers) and is a minor break from 0.7.0,
  where `KMeans(numeric_mode="fast")` worked.
- Removed the runtime NumPy dependency. Estimators return `mojolearn.Array`
  and accept supported buffer inputs; `numpy.asarray(result)` provides a
  zero-copy view. Shared native conversion, validation and row-gather helpers
  support the GPU paths. New builds and qualification are required; this
  changes the array return API from 0.7.0.
- Added optional GPU `parallel_groves` prediction for Random Forest and Extra
  Trees, sharing resident forest storage, vector-leaf traversal and reusable
  prediction buffers. The default remains `sequential`; the two engines use
  different floating-point reduction orders. Packed-node traversal remains an
  experimental build option.
- Added bounded Random Forest classifier `class_weight` support, per-tree GBDT
  feature sampling and optional minimum child Hessian eligibility for Newton
  depthwise/lossguide growth. Unsupported combinations raise explicitly.
- Added GBDT classifier/regressor adapters and sklearn-style forest parameter
  and scoring protocols. Fitted numeric modes are retained for prediction.
- Added GPU regression errors, classification counts and scores, log loss,
  binary ROC AUC and precision-recall curves, with mode-aware arithmetic.
- Added GPU `StandardScaler` and `MinMaxScaler` with transformer protocols,
  and bounded serial GPU `cross_val_score` support for compatible pipelines.
- Improved automatic neighbor query batching and IDENTICAL wide full PCA.
- Added compiled host buffer conversion helpers and staged the NumPy-free
  buffer core. The estimator layer is not yet NumPy-free.
- Added `LanguageModelConfig` and `LanguageModelTrainer` aliases with
  configurable layer counts and token vocabularies, plus optional resident
  IDENTICAL model/optimizer sessions across Python calls. Inter-layer
  gradients stay on device; attention and prefill workspace allocation is
  reduced. Existing small byte-model defaults remain supported. Large-model
  fit and full-training time remain unqualified.
- The macOS release workflow now includes the IDENTICAL language-model
  extension and checks three-layer/vocab257 resident and stateless training
  in the installed wheel on each supported Python interpreter.
- Extended IDENTICAL IVF selection through k=1024, added the embedding
  backward total-key sort plan with scan/sort identity checks, and routed
  UMAP host math through portable binary64 seams including power.
- Flattened implementation directories and corrected release build policy pins.

## 0.7.0 (published 2026-09-09)

Linux x86-64 wheel (CUDA sm_89, CUDA sm_90a, HIP gfx942) from commit fe6067ba;
macOS arm64 wheel from tag v0.7.0. Installed per-architecture qualification
was not run for the Linux wheel.

- The identical path no longer calls the host C library. The seven host
  calls for float64 log and log2, float32 log and exp, and ceil (random-forest
  feature rule, boosting loss constant, min-entropy bin construction,
  extremely-randomized-trees builder) use the library's own portable
  implementations, the same code the device runs under identical mode, so
  every host computes the same bits by construction (DEVIATIONS 2260 to 2266).
  Verified on the M4 and an H100: the CatBoost bias oracle matches to the bit
  on every implemented arm, min-entropy and GreedyLogSum borders match CatBoost on
  every case, the RF predict check passes in both modes. Fast-mode bits at
  those three sites move; the fast profile carries no bit promise across
  versions.
- Mamba-3 deterministic mode on CUDA: the stable small-dt softplus
  spelling (float32 log1p of the vendor exp) now covers DETERMINISTIC as
  well as FAST. Before this, DETERMINISTIC evaluated log(exp(x) + 1) and the
  installed qualification failed the Mamba-3 key-state report against the
  float64 reference at one element on both an L40S and an H100, with the
  same excess FAST had shown before its own repair (DEVIATION 2300).
  IDENTICAL bits are untouched; DETERMINISTIC dt bits move on every vendor,
  within its same-box same-build contract.
- One Linux x86-64 wheel carrying CUDA sm_89, CUDA sm_90 and HIP gfx942, each
  in fast, deterministic and identical, plus the identical-mode byte-LM
  trainer extension per architecture (the combined-Linux profile authored
  for 0.6.1, published as 0.7.0). The 0.6.1 version number was never
  published.
- Repository size fences: pre-commit and pre-push hooks under tools/hooks
  refuse oversized blobs and wheels, tarballs or fixture dumps under
  bench/results; the incident is recorded in CONTRIBUTING.md.
- README rewritten around the gap the implementation fills and the identity contract,
  with a project-status section.
- Leg tool: a `checks` family runs named conformance checks on a rented GPU;
  `--allow-concurrent` covers recorded leases; the release-build gate ignores
  bench/results; the pixi installer has a 300 s budget with one retry.

### 0.6.1 (unpublished candidate)

- Version bump, alpha overlay, Linux and macOS packaging, serial job guards
  and release qualification tooling. Superseded by 0.7.0 without a PyPI
  release.

## 0.6.0 (published 2026-09-06)


- Added `UMAP.transform` to embed unseen samples against a frozen fitted model.
  Training input, embedding and fitted parameters are retained privately;
  changed parameters or numeric mode require refitting.
- Public UMAP fitting now stores the fuzzy graph in CSR form, using
  O(n_samples * n_neighbors) graph space. Exact neighbor computation remains
  quadratic; sparse storage is not an approximate-neighbor implementation.
- Preserved named IDENTICAL fit layouts and added held-out transform quality
  checks. Source transform fixtures match across Apple, NVIDIA and AMD;
  installed artifacts are qualified separately before publication.
- Bounded binding compilation to two workers by default, configurable through
  `MOJOLEARN_COMPILE_JOBS`.
- Retained an experimental specialized small-k selector behind an explicit
  build flag. It is not enabled in normal wheel builds.
- Removed the identical path's last host-libm dependency (IDENTITY_PATHS row 18,
  DEVIATION 2260). The eight `external_call` sites for `log`, `log2`, `logf`,
  `expf` and `ceil` in the random forest `max_features='log2'` rule, the extra
  trees host feature sampler, the MinEntropy border penalty and the
  boost-from-average logit now use the library's own portable logarithm and
  exponential (new `portable_log2_64`, exact at powers of two) and an exact
  `ceil`, so those bits are the same on every host and device. The two
  CatBoost-fidelity sites may differ from CatBoost's libm-computed value by one
  ulp on a near-tie; the CatBoost oracle cards are owed a re-baseline run.

## 0.5.0 — 2026-09-05

- Published the macOS arm64 wheel with 15 native extensions in all three modes.
  All Python 3.10–3.14/mode combinations passed isolated installed-wheel checks.
  The downloaded PyPI artifact matched the publication digest and passed smoke,
  Mamba, and Transformer API suites in all modes on Python 3.12 / Apple M4.
  A refreshed Linux wheel remains pending.

- Added `mojolearn.UMAP.fit` and `fit_transform` for dense Euclidean input,
  spectral initialization, and 2D/3D embeddings, with per-estimator numeric modes.
- Reject non-finite UMAP inputs and optimizer parameters before numerical work;
  gate the installed API against the named IDENTICAL layout fixture.
- Retained three-vendor Mamba backward and UMAP source certificates: five Mamba
  cases, 54 gradient tensors, and 186 UMAP stage cells match bitwise on Apple M4,
  NVIDIA RTX 4090, and AMD MI300X at `718495cd`. These are source-fixture claims,
  separate from installed-wheel platform coverage.

- Consolidated active documentation around one roadmap, support matrix, verification guide, and
  normative numerical contracts.
- Added and expanded Mamba, Transformer, training, embedding, and packaging validation lanes.
- Distinguished FAST, deterministic, and IDENTICAL promises across bindings and release tooling.
- Added guarded multi-vendor evidence collection and interleaved FAST/IDENTICAL performance harnesses.
- Added artifact admission checks for wheel contents, digests, platform tags, and native extensions.

## 0.4.0 — 2026-09-02

- Expanded cross-vendor identity coverage across classical ML, linear algebra, tree, sequence, and
  training components.
- Added Mamba 1/2/3 and Transformer forward/backward implementation work and Python bindings.
- Added the 15-extension packaging surface and stricter release refusal checks.
- Added representative price lanes for classical, unsupervised, linear-algebra, and tree workloads.

## Earlier releases

Versions 0.1.0 through 0.3.2 established the Mojo GPU implementation, derivation/refusal ledgers, identity-card
methodology, Python packaging, and the initial Apple/AMD/NVIDIA evidence. Exact changes are preserved
by Git tags and history rather than duplicated here.
