# Changelog

This file records release-level changes, not the development diary. Git history and archived evidence
contain the detailed investigation record.

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
