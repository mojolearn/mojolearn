# Changelog

This file records release-level changes, not the development diary. Git history and archived evidence
contain the detailed investigation record.

## Unreleased

### 0.6.0 release candidate

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

Versions 0.1.0 through 0.3.2 established the Mojo GPU port, derivation/refusal ledgers, identity-card
methodology, Python packaging, and the initial Apple/AMD/NVIDIA evidence. Exact changes are preserved
by Git tags and history rather than duplicated here.
