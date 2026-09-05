# Roadmap

This is the only live project plan. Historical plans and handoffs are not
current instructions; git history and `archive/` retain their evidence.

## Resumed implementation and identity checks (2026-09-05)

- CatBoost's experimental two-level FeatureFreq fit now accepts sample
  weights. Native and Python checks cover unequal weights, unit-weight
  equivalence, and occupied zero-weight leaves with zero regularization.
- Mamba2 exposes an incoming-state cotangent at the L257 chunk boundary.
  Mamba2/3 backward gates separate independent calculus checks from pinned
  reduction checks; the certificate retains native gradient bytes for
  comparisons across matching source snapshots and named devices.
- UMAP now has an end-to-end eight-stage bit capture, with finite-parameter
  refusal checks. Local repeatability is measured; cross-vendor status must
  come from the captured stage comparison, not a repeated local run.
- Stale optimizer comments and startup messages have been corrected while
  preserving independent-corpus limitations and historical evidence.

Apple M4 and NVIDIA RTX 4090 at `718495cd` passed all five backward gates;
all 54 retained gradient tensors and all 186 UMAP stage cells matched by
bits. See [the comparison record](bench/results/resume/2026-09-05/cross-device.json).

The subsequent AMD MI300X run at the same `718495cd` source completed all
five backward gates and the UMAP capture. Its recovered certificate matches
all 54 gradient tensors and 186 UMAP cells from Apple/NVIDIA; see the
[three-vendor record](bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json).
The completed AMD pod was terminated and its absence verified by HTTP 404.
The ROCm 6.4 image with SSH bootstrap therefore has a successful deployment;
the earlier runtime/inventory failures remain historical evidence.

UMAP's non-finite input changes are now checked on Apple M4: both
optimizer entries and public data entries reject all 42 NaN/infinity cases
in FAST and IDENTICAL builds, and the named fixture still matches all 186
baseline cells. The [local record](bench/results/umap/2026-09-05-finite-input-resume/metadata.json)
retains the dirty source snapshot. This later patch has no new remote claim.
The `0.5.0` macOS wheel at `529ec5ec` passed the clean installed-wheel gate:
all 15 extensions in three modes, Python 3.10 through 3.14, with no skipped
interpreter. The new `UMAP.fit` / `fit_transform` API passed its six test
groups in every combination, including the 16 pinned layout cells in
IDENTICAL mode. The installed Mamba and Transformer surface suites also
passed in all three modes on Python 3.12. See the
[qualification record](bench/results/wheels/2026-09-05-umap-api/release-status.json).
The tagged PyPI publication workflow is in progress; Linux installed-wheel
qualification remains separate.

## UMAP follow-up priority

UMAP is the next feature focus after the wheel release. Extend the identity
fixtures beyond the current 8x1 case to multidimensional data, 3D output,
multiple seeds and parameter settings, alongside independent embedding-quality
checks. Use RunPod for NVIDIA and **DigitalOcean for AMD**, with tests and
measurements in the main lane only.

Then add fitted-state `transform` following the pinned cuML implementation:
new-to-training k-NN, membership weights, weighted initialization from the
training embedding, and layout optimization against the retained embedding.
The current API intentionally refuses that operation. Quadratic graph storage
also needs a sparse path before claiming large-dataset scalability.

## Now: release truth and artifact closure

1. Build a clean wheel and verify all 15 native extensions in `fast`,
   `deterministic`, and `identical` mode from an isolated installation.
2. Stamp native artifacts with source/build identity and refuse stale
   artifacts at import or release time.
3. Generate extension, public-surface, architecture, and certification tables
   from machine-readable registries instead of repeating those facts in prose.
4. Classify remote results as `PASS`, `EXPECTED_DIVERGENCE`, `INFRA_FAILURE`,
   or `ALGORITHM_FAILURE` before they enter summaries.

## Next: close claims already exposed

- Run the dedicated ARIMA optimizer/fit correctness gate. The Python fit
  surface has only been smoke-tested on one Apple M4; NVIDIA and AMD remain.
- Close current NVIDIA legs for Holt-Winters, spectral clustering, TSA,
  Mamba 3, transformer bindings, and other recently exposed surfaces.
- Localize and fix the recorded AMD `_mojolearn_mamba` binding memory fault
  before making an AMD Python-surface claim.
- Add independent numerical references where cards currently prove stable
  bits without validating the calculus or accuracy, especially transformer
  backward and training/embedding paths.
- Exercise shipped-scale and plan-invariance fixtures for newer neural and
  training operators.

## NVIDIA performance campaign

Use one guarded NVIDIA rental only after the artifact gates above are green.
Every method gets exactly three interleaved arms:

1. mojolearn `fast`;
2. mojolearn `identical`;
3. one external comparator appropriate to the method.

Use CatBoost GPU for GBDT, cuML for an equivalent classical estimator when it
exists, PyTorch CUDA for GEMM/training/transformer operations, mamba-ssm for
Mamba, and scikit-learn CPU only where no GPU equivalent exists. Record warmup,
at least seven samples, explicit synchronization, median and IQR, accuracy,
output hashes, versions, driver, device, architecture, and commit. Preserve
raw output; publish no ratio from separate rental sessions.

FAST is intended to be the fastest mojolearn tier. Treat a repeatable
`identical/fast < 1.0` ratio as a performance defect, not an interesting
anomaly. Require five interleaved rounds and three representative sizes
before changing a default; ratios whose ranges overlap remain inconclusive.
Current candidates are GBDT's row-index-only route and DBSCAN's scheduling
path. Do not weaken IDENTICAL to make the comparison green.

## Algorithmic work after closure

- Complete CatBoost ordered boosting: per-fold approximation cursors, weak
  targets, leaf estimation, and model averaging. Until then, comparisons pin
  CatBoost to plain boosting.
- Complete tree CTR/feature-combination wiring if categorical parity remains
  a product priority.
- Consider AutoARIMA/search only after the existing ARIMA fit is independently
  validated across vendors.
- Optimize only profiles that show a representative workload gap. Do not add
  estimators merely because an upstream implementation exists.

## Release rule

Evidence does not transfer automatically between source checks, bindings,
built artifacts, and installed wheels. A release claim requires the installed
wheel layer to pass. Cross-vendor identity is claimed only for a named profile,
fixture, commit, and recorded hardware column.
