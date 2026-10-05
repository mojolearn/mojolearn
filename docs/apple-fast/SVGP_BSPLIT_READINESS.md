# SVGP BSPLIT: source review, not queue admission

Reviewed 2026-10-04 against origin/main
`34795a43c23f64790859977f5520048c51378771`. Candidate originated on
`lane/apple-fast-w2-svgp@146898d2c`; its implementation is already in main.
No kernel cherry-pick is needed. No build, numerical run, timing, cloud access,
queue insertion or promotion was performed for this review. SVGP already wins
its board comparisons; remaining losing rows take priority.

## Route and hypothesis

`SVGP.fit` in `python/mojolearn/_expansion_neighbors.py` calls `svgp_fit_ff`.
`x_neighbors/iter_device.mojo` selects BSPLIT only with FAST, Apple, SYMTILE
and `MOJOLEARN_SVGP_FAST_BSPLIT`. Keep current RBFTILE/BLKCHOL defaults and
SYMTILE enabled; do not bundle rollback flags into this experiment.

B = Kfu-transpose times Kfu uses four row slices; b = Kfu-transpose times y
uses 32. Each slice retains its float-float partial across row tiles, then
partials fold in ascending slice order. Integer slice boundaries cover rows
without overlap, including tails/empty slices. Symmetric tile code stores both
triangles. The reduction reassociates arithmetic, so output bits may change.
No obvious source indexing defect was found; this is not numerical evidence.
This path uses dedicated float-float statistics kernels, not shared GEMM;
shared float32 GEMM wins cannot be credited to it.

## Exact artifact preparation contract

Choose the full final harness/source commit H after review. Build both arms
from H on M2 through `~/mojolearn-evidence/compile_slot.sh`, with
`MOJOLEARN_COMPILE_JOBS=1`, `MOJOLEARN_NUMERIC_MODE=fast`,
`MOJOLEARN_SKIP_BUILD_GATE=1` (compile only), invoking
`bash bindings/build_x_neighbors.sh`. Use a persistent owned TMPDIR.
`MOJOLEARN_BUILD_EXTRA_DEFINES` is empty for A and exactly
`-D MOJOLEARN_SVGP_FAST_BSPLIT` for B. Clear unrelated inherited build flags.
Copy each completed artifact before the next build overwrites it.

Stage `~/mq/verified-arms/H/x_neighbors/{A.so,B.so,manifest.json}`.
Manifest fields: `source_sha=H`, `binding=x_neighbors`, `numeric_mode=fast`,
`defines_A=""`, `defines_B="-D MOJOLEARN_SVGP_FAST_BSPLIT"`, and `hashes`
with exact SHA256 values for A/B. Also retain compiler/environment provenance.
Do not reuse the old branch's binary against modern Python/native sources.

Explicitly prepare the IDENTICAL base dependency from H using
`MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh` through the same
compile semaphore, compile-only setting and empty candidate defines.
The package's `_buffer._native` resolves conversions against IDENTICAL even
in FAST: possible requirements include `cast_f64_to_f32`,
`cast_colmajor_f64_to_f32`, `transpose_f32`, and finiteness helpers. An x_neighbors
FAST binary alone does not establish a complete fresh installation.
Use the single-artifact manifest schema in JOB_PREFLIGHT.md with binding
`ibase`, artifact `_mojolearn.so`, numeric_mode `identical`, defines empty,
source H and exact hash; install at `python/mojolearn/identical/_mojolearn.so`.
Verify exports/import in the final case helper, not merely manifest strings.

## Harness contract and outstanding infrastructure

Existing invocation contract is:

```
python tools/svgp_fast_pair.py quality H QUALITY_TAG
python tools/svgp_fast_pair.py timing H QUALITY_TAG TIMING_TAG taxi
python tools/svgp_fast_pair.py timing H QUALITY_TAG TIMING_TAG istella
```

Quality captures both board datasets, then compares predictions against query
labels using independent NumPy R2/RMSE calculations and compares ELBO. Existing
one-sided 1e-4 gates are unchanged. Fitted arrays must be finite; their relative
differences remain diagnostic. This is main-versus-candidate quality, not a new
opponent measurement or an independent posterior/ELBO oracle.

The v2 fixture additionally binds X/y/Xq/inducing points/hyperparameters across
arms and checks matching shapes and finite baseline state. Old PASS receipts
are intentionally incompatible. The pair helper now restores a prior binary
if one exists, or removes its installed arm on exit if the tree was fresh.

**Not yet ready for pinned-runner submission:**

- `svgp_fast_pair.py` is not allowlisted by `apple_fast_job_policy.py`.
- The helper does not yet install/hash/revalidate the IDENTICAL dependency;
  installation must be implemented with restoration before adding an allowlist.
- Timing still delegates to external `~/mq/verified_arms.py`; review its exact
  version/build suppression and dependencies or replace with a committed,
  pinned no-build timing path before admitting it.
- Data currently uses the legacy board-0834/board-0833 directory search. Pin
  explicit dataset paths/hashes in the submission/runtime receipt; arm input
  equality alone does not pin the historical board fixture.
- All artifacts are unbuilt here. No artifact hashes, successful native exports,
  actual route counters, quality PASS or timing admission exist from this review.

Manager next steps: finish these helper prerequisites, build and stage artifacts
in the serial transfer window, then prepare the metadata preflight with both
artifacts and exact data requirements. Queue quality first. After valid quality,
run one scored sample per arm per dataset, no opponent rerace. Preserve any
failures under fresh tags; do not turn an infrastructure failure into HOLD-quality.
Additional tail/inducing-count boundary quality is owed before any broad default.
