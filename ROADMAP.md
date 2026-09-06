# Roadmap

This is the only live project plan. Historical plans and handoffs are not
current instructions; git history and `archive/` retain their evidence.

## Installed API gap closure (2026-09-06, in progress)

The audit confirmed that source capabilities exceed the published artifacts:
PyPI currently has macOS 0.5.0 and Linux 0.3.1, with no published 0.6.0.
UMAP transform and CSR fitting therefore remain delivery priorities even
though their named native/source checks pass.

Source `eb835021` adds the narrow public `OrderedRMSE` estimator, retaining
explicit permutation, numeric RMSE, sample weights and IDENTICAL selection.
It is distinct from general CatBoost ordered boosting. Thirty lightweight
Python boundary checks pass. All 45 AMD extensions now build and all 24
installed jobs pass, including the new ABI in every mode; matching NVIDIA
qualification is in progress. Native backward does not expose Python Mamba backward.

The installed Linux gate now requires all 45 extensions, isolated package and
binding hashes, 24 serial jobs, UMAP fit/transform and six expanded held-out
quality fixtures in every mode, ordered RMSE, Mamba, Transformer and fitted
ARIMA. A retained comparison checks IDENTICAL UMAP input/embedding bytes
across AMD/NVIDIA. Missing statuses, changed sources and incomplete evidence
are refusals, never inferred passes. Six synthetic evidence checks pass.

The main operator alone builds/tests/measures: one GPU rental at a time,
CPU affinity at most four, compiler workers two, BLAS/OpenMP one, and no
parallel tier builds. The repaired AMD wheel added eight empty ZIP directories outside RECORD;
normalization removes only those entries and verifies all 91 payload files
unchanged. The qualified AMD wheel SHA256 is
`7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`.
AMD was deleted and confirmed absent before NVIDIA was created. Current
candidates are not publication or universal identity claims. Results are retained under
`bench/results/resume/2026-09-06-installed-gap-closure/`.

## Native certification continuation (2026-09-06)

Mamba source `395d9421` passes all five baseline and both long cases on AMD
MI325X and NVIDIA RTX 4090. All 54 baseline and 21 long-profile gradient
tensors match by bits. Mamba3 L65 additionally matches all 86 diagnostics/public
gradients and nine forward operands. Corrected continued-check source
`6dd44ac5` passes ordered RMSE, weighted CTR and all four kNN flag combinations
on both GPUs: 130 ordered records and 5,440 selected index/distance pairs per
kNN arm match exactly. See the [complete record](bench/results/resume/2026-09-06-ordered-mamba-knn/README.md).

Mamba3 L65 now retains all 76 diagnostics and ten public gradients. Every
public gradient must pass the independent whole-forward float64 oracle.
Thirteen intermediate outputs use an explicit compositional contract:
independently validate their operands, reconstruct the prescribed float32
wrapped-angle recurrence and FMA/reduction DAG, and require exact output
bits. Direct reference differences remain visible; this does not claim
that all intermediates satisfy direct whole-float64 tolerance. No tolerance
was widened and no failing output was omitted. See the
[contract](mamba/BACKWARD_CERTIFICATION.md).

The native [ordered RMSE entry](gbdt/ORDERED_RMSE.md) now trains with
independent per-fold approximation cursors, prefix-only leaf estimation,
and a separate exported-model cursor. Its gate checks independent replay,
weighted fit/predict, leakage controls, zero-mass prefixes, and constant-tree
prediction. This closes the numeric single-permutation implementation;
multiple categorical permutations and full external CatBoost parity remain.

Both kNN compile-time flags work with IDENTICAL. The new adversarial gate
checks 5,440 selected distance/index pairs across 32 cases, including
non-dyadic data, duplicate rows, high offsets and odd dimensions. All four
arms agree exactly on both GPUs. See [build usage](neighbors/README.md). These flags
are compiler defines, not runtime Python settings or rebuilt wheel options.
Existing UMAP expanded quality/identity evidence remains current for its
recorded source; this continuation does not repeat that performance work.

Only the main operator runs checks; GPU work is serial, remote CPU affinity
is four cores, compiler jobs are limited, and BLAS/OpenMP use one thread.

## Resumed implementation and identity checks (2026-09-05)

**Mamba3 certification correction:** the L65 diagnostic investigation found
a missing `trap.scale -> gamma -> dt/sigma` contribution in the backward
join. The staged float32 reference repeated it, so previous Mamba3 byte
matches and staged-reference passes do not establish a correct full gradient.
The independently differentiated float64 forward rejects the retained old
`x`, `block_norm.weight`, `in_proj.weight` and `dt_bias` outputs. The native
join and staged references are corrected at `eebd7c92`. AMD and NVIDIA now
pass all five baseline cases and match all 54 native tensors, with every
public Mamba3 leaf also checked against the independent whole float64 forward
at unchanged tolerances. See the [corrected comparison](bench/results/resume/2026-09-05-next-certification/corrected-backward-cross-device.json).
Apple has not been rerun for the correction. Historical Mamba3 byte equality
is retained as evidence, but its old gradient-correctness claim is superseded.

Mamba1 L64 passes on both GPUs. All ten Mamba3 L65 public gradients now pass
the independent forward oracle, and the 21 long-case public tensors match
across AMD/NVIDIA. At that historical source the complete long certificate was RED on 13
intermediate comparisons. The explicit September 6 contract above supersedes
that gate policy; the original direct differences remain retained.

UMAP's self-neighbor fix passes all six expanded quality fixtures in all
three modes on both GPUs. Both native stage fixtures (186 and 690 cells) and
all six IDENTICAL held-out embeddings match. See the [UMAP comparison](bench/results/resume/2026-09-05-next-certification/fixed-cross-device.json);
its older Mamba3 gate results are superseded by the correction above.
The [weighted CatBoost slice](bench/results/e1/2026-09-05_235251-amd-catboost-fixed-partition/README.md)
now passes on AMD, including fixed occupied zero-mass leaf estimation at
L2=0 and L2=3. A split optimizer is no longer required to choose that corner
case for its estimator to receive coverage.

Next work, with only the main operator testing/measuring and serial GPU jobs:

1. Re-run corrected Apple Mamba evidence separately when that hardware is
   in scope. The named AMD/NVIDIA native backward profiles are now closed.
2. Complete kNN installed-artifact/external and larger-scale coverage. The
   new adversarial distribution/dimension gate passes both GPUs, but neither
   experimental flag is promoted to default dispatch by correctness alone.
3. Extend the implemented numeric ordered RMSE path only with explicit
   coverage for additional permutations, categorical CTR and objectives.
   Keep general external comparator runs in plain mode until parity is scoped.
4. Keep 0.6.0 publication and Linux installed-wheel qualification separate
   from these native/source certificates.

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
The tagged [PyPI publication workflow](https://github.com/mojolearn/mojolearn/actions/runs/33958208803)
succeeded. The downloaded PyPI wheel matched the publication digest and passed
all nine smoke/Mamba/Transformer checks across the three modes on Python 3.12;
see the [post-publication record](bench/results/wheels/2026-09-05-umap-api/postpublish/results.json).
Linux installed-wheel qualification remains separate.

## UMAP follow-up priority

Resumed campaign after the crash (main operator only; serial GPU
workloads; AMD on DigitalOcean, NVIDIA on RunPod):

1. Close AMD's missing four-arm kNN layout qualification against the retained
   Apple/NVIDIA evidence. Keep the selector and transpose flags opt-in.
2. Run the five existing Mamba backward cases and a separate
   `long-sequence-v1` certificate: Mamba1 `base_b1_l64_d8` and Mamba3
   `m3_base_b1_l65_d64`, requiring every public-prefill gradient. These 21
   additional tensors must not be folded into the historical 54-tensor claim.
3. Run UMAP's opt-in `expanded` held-out quality profile in all three modes:
   the original two cases plus 128-training-row cubic and saddle fixtures,
   each at two additional seeds, 15 neighbors and min_dist 0.2. Retain the
   original thresholds and both correspondence-breaking controls. Compare
   IDENTICAL inputs and embeddings only after both hardware legs finish.
4. The later September 6 continuation implements numeric ordered RMSE
   with per-fold cursors and prefix-only leaf estimation. Remaining
   categorical/permutation scope and external parity are still open.

The first expanded NVIDIA run at `5658d28e` passed all five baseline Mamba
cases and Mamba1 L64, but exposed an obsolete partial manifest in the Mamba3
L65 driver. Expanded UMAP passed all six IDENTICAL cases; the larger cubic
fixtures failed in FAST and DETERMINISTIC because the raw same-data kNN result
did not put self first. See the
[retained failures](bench/results/e1g/2026-09-05_175405-nvidia-mamba/README.md).
The first fixes normalized UMAP's self slot in both graph adapters and exposed
the already-computed Mamba3 public gradients. The later independent-gradient
investigation and corrective results are recorded above.
AMD at `6a3a2d30` passed the self-neighbor regression and all six expanded
UMAP cases in every mode, plus all five baseline backward cases. Its long
certificate failed before execution because host `python` was absent from
PATH; the launcher now runs inside pixi. The weighted CatBoost fixture
did not exercise its required zero-weight leaf and was RED; the fold-axis
gate passed. See the [AMD record](bench/results/e1/2026-09-05_223146-mojolearn-e2-amd/README.md).
The matching `d88c7883` campaign closed the expanded UMAP matrix and exposed
the Mamba3 chain-rule defect. No numerical threshold was changed in response
to either failure.
The serial follow-up payload uses its former two-arm kNN timing slot for the
long-sequence certificate; kNN timings belong to the dedicated four-arm leg.

UMAP and sequence certification are the current feature focus; artifact
publication remains a separate release gate. Extend beyond the recorded
small fixtures alongside independent embedding-quality checks. Use RunPod
for NVIDIA and **DigitalOcean for AMD**, with tests and
measurements in the main lane only.

The 0.6.0 source candidate now implements fitted-state `transform` and CSR
graph storage for public fitting. Integrated fit/transform API checks passed
in all three modes on Apple, NVIDIA and DigitalOcean AMD. Named held-out
IDENTICAL inputs and embeddings also match between those source bindings
and the exact installed macOS candidate; see the
[held-out comparison](bench/results/wheels/2026-09-05-umap-060/cross-vendor-heldout.json).
These fixture results do not establish large-dataset scalability.

The exact macOS 0.6.0 candidate passed the Python 3.10–3.14 smoke matrix and
additional installed fit/transform and held-out quality checks in all three
modes. Its build-only workflow failed during artifact upload after the build
and smoke checks passed; see the
[candidate qualification](bench/results/wheels/2026-09-05-umap-060/README.md).
Complete artifact delivery, publication and fresh Linux installed-wheel
qualification next. Broaden quality fixtures and measure scalability separately.

The later build-only workflow `33974940904` passed its standard smoke matrix
but failed the additional UMAP installation when the dependency index could
not resolve. The preserved exact wheel now passes all nine installed UMAP
checks using an offline dependency wheelhouse, with unchanged named IDENTICAL
inputs and embeddings versus the prior candidate. See the
[recovery evidence](bench/results/wheels/2026-09-05-umap-060-install-recovery/README.md).
The qualifier records dependency hashes/versions and checks dependency
consistency; the workflow also now refuses multiple candidate wheels correctly.
This local recovery does not turn either failed workflow into a publication.

The latest k-NN work adds opt-in transposed distances alongside the opt-in
small-k selector. Qualify baseline, selector-only, transpose-only and combined
builds with the four-arm public driver before changing dispatch defaults.
Earlier two-arm selector results do not qualify the new distance layout.
The resumed dirty source passed all four IDENTICAL correctness arms on Apple
M4: all 143,628 selected distance/index pairs matched exactly. See the
[local qualification](bench/results/resume/2026-09-05-layout-local/results.json).
The subsequent full Apple/NVIDIA campaign at `9fe07a33` passed all four
correctness arms and 108 timing invocations per vendor. Every selected output
bit matches across vendors, arms and rounds; see the
[full comparison](bench/results/resume/2026-09-05-layout-apple-price/cross-vendor-summary.json).
At 1,000 queries, NVIDIA median native request times were 759.333 ms baseline,
17.241 ms selector-only and 9.273 ms combined. Apple did not reproduce those
gains, and its combined arm was slower at 32 queries. Keep both experiments
opt-in: broader datasets and installed-artifact gates remain pending.
The resumed DigitalOcean AMD MI325X campaign at `64e70035` now passes all
four correctness arms and 108 timing invocations. Complete outputs match
NVIDIA and the validated Apple output hashes. At 1,000 queries AMD medians
were 66.968 ms baseline, 65.703 ms selector, 6.671 ms transpose and 5.522 ms
combined: its benefit is primarily from transposition. See the
[AMD record](bench/results/e1/2026-09-05_215006-mojolearn-e2-amd/README.md).
Collection completed and droplet deletion was verified by GET 404.
The separate [mode-isolation gate](bench/results/resume/2026-09-05-layout-modes/results.json)
passed on Apple M4 in FAST and DETERMINISTIC, with neither and both
experimental defines: both effective flags stayed disabled, and public
host-reference, alternate-method and query-tile checks passed. The guarded
NVIDIA layout controller also passed its dry run. The later rental completed
successfully and was deleted with verified HTTP 404. Its original controller
reported an empty console as missing, and the initial parser rejected an
allocator warning before the benchmark header. Both reporting defects now
have regression controls; the unchanged raw evidence passes revalidation.

## Now: release truth and artifact closure

Crash-resume checkpoint (2026-09-05): the release workflow now refuses a
macOS upload candidate that differs from its passing UMAP qualification.
The new read-only `tools/verify_umap_qualification.py` gate checks the wheel
digest, complete successful job inventory, frozen qualification sources,
and installed wrapper/binding hashes in all three numeric modes. The retained
0.6.0 recovery candidate (`dab65d03...18ea02`) passes; nine new artifact and
workflow controls, three existing qualification command tests, and 15 release
artifact controls pass. No build or GPU measurement was run for this checkpoint.
GitHub runs 33974940904 and 33972158535 remain failed. This closes the gap
between qualification and upload hashing; native source/build identity stamps
and the remaining release/remote gates below are still open.

1. Complete 0.6.0 artifact delivery/publication and refreshed Linux wheel and
   installed-API gates on NVIDIA and DigitalOcean AMD. The exact macOS 0.6.0
   candidate's installed checks passed; its upload workflow failed.
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
- Complete AMD installed-wheel Mamba qualification. The binding fix and newer
  source forward/state API checks passed; source checks do not certify a wheel.
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

- Extend the native single-permutation ordered RMSE implementation to the
  remaining categorical/permutation and objective scope. General comparisons
  still pin CatBoost to plain boosting; full parity is not established.
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
