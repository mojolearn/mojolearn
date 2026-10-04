# Shared GEMM downstream quality checkpoint (source only)

Base: shared dispatcher repair `a9e64922f`. No changes to G1/G5 kernels,
product defaults, numerical modes or fallback policy. No builds, models or
numerical tests have run for this checkpoint. Matrix-screen quality/timing
selects the variant and caller shapes before downstream M2 builds.

## Actual binding matrix and build order

| Priority | Family / extension / build script | Cases and actual routes | Coverage caveats |
|---|---|---|---|
| 1 | core / `_mojolearn` / `bindings/build.sh` | `kmeans`: kmeans++ candidate seeding calls core NT (route 0); `knn-wide-k`: tiled brute distance product calls core NT | KMeans assignment, predict and transform are fused/direct cells, not shared GEMM. Default small-k KNN Apple distance specializations bypass it. Reach must come from measured phase deltas. |
| 1 | estimators / `_mojolearn_estimators` / `bindings/build_estimators.sh` | `ols`: inverse solve product; `pca`: transform and inverse call core NT; `ridge` and `kde`: controls under current defaults | OLS prediction n=1 takes GEMV unchanged; PCA TN Gram and direct decomposition kernels are not this adapter. Ridge NO_U removes its NT product. KDE default fused path bypasses NT at all fixture widths. |
| 2/control | kernel_methods / `_mojolearn_kernel_methods` / `bindings/build_kernel_methods.sh` | `rbf`: actual RBFSampler resident fit_transform and ordinary transform; vendor NN is route 2 | Resident `_rbf_gemm` tries AFN first. An AFN hit is an intentional non-reaching control, not proof that shared G1/G5 accelerated RBF. Do not disable AFN just to force coverage for a board claim. |
| 2 | svm / `_mojolearn_svm` / `bindings/build_svm.sh` | `svc`: actual binary RBF fit, decisions and predictions through kernel_op/core NT | SVC/SVR are **not** in estimators. LinearSVC is a different algorithm. RBF d<=64 can take fused tiles; d=65/220 supplies a plausible NT case, while d=11 is a useful control. |

Start with only core and estimators for the chosen variant: two diagnostic
arms each, not all variants × bindings × datasets. Kernel methods and SVM
are separate follow-ups if the matrix screen motivates them. Existing
board-quality holds remain holds; this is not opponent-quality accounting.

## Static instrumentation, not a runtime selector

New diagnostic define `MOJOLEARN_APPLE_FAST_SHARED_GEMM_COUNTERS` is
orthogonal to production G1/G5. Compile A with COUNTERS only, B with
COUNTERS plus exactly the selected `MOJOLEARN_APPLE_FAST_SHARED_GEMM_G1`
or `_G5`. Do not pass AUDIT. The manager's compile helper must permit the
nonempty A define list; no fake empty-A shortcut is provided here.

Five diagnostic exports are registered inside each **actual extension**:
`shared_gemm_reset`, `shared_gemm_count(route,column)`, `shared_gemm_variant`,
`shared_gemm_mode`, `shared_gemm_vendor`. Each extension reads its own
shared-dispatch state. A custom probe extension's counters are never used.
Reset/read and calls must be serial. Four route rows remain core NT,
core Gram, vendor NN, vendor NT; columns remain fallback, G1, G5, total
candidate. A COUNTERS-only build records incumbent arrivals even though
production candidate dispatch is disabled. Imports, variant metadata and
zero candidate counters cannot satisfy the reach gate. The AUDIT-only
matrix harness retains its original selector and counters behavior.

COUNTERS absent: no new counter updates or diagnostic exports. Production
G1/G5 remain OFF by default. Final timings should use separately compiled
non-diagnostic production arms, with provenance distinguishing them from
quality binaries; this checkpoint supplies no timing function.

## Unscored capture adapter

`tools/shared_gemm_downstream_quality.py` calls real Python estimators and
records fit/query counters independently, saves full caller-owned output
copies and fitted state, and binds records to source HEAD, actual loaded
binary SHA, numeric mode/vendor, fixture hash and capture SHA. Source must
be clean. Run fresh processes for A and B after manager-controlled verified
installation. Both arms must come from this same source, with the same
fixture width. Keep the existing matched base dependency bindings installed;
this helper neither compiles nor swaps them.

```
MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python python tools/shared_gemm_downstream_quality.py dump SOURCE A_BINARY_SHA 0 ols evidence/A.npz --features 65
MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python python tools/shared_gemm_downstream_quality.py dump SOURCE B_BINARY_SHA 1 ols evidence/B.npz --features 65
python tools/shared_gemm_downstream_quality.py compare evidence/A.npz evidence/B.npz evidence/report.json
```

Use variant 5 instead of 1 when chosen. Cases are `kmeans`, `knn`, `ols`,
`ridge`, `pca`, `kde`, `svc`, `rbf`. Fixed synthetic fixtures: 513 training
rows, 73 query rows, widths 11/65/220 (default 65), seed 724190, with 7
clusters/neighbors/components or 67 RBF features. These are small,
algorithm-level quality gates, not board-shaped performance evidence.
Choose width/cases before execution. No repeated scored calls exist; all
fits here are unscored quality. No automatic full crossproduct is supplied.

## Fixed quality contract

Every numeric metric reports A/B relative L2 and max-absolute error
independently, and B must be no worse than A in **both**, with zero added
tolerance. For a zero-valued algebra-residual oracle, the L2 denominator is
1 (absolute L2). Every saved value must be finite. No metric averaging,
post-hoc epsilon, solver tolerance change or accepted noise band is allowed.

- OLS/ridge: float64 centered least squares / alpha=1 ridge coefficient,
  intercept and full prediction oracles.
- Brute KNN: exact neighbor indices against A and the float64 all-pairs
  oracle, plus independent distance errors. Stable index ordering is fixed.
- KMeans: exact labels, query labels and iteration count; float64 empirical
  centers for the common assignment, inertia, transform distances, and
  oracle query labels. Empty clusters explicitly refuse this oracle.
- PCA: float64 SVD projector, means, variance, singular values and query
  reconstruction; transform algebra residual checked separately using
  each arm's components. Projector avoids arbitrary singular-vector signs.
- KDE: float64 Gaussian logsumexp oracle at bandwidth 2, all query scores.
- SVC: support indices/vectors, dual coefficients, intercept, classes,
  support counts and predictions must be byte-identical. Decision errors
  compare against float64 RBF evaluation of that common saved model. This
  validates unchanged fit state and downstream evaluation; it does not
  claim an independent convex optimization oracle.
- RBF: exact sampled weights/offsets/scale, then float64 projection+cosine
  oracle for all fit_transform and query outputs.

A must launch zero candidates. B must launch only its selected variant,
and at least one actual estimator phase must launch a candidate. Reports
retain all phase counters. `NO_REACH` is a non-pass/control result even if
numerics agree; `HOLD` means reached but a strict metric/exact gate failed.
`PASS` means only this fixture/variant passed. No defaults, board promotion,
production precision claim or timing authorization follows automatically.

## Deliberate limits before broader rollout

The adapter is source-ready, not M2/M3 validated. Installed SDK compilation
and Python API integration remain gates. It does not implement a new serial
pair installer, provenance intake, dataset caching or board-row promotion:
manager-owned tooling handles those. It does not yet test multiclass GLM,
SVR, randomized PCA, neural callers, nonfinite inputs, rank-deficient linear
fixtures, or cached full taxi/istella fitted-estimator rows. Those are
separate predeclared fixtures after the first focused cases pass. No
production expansion into MCD/other batched reductions is included.

## Variant-pinned pair build and installer contract

The G1 and G5 follow-up branches differ only in tracked
`tools/shared_gemm_variant.json`; each therefore has its own source SHA and
`~/m2-arms/SOURCE/FAMILY` / `~/mq/verified-arms/SOURCE/FAMILY` namespace.
The config fixes variant, base checkpoint, exact defines and allowed
families. Build manifests include its SHA-256; M3 requires byte-identical
config identity, actual artifact hashes, and runtime variant metadata.
Do not build a second variant into an existing source/family directory.
No manifest/arm overwrite is allowed, including a failed partial build.

On the manager's **M2 private checkout** pinned to the selected G1 or G5
branch, run these compile-only commands, one family at a time as selected:

```
SOURCE=$(git rev-parse HEAD)
bash tools/shared_gemm_build_pair.sh "$SOURCE" core
bash tools/shared_gemm_build_pair.sh "$SOURCE" estimators
```

The script verifies Apple M2 hardware, exact clean source, disk >=8 GiB,
and takes the existing build lock. It invokes the real build scripts with
FAST, `MOJOLEARN_SKIP_BUILD_GATE=1`, `MOJOLEARN_COMPILE_JOBS=1`, explicit
`metal:1` accelerator, Apple column, and cleared inherited compiler flags.
No extension import, model execution or numerical check occurs on M2.
A receives exactly COUNTERS; B COUNTERS plus the pinned G1 or G5. The
Apple column define is separately declared target metadata shared by both.
It checks the defined CPython init symbol with `nm -gU` before writing the
manifest. Core maps to `bindings/build.sh` and **`_mojolearn.so`**, never
`_mojolearn_core.so`; estimators maps to `bindings/build_estimators.sh` and
`_mojolearn_estimators.so`.

This dedicated contract avoids the old `compile_arms_m2.sh` assumptions of
empty A defines and `_mojolearn_${family}.so`. The manager still owns source
transfer to M2, invocation, artifact transfer to M3, and serial queue intake.
Copy each completed directory intact to M3's verified-arms namespace during
its allowed transfer window. Do not use old `verified_arms.py`, which assumes
empty A defines, to install these diagnostic pairs.

M3 unscored pair, inside the serial queue at the same source pin:

```
MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python tools/shared_gemm_downstream_pair.py SOURCE UNIQUE_TAG ols --features 65
```

Case selects the family: core `kmeans|knn`, estimators
`ols|ridge|pca|kde`. Source config selects the variant, not a runtime flag.
The helper checks manifest/source/config/defines/mode/target/hash bindings
before installation. It acquires a nonblocking pair lock, refuses existing
evidence tags and symlinked installed bindings, backs up an existing target,
atomically installs each verified arm, and starts a fresh capture process.
The capture verifies the actual loaded path and SHA. Comparison is unscored
and has no timing path. The original target is restored and hash-checked in
`finally`; when no original existed the temporary target is removed.
`restore.json` records successful restoration. `PASS.json` can be issued only
after comparison PASS **and** successful restoration; HOLD/NO_REACH cannot
issue PASS. Exceptions preserve logs and the backup, then propagate failure.
A hard kill/power loss cannot run `finally`: retained `intake.json` and
`original.so` support manager recovery; never blindly rerun the same tag.

Current synthetic fixture choices and strict oracle rules above remain
unchanged. Select a small case/width set before execution using the matrix
screen; this script does not automatically run either variant or a full
crossproduct. Final production timings must use separately declared builds
without COUNTERS and still need the appropriate downstream quality gate.

## Predeclared first downstream set and corrected tooling revision

This source audit is prior to downstream quality results. Runtime counters
remain required; the following expected routes are not measured claims.
Use the same selected native variant for this small set, then inspect all
metrics before extending coverage:

| Order | Case / width | Expected candidate phase and shape | Intentional non-reaching phases |
|---|---|---|---|
| 1 | `ols --features 65` | fit inverse product, core NT route 0, 65x65 output with K=65 (`lstsq.mojo:571`) | single-target predict GEMV |
| 2 | `pca --features 65` | transform core NT: M=73,N=7,K=65; inverse core NT: M=73,N=65,K=7 (`decomposition/estimator.mojo:453`) | covariance TN/fused fit stages do not prove shared reach |
| 3 | `knn-wide-k --features 65` | k=65 excludes certified MMA (max24), FAST MMA (max64), FAST top-k and fused-L2 (max64); tiled L2 product core NT M=73,N=513,K=65 | index fit has no candidate launch |
| 4 | `kmeans --features 65` | default scalable seeding can invoke classic kmeans++ on candidate rows (`kmeans.mojo:1141`), NT with N=n_trials=4,K=65 | main assignment, predict and transform are fused/direct cells; transform is **squared L2** for this API's default `euclidean` metric |

If scalable initialization does not collect more than seven candidates,
it may skip the classic seed reduction; count zero is NO_REACH, never
permission to claim a KMeans speedup. The fixture uses the real default
oversampling=2 and does not force a branch just to produce a reach count.

Minimal control: `knn --features 11` (k=7) should take certified MMA before
shared dispatch, hence NO_REACH. Further cheap controls when useful:
`ridge --features 65` has default NO_U, TN Gram plus vector reductions,
and `kde --features 65` has the default fused pass. KDE also remains fused
at d=11 and d=220 (`n_features <= KDE_FUSED_TILE_FLOATS`); no need to
blindly run all widths. All NO_REACH cases remain non-passing for promotion
and timing regardless of oracle agreement. Ridge d=220 also uses default
Apple TN-v1 rather than this shared adapter; do not interpret that as NN.

These initial cases cover **core NT route 0**, not every shared route.
PCA inverse is mathematically an NN reconstruction, but it transposes the
components explicitly and enters core NT; it does not exercise vendor NN
route 2. Aliased Gram route 1 needs a separate underdetermined OLS fixture
(the current tall fixture does not reach it). Vendor NN/NT routes 2/3 need
real eligible callers beyond this first group, with AFN/fused precedence
checked before selecting them. RBF's AFN-first path is a control unless
actual counts demonstrate otherwise. This is a bounded first gate, not an
'all callers passed' statement. Default small-k KNN cannot benefit from
this adapter merely because its fallback imports core GEMM.

`shared-downstream-f64-independent-errors-v2` corrects the KMeans transform
oracle from sqrt distance to the public API's squared distance. This is a
source-contract correction discovered before execution, not a tolerated
quality regression. It adds the predeclared k=65 KNN case; original k=7
remains the control. Every metric still requires B relative-L2 and
max-absolute error <= A independently, without an epsilon. Exact KNN
indices, KMeans labels/query labels/iteration counts and saved-state gates
remain. PCA sign-independent projector/reconstruction checks and its
separate transform-arithmetic residual remain. No metric is averaged.

### Reuse unchanged native artifacts with newer quality tooling

`SOURCE` in the pair/dump CLI is now explicitly the **compiled source**.
The running harness HEAD may be a descendant only if
`shared_gemm_source_contract.validate_source` verifies ancestry, clean
tracked source and zero diff outside an exact allowlist of the two quality
helpers, source-contract validator and this documentation file. Untracked
possible source files are rejected. Any `.mojo`, production Python,
build/config/lockfile, or variant-config change outside that exact allowlist
rejects reuse. G1 and G5 retain their separate tracked variant identities;
there is no cross-variant reuse. The manifest/config/binary SHA and actual
runtime variant checks are unchanged. Captures, reports and receipts store
both compiled and harness source SHAs plus hashes of every allowlisted
harness file. A/B must use identical harness provenance. Old v1 captures
cannot be compared under v2. Do not rebuild native code for this tools-only
repair; compile sources remain G1 `30e4562c2129569ed03d93d878ec6a903ea51691`
and G5 `495c30c33a8805a1944b45f9bc911446f7e89ed8`.
