# Shared GEMM downstream quality checkpoint (source only)

Base: shared dispatcher repair `a9e64922f`. No changes to G1/G5 kernels,
product defaults, numerical modes or fallback policy. No builds, models or
numerical tests have run for this checkpoint. Matrix-screen quality/timing
selects the variant and caller shapes before downstream M2 builds.

## Actual binding matrix and build order

| Priority | Family / extension / build script | Cases and actual routes | Coverage caveats |
|---|---|---|---|
| 1 | core / `_mojolearn` / `bindings/build.sh` | `kmeans`: kmeans++ and transform call core NT (route 0); `knn`: brute distance product calls core NT | KMeans fused assignment is not shared GEMM. KNN Apple distance specializations can bypass it. Reach must come from measured phase deltas. |
| 1 | estimators / `_mojolearn_estimators` / `bindings/build_estimators.sh` | `ols`, `ridge`: solve products; `pca`: dense full fit, transform and inverse; `kde`: expanded distance calls core NT | OLS prediction n=1 takes GEMV unchanged; PCA TN Gram and direct decomposition kernels are not this adapter. KDE fused distance/density paths may bypass NT. |
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
