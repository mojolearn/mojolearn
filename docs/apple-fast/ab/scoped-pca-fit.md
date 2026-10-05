# Scoped public PCA fit: remote-only harness

Compiled source C: `201fe7367461af236f4adf0331f367d8a02cd29d`.
Remote branch: `lane/apple-fast-scoped-pca-remote-harness`, descendant of C.
H denotes its final reviewed harness commit. Reconstructed on M2 without
reading laptop-only helper/WIP. All new source work and metadata checks happen
on M2. No GPU/model runtime or timing was performed by this lane.

## Reach before builds

PCA board uses `PCA(n_components=10, svd_solver='covariance_eigh', whiten=False,
random_state=7).fit(X)` on the full unscaled `big-istella` block. Its 220-column
covariance is aliased TN M=N=220, K=fit-row-count, reaching G1_GRAM+SPLIT+PCA.
The helper reads every stored X row without sampling, verifies 220 columns,
and requires exact runtime route2 counters and current split/stride metadata.
The manager must supply the authoritative full board block and recorded hash;
shape alone does not establish that a file is the historical board dataset.
Taxi's <=128-column fused covariance remains a NO_REACH control.

Do not build expensive RSVD/LLE cases for this selector yet. Board randomized
SVD uses rank8+oversampling10=18: its NN range products miss TALL N>=32 and
NARROW N<=16; the small decompositions miss GRAM M>=129. Board standard LLE
uses n=10000 and nc=2 with thin iterative/vector products; dense distance
products exceed windows and narrow products have unsupported orientations or
K outside128..512. This is source analysis, not a claimed runtime NO_REACH
certificate. A different selector requires independent review/quality.

## Minimal build/artifact contract

Only FAST `estimators` A/B are needed for this exact public fit on C-contiguous
float32 input, whiten=False. No core, x_decomp or IDENTICAL base installation
is required by this specific call path; no input casts/whitening helpers run.
Each subprocess verifies the loaded estimators module path/hash/vendor/mode
and diagnostic exports before fitting. Both arms get metadata preflight before
either fit. A missing symbol/import is infrastructure failure, never quality.

Build at C with compile semaphore, `MOJOLEARN_COMPILE_JOBS=1`,
`MOJOLEARN_NUMERIC_MODE=fast`, `MOJOLEARN_SKIP_BUILD_GATE=1`, and
`bash bindings/build_estimators.sh`. Flags via MOJOLEARN_MOJO_BUILD_FLAGS:

- A: `-D MOJOLEARN_SCOPED_GEMM_AUDIT`
- B: A plus `-D MOJOLEARN_SCOPED_GEMM_G1_GRAM -D MOJOLEARN_SCOPED_GEMM_SPLIT -D MOJOLEARN_SCOPED_GEMM_PCA`

Remote M2 manifest reviewed under `~/m2-arms/C/estimators/manifest.json`:
A SHA256 `1226b5c1fe2e61e9ecd03a9f7120889071aa035201f2faaecf6ea864514b55c4`;
B SHA256 `37cba76b51ef8e8cee5b76f47a2a1d5393c539cf1efba7916b542ddeb3199d94`.
Runtime rechecks full source/binding/mode/ordered flags and both hashes.
Manager stages unchanged artifacts at `~/mq/verified-arms/C/estimators/`.

H permits C reuse only through ancestor proof, an explicit tools/docs allowlist
and zero tracked source drift. Any native/production/config difference is
rejected. New helper installs into its own pinned tree, preserves an existing
binary if present, removes its installation when none existed, and restores
on exceptions. No M3 native build, download, dependency-install or fallback.

## Quality, then separate one-call timing

```
python tools/scoped_pca_fit_spec.py C UNIQUE_Q_TAG quality /ABS/big-istella.npz DATA_SHA256
python tools/apple_fast_pinned_job.py H tools/scoped_pca_fit.py C UNIQUE_Q_TAG quality /ABS/big-istella.npz DATA_SHA256
```

These placeholders must be replaced with full commits/paths/hashes. Spec output
is JSON for `apple_fast_job_preflight.py`. It declares estimators pair and pinned
data file. Manager alone stages/preflights/enqueues. Deploy the policy addition
additively so newer main policy registrations remain intact.

The unscored quality run fits each arm once. Independent FP64 mean/covariance
uses all rows in bounded chunks, then eigvalsh provides the reference spectrum.
Public components, mean, explained variance/ratio, singular values and noise
are captured. Mean/spectrum, eigen-residual, orthogonality and reconstruction
errors are independently computed, with finite outputs, unchanged caller input,
5e-6 bounds on relative numerical errors and **zero error-regression allowance**
for every metric, including maximum-absolute errors. Reconstruction residual
is an approximation objective and is gated B<=A without an arbitrary 5e-6
rank10 approximation requirement. No sign-sensitive eigenvector comparison.
Atomic nondeterminism may produce HOLD; do not loosen gates or rerun for a win.

```
python tools/scoped_pca_fit_spec.py C UNIQUE_T_TAG timing /ABS/big-istella.npz DATA_SHA256 --quality-report /ABS/Q-quality/report.json --quality-sha REPORT_SHA256
python tools/apple_fast_pinned_job.py H tools/scoped_pca_fit.py C UNIQUE_T_TAG timing /ABS/big-istella.npz DATA_SHA256 --quality-report /ABS/Q-quality/report.json --quality-sha REPORT_SHA256
```

Timing validates exact report hash, source/H/helper/data/manifest/flags, PASS,
all metric gates, packets and positive candidate reach. A cross-tag exclusive
reservation blocks replay of the same source/data/artifact scored pair even
if interrupted. Do not remove reservations to replay completed arms; partial
failures need separately reviewed recovery of only unmeasured work.

One cold public constructor+fit+first complete fitted-output copy is timed per
arm in fresh processes. Imports/data loading/metadata checks are outside the
span; there are no warmups/repeats/opponent runs. Timing outputs are rechecked
against the saved independent oracle. This is a declared caller measurement,
not automatic replacement of board cells. Existing frozen-model transform/
inverse timings are a different operation and are not repeated.

`TAG-quality/report.json` / `TAG-timing/report.json` retain C/H/helper hashes,
exact data identity, complete manifest, packet hashes, shape, reach/metadata,
metrics and PASS/HOLD, with `board_admitted=false`. A quality failure writes a
HOLD report; infrastructure failures retain logs without fabricated measurements.

Before any promotion: compare native source C to current main, resolve any
relevant drift, assess authoritative dataset/opponent-quality status, validate
production non-audit default and rollback, and admit only useful measured scope.
Probe PASS alone is not estimator acceptance; estimator PASS alone is not speed.


## Serialization repair after first full Istella capture

`scoped-pca-fit-istella-q-v1` captured both arms and the independent oracle,
then failed while writing `report.json`. Orthogonality normalization produces
a NumPy float64; comparison produces a NumPy bool (named `bool` in NumPy2),
which the standard JSON encoder rejects. The repaired comparator converts
scalar representations to built-in float before unchanged comparisons; no
values, tolerance or metric definitions are relaxed. Receipt serialization
now completes before a file is created, avoiding encoder-truncated receipts.

The original M3 output is preserved untouched. Both A/B NPZ packets and their
metadata exist, as does oracle.npz. The partial report already records HOLD:
for example eigen-residual errors exceed5e-6, and a maximum-absolute error
regresses. Do not classify this as a passing quality job or replay the fits.
The full metrics can be reconstructed from saved A/B/oracle packets through
`compare(existing_output_directory)` without GPU work, fitting, timing or
rebuilding the full-data covariance. Any recovered report must be a NEW file
with original capture harness c982d77989a42e89034082dcdc86cae6aaf6bc7d, recovery
harness separately recorded, and hashes of all saved packets and original
partial report; never overwrite the failed report or substitute recovery
source for capture provenance. Such recovery remains manager-owned serialized
host analysis; it was not executed by this repair lane.


### Frozen-output recovery command

`tools/scoped_pca_recover.py H UNIQUE_TAG --spec` generates the metadata
preflight spec declaring all six original files and exact SHA256s. After
manager staging/preflight, the serial command is:

```
python tools/apple_fast_pinned_job.py H tools/scoped_pca_recover.py H scoped-pca-fit-istella-recover-v1
```

H is the final recovery commit, not the old capture or compiled commit.
It checks exact capture identity, repaired comparator hash, all packet hashes,
loaded-binary provenance from metadata, route/split records and input-preserved
flags. It performs only CPU comparison of saved fitted outputs against saved
220x220 oracle data. No native modules, fits, full dataset read, full covariance
rebuild or timing. The partial report is never opened for writing. The NEW
`TAG-recovery/report.json` keeps original C/capture-H/helper identity and adds
separate recovery source/helper/input hashes. A known-HOLD guard rejects an
unexpected PASS. Exit0 means reconstruction succeeded, never quality passed.
The spec's reference policy requires no native artifacts. This recovery was
prepared but not executed by the lane; the manager owns its serial execution.
