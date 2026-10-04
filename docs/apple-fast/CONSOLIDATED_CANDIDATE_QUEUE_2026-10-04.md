# Consolidated Apple candidate queue

User-requested intake on 2026-10-04. Source is
`experiment/apple-kernel-lanes-20261004@9ab2d3d3fb770498ef025db08f595a0149792bb7`,
`/Users/andrewhendel/CascadeProjects/mojolearn-apple-experiments/experiments/README.md`.
The source tree stays intact. These **19 source-only ideas are in the review
and A/B backlog**, not assertions that 19 runnable M3 jobs already exist.
The user now requests future A/B testing; the original source-only task's
deferred-execution boundary is superseded for this intake.

For each candidate: trace its actual current-main caller, isolate the opt-in
FAST/Apple source, compile on M2, verify outputs and refusal behavior on M3,
then permit one scored M3 run per arm under a unique tag. Preserve exact
source SHA, defines and binary hashes. Do not rerace opponents. Any changed
output storage is judged on call plus completion and first read. Promotion
also requires no quality degradation, parallel GPU computation, an `_OFF`
rollback and both promotion builds. No new machines.

## GEMM variants

All require `MOJOLEARN_APPLE_GEMM_EXPERIMENT`; modifiers below have prefix
`MOJOLEARN_APPLE_GEMM_`. Review the private intrinsic ABI on the installed M2
compiler. Direct controls and staged candidates need the same independent
float64 oracle, including rectangular/ragged/tail/zero-K, cancellation,
dynamic range and aliased-input Gram cases. Confirm the tested shape reaches
this core GEMM hook: decomposition MMA, LU and Cholesky have other launchers.

| ID | Modifiers | Tile M x N x K | Intake state |
| --- | --- | --- | --- |
| G1 | DIRECT | 64 x 64 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G2 | DIRECT, SMALL | 32 x 32 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G3 | DIRECT, WIDE | 64 x 128 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G4 | DIRECT, TALL | 128 x 64 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G5 | none | 64 x 64 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G6 | DEEP | 64 x 64 x 32 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G7 | PADDED | 64 x 64 x 16 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G8 | WIDE, DEEP | 64 x 128 x 32 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G9 | TALL, DEEP | 128 x 64 x 32 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |
| G10 | DEEP, PADDED | 64 x 64 x 32 | M2 A/B PASS9f1a1657c; resident screen fa390736 all66 records exact; standalone n1 HOLD retained; no broad default |

Overlap review: `DECOMP_FAST_MMA_K16@d487c814f` targets decomposition's
non-split launcher; `LU_FAST_MMA_DBUF@5d4e5d5d5` targets LU's subtract-update
kernel. These are distinct callers, not evidence that G1–G10 work or win.
Prefer initial standalone control/staging evidence before expensive caller
comparisons. Keep all ten ideas individually tracked rather than treating a
bundle's best result as validation of every variant.

## Kalman variants

These are isolated Metal shader sources with no production adapter yet.
The older frozen-gain scan remains rejected. Compare against the new exact
rank-one scan design (`b76e8fd38`) to share oracle coverage, not to substitute
one formulation's evidence for another. Preserve initializer, differencing,
likelihood convention, search/optimizer inputs and failure handling. The
source's serial likelihood reduction must be replaced before default
promotion. Forecast quality remains a separate acceptance gate.

| ID | Candidate | Intake state |
| --- | --- | --- |
| K1 | Full Gaussian associative prefix scan | Full Gaussian oracle HOLD: 6 math / 48 float32 / 14 gradient failures; fixed diagnostic controls completed; conditioning/model repair required before GPU admission |
| K2-B8 | Blocked Gaussian scan, block size 8 | Equation/oracle and adapter backlog |
| K2-B16 | Blocked Gaussian scan, block size 16 | Equation/oracle and adapter backlog |
| K2-B32 | Blocked Gaussian scan, block size 32 | Equation/oracle and adapter backlog |
| K3 | Scalar exact-observation specialization | Actual GPU102e0d70a HOLD49 gradient components; compensated reference9723a738 PASS13 groups on M3; compensated GPU implementation and full fit/forecast owed |

## Shared-call variants

Pinning alone does not establish a speed gain: the prior pinned-output
experiment lost after the caller's first read. Use ordinary caller-owned
output where intended, retain all buffers through completion, and validate
reuse, exception cleanup, empty/tail inputs and full output initialization.
These APIs need explicit FAST/Apple guards at any production integration.

| ID | Candidate | Intake state |
| --- | --- | --- |
| C1 | ResidentCallSlot: retained transfers and scratch | r5 lifecycle quality PASS3491a4d4c; A probe is no-op, not production timing baseline; caller integration/reach review required |
| C2 | wait_pair: one wait for two independent calls | r5 lifecycle quality PASS3491a4d4c; A probe is no-op, not production timing baseline; caller integration/reach review required |
| C3 | PackedReadback: retained grouped readback slab | r5 lifecycle quality PASS3491a4d4c; A probe is no-op, not production timing baseline; caller integration/reach review required |
| C4 | Resident MinMax transform adapter | r5 lifecycle quality PASS3491a4d4c; A probe is no-op, not production timing baseline; caller integration/reach review required |

The sibling IDENTICAL call-path worktree is not included in this FAST intake.
Manager owns queue edits and merges; delegated reviewers own isolated source
and harness preparation. Record new jobs and verdicts in `EXPERIMENTS.md`
when their exact compiled source and gates are ready.

## Historical first intake evidence (superseded by current table)

G1/G5 probe source `6abb76673038a3f7a3eb6ebc5be331472e32747e`, binding
`gemm_probe`, define `MOJOLEARN_APPLE_GEMM_PROBE`: M2 A/B rc0. This
standalone probe compares explicit incumbent, direct and staged kernels;
production dispatch is unchanged. Earlier reserved-identifier parse failures
produced no measurements. Twelve M3 oracle fixtures must pass before timing.

C1–C4 probe source `89e7d080b1bdb99fb84e956b5ba64371453857e2`, binding
`callpath_probe`, define `MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES`: M2
A/B rc0. M3 bit/lifecycle checks are still owed. Compilation is not evidence
of safe reuse, output identity, or end-to-end speed.

The separate rank-one Kalman reference oracle `45b61672d`, tag
`arima-assoc-oracle-v1`, is HOLD: float64 algebra passes all 68 fixtures,
but 47 float32 checks and 9 finite-difference gradient checks fail. These
are NumPy emulations, not actual-main kernel comparisons. Exact-rank
elimination changes the rounded-Q model; do not use this formulation as
validation for catalog K1/K2. Retaining full conditional covariance is the
next mathematical comparison. Scalar K3 remains independently gated.

## Shared reach priority

User steering: test improvements across every eligible caller and reduce
avoidable duplication. See [GEMM reach audit](GEMM_REACH_AUDIT_2026-10-04.md).
Next integration targets both unfused SDK entrances with one opt-in candidate
dispatcher and per-route reach counters. Matrix quality is followed by actual
estimator quality before scored caller A/B runs. Specialized LU/Cholesky/MCD
adapters must retain fused epilogues, batching, strides and reduction policy.
No claim of broad runtime reach follows from source imports alone.

Queued G1/G5 and C1–C4 quality jobs now use `tools/apple_fast_pinned_job.py`
from an already prepared queue branch. Their own exact source is checked out
privately and their M2 binary manifests/hashes are verified by the original
probe helper; no redundant native build or scored replay is involved.
K3 `6f09497ab` passed M2 A/B and is staged/queued for kernel quality only.

## Latest shared-path decisions

M3 catalog-matrix-t-v1 finished all66 predeclared matrix-only records with
exact incumbent output hashes. Cold context/transport dominates23–30ms;
no universal geometry wins and no estimator is admitted from this screen.
A distinct resident-input contract is under preparation; it still includes
completion and first read, with one scored call per arm/shape after quality.
Actual core/estimators counter-quality harness0505c9427 is source-only.
Shared dispatcher a9e64922f passed43 route fixtures, including preserved
GEMV/TN/K0 fallbacks. Decomposition and PCA bypass these SDK entrances;
their next adapters must preserve strides, split policy and atomic behavior.

## Manager execution checkpoint — 2026-10-04

All19 original ideas remain individually listed above; no side branch is deleted.
The 19-row catalog is an accounting of source ideas, not 19 runnable jobs.
K2 block8/16/32 remain explicit oracle/adapter backlog. C1/C3 mostly duplicate
already pooled VAR behavior; C4 must preserve mutable model attributes and finite
checks and needs a current-MinMax baseline. Do not score the no-op C probe.

Four actual PCA transform/inverse G1/G5 jobs completed on M3: transform G1
32.633750->29.887000ms and G5 32.698166->30.608875ms; inverse G1
34.177417->36.050709ms and G5 35.538000->36.727542ms. All output gates pass;
diagnostic counters enabled, one cold call plus first full copy, no board admission.
Tags are `g{1,5}-pca-tall-{transform,inverse}-t-v1`. Keep inverse incumbent.

Scoped actual AFN decomp/PCA adapter compiled28f06e1923 on M2 A/B, pinned
harness5776a5d1cb; M3 `scoped-r2-all-q-v1` PASS with no failures. Metadata
preflight covered28 fixtures; report contains38 output checks. Actual PCA.fit
quality/reach remains the next gate. Board RSVD rank18 intermediates fall outside
the current tall/narrow selector; do not queue an expensive NO_REACH timing.

`arima-k3-df-reference-v1` completed PASS13groups at9723a738, zero error allowance,
reference emulation only. Original actual-GPU K3 HOLD remains in force.

Recovered `MOJOLEARN_RESAMPLE_FAST_GATHER` source8605a3581 is now under M2
A/B compilation; its old taxi request failed parsing, with no timing. Source and
quality harness are consolidated opt-in into lane/apple-fast, not enabled or
merged into production main. Scoped GEMM source/harness are also consolidated
there. M3 legacy lane ref is deliberately not moved just for source consolidation;
new jobs use exact pinned source and verified artifacts.

SVGP_BSPLIT already exists opt-in on main; readiness reviewb1487e2e is retained,
not admitted. Most other unmatched request tags are neural work outside this
classical/tree effort. Remaining classical/tree dispositions are in
`notes/remaining-branch-audit-20261004.md`; GPU resample is the concrete recovery.


## Main integration checkpoint, remote manager 2026-10-04

User-authorized source consolidation preserves all candidate flags default OFF;
this does not promote unmeasured/held code or claim a new opponent win.
Integration descends GitHub main cb88add1c. M2 bare main may advance separately
from GitHub: GitHub write authentication is unavailable in the remote session.
Existing accepted defaults and rollback flags are preserved.

| Current source | Evidence and exact state | Inline outcome location |
|---|---|---|
| GPU resample compiled7eacaa2b2, harness0c7066aae | q-r2 PASS exact outputs/refusals/lifetime/native reach. Initial taxi/istella timing attempts failed NumPy header parsing before scoring; repaired public metadata API, no numerical loss and no speed claim. New timing jobs use matched board warmup+one scored round. | resample/estimator.mojo RESAMPLE_GPU_GATHER |
| Scoped decomp/PCA201fe736 and harness4c69e4387 | scoped mechanism PASS, actual PCA.fit HOLD on no-regression singular/noise gates; report recovery remains separate. No PCA default/board promotion. | scoped_dispatch.mojo and PCA/decomp caller gates |
| Private compensated K3 GPU fe5df7ab0 | arima-k3-df-gpu-q-v1 PASS13groups/4controls; kernel-only. No production AutoARIMA integration or estimator timing. Original scalarK3 HOLD49 worse gradient components remains. | arima/impl/fast_scalar_df.mojo and gated binding export |
| MCD G1 candidate35c712d9 with comments5d5a06c6 | SOURCE-READY/UNBUILT. Actual batched covariance/fitted-state/support/rank quality and timing owed; no import of held ordered covariance/PCA atomic route. | x_decomp/mcd_bmma.mojo |
| Softmax G2 candidate4bfc1424 with comments574b63b8 | SOURCE-READY/UNBUILT. Matrix G2 lead does not establish optimizer quality/speed. Actual-caller quality and timing owed. | experiments/apple_fast/gemm/softmax_narrow.mojo |
| Existing MBK_LABRG, CholeskyNB512 | MBK qualityPASS but negligible/mixed speed; NB512260.4->265.7ms. Stay opt-in. | x_cluster/minibatch_fast.mojo; cholesky/checks/potrf.mojo |
| Existing SVGP_BSPLIT / PREP2_EIGH_BLOCK | OPEN with no judged timing admission; historical queued/readiness status is not a result. | x_neighbors/iter_device.mojo; x_prep/fastprep2.mojo |

No deleted rejected implementations were recreated just to add comments:
SHAP_PIPE, PINNED_OUT, host ROW_GATHER, VAR_FUSED, LU_TSLU/LU_DBUF, older K1,
ordered MCD and abandoned sparse/RBF variants remain in their recorded branch
history. Existing accepted SHAP/LU/PCA/label/target/RBF defaults retain their
prior measured evidence and rollback switches. Source comments distinguish
shared G1/G5 transform wins from inverse losses and scoped PCA-fit HOLD.

Source-only validation: merge conflicts reconciled additively; unchanged board
and quality thresholds; Python AST, conflict-marker/whitespace checks and
no-host-routes hook. No native build or GPU job was launched by integration.
New softmax/MCD compilation and combined-source native builds remain owed to
the manager; source integration is not binary qualification. No board cells or
351/377 headline are changed by this integration.


## Completed matched timing, 2026-10-04

Source7eacaa2b2, harness0c7066aae; r2-20261004 tags. One unscored warmup
and one scored call per arm, existing board worker including full output reads.
Both receipts PASS exact output summaries/digests. Taxi A68.48829198861495ms
B58.8205840322189ms; istella A390.82008303375915ms B805.391583009623ms.
Mixed result: wide-row regression blocks broad default. Toggle remains OFF;
no board promotion or dataset-specific dispatch added.

Receipt SHA256 taxi a8077db5a76c48ed431a7736597fe241d301aa1f01aa20f255d10c67431aeaea
Receipt SHA256 istella 28e20bec22f34b67c1b8123f1d71ead38a4e402946777554f09fa412eadfd7c7
Paths: ~/mq/out/resample-gpu-recovered-t-{taxi,istella}-r2-20261004-timing/PASS.json
