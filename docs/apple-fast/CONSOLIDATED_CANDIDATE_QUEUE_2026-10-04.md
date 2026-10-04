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
| G1 | DIRECT | 64 x 64 x 16 | M2 A/B PASS at 6abb76673; M3 quality next |
| G2 | DIRECT, SMALL | 32 x 32 x 16 | Review/compile backlog |
| G3 | DIRECT, WIDE | 64 x 128 x 16 | Review/compile backlog |
| G4 | DIRECT, TALL | 128 x 64 x 16 | Review/compile backlog |
| G5 | none | 64 x 64 x 16 | M2 A/B PASS at 6abb76673; M3 quality next |
| G6 | DEEP | 64 x 64 x 32 | Review/compile backlog |
| G7 | PADDED | 64 x 64 x 16 | Review/compile backlog |
| G8 | WIDE, DEEP | 64 x 128 x 32 | Review/compile backlog |
| G9 | TALL, DEEP | 128 x 64 x 32 | Review/compile backlog |
| G10 | DEEP, PADDED | 64 x 64 x 32 | Review/compile backlog |

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
| K1 | Full Gaussian associative prefix scan | Equation/oracle and adapter backlog |
| K2-B8 | Blocked Gaussian scan, block size 8 | Equation/oracle and adapter backlog |
| K2-B16 | Blocked Gaussian scan, block size 16 | Equation/oracle and adapter backlog |
| K2-B32 | Blocked Gaussian scan, block size 32 | Equation/oracle and adapter backlog |
| K3 | Scalar exact-observation specialization | Exact scalar Mojo adapter; M2 signature repair pending; quality/forecast gates owed |

## Shared-call variants

Pinning alone does not establish a speed gain: the prior pinned-output
experiment lost after the caller's first read. Use ordinary caller-owned
output where intended, retain all buffers through completion, and validate
reuse, exception cleanup, empty/tail inputs and full output initialization.
These APIs need explicit FAST/Apple guards at any production integration.

| ID | Candidate | Intake state |
| --- | --- | --- |
| C1 | ResidentCallSlot: retained transfers and scratch | M2 probe A/B PASS at 89e7d080b; M3 quality next |
| C2 | wait_pair: one wait for two independent calls | M2 probe A/B PASS at 89e7d080b; M3 quality next |
| C3 | PackedReadback: retained grouped readback slab | M2 probe A/B PASS at 89e7d080b; M3 quality next |
| C4 | Resident MinMax transform adapter | M2 probe A/B PASS at 89e7d080b; M3 quality next |

The sibling IDENTICAL call-path worktree is not included in this FAST intake.
Manager owns queue edits and merges; delegated reviewers own isolated source
and harness preparation. Record new jobs and verdicts in `EXPERIMENTS.md`
when their exact compiled source and gates are ready.

## First intake evidence

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
