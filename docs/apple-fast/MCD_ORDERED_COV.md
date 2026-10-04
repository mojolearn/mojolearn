# MCD ordered covariance candidate

Status: source only, uncompiled and unmeasured. Base main `8b3079d02`.
Binding: `x_decomp`. Opt-in: `MOJOLEARN_MCD_ORDERED_COV` (FAST + Apple).
No SKIP_PINVH, DEFLATE, convergence tolerance, rank cutoff, or eigensolver change.

## Hypothesis and source scope

Saved MCD A/A captures establish actual covariance variability. The unchanged
split-K covariance kernels add FP32 partials using atomics; a different add
order perturbs the covariance and can amplify precision/score differences.
Two saved baselines do not establish an acceptable noise bound. SKIP's oracle
HOLD remains; this is a separate accuracy/reproducibility experiment.

`mcd_bmma_kernel[True, True]` keeps the current MMA tiles, window partition and
operands, but writes each candidate/split to its own region. A second grid
assigns one independent GPU thread to each output cell and folds split indices
in ascending order with Neumaier compensation. Addition is expressed through
`llvm.fma.f32(1,a,b)` without fast-math flags, preventing cancellation of the
compensation by reassociation. At most 640 splits feed any cell. This is not a
single global reduction or one-block cooperative algorithm; the grid scales
with candidate count and output cells. Small matrices naturally occupy fewer
blocks. No host numerical computation or host scratch reduction is added.

Only these covariance call sites change:

- Raw MCD C-step `_mma_covariance`: phase-owned `part` workspace is enlarged
  when necessary and reused AFTER its column-moment consumers have completed
  on the same in-order stream. No allocation is added when the define is off.
- Final MCD `_masked_cov` (including assume-centered): an optional resident
  `x_decomp_dev_mcd_cov` entry launches the same ordered Gram. The export exists
  only for the opt-in FAST Apple build. Its temporary uses the existing
  same-context resident pool and queued reuse rules. Small/nonresident inputs
  retain the current path.

Other generic GEMMs, precision construction and distance kernels are unchanged.
An unsplit covariance uses the original kernel exactly. Inactive candidates
are skipped in BOTH partial production and folding; their uninitialized
partials are never consumed. Existing gated publication is preserved.

Workspace: `candidate_count * splits * d * d` float32 values; the size is
checked against Int32 addressing. For d=220, 40 splits, ten final candidates,
the partial region is approximately 77.4 MB (decimal), reused within a phase.
The resident single-model final covariance needs about 7.74 MB at that shape.
No memory is exposed to Python; output ownership and first-read semantics stay
ordinary. Extra launch/bandwidth may cost time. This candidate does not repair
MMA partial-sum rounding, covariance centering errors, or eigensolver error.
Compiler acceptance and actual preservation of compensated arithmetic remain
unverified until M2 build and M3 quality checks.

## Required quality plan before any timing

Pre-register a NEW fixture/receipt, `mcd-ordered-cov-v1`; existing permissive
`mcd_compat_quality.py` PASS is insufficient. No scored timing or opponent fit
is authorized by this source branch. Manager owns all machine work.

1. M2 builds current-source A without define and B with the define. Record
   compiler/flags, full source SHA, and binary hashes. Verify B's optional
   resident entry exists and A's does not. Both must report FAST and Apple.
2. M3 quality-only Gram fixtures compare A/B to the SAME float64 reference:
   deterministic finite inputs, mixed signs/cancellation, different column
   scales, exact-zero columns, rank deficiency, tails and pool reuse. Cover
   n=1023/1024/1025/3000 and d=11/65/220, plus a long n with a small d. Exercise
   batched active/inactive candidates and poison unused partial storage.
   Require every B relative-L2 AND max-absolute covariance error <= A's error,
   without a tolerance. Nonfinite values or shape discrepancies fail.
3. In that quality job, repeat identical B inputs after unrelated-sized work
   and require all covariance output words identical. This is a reproducibility
   check, not repeated scored measurements. Retain separate captures, including
   scratch reuse/alternating active masks. Do not declare determinism from code
   structure alone.
4. Full fitted-state fixtures: MCD and EE istella cap3000, MCD and EE taxi,
   and a small synthetic rank-deficient case. Preserve input hash/source/arm
   and execution-policy metadata. Require A/B raw support, final support and
   flags EXACTLY identical. Changed masks are HOLD, irrespective of objective.
5. With a common saved raw support, reconstruct float64 raw mean/covariance
   directly from original X. With a common final support, reconstruct corrected
   float64 final mean/covariance, precision at full-d float32-epsilon cutoff,
   distances, and EE offset/decision function. Require B <= A separately for
   relative-L2 AND max-absolute error in EVERY numeric field, raw and final.
   No error averaging, relaxed threshold, inferred noise allowance or new
   opponent fit. Numeric raw covariance may change intentionally; unlike the
   old SKIP-only exact-raw gate, it is judged against this predeclared oracle.
6. Repeat B fitted-state fixtures as quality-only captures and require identical
   raw/final arrays and decisions. Any remaining variation is HOLD and requires
   locating another unordered path before timing. This candidate stabilizes
   covariance only and does not claim the entire estimator is already stable.
7. Only after those gates, review current-main source drift and request a new
   single scored A/B comparison. Evaluate this candidate independently. SKIP
   must later be tested separately against an accepted stable baseline; never
   attribute combined speed gains to ordered reduction alone.

The strict ordered-cov oracle/reproducibility harness is still RUN OWED /
IMPLEMENTATION OWED. This document is the fixed validation contract, not a
claim that the current generic quality script enforces it.
