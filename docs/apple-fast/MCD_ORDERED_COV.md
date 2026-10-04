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

The specialized harness is implemented in the quality follow-up branch, but
M2 compilation and M3 execution remain owed. Generic permissive MCD PASS
receipts do not satisfy this contract.


## Specialized quality harness (source only)

`tools/mcd_ordered_pair.py SOURCE TAG CASE [DIRECT_PASS_TAG]` verifies exact
source, binary hashes, defines and FAST mode, then executes isolated workers.
The worker verifies the binding's Metal vendor, opt-in flag and runtime raw/
final route counters (including split invocations). No clocks are read and no
fit times are recorded. A is captured once; B twice strictly as an unscored
reproducibility check, with intervening different-sized resident allocations.

Run `direct` first. It covers n/d pairs (1023,11), (1024,65), (1025,11),
(3000,220), (5001,65), (100000,11), each with three candidates and changing
active masks. The raw probe poisons partials and inactive outputs, and calls
the actual batched baseline or ordered helper. The final probe calls the real
resident entry with pool reuse. B reverses fixture order for its second pass.
Every direct output must be reproducible and no worse by both float64 error
metrics; inactive outputs must preserve their sentinel exactly.

Subsequent cases require that hash-bound direct PASS receipt:
`mcd-istella`, `ee-istella` (3000 rows), `mcd-taxi`, `ee-taxi` (100000 rows),
`mcd-synthetic` (1100x17 rank deficient). Each tests the actual estimator fit,
state, query predictions and repeated B determinism. Runtime counters require
both raw and final routes; non-synthetic cases also require raw split work.
`tools/mcd_ordered_oracle.py` enforces the predeclared exact masks and strict
numeric no-worse criteria, including raw mean/covariance. Capture metadata
includes actual loaded binary hash, source, fixture input hashes, arm, reach
and unscored execution policy. Reports are never overwritten.

No timing action exists in this helper. A passing individual receipt is not
permission to time while other required quality cases remain unresolved.
The algorithm could later serve other long-K Gram/covariance consumers, but
this branch changes only MCD's two covariance seams. Generalizing it would
need caller-specific quality, memory and timing evidence; no other algorithm
is silently opted in.

### Compile repair r1

The first M2 arm A build of a47cdb339 failed before execution: `out` was used as a parameter name, and inferred immutable `DeviceBuffer.unsafe_ptr()` results did not match the existing mutable-pointer launcher API. The r1 source renames the parameter to `dst` and explicitly borrows each locally owned mutable buffer through `unsafe_ptr[True]()`, retaining a single input pointer for both X operands. No const cast or address reconstruction is introduced. Buffer ownership stays local through synchronization. This matches the [DeviceBuffer pointer API](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceBuffer/). Arithmetic, fixtures, strict gates and scoring policy are unchanged. New M2 compilation and M3 quality remain required; the compile failure is not numerical evidence.

### Installed-SDK compile repair r2

M2 rejected r1's `unsafe_ptr[True]`: its installed `max/mojo/max/gpu/host/device_context.mojo:1858` declares the explicit parameter as `origin: MutOrigin`, not `mut: Bool`. The r1 linked web API belongs to a different SDK revision and does not establish compatibility here. r2 requests `unsafe_ptr[MutAnyOrigin]()` directly from each locally owned mutable buffer, matching the installed compiler diagnostic and the existing `F32Ptr` / `I32Ptr` aliases. This selects the mutable-origin overload; it is not a const cast. Owners remain alive through synchronization. No fixture or threshold changes. Compilation remains owed.

### Non-null gate repair r3

r2 compiled arm A, but arm B instantiated `dev_mcd_cov_py` and rejected its fabricated null `I32Ptr`. r3 lazily allocates one real int32 gate word on `xd_ctx`, initializes it to 1 on the same stream, and retains ownership in a dedicated global buffer holder. The ordered covariance call borrows that allocation with the installed SDK's mutable-origin API. The global owner survives the asynchronous launch through subsequent download/synchronization; there is no per-call gate free or extra synchronize. `gate_all=True` still preserves the original dispatch behavior, but the gate address is now valid even though the kernel does not read it in this mode. No arithmetic or quality-gate changes. M2 arm B compilation and M3 quality remain owed.

### Split-chain repair r4 (lane apple-fast-rec-misc, 2026-10-04)

r3 compiled and ran: `mcd-ordered-direct-q-r3` returned HOLD with `repeat_identical` true. B kept main's split
count (`DFG_BLOCK_TARGET` / `DFG_MIN_SPLIT_STEPS`: 1, 2, 2, 5, 9 and 184 splits at the six direct shapes). With 2
partials the Neumaier fold equals the plain sum; with 5..9 the fp32 accumulation inside each split's MMA chain
(about 550..600 rows) is larger than the fold's rounding, so B's error differed from A's by noise and the per-key
max-abs comparison, with no tolerance, could go either way. r4 cuts every split to `MCD_ORD_CHAIN` = 4 K windows
(128 rows at `AFN_GEMM_KB` 32), never fewer splits than main, with the partial region capped at the kernels' Int32
bound / 8 (`MCD_ORD_MAX_WORDS`). The fold, the gates and the harness are unchanged. Quality is still owed, direct
first.
