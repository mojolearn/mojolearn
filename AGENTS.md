# Agent working rules

## Logs and context

- Save complete build, test, and GPU logs to files. Inspect the exit status and
  structured summary first; do not stream or paste entire logs into context.
- Prefer targeted `rg` searches (or `grep` when unavailable), bounded context,
  and short tails. For example:
  `rg -n -i -m 20 -C 3 --max-columns 240 --max-columns-preview 'error|fail|refused|divergent|summary' run.log`
  or `tail -n 20 run.log`. Narrow to relevant files before searching directories.
- Parse large JSON reports for selected fields, counts, and failures rather
  than printing whole documents. Keep full evidence on disk and report its path.
- Expand the relevant excerpt when needed to establish the cause. A filtered
  excerpt or a few passing lines never proves the whole run passed: preserve
  exit codes, expected coverage, failures, and incomplete or skipped work.
- Never read whole subagent transcripts or `tasks/*.output` files. Request a
  concise status or inspect the specific retained artifact instead.

## Lane and subagent prompts

Include this reminder in every lane brief, spawn prompt, and resumed assignment,
and ask delegated agents to propagate it to their own subagents:

> Keep logs out of context: save complete output to files, use targeted rg/grep
> with bounded surrounding lines and short tails, and summarize exit status,
> coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
> never hide failures or infer full success from filtered output.

Final handoffs should give the branch/commit, changes, validation status,
remaining work, and evidence paths without pasting full logs or diffs.

## Wait for Mojo and Modular

If Mojo or Modular does not support something yet, do not build a workaround for it; wait for their support.
Examples: an AMD portable kernel path needs Mojo to keep kernel IR (or bitcode) and to accept generic gfx targets;
until Modular ships that, we do not pursue it (decided 2026-10-04; the parked prototype is on `lane/amd-portable`,
with the evidence in `docs/AMD_PORTABLE_PATH.md` on that branch). Do not hand-rewrite compiler output, patch
toolchain internals, or ship unsupported build modes. Record the ask for Modular instead and move on.

## No dimension targeting, in any mode

No dispatch, tile, threshold, cap or route rule may key on an exact benchmark dimension, a size chosen to sit just above or
below a board row (rows, features, classes, k, vocabulary), or a board dataset name. This applies to FAST and to IDENTICAL on
every vendor. A rule must come from size, hardware or cost reasoning that covers neighboring shapes, stated in a comment.
In IDENTICAL, removing such a rule may change bits: that is allowed, because bits only have to match across NVIDIA and
AMD within one version, never across versions (Andrew, 2026-10-07: identity across the two GPU vendors only). Change all columns together. Each removal gets an
A/B with the old rule as the B arm, timed on neighboring shapes and one non-board dataset.

## Measurement process (owner, 2026-10-05)

### Full-dataset candidate evaluation (owner clarification, 2026-10-06)

- ALWAYS evaluate performance candidates with end-to-end A/B measurements on the full dataset for each affected
  estimator. This applies to AMD, NVIDIA default, NVIDIA architecture-specific, and applicable Apple FAST candidates.
  Component timings, synthetic substitutes, tiny public-caller fixtures, and opponent-only runs do not complete this work.
- Before timing, map each candidate to its affected estimators and saved full-workload recipes: dataset/version/hash,
  actual rows/features/classes or sequence dimensions, estimator settings, numeric mode, A/B toggles, and timed boundary.
  See `experiments/performance_ideas/README.md` for the recipe locations. Audit intrinsic lane caps: `--rows full` alone
  does not prove the full intended dataset was measured. Missing or ambiguous coverage stays pending; do not silently
  substitute a smaller case and report completion.
- Include preparation, fit/training, required synchronization, and consumed outputs in the declared whole-operation
  boundary; report fit, inference, cold and repeated use separately where applicable. Measure relevant toggle combinations
  and the proposed complete default configuration: independent component wins can interact and regress a full workload.
- Do not promote a default from component or reduced-workload gains. Require the full-dataset evidence above and the
  existing vendor/quality acceptance rules. Record winners AND losers beside switches and in the boards, with scope,
  sample counts and pending coverage. Say "component screening complete" when only screening is complete, never
  "measurement campaign complete" or "full board complete".
- Reuse accepted compilation and identity evidence. Do not repeat compilation/identity validation merely to evaluate
  performance; repair only actual measurement failures, preserving completed cells and original failure evidence.


1. Freeze one commit per A/B round. Compile it once on cheap boxes (Apple on the M2; NVIDIA/AMD on cheap fast-CPU boxes).
   Only a green frozen build goes to the timing GPUs. New code waits for the next freeze.
2. IDENTICAL switches are decided by NVIDIA and AMD together: combined faster, and neither vendor materially slower.
   Apple never votes on IDENTICAL switches and IDENTICAL is never tuned for Apple; Apple must only match bits.
3. Measure IDENTICAL on NVIDIA, AMD and Apple and update every board (main board included) as results land, through the
   board tools only. A full-board run is IDENTICAL on the three; FAST is not rerun.
4. Standing order: when a problem is found, fix it. Do not just comment on it or defer it.
5. Read logs with grep and short tails; never paste whole logs. Tell every subagent the same.

## Incremental experiment delivery (owner, 2026-10-05)

- Keep experiment code, the selectable runner, decisions and retained result summaries in `main`.
  Experimental code may merge after existing compile evidence on one device, with untested targets and measurements
  explicitly marked pending and unproven defaults disabled. Do not rebuild merely to merge already compiled code.
  A compile pass on one target is not a claim that every target works.
- Run selected A/B candidates through the actual full-size measurement harness in IDENTICAL mode on NVIDIA and AMD,
  using one excluded warmup and one scored sample. Reuse verified frozen binaries when numerical source, flags,
  compiler and target match. Run the vendors in parallel on separate owned boxes and cells serially within each GPU.
  Capture correctness/identity evidence from the scored execution. Apple identity matters; Apple timing does not vote.
- As each cell lands, retain its source, binary, hardware and harness provenance, timing, correctness and failure status.
  Update the experiment board and applicable main/vendor boards through board tools immediately. Pending, unsupported,
  failed or unqualified cells must remain visibly separate from admitted measurements; preserve the original evidence.
  Do not invent opponent ratios from own-only A/B runs or overwrite incomparable historical opponent measurements.
- Evaluate each candidate as soon as its required full-dataset, end-to-end baseline/candidate measurements and accepted
  quality evidence are available on BOTH NVIDIA and AMD for all affected estimator workloads and relevant combinations.
  Do not wait for unrelated candidates or the entire
  campaign to finish. Apply the existing rule: combined improvement, neither vendor materially slower, required identity
  and quality preserved. A single partial cell, compile pass or synthetic-only gain is not enough to flip a default.
- When that evidence supports a change, flip the applicable switch promptly and commit/push it to `main`. Put the reason
  INSIDE the source file beside the switch: experiment/source IDs, cases, timing ratios for both vendors, identity/quality
  outcome, sample count and limitations. Record failed, neutral and rejected candidates there too, including why they stay
  off. Also commit the corresponding machine-readable results and board-tool updates. Do this as decisions become ready.
- A promoted source change starts a new freeze; never mutate binaries or settings inside an active frozen run. Repair and
  rerun only affected cells, preserving valid completed evidence. Do not repeat already decided experiments unchanged.

## Full-machine CPU race resources (owner, 2026-10-05)

- Every CPU race arm gets the full machine allocation: opponents and our CPU arms wherever measured.
  No inherited laptop/test caps or `MOJOLEARN_BENCH_THREADS=1` overrides in a race. Use all-core
  estimator settings where supported and run measurement arms serially; do not introduce competing jobs.
- On dedicated Apple machines, leave CPU libraries unrestricted. On rented Linux machines respect the
  actual cgroup allocation, not the host's larger visible CPU count. Do not claim all cores are busy:
  serial algorithms, library defaults and nested parallelism may legitimately use fewer threads.
- Preserve algorithm semantics (including seeded UMAP) and documented nested-pool settings (implicit's
  BLAS=1 inside its parallel solver). Do not remove correctness settings just to raise utilization.
- Record worker resource policy, thread environment and effective pools with measurement evidence.
  A controller's environment alone does not prove an old worker's thread count. Keep unproven historical
  resource claims explicit; do not label every previous CPU result as one-core without worker evidence.
- These resource rules do not expand the selected race arms or alter another session's active frozen run.

## Measurement artifact retention before teardown

Before terminating a measurement machine, prove where every required output,
model-state artifact, accepted binary and unique input is stored. An attached EBS
volume with DeleteOnTermination=false does not prove that a workspace is on it.
Map the actual filesystem through its physical device to the provider volume ID;
directory names and mount labels are not evidence. For instance-local storage,
verify the retained bytes off the machine before termination. Metadata and hashes
alone are not retained array/model/binary bytes. Keep any incomplete retention
explicit and stop teardown until the required evidence is durably preserved.

## No Python in the runtime

Owner clarification (2026-10-06, supersedes earlier Python API/glue exceptions):
Python is ONLY for PyPI packaging/distribution and testing. NEVER USE PYTHON IN
PRODUCT EXECUTION. This includes public API wrappers, argument/data preparation,
binding selection/orchestration, fit, transform, predict, score, training and
consumed-output processing. Product execution belongs in compiled Mojo/native
bindings; no Python loops, callbacks, worker threads or NumPy/SciPy/scikit-learn
computation, conversions or fallbacks may execute on a product path, in any mode
or on any vendor. Preserve supported interfaces and semantics through native
implementation; do not remove features to claim compliance.
Product orchestration is runtime work: batch/epoch loops, kernel launch sequences,
buffer/workspace management, synchronization and stopping decisions must also run
in Mojo. Moving arithmetic to Mojo while Python still drives these operations is
only a partial repair, not compliance. Measure performance gains; do not assume
that removing Python makes every full workload faster.

Python test/benchmark controllers, test-data preparation and independent test
oracles remain allowed as testing tools. Keep their work explicitly distinguished
from product execution and accurately included in or excluded from declared timing
boundaries. Do not move product work outside a timer or relabel it as testing to
hide a violation. Installing a missing Python package is not a runtime repair.

Audit transitive callees and the actual whole-operation boundary; a helper's old
"outside the clock" comment is not evidence. Existing checker baseline entries
are debt to remove, never exemptions, and an empty baseline is not proof that
all product paths are native. Repair in separate source freezes, preserving
original measurement/failure evidence and valid completed cells. If supported
Mojo/Modular functionality is missing, record the blocker and wait for support;
do not invent a Python fallback or unsupported toolchain workaround.
