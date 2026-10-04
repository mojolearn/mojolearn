# Apple FAST experiment and toggle process

Scope: classical ML and trees in FAST mode on Apple. This is the working
process for new lanes and manager reviews. Historical outcomes live in
[EXPERIMENTS.md](EXPERIMENTS.md); accepted measurements live in
[BOARD_M3_FAST.md](BOARD_M3_FAST.md). A passing build is not a passing experiment.

## Before implementation

1. Search the experiment ledger and nearby toggle comments by algorithm and
   define. Read previous failures before repeating an idea. Record what changed
   that makes a retry useful; use a new tag, preserving the previous result.
2. Branch from current main in a private worktree. Record the base SHA, hypothesis,
   affected datasets, define, and relevant quality checks. Check all places using
   a bundled define; splitting a bundle requires independent evidence.
3. Keep candidates opt-in and scoped to FAST + Apple. Preserve IDENTICAL and
   other vendor behavior. Do not merge an unvalidated classical/tree candidate
   merely because it is behind a toggle.

## Establish trustworthy evidence

- Arm A is current main behavior; B is the candidate. Record exact source SHAs,
  datasets/fixtures, mode, vendor, defines, compiler environment and binary hashes.
  Validate the M3 checkout against the built arms: a pushed GitHub branch does
  not imply that the cloud mirror has that revision.
- Measurement isolation includes manager activity: do not run disk scans, cleanup,
  artifact compression, dependency installation, bulk transfers or other heavy
  maintenance alongside scored M3 work. Heavy maintenance must be its own job
  in the same serial queue, or the runner must first be paused between jobs and
  all of its workers confirmed exited. Keep it paused until maintenance finishes.
  Do not start heavy work through separate SSH while the queue is running, even
  if it currently appears to be building or between arms. Subagents follow the
  same rule. Lightweight, bounded progress checks are allowed.
- If isolation is uncertain, retain raw results but mark the affected comparison
  inconclusive. One sample per arm does not estimate variance or establish a
  small gain/regression; do not call the noise range known without evidence.
  Any additional measurements must respect the current run budget and replay
  policy; do not silently rerun pairs until a preferred result appears.
- M3 is the timing authority; M2 may compile. Serialize scored M3 work. Under the
  current run budget, use one scored run per arm; do not replay completed arms,
  rerace opponents or rerun the whole board to improve a result. If evidence is
  too noisy, leave the verdict inconclusive rather than claiming a win.
- Old-base gains are provisional. Rebase/merge current main and validate again,
  unless a recorded source review establishes that intervening changes cannot
  affect the measured path. A stale gain already present in main is not a win.
- Run meaningful quality checks before expensive timings when possible. Test
  actual outputs, predictions and fitted state as appropriate: AUC/logloss,
  forecast error, recall, residuals, reconstruction, support masks or parameters.
  Shape, finiteness and successful compilation alone are insufficient.
- Changed bits are allowed in FAST. Quality must not worsen beyond demonstrated
  noise. Record the metric, tolerance and evidence for that tolerance before
  accepting a difference; do not loosen a threshold after a failure to pass it.
  Changed anomaly behavior is not excused solely by a better optimizer objective.
- Compare candidate quality to main and keep opponent quality visible separately.
  An optimization may preserve main exactly while an existing opponent-quality
  deficit remains. Such a row stays held from the quality-qualified headline.
- A timeout, missing output, source mismatch, build failure or harness failure is
  not a valid measurement. Repair infrastructure separately and preserve logs.
  Retry only the work that did not produce valid evidence, using explicit tags.

## Efficient execution for new submissions

These rules apply to future submissions. Do not restart, rewrite or interrupt
claimed jobs to adopt them; preserve existing source pins, output directories
and evidence. The manager remains the sole queue writer.

1. **Prepare complete artifacts on M2.** Compile only the affected bindings and
   explicit auxiliary dependencies. Record mode, defines, full source and binary
   hashes for every artifact. FAST estimators can depend on an IDENTICAL base
   binding for existing API validation; FAST core alone does not supply that
   dependency. Check the actual API and required exports before submission.
2. **Use verified binaries on M3.** New jobs use an allowlisted pinned-source
   helper from an already prepared queue branch. No automatic native build,
   dependency installation or build fallback on M3. A fresh isolated source tree
   is acceptable; the helper installs/restores the verified artifacts. Do not
   modify the running legacy runner to retrofit this rule into claimed jobs.
3. **Preflight before queue insertion.** Verify the source exists in the M3
   mirror, all required artifacts are staged and hash-correct, helper arguments
   match their source contract, and the tag has no prior queue/result/output.
   Missing prerequisites are infrastructure errors, never numerical failures.
   A preflight receipt is readiness evidence only: quality and timing helpers
   must revalidate source and artifacts when executing.
4. **Reuse unchanged builds safely.** A tools-only harness repair may reuse an
   older compiled source only after ancestor checks, an explicit changed-file
   allowlist, zero native/production/config drift and exact manifest/hash checks.
   Record both harness and compiled sources. Use a fresh tag for the repaired
   job; never replay successful scored arms.
5. **Prove useful reach before scaling out.** Use actual caller counters and
   independent output-quality checks. NO_REACH is a control result, not admission
   for timing. Start with representative reachable operations; expand to each
   affected route and precision/layout contract before any broad default.
6. **Separate measurement contracts.** Host-input and resident-input routes have
   different setup costs. Declare boundaries and shapes before results; retain
   completion and first output read. Do not infer kernel throughput from a
   context/transfer-dominated total or create a new label just to repeat a score.
7. **Keep compilation ahead of measurement.** Maintain source-ready,
   build-verified, quality-passed and timing-admitted states. Compile the next
   reviewed candidate on M2 while M3 works serially. Bound build concurrency by
   memory/disk; remove only completed owned build trees, keeping source refs,
   binary manifests and evidence. M3 cleanup/transfers remain serial queue jobs.
8. **Batch readiness work, not timing workers.** Stage several completed pairs
   in one transfer window and release it promptly. Reuse a prepared runner
   branch and fetch bounded result summaries. Do not repeatedly copy full logs
   or rebuild every family for a one-binding quality check.

A passing miniature fixture does not certify all board shapes. Preserve broad
coverage where needed; efficiency comes from removing redundant setup and
uninformative experiments, not from shrinking quality requirements.

## Decide and maintain the toggle

| State | Runtime default | Required action |
|---|---|---|
| Candidate / open | Off, on experiment branch | Record hypothesis and owed checks. |
| Hold / inconclusive | Off | Record the exact uncertainty or failed gate and next step. |
| Accepted | On for validated FAST + Apple scope | Keep a named `_OFF` rollback switch and evidence comment. |
| Rejected / abandoned | Off until reviewed cleanup | Preserve branch + commit + evidence; remove dead implementation from main in a separate reviewed cleanup. |

A passing experiment needs both a useful speed improvement and acceptable
quality, with no unsupported extrapolation to unmeasured datasets or bundles.
A deliberate quality improvement with a speed cost is a separate decision;
do not count it as a speed win.

For accepted changes, validate both default and `_OFF` builds, review every use
of the gate, and make the off switch restore the prior implementation. Merge
with `--no-ff`, never force-push, and merge main into `lane/apple-fast`. Retain
rollback switches while useful for diagnosis; removing one is a separate review
with an explicit replacement or reason, not automatic cleanup.

For held or failed code that remains, put a short comment next to the toggle.
Comments should explain **why it is off**, not merely say “experimental.” Do not
restore deleted code solely to annotate it. Branches and recorded commits retain
failed experiments without requiring every abandoned path to remain on main.

```text
HOLD-quality, YYYY-MM-DD, <tag>, <dataset>, source <sha>:
A <time> -> B <time>; <quality metric A -> B>, required <tolerance>.
Disabled because <specific failure/uncertainty>. Next: <repair/check>.
See docs/apple-fast/EXPERIMENTS.md (<define>).
```

For a winner, use the same evidence fields with `KEEP`, the validated scope and
rollback define. A bundled failure does not establish that every component fails
alone. Distinguish quality regression, slower, noise, stale base and harness error.

## Close the loop

1. Update the ledger with define(s), branch/base/measured/merged SHAs, tags,
   datasets, A/B times, quality metrics, verdict, evidence location and next step.
   Keep historical attempts rather than overwriting failed results with retries.
2. Update only measured, accepted table cells from raw successful arm results.
   Preserve all quality holds. Record whether a headline change is a new winner,
   a newly excluded row or a changed comparison set.
3. Rebuild canonical evidence and the local page. State hosted publication
   separately; a rebuilt HTML file is not a published page.
4. Keep the handoff current. The manager owns queue edits, cloud synchronization
   and main merges; agents use private worktrees and bounded `rg`/`grep` reports.
   The M3 watchdog checks every 20 minutes for progress and low disk space.
   It must not blindly kill live workers or replay claimed scored jobs.

This process is the target standard, not a claim that every historical toggle
already has complete comments or evidence. Fill gaps during review; do not label
untested legacy code as validated.

## Future submission preflight

Before submitting a new pinned job, use the read-only metadata gate described
in [JOB_PREFLIGHT.md](JOB_PREFLIGHT.md). Declare exact source, script/arguments,
all pair/single artifacts and each case's prerequisites. READY is a snapshot,
not a queue reservation or native-symbol/quality/reach certification. The
case helper retains the final runtime and numerical gates. Apply this process
to future submissions; do not rewrite or replay in-flight jobs.
