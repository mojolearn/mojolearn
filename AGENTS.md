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
In IDENTICAL, removing such a rule may change bits: that is allowed, because bits only have to match across NVIDIA, AMD,
Apple and the host column within one version, never across versions. Change all columns together. Each removal gets an
A/B with the old rule as the B arm, timed on neighboring shapes and one non-board dataset.

## Measurement process (owner, 2026-10-05)

1. Freeze one commit per A/B round. Compile it once on cheap boxes (Apple on the M2; NVIDIA/AMD on cheap fast-CPU boxes).
   Only a green frozen build goes to the timing GPUs. New code waits for the next freeze.
2. IDENTICAL switches are decided by NVIDIA and AMD together: combined faster, and neither vendor materially slower.
   Apple never votes on IDENTICAL switches and IDENTICAL is never tuned for Apple; Apple must only match bits.
3. Measure IDENTICAL on NVIDIA, AMD and Apple and update every board (main board included) as results land, through the
   board tools only. A full-board run is IDENTICAL on the three; FAST is not rerun.
4. Standing order: when a problem is found, fix it. Do not just comment on it or defer it.
5. Read logs with grep and short tails; never paste whole logs. Tell every subagent the same.
