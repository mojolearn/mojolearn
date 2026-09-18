# Release 0.8.7 proof repair checkpoint

2026-09-18: selectively backported the byte-LM sabotage workflow opt-in and
reference-sharded neighbor numerical mismatch reporting from main commits
41588b491 and 4aba74ba3. No new algorithm/native implementation was imported.
All 85 reporting/orchestration tests passed on this release branch.

The native freeze remains d9185528e1ba78e61e091b37489b52daebde342d. However,
identity_break.py changed, so the qualification tool snapshot differs from
aff968968. The previous GPU receipts must not be silently treated as witnesses
for these updated tools. Requalify the exact artifact with compatible source
witnesses before publication. No publication or completed certification claimed.

Release run 35350125464 is still collecting the older source's CPU matrix.
ARM64 clean sweep passed, classical UMAP expectations and sabotage sweep failed.
The new reporting fixes passed targeted native Apple M4 controls on main; see
main bench/results/identity_break/2026-09-18-cpu-proof-followup. The frozen
release still needs its full certification rerun after all repairs are ready.

Remaining: fresh NVIDIA/AMD UMAP recordings and explicit supersession of nine
stale saved-model expectations; inspect remaining matrix outcomes; requalify
updated tool source and rerun exact release gates. The six new CPU kernel
variants and other new main features belong to a later qualified wheel.

Follow-up 2026-09-18: backported main 434487ebf as 1a49e0e35. The classical
saved-model gate's plain --expect-mismatch could pass solely because an optional
GPU column disagreed, even when the CPU output matched the saved-model recording.
The negative verdict now also requires a nonempty list of actual CPU output
differences. Empty evidence fails. Per-lane/per-fixture requirements and the
clean gate's column disagreement failure remain intact.

Regression reproduced before the repair: the unchanged-CPU/changed-GPU-column
case exited 0 under --expect-mismatch. Tests now require exit 1, retain exit 1
for a clean column mismatch, and retain exit 0 for a changed CPU prediction.
All 41 orchestration tests pass via the workflow's binding-free unittest
command on this release branch. This is mocked inference and real temporary
reference files, not a fresh native certification. The current workflow does
not pass GPU columns to its sabotage invocation; this closes a tool-level
false-pass path and does not itself resolve the current UMAP failures.

The native freeze is unchanged. Qualification tools changed again; refresh
source-matched witnesses for the final repaired release source. No fresh rental
or heavy local build was started while run 35350125464 remained active and
local memory pressure remained at warning level.

## Timeout diagnosis and prepared UMAP capture

GitHub cancelled hosted Apple job 105616007971 at its 90-minute job timeout
while the fault sweep was running. The annotation and job metadata are retained
under bench/results/cpu_certification/2026-09-18-apple-timeout. x86 started after
that job ended. The workflow now allows 180 minutes for the two 50-minute
column sweeps plus builds/additional gates; serial matrix/two-shard limits are
unchanged. The active old run still uses its original budget.

tools/release_installed_checks.sh now captures a repeated all-fixture UMAP
column, records all nine saved GPU models, and replays them through the installed
CPU host binding against the fresh column. This runs on each existing bounded
qualification leg after the property captures. Source archives explicitly set
MOJOLEARN_GATE_COMMIT for saved-model report provenance. Outputs are
umap-column.json, umap-saved-models/, umap-cpu-replay.json and their logs.

Two binding-free shell orchestration tests pass, including five failure-stage
subcases proving supplemental failure leaves exit_code=1 and stops later work.
bash -n passes. Native UMAP execution is still OWED; no new rental was launched.
These commands retain evidence but do not overwrite/admit historical hashes.
Next: finish x86 diagnosis, run the final source-matched GPU qualifications,
review cross-vendor equality and explicitly supersede the nine stale UMAP
expectations, then rerun CPU certification and exact-artifact publication gates.
Keep the rental deadlines; the extra bounded steps do not authorize runaway
jobs or automatically extend a rental lease.

## Checkpointing and revised worker policy

User requested retaining intermediate evidence and parallel architecture jobs.
Backports eca5bbfde and 92bc7306e add three stage artifact uploads, strict local
shard resume, and max-parallel=3 with two shards/job (up to six workers). The
earlier two-worker/serial constraint is superseded for this architecture matrix.
No new Apple run has been dispatched. Keep the existing run's evidence.

run-column --resume preserves shard JSON and appends logs, starts missing shards
fresh, and recomputes the merged verdict. The harness still refuses incompatible
source/binary bytes, environment, machine provenance and protocol. Failed cells
are not converted to passes. All 43 binding-free orchestration tests pass.
Intermediate uploads preserve completed stages but automatic cross-run artifact
restore and safe stage selection are not yet implemented; a killed running
stage can still lose cells not yet uploaded. Those are the next staging tasks.

Main's --inference/--training CLI aliases and small-fixture-profile design are
separate next-wheel work; neither was imported into this frozen release branch.
