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
