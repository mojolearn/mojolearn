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
