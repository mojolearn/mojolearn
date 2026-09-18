# Next-wheel coverage audit and pending qualification

Updated 2026-09-18. Andrew asked to find APIs beyond the 246 appendix variants,
expose finished work in wheels, and finish reference/parallel qualification.
Worktree: `~/mojolearn-wt/next-wheel-coverage`, branch `lane/next-wheel-coverage`.
Commits `68a9241ac` and `5f7eb5ee3` are pushed and integrated into main.
Do not merge this new Python inventory change into the already-qualified
`release/087-final` branch without rebuilding its native source witnesses.

Completed: corrected nested-package API inventory, safe AST-only wheel export
comparison, per-property/per-device numerical reference counts in installed
coverage, and current-table rather than historical hold reasons. Full audit:
`docs/NEXT_WHEEL_COVERAGE.md`. 48 targeted tests passed, then 29 coverage tests
passed after the hold-reason correction. Both tools and source CLI were run.

Measured inventory: 284 public names, including aliases/helpers/constants;
229 filtered callable API entries and 183 implementation symbols, neither a
count of distinct algorithms. Main has 228 lanes and 18 additional lanes
outside the appendix. Frozen 0.8.7 has 229 (one explicit refusal lane later
removed from main). Actual published 0.8.5 contains 141 public names and one
host binding; the 0.8.7 Linux candidate contains all 284 names and 32 host
bindings. No discovered current public name is missing from that candidate.

The ordinary pending routes were NOT promoted. Against current main, scoped
admission of five neural routes using the new three-GPU reference records
still refuses missing batchgrad, batchscale, ragged and stepfull. Frozen
0.8.7's older table lacks only stepfull. Twelve other holds need independent
GPU witnesses and complete replay. Seventeen logical-shard CPU routes are
implemented; 33 parallel routes still require GPU execution and further CPU
implementation. Do not turn serial CPU fits into alleged multi-GPU proof.

## Active release / evidence jobs

The original 0.8.7 Linux release has passed all 162 CPU lanes, models and
self-test, all three GPU qualification targets, final admission and current
reference comparisons. All rentals are deleted. Exact SHA and receipts are
under `~/mojolearn-evidence/release-087-final/`, staging `linux-publish-v4`.

Standard publishing workflow **35350125464** was dispatched on `aff968968`:
https://github.com/mojolearn/mojolearn/actions/runs/35350125464
Real-Metal ephemeral runner is session **60355**; log `logs/pypi-release-runner.log`.
The build and full CPU certification remain active. No publication yet.

ARM64 job **105616007904** failed the classical saved-model check: exactly
nine UMAP fixtures, 45 DIFFER output lines including identity-column checks;
other 600 fixtures passed. The expectations date from 2026-09-15, before
`24bd362b3` deliberately repaired UMAP transform batch invariance on Sep 16.
ARM64 now matches the later CPU reference (`umap/base` infer
`01d01e040df02560`, old `5a4ef62db605973a`). Fresh GPU recording must confirm
before replacing old expectations. Do not weaken or skip the failed check.

Direct partial job logs ARE available while the run is active:
`gh api repos/mojolearn/mojolearn/actions/jobs/105616007904/logs`.
`gh run view --log-failed` says to wait, but the API above works. The retained
partial log is `logs/arm64-live-attempt.log`. Inspect completed steps for more
failures; let full certification provide its evidence.

A guarded Apple recording queue is active, session **11099**, external script
`~/mojolearn-evidence/next-wheel-coverage/queue_apple_pending.py`, log
`queue-apple-pending.log`. It waits for the Mac BUILD job (not the overall run)
to succeed, verifies checkout commit, copies the exact new Mac wheel, and takes
a single Metal slot with nice 19 and one numerical worker. It records fresh
UMAP/PCA saved models and checks their host inference, then records UMAP/PCA
and all 17 pending ordinary lanes across nine fixtures twice with all extra
property flags. Installed origin/backend/harness digest are asserted. Every
record must be complete/stable or N/A; refusal/mismatch stops the capture.
Two-hour execution cap, per-stage limits; no rental. It does not admit tables,
promote routes or publish. Output: `apple-installed-reference-capture/`.
If the build fails, the queue stops; inspect and repair rather than assuming it
has run. Do not start duplicate Metal work or additional cloud workers during
CPU certification (matrix serial, at most two cloud numerical workers).

Next: resolve UMAP saved-model references with measured GPU/host equality,
refresh workflow reference inputs with explicit supersession while preserving
historical records, rerun required release checks, publish and verify both
wheels, tag successful source and merge release into then-current main without
discarding newer work. New pending-route records still require independent
NVIDIA/AMD/CPU comparisons and negative controls before promotion. Physical
parallel claims require real one-vs-two-GPU records on NVIDIA and AMD.
