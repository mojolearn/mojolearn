# GEMM kernel speed, 2026-09-18

Branch `lane/gemm-single-leaf`, isolated worktree `~/mojolearn-wt/gemm-single-leaf`, based on main c7442abed.
The branch name records the initial proposal, now rejected before implementation.

## Correction before work

The single-leaf proposal in the preceding assistant response was wrong.
`group_leaves=1` is not `P=1`. Target projection calls have P=6 or P=16;
the fold performs real additions and cannot be removed that way. Current
LANE_STATUS_gemm-next.md section 6(a) and brief section 23 explicitly retract it.
No production change implements the proposal.

## Registered experiment 1, before rental

Rebase the diagnostic body on shipped kpack_hg (gather staging, hardware fold
flush, shipped group sizes). Verify its full outputs against shipped at all
12 target GEMM calls. Before each match, deliberately sabotage the device
kernel; the same comparator must reject it. Print both output digests for
every matching call, and the moved cells/first index for every negative.

Predictions: exactly 12 clean full-output matches and 12 rejected device
sabotages; zero poison cells. Any mismatch or accepted sabotage falsifies the
harness and prevents drawing timing conclusions. H100 weighted base GEMM sum
predicted 110–135 ms (historical approximately 119 ms); outside that range
requires explanation, not silent comparison against history.

Working performance hypothesis: removing staging/prefetch/barriers reduces
the weighted diagnostic sum by at least 5 percent; less than 5 percent falsifies
that hypothesis. Removing per-step shared loads predicted at least 3 percent;
less falsifies it. These deliberately incorrect diagnostic programs are not
candidate optimizations, and their savings are neither additive nor a promised
speedup. Initialize shared pages for no-stage variants to avoid undefined reads.
Two complete price sweeps must agree on the direction before choosing an arm.
No training-step share measurement requested or planned.

No corpus needed for this microbenchmark. R2 staging is strict; training
qualification, if a candidate merits it, will stage pinned corpora from R2 and
run at least 700 steps on both corpora with per-step witnesses.

## Dispatch and Apple

Python _backend._layout caches backend selection, including architecture;
Mojo TARGET_COLUMN and GEMM matrix selections are compile-time constants.
No vendor detection occurs in the GEMM product loop.
Apple remains a possible bounded explicit diagnostic through the shared Metal
scheduler; no full Apple column or heavy Mac compute is planned. GPU core
partitioning has not been established. The known Apple FMA boundary discrepancy
prevents a universal cross-vendor identity claim; do not silently change it.
