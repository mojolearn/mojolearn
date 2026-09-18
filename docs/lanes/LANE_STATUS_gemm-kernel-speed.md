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

## Registered experiment 2: AMD operand-flush transport

Source inspection: TUNED_STAGE_FTZ defaults on only for NVIDIA. AMD's gather
body currently flushes at operand use; the already implemented opt-in moves
that same operation to staging. This is not the shipped AMD post-FMA class
flush and does not replace its RN-FMA then post-round flush seam.

Prediction before rental: enabling existing MOJOLEARN_GEMM_STAGE_FTZ on
MI300X lowers the fixed 12-call weighted GEMM sum by at least 3 percent from
the same-device legacy-stage build. Two ABBA passes must agree. A smaller
change is INERT for this hypothesis; a regression rejects the default flip.
The 144 gather fixtures (3 operations, 8 K lengths, 2 operand swaps, 3 group
sizes) must match PLAN_FLAT and the host contract word for word in both
builds. Omit the actual gather flush deliberately and require a mismatch;
sabotage a priced device arm and require rejection. Print all 144 matching
triples of digests in each clean build, and all 12 cross-build price matches.
Any changed word is a defect. Both corpora at 700 steps remain required
before shipping a measured candidate. No training run is owed until it is
registered and launched.

## Experiment 1 result and registered experiment 3

H100 em6ro8lqzvh0ml completed both sweeps and was deleted (DELETE 204, GET 404).
Base weighted sum 120.861107 / 120.840876 ms; no-stage ratios
0.722425 / 0.722484, no-shared-load ratios 0.989565 / 0.989932.
All 24 base/shipped full-output comparisons matched, all 24 device sabotages
failed. Per-step shared-load >=3% hypothesis falsified. Staging/prefetch/barrier
>=5% hypothesis supported as an aggregate diagnostic, not an isolated cause or
achievable optimization. Diagnostics intentionally compute different answers.
Evidence: bench/results/e1g/2026-09-18-nvidia-h100-gemm-hg-diag-retry.

NVIDIA candidate: on a full operand tile, perform one block-uniform bounds
check instead of repeating the outer-index check at each of 8 scalar gathers.
Keep the masked loads on ragged tiles, all operand bits, step sequence, FMA
seam and fold unchanged. Opt-in MOJOLEARN_GEMM_GATHER_FULL_TILE only.
Prediction before rental: >=2% lower weighted 12-call GEMM sum in both ABBA
pairs. Less is INERT for the hypothesis; regression rejects default flip.
288 transport fixtures now include 129x131 full-plus-ragged tiles; all must
match the untuned flat kernel and host oracle. A deliberate full-tile load
corruption must fail before the clean gate. The independent price comparator
also receives actual sabotaged device output and must reject it.

Experiment 2 (AMD) was already launched at 8c9c1cbf9 with its preregistered
144 small ragged fixtures, no full-tile optimization. Experiment 3's expanded
fixture is independent; no retrospective claim it ran on experiment 2.
