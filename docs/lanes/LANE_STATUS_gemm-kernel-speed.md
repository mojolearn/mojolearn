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

## Experiment 2 result; registered training qualification

MI300X source 8c9c1cbf9 completed, body exit 0; VM
86dc3471-c47c-4bee-bc3a-a56de6ea86dc deleted and verified 404/list absent.
ABBA shipped-column weighted GEMM sums: legacy 488.722683 / 490.584033 ms,
staged 380.639570 / 379.496331 ms. About 22.4% less GEMM time; >=3% hypothesis
supported. All 12 cross-build output digests match at each pass; 144 transport
fixtures pass on both explicit settings, actual omitted-flush defect fails
(got 654311424, flat/oracle 0) and price sabotage moves 576 cells and is rejected.
This is microbenchmark evidence, not a whole-training improvement yet.

Before training rentals: 700 steps per arm per corpus on MI300X, B1 L2048
DM768 H12 KV12 HD64 FF2048 layers12 V50257, seed20260917. Two independent
one-corpus leases avoid placing four long runs against one 60-minute watchdog.
R2 corpus/enwik8/input.txt (100000000 bytes, SHA256
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8) and
corpus/pile_github/input.txt (97124565 bytes, SHA256
52a5b4c36ab9119c15505331c10e3b23690377d40fbac3e598c7fafe13a324df).
Strict staging plus on-box --check; no origin downloads.

Prediction: at least 10% lower last-200 median whole-step time on each corpus.
Smaller falsifies this magnitude prediction; a regression on either corpus
rejects the default flip. All 700 loss words, 6 endpoint state hashes at steps
0 and699 and each step's complete attention witness must match exactly.
Any bit difference rejects the optimization. Binary read-back must show
stage_ftz False/True and different loaded binary hashes; same-arm comparisons
are rejected. The comparator first corrupts each witness type and verifies
the corresponding check fails. No component/step-share timing is performed.

A historical attention result was used only to exercise the comparator with
synthetic dispatch labels; all 13 intentional corruptions were rejected for
their expected reasons. This is a harness test, not a new training result.
