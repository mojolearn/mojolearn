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

## Experiment 3 result; registered experiment 4: reusable grouped scratch

Full-tile predicate H100 ABBA base 121.665776 / 121.676622 ms,
full-tile arm 120.262619 / 120.255100 ms. All digests and 288 transport
fixtures match, both device defects rejected. The ~1.2% difference falls
below the registered >=2% hypothesis: INERT for that hypothesis, no default
flip. Pod 02i56zs2nj2cp5 completed and was deleted/verified gone.

Source exposes a separate overhead: _shipped_body_kpack_hg ignores the caller
workspace and _kpack_run allocates m*n*groups floats plus two host fences per
call. Candidate reuses sufficiently sized caller scratch, launching the exact
same group and fold kernels, on the same ordered context, without those two
fences. The workspace size helper includes the actual default group count;
older small-buffer callers retain the safe allocating path. GemmWorkspace
already fences before growth and retains scratch across calls. This may
increase retained session scratch; it must be reported with training memory.
No leaf, group rule, fold, kernel body or floating-point operation changes.

Prediction before rental: >=5% lower weighted H100 GEMM sum. Less falsifies
this magnitude; no actual improvement rejects the default flip. ABBA builds
plus alternating old/new calls within each price process. All 12 real-shape
outputs must match, including each cross-build digest. A dedicated gate queues
two different products through the same scratch, checks both full outputs,
grows then reuses scratch, and exercises old undersized-buffer fallback.
Deliberately omit the candidate fold and require poisoned/mismatched outputs;
also sabotage the priced old arm and require price-comparator rejection.
Print required and retained workspace sizes; a non-grouped shape is refused
by the dedicated gate. Registered before launch, not measured yet.

## Experiment 4 result and registered pipeline qualification

H100 per-call synchronized sums: base 121.600850 / 121.724870 ms,
reuse 120.570859 / 120.573541 ms. In the same reuse-build process,
legacy/reuse ratios 1.009656 / 1.010550. The predicted >=5% reduction is
falsified: INERT for that magnitude hypothesis; no default flip justified by
this result. All cross-build output digests match. Nine dedicated queued,
growth/reuse and undersized-buffer matches pass; omitting the actual fold
leaves 262144 poisoned cells and is rejected. Price sabotage also rejects.
Required scratch 1835008 -> 5128461 -> 786432 floats; retained capacity grows
1835008 -> 5128461 -> 5128461 as intended. Exact same device kernels.

The per-call harness synchronizes after every GEMM, so it cannot determine
the proposed benefit of enqueueing consecutive training operations without
the former internal fences. Before the distinct full-step experiment:
H100 80GB HBM3, both pinned R2 corpora, 700 steps per arm each, target shape
and seed20260917 as above. Predict >=3% lower last-200 median on BOTH corpora
and <=1024 MiB extra sampled device memory. Less falsifies the speed magnitude
prediction; no significant full-step gain means keep this path off. All 700
loss words, all 6 endpoint witnesses and complete per-step attention reports
must match. The loaded binaries must read reuse False/True and have different
hashes. Builds reused between corpora, no repeat compilation and no component
share timing. This is a new pipeline hypothesis, not a kernel-speed claim.

R2 reconfirmed at Andrew's request during the run: enwik8 staged in 7s,
Pile GitHub in 8s; both exact pinned sizes and SHA256 values, one linked key
per training lease. The microbenchmarks generate inputs, so stage zero keys.

## Final default gate registered before execution

Pending the two AMD training comparisons, qualify the actual AMD default
with both the gather transport fixtures (288 cases) and all 20 named GEMM
plans (840 cases, 471240 words). First omit the actual tuned-loader flush
and gather-loader flush in separate device builds; each must produce a
bit mismatch against flat/oracle. All clean output digests must match.
Read back STAGE_FTZ False/True from both builds, rejecting missing, duplicate
and wrong read-backs before accepting them. Price sabotage must also fail.
ABBA prices must retain >=3% lower weighted GEMM time on both pairs, with
all 12 call digests identical across builds and to recorded NVIDIA values.
This gate runs only after a training-supported AMD default change is committed
and pushed. No Apple or RDNA default changes are proposed.

H100 workspace training source 06dee7da6 stages both corpora in 10 seconds.
Its stage.log verifies both exact sizes and SHA256 pins and links the data
into the running checkout. MOJOLEARN_STAGE_STRICT=1; the training script
uses only fetch_corpus_* --check.

## AMD training result and default decision

Source 8bd54f134, 700 steps per arm on both pinned corpora, completed.
Last-200 medians: enwik8 0.8319991125 -> 0.7159650970 seconds (ratio
0.8605358903); Pile GitHub 0.7372150485 -> 0.6244404975 (0.8470262494).
Geomean 0.8537543485: **14.6246% lower whole-step time**. The >=10%
prediction is supported on both. All 700 losses, 6 endpoint hashes and
every per-step attention report match. Nineteen intentional comparator
defects fail for their expected reasons before clean matches are printed.
AMD device footprint was not sampled by this probe (null), so no measured
AMD memory-change claim is made. These runs keep AMD's existing attention
path; the separate attention repair remains NVIDIA-only.

Direct enwik8 NVIDIA (workspace baseline source 06dee7da6) versus AMD
staged training also matches all 700 losses and all endpoint hashes. The
cross-vendor comparator rejects 15 deliberate defects first. Pile cross-
vendor comparison awaits the H100 run. This covers these training routes
and the 12 GEMM price outputs, not arbitrary-input universal identity.

AMD training VMs 7cef1b85-ef7e-43a1-b943-b0f84f913daa and
e7d39c7f-acb5-4f9b-b7bd-66d175841f2a both completed and were deleted,
verified GET 404 and absent from the VM list. No owed work was cancelled.

FLIP AMD operand staging ON through lib_gemm_stage_ftz_for. NVIDIA retains
its existing enabled path; Apple/RDNA/other columns stay unchanged. The
actual default is now subject to the preregistered all-plan/gather device
gates and ABBA prices before integration. Evidence: amd-training/ under
bench/results/gemm_kernel_speed_2026-09-18.

## NVIDIA pipeline result and bounded default qualification

Source 06dee7da6 completed all four 700-step runs. Last-200 medians:
enwik8 0.1968112644 -> 0.1948784720 s (0.9901794627); Pile GitHub
0.1965412674 -> 0.1956763240 s (0.9955991762). Geomean 0.9928856215:
**0.7114% lower step time**, a modest result. The >=3% pipeline prediction
is falsified, as the earlier >=5% microbenchmark prediction was. Neither
large-magnitude hypothesis is rescued by bit equality. The smaller measured
win on both corpora and repeated ~1% microbenchmark difference support the
repository's two-corpus default rule, subject to final capacity qualification.

All 700 loss words, all six endpoint witnesses and all per-step attention
reports match. All 19 comparator defects reject first. Peak sampled device
MiB: enwik8 15153 -> 16177 (+1024), Pile 15681 -> 16177 (+496). The
registered <=1024 MiB increase holds, at its bound on enwik8. All 700 losses
and endpoint states also match the AMD staged run on both corpora, after
15 cross-vendor negative controls per corpus. Pod 8dqkcbscpgbqu2 completed,
was deleted (204) and verified absent (404).

Enable the NVIDIA workspace row on this branch for final qualification.
Before the next H100 rental: actual-default dedicated reuse and undersized
buffer gate, real omitted-fold and price defects must fail, clean full outputs
must match. ABBA fixed-call prices must reproduce >=0.5% lower weighted time
in both pairs, a replication of the measured small effect, not a revision of
the falsified >=5% prediction. Then default B4 target, pinned R2 enwik8, 2000
steps, per-step attention witness. Predict all 2000 finite losses, eager
capacity 432 B, aexp separately 9663676416 B, sampled peak <=45000 MiB.
Compare all 2000 loss words to the committed prior B4 endurance record; any
bit movement is a defect. Timing versus that older rental is descriptive,
not an isolated speed comparison. Batch 4 is a tested survivor, not a claim
of the maximum batch boundary. Failure rejects the default before main.

FLIP decision on this branch: NVIDIA workspace reuse; geomean 0.9928856215,
quality witnesses identical, with the explicit memory tradeoff above. AMD
workspace default remains off. Full-tile bounds remains off (INERT for its
registered >=2% hypothesis). Apple remains unmeasured and unchanged.

## Final shipping decision (supersedes the proposed NVIDIA default above)

Andrew requested advice, stopped further experiments, then authorized shipping
the verified AMD work. The NVIDIA default proposal in f95ce5dc5 is withdrawn:
workspace reuse remains opt-in. Its default/endurance scripts are removed;
no such rental or 2000-step run was started, so none was cancelled. The
measured 0.71% gain does not justify shipping the memory tradeoff here.
NVIDIA and Apple production defaults remain unchanged by this work.

AMD actual-default qualification at 8777fcbdf completed successfully:
288 gather cases (2514240 words) and 840 cases across all 20 named plans
(471240 words), both legacy and default. Omitting either actual loader's
flush fails with got=654311424 versus flat=oracle=0. The priced device
sabotage also fails. Read-back prints legacy STAGE_FTZ=False and default
True, after missing/wrong/duplicate read-backs are rejected.

ABBA weighted GEMM sums: legacy 488.486730 / 490.933474 ms; default
379.378933 / 379.965076 ms. Both exceed the preregistered >=3% reduction.
All 12 output digests match across settings and the recorded H100 outputs.
Training evidence remains the two 700-step corpus comparisons above:
14.6246% less step time, with exact loss/state witnesses.

VM 66f2535a-1d64-49f2-a707-706a3e1b860b completed and was deleted,
verified GET 404 and list absent at 14:26:13 UTC. Every rental owned by
this lane has finished and been removed. No owed run was cancelled.
Final default evidence: bench/results/gemm_kernel_speed_2026-09-18/amd-default.

Integration: main 50faa970b adds only a CPU coverage-priorities record since
the prior merge; it merges without conflicts. Final production GEMM source
and matrix are identical to the verified 8777fcbdf build. Shell syntax and
source whitespace checks pass. The broad docs_facts check reported inherited
CPU training-lane/host-module list drift after merging main 9479ee119; these
generated lists are outside this change and were not edited. This is not
a numerical verification failure or an all-repository-checks-pass claim.
