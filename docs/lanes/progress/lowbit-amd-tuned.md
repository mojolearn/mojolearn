# lowbit-amd-tuned: the int8 unit plans for a wavefront of 64 (MI325X)

Lane F, branch `lane/lowbit-amd-tuned`, worktree `~/mojolearn-wt/lowbit-amd-tuned`.
Brief in force: `~/mojolearn-evidence/lowbit-units/brief_current.md`. Lane files
`~/mojolearn-evidence/lowbit-amd-tuned/`. Box: `do-amd` (MI325X, gfx942) through
the steward, one job of this lane at a time. Results in the repo:
`bench/results/lowbit_amd_tuned/`.

## Where it starts

Lane D's one MI325X batch (steward request 1790656243553, commit 464d5002a), ONE
run, filed at `bench/results/lowbit_mma_speed/2026-09-29/mi325x/lane_d_batch_1790656243553/`.
Over fp32.v1 at the four 512-token rows (fp32.v1 0.958, 2.634, 2.632, 2.903 ms):
reference unit plan 0.28 to 0.35; best one-product tuned plan 0.050 to 0.086;
four products one staging, best plan per row, 0.163 to 0.196; the complete
operation on the H100's launcher plan 0.191 to 0.299.

## What exists (the handover, commits bec341020, c02921536, d4a0f764e)

| Piece | File | State |
|---|---|---|
| AMD plans: direct kernel (one product, four), tuned file's staged kernels at AMD geometries, the 16-wave form of the 32-warp lever | `gemm/checks/gemm_int8_mma_amd.mojo` | NOT BUILT at the handover |
| Its gate (six builds, a reach gate for the scheduling arms) | `gemm/checks/gemm_int8_mma_amd_check.mojo` | NOT BUILT |
| Timing harness: every arm of `bench/gemm_lowbit_price_main.mojo` plus the AMD plans and the complete operation on every four-product plan | `bench/gemm_lowbit_amd_price_main.mojo` | NOT BUILT |
| Job phases `amd-gate`, `amd-price` | `tools/lowbit_mma_speed/{box_job,gate_job,price_job}.sh` | NOT RUN |

`amd-asm` (named in box_job.sh at the handover) called a script that never
existed; the phase is removed. `amd-fault-repro` (`tools/lowbit_amd_tuned/fault_repro.sh`)
is in its place.

## Hunks in files this lane does not own

All AMD-only (`comptime if TARGET_COLUMN == COLUMN_AMD`); no other column's
code path moves.

| File (owner) | Hunk |
|---|---|
| `gemm/checks/gemm_int8_mma_tuned.mojo` (Lane D) | import `COLUMN_AMD` from `checks.kernel_matrix` |
| same | `int8_pieces_dispatch`: on AMD, m > 16 and n > 8192 takes `INT8_PIECES_PLAN_FRAG2` |
| same | `int8_tuned_dispatch`: on AMD, m > 16 takes `INT8_TUNED_PLAN_FRAG2_K64`, and `INT8_TUNED_PLAN_K64` where n > 14336 |
| (withdrawn) `gemm/checks/gemm_int15_tuned_check.mojo`, `gemm/checks/gemm_int15_tuned.mojo` | This lane's fix of the refusal gate and its AMD branch in the fifteen-bit launcher were DROPPED at the orchestrator's word: Lane C fixed the gate at its root in its own branch (MI325X job 1790658677079) and rebuilt the launcher on the fused form; both files are now lane/lowbit-int15's as merged. |

The AMD pieces launcher now has Lane D's two forms: the SUMS form
(`identical_gemm_int8_pieces_amd_with_plan/_into`) and the FUSED form
(`identical_gemm_int8_pieces_amd_fused_with_plan/_into`, the seam
`int15_store_cell` at the last step, the tuned file's `_store_cell_of_sums`),
for the direct kernels and the staged ones alike. The gate checks every plan's
fused form, and the dispatched one, against the seam computed on the host's
sums.

## What ran

| Box | Request | Commit | What | Verdict |
|---|---|---|---|---|
| MI325X | 1790657510941 | 0a29ffdf3 | amd-gate, full-price (run of record 2), amd-price | gate: clean GREEN, four arms seen failing, byte-path arm RED (fault); full-price GREEN; amd-price did not build |
| MI325X | 1790659053624 | 72b659823 | lane/lowbit-int15's tuned gate (before its fix was merged here), the fault's reproduction (4 builds x 3 runs, byte path and stated loads) | tuned gate GREEN with this lane's refusal-gate fix (since withdrawn for Lane C's); the fault CAME BACK once in 24 runs (Failures 1) |
| MI325X | 1790658495381 | 6153fb859 | Lane D's unit and pieces gates (the AMD column), amd-gate, amd-price on the new dispatch, int15 tuned gate, int15 price | every lowbit-mma-speed and AMD gate GREEN with every arm as expected; amd-price GREEN (1380 of 1380 cold == record, 345 of 345 sabotage seen); int15 tuned gate RED at the stale refusal gate only; int15 price GREEN (372 digests warm == timed, sabotage seen) |
| MI325X | 1790657862351 | 6c6f2bb88 | amd-gate, amd-price (every plan of both files, run 3 of the shared arms), lane/lowbit-int15's tuned gate | amd-gate GREEN with all six arms as expected (byte-path arm passed); amd-price GREEN: cold == record at 1380 of 1380, sabotage seen at 345 of 345; int15 tuned gate RED (Failures 3) |

## Gate verdicts (MI325X, job 1790657510941)

`gemm/checks/gemm_int8_mma_amd_check.mojo`, 21 one-product plans and 13
four-product plans, built and passed FIRST TRY in the clean build: 8 gates, 0
failed (621 quantized-fixture cases over 27 shapes, 2576 planted, 115 minus-128,
560 hashed-plane four-product, 658 planted four-product, the batch-invariance
gate across the launcher's two plans, the refusal of k = 65537 by name).

| arm | exit | verdict |
|---|---|---|
| clean | 0 | 8 gates, 0 failed |
| direct loads' padding broken (`MOJOLEARN_INT8_AMD_SABOTAGE`) | 1 | SEEN failing: 6 of 8 gates; reach gate: 1886 cases reachable, 1808 differed, 0 unreachable cases differed |
| staging padding broken (`MOJOLEARN_INT8_TUNED_SABOTAGE`) | 1 | SEEN failing: 5 of 8; reach: 1185 reachable, 627 differed, 0 unreachable differed |
| middle sum takes HL twice (`MOJOLEARN_INT8_PIECES_SABOTAGE`) | 1 | SEEN failing: the two four-product gates only; the one-product gates pass |
| every value flipped (`MOJOLEARN_LOWBIT_SABOTAGE`) | 1 | SEEN failing: 6 of 8 |
| byte path forced (`MOJOLEARN_INT8_TUNED_UNSTATED`) | 139 | RED: GPU memory access fault (Failures 1) |

## Run of record 2 (task 1): MI325X, the same arms as Lane D's run 1

Job 1790657510941 phase full-price, commit 0a29ffdf3, against request
1790656243553 (464d5002a). Digests: 336 arm-and-row pairs present in both runs,
0 disagree. Times over fp32.v1 in the same run, run 1 / run 2 (fp32.v1 run 2:
0.961, 2.496, 2.519, 2.910 ms):

| arm | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 |
|---|---:|---:|---:|---:|
| reference unit plan | 0.279 / 0.278 | 0.335 / 0.360 | 0.322 / 0.336 | 0.349 / 0.349 |
| staged 32x32 per wave, block 64x64, k64 l16 | 0.086 / 0.083 | 0.050 / 0.055 | 0.086 / 0.091 | 0.101 / 0.100 |
| staged 64x64, block 128x128, k64 l16 | 0.165 / 0.162 | 0.072 / 0.078 | 0.168 / 0.176 | 0.075 / 0.075 |
| staged 32x32, block 128x128 (the H100's launcher plan) | 0.112 / 0.111 | 0.087 / 0.092 | 0.130 / 0.136 | 0.096 / 0.094 |
| four products one staging, 16x32 block 64x128 (H100 launcher) | 0.173 / 0.171 | 0.207 / 0.219 | 0.196 / 0.206 | 0.256 / 0.252 |
| four products one staging, 32x32 block 64x128 | 0.188 / 0.186 | 0.163 / 0.172 | 0.202 / 0.218 | 0.190 / 0.186 |
| complete op, inference (quantize A + four products one staging + stand-in recombination, H100 plan) | 0.191 / 0.189 | 0.252 / 0.233 | 0.228 / 0.220 | 0.299 / 0.263 |
| complete op, training (A and B quantized) | 0.223 / 0.222 | 0.295 / 0.285 | 0.295 / 0.294 | 0.330 / 0.305 |

Every ratio of run 2 is within 0.02 of run 1 but the small-tile plans at
mlp_up.t512 (16x16 block 32x32, k32 l4: 0.137 / 0.190; k64 l16: 0.101 / 0.134).
The full table of both runs: `bench/results/lowbit_amd_tuned/2026-09-29/mi325x/`.

## THE AMD LEVER TABLE (job 1790657862351, one run; ms over fp32.v1, same run)

fp32.v1 ms: qkv.t512 0.985, mlp_up.t512 2.441, mlp_down.t512 2.458, lm_head.t512
2.838 (n capped 16032); qkv.t1 0.067, qkv.t8 0.082, mlp_up.t1 0.174, t8 0.178,
mlp_down.t1 0.167, t8 0.199, lm_head.t1 1.421, t8 1.409. ONE LAUNCH AND ONE WAIT
(`probe.amd.launch-floor`) is 0.020 to 0.022 ms: 0.30 of fp32.v1 at qkv.t1.

One product, one lever at a time:

| lever | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 | qkv.t1 | mlp_down.t1 | lm_head.t1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0. the reference unit plan | 0.272 | 0.365 | 0.344 | 0.359 | 0.867 | 0.948 | 0.206 |
| control: its schedule respelled here, alignment not stated | 0.283 | 0.384 | 0.362 | 0.389 | 0.791 | 0.987 | 0.224 |
| A. the same, alignment stated | 0.278 | 0.380 | 0.352 | 0.372 | 0.729 | 0.824 | 0.225 |
| B. 16-byte loads, two unit steps a load | 0.182 | 0.200 | 0.218 | 0.194 | 0.575 | 0.588 | 0.175 |
| C. four windows' loads issued before the first unit step | 0.149 | 0.195 | 0.186 | 0.188 | 0.465 | 0.426 | 0.136 |
| C, one wave a block | 0.158 | 0.201 | 0.188 | 0.204 | 0.481 | 0.429 | 0.125 |
| D. 32x32 per wave, 2x2 waves (nothing staged) | 0.096 | 0.107 | 0.112 | 0.123 | 0.703 | 0.740 | 0.153 |
| D, two windows a turn | 0.094 | 0.107 | 0.111 | 0.111 | 0.592 | 0.632 | 0.151 |
| E. 64x64 per wave, one wave a block (nothing staged) | 0.110 | 0.073 | 0.122 | 0.087 | 0.931 | 1.057 | 0.164 |
| F. staged in threadgroup memory: 32x32 per wave, block 64x64, k64 | 0.085 | 0.059 | 0.093 | 0.104 | 0.976 | 1.168 | 0.174 |
| F, k128 | 0.082 | 0.069 | 0.098 | 0.084 | 0.917 | 1.101 | 0.179 |
| G. MORE WAVES A BLOCK: 8 waves of 32x32, block 64x128 | 0.087 | 0.052 | 0.102 | 0.086 | 1.054 | 1.207 | 0.183 |
| G. 16 waves of 16x32, block 64x128 | 0.097 | 0.106 | 0.114 | 0.137 | 0.961 | 1.137 | 0.210 |
| G. 16 waves of 16x32, block 128x64 | 0.094 | 0.118 | 0.114 | 0.136 | 0.879 | 1.067 | 0.274 |
| G. 16 waves of 16x16, block 64x64 | 0.132 | 0.166 | 0.186 | 0.176 | 0.811 | 1.081 | 0.276 |
| G. 16 waves of 32x32, block 128x128 (THE H100'S PLAN) | 0.113 | 0.096 | 0.141 | 0.099 | 1.115 | 1.293 | 0.234 |
| H. tall wave: 64x32 per wave, block 128x64 | 0.108 | 0.055 | 0.132 | 0.057 | 1.086 | 1.419 | 0.171 |

Read plainly:
- STATING THE ALIGNMENT, the H100's largest lever, bought NOTHING on the MI325X
  at the wide rows (0.283 to 0.278) and some at the decode rows (0.79 to 0.73).
- The load WIDTH and ISSUING A TURN'S LOADS TOGETHER are the decode rows'
  levers: 0.79 to 0.47 at qkv.t1, 0.99 to 0.43 at mlp_down.t1.
- THE MANY-WAVES LEVER (task 3), in its AMD form: a block of sixteen waves
  (1024 threads, the most a block holds) cost time at every 512-token row
  against the same block cut into eight or four larger waves (0.097 against
  0.087 in a 64x128 block at qkv, 0.106 against 0.052 at mlp_up). The H100's
  plan, sixteen warps of 32x32 in 128x128, is 1.3 to 1.8 times the best AMD
  plan here. The two 32-warp plans cannot run (2048 threads); what they test,
  the smallest tiles in the most waves, is the 16x32 and 16x16 rows above and
  it loses.
- STAGING wins at the 512-token rows by a little (0.085 against 0.094 nothing
  staged); nothing staged wins at every decode row by a factor near two.

Four products (one launch, three Int32 sums a cell), the best plan per row
against the H100's launcher plan:

| row | H100's plan (16x32, block 64x128, staged) | best AMD plan | which |
|---|---:|---:|---|
| qkv.t512 | 0.171 | 0.162 | staged 16x32, block 32x128 |
| mlp_up.t512 | 0.225 | 0.158 | staged 32x32, block 64x64 |
| mlp_down.t512 | 0.212 | 0.185 | staged 32x32, block 64x64 |
| lm_head.t512 | 0.261 | 0.166 | staged 32x32, block 128x64 |
| qkv.t1 / t8 | 1.787 / 1.483 (the H100's small plan: 1.461 / 1.342) | 0.712 / 0.708 | direct 16x16, block 32x32, 16-byte loads |
| mlp_up.t1 / t8 | 0.857 / 0.821 (0.689 / 0.736) | 0.402 / 0.406 | the same |
| mlp_down.t1 / t8 | 2.266 / 1.896 (1.941 / 1.625) | 0.774 / 0.746 | the same |
| lm_head.t1 / t8 | 0.419 / 0.458 (0.350 / 0.359) | 0.254 / 0.309 | the same |

THE COMPLETE OPERATION (int8 stand-in: quantize A in parallel, four products in
one launch, the stand-in recombination; one wait), best plan per row, this run:
inference 0.173, 0.176, 0.196, 0.181 at the 512-token rows; 0.820, 0.796,
0.427, 0.447, 0.901, 0.832, 0.255, 0.315 at the decode rows (qkv, mlp_up,
mlp_down, lm_head; t1 then t8): UNDER fp32.v1 AT ALL TWELVE ROWS. Training
(both operands quantized per call) 0.200, 0.218, 0.273, 0.222; decode 1.09 to
1.74 but the head (0.87, 0.94): the right operand's quantizer alone is 0.58 to
0.79 of fp32.v1 there.

## THE PLAN CHOICE FOR A WAVEFRONT OF 64 (task 2)

`int8_amd_dispatch` and `int8_amd_pieces_dispatch` (this lane's launcher) and
the AMD column of Lane D's two dispatchers (hunks above) now read this
measurement. One product: m <= 16 the direct plan, four windows a turn; m > 16
staged 32x32 in 64x64 with k128, or 64x32 in 128x64 where n > 8192. Four
products: m <= 16 direct 16x16, 16-byte loads; m > 16 staged 32x32 in 64x64,
or in 128x64 where n > 14336. Lane D's launcher can take only its own file's
plans (the direct kernels live here and this file imports that one); its AMD
column takes the best of them.

## THE COMPLETE OPERATION ON AMD'S LAUNCHER (tasks 4 and 6), job 1790658495381

One call, one wait, over fp32.v1 in the same run. The launcher takes AMD's plan
per row (`int8_amd_pieces_dispatch`); beside it the same operation on the H100's
launcher plan. fp32.v1 ms: 1.004, 2.509, 2.521, 2.861; decode 0.060, 0.073,
0.169, 0.180, 0.163, 0.197, 1.411, 1.409.

| operation | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 | qkv.t1 | qkv.t8 | mlp_up.t1 | mlp_up.t8 | mlp_down.t1 | mlp_down.t8 | lm_head.t1 | lm_head.t8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 stand-in, inference, AMD's plan | 0.179 | 0.176 | 0.211 | 0.193 | 0.935 | 0.890 | 0.518 | 0.496 | 1.046 | 0.956 | 0.274 | 0.331 |
| the same, the H100's plan | 0.180 | 0.184 | 0.220 | 0.195 | 1.825 | 1.483 | 0.742 | 0.764 | 1.964 | 1.708 | 0.353 | 0.361 |
| int8 stand-in, training (both operands quantized), AMD's plan | 0.198 | 0.214 | 0.256 | 0.227 | 1.459 | 1.305 | 1.111 | 1.054 | 1.713 | 1.508 | 0.870 | 0.929 |
| one product, AMD's plan, quantize A included | 0.104 | 0.060 | 0.116 | 0.060 | 0.561 | 0.494 | 0.240 | 0.243 | 0.511 | 0.436 | 0.127 | 0.150 |
| one launch and one wait (the floor) | 0.017 | 0.007 | 0.007 | 0.007 | 0.192 | 0.153 | 0.087 | 0.087 | 0.149 | 0.073 | 0.011 | 0.012 |

THE FIFTEEN-BIT PROFILE ITSELF (task 5; lane/lowbit-int15's harness, the real
quantizer to planes, the four products on AMD's launcher through this lane's
branch in `gemm_int15_tuned.mojo`, the profile's own recombination and pinned
seam as its epilogue launch), same job; fp32.v1 ms 0.962, 2.355, 2.433, 2.673;
decode 0.062, 0.073, 0.169, 0.170, 0.159, 0.195, 1.421, 1.381:

| operation | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 | qkv.t1 | qkv.t8 | mlp_up.t1 | mlp_up.t8 | mlp_down.t1 | mlp_down.t8 | lm_head.t1 | lm_head.t8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inference.int15i64.v1.tuned (THE COMPLETE CALL) | 0.273 | 0.208 | 0.243 | 0.214 | 1.396 | 1.503 | 0.669 | 0.843 | 1.250 | 1.247 | 0.301 | 0.376 |
| int15i64.v1.tuned (the product and its epilogue, planes given) | 0.184 | 0.169 | 0.185 | 0.176 | 0.755 | 0.726 | 0.395 | 0.463 | 0.793 | 0.731 | 0.252 | 0.326 |
| inference.int15i64.v1.planes (the reference unit plan, four launches) | 0.633 | 0.829 | 0.752 | 0.815 | 1.850 | 1.837 | 0.805 | 0.943 | 1.765 | 1.668 | 0.456 | 0.572 |

At the 512-token rows the complete fifteen-bit call is 0.21 to 0.27 of fp32.v1
(the brief's stand-in number was 0.19 to 0.30). At the decode rows it is OVER
fp32.v1 at qkv and mlp_down (1.25 to 1.50): the fifteen-bit parallel quantizer
to planes alone takes 0.043 to 0.080 ms there (0.70 of fp32.v1 at qkv.t1), and
the call is three launches (quantize, sums, epilogue) where fp32.v1 is one. The
int8 stand-in's quantizer at the same row is under half of that.

## Failures, each with its cause

1. Job 1790657510941, `amd-unstated` (byte path forced): exit 139, "Memory
   access fault by GPU node-1 ... address ...340000", within the first gate;
   stdout was buffered so the plan is not named. Cause NOT established. The
   clean build with the same kernels passes every case. Next: the check names
   every launch, flushed, under `-D MOJOLEARN_INT8_AMD_TRACE=1` (the unstated
   arm builds with it).
2. Job 1790657510941, `amd-price`: the harness did not parse:
   `_launch_floor_kernel(out: ...)`, `out` is a keyword. Renamed `cell`.
3. Job 1790657862351, lane/lowbit-int15's tuned gate on the MI325X: RED at
   ONE gate, `check_int15_tuned_refuses` ("only 2 of 3 launches refused").
   The oracle and planted gates PASSED and all four sabotage arms were seen
   failing. Cause: the gate expects the sums kernel to refuse k = 65536, and
   lane/lowbit-mma-speed raised that kernel's bound to 65536 (commit
   520406a38). The gate is stale, not the arithmetic. Worse, that call then
   LAUNCHES at k = 65536 on buffers of one byte: reads out of bounds. Seen
   again in job 1790658495381. FIXED at its root on this branch (rule 16): the
   gate reads both bounds, holds every code the largest `k` reads, and checks
   that each refusal names its bound; the patch is sent to the orchestrator
   for Lane C.

## Owed

1. Second run of record on the MI325X.
2. The plan choice for a 64-wide wavefront, as an AMD column of the launcher.
3. The AMD form of the many-warps-per-block lever.
4. The complete operation with AMD's best plan per row, over fp32.v1, twelve rows.
5. The 15-bit profile's tuned plan on the MI325X (merge `origin/lane/lowbit-int15`).
6. The decode rows.
