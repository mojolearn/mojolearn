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

`amd-asm` (named in box_job.sh) calls `tools/lowbit_amd_tuned/asm_probe.sh`, which
does not exist; the phase is not used.

## Hunks in files this lane does not own

None yet. (Task 2 will add AMD branches to Lane D's launcher dispatch; every hunk
will be listed here.)

## What ran

| Box | Request | Commit | What | Verdict |
|---|---|---|---|---|
| MI325X | 1790657510941 | 0a29ffdf3 | amd-gate, full-price (run of record 2), amd-price | gate: clean GREEN, four arms seen failing, byte-path arm RED (fault); full-price GREEN; amd-price did not build |

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

## Failures, each with its cause

1. Job 1790657510941, `amd-unstated` (byte path forced): exit 139, "Memory
   access fault by GPU node-1 ... address ...340000", within the first gate;
   stdout was buffered so the plan is not named. Cause NOT established. The
   clean build with the same kernels passes every case. Next: the check names
   every launch, flushed, under `-D MOJOLEARN_INT8_AMD_TRACE=1` (the unstated
   arm builds with it).
2. Job 1790657510941, `amd-price`: the harness did not parse:
   `_launch_floor_kernel(out: ...)`, `out` is a keyword. Renamed `cell`.

## Owed

1. Second run of record on the MI325X.
2. The plan choice for a 64-wide wavefront, as an AMD column of the launcher.
3. The AMD form of the many-warps-per-block lever.
4. The complete operation with AMD's best plan per row, over fp32.v1, twelve rows.
5. The 15-bit profile's tuned plan on the MI325X (merge `origin/lane/lowbit-int15`).
6. The decode rows.
