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

## Failures, each with its cause

## Owed

1. Second run of record on the MI325X.
2. The plan choice for a 64-wide wavefront, as an AMD column of the launcher.
3. The AMD form of the many-warps-per-block lever.
4. The complete operation with AMD's best plan per row, over fp32.v1, twelve rows.
5. The 15-bit profile's tuned plan on the MI325X (merge `origin/lane/lowbit-int15`).
6. The decode rows.
