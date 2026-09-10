# GEMM lane handoff (2026-09-09)

Branch `lane/gemm-identical`, worktree `.claude/worktrees/agent-a7f6881ef7f397d7d`.
Every build and run below was on rented NVIDIA L40S pods (driver 580.126.09,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, torch
2.4.1+cu124, cuBLAS 120402, cupy 14.2.0, Mojo 1.0.0). Nothing was built or run
on the Mac. No H100 run happened (the session was cut by a rate limit and then
wound down), so the H100 reference table of the brief is NOT DONE.

## Commits (`%h parent %p`)

- `2ca3ff2a parent f3d76e8d` identical speed arms, direct cuBLAS opponents,
  hardware ftz row, tuned kernel with local fold, Gram past 128 features via v1
- `ed8dec68 parent 2ca3ff2a` the tuned register-blocked plans are real plans
  8-10 of the profile (`gemm_identical_tuned.mojo` deleted)
- the handoff commit (this file, plus the L40S logs under `bench/results/e1g/`)

## What landed

- `bench/speed/gemm_speed_main.mojo`: under `-D MOJOLEARN_NUMERIC_IDENTICAL=1`
  two arms of ours, `ours-v1-identical` (`identical_gemm_into`, every row) and
  `ours-core-identical` (`core/gemm.mojo`'s route; named
  `ours-core-swap537-identical` when built with `-D MOJOLEARN_537_GEMM_IDENT_SWAP=1`).
  `MOJOLEARN_SPEED_GEMM_ARMS=v1,core`, `MOJOLEARN_SPEED_SHAPES=a,b`. FAST arm untouched.
- `tools/speed_gemm_arm.py`: arms `cublas-fp32/tf32` (torch.matmul, cuBLAS
  preferred), `torch-fp32/tf32` (torch default backend), `cublas-sgemm-fp32/tf32`
  (cublasSgemm through cupy, math mode DEFAULT / TF32_TENSOR_OP, checked
  against torch fp32 before timing). Versions on every NOTE line. `--shapes`, `--arms`.
- `checks/kernel_matrix.mojo`: `lib_hardware_ftz_fma_for[column]`, True on
  NVIDIA only.
- `gemm/checks/gemm_identical.mojo`: plans 8-10 `PLAN_TUNED_32_2X2`,
  `PLAN_TUNED_64_4X4`, `PLAN_TUNED_128_8X8` (register blocking across cells,
  two shared pages, one barrier per window, VEC-wide loads, operand flushes
  once per staged value, fold stack in thread-local memory, per-step seam
  `fma.rn.ftz.f32` on NVIDIA via `llvm.nvvm.fma.rn.ftz.f`). `choose_gemm_plan`
  picks them for `m >= 32, n >= 32, m*n >= 131072` (8x8 when both >= 128,
  else 4x4 when both >= 64, else 2x2); `choose_gemm_plan_untuned` is the old
  rule. `GEMM_PLAN_COUNT = 11`, so every existing all-plans gate covers them.
- `gemm/checks/gemm_device_check.mojo`: `check_tile_fold_is_the_contract_tree`
  (thread-local 4-cell fold vs `fold_balanced_tree`, P in 1..2049).
- `gemm/checks/gemm_unpinned.mojo`: the control arm mirrors
  `choose_gemm_plan_untuned` (the tuned plans have no unpinned counterpart).
- `gemm/checks/gemm_tuned_probe.mojo`: untuned plan vs dispatcher, bits and
  time; `MOJOLEARN_GEMM_PLAN=<id>` forces a plan.
- `core/gemm.mojo`: `gemm_tn` under IDENTICAL routes `m > GRAM_MAX_COLS`
  (128) through v1 OP_TN (`gemm_tn_identical_v1`) instead of raising; split-K
  path for `m <= 128` untouched. `core/gemm_identity_check.mojo`
  (`check_gemm_tn_over_capacity_takes_v1`, bits vs `gemm_oracle`),
  `glm/checks/ols_check.mojo` + `glm/ols_main.mojo`
  (`check_ols_over_capacity_fits`) replace the refusal assertions.
- `GEMM_IDENT_SWAP_537` NOT flipped (owner's decision); only measured.

## Measured (L40S), with log paths

Run 1, pod nsnjnhau1tm277, tree f3d76e8d + the driver/opponent edits of 2ca3ff2a
(kernels unchanged), logs `bench/results/e1g/2026-09-09_092558-nvidia-l40s-identical-gemm/`:

- `l40s.card` body byte-equal to the known-good NVIDIA card
  `bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card`;
  `device_check.log` all 6 gates green [IDENTICAL].
- medians of 5 rounds, ms (`speed_v1.log`, `speed_core.log`, `opponents.log`):

| shape | v1 identical | core identical | cublas-fp32 | cublas-tf32 | torch-fp32 | torch-tf32 | sgemm-fp32 | sgemm-tf32 |
|---|---|---|---|---|---|---|---|---|
| gram.32x32x1M | 138.50 | 0.367 | 0.375 | 0.363 | 0.376 | 0.363 | 0.374 | 0.361 |
| gram.128sq.x100003 | 5.790 | 0.174 | 0.123 | 0.083 | 0.132 | 0.121 | 0.245 | 0.101 |
| ols.step1.16x16x64K | 0.383 | 0.032 | 0.024 | 0.023 | 0.024 | 0.023 | 0.024 | 0.021 |
| pca.transform.wide.8192x64x128 | 0.073 | 0.215 | 0.019 | 0.016 | 0.019 | 0.015 | 0.017 | 0.014 |
| kmeans.dist.4096x64x64 | 0.030 | 0.065 | 0.015 | 0.014 | 0.015 | 0.014 | 0.014 | 0.013 |
| llama8b.qkv.t512 | 6.399 | 32.08 | 0.518 | 0.151 | 0.515 | 0.155 | 0.496 | 0.148 |
| llama8b.mlp_up.t512 | 22.87 | 223.7 | 1.813 | 0.715 | 1.806 | 0.715 | 1.796 | 0.718 |
| llama8b.mlp_down.t512 | 23.33 | 114.5 | 1.649 | 0.560 | 1.651 | 0.564 | 1.637 | 0.554 |
| llama8b.lm_head.t512 | 220.8 | skipped | 16.57 | 5.138 | 16.28 | 5.179 | 16.20 | 5.160 |

  (all 20 rows are in the logs; core arm's lm_head.t512 skipped by the 40 GMAC cap.)
- `tuned_probe.log`: the tuned kernel as shipped in the tree (software ftz,
  SIMD fold stack) BITS MATCH v1 at all 20 rows; qkv.t512 6.40 -> 4.41 ms.
- The same pod's later probes (hardware ftz: qkv.t512 2.78 ms; local fold:
  1.90 ms; forced-plan sweep: 4x4 1.51 ms, 8x8 1.27 ms, 128x128 TN row loses on
  every tuned tile) and the Gram-routing checks (`gemm_identity_check`
  IDENTICAL and FAST both exit 0; `ols_main` IDENTICAL exit 0, FAST exit 1,
  log lost) were on the pod when it expired: their LOGS ARE LOST, the numbers
  above are from the session transcript only and are re-run owed.

Run 2, pod 6kd9espvn0e04x, tree ed8dec68 (the merged dispatcher), logs
`bench/results/e1g/2026-09-09_123601-nvidia-l40s-identical-gemm-merged/`:

- `l40s2.card` body byte-equal to the known-good NVIDIA card (the dispatcher
  now sends 7 of the 20 rows to the tuned plans; bits unchanged).
- `device_check_l40s2.log`: 7 gates green [IDENTICAL], including
  `check_tile_fold_is_the_contract_tree` and every TUNED plan line of the
  oracle and launch-invariance gates (27 OK lines mention TUNED).
- `probe_l40s2.log`: 20 match, 0 MOVED, 0 refused (untuned plan vs dispatcher).
- `gemm_identity_l40s2.log`: `check_gemm_tn_over_capacity_takes_v1 OK [IDENTICAL]`.
- `ols_main_l40s2.log`: `check_ols_over_capacity_fits OK [IDENTICAL]: 512 x 130 fits`.
- `ols_main_fast_l40s2.log` exit 1: `check_ols_dispatch_routes_special_shapes`
  raises `gemm_nt_gram: n == 1 is not a Gram shape` under FAST on NVIDIA.
  PRE-EXISTING: FAST on non-Apple takes `gemm_tn_via_transpose` at every
  shape and that entry refuses n == 1; this lane's change is inside the
  IDENTICAL branch only. Not fixed; report it to the OLS owner.
- medians of 5 rounds, ms (`speed_v1_l40s2.log`, `speed_core_l40s2.log`,
  `speed_swap_l40s2.log`, `opponents_l40s2.log`); v1's plan per row is on
  its `FSPEED-NOTE ... plan=` line:

| shape | v1 identical (plan) | core identical | core swap537 | cublas-fp32 | cublas-tf32 | torch-fp32 | sgemm-fp32 | sgemm-tf32 |
|---|---|---|---|---|---|---|---|---|
| gram.32x32x1M | 138.47 (SPLITK) | 0.368 | 0.368 | 0.378 | 0.363 | 0.377 | 0.374 | 0.362 |
| gram.128sq.x100003 | 5.775 (TILE 16x16) | 0.174 | 0.175 | 0.123 | 0.083 | 0.133 | 0.248 | 0.100 |
| ols.step1.16x16x64K | 0.381 (SPLITK) | 0.032 | 0.032 | 0.024 | 0.023 | 0.024 | 0.023 | 0.022 |
| pca.transform.wide.8192x64x128 | 0.0245 (TUNED 64x64) | 0.215 | 0.030 | 0.019 | 0.017 | 0.019 | 0.017 | 0.014 |
| kmeans.dist.4096x64x64 | 0.0172 (TUNED 64x64) | 0.065 | 0.022 | 0.015 | 0.015 | 0.015 | 0.015 | 0.014 |
| llama8b.qkv.t512 | 1.263 (TUNED 128x128) | 32.10 | 1.268 | 0.536 | 0.153 | 0.518 | 0.507 | 0.149 |
| llama8b.mlp_up.t512 | 4.866 (TUNED 128x128) | 223.2 | 4.878 | 1.828 | 0.725 | 1.840 | 1.835 | 0.729 |
| llama8b.mlp_down.t512 | 4.355 (TUNED 128x128) | 117.0 | 4.358 | 1.686 | 0.571 | 1.683 | 1.681 | 0.574 |
| llama8b.lm_head.t512 | 35.60 (TUNED 128x128) | skipped | skipped | 16.89 | 5.251 | 17.31 | 17.37 | 5.314 |

  So on the L40S the identical v1 is now 2.1x to 2.7x cuBLAS FP32 at the
  t512 rows (was 12x to 14x in run 1) and 1.1x to 1.3x at pca.wide/kmeans;
  m <= 8 rows and the TN rows are unchanged. The swap-537 arm is v1 through
  the synchronizing `identical_gemm` form (same bits as v1, within 1% at
  the t512 rows, 1.2x to 1.5x slower than `identical_gemm_into` at the two
  small NT rows because of the per-call workspace and wait); it does not
  touch the TN, gram or gemv rows. Flag NOT flipped.

## RUN OWED on the Apple M4 (orchestrator)

    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_device_check.mojo
    MOJOLEARN_GEMM_CARD_HOST_CAP=1 tools/gemm_card.sh device /tmp/apple.card
      then diff /tmp/apple.card (comments stripped) against
      bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
    tools/with_identical_mode.sh pixi run check-gemm-identity
    pixi run check-gemm-identity
    tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo
    pixi run mojo run -I . glm/ols_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_tuned_probe.mojo
    tools/with_identical_mode.sh pixi run check-batch-invariance
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_backward_check.mojo

The Apple column runs the software seam (`lib_hardware_ftz_fma_for` is False
there) and one shared page at KS=16 fits, so the tuned plans compile there
unchanged; the card must stay byte-equal.

## Next, in order, for a fresh agent

1. Rent an H100 (`NVIDIA H100 80GB HBM3`), ship `lane/gemm-identical` HEAD,
   `pixi install`, `pip install cupy-cuda12x`, then build and run under
   `-D MOJOLEARN_NUMERIC_IDENTICAL=1`: `gemm/checks/gemm_device_check.mojo`,
   `bench/gemm_card_main.mojo` (device arm, HOST_CAP=1, diff vs the known-good
   card), `gemm/checks/gemm_tuned_probe.mojo`, `bench/speed/gemm_speed_main.mojo`
   (arms v1 then core, 5+ rounds), `python3 tools/speed_gemm_arm.py --rounds 5`,
   the swap build (`-D MOJOLEARN_537_GEMM_IDENT_SWAP=1`, arm core),
   `core/gemm_identity_check.mojo` and `glm/ols_main.mojo` in both modes.
   Commit the logs under `bench/results/e1g/<stamp>-nvidia-h100-identical-gemm/`.
2. Task 3's second half: time OLS/PCA at 256 and 512 features identical vs
   `torch.linalg.lstsq` / covariance `eigh` on the same box (the estimators
   now run there; no driver for those widths exists yet, `bench/speed/classical_speed_main.mojo` is pinned to 32 columns).
3. Task 4 at estimator level: `bench/speed/classical_speed_main.mojo` lanes
   ols/pca/knn under IDENTICAL with and without the swap define (GEMM-level
   swap numbers come from step 1's `speed_swap` log).
4. Kernel work still open: TN rows (no vector loads, 128x128 loses on every
   tuned tile), the m < 32 rows (t1/t8 stay on the old plans), KS=32 and
   larger shared pages on NVIDIA/AMD, and an unpinned counterpart of the
   tuned plans for `gemm_unpinned.mojo`. AMD and Apple columns of the tuned
   plans are unmeasured; the hardware-ftz row is NVIDIA only by design.
