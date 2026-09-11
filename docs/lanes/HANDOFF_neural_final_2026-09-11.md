# Handoff: neural final optimization pass, 2026-09-11 (NVIDIA H100, wound down)

Andrew, about 20:10Z: one last optimization pass, measurements on RunPod
NVIDIA. About 20:30Z: no new lanes, let the running ones wind down, then a
handoff with next steps. About 20:50Z: wind the lanes down now. Trees and
classical ML belong to the other session (mojolearn-83, session bd222cd8) and
are not covered here. The full history of the day is
`docs/lanes/HANDOFF_neural_perf_2026-09-11.md` (items 0 to 0f).

The step-by-step plan to continue this work in another session, with the
kickoff prompt, commands and read-backs, is
`docs/lanes/PLAN_neural_final_continue_2026-09-11.md`.

## 1. Where the step stands

Byte LM training step at the target shape, lean step, steady median, RunPod
H100 80GB HBM3 at 1980 MHz, IDENTICAL (bits equal on Apple, NVIDIA and AMD).

| when (2026-09-11) | what shipped on NVIDIA | enwik8 s | Pile GitHub s |
|---|---|---|---|
| morning | device-owned step, `stash_tiled` attention | 0.383 | 0.380 |
| afternoon | GEMM `ksplit` (2595) plus attention round 3 (2534) | 0.295 | 0.295 |
| evening | attention `_kvgrid_r32` dk/dv (2597, word 52327) | 0.2919 | 0.2906 |

Shipped defaults now. NVIDIA is GEMM `ksplit(S=132) else tuned128` with
attention `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`. AMD is `ksplit(110)` with
the same attention word. Apple is the `tuned128` plan with `stash_tiled`.

Torch on the same pod (`bench/OPPONENT_REFERENCE.md`, 19:32Z section). Our step
takes 14.5x compile_bf16, 9.1x compile_tf32, 7.7x eager_tf32, 5.1x
compile_fp32 and 4.8x eager_fp32 on enwik8.

Where the time goes (step breakdown leg, DEVIATION 2630, previous attention
default, per step). GEMM 145 ms (48 percent), attention kernels 116 ms (39
percent), refusal and validation scans 6.4, RMSNorm forward 6.1, remainders
4.3, RMSNorm backward 4.0, AdamW 4.0, nothing else over 3 ms. Within attention
on the current default (enwik8, per step) zdot 66.4 ms, forward kernel 20.7,
dq 12.5, dk/dv 9.9. Within GEMM the ksplit fold launch is 13.3 ms.

Progress was slowing. The evening round of three lanes gave 1 percent, and six
of the last eight trial arms were flat or slower (2528, 2540-2544, 2598, 2599
`kpack` 1.24x and `kpack_wide` 1.40x, `_kvsplit`).

## 2. What landed on main today in this window

- 6d4bd867 zdot trial arms `_zdefer`, `_zlag` (DEVIATION 2598, trial only).
- e6ffb6f4 GEMM kernel trial arms `kpack`, `kpack_wide` (DEVIATION 2599, trial only).
- c8b9cba7 step phase timers and the H100 breakdown leg (DEVIATION 2630, timers only).
- 272011ae NVIDIA attention default flipped to `_kvgrid_r32` (lean step geomean 0.9908).
- c7632929 GEMM kernel leg, both arms NO FLIP, brief section 12.
- 4bce8278 same-pod confirmation of the flip plus every torch column, handoff item 0f.

## 3. The final pass lanes (nothing built, nothing measured)

Each lane read, designed and wrote code on its own pushed branch. By rule,
lane code is never built by the lane; the orchestrator builds on the M4 and
runs H100 legs. All three were told to stop before any of that happened, so
none of the estimates below is measured. None of the branches is merged.

### 3.1 GEMM, ARMS WRITTEN (`lane/gemm-final-h100`, f470a7aa, brief 05f5bf9c, DEVIATIONS 2640 and 2641)

Brief `docs/lanes/BRIEF_gemm_final_2026-09-11.md` (section 6 lists what is
unfinished).

- **The reading.** The ksplit group launches already run at the fastest
  schedule 132 blocks side by side allow on every call except head_dA (320
  leaf-rounds where 288 would do). The fold launch (13.3 ms per step) is
  outside that limit. It runs one thread per output cell with a 12-level stack,
  and every retained fold line fits `8 µs + ceil(cells / 270,336) × t(G)`.
  Seven groups with serial loads cost the same per round as eight batched, which
  reads as per-thread push work setting the fold time rather than load latency.
  That is a fit, not a measurement.
- **`kfoldv` (2640).** The shipped group launch and rule unchanged; the group
  nodes are folded 16 cells per thread through a lane-wise 8-level register
  stack, about 5x fewer instructions per cell by source count.
- **`kfoldv_leaf` (2641).** The same fold under `ksplit_leaf`'s finer rule, to get
  the one-leaf walk rate and head_dA's measured 1.33 ms back.
- **Expected saving.** `kfoldv` 9.3 ms (3.1 percent) if the fold is compute
  bound, 5.8 ms at half that gain, 0 if memory traffic binds it. `kfoldv_leaf`
  +14.1 ms if compute bound, a 2.2 ms loss if traffic binds it. The leg's PHASE
  `fold_ms` separates the two cases.
- **Written.** Both arms in `gemm/checks/gemm_identical.mojo`, the arms check
  (`gemm/checks/gemm_step_arms_check.mojo`), the price harness
  (`bench/gemm_step_price_main.mojo`), `tools/gemm_final_leg.sh`.
- **Not written.** The host check that the lane fold equals
  `_fold_push`/`_fold_drain` for G from 1 to 255, the host check of the rule
  hand counts, resources rows for both fold kernels, and docstring paragraphs.
  The device ragged check still holds the arms' bits against the old plan.

### 3.2 Attention, DESIGN ONLY (`lane/attention-final-h100`, 2316320f, DEVIATIONS 2650 to 2652)

Brief `docs/lanes/BRIEF_attention_step_2026-09-11.md` section 20 (20.1 ranking,
20.2 reading, 20.3 mechanism, 20.4 saving, 20.5 identity argument, 20.8 M4
commands, 20.9 H100 leg, 20.10 risks). No code.

- **The reading.** The H100 fits `floor(256 / pad8(regs))` 256-thread blocks per
  SM, and all 11 resource readbacks on file agree. The shipped zdot kernel
  reads 134 registers, so it gets one block per SM, 6 registers short of two.
  Tiling (2528) and deferral (2598) never changed that count, which is why they
  did not move zdot. `_kvgrid_r32` went from 2 to 4 blocks per SM, and dk/dv
  moved.
- **The mechanism.** `_estash` (2650) keeps the exp stash the forward already
  computes and frees, so the backward takes y from one divide with no score
  chain, no exp and no K staging, 8 rows per block instead of 4.
  `_estash_dres` (2651) also moves dctx rows to shared memory to free 64
  registers per thread. 2652 is the step plumbing (one host field on
  `LlamaDeviceStages`, trial builds only).
- **Expected saving.** zdot 66.4 to about 33 ms, about 33 ms per step (11
  percent). About 50 ms if registers fall to 128 or fewer. At least 11 ms if the
  stash stores dominate. Costs 2.42 GB of device memory held for the stash.

### 3.3 Step glue, HALF WRITTEN (`lane/step-glue-h100`, b3892ba8, DEVIATIONS 2645 to 2647)

Brief `docs/lanes/BRIEF_step_glue_2026-09-11.md` (section 3 lists what is not
built; sections 4 and 5 the identity and refusal arguments; 7 the leg; 8 the M4
commands; 9 risks).

- **The 6.4 against 3.4 ms question.** Two different sets of scans. The declined
  3.4 ms was the attention launchers' regime scans (still 3.32 ms); the 6.4 ms
  is the step's own refusal scans. Nothing regressed.
- **`rows` (2645).** RMSNorm row kernels launch 128 threads per block, so 2,048
  rows are 16 blocks on 132 SMs; the traffic model predicts 263 µs for the
  forward norm and the box measured 256 µs. 128 blocks would save about 7 ms
  per step, or nothing if the runtime already packs several blocks per SM.
- **`optskip` (2646).** The optimizer's four entry scans can never fire on this
  step (the gradient scan runs just before, and param, m and v were validated
  when written). 2.18 ms.
- **`noshadow` (2647).** AdamW writes the new state into the shadow buffers and
  the step swaps handles, so the shadow copy disappears with rollback and
  refusal unchanged. 2.31 ms.
- **Total.** About 11.5 ms (3.9 percent), or 4.5 ms if `rows` does nothing.
- **Written.** The brief, `core/step_glue.mojo` (define, arm names, sabotage),
  `adam_update_oop_kernel` in `training/checks/optimizer.mojo` (uncalled),
  `byte_lm_step_glue_arm_binding` (not registered), trainer metadata fields,
  `tools/step_glue_leg.sh`.
- **Not written.** The `rows` branches in `llama_rms_norm` and `bwd_rms_norm`,
  `_byte_glue_update` and its branch in `_byte_step_device`, the binding
  registration line, the probe's `result.json` fields,
  `training/checks/step_glue_check.mojo`.

The three branches touch different files (GEMM in `gemm/`, attention in
`transformer/impl/llama/fused_attention.mojo` and the step stages, glue in
`core/`, `training/` and the RMSNorm functions), so they can be merged in any
order, but each merge still needs the M4 gates on the merged tree.

## 4. Next steps, in order

Together the three designs estimate about 20 to 55 ms off the 290 ms step if
their readings hold. That would still leave our step at roughly 3.5 to 4x torch
eager fp32 and 11 to 13x compile bf16. The fixed cost of identity remains.

1. **GEMM `kfoldv`, the most ready.** The arms are written. Run the M4 gates
   one at a time from a worktree of f470a7aa:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-final-check`,
   then `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-final-check`
   (expect `REACH ragged` N/N for both new geometries), the same build without
   the trial define (expect FAIL naming the define), and `gemm_device_check`
   green. Then the Apple card (`tools/gemm_card.sh device /tmp/gemm-final-apple.card`)
   and the H100 leg with `MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_final_leg.sh`
   (about 25 minutes). Watch for a spilled 128-wide register stack in the
   resources lines, and read `fold_ms` to see whether the fold is compute or
   traffic bound. The unfinished host checks in 3.1 should land before any
   flip, not before the leg.
2. **Attention `_estash`, the largest.** Before writing the kernel, check the
   register count the resources readback will report for the new zdot kernel,
   because the whole saving depends on blocks per SM. Then write brief 20.6
   (kernel, launchers, step plumbing, arms check with reach and sabotage, fused
   check, harness fields, `tools/attention_final_leg.sh`), run the 20.8 M4
   gates, and the 20.9 H100 leg (about 20 to 25 minutes). Check memory headroom
   for the 2.42 GB stash on each column before any flip.
3. **Step glue `optskip` and `noshadow`.** Host-side logic with a written
   refusal argument, about 4.5 ms together. Finish the unwritten items in 3.3,
   write `step_glue_check.mojo`, gate on the M4 (brief section 8), run
   `tools/step_glue_leg.sh` (about 30 minutes). Read `adam_update_oop_kernel`
   line by line against the shipped update, because a random fixture cannot
   catch a fused-versus-unfused slip in its last line; the leg's per-step
   witnesses are the real check.
4. **Step glue `rows`.** Only a box says whether 16-thread blocks already spread
   across SMs. The resources readbacks from steps 1 and 2 bear on it.
5. **After any flip.** A flip changes the NVIDIA row only. Run the shipped fused
   check and arms check on the M4, a same-pod H100 confirmation with every torch
   column (as 4bce8278 did), and add the row to `bench/OPPONENT_REFERENCE.md`.
6. **If none of 1 to 3 flips, stop tuning the NVIDIA step** and record the
   current number as the price of identity.
7. **Owed, not speed.** The AMD shipped-binding confirmation of `_kvgrid_r32`
   (blocked by the NVIDIA-only order); a clean 1x MI300X torch ROCm row; branch
   `lane/cpu-speed` (byte LM CPU host kernels, Claude-Session
   01HKT67ojRtb5zcS3ryzVXJ2, owner unknown) reuses DEVIATION 2624, which main
   already holds for the pointwise fix, and must renumber before it merges.

## 5. How this work runs (binding)

- **Measurement.** RunPod NVIDIA H100 only, and only same-pod ratios. H100 pods
  come at 1980 MHz and 1590 MHz, about 20 percent apart.
- **Flip rule** (ENGINEERING_RULES 9). The geomean of the enwik8 and Pile GitHub
  lean step ratios below 1 on one pod, with every step witness equal.
- **Builds.** Lanes never build or test on the Mac. The orchestrator runs M4
  gates one at a time with
  `pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml --frozen`
  from the worktree. Byte LM binding builds are too heavy for the Mac and
  `tools/macos_serial_guard.py` admits tiny jobs only, so they are proven on
  the GPU box.
- **Arms.** Trial only behind `MOJOLEARN_ATTN_ARM_TRIAL`,
  `MOJOLEARN_GEMM_ARM_TRIAL` or `MOJOLEARN_STEP_GLUE_TRIAL`; shipped kernels
  unchanged until a flip. Every GPU SIMD width must be a power of two (a width-12
  register broke the 2599 build).
- **Legs.** `tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60
  --gpu "NVIDIA H100 80GB HBM3" --local-card <apple card>` with
  `MOJOLEARN_GEMM_LEG_EXTRA=<body>` and `MOJOLEARN_GPU_ARCHS=sm_90a`. It ships a
  `git archive` of local HEAD, has no extra-env plumbing (bodies set their own
  env), and must end with the pod verified gone (HTTP 404).
- **Evidence.** Files over 900 KB (operand dumps) stay in `~/mojolearn-evidence/`
  under the leg name.
- **DEVIATION numbers.** Neural holds 2640 to 2659; trees and classical hold 2625
  to 2629 and 2631 to 2639.

## 6. Evidence from this window

- `bench/results/e1g/2026-09-11_185833-nvidia-h100-80gb-hbm3-attention-zdot`
- `bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown`
- `bench/results/e1g/2026-09-11_191151-nvidia-h100-80gb-hbm3-gemm-kernel`
- `bench/results/e1g/2026-09-11_193203-nvidia-h100-80gb-hbm3-kv-default-torch`
