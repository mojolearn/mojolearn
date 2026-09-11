# Handoff: neural IDENTICAL training speed, 2026-09-11 (wound down on Andrew's order)

Written by the orchestrator session that took over the neural performance
lane on Sep 11. Andrew asked for a gentle wind-down with a plan, next steps
and recommendations. Nothing is renting. Read this file, then
docs/lanes/BRIEF_attention_step_2026-09-11.md sections 10 and 11, then
ENGINEERING_RULES.md section 9, before touching the lane.

## 1. State at wind-down

- main at the commit that adds this file (parents are in `git log`).
- No DigitalOcean droplet exists. No RunPod pod of this lane exists (the
  `samba-train-*` pod belongs to another session; never touch it).
- RunPod balance about $399; DigitalOcean bills the card on file
  automatically (about $41 of prepaid credit left), no top-up needed.

## 2. What landed today

| item | commit on main | result |
|---|---|---|
| attention `stash_tiled` default (DEVIATIONS 2525 to 2527) | a70a96e0 | H100: lean target step 0.5619 -> 0.3829 s (English text), 0.5593 -> 0.3804 s (source code), every step witness equal; price fwd+bwd 28.5 -> 13.6 ms on both corpora; evidence bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step |
| torch opponent harness (`tools/torch_lm_step_opponent.py`, `_leg.sh`) | 1da28b6d | vendor-detecting; ROCm torch pinned to the 2.6.0+rocm6.4.1 cp312 wheels used on the MI325X Sep 7; Mac CPU control-shape smoke passed on both corpora, eager and compile columns (TF32 exits 4, not applicable, by design); NOT YET RUN ON ANY GPU |
| attention AMD reading and order of attack | 9e2e7323 | brief section 11; design only, no kernels |

Declined on measurement (do not reopen without a new mechanism):

- DEVIATION 2529 persistent attention scratch: allocation is 0.14 ms per
  direction per step.
- One fused regime scan and deferred flag readback: scans plus corner flags
  are 3.3 ms per step.

## 3. Branches not merged

| lane | branch | commit | state |
|---|---|---|---|
| GEMM occupancy and head arms (DEVIATIONS 2540 to 2544) | MERGED to main | 8c2a1922 (branch tip 722162c1) | design only: docs/lanes/BRIEF_gemm_step_2026-09-11.md with the identity argument per arm; no hook, kernel, check, price harness or leg exists; nothing built or run |
| DigitalOcean extra-body leg runner (`tools/do_extra_leg.sh`) | MERGED to main | aeb4dd90 (branch tip 93786fd8) | Mac `--dry-run` GREEN (bundle 9,512,631 bytes, no token, no API call); never run against a droplet, so the first paid run is also its bring-up; see step 1 of section 6 |

## 4. Decisions Andrew made today (binding)

- Tune neural IDENTICAL speed on AMD (Instinct MI325X, DigitalOcean tor1,
  build target `gfx942`). NVIDIA H100 is the confirmation column. His
  reason: everyone else tunes to NVIDIA. IDENTICAL keeps the bits equal on
  every vendor, so this costs no correctness. Geometry that differs by
  vendor is a kernel matrix row, never an inline vendor branch.
- Two ordinary corpora of different kind for every number: English text
  (training/corpus/tinyshakespeare, committed) and source code
  (training/corpus/cpython312_lib, fetched by
  tools/fetch_corpus_cpython312_lib.sh, pinned by sha256). Step timing does
  not depend on corpus size (each step reads 2,048 bytes).
- Flip rule (ENGINEERING_RULES 9): the geometric mean of the two
  after/before step time ratios below 1, bits equal on both, flips the
  default in the same session without asking.

## 5. Where the step's time is now (H100, default `stash_tiled`, per step)

| line | ms |
|---|---:|
| whole native call | 382 |
| attn.bwd_zdot_stash | 89.4 |
| head GEMMs, forward and backward (2048 x 768 x 50257) | about 46 |
| bwd.after_attention (o_proj and qkv backward GEMMs) | 40.5 |
| attn.fwd_sstash_kernel | 36.8 |
| block.mlp_and_residuals | 26.4 |
| attn.bwd_dkdv_tiled + attn.bwd_dq_tiled | 33.9 |
| attn.qkv_proj | 10.0 |

No AMD timing of this step exists at the target shape.

## 6. Plan, next steps in order

1. The DigitalOcean runner is merged and its dry run is green (section 3);
   its first paid run is also its bring-up. Tuning on AMD is now a repo rule
   for every lane (ENGINEERING_RULES.md section 10), and the account allows
   one GPU droplet at a time, shared with the trees and classical lanes:
   every DigitalOcean GPU leg takes `mkdir /tmp/mojolearn-do-gpu.lock`
   (owner file inside with lane name and UTC time) before the create and
   removes it only after the destroy is verified; a lock older than 100
   minutes with zero droplets live may be broken. `tools/do_extra_leg.sh`
   does NOT take this lock yet (its preflight only refuses while a GPU or
   mojolearn droplet exists); add the lock to it, or take it by hand around
   the run, before its first paid run. Rerun the dry run from the
   clean checkout you launch from:
   `MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh bash tools/do_extra_leg.sh amd --dry-run`.
   The real command (from `git worktree add --detach`):
   `MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_GEMM_LEG_EXTRA=<leg sh> MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-amd-mi325x-<lane> bash tools/do_extra_leg.sh amd --minutes 60`.
   A 60-minute lease is tight: the 9.5 MB bundle takes about 5 minutes over
   the Mac uplink before bring-up, and the attention leg took about 12
   minutes of body on the H100 after its builds; consider `--skip-gates`.
   Unknown until the box: whether image 188571990 has pixi (the body
   installs it if not) and how long HIP builds take. The runner writes the
   rocm-smi product name to `gpu.txt` and `device.txt`; the attention leg's
   `gpu_before.csv` and `gpu_after.csv` will hold nvidia-smi errors on AMD
   until step 2 lands.
2. Make tools/attention_step_leg.sh vendor-agnostic. It calls nvidia-smi and
   has a CUDA-only assembler block (brief section 11), so it does not run on
   AMD unmodified. Rehearse every command line it runs on the Mac at the
   control shape.
3. AMD leg 1, attention: price `stash_tiled` against `baseline` on the
   MI325X on both corpora's real activations, then the lean step for both
   arms with witnesses. The default was flipped on H100 evidence only; if
   it loses on AMD, the AMD default becomes a kernel matrix row. Record the
   device's real properties (core count, page limits) the brief could only
   transcribe from MI250X numbers.
4. AMD leg 2, opponent: `tools/torch_lm_step_opponent_leg.sh` on the MI325X
   through the runner (`MOJOLEARN_GEMM_LEG_EXTRA=tools/torch_lm_step_opponent_leg.sh`,
   output defaults to /root/gemm_leg_out/torch-lm-step, which the runner
   fetches; it installs the pinned ROCm torch into a throwaway venv if the
   box's torch differs) (eager FP32 is the row; compile is an extra column and may need
   `MOJOLEARN_TORCH_LM_DEADLINE` above 300 s on ROCm). Add the row to
   bench/OPPONENT_REFERENCE.md (item 5 today) with GPU, ROCm, torch and
   evidence path. Until this row exists there is no ratio against torch.
5. Attention arms in the order of brief section 11: DEVIATION 2528 tiled
   zdot (price the 32-row AMD variant and the 64-row one), 2531 32-row
   forward grid, 2533 preflushed seams, 2530 forward Q residency at 32 rows
   only. 2532 (keep y for the backward) needs the callers' stage structs and
   2.42 GB of device memory; out of this lane's files.
6. GEMM arms per docs/lanes/BRIEF_gemm_step_2026-09-11.md. Every GEMM in
   the step reaches `identical_gemm_into` (gemm/checks/gemm_identical.mojo),
   so one trial hook covers the head, block forward and block backward
   calls. Its model (fitted to H100 timers, not measured): per-layer calls
   are block-count bound (36 to 96 blocks against 132 SMs) and the shipped
   kernel fits one block per SM at 255 registers. Arm A (2540: `lfold`,
   `half`, `half_ks16`, `quarter`) folds leaf cells through thread-local
   memory to cut registers; arm B (2541: `head`, `half_head`) tiles the
   three vocab-sized calls along the long axis. To build: the hook and
   `MOJOLEARN_GEMM_ARM` selector (2542), the kernel, check, price and
   resource harnesses (2543), a vendor-agnostic tools/gemm_step_leg.sh and a
   `gemm_arm` probe field (2544). Register counts per arm and all AMD
   occupancy facts are unknown until a box reads them back; ptxas is an
   NVIDIA-only instrument. First M4 command once the hook exists:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_tuned_probe.mojo -o /tmp/gemm_shipped_probe`.
7. NVIDIA confirmation: the torch H100 row
   (`tools/torch_lm_step_opponent_leg.sh` through tools/gemm_remote_leg.sh)
   and one H100 leg for every arm that flips on AMD.
8. After any flip: the shipped fused check and the arms check on the M4
   (they caught two 32 KB shared-memory bugs on Sep 9).

## 7. Recommendations

- Cost. DigitalOcean is the expensive AMD host: MI325X $3.80/hr, H100
  $4.41/hr there, against RunPod H100 $2.69 to $3.49/hr. Hot Aisle lists
  MI300X at $1.99/hr billed per minute and TensorWave MI325X near $2.25/hr
  (quote based). If AMD legs become routine, an account there roughly halves
  every leg; the runner would need a provider backend. Decide after AMD
  leg 1 shows the dollars per step.
- DigitalOcean allows one GPU droplet at a time on this account, so AMD
  legs serialize. Ask DigitalOcean support to raise the GPU quota if
  parallel legs are wanted.
- One orchestrator per lane. On Sep 11 two sessions acted on the same leg
  and nearly duplicated a flip; before acting on a leg another session
  launched, check `git status` of the owned files and `ListAgents`, then
  message the owner.
- Rehearse the harness, not the new code, before renting: every command
  line the leg runs, on the Mac, at the control shape. macOS has no
  `timeout` binary; smoke without it.
- Do not quote the 1.47x as a speed claim against anyone. It is our
  before and after on one vendor.

## 8. Evidence and commands

- Attention leg: bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step
  (operand dumps outside the repo at
  ~/mojolearn-evidence/attention-step-2026-09-11_113013/).
- Torch Mac smoke:
  `nice -n 19 pixi run --frozen -e skgpu python tools/torch_lm_step_opponent.py --device cpu --shape control --corpus tinyshakespeare --column eager_fp32 --warmup 1 --steps 1 --out <scratch json>`
- H100 legs (RunPod, one-hour lease, key never exported):
  `MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_GEMM_LEG_EXTRA=<leg sh> MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-<lane> sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 --gpu "NVIDIA H100 80GB HBM3"`
  from a `git worktree add --detach` checkout.
- AMD legs: through the runner in section 3 with `MOJOLEARN_GPU_ARCHS=gfx942`.
