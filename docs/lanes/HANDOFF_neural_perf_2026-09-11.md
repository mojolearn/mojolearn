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
- Two benchmark corpora of different kind for every number (replaced
  2026-09-11 night, ENGINEERING_RULES 9): English text
  (training/corpus/enwik8, 100 MB, tools/fetch_corpus_enwik8.sh) and
  source code (training/corpus/pile_github, 97 MB of the Pile's GitHub
  component, tools/fetch_corpus_pile_github.sh), both pinned by sha256 and
  fetched on the box. The H100 attention numbers above were measured on the
  retired pair (tinyshakespeare, cpython312_lib). Step timing does not
  depend on corpus size (each step reads 2,048 bytes).
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

0. DONE 2026-09-11 night (main 7c0a5af4, resumed session). Andrew's
   corpus decision ("we should take 2 corpora that generalize and that have
   benchmarks"): the two neural corpora are now enwik8 and the Pile's
   GitHub component (ENGINEERING_RULES 9, training/corpus/enwik8 and
   training/corpus/pile_github, fetch scripts pin both sha256);
   tinyshakespeare and cpython312_lib are retired as timing corpora.
   `tools/do_extra_leg.sh` now takes the shared lock itself and passes
   MOJOLEARN_DO_EXTRA_ENV knobs to the body. `tools/attention_step_leg.sh`
   is vendor-agnostic (steps 1 and 2 below are done except the paid run).
   Dry runs green; torch harness Mac smoke green on both new corpora.
   Lock order agreed with the trees and classical session (mojolearn-83):
   release-0.8.1 holder, their amd-trees leg, our attention leg, their
   GBDT A/B, our torch leg; wait 180 s after our own leg before retrying.
   Lanes launched in worktrees, code only, nothing built: attention
   DEVIATIONS 2528, 2531, 2533, 2530 (section 11 order) and GEMM
   DEVIATIONS 2540 to 2544. AMD leg 1 invocation:
   `MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_ATTN_LEG_ARMS=stash_tiled MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1" MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-amd-mi325x-attention-step bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates`.
0b. MERGED 2026-09-11 night after orchestrator M4 gates (lane code is never
   compiled by the lanes): DEVIATION 2528 zdot tiled, trial arms
   `stash_tiled_ztiled_r64` / `_r32` (25a0356c; transformer_fused_check
   PASS, arms check PASS with sabotage_new moving only the backward; the
   parser needed a String aliasing fix before it compiled), and GEMM
   DEVIATIONS 2540 to 2544, trial only (63e9077f; step arms check PASS on
   102 ragged cases x 6 geometries with reach 102/102, the no-trial build
   fails naming the define, gemm_device_check green). Defaults unchanged.
   2531, 2533 and 2530 are not built. AMD leg 1 (attention, from commit
   63325cf7) waits behind amd-trees-leg through a launcher; its env prices
   stash_tiled and both 2528 row counts against baseline and runs the LM
   step for baseline and stash_tiled on both corpora. Next AMD legs, in
   the agreed alternation: the torch opponent row, then
   `MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_step_leg.sh MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,lfold,half,half_ks16,quarter,head,half_head MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=auto" MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-amd-mi325x-gemm-step bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates`,
   then the 2528 LM step leg against stash_tiled (brief section 12).
0c. NVIDIA FIRST, 2026-09-11 ~13:30Z. Andrew: "use runpod then and just do
   nvidia for now" while the trees lane held the DigitalOcean GPU. The AMD
   attention launcher was stopped before it took the lock and mojolearn-83
   was told neural yields its AMD turns until neural messages again. Two
   concurrent RunPod H100 legs at commit cd086f67, both pods terminated and
   verified, both device cards IDENTICAL to the M4 card from the same commit:
   - Attention plus torch, one pod
     (bench/results/e1g/2026-09-11_133041-nvidia-h100-attention-torch; attention
     brief section 13). `stash_tiled` holds on the benchmark corpora: lean step
     0.5625 -> 0.3835 s (enwik8), 0.5628 -> 0.3833 s (Pile GitHub), witnesses
     equal. DEVIATION 2528 NO FLIP on NVIDIA: r64 step 1.084x stash_tiled on
     both corpora (y/dy kernel 115.5 + fold 6.6 ms against the stash read's
     89.9 ms); r32 prices worse than r64.
   - GEMM step arms (bench/results/e1g/2026-09-11_133216-nvidia-h100-gemm-step;
     GEMM brief 10.7). NO FLIP: quarter step 1.128x shipped, every arm slower
     in price (quarter 1.27 to lfold 2.86), every arm still one block per SM
     at 214 to 254 registers. The step check passed on the H100 including the
     three vocab-sized head calls.
   - The first torch row (bench/OPPONENT_REFERENCE.md, H100 section). Same pod,
     same shape, same corpora: our IDENTICAL step takes 10.1x torch eager TF32's
     time (0.383 s against 0.038 s), 6.7x compile fp32's and 6.2x eager fp32's.
     Compile with TF32 and bf16 autocast were not measured.
   What that says about where to work next: the GEMM price harness puts the
   twelve GEMM call kinds at about 219 ms of the 384 ms step (the three vocab
   calls alone about 55 ms) and the attention timers put attention at about
   165 ms, so the matrix multiplies are the larger share against a whole torch
   step of 38 ms. Still owed: the AMD legs (attention 12.7 second leg for 2528
   at 32 rows, GEMM 10.5 for the first AMD register readback, the torch ROCm
   row) whenever neural takes an AMD turn again, and a shipped non-trial
   binding probe to explain the GEMM leg's 0.457 s shipped step (GEMM brief
   10.7). `tools/gemm_remote_leg.sh` has no extra-env plumbing; the attention
   settings rode in a wrapper body the leg copied to `extra_body.sh`.
0d. WOUND DOWN 2026-09-11 ~14:35Z on Andrew's order ("please tell all lanes to
   wind down quickly"). Nothing got faster after 0c. State:
   - Providers. Andrew: neural legs run on NVIDIA on RunPod ("don't attempt amd
     JUST DO NVIDIA ON FUCKING RUNPOD"); then Hot Aisle AMD was set up for new
     legs. RunPod had no AMD MI300X stock (create: "There are no instances
     currently available") and briefly no H100 80GB HBM3 stock either.
     Hot Aisle: SSH to admin.hotaisle.app accepts ~/.ssh/id_ed25519; the API
     key is NOT on disk yet (expected at ~/.mojolearn_hotaisle_key, mode 600,
     created in the admin TUI under personal settings). Base
     https://admin.hotaisle.app/api/, header `Authorization: Token <key>`,
     spec admin.hotaisle.app/api/docs/swagger.json, billed per minute, GPU
     MI300X. Opponent rows measured there are a new tuple.
   - Trial-binding overhead, PARTIAL
     (bench/results/e1g/2026-09-11_141443-nvidia-h100-trial-overhead; operand
     dumps outside the repo). The attention trial binding's shipped
     `stash_tiled` step is 0.3801 / 0.3799 s on this third H100 pod. The GEMM
     half ran its check and shipped price but no LM probe:
     `tools/gemm_step_leg.sh` drops `shipped` from LM_ARMS (it is always the
     bracket), so `LM_ARMS=shipped` resolves to none. Use
     `MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=quarter` to get the shipped bracket on
     the GEMM trial binding (scratchpad body nvidia_trial_overhead_body2.sh
     did; its pod f8cgynz7rl2h99 was reaped at wind-down before any work and
     verified gone). Whether the GEMM trial build costs the 0.457 s against
     0.380 s is still OPEN.
   - GEMM long inner dimension lane (DEVIATIONS 2590 to 2594), NOT MERGED,
     branch `lane/gemm-long-k` (48aa0325): brief
     docs/lanes/BRIEF_gemm_long_k_2026-09-11.md with the reading (proj_dB at
     2.7x proj_fwd's time for equal flops; a fitted model of one block per SM
     over 132 SMs, not measured), the `ksplit` / `ksplit_leaf` design and its
     identity argument. Code on the branch is a kernel matrix row and control
     shapes only, never compiled; the arm itself is NOT BUILT. Its predicted
     GEMM sum of about 162 ms (from 219) is a model, not a measurement.
1. The DigitalOcean runner is merged and its dry run is green (section 3);
   its first paid run is also its bring-up. Tuning on AMD is now a repo rule
   for every lane (ENGINEERING_RULES.md section 10), and the account allows
   one GPU droplet at a time, shared with the trees and classical lanes:
   every DigitalOcean GPU leg takes `mkdir /tmp/mojolearn-do-gpu.lock`
   (owner file inside with lane name and UTC time) before the create and
   removes it only after the destroy is verified; a lock older than 100
   minutes with zero droplets live may be broken. `tools/do_extra_leg.sh`
   takes this lock itself since main 7c0a5af4. Rerun the dry run from the
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
