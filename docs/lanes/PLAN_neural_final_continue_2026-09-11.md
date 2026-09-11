# Plan: continue the neural final optimization pass (NVIDIA H100, IDENTICAL)

Written 2026-09-11 about 21:10Z by the neural session (mojolearn-df) for Andrew
to continue in another session. State and numbers are in
`docs/lanes/HANDOFF_neural_final_2026-09-11.md`; read it first. This file is the
operating plan: what to run, in what order, what to read back, and when to stop.

## 0. Kickoff prompt (paste into the new session)

```
Continue the neural final optimization pass in /Users/andrewhendel/CascadeProjects/mojolearn.
Read docs/lanes/PLAN_neural_final_continue_2026-09-11.md and
docs/lanes/HANDOFF_neural_final_2026-09-11.md, then execute the plan from phase 0.
Neural only (trees and classical belong to session mojolearn-83). Measurements on
RunPod NVIDIA H100 only, at most 2 pods at once. No heavy compute on the Mac;
M4 gates one at a time. Commit and push what passes; report honestly whether the
step got faster, with same-pod numbers.
```

## 1. Goal and stop rule

Shave time off the H100 byte LM training step (now 0.2919 s enwik8, 0.2906 s
Pile GitHub) with results still bit-equal on Apple, NVIDIA and AMD. Three
unmerged, unbuilt branches carry the candidates. Take them in order of
readiness, measure each on one H100 pod, flip only what wins by the flip rule.
If none of the three flips, stop tuning the NVIDIA step and record the current
number as the price of identity.

| order | branch | commit | DEVIATIONS | state | estimate per step |
|---|---|---|---|---|---|
| 1 | `lane/gemm-final-h100` | f470a7aa | 2640, 2641 | arms written, some host checks missing | 0 to 9.3 ms |
| 2 | `lane/attention-final-h100` | 2316320f | 2650 to 2652 | design only (brief section 20) | 11 to 50 ms |
| 3 | `lane/step-glue-h100` | b3892ba8 | 2645 to 2647 | half written | 4.5 to 11.5 ms |

## 2. Binding rules (from Andrew and the memory)

- RunPod NVIDIA H100 only; at most 2 neural pods at once (the trees and classical
  session runs its own). Every pod self-expires (the runner's 60 minute lease and
  dead-man) and every leg ends with the pod verified gone (HTTP 404).
- Never print, export or pass on argv the keys in `~/.mojolearn_runpod_key`,
  `~/.mojolearn_do_token`, `~/.mojolearn_hotaisle_key`. Pass the file path only
  (`MOJOLEARN_RUNPOD_KEY_FILE`). Never touch any `samba-train-*` pod.
- Same-pod ratios only. H100 pods come at 1980 MHz or 1590 MHz, about 20 percent
  apart (`gpu.txt` records `clocks.current.sm`).
- Flip rule (ENGINEERING_RULES 9): geomean of the enwik8 and Pile GitHub lean step
  ratios below 1 on one pod, every step witness equal. A flip changes the NVIDIA
  row only.
- No heavy compute on the Mac. Subagents never build or test locally; the
  orchestrator runs M4 gates one at a time. Byte LM binding builds only on the
  GPU box (`tools/macos_serial_guard.py` admits tiny jobs only).
- Git: never `git add -A`, never rewrite history, merges in separate worktrees
  from origin/main, push with fetch, merge and retry (main moves often).
  Never commit another session's uncommitted files in the shared checkout.
- Evidence files over 900 KB go to `~/mojolearn-evidence/<leg name>/`, never the
  repository.
- Neural DEVIATIONS are 2640 to 2659. Trees and classical hold 2625 to 2629 and
  2631 to 2639. Message `mojolearn-83` before editing `gemm/checks/gemm_identical.mojo`
  if their lanes might need it (they said they will not).
- Writing for Andrew: American English, never say "we are faster" (give ratios),
  no dashes and no colons in prose.

## 3. Phase 0, setup (10 minutes)

1. `git fetch origin` and confirm the three branch tips in section 1.
2. `ListAgents`; if `mojolearn-83` is live, tell it the neural pass resumed and
   how many pods you will use.
3. Recreate the two helper scripts below in the session scratchpad (they lived
   only in the old session's scratchpad).

### 3.1 `runpod_nvidia_leg.sh` (one H100 leg, SKU fallback only when no pod was created)

```bash
#!/usr/bin/env bash
#   bash runpod_nvidia_leg.sh <worktree> <expected commit> <body> <out suffix> <card>
set -u
W=$1; EXPECT=$2; BODY=$3; SUFFIX=$4; CARD=$5
LOG_DIR=$(dirname "$0")
stamp() { date -u +%H:%M:%SZ; }
head=$(git -C "$W" rev-parse HEAD)
[ "$head" = "$EXPECT" ] || { echo "REFUSED: worktree at $head, expected $EXPECT"; exit 2; }
[ -z "$(git -C "$W" status --porcelain --untracked-files=no)" ] || { echo "REFUSED: worktree is dirty"; exit 2; }
[ -f "$CARD" ] || { echo "REFUSED: no card $CARD"; exit 2; }
round=1
while [ "$round" -le 10 ]; do
  for g in "NVIDIA H100 80GB HBM3" "NVIDIA H100 NVL"; do
    tag=$(echo "$g" | tr 'A-Z ' 'a-z-' | sed 's/nvidia-//')
    OUTREL="bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-$tag-$SUFFIX"
    LOG="$LOG_DIR/nv_${SUFFIX}_r${round}_$tag.log"
    echo "LAUNCH $(stamp) round $round gpu '$g' commit $EXPECT body $BODY out $W/$OUTREL"
    (
      cd "$W" || exit 9
      MOJOLEARN_RUNPOD_KEY_FILE="$HOME/.mojolearn_runpod_key" \
      MOJOLEARN_GPU_ARCHS=sm_90a \
      MOJOLEARN_GEMM_LEG_EXTRA="$BODY" \
      MOJOLEARN_GEMM_LEG_OUT="$OUTREL" \
      sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "$g" --local-card "$CARD" --allow-concurrent
    ) > "$LOG" 2>&1
    rc=$?
    echo "RUNNER EXIT $rc $(stamp) (log $LOG)"
    if [ "$rc" != 0 ] && grep -q "create FAILED" "$LOG" && ! grep -q "pod .* created" "$LOG"; then
      continue
    fi
    grep -E "pod .* created|terminat|MAY STILL BE BILLING|leg done|remote_(extra|body)_exit|404" "$LOG" | tail -10
    echo "OUT $W/$OUTREL"
    exit "$rc"
  done
  echo "no stock on either SKU; retry in 120 s"; sleep 120; round=$((round + 1))
done
exit 1
```

Run it in the background (`run_in_background`), one call per leg. The body path
may be absolute. `tools/gemm_remote_leg.sh` ships a `git archive` of the
worktree's HEAD, so a local, unpushed merge commit can be legged. It exports no
extra env into the body, so a body sets its own env.

### 3.2 M4 gate helpers (bash, never zsh)

```bash
R=/Users/andrewhendel/CascadeProjects/mojolearn
M=(pixi run --manifest-path "$R/pixi.toml" --frozen)
"${M[@]}" bash -c pwd   # must print the worktree, not $R; refuse otherwise
bld() { n=$1; shift; nice -n 19 "${M[@]}" mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" -o "$T/$n" > "$T/build_$n.log" 2>&1; }
run() { n=$1; tag=$2; shift 2; env "$@" nice -n 19 "${M[@]}" "$T/$n" > "$T/run_$tag.log" 2>&1; }
```

One build at a time. `T` is a scratchpad directory per gate set.

## 4. Phase 1, GEMM `kfoldv` (about 1.5 hours with the leg)

Read `docs/lanes/BRIEF_gemm_final_2026-09-11.md` on the branch (section 6 lists
what is unfinished; the lane's gate and leg commands are in its report and brief).

1. **Merge worktree.** `git worktree add -b merge/gemm-final <scratch>/wt-gemm-final origin/lane/gemm-final-h100`,
   then `git merge --no-edit origin/main`. Expect a clean merge (only `gemm/`,
   `bench/gemm_step_price_main.mojo`, `tools/gemm_final_leg.sh`, the brief). If it
   conflicts in kernel code, resolve by hand and re-read both sides.
2. **M4 gates on the merged tree**, in this order:
   - `bld check -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo`
     then `run check check MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000`.
     Expect PASS and `REACH ragged` N/N for `kfoldv` and `kfoldv_leaf`.
   - `bld notrial -I . gemm/checks/gemm_step_arms_check.mojo` then the same run. Expect a
     FAILURE whose lines name `MOJOLEARN_GEMM_ARM_TRIAL`.
   - `bld dev -I . gemm/checks/gemm_device_check.mojo` then `run dev dev`. Expect
     `all green [IDENTICAL]`.
   - `bld price -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo` and
     `bld resources -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo` (build only).
   - If a build fails on a SIMD width, look for a comptime width that is not a
     power of two (the 2599 failure was `simd<12>`); the lane flagged a 128-wide
     `mut` register stack argument and 16-wide global loads as untested.
3. **Commit the merge** (explicit paths if you fixed anything), push to main with
   fetch and retry. Trial-only code may merge after green M4 gates, as 2598 and
   2599 did.
4. **Apple card** at that commit: `nice -n 19 "${M[@]}" sh tools/gemm_card.sh device <scratch>/apple_<commit>.card`.
5. **H100 leg.** Detached worktree at the pushed commit, then
   `bash <scratch>/runpod_nvidia_leg.sh <wt> <commit> tools/gemm_final_leg.sh gemm-final <scratch>/apple_<commit>.card`
   (about 25 minutes of body).
6. **Read back** from `<out>/remote/gemm-final/` (or the directory the body names):
   - `status.tsv` every item 0; `diff_apple_vs_nvidia.txt` says `RESULT: IDENTICAL`.
   - `lm_summary.tsv` is not a table: each line is `<run>\tkey=value...`. Read
     `witnesses_equal_baseline=True` for every arm and corpus, and the
     `verdict\t<arm>\t<FLIP|NO FLIP>\tgeomean=...` lines.
   - `price_step.txt` STEP lines (GEMM per step against shipped) and the PHASE
     lines' `fold_ms` for the arms against shipped. A fold that does not drop
     means memory traffic binds it, and `kfoldv` cannot win.
   - `resources_lines.txt` for the fold kernels: `regs` at or below 255,
     `local`, `blocks_per_sm_256`.
7. **Evidence.** Move files over 900 KB to `~/mojolearn-evidence/<leg name>/`,
   commit the leg directory and a results section in the brief (table like
   `BRIEF_gemm_kernel_2026-09-11.md` section 12), push.
8. **If an arm flips.** First finish the brief's section 6 items (the host check
   that the lane fold equals `_fold_push`/`_fold_drain` for G from 1 to 255, the
   rule hand counts, resources rows). Then one commit that makes the shipped
   NVIDIA dispatch take the winning fold (a NVIDIA-only row in
   `checks/kernel_matrix.mojo`, every other column off, trial arm kept), M4 gates
   again (no-trial `gemm_device_check`, trial step check), push, and phase 4.

## 5. Phase 2, attention `_estash` (code, then about 1.5 hours of gates and leg)

Everything is in `docs/lanes/BRIEF_attention_step_2026-09-11.md` section 20 on
`lane/attention-final-h100`. The code is not written. Starting a code lane for
it is Andrew's call; if a lane writes it, the lane never builds, and the
orchestrator gates.

1. **Decide first whether the saving is reachable.** The saving depends on the new
   zdot kernel leaving one block per SM (`floor(256 / pad8(regs))`, 20.2). Write
   the kernel and its resources readback before the step plumbing, and get the
   register count from one cheap H100 resources run if the brief's count by
   reading is close to a boundary (128 or 136).
2. **Write 20.6.** Kernel (DEVIATION 2650, and 2651 `_estash_dres`), launchers with
   `step_count_launch` before every launch and `step_count_sync` before every
   synchronize, the step plumbing (2652, one host field on `LlamaDeviceStages`,
   trial builds only), arms named on word 52327 and refused by default through
   `ATTN_ARM_DEFAULT_REFUSED_BITS`, reach and sabotage in
   `transformer/checks/transformer_attention_arms_check.mojo`, fused check and
   harness fields, `tools/attention_final_leg.sh` modeled on
   `tools/attention_zdot_leg.sh`.
3. **M4 gates** per 20.8: no-trial `transformer_fused_check.mojo` PASS with
   `DEFAULT column=apple arm=stash_tiled`, the `MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1`
   knob build PASS, trial arms check PASS (reach per branch), `gemm_device_check`
   green.
4. **H100 leg** per 20.9 with the helper script (about 20 to 25 minutes). The
   reference is the NVIDIA default word 52327.
5. **Read back** as in phase 1, plus the attention timers: zdot ms per step for
   `_estash` against 66.4, and device memory (the stash holds 2.42 GB).
6. **Flip** only on the flip rule, NVIDIA row only, after checking memory headroom.
   Apple and AMD stay on their words until measured there.

## 6. Phase 3, step glue (code, then about 1.5 hours)

`docs/lanes/BRIEF_step_glue_2026-09-11.md` on `lane/step-glue-h100`.

1. Merge origin/main into the branch first; if attention's step plumbing landed,
   expect conflicts in `transformer/impl/llama/modeling_llama.mojo` and
   `training/byte_lm.mojo`.
2. Finish section 3's list: the `rows` branches in `llama_rms_norm` and
   `bwd_rms_norm`, `_byte_glue_update` and its branch in `_byte_step_device`,
   the binding registration line, the probe's `result.json` fields, and
   `training/checks/step_glue_check.mojo` (section 6).
3. Read `adam_update_oop_kernel` line by line against the shipped AdamW update.
   A random fixture cannot catch a fused-versus-unfused slip in its last line;
   the leg's per-step witnesses are the real check.
4. M4 gates per section 8 (`-D MOJOLEARN_STEP_GLUE_TRIAL=1 -I . training/checks/step_glue_check.mojo`,
   plus the no-trial fused check and the optimizer check).
5. H100 leg with `tools/step_glue_leg.sh` (about 30 minutes). `optskip` and
   `noshadow` are the likely wins (4.5 ms); `rows` wins only if the leg's timers
   show the RMSNorm phases dropping (the runtime may already pack blocks per SM).

## 7. Phase 4, same-pod confirmation after any flip

After a flip commit is pushed, run one pod that proves the shipped build takes
the new default and re-measures every torch column on the same pod. The
attention confirmation body used for 272011ae, as a template (edit the arm names;
for a GEMM flip use `tools/gemm_step_leg.sh` with the old plan's trial arm as the
reference instead of `tools/attention_step_leg.sh`):

```sh
#!/bin/sh
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=<previous default arm> \
MOJOLEARN_ATTN_LEG_ARMS=<new default arm> \
MOJOLEARN_ATTN_LEG_LM_ARMS=<previous default arm>,<new default arm> \
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$t" = 0 ]
```

Read `gate.txt` (`shipped_check: DEFAULT column=nvidia arm=<new>`),
`lm_summary.tsv` (`arm_is_default=True` on the new arm, witnesses equal) and
`torch-lm-step/summary.tsv`. Add a same-pod section to `bench/OPPONENT_REFERENCE.md`
like the 19:32Z one (ours over each torch column, both corpora), move operand
dumps out of the repository, commit, push.

## 8. Phase 5, close

1. Update `docs/lanes/HANDOFF_neural_final_2026-09-11.md` with each phase's
   verdict and the new step numbers, and the memory file
   `mojolearn-neural-perf-lane-sep11.md`.
2. Tell Andrew plainly: the step before and after on one pod, what flipped, what
   did not, and the torch ratios. If nothing flipped, say the NVIDIA step
   tuning is done and the remaining gap is the price of identity.
3. Remove finished scratchpad worktrees (`git worktree remove`), and confirm no
   pod is left (`tools/gemm_remote_leg.sh` prints `VERIFIED ... gone (HTTP 404)`).

## 9. Pitfalls met today

- A lane's `worktree-agent-*` branch can stay at its base commit while the work
  sits on the named `lane/...` branch. Merge the named branch.
- `pixi run` from a worktree must resolve to that worktree (`bash -c pwd` check),
  or the gate builds the shared checkout.
- zsh does not split a command string; gate scripts are bash.
- A push can be rejected because another session pushed seconds earlier. Fetch,
  merge, push again; never force.
- `tools/gemm_step_leg.sh` drops `shipped` from its LM arm list; the shipped LM
  step is still measured as the reference (`lm-shipped-*`, `lm-shippedclose-*`).
- DEVIATION 2624 is taken on main (pointwise fix). Branch `lane/cpu-speed` (byte
  LM CPU host kernels, owner unknown) reuses it and must renumber before merging.
- A GPU kernel SIMD width that is not a power of two fails in the offload pass,
  on Apple and NVIDIA alike.
