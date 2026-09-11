# Step breakdown lane: a complete per-component timing of the shipped byte LM step (DEVIATION 2630)

Source-only lane, September 11, 2026, IDENTICAL only. STATUS: WRITTEN, NOT
BUILT, NOT RUN. Nothing here was compiled or run on the Mac or on a GPU.
The orchestrator's M4 build is the first compile (section 8), then one
NVIDIA H100 leg (section 7). DEVIATION 2630 is this lane's number; renumber
at merge if another lane took it.

Branch `lane/step-breakdown-h100` (from origin/main 68be1c79).

## 1. Why

The shipped byte LM training step on an H100 (RunPod) is 0.2950 / 0.2949 s
on enwik8 / Pile GitHub
(bench/results/e1g/2026-09-11_164101-nvidia-h100-80gb-hbm3-new-defaults-torch/remote/attention-step/lm_summary.tsv).
The next optimization targets have to come from a measured breakdown of
that step, with GEMM time measured INSIDE the step rather than estimated
from the separate price harness
(bench/results/e1g/2026-09-11_163310-nvidia-h100-80gb-hbm3-gemm-ksplit-default/remote/gemm-longk/price_tables.txt,
about 143 ms per step summed over the twelve call kinds).

## 2. What was already timed (read from the H100 default run, not new)

The run-time switch `MOJOLEARN_TRANSFORMER_TIMING=1` already prints
synchronized `timing <name> <ms> ms` lines from the Python wrapper, the
binding, the trainer, the loss, the optimizer and the blocks (DEVIATION
2499), and `-D MOJOLEARN_ATTN_PHASE_TIMERS=1` adds the fused attention's
per-kernel lines. Summing
`lmtiming-stash_tiled_fgrid_r32_qres_pf-enwik8/worker.log` by name:

| interval | ms per step | covered by |
|---|---:|---|
| `envelope.native_call` | 293.437 | the Python wrapper |
| `step.*` inside the envelope (head GEMMs 42.10, embedding, CE, AdamW, scans, copies, binding) | 58.977 | trainer, loss, optimizer, binding ticks |
| `envelope.blocks_forward` | 64.930 | `block.norm1` 3.690 + `block.attention_total` 36.740 + `block.mlp_and_residuals` 24.436 = 64.866 |
| `envelope.blocks_backward` | 169.213 | `bwd.before_attention` 0.035 + `bwd.attention` 93.213 + `bwd.after_attention` 23.108 = 116.356 |
| envelope minus all of the above | 0.317 | nothing |

So the "about 35 ms never measured" was mostly measured at the top level
already (0.3 ms of the envelope has no tick). What no tick brackets is
INSIDE the blocks:

- **52.857 ms per step in the backward blocks**: in
  `llama_decoder_layer_backward_device` the clock starts at the attention
  (`var tk` right before `bwd.before_attention`), so the refusal scan of
  the incoming gradient, the MLP backward (down, gate and up GEMMs, the
  gate product, SiLU), the post-attention RMSNorm backward and the o_proj
  backward print nothing, 12 layers times about 4.4 ms.
- `block.mlp_and_residuals` (24.4 ms), `block.norm1` (3.7 ms) and
  `bwd.after_attention` (23.1 ms) are single lines with GEMMs, norms, SiLU
  and residual adds inside.
- No GEMM inside the blocks has its own line, and nothing counts launches,
  synchronizes, copies or allocations.

## 3. What this lane adds

### 3.1 The switch

`core/step_phase.mojo` (new). Everything in it is compiled only under
`-D MOJOLEARN_STEP_PHASE_TIMERS=1` (`comptime STEP_PHASE_TIMERS`) and
switched on at run time by the same `MOJOLEARN_TRANSFORMER_TIMING=1`, the
`MOJOLEARN_ATTN_PHASE_TIMERS` pattern. It holds:

- `StepPhaseClock(ctx)`: `mark(ctx)` synchronizes and restarts;
  `tick(ctx, name[, gemm])` synchronizes, prints `timing <name> <ms> ms`,
  `timing launches.<name> <n> count` and `timing syncs.<name> <n> count`
  (launches and code synchronizes counted inside the interval), with
  `gemm` also `timing gemm.<kind> <ms> ms` and its two count lines for the
  same interval, then restarts AFTER printing (the print lands in the
  parent's remainder, not in a leaf).
- Seven counters in a `std.ffi._Global` slot (the trees, RF and byte LM
  bindings' pattern): launches, synchronizes, H2D, D2H and D2D copies,
  device and host buffer creations. `step_count_*()` increments one.
- `step_counts_now()` and `step_counts_report(start)`: the binding prints
  `timing count.launches`, `count.synchronizes`, `count.copies_h2d`,
  `count.copies_d2h`, `count.copies_d2d`, `count.device_allocs`,
  `count.host_allocs` and `count.native_steps 1` per resident step.

On a build without the define, every counter helper and every clock method
has an empty body, `step_phase_on()` returns False without reading the
environment, and the clock constructor stores three fields nothing reads.

### 3.2 The new timed intervals

Names follow the existing prefixes: `fwd.*` are sub-phases of the forward
block parents, `grad.*` of the backward parents, `gemm.*` re-labels GEMM
intervals by call kind (a cross-cut, never added), `bwd.mlp_through_oproj`
is a new parent beside `bwd.attention`.

| file, function | new lines (per layer unless noted) | parent |
|---|---|---|
| modeling_llama `llama_decoder_layer_forward_planted` | `fwd.refuse_call`, `fwd.norm1` | `block.norm1` |
| modeling_llama `llama_attention_forward` | `fwd.q_proj`, `fwd.k_proj`, `fwd.v_proj` (`gemm.proj_fwd` x3) | `attn.qkv_proj` |
| same | `fwd.o_proj` (`gemm.proj_fwd`) | `attn.o_proj` |
| modeling_llama layer forward and `llama_mlp_forward` | `fwd.residual1_add`, `fwd.norm2`, `fwd.gate_proj`, `fwd.up_proj` (`gemm.gateup_fwd` x2), `fwd.silu`, `fwd.gate_mul`, `fwd.down_proj` (`gemm.down_fwd`), `fwd.residual2_add` | `block.mlp_and_residuals` |
| transformer_backward `llama_decoder_layer_backward_device` | `bwd.mlp_through_oproj` (new parent, entry to the attention) | `envelope.blocks_backward` |
| same | `grad.refuse_scan`, `grad.residual2_copy`, `grad.down_dA` (`gemm.down_dA`), `grad.down_dB` (`gemm.down_dB`), `grad.gate_mul`, `grad.silu`, `grad.gate_dB`, `grad.up_dB` (`gemm.gateup_dB` x2), `grad.gate_dA`, `grad.up_dA` (`gemm.gateup_dA` x2), `grad.gateup_fanin`, `grad.norm2_kernels`, `grad.norm2_dW` (`gemm.norm_dW`), `grad.residual1_add`, `grad.o_copy`, `grad.o_dA` (`gemm.proj_dA`), `grad.o_dB` (`gemm.proj_dB`) | `bwd.mlp_through_oproj` |
| same | `grad.kv_slice`, `grad.rope`, `grad.q_dB`, `grad.k_dB`, `grad.v_dB` (`gemm.proj_dB` x3), `grad.q_dA`, `grad.k_dA`, `grad.v_dA` (`gemm.proj_dA` x3), `grad.qkv_fanin`, `grad.norm1_kernels`, `grad.norm1_dW` (`gemm.norm_dW`), `grad.x_add` | `bwd.after_attention` |
| transformer_backward `bwd_rms_norm[which]` | the `grad.norm{1,2}_kernels` and `grad.norm{1,2}_dW` ticks above (`which` is a new comptime parameter, default 0, naming the call site; `training/samba_ops.mojo` keeps the default and prints `grad.rmsnorm_*`) | the caller's parent |
| training/byte_lm `_byte_forward_loss`, `_byte_step_device` (per step) | `gemm.head_fwd`, `gemm.head_dA`, `gemm.head_dB` (inside the existing `step.head_forward`, `step.head_backward_da`, `step.head_backward_db`) | cross-cut |
| bindings/_mojolearn_byte_lm `byte_lm_session_step_binding` (per step) | the `count.*` lines | none |

The GEMM kinds are the price harness's twelve (`proj_fwd`, `proj_dA`,
`proj_dB`, `gateup_fwd`, `gateup_dA`, `gateup_dB`, `down_fwd`, `down_dA`,
`down_dB`, `head_fwd`, `head_dA`, `head_dB`) plus `norm_dW`, the RMSNorm
weight gradient (`identical_gemm` at `(1, dm, M)` OP_NN), which the harness
does not price. `lines_per_step` of a `gemm.*` name is its calls per step.

Already timed and kept as they are: embedding lookup and backward
(`step.embedding_forward`, `step.embedding_backward`), the logits softmax
and cross entropy with its refusal scan and backward (`step.ce_refuse_scan`,
`step.ce_forward`, `step.ce_backward`, `step.loss_download`), the AdamW
update (`step.opt_refuse_scan`, `step.shadow_copy`, `step.optimizer`), the
copies (`step.upload_inputs` H2D, `step.unpack_weights` and
`step.pack_grads` D2D, `step.loss_download` D2H), the scans and the
binding's admission, final wait and publish. The attention keeps its
`attn.*` lines.

### 3.3 The counters

Every `enqueue_function`, `enqueue_fill`, `synchronize`, `enqueue_copy`
(direction read from its keywords) and buffer creation in the step's files
carries a `step_count_*()` line before it: 418 lines in
`core/device_scan.mojo` (28), `embedding/checks/embedding_identical.mojo`
(12), `training/checks/loss.mojo` (17), `training/checks/optimizer.mojo`
(20), `training/checks/train_loop.mojo` (38),
`gemm/checks/gemm_identical.mojo` (28),
`transformer/impl/llama/fused_attention.mojo` (92),
`transformer/impl/llama/modeling_llama.mojo` (72),
`transformer/checks/transformer_backward.mojo` (83), `training/byte_lm.mojo`
(27) and the step binding's final wait (1). The timer helpers' own waits
(`timing_tick`, `_attn_tick`, `_step_timing_tick`, the operand dump) carry
none, so `syncs` counts the code's waits, not the timers'. Every site is in
a function that already `raises`; the patcher refused otherwise. A launch or
wait the MAX runtime makes internally is invisible here (section 9).

## 4. Why no computed bit can change

- The define is off in every shipped build. Without it the counter helpers
  have empty bodies, `step_phase_on()` is `return False`, and the clock's
  methods return immediately; no kernel, operand, geometry, launch order or
  buffer changes. `bwd_rms_norm` gained a comptime parameter that selects a
  timer name and nothing else; its body is unchanged.
- With the define, a tick is a `ctx.synchronize()` plus a host clock read
  and prints, and a counter is a host integer increment. A synchronize
  waits for work already enqueued on the context's in-order stream; it
  changes when the host observes completion, not what the device computed.
  The fused attention's phase timers and the block timers already
  synchronize the same way under the same switch, and the attention legs'
  witnesses were equal with them compiled in.
- No timer reads or writes a device buffer, and every buffer a clock
  outlives is already kept alive by the unchanged code
  (`[[mojo-buffer-freed-at-last-use]]` is not touched: no new `_ =` lines,
  no new buffers).
- The leg checks it on the box: the shipped build, the timers build with the
  switch off, and the timers build with the switch on each hash every step
  (loss, gradients, parameters, both moments, flags) and
  `witnesses.tsv` requires them equal per step (section 7).

## 5. The probe

`tools/lm_step_memory_probe.py`:

- `--component-timing-steps N` (default 1): the component timing runs N
  consecutive steps under the switch, batches continuing the schedule after
  `--steps`. Each name's lines split into N equal consecutive groups (every
  step prints the same lines in the same order; a name that does not is
  listed in `component_timing_uneven`), and `component_timing_ms`,
  `component_bytes` and the new `component_counts` are the MEDIAN per step.
  `component_timing_ms_per_step`, `component_timing_step_seconds_all` and
  `component_timing_steps` are new. For N = 1 every field keeps its old
  value, so `tools/attention_step_leg.sh`'s `lm_summary.tsv` reads the same.
- Under `--witness-every-step` the timed steps are witnessed too, outside
  their wall time and with the switch off for the exports
  (`component_timing_witnesses`).
- `fwd.`, `grad.` and `gemm.` join `envelope.` and `attn.` outside
  `component_timing_total_ms`.

`tools/step_breakdown_summary.py` (new) turns the runs into
`breakdown.tsv` and `witnesses.tsv`. The breakdown is the tree of section
3.2 rooted at `envelope.native_call`: each parent, its children, and
`remainder:<parent>` (parent minus children per step, then the median:
timer prints, host work between ticks, anything no tick brackets), so
leaves plus remainders telescope to the envelope and the top-level
remainder is the unattributed time. Then the leaves by category (GEMM,
embedding, RMSNorm forward, RMSNorm backward kernels, SwiGLU forward and
backward, residual and fan-in adds, rope and KV, attention kernels, refusal
and validation scans, softmax and cross entropy, AdamW, copies, binding,
remainders), the `gemm.*` kinds with calls and ms per call, the `count.*`
lines, the lean medians and the two timer overheads. Names the tree does not
reach print as `untreed` rows instead of vanishing.

## 6. What the numbers mean

The switch-on steps are a breakdown, never a price: every tick waits. The
price is the lean step. The leg runs three things per corpus on one pod:

| run | build | switch | what it says |
|---|---|---|---|
| `lean-shipped-<corpus>` | shipped | off | THE price (steady median of 3 steps after 1 warmup) |
| `lean-timers-<corpus>` | timers | off | minus shipped = what compiling the counters in costs |
| `timing-<corpus>` | timers | on | 1 warmup, then 3 timed steps: the tree; wall minus shipped = the switch-on overhead |

plus `lean-shipped2-enwik8` at the end (the shipped build again: pod
drift). Only same-pod numbers compare; `nvidia_smi.txt` and `host.txt`
(CPU model and cores; launch overhead is host work) name the pod.

## 7. The leg

`tools/step_breakdown_leg.sh`, a POSIX sh body for
`tools/gemm_remote_leg.sh`'s `MOJOLEARN_GEMM_LEG_EXTRA`. That runner exports
neither the column nor the arch, so the body reads the vendor from the box,
exports `MOJOLEARN_TARGET_COLUMN` (it selects the kernel-matrix rows,
including the attention default arm and the GEMM ksplit default, so a build
without it is not the shipped build) and derives the arch from nvidia-smi
(9.0 is `sm_90a`). Phases, each with an exit code in `status.tsv`: base
binding build, byte LM binding shipped and timers builds
(`-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1`, no
trial define) into `bin/shipped` and `bin/timers`, both corpora fetched and
verified, the section 6 runs (the binding copied into the package before
each), the summary, and the binaries removed before the fetch. Estimated
lease: three builds about 9 min, seven target probes about 1.5 min each,
corpora about 2 min.

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_GEMM_LEG_EXTRA=tools/step_breakdown_leg.sh MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-step-breakdown sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 --gpu "NVIDIA H100 80GB HBM3" --local-card <card>

from a `git worktree add --detach` checkout at the lane's commit. Read
`remote/step-breakdown/witnesses.tsv` first (`all bits_identical True`),
then `breakdown.tsv` (`category` and `remainder:envelope.native_call`
rows), then the `lean` rows.

## 8. RUN OWED, in order (M4 light checks, one at a time)

1. The shipped path still builds and is still bit-identical (no define):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check` (expect PASS as before this lane).
2. The GEMM device check, no define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check`
   then `nice -n 19 /tmp/gemm-device-check` (expect PASS).
3. The backward check, no define (`bwd_rms_norm` is now parametric):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/backward-check`
   then `nice -n 19 /tmp/backward-check` (expect PASS).
4. A timers build compiles and still passes, switch off and on:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/backward-check-timers`
   then `nice -n 19 /tmp/backward-check-timers` and
   `MOJOLEARN_TRANSFORMER_TIMING=1 nice -n 19 /tmp/backward-check-timers`
   (expect PASS both; the second prints `fwd.*`, `grad.*`, `gemm.*`,
   `launches.*` and `syncs.*` lines).
5. The byte LM binding builds with the timers (macOS guard):
   `MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BYTE_LM_OUTDIR=/tmp/step-timers-bytelm MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1" sh bindings/build_byte_lm.sh`, on a GPU box and not on the Mac: a full binding build is heavy, and `tools/macos_serial_guard.py` admits tiny jobs only (1 to 180 s, 1 to 4 GiB), so the guarded spelling this item first carried can never run.
6. Only if the M4 already runs the probe at the control shape (the neural
   handoff's rehearse-before-renting rule), in a detached worktree with
   that binding installed as `python/mojolearn/identical/_mojolearn_byte_lm.so`
   and enwik8 fetched:
   `MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$PWD/python:$PWD nice -n 19 pixi run python tools/lm_step_memory_probe.py --out /tmp/step-breakdown-rehearsal/timing-enwik8 --resident-lean --witness-every-step --steps 1 --component-timing --component-timing-steps 2 --corpus training/corpus/enwik8/input.txt --budget-seconds 600`
   then `pixi run python tools/step_breakdown_summary.py /tmp/step-breakdown-rehearsal`
   (expect no `untreed` rows inside the envelope and
   `component_timing_uneven` empty; `bits_identical` reads False there
   because no shipped run sits beside it).
7. `sh -n tools/step_breakdown_leg.sh` (the runner does this too).

**Merge note (2026-09-11, branch `lane/step-breakdown-h100-merged`).**
This lane was merged with origin/main at afba564c (the AMD attention
default `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, attention brief
section 18). The only conflicts were three hunks in
`transformer/impl/llama/fused_attention.mojo`, where main moved the
sabotage copies in `_launch_bwd_stash_zdq_pf` and `_launch_bwd_stash_tiled_kv`
under `comptime if ATTN_ARM_TRIAL`. They were resolved as main's structure
with this lane's `step_count_launch()` inside each trial branch. Main's
shipped `_launch_bwd_stash_tiled_kv[64, ATTN_DEFAULT_KV_KEYS,
ATTN_DEFAULT_KV_SPLIT]` call carries no counter of its own, because every
allocation, synchronize and launch inside that helper (and inside
`_launch_bwd_stash_zdq_pf`) is already counted. Nothing was built. The
orchestrator runs these M4 light checks on the merged branch, ONE AT A
TIME, before the H100 leg:

1. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check` (expect PASS).
2. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-kv`
   then `nice -n 19 /tmp/fused-check-kv` (expect PASS naming
   `bwd_stash_tiled_pf_kvgrid_r32` and `fused_bwd_dkdv_r2_kernel[64,32]`).
3. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check`
   then `nice -n 19 /tmp/gemm-device-check` (expect green).
4. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/backward-check`
   then `nice -n 19 /tmp/backward-check` (expect PASS).
5. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/backward-check-timers`
   then `nice -n 19 /tmp/backward-check-timers` and
   `MOJOLEARN_TRANSFORMER_TIMING=1 nice -n 19 /tmp/backward-check-timers`
   (expect PASS both).
6. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check` (expect
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 18 arms`).
7. The byte LM timers binding build:
   `MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BYTE_LM_OUTDIR=/tmp/step-timers-bytelm MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1" sh bindings/build_byte_lm.sh`, on a GPU box and not on the Mac: a full binding build is heavy, and `tools/macos_serial_guard.py` admits tiny jobs only (1 to 180 s, 1 to 4 GiB), so the guarded spelling this item first carried can never run.

Not covered by these checks: the counters inside the shipped kv helper are
compiled with the define only on a column whose default carries a
DEVIATION 2597 token (AMD), or on a trial build. No M4 command above
combines `-D MOJOLEARN_STEP_PHASE_TIMERS=1` with
`-D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` or
`-D MOJOLEARN_ATTN_ARM_TRIAL=1`. The first timers build on AMD compiles
them for the first time.

## 9. Risks only a build or a box can settle

- `std.ffi._Global` from a non-binding module (`core/step_phase.mojo`) has
  not been compiled in this tree, and whether `get_or_create_ptr` raises is
  not known here (the helpers are declared `raises` either way). Two shared
  libraries built with the define in one process would share the slot name
  `MojolearnStepPhaseCountsV1`; only the byte LM binding is built with it.
- `comptime if which == 1:` on a defaulted comptime parameter and the call
  spelling `bwd_rms_norm[2](...)` are uncompiled.
- The zero-cost claim for the shipped build rests on the compiler dropping
  three unread field stores per clock. The leg compares the shipped and
  timers builds, not this lane's shipped build against the parent commit's;
  the 0.2950 s reference is from another pod.
- The counters see only call sites in the step's files. The MAX runtime
  may wait or launch inside `enqueue_create_buffer`, frees at last use,
  host buffer creation or copies; none of that is counted.
- Per-step grouping assumes every timed step prints the same lines in the
  same order (a refusal or a fallback path would break it); the probe
  lists any name that violates it.
- In the timers build the counters also increment with the switch off (a
  global slot lookup per site, about 2,000 per step here);
  `lean-timers` minus `lean-shipped` measures it.
- `transformer/impl/llama/fused_attention.mojo` (92 inserted lines) and
  `gemm/checks/gemm_identical.mojo` are active lane files; expect merge
  conflicts that are mechanical.
- The body handles an AMD box when `MOJOLEARN_GPU_ARCHS` is given, but only
  the NVIDIA H100 leg is written and costed.

## 10. Files

- `core/step_phase.mojo` (new)
- `transformer/impl/llama/modeling_llama.mojo`,
  `transformer/checks/transformer_backward.mojo`, `training/byte_lm.mojo`,
  `bindings/_mojolearn_byte_lm.mojo` (ticks and counters)
- `transformer/impl/llama/fused_attention.mojo`,
  `gemm/checks/gemm_identical.mojo`, `core/device_scan.mojo`,
  `embedding/checks/embedding_identical.mojo`, `training/checks/loss.mojo`,
  `training/checks/optimizer.mojo`, `training/checks/train_loop.mojo`
  (counters only)
- `tools/lm_step_memory_probe.py` (multi-step component timing, counts,
  witnesses)
- `tools/step_breakdown_summary.py`, `tools/step_breakdown_leg.sh` (new)
