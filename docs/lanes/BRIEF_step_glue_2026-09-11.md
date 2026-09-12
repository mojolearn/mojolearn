# Step glue lane: the byte LM step outside GEMM and attention (DEVIATIONS 2645 to 2648)

Source-only lane, September 11, 2026, IDENTICAL only. STATUS: BUILT, RUN,
MEASURED AND FLIPPED (section 11). The lane was wound down half written and
resumed the same day on Andrew's order; main was merged in at 4dc4346a (the
attention `_estash` work) with no conflict.

**THIS PARAGRAPH USED TO SAY "NOT BUILT, NOT RUN" AND "DEVIATION 2649 IS
RESERVED AND UNUSED". BOTH ARE NOW FALSE AND THE CORRECTION IS THE POINT.**
It read that way while the code was unbuilt; the M4 compiled it, the H100
leg of section 7 ran every arm on 2026-09-11 (commit 030079af), and
DEVIATION 2649 flipped the winner into the shipped NVIDIA path on
2026-09-12. Section 11 is the record.

DEVIATIONS 2645 to 2648 are this lane's arms; 2649 is the flip.

Branch `lane/step-glue-h100` (from origin/main 36ca51fd).

This brief was written BEFORE the code, as the lane's rules require. The
identity argument (section 4) and the refusal argument (section 5) are the
reason each arm may exist at all.

## 1. Where the time is (read, not new)

Evidence
bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown/remote/step-breakdown/breakdown.tsv
(H100 80GB HBM3, 1980 MHz, lean shipped step 295.8 / 294.7 ms, timed
envelope 299.8 ms, witnesses equal with timers compiled in, off and on).
Target shape B2 L1024 DM768 FF2048 V50257, 12 layers, n = 162,147,840
parameters, M = 2,048 token rows.

Outside GEMM (144.9 ms) and the attention kernels (116.2 ms), per step:

| component | ms | calls | per call |
|---|---:|---:|---:|
| `fwd.norm1` + `fwd.norm2` (RMSNorm forward, one launch each) | 6.139 | 24 | 256 us |
| `grad.norm1_kernels` + `grad.norm2_kernels` (three launches each) | 4.044 | 24 | 169 us |
| `step.opt_refuse_scan` (param, grad, m, v scans) | 2.176 | 1 | 4 scans, 0.54 ms each |
| `step.validate_after_scan` (param, m, v finite, v not negative) | 2.150 | 1 | 4 scans |
| `step.shadow_copy` (param, m, v copied, 3 launches) | 2.308 | 1 | |
| `step.optimizer` (the AdamW launch and its wait) | 1.714 | 1 | |
| `step.embedding_backward` | 2.522 | 1 | |
| `step.unpack_weights`, `step.pack_grads` | 2.281 | 1 each | |
| `fwd.refuse_call`, `grad.refuse_scan`, `step.ce_refuse_scan`, `step.validate_grads_scan` | 2.030 | | |

### 1.1 The 3.4 ms against 6.4 ms question

They are two different sets of scans and both numbers still stand. The
declined "fused regime scans / deferred flag read" (handoff section 2) was
priced on the ATTENTION launchers' regime scans and corner flags. In this
breakdown those are `attn.fwd_regime_scan` 1.141 + `attn.bwd_regime_scan`
1.578 + `attn.fwd_corner_flag` 0.280 + `attn.bwd_corner_flag` 0.319 = 3.318
ms, still 3.3 ms, and they sit in the attention category. The 6.356 ms
"refusal and validation scans" category is the STEP's own refusals:
`fwd.refuse_call` 0.691, `grad.refuse_scan` 0.396, `step.ce_refuse_scan`
0.388, `step.validate_grads_scan` 0.555, `step.opt_refuse_scan` 2.176,
`step.validate_after_scan` 2.150. Nothing changed; no gap exists.

### 1.2 A traffic model that fits the step

Words moved per second, per SM (H100 132 SMs, attention brief 3.1), from
leaves whose launches cover at least 132 blocks:

| leaf | words per call | ms | G words/s | per SM |
|---|---:|---:|---:|---:|
| `fwd.residual1_add` (1.57M cells, 12,288 blocks) | 4.7M | 0.018 | at least 260 | at least 2.0 |
| `fwd.silu` (4.2M cells) | 12.6M | 0.031 | about 410 | about 3 |
| `step.validate_grads_scan` (162M strided loads) | 162M | 0.555 | 292 | 2.2 |
| `step.optimizer` (4 loads, 3 stores per element) | 1,135M | 1.714 | 662 | 5.0 |
| `step.ce_forward` (row max, shift and exp, the GEMM fold) | about 310M | 1.711 | about 180 | 1.4 |

RMSNorm forward launches ONE THREAD PER TOKEN ROW at `LLAMA_TPB` 128, so
2,048 rows are 16 blocks and can occupy at most 16 SMs. Its traffic per
call is x read twice and the weight once per cell plus one store, about
4 x 1.57M = 6.3M words. At the 1.4 to 2.0 G words/s per SM the table
reads, 16 SMs predict 6.3M / (16 x 1.5 G/s) = 263 us. Measured: 256 us.
The backward `bwd_norm_dot_kernel` is the same shape (one thread per row,
16 blocks, dh and x read, about 3.2M words): 130 us predicted inside a 169
us three-kernel leaf whose other two kernels are elementwise over at least
132 blocks (about 40 us together, from the residual add's floor).

Nothing else outside GEMM and attention launches fewer blocks than SMs at
the target (every other launch in the step is elementwise over at least
1.57M cells, or per row of the vocabulary at 2,048 blocks).

The model is a FIT, not a proof. The alternative reading is that the CUDA
runtime packs low-register blocks several to an SM (the occupancy query
allows up to 8 blocks of 256 threads when registers are few), in which case
16 blocks may not be SM-bound and more blocks buy nothing. The leg decides
it with `fwd.norm1` under the timers (section 7).

## 2. The arms

One trial define, `-D MOJOLEARN_STEP_GLUE_TRIAL=1`, and one runtime
selector, `MOJOLEARN_STEP_GLUE_ARM`, a name made of tokens in this order,
joined by `_`: `optskip`, `noshadow`, then one of `rows16`, `rows8`,
`rows4`. `shipped` or unset is no token. Examples: `rows16`,
`optskip_noshadow`, `optskip_noshadow_rows16`. An unknown token, a repeated
token, two rows tokens or tokens out of order raise. A build without the
define compiles none of the new launch paths and never reads the variable.
`MOJOLEARN_STEP_GLUE_ARM_SABOTAGE=1` (trial builds only) is the reach
sabotage of section 6.

| token | DEVIATION | what changes | expected saving, ms per step |
|---|---|---|---:|
| `rows16` / `rows8` / `rows4` | 2645 | the two row kernels launch R threads per block, so 2,048 rows are 128 / 256 / 512 blocks | about 7 if section 1.2 holds, 0 if blocks pack |
| `optskip` | 2646 | the optimizer's four entry scans are not run on the byte LM step | 2.18 |
| `noshadow` | 2647 | AdamW writes the new state INTO the shadow buffers and the step swaps handles; no shadow copy | 2.31 |
| (infrastructure) | 2648 | `core/step_glue.mojo`, the binding read-back, the probe field, the check, the leg | 0 |

The rows arithmetic. RMSNorm forward at 128 blocks: 6.3M / (128 x 1.5 G/s)
= 33 us, plus a launch and wait floor of about 18 us (the residual add
leaf), so about 51 us against 256 us, 0.205 ms x 24 = 4.9 ms. The dot
kernel at 128 blocks: about 17 us against about 130 us, 0.11 ms x 24 = 2.7
ms, bounded by the 4.04 ms leaf. Sum about 7 ms. rows8 and rows4 exist
because the SM count is 132 and 128 blocks leave four idle, and because a
16-thread block is half a warp; the leg picks.

Total if every arm reads as predicted: 7 + 2.2 + 2.3 = 11.5 ms, 3.9
percent of the 292 ms step. Without the rows model: 4.5 ms, 1.5 percent.
Either clears the lane's 1 percent bar (about 3 ms).

## 3. What is NOT built, with the reading

- A fused `validate_after` (param, m, v and v negative in one kernel over
  three buffers): 486M strided words instead of 648M, about 0.54 ms. Under
  the bar alone; left out to keep one new scan kernel out of the tree.
- Weight and gradient VIEWS instead of `step.unpack_weights` and
  `step.pack_grads` (2.28 ms): `LlamaDeviceWeights` and
  `LlamaBackwardStages` would take sub-buffers of `param` and `grad`.
  Views of `param` conflict with `noshadow`'s handle swap unless the views
  are rebuilt after every swap; the gradient half (1.12 ms) does not
  conflict. Recommended as the next glue arm if `noshadow` flips.
- The embedding backward (2.52 ms): its counts kernel walks all 2,048 ids
  for each of 50,257 vocabulary threads (103M loads) and the backward
  kernel loads two run bounds for all 38.6M cells, though at most 2,048
  rows are non-empty. A host-built run table (the step already holds the
  ids on the host) and a launch over the non-empty rows only would keep the
  per-cell fold and cut most of the traffic, maybe 1.5 ms. Not built: the
  embedding lane's contract names the device seams E0 to E2.
- The copy kernel's block size (`TRAIN_TPB` 128): the shadow copy moves 972M
  words in 2.31 ms (421 G/s) while the AdamW launch at 256 threads per
  block moves 1,135M in 1.71 ms (662 G/s). A 256-thread copy might take a
  third off every copy leaf; unmeasured, not built.
- Anything that needs `gemm/checks/gemm_identical.mojo` or the attention
  kernels. The GEMM calls own 988 of the step's 2,021 synchronizes (four
  per call) and 520 of its 1,324 launches, and a GEMM call's fixed cost
  fits about 94 us (proj_fwd 0.271 ms at 1.21 GFLOP against gateup 0.565
  ms at 3.22 GFLOP). Host and device run in lockstep because every call
  waits; only those files can change that.

## 4. Why no computed bit can change

### 4.1 rows (DEVIATION 2645)

`llama_rms_norm_kernel` and `bwd_norm_dot_kernel` read their row as
`t = block_idx.x * block_dim.x + thread_idx.x`, return when `t >= m`, and
keep the whole row fold (contract S1, ascending `j` from +0.0) inside the
thread's registers. No expression reaches `block_dim` except that index.
Any launch whose blocks times threads covers `[0, m)` computes every row
exactly once with the same operands in the same order, so `sumsq`, `out`,
`dot`, `rstd` and `dvcoef` are the same bits at 128, 16, 8 or 4 threads per
block. The kernels' own docstrings state it ("no launch geometry can
reorder it"). Every thread count is a power of two, well under every
column's block cap (Metal 1,024). No other kernel's geometry moves.

### 4.2 optskip (DEVIATION 2646)

The four scans compute a refusal, never a stored value. Removing them
cannot change a bit of any buffer. Section 5 says why they can never fire
on this path.

### 4.3 noshadow (DEVIATION 2647)

`adam_update_oop_kernel` (training/checks/optimizer.mojo, next to
`adam_update_kernel`) is the clean path of `adam_update_kernel`, seam for
seam: O1 to O3 loads with `ftz`, O4b `ftz(identical_mul(decay_mul, p))` or
O4a `ftz(identical_mul_add(weight_decay, p, g))`, O5 and O6
`ftz(identical_mul_add(c1, g, ftz(identical_mul(beta1, mp))))`, O7 to O9
`ftz(identical_mul_add(c2, ftz(identical_mul(g, g)), ftz(identical_mul(beta2, vp))))`,
O10 to O12 `ftz(ftz(identical_div(ftz(identical_sqrt(v)), rt_bc2)) + eps)`,
O13 `ftz(identical_div(m, dn))`, O14 `ftz(identical_mul_add(-step_size, q, p))`,
stores `p_out`, `ftz(m)`, `ftz(v)`. It reads element `i` of `param`,
`grad`, `m_state`, `v_state` and writes element `i` of three OTHER buffers.
The shipped kernel reads and writes the same buffer, but it loads all four
operands before its first store, so the values it reads are the pre-update
values in both kernels. One thread per element, no shared memory, no
reduction: the same argument as the shipped kernel's docstring.

What the transcription does NOT carry: the sabotage arms (the byte LM
refuses every sabotage build, `_require_profile`) and the recorded
intermediates (`OPT_RECORD_INTERMEDIATES`; the trial path refuses the
combination at compile time).

READ LINE BY LINE against `adam_update_kernel` (lane, 2026-09-11, after
writing it). Walking the shipped kernel top to bottom with every sabotage
define off: `SAB_ADAMW_AS_ADAM` (skipped, `is_adamw` from the argument),
the four loads with `ftz` in the order g, p, mp, vp (same order), the
`SAB_FTZ_LATE` reloads (skipped), the decay branch on `weight_decay !=
0.0` with O4b under the `SAB_DECAY_ADD_FORM` else and O4a (same
expressions), `SAB_MOMENT_LERP` else `ms` then `m` (same), `vs` formed
before the `SAB_SQ_ASSOC` else `g2` then `v` (same order), the
`SAB_FTZ_LATE` recompute (skipped), `SAB_MHAT_FORM` (skipped, its early
return is the sabotage's), `SAB_EPS_INSIDE_SQRT` else `SAB_RSQRT` else
`s`, `sd`, `dn` (same), `SAB_RECIP_MUL` else `q` (same),
`SAB_UNFUSED_UPDATE` else O14 `identical_mul_add(-step_size, q, p)` (same),
the stores `p_out`, `ftz(m)`, `ftz(v)` (same values, other buffers),
`OPT_RECORD_INTERMEDIATES` (not carried). The shipped `var m =
Float32(0.0)` and `var v = Float32(0.0)` declarations are overwritten
before any use and compute nothing. No operand, operation, flush or order
differs. The only plain arithmetic operators in either clean path are
`sd + eps` and the negation `-step_size`, both identical, and neither has a
multiply operand a compiler could contract.

The file's own warning applies: a random fixture cannot see a fused against
unfused O14 (zero of 2^20 hashed patterns in `check-ieee-arith`). So the
transcription rests on reading it beside the shipped kernel, as the shipped
kernel's own O14 does, plus the check's bit comparison on planted values and
the H100 leg's witnesses over 486M updated elements per corpus.

After the kernel's wait, `swap(tr.buffers.param, tr.buffers.shadow_p)` and
the same for `m_state` and `v_state`. From then on `param`, `m_state`,
`v_state` hold exactly what the shipped in-place update left in them, and
`shadow_p`, `shadow_m`, `shadow_v` hold exactly what the shipped shadow copy
left in them (the pre-update state). Every later reader (the next step's
unpack, `validate_device_state`, `export_state`, `byte_rollback`) reads the
fields at call time; nothing in the trainer, the binding or the Python layer
keeps a pointer or a view of those buffers across calls (unpack copies,
exports stage per call, the weights are separate buffers). The buffers are
allocated once with the same length, so a swap is legal wherever a copy
was.

## 5. Why every refusal is the same

### 5.1 The inputs of a step

A resident step's inputs are the token ids (host-validated by
`byte_validate_tokens` before any device work, unchanged) and the trainer
state `param`, `m_state`, `v_state`, flags and step, which the trainer only
holds after one of three write events:

1. `ByteTrainer.__init__` (open): `byte_validate_state` on the host Lists
   BEFORE the upload (finite param, finite m, finite v, v not negative).
2. A successful step: the update, then `validate_device_state` (the same
   four predicates on the device) passes, or the step raises and rolls back.
3. `byte_rollback`: three copies from the shadow, then
   `validate_device_state`; if it raises the session is lost and no step
   can run.

No other code writes those three buffers. The forward, CE, backward, pack,
`validate_grads_scan` and the shadow copy only read them (the shadow copy
writes `shadow_*`, distinct allocations). `export_state` and
`export_gradients` read. Eval reads. The stateless `byte_train_step` adds
the host `byte_validate_state` before the same body.

### 5.2 optskip refuses exactly what shipped refuses

`opt_refuse_device_inputs` raises when param, grad, m or v holds a NaN or an
infinity (by bits, `|bits| >= 0x7F800000`).

- grad: `step.validate_grads_scan` runs the same predicate with the same
  kernel (`nonfinite_partial_kernel`) over the same `n` elements of the same
  buffer immediately before, and raises first. Between the two scans only
  the shadow copy runs, which does not write `grad`. So on every input the
  optimizer's grad scan finds nothing.
- param, m, v: at the optimizer they hold the bits of the last write event
  (5.1), and every write event ends with a pass of predicates that include
  the optimizer's (finite param, m, v). So the scans find nothing.

So on every reachable input, shipped's optimizer scans never raise, and
skipping them refuses the same set of inputs with the same messages in the
same order. The only way to reach them is to write the state between events,
which exists only in the fault-injection build (`opt_refuse` plants a NaN in
`m_state` after the shadow copy). The trial path refuses to compile with
`-D MOJOLEARN_BYTE_LM_FAULT_INJECT=1` (a `comptime assert`), so the G4 gate
build keeps its controls and its messages; it runs the shipped path.

### 5.3 noshadow refuses and rolls back exactly as shipped does

Refusals that can fire after the grad scan on this path: `validate_after`
(an update can overflow, for example `g * g` past float max or an `lr` large
enough that `step_size` is infinite; the check plants the second). Shipped
order: shadow copy, `shadow_valid = True`, optimizer, `validate_after`
raises, `_byte_recover` calls `byte_rollback`, which copies shadow into
state, restores flags and step, re-scans, raises the step's message.
noshadow order: `flags_before` and `shadow_step` set, kernel into shadows,
wait, three swaps, `shadow_valid = True`, `validate_after` raises (it reads
the post-swap `param`, `m_state`, `v_state`, which are the bits shipped's
in-place update produced), `_byte_recover` calls the same `byte_rollback`,
which copies `shadow_*` (the pre-update state, the bits shipped's shadow
copy held) into state. Same message, same restored bits, same flags, same
step, same `shadow_valid` afterwards.

A raise before the swap (the kernel launch or its wait failing, a lost
context) leaves `param`, `m_state`, `v_state` untouched and `shadow_valid`
False, so `_byte_recover` takes its "before the shadow point" branch: re-scan
and raise. Shipped would have rolled back a bitwise-equal shadow first; the
restored state and the message are the same.

`noshadow` without `optskip` runs the four scans BEFORE the kernel with
`shadow_valid` False. They cannot fire (5.2). If memory corruption made one
fire, shipped reports "rollback re-scan failed" and the arm reports the
recover re-scan's "session lost"; both mark the session lost. This is the
only message difference, and it is unreachable.

An explicit `byte_lm_session_rollback` after a SUCCESSFUL step restores the
pre-step state in both (shipped: shadow holds the copy; noshadow: shadow
holds the swapped-out buffers). The check exercises this.

### 5.4 rows

A launch geometry change raises nothing and refuses nothing.

## 6. The check, `training/checks/step_glue_check.mojo`

Built WITH the trial define (without it `main` fails at once, naming the
define). Clauses, each printing PASS or FAIL, then one summary line.

- (a) NAMES, host only: `step_glue_arm_parse` and `step_glue_arm_name` are
  inverses on all 16 valid words; eight invalid spellings raise.
- (b) RMSNorm FORWARD: `llama_rms_norm` under `shipped` and under rows16,
  rows8, rows4 on (m, dm) = (45, 24), (300, 64), (2048, 768) with hashed
  operands including +0, -0, subnormals and large magnitudes: `sumsq` and
  `out` bit-equal. REACH by sabotage: with
  `MOJOLEARN_STEP_GLUE_ARM_SABOTAGE=1` the rows launch uses floor(m / R)
  blocks, so at m = 45 the FIRST MOVED ROW of `out` must be exactly
  floor(45 / R) x R (32, 40, 44). That names R, so it proves the arm's
  threads per block reached the launch and not merely that something moved.
  The shipped geometry under the same variable must move nothing.
- (c) RMSNorm BACKWARD: `bwd_rms_norm` the same way (dot, dx, dW bit-equal;
  reach: first moved row of `dot` is floor(45 / R) x R).
- (d) UPDATE: on n = 1,000 elements over 3 tensors with hashed finite
  operands, some gradients near 1e20 so `v` overflows to infinity in the
  output: shipped `identical_optimizer_step` against the arm's in-place
  launch (`optskip`) and the out-of-place launch (`noshadow`), param, m, v
  bit-equal including the infinities, and the out-of-place launch leaves its
  input param untouched. VACUOUS unless the shipped output holds an
  infinite moment. REACH: under sabotage the out-of-place launch covers
  floor(n / 256) blocks, so the first moved element is exactly 768; the
  in-place launch (shipped kernel and geometry) under the same variable
  must move nothing.
- (e) STEP, end to end, at the default ByteConfig (B2 L32 DM32 H4 KV2 FF64
  V256, 2 layers, 34,944 parameters): the same state and the same three
  batches under `shipped` and every arm in section 7's list; loss bits and
  the downloaded gradient, param, m and v after every step equal. VACUOUS if
  param did not move.
- (f) REFUSAL, end to end, two admitted configurations. Case 0, `lr =
  3.4e38`, `beta1 = beta2 = 0`, `weight_decay = 1`: decoupled decay takes a
  norm weight near 1 to about -3.4e38 and the step overflows it, so
  `validate_after` refuses and the rollback runs AFTER noshadow's swap.
  Case 1, `lr = 3e38`: the step size overflows. The shipped outcome must be
  a clean refusal (raised, state restored, `completed_steps` 0, healthy, a
  second `byte_rollback` False), and every arm must match it field by
  field, message included.
- (g) ROLLBACK after success: one good step then `byte_rollback` returns
  True and restores the pre-step bits, in every arm.

## 7. The leg, `tools/step_glue_leg.sh`

A POSIX sh body for `tools/gemm_remote_leg.sh`'s `MOJOLEARN_GEMM_LEG_EXTRA`,
modeled on `tools/step_breakdown_leg.sh`. Phases, each with an exit code in
`status.tsv`:

1. builds: the base binding; the byte LM binding shipped (`bin/shipped`);
   with `-D MOJOLEARN_STEP_GLUE_TRIAL=1` (`bin/glue`); with the trial define
   and both timer defines (`bin/gluetimers`); the check with the trial
   define, then run (`glue-check.log`).
2. both corpora fetched and verified.
3. per corpus: `lean-shipped` (shipped binding, 1 + 3 steps, witness every
   step), `lean-glue-shipped` (trial binding, arm `shipped`: the reference
   of the verdict), then `lean-glue-<arm>` for every arm in
   `MOJOLEARN_STEP_GLUE_LEG_ARMS` (default
   `optskip_noshadow,rows16,rows8,optskip_noshadow_rows16,optskip_noshadow_rows8`).
4. on enwik8, the timers binding: `timing-shipped` and `timing-<best
   combined arm>` (default `optskip_noshadow_rows16`), 1 warmup and 3 timed
   steps, for the attribution table.
5. `summary.tsv`: per arm and corpus the steady median, the ratio against
   `lean-glue-shipped` and against `lean-shipped`, witnesses equal to
   `lean-shipped` on every step, the geometric mean of the two
   `lean-glue-shipped` ratios, and `verdict <arm> FLIP` or `NO FLIP`
   (ENGINEERING_RULES 9: geomean below 1 and every witness equal on both
   corpora). `attribution.tsv`: the rows of section 1 from both timing runs
   side by side.

Estimated body: builds about 6 min (four builds from about 1 min each on the
breakdown leg, the check about 2 min), corpora 0.5 min, 14 lean probes at
about 85 s each (20 min), 2 timing probes (3 min): about 30 min.

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_GEMM_LEG_EXTRA=tools/step_glue_leg.sh MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-step-glue sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 --gpu "NVIDIA H100 80GB HBM3" --local-card <card>

from a `git worktree add --detach` checkout at the lane's commit. Read
`glue-check.log` first, then `summary.tsv` (`witnesses_equal_shipped True`
on every row), then `attribution.tsv`.

## 8. RUN OWED (M4 light checks, one at a time)

1. The trial check:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_GLUE_TRIAL=1 -I . training/checks/step_glue_check.mojo -o /tmp/step-glue-check`
   then `nice -n 19 /tmp/step-glue-check` (expect `step_glue_check: PASS`).
2. The same source without the define must fail at run time naming it:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/step_glue_check.mojo -o /tmp/step-glue-check-shipped`
   then `nice -n 19 /tmp/step-glue-check-shipped` (expect exit 1 and
   `MOJOLEARN_STEP_GLUE_TRIAL`).
3. The shipped paths still build and pass, no define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/backward-check`
   then `nice -n 19 /tmp/backward-check`, and
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/optimizer_check.mojo -o /tmp/optimizer-check`
   then `nice -n 19 /tmp/optimizer-check` (expect both as before).
4. `sh -n tools/step_glue_leg.sh`.

The byte LM binding builds (shipped, trial, trial plus timers) run on the
GPU box only (tools/macos_serial_guard.py admits tiny jobs only).

## 9. Risks only a build or a box can settle

- Whether 16-thread blocks really spread over SMs on the H100 (section 1.2);
  if the runtime packs them, the rows arms read 1.00.
- `swap` on two fields of one `mut` struct argument (the builtin); the same
  file already passes two fields of `tr.buffers` as `mut` arguments to
  `_copy_into`.
- `String.split` on the arm name and `setenv` in the check (both used
  elsewhere in the tree: `split` in bindings, `setenv` in
  `cluster/checks/kmeans_identity_check.mojo`).
- `comptime if` around a runtime `if` that returns early from a launcher
  (the attention launchers do the same).
- The check's end-to-end clauses run the eager attention path at head_dim 8;
  the H100 leg is the only run at head_dim 64 with the fused kernels.
- Check spellings not yet compiled in this tree: `String(Bool)`, a
  `@fieldwise_init` struct with a `String` field returned by value,
  `bwd_rms_norm(...)` called without its comptime `which` (the samba ops
  spelling), `_grid_for as _opt_grid_for` imported from the optimizer, and
  a `ByteTrainer` built and returned from a helper.
- Case 1 of clause (f) may be refused by the host scalars (a nonfinite step
  size) rather than by `validate_after`; either way every arm must match
  shipped, and case 0 is the one that must reach the post-swap rollback.
- On the H100 the new launches add no blocks per SM for the update (the
  out-of-place kernel keeps 256 threads per block), but they do write to
  the shadow allocations instead of the state allocations; whether page
  placement differs is unknown and only the lean step says.
- `transformer/impl/llama/modeling_llama.mojo` and
  `transformer/checks/transformer_backward.mojo` gain a few lines each and
  `training/checks/optimizer.mojo` one kernel; merge conflicts with active
  lanes should be mechanical.

## 10. Files

- `docs/lanes/BRIEF_step_glue_2026-09-11.md` (this file)
- `core/step_glue.mojo` (new): define, arm bits, parse, name, env, row
  geometry, sabotage switch
- `transformer/impl/llama/modeling_llama.mojo` (`llama_rms_norm` launch)
- `transformer/checks/transformer_backward.mojo` (`bwd_rms_norm` dot launch)
- `training/checks/optimizer.mojo` (`adam_update_oop_kernel`)
- `training/byte_lm.mojo` (the glue update path in `_byte_step_device`)
- `bindings/_mojolearn_byte_lm.mojo` (`byte_lm_step_glue_arm` read-back)
- `python/mojolearn/_byte_lm_impl.py`, `tools/lm_step_memory_probe.py`
  (the arm in `run_metadata` and `result.json`)
- `training/checks/step_glue_check.mojo` (new), `tools/step_glue_leg.sh` (new)

## 11. The leg, the flip and the shipped branch (2026-09-11 and 12, measured): DEVIATION 2649

THE LEG. RunPod NVIDIA H100 80GB HBM3, pod zlzx4eqhs62ahx, commit 030079af,
one pod and one heat window, every arm timed against the SAME build's
`shipped` arm. Evidence
`bench/results/e1g/2026-09-11_215139-nvidia-h100-80gb-hbm3-step-glue`.

| arm | enwik8 | pilegithub | geomean |
|---|---:|---:|---:|
| `optskip_noshadow` | 0.9846 | 0.9852 | 0.9849 |
| `rows16` | 0.9853 | 0.9851 | 0.9852 |
| `rows8` | 0.9887 | 0.9859 | 0.9873 |
| `optskip_noshadow_rows8` | 0.9732 | 0.9737 | 0.9734 |
| `optskip_noshadow_rows16` | 0.9744 | 0.9703 | **0.9723** |

Every arm FLIPs. `witnesses_equal_shipped=True` on every arm and both
corpora, `arm_named=True`, drift arm 1.0054 over the window, all 26 stages
exit 0. Lean step 0.2911 / 0.2921 -> 0.2837 / 0.2834 s. ENGINEERING RULES 9
takes the winner, `optskip_noshadow_rows16`.

Attribution accounts for the whole 8.0 ms (timers build, same box,
`timing_witnesses_equal=True`), enwik8, ms: `fwd.norm1` 3.068 -> 1.554,
`fwd.norm2` 3.077 -> 1.562, `grad.norm1_kernels` 2.026 -> 1.399,
`grad.norm2_kernels` 2.038 -> 1.393, and `step.shadow_copy` (2.309) and
`step.opt_refuse_scan` (2.175) gone. The optimizer, the scans and the
packing are unchanged to within 0.013 ms. Section 1.2's risk (the runtime
packing 16-thread blocks so the rows arms read 1.00) did not happen.

THE SHIPPED BRANCH (what the flip owed, the DEVIATION 2657 shape). The arms
were reachable only under `-D MOJOLEARN_STEP_GLUE_TRIAL=1`, and
`step_glue_arm_from_env` opened with `comptime if not STEP_GLUE_TRIAL:
return STEP_GLUE_SHIPPED`, so moving a default alone would have changed
nothing at all. DEVIATION 2649 adds:

- `step_glue_default_arm_for` in `checks/kernel_matrix.mojo`, the ROUTING
  row, with the word literal `STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16`
  (7) and the check knob `MOJOLEARN_STEP_GLUE_DEFAULT_EVERY_COLUMN`. NVIDIA
  returns the winner; every other column returns `shipped`, so Apple and AMD
  are unmoved by construction.
- `STEP_GLUE_ARM_DEFAULT`, `STEP_GLUE_SHIPPED_UPDATE`, `STEP_GLUE_ROWS_FIT`
  and `STEP_GLUE_SHIPPED_ROWS` in `core/step_glue.mojo`, mirroring
  `ATTN_ARM_DEFAULT` and `ATTN_SHIPPED_BWD_ESTASH`. The rows constants sit
  after `step_glue_rows_of` because they call it.
- `step_glue_arm_from_env` returns `STEP_GLUE_ARM_DEFAULT` instead of the
  literal `STEP_GLUE_SHIPPED`, and asserts the matrix literal against this
  file's composition. The early return stays ahead of `getenv`: the two
  RMSNorm launchers call it once per launch.
- The three entry points take `comptime if TRIAL or SHIPPED_*`.
- `step_glue_sabotage()` does NOT widen. It stays trial-only, so a shipped
  build never takes the floor in `step_glue_blocks`.

A CORRECTNESS HOLE FOUND WHILE BUILDING IT. `_byte_glue_update`'s
`comptime assert not BYTE_LM_FAULT_INJECT` sat INSIDE
`comptime if STEP_GLUE_TRIAL`. A shipped glue build would have reached the
glue update with that refusal silently gone, and section 5.2's `optskip`
argument depends on it: the `opt_refuse` fault site writes `m_state`
between the shadow copy and the optimizer, which is the one write the
argument excludes. Both that assert and the `OPT_RECORD_INTERMEDIATES` one
now carry the same two-part condition as the path they guard.

THE M4 GATES (2026-09-12, on origin/main 5b7e1e41 plus the flip, one build
at a time):

1. `transformer_backward_check`, no trial define, knob ON: PASS, 17 cases,
   37/37 stages bit-identical to the oracle on all 412,172 cells. The rows
   branch is compiled into a SHIPPED build and changes no bit.
2. The same check with no knob: PASS, identical clause line. The Apple
   default is untouched.
3. `transformer_fused_check`, no trial define, knob ON: PASS, 15 cases,
   every compared buffer bit-identical.
4. `step_glue_check` under `-D MOJOLEARN_STEP_GLUE_TRIAL=1`: PASS (names,
   norm forward and backward with reach, update with reach, step, refusal,
   rollback). The 16 trial arms are unchanged by the flip.
5. The byte LM binding, no trial define, knob ON:
   `byte_lm_step_glue_arm()` reads back `arm=optskip_noshadow_rows16
   trial=0`.
6. The same binding with no knob: `arm=shipped trial=0`.

GATES 5 AND 6 ARE THE EVIDENCE THE KNOB WORKS, AND BINARY HASHES ARE NOT.
Comparing the knob build against the no-knob build by sha256 looked like a
clean control and is worthless: a CONTROL REBUILD OF THE SAME CONFIGURATION
PRODUCED A DIFFERENT HASH, so this compiler's output is not byte
reproducible and a hash difference carries no information about the define.
The readback pair does: it is a runtime observation of a comptime constant
resolving two different ways, and `STEP_GLUE_SHIPPED_ROWS` and
`STEP_GLUE_SHIPPED_UPDATE` are derived from that same constant.

WHAT THE M4 DID NOT OBSERVE. The 16-thread launch itself. The reach clause
that names `threads_per_block=16` and `first_moved_row=32` is in
`step_glue_check`, which refuses a non-trial build by design, so the rows
geometry was directly observed only on the TRIAL build (on this M4 in gate
4, and on the H100 during the leg). On a shipped build the chain is
inferred, though every link is checked: the readback proves the default
resolves to the winner, `step_glue_rows_of` of that word is 16, the two
launchers call exactly those two functions, and gates 1 and 3 prove the
result is bit-identical with the branch compiled in.

WHAT IS NOT CLAIMED. The rows arms move the launch geometry of
`llama_rms_norm` and `bwd_rms_norm` for EVERY caller on an NVIDIA build,
including `training/samba_ops.mojo`, which the leg never timed. Bits are
safe there by section 4.1 (one token row per thread, the fold never leaves
the thread, `step_glue_blocks` never returns 0), and this is a schedule and
never a numeric term, but the PRICE on the samba path at 16 threads per
block is unmeasured. Separately, the flip makes dead code of the shipped
shadow copy and `identical_optimizer_step` inside `_byte_step_device` on
NVIDIA without deleting either; both stay compiled, and
`identical_optimizer_step` is exported and used elsewhere regardless.
`adam_update_oop_kernel` was already compiled into every build (`byte_lm.mojo`
imports it unconditionally); what changed is that a shipped NVIDIA build now
reaches it.
