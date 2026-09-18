# LANE STATUS: lane/lm-step-memory-build, paused 2026-09-17

Written for a session with NO CONTEXT. Branch `lane/lm-step-memory-build`,
off `origin/main` at `712eedd16`. Two commits, both pushed. Worktree
`~/mojolearn-wt/lm-step-memory-build`. Evidence
`~/mojolearn-evidence/lm-step-memory-build/`.

**NOTHING IN THIS LANE HAS RUN ON A GPU.** The Mojo compiles (both arms,
checked locally, see "What was verified"); the Python is tested on the host.
Every device number below is someone else's measurement and is attributed.

---

## 0. CONFIRMED: the unexplained 2.03x is the eager attention fallback

H100 sm_90a, pod 4tshk1f4ioag5k, commit 131568247, 2026-09-18, target shape
(162,147,840 parameters, B1 L2048 V50257), enwik8 100,000,000 bytes staged
from R2, 700 consecutive steps, one resident lean session.
Evidence: `~/mojolearn-evidence/lm-step-memory-build/eager_hypothesis_CONFIRMED.md`
and the filed leg directory.

`lane/lm-training-shakedown` measured a run stepping from 16.949 to 34.397 GB
of device memory and 0.207 to 0.44 s a step between steps 210 and 480, then
plateauing, and recorded the cause as NOT ESTABLISHED. It is the eager
attention fallback: the `[B, n_heads, L, S]` arrays growing one LAYER at a
time and never being released.

| step | s/step | device MB | layers grown fwd/bwd | `eager_bytes` |
|---:|---:|---:|---:|---:|
| 0 | 15.32 (setup) | 15,151 | 0 / 0 | 432 |
| 69 | 0.5275 | | 1 / 1 | 1,442,840,972 |
| 100 | 0.2057 | 16,431 | 1 / 1 | 1,442,840,972 |
| 201 | 0.2330 | | 2 / 2 | 2,885,681,512 |
| 219 | 0.2332 | | 6 / 6 | 8,657,043,672 |
| 300 | 0.3090 | 27,439 | 9 / 9 | 12,985,565,292 |
| 309 | 0.3918 | | 11 / 11 | 15,871,246,372 |
| 509 | 0.4386 | | **12 / 12** | **17,314,086,912** |
| 699 | 0.4646 | 31,535 | 12 / 12 | 17,314,086,912 |

**THE PREDICTION WAS WRITTEN DOWN BEFORE THE RUN AND IT LANDED ON THE BYTE.**
One layer's eager set is `3*B*H*L*S + L*S` floats forward (620,756,992 B) plus
`4*B*H*L*S + L*S` backward (822,083,584 B). Twelve layers is
**17,314,086,912 B**, registered before the leg started. The witness reads
17,314,086,912. Not close: equal. Every increment is 1,442,840,540 B, one
layer, and the 36 B shortfall against the formula is the nine one-element
placeholders that stop counting once a layer has grown.

**THE DEVICE AGREES INDEPENDENTLY.** 16,431 MB at one grown layer to 27,439 MB
at nine is 11,008 MB over eight layers, 1,376 MB a layer; one layer's eager
set is 1,376 MiB. Against the other lane's 17.448 GB step the sum is 0.77%
low, the remainder being allocator and context overhead that device-wide
sampling includes and a buffer-length sum does not. The step time lands too:
0.2057 s in phase 1 against their 0.207, 0.4646 s at the end against their
0.440 to 0.466 plateau.

**THE COMPETING ARITHMETIC IS REFUTED, NOT MERELY OUTSCORED.** Counting all
TEN arrays gives 19.730 GB, 13.1% ABOVE their step. `layers_full_aexp` reads
**12 at step 0**, `forward_aexp_bytes` = 2,415,919,104 = 12 x 201,326,592:
`aexp` is allocated full size from the first step by the DEVIATION 2652 exp
stash and was never part of the growth. Reporting it apart from the other
three is what made this legible; summed, step 0 would have read "2.4 GB of
quadratic attention stages held" and looked like a fallback that had already
happened.

**THE TRIGGER IS DATA DEPENDENT AND THAT IS ITSELF THE FINDING.** The
transition ran from step 69 to 309 here and from about 210 to about 480
there: same mechanism, different step. The step at which a run doubles its
memory and halves its speed is therefore NOT a property of the shape, so no
fixed step budget can be relied on and no three-step cell can see it.

**WHY IT COSTS 2.13x RATHER THAN THE EAGER PATH'S OWN SPEED.** On a
non-`FUSED_RAN` status the fused kernel has ALREADY run and been paid for,
and then `attention_eager_core` runs as well
(`transformer/impl/llama/modeling_llama.mojo:3588-3592`). The step pays both.

**WHAT IS STILL NOT MEASURED, AND IS NOT GUESSED AT HERE.** WHICH branch asks
for the eager path. The regime bound cannot be it (`regime_product_ok` is
`hd * a_max * b_max < 2^100`; a healthy run never approaches it). The
candidate in the source is `FUSED_CORNER`
(`transformer/impl/llama/fused_attention.mojo:1908-1910`), raised when an
output context cell has the exact bits of `-0.0` while its row range does not
cover the whole sequence. **This run does not show that.** It shows the
arrays grew; it does not name the branch. Separating "the arrays are grown"
from "the eager path ran this step" needs `stages.attn_materialized` on the
report, a one-line addition, and it is OWED. The trigger itself lives in
`lane/attention-speed`'s file.

## 1. The single most important thing this lane found

**Rank 7 is ALREADY LANDED. Do not build it.** The task brief said
`ce_refuse_device_inputs` still downloads the full logits matrix every step
(785 MiB at the target plus a 4.9 GiB transient). It does not. DEVIATION 2514
step 2 moved it onto the device: `training/checks/loss.mojo:1261`
`ce_refuse_device_inputs` now calls `device_first_nonfinite` and downloads
only `n_rows` int32 targets (8 KB at the target shape). Its own docstring at
`:1298-1305` says so: "The `(n_rows * vocab)`-float host mirror and its List
copy that DEVIATION 1495 paid (0.41 GB at the LM target, 264 ms per step
measured 2026-09-10) are gone." The optimizer refusal went the same way
(`training/checks/optimizer.mojo:105-112`, four device scans). The phase
timer is now `step.ce_refuse_scan`, not `step.ce_refuse_download`.

**The second most important thing: the memory wall is not where the brief
put it, and `lane/lm-training-shakedown` has already measured it.** Read
`bench/results/lm_training_shakedown_2026-09-17/MEASURED.md` and `GO_NO_GO.md`
on that lane's branch before doing anything here. Ground truth, H100 80 GB,
162,147,840 parameters, L2048, enwik8 from R2:

| batch | s/step | tokens/s | device peak |
|---|---|---|---|
| 1 | 0.20708 | 9,890 | 16.95 GB |
| 2 | 0.37504 | 10,922 | 26.61 GB |
| 4 | 0.65732 | 12,463 | 45.94 GB |
| 8 | 1.31258 | 12,482 | 84.33 GB |

`device_GB = 7.367 + 9.624 * batch`. Batch 8 FITS; batch 9 does not.
Throughput saturates at batch 4, and **batch 4 is the operating point**.
`bench/results/lm_capacity_2026-09-10/b8-l2048.json`'s 161.75 GiB with
`fit_admitted: false` is retired by `SUPERSEDED.md` in that directory.

So the ranked savings are worth a batch step or two, not a wall. They are
still worth landing (they matter on a 40 GB A100, at longer L, at a bigger
vocab, and they compound with batch) but nobody should describe them as
unblocking the target shape on an H100. It is not blocked.

---

## 2. Ranks: what is landed, started, and not done

| rank | what | state |
|---|---|---|
| 7 | device scan for the CE and optimizer refusals | **ALREADY ON MAIN** (DEVIATION 2514 steps 2-3). Nothing owed. |
| 3 | in-place aliasing of the CE chain | **WRITTEN, COMPILES, NOT RUN.** Commit `45c29dcfb`. |
| 1 | backward stage reuse across layers (2,432 MiB) | **NOT LANDED, NOT STARTED, AND UNOWNED.** Still one `LlamaBackwardStages` per layer (`training/byte_lm.mojo:610, 634, 641-643`) and `_pack_block` still runs in a separate loop after the backward loop (`:1040-1042`). The brief names the owner as "byte_lm.mojo (other lane)" with no lane; `DESIGN_lm_device_owned_step_2026-09-11.md:504-506` defers it explicitly. **It is the largest single item left and nobody has it.** |
| 2 | row-tiled loss and head (1,374 MiB at R=256) | **NOT STARTED.** Deliberately last per the task's order; it is the only item that touches workspace planning (`head_ws`/`head_bwd_ws` must be sized for chunk shapes; at R=128 the dA shape becomes SPLIT_64_4X4 with a 154 MB workspace, at R>=256 it stays TUNED). |
| 4, 5, 6 | parameter/gradient views, shared forward stages | not started, not asked for. |

### Rank 3 as built (DEVIATION 3011)

`training/byte_lm.mojo`, constructor of `ByteBuffers`:

- `ce_shift` is `logits.create_sub_buffer[DType.float32](0, M * V)`
- `ce_weights` and `ce_dlogits` are both
  `ce_expo.create_sub_buffer[DType.float32](0, M * V)`

Five `[M, V]` allocations become two: `3 * M * V * 4` bytes, **1,178 MiB at
B1/L2048/V50257 and about 9.2 GiB at the batch-4 operating point.** Trainer
side only; no kernel edited, no launcher signature changed.

Why the bits do not move is written in the code beside the change, not only
in the brief: each of the three kernels owns one CELL, loads its input cell
into a register and then stores, so writing back over the input is the same
arithmetic in the same thread (`ce_shift_exp_kernel` loss.mojo:660-668,
`ce_weights_kernel` :1020-1032, `ce_dlogits_kernel` :1034). `expo` is dead at
L14 and `weights` at L16 because the two backward kernels are enqueued back
to back on one in-order context.

**The brief's flagged level-2 risk is closed by a check, not a promise.** It
worried that the refused `SAB_NLL_*` arms read `logits` after `shift` has
overwritten it. `_require_profile` (`training/byte_lm.mojo:297-303`) raises
"byte LM: numerical sabotage build refused" for any build carrying
`ANY_LOSS_SABOTAGE`, which includes `SAB_NLL_VIA_ADDBACK` and
`SAB_NLL_VIA_LOG_W` (`loss.mojo:335-347`). A sabotage trainer cannot exist.
The loss lane's own gate fixtures pass five distinct buffers and are
untouched.

`-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1` restores the five separate
allocations. That define exists so the A/B has two arms;
`byte_lm_ce_aliased()` is the runtime witness, reachable from Python through
`byte_lm_session_info`'s binding sibling `byte_lm_ce_aliased`.

---

## 3. The binary checkpoint: the codec at DESIGN:165 was NOT built, and now it is

**Answer to the question as asked: `DESIGN_lm_device_owned_step_2026-09-11.md`
lines 163-166 do NOT design a binary codec.** They design
`export_checkpoint(path)` over the EXISTING JSON/hex codec with
`save_checkpoint` as an alias, and then concede in as many words that at the
target shape it refuses and "`export_state()` arrays are the checkpoint
path". Everything in that bullet is built EXCEPT that last clause: there was
no writer, no reader, no format and no gate for those arrays anywhere in
Python.

What exists in the tree, and why neither piece covered the byte LM at scale:

1. `training/checkpoint.mojo` (1,436 lines) IS a real byte-exact binary v1:
   magic `MLCKPT01`, version, little-endian by construction, two FNV-1a64
   hashes, param + m + v + flags + optimizer descriptor + step `t`. Spec at
   `training/CHECKPOINT_FORMAT.md`, gate at
   `training/checks/checkpoint_check.mojo`. **It declares itself never
   compiled and never run** (`training/checkpoint.mojo:3-11`), it is imported
   by no binding (`grep -rn checkpoint bindings/` finds only prose), and
   `byte_checkpoint` at `training/byte_lm.mojo:1309` is not in the binding's
   import list. **Zero Python reachability.** It also carries no model shape,
   so two architectures with the same flat count are indistinguishable to it.
2. `python/mojolearn/_byte_lm_impl.py` JSON/hex, capped at 2 MiB
   (`_CHECKPOINT_LIMIT` at `:70`, raising at `:1043`, `:1055`, `:1103`). Hex
   doubles the payload, so the cap is about 87,381 parameters.
   `lane/lm-training-shakedown` confirmed on the H100: "the path
   export_checkpoint refuses above 87,381 parameters".

**What this lane built (commit `3e95eb7af`):
`python/mojolearn/_byte_lm_checkpoint.py`**, schema
`mojolearn.byte-lm-stream.v1`, modeled on the shipping
`python/mojolearn/_samba_checkpoint.py`: magic, a bounded canonical JSON
header with its own SHA-256, four raw little-endian arrays each with its own
SHA-256, atomic temp-and-replace, and reads that verify magic, header digest,
every array digest and the exact file size before allocating anything. Array
lengths come from the admitted registry (`state_shape`), never from the file.
12 bytes per parameter instead of the envelope's 24, so about 1.95 GB at the
target shape.

Surface: `LanguageModelTrainer.export_checkpoint_binary(path)` and
`LanguageModelTrainer.from_checkpoint_binary(path, *, resident=False)`. The
JSON envelope, its bytes, its bound and all its refusals are UNTOUCHED; this
is a second named format beside it.

The moments are in the file, and that is the point:
`lane/lm-training-shakedown` measured on 2026-09-17 that a restore dropping
`m` and `v` produces the SAME loss and the SAME gradient on its first resumed
step and a DIFFERENT update, so the divergence surfaces a step later looking
like nondeterminism. **That lane also already proved bit-exact resume at
162,147,840 parameters through `export_state()` + `load_state_dict`** (save
11.19 s, restore 7.63 s, every tail-step witness matched, missing-moments
control separated). What was missing was only the durable file, and the
process-boundary and cross-vendor legs of that claim are still owed here.

---

## 4. DEVIATION 3010: the eager-fallback witness, and the hypothesis it exists to settle

`lane/lm-training-shakedown`'s 2,000-step run found a regime change nobody
has explained: somewhere between steps 210 and 480 the device memory steps
from 16.949 GB to 34.397 GB (2.03x) and the step from 0.207 s to 0.44 s
(2.13x), and both then plateau. Not clocks, not thermal, not power, not host
CPU, not host memory. Their `GO_NO_GO.md` records the cause as NOT
ESTABLISHED.

**HYPOTHESIS, NOT A CONCLUSION: it is the eager attention fallback.**
`BRIEF_lm_step_memory_2026-09-10.md` section 1.5 predicted exactly this
shape. Ten `[B, n_heads, L, S]` arrays are allocated at ONE element under
`lean=True` and grow on demand
(`ensure_attention_stage_capacity`, `transformer/impl/llama/modeling_llama.mojo:1726-1738`;
`ensure_backward_attention_capacity`, `transformer/checks/transformer_backward.mojo:1929-1941`),
the growth is data dependent and therefore step dependent, and once grown
they are **never released while the session lives**. The trigger that is
plausible here is not the regime bound (`regime_product_ok` is
`hd * a_max * b_max < 2^100`, unreachable in a healthy run) but
**`FUSED_CORNER`**: the fused forward raises it when an output context value
has the exact bits of `-0.0` while the row range does not cover the whole
sequence (`transformer/impl/llama/fused_attention.mojo:1908-1910`), which is
data dependent and gets likelier as the model trains. On a corner the caller
runs the eager path as well, which both grows the arrays and costs the extra
time.

Rough arithmetic, NOT a measurement: at B1, H12, L=S=2048, one layer's
forward eager set is 4 x 201.33 MB + 16.78 MB = 822 MB and the backward set
is the same, so twelve layers in both directions is 19.7 GB against a
measured delta of 17.45 GB. If `aexp` were already full size from the
DEVIATION 2652 exp stash, the remaining growth is about 17.32 GB, which is
within 0.8% of the measured delta. **That closeness is suggestive and is not
evidence.** It is a reason to run the witness, not a reason to believe it.

So the witness was built rather than the argument continued:

- `byte_attention_eager_cells()` (`training/byte_lm.mojo:667`) sums the
  buffer lengths the trainer already owns and returns
  `[forward_eager_cells, forward_aexp_cells, backward_eager_cells,
  layers_grown_forward, layers_grown_backward, layers_full_aexp]`.
- `byte_lm_session_info` appends those six integers (positions 4 to 9);
  positions 0 to 3 are unchanged and existing callers are unaffected.
- `LanguageModelTrainer.attention_stage_report()` returns them as a dict with
  byte counts, or `None` against a binding that predates this.
- `aexp` is reported APART from the other three because the DEVIATION 2652
  exp stash grows it alone on a call that refused nothing; summing them would
  read a healthy stash as a fallback.

Host metadata only: nothing launched, downloaded or synchronized, and no step
path calls it.

**If this hypothesis is right, it is worth more than every rank in the brief
combined** (17.4 GB and 2.13x at the operating point, against rank 3's
1.15 GiB), and it is one long run away from being settled instead of argued.

---

## 5. What was actually verified, and how

### Binary checkpoint (host, M4, one core, nice 19)

`python/mojolearn/tests/test_byte_lm_checkpoint_binary.py`, 11 tests, pass.
322 byte-LM and host-surface tests pass with them
(`.pixi/envs/test/bin/python3 -m pytest mojolearn/tests/ -k "byte_lm or host_surface"`
from `python/`; the binaries were COPIED out of the shared checkout into the
worktree, never rebuilt into it, and removed afterwards).

**Nine sabotage arms, each run against the unfixed side and watched failing**
(`~/mojolearn-evidence/lm-step-memory-build/checkpoint_sabotage.log`): magic,
header digest, array digest, file size, descriptor shape, header bound,
numeric mode, save-through-decimal, load-zeroes-one-cell. Each separated a
named test.

**Two arms came back BLIND on the first pass and changed the TEST, not the
code.** Both are recorded because they are the reason the gate is worth
anything:

- the numeric-mode test matched `profile/registry/mode`, which is also
  `_validate_state`'s spelling, so it passed with the codec's own check
  deleted. It now matches `checkpoint profile/registry/mode`.
- the round-trip test started from an all-zero parameter vector, so a
  sabotage that zeroed one cell round-tripped clean. It now starts from a
  non-uniform vector and both value arms separate it.

**REPORTED INERT:** the final one-byte read at the end of `load()` is
unreachable given the `fstat` size check; no sabotage of it fails a test. It
is kept because the Samba codec keeps it, and it is named here rather than
claimed as covered.

### Rank 3 and the witness, MEASURED on the Apple column (M4, Metal, one core)

Run 2026-09-18 under `mac_slot.py metal` at shape `2 32 256 4 2 64 64 2 256`
(head_dim 64, so the FUSED path), five steps, witnessed every step.
Evidence: `~/mojolearn-evidence/lm-step-memory-build/ab_sabotage.log` and
`metal-smoke*/verdict.json`.

**DEVIATION 3011, the CE aliasing A/B: PASSED, and watched failing.**

| check | clean | sabotaged |
|---|---|---|
| `ce_arms_are_two_arms` (`ce_aliased` true vs false) | PASS | PASS |
| `ce_bits_unmoved` (5 steps, six hashes each) | PASS, 0 differing | **FAIL, all 5 steps differ on `loss`** |

The sabotage aliased `ce_shift` onto `ce_expo` instead of `logits`, so
`ce_shift_exp_kernel`'s `expo` store lands on its own `shift` store at the
same cell and L6/L7 reads `exp(s)` where it must read `s`. Loss moved at
every step; the gradient did not, which is right: `expo` still ends up
correct, so only the nll seam is wrong. That is a precise separation and it
is the reason this arm is worth having.

**DEVIATION 3010, the eager witness: it discriminates.**

| arm | layers grown fwd/bwd | `eager_bytes` |
|---|---|---|
| `--attention-path fused` | 0 / 0 | 72 |
| `--attention-path eager` | 2 / 2 | 475,136 |

72 bytes is 18 one-element buffers: nine per layer, two layers, exactly lean.
The two arms' losses are bit-identical, which is the IDENTICAL contract
holding and also proves both arms really ran.

**A SHAPE THAT MADE THE WITNESS BLIND, found here and not on a rented box.**
At `head_dim = 8` the fused kernel is not instantiated at all
(`fused_supported_head_dim` admits 16, 24, 64 and 128 only,
`fused_attention.mojo:1542-1554`), so the "fused" arm falls back to eager and
BOTH arms read every layer grown. A witness that reads the same in both arms
is not a witness. The leg's control and target shapes both use head_dim 64.

**The binary checkpoint, through the device this time.** `ckpt-save` (3
steps) then a separate process `ckpt-resume` (+2): `resume_matches` PASS,
both tail steps equal the uninterrupted run hash for hash. The
`--drop-moments` control SEPARATES exactly where the arithmetic says it must:
the first resumed step differs on `parameters`, `m` and `v` while `loss` and
`gradients` still MATCH (the parameters were restored and the moments only
enter the update), and by the second step `loss` and `gradients` differ too.

**REPORTED INERT, not passed.** A second sabotage moved `ce_dlogits` to a
view at offset V inside an oversized backing buffer, meaning to shift the
gradient by one row. `ce_bits_unmoved` still passed. The arm is inert BY
CONSTRUCTION: `byte_lm.mojo` hands the SAME `ce_dlogits` field to the writer
kernel and to both reader GEMMs, so a constant offset on that one view is a
relabelling both sides agree on. Moving the gradient needs two DIFFERENT
views, which is a second field and a call-site edit, not a build flag. The
gradient, parameter, `m` and `v` hashes are nonetheless live in the
comparator, because the `ckpt-control` arm above was watched separating on
exactly those names. `flags` has never been observed differing and is
therefore NOT COVERED, which is said rather than counted.

### Mojo (compile only, M4, one core, nice 19, no GPU, no Metal lock taken)

Both arms of DEVIATION 3011 COMPILE for the Apple column, from the shared
checkout's pixi env against this worktree's sources, into scratch output
directories:

- clean: exit 0, only pre-existing warnings
- `-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1`: exit 0, `_mojolearn_byte_lm.so`
  4,068,840 bytes

A compile is not a run. No bit has been compared.

### Measured device peak / host RSS obtained by this lane

**NONE.** Every device figure in this document is
`lane/lm-training-shakedown`'s. The before/after peaks this lane owes are the
first thing the leg below produces.

---

## 6. THE EXACT NEXT STEP

Everything is committed and the leg body is written. The next session runs
ONE H100 leg and reads one file.

```sh
cd ~/mojolearn-wt/lm-step-memory-build
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_ce_alias_body.sh \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 75 \
    --gpu "NVIDIA H100 80GB HBM3"
```

Run it from THIS worktree: the leg ships `git archive` at `HEAD` and refuses
a dirty tree under `--rent`. `lane/r2-opponent-hygiene` (merged, `744f44945`)
made opponent installs opt-in behind `--opponents`; this body installs none.

Then read `<leg out>/remote/lm-ce-alias/verdict.json`. Six checks, each
naming what would have made it fail:

| check | passes when | and would have failed if |
|---|---|---|
| `ce_arms_are_two_arms` | aliased reports `ce_aliased: true`, unaliased `false` | one build was compared with itself, which reads exactly like a passed identity gate |
| `ce_bits_unmoved` | every step's loss/gradients/parameters/m/v/flags hash equal across the arms | aliasing moved a bit |
| `ce_memory_moved` | the aliased arm's device peak is LOWER | the aliasing never reached the allocator, whatever the source says |
| `eager_witness_moves` | zero layers grown under `--attention-path fused`, EVERY layer grown under `eager` | the witness reads zero in both arms, i.e. it is blind |
| `resume_matches` | the 162M binary-checkpoint resume's tail equals the uninterrupted tail | the file does not carry the state |
| `resume_control_separates` | the moments-dropped control DIFFERS | the comparison is not reading `m` and `v`, and `resume_matches` proves nothing |

`tools/lm_ce_alias_body.sh` builds the byte LM binding twice (clean, then
with the unaliased define) via `MOJOLEARN_BUILD_EXTRA_DEFINES`, runs five
steps at the target shape against each, runs the two attention arms at the
control shape, and does the checkpoint save / resume / drop-moments control
across three separate processes. `tools/lm_ce_alias_probe.py` is the probe and
`tools/lm_ce_alias_compare.py` writes the verdict.

**Do not read the aliasing result from a `.so` digest.** Two builds of
identical source always differ, and a digest does not prove a define; the
witness is `byte_lm_ce_aliased()` read from INSIDE the process that loaded
the binary, which every `result.json` carries.

File the fetched directory under `bench/results/e1g/` with its
`extra_body.sh` copy. The 1.95 GB checkpoint is deleted on the box; its
digest and size come home in `status.txt` and `ckpt-save/checkpoint.json`.

### After that leg, in order

1. **Tell `lane/lm-training-shakedown`** what `eager_witness_moves` says, and
   ask them to call `attention_stage_report()` each step in their long-run
   body. That is the whole cost of settling their 2.03x.
2. **Rank 1** (2,432 MiB, the largest item left, unowned). One
   `LlamaBackwardStages` reused across layers with per-layer `d_x`, and
   `_pack_block` moved inside the backward loop.
3. **Rank 2** (1,374 MiB at R=256), last, because it is the only one that
   touches workspace planning.

## 7. Carried forward so nobody rediscovers it

- **Gradient accumulation is NOT missing.** `accumulate_grads` and
  `accumulation_is_aligned` at `python/mojolearn/_training_impl.py:1667,1677`,
  exported through `training.py`, used by `_samba_impl.py` and
  `parallel_training.py`, covered by four test files, reached by
  `tools/identity_break.py` as the `batchgrad` part. The claim that it was
  missing came from `archive/docs/lanes/HANDOFF_lm_capacity_2026-09-10.md:92`,
  which is ARCHIVED and superseded. `lane/lm-training-shakedown` corrected the
  same claim independently (`764f17ac7`).
- **The 2 MiB cap is real**: `_CHECKPOINT_LIMIT` at `_byte_lm_impl.py:70`,
  raising at `:1043`, `:1055`, `:1103`.
- **The analytic capacity json is retired**, not merely loose: it said
  161.75 GiB and `fit_admitted: false` for batch 8, the measurement is
  78.5 GiB and batch 8 fits.
  See `bench/results/lm_capacity_2026-09-10/SUPERSEDED.md` on
  `lane/lm-training-shakedown`.
- **DEVIATION numbers used by this lane: 3010 and 3011.** The highest in the
  tree before them was 3003.
