# Decode on one runtime path, the Mamba3/Samba door, the GLM tile order and the noise cells

Ideas lane `ideas-decode-door-glm` (2026-11-08), read-only: code and history on origin/main `ce65fede31`, the
diagnosis `docs/plans/REGRESSIONS_0866_VS_0855.md` and `bench/results/paper_compare/REGRESSIONS-amd.txt`
(AMD MI300X, 0.8.55 -> 0.8.66, our arm, same explicit parameters). Nothing was run, compiled or measured.
File:line references are on main unless they say 0.8.55 (`4e927c92495`).

## 0. One screen

1. **Decode on the executor (Andrew's first question): yes, and it is the right end state, but not the first
   step.** The autograd executor (`autograd/ops_api.mojo` `OpsSession`, `autograd/program.mojo` captured programs,
   `autograd/program_control.mojo` bounded loops with loop-carried state, inference mode `tape.no_grad`) already does
   what a decode loop needs from a runtime: parameters uploaded once and read in place, inputs cast and uploaded in one
   call, outputs downloaded once, a `while_loop` over a body program with carried state tensors, no recording. It lacks
   three things for decode: (a) block-level *step ops* (a Mamba1/2/3 L = 1 step over a resident SSM state, an attention
   step over a resident KV cache) -- the tape never launches a Mamba or attention forward today, it only records
   backward rules over hand-launched stage buffers (`autograd/rules_mamba3.mojo:1-30`, `rules_samba.mojo:1-25`,
   `rules_transformer.mojo`); (b) *in-place carried state*: a program parameter is read in place but only the optimizer
   writes one in place (`ops_api.mojo:954`), and `while_loop` carries state by one `OP_COPY` plus a sync per state tensor
   per iteration (`program_control.mojo:57-85`), fine for a 393 KB SSM state, wrong for a KV cache; (c) a token index
   register (the `LOOP count` of `docs/plans/AUTOGRAD_API.md:440-470` is a design note, not implemented). The resident
   decode sessions that already exist (`Mamba1/2/3DecodeSession`, `TransformerDecodeSession`) are exactly the step-op
   bodies: `_m3_session_step_run` (`mamba/impl/mamba_shell_impl.mojo:1514-1541`) is one x upload, one block launch
   set, five downloads, no weight or state traffic. So the order is: make those sessions the public and board path for
   every family now (S), put a native N-token loop over them (M), then lift them onto the executor as step ops with
   in-place state so forward, decode and training share one path (L). Per-door caching is the fallback only.
2. **The mamba3-decode and samba-decode recipes never touch a resident session.** The board builds a `decode_session`
   only for transformer and mamba1 (`tools/bench_board_neural.py:2387-2401`); Mamba2/3 run `block.step(xt, state)` per
   token (:2463-2466) because `Mamba3Block.decode_session` is private (`_decode_session`, `_mamba_impl.py:1477`, kept
   private since `8e8516f6a3`, 2026-09-22, "until its lane is recorded", `_RESIDENT_SESSION_NOTE` :969-998). Samba's
   `step` drives the stack layer by layer per token (`bindings/samba_inference_native.mojo:270-345`), each Mamba3 layer
   through `_m3_state_call` (:105-180) and each attention layer through the per-call `neural_block_forward_binding`.
   The device half of the per-call Mamba3 path is byte-for-byte the 0.8.55 op list (diffed: 9 weight uploads, 10 state
   uploads, stage allocation, x upload, block, 15 downloads, one sync, `mamba_shell_impl.mojo:1122-1298` against
   0.8.55 `bindings/_mojolearn_mamba.mojo`), so the +1.6 ms per token is the 0.8.66 host door; the exact line is not
   settled by reading (section 2.3 lists every per-token host step with its line) and one phase-timer run is owed.
   The same door tax at session-step size is visible on transformer-decode: (2699 - 2445) ms / 2048 = 0.124 ms per
   token through `transformer_decode_run` (`providers/python/transformer_decode_pyapi.mojo:596-640`), so 1.10x there
   is probably real, not noise.
3. **C6 is misattributed.** An IDENTICAL single-device `C == 1` fit never reaches `linear_bwd`'s tile-order arm:
   `qn_tiled_applies(D, C)` is `C == 1` under `QN_TILED` (`glm/impl/qn/glm_base.mojo:479,562-564`), so `evaluate`
   takes `enqueue_tiled` (:2106, :2127, :2149), and the `C == 1` branch of `linear_bwd` that `4b8a8e8f2b` moved to
   `qntm_tiled_into` (:1752-1764) is reached only with `QN_TILED` off (the tombstone says so:
   `docs/TOMBSTONES.md:430-440`). The only changes on the measured path since 0.8.55 are a `dplan.single()` string
   test per evaluation (:2526-2528, `core/device_plan.mojo:395-400`) and the workspace bound (:1912-1913), neither of
   which costs 100 ms. The LinearSVR deltas (+102 ms on 449.5M elements, +6.3 ms on 44M) scale like one more pass over
   X at 17-29 GB/s (a host memcpy or a second upload), and the cell's clock is fit plus `predict(Xq)`
   (`tools/bench_board_more.py:1270-1276`). The A/B with `experiments/removed/MOJOLEARN_QN_TILED_MULTI_OFF.patch` is
   still cheap and is kept as the confirming arm, but the lane must first find the pass over X.
4. **Noise: both arms are one timed round after one untimed warm-up.** The paper board ran `rounds: 1`
   (`bench/results/bench_board/main-amd-mi300x/board.json` config), the compare ran `rounds: 1`
   (`tools/paper_compare.py:242`), the board default is 1 (`tools/bench_board.py:128`, `bench_board_neural.py:6437`).
   No stored A/A floor exists for the neural lanes on AMD; the six-lane grid falls back to a 5% default
   (`tools/six_lane_grid_decide.py:111-113`). One repeat is therefore not enough to call 1.1-1.2x noise: the rule is
   one A/A pair per wheel (new wheel twice, old wheel twice, interleaved, same box, same session); a cell is noise only
   when the A/A spread of either wheel covers the A/B delta. transformer-decode has a mechanism (item 2); gemm-bf16,
   gemm-int8 and lm-forward have none found and get the A/A before any code work.

Bits: nothing in this plan changes the bytes a kernel reads or the fold orders; identity is unchanged by every lane.

## 1. Decode as a forward-only program on the executor

### 1.1 What the executor is today (file:line)

- `OpsSession` (`autograd/ops_api.mojo`): one `Tape` over one `Exec` (host or device), `record=False` is an inference
  session (`ops_api.mojo:340,360`, `tape.no_grad()` `autograd/tape.mojo:424`: "`_record` keeps no parents, allocates no
  gradient storage ... a recorded and a non-recorded forward give the same bits by construction",
  `docs/plans/NATIVE_AUTOGRAD.md` "Inference").
- Captured programs (`autograd/program.mojo:1-40`): `capture(inputs, outputs)` keeps the logged ops; parameters are
  "a session tensor made by `upload` ... read IN PLACE at each run, so an optimizer step between runs is seen by the
  next run"; `run_program` (`ops_api.mojo:2036`) uploads the inputs through `CastIn` descriptors into a scratch tape
  whose storage is reused run after run, replays every node through the same session entry, downloads the outputs:
  one binding call per run. Ids inputs exist (`PROGRAM_IN_IDS`, `embedding`/`take` with an ids tensor).
- One executor per device (`autograd/program_executor.mojo:1-40`): the distributed and partitioned programs compose
  these.
- Control flow (`autograd/program_control.mojo:1-200`, Python `Session.while_loop(*arrays, condition, max_iterations)`
  `python/mojolearn/ops.py:1386` and `run_if` :1370): a body program, a condition program, a bound; the body's outputs
  are the next iteration's inputs. `_copy_state` (:57-85) copies each loop-carried state tensor with one `OP_COPY` and
  one `sync` per iteration; `_predicate` reads one device scalar.
- Ops: matmul, elementwise with broadcast, softmax, losses, embedding, data (`permutation`, `gather_rows`), transformer
  primitives (`OP_TFP_RMS`, `OP_TFP_ROPE`, `OP_TFP_LINEAR`, `OP_TFP_SOFTMAX`, `autograd/transformer_primitives.mojo`),
  CNN and rank-n ops, MoE routing (`route_ops.mojo`). No attention-with-cache op, no Mamba step op.
- Mamba3, Samba and transformer on the tape are *backward* rules over the hand forward's stage buffers: "The forward is
  the hand path's (`mamba3_block_forward`, stages recorded); the tape records nodes over those stage buffers and
  relaunches nothing" (`autograd/rules_mamba3.mojo:28-30`); the Samba train step records one tape over the resident
  session (`rules_samba.mojo:1-25`, `training/samba_resident.mojo`).

### 1.2 Which families already run forward through a resident path (the diagnosis's faster lanes)

| lane | 0.8.66/0.8.55 | path | on the tape? |
|---|---|---|---|
| program-run | 0.66x | `ops.Session.capture` + `Program.run`: one native call, parameters resident (`program.mojo`) | yes (inference session) |
| sequential-predict | 0.64x | `mlp_predict_autograd` (`sequence/mlp_fit.mojo`, `NATIVE_AUTOGRAD.md` "Inference") | yes |
| mlp-train-step | 0.29x | `SmallMLPTrainer`, one native call per method, resident device state (`bindings/small_mlp_native.mojo`) | no (hand resident) |
| transformer-forward | 0.72x | `_neural_plain` `forward_session` (`bindings/neural_block_public.mojo:647-720`) | no (hand resident) |
| samba-train-step | SAME | `samba_resident_train_step` forward hand-launched, backward on one tape | backward only |
| transformer-decode | 1.10x | `TransformerDecodeSession` (`_transformer_impl.py:1192`, `providers/python/transformer_decode_pyapi.mojo:596`) | no (hand resident) |
| mamba1-decode | n/a | `Mamba1DecodeSession` (`_mamba_impl.py:598`, `mamba_public_native.mojo:712-930`) | no (hand resident) |
| mamba2/3-decode, samba-decode | 2.17x, 1.57x | per-call door, no session on the recipe | no |

The lesson of the table: every lane that got faster replaced per-call host work with a resident device state behind
one native call; the two decode lanes that regressed are the two whose public path has no session.

### 1.3 What the executor lacks for decode, and what the per-layer door does that the executor already does

| need | executor today | per-layer door today (Mamba3 step) |
|---|---|---|
| admit the block once | parameters uploaded once (`program.mojo:15`), shapes fixed at capture | `neural_block_snapshot_prepared` per token (`neural_block_public.mojo:1031-1070`): 44 field reads, identity-key match, array namespace walk (`native_identity_key.mojo:261-300`) |
| weights resident | yes | `_m3_load_weights` per token (`mamba_shell_impl.mojo:1092-1106,1166`): 9 uploads, about 4 MB at d_model 384 |
| input admission | `CastIn` upload at run start (`autograd/cast_ops.mojo`) | `_neural_input` -> `neural_block_f32` copies a NumPy x into a zero-filled `generation_array` per token (`neural_block_admission.mojo:404-412`) |
| carried state | loop-carried tensors by copy + sync per iteration (`program_control.mojo:57-85`); no in-place write op except the optimizer (`ops_api.mojo:954`) | 10 state pieces snapshotted twice per token (`_mamba_state_fields` `mamba_public_native.mojo:105-124`, `_mutable` `generation_buffers.mojo:325-360`), uploaded and downloaded per token (`mamba_shell_impl.mojo:1232-1250,1283-1297`) |
| step op | none for Mamba1/2/3 or attention-with-cache; `OP_TFP_*` cover RMS, RoPE, linear, softmax of a prefill | the session bodies exist: `_m3_session_step_run` (`mamba_shell_impl.mojo:1514-1541`), `transformer_decode_session_step_binding` (`bindings/_mojolearn_transformer.mojo:1045`), Mamba1/2 session steps (`mamba_public_native.mojo:838-900`) |
| N tokens per call | `while_loop` with a bound; no iteration register (design note `AUTOGRAD_API.md:440-470`) | one Python call per token per block; Samba one call per token per stack |
| recording off | `OpsSession(ex, False)`, `tape.no_grad()` | n/a |
| output | downloaded once at the end of the run | 5 zero-filled report arrays per token (`samba_inference_native.mojo:142-146`), 15 downloads, one sync |
| training reuse | the same tape, recording on | `rules_mamba3` records the hand forward's stages; the resident weights of a step op would be the same buffers the training forward reads |

### 1.4 The answer and the order

Decode can run as a forward-only program on the executor, and should: one runtime path for forward, decode and
training, admission paid once per model, the per-layer native door gone from the inference path. What makes it a
three-step program rather than one lane is that the executor has no block step ops and no in-place carried state,
while the resident hand sessions already hold exactly the per-token device work a step op must do. So the step ops
*are* the sessions, lifted:

1. **Lane D1 (S): sessions on every public decode path.** Make `Mamba2Block.decode_session` and
   `Mamba3Block.decode_session` public (rename the underscore doors and classes back, re-export from `mojolearn.mamba`,
   as `_RESIDENT_SESSION_NOTE` prescribes), and give `SambaStack` a `decode_session(state)` that holds one resident
   session per layer (Mamba3 and transformer) plus the embedding, final norm and head weights on the device, with the
   hidden state passed device to device (today each layer goes through a new host array, owed row N094). Change the
   board recipes for mamba2-decode, mamba3-decode and samba-decode to open the session before the clock and call
   `session.step` per token, exactly as transformer-decode and mamba1-decode do (`bench_board_neural.py:2387-2401`),
   and update `LANE_TEXT` (:980-983). Bits: a session step is the same `mamba3_block_forward(..., L=1)` on the same
   bytes; the note's own test `python/mojolearn/tests/test_mamba23_decode_sessions.py` holds it to the per-call step
   byte for byte. The identity lanes the note asks for (`mamba2-decode-session`, `mamba3-decode-session` in
   `tools/identity_break.py`) are added in the same lane and recorded at the one pre-release verify.
2. **Lane D2 (M): N tokens per native call on the sessions.** One entry `decode(x_or_ids, state, n_tokens)` on the
   Mamba, transformer and Samba session handles: the host validates once, then a `# driver-loop: over tokens` runs
   the session step bodies with the input slab uploaded once (B x N x d_model, or B x N ids), the per-token outputs
   left in one device output slab, one download and one sync at the end. The session's five report downloads per
   token (`mamba_shell_impl.mojo:1533-1537`) move to the end as well (one download per report, the last token's).
   This is the N-token loop the executor's `LOOP` note describes, done on the hand sessions first because they exist.
3. **Lane D3 (L): the step ops on the executor, in-place state, one path.** Add `OP_*_STEP` nodes whose bodies are
   the session steps (Mamba1/2/3 over a `Mamba3DeviceState`-shaped session tensor set, attention over a KV-cache
   tensor pair), an in-place state write rule on the tape (the `*_into` family already writes caller storage:
   `unary_into`, `binary_into`, `custom_into`, `NATIVE_AUTOGRAD.md` "Caller storage"), and an iteration register for
   `while_loop` so the body can `take` token t from the input slab. `Program.decode(ids, state, n)` is then
   `while_loop` with recording off; `train_step` is the same program with recording on, and `rules_mamba3` /
   `rules_transformer` record the step op's stage buffers as they record the hand forward's today. Weights are the
   program's parameters, uploaded once, read in place by decode and updated in place by the optimizer: the training
   step stops re-uploading weights per block call as well.

**Per-door caching is the fallback only.** A `NativeBlockPlan`-style cache of the Mamba3 plan removes the per-token
snapshot matching and entry lookup (roughly 0.1-0.3 ms by reading) but keeps the 4 MB weight upload, the 10 state
uploads, about 25 device allocations and frees, 15 downloads and the sync per token, and buys nothing for Samba's
attention layers or the transformer. It is what to do if D1 is blocked (it is not: the sessions exist and are tested).

## 2. The mamba3-decode and samba-decode call paths and the per-token cost (question 1)

### 2.1 mamba3-decode

1. `tools/bench_board_neural.py:2387-2401`: `decode_state` is built only for `transformer` and `mamba1`, so only those
   get `decode_session`; `:2463-2466`: Mamba2/3 run `state = allocate_state(B)`; `step(xt, state)` per token.
2. `python/mojolearn/_mamba_impl.py:1457-1475` `Mamba3Block.step` -> `mamba_native_call("step_public", (self, x,
   state))` (binding module `_mojolearn_mamba` via `_native_neural_bridge.py:26`).
3. `bindings/mamba_public_native.mojo:1121` `mamba_native_call`: `_mamba_generation_ready` (:1104), then :1179-1189
   `mamba3_forward(...)` and four `py_setitem` of the report fields.
4. `mamba3_forward` :333 -> `neural_block_snapshot_prepared` (`neural_block_public.mojo:1031`: `neural_block_fields`
   over 44 names `neural_block_admission.mojo:163-209`, `prepared_get`, `_neural_snapshot_matches` :1020 with
   `arrays_match` `native_identity_key.mojo:261`) -> `_mamba3_forward_captured` :289: `_neural_input`
   (`neural_block_public.mojo:467`, NumPy x copied by `neural_block_f32` `neural_block_admission.mojo:404-412`),
   `_m3_entries` :171 (`PyType_IsSubtype`, `neural_training_entries` `neural_block_public.mojo:835`),
   `_mamba_state_fields` :105 (12 names, 10 `neural_array_snapshot`s), `_m3_state_call`.
5. `bindings/samba_inference_native.mojo:105-180` `_m3_state_call`: 9 `generation_weight_address`, 10 `_mutable`
   (a second snapshot per piece, `generation_buffers.mojo:325-360`), 5 zero-filled `generation_array`s (output and
   four reports, `generation_buffers.mojo:152-190`, `native_array_zeros`), `py_call(mamba3_decode_step, 25 addresses,
   4 words)`, `_writeback`, two `py_setitem`.
6. `bindings/_mojolearn_mamba.mojo:989` `mamba3_decode_step_binding` -> `mamba/native_state_dispatch.mojo:9` ->
   `mamba/impl/mamba_shell_impl.mojo:1122` `_mamba3_run`: `_m3_load_weights` (:1092-1106, 9 uploads), `_mamba3_run_on`
   :1216: `Mamba3DeviceState` plus 10 state uploads (:1232-1250), `Mamba3DeviceStages` (:1252), x upload,
   `mamba3_block_forward`, 15 downloads, `ctx.synchronize()` (:1283), buffer destruction. Kernels:
   `mamba/impl/modules/mamba3.mojo`.

### 2.2 samba-decode

1. `bench_board_neural.py:2549-2552`: `state = allocate_state(B, L)`; `model.step(ids_t, state)` per token.
2. `python/mojolearn/_samba_impl.py:538` -> `samba_native_call("step")` -> `bindings/samba_public_native.mojo:51` ->
   `samba_forward(..., step=True, ...)`.
3. `bindings/samba_inference_native.mojo:270-345`: `_samba_inference_ids`, `generation_array` hidden, embedding launch,
   then `# driver-loop: over native model layer execution owners`: an attention layer calls
   `neural_block_forward_binding` (`neural_block_public.mojo:829` -> `neural_block_execution` :253 (owner slot) ->
   `_neural_plain` :647 -> `_neural_layer_run` :423 -> `causal_layer_run_fields` `causal_lm_public.mojo:764`, weights
   by host address per token); a Mamba3 layer calls `_m3_state_forward` :182 -> `_m3_state_call` (2.1 step 5, without
   the snapshot). Each layer's output is a new host array (download then upload: owed row N094). Then the final norm
   and the head launches, each from a host array.
4. The resident Samba path (`_samba_resident_forward` :193, `training/samba_resident.mojo`) exists for stateless
   `forward` and `train_step` only; `step` never reaches it.

### 2.3 Where the ~1.6 ms per Mamba3 layer call goes (reading; medium confidence on the layer, low on the line)

The device op list of the per-call path is unchanged since 0.8.55 (section 0, item 2), and 0.8.55's
`_mamba_impl.py:1189` handed the 25 addresses to `ext.mamba3_decode_step` with NumPy-level checks only. So the
delta is the 0.8.66 host door. Per token, by reading:

| step | lines | per-token work | reading estimate |
|---|---|---|---|
| generation registry check | `mamba_public_native.mojo:1104-1118` | dict-watcher test | microseconds |
| block admission | `neural_block_admission.mojo:163-209` | 44 field reads, `neural_plain_value` on about 40 scalars | tens of us |
| snapshot match | `neural_block_public.mojo:1020-1046` | identity key over 6 names at depth 1, `arrays_match` walks the Array class namespaces (`native_identity_key.mojo:261-300`), dict copy and update | tens to 100+ us |
| x admission | `neural_block_public.mojo:467-494`, `neural_block_admission.mojo:404-412` | exporter admission, zero-filled alloc, copy of 1.5 KB, a reshaped Array object with 13 attributes | tens of us |
| entries | `mamba_public_native.mojo:171-196`, `neural_block_public.mojo:835-930` | subtype test, prepared-slot reads, `getenv` | microseconds |
| state fields | `mamba_public_native.mojo:105-124` | record fields, dict copy, 10 `neural_array_snapshot`s (each reads every Array slot through a descriptor, `native_array_snapshot.mojo:54-80`) | about 100 us |
| state call | `samba_inference_native.mojo:105-180` | 10 more snapshots in `_mutable`, 5 zero-filled arrays (about 400 KB: h_last is B x 12 x 64 x 128 floats), 25-address list, the call | about 100-200 us |
| publication | `mamba_public_native.mojo:321-330,1184-1188` | 7 + 4 `py_setitem` | microseconds |

That is roughly 0.3-0.5 ms accounted; the rest is not settled by reading. Candidates the timer run decides: the
`arrays_match` namespace walk per token, a snapshot miss (if `_neural_snapshot_matches` fails, `neural_block_snapshot`
rebuilds the identity key and the weight snapshots every token: :1047-1068), and the zero fills. The fix of section 1
removes the whole table from the per-token path, which is why the line does not have to be found before D1.

**RUN OWED (before or alongside D1, one run):** mamba3-decode on the 0.8.66 wheel with `-D MOJOLEARN_MAMBA3_PHASE_TIMERS`
(prints `M3_PHASE surface.*` per call, `mamba_shell_impl.mojo:1154-1175`) plus a wall timer around
`mamba3_forward` in `mamba_native_call` (:1183) and around the `mamba3_decode_step` call in `_m3_state_call` (:170),
printed per token; the three numbers split door, binding entry and device. 2048 tokens, one run.

### 2.4 Why transformer-decode's 1.10x is probably the same tax, not noise

`transformer_decode_run` (`providers/python/transformer_decode_pyapi.mojo:596-640`) per token: owner lookup,
`_transformer_decode_unchanged`, `_neural_input` (the NumPy copy above), `generation_array` for y, the call. The delta is
254 ms over 2048 tokens = 0.124 ms per token, the size of that work. 0.8.55's session step did the same in Python with
NumPy borrows. Lane D2 removes it (one call per N tokens); the A/A of section 4 confirms it is real.

## 3. C6, LinearSVR (question 3)

### 3.1 The tile-order change is not on the measured path

- `QN_TILED = IDENTICAL and not QN_TILED_OFF` (`glm/impl/qn/glm_base.mojo:479-482`); `qn_tiled_applies(d, c)` is
  `c == 1` under it (:562-564). `GLMWithData.evaluate` picks `tiled` (:2106) and runs `enqueue_tiled` (:2127, :2149),
  the fused tiled objective (:2507-2540). `linear_bwd` runs only on the `else` arm (:2131, :2151): a `C > 1` fit, or a
  `C == 1` fit with `QN_TILED` off.
- `4b8a8e8f2b` changed `linear_bwd`'s `C == 1` arm (:1752-1764, "reached only with QN_TILED off") and the `C > 1` arms,
  and added `qntm_tiled_into` chunking. LinearSVR (`C == 1`, `QN_LOSS_SVR_L2`) in an IDENTICAL build never calls it.
  LinearSVC taxi's 0.29x is binary and `C == 1` too, so it cannot be "the narrow shape fills the GPU" of the tile
  order either; whatever moved those two cells is in `enqueue_tiled`'s surroundings or outside the GLM.
- What did change on the `enqueue_tiled` path since 0.8.55 (`git diff 4e927c92495 HEAD -- glm/impl/qn/glm_base.mojo`):
  `self.dplan.single()` per evaluation (:2526-2528; a string compare, `core/device_plan.mojo:395-400`) and the
  workspace bound `qntm_ws_floats(n_rows, C * D)` joining the `max` (:1912-1913; same order of size as
  `qnt_workspace_floats`). Neither explains 100 ms.
- The cell's clock is `fit(X, y)` plus `predict(Xq)` (`tools/bench_board_more.py:1270-1276`). The fit door
  (`bindings/svc_linear_native.mojo:265-300`, `svc_f32` -> `family_f32_snapshot` `family_common.mojo:329-333`) does
  not cast or copy a float32 C-order X (`family_snapshot_f32` :317-326 casts only when `not f32_c()`), and
  `qn_fit_binding` is unchanged since 0.8.55 (diffed). The predict (`_qn_scores`, `svm_mixture_native.mojo:22-48`)
  calls the same `qn_decision_function` device entry 0.8.55 called (`3a59ed589e` only split it into
  `qn_decision_function_device`, `glm/estimator.mojo`).
- The deltas, +102.3 ms on 2,043,304 x 220 and +6.3 ms on 4,000,000 x 11, are 17.6 and 28 GB/s over X: one more
  host pass or one more upload of X. The reading did not find it in the LinearSVR files; it is somewhere the fit or
  predict door shares with the other classical2 lanes that did not move (svr 1.00x, knn 1.02x), so it is specific to
  the QN entry's arguments (y as float32: `svc_f32(args[2], 1, "y")`, the `sample_weight` path, `_linear_strengths`)
  or to `qn_fit_host`'s pageable `enqueue_copy` of X (`glm/estimator.mojo:686-700`) behaving differently for the two
  wheels (pinned vs pageable source, driver). Low confidence; the timer run decides.

### 3.2 The A/B design (kept as the confirming arm) and the timer run

- **Arm A**: the 0.8.66 wheel (IDENTICAL). **Arm B**: 0.8.66 with `git apply
  experiments/removed/MOJOLEARN_QN_TILED_MULTI_OFF.patch` (restores `xty_kernel` / `xtdz_multi_kernel` wherever the
  gradient took them). Expected: A = B on linearsvr and linearsvc (the arm is not on their path); a difference means
  the reading above is wrong and the lane follows the tile order after all.
- Record `n_iter_` per fit (LinearSVR exposes it: `python/mojolearn/svm.py:294`; `_qn_fit_one_target` returns it,
  `svc_linear_native.mojo:234-262`); the bench stores none today, so the A/B harness prints it beside the time.
- Shapes: istella and taxi at the board rows, plus the neighbors 1M x 220, 2M x 110, 2M x 440 (one non-board width
  on each side) and one non-board dataset (HIGGS 1M x 28, already on the box for the fit lanes). NVIDIA and AMD; the
  verdict is the average of the two vendors' ratios (CLAUDE.md measurement process). One run per arm.
- **RUN OWED first, both wheels, same box:** the LinearSVR istella cell with a wall timer around `fit` and `predict`
  separately (the recipe times them together), and inside the fit around `svc_f32(X)`, `_qn_fit_one_target` and the
  `qn_fit` binding call (`svc_linear_native.mojo:274,290,251`), and `MOJOLEARN_GLM_PHASE` style prints around
  `qn_fit_host`'s uploads and the solver loop (`glm/estimator.mojo:686-700`). Five numbers per wheel locate the pass.
- **The likely fix once located:** if it is a host pass over X (a cast, finite scan or copy that 0.8.55 did not do),
  it joins F1/F3 of the regressions plan (parallel SIMD, or no copy at all); if it is the upload, pin or borrow the
  source as 0.8.55's `as_f32_c` did. If, against the reading, the tile order is the cause on wide shapes, the fix is
  the workspace and tile shape by (rows, features) reasoning: size `QNTM_MAX_WS` from device memory so a wide
  `C == 1` gradient is one chunk (`qn_tiled_rule.mojo:30-42`), and choose the tile rows so `tiles * cells` fills the
  device at the smallest board-independent width, never by a board dimension; the narrow-shape gain stays.

## 4. The 1.1-1.2x cells and the noise floor (question 4)

- Both arms are a single timed round after one untimed warm-up: the 0.8.55 paper board (`board.json` config
  `"rounds": 1`), the 0.8.66 compare (`tools/paper_compare.py:242`), the defaults (`tools/bench_board.py:128`
  `DEFAULT_ROUNDS = 1`, `tools/bench_board_neural.py:6437`). `median_ms` of one round is that round.
- No A/A floor is stored for the neural lanes on AMD: `tools/six_lane_timing.py` keeps A/A floors per
  `workload_id|vendor` for the six-lane grid, and the grid decider falls back to a 5% default when none is stored
  (`tools/six_lane_grid_decide.py:111-113`). The paper compare has no floor field at all (`compare-amd.json` rows:
  `old_ms`, `new_ms`, `ratio`).
- So one repeat is not enough. The rule: an A/A pair per wheel, interleaved (new, old, new, old) in one session on one
  box, each a one-round race with its warm-up. A cell is noise when max(A/A spread of new, A/A spread of old) covers
  |new - old|; otherwise it is real and gets a lane. Expected from the mechanisms found: transformer-decode real
  (0.124 ms per token door tax, section 2.4); gemm-bf16 (+8.7 ms), gemm-int8 (+6.6 ms), lm-forward (+8.4 ms) no
  mechanism found and code-equivalent per the diagnosis; the candidate `lin_call` descriptor route
  (`bindings/linalg_ops_lowbit.mojo:47`) is tested by the same A/A (if real, time the door with a wall timer around
  `lin_call`).
- Cost: four one-round races per lane, six lanes (the three above, transformer-decode, linearsvr istella and taxi),
  about 15 minutes of MI300X including warm-ups. Queue as one `lq add amd CMD` on the frozen commit.

## 5. Table

| item | layer | cause hypothesis, evidence | fix | lane | A/B or run owed |
|---|---|---|---|---|---|
| mamba3-decode 2.17x (+1.64 ms/token) | public door (host), not the tape, not the device | per-token admission and snapshotting in `mamba3_forward` -> `_m3_state_call` (section 2.3); device op list identical to 0.8.55 (diffed) | sessions on the public path, N tokens per call, then executor step ops | D1 (S), D2 (M), D3 (L) | phase-timer run (2.3); A/B board: recipe with `decode_session` vs per-call step, AMD and NVIDIA, one run |
| samba-decode 1.57x (+4.2 ms/token) | stack driver (host) | 4 per-layer doors per token (2 Mamba3 `_m3_state_call`, 2 attention `neural_block_forward_binding`), hidden state through host arrays (N094) | `SambaStack.decode_session` with per-layer resident sessions and device-to-device hidden | D1, D2 | same A/B as above on samba-decode |
| mamba*/samba-forward 1.27-1.55x | public door | the same door per call plus the NumPy x copy (C5, `neural_block_admission.mojo:404-412`) | D1 removes the door from step; F5 of the regressions plan (borrow x) covers forward | D1 + F5 | one forward run with the timer |
| transformer-decode 1.10x (+0.124 ms/token) | session door | per-token `_neural_input` copy + `generation_array` + call (`transformer_decode_pyapi.mojo:596-640`) | D2 (one call per N tokens) | D2 | A/A pair (section 4) |
| linearsvr 1.98x / 1.16x | not the tile order; fit-or-predict door or the X upload | `enqueue_tiled` serves IDENTICAL `C == 1` (`glm_base.mojo:2106,2127`); deltas scale as one pass over X at 17-28 GB/s | find the pass (timer run), then F1/F3-style removal; tile A/B only confirms | G1 (S) | timer run both wheels; A/B A = 0.8.66, B = `+ MOJOLEARN_QN_TILED_MULTI_OFF.patch`, `n_iter_` recorded, NVIDIA + AMD, neighbors + HIGGS |
| gemm-bf16/int8, lm-forward 1.14-1.18x | unknown | none found; n = 1 both arms | A/A first | N1 (XS) | A/A pair per wheel |
| the tape for training (Andrew's question) | executor | training already runs backward on the tape over hand forwards; forward and decode do not | D3: step ops + in-place state + iteration register; weights as program parameters | D3 | A/B train step: tape-with-step-ops vs today, same bits |

## 6. Lanes (one paragraph each)

**D1 `decode-sessions-public` (S, no bits change).** Make `Mamba2Block.decode_session` and
`Mamba3Block.decode_session` public: rename `_decode_session` and `_Mamba2DecodeSession` / `_Mamba3DecodeSession`
(`python/mojolearn/_mamba_impl.py:956-1125,1477-1560`), re-export from `mojolearn.mamba`, write user docstrings in the
voice of `Mamba1Block.decode_session` (:598-620), add the `mamba2-decode-session` and `mamba3-decode-session` lanes to
`tools/identity_break.py` and the host-surface descriptions as `_RESIDENT_SESSION_NOTE` (:969-998) lists, and remove
the note. Add `SambaStack.decode_session(state)` (`_samba_impl.py`, `bindings/samba_inference_native.mojo`): one
native handle holding one Mamba3 or transformer decode session per layer plus the embedding, `norm_f` and head on the
device, `step(ids)` running embedding -> layers -> norm -> head with the hidden state device-resident (the
`samba_resident` session's buffers are the model: `training/samba_resident.mojo`), outputs (B, vocab) logits; `close`
and `sync_state` as the block sessions. Change the three board recipes (`tools/bench_board_neural.py:2387-2401,
2549-2552`, `LANE_TEXT` :980-983) to open the session before the clock. Tests (unrun): session step equals per-call
step byte for byte for Mamba2, Mamba3 and the Samba stack (`test_mamba23_decode_sessions.py` pattern). Bindings:
`_mojolearn_mamba`, `_mojolearn_runtime_neural*`, the samba binding group.

**D2 `decode-n-tokens` (M, no bits change).** On each session handle (Mamba1/2/3 `MambaDecodeHandle`
`mamba_public_native.mojo`, `TransformerDecodeSession`, the new Samba session) add `decode(x_or_ids, n_tokens)`: host
admission once, the input slab uploaded once, `# driver-loop: over tokens` calling the session step body
(`_m3_session_step_run` without its downloads, `_session_step_run`), one output slab, one download, one sync at the
end; reports are the last token's. `Mamba*DecodeSession.step` stays as the one-token case. The board recipes keep one
public call per token (the lane's declared operation) and a second operation `decode_n` is added for an A/B of the
two shapes; the paper recipe is not changed by this lane. Tests (unrun): `decode(x, L)` equals `L` steps byte for
byte, state equal after. Bindings: as D1.

*Status 2026-11-09: written on `lane/d2-generate-on-sessions` (uncompiled, unrun). `decode(x)` on the Mamba1/2/3,
Transformer and Samba sessions; one `<kind>_session_decode` / `transformer_decode_session_decode` /
`samba_decode_tokens` call per N tokens, token t moved on the device by `core/decode_slab.mojo`; board A/B
operation `decode_n`. CausalLM.generate on transformer checkpoints already reads whole token blocks on the
resident sessions (`_read_resident`); mamba1/mamba2 CausalLM generation still runs the per-layer native engine
(`transformer/native_generation_engine.mojo`) and moves with X4/X5.*

**D3 `decode-on-executor` (L, no bits change; after D1/D2 are green).** In `autograd/`: step ops `OP_M1_STEP`,
`OP_M2_STEP`, `OP_M3_STEP`, `OP_ATTN_STEP` whose bodies are the session step launch sets over session tensors that
hold the state (`Mamba3DeviceState` pieces, K/V cache pair) and an in-place state write rule (a `*_into` node writing
its own input storage, declared so the tape never reads a stale copy; `zeroed_destination` style declaration), an
iteration register for `while_loop` (`program_control.mojo`) so the body can `take` token t, and
`Program.decode(ids, state, n)` in `python/mojolearn/ops.py` as `while_loop` with recording off. Training: the same
program with recording on; `rules_mamba3`, `rules_transformer` record the step op's stage buffers as they record the
hand forward's now. Weights are program parameters (uploaded once, read in place, updated in place by the optimizer).
A/B: decode through the program vs D2's `decode_n` (same bits); train step through the program vs today's resident
step (same bits). This lane is the "one runtime path" and is where the per-layer door leaves the inference path.

**G1 `linearsvr-pass-over-x` (S).** Queue the two-wheel timer run of section 3.2 first; read the five numbers; remove
the pass (parallel SIMD finite scan or no copy, as F1/F3; pinned or borrowed upload source if it is the copy-in);
then run the confirming tile-order A/B with `n_iter_` recorded on NVIDIA and AMD over the neighbor shapes and HIGGS.
Only if the A/B shows the tile order matters does the lane touch `qn_tiled_rule.mojo` (workspace from device memory,
tile rows by (rows, features) reasoning, never a board dimension). Bindings: `_mojolearn_estimators`.

**N1 `aa-pairs` (XS, no code).** One `lq add amd CMD` on the frozen commit running the interleaved A/A pairs of
section 4 for gemm-bf16, gemm-int8, lm-forward, transformer-decode, linearsvr istella and taxi on both wheels;
publish the six spreads next to the deltas in `bench/results/paper_compare/`; cells inside the spread are closed as
noise, the others open lanes (lm-forward and the gemms: a wall timer around `lin_call`).

## 7. A/Bs owed, exact arms

| A/B | arm A | arm B | where | what is recorded |
|---|---|---|---|---|
| decode sessions on the recipe | mamba3-decode and samba-decode as today (per-call `step`) | the D1 recipe (`decode_session` opened before the clock) | AMD MI300X and NVIDIA L40S, one run each | median_ms, digest (must be equal), per-token ms |
| N tokens per call | D1 recipe, one `step` per token | D2 `decode_n(x, L)` | same | same, plus state equality |
| tile order (confirming) | 0.8.66 IDENTICAL | 0.8.66 + `experiments/removed/MOJOLEARN_QN_TILED_MULTI_OFF.patch` | NVIDIA + AMD; istella, taxi, 1M x 220, 2M x 110, 2M x 440, HIGGS 1M x 28 | ms, `n_iter_`, objective, r2 / accuracy; verdict = average vendor ratio |
| A/A pairs | 0.8.66 wheel twice | 0.8.55 wheel twice, interleaved | AMD, one session | the two spreads vs the delta |
| executor decode (D3) | D2 `decode_n` | `Program.decode` | both vendors | ms, digest equal |

## 8. Runs owed (exact)

1. mamba3-decode on the 0.8.66 wheel, `-D MOJOLEARN_MAMBA3_PHASE_TIMERS`, wall timers at `mamba_public_native.mojo:1183`
   and `samba_inference_native.mojo:170`, 2048 tokens, one run, AMD.
2. linearsvr istella on both wheels, timers at `svc_linear_native.mojo:274,290,251` and `glm/estimator.mojo:686-700`,
   `fit` and `predict` timed separately, one run each, AMD.
3. The A/A pairs of section 4 (six lanes, two wheels, interleaved), AMD, one `lq` CMD.
4. The tile-order A/B of section 3.2 after run 2, NVIDIA and AMD.

None of these runs happens on the Mac; all go through `lq` on the frozen commit after the release record.
