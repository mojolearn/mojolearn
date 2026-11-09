# One runtime path per family: move onto the executor, delete what no longer executes

Ideas lane `ideas-executor-consolidation` (2026-11-08), read-only. Code and history on origin/main `22ac89b54c`;
the diagnosis `docs/plans/REGRESSIONS_0866_VS_0855.md`, the decode plan `docs/plans/DECODE_DOOR_AND_GLM_PLAN.md`
(lanes D1-D3, which this plan extends and does not repeat), the program `docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md`
(workstreams A-E, lessons 1-15). Nothing was run, compiled or measured. Every file:line is on main unless it says so.

Andrew (2026-11-08 20:40): "are there a lot of places where we can and should refactor to use autograd? we should do
that wherever we can and get RID OF the old stale code that doesn't execute. CLEAN UP not just add." And (20:50):
"are there other things that should be running on autograd?"

## 0. One screen

1. **Every backward is already on the tape; no forward is.** The nine rule modules (`autograd/rules_{transformer,
   mamba,mamba3,samba,lm,cnn,conv2d,recurrent,small_mlp}.mojo`) are the only backward of their block; the hand
   backward walks were deleted 2026-10-19 (`docs/TOMBSTONES.md`, `MOJOLEARN_AUTOGRAD_*_OFF`). But each rule records
   nodes over stage buffers that a *hand-launched forward* already wrote (`rules_mamba3.mojo:28-30`,
   `rules_transformer.mojo:1-6`): the forward of Transformer, Mamba1/2/3, Samba, the byte LM, the CNN and the small
   MLP is still `llama_decoder_layer_forward` (31 call sites), `mamba3_block_forward` (11), `_mamba{1,2}_run` (7 each),
   `conv_block_forward_into` (7). Only three paths run forward AND backward through the executor today: MLP/Sequential
   fit (`sequence/mlp_fit.mojo:807,889`), GRU/LSTM/RNN fit (`autograd/rules_recurrent.mojo:439,442`) and
   `ops.Session`/`Program` (`autograd/ops_api.mojo:295`, `program_entries.mojo:205`). Section 1 is the table.
2. **The executor is a complete training runtime already**: captured programs with parameters read in place
   (`autograd/program.mojo:1-40`), `Program.fit` as one native call over epochs and minibatches
   (`autograd/program_runtime.mojo:139,243-260`), the in-place optimizer step (`ops_api.mojo:953`), `while_loop`/
   `run_if` (`program_control.mojo:121-248`), 90 session op codes (`OPS_*`, `ops_api.mojo`), distributed and
   partitioned programs. What it lacks for the block families is in section 2: block **step ops** (none of `OP_M1_`,
   `OP_M2_`, `OP_M3_`, `OP_ATTN` exists; `OP_TFP_*` cover RMS/RoPE/linear/softmax of a prefill only), **in-place
   carried state** (`while_loop` copies every state tensor with `OP_COPY` + one sync per iteration,
   `program_control.mojo:55-85`; integer state is refused :66,147), an **iteration register**, and **weights as
   program parameters** (per-call upload in `_mamba{1,2}_run` `mamba_shell_impl.mojo:231-237`, `_mamba3_run` :1165).
3. **Classical and trees: there is no admission layer, there are twelve.** Four byte-assembled cast loops
   (`core/classical_arrays_impl.mojo:11`, `core/family_cast_impl.mojo:18`, `core/xseq_marshal.mojo:30`,
   `bindings/linalg_ops_buffers.mojo:375`) plus a trees path that leaves Mojo to call the base binding's
   `cast_elements` through Python with `_tasks = 1` (`trees_public_inputs.mojo:395-406`, `core/hotpath_impl.mojo:62`);
   **nine finite scans**, one SIMD (`p3nc_scan.mojo:20`), one parallel (`family_cast_rows_finite`
   `family_cast_impl.mojo:27`, called from two files), one on the device (`core/input_device.mojo:117`) that no
   classical or family helper reaches; three materializers of the same input types, with
   `bindings/classical_labels.mojo:186-600` a near-verbatim copy of `trees_public_inputs.mojo:139-600`; four metadata
   structs; six sample-weight admitters; eight zero-fill allocators. `AdmittedCall.inputs` is written
   (`classical_validate.mojo:128`, `family_public_lifecycle_pyapi.mojo:236`) and read by nothing, so the forest door
   casts X in `classical_check_x` and re-views it with `trees_f32` (`trees_public_lifecycle_native.mojo:85,96`);
   DeviceRows folds are materialised to host rows in both doors and re-uploaded by every family's kernel driver.
   Section 3 designs ONE path (`admit_x`: metadata once, borrow or parallel typed cast, finite on the device after the
   upload the kernel makes anyway, one upload, DeviceRows passed through) and the lanes that move each family onto it
   and delete its copy. 61 `# native input marshaling` loops: 2 parallel, 59 serial, all listed.
4. **Outside neural, the executor's model fits in three places** (section 4): (a) re-admission defects that need
   no new op (LDA online re-uploads X every epoch `x_decomp/kit_device.mojo:1501`; cross-validation downloads the
   gathered fold rows for every classical estimator `classical_validate.mojo:94-120`; RFE re-puts X per round
   `prep_select_native.mojo:112`; Stacking/OvR/MultiOutput/Voting re-admit the same X per member; AdaBoost rebuilds
   weights on the host per round `trees_public_boost.mojo:1070-1075`); (b) loops that already have exactly the
   `while_loop` shape, one scalar read per iteration, same kernels (GMM EM, BGMM, MDS, CD sweeps, GLM line-search
   trials, GP/GPC Newton, SMO outer, ARIMA lock-step); (c) host eigen/SVD round trips that need one device `eigh`
   op (FA EM, FastICA, rotation, IPCA, KPCA/Isomap/Spectral Lanczos). Already resident and chunk-polled (KMeans DC,
   AP, mean shift, DBSCAN, UMAP, t-SNE, GBDT levels, IsolationForest): nothing to gain; `while_loop`'s one scalar per
   iteration would regress them until the body can unroll a chunk.
5. **Dead code with zero callers, verified by rg** (section 5): 13 `byte_lm_parallel_*` entries, 3
   `mamba{1,2,3}_decode_native_handle_v1`, 5 `vocab_*_public`, the `_mojolearn_scoped_gemm_probe` binding, 5 orphan
   functions, 7 experiment defines whose EXPERIMENTS.md verdict is a loser but whose gate is still in the code, and
   their witness exports; `classical_host_weights` (0 callers); two Python Lanczos/rSVD loops that nothing live calls
   (`_expansion_decomp.py:2691,4002`); 140 of 808 non-test tools with no inbound reference. Plus the duplicate paths
   that are dead *by construction* once a decision is taken: hand `mlp_predict` vs `mlp_predict_autograd`
   (`MOJOLEARN_AUTOGRAD_PREDICT`, default off, no A/B record anywhere), the per-call Mamba3 step vs the session
   step, Samba's three-tier forward. Each deletion follows the tombstone rules (inline marker, `TOMBSTONES.md`
   section, `experiments/removed/` patch when it reached main, guard refusal for defines).
6. **Bits.** No lane in this plan changes the bytes a kernel reads or a fold order: step ops are the session bodies
   lifted, the admission path produces the same float32 C-order bytes, the dead code has no live path. The one
   pre-release identity run is the proof; section 6 names the three lanes that *could* move bits and how each avoids it.
7. **The name** (section 7): `autograd/` is the Mojo package of a resident program executor whose gradient tape is
   one feature; the public surface is already `mojolearn.ops`. A split is proposed for Andrew to decide; no rename
   is made or scheduled by this plan.

Section 8 is the table. Order: E-admit (one admission path) and the zero-caller deletions first (no numerics, every
lane pays); then D1/D2 of the decode plan; then D3 in family order small MLP -> transformer -> Mamba -> Samba ->
byte LM -> CNN, each lane moving the forward onto the executor and deleting the hand forward and its door in the
same change; the non-neural executor moves after D3 proves the step-op pattern.

## 1. Neural inventory: every public path and the runtime it takes

Runtime paths: **executor** = `ops.Session` / captured `Program` on `ProgramExecutor`, or the `Exec`-level tape
with a recorded forward; **resident session** = hand kernels over device-resident weights/state behind one native
handle; **per-call door** = admission, weight upload and launch per public call; **hand chain** = fixed op sequence
on the `Exec` with no tape; **host driver loop** = a Mojo `for`/`while` over tokens, layers or workers.

| family | public method | native door | path today | fwd / bwd on tape | executor: as is / after D3 (needs) / never |
|---|---|---|---|---|---|
| MLP (sklearn-style) fit | `MLPClassifier.fit` `_x_sequence_mlp.py:285`, `MLPRegressor.fit` :154 | `xseq_neural_native.mojo` -> `sequence/ops_impl.mojo` -> `sequence/mlp_fit.mojo:651` | executor (fused tape node per layer :807/:889) | yes / yes | as is; optimizer on the executor only under `-D MOJOLEARN_AUTOGRAD_OPTIM` (:157) |
| MLP predict | `.predict/.predict_proba` :290-295 | `ops_impl.mojo:773-780` | hand `mlp_predict` -> hand `mlp_forward` (`mlp_fit.mojo:985,1015`) unless `-D MOJOLEARN_AUTOGRAD_PREDICT` (:92, default off) | no | as is via `mlp_predict_autograd` (:1034) once the define flips; **two predict paths, one dead by construction** (section 5.6) |
| Sequential | `.fit` `_x_sequence_model.py:331`, `.predict` :597, `.predict_proba` :616 | `_native_runtime_sequential_bridge` :634 -> same `mlp_*`/`rnn_*` routines | executor | yes / yes | as is |
| SmallMLPTrainer | `.train_step` `_mlp_impl.py:348` | `small_mlp_native.mojo:1045`, resident :1057; kernels `training/mlp_ops.mojo:332` (hand forward :140-270), backward :572 = `rules_small_mlp` | resident session | no / yes | after D3: weights as parameters, in-place optimizer state (already resident); forward is 3 GEMMs + bias + ReLU, every op exists (`OPS_LINEAR`, `OPS_RELU`) |
| TransformerBlock | `.forward` `_transformer_impl.py:1055`, `.step` :1083, `.backward` :1161 | `neural_block_public.mojo:785` -> `_neural_plain` :647 -> `causal_lm_public.mojo:824,912-922` -> `_mojolearn_transformer.mojo:769/589` -> `transformer_binding_impl.mojo:1284/837`; backward `neural_block_public.mojo:1074` -> `transformer_binding_impl.mojo:963/1000` -> `llama_decoder_layer_backward_autograd` :984,1073 | per-call door over a retained block session (weights, cache, RoPE retained; one block per call) | no / yes | after D3: forward on the tape (today the rule replays from hand-saved stages) + weights as parameters; `OP_TFP_*` exist (`transformer_primitives_ops.mojo:12-21`), no fused attention op |
| TransformerDecodeSession | `.step` :1297, `.forward` :1307 | `transformer_decode_pyapi.mojo:596` -> `_mojolearn_transformer.mojo:1045` -> `transformer_binding_impl.mojo:1186` (resident IO gated `IDN_DECODE_RESIDENT_IO`, `fused_attention.mojo:2598`) | resident session | no / n.a. | after D3: `OP_ATTN_STEP` with in-place KV state + position register |
| CausalLM | `.forward` `models/causal_lm.py:302`, `.step` :321, `.generate` :325 | `causal_lm_public.mojo:1811` -> `_run_prepared` :1714 (resident :1771 or per-layer `for` :1802); generate :2859 -> stream :2181 -> engine `native_generation_engine.mojo:269` (token `while` :276, layer loop in `forward_into`) or `_read_resident` :2301 | host driver loop (tokens x layers) | no / n.a. | after D3: step ops with carried KV/SSM state, iteration register, argmax on the executor (`OPS_ARGMAX` exists; `causal_lm_pick_step` :279 runs between passes today) |
| ParallelCausalLM | `parallel_causal_lm.py:98` | `parallel_causal_native.mojo:540,480` | host driver loop over worker RPCs | no | never as one program across processes; a `distributed_program` partition is the shape |
| Mamba1Block | `.forward` `_mamba_impl.py:527`, `.step` :578, `.backward` :554, `.decode_session` :598 (public) | `mamba_public_native.mojo:1122` -> `_neural_plain` -> `causal_lm_public.mojo:941-965` -> `_mojolearn_mamba.mojo:407/462` -> `mamba_shell_impl.mojo:194` `_mamba1_run` (weights + state uploaded per call :231-237); backward :311 -> `mamba1_tape_backward` :335 (`rules_mamba.mojo:808`); session :463 `_m1_session_step_run` | forward/step: per-call door; decode_session: resident session | no / yes (re-runs the forward, then the tape) | after D3: `OP_M1_STEP` over (h, conv window) state in place, weights as parameters |
| Mamba2Block | `.forward` :885, `.step` :936, `.backward` :912, `._decode_session` :956 (private) | same route, `_mojolearn_mamba.mojo:649/721` -> `mamba_shell_impl.mojo:500` `_mamba2_run`; backward :2112 -> `mamba2_tape_backward`; session :1035 | per-call door; private session | no / yes | after D3: as Mamba1; D1 makes the session public first |
| Mamba3Block | `.forward` :1406, `.step` :1457, `.backward` :1433, `._decode_session` :1477 (private), `_Mamba3DecodeSession.step` :1517 | forward `mamba_public_native.mojo:333` -> `_mamba3_forward_captured` :289 (prefill session, weights retained: `_mojolearn_mamba.mojo:1258` -> `mamba_shell_impl.mojo:1194`); step -> `_m3_state_call` (`samba_inference_native.mojo:105`) -> `_mojolearn_mamba.mojo:989` -> `_mamba3_run` :1122 (9 weight + 10 state uploads, 15 downloads, 1 sync per token; decode plan 2.1); backward `neural_block_public.mojo:1246-1270` -> `mamba_shell_impl.mojo:1963,2003` -> `rules_mamba3.mojo:818` (:438 records the already-run forward as 5 stage nodes); session step :1514 | forward: resident prefill session; step: per-call door; backward: tape over hand stages | no / yes | after D3: `OP_M3_STEP` whose body is `_m3_session_step_run`, state (h, k, v, theta) in place, weights as parameters |
| SambaStack | `.forward` `_samba_impl.py:495`, `.step` :538, `.loss` :581, `.train_step` :624 (no `generate`) | `samba_inference_native.mojo:264`: stateless + resident -> `_samba_resident_forward` :193 -> `training/samba_resident.mojo:968`; else `for` over layers :311 (attention: `neural_block_forward_binding`; Mamba3: `_m3_state_forward` :182); train_step `samba_step_native.mojo:236` -> `samba_resident.mojo:1002` (tape :1095, optimizer :1104 `identical_optimizer_step`), fallback `samba_training_native.mojo:94` | forward/step with state: per-call per-layer door (host driver loop over blocks); stateless forward + train_step: resident session | no / yes (one tape, `rules_samba`) | after D3: the transformer and Mamba3 step ops + weights as parameters; the optimizer is already in-executor. Three fallback tiers in one driver (:283-295) collapse to one |
| GRU/LSTM/RNN fit | `_x_sequence_rnn.py:91-128` | `xseq_neural_native.mojo:414` -> `ops_impl.mojo:102,148` -> `sequence/recurrent.mojo:534` | executor (`rnn_forward_autograd`/`rnn_backward_autograd`, `rules_recurrent.mojo:439,442`; cell ops `OP_CELL_*` `sequence/ops.mojo:125-180`) | yes / yes | as is |
| GRU/LSTM/RNN predict | same, `.predict` | `recurrent.mojo:640` `rnn_predict` -> hand `forward` :324 + `head` :393 | hand chain | no | as is through an inference-mode tape forward (same move as MLP predict) |
| CNNClassifier | `.fit/.predict_proba/.predict` `_expansion_cnn.py:1026-1039`; `Conv2d.forward/backward/fit` :633-672; pools :756-826 | `xcnn_classifier_native.mojo`, `_mojolearn_x_cnn.mojo` -> `x_cnn/device_entries_impl.mojo:151-185` (forward `conv_block_forward_into` :162 per block, `cnn_backward_autograd` :173 = `rules_cnn.mojo:248`, optimizer :177-185); layer doors `xcnn_conv_native.mojo:747` | hand chain inside a Mojo step loop | no / yes | after D3: record the forward (`OP_RN_CONV_*`, `OP_RN_COL2IM`, pools exist: `rankn_conv_ops.mojo`, 10 codes) + weights as parameters |
| GCN / SAGE | `GCNConv.forward/backward` `_expansion_cnn.py:1522/1572` | `xcnn_graph_native.mojo:450,592,1016` | per-call door, hand backward door | no / no | after D3 + one sparse aggregate op (missing op kind) |
| MoEBlock | `.forward` `_x_sequence_moe.py:180`, `.backward` :200 | `ops_impl.mojo:1899,1922` -> `:2024/:1946` (0 `Tape` refs); ops `OP_MOE_ROUTE/HIDDEN/OUT` `sequence/ops.mojo:174-176`, 17 `OP_MOEB_*` `autograd/kernels/moe.mojo:57-73` | hand chain (fixed op sequence, hand VJP chain) | no / no | as is for the kernels; needs tape rules to become a recorded program (`AUTOGRAD_MOE` gate `route_ops.mojo:73`) |
| ops.Session | `.backward` `ops.py:859`, `.step` :1274, `.capture` :1015, `.run` :1055 | `autograd_ops_frontend_pyapi.mojo:395` (70-branch `elif` chain :413-1028) -> `ops_api.mojo:295` (`backward` :600, `step` :953, `capture` :1822, `run_program` :2036, `op` :1096 -> `_op_run` :1105 with 59 branches) | executor | yes / yes | as is; the two dispatch chains are the compile-memory suspect of lesson 12 |
| Program | `.run` `ops.py:1358`, `.run_if` :1370, `.while_loop` :1386, `.fit` :1390, `.partition` :1447, `.distribute` :1459 | `autograd_program_public_request_pyapi.mojo:11`; control `program_control.mojo:164` (`while True` :205, `_predicate` :44 downloads one scalar); fit `program_entries.mojo:205` -> `program_runtime.mojo:243` epochs / :253 minibatches (`gather_rows` on the executor :260); distributed `distributed_program.mojo:326`; partition `partition_runtime.mojo:238` | executor, with a Mojo driver loop for while/if and epochs | yes / yes | as is; `while_loop` needs the iteration register and in-place state to drop the per-iteration copy + sync |
| import_model | `__init__.py:215` -> `autograd/import_api.mojo:162-556` | runs as an `OpsProgram` | executor | yes / yes | as is |
| Tape user model | through `Program.fit`; `autograd/model_execution.mojo:46,108` -> `autograd/model.mojo:917` (epochs :1112, batches :1127), `model_forward` :601 | executor (fused unless `-D MOJOLEARN_AUTOGRAD_MODEL_UNFUSED` :59) | yes / yes | as is |
| Byte LM trainer | `SmallByteLanguageModelTrainer.train_step` `_byte_lm_impl.py:1183` | `byte_lm_trainer_native.mojo:1110` -> `_run_resident` :1125 (session :1134) -> `training/byte_lm.mojo` (forward `llama_decoder_layer_forward` per layer, 8 sites; backward :1908 -> `rules_lm.mojo:434`; optimizer per layer :1945); `train_steps` :1304 is a Mojo `while k < n_steps` :1250 | resident session | no / yes | after D3: the transformer forward on the tape + weights as parameters |
| ParallelNeuralTrainer / ParallelByteLM | `parallel_training.py:36,204` | `parallel_neural_native.mojo:570,598`, `neural_ordered_sum` `parallel_mlp_native.mojo:384` | host driver loop over shard workers | per shard as the wrapped model | never as one program across processes; each shard body follows its family row |
| training.py optimizers / losses / layers | `SGD/Adam/AdamW.step` `_training_impl.py:536`, `cross_entropy` :974, `embedding_backward` :1607, `rms_norm_*` :1631/1653, `linear_backward` :1834 | `_mojolearn_training.mojo:225,436/471,394/409,882-930` | per-call hand kernels (standalone doors) | no / no | as is: the kernels are `OP_TFP_RMS*`, `OP_TFP_LINEAR`, the loss ops; the doors stay as the primitive API |
| Embedding | `Embedding.forward` `embedding.py:174`, `.backward` :193 | `embedding_layer_native.mojo:880/855` -> :237/:683; tape `OP_EMB_*` (14 codes, `embedding_ops.mojo:51-52`, gate `AUTOGRAD_EMBEDDING` `_mojolearn_embedding.mojo:596`, default on) | per-call hand kernel forward; backward through `OP_EMB` | no / yes | as is |

Facts the table rests on:

- Hand backward paths beside a tape rule: none left. `llama_decoder_layer_backward_device` survives only in two
  stale comments (`training/samba_resident.mojo:35`, `transformer/impl/llama/modeling_llama.mojo:1757`); `HandStep`
  only in a refusal string (`core/experiment_guards_neural.mojo:226`); `mlp_backward(` has 0 callers
  (`mlp_fit.mojo:84` tombstone). `_transformer_saved_backward` (`transformer_binding_impl.mojo:963`) is a tape wrapper
  reached only by `forward_with_tape`/`backward_from_tape`, which `docs-site/api_exclude.txt` marks "tape API not in
  the shipped binding; unverified" (callers `_transformer_impl.py:1106`, `tests/test_transformer_public_boundary.py:112-116`,
  `_verify_api_neural.py:55`).
- Hand forwards still live: `llama_decoder_layer_forward` 31 call sites (`training/byte_lm.mojo` 8,
  `transformer_binding_impl.mojo` 10, `byte_lm_logits` 4), `mamba3_block_forward` 11, `_mamba3_run` 12,
  `_mamba3_run_owned` 2, `_mamba1_run` / `_mamba2_run` 7 each, `conv_block_forward_into` 7, `mlp_forward` 1
  (inside `mlp_predict`), `mlp_train_step_host` 2, recurrent hand `forward` 1 (`rnn_predict`).
- `MOJOLEARN_AUTOGRAD_*` defines in `.mojo`: 41 names, 300+ mentions, most of them tombstones and refusal asserts.
  Live opt-in, default off: `AUTOGRAD_PREDICT` (`mlp_fit.mojo:92`), `AUTOGRAD_LOSS` (:136, `recurrent.mojo:80`),
  `AUTOGRAD_OPTIM` (:157, `recurrent.mojo:66`), `AUTOGRAD_DATA` (`recurrent.mojo:505`), `MODEL_UNFUSED`
  (`model.mojo:59`), `MLP_UNFUSED` (`mlp_fit.mojo:114`), `GEMM_DISPATCH_V1` (`sequence/gemm_tiled.mojo:379`),
  `HAS_INFERENCE`, `PROBE_CPU`. Live default-on `_OFF` rollbacks: `TRANSPOSE_VIEW_OFF`, `CURVE_WORD_OFF`,
  `FOLD_V2_OFF`, `LOSS_REDUCED_OFF` (`autograd/ops.mojo:156-157,271,275`), `GRAD_ALIAS_OFF`, `PAIR_OFF`
  (`tape.mojo:75,94`), `MOE_OFF` (`route_ops.mojo:73`), `OPT_FUSE_OFF`, `OPT_CLIP_FUSE_OFF` (`optim.mojo:68,90`),
  `REFUSE_DEFER_OFF` (`rules_lm.mojo:145`), `EMBEDDING_OFF`. None of the four opt-in defines has an EXPERIMENTS.md
  row; section 5.6 decides them.
- Executor op families for blocks: `OP_TFP_*` 10 codes, `OP_CELL_*` 6, `OP_MOE_*` 3 + `OP_MOEB_*` 17, `OP_RN_CONV_*`
  and pools 10, `OP_EMB_*` 14; generic `autograd/ops.mojo` 43, `import_ops` 56, `loss_ops` 28, folds 13. Session
  codes `OPS_*`: 90 (1..80 with gaps; `program.mojo:56-58` caps programs at 80). No Mamba, attention-with-cache or
  routing step op.
- `docs/plans/TAPE_EVERYWHERE_PLAN.md` is cited by five modules (`rules_cnn.mojo:3`, `rules_samba.mojo:3`,
  `rules_mamba3.mojo:3`, `rules_small_mlp.mojo:3`, `route_ops.mojo:4`) and never existed in git (`git log --all --
  '*TAPE_EVERYWHERE*'` is empty). The citations point at this plan and the decode plan after they merge.

## 2. D3: what the executor must add, and the lane order

The decode plan's D3 is one lane; here it is split so that **each lane moves one family's forward onto the executor
and deletes that family's hand forward, its per-call door and its duplicate path in the same change** (Andrew: "CLEAN
UP not just add"). A lane that adds an op and leaves the old path is not finished.

### 2.1 The four executor additions (shared, one lane, before any family moves)

| addition | today | design | files |
|---|---|---|---|
| **In-place carried state** | `while_loop` carries state by `OP_COPY` + `sync` per state tensor per iteration (`program_control.mojo:55-85`), allocates and frees the previous buffers each iteration (:209-213); integer state refused (:66,147,153); the only in-place writers are the optimizer (`ops_api.mojo:953-990`) and `load_into` (:504) | a `state` kind of program value: a session tensor the body may write in place through `*_into` nodes (`unary_into`, `binary_into`, `custom_into` already write caller storage, `NATIVE_AUTOGRAD.md` "Caller storage"), declared at capture so the tape never reads a stale copy (`zeroed_destination`-style flag); `while_loop` then skips `_copy_state` for declared state and keeps the copy for the rest. Integer-word state (int32 ids, KV position, cluster labels) allowed for `state` values, since nothing differentiates them | `autograd/program.mojo` (value kind), `program_control.mojo`, `program_runtime.mojo`, `ops_api.mojo` |
| **Iteration register** | design note only (`AUTOGRAD_API.md:440-470` `LOOP count`) | one int32 program value `iteration`, written by `control_run` before each body run (one `OP_FILL` of a (1,1) tensor, no sync); `take(slab, iteration)` selects token t from the uploaded input slab (`OPS_TAKE` 41 and `OPS_INDEX` 76 exist); the condition program may read it, so `max_iterations` becomes a program value too | `program_control.mojo`, `program.mojo` |
| **Block step ops** | none; the session bodies exist as hand functions: `_m3_session_step_run` (`mamba_shell_impl.mojo:1514`), `_m1_session_step_run` :463, `_m2_session_step_run` :1035, `transformer_decode_session_step_binding` (`_mojolearn_transformer.mojo:1045`) | `OPS_M1_STEP`, `OPS_M2_STEP`, `OPS_M3_STEP`, `OPS_ATTN_STEP` (codes above 80; raise the program cap at `program.mojo:56-58` and bump `PROGRAM_RECORD_VERSION`): inputs = x_t, the state values (in place), the weight parameters; the body is the session step launch set, unchanged; the backward rule is the existing stage rule (`rules_mamba3.mojo:438` records 5 stage nodes; `rules_transformer`) recording the op's stage buffers as it records the hand forward's today. Prefill (L > 1) is the same op with L > 1, so `forward`, `step`, `decode` and `train_step` are one op per block | `autograd/ops_api.mojo` (`_op_run` branch), a new `autograd/block_step_ops.mojo` wrapping the session bodies, `rules_*` |
| **Weights as program parameters** | per-call upload: `_mamba{1,2}_run` (`mamba_shell_impl.mojo:231-237`), `_mamba3_run` :1165 (9 uploads per token, decode plan 1.3); the retained sessions hold their own copies (`neural_block_public.mojo:647-720`, `byte_lm_trainer_native.mojo:1134`, `small_mlp_native.mojo:622`) | the block's weights are `upload`ed once into the session as parameters (`program.mojo:15`: read in place at each run, updated in place by `step`); the family's resident-session copy is deleted, since the program's parameters ARE the resident copy; `parameters()` / `state_dict()` download on request (workstream C) | the family bindings |

Compile notes (lessons 8, 12): the two dispatch chains (`autograd_ops_frontend_pyapi.mojo:413-1028` 70 branches,
`ops_api.mojo:1105-1588` 59 branches) grow by four ops each; the lane dispatches the new codes through a table, not
more `elif`.

### 2.2 Family lanes (each: move AND delete)

| lane | moves | deletes in the same change | size |
|---|---|---|---|
| **X1 `executor-state-and-step-ops`** | the four additions of 2.1; `Program.decode(ids, state, n)` in `ops.py` as `while_loop` with recording off | nothing yet (shared layer) | M |
| **X2 `small-mlp-on-executor`** | `SmallMLPTrainer.train_step` as a captured program (3 `OPS_LINEAR`/`OPS_RELU` + loss + `step`), weights as parameters; `predict_logits` as the same program with recording off | hand forward `training/mlp_ops.mojo:140-270`, `mlp_train_step_host` :332 and its two callers, `rules_small_mlp.mojo` (its launch inventory becomes the program), the resident-session plumbing of `small_mlp_native.mojo:622-1092` that the program's parameters replace | S |
| **X3 `mlp-rnn-predict-on-tape`** | MLP and GRU/LSTM/RNN predict through the inference-mode tape forward (`mlp_predict_autograd` `mlp_fit.mojo:1034`; an `rnn_predict_autograd` over `rnn_forward_autograd`) | hand `mlp_predict` :985, hand `mlp_forward` :232 (1 caller), `MOJOLEARN_AUTOGRAD_PREDICT` (:92, folded in: record the A/B, tombstone), hand recurrent `forward` :324 + `head` :393 | S |
| **X4 `transformer-on-executor`** | `TransformerBlock.forward/step/backward` and `TransformerDecodeSession.step` as `OPS_ATTN_STEP` over parameters + in-place KV state; `CausalLM.generate` as `Program.decode` (greedy pick = `OPS_ARGMAX` 72 in the body) | `_neural_plain` / `_neural_layer_run` for transformer (`neural_block_public.mojo:423-720`), the `forward_fresh` / `session_forward_fresh` pair and `MOJOLEARN_TRANSFORMER_LEGACY_SETUP`, `transformer_decode_pyapi.mojo:596-663` three-way door, the engine token loop `native_generation_engine.mojo:269-340` and `_read_resident` (`causal_lm_public.mojo:2301`), the excluded tape API (`forward_with_tape`/`backward_from_tape`, `_transformer_saved_backward` :963, `native_tape_dispatch.mojo:73`) | L |
| **X5 `mamba-on-executor`** | Mamba1/2/3 `forward/step/backward` and the three decode sessions as `OPS_M{1,2,3}_STEP` (after D1 made the sessions public and D2 the N-token entry) | `_mamba{1,2,3}_run` per-call bodies (`mamba_shell_impl.mojo:194,500,1122`) and their `*_forward_binding` / `*_decode_step_binding` registrations, `_m3_state_call` (`samba_inference_native.mojo:105-180`), the `_RESIDENT_SESSION_NOTE`, the host-side state snapshot helpers (`mamba_public_native.mojo:105-124`), `mamba{1,2,3}_decode_native_handle_v1` (section 5.1) | L |
| **X6 `samba-on-executor`** | `SambaStack.forward/step/train_step` as one program of embedding, per-layer step ops, final norm and head, with the hidden state device-resident (owed row N094) | the three-tier driver `samba_inference_native.mojo:283-295` (resident, provider, per-layer primitives), `samba_training_native.mojo:94-221` fallback walk, `training/samba_resident.mojo` session buffers that the program's parameters replace, `rules_samba.mojo` launch-inventory wrapper (the per-layer rules stay) | M |
| **X7 `byte-lm-on-executor`** | `SmallByteLanguageModelTrainer.train_step/train_steps` as `Program.fit` over the transformer step op (`train_steps` = epochs of one batch) | the `ByteTrainer` per-layer forward loop (`training/byte_lm.mojo` 8 `llama_decoder_layer_forward` sites), `_byte_backward_tape` :1908 wrapper, the `while k < n_steps` driver (`byte_lm_trainer_native.mojo:1250`) | M |
| **X8 `cnn-on-executor`** | `CNNClassifier.fit/predict` as a captured program over `OP_RN_CONV_*` + pools + `OPS_LINEAR` + `OPS_CROSS_ENTROPY`, `Program.fit` for the epochs | `x_cnn/device_entries_impl.mojo:151-185` epoch step and `conv_block_forward_into` (7 sites), `rules_cnn.mojo:248` wrapper, the per-layer `Conv2d.forward/backward` doors kept only as the primitive API | M |
| **X9 `moe-and-gcn-rules`** | MoE forward/backward as tape rules over `OP_MOE_*` (the `AUTOGRAD_MOE` route op `route_ops.mojo:73` becomes the only path); GCN/SAGE after one sparse aggregate op | `moe_backward_run_impl` hand VJP chain (`ops_impl.mojo:1946`), `xcnn_graph_native.mojo` hand backward door | M |

Order: X1, then X2 and X3 (small, prove the pattern on lanes the board already races: `mlp-train-step`, `mlp-predict`,
`sequential-predict`, `gru-fit`, `lstm-fit`), then X4 and X5 in parallel (disjoint files), then X6 (needs both),
X7 (needs X4), X8, X9. Each lane's A/B is "program vs today's path, same bits, ms per call" on NVIDIA and AMD, one
run per arm, on the lane's board rows; identity lanes `transformer`, `mamba1/2/3`, `samba`, `mlp`, `byte-lm`
(`tools/identity_lanes/neural.core`) are the proof at the one pre-release verify.

## 3. Classical and trees: ONE admission path

### 3.1 The copies (what a public door does to X, y and sample_weight today)

| helper (def) | cast | finite scan | copy vs borrow | per-element loop | call sites / files |
|---|---|---|---|---|---|
| `classical_host_f32` `bindings/classical_arrays.mojo:288` (-> `classical_check_x` `classical_validate.mojo:82,104`, `_cast_x` :68, `classical_check_y` :147, `classical_check_weights_input` :190, `family_cast_f32` `family_common.mojo:197`) | byte-assembled per element | inline per element, **runs even when X is a borrowed float32 C-order array** (`core/classical_arrays_impl.mojo:11-21`; the 13 ns/element pass, REGRESSIONS C1) | borrow when f32 C (:321) else `classical_array` zero-fill + copy | serial rows x cols x bytes | 58 / 27: ridge, ols, score, qn, logistic (x5), ipca (x7), kpca (x5), fa, ica, nmf, lda, rp, tsvd, pca_exact, prep_*, mcd, linalg_decomp, parallel_public |
| `family_f32_snapshot` `family_common.mojo:329` (-> `family_snapshot_f32` :317, `family_input_snapshot` :277, aliases `family_as_f32` :336, `family_as_f32_dense` :341, `svc_f32` / `svc_snapshot` `svc_common.mojo:164/173`) | `family_cast_rows` `core/family_cast_impl.mojo:18` byte-assembled | **none**; callers add `family_all_finite_f32` (:109, serial scalar, 28 sites / 20 files) or `p3nc_all_finite_f32` (`p3nc_scan.mojo:20`, SIMD x8 one thread, 14 / 8) | borrow when `snapshot.f32_c()` else `family_array` zero-fill + copy | serial | 42 + 59 + 10 + 10 + 5 sites: clustering, neighbors, SVM, GP, mixture, kernel methods, manifold, linalg, msel, parallel_shard (x9), lifecycle pyapis |
| `family_cast_rows_finite` `core/family_cast_impl.mojo:27` | typed cast + finite in one pass | in the same pass | copy into caller storage | **host_parallelize over row tasks with the grain floor** (:42-44,68): the only parallel cast | 2 / 3: `pca_input_pyapi.mojo:81`, `decomp_input_pyapi.mojo:84` |
| `trees_f32` `bindings/trees_public_inputs.mojo:425` (-> `trees_view` :319, `trees_f32_view` :435, `trees_forest_layout` :494, `trees_empty` :139) | **py_call into the base binding** `cast_elements` (`trees_helper` :96 -> `core/hotpath_impl.mojo:221`, SIMD but `_tasks` returns 1, :62-65) | none; callers add `trees_all_finite` :507 (py_call to the base scan, 20 / 8) | borrow when `<f4` in the wanted order, else `trees_empty` (`array.array * count` zero-fill) + cast | SIMD, one task | 46 / 11: forest, boost, gbdt, gbdt_fit, iforest, meta, score, auxiliary, agnostic, explainer, lifecycle |
| `labels_f32` `bindings/classical_labels.mojo:460` (-> `labels_view` :354, `labels_f32_view` :470, `labels_empty` :186, `labels_all_finite` :529, `labels_check_sample_weight` :540) | as trees (near-verbatim copy of :139-600: 81 of ~415 lines differ after renaming) | as trees | as trees | as trees | x_linear (x6), pls, prep_encode, prep_select, mcd, linalg_decomp, neighbors_knn/local, parallel_neighbor_worker |
| `pca_input_snapshot` `providers/python/pca_input_pyapi.mojo:22`, `decomp_input_snapshot` `decomp_input_pyapi.mojo:18` | `family_cast_rows_finite` (parallel) | `family_all_finite_f32` (serial) when borrowed (REGRESSIONS C2) | borrow or parallel copy | parallel cast, serial scan | 2 / 3 and decomp lifecycle |
| `LinAdmission` `bindings/linalg_ops_buffers.mojo:483` (-> `lin_admitted_c` :629, `lin_borrow_array` :591, `lin_convert` :332) | memcpy per contiguous run, else element loop (:375) | none (metrics pyapis add `family_all_finite_f32`) | borrow when same dtype + C-contiguous (zero-copy `PyMemoryView_FromMemory`), pin | serial element path | 20 / 8 and 12 / 7: linalg ops, lowbit, resample, metrics |
| `xseq_cast_nd` `core/xseq_marshal.mojo:30` (-> `xseq_f32` `xseq_native_common.mojo:125`, `tsa_buffer` `tsa_admission.mojo:56`, `training_f32_cast_owned` `training_buffer_pyapi.mojo:62`) | byte-assembled N-d odometer | `xseq_all_finite_f32` :123 serial (training_buffer only) | always a copy except `xseq_f32` | serial | 9 / 10 + 11 + 8: TSA, forecast, embedding, xcnn, xseq_neural |
| `neural_block_f32` `bindings/neural_block_admission.mojo:380` (-> `_neural_input` `neural_block_public.mojo:467`) | none (f32 only, refuses other dtypes) | none | Array C -> borrow; owned -> memcpy; F-order -> gather; buffer -> pin + `layout.copy_to` into a zero-filled `generation_array` (REGRESSIONS C5) | memcpy | 9 / 5 and 8 / 5 |
| `prep_float_finite` `core/prep_boundary_impl.mojo:8` (+ `prep_gather_rows` :32, masks, categories) | n/a (finite on original words) | serial per element with a per-element rank loop and byte loop (:16-24) | n/a | serial | prep_target, prep_transform (x2) pyapi |
| `mcls_read_i64` / `mcls_narrow_i32` `core/mcls_impl.mojo:11/31`, `ann_candidates_i32` `core/neighbors_ann_impl.mojo:30` | integer labels / ids | n/a | copy | serial, two passes | mcls_common, metrics_vector_input, mcls_cluster, neighbors_ann |
| sample weights | `classical_check_weights_input` (:190), `classical_host_weights` (`classical_arrays.mojo:337`, **0 callers**), `family_sample_weight[_snapshot]` (`family_common.mojo:385/467`, serial fill/finite/min-max `core/family_weights_impl.mojo:15-58`), `trees_check_sample_weight` (:518, py_call `prepare_targets` for finite/sign/zero), `labels_check_sample_weight`, `labels_sample_weight_f32` (:1505): six admitters | | | | |
| label encoders | `classical_encode_labels` (`classical_labels.mojo:2082`), `labels_encode_labels` :1110, `trees_encode_labels` (`trees_public_inputs.mojo:834`), `svc_encode_1d` (`svc_common.mojo:215`), `neural_encode_labels` (`neural_labels_native.mojo:137`): five | | | | |
| zero-fill allocators | `native_array_zeros` (`native_array_storage.mojo:42`, `PySequence_Repeat`), `trees_empty`, `labels_empty`, `classical_array`, `family_array`, `sequential_array`, `lin_new`, `generation_array`, `tsa_array`: eight (REGRESSIONS C3) | | | | |
| metadata structs | `ArrayMetadata` (`array_metadata_pyapi.mojo:13`, 35 sites), `TensorView` (`optimizer_public_door.mojo:284`), `TreesView`/`TreesMatrix` (`trees_public_inputs.mojo:186/408`), `LabelsView`/`LabelsMatrix` (`classical_labels.mojo:229/443`), plus `CastIn` (`core/cast_in.mojo:88`): four plus the executor's own | | | | |

The 61 `# native input marshaling` loops, by owner (file:line): `classical_arrays_impl.mojo` 11, 12 (`classical_cast_rows`),
29 (weights); `family_cast_impl.mojo` 18, 19 (`family_cast_rows`), 51, 52 (parallel, `family_cast_rows_finite`), 80
(`family_min_f32`), 92, 93 (f64 cast), 112 (`family_all_finite_f32`), 122 (`family_max_f32`); `family_weights_impl.mojo`
15, 21, 26, 40, 45, 58; `prep_boundary_impl.mojo` 16, 19, 24 (`prep_float_finite`), 36, 40, 42, 43 (`prep_gather_rows`),
49, 62, 64, 75, 104, 111, 121; `xseq_marshal.mojo` 26, 30, 33, 38 (`xseq_cast_nd`), 50, 56, 93, 94, 97
(`neural_cast_i32_2d`), 106, 107; `mcls_impl.mojo` 16, 19, 35, 46, 55; `neighbors_ann_impl.mojo` 16, 38, 39, 43, 44;
`classical_pandas_impl.mojo` 26, 36; `pca_exact_impl.mojo` 8, 16; `neighbors_graph_impl.mojo` 11; `mani_impl.mojo` 9;
`decomp_native_impl.mojo` 20; `cfb_impl.mojo` 18. Untagged serial scans doing the same job: `xseq_marshal.mojo:125,134,147`,
`core/host_helpers_impl.mojo:30,40` (`all_finite_f32/f64`, 48 sites / 35 files), `linalg_ops_buffers.mojo:375`,
`core/host_sparse_impl.mojo:49`.

What exists to build on: `AdmittedCall` (`bindings/admitted_call.mojo:105`, `AdmittedInput` :68, built at 8 sites,
`inputs` filled at `classical_validate.mojo:128` and `family_public_lifecycle_pyapi.mojo:236`, **read by no cast
helper**); `array_metadata_once` (`array_metadata_pyapi.mojo:135`, one `CastIn` capture, `f32_c()`); the device finite
scan `device_all_finite` (`core/input_device.mojo:117`, wired only behind the base binding's `all_finite_f32`
`bindings/_mojolearn.mojo:1378` -> `hotpath_device.mojo:869`) plus `device_transpose_to_f32` :187 and
`device_strided_copy` :281; the grain floor (`core/host_predict_threads.mojo:64,148,172,186`, `core/host_parallel.mojo`);
`family_lifecycle_begin` (`family_public_lifecycle_pyapi.mojo:177`, X at :231-253: DeviceRows, sparse, snapshot, scan,
replace, records on `AdmittedCall`; 7 callers), the closest thing to the one path; `device_rows_door.mojo:43-92`
(`DoorRows`, `door_rows_host`); `bindings/native_model_records.mojo` (`records_slot_resident` :566) for a remembered
upload; `p3nc_fitted_freeze.mojo:29` for the read-only rule.

### 3.2 The one path

`admit_x(call: AdmittedCall, arg, spec) -> AdmittedX` in `bindings/admitted_input.mojo` (new; the module name follows
the `AdmittedCall` contract of the ne1 brief; a lane that needs more adds a field and says so):

1. **Metadata once**: `array_metadata_once` on the exporter (ndarray, Array, buffer, pandas, torch, list through
   the existing `classical_view` materializer, DeviceRows through `door_x`); record it in `call.inputs[k]`. Every
   later helper reads `call.inputs[k]`; a second cast of the same argument in the same call is a refusal in a tools
   source guard (`tools/check_admitted_call.py`, workstream E).
2. **Borrow or cast**: when `f32_c()` (or the family's wanted order), borrow: no copy, no host pass, the exporter is
   pinned for the call (as `LinAdmission` does). Otherwise one typed cast into **uninitialised** storage (one new
   allocator replacing the eight zero-fill allocators; REGRESSIONS F3), `host_parallelize` over row tasks with the
   grain floor, a typed load per item kind (never byte assembly), `family_cast_rows_finite`'s shape generalised to
   N-d strides (so `xseq_cast_nd` and `lin_convert` fold in).
3. **Finite**: for GPU doors, after the upload the kernel driver makes anyway: `device_all_finite_span`
   (`hotpath_device_impl.mojo:9`), one status word read back with the result the door already downloads (no new
   sync). For host doors and the CPU column, the finite test rides in the parallel cast (borrowed input: one SIMD
   parallel scan, `p3nc_all_finite_f32` made parallel). Refusal wording unchanged (`_check_x_view` :132 is the one
   place for the sklearn text).
4. **Upload once**: `AdmittedX` carries the device buffer handle when the door is a GPU door; a DeviceRows fold is
   passed through as that handle, never materialised to host rows (`door_rows_host` deleted; the arena program's
   `_rows_gather` is the model). Where a family keeps a resident copy of X (k-NN, SVC), the handle goes to the
   family's `records_slot_resident`, with the read-only rule.
5. **y and sample_weight**: `admit_y` (float target: same path, `ndim=1`; labels: ONE encoder, `classical_encode_labels`
   kept, the other four become aliases then go) and `admit_weights` (one admitter: fill / finite / sign / min-max as
   one parallel pass on the device for GPU doors) with the same `AdmittedCall` record.
6. **Numerics**: the bytes reaching the kernel are the same float32 C-order (or F-order for GBDT) bytes; the finite
   test is a predicate, not arithmetic. No fold order is touched. The host column casts the same bytes.

### 3.3 Lanes (each: move the family's doors onto `admit_x`, delete its copy)

| lane | moves | deletes | size |
|---|---|---|---|
| **A0 `admit-x-shared`** | `bindings/admitted_input.mojo` (`admit_x`, `admit_y`, `admit_weights`, `AdmittedX`), the uninitialised allocator, the parallel typed cast, the device finite word, `classical_check_x` and `family_lifecycle_begin` rewritten as thin callers; `classical_host_f32` becomes `admit_x` for its 58 sites with the same signature | `classical_cast_rows` (`classical_arrays_impl.mojo:7-21`), `family_cast_rows` (:18-19) and `_f64`, `family_cast_rows_finite` (absorbed), `family_all_finite_f32`, `p3nc_all_finite_f32` (callers become no-ops: the scan is on the device), `classical_host_weights` (0 callers, now), `door_rows_host` | M |
| **A1 `admit-trees-labels`** | trees and x_linear/pls/prep/neighbors doors onto `admit_x` / `admit_y` / `admit_weights` | `trees_f32`, `trees_view`, `trees_f32_view`, `trees_empty`, `trees_all_finite`, `trees_check_sample_weight`, `trees_prepare_weights`, `TreesView`/`TreesMatrix` (`trees_public_inputs.mojo:139-600`), the whole `classical_labels.mojo:186-600` copy (`labels_*`, `LabelsView`/`LabelsMatrix`), the py_call cast path (`trees_helper` :96 -> base `cast_elements`), `_tasks = 1` (`core/hotpath_impl.mojo:62`) | M |
| **A2 `admit-family`** | clustering, neighbors, SVM, GP, mixture, kernel methods, manifold, msel, parallel_shard onto `admit_x` | `family_f32`, `family_cast_f32`, `family_f32_snapshot`, `family_snapshot_f32`, `family_as_f32[_dense]`, `svc_f32`, `svc_snapshot`, `family_cast_view[_f64]`, `family_owned[_snapshot]` (the borrow rule lives in `AdmittedX`), `family_sample_weight[_snapshot]`, `_family_weight_buffer`, `core/family_weights_impl.mojo`, `labels_encode_labels`, `svc_encode_1d` | M |
| **A3 `admit-decomp-prep-metrics`** | PCA/decomp/IPCA/KPCA/FA/ICA/NMF/LDA/RP/TSVD/MCD/PLS, prep, metrics, linalg ops onto `admit_x` | `pca_input_snapshot`, `decomp_input_snapshot`, `LinAdmission`/`lin_admitted_c`/`lin_borrow_array`/`lin_convert`/`TensorView`, `metric_input_snapshot`, `prep_float_finite`, `prep_gather_rows`, `mcls_read_i64`/`mcls_narrow_i32`/`ann_candidates_i32` (one `admit_ids`), `host_sparse_dense` zero-fill | M |
| **A4 `admit-sequence-tsa-neural`** | TSA, forecast, embedding, xcnn, xseq_neural, byte-LM trainers, neural block inputs onto `admit_x` | `xseq_cast_nd`, `xseq_f32`, `tsa_buffer`, `_series_major`, `training_f32_cast_owned`, `neural_cast_i32_2d` (-> `admit_ids`), `neural_block_f32[_owned]`, `_neural_input`'s zero-filled `generation_array` (REGRESSIONS C5), `xseq_all_finite_f32`, `neural_encode_labels` | S |

A0 compiles first and alone (it is the shared layer; lesson 8); A1-A4 have disjoint files and run in parallel. Every
`# native input marshaling` tag goes with its loop; the tag itself is retired (the one path has no per-element host
loop over X). Proof: every identity lane (the door is on every path); REGRESSIONS F1/F2/F3/F5 are closed by A0-A4
rather than patched in place, so the regressions lane's per-helper fixes are subsumed (F1 as a stopgap is fine
before A0 if the release needs it; A0 then deletes the stopgap too).

## 4. Beyond neural: iterative device loops that fit the executor

The executor's model: resident state, a bounded `while_loop` with loop-carried buffers, a convergence criterion as a
program value, one native call per fit. What `while_loop` costs today: one scalar read per iteration (`_predicate`
`program_control.mojo:44-52`) and one copy + sync per carried tensor (:55-85); after X1 (in-place state) the copy goes,
and a body may unroll k iterations between predicate reads (the chunk-polling the hand loops already do). Op kinds
present beyond the 90 session codes: row argmax (`OP_AGM_ROW_ARGMAX`), bitonic sorts (`OP_LS_BITONIC`,
`OP_ROW_BITONIC_BLOCK`), sort-by-id + fold = segmented reduce by key (`autograd/embedding_ops.mojo:23-31,54-61`), the
64-term blocked fold (`autograd/fold.mojo:12-19`), cumsum, Philox dropout RNG. Missing everywhere below: Cholesky +
trsm, eigh (any size), QR/TSQR, Jacobi SVD, CSR SpMV/SpMM, stream compaction with a device-side count, tree and
histogram builders, time-serial recursions (Kalman), digamma/lgamma/gamma draws.

| algorithm | driver loop today | per iteration: device / host crossings / re-admission | executor needs | bits risk | gain | wave |
|---|---|---|---|---|---|---|
| **LDA online** | `x_decomp/lda_fit_dev.mojo:20-28,114` -> `kit_device.mojo:1478,1506` | ~15 launches / C, E[log beta] k·v downloaded twice per epoch / **X n·v re-uploaded every epoch** (`kit_device.mojo:1501`), C/ED uploaded twice (:1502-1503) | nothing new: X as a parameter, C/ED as in-place state | none | X once instead of max_iter times; 4 transfers per epoch gone | **first** (a defect) |
| **cross_validate / GridSearchCV / RandomizedSearchCV / learning_curve / permutation_test_score** | `bindings/msel_cv.mojo:504` folds, `msel_search.mojo:321-324,522-535,820-842` | one gather per fold side from resident X (`MselFoldRows` :281-397) / **classical estimators download the gathered fold rows** (`classical_check_x` -> `door_rows_host` `classical_validate.mojo:94-120`), re-check, re-upload; only x_prep reads DeviceRows on the device (`_mojolearn_x_prep.mojo:355`); fold host copy not cached (:338-345) | A0 step 4 (DeviceRows passed through as a device handle); the permutation program `msel_search.mojo:740-766` is the model | none | candidates x folds x (download + re-cast + upload) of the fold gone; the largest classical gain outside the door | **first** (A0 closes it) |
| **RFE** | `bindings/prep_select_native.mojo:264` | gather program + public fit + step/rank / support + ranking down per round / **X re-put every round** (:112,270) | X as a parameter of one program | none | X once | **first** (defect) |
| **Stacking / Voting / OneVsRest / MultiOutput / CalibratedClassifierCV** | `trees_public_meta.mojo:863,886` (folds x members, `_fit_member` :344-349), :633, :1525-1527, :1130-1276, :1801-1817 | host `x_trees_gather_f32` of Xtr/Xte per fold (`trees_public_boost.mojo:213-219`); one public fit per member / class / column re-admits the same X | A0 (admit once per call, members take `AdmittedX`); a batched k-target fit for linear members (ridge multi-RHS path exists :1123) | none | m / k / columns re-admissions of X gone | first (A1) |
| **AdaBoost** | `trees_public_boost.mojo:1147` (clf), :1222 (reg) | RF tree fit + predict + SAMME step / codes n + 4 f64 stats + weight scale per round (:1162-1179) / **weights rebuilt f32 on the host per round** (:1070-1075,1150); regressor re-uploads rows + gathered y per round | weights as in-place state; SAMME step as device ops (reduce exists); the RF builder stays a hand kernel called from the program | SAMME fold order kept (same kernels) | 3 round trips + a Python member clone per round gone | second |
| **Logistic OvR** | `glm/estimator.mojo:817` | one QN fit per class / weights per class / **X copied device-to-device per class** (:821-824) | X as one parameter shared by the k fits | none | k·n·d copies gone | first (defect) |
| **GMM EM** | `mixture/estimator.mojo:1321` | 12-20 launches (E-step gemm/logsumexp `checks/estep.mojo:1148+`, M-step colsums/gemm/cov/fused Cholesky `checks/mstep.mojo:1903`) / 1 drain: mean loglik + pivot flags (:1364-1367, `mstep.mojo:1914-1917`) / none | `while_loop` shape exactly; needs Cholesky + logdet as ops (the fused per-component kernel can be the op body), the collapse flag as a state word | none with the same kernels; **high** if the executor's gemm/softmax/fold replaced `identical_gemm`/logsumexp | max_iter drains -> 1 | second (after X1; the docstring :1150-1161 defends the host test, so the criterion becomes a program value) |
| **BayesianGMM** | `x_cluster/bgmm.mojo:359` | ~8 / 2·SC_SIZE floats per iter (:377) / none | as GMM + digamma | ff fold order kept | max_iter -> 1 | second |
| **Lasso / ElasticNet CD** | `solver/impl/cd.mojo:1778,1843,1883` (feature loops :1786,1845,1886) | 2-3 launches per feature per sweep / 1 sync per sweep: coef_max, diff_max (:1827,1872,1923) / none | max-to-scalar reduce (blocked fold exists); the sequential feature dependence stays in the body; the IDENTICAL Gram arm `:1721` is already one fused kernel per `CD_IDN_GRAM_EPOCHS` with the stop word in the launch (:1731-1735): the model | low with the same kernels; a reorder is bits | 1 sync per sweep -> 1 per fit for the non-Gram arms | second |
| **GLM quasi-Newton (logistic, QN SVC/SVR)** | `glm/impl/qn/qn_solvers.mojo:210` (OWL-QN :477), via `glm/estimator.mojo:641` | 1 direction + 5-8 per line-search trial (`glm_base.mojo:2063-2194`) / 1 sync per trial: loss, reg, gnorm (:2179), Wolfe dot (`qn_linesearch.mojo:118`), convergence on host (`qn_solvers.mojo:100`) / none | the line search on the device as a body with a trial counter; `qn_dconv.mojo:427-461` (Apple FAST: 1 sync per 8 iterations) IS the target shape and could be the IDENTICAL body too | none with the same one-block folds; matmul-based dots would change bits | iters x trials syncs -> iters / 8 | second |
| **GLM Poisson/Gamma/Tweedie, Huber grid, LogisticRegressionCV, BayesianRidge** | `x_linear/device.mojo:3568,3617-3645,5612-5705`, `lbfgs_device.mojo:212-231` | flag + slope per iter, >= 1 sync per Armijo trial on the host | as QN; p x p solve as an op (small), Jacobi eigh for BayesianRidge | none | (1 + trials) syncs per iter -> 1 per fit | second |
| **SGD / Perceptron / PA** | `x_linear/device.mojo:1852,3028,1528` | permutation + gather X + gather y per epoch (:1868-1886) / 1 sync per epoch: 2 flags (:2120) / X re-gathered D2D per epoch n·d | `Program.fit` already does epochs, shuffle, minibatches on the executor (`program_runtime.mojo:243-260`): this IS a `Program.fit` with the hinge/log/epsilon losses as loss kinds | minibatch fold order is load-bearing: keep the same gather + kernel sequence | epochs syncs -> 1; one code path with `Program.fit` | second |
| **GP regressor / GPC Laplace** | `gaussian_process/gp_optim.mojo:425,429` (L-BFGS-B evaluations), `classifier.mojo:347`, `gpc_ovr.mojo:336` | 12-15 launches incl. potrf + 2 cho_solve / 1 stop word per evaluation (:475) + potrf `info` drains (1 + ceil(n/nb), `cholesky/checks/potrf.mojo:1989,2133`; deferred only on Apple :1124,2021) / none | Cholesky + trsm + logdet as ops; the `CHOL_DEFER_INFO` deferral on every vendor (the info word joins the predicate) | none with the same kernels; a refold of grad/logdet is bits | evals x (2 + ceil(n/nb)) syncs -> 1 | third |
| **KRR** | no loop; potrf panel loop `potrf.mojo:2039` | `info` per panel (:2133) | `CHOL_DEFER_INFO` on every vendor | none | ceil(n/nb) syncs per fit | third (shares the Cholesky op) |
| **SVC / SVR SMO** | `svm/impl/smosolver.mojo:856` outer; inner steps in `smo_grid_solve_kernel` :389,878 | 15-25 launches (working set sort `workingset.mojo:246-410`, tile gather + GEMM `kernelcache.mojo:308-365`, block solve, compaction, update_f) / 1 drain per outer iter inside `select_i32` (`svm/checks/device_select.mojo:355`: the nnz count sizes later grids) / kernel tile recomputed each iter (:869) | compaction with a device-side count (grids fixed to n_ws so no size comes back), the sort as an op, the SMO block solve as a kernel body | keep the sort and `fold_order` (DEVIATION 634), update_f gemm order | n_outer syncs -> 1 | third |
| **ARIMA lock-step L-BFGS** | `arima/impl/batched_fit.mojo:611,628` | prelude + per trial candidate / eval / accept (`eval_batch_device` :219; D2D y replication :274-283) / any_active flag per iter (:612,417), any_searching per trial (:629) + eval drains (:321,366,373) | the async FAST arm (`fast_lbfgs_async.mojo:455`, flags every 4 rounds :93) is the shape; Kalman stays a kernel body | Kalman order kept | (1 + 2·max_ls) syncs per iter -> per 4 | third |
| **MDS SMACOF** | `x_decomp/mds_dev.mojo:78,116` | ~14 (+6 isotonic) / stress + ssd words (:93,95) / none | isotonic (sort + group sums: both exist as op kinds) | none | 2-3 syncs per iter -> 1 | second |
| **Label propagation** | `x_neighbors/iter_device.mojo:254` (C33 chunk :234-246), compact fallback `neighbors_semi_native.mojo:784` | ~5 / 1 absdiff word per iter (:257-258); C33: per 16 / fallback allocates n·c per iter (:793) | ELL SpMM as an op; the C33 chunk pattern is the body | fixed tree fold kept | 1 sync per iter removable; the fallback's allocation per iter gone | second |
| **NMF multiplicative / CD** | `x_decomp/nmf_dev.mojo:65,182` | 6 gemm + 2 ew; CD: 2 gemm + cd_rows + total per side (`kit_device.mojo:783-803`) / 1-3 words per 10 iters; CD: 1-2 words per iter + perm download (:776) / **CD re-uploads the permutation k i32 per side** (:792-794), identity k·k when l2 != 0 (:158) | perm as a device value (Philox + sort exist) | keep `cd_rows_kernel` | 1-4 syncs + 1-4 uploads per iter gone | second |
| **FastICA parallel / deflation, FA EM, FA rotation, IPCA, KernelPCA Lanczos, Isomap, Spectral Lanczos, LLE** | `x_decomp/ica_dev.mojo:60,88`, `fa_em_dev.mojo:32`, `rotation_dev.mojo:47`, `ipca_owner.mojo:348`, `kpca_owner.mojo:107` + `lanczos_dev.mojo:96`, `mani_decomp_native.mojo:257`, `spectral/.../lanczos.mojo:2510`, `lle_iter_dev.mojo:57` | the eigen / SVD step goes **home**: w, V downloaded, sorted or reversed on the host, re-uploaded (`kit_device.mojo:915-943`, :555-569; `ipca_owner.mojo:79-111`; `lanczos_dev.mojo:112-144`; `lanczos.mojo:945-960,2159-2162`; `lle_iter_dev.mojo:37-72`) per iteration / restart / batch | **one device `eigh` op** (k x k and tridiagonal; `_eigh_on` exists on the device for ICA, `x_linear/ops.mojo:516` Jacobi for BayesianRidge), a device sort, QR for LLE | **the eigh itself must be the same algorithm on all three vendors**; moving a host LAPACK/NumPy eigh onto the device changes bits (allowed between versions, host column changed with it) | >= 2 syncs + 2-4 uploads per iteration gone | third (one lane for the eigh op, then the families) |
| **PCA / TSVD randomized** | `x_decomp/pca_randomized.mojo:698` (TSVD `tsvd_owner.mojo:58`; binding `linalg_decomp_native.mojo:119` dispatches each op through `py_call`, `decomp_kit_native.mojo:695,1369`) | 2 gemm + 2 orth / 2-6 orth syncs that read **nothing** (`s_linalg_fast.mojo:265`, `device.mojo:2445,2452`) / none | QR/TSQR orth as an op; the syncs are deletable now | none with the same `qr_factor` + gemm order | 18-54 syncs per fit; 4 Python dispatches per iteration on the binding path | **first** (delete the bare syncs; the binding's `py_call` loop joins A3) |
| **ALS** | `x_decomp/als_dev.mojo:79,85` | gemm + per-row f x f solve x 2 / **two bare `synchronize()` reading nothing** (:31,49) + 3 loss words (:59-61) / none; `flags` written, never read (:22-30) | nothing (already resident) | none | 2 syncs per iter deleted; dead `flags` buffer deleted | **first** (defect) |
| **Dictionary learning** | `x_decomp/dictl_dev.mojo:97,130` | encode (lasso/LARS/OMP rows) + 2 gemm + update / 3-4 words per iter (`kit_device.mojo:639`) / none | row solvers as kernel bodies | none | 3-4 syncs per iter | third |
| **GBDT rounds** | `gbdt/methods/doc_parallel_boosting.mojo:3342` | ~15-25 + 6·depth / 4-5 syncs per round: split winners (`searcher :1162-1170`), partition bounds (`doc_parallel_leaves_estimator.mojo:416`), leaf values (:4489), held-out loss (:645; fused: 1 per 1024 rounds :715-751) / split records uploaded 5 copies per tree (`sym_iter_fast.mojo:270-276`) | histogram (segmented reduce by bin), split argmax, stable partition, tree gather, bootstrap RNG, the host gate walk (:1193-1230) as device ops: a tree builder is not an op the executor has | none with the same kernels; a device early-stop detector that refolds the loss is bits | ~4T syncs per fit | later (not an executor fit until the builder is an op; the depth levels :881 are already resident) |
| **RandomForest / ExtraTrees / DecisionTree levels** | `ensemble/randomforest.mojo:3163,3281`, `builder.mojo:3125` (`LOOP_K=4`), `extratrees/.../builder.mojo:7074,6691` | 1 drain per cycle shared by in-flight trees (:3296), 16-word header per 4 batches (:3321,3340-3350) | as GBDT | none | 1 drain per tree + 1 per 4 batches: little | later |
| **KMeans Lloyd (DC arm), kmeans++ NOSYNC, k-means\|\|, AP, mean shift, DBSCAN CC, OPTICS, HDBSCAN Boruvka, MiniBatch FAST, QN dconv, ARIMA async, UMAP, t-SNE, Holt-Winters, Prophet default, GARCH/ETS/Theta, IsolationForest, GBDT depth levels, IterativeImputer built-in** | `cluster/impl/detail/kmeans.mojo:1767-1850` (2 int32 per 8 iters), `x_cluster/device_ops.mojo:3305-3325` (2 int32 per 8), `meanshift_idn.mojo:252-278` (per 16), `dbscan/.../csr.mojo:410-449` (per 8), `sparse_mr_mst.mojo:1499,1599` (1 per round, log m rounds), `umap/optimizer_identical_device.mojo:639-684` (0), `x_ann/tsne_device.mojo:874` (0), `hw_optim.mojo:465` (in-kernel), `isolation_forest.mojo:1105-1150` (1 after), `prep_iterimp_native.mojo:726-779` (device-gated stages, one `q.run`) | already resident and chunk-polled or sync-free | nothing to gain; `while_loop`'s one scalar per iteration would **regress** them until X1's unrolled body lands | none | none | never for speed; `IterativeImputer` (:779) is already the resident-program shape and the model for the others |
| **MiniBatchKMeans generic** | `x_cluster/minibatch.mojo:381` | ~6 per step / per group <= 16: energy, weights, **centers g·k·d downloaded** (:411) / batch row ids uploaded every step when weighted (:392-397) | device RNG for the weighted draw; scatter-add by label (segmented reduce exists) | host f64 EWA (`minibatch_fast.mojo:683-710` in the FAST arm) is a stop rule: moving it changes stop bits | g·k·d downloads per group gone | second |
| **Spectral discretize** | `x_cluster/spectral_assign.mojo:232-275` | **host loops over n·k** | k x k SVD op | n/a (host today) | move to device | third (with the eigh op) |
| **Prophet FAST (N >= 16384)** | `sequence/ops_impl.mojo:1546-1609` | 2 launches per f/g evaluation / g (P) + SSE down + sync per evaluation (:1537-1539) / **theta (P) uploaded every evaluation** (:1528) | L-BFGS two-loop + Armijo as a body (`OP_PROPHET_FG_SUM` order kept) | none | <= 40 (upload + sync) per iteration | second |
| **IterativeImputer with a user estimator** | `prep_iterimp_native.mojo:690-700` (`cpu-route`) | take + public fit + put per feature per round | an estimator-in-program (the built-in BR path :726-779 is that already) | none | rounds x features re-admissions | later |

Python iteration loops (defects, `python/mojolearn/`): `_expansion_decomp.py:2691` (`_rsvd_core`, `for _ in
range(n_iter)` over `k.mm` / `k.orth` device calls) is **dead**: public `randomized_svd` :2670 routes to the native
`linalg_decomp_native.mojo:60,119`; its only caller `_randomized_decompose` :5080,5092 has no callers and
`tests/test_pca_exact_native_owner.py:329-330` patches both to fail if reached. `_expansion_decomp.py:4002`
(`_lanczos_top` :3957, `_lanczos_batch` :3944, a `while True` over basis doublings) is dead on the fit path (live twin
`mani_decomp_native.mojo:257`), used only by `tools/kernel_pca_gpu_trial.py:23` and `bench/decomp_fast_quality.py`.
`model_selection.py:513` (`_prepare_folds` :480) has no caller next to native `msel_cv_folds`. No other Python loop
over iterations, epochs, rounds, trees, members, candidates or folds exists in the product modules (test-only hits:
`tests/_svm_reference.py:139`, `tests/_legacy_neighbors_semi.py:247`, `tests/test_ops_optimizer.py:94`). Section 5.4
deletes the three.

Waves: **first** = the re-admission defects and bare syncs (LDA online, RFE, Logistic OvR, ALS, PCA randomized
syncs) plus everything A0 closes (CV folds, meta-estimators); they need no executor change. **Second** = the loops
with exactly the `while_loop` shape (GMM, BGMM, CD, QN and the GLM/Huber/LogCV line searches, SGD as `Program.fit`,
MDS, label propagation, NMF, MiniBatch generic, Prophet FAST, AdaBoost), after X1 lands in-place state and the
unrolled body. **Third** = the Cholesky / eigh / sort / compaction op lane (one lane adds the ops with the same
algorithm on every vendor and the host column; then GP/GPC/KRR, SMO, the decomposition eigen round trips, Spectral
discretize, dictionary learning, ARIMA lock-step). **Later** = tree builders (GBDT, RF, ET) and the estimator-in-program
imputer, which need op kinds the executor is not going to grow this year. **Never for speed** = the chunk-polled and
sync-free loops; they are listed so nobody moves them.

## 5. Dead code: paths nothing reaches

Method: reachability by `rg -w` over `python/mojolearn/**` (public API and tests), `tools/identity_lanes/**` and
`tools/identity_break.py` (identity lanes), `tools/bench_board*.py`, `tools/paper_compare.py`,
`tools/classical_two_datasets.py`, `bench/speed/*.py` (board recipes), the verifier (`tools/verify_*.py`,
`_verify*.py`), `tools/tests/**`; 2,700 native registrations harvested from `def_function[...]("name")` /
`def_py_c_function` in `bindings/` and `providers/python/`, crossed against a word set of `python/ tools/ bench/speed
packaging/ .github pixi.toml`, then every zero-Python name re-checked for a Mojo-side string caller (`family_entry`,
`trees_entry`, `gp_entry`, `kit.entries[...]`). Already deleted and not re-listed: the 9 + 218 entries of
`docs/TOMBSTONES.md` "Dead native entries" (two batches, 2026-10-26); `coverage_declared.tsv` has 0 `dead` rows left;
75 of the 79 "Owed deletions" defines (`TOMBSTONES.md:3192-3288`) are gone from main.

### 5.1 Native entries with zero callers (19 of 61 zero-Python names; the other 42 are reached by a Mojo owner by string)

| entry | defined at | evidence | first on main | action |
|---|---|---|---|---|
| `byte_lm_parallel_apply_gradient`, `_close`, `_fold_add`, `_fold_reset`, `_reduction_pool_available`, `_rollback`, `_shard_gradient`, `_shard_gradient_fold`, `_open`, `_open_pooled` | `bindings/_mojolearn_byte_lm.mojo:758-1011` (registrations :1800-1814) | 0 refs outside the file (only `tools/byte_lm_export_fast/change.patch`, a stale patch); live siblings `_create` (1), `_set_lr` (1), `_step` (5), `_export` (2), `_fold_export` (2), `_ownership` (1) | `5b7c82abe9` 2026-09-21 | delete the ten, one commit; tombstone + `experiments/removed/dead_entries_byte_lm_parallel.patch` |
| `mamba1_decode_native_handle_v1`, `mamba2_...`, `mamba3_...` | `bindings/_mojolearn_mamba.mojo:490,494,498` (regs :1557,1570,1604) | 0 code refs; only an allow-list f-string `tests/test_cpu_training_mamba.py:81`; the transformer twin IS used (`transformer_resident_pyapi.mojo:22`) | `35735229d5` 2026-10-27 | delete (X5 deletes them anyway; do it now) |
| `vocab_encode_public`, `vocab_decode_public`, `vocab_pretokenize_public`, `vocab_pretrained_public`, `vocab_pattern_name` | `bindings/_mojolearn_tokenizer_host.mojo:844,851,1187,1146,1174` (C regs :1347-1352, dispatcher :1205-1217) | 0; `python/` uses only `vocab_size/sample_bytes/bpe/build/fingerprint` | `12fa2ed0f5` 2026-10-20 | delete the five exports and the `_vocab_public_c` arms (the tokenizer host group is the 12 GiB compile, lesson 11: every deletion there helps) |
| `scoped_gemm_probe_numeric_mode`, `scoped_gemm_probe_vendor` = the whole `bindings/_mojolearn_scoped_gemm_probe.mojo` (137 lines) + `build_scoped_gemm_probe.sh` + `tools/scoped_gemm_quality.py` | | 0 product refs; `tools/test_release_no_hands.py:146` lists its `.so` as STALE; not in the release build | `04d5843e14` 2026-10-04 | delete with `MOJOLEARN_SCOPED_GEMM_AUDIT` (5.5) |
| `solver_numeric_mode` | `bindings/_mojolearn_solver.mojo:252` | constructed by the loader (`_backend.py` `prefix + '_numeric_mode'`) | | keep |
| `forest_export_legacy` registration | `_mojolearn_rf.mojo:1777`, `_mojolearn_rf_host.mojo:1098` | live via `trees_public_forest.mojo:536` although the `FOREST_EXPORT=legacy` arm is gone (batch 2) | | keep until that reader goes (A1 touches the file) |

### 5.2 Orphan functions (def only, zero references outside the defining file)

| symbol | defined at | action |
|---|---|---|
| `classical_host_weights` | `bindings/classical_arrays.mojo:337` | delete (A0 deletes the file's loops anyway) |
| `pca_query_fields` | `providers/python/pca_model_pyapi.mojo` | delete |
| `prep_check_info`, `prep_scalar_dtype` | `providers/python/prep_metadata_pyapi.mojo` | delete |
| `rankn_mean`, `rankn_softmax` | `autograd/rankn_fold.mojo:379,456` (def + one check file each) | delete or route `OPS_MEAN_N` / `OPS_SOFTMAX_N` to them: today they are unreachable from any op |
| `flags` buffer in ALS | `x_decomp/als_dev.mojo:22-30` (written, never read) + two bare `synchronize()` :31,49 | delete |
| 147 of 150 zero-ref public fns in `providers/python/*.mojo` | | in-file dispatch helpers (e.g. 30 `host_compat_*`, 8 `low_*`): not dead |
| `autograd/checks` (72 files), `autograd/examples` | run by `tools/autograd_smoke.py` / `tools/check_autograd.py` (`tools/release_requirements.json`) | live, keep |

### 5.3 Stale references

`llama_decoder_layer_backward_device` in two comments (`training/samba_resident.mojo:35`, `modeling_llama.mojo:1757`);
`TAPE_EVERYWHERE_PLAN.md` in five module headers (section 1); `tools/byte_lm_export_fast/change.patch` naming deleted
entries. Fix with the deletion commits (comments are not tombstones; no record needed).

### 5.4 Python

`_expansion_decomp.py:2674-2691` (`_rsvd_core`) and :5080-5092 (`_randomized_decompose`): dead (section 4).
`_expansion_decomp.py:3944-4002` (`_lanczos_batch`, `_lanczos_top`): dead on the product path; move
`tools/kernel_pca_gpu_trial.py` and `bench/decomp_fast_quality.py` to the native binding or delete them with it.
`model_selection.py:480-513` (`_prepare_folds`): no caller. No module in `python/mojolearn/` is unreferenced
(`_surface_metrics.py` is loaded by name through `host_surface.py:136`). `docs-site/api_exclude.txt` (~99 entries) is
the list of shipped-but-unverified surfaces; the strongest deletion candidates are the oldest: the 15 CNN/GNN layer
names ("no NVIDIA+AMD reference records in 0.8.41"), `GPT2Tokenizer` (deprecated alias, blank page), the 11
`parallel_*` facades + `DistributedIVFIndex` ("physical qualification pending"; their reader `_verify_par.py` was
removed 2026-10-31), the transformer tape API (`forward_with_tape`, `backward_from_tape`: X4 deletes it),
`Mamba3Block.*_owned_weights` (X5). These need Andrew's word per name; the plan lists them, it does not decide them.

### 5.5 Experiment defines with a loser verdict whose gate is still in the code

| define | EXPERIMENTS.md verdict | live gate | witness exports to delete with it |
|---|---|---|---|
| `MOJOLEARN_PT_COLBATCH` | DROP (quality) :377; in the TOMBSTONES owed table | `x_prep/fastpt.mojo:79` | |
| `MOJOLEARN_EIGH_FAST_PANEL_DF` | DROPPED, +59.8 % slower | `x_decomp/eigh_tridiag.mojo:180` | |
| `MOJOLEARN_MCD_ORDERED_COV` | NEUTRAL, "still owes its own deletion" (TOMBSTONES batch 1) | `x_decomp/mcd_bmma.mojo:97` | `x_decomp_mcd_cov_probe`, `_reach`; `tools/mcd_ordered_quality.py` |
| `MOJOLEARN_MCD_FAST_G1_GRAM` | NEUTRAL | `x_decomp/mcd_bmma.mojo:44` | `mcd_g1_gram_*` (3) |
| `MOJOLEARN_LLE_FAST_DEV_LU` | HOLD-quality (real drop) | `x_decomp/w4_fast.mojo:67` | |
| `MOJOLEARN_MCD_BATCH_COMPAT` | HOLD-quality | `x_decomp/mcd_fast.mojo:120` | |
| `MOJOLEARN_CHOL_FAST_NB512` | HOLD-speed, no benefit | `cholesky/checks/potrf.mojo:462` | |
| `MOJOLEARN_PRODUCT_DF_TAIL` (F16) | left off; 0 live gates | | `arima_product_df_count` |
| `MOJOLEARN_SCOPED_GEMM_AUDIT`, `SOFTMAX_G2_AUDIT`, `LU_FAST_SHARED_SUB` (F02), `PCA_FAST_COMPENSATED_COV` (F11, "measured slower"), `MCD_FAST_ACTIVE_COMPACT` / `_BOUND_BATCH` (F06), `IVF_FAST_BALANCED_TASKS` (F14, never run), `ARIMA_FAST_K3_DF_PROBE`, `FA_FAST_GRAM_SPREAD` (OPEN), `KSHAP_FAST_DEVICE_MODEL` (OPEN) | no verdict row (`experiments/performance_ideas/F01..F20` hold caller.py / manifest.json only) | one gate each | `scoped_gemm_count/_flags/_metadata`, `softmax_g2_*` (4), `pca_compensated_cov_count`, `gp_shared_sub_count`, `x_decomp_shared_sub_count`, `mcd_experiment_count`, `mcd_fast_batch_count`, `ivf_fast_balanced_hits`, `_arima_k3_df_probe`, the 19 `device_plan_log` rows (reader removed) |
| `MOJOLEARN_HDB_ONE_SYNC`, `HDB_SELECT_DEVICE`, `TREES_T29` | owed table says DROP, later EXPERIMENTS rows say KEPT (`hdbscan/impl/detail/fast_apple.mojo:58,82`, `gbdt/trees_identical_switches.mojo:74`) | live | keep; correct the owed table |

Apple FAST investment is stopped (AGENTS.md 2026-10-22), so the "no verdict" FAST switches are not going to be
measured: Andrew decides measure-once or delete; the plan recommends delete (a FAST switch nobody will A/B is dead).

### 5.6 Duplicate paths, dead by construction once decided

| pair | the decision | deletes |
|---|---|---|
| hand `mlp_predict` / `mlp_forward` vs `mlp_predict_autograd` (`MOJOLEARN_AUTOGRAD_PREDICT` `mlp_fit.mojo:92`, default off, no EXPERIMENTS or TOMBSTONES row) | the tape forward is the one path (X3); A/B recorded in EXPERIMENTS.md with the lane's one run | hand `mlp_predict` :985, `mlp_forward` :232, the define (tombstone) |
| `MOJOLEARN_AUTOGRAD_LOSS`, `AUTOGRAD_OPTIM`, `AUTOGRAD_DATA` (default off, `mlp_fit.mojo:136,157`, `recurrent.mojo:66,80,505`) | each gates an executor path beside a hand path inside an executor fit; same decision as PREDICT: measure once on the lane's board rows (mlp-fit, gru-fit), keep one | the losing arm per define |
| per-call Mamba3 step (`_m3_state_call` -> `_mamba3_run`) vs `_m3_session_step_run` | D1 makes the session the public path; X5 lifts it | `_mamba3_run` and its door (X5) |
| Samba forward tiers (`samba_inference_native.mojo:283-295`: resident, provider, per-layer primitives) | X6: one program | the two fallback tiers |
| `CausalLM` engine loop vs `_read_resident` (`causal_lm_public.mojo:2223 / 2301`) | X4: `Program.decode` | both |
| transformer `forward_fresh` / `session_forward_fresh` / `LEGACY_SETUP` (`neural_block_public.mojo:672-714`, `native_block_dispatch.mojo:102`) | X4 | all three arms |
| the excluded tape API (`forward_with_tape` / `backward_from_tape` / `_transformer_saved_backward`) | X4: `Session.capture` over `OPS_ATTN_STEP` is the tape API | the three |

### 5.7 Tools

1,306 files under `tools/`; 498 `test_*.py` are collected by pytest and live. Of the 808 non-test tools, **140 have no
inbound reference** by stem across `tools/ .github/ docs/ python/ packaging/ bindings/ pixi.toml`, 170 carry probe
names (`probe`, `ab`, `check`, `audit`, `capture`, `diagnostic`). The first sixty by name, with first commit where it
dates them (one-off probes from August-October): `add_source_headers` (08-31), `af_board_full_sync` (10-04),
`af_ident_to_afb`, `agglo_device_check` (10-02), `appendix_delta` (09-19), `apple_byte_lm_first_step` (09-07),
`apple_fast_tree_capture`, `arima_fit_identity_capture` (09-05), `audit_portable_primitive_calls`, `bench_boundary_tax`,
`bench_numpy_free_conversion`, `board_quality_backfill` (10-12), `bpe_trainer_determinism`, `byte_lm_capture_regression`,
`byte_lm_config_binding_check`, `byte_lm_model_pool_capacity_check`, `byte_lm_runtime_numerical_check`,
`byte_lm_session_bench`, `capture_small_training`, `catboost_cat_oracle` (08-21), `check_not_implemented`,
`check_serialization`, `check_trees_resident` (11-06), `checks_that_cannot_fail_probe`, `chol_fast_tall_pair`,
`compare_umap_portable_host_math`, `cpu_public_fit_smoke`, `criteo_dense_arms`, `do_byte_lm_leg`, `fa_diag`,
`fast_speed_diff`, `forest_experiment_board`, `forest_ordered_default_smoke`, `full_ab_watch`, `gbdt_cpu_parity_matrix`,
`gbdt_cpu_refusal_probe`, `gbdt_partition_cache_ab`, `gbdt_prepared_check`, `gbdt_small_probe`,
`gemm_class_section_gate`, `gemm_identical_table`, `gemm_training_cross_vendor`, `hdbscan_idcheck`, `hdbscan_sparse_taxi`,
`identical_candidate_gate`, `identical_dart_wave_receipt`, `identical_family_bit_diagnostic`,
`identical_gmm_power_bit_diagnostic`, `identical_pca_bit_diagnostic`, `identical_quality_batch`,
`identical_source_bit_checks`, `identical_wave_identity_merge`, `identical_wave_v2_stage`, `identity_overhead2`,
`inert_condition_audit` (08-25, 1148 lines), `inf_gbdt_wheel_models`, `istella_rank_summary`, `kernel_pca_gpu_trial`,
`kmeans_linear_profile`, `knn_coalesced_columns_summary`. Caveat: a tool invoked only from `~/mojolearn-ops` or a brief
under `~/mojolearn-evidence/` is invisible to the scan, so the tools lane confirms each against `~/mojolearn-ops/` and
the briefs before deleting; tools are not experiments (no patch), but the deletion commit lists them in one
`TOMBSTONES.md` section as the 2026-10-26 batches did.

### 5.8 Deletion plan (CLAUDE.md rules, applied)

One lane **K0 `dead-code-batch-3`** (S), before any executor lane, one commit per group of 5.1 / 5.2 / 5.3 / 5.4 /
5.5 / 5.7, each commit leaving: (1) an inline `# TOMBSTONE: <name> (<verdict>) deleted <date> by <sha>; code
recoverable at <sha>^.` + `# Restore:` at every registration block and definition site; (2) one `docs/TOMBSTONES.md`
section "Dead native entries, batch 3 (2026-11)" with an index row per entry group and the define rows, plus the
`experiments/tombstones/index_count.txt` bump (367 today); (3) `experiments/removed/<group>.patch` = the reverse of
the commit restricted to code files (everything in 5.1, 5.2, 5.5 reached main; the Python loops of 5.4 too); (4)
for the defines of 5.5: a `removed` refusal in `core/experiment_guards_<family>.mojo` (`decomposition`, `linear`,
`kernel`, `series`, `trees`, `base`), the `grid_controls/*.json` entry moved to `removed`, the
`docs/apple-fast/EXPERIMENTS.md` row (DELETED). The duplicate paths of 5.6 are deleted by their executor lane (X3-X6)
with the same four items, since each has a define or a public door. Bits: none move (no live path changes);
`tools/test_tombstones_preserved.py` and the host manifest test (`python/mojolearn/tests/test_host_surface.py`) are
the fence, as for the 2026-10-26 batches.

## 6. Risk and proof

Bits must not change inside a version; the one pre-release identity run on the frozen source is the proof (CLAUDE.md
"Verification is pre-release only"; no `lq add ... ID` during development). What each lane does to stay inside that:

| lane | could it move bits? | how it does not |
|---|---|---|
| A0-A4 (admission) | no: the kernels read the same float32 bytes; the finite test is a predicate | the borrow case produces no bytes at all; the cast case is a typed load of the same values; F-order for GBDT is the same `transpose_f32`; the host column casts the same way. Risk: a family that depended on a zero-filled copy (reading past the admitted shape) now reads uninitialised memory: the uninitialised allocator is used only where the destination is fully written before it escapes (REGRESSIONS F3 rule), and the compile's sanitizer-free check is one `verify` of every lane |
| K0 (deletions) | no: no live path | the `rg -w` evidence per entry is in 5.1-5.5; the host manifest test refuses a registration the manifest no longer lists |
| X1 (executor state, iteration register) | no for existing programs: `while_loop` keeps `_copy_state` for undeclared state; the register is a new value | the board lane `program-run` and the identity lanes over `ops.Session` (`optim-*`, `training-primitives`) pin the existing programs |
| X2-X8 (step ops) | **yes if** a step op's body were rewritten; **no** when the body IS the session launch set (`_m3_session_step_run`, `transformer_decode_session_step_binding`) and the backward rule records the same stage buffers | the decode plan's D1 test (`test_mamba23_decode_sessions.py`: session step equals per-call step byte for byte) is extended to "program step equals session step" per family; a stage count check per op (the launch inventories at the top of each `rules_*.mojo`) is a tools test. The one place a fold order could change is weights-as-parameters replacing a per-call upload: same bytes, different buffer, no arithmetic |
| X3 (predict on the tape) | no: `tape.no_grad()` launches the same kernels (`NATIVE_AUTOGRAD.md` "Inference": "a recorded and a non-recorded forward give the same bits by construction") | the board lanes `mlp-predict`, `sequential-predict` and the identity lane `mlp` |
| section 4 wave 2 (while_loop bodies) | **yes if** a dot or max reduction were replaced by a matmul or a different fold | the body uses the family's existing one-block folds and kernels as op bodies; `qn_dconv.mojo` shows the shape with the same kernels. Each lane's A/B records the digest of arm A and arm B; a differing digest fails the lane, it is not re-measured |
| section 4 wave 3 (eigh / Cholesky ops) | **yes**: the eigh algorithm moves from the host (LAPACK / NumPy) to the device | allowed between versions; the lane changes NVIDIA, AMD, Apple and the host column together (same Jacobi / tridiagonal algorithm, same sweep order), and the affected identity lanes (`x-decomp-eigh-wide-*`, `gmm`, `tsvd`, `umap`, `spectral`, `linalg-eigh`) re-record at the one pre-release run like every lane does anyway |

The release identity run is the only check; a lane that wants a same-bits check during development has misread the
rules and gets the decode plan's answer: fix in code, verify once at the end.

## 7. The name

`autograd/` (99 Mojo modules, imported from 300 files) is the package of the resident program executor: `OpsSession`,
captured programs, `Program.fit`, control flow, distributed and partitioned programs, the importer, checkpoints,
rank-N layouts, the data ops. The gradient tape (`tape.mojo`, `rules_*.mojo`, `higher*.mojo`) is one of its features,
and after this plan it is the feature that records a forward the executor ran. The Python surface already says so:
the public module is `mojolearn.ops` (`Session`, `Program`), `autograd` appears only in comments and in the user guide
page `docs-site/user_guide/autograd.md`. Proposal for Andrew to decide, no renames in code or docs by this plan:
split the name into **the program runtime** (working name `executor`: sessions, programs, control, fit, distribution,
import, checkpoints, the op table) and **the tape** (`autograd` kept for `tape.mojo`, the rules and higher-order
differentiation, as the gradient feature of the executor). Either the Mojo package is renamed in one mechanical lane
after the X lanes merge (300 import lines; one compile) or only the docs and plan vocabulary change and the package
name stays, with `docs/plans/AUTOGRAD_API.md` and the user guide retitled. The plan's own vocabulary uses "executor"
for the runtime and "tape" for the gradient feature from here on.

## 8. The table

| item | today | target | lane | deletes | size | order |
|---|---|---|---|---|---|---|
| Admission of X, y, sample_weight on every classical / trees / neural door | 12 helpers, 4 cast loops (3 serial byte-assembled + 1 py_call), 9 finite scans (1 parallel), 3 materializers, 6 weight admitters, 8 zero-fill allocators (section 3.1) | `admit_x` / `admit_y` / `admit_weights` on `AdmittedCall`: metadata once, borrow or parallel typed cast into uninitialised storage, finite on the device, one upload, DeviceRows through | A0 shared, then A1 trees + labels, A2 family, A3 decomp + prep + metrics + linalg, A4 sequence + TSA + neural | the 11 copies, 59 serial marshaling loops, `door_rows_host`, `classical_host_weights`, `classical_labels.mojo:186-600`, `TreesView`/`LabelsView`/`TensorView` | M + 4 x (M, M, M, S) | 1 (A0 alone), then A1-A4 parallel |
| Zero-caller native entries, orphans, loser defines, dead Python loops, stale comments, orphan tools (5.1-5.5, 5.7) | registered / gated / present, nothing reaches them | gone, with tombstones, patches, guard refusals | K0 `dead-code-batch-3` | 19 entries, 6 functions, 8 defines + 3 unverdicted FAST groups + their witnesses, the scoped-gemm probe binding, 3 Python functions, confirmed orphan tools | S | 1 (parallel with A0) |
| Re-admission defects outside neural (LDA online X per epoch, RFE X per round, Logistic OvR D2D X per class, ALS bare syncs + dead flags, PCA randomized orth syncs, NMF-CD perm uploads, Prophet FAST theta per eval) | section 4, first wave | admit / upload once; delete the syncs that read nothing | N0 `readmission-defects` | the per-iteration uploads and syncs | S | 1 |
| Executor: in-place carried state, iteration register, block step ops, weights as parameters, `Program.decode` | `_copy_state` + sync per state per iteration; no register; no block op; per-call weight uploads | section 2.1 | X1 | nothing yet | M | 2 (after D1/D2 of the decode plan) |
| SmallMLPTrainer train_step / predict_logits | resident hand session, hand forward, tape backward | captured program, weights as parameters | X2 | `training/mlp_ops.mojo:140-332`, `rules_small_mlp` wrapper, resident plumbing in `small_mlp_native.mojo` | S | 3 |
| MLP and RNN predict | hand `mlp_predict` / recurrent `forward`; `AUTOGRAD_PREDICT` off | inference-mode tape forward | X3 | hand `mlp_predict`, `mlp_forward`, recurrent `forward` + `head`, the define (A/B recorded) | S | 3 |
| `AUTOGRAD_LOSS` / `_OPTIM` / `_DATA` dual arms inside executor fits | two arms each, no verdict | one arm | X3 | the losing arm per define, tombstoned | S | 3 |
| Transformer block, decode session, CausalLM generate | per-call door over a retained session; resident session; Mojo token x layer loop | `OPS_ATTN_STEP` program; `Program.decode` | X4 | `_neural_plain` transformer arms, `forward_fresh` trio, the three-way decode door, the engine loop + `_read_resident`, the excluded tape API | L | 4 |
| Mamba1/2/3 forward / step / backward / decode | per-call door (weights per call) + three sessions (two private) | `OPS_M{1,2,3}_STEP` program | X5 (after D1, D2) | `_mamba{1,2,3}_run` + bindings, `_m3_state_call`, state snapshot helpers, `_RESIDENT_SESSION_NOTE`, `*_decode_native_handle_v1` | L | 4 (parallel with X4) |
| SambaStack forward / step / train_step | three-tier driver; resident train step | one program over step ops | X6 | tiers 2-3 of `samba_inference_native.mojo:283-295`, `samba_training_native.mojo:94-221`, session buffers | M | 5 |
| Byte LM trainer train_step / train_steps | resident session, per-layer hand forward, Mojo step loop | `Program.fit` over the transformer step op | X7 | the per-layer forward loop, `_byte_backward_tape`, the `while k < n_steps` driver | M | 5 |
| CNNClassifier fit / predict | hand chain in a Mojo epoch loop | captured program + `Program.fit` | X8 | `device_entries_impl.mojo:151-185`, `conv_block_forward_into` sites, `rules_cnn` wrapper | M | 5 |
| MoE and GCN/SAGE | hand op chains, hand backward door | tape rules; one sparse aggregate op | X9 | `moe_backward_run_impl`, the GCN backward door | M | 6 |
| Loops with the `while_loop` shape (GMM, BGMM, CD, QN + GLM/Huber/LogCV line searches, SGD as `Program.fit`, MDS, label propagation, NMF, MiniBatch generic, Prophet FAST, AdaBoost) | one scalar sync per iteration or trial on the host (section 4) | body programs with the same kernels, criterion as a program value, unrolled chunk | N1-N6 (one lane per family group) | the host driver loops and their per-iteration downloads | 6 x S-M | 6 (after X1) |
| Cholesky / eigh / sort / compaction as executor ops; then GP, GPC, KRR, SMO, FA, ICA, rotation, IPCA, KPCA, Isomap, Spectral Lanczos + discretize, LLE, dictionary learning, ARIMA lock-step | host round trips of w, V per iteration; potrf `info` drains; SMO count drain | device ops with one algorithm on every vendor and the host column | N7 (ops), N8-N10 (families) | the host eigh/sort/reverse code paths and re-uploads | M + 3 x M | 7 (bits change between versions, all columns together) |
| Tree builders (GBDT rounds, RF/ET cycles), estimator-in-program imputer | 4-5 host gates per round | not an executor fit this year | none | | | later |
| Chunk-polled and sync-free loops (KMeans DC, AP, mean shift, DBSCAN, HDBSCAN MST, UMAP, t-SNE, HW, Prophet, IsolationForest, IterativeImputer built-in, GBDT levels) | already the target shape | unchanged | none | | | never for speed |
| The name `autograd` | package of the executor; public module already `mojolearn.ops` | "executor" for the runtime, "tape" for the gradient feature | Andrew decides; one mechanical lane or docs only | | S | after the X lanes |
| Dangling `TAPE_EVERYWHERE_PLAN.md` citations, stale `llama_decoder_layer_backward_device` comments | five module headers, two comments | point at this plan and the decode plan | K0 | | XS | 1 |
