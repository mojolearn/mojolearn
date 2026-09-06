# Public FP32 Transformer training and cross-vendor resume

Source audit: 2026-09-06. This is an implementation plan, not a qualification record. No tests, builds, model execution, measurements, or provisioning were performed for this audit. Execution belongs exclusively to the root agent on remote NVIDIA/AMD; no Apple testing. Every implementation subagent must receive the explicit instruction **NEVER RUN TESTS, MEASUREMENTS, BUILDS, MODEL/GPU CODE, OR PROVISIONING**.

## What exists

| Surface | Exact source and entry points | Current boundary |
|---|---|---|
| Python Transformer | `python/mojolearn/_transformer_impl.py::TransformerBlock`, re-exported by `python/mojolearn/transformer.py` | Nine caller-supplied weights; forward, single-token step, explicit KV state. No public backward or training method. Shapes are configurable within its profile. |
| Transformer binding | `bindings/_mojolearn_transformer.mojo::_transformer_run`, `transformer_forward_binding`, `transformer_decode_step_binding` | Synchronous host-buffer fold, separate mode/vendor readback. No backward binding. |
| Native Transformer backward | `transformer/checks/transformer_backward.mojo::LlamaBackwardStages`, `llama_decoder_layer_backward` | Requires the exact forward stages, original block input, RoPE buffers, weights, and incoming host `List[Float32]` cotangent. This is implemented source; stale comments saying “never run” are not current evidence. |
| Public optimizer/loss | `python/mojolearn/_training_impl.py`, `bindings/_mojolearn_training.mojo`, `training/estimator.mojo` | Optimizer steps and cross-entropy are public primitives, not a Transformer training step. `_Optimizer.state_dict()` omits model weights/configuration/registry. `load_state_dict()` is not a complete training-checkpoint validator. |
| Complete internal step | `training/checks/train_loop.mojo::TrainBuffers`, `TrainConfig`, `train_step`, `unpack_params`, `pack_grads` | Embedding → one Llama block → untied LM head → causal CE → all backward passes → AdamW. Frozen toy profile, explicit twelve-stage schedule. |
| Checkpoint codec | `training/checkpoint.mojo::Checkpoint`, `save_checkpoint`, `load_checkpoint`, `compare_checkpoint_files` | Portable host-only versioned binary format for flat weights/moments/flags and descriptor. `load_checkpoint` checks expected names/offsets, but caller still must check intended configuration. |
| Resume driver | `training/checks/checkpoint_check.mojo::checkpoint_of`, `restore_into`, `train_head`, `train_tail`, `clause_iii` | File save/reload and deliberate bad resumes exist within one device process. Main accepts output-file and two-file comparison modes, but no external-file continuation mode. |
| Existing witnesses | `training/checks/train_step_check.mojo`, `training/checks/checkpoint_check.mojo`, `transformer/checks/transformer_backward_check.mojo`, `embedding/checks/embedding_identical.mojo` | Separate numerical, state-transfer, and negative controls must remain distinct. Files under `bench/results/checkpoint_2026-09-03/` exist for three devices; file presence alone does not establish current source/binary provenance or correctness. |

The internal profile has vocabulary 64, width 32, four query heads, two KV heads, head width 8, intermediate width 64, batch 2, sequence length 8, RoPE table length 64, RMS epsilon 1e-6, and RoPE theta 10000. It is not the public TransformerBlock's entire configurable shape space.

## Smallest safe public API

First expose exactly the existing fixed-shape, single-block, zero-cache, FP32 IDENTICAL training composition. Do not present it as general Llama training or silently add training to an inference `TransformerBlock` instance.

Proposed additions to `python/mojolearn/training.py` and `_training_impl.py`:

```python
state = TransformerTrainingState.from_weights(
    weights, numeric_mode="identical", lr=1e-3, weight_decay=0.01,
    max_norm=0.0,
)
result = transformer_train_step(state, token_ids)  # int32, shape (2, 9)
# result.loss is np.float32; result.completed_steps == state.completed_steps
state.save_checkpoint(path, run_descriptor=descriptor)
restored = TransformerTrainingState.load_checkpoint(path, expected_run=descriptor)
```

The API takes actual caller weights and token IDs. Deterministic fixture initialization and generated batches remain harness features. `token_ids[:, :8]` are inputs and `token_ids[:, 1:9]` are next-token targets, matching `batch_inputs` / `batch_targets` exactly. Reject IDs outside `[0, 64)`, wrong shape/dtype, ignored labels, and inconsistent configuration before device work. No hidden token source, RNG, automatic optimizer call through the existing Python optimizer, or duplicate numerical implementation.

Start with the AdamW configuration already built by `TrainConfig.optimizer`: fixed betas `(0.9, 0.999)`, epsilon `1e-8`, caller-specified finite FP32 learning rate/weight decay/max norm with the existing optimizer's domain constraints. Pass `ARM_NONE` only. Do not expose sabotage arms, SGD, Adam, parameter groups, microbatching, loss scaling, reduced precision, stochastic layers, or carried KV training until separately implemented and qualified. FAST remains explicitly unsupported by this initial public step; fast inference is a different supported operation.

## Ownership and state contract

The flat optimizer buffer is authoritative. Do not checkpoint `LlamaDeviceWeights`: it is refreshed from the flat buffer at the beginning of each step and is stale immediately after an update.

`TransformerTrainingState` owns writable C-contiguous float32 arrays `param`, `exp_avg`, and `exp_avg_sq`, an int32 vector of eleven 0/1 momentum-initialization flags, an immutable parameter registry, immutable profile/configuration, and an integer count of completed steps. A successful call supplies `completed_steps + 1` as native optimizer `t`, then updates all three arrays, flags, and step count together. Save before the first update uses `t=0`; resume uses the saved count without restarting Adam bias correction.

Constructor inputs are copied into canonical storage once. Exposed named weight views alias that storage deliberately; independent snapshots are explicit copies. Refuse writable aliases between parameter/moment/flag arrays and call inputs. Serialize access to an individual state object; no concurrent steps or mutation during an in-flight call. No hidden persistent GPU cache in the first implementation.

Freeze the existing `param_id_name` order, not Python dictionary order or TransformerBlock's weight order:

| ID | Checkpoint name | Public weight key | Shape |
|---:|---|---|---|
| 0 | embed | embed.weight | (64, 32) |
| 1 | norm1_w | input_layernorm.weight | (32,) |
| 2 | w_q | q_proj.weight | (32, 32) |
| 3 | w_k | k_proj.weight | (16, 32) |
| 4 | w_v | v_proj.weight | (16, 32) |
| 5 | w_o | o_proj.weight | (32, 32) |
| 6 | norm2_w | post_attention_layernorm.weight | (32,) |
| 7 | w_gate | gate_proj.weight | (64, 32) |
| 8 | w_up | up_proj.weight | (64, 32) |
| 9 | w_down | down_proj.weight | (32, 64) |
| 10 | lm_head | lm_head.weight | (64, 32) |

Use `train_offsets()` and `param_id_count()` as the native authority, with independently declared Python names/shapes checked at the boundary. There are 13,376 FP32 elements in each authoritative flat array. No tied embedding/head alias is part of this profile.

A step allocates fresh native device owners, uploads caller state, calls `unpack_params`, then the existing `train_step`. Keep all owners alive through synchronization and complete host downloads using explicit keep-alives. Stage outputs into fresh host arrays first, then publish to the Python state after the complete successful native return. Device errors must leave the caller's previous checkpointable state intact. The first wrapper may retain the existing small host round trip for the block-output cotangent; removing it is a later numerical-preservation change.

The gradient buffer, activations, loss scratch, and KV cache are transient. Cache starts at `pos0=0` every training step. Momentum flags are serialized even though they are numerically inert for the initial AdamW API.

## Checkpoint and data continuation

Reuse the existing checkpoint format rather than pickle or a parallel NumPy codec. Preserve FP32 payload bits and descriptor float bits, the canonical names/offsets, all moments/flags, and `t`. Reject truncated/corrupted files, layout differences, nonfinite payload/config values, unsupported mode/profile, and changed optimizer configuration before uploading. Existing codec validity is necessary but does not replace the public API's intended-run checks.

Existing v1's `(seed, t)` is sufficient only for `train_batch_ids(seed, step_index)`. Caller-provided data cannot honestly inherit that guarantee. Define a required, versioned run manifest alongside the unchanged native v1 file, containing model/profile identifier, numeric mode, canonical registry, optimizer descriptor, completed-step count, data-schedule identifier, dataset/token-byte digest, batch schedule/next batch index, and the checkpoint SHA-256. Exact resume requires the same remaining token schedule. The harness uses the existing stateless generated schedule; external users supply their data descriptor and resume cursor.

Write payload and manifest to unique temporary paths; flush/close, validate, then publish the manifest last as the commit point. Refuse an incomplete/mismatched pair. A saved state can be used to start a different data run only through an explicit fork operation, with no exact-resume claim. Do not add wall time, hostname, device identity, or paths to canonical state bytes. Store provenance separately from the files whose byte equality is being compared.

## Implementation sequence and ownership

1. **Shared host adapter, native implementation owner.** Add `training/train_step_host.mojo`. Reuse `TrainBuffers`, `train_dims`, `unpack_params`, `train_step`, and existing device-weight construction. Extract reusable upload/download/checkpoint assembly helpers currently in `training/checks/checkpoint_check.mojo` into this adapter or a small `training/train_state.mojo`; update the gate to call the shared helper rather than copy arithmetic. Do not move the numerical kernels from `checks/` during this work.
2. **Native ABI, same owner.** Extend `bindings/_mojolearn_training.mojo` with a folded buffer-address/scalar train-step entry and checked checkpoint read/write adapters. Explicit mode/vendor readback; IDENTICAL-only refusal precedes allocations. Guard imported Transformer backward, embedding, optimizer, loss, and GEMM sabotage flags at binding initialization. The existing training extension now pulls in the Transformer composition, so record the larger build dependency set; do not compile concurrently with another extension.
3. **Python API, separate source-only owner.** Extend `python/mojolearn/_training_impl.py`, `python/mojolearn/training.py`, and appropriate package exports. Implement strict shape/dtype/config/state validation, fixed registry, per-state call lock, transactional publication, and checkpoint/run-manifest policy. Keep the existing inference class untouched. Add authored tests in `python/mojolearn/tests/test_transformer_training_surface.py`; host mocks cover registry/order/ownership/refusals, while root-only remote cases cover numerical execution.
4. **Independent numerical gate, root executes only.** For one supplied-weight/input step, compare loss and all eleven gradients with an independent FP64 autograd transcription on the selected remote GPU. Separately compare public-step bytes against native-step state/loss bytes; no tolerance substitutes for that check. Preserve Transformer-backward and embedding gradient checks so agreeing optimizers cannot hide an incorrect gradient.
5. **Resume harness, source-only tooling owner.** Add `training/checks/public_train_resume.mojo` (or extend the existing gate without changing its default behavior) and `tools/training_cross_vendor_resume.py`. Separate modes must support starting at zero, writing at a requested completed step, loading an externally supplied checkpoint, continuing a specified remaining data interval, and comparing final canonical files. Reuse checked restore and step functions. No provisioning embedded in the comparator.
6. **Integration and qualification, root only.** Snapshot source, build scripts/flags, binary hashes, mode/vendor reports, runtime versions, GPU model, fixture and schedule hashes. Run sequential bounded jobs remotely, collect complete artifacts, classify failed/skipped/unrun gates explicitly, then update public documentation/support claims to the exact evidenced scope.

Subagent work can overlap only on disjoint source files; binding/adapter changes have one owner. Every test, build, benchmark, GPU call, and cloud operation stays in the main/root thread. No Apple execution is part of this plan.

The bounded companion-driver implementation and guarded root commands are documented in
[PUBLIC_TRAINING_RESUME_COMMANDS.md](PUBLIC_TRAINING_RESUME_COMMANDS.md).
It does not yet expose the proposed public training API.

## NVIDIA ↔ AMD resume gate

Use the same frozen profile, IDENTICAL binary mode, configuration, seed, source snapshot, and ordered batches on both devices. The minimum meaningful split is **8 completed steps followed by 8 resumed steps**, comparing at step 16; this exercises the Adam counter/bias-correction seam beyond step six and the optimizer contract's stated minimum split.

Root performs one GPU job at a time:

1. NVIDIA uninterrupted steps 1–16; retain losses and complete final canonical checkpoint.
2. AMD uninterrupted steps 1–16; compare final parameter/moment/flag bytes and canonical checkpoint bytes to NVIDIA.
3. NVIDIA steps 1–8, save, terminate the process; transfer the validated checkpoint/manifest to AMD; new process resumes 9–16. Compare every retained step loss/state digest and final bytes to the uninterrupted references.
4. AMD steps 1–8, save, terminate; NVIDIA resumes 9–16 in a new process and makes the same comparisons.
5. Negative controls independently discard moments, restart `t`, change the next batch, change optimizer configuration, permute registry entries, corrupt payload bytes, and truncate the file. Structural/configuration corruptions must refuse; numerically wrong but well-formed resumes must demonstrably diverge. Do not count AdamW momentum-flag mutations as an effective numerical control: that flag is inert here.

For each equality assertion retain actual arrays/files and the first differing named tensor/element, not just aggregate hashes. Report independently: correct gradient evidence, public/native agreement, same-device restart, cross-vendor state identity, and opposite-vendor continuation. These are distinct claims.

Set explicit host/GPU memory ceilings, process deadlines, and single-worker/thread limits in the root controller. Build and execute serially; avoid broad suite fan-out. Derive temporary-space requirements from the fixed shapes before launching. Capture remote failure artifacts before teardown. A terminated or incomplete run never becomes a pass. Performance work is deferred until these correctness gates pass and would be a separate root-only remote campaign.

## Completion criteria

The first release slice is complete when callers can perform the fixed-profile supplied-data step, save all authoritative state, reload it in another process, and continue on NVIDIA or AMD with the promised IDENTICAL bits; numerical gradients and the public/native adapter have independent evidence. It does not close configurable model dimensions, full pretrained backbones, FAST training, reduced precision, distributed training, arbitrary microbatch schedules, or training through inference caches.
