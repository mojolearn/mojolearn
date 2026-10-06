# Neural training lane source handoff

NN49–NN64 selected source arms; all new flags require IDENTICAL and respect ALL_OFF. Existing candidates remain identified as existing. No build, static verification, test, candidate execution or timing was run; no defaults promoted.

Branch `ideas/neural-identical-ab-20261006-r3`, base main `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.

## NN49 — wired_draft

Four independent adjacent feature chains share each canonical run lookup in the actual touched-row backward route. Existing radix/touched infrastructure is reused; no new identity/quality claim.

Sources: `embedding/checks/embedding_identical.mojo`.

A defines: `MOJOLEARN_NN49_EMB_VECTOR4=1`.

B defines: incumbent, new flag absent.

- Route only runs when existing touched-row dispatch is active (n_positions < vocab); other shapes retain incumbent.
- Full LM/embedding callers and all-vendor qualification remain unrun; include zeroing/sort/readback in timing.

## NN50 — wired_draft

ByteTrainer owns immutable grouping snapshots across gradient calls. A device byte comparison admits reuse; changed IDs replace the owner and rebuild groups. The actual embedding backward consumes retained groups.

Sources: `embedding/owned_runs.mojo`, `embedding/checks/embedding_identical.mojo`, `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN50_OWNED_RUNS=1`.

B defines: incumbent, new flag absent.

- Cold construction and repeated same/different-batch qualification intentionally unrun.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN51 — wired_draft

Byte-LM embedding forward/backward and CE target admission use the same-operation device token validation. CE still refuses nonfinite logits first and the general entry retains full validation.

Sources: `training/byte_lm.mojo`, `training/checks/loss.mojo`.

A defines: `MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION=1`.

B defines: incumbent, new flag absent.

- Other model upload owners are optional extensions; selected byte-LM caller is wired.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN52 — wired_draft

Actual identical_ce_backward_into computes/stores weights and dLogits in one pass with L14/L16 seams and ignored rows preserved; sabotage builds use the original kernels.

Sources: `training/checks/loss.mojo`.

A defines: `MOJOLEARN_NN52_CE_WEIGHT_GRAD=1`.

B defines: incumbent, new flag absent.

- Selected complete arm is backward weights/dLogits fusion; forward shift-exp/target fusion is an additional research variant.
- Aliased/unaliased buffers, full loss semantics and complete training quality unverified.

## NN53 — wired_draft

Existing opt-in chunked head gains separate 512/2048 panel arms versus incumbent 1024. This is a storage/launch extension to the existing profile, not initial head implementation.

Sources: `training/chunked_lm_head_v2.mojo`, `training/byte_lm_config.mojo`.

A defines: `MOJOLEARN_NN53_HEAD_CHUNK512=1`.

B defines: incumbent, new flag absent.

- ByteConfig.chunked_lm_head_v2=True in BOTH arms; normal head does not reach these panels.
- Complete old-head-versus-v2 arithmetic/model qualification is not supplied by panel geometry.
- Host/head/loss/gradient and full-model quality remain unrun.

## NN54 — wired_draft

Fixed 128-value ascending leaves and adjacent-pair/odd-carry total now reach actual CE L12 on device, host oracle and optimized host row/byte-LM routes. All normalization, smoothing and ignored-row seams retain their existing meaning. Caller-owned CE workspace retains scratch through completion.

Sources: `training/neural_ab_profiles.mojo`, `training/neural_ab_profile_contract.mojo`, `training/checks/loss.mojo`, `training/checks/loss_oracle.mojo`, `training/loss_host_rows.mojo`, `training/byte_lm_host_kernels.mojo`.

A defines: `MOJOLEARN_NN54_LOSS_PROFILE=1`.

B defines: incumbent, new flag absent.

- Selected arm changes row-total order only; denominator and vocabulary folds remain their existing selected GEMM contract.
- Alternate leaf sizes are additional research arms, not implemented switches.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN55 — wired_draft

Actual ByteTrainer grouped OOP AdamW update optionally emits/folds block status, swaps the tentative state, then uses existing finish/refusal and rollback. NN56 is common in both NN55 arms; control uses normal output scans.

Sources: `training/neural_ab_optimizer.mojo`, `training/checks/optimizer.mojo`, `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`, `MOJOLEARN_NN55_BLOCK_STATUS=1`.

B defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

- No status or rollback execution was run; faults retain incumbent route.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN56 — wired_draft

Actual ByteTrainer tensor registry forms optimizer groups with canonical scalar tables. Device source supports different per-group configurations; this caller repeats its admitted AdamW configuration. Existing input refusal, old-state handles, counters and rollback remain owned by the trainer.

Sources: `training/neural_ab_optimizer.mojo`, `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

B defines: incumbent, new flag absent.

- The selected complete model is byte-LM AdamW; general SGD/group-API expansion remains an optional separate idea.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN57 — wired_draft

Actual serial/batched/multi-GPU clipping uses fixed 128-value rounded-square leaves for per-tensor and final norm reductions, with matching host graph. Tensor order, intermediate sqrt, coefficient, finite-scalar admission and gradient scaling stay explicit; workspace sizes include both scratch planes.

Sources: `training/neural_ab_profiles.mojo`, `training/neural_ab_profile_contract.mojo`, `training/checks/optimizer.mojo`, `training/checks/optimizer_oracle.mojo`, `training/clip_multi_gpu.mojo`.

A defines: `MOJOLEARN_NN57_NORM_PROFILE=1`.

B defines: incumbent, new flag absent.

- This selected v2 arm preserves the two-level tensor norm structure; a flattened norm is an additional arithmetic experiment.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN58 — wired_draft

Actual Samba accumulation keeps its original balanced FMA tree and reports integer first-nonfinite status while producing nodes. The caller consumes status before publishing output; host uses the same field/index refusal. NN63 combination falls back to final device scan after canonical merge.

Sources: `training/neural_ab_pointwise.mojo`, `training/samba_ops.mojo`, `training/host/samba_ops_oracle.mojo`.

A defines: `MOJOLEARN_NN58_ACCUMULATE_STATUS=1`.

B defines: incumbent, new flag absent.

- The new opt-in profile explicitly refuses computed nonfinite gradients; baseline input admission is retained. NN58+NN63 avoids pretending the independent fused status win composes for free.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN59 — wired_draft

Public residual_dropout and residual_dropout_backward execute a complete explicit residual/dropout layer with the shared Philox mapping, native shape/finite/probability/stream admission and paired input/residual gradients. NN59 selects fused A versus materialized B on GPU; host uses the same scalar contract. Both native training builders register the operations and Python only passes buffers/metadata.

Sources: `training/neural_ab_pointwise.mojo`, `training/residual_dropout.mojo`, `training/residual_dropout_contract.mojo`, `training/residual_dropout_host.mojo`, `bindings/residual_dropout_boundary.mojo`, `bindings/residual_dropout_boundary_host.mojo`, `python/mojolearn/_residual_dropout_impl.py`, `python/mojolearn/training.py`.

A defines: `MOJOLEARN_NN59_DROPOUT_RESIDUAL=1`.

B defines: incumbent, new flag absent.

- Existing transformer dropout refusal is preserved. This is an explicit callable layer; arbitrary CNN/transformer residual-model adoption is an additional experiment.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN60 — wired_draft

ByteTrainer actual block parameter/gradient and emb/head view arms rebind flat parameter views after handle swaps, using existing native ownership and transaction boundaries.

Sources: `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN60_BLOCK_VIEWS=1`.

B defines: incumbent, new flag absent.

- Other model/offload owners are separate extensions; no new default enabled.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN61 — reused_existing

Selected source arm reuses I10's existing actual byte-LM finish/status owner and paired native recipes. It is intentionally not claimed as new work or a newly qualified win.

Sources: `training/byte_lm.mojo`, `core/step_glue.mojo`, `experiments/performance_ideas/I10/native_arms.json`.

A defines: `MOJOLEARN_TRAIN_LIVE_STATUS=1`, `MOJOLEARN_STEP_GLUE_TRIAL=1`.

B defines: `MOJOLEARN_STEP_GLUE_TRIAL=1`.

- Broader all-model drain consolidation is an optional extension.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN62 — wired_draft

ByteBuffers owns an explicit live-interval arena for CE and optimizer workspaces. CE phase 0 and clipping phase 2 reuse a slab on one ordered context, retained across full forward/backward/update/rollback; replay does not retain either scratch region. All scratch producers initialize their consumed cells.

Sources: `training/neural_ab_lifetime.mojo`, `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN62_LIFETIME_ARENA=1`.

B defines: incumbent, new flag absent.

- Other activation/stage intervals are additional arena applications, not needed to select this concrete complete-model arm.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN63 — wired_draft

Actual Samba resident accumulation, including supported column-sharded multi-GPU callers, supplies a complete canonical leaf permutation by construction and invokes reassembly plus fixed-tree fold before publishing output. Existing supported transport and completion ownership remain in place.

Sources: `training/neural_ab_shards.mojo`, `training/samba_ops.mojo`, `training/accumulate_multi_gpu.mojo`.

A defines: `MOJOLEARN_NN63_CANONICAL_SHARD_MERGE=1`.

B defines: incumbent, new flag absent.

- No new collective/transport backend. Different partition/topology recipes must preserve logical leaves; no count-invariance evidence claimed.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

## NN64 — wired_draft

Public mlp_inference_sessions / SmallMLPTrainer.predict_logits_sessions runs a complete shared-weight Linear/ReLU/Linear model. A batches independent sessions through both projections; B dispatches each session separately. Native admission, upload, activation, output status and consumed concatenated logits are included. Matching host model is authored, and zero-row sessions are admitted.

Sources: `training/neural_ab_lifetime.mojo`, `training/neural_session_mlp.mojo`, `training/neural_session_mlp_host.mojo`, `bindings/_mojolearn_training.mojo`, `bindings/_mojolearn_training_host.mojo`, `python/mojolearn/_training_impl.py`, `python/mojolearn/_mlp_impl.py`.

A defines: `MOJOLEARN_NN64_SESSION_PACK=1`.

B defines: incumbent, new flag absent.

- Selected arm is stateless feedforward inference: no cache or RNG exists to merge. Stateful decode/cache batching remains a separate unimplemented research direction in the broad idea list.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

> Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
