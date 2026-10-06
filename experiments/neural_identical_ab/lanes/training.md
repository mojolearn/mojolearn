# Neural training lane source handoff

NN49–NN64 source drafts; all new flags require IDENTICAL and respect ALL_OFF. Existing candidates remain identified as existing. No build, static verification, test, candidate execution or timing was run; no defaults promoted.

Branch `ideas/neural-identical-ab-20261006-r3`, base main `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.

## NN49 — wired_draft

Four independent adjacent feature chains share each canonical run lookup in the actual touched-row backward route. Existing radix/touched infrastructure is reused; no new identity/quality claim.

Sources: `embedding/checks/embedding_identical.mojo`.

A defines: `MOJOLEARN_NN49_EMB_VECTOR4=1`.

B defines: incumbent, new flag absent.

- Route only runs when existing touched-row dispatch is active (n_positions < vocab); other shapes retain incumbent.
- Full LM/embedding callers and all-vendor qualification remain unrun; include zeroing/sort/readback in timing.

## NN50 — component_draft

OwnedEmbeddingRuns snapshots device IDs, validates them on GPU with a two-word error readback, sorts once, and calls the canonical backward fold with retained groups. No Python data work.

Sources: `embedding/owned_runs.mojo`, `embedding/checks/embedding_identical.mojo`.

A defines: `MOJOLEARN_NN50_OWNED_RUNS=1`.

B defines: incumbent, new flag absent.

- Public LM/microbatch owner integration pending.
- Owner must not expose/mutate its internal IDs/groups and must retain all buffers through completion.
- Construction/copy/sort costs must be included in cold timing.

## NN51 — wired_draft

Native byte-LM embedding forward/backward uses existing prerefused APIs after byte_require_tokens_device validates owned IDs/targets in the same forward/gradient operation.

Sources: `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION=1`.

B defines: incumbent, new flag absent.

- Loss-side redundant validation and reusable upload-buffer extension remain pending.
- Failure priority and full train-step identity/quality not verified.

## NN52 — wired_draft

Actual identical_ce_backward_into computes/stores weights and dLogits in one pass with L14/L16 seams and ignored rows preserved; sabotage builds use the original kernels.

Sources: `training/checks/loss.mojo`.

A defines: `MOJOLEARN_NN52_CE_WEIGHT_GRAD=1`.

B defines: incumbent, new flag absent.

- Forward shift-exp/target contribution fusion not implemented by this arm.
- Aliased/unaliased buffers, full loss semantics and complete training quality unverified.

## NN53 — wired_draft

Existing opt-in chunked head gains separate 512/2048 panel arms versus incumbent 1024. This is a storage/launch extension to the existing profile, not initial head implementation.

Sources: `training/chunked_lm_head_v2.mojo`, `training/byte_lm_config.mojo`.

A defines: `MOJOLEARN_NN53_HEAD_CHUNK512=1`.

B defines: incumbent, new flag absent.

- ByteConfig.chunked_lm_head_v2=True in BOTH arms; normal head does not reach these panels.
- Complete old-head-versus-v2 arithmetic/model qualification is not supplied by panel geometry.
- Host/head/loss/gradient and full-model quality remain unrun.

## NN54 — component_draft

nn_reduce_into[LEAF,False] and nn_reduce_host share fixed ascending leaves and adjacent-pair/odd-carry arithmetic. LEAF is a versioned template parameter, e.g. 64/128/256 in independent arms.

Sources: `training/neural_ab_profiles.mojo`.

A defines: `MOJOLEARN_NN54_LOSS_PROFILE=1`.

B defines: incumbent, new flag absent.

- Public loss normalization/weights/ignore-index and matching caller host migration pending; component is a raw admitted-value sum only.
- Full loss and multi-step task quality cannot be inferred from this component.

## NN55 — component_draft

Grouped OOP canonical Adam cells emit four per-block integer status minima and fold them. Separate existing I10 atomic-status path is retained as prior work.

Sources: `training/neural_ab_optimizer.mojo`, `training/checks/optimizer.mojo`.

A defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`, `MOJOLEARN_NN55_BLOCK_STATUS=1`.

B defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

- Public commit/rollback/status owner integration pending; updates remain tentative buffers.
- NN56 common in both arms to isolate NN55; control is STATUS=False plus incumbent post-scan of the same outputs.
- No removal of input checks or failure semantics authorized.

## NN56 — component_draft

One canonical step-scalar table per group and grouped grid.y OOP Adam/AdamW using the existing _adam_oop_cell. Source supports per-group config and step metadata.

Sources: `training/neural_ab_optimizer.mojo`.

A defines: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

B defines: incumbent, new flag absent.

- Owner must admit monotonic complete offsets, scalar/kind table and input finite values; then commit only after success.
- SGD/additional optimizer extension and public optimizer integration pending.
- Control is existing per-group updates with the same scalar tuples and outputs.

## NN57 — component_draft

nn_reduce_into[LEAF,True] and matching host helper produce a fixed-profile sum of explicitly rounded squares.

Sources: `training/neural_ab_profiles.mojo`.

A defines: `MOJOLEARN_NN57_NORM_PROFILE=1`.

B defines: incumbent, new flag absent.

- Tensor-order flattening, sqrt/clip coefficient/nonfinite admission and full native clipping integration pending.
- No independent shard clipping or microbatch regrouping; all profile consumers must migrate together.

## NN58 — component_draft

One prescribed accumulation-tree edge writes the original FTZ add and per-cell integer nonfinite records while values are live.

Sources: `training/neural_ab_pointwise.mojo`.

A defines: `MOJOLEARN_NN58_ACCUMULATE_STATUS=1`.

B defines: incumbent, new flag absent.

- Owner must preserve the actual existing microbatch/shared-parameter tree; this is not a serial running-sum replacement.
- Compact status reduction and public accumulation/commit integration remain pending.
- Extra integer output bytes may lose; no performance claim.

## NN59 — component_draft

Portable Philox dropout plus rounded residual-add forward, and mask-replayed paired backward; shared scalar functions are host callable.

Sources: `training/neural_ab_pointwise.mojo`.

A defines: `MOJOLEARN_NN59_DROPOUT_RESIDUAL=1`.

B defines: incumbent, new flag absent.

- Public model caller/config/refusal integration and complete host operation wrapper pending.
- Caller preserves exact stream/offset and original residual/activation order; no reordered RNG stream.

## NN60 — wired_draft

Independent block parameter/gradient views and non-Apple emb/head parameter-view arms use existing sub-buffer mechanisms. Block gradients write into current flat grad; parameter views rebind after swaps. Existing FAST behavior remains separate.

Sources: `training/byte_lm.mojo`.

A defines: `MOJOLEARN_NN60_BLOCK_VIEWS=1`.

B defines: incumbent, new flag absent.

- Compile/API support on each target and lifetimes/handle-swap/full-step quality unverified.
- Existing Apple emb/head views remain inherited ON; adding its flag on Apple is not a new reached arm.
- Pooled/offload owners and complete multi-model lifecycle coverage remain pending.

## NN61 — reused_existing

Reuse I10's actual finish/status update owner and its existing paired native recipes. No new complete one-drain public model was invented.

Sources: `training/byte_lm.mojo`, `core/step_glue.mojo`, `experiments/performance_ideas/I10/native_arms.json`.

A defines: `MOJOLEARN_TRAIN_LIVE_STATUS=1`, `MOJOLEARN_STEP_GLUE_TRIAL=1`.

B defines: `MOJOLEARN_STEP_GLUE_TRIAL=1`.

- Broader status/loss consolidation and full-step reach/transactional coverage remain pending.
- Retain I10 exact arm environment/settings and source matching; component receipts do not qualify the full LM/Samba/MLP.

## NN62 — component_draft

A bounded model-owned native arena assigns deterministic byte offsets to explicit tensor live intervals and returns guarded sub-buffer views.

Sources: `training/neural_ab_lifetime.mojo`.

A defines: `MOJOLEARN_NN62_LIFETIME_ARENA=1`.

B defines: incumbent, new flag absent.

- Model-wide liveness intervals including asynchronous consumers/backward/replay and error paths must be supplied and integrated.
- The component does not prove caller lifetime metadata or automatically omit clears; no public model currently opts in.

## NN63 — component_draft

Device logical-leaf reassembly plus fixed pair-tree merge and matching host routine after supported shard transport. Physical arrival order does not become arithmetic order.

Sources: `training/neural_ab_shards.mojo`.

A defines: `MOJOLEARN_NN63_CANONICAL_SHARD_MERGE=1`.

B defines: incumbent, new flag absent.

- Shard leaf production, complete unique permutation admission, transport/completion ownership and public distributed model integration pending.
- No new communication backend, unsupported collectives or toolchain patch.
- Source/binary profile matching to the single-device contract is required before claiming count/topology invariance.

## NN64 — component_draft

GPU pack/scatter over compatible ragged session rows using admitted prefix/base descriptors; exact copies with empty sessions supported.

Sources: `training/neural_ab_lifetime.mojo`.

A defines: `MOJOLEARN_NN64_SESSION_PACK=1`.

B defines: incumbent, new flag absent.

- Session/cache/RNG owner admission and batched model execution remain pending; this is movement, not a complete batched decode implementation.
- Disjoint scatter ranges/valid capacities and tail/latency quality must be established later.

> Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
