# What model integration means in this branch

A standalone kernel cannot be selected by an ordinary training or inference
call. A wired source arm has a native caller, an A/B switch, admitted inputs,
owned scratch/state, consumed outputs, and a matching host arithmetic path when
it changes the numerical graph. Training changes also need backward and the
existing transaction or checkpoint rules. Source integration does not establish
that a build succeeds or that identity, quality, or performance pass.

The completion pass connects the selected experiments through these callers:

- Neural GEMM dispatch and backward surfaces feed transformer, Mamba, CNN,
  byte-LM and MLP native callers; the small MLP's Python API calls the neural
  binding for IDENTICAL matrix operations. Classical GEMM callers retain their
  own dispatch. Leaf/chain changes use a common host/device graph.
- Transformer attention/norm profiles reach host/device forward and backward.
  Explicit `forward_with_tape` / `backward_from_tape` native session APIs own
  retained/recomputed stages, input/weight snapshots and single-use generations.
- Mamba and CNN integration details, including prefix-state versions and the
  bounded im2col owner, are in the state/CNN handoff. Generated repository host
  source is ordinary source; no compiler IR or toolchain is patched.
- Byte-LM owns retained embedding runs and admits reuse by device ID comparison.
  Its CE caller can reuse same-operation token admission. Its grouped AdamW
  experiment keeps old state for rollback and consumes the block-status result.
- Loss totals and clipping reductions have matching pure host arithmetic and
  device callers with sized, caller-owned workspace. Profile-changing flags
  contribute to byte-LM/MLP checkpoint admission and the actual native binding's arithmetic identity.
- Samba accumulation keeps canonical microbatch order, consumes live integer
  status, and can use logical-leaf reassembly inside the existing supported
  multi-GPU ownership/transport scheme.
- Byte-LM CE/optimizer scratch has a concrete arena owner with disjoint declared
  phases on one ordered device context. The byte-LM inference scratch owner has
  a real redundant-clear elimination arm.
- Public `residual_dropout` / `residual_dropout_backward` implement an explicit
  complete layer with fused/materialized controls, a pure shared Philox
  contract, native admission and both input/residual gradients. Existing
  transformer dropout refusal stays in force.
- `mlp_inference_sessions` and `SmallMLPTrainer.predict_logits_sessions` execute
  an entire shared-weight Linear/ReLU/Linear model. A combines independent
  sessions' projections; B runs separate projections. The native boundary
  includes admission, copies, activation, completion and consumed logits.

Each numbered idea selects a concrete experiment arm, described in its lane
record. A broad idea can also name further research variants. For example,
NN52's implemented arm fuses CE weights and logits gradients; separate forward
fusion variants are additional work. NN56's complete-model caller is byte-LM
AdamW; arbitrary SGD group APIs are additional work. NN64 implements stateless
MLP session batching; stateful decoder cache/RNG batching remains a separate
unimplemented direction. These extensions are not claimed as programmed.

Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and its per-card
handoffs for the final source status, including any incomplete selected arm.
None of this branch's new source was compiled, identity-checked, quality-tested
or timed. No new experiment default was promoted. Prior failures and rejected
candidates remain recorded; they do not become wins because new callers exist.

The complete source-authoring output for the Mamba host refresh is retained at
[source_authoring/mamba_host_20261006.log](source_authoring/mamba_host_20261006.log).
It is an authoring log, not build, verification, identity or timing evidence.
