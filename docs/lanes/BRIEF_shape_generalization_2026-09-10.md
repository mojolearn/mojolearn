# Lane brief: generalize the trainer shape and add a resident-state API

Date 2026-09-10. Owner: the shape-generalization lane. Root (the orchestrator)
runs every build, test, and GPU job. This brief is self-contained; read it
before touching any file it names.

## Goal

Turn the fixed two-block byte language model trainer into a config-driven
trainer that can build a 125M-parameter, 12-layer, 50257-vocabulary model,
without changing one arithmetic operation. The end use is a GPT-3 Small scale
training run on rented NVIDIA and AMD GPUs whose every step is bitwise
identical across vendors. The Mac verifies a few hundred steps of that run;
it never trains the model.

## What is already runtime and must not be re-done

`training/byte_lm_config.mojo::ByteConfig` already carries batch, length,
d_model, n_heads, n_kv, head_dim and intermediate as runtime fields. The
Llama block (`transformer/impl/transformers/models/llama/modeling_llama.mojo`)
accepts those dimensions, supports sliding window, and dispatches to the
fused three-pass attention on main (same bits as eager; eager stays the
oracle). The Python optimizer in `python/mojolearn/_training_impl.py`
already has a learning-rate schedule hook, `accumulation_steps`,
`step_accumulated`, and `clip_grad_norm_`. Do not rebuild any of that.

## What is fixed and is this lane's job

Three things in `training/byte_lm.mojo` and its callers are compile-time
constants that pin the trainer to one model:

1. Layer count. Two blocks are spelled as separate fields (`forward0`,
   `forward1`, `backward0`, `backward1`) with hand-written forward and
   backward composition. There is no loop over layers.
2. Vocabulary. `BYTE_V = 256` sizes the embedding table, the LM head, and
   the cross-entropy configuration.
3. The parameter registry. `BYTE_J = 20`, `BYTE_N_TOTAL = 34944`, the
   positional names in `byte_param_name`, the offsets, the momentum-flag
   list length, and the profile string all assume exactly two blocks and a
   byte alphabet. The checkpoint descriptor and the Python binding
   (`bindings/_mojolearn_byte_lm.mojo`, `python/mojolearn/_training_impl.py`)
   inherit those assumptions.

Deliverable 1 removes all three across Mojo, the Python surface, and the
checkpoint metadata:

- `ByteConfig` gains `n_layers` and `vocab` (rename the struct and profile
  prefix if "byte" is now wrong; keep the old default values so the old
  profile is one configuration of the new code).
- Forward is a loop: embedding, then layer i for i in 0..n_layers, then head,
  then loss. Backward is the reverse loop. Stages and caches are lists sized
  from the config.
- The registry is generated: index 0 embedding, then nine weights per layer
  in the existing per-block order, then the LM head. `2 + 9 * n_layers`
  tensors. Names, offsets, and parameter count derive from the config.
- The checkpoint descriptor records the full config so a file says which
  model it belongs to. `load_checkpoint` refuses a config mismatch by name.
- The Python binding and the Python training class accept the new config
  fields and size every array from the registry.

## The second finding: resident state

The lane reported that the Python training API rebuilds native state on
every call and downloads full arrays each step. That is correct and it is
by design for the fixture path: a step is a correctness capture, not a
throughput path. It cannot carry a long run.

Deliverable 2, after deliverable 1 is gated, is a resident-state trainer:

- Parameters, moments, flags, and optimizer scratch live on the device for
  the lifetime of the trainer object. Nothing row-scaled or parameter-scaled
  crosses the bus on a normal step.
- A normal step returns the scalar loss and nothing else. Full-state
  capture becomes an explicit separate method, used by the gates and by
  checkpoint writes.
- Stage recording is off by default. The recording path stays, selected
  explicitly, because every existing gate reads it.
- One host synchronize per step at most. Today `byte_train_step` has about
  fifteen. Token batches are uploaded ahead of the step they feed, from a
  buffer the caller fills while the previous step runs.
- Weights are uploaded once at construction, never per call. The public
  `TransformerBlock` re-uploads nine weights every call; the trainer must
  not inherit that pattern.

Deliverable 2 is host and control-plane work only. It changes when data
moves, never what arithmetic runs.

## The one hard constraint

No arithmetic changes. Every GEMM, softmax fold, loss reduction, and AdamW
update is called with the same operands in the same order with the same
rounding as today. More calls on bigger buffers is fine. A different fold
order, a fused epilogue, a different reduction tree, or a "cheaper
equivalent" is not, even if it looks bit-identical by inspection. Speed
work on kernels belongs to the GEMM and attention lanes and is out of scope
here.

If a change to arithmetic looks unavoidable, stop and write it up as a
question for root instead of making it.

## Acceptance gate

The generalized trainer, configured to the old profile (batch 2, length 32,
d_model 32, 4 heads, 2 KV heads, head_dim 8, intermediate 64, vocabulary
256, 2 layers), must reproduce the retained three-vendor 128-step record
byte for byte: every retained raw state, every held-out loss, and the final
checkpoint bytes in
`bench/results/resume/2026-09-07-root-byte-lm-three-vendor/`. Held-out loss
5.5413 to 2.8436 on the pinned Tiny Shakespeare schedule.

If the bytes match, no arithmetic moved and the cross-vendor identity proof
carries over to the new code. If they differ, the lane changed arithmetic
somewhere and that is the bug, regardless of how small the difference is.

A second gate for deliverable 1: a config with 3 layers and a vocabulary of
512 at the same small width must build, run one step on the device, and
produce gradients that agree with `tools/byte_lm_gradient_oracle.py` (the
independent FP64 first-step oracle) the way the existing profile does.

Deliverable 2's gate is the same 128-step record, produced through the
resident path with capture taken only at step 128.

## Rules of the road

- The lane never builds, tests, runs models, measures, or provisions. Write
  every needed run as `RUN OWED` with the exact command, working directory,
  environment variables, and the file the result should land in. Root runs
  one job at a time. The Mac runs only host-side checks under the tiny-job
  guard; GPU legs go to rented NVIDIA and AMD.
- No FAST tier, no bf16, no TF32, no tensor cores. FP32 IDENTICAL only. This
  is not a speed lane; do not report timings.
- Every commit names its parent (`%h parent %p`). Explicit paths only,
  never `git add -A`. Fix stale docs and comments you find in the same
  commit, across the whole tree, not only in the files you were sent to.
- Do not touch `gemm/`, `transformer/impl/`, the loss and optimizer kernels
  under `training/checks/`, or `numerics/`. Read them; do not edit them.
- Keep the old profile string reproducible from the new code so old
  checkpoints load. Old records must not be rebound to new source; the
  acceptance gate re-derives them.

## Files

Read first: `training/BYTE_LM_IMPLEMENTATION.md`,
`training/PUBLIC_TRAINING_IMPLEMENTATION_PLAN.md`,
`training/CHECKPOINT_FORMAT.md`, `docs/LM_TRAINING_CLAIM_PLAN.md`,
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` section 11.

Edit: `training/byte_lm.mojo`, `training/byte_lm_config.mojo`,
`training/checkpoint.mojo`, `training/checks/byte_lm_config_check.mojo`,
`bindings/_mojolearn_byte_lm.mojo`, `python/mojolearn/_training_impl.py`
and its public re-export, `bindings/build_byte_lm.sh` if the build inventory
changes, and the docs above.

## Report

One handoff file, `docs/lanes/HANDOFF_shape_generalization_<date>.md`, with:
the commit list with parents; the exact config fields added; the registry
rule; every RUN OWED command in the order root should run them, the
acceptance gate first; and a list of everything the lane found stale and
fixed. Numbers only where a reader must act on them.
