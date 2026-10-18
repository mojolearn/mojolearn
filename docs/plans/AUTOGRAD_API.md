# Autograd public surface: proposal

Scope: what `autograd/` exposes in Mojo today, the Python surface that keeps
the no-Python-in-the-runtime rule, the trainer migration order, and the A/B
that decides each migration. Implementation record: `NATIVE_AUTOGRAD.md`.

## (a) Mojo API as it stands

- `Tape` (`autograd/tape.mojo`): records FP32 matrix/scalar ops on one
  `Exec` (HostExec or DeviceExec) and runs first-order reverse mode.
  Leaves: `variable` (borrowed executor memory), `tensor` (host list copy),
  `scalar`. Ops: `matmul`, `add/subtract/multiply/divide` with row, column
  and scalar broadcast, `relu/sigmoid/tanh/exp/log/sqrt/scale`, `transpose`,
  `reshape`, `detach`, `sum/mean` over an axis or all, `softmax`,
  `log_softmax`, `mse_loss`, `cross_entropy`. Backward: `backward` (scalar),
  `backward_from` (explicit seed), `*_with_rules` variants for custom ops.
  `reset()` reuses storage across steps; stale handles are refused.
- `Tensor`: an opaque handle (owner, generation, index); shape via
  `tape.rows/cols`, storage via `tape.value/gradient`.
- `BackwardRules` trait + `tape.custom(inputs, output, rows, cols, tag,
  saved)`: an already-executed native op with a VJP. The tape hands the rule
  one zeroed destination per input and owns ordering and accumulation.
- Adapters: `autograd/layers.mojo` (`layer_norm`, `NeuralRules`) and
  `autograd/rules_recurrent.mojo` (`rnn_linear`, `rnn_layer`,
  `rnn_last_step`, `RecurrentRules`, `AutogradRnnStep`).
- Trainers on the tape, opt-in: MLP (`-D MOJOLEARN_AUTOGRAD_MLP`,
  `sequence/mlp_fit.mojo`), recurrent (`-D MOJOLEARN_AUTOGRAD_RNN`, the
  `RnnStep` seam in `sequence/recurrent.mojo::rnn_fit`).

Constraints that carry into any public surface:

- Every op is a parallel launch with a fixed fold order; identity across
  Apple, NVIDIA and AMD comes from the ops, not from the tape.
- Tags of custom rules are per family (LayerNorm 1, recurrent 16-18). A
  composite `NeuralRules` that dispatches every family is the target once
  the module graph allows it without an import cycle: today
  `autograd.rules_recurrent` imports `sequence.recurrent`, which imports no
  autograd module; the `RnnStep` seam lets `sequence/pyapi.mojo` pick the
  step, so `sequence.recurrent` never imports the rules.
- Metal: the combined `OP_AUTOGRAD` launch is blocked (NATIVE_AUTOGRAD.md,
  ask for Modular). The recurrent path issues no `OP_AUTOGRAD` launch (only
  OP_FILL/OP_COPY plus the existing kernels, and no tensor has two
  consumers, so no ACCUMULATE); the MLP path does (`tape.add`).

## (b) Python surface

Python describes a model; Mojo trains it. No per-op Python call happens
inside a training step, and Python computes no number.

1. `mojolearn.nn.Graph` (name provisional) builds a DESCRIPTION: an ordered
   list of node specs, each `(kind, inputs, attrs)`. Kinds map 1:1 to tape
   ops and adapters: `input`, `param(shape, init)`, `linear`, `rnn(cell,
   hidden, layers)`, `layer_norm`, `embedding`, activations, reductions,
   `loss(mse|cross_entropy)`. Python checks argument types and names only;
   shape inference, shape errors and initial values come from Mojo.
2. The description is encoded as two flat arrays (int32 codes/shapes,
   float32 attrs) plus the data buffer addresses, and passed to ONE binding:
   `graph_fit(addrs, ints, floats)`. In Mojo, the entry decodes the spec,
   allocates parameters on the device, and runs the whole loop: epoch
   order (`op_mlp_perm`), batch gather, `tape.reset()`, graph record,
   loss + seed, `backward_with_rules`, optimizer (`opt_step`), stopping.
   It returns trained parameters and per-step losses by address.
3. `graph_grad(addrs, ints, floats)` returns loss and flat gradients for
   one batch, for callers that keep their own optimizer loop. Its output is
   exactly what `step(gradients)` in `python/mojolearn/_training_impl.py`
   takes, so `step(gradients)` stays as the public optimizer contract; no
   `.grad` attributes and no `loss.backward()` in Python.
4. `graph_predict(addrs, ints, floats)` runs the recorded forward without a
   tape (or with `requires_grad=False` leaves).
5. Custom user ops are Mojo only: a user writes a `BackwardRules` struct and
   builds their own binding. Python never supplies a VJP.

Docstrings that say there is no autograd, to reword when (b) ships (not
edited in this lane):

- `python/mojolearn/_training_impl.py:80` and `:1379` (`step()` rationale).
- `python/mojolearn/_mlp_impl.py:6` ("not autograd").
- `python/mojolearn/embedding.py:11` ("There is no autograd here").

The public surface is listed in `docs-site/api_exclude.txt` until its
gradient checks pass on Apple, NVIDIA and AMD.

## (c) Migration order

What moves is model-level assembly: the hand-chained order of head GEMMs,
column sums, residual adds, per-layer deltas and the calls into the block
kernels. What stays hand-written becomes a custom rule.

1. Recurrent (`sequence/recurrent.mojo::backward`): done under
   `MOJOLEARN_AUTOGRAD_RNN`. Stay: `op_cell_fwd_h`/`op_cell_bwd_h`, the scan
   `OP_CELL_FWD_SCAN`/`OP_CELL_BWD_SCAN`, `wgrad_blocked`. The rules reuse
   every hand launch with its arguments, so the expected result is the same
   bits as the hand path.
2. Transformer (`transformer/checks/transformer_backward.mojo::
   llama_decoder_layer_backward_device` and its caller
   `bindings/_mojolearn_transformer.mojo::_transformer_saved_backward`, which
   walks `TransformerForwardTape`): migrate the decoder-stack glue (residual
   adds, norm -> attention -> MLP chain, embedding and LM head). Stay: fused
   attention forward/backward (`transformer/impl/llama/fused_attention.mojo`
   `fused_bwd_*`), RMSNorm row kernels, RoPE, SwiGLU fused kernels, each as
   one rule. Prerequisite: the tape records over a `DeviceExec` that shares
   the transformer's `DeviceContext`, and `TransformerForwardTape`'s saved
   activations become rule `saved` buffers.
3. Mamba (`mamba/impl/modeling/modeling_mamba_prefill_backward.mojo::
   mamba1_prefill_backward`, `mamba/impl/modules/mamba2_backward.mojo`):
   migrate the block assembly (norm, in/out projections, residual, block
   stack). Stay: the selective-scan and SSD backward kernels
   (`mamba/impl/ops/selective_scan_backward.mojo`,
   `mamba/impl/ops/mamba2_ssd_backward.mojo`), the conv tap kernels and the
   `mamba_bwd_*_into` gate/norm kernels, each as one rule.
4. Language-model trainers last: they compose 2 and 3 plus embedding and
   loss; nothing new stays hand-written.

Each migration keeps the trainer's loss scaling, shuffle, batching,
stopping and optimizer unchanged; only the backward assembly is swapped
behind a seam like `RnnStep`.

## (d) A/B per migration

Each define is default off; arm A is the hand path, arm B the tape path,
one run per arm, same frozen commit.

| Define | Recipes | Identity | Speed | Quality |
|---|---|---|---|---|
| `MOJOLEARN_AUTOGRAD_MLP` | MLP classifier/regressor board rows | nv/amd ID check, arm B digest vs arm A | NV + AMD full-size race | board metric unchanged |
| `MOJOLEARN_AUTOGRAD_RNN` | rnn/lstm/gru clf + reg rows, L = 1 and 2 | ID check: arm B digest equals arm A (same launches) on nv and amd, and nv = amd | NV + AMD full-size race; the tape adds fills/copies per node | unchanged by construction; recorded |
| `MOJOLEARN_AUTOGRAD_TRANSFORMER` | transformer LM rows | nv = amd = Apple digest; arm A equality expected only where launches are reused verbatim | NV + AMD | loss / perplexity no material drop |
| `MOJOLEARN_AUTOGRAD_MAMBA` | Mamba-1/2 LM rows | as transformer | NV + AMD | as transformer |

Verdict: IDENTICAL switches by the average NV/AMD ratio against the noise
floor; a winner becomes the default with an `_OFF` define, a loser is
deleted with the tombstone procedure. A digest change between arms is a
bit version only when the define intentionally changes a fold; for the
RNN define it is a bug. Rows go into `docs/apple-fast/EXPERIMENTS.md` when
the A/B runs.
