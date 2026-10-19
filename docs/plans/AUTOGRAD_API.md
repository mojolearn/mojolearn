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
  autograd module but `autograd.losses` (which imports only the tape, the
  fold and the loss bodies); the `RnnStep` seam lets `sequence/pyapi.mojo`
  pick the step, so `sequence.recurrent` never imports the rules.
- Metal, NVIDIA and AMD: the 97-case gradient suite gives the same digests
  on all three and on the host (NATIVE_AUTOGRAD.md, "GPU status").
- `python/mojolearn/_training_impl.py:80` and `:1379` (`step()` rationale).
- `python/mojolearn/_mlp_impl.py:6` ("not autograd").
- `python/mojolearn/embedding.py:11` ("There is no autograd here").

The public surface is listed in `docs-site/api_exclude.txt` until its
gradient checks pass on Apple, NVIDIA and AMD.

## Losses and metrics

One set, used by the tape for training losses and by the trainers for the
losses they report, so the numbers users compare across machines are
identical by construction.

- `autograd/losses.mojo` (tape): `mse`, `mae`, `huber(delta)`,
  `binary_cross_entropy_with_logits`, `cross_entropy` (logits and integer
  class codes, held as float values in a (rows, 1) tensor), each with
  `REDUCTION_MEAN` or `REDUCTION_SUM`, and `cross_entropy_dense` (the tape's
  dense-target `cross_entropy`). Each is a loss node of the tape
  (`Tape.loss_terms`, `Tape.cross_entropy_rows`: per-element or per-row
  terms with their VJPs) followed by `Tape.mean` or `Tape.sum`. `mse`
  equals `Tape.mse_loss` bit for bit for same-shape inputs.
- `loss_value_into` (trainers, no tape): the same term bodies and the same
  fold over executor buffers into one word, plus the forward-only
  clipped-probability kinds `LS_LOG_PROB` / `LS_BINARY_LOG_PROB` of the MLP
  trainer. Its value equals the tape loss of the same kind and reduction bit
  for bit. Scratch: `loss_scratch(ex, max_terms)`.
- `autograd/metrics.mojo` (forward only, one download of one value each):
  `accuracy`, `accuracy_from_scores` (first maximum), `log_loss` (clipped to
  [2^-23, 1 - 2^-23], k-class or the positive column), `mean_squared_error`,
  `root_mean_squared_error`, `mean_absolute_error`, `r2_score` (one output),
  `roc_auc` (ties count one half). Rows are limited to 2^24 - 1 because
  codes, counts and indices travel as float values.
- Bodies: `autograd/loss_ops.mojo`, opcodes 1200..1211, a second range in
  `sequence/dispatch.mojo` (asserted disjoint from the autograd ops and the
  recurrent scan codes).

Fold convention. Every sum is the tape's blocked fold
(`autograd/fold.mojo::blocked_fold`, the launches `Tape` itself uses):
blocks of `ag_fold_block(extent)` consecutive terms (the smallest power of
two R >= 64 with R * R >= extent) added ascending from +0.0, then the block
partials added ascending from +0.0 times the factor. The mean's factor is
`identical_div(1, count)`. Per-row terms (cross-entropy, log loss, argmax)
are one thread per row over the classes in ascending order. The AUC sorts
(score, label) with a bitonic network on the executor (padding to a power
of two sorts last), counts positives before each row with a blocked scan,
finds each positive row's tie group by binary search, and folds its tie-aware
pair count over twice the negatives; the integer counts are exact, the final
sum is the fixed fold. The metrics in `metrics/impl/` (DeviceContext kernels
with shared-memory trees) are not used here because they do not run on
HostExec; the terms are theirs, the fold order is the tape's.

Trainers under `-D MOJOLEARN_AUTOGRAD_LOSS` (default off): `sequence/
mlp_fit.mojo` takes each batch's data term from `loss_value_into` (sum of
per-element terms) and keeps op_mlp_bloss for the L2 term and the divide;
`sequence/recurrent.mojo::rnn_fit` takes `losses[k]` from `loss_value_into`
(mean) instead of the one-thread OP_SUM. Gradient seeds are unchanged in
both. New bits in the reported losses only; n_iter of the MLP can move with
them through the stopping rule.

Checks: `autograd/checks/loss_cpu.mojo` and `loss_gpu.mojo`
(`run_loss_checks`): float64 oracles for every loss value and gradient,
bit equality of `loss_value_into` and the tape, and the metrics on small
arrays with ties.

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
