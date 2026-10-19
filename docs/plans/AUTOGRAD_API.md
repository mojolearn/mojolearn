# Autograd public surface

Scope: what `autograd/` exposes in Mojo today, the Python model API that
keeps the no-Python-in-the-runtime rule (implemented, section (b)), the
trainer migration order, and the A/B that decides each migration.
Implementation record: `NATIVE_AUTOGRAD.md`.

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
- Metal, NVIDIA and AMD: the 97-case gradient suite gives the same digests
  on all three and on the host (NATIVE_AUTOGRAD.md, "GPU status").
- `python/mojolearn/_training_impl.py:80` and `:1379` (`step()` rationale).
- `python/mojolearn/_mlp_impl.py:6` ("not autograd").
- `python/mojolearn/embedding.py:11` ("There is no autograd here").

The public surface is listed in `docs-site/api_exclude.txt` until its
gradient checks pass on Apple, NVIDIA and AMD.

## (b) Python model API: the model description (implemented)

`mojolearn.Sequential` (`python/mojolearn/_x_sequence_model.py`) records
layers and options; one Mojo call runs the whole fit
(`autograd/pyapi.mojo` `model_fit_py` -> `autograd/model.mojo`
`model_fit`) or prediction (`model_predict`). Python never computes on data
and never generates weights. Listed in `docs-site/api_exclude.txt` until its
checks pass on Apple, NVIDIA and AMD.

Description, shared by every entry (validated by `parse_model`, which raises
naming the first bad layer):

- `prog` (ints) = `[1 (version), input features, layer count]` + per layer
  `(code, arg0, arg1, 0)`.
- `ptab` (floats) = one per layer.

| Code | Layer | arg0 | arg1 | float | Parameters (flat vector, in layer order) |
|---|---|---|---|---|---|
| 1 | Linear | output units | bias 0/1 | - | weight (in, out) row-major, then bias (out) |
| 2 | ReLU | - | - | - | - |
| 3 | Sigmoid | - | - | - | - |
| 4 | Tanh | - | - | - | - |
| 5 | Softmax (rows) | - | - | - | - |
| 6 | LayerNorm | - | - | epsilon | weight (width), then bias (width) |
| 7 | Dropout | - | - | drop probability | - (needs `-D MOJOLEARN_AUTOGRAD_HAS_DROPOUT`, lane ag-data's `Tape.dropout`; refused by name otherwise) |
| 8 | Residual | source (-1 the input, else an earlier layer) | - | - | - |

GELU is left out: the tape has no GELU op yet.

Entries (`bindings/_mojolearn_x_sequence.mojo` and the host twin):

- `model_n_params(prog, ptab)` -> `[parameter count, output width]`.
- `model_fit([X, Y, params, loss_curve], prog, ip, fp)` -> epochs run.
  `ip = [N, loss, target, optimizer, optimizer flags, schedule, schedule
  step, batch, epochs, shuffle, seed_lo, seed_hi, n_iter_no_change, init]`;
  `fp = [lr, f1 (momentum / beta1), f2 (beta2), eps, weight_decay,
  f7 (dampening), tol, schedule gamma, huber delta] + ptab`.
  Loss 0 mse (mean over elements), 1 cross_entropy (logits, mean over rows),
  2 bce_with_logits (mean over elements), 3 huber (mean over elements).
  Target 0 dense `(N, out)` float32, 1 N int32 class codes (one-hot built on
  the executor, cross_entropy), 2 N int32 codes as the `(N, 1)` float target
  (bce_with_logits). Optimizer = `sequence/ops.mojo` `OPT_SGD` (momentum in
  f1, flag bit 0 Nesterov), `OPT_ADAM`, `OPT_ADAMW`, run by
  `sequence/recurrent.mojo` `opt_step` (lane ag-optim's `autograd/optim.mojo`
  replaces it when it lands). Schedule 0 constant, 1 step (`lr gamma^(epoch //
  step)`), 2 exponential (`lr gamma^epoch`), a per-epoch scalar in Mojo.
  `n_iter_no_change = k > 0` stops after k epochs without an improvement
  greater than tol (one word read back per epoch); 0 runs every epoch.
- `model_predict([X, params, out], prog, ptab, [N, chunk, transform])`;
  transform 0 raw, 1 row softmax, 2 sigmoid, 3 the `(N, 2)` `[1 - p, p]` of a
  one-output sigmoid.

Fit, per epoch: the row order on the executor (`OP_MLP_PERM` keyed by
`mlp_epoch_key(seed, epoch)`, or the identity order); per batch one gather
of X and Y rows, `tape.reset()`, the forward recorded from the description
(parameters are `variable_into` leaves whose gradients land in the gradient
vector), the loss, `backward_scalar_with_rules(NeuralRules)`, the batch loss
times its rows into a slot, and one optimizer step. The epoch loss is a
blocked fold (`tape.sum`) of the slots times 1/N. bce_with_logits and huber
are compositions of tape ops (`max(x,0) - xy + log(1 + exp(-|x|))`;
`0.5 q^2 + delta (|r| - q)`, `q = min(|r|, delta)`), so every fold is the
tape's blocked fold (lane ag-loss's `autograd/losses.mojo` replaces them
when it lands).

Randomness: parameters are initialised on the executor by
`OP_AGM_INIT_UNIFORM` (`autograd/model_ops.mojo`, opcode 1040): Philox4x32-10
word 0 at counter `(flat parameter index, 0, 0x4D4F4445, 0)` under the 64-bit
seed, `u = (word >> 8) / 2^24`, value `bound (2u - 1)` with
`bound = sqrt(6 / (in + out))` (Linear weight and bias); LayerNorm weight 1,
bias 0. Each value depends on the seed and its position only, so the same
seed gives the same parameters on every executor.

Prediction records the forward with gradient-free leaves, chunked by rows
(lane ag-forward's inference mode is used under
`-D MOJOLEARN_AUTOGRAD_HAS_INFERENCE`). Host loops run over layers, batches,
epochs and row chunks only (`# driver-loop:` tags).

Checks: `python/mojolearn/tests/test_autograd_model.py` (loss falls, shapes,
same digest for the same seed across two fits, refusals). Cross-vendor
identity: its `MODEL_DIGEST` lines must be equal on Apple, NVIDIA and AMD.

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
