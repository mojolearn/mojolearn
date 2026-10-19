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
- Fused nodes: `linear` / `linear_into` (see "Fused nodes" below) and
  `backward_from_preactivation`, `preactivation_gradient`.
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

### Fused nodes

Model builders should record dense layers with `tape.linear(ex, x, W, b,
act)` (or `linear_into`, output in caller storage), not with `matmul`, `add`
and an activation. The tape never pattern-matches three plain ops into one
node at record time: fusion is opt-in by API, and the plain ops keep their
own launches and fold order.

- One node computes `act(x @ W + b)`; `act` is 0 (none), `RELU`,
  `SIGMOID`, `TANH` or `SOFTMAX`; the bias is optional.
- Forward: one launch, the MLP trainer's `gemm_epi` with `EPI_BIAS_ACT`
  (softmax adds its row launch; without a bias, `OP_GEMM` + `OP_ACT`).
- Backward: three launches, the hand MLP's. dW = `gemm_epi EPI_L2GRAD`
  (`OP_GEMM` when unscaled), db = `OP_COLSUM_DIV` (`OP_COLSUM`), dX =
  `gemm_epi EPI_ACT_BWD`, which applies the input layer's activation
  derivative in the epilogue when that input is a linear node consumed only
  here; its gradient buffer then holds dL/dZ (`preactivation_gradient`;
  `gradient` refuses it). Otherwise dZ is formed in place (`OP_ACT_BWD`) or,
  for softmax, by the softmax VJP into scratch: one more launch.
- `l2` and `divisor` put the trainer's scaling in the epilogue: the layer's
  weight gradient is (x^T dZ + l2 W) / divisor and its bias gradient
  colsum(dZ) / divisor, so a trainer exports nothing after backward.
- `backward_from_preactivation(ex, out, seed)`: the seed is dL/dZ of the
  output layer (the softmax or sigmoid with log loss derivative p - y), read
  in place without the root copy.
- Per layer: hand MLP 1 + 3 launches, unfused tape 3 + 9, fused node 1 + 3
  (2 backward for a first layer). The MLP uses it under
  `-D MOJOLEARN_AUTOGRAD_MLP_FUSED`; recurrent gates and transformer MLPs
  are the next users.
- Opcodes: none new. The node reuses `OP_GEMM_EPI`, `OP_COLSUM_DIV`,
  `OP_COLSUM`, `OP_ACT`, `OP_ACT_BWD`, `OP_SOFTMAX` and `OP_AG_SOFTMAX_VJP`
  (range 1200..1219 stays reserved for this lane's fused bodies).
- Check: `autograd/checks/fused_cpu.mojo`, `fused_gpu.mojo` (fused vs
  unfused tape), and `mlp_cpu` / `mlp_gpu` `DIGEST` lines across the
  `_MLP_OFF`, default and `_MLP_FUSED` builds.

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
| `MOJOLEARN_AUTOGRAD_MLP_FUSED` | MLP rows; three arms: hand (`_MLP_OFF`), tape, fused tape | fused digest equals the hand digest (same launches and arguments), nv = amd | NV + AMD full-size race | equal to the hand path by construction |
| `MOJOLEARN_AUTOGRAD_RNN` | rnn/lstm/gru clf + reg rows, L = 1 and 2 | ID check: arm B digest equals arm A (same launches) on nv and amd, and nv = amd | NV + AMD full-size race; the tape adds fills/copies per node | unchanged by construction; recorded |
| `MOJOLEARN_AUTOGRAD_TRANSFORMER` | transformer LM rows | nv = amd = Apple digest; arm A equality expected only where launches are reused verbatim | NV + AMD | loss / perplexity no material drop |
| `MOJOLEARN_AUTOGRAD_MAMBA` | Mamba-1/2 LM rows | as transformer | NV + AMD | as transformer |

Verdict: IDENTICAL switches by the average NV/AMD ratio against the noise
floor; a winner becomes the default with an `_OFF` define, a loser is
deleted with the tombstone procedure. A digest change between arms is a
bit version only when the define intentionally changes a fold; for the
RNN define it is a bug. Rows go into `docs/apple-fast/EXPERIMENTS.md` when
the A/B runs.
