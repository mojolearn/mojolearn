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
  `autograd/rules_mamba.mojo` (`Mamba1Rules`, `Mamba2Rules`, the
  `mamba*_tape_backward[_stack]` assemblies).
- Trainers on the tape, opt-in: MLP (`-D MOJOLEARN_AUTOGRAD_MLP`,
  `sequence/mlp_fit.mojo`), recurrent (`-D MOJOLEARN_AUTOGRAD_RNN`, the
  `RnnStep` seam in `sequence/recurrent.mojo::rnn_fit`).

Constraints that carry into any public surface:

- Every op is a parallel launch with a fixed fold order; identity across
  Apple, NVIDIA and AMD comes from the ops, not from the tape.
- Tags of custom rules are per family (LayerNorm 1, recurrent 16-18, Mamba 48-63). A
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
   Code-complete under `MOJOLEARN_AUTOGRAD_MAMBA` (lane ag-mamba, not yet
   compiled or run): `autograd/rules_mamba.mojo` records each block over the
   forward's own stage buffers as custom nodes (Mamba-1: norm, in_proj,
   conv, x_proj, dt, scan, gate, out_proj, tags 48-55; Mamba-2: norm,
   in_proj, SSD, gated norm, out_proj, tags 56-60), the residual as a tape
   add, blocks chained through it (`mamba1_tape_backward_stack`,
   `mamba2_tape_backward_stack`), one `backward_with_rules`. The tape runs
   on `DeviceExec(ctx)` over the binding's own context. Every rule launches
   the hand kernels with their arguments; the tape owns the three joins the
   hand path fused (residual into the norm, du's D/scan/x_proj join, the
   dP / dXP / in_proj packs), in the hand order, with -0.0 standing in for
   the terms another node supplies. Expected: same bits as the hand path;
   the only possible differences are IDENTICAL-mode subnormals that the
   tape flushes and a hand copy did not (header of the module).
   Check: `pixi run check-mamba-tape-backward`. Mamba-3 stays hand-chained
   (it shares neither kernel set).
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
