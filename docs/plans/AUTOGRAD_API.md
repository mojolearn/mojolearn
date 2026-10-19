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
  `log_softmax`, `mse_loss`, `cross_entropy`, `dropout`. Backward: `backward` (scalar),
  `backward_from` (explicit seed), `*_with_rules` variants for custom ops.
  `reset()` reuses storage across steps; stale handles are refused.
- Data layer (`autograd/data.mojo`): `permutation(ex, seed, epoch, n,
  dst_idx)`, `epoch_order(ex, shuffle, seed, epoch, n, dst_idx)` and
  `gather_rows(ex, X, idx, start, count, D, dst)`, forward only. See
  "Randomness" below.
- `Tensor`: an opaque handle (owner, generation, index); shape via
  `tape.rows/cols`, storage via `tape.value/gradient`.
- `BackwardRules` trait + `tape.custom(inputs, output, rows, cols, tag,
  saved)`: an already-executed native op with a VJP. The tape hands the rule
  one destination per input and owns ordering and accumulation; it zeroes a
  destination unless `zeroed_destination(tag, saved, edge)` says the rule
  overwrites it. `tape.custom_into` also names caller storage for the op's
  gradient; a seed that is that storage is used in place.
  `binary_recorded` records an add/subtract/multiply/divide whose value a
  caller kernel already computed (no launch, the tape's own VJP).
- Adapters: `autograd/layers.mojo` (`layer_norm`, `NeuralRules`) and
  `autograd/rules_recurrent.mojo` (`rnn_linear`, `rnn_layer`, `rnn_head`,
  `rnn_last_step`, `RecurrentRules`, `AutogradRnnStep`).
  `autograd/rules_transformer.mojo` (`TransformerRules`, tags 32-37, and
  `llama_decoder_layer_backward_autograd`).
  `autograd/rules_mamba.mojo` (`Mamba1Rules`, `Mamba2Rules`, the
  `mamba*_tape_backward[_stack]` assemblies).
- Trainers on the tape, opt-in: MLP (`-D MOJOLEARN_AUTOGRAD_MLP`,
  `sequence/mlp_fit.mojo`), recurrent (`-D MOJOLEARN_AUTOGRAD_RNN`, the
  `RnnStep` seam in `sequence/recurrent.mojo::rnn_fit`), transformer
  decoder layer (`-D MOJOLEARN_AUTOGRAD_TRANSFORMER`, the two saved-stage
  backwards in `bindings/_mojolearn_transformer.mojo`).

Constraints that carry into any public surface:

- Every op is a parallel launch with a fixed fold order; identity across
  Apple, NVIDIA and AMD comes from the ops, not from the tape.
- Tags of custom rules are per family (LayerNorm 1, recurrent 16-18,
  transformer 32-47, Mamba 48-63). A
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

## Randomness

Contract: the same seed gives the same epoch order, the same batches and the
same dropout masks on Apple, NVIDIA, AMD and the host, in every run. Every
draw is a pure function of its position, never of launch geometry, thread
count or executor.

- Epoch order: `permutation(ex, seed, epoch, n, dst)` launches the trainers'
  `OP_MLP_PERM` (`sequence/mlp.mojo`) with `mlp_epoch_key(seed, epoch)`: six
  Feistel rounds on the bits of each position, cycle-walked into `[0, n)`,
  one thread per position, integers only. It is the MLP's and the recurrent
  trainer's order bit for bit, a different order per epoch and per seed.
  Rows travel as float32 words, so `n < 2^24`. `epoch_order` with
  `shuffle=False` is `0, 1, ..., n-1`.
- Batches: `gather_rows` (`OP_GATHER_ROWS`) copies rows `idx[start ..
  start + count)` of a row-major `X`, one thread per (row, column).
- Dropout: `tape.dropout(ex, x, p, seed, counter)` keeps element `e` when
  the uniform word of element `e` in the neural Philox4x32-10 stream
  (`core/philox_neural.mojo`: counter `(e // 4, stream = counter, 0)`, key
  `seed`, word `e % 4`, mapped exactly to `[0, 1)` as `(x >> 8) * 2^-24`) is
  `>= p`, and scales kept values by `s = 1 / (1 - p)` (one float32 division
  at record time, one rounded multiply per element); dropped elements are
  `+0.0`. The values equal `neural_rng`'s dropout arms for the same seed,
  stream and `p`. The VJP re-derives the mask from the same `(seed,
  counter)` instead of storing it, so no mask buffer lives between forward
  and backward. Callers pass a new `counter` per step (the step number) and
  the same one when a step is replayed; `0 <= p < 1`, `0 <= counter < 2^32`.
- Opcodes: dropout is `OP_AG_DROPOUT_FWD`/`OP_AG_DROPOUT_VJP` (1100/1101,
  `autograd/data_ops.mojo`), a range of its own beside the tape family
  (1000..1010), asserted disjoint in `sequence/dispatch.mojo`.
- Trainers: `-D MOJOLEARN_AUTOGRAD_DATA` routes the recurrent trainer's epoch
  order through `epoch_order` (same launches, same bits expected). The MLP
  keeps its own `OP_MLP_PERM` launch: `permutation` is that launch with the
  same key, so moving the MLP onto it would change no bits and needs no
  define.
- Checks: `autograd/checks/data_cpu.mojo` (HostExec) and `data_gpu.mojo`
  (DeviceExec) print a digest per case; the digests must match across the
  host and the three vendors.
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
2. Transformer decoder layer (`transformer/checks/transformer_backward.mojo::
   llama_decoder_layer_backward_device`, called by
   `bindings/_mojolearn_transformer.mojo::_transformer_saved_backward` and
   `_transformer_run_session_backward`): written under
   `MOJOLEARN_AUTOGRAD_TRANSFORMER` (lane ag-transformer, compile and A/B
   owed). `llama_decoder_layer_backward_autograd` takes the hand entry's
   arguments, records the saved forward on a `Tape` over `DeviceExec(ctx)`
   (the transformer's own context and queue) and runs one
   `backward_with_rules`. Moved to the tape: the stage order, the residual
   fan-outs (two `binary_recorded` adds over `fwd.residual1/2`, no forward
   relaunch) and every fan-in sum (gate + up, the norm2 residual, q + k + v,
   the norm1 residual), with gradients landing in the same
   `LlamaBackwardStages` fields (`custom_into`, `variable_into`). Stay, one
   rule each (`autograd/rules_transformer.mojo`, tags 32-37): RMSNorm
   (`bwd_rms_norm_routed`, residual unfused), each projection (`_route_a` /
   `_route_b`), RoPE (`bwd_rope_kernel`), attention (`bwd_attention_dispatch`,
   the fused/eager decision moved out of the hand path verbatim, then
   `bwd_kv_slice2_kernel`), SwiGLU (the fused or split S4-6 kernels). The
   q/k/v projections are recorded v, k, q so the reverse traversal sums
   `(q + k) + v`, the hand path's pinned order; `OP_AG_ACCUMULATE` is
   `bwd_add2_kernel`'s arithmetic, so the expected result is the hand
   path's bits. The tape adds a seed copy, four ADD VJP copies and 25 zero
   fills of custom destinations per layer. Traced or materialized calls and
   non-default block options are refused (they keep the hand path). The
   embedding gather and LM head are not in this backward; they move with
   item 4.
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
| `MOJOLEARN_AUTOGRAD_DATA` | rnn/lstm/gru clf + reg rows with shuffle | ID check: arm B digest equals arm A (same launches) on nv and amd, and nv = amd | NV + AMD; routing only, expected noise | unchanged by construction |
| `MOJOLEARN_AUTOGRAD_RNN` | rnn/lstm/gru clf + reg rows, L = 1 and 2 | ID check: arm B digest equals arm A (same launches) on nv and amd, and nv = amd | NV + AMD full-size race; the tape adds fills/copies per node | unchanged by construction; recorded |
| `MOJOLEARN_AUTOGRAD_TRANSFORMER` | transformer LM rows | nv = amd = Apple digest; arm A equality expected only where launches are reused verbatim | NV + AMD | loss / perplexity no material drop |
| `MOJOLEARN_AUTOGRAD_MAMBA` | Mamba-1/2 LM rows | as transformer | NV + AMD | as transformer |

Verdict: IDENTICAL switches by the average NV/AMD ratio against the noise
floor; a winner becomes the default with an `_OFF` define, a loser is
deleted with the tombstone procedure. A digest change between arms is a
bit version only when the define intentionally changes a fold; for the
RNN define it is a bug. Rows go into `docs/apple-fast/EXPERIMENTS.md` when
the A/B runs.

## Optimizer sites

Survey for lane ag-optim (2026-10-19), formula level. `g` gradient, `p`
parameter, `m`/`v` moments, `t` one-based step. Every site flushes
denormals (`ftz`) on load and store and pins products and fused adds
(`identical_mul`, `identical_mul_add`) under IDENTICAL.

| Site | Kinds | Momentum / Nesterov | Weight decay | Clip | LR schedule | Bias correction | eps |
|---|---|---|---|---|---|---|---|
| `sequence/mlp_fit.mojo` (OP_OPT with OPT_SK_ADAM / OPT_SK_SGD) | sklearn Adam, sklearn SGD | SGD: velocity form, Nesterov flag | none (L2 `alpha` is added to the gradient before the step) | none | host, per epoch: constant, invscaling `lr0 / (t_samples + 1)^power_t`, adaptive (`lr / 5` on plateau) | running products `b1^t`, `b2^t` (one `identical_mul` a step); `lr_t = lr sqrt(1 - b2^t) / (1 - b1^t)` | outside the root |
| `sequence/recurrent.mojo` (OptConfig / opt_scalars / opt_step, OP_OPT) | torch SGD, Adam, AdamW, RMSprop (centered), Adagrad, Lion, Adamax, NAdam | SGD: torch buffer form with dampening, Nesterov; RMSprop momentum | coupled (`g += wd p`) for SGD/Adam/RMSprop/Adagrad/Adamax/NAdam; decoupled (`p *= 1 - lr wd`) for AdamW, Lion, NAdam flag | none | per-step `lrs[k]` array from the caller | running products; `f5 = lr / (1 - b1^t)`, `f6 = sqrt(1 - b2^t)` | outside the root |
| `training/estimator.mojo` -> `training/checks/optimizer.mojo` (`identical_optimizer_step`) | SGD, Adam, AdamW (`mojolearn.identical.optimizer.fp32.v1`) | SGD torch buffer form (first step copies g), dampening, Nesterov | coupled for SGD/Adam, decoupled for AdamW | global norm: per-tensor sumsq, per-tensor sqrt, norm of norms by GEMM, `coef = min(1, max_norm / (norm + 1e-6))`, unconditional scale | caller sets `lr` per step | `b^t` by square-and-multiply (`pow_int_f32`), not a running product; `step = lr / bc1`, `denom = sqrt(v) / sqrt(bc2) + eps` | outside the root |
| `training/afn_optim.mojo` (Apple FAST only) | Adam, AdamW, SGD | as the certified step | as the certified step | fused free-order norm, coefficient on device | caller | as the certified step, unpinned | outside the root |
| `training/mlp_fast.mojo` (FAST fused MLP) | AdamW only, fused with the gradient fold | none | decoupled | none | caller | `_pow_f32` square-and-multiply, unpinned | outside the root |

Update formulas (the OP_OPT body, `sequence/ops.mojo::op_opt`, which both
sequence trainers share):

- torch SGD: `g += wd p`; `buf = g` at `t = 1`, else `buf = mu buf + (1 - damp) g`; Nesterov `g = g + mu buf`, else `g = buf`; `p = p - lr g`.
- torch Adam / AdamW: AdamW `p *= 1 - lr wd`, Adam `g += wd p`; `m = b1 m + (1 - b1) g`; `v = b2 v + (1 - b2) g g`; `p = p - (lr / bc1) m / (sqrt(v) / sqrt(bc2) + eps)`.
- sklearn Adam: `m`, `v` as Adam; `p = p + (-lr_t m) / (sqrt(v) + eps)`.
- sklearn SGD: `vel = mu vel - lr g`; Nesterov step `mu vel - lr g` with the new `vel`, else `vel`; `p = p + step`.
- RMSprop, Adagrad, Lion, Adamax, NAdam: torch's rules as written in `op_opt`.

The certified LM step differs from OP_OPT's Adam in two places: its
`b^t` is square-and-multiply rather than a running product, and its
quotient is `m / (sqrt(v) / sqrt(bc2) + eps)` with `step_size = lr / bc1`
(the same algebra as OP_OPT's `f5`/`f6`, with a different `b^t` rounding).
So the LM trainers keep their own step until a separate A/B moves them.

`autograd/optim.mojo` is the tape trainers' set: every OP_OPT kind (a
superset of what the MLP and recurrent trainers use), the same element
body and the same host scalars, plus a global-norm clip (blocked fold, two
launches, then one scale launch) and host learning-rate schedules
(constant, step, cosine, linear warmup). Under `-D MOJOLEARN_AUTOGRAD_OPTIM`
the MLP and recurrent trainers update through it; with the clip off and a
constant schedule the launches and their arguments are the ones the hand
update issues, so the bits are the same.
