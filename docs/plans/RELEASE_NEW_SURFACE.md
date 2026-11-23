# New public surface: the checklist the release walks

Everything public added from 2026-11-22 on. Each name is in `docs-site/api_exclude.txt` with the one reason
"not yet in a release verification". The release's one verification records every row below on every vendor
and the CPU column. Then each row's GPU run is done, its gaps are closed or named, and all the names leave
`api_exclude.txt` in ONE commit. Nothing below has been recorded. Do not record before the release.

Columns:
- **verify lane**: the identity lane (`tools/identity_lanes/sequence.py`, routine profile).
- **endpoint row**: the bounded fixture of the verify public endpoints
  (`_verify_api_neural.SEQUENTIAL_LAYER_FIXTURES`, a row of `_verification_api_targets.py`). Its record comes
  from `python -m mojolearn verify` at the release.
- **sabotage**: the fault-injection define shown CAUGHT on the CPU column.
- **recipe**: a shipped `verify --recipe example:<name>`.
- **native case**: the TestSuite file under `checks/native_tests/`.
- **boundary**: the Python test under `python/mojolearn/tests/`.
- **GPU**: the L40S run that is owed.

Lane short names used in the table: sb = seq-blocks, sb3 = seq-blocks-3, nb = neural-blocks, nb2 = neural-blocks-2,
nb3 = neural-blocks-3.

## Sequential layer methods (2026-11-22)

| name | family (lane) | verify lane | endpoint row | sabotage | recipe | native case | boundary | GPU owed |
|---|---|---|---|---|---|---|---|---|
| `Sequential.conv1d` | 1-D conv (sb) | sequence-blocks-cnn1d, -resnet1d, -inceptiontime, sequence-composed-conv1d | yes | `MOJOLEARN_SEQ_BLOCKS_SABOTAGE_CONV` | seq-conv1d-residual | sequence_blocks, neural_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.max_pool1d` | pools (sb) | sequence-blocks-cnn1d, -inceptiontime | yes | **none (owed: a pool arm)** | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.avg_pool1d` | pools (sb) | **none** | yes | **none** | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.adaptive_avg_pool1d` | pools (sb) | sequence-blocks-cnn1d, -resnet1d, -inceptiontime | yes | **none** | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.branch` | containers (sb) | sequence-blocks-resnet1d (through residual_block) | yes | **none (owed: a renumbering arm)** | seq-conv1d-residual | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.concat` | containers (sb) | sequence-blocks-inceptiontime (through parallel) | yes | **none** | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.residual_block` | containers (sb) | sequence-blocks-resnet1d | yes | **none** | none (the recipes use `residual(source)`) | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.parallel` | containers (sb) | sequence-blocks-inceptiontime | yes | **none** | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.gated` | containers (nb) | **none** | yes | **none** | none | neural_blocks | layers | neural_blocks device column |
| `Sequential.silu` | activations (nb) | sequence-composed-transformer, -conv1d | yes | **none (owed: an activation arm)** | seq-transformer | neural_blocks | layers | neural_blocks device column |
| `Sequential.gelu` | activations (nb) | **none** | yes | **none** | none | neural_blocks | layers | neural_blocks device column |
| `Sequential.softplus` | activations (nb) | sequence-composed-mamba1, -mamba2 | yes | **none** | seq-mamba2 | neural_blocks | blocks, layers | neural_blocks device column |
| `Sequential.rms_norm` | norms (nb) | sequence-composed-transformer | yes | **none** | seq-transformer | neural_blocks | layers | neural_blocks device column |
| `Sequential.rotary` | attention (nb) | sequence-composed-transformer (inside attention) | yes | **none** | seq-transformer (inside attention) | neural_blocks | layers | neural_blocks device column |
| `Sequential.multiply` | containers (nb) | **none** | yes | **none** | none | neural_blocks | layers | neural_blocks device column |
| `Sequential.narrow` | shape (nb) | sequence-composed-conv1d, sequence-blocks-*-lengths | yes | **none** | seq-conv1d-residual | neural_blocks, sequential_lengths | blocks, layers | neural_blocks device column |
| `Sequential.embedding` | embedding (nb) | **none** | yes | **none** | none | neural_blocks | layers | neural_blocks device column |
| `Sequential.attention` | attention (nb) | sequence-composed-transformer | yes | **none (owed: an attention arm)** | seq-transformer | neural_blocks (4 forms) | layers | neural_blocks device column; transformer_check |
| `Sequential.lstm` | recurrent (nb, sb3) | sequence-composed-recurrent, sequence-blocks-lstm-lengths | yes | `MOJOLEARN_SEQ_BLOCKS_SABOTAGE_MASK` (lengths only) | seq-recurrent | neural_blocks, sequential_lengths, sequence_blocks | layers | sequential_lengths device column |
| `Sequential.gru` | recurrent (nb, sb3) | sequence-composed-recurrent, sequence-blocks-gru-lengths | yes | `MOJOLEARN_SEQ_BLOCKS_SABOTAGE_MASK` (lengths only) | seq-recurrent | neural_blocks, sequential_lengths, sequence_blocks | blocks, layers | sequential_lengths device column |
| `Sequential.rnn` | recurrent (nb) | **none** | yes | **none** | none | neural_blocks, sequential_lengths | layers | sequential_lengths device column |
| `Sequential.batch_norm1d` | norms (sb3) | sequence-blocks-resnet1d, -inceptiontime | yes | `MOJOLEARN_SEQ_BLOCKS_SABOTAGE_BN` | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.batch_norm2d` | norms (sb3) | **none** | yes | `MOJOLEARN_SEQ_BLOCKS_SABOTAGE_BN` | none | sequence_blocks | blocks, layers | sequence_blocks device column |
| `Sequential.selective_scan` | Mamba-1 scan (nb2) | sequence-composed-mamba1 | yes | `MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED` | none | neural_blocks | blocks, layers | neural_blocks device column; mamba_tape_backward_check |
| `Sequential.ssd_scan` | Mamba-2 scan (nb3) | sequence-composed-mamba2 | yes | `MOJOLEARN_MAMBA2_SABOTAGE_SEGSUM_DESCENDING` | seq-mamba2 (with rms_norm after the scan since lane mamba2-nan) | neural_blocks | layers | neural_blocks device column; mamba2_check |
| `Sequential.trapezoid_scan` | Mamba-3 scan (nb3) | sequence-composed-mamba3 | yes | `MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY` | none | neural_blocks | layers | neural_blocks device column; mamba3_check |
| `Sequential.lm_head` | LM head (nb3) | **none** | yes | `MOJOLEARN_LMHEAD_SABOTAGE_LEAF_DESCENDING` | none | neural_blocks (lm_head_*) | layers | neural_blocks device column |

`blocks` = `test_sequential_blocks_boundary.py`. `layers` = `test_sequential_layers_boundary.py`: every method
is admitted and named, refuses bad arguments in clear words, and gives the same predictions bit for bit after
checkpoint and export.

## Arguments and losses (2026-11-22)

| name | family | verify lane | sabotage | native case | boundary | GPU owed |
|---|---|---|---|---|---|---|
| `Sequential.fit / predict / predict_proba (lengths=)` | recurrent (sb3) | sequence-blocks-gru-lengths, -lstm-lengths | `..._SABOTAGE_MASK` | sequential_lengths | blocks | sequential_lengths device column; `tools/ops/sequence_lengths_timing.py --label lengths` |
| `LSTMRegressor / GRURegressor / ... (lengths=)` | recurrent estimators (seq-blocks-2) | the estimators' sequence_lengths digests | `..._SABOTAGE_MASK` | sequence_lengths | **none (owed: an estimator lengths boundary test)** | sequence_lengths device column |
| `compile(loss="token_cross_entropy")` | losses (nb3-lm) | none | `MOJOLEARN_LMHEAD_SABOTAGE_LEAF_DESCENDING` | neural_blocks lm_head_token_loss_0/_1 | layers (lm_head fixture) | neural_blocks device column |

## Verify tooling and attributes (2026-11-22)

| name | family | tests | GPU owed | docs flag |
|---|---|---|---|---|
| `recipe` / `verify --recipe` | verify tool (verify-recipe) | `tools/tests/test_verify_recipe.py` (27); feature proof of the 8 shipped recipes | `verify --recipe example:<name> --self-test` for the 8 names on the L40S, then `--against` on a CPU box | api_exclude `recipe` |
| `training_digest_` | digest (digest-answers) | `tests/api/test_digest_answers.py` | from a built wheel on a GPU | api_exclude `training_digest_` (a member name; `gen_api.py` accepts it) |

## Consolidated trainers (2026-11-22)

`LanguageModelTrainer(device="cpu")` replaces `LanguageModelHostTrainer`. `ParallelByteLanguageModelTrainer(schedule=
"replica"|"model_pool"|"offload")` replaces `PooledByteLanguageModelTrainer` and `OffloadedByteLanguageModelTrainer`.
The blocks take `device="cpu"` in place of the `*BlockInference` classes. Verify lane byte-lm-host-train is
verified against the shipped table. Owed: GPU runs of the pooled and offloaded schedules.
`ParallelByteLanguageModelTrainer` keeps its multi-GPU exclusion line.

## Modules of the nn-* lanes (2026-11-23)

Each nn-* lane fills its own rows here when it lands. Its rows are: its layer codes, its `Sequential` methods
or `compile` options, its verify lane, sabotage arm, recipe, native case, boundary test and GPU checks. Each
lane adds its own `api_exclude.txt` lines with the one reason, plus rows in `SEQUENTIAL_LAYER_FIXTURES` and
`_verification_api_targets.py` (the 27 rows above show the form).
- nn-activations: layer codes 47 to 53.
- nn-norms: 54 to 55.
- nn-conv: 56 to 66.
- nn-attention: 67 to 72.
- nn-conv3d: 73 to 76.
- nn-losses: loss codes.
- nn-optim: optimizer and schedule codes.
- nn-recurrent: arguments.
- nn-losses-2: layer code 77 (log_softmax); loss code 9 (ctc). Rows:

| name | family (lane) | verify lane | endpoint row | sabotage | recipe | native case | boundary | GPU owed |
|---|---|---|---|---|---|---|---|---|
| `Sequential.log_softmax` | activations (nn-losses-2) | sequence-composed-log-softmax-ctc | yes | `MOJOLEARN_NN_LOSSES_SABOTAGE_LOG_SOFTMAX` | seq-nll-log-softmax, seq-ctc | nn_losses_2 | sequential_ctc, layers | nn_losses_2 device column |
| `compile(loss="ctc", blank=, zero_infinity=)`, `fit(target_lengths=)` | losses (nn-losses-2) | sequence-composed-log-softmax-ctc | yes (`compile:loss=ctc`) | `MOJOLEARN_NN_LOSSES_SABOTAGE_CTC_REPEAT` | seq-ctc | nn_losses_2 | sequential_ctc | nn_losses_2 device column |
| `compile(loss="l1" / "smooth_l1" / "nll" / "kl_div" / "bce_with_logits")`, `pos_weight=`, `beta=`, `reduction=` | losses (nn-losses) | sequence-composed-losses (and -log-softmax-ctc for nll, kl_div) | yes (`compile:loss=<name>`, added by nn-losses-2) | `MOJOLEARN_NN_LOSSES_SABOTAGE_SMOOTH_L1_HALF` | seq-l1, seq-smooth-l1, seq-nll-log-softmax | nn_losses, nn_losses_2 | sequential_losses | nn_losses device column |

## Records to take at the release (named, not recorded here)

1. `python -m mojolearn verify`. The public endpoints include the 27 `mojolearn.Sequential.<method>` rows above.
   The neural family's source closure changed, so every neural endpoint record is new for this release.
2. `python -m mojolearn verify --lanes sequence-blocks-cnn1d,sequence-blocks-resnet1d,sequence-blocks-inceptiontime,`
   `sequence-blocks-gru-lengths,sequence-blocks-lstm-lengths,sequence-composed-transformer,sequence-composed-mamba1,`
   `sequence-composed-mamba2,sequence-composed-mamba3,sequence-composed-recurrent,sequence-composed-conv1d`
   on every vendor and the CPU column.
3. `python -m mojolearn verify --recipe example:<name> --self-test` for the 8 shipped recipes on every device.
4. `python -m mojolearn verify --self-test` ("THE VERIFIER WORKS").

## Gaps to close before the names leave api_exclude

- Sabotage arms: pools, containers (branch / concat / residual_block / parallel / gated / multiply), activations,
  rms_norm, rotary, embedding, attention, the recurrent cells without lengths.
- Verify lanes: avg_pool1d, gated, gelu, multiply, embedding, rnn, batch_norm2d, lm_head.
- An estimator `lengths` boundary test.
- Every GPU column named above, on the L40S.
