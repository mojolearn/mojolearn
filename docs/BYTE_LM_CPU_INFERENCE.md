# Byte LM inference on the CPU

`mojolearn.LanguageModelInference` runs the byte-level decoder language model's
forward pass on a CPU with no GPU. It loads the parameters of a checkpoint
written by `LanguageModelTrainer.export_checkpoint` and returns logits, the
mean next-byte loss, or greedy next bytes.

```python
import mojolearn
model = mojolearn.LanguageModelInference.from_checkpoint("final.checkpoint.json")
logits = model.logits(ids)          # int32 [batch, length] -> float32 [batch, length, 256]
bits = model.loss_bits(batch_ids)   # int32 [2, 33] -> IEEE-754 bits of the mean loss
```

## What it computes and why it can match the GPUs

There is no new arithmetic. The CPU forward calls the host FP32 oracles that
the Metal, CUDA and HIP kernels are gated against bit for bit, in the order
`training/byte_lm.mojo::_byte_forward_loss` launches those kernels.

| stage | CPU (this path) | GPU kernel it mirrors |
|---|---|---|
| embedding | `emb_forward_oracle` | `identical_embedding_forward_into` |
| each block | `transformer_block_oracle`, fresh cache, prefill at 0 | `llama_decoder_layer_forward` |
| head | `gemm_oracle(OP_NT)` | `identical_gemm_into(..., OP_NT)` |
| loss | `ce_forward_oracle(causal_lm)` | `identical_ce_forward_into` |

That composition is a prediction until the gate below runs on a CPU. A CPU is
certified only by its own row in the table at the end of this file.

## Scope

- One model family, the byte LM profile
  `mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`.
- FP32, forward only. No training, no backward pass, no optimizer on the CPU.
- The oracles are single-threaded scalar loops. This path is correct first and
  slow; it is not a performance claim of any kind.
- Every GPU estimator, block and trainer still requires a GPU. On a CPU-only
  install they raise by name on use (DEVIATION 2615).

## The gate

`tools/byte_lm_host_gate.py` (DEVIATION 2613) loads parameters from the retained
three-vendor capture
`bench/results/resume/2026-09-07-root-byte-lm-three-vendor/`, checks every file
against the SHA-256 its own manifest recorded, and requires the CPU loss to
reproduce the recorded 4 loss bytes for

- 8 held-out batches before training (initial parameters),
- 8 held-out batches after 128 steps (final parameters),
- the training batches of the selected steps (parameters before each step).

The capture is Apple's, and its `comparison.json` records that CUDA and HIP
matched it on every one of these bytes, so a match is a match against all three.
The gate also records SHA-256 of fixed logits probes so CPUs can be compared at
full `[batch, length, vocab]` resolution.

The negative control (DEVIATION 2612) is a build with
`-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1`, which folds the head product in reverse
order. The gate run on that build must find a mismatch.

## Running it

On a rented CPU-only droplet (both builds, both gates, logs come home):

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_host_leg.sh \
MOJOLEARN_DO_EXTRA_PAYLOAD=/path/to/byte_lm_host_capture_payload.tgz \
bash tools/do_extra_leg.sh cpu-intel --minutes 45
```

`cpu-amd` is the AMD twin. The payload is the capture subset `git archive`
excludes; see DEVIATION 2614 in `tools/do_extra_leg.sh`.

## Certified CPUs

| CPU | host | build | loss bytes equal | sabotage caught | evidence |
|---|---|---|---|---|---|
| none yet | | | | | |
