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
- The reference path is the oracles as written, single-threaded scalar loops.
  `threaded=True` (DEVIATION 2616) splits the same arithmetic across cores
  along axes the contracts make independent: one task per batch row, or one
  task per token row of the head product for a single sequence. No float
  crosses a thread and no fold changes order, and the gate requires both
  paths to reproduce the capture bytes and each other's logits. Neither path
  vectorizes, and neither is a performance claim of any kind.
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
| Intel x86_64, DigitalOcean Premium Intel (model name masked by QEMU; AVX2, FMA), 4 vCPU | Ubuntu 24.04, Linux 6.8, Python 3.12.3 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `146becba` | 33 of 33 (16 held-out, 17 training steps) | yes, 9 of 33 differ | `bench/results/e1g/2026-09-11_163928-cpu-intel-byte-lm-host` |

Logits probes on the final parameters, for comparing CPUs with each other at
full resolution (SHA-256 of the float32 bytes): held-out batch 00 `[2, 32]`
`2e2408f47392b4a1b69a791cf4672cdf6e49fcc0f576e9bcd6b7018f0ffaa6c4`, its first
token alone `[1, 1]`
`8f46027a73524437c25bc9988e309fdc3f37a4210447a56577f2ed1a1401e839`.

Operational timing on that box, not a benchmark: one `[2, 32]` forward plus
the loss took 29 to 56 ms on one thread through the Python surface.
