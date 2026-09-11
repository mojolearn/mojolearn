# Byte LM logits on a GPU

`LanguageModelTrainer.logits(ids)` and `next_bytes(ids)` (DEVIATION 2660) run
the model's forward pass on the GPU and return what it predicts, without
training and without changing any state.

```python
trainer = mojolearn.LanguageModelTrainer(parameters, data_schedule=schedule, resident=True)
scores = trainer.logits(ids)        # int32 [batch, length] -> float32 [batch, length, 256]
nxt = trainer.next_bytes(ids)       # the greedy next byte after each row
```

Before this the GPU side could train and report a loss, so the model's own
predictions could only be read on the CPU, through
`LanguageModelInference` (docs/BYTE_LM_CPU_INFERENCE.md). That is the gap this
closes, and it is what lets the inference claim name GPUs as well as CPUs.

## What it computes

`training/byte_lm.mojo::_byte_forward_loss` launches three things before the
loss: `identical_embedding_forward_into`, one `llama_decoder_layer_forward`
per block as a fresh prefill from absolute position 0, and the head
`identical_gemm_into(..., OP_NT)`. `training/byte_lm_logits.mojo` launches the
same three on the same weights and stops there. Training is untouched.

The block stages and the KV cache are allocated for the call's own
`[batch, length]`, because `llama_decoder_layer_forward` refuses stages built
for another shape. So every shape computes exactly what the CPU reference path
computes for it, with no padding and no dependence on the training batch.

Two entry points, both writing no parameter, moment, flag or step. The
stateless one builds one device context per call, drains its pending frees and
destroys it (DEVIATION 2520). The resident one runs on an open session and
takes the caller's committed step, refusing a mismatch, which is the scalar
half of the admission `byte_lm_session_eval` performs.

A non-finite logit is refused rather than returned. A computed NaN's payload
is vendor-shaped (IDENTITY_PATHS row 39), so it can never be part of an
identity claim. The CPU class returns such values; this path does not.

## How it is checked

`tools/byte_lm_gpu_logits_sweep.py` requires the GPU to equal the CPU
reference path, `LanguageModelInference(threaded=False)`, the oracles as
written, byte for byte:

- logits at every batch from 1 to 8 and every length from 1 to 32, on a
  resident trainer, and at a sampled subset on a stateless one,
- `next_bytes` at each of those shapes,
- the IEEE-754 bits of `evaluate()` against `loss_bits` on random batches,
- for three parameter states of the retained three-vendor capture (before
  training, mid-training and after 128 steps), with seeded ids that reach all
  256 byte values.

The ids are drawn in the order `tools/byte_lm_host_path_sweep.py` draws them,
so the SHA-256 per state over the GPU logits must equal the value that CPU
sweep recorded. That ties the GPU to every CPU already certified, at every
shape, not only to the CPU in the same box.

A negative control fails the sweep unless the logits of two states with
different parameters differ on the same ids, so a pass cannot come from a
comparison that compares nothing.

`tools/byte_lm_gpu_logits_leg.sh` runs all of it on a rented GPU: it builds
the three bindings on the box, fetches the three parameter states from GitHub
at the leg's own commit and verifies their SHA-256, then runs this sweep and
the CPU path sweep, recording each phase's exit code.

## Certified GPUs

| GPU | build | sweep against the CPU reference | per-state logits equal the CPU sweep | negative control | evidence |
|---|---|---|---|---|---|
| AMD MI325X (HIP, gfx942), DigitalOcean | `--target-accelerator gfx942`, Mojo 1.0.0, commit `652b93fe` | PASS, 1680 of 1680 (768 resident logits, 96 stateless, 768 next_bytes, 48 loss bits) | yes: `6db55997`, `30a89281`, `b518e71e` | differs | `bench/results/e1g/2026-09-11_1800-amd-mi325x-do-gpu-logits` |
| NVIDIA H100 80GB HBM3 (CUDA, sm_90a), DigitalOcean | `--target-accelerator sm_90a`, Mojo 1.0.0, commit `652b93fe` | PASS, 1680 of 1680, the same counts | yes: `6db55997`, `30a89281`, `b518e71e` | differs | `bench/results/e1g/2026-09-11_1806-nvidia-h100-do-gpu-logits` |
| Apple M4 (Metal) | | OWED | | | |

Each row also carries a CPU path sweep from its own box, 4752 of 4752 with the
same digests: an AMD EPYC 9575F beside the MI325X, an Intel Xeon Platinum 8468
beside the H100.

Until the Apple row is filled, this file supports two GPU vendors, AMD and
NVIDIA. A measurement on one column is a result about that column only.

## Scope

- One model family, the byte LM profile
  `mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`, at most 32
  tokens per row.
- Forward only. No training, backward pass or optimizer through this surface.
- Whole sequences only. There is no KV-cache decode, so generating text means
  calling again on the growing prefix, which recomputes the prefix.
- `batch` at most 1024 and `batch * length * vocab` at most 268435456, refused
  in Python before any native call, as are ids outside `[0, 256)`.
