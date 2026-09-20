# Chunked LM-head v2 device forward/backward evidence

- Date: 2026-09-20
- Host: macOS 26.5.2 arm64, local Metal device
- Numeric mode: IDENTICAL
- Shape/seed rule: rows=256, vocab=8192, width=256; integer modular fixtures
- Command: `/usr/bin/time -l pixi run python bench/chunked_lm_head_v2_device.py --rows 256 --vocab 8192 --width 256 --reps 3`
- Three post-warmup forward+backward seconds: 0.938357, 0.946982, 0.957327
- Exact witness SHA-256 (loss, row statistics, and both gradients):
  `75375219dcbac3254b6b6cd338ebf5349b1f38c39f32f1e9f79e49a83b437cd9`
- Same-batch loss before/after a float32 SGD update at 1e-3:
  20.244848251342773 -> 20.24451446533203
- Process maximum RSS (`time -l`): 132,399,104 bytes
- Device stage scratch beyond inputs and returned gradients: 3,076 bytes (`(3*rows+1)*4`)
- Full logits allocation avoided at this shape: 8,388,608 bytes

The independent device/CPU gate uses rows=7, vocab=513, width=17 so it
crosses two full 256-token chunks and a one-token tail. Loss, every row
maximum, denominator, dHidden cell, and dWeight cell match the normative CPU
oracle bit-for-bit; repeated device loss and gradient bits also match. Its success sentinel is
`CHUNKED_LM_HEAD_V2_DEVICE_OK`.

This is a bounded-memory training-stage result, not a speedup claim against v1
LM-head: the repository's v1 path materializes logits between a GEMM and CE,
whereas this stage deliberately exchanges recomputation for bounded memory.
The explicit opt-in selector is `chunked_lm_head_v2_train`; existing trainers
and V1 remain unchanged by default.
