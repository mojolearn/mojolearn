# Chunked LM-head v2 device forward/loss evidence

- Date: 2026-09-20
- Host: macOS 26.5.2 arm64, local Metal device
- Numeric mode: IDENTICAL
- Shape/seed rule: rows=64, vocab=4096, width=128; integer modular fixtures
- Command: `/usr/bin/time -l pixi run python bench/chunked_lm_head_v2_device.py`
- Five post-warmup seconds: 0.101870, 0.106409, 0.106093, 0.106419, 0.106445
- Exact witness SHA-256 (loss + row maxima + denominators):
  `f5bbce4d8bd2b3b5f44d617270fd1ebc0a4b9a717fae3a6d0a99173ba31fc09d`
- Loss bits: `0x41a3977f`
- Process maximum RSS (`time -l`): 70,942,720 bytes
- Device stage scratch beyond inputs: 772 bytes (`(3*rows+1)*4`)
- Full logits allocation avoided at this shape: 1,048,576 bytes

The independent device/CPU gate uses rows=7, vocab=513, width=17 so it
crosses two full 256-token chunks and a one-token tail. Loss, every row
maximum, and every denominator match the normative CPU oracle bit-for-bit;
the repeated device loss bits also match. Its success sentinel is
`CHUNKED_LM_HEAD_V2_DEVICE_OK`.

This is forward/loss stage evidence, not a speedup claim against a fused v1
LM-head: the repository's v1 path materializes logits between a GEMM and CE,
whereas this stage deliberately exchanges recomputation for bounded memory.
Backward and production trainer selection remain outside this commit.
