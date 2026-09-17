# LM training shakedown: what an H100 actually did

Lane `lane/lm-training-shakedown`. Leg 1 ran 2026-09-17 20:07Z on a RunPod
NVIDIA H100 80GB HBM3 (sm_90a), commit 025910137, driver and card in the leg
directory. Corpus enwik8, staged from R2, 100,000,000 bytes, sha256
`2b49720e...`. Every arm is the IDENTICAL shipped default; nothing here is an
opponent ratio or a default gate.

Target shape throughout: batch B, length 2048, d_model 768, 12 heads, 12 KV,
head_dim 64, intermediate 2048, 12 layers, vocab 50257, 162,147,840
parameters. Resident session, `step_result='lean'`, no per-step witnesses,
three steps per cell with the first (which carries setup) excluded from the
median.

## 1. The batch sweep, and it is nothing like the analytic bound

| batch | tokens/step | median s/step | tokens/s | device peak (this process) | host RSS | first call |
|---|---|---|---|---|---|---|
| 1 | 2,048 | 0.20708 | 9,890.0 | 16.95 GB | 8.32 GB | 13.51 s |
| 2 | 4,096 | 0.37504 | 10,921.6 | 26.61 GB | 8.27 GB | 12.31 s |
| 4 | 8,192 | 0.65732 | 12,462.7 | 45.94 GB | 8.30 GB | 13.03 s |
| 8 | 16,384 | 1.31258 | 12,482.3 | 84.33 GB | 8.27 GB | 13.67 s |

**All four fit.** Device peak is polled `nvidia-smi --query-compute-apps` for
this pid; the device-wide figure is within 12 MB of it in every row, so
nothing else was on the card.

**`bench/results/lm_capacity_2026-09-10/b8-l2048.json` is wrong by 2.06x and
should be read as retired.** It puts batch 8 at 161.75 GiB in the attention
subset alone and records `fit_admitted: false`. The measurement is 84.33 GB =
78.5 GiB for the WHOLE step, and it ran. That file labels itself "host
arithmetic only; not measured memory or throughput", which is exactly right;
the number inside it was never an OOM observation and must not be quoted as
one. The eight per-head attention matrices per layer it sums are not all live
at once.

**Memory is linear in batch with a fixed floor.** Least squares over the four
cells:

    device_GB = 7.367 + 9.624 * batch      (residual under 0.3 GB in every row)

The 9.624 GB per batch unit is the activation and attention working set at
L2048; the 7.367 GB floor is close to the 2.594 GB of parameters, gradient and
the two AdamW moments plus the fixed workspaces. On this card (81,559 MiB =
85.52 GB) that fit admits **batch 8 and refuses batch 9** (93.9 GB predicted).
Batch 8 ran at 98.6% of the device, which is a real run but not a margin
anyone should plan a 200-hour job around.

**Throughput saturates at batch 4.** 9,890 to 10,922 to 12,463 tokens/s is a
26% gain, and batch 8 adds 0.15% on top of batch 4 for 83% more memory. Batch
4 is the operating point: the same tokens per second as batch 8 at 45.94 GB
instead of 84.33 GB.

**Host RSS does not move with batch**: 8.27 to 8.32 GB across a 5x change in
device memory. The 21.9 GB host RSS recorded in the 2026-09-10 probe runs was
the STATELESS (`resident=False`) path, which copies the whole state per call;
the resident path costs 8.3 GB and is the one to use. That is a resolved
question, not an open risk.

## 2. Bitwise identity still holds at 162M, five days and one card later

The three-step witness at batch 1 on enwik8 reproduces the pinned
2026-09-12 record (`bench/results/e1g/2026-09-12_133007-nvidia-h100-owed-rest/
remote/attention-step/.../lm_summary.tsv`, step 3) hash for hash on a
different physical H100 at a different commit:

    gradients  80502bcbaa38a7f5c40a0f9886a7300b13ea31aaef29ac93a241adc4fe601b0d
    parameters cb6cfb17e14fc800c38b68bb37a6453a782a8d13de0107dd077aa5a2759ac808
    m          62a57517ac0891e078a2d17199044887b3e460bf5e42af74859ccf584be9a6eb
    v          eca204720d5c1c692d2ae808d79fb51bf90a502961202b0450615b7f0e4c2d6d
    flags      360d579dbd14759b41afdf7fb5e80c0101e15150ae401d59f92a1e32d129f7cb

This was not an arm of this lane; it fell out of the batch-1 cell and is
recorded because it is evidence.

## 3. The step is 11% faster than the number everyone is quoting

`bench/OPPONENT_REFERENCE.md` carries 0.2326 s for our IDENTICAL arm on
enwik8, measured at commit bb679f19 on 2026-09-12. The same probe, same
shape, same corpus, same arm, on this card at 025910137 gives **0.20708 s**,
which is 11.0% faster. Opponent columns are measured once per tuple and are
NOT re-measured here, so no ratio in that file is restated: our cell moved,
theirs was not observed, and the two cannot be divided across pods
(`bench/OPPONENT_REFERENCE.md` says so itself about `compile_bf16`).

## 4. Recosting from the measured throughput

| corpus | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|
| FineWeb-Edu 10BT, H100-hours | 280.9 | 254.3 | 222.9 | 222.5 |
| FineWeb-Edu 10BT, optimizer steps | 4,882,812 | 2,441,406 | 1,220,703 | 610,352 |
| 25B tokens, H100-hours | 702.2 | 635.8 | 557.2 | 556.3 |
| 25B tokens, optimizer steps | 12,207,031 | 6,103,516 | 3,051,758 | 1,525,879 |

Against the 999,999-step ceiling (SOURCE_BLOCKERS.md blocker A), **only batch
8 can reach 10B tokens at all**, and **no batch can reach 25B**: batch 13
would be needed and the card refuses batch 9.

At the $2.2 to $2.5 per H100-hour the original estimate was costed with,
10BT at batch 4 is about **$490 to $560** against the $700 quoted at batch 1,
and 25B is about **$1,230 to $1,390** against $2,000. The bill is 21% lower
than the batch-1 estimate and remains the least interesting number here.
