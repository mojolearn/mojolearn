# These three files were never measurements, and the batch-8 one is wrong by 2.06x

Added 2026-09-17 by `lane/lm-training-shakedown`. The JSON files here are left
exactly as they were recorded; this note is the correction beside them, because
the numbers inside were being read as a capacity verdict and they are not one.

Each file says so itself, in its own `qualification` field:

    "qualification": "host arithmetic only; not measured memory or throughput"

`b8-l2048.json` sums an "eight attention matrices per layer" term to
154,618,822,656 bytes, totals 161.75 GiB and records `fit_admitted: false`.
That `false` is an ARITHMETIC REFUSAL, not an out-of-memory observation. It was
being quoted as though batch 8 had been tried and had failed. It had not been
tried at all.

On 2026-09-17 it was tried, on a RunPod NVIDIA H100 80GB HBM3 at commit
025910137, three complete IDENTICAL steps per cell at the same shape, resident
session, enwik8 from R2:

| batch | analytic subset total here | MEASURED device peak | it ran |
|---|---|---|---|
| 1 | 22.33 GiB | 16.95 GB = 15.79 GiB | yes |
| 2 | (not filed) | 26.61 GB = 24.78 GiB | yes |
| 4 | (not filed) | 45.94 GB = 42.78 GiB | yes |
| 8 | 161.75 GiB | 84.33 GB = 78.54 GiB | **yes** |

Batch 8 fits, at 98.6% of the card. The eight per-layer attention matrices the
analytic term sums are not all live at once, and measured memory is linear at

    device_GB = 7.367 + 9.624 * batch

which admits batch 8 and refuses batch 9 on an 80 GiB card.

Full table, the throughput that goes with it, and the reason the batch sweep is
itself only a first-210-steps measurement:
`bench/results/lm_training_shakedown_2026-09-17/MEASURED.md`.
