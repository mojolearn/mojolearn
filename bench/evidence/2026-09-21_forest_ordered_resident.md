# IDENTICAL ordered resident RF/ExtraTrees inference

**Verdict: promoted on NVIDIA IDENTICAL.** The route retains the resident GPU
model and I/O workspaces while using the existing strict increasing-tree
aggregation kernel. NVIDIA IDENTICAL `inference_engine="auto"` selects it by
default. `-D MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF=1` restores the former
sequential AUTO path and resident 32-grove graph. Other vendors and numeric
modes retain their previous defaults; the positive define remains available
for explicit IDENTICAL experiments on them.

The run reused the warm RunPod H100 and the two Cloudflare R2 objects already
staged by `tools/stage_from_r2.sh`: Taxi and Istella-S. Models were fitted once
on the first 1,000,000 training rows. Repeated inference used 1,000,000 rows
from each source: Taxi's 11 numeric columns and Istella-S's 220 columns. Each
cell has three alternating baseline/candidate processes, one excluded warmup,
and five retained calls per process. Training and dataset loading are outside
the clock; the public call, validation, upload, inference, readback, and
synchronization are inside it.

| dataset | model | operation | sequential ms | ordered resident ms | speedup | conservative candidate/baseline | quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxi | RF | predict | 2777.501 | 16.049 | 173.06x | 0.0069 | accuracy 0.784290 |
| Taxi | RF | proba | 2934.005 | 12.934 | 226.84x | 0.0057 | logloss 0.4899468627365843 |
| Taxi | ET | predict | 2171.792 | 13.685 | 158.70x | 0.0073 | accuracy 0.779572 |
| Taxi | ET | proba | 2105.932 | 12.096 | 174.10x | 0.0071 | logloss 0.5114427944488870 |
| Istella-S | RF | predict | 2833.183 | 152.595 | 18.57x | 0.0581 | accuracy 0.965139 |
| Istella-S | RF | proba | 2816.891 | 149.195 | 18.88x | 0.0581 | logloss 0.1061623170680610 |
| Istella-S | ET | predict | 2686.771 | 148.582 | 18.08x | 0.0650 | accuracy 0.938552 |
| Istella-S | ET | proba | 2479.831 | 146.159 | 16.97x | 0.0619 | logloss 0.1700072624003231 |

Every warmup and retained full output buffer had the same SHA-256 within and
across arms. That covers 36 buffers per cell and all eight cells. Accuracy and
logloss therefore remain exactly equal as well. The CUDA analytic gate also
uses a cancellation fixture whose ordered and 32-grove answers differ, and
proved that both RF and ExtraTrees selected the ordered graph. The same gate
passed locally on Metal. Candidate process-median max/min spreads were
1.023–1.065. The conservative ratio is the slowest candidate process median
divided by the fastest sequential process median, so the conclusion does not
depend on favorable baseline drift.

A first attempt used all 4,000,000 Taxi inference rows. It was stopped before
a complete cell because the current sequential baseline is CPU-heavy and the
planned three-pass matrix would have exceeded the warm lease. It produced no
quoted timing. The completed 1M rung is the established large forest
inference size and uses both requested R2 datasets.

Compact receipts are in
`bench/results/forest_ordered_resident_2026-09-21/`; raw logs and per-call
JSON remain outside git under
`~/mojolearn-evidence/2026-09-21_forest_ordered/`.
