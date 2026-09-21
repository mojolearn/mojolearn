# Ordered resident RF/ExtraTrees inference

**Verdict: promoted on NVIDIA IDENTICAL and Apple FAST/IDENTICAL.** The route retains the resident GPU
model and I/O workspaces while using the existing strict increasing-tree
aggregation kernel. NVIDIA IDENTICAL `inference_engine="auto"` selects it by
default. Apple FAST and IDENTICAL select the same strict ordered resident
route. `-D MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF=1` restores the former
sequential IDENTICAL AUTO path and resident FAST 32-grove graph. AMD and
other vendors retain their previous defaults; the positive define remains
available for explicit IDENTICAL experiments on them.

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

After promotion, fresh no-define NVIDIA RF and ExtraTrees bindings were built
on the same H100. The analytic gate reported the ordered route selected, and
a public `inference_engine="auto"` smoke compared it with explicit sequential
inference on 100,000 rows from both datasets. All eight RF/ET predict/proba
cells matched full-buffer SHA-256 for three AUTO repeats. This verifies the
promoted default rather than only the experimental define.

## Apple Metal support

The no-define Apple IDENTICAL route was also compared with the former
sequential route on the same two datasets. Models were fitted once on 250,000
rows and each process predicted 1,000,000 rows. The process pattern remained
three alternating arms, one excluded warmup, and five retained calls.

| dataset | model | operation | sequential ms | ordered resident ms | speedup | conservative candidate/baseline | quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxi | RF | predict | 5524.572 | 173.112 | 31.91x | 0.0427 | accuracy 0.779465 |
| Taxi | RF | proba | 5824.373 | 202.813 | 28.72x | 0.0389 | logloss 0.5126321444971583 |
| Taxi | ET | predict | 3915.889 | 215.645 | 18.16x | 0.0600 | accuracy 0.777343 |
| Taxi | ET | proba | 3827.299 | 196.738 | 19.45x | 0.0591 | logloss 0.5195547196918995 |
| Istella-S | RF | predict | 6188.296 | 641.641 | 9.65x | 0.1238 | accuracy 0.945320 |
| Istella-S | RF | proba | 6214.627 | 607.458 | 10.23x | 0.1225 | logloss 0.1429750222767868 |
| Istella-S | ET | predict | 4105.800 | 568.655 | 7.22x | 0.1570 | accuracy 0.926648 |
| Istella-S | ET | proba | 4301.406 | 562.888 | 7.64x | 0.1581 | logloss 0.1837667854902221 |

All warmup and retained full-buffer hashes match across the two Apple arms,
so accuracy and logloss are bitwise-derived from the same predictions. The
mechanical verdict is `reject` because local contention put the cross-process
candidate median spread above 1.10 in five cells. The sequential arm was also
above 1.10 in seven cells. This run is therefore supporting performance
evidence rather than the promotion gate. Even its conservative slowest
candidate / fastest baseline ratios are 0.0389--0.1581.

Fresh no-define Apple FAST bindings then fitted separate FAST models and
compared public AUTO with explicit sequential inference on 100,000 rows.
All eight Taxi/Istella-S RF/ET predict/proba cells matched full-buffer SHA-256
for three AUTO repeats. This proves that the FAST default keeps strict tree
order and exact output bytes. The pure policy gate passed, and explicit `_OFF`
FAST and IDENTICAL builds reported `RESIDENT_ORDERED False` while passing the
resident lifecycle and routing checks, proving the restore path remains live.
Compact Apple receipts are in
`bench/results/forest_ordered_apple_2026-09-21/`; raw logs remain in
`~/mojolearn-evidence/2026-09-21_forest_ordered_apple{,_fast}/`.

The reused H100 was terminated after the NVIDIA work. RunPod returned HTTP
204 for deletion and HTTP 404 on the verification lookup; the teardown receipt
is tracked beside the NVIDIA results.
