# Every public estimator on three GPUs, third run (2026-09-14, 47 lanes, 423 train cells and 585 infer and model cells, all identical)

The record the CPU identity gate diffs against from this commit on. Same `tools/identity_break.py`
protocol as the 2026-09-13 and 2026-09-14 46-lane runs (nine hostile fixtures, each cell fitted twice,
train, infer and model columns), at commit 7bf4f4cc9 with everything the day added: the `pca-whiten`
lane, the Holt-Winters and ARIMA forecast probes (infer cells for both forecasters), save and load on
ols, ridge, tsvd, logistic, pca, kde, svc and the three k-NN classes (model cells), the CPU host set
built beside the GPU set on every box (the byte LM host lanes have every cell).

| column | box | commit | cells |
|---|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal, two-core cap, the 17 IDENTICAL bindings the 0.8.5 macOS release built at 8d16ce2f (no GPU source changed since) | 7bf4f4cc9 | 423: stable 423, refused 0; infer 369, model 216 |
| nvidia-h100-sm_90a | RunPod H100 80GB HBM3 (`bench/results/e1g/2026-09-14_103118-nvidia-h100-identity-47-lanes`) | 7bf4f4cc9 | 423: stable 423, refused 0 |
| amd-mi300x-gfx942 | Hot Aisle MI300X, rocm/dev-ubuntu-22.04 container (`bench/results/e1g/2026-09-14_103118-amd-mi300x-hotaisle-identity-47-lanes`) | 7bf4f4cc9 | 423: stable 423, refused 0 |

`diff.apple-nvidia-amd.txt`: `summary: IDENTICAL=423` and `summary (infer/model): IDENTICAL=585,
N/A=261`. No DIVERGENT, MOVED or ONE-COLUMN cell. Compared with the 46-lane run of the same
morning: 9 more train cells (pca-whiten), 126 more infer and model cells (18 forecaster infer cells,
9 pca-whiten infer, 99 model cells from the new save and load).

The seven lanes with no infer column, by construction: kmeans (fit and fit_predict only), dbscan,
agglomerative and spectral (transductive; spectral's predict raises), gemm-pinned and metrics
(functions). See `tools/identity_break.py`'s docstring for the audit.
