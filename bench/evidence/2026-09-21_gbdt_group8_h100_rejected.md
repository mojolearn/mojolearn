# Eight-tree grouped GBDT inference on H100: rejected

The experiment widened IDENTICAL symmetric resident GBDT inference from four
trees per launch to eight behind `MOJOLEARN_GBDT_GROUP8=1`. It preserved the
original tree order and Float32 addition order. The candidate is rejected and
its implementation remains out of the final tree.

The run used an NVIDIA H100 80GB HBM3 with driver 580.126.09. It reused the
warm RunPod box from the preceding GBDT trial and the same two medium-large
datasets staged from Cloudflare R2:

| dataset | R2 source shape | timed shape |
|---|---:|---:|
| Taxi | 4,000,000 x 11 | 1,000,000 x 16 |
| Istella-S | 2,043,304 x 220 | 1,000,000 x 220 |

The baseline and candidate used the same prepared models and arrays. For each
public path, three alternating process pairs each excluded one warmup and then
timed five samples of five consecutive calls. Values below are the median of
the three process medians, in per-call milliseconds.

| dataset / public path | group4 -> group8 ms | group4/group8 | baseline spreads | candidate spreads | qualified |
|---|---:|---:|---:|---:|---:|
| Taxi `predict` | 13.146807 -> 12.475643 | 1.053798 | 1.163, 1.272, 1.222 | 1.076, 1.021, 1.015 | no |
| Taxi `predict_proba` | 22.921026 -> 24.666635 | 0.929232 | 1.264, 1.197, 1.285 | 1.301, 1.321, 1.198 | no |
| Istella-S `predict` | 73.366614 -> 70.090014 | 1.046748 | 1.035, 1.038, 1.360 | 1.083, 1.237, 1.038 | no |
| Istella-S `predict_proba` | 87.192305 -> 81.096122 | 1.075172 | 1.120, 1.122, 1.188 | 1.098, 1.265, 1.072 | no |

The strict gate requires every process in both arms to have maximum/minimum
spread at most 1.10. Every cell failed that gate. Taxi `predict_proba` also
regressed by about 7.1%. The unqualified four-cell geometric mean is 1.024591x;
it is diagnostic only and does not support promotion.

## Identity and quality

Every output hash was constant across all repeated calls and equal across the
two builds. Logloss and accuracy were exactly equal for baseline and candidate
on both datasets and both public paths. The binding hashes differed
(`9fe002284c14a6c5...` baseline and `e8f7aff05fbbb73b...` candidate), confirming
that both compiled arms actually ran.

Compact receipts are in `bench/results/gbdt_group8_2026-09-21/`. They retain
all timed samples, spreads, ratios, hashes, quality observations, build hashes,
and R2 provenance. Full logs are outside the repository at
`/Users/andrewhendel/mojolearn-evidence/2026-09-21_gbdt_group8/`.
