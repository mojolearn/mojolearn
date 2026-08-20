
## bench_sklearn.py vs /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a9253d3f-96a9-4297-8b71-18d54c0bf0ed/scratchpad/bench_main_t3 (3 rounds, arms alternated per round)

| arm | ours ms | sklearn ms | ratio | verdict | ours [min, max] | sklearn [min, max] | n |
|---|---|---|---|---|---|---|---|
| dbscan | 29.47 | 11.80 | 0.40x | INDISTINGUISHABLE | [9.47, 66.67] | [11.14, 40.75] | 15/15 |
| dbscan_brute | -- | 11.44 | -- | ONE-SIDED | [10.73, 46.51] n=15 | | |
| kmeans | 1102.47 | 3110.72 | 2.82x | ours faster | [979.49, 2163.43] | [2881.23, 4866.46] | 15/15 |
| knn | 975.82 | 1493.37 | 1.53x | INDISTINGUISHABLE | [926.29, 1467.64] | [1377.47, 1998.79] | 15/15 |
| ols | 56.40 | 1116.69 | 19.80x | ours faster | [43.24, 78.97] | [1070.20, 2057.04] | 15/15 |
| ols_normal_eq | -- | 189.07 | -- | ONE-SIDED | [182.80, 419.98] n=15 | | |
| pca | 62.99 | 220.54 | 3.50x | ours faster | [47.99, 80.60] | [160.88, 451.96] | 15/30 |

ratio > 1 means ours is faster; INDISTINGUISHABLE means the min..max ranges overlap and the ratio is not a finding.

```
recorded_at=2026-08-20T12:38:13Z
uname=Darwin Mac 25.5.0 Darwin Kernel Version 25.5.0: Tue Jun  9 22:26:22 PDT 2026; root:xnu-12377.121.10~1/RELEASE_ARM64_T8132 arm64
model=Mac16,12
chip=Apple M4
cores=10
memory_bytes=17179869184
macos=26.5.2
thermal=Note: No thermal warning level has been recorded;Note: No performance warning level has been recorded;Note: No CPU power status has been recorded;
power=Now drawing from 'AC Power'
python=Python 3.14.6
numpy=2.4.4
```
