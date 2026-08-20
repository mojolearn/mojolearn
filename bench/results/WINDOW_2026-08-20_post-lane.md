
## bench_sklearn.py vs /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a9253d3f-96a9-4297-8b71-18d54c0bf0ed/scratchpad/bench_post (3 rounds, arms alternated per round)

| arm | ours ms | sklearn ms | ratio | verdict | ours [min, max] | sklearn [min, max] | n |
|---|---|---|---|---|---|---|---|
| dbscan | 12.73 | 10.02 | 0.79x | INDISTINGUISHABLE | [8.76, 26.50] | [8.34, 18.48] | 15/15 |
| dbscan_brute | -- | 9.09 | -- | ONE-SIDED | [8.22, 15.35] n=15 | | |
| kmeans | 1657.90 | 2309.56 | 1.39x | ours faster | [1541.67, 2035.98] | [2053.14, 2784.51] | 15/15 |
| knn | 756.00 | 1141.22 | 1.51x | ours faster | [697.17, 916.68] | [954.16, 1467.82] | 15/15 |
| ols | 62.61 | 906.24 | 14.48x | ours faster | [57.66, 74.78] | [834.36, 1110.82] | 15/15 |
| ols_normal_eq | -- | 151.46 | -- | ONE-SIDED | [141.27, 174.69] n=15 | | |
| pca | 75.80 | 147.04 | 1.94x | ours faster | [67.44, 91.86] | [118.08, 236.37] | 15/30 |

ratio > 1 means ours is faster; INDISTINGUISHABLE means the min..max ranges overlap and the ratio is not a finding.

```
recorded_at=2026-08-20T11:24:30Z
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
