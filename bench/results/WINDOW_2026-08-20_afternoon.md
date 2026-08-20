
## bench_sklearn.py vs /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a9253d3f-96a9-4297-8b71-18d54c0bf0ed/scratchpad/bench_main_w (3 rounds, arms alternated per round)

| arm | ours ms | sklearn ms | ratio | verdict | ours [min, max] | sklearn [min, max] | n |
|---|---|---|---|---|---|---|---|
| dbscan | 10.48 | 10.23 | 0.98x | INDISTINGUISHABLE | [8.23, 17.26] | [8.33, 20.23] | 15/15 |
| dbscan_brute | -- | 10.21 | -- | ONE-SIDED | [8.50, 16.98] n=15 | | |
| kmeans | 764.35 | 2488.42 | 3.26x | ours faster | [749.38, 1246.88] | [2022.30, 3405.42] | 15/15 |
| knn | 705.66 | 1196.35 | 1.70x | ours faster | [689.52, 917.88] | [969.82, 1583.26] | 15/15 |
| ols | 61.98 | 913.06 | 14.73x | ours faster | [58.63, 78.24] | [830.25, 1155.74] | 15/15 |
| ols_normal_eq | -- | 166.85 | -- | ONE-SIDED | [139.88, 361.15] n=15 | | |
| pca | 63.89 | 144.37 | 2.26x | ours faster | [60.30, 80.87] | [119.94, 282.68] | 15/30 |

ratio > 1 means ours is faster; INDISTINGUISHABLE means the min..max ranges overlap and the ratio is not a finding.

```
recorded_at=2026-08-20T11:53:51Z
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
