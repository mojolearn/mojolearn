# CPU weighted accuracy, 10M rows

Base: `a46f2af2e97ebc919bf8eab1183ddcf91a8ac3e0`. Hardware: Apple M4,
10 logical CPUs, arm64, macOS 26.5.2 (25F84). Workload: 10,000,000 rows,
11 int32 classes, float32 weights, NumPy PCG64 seed `20260920`; combined
input SHA-256
`83ce744083d10bae35a39a400fefa2391c3d0ae98b7cf8f866a7c18deacd4faf`.

Command, alternated baseline then candidate in five fresh process pairs:

```sh
.pixi/envs/default/bin/python bench/bench_weighted_accuracy_host.py LIBRARY --repeats 1
```

Raw seconds:

- baseline: 0.159650750, 0.158281667, 0.194060167, 0.154948625, 0.163748542
- candidate: 0.031908708, 0.032124458, 0.032124875, 0.032114292, 0.033319959

Medians were 0.159651 and 0.032124 seconds, a 79.9% reduction. Every run
returned exactly `0.2501491904258728`. Median peak RSS was 408,944,640 bytes
baseline and 208,601,088 bytes candidate, 200,343,552 bytes lower.

The original binding copied two label arrays and the weights, then allocated
two n-value term arrays. The candidate borrows the validated inputs and forms
the numerator and denominator directly in two 256-value slabs. Each slab uses
the same halving tree, chunk partials retain ascending order, and the final
fold remains ascending. The negative-control shifted indexing and dropped
sample-zero weight are retained.

Baseline and candidate produced identical errors for `n=0` and zero total
weight: `metrics: n must be positive, got 0` and
`weighted accuracy_score: the weights must have positive total`. A 100-row,
five-repeat regression also matched exact results and errors.

`./bindings/build_metrics_host.sh` and `git diff --check` passed.
