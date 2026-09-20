# CPU entropy, 10M labels

Base: `3aac9357132b6a71a6665baaae7ba45e75f0e0e5`. Hardware: Apple M4,
10 logical CPUs, arm64, macOS 26.5.2 (25F84). Workload: 10,000,000 int32
labels across 64 classes from NumPy PCG64 seed `20260920`; input SHA-256
`a26b22e9ed7f11f65969a842fd4191c354d9abb1ba4ea324242acc849cf317d2`.

Command, alternated baseline then candidate in five fresh process pairs:

```sh
.pixi/envs/default/bin/python bench/bench_entropy_host.py LIBRARY --repeats 1
```

Raw seconds:

- baseline: 0.033874958, 0.031455041, 0.028796375, 0.028678167, 0.028478084
- candidate: 0.013954417, 0.016570875, 0.013572625, 0.014108333, 0.013058292

Medians were 0.028796 and 0.013954 seconds, a 51.5% reduction. All A/B
runs returned exactly `4.15887975692749`. Median peak RSS was 128,876,544
bytes baseline and 89,145,344 bytes candidate, 39,731,200 bytes lower.

The candidate borrows the already validated contiguous label array instead of
copying its 40,000,000 bytes. Histogram construction stays in ascending row
order, and the probability/log accumulation stays in ascending class order.
The sabotage row-zero substitution is retained. Both builds produced the same
errors for `n=0` and an upper class range below the lower range:
`entropy: n must be positive, got 0` and
`metrics: upper_class_range (3) is below lower_class_range (4)`.

`./bindings/build_metrics_host.sh` and `git diff --check` passed.
