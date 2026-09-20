# CPU mutual information, 10M rows

Base: `b26e6b12ac740d7cbd06a4669f14dffbf1745c46`. Hardware: Apple M4,
10 logical CPUs, arm64, macOS 26.5.2 (25F84). The requested Gaussian,
multinomial and Bernoulli naive Bayes, LDA/QDA, isotonic and Platt estimator
paths are not present in this checkout, so the sweep continued to this
uncovered production probability metric.

The deterministic workload has 10,000,000 int32 truth and prediction labels,
32 classes, NumPy PCG64 seed `20260920`, and combined input SHA-256
`0a5797d4d4b7e0d6f0144411db5127d6a1289645008e848114b3296caf238375`.
The command was:

```sh
.pixi/envs/default/bin/python bench/bench_mutual_info_host.py LIBRARY --repeats 1
```

Five fresh-process baseline/candidate pairs were alternated. Raw timed seconds:

- baseline: 0.029327375, 0.030047875, 0.030602208, 0.029313792, 0.029518834
- candidate: 0.008161083, 0.008408375, 0.008482500, 0.008309416, 0.008290208

Medians are 0.029519 and 0.008309 seconds, a 71.8% reduction. Every run
returned the exact Float64 value `1.2685227394104004`. Median process peak RSS
was 327,794,688 bytes baseline and 247,840,768 bytes candidate, 79,953,920
bytes lower. The input arrays account for 80,000,000 bytes; the removed copies
therefore explain the memory change.

Both builds returned byte-for-byte identical errors, in the same validation
order, for `n=0` and for an upper class bound below the lower class bound:
`metrics: n must be positive, got 0` and
`metrics: upper_class_range (3) is below lower_class_range (4)`.

The candidate borrows the two already validated contiguous arrays at the
binding boundary. Its contingency loop and Float32 epilogue are statement-for-
statement in the original row/class order, including the sabotage read. Build
passed with `./bindings/build_metrics_host.sh`; `git diff --check` passed.
